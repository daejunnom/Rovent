//! Bounded actual CPU NN integration execution and independent evidence audit.
//! Process cleanup, B's NN evidence and A's PGN audit remain separate gates.
//! Neither this API nor a successful integration receipt permits strength runs.

use crate::native_launch::NativeLaunchDeclaration;
use crate::{
    ArenaError, CleanupStatus, NativeLaunchOwner, NativeSnapshotReceipt, PairPgnAudit,
    ProcessOutput, ProcessReceipt,
};
use serde::Serialize;
#[cfg(target_os = "linux")]
use std::sync::Arc;
use std::{
    fmt,
    sync::{
        Mutex, TryLockError,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

#[cfg(any(test, target_os = "linux"))]
fn clamp_original_wall(
    declared_ms: u64,
    deadline: Option<std::time::Instant>,
    now: std::time::Instant,
) -> Result<u64, ArenaError> {
    let Some(deadline) = deadline else {
        return Ok(declared_ms);
    };
    let remaining = deadline
        .checked_duration_since(now)
        .and_then(|d| u64::try_from(d.as_millis()).ok())
        .filter(|n| *n > 0)
        .ok_or_else(|| {
            ArenaError::Budget("original provider run window expired before child dispatch".into())
        })?;
    Ok(declared_ms.min(remaining))
}

#[cfg(target_os = "linux")]
fn earlier_deadline(
    first: Option<std::time::Instant>,
    second: Option<std::time::Instant>,
) -> Option<std::time::Instant> {
    match (first, second) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) | (None, Some(a)) => Some(a),
        (None, None) => None,
    }
}

#[cfg(any(test, target_os = "linux"))]
fn v6_effective_deadlines(
    run_until: std::time::Instant,
    cleanup_until: std::time::Instant,
    provider_until: Option<std::time::Instant>,
) -> Result<(std::time::Instant, std::time::Instant, u64), ArenaError> {
    let reserve = cleanup_until
        .checked_duration_since(run_until)
        .ok_or_else(|| ArenaError::Budget("V6 original cleanup window is invalid".into()))?;
    let run = provider_until.map_or(run_until, |deadline| deadline.min(run_until));
    let cleanup = run
        .checked_add(reserve)
        .ok_or_else(|| ArenaError::Budget("V6 effective cleanup window overflow".into()))?
        .min(cleanup_until);
    // The existing process supervisor permits twice its integer-millisecond
    // grace. Round up only enough to cover the original whole deadline.
    let grace_ms = u64::try_from(reserve.as_nanos().div_ceil(2_000_000))
        .map_err(|_| ArenaError::Budget("V6 cleanup grace overflow".into()))?;
    Ok((run, cleanup, grace_ms))
}
#[cfg(test)]
mod original_window_tests {
    use super::*;
    #[test]
    fn original_window_only_reduces_time_and_refuses_expired_dispatch() {
        let now = std::time::Instant::now();
        let deadline = now + std::time::Duration::from_millis(37);
        assert_eq!(clamp_original_wall(100, Some(deadline), now).unwrap(), 37);
        assert_eq!(clamp_original_wall(9, Some(deadline), now).unwrap(), 9);
        assert_eq!(clamp_original_wall(100, None, now).unwrap(), 100);
        assert!(clamp_original_wall(100, Some(deadline), deadline).is_err());
        assert!(
            clamp_original_wall(100, Some(now + std::time::Duration::from_nanos(1)), now).is_err()
        );
    }

    #[test]
    fn v6_stricter_deadline_and_odd_cleanup_do_not_extend_the_original_window() {
        let now = std::time::Instant::now();
        let run = now + std::time::Duration::from_secs(10);
        let cleanup = run + std::time::Duration::from_millis(5);
        let earlier = now + std::time::Duration::from_secs(8);
        assert_eq!(
            v6_effective_deadlines(run, cleanup, Some(earlier)).unwrap(),
            (earlier, earlier + std::time::Duration::from_millis(5), 3)
        );
        assert_eq!(
            v6_effective_deadlines(run, cleanup, None).unwrap(),
            (run, cleanup, 3)
        );
        assert!(v6_effective_deadlines(cleanup, run, None).is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn v6_control_busy_does_not_block_or_replace_the_original_owner() {
        struct Control(Arc<std::sync::atomic::AtomicUsize>);
        impl NativeV6Control for Control {
            fn poll_run(
                &mut self,
                _: &crate::process::SpawnCustody,
                _: std::time::Instant,
            ) -> Result<(), ArenaError> {
                self.0.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            fn poll_cleanup(
                &mut self,
                _: &crate::process::SpawnCustody,
                _: std::time::Instant,
            ) -> Result<(), ArenaError> {
                self.0.fetch_add(10, Ordering::Relaxed);
                Ok(())
            }
            fn pre_signal_enabled(&self) -> bool {
                true
            }
            fn poll_pre_signal(
                &mut self,
                _: &crate::process::SpawnCustody,
                _: std::time::Instant,
            ) -> Result<crate::process::PreSignalProgress, ArenaError> {
                self.0.fetch_add(100, Ordering::Relaxed);
                Ok(crate::process::PreSignalProgress::Acknowledged)
            }
        }
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let control: Arc<Mutex<Box<dyn NativeV6Control>>> =
            Arc::new(Mutex::new(Box::new(Control(Arc::clone(&calls)))));
        let custody = crate::process::SpawnCustody::default();
        let held = control.lock().unwrap();
        poll_v6_control(&control, &custody, std::time::Instant::now(), false).unwrap();
        assert_eq!(
            poll_v6_pre_signal(&control, &custody, std::time::Instant::now()).unwrap(),
            crate::process::PreSignalProgress::Pending { partial: false }
        );
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        drop(held);
        poll_v6_control(&control, &custody, std::time::Instant::now(), false).unwrap();
        poll_v6_control(&control, &custody, std::time::Instant::now(), true).unwrap();
        assert_eq!(
            poll_v6_pre_signal(&control, &custody, std::time::Instant::now()).unwrap(),
            crate::process::PreSignalProgress::Acknowledged
        );
        assert_eq!(calls.load(Ordering::Relaxed), 111);
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeProviderSessionAudit {
    pub role: String,
    pub process_id: u32,
    pub process_run_id: String,
    pub startup_sha256: String,
    pub termination_sha256: String,
    pub completed_by_runtime: u64,
    pub actual_cpu_inference_observed: bool,
    pub physical_drain: String,
}

/// Read-only view of an actual launch-owner input pin. The path names the
/// private snapshot, not the source asset or the helper's registered program.
#[cfg(target_os = "linux")]
#[doc(hidden)]
pub struct NativeProviderInputView<'a> {
    pub artifact: &'a rz_experiments::ArtifactRef,
    pub path: &'a std::path::Path,
    pub file: &'a std::fs::File,
}

#[cfg(target_os = "linux")]
#[doc(hidden)]
pub struct NativeProviderInputContext<'a> {
    pub directory: &'a cap_std::fs::Dir,
    pub pins: &'a [NativeProviderInputView<'a>],
}

/// Provider-specific wire acceptance is distinct from shared process/PGN gates.
#[doc(hidden)]
pub trait NativeProviderDeclaration: NativeLaunchDeclaration {
    type Audit: Clone + fmt::Debug + Serialize + Send + 'static;
    fn scope(&self) -> &'static str;
    fn receipt_filename(&self) -> &'static str;
    fn receipt_version(&self) -> u32 {
        1
    }
    fn contract_revision(&self) -> &str {
        "0.1"
    }
    /// An optional original run window that started before common preparation
    /// checks. Historical providers keep None and their existing timing. This
    /// only reduces dispatch time; the separately registered cleanup reserve is
    /// never increased or removed.
    fn original_wall_deadline(&self) -> Result<Option<std::time::Instant>, ArenaError> {
        Ok(None)
    }
    #[cfg(target_os = "linux")]
    fn validate_runtime_admission(&self) -> Result<(), ArenaError> {
        Ok(())
    }
    /// Consume provider-specific evidence from the owner-pinned preflight root
    /// after both probes have closed, before admitting any game process. The
    /// default preserves historical Own/V1/V2 behavior and creates no evidence.
    #[cfg(target_os = "linux")]
    fn validate_external_preflight(
        &self,
        _role: rz_experiments::NativeEngineRole,
        _preflight: &crate::ExternalUciPreflight,
        _runtime_directory: &cap_std::fs::Dir,
        _inputs: &NativeProviderInputContext<'_>,
    ) -> Result<Vec<(String, Vec<u8>)>, ArenaError> {
        Ok(Vec::new())
    }
    fn expected_provider_sessions(&self) -> usize {
        4
    }
    /// Opt-in only: this provider can launch work whose physical completion
    /// cannot be inferred from OS process-group cleanup. Historical providers
    /// retain their original behavior until they explicitly implement this.
    fn requires_physical_lifetime_guard(&self) -> bool {
        false
    }
    /// Verify the actual supervisor-owned preflight child has not created a
    /// GPU session. An argument or a child-reported flag alone is insufficient.
    #[cfg(target_os = "linux")]
    fn verify_preflight_no_gpu(
        &self,
        _role: rz_experiments::NativeEngineRole,
        _probe: NativePreflightProbe,
        _process: &ProcessOutput,
        _runtime_directory: &cap_std::fs::Dir,
        _inputs: &NativeProviderInputContext<'_>,
    ) -> Result<(), ArenaError> {
        Err(ArenaError::Invalid(
            "guarded preflight GPU lifetime is unknown".into(),
        ))
    }
    /// Independently verify physical completion from the original supervised
    /// process, role-bound sessions and pinned inputs. This must not depend on
    /// unrelated PGN, logging or model-output acceptance succeeding.
    #[cfg(target_os = "linux")]
    fn verify_pair_physical_completion(
        &self,
        _process: &ProcessOutput,
        _baseline_runtime: &cap_std::fs::Dir,
        _candidate_runtime: &cap_std::fs::Dir,
        _inputs: &NativeProviderInputContext<'_>,
    ) -> Result<(), ArenaError> {
        Err(ArenaError::Invalid(
            "guarded pair GPU lifetime is unknown".into(),
        ))
    }
    fn uses_provider_records(
        &self,
        role: rz_experiments::NativeEngineRole,
    ) -> Result<bool, ArenaError> {
        Ok(self.engine_view(role)?.external.is_none())
    }
    fn provider_export_artifact(
        &self,
        role: rz_experiments::NativeEngineRole,
    ) -> Result<Option<&rz_experiments::ArtifactRef>, ArenaError> {
        let engine = self.engine_view(role)?;
        if engine.external.is_some() {
            Ok(None)
        } else {
            Ok(Some(engine.artifact(
                rz_experiments::NativeArtifactRole::ExportManifest,
            )?))
        }
    }
    #[cfg(target_os = "linux")]
    fn validate_conversion_manifest(
        &self,
        _role: rz_experiments::NativeEngineRole,
        startup: &[u8],
        manifest: &[u8],
        batch_experiment: bool,
    ) -> Result<(), ArenaError> {
        linux::verify_conversion_provenance(
            &linux::json(startup)?,
            &linux::json(manifest)?,
            batch_experiment,
        )
    }
    fn role_startup_filename(&self, _role: rz_experiments::NativeEngineRole) -> &'static str {
        self.startup_filename()
    }
    fn role_termination_filename(&self, _role: rz_experiments::NativeEngineRole) -> &'static str {
        self.termination_filename()
    }
    fn startup_filename(&self) -> &'static str;
    fn termination_filename(&self) -> &'static str;
    fn claim_policy(&self) -> rz_experiments::ClaimPolicy {
        rz_experiments::ClaimPolicy::ExplicitClaim
    }
    fn validate_clock_trace(
        &self,
        _stdout: &[u8],
        _pgn: &PairPgnAudit,
    ) -> Result<Option<crate::NativePilotClockAudit>, ArenaError> {
        Ok(None)
    }
    fn audit_failure_trace(
        &self,
        _stdout: &[u8],
        _pair: &crate::PairSpec,
    ) -> Result<Option<crate::NativePilotFailureAudit>, ArenaError> {
        Ok(None)
    }
    #[cfg(target_os = "linux")]
    fn validate_records(
        &self,
        role: rz_experiments::NativeEngineRole,
        startup: &[u8],
        termination: &[u8],
        session: &str,
    ) -> Result<(Self::Audit, u32), ArenaError>;
    /// Context-aware production entry. Historical providers keep their exact
    /// validator; inputs and supervisor output never become wire declarations.
    #[cfg(target_os = "linux")]
    fn validate_records_with_context(
        &self,
        role: rz_experiments::NativeEngineRole,
        startup: &[u8],
        termination: &[u8],
        session: &str,
        _inputs: &NativeProviderInputContext<'_>,
        _supervisor_stdout: &[u8],
    ) -> Result<(Self::Audit, u32), ArenaError> {
        self.validate_records(role, startup, termination, session)
    }
    #[cfg(target_os = "linux")]
    fn validate_provider_session_set(
        &self,
        _sessions: &[Self::Audit],
        _preflight: &[crate::ExternalUciPreflight],
        _runtime_root: &cap_std::fs::Dir,
        _inputs: &NativeProviderInputContext<'_>,
        _supervisor_stdout: &[u8],
        _artifacts: &[rz_experiments::ArtifactRef],
    ) -> Result<(), ArenaError> {
        Ok(())
    }
    #[cfg(target_os = "linux")]
    fn verify_session_evidence(
        &self,
        _role: rz_experiments::NativeEngineRole,
        _pid: u32,
        _startup: &[u8],
        _runtime_directory: &cap_std::fs::Dir,
    ) -> Result<Vec<(String, Vec<u8>)>, ArenaError> {
        Ok(Vec::new())
    }
    #[cfg(target_os = "linux")]
    fn external_exit_ids(
        &self,
        _stdout: &[u8],
        _native_pids: &[u32],
    ) -> Result<Vec<u32>, ArenaError> {
        Ok(Vec::new())
    }
    #[cfg(target_os = "linux")]
    fn external_game_process_ids(
        &self,
        _role: rz_experiments::NativeEngineRole,
        external_ids: &[u32],
        _root_output: &cap_std::fs::Dir,
    ) -> Result<Vec<u32>, ArenaError> {
        Ok(external_ids.to_vec())
    }
    /// Additional provider-specific acceptance of the supervisor-owned stdout.
    /// CPU V1 keeps its existing gate; CUDA requires the pinned runner's exit
    /// observations for the same four identities validated from native records.
    #[cfg(target_os = "linux")]
    fn validate_process_exit_trace(
        &self,
        _stdout: &[u8],
        _expected_pids: &[u32],
    ) -> Result<(), ArenaError> {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativePreflightProbe {
    Identification,
    Readiness,
}
impl NativeProviderDeclaration for rz_experiments::LockedIntegrationPairSpecV1 {
    type Audit = NativeProviderSessionAudit;
    fn scope(&self) -> &'static str {
        "cpu_nn_pair_integration_process_provider_and_native_rules"
    }
    fn receipt_filename(&self) -> &'static str {
        "native-pair-receipt.v1.json"
    }
    fn startup_filename(&self) -> &'static str {
        "native-cpu-startup.v1.json"
    }
    fn termination_filename(&self) -> &'static str {
        "native-cpu-termination.v1.json"
    }
    #[cfg(target_os = "linux")]
    fn validate_records(
        &self,
        role: rz_experiments::NativeEngineRole,
        startup: &[u8],
        termination: &[u8],
        session: &str,
    ) -> Result<(Self::Audit, u32), ArenaError> {
        let audit = validate_native_provider_record_fields(
            startup,
            termination,
            self.input().engine(role)?,
            session,
        )?;
        let pid = audit.process_id;
        Ok((audit, pid))
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct NativePairReceipt<A = NativeProviderSessionAudit> {
    pub receipt_version: u32,
    pub execution_ready: bool,
    pub strength_eligible: bool,
    pub validation_scope: String,
    pub input_sha256: String,
    pub pair_id: String,
    pub contract_revision: String,
    pub runner_source_commit: String,
    pub runner_binary_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub match_resources: Option<rz_experiments::MatchResourcePlan>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ponder_enabled: Option<bool>,
    pub process: ProcessReceipt,
    pub snapshots: Vec<NativeSnapshotReceipt>,
    pub snapshot_cache_hints: Vec<SnapshotCacheHint>,
    pub artifacts: Vec<rz_experiments::ArtifactRef>,
    pub provider_sessions: Vec<A>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical_lifetime_error: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub external_preflight: Vec<crate::ExternalUciPreflight>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub external_execution: Vec<crate::ExternalGameObservation>,
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub environments: std::collections::BTreeMap<String, crate::EngineEnvironmentObservation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_audit_error: Option<String>,
    pub provider_audit_error: Option<String>,
    pub pgn_audit: Option<PairPgnAudit>,
    pub pgn_audit_error: Option<String>,
    pub integration_checks_passed: bool,
    /// Integration-only runs grant no scoring authority, including valid draws.
    pub scored_games: u32,
    pub incomplete_games: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clock_audit: Option<crate::NativePilotClockAudit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clock_audit_error: Option<String>,
    /// Failure accounting only; this never grants provider, clock or score acceptance.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_audit: Option<crate::NativePilotFailureAudit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_audit_error: Option<String>,
    pub primary_error: Option<String>,
    pub cleanup_verified: bool,
    pub unresolved_owner_retained: bool,
    pub limitations: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SnapshotCacheHint {
    pub phase: String,
    pub game: Option<u8>,
    pub eligible_files: usize,
    pub advised_files: usize,
    pub errors: Vec<String>,
    pub scope: &'static str,
}

#[cfg(target_os = "linux")]
impl SnapshotCacheHint {
    pub(crate) fn new(phase: &str, game: Option<u8>) -> Self {
        Self {
            phase: phase.into(),
            game,
            eligible_files: 0,
            advised_files: 0,
            errors: Vec::new(),
            scope: "private_verified_readonly_snapshot_fds_best_effort_not_reclamation_proof",
        }
    }

    // Post-run callers supply eligibility only after owned cleanup. An
    // unresolved child retains all pins and receives no new rolling advice.
    pub(crate) fn rolling(eligible: bool, phase: &str) -> Option<Self> {
        (cfg!(feature = "experimental-snapshot-reclaim") && eligible)
            .then(|| Self::new(phase, None))
    }

    pub(crate) fn advise(&mut self, file: &std::fs::File, bytes: u64) {
        if bytes >= 8 * 1024 * 1024 {
            self.record_advice(linux::advise_input_cache(file));
        }
    }

    fn record_advice(&mut self, result: Result<(), ArenaError>) {
        self.eligible_files += 1;
        match result {
            Ok(()) => self.advised_files += 1,
            Err(error) => self.errors.push(error.to_string()),
        }
    }

    pub(crate) fn emit(&self) {
        // Small status survives with tracing disabled; failure never cancels play.
        if let Ok(json) = serde_json::to_string(self) {
            eprintln!("native_cache_hint={json}");
        }
    }
}

/// The original V6 control owner remains in the same bundle as the runner.
/// Its callbacks must be bounded/nonblocking and do not attest completion.
#[cfg(target_os = "linux")]
pub(crate) enum NoGameProgressReadback {
    Observed(crate::pals_adaptive_launch::SmokeControlProgress),
    Busy,
    Poisoned,
    Unavailable,
}

#[cfg(target_os = "linux")]
pub(crate) trait NativeV6Control: Send {
    /// The actual original execution allocation held by this control owner.
    /// A missing owner is never accepted as an equivalent declaration.
    fn original_execution(
        &self,
    ) -> Option<&Arc<crate::pals_adaptive_launch::PalsV6ExecutionStart>> {
        None
    }
    /// The same original role listeners must be retained by this control on
    /// every completion or quarantine path; names alone grant no authority.
    fn original_role_sockets(
        &self,
    ) -> Option<&Arc<crate::pals_adaptive_launch::PalsV6RoleSockets>> {
        None
    }
    /// This is a borrow of optional NoGame frame observations, never an
    /// acknowledgement, process completion, or generic V6 game receipt.
    fn no_game_grants(
        &self,
    ) -> Option<&[Option<crate::pals_adaptive_launch::SmokeGrantObservation>; 2]> {
        None
    }
    fn no_game_progress(&self) -> NoGameProgressReadback {
        NoGameProgressReadback::Unavailable
    }
    fn poll_run(
        &mut self,
        custody: &crate::process::SpawnCustody,
        until: std::time::Instant,
    ) -> Result<(), ArenaError>;
    fn poll_cleanup(
        &mut self,
        custody: &crate::process::SpawnCustody,
        until: std::time::Instant,
    ) -> Result<(), ArenaError>;
    /// Declare support for this control's whole lifetime. A control may
    /// acquire its original role channels later, after the child is spawned;
    /// its callback must refuse to send without an authenticated sender.
    /// Existing controls keep prior behavior.
    fn pre_signal_enabled(&self) -> bool {
        false
    }
    fn poll_pre_signal(
        &mut self,
        _custody: &crate::process::SpawnCustody,
        _until: std::time::Instant,
    ) -> Result<crate::process::PreSignalProgress, ArenaError> {
        Err(ArenaError::Invalid(
            "V6 pre-signal control unavailable".into(),
        ))
    }
}

#[cfg(target_os = "linux")]
pub(crate) struct NativeV6Supervisor {
    execution: Arc<crate::pals_adaptive_launch::PalsV6ExecutionStart>,
    control: Arc<Mutex<Box<dyn NativeV6Control>>>,
    role_sockets: Option<Arc<crate::pals_adaptive_launch::PalsV6RoleSockets>>,
    // If pinned input inspection fails, retain the same owner/control in the
    // normal failure bundle without starting preflight or a child.
    original_pin_error: Option<ArenaError>,
    pre_signal_enabled: bool,
    pre_signal_observation: crate::process::PreSignalObservation,
    no_game_evidence: Option<NoGameEvidence>,
    // Compiled source-byte observation only; it does not attest the separate
    // proxy executable or affect admission and older canonical records.
    cancel_source_observation: crate::pals_adaptive_launch::CancelSourceObservation,
}

/// Rejected construction returns the exact original control owner. The
/// caller retains its own execution Arc because construction only borrows it.
#[cfg(target_os = "linux")]
#[must_use = "retain or return the original V6 control owner"]
pub(crate) struct NativeV6SupervisorRejection {
    pub(crate) error: ArenaError,
    pub(crate) control: Box<dyn NativeV6Control>,
}

#[cfg(target_os = "linux")]
impl fmt::Debug for NativeV6SupervisorRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeV6SupervisorRejection")
            .field("error", &self.error)
            .field("original_control_retained", &true)
            .finish()
    }
}

#[cfg(target_os = "linux")]
impl NativeV6Supervisor {
    pub(crate) fn new(
        execution: &Arc<crate::pals_adaptive_launch::PalsV6ExecutionStart>,
        control: Box<dyn NativeV6Control>,
    ) -> Result<Self, NativeV6SupervisorRejection> {
        let capabilities = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (
                control
                    .original_execution()
                    .is_some_and(|original| Arc::ptr_eq(original, execution)),
                control.pre_signal_enabled(),
                control.original_role_sockets().cloned(),
            )
        }));
        let (same_owner, pre_signal_enabled, role_sockets) = match capabilities {
            Ok(value) => value,
            Err(_) => {
                return Err(NativeV6SupervisorRejection {
                    error: ArenaError::Io("V6 control capability observation panicked".into()),
                    control,
                });
            }
        };
        if !same_owner {
            return Err(NativeV6SupervisorRejection {
                error: ArenaError::Integrity(
                    "V6 control and supervisor execution owners differ".into(),
                ),
                control,
            });
        }
        Ok(Self {
            execution: Arc::clone(execution),
            control: Arc::new(Mutex::new(control)),
            role_sockets,
            original_pin_error: None,
            pre_signal_enabled,
            pre_signal_observation: crate::process::PreSignalObservation::default(),
            no_game_evidence: None,
            cancel_source_observation:
                crate::pals_adaptive_launch::compiled_cancel_source_observation(),
        })
    }
}

#[cfg(target_os = "linux")]
fn poll_v6_pre_signal(
    control: &Mutex<Box<dyn NativeV6Control>>,
    custody: &crate::process::SpawnCustody,
    until: std::time::Instant,
) -> Result<crate::process::PreSignalProgress, ArenaError> {
    let mut owner = match control.try_lock() {
        Ok(owner) => owner,
        Err(TryLockError::WouldBlock) => {
            return Ok(crate::process::PreSignalProgress::Pending { partial: false });
        }
        Err(TryLockError::Poisoned(_)) => {
            return Err(ArenaError::Io("v6 original control owner poisoned".into()));
        }
    };
    owner.poll_pre_signal(custody, until)
}

#[cfg(target_os = "linux")]
fn poll_v6_control(
    control: &Mutex<Box<dyn NativeV6Control>>,
    custody: &crate::process::SpawnCustody,
    until: std::time::Instant,
    cleanup: bool,
) -> Result<(), ArenaError> {
    let mut owner = match control.try_lock() {
        Ok(owner) => owner,
        // The original owner remains intact and the supervisor will poll
        // again; transient contention is not a process failure or completion.
        Err(TryLockError::WouldBlock) => return Ok(()),
        Err(TryLockError::Poisoned(_)) => {
            return Err(ArenaError::Io("v6 original control owner poisoned".into()));
        }
    };
    if cleanup {
        owner.poll_cleanup(custody, until)
    } else {
        owner.poll_run(custody, until)
    }
}

struct NativeRunBundle<S: NativeProviderDeclaration> {
    owner: Option<NativeLaunchOwner<S>>,
    process: Option<ProcessOutput>,
    receipt: Option<NativePairReceipt<S::Audit>>,
    receipt_artifact: Option<rz_experiments::ArtifactRef>,
    original_error: Option<String>,
    guarded_preflight_failures: Vec<String>,
    probe_evidence: Option<GuardedProbeReservation>,
    /// Set by the original process owner immediately after actual spawn and
    /// cleared only by the matching physical/no-GPU evidence verifier.
    spawn_custody: crate::process::SpawnCustody,
    physical_guard_required: bool,
    #[cfg(target_os = "linux")]
    v6_supervisor: Option<NativeV6Supervisor>,
}

/// Independent NoGame preparation cannot use NativeRunBundle's paired owner.
/// Keep the selected snapshot, execution and listeners together, and use the
/// same quarantine as the paired runner if a future supervised child remains
/// unresolved. Construction alone neither starts a child nor applies resources.
#[cfg(target_os = "linux")]
struct IndependentNoGameOwners {
    _snapshot: Arc<crate::pals_adaptive_launch::ProxyEntrySmokeSnapshotOwnerV1>,
    _execution: Arc<crate::pals_adaptive_launch::NoGameExecutionStartV1>,
    _sockets: Arc<crate::pals_adaptive_launch::NoGameRoleSocketsV1>,
    // The control itself owns these same original Arcs and selected proxy FD.
    _control: Arc<Mutex<crate::pals_adaptive_launch::SmokeDeadlineControl>>,
}

#[cfg(target_os = "linux")]
pub(crate) struct PreparedIndependentNoGameOwner {
    owners: Option<IndependentNoGameOwners>,
    process: Option<ProcessOutput>,
    spawn_custody: crate::process::SpawnCustody,
    // Vacuously true only while no child has been started. Any future spawn
    // adapter must clear this before dispatch and restore it only on the
    // original owner's verified physical fence.
    physical_fence_confirmed: bool,
    original_error: Option<String>,
}

#[cfg(target_os = "linux")]
impl PreparedIndependentNoGameOwner {
    /// All inputs are borrowed: on rejection the caller still owns the exact
    /// original Arcs. There is intentionally no process-dispatch method yet.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        snapshot: &Arc<crate::pals_adaptive_launch::ProxyEntrySmokeSnapshotOwnerV1>,
        execution: &Arc<crate::pals_adaptive_launch::NoGameExecutionStartV1>,
        sockets: &Arc<crate::pals_adaptive_launch::NoGameRoleSocketsV1>,
        proxy_pin: &Arc<std::fs::File>,
        owner_id: [u8; 16],
        owner_epoch: u64,
        request_ids: [u64; 2],
    ) -> Result<Self, ArenaError> {
        let control = crate::pals_adaptive_launch::SmokeDeadlineControl::new_no_game(
            snapshot,
            execution,
            sockets,
            proxy_pin,
            owner_id,
            owner_epoch,
            request_ids,
        )?;
        Ok(Self {
            owners: Some(IndependentNoGameOwners {
                _snapshot: Arc::clone(snapshot),
                _execution: Arc::clone(execution),
                _sockets: Arc::clone(sockets),
                _control: Arc::new(Mutex::new(control)),
            }),
            process: None,
            spawn_custody: crate::process::SpawnCustody::default(),
            physical_fence_confirmed: true,
            original_error: None,
        })
    }

    fn unresolved(&self) -> bool {
        !self.physical_fence_confirmed
            || self.spawn_custody.pending.load(Ordering::Acquire)
            || self.spawn_custody.child().is_some()
            || self.process.as_ref().is_some_and(|process| {
                process.receipt.group_cleanup != CleanupStatus::Gone
                    || process.pending_child.is_some()
            })
    }
}

#[cfg(target_os = "linux")]
impl Drop for PreparedIndependentNoGameOwner {
    fn drop(&mut self) {
        if !self.unresolved() {
            return;
        }
        let physical_lifetime_unknown = !self.physical_fence_confirmed
            || self.spawn_custody.pending.load(Ordering::Acquire)
            || self.spawn_custody.child().is_some();
        crate::native_launch::close_native_admission();
        let retained = Box::new(RetainedNativeScope {
            _owner: Box::new(
                self.owners
                    .take()
                    .expect("NoGame owns its original snapshot"),
            ),
            process: self.process.take(),
            spawned_child: self.spawn_custody.take_child(),
            last_spawn_pid: self.spawn_custody.last_spawn_pid,
            physical_lifetime_unknown,
            receipt: None,
            original_error: self.original_error.take(),
            guarded_preflight_failures: Vec::new(),
            interrupted_by_unwind: std::thread::panicking(),
            guarded_probe_scope: None,
            _v6_supervisor: None,
        });
        let mut slots = QUARANTINE.lock().unwrap_or_else(|error| error.into_inner());
        retain_in_slot(&mut slots, retained);
    }
}

/// Retained inside the same supervisor as the original NoGame control, also
/// when NativeRunBundle moves an unresolved owner into quarantine. These are
/// declared endpoint resources; no installation or readback is inferred.
#[cfg(target_os = "linux")]
struct NoGameEvidence {
    smoke: Arc<Mutex<crate::pals_adaptive_launch::ProxyEntryFunctionalSmokeOwnerV1>>,
    declared_resources: rz_experiments::PalsResourcePlanV5,
    transport: Option<crate::process::OriginalSmokeTransportObservation>,
}

/// A read-only projection of an already sent NoGame frame. It is neither a
/// proxy acknowledgement nor evidence that a process or device is finished.
#[cfg(target_os = "linux")]
#[derive(Serialize)]
struct NoGameFrameSnapshot {
    phase: &'static str,
    role: &'static str,
    request_id: u64,
    generation: u64,
    peer_pid: u32,
    peer_start_ticks: u64,
    original_monotonic_deadline: &'static str,
    completion_observed_during_cleanup: bool,
}

#[cfg(target_os = "linux")]
#[derive(Serialize)]
struct NoGameFrameProgressSnapshot {
    challenge_written: Option<usize>,
    request_read: Option<usize>,
    grant_written: Option<usize>,
    retained_grants: usize,
    first_failure: Option<String>,
}

#[cfg(target_os = "linux")]
#[derive(Serialize)]
struct NoGameObservationSnapshot<'a> {
    domain: &'static str,
    smoke_lock: Option<serde_json::Value>,
    smoke_lock_observation: &'static str,
    declared_endpoint_resources: &'a rz_experiments::PalsResourcePlanV5,
    resource_application: &'static str,
    resource_readback: &'static str,
    transport_attempt_started: Option<bool>,
    transport_attempt_returned: Option<bool>,
    transport_spawn_timestamp_semantics: &'static str,
    transport: Option<&'a crate::process::OriginalProcessObservation>,
    process: Option<&'a ProcessReceipt>,
    frame_observation: &'static str,
    frames: Option<Vec<NoGameFrameSnapshot>>,
    frame_progress: Option<NoGameFrameProgressSnapshot>,
    frame_progress_observation: &'static str,
    physical_completion: &'static str,
    eligible_for_paired_or_performance_acceptance: bool,
}

/// NoGame preparation reuses the original bundle's existing quarantine on
/// drop. It deliberately has no execution entrypoint or paired receipt.
#[cfg(target_os = "linux")]
pub(crate) struct PreparedNoGameSmokeOwner {
    bundle: NativeRunBundle<crate::pals_adaptive_launch::LockedPalsSystemArenaPairV6>,
    proxy: Option<Arc<std::fs::File>>,
    arguments: Vec<std::ffi::OsString>,
    rejected_control: Option<Box<dyn NativeV6Control>>,
}

/// Failure returns the same preparation owner; a future spawn must never be
/// hidden behind this declaration-only error path.
#[cfg(target_os = "linux")]
pub(crate) struct NoGameSmokePreparationFailure {
    pub(crate) error: ArenaError,
    pub(crate) original: PreparedNoGameSmokeOwner,
}

#[cfg(target_os = "linux")]
struct NoGameRetainedControl {
    execution: Arc<crate::pals_adaptive_launch::PalsV6ExecutionStart>,
    sockets: Arc<crate::pals_adaptive_launch::PalsV6RoleSockets>,
    control: Arc<Mutex<crate::pals_adaptive_launch::SmokeDeadlineControl>>,
    /// Frame-delivery observations stay with this owner; they do not prove
    /// process exit, acknowledgement, or device completion.
    grants: [Option<crate::pals_adaptive_launch::SmokeGrantObservation>; 2],
}

#[cfg(target_os = "linux")]
impl NoGameRetainedControl {
    fn poll_smoke(
        &mut self,
        custody: &crate::process::SpawnCustody,
        until: std::time::Instant,
        cleanup: bool,
    ) -> Result<(), ArenaError> {
        let observed = {
            let mut control = match self.control.try_lock() {
                Ok(control) => control,
                // The same control and every partial frame remain owned; the
                // bounded supervisor can retry on a later tick.
                Err(TryLockError::WouldBlock) => return Ok(()),
                Err(TryLockError::Poisoned(_)) => {
                    return Err(ArenaError::Io(
                        "NoGame original control owner poisoned".into(),
                    ));
                }
            };
            if cleanup {
                control.poll_cleanup(custody, until)?
            } else {
                control.poll_run(custody, until)?
            }
        };
        if let Some(observation) = observed {
            let index = match observation.scope.phase {
                rz_contracts::pals::proxy_entry_smoke_grant_v1::ProxyEntrySmokePhase::Run => 0,
                rz_contracts::pals::proxy_entry_smoke_grant_v1::ProxyEntrySmokePhase::Cleanup => 1,
            };
            if self.grants[index].is_some() {
                return Err(ArenaError::Integrity(
                    "duplicate NoGame grant observation".into(),
                ));
            }
            self.grants[index] = Some(observation);
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
impl NativeV6Control for NoGameRetainedControl {
    fn original_execution(
        &self,
    ) -> Option<&Arc<crate::pals_adaptive_launch::PalsV6ExecutionStart>> {
        Some(&self.execution)
    }

    fn original_role_sockets(
        &self,
    ) -> Option<&Arc<crate::pals_adaptive_launch::PalsV6RoleSockets>> {
        Some(&self.sockets)
    }

    fn no_game_grants(
        &self,
    ) -> Option<&[Option<crate::pals_adaptive_launch::SmokeGrantObservation>; 2]> {
        Some(&self.grants)
    }

    fn no_game_progress(&self) -> NoGameProgressReadback {
        match self.control.try_lock() {
            Ok(control) => NoGameProgressReadback::Observed(control.progress()),
            Err(TryLockError::WouldBlock) => NoGameProgressReadback::Busy,
            Err(TryLockError::Poisoned(_)) => NoGameProgressReadback::Poisoned,
        }
    }

    fn poll_run(
        &mut self,
        custody: &crate::process::SpawnCustody,
        until: std::time::Instant,
    ) -> Result<(), ArenaError> {
        self.poll_smoke(custody, until, false)
    }

    fn poll_cleanup(
        &mut self,
        custody: &crate::process::SpawnCustody,
        until: std::time::Instant,
    ) -> Result<(), ArenaError> {
        self.poll_smoke(custody, until, true)
    }
}

#[cfg(target_os = "linux")]
impl PreparedNoGameSmokeOwner {
    /// Prepare without spawning. The exact owner, live control and any later
    /// unresolved custody/output use NativeRunBundle's existing retention path.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn assemble(
        owner: NativeLaunchOwner<crate::pals_adaptive_launch::LockedPalsSystemArenaPairV6>,
        smoke: &Arc<Mutex<crate::pals_adaptive_launch::ProxyEntryFunctionalSmokeOwnerV1>>,
        execution: &Arc<crate::pals_adaptive_launch::PalsV6ExecutionStart>,
        sockets: &Arc<crate::pals_adaptive_launch::PalsV6RoleSockets>,
        role: rz_experiments::NativeEngineRole,
        owner_id: [u8; 16],
        owner_epoch: u64,
        request_ids: [u64; 2],
    ) -> Result<Self, Box<NoGameSmokePreparationFailure>> {
        let mut original = Self {
            bundle: NativeRunBundle {
                owner: Some(owner),
                process: None,
                receipt: None,
                receipt_artifact: None,
                original_error: None,
                guarded_preflight_failures: Vec::new(),
                probe_evidence: None,
                spawn_custody: crate::process::SpawnCustody::default(),
                physical_guard_required: true,
                v6_supervisor: None,
            },
            proxy: None,
            arguments: Vec::new(),
            rejected_control: None,
        };
        let prepared = {
            let source = original
                .bundle
                .owner
                .as_ref()
                .expect("fresh original owner");
            crate::pals_adaptive_launch::DirectSmokeBinding::bind(
                source,
                smoke,
                execution,
                sockets,
                &mut original.bundle.spawn_custody,
                role,
                owner_id,
                owner_epoch,
                request_ids,
            )
            .map(|binding| {
                (
                    Arc::clone(binding.proxy()),
                    Arc::clone(binding.control()),
                    binding.common_arguments().to_vec(),
                )
            })
        };
        let (proxy, control, arguments) = match prepared {
            Ok(value) => value,
            Err(error) => return Err(Box::new(NoGameSmokePreparationFailure { error, original })),
        };
        original.proxy = Some(proxy);
        original.arguments = arguments;
        let control: Box<dyn NativeV6Control> = Box::new(NoGameRetainedControl {
            execution: Arc::clone(execution),
            sockets: Arc::clone(sockets),
            control,
            grants: [None, None],
        });
        match NativeV6Supervisor::new(execution, control) {
            Ok(mut supervisor) => {
                let declared_resources = original
                    .bundle
                    .owner
                    .as_ref()
                    .expect("fresh original owner")
                    .spec()
                    .endpoint(role)
                    .startup_lock
                    .config
                    .endpoint
                    .resources
                    .clone();
                supervisor.no_game_evidence = Some(NoGameEvidence {
                    smoke: Arc::clone(smoke),
                    declared_resources,
                    transport: None,
                });
                original.bundle.v6_supervisor = Some(supervisor);
            }
            Err(rejection) => {
                original.rejected_control = Some(rejection.control);
                return Err(Box::new(NoGameSmokePreparationFailure {
                    error: rejection.error,
                    original,
                }));
            }
        }
        Ok(original)
    }

    /// Reserve the transport observation in the same retained supervisor as
    /// the child owner. The input uses the original execution's exact window;
    /// reserving it neither applies resources nor starts a process.
    pub(crate) fn reserve_transport<'a>(
        &mut self,
        bytes: &'a [u8],
    ) -> Result<crate::process::OriginalProcessInput<'a>, ArenaError> {
        let supervisor =
            self.bundle.v6_supervisor.as_mut().ok_or_else(|| {
                ArenaError::Invalid("NoGame original supervisor unavailable".into())
            })?;
        supervisor.execution.require_running()?;
        let window = crate::process::OriginalProcessWindow::new(
            supervisor.execution.started(),
            supervisor.execution.run_until(),
            supervisor.execution.cleanup_until(),
        )?;
        let evidence = supervisor.no_game_evidence.as_mut().ok_or_else(|| {
            ArenaError::Invalid("NoGame original evidence owner unavailable".into())
        })?;
        if evidence.transport.is_some() {
            return Err(ArenaError::Invalid(
                "NoGame transport already reserved".into(),
            ));
        }
        let input = crate::process::OriginalProcessInput { bytes, window };
        evidence.transport = Some(crate::process::OriginalSmokeTransportObservation::new(
            input,
        )?);
        Ok(input)
    }

    /// A later direct supervisor may update only this already reserved
    /// observation; unresolved Drop retains it with the original supervisor.
    pub(crate) fn transport_observation_mut(
        &mut self,
    ) -> Option<&mut crate::process::OriginalSmokeTransportObservation> {
        self.bundle
            .v6_supervisor
            .as_mut()?
            .no_game_evidence
            .as_mut()?
            .transport
            .as_mut()
    }

    /// Transfer an observed output into the existing original bundle. This
    /// local PID check is not proof of the output's origin; a future execution
    /// path must keep the adapter return and owner in the same call. An error
    /// returns the same output, including any pending child handle.
    fn retain_process_output(
        &mut self,
        output: ProcessOutput,
    ) -> Result<(), (ArenaError, ProcessOutput)> {
        if self.bundle.process.is_some()
            || self.bundle.spawn_custody.last_spawn_pid != Some(output.receipt.pid)
        {
            return Err((
                ArenaError::Integrity("NoGame output does not match original custody".into()),
                output,
            ));
        }
        self.bundle.process = Some(output);
        Ok(())
    }

    /// Borrow raw output while the same owner remains live. A borrow or a
    /// process receipt does not establish durable recovery or physical finish.
    pub(crate) fn retained_process_output(&self) -> Option<&ProcessOutput> {
        self.bundle.process.as_ref()
    }

    /// Produce a bounded in-memory observation snapshot. A busy observation
    /// lock is reported as unavailable and never blocks unrelated execution.
    /// Raw output remains in the original bundle; this is not durable storage.
    pub(crate) fn observation_json(&self) -> Result<Vec<u8>, ArenaError> {
        let supervisor =
            self.bundle.v6_supervisor.as_ref().ok_or_else(|| {
                ArenaError::Invalid("NoGame original supervisor unavailable".into())
            })?;
        let evidence = supervisor.no_game_evidence.as_ref().ok_or_else(|| {
            ArenaError::Invalid("NoGame original evidence owner unavailable".into())
        })?;
        let (smoke_lock, smoke_lock_observation) = match evidence.smoke.try_lock() {
            // The same Mutex can contain a replacement object. Until the
            // control's original identity is checked, publish no lock value.
            Ok(_) => (None, "unknown_original_identity"),
            Err(TryLockError::WouldBlock) => (None, "unavailable_busy"),
            Err(TryLockError::Poisoned(_)) => (None, "unavailable_poisoned"),
        };
        let (frames, frame_progress, frame_observation, frame_progress_observation) =
            match supervisor.control.try_lock() {
                Ok(control) => match control.no_game_grants() {
                    Some(grants) => {
                        let (progress, progress_status) = match control.no_game_progress() {
                            NoGameProgressReadback::Observed(value) => (
                                Some(NoGameFrameProgressSnapshot {
                                    challenge_written: value.challenge_written,
                                    request_read: value.request_read,
                                    grant_written: value.grant_written,
                                    retained_grants: value.retained_grants,
                                    first_failure: value.first_failure,
                                }),
                                "original_control_observed",
                            ),
                            NoGameProgressReadback::Busy => (None, "unavailable_busy"),
                            NoGameProgressReadback::Poisoned => (None, "unavailable_poisoned"),
                            NoGameProgressReadback::Unavailable => (None, "unavailable"),
                        };
                        (
                    Some(grants.iter().flatten().map(|grant| NoGameFrameSnapshot {
                        phase: match grant.scope.phase {
                            rz_contracts::pals::proxy_entry_smoke_grant_v1::ProxyEntrySmokePhase::Run => "run",
                            rz_contracts::pals::proxy_entry_smoke_grant_v1::ProxyEntrySmokePhase::Cleanup => "cleanup",
                        },
                        role: match grant.scope.role {
                            rz_contracts::pals::deadline_grant_v1::DeadlineRole::Baseline => "baseline",
                            rz_contracts::pals::deadline_grant_v1::DeadlineRole::Candidate => "candidate",
                        },
                        request_id: grant.scope.request_id,
                        generation: grant.scope.generation,
                        peer_pid: grant.peer_pid,
                        peer_start_ticks: grant.peer_start_ticks,
                        original_monotonic_deadline: "retained_in_original_control",
                        completion_observed_during_cleanup: grant.completion_observed_during_cleanup,
                    }).collect()),
                    progress,
                    "original_owner_borrowed",
                    progress_status,
                )
                    }
                    None => (None, None, "unavailable", "unavailable"),
                },
                Err(TryLockError::WouldBlock) => {
                    (None, None, "unavailable_busy", "unavailable_busy")
                }
                Err(TryLockError::Poisoned(_)) => {
                    (None, None, "unavailable_poisoned", "unavailable_poisoned")
                }
            };
        let transport = evidence.transport.as_ref();
        let snapshot = NoGameObservationSnapshot {
            domain: "rz-pals-nogame-original-observation/1",
            smoke_lock,
            smoke_lock_observation,
            declared_endpoint_resources: &evidence.declared_resources,
            resource_application: "unknown",
            resource_readback: "unknown",
            transport_attempt_started: transport.map(|v| v.attempt_started()),
            transport_attempt_returned: transport.map(|v| v.attempt_returned()),
            transport_spawn_timestamp_semantics: "supervisor_pre_spawn_not_os_spawn",
            transport: transport.map(|v| v.transport()),
            process: self.bundle.process.as_ref().map(|v| &v.receipt),
            frame_observation,
            frames,
            frame_progress,
            frame_progress_observation,
            physical_completion: "unknown",
            eligible_for_paired_or_performance_acceptance: false,
        };
        serde_json::to_vec(&snapshot)
            .map_err(|_| ArenaError::Io("NoGame observation serialization failed".into()))
    }
}

/// Original probe output retained after a storage failure.  Taking this value
/// transfers evidence custody to the caller; it is not a GPU reservation.
pub struct GuardedPreflightEvidence {
    pub run_id: u64,
    pub input_sha256: String,
    pub role: &'static str,
    pub probe: &'static str,
    pub declared_output_limit_bytes: u64,
    pub observed_retained_output_bytes: u64,
    pub interrupted_by_unwind: bool,
    pub original_error: Option<String>,
    pub failures: Vec<String>,
    pub process: ProcessOutput,
    /// Original generic pair receipt, if one was constructed before failure.
    pub pair_receipt: Option<Box<dyn std::any::Any + Send>>,
}
impl fmt::Debug for GuardedPreflightEvidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GuardedPreflightEvidence")
            .field("run_id", &self.run_id)
            .field("role", &self.role)
            .field("probe", &self.probe)
            .field("pair_receipt_retained", &self.pair_receipt.is_some())
            .finish_non_exhaustive()
    }
}

enum GuardedEvidenceSlot {
    Retained(GuardedPreflightEvidence),
}
struct GuardedEvidenceEntry {
    run_id: u64,
    role: &'static str,
    probe: &'static str,
    retained_payload_bytes: u64,
    slot: GuardedEvidenceSlot,
}
/// Bounded process-local evidence capacity; this is not a GPU or VRAM limit.
const GUARDED_EVIDENCE_CAPACITY: usize = 16;
const GUARDED_EVIDENCE_PAYLOAD_LIMIT_BYTES: u64 = 512 * 1024 * 1024;
static GUARDED_EVIDENCE_SLOTS: Mutex<Vec<GuardedEvidenceEntry>> = Mutex::new(Vec::new());
static GUARDED_EVIDENCE_NEXT_RUN: AtomicU64 = AtomicU64::new(1);
static GUARDED_EVIDENCE_FALLBACK_LOST: AtomicU64 = AtomicU64::new(0);
static GUARDED_EVIDENCE_LOSSES: Mutex<[Option<GuardedEvidenceLoss>; GUARDED_EVIDENCE_CAPACITY]> =
    Mutex::new([const { None }; GUARDED_EVIDENCE_CAPACITY]);

#[derive(Clone, Debug, Serialize)]
pub struct GuardedEvidenceLoss {
    pub run_id: u64,
    pub input_sha256: String,
    pub role: &'static str,
    pub probe: &'static str,
    pub observed_retained_output_bytes: u64,
    pub reason: GuardedEvidenceFallbackError,
    pub interrupted_by_unwind: bool,
}

/// Recent bounded observations. The separate lost counter includes entries
/// that have rolled off and is unrelated to GPU or storage admission.
pub fn guarded_evidence_recent_losses() -> Vec<GuardedEvidenceLoss> {
    GUARDED_EVIDENCE_LOSSES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .flatten()
        .cloned()
        .collect()
}

fn retain_guarded_evidence(
    evidence: GuardedPreflightEvidence,
) -> Result<(), (GuardedEvidenceFallbackError, GuardedPreflightEvidence)> {
    let mut slots = GUARDED_EVIDENCE_SLOTS
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if slots.iter().any(|e| {
        e.run_id == evidence.run_id && e.role == evidence.role && e.probe == evidence.probe
    }) {
        return Err((GuardedEvidenceFallbackError::ScopeInUse, evidence));
    }
    if slots.len() >= GUARDED_EVIDENCE_CAPACITY {
        return Err((GuardedEvidenceFallbackError::TotalSlotsExhausted, evidence));
    }
    let total = slots
        .iter()
        .try_fold(0u64, |sum, e| sum.checked_add(e.retained_payload_bytes))
        .and_then(|sum| sum.checked_add(evidence.observed_retained_output_bytes));
    if total.is_none_or(|sum| sum > GUARDED_EVIDENCE_PAYLOAD_LIMIT_BYTES) {
        return Err((
            GuardedEvidenceFallbackError::TotalPayloadCapacityExhausted,
            evidence,
        ));
    }
    slots.push(GuardedEvidenceEntry {
        run_id: evidence.run_id,
        role: evidence.role,
        probe: evidence.probe,
        retained_payload_bytes: evidence.observed_retained_output_bytes,
        slot: GuardedEvidenceSlot::Retained(evidence),
    });
    Ok(())
}

/// Count of dropped fallback evidence when callers did not take it and the
/// bounded in-memory fallback could not accept it; not a GPU failure count.
pub fn guarded_evidence_fallback_lost_count() -> u64 {
    GUARDED_EVIDENCE_FALLBACK_LOST.load(Ordering::Relaxed)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardedEvidenceFallbackError {
    ScopeInUse,
    TotalSlotsExhausted,
    TotalPayloadCapacityExhausted,
}
impl fmt::Display for GuardedEvidenceFallbackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ScopeInUse => "guarded_evidence_scope_in_use",
            Self::TotalSlotsExhausted => "guarded_evidence_total_slots_exhausted",
            Self::TotalPayloadCapacityExhausted => {
                "guarded_evidence_total_payload_capacity_exhausted"
            }
        })
    }
}

struct GuardedProbeReservation {
    run_id: u64,
    input_sha256: String,
    role: &'static str,
    probe: &'static str,
    output_limit_bytes: u64,
    persistence_failed: bool,
    process_available: bool,
}

impl GuardedProbeReservation {
    fn acquire(
        run_id: u64,
        input_sha256: String,
        role: &'static str,
        probe: &'static str,
        output_limit_bytes: u64,
    ) -> Self {
        Self {
            run_id,
            input_sha256,
            role,
            probe,
            output_limit_bytes,
            // Until every write succeeds, unwind must preserve the original.
            persistence_failed: true,
            process_available: false,
        }
    }
}

/// Returns owned output for caller-directed recovery. A reservation in use is
/// not disturbed. No process or GPU resource owner is included.
pub fn take_guarded_preflight_evidence(
    run_id: u64,
    role: &str,
    probe: &str,
) -> Option<GuardedPreflightEvidence> {
    let mut slots = GUARDED_EVIDENCE_SLOTS
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let index = slots.iter().position(|e| {
        e.run_id == run_id
            && e.role == role
            && e.probe == probe
            && matches!(e.slot, GuardedEvidenceSlot::Retained(_))
    })?;
    let GuardedEvidenceSlot::Retained(evidence) = slots.remove(index).slot;
    Some(evidence)
}

#[derive(Debug, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum GuardedPreflightEvidenceStatus {
    Retained {
        run_id: u64,
        input_sha256: String,
        role: &'static str,
        probe: &'static str,
        declared_output_limit_bytes: u64,
        observed_retained_output_bytes: u64,
        interrupted_by_unwind: bool,
        failures: Vec<String>,
    },
}

/// Payload bytes are observed retained stream lengths, not process RSS.
pub fn guarded_preflight_evidence_status() -> (usize, u64, Vec<GuardedPreflightEvidenceStatus>) {
    let slots = GUARDED_EVIDENCE_SLOTS
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let statuses = slots
        .iter()
        .map(|entry| match &entry.slot {
            GuardedEvidenceSlot::Retained(e) => GuardedPreflightEvidenceStatus::Retained {
                run_id: e.run_id,
                input_sha256: e.input_sha256.clone(),
                role: e.role,
                probe: e.probe,
                declared_output_limit_bytes: e.declared_output_limit_bytes,
                observed_retained_output_bytes: e.observed_retained_output_bytes,
                interrupted_by_unwind: e.interrupted_by_unwind,
                failures: e.failures.clone(),
            },
        })
        .collect();
    (
        GUARDED_EVIDENCE_CAPACITY,
        GUARDED_EVIDENCE_PAYLOAD_LIMIT_BYTES,
        statuses,
    )
}

impl<S: NativeProviderDeclaration> NativeRunBundle<S> {
    fn unresolved(&self) -> bool {
        (self.physical_guard_required
            && (self.spawn_custody.pending.load(Ordering::Acquire)
                || self.spawn_custody.child().is_some()))
            || self.process.as_ref().is_some_and(|p| {
                p.receipt.group_cleanup != CleanupStatus::Gone || p.pending_child.is_some()
            })
    }
    fn take_evidence(&mut self) -> Option<GuardedPreflightEvidence> {
        if self.unresolved()
            || !self
                .probe_evidence
                .as_ref()
                .is_some_and(|r| r.persistence_failed && r.process_available)
        {
            return None;
        }
        let reservation = self.probe_evidence.take()?;
        let process = self.process.take()?;
        let observed = u64::try_from(process.stdout.len().saturating_add(process.stderr.len()))
            .unwrap_or(u64::MAX);
        Some(GuardedPreflightEvidence {
            run_id: reservation.run_id,
            input_sha256: reservation.input_sha256,
            role: reservation.role,
            probe: reservation.probe,
            declared_output_limit_bytes: reservation.output_limit_bytes,
            observed_retained_output_bytes: observed,
            interrupted_by_unwind: std::thread::panicking(),
            original_error: self.original_error.take(),
            failures: std::mem::take(&mut self.guarded_preflight_failures),
            process,
            pair_receipt: self
                .receipt
                .take()
                .map(|r| Box::new(r) as Box<dyn std::any::Any + Send>),
        })
    }
}

#[cfg(test)]
mod physical_guard_tests {
    use super::*;

    #[test]
    fn guarded_spawn_without_process_output_stays_unresolved() {
        // No owner is constructed in this state-only test. Clear the pending
        // flag before Drop so the normal quarantine invariant is untouched.
        let bundle = NativeRunBundle::<rz_experiments::LockedIntegrationPairSpecV1> {
            owner: None,
            process: None,
            receipt: None,
            receipt_artifact: None,
            original_error: None,
            guarded_preflight_failures: Vec::new(),
            probe_evidence: None,
            spawn_custody: crate::process::SpawnCustody::default(),
            physical_guard_required: true,
            #[cfg(target_os = "linux")]
            v6_supervisor: None,
        };
        assert!(!bundle.unresolved()); // no child was spawned
        bundle.spawn_custody.pending.store(true, Ordering::Release);
        assert!(bundle.unresolved()); // spawn returned, output still absent
        bundle.spawn_custody.pending.store(false, Ordering::Release);
        assert!(!bundle.unresolved()); // caller recorded independent proof
    }

    #[test]
    fn historical_unprotected_provider_does_not_inherit_new_gate() {
        let bundle = NativeRunBundle::<rz_experiments::LockedIntegrationPairSpecV1> {
            owner: None,
            process: None,
            receipt: None,
            receipt_artifact: None,
            original_error: None,
            guarded_preflight_failures: Vec::new(),
            probe_evidence: None,
            spawn_custody: crate::process::SpawnCustody::default(),
            physical_guard_required: false,
            #[cfg(target_os = "linux")]
            v6_supervisor: None,
        };
        bundle.spawn_custody.pending.store(true, Ordering::Release);
        assert!(!bundle.unresolved());
    }

    #[test]
    fn previous_probe_is_not_relabelled_as_current_even_with_reused_pid() {
        assert!(guarded_process_is_previous(true, true, Some(41), Some(41)));
        assert!(guarded_process_is_previous(true, false, Some(42), Some(41)));
        assert!(!guarded_process_is_previous(
            true,
            false,
            Some(42),
            Some(42)
        ));
        assert!(!guarded_process_is_previous(
            false,
            true,
            Some(41),
            Some(41)
        ));
    }

    #[test]
    fn failed_probe_storage_retains_original_streams_without_gpu_quarantine() {
        let mut reservation = GuardedProbeReservation::acquire(
            u64::MAX,
            "input".into(),
            "candidate",
            "readiness",
            256 * 1024,
        );
        let independent = GuardedProbeReservation::acquire(
            u64::MAX - 1,
            "other".into(),
            "baseline",
            "identification",
            256 * 1024,
        );
        assert_eq!(independent.output_limit_bytes, 256 * 1024);
        let large = GuardedProbeReservation::acquire(
            u64::MAX - 2,
            "other".into(),
            "pair",
            "runner",
            GUARDED_EVIDENCE_PAYLOAD_LIMIT_BYTES + 1,
        );
        assert!(large.output_limit_bytes > GUARDED_EVIDENCE_PAYLOAD_LIMIT_BYTES);
        reservation.persistence_failed = true;
        reservation.process_available = true;
        let process = ProcessOutput {
            receipt: crate::ProcessReceipt {
                supervisor_version: 1,
                pid: 17,
                elapsed_ns: 0,
                cleanup_elapsed_ns: None,
                cleanup_timing_unavailable_reason: None,
                stop: crate::ProcessStop::Exited,
                exit_code: Some(0),
                exit_signal: None,
                group_cleanup: CleanupStatus::Gone,
                stdout_bytes: 3,
                stderr_bytes: 2,
                observed_output_bytes: 5,
                descendant_cleanup_required: false,
                child_limit_enforcement: "test".into(),
                watched_artifact_bytes: None,
                artifact_limit_enforcement: None,
                errors: Vec::new(),
            },
            stdout: b"out".to_vec(),
            stderr: b"er".to_vec(),
            pending_child: None,
        };
        let bundle = NativeRunBundle::<rz_experiments::LockedIntegrationPairSpecV1> {
            owner: None,
            process: Some(process),
            receipt: None,
            receipt_artifact: None,
            original_error: None,
            guarded_preflight_failures: vec!["storage failed".into()],
            probe_evidence: Some(reservation),
            spawn_custody: crate::process::SpawnCustody::default(),
            physical_guard_required: true,
            #[cfg(target_os = "linux")]
            v6_supervisor: None,
        };
        assert!(!bundle.unresolved());
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _bundle = bundle;
            panic!("simulated guarded probe persistence panic");
        }));
        assert!(unwind.is_err());
        let evidence = take_guarded_preflight_evidence(u64::MAX, "candidate", "readiness")
            .expect("raw evidence retained");
        assert_eq!(evidence.process.stdout, b"out");
        assert_eq!(evidence.process.stderr, b"er");
        assert_eq!(evidence.observed_retained_output_bytes, 5);
        assert!(evidence.interrupted_by_unwind);
        assert_eq!(evidence.failures, ["storage failed"]);
        assert!(!guarded_preflight_evidence_status().2.iter().any(|e| matches!(e, GuardedPreflightEvidenceStatus::Retained { run_id, .. } if *run_id == u64::MAX)));
        let mut next = GuardedProbeReservation::acquire(
            u64::MAX - 3,
            "input".into(),
            "pair",
            "runner",
            GUARDED_EVIDENCE_PAYLOAD_LIMIT_BYTES + 1,
        );
        next.process_available = true;
        let mut bundle = NativeRunBundle::<rz_experiments::LockedIntegrationPairSpecV1> {
            owner: None,
            process: Some(evidence.process),
            receipt: None,
            receipt_artifact: None,
            original_error: Some("save failed".into()),
            guarded_preflight_failures: Vec::new(),
            probe_evidence: Some(next),
            spawn_custody: crate::process::SpawnCustody::default(),
            physical_guard_required: true,
            #[cfg(target_os = "linux")]
            v6_supervisor: None,
        };
        let caller_owned = bundle
            .take_evidence()
            .expect("caller can recover without fallback capacity");
        assert_eq!(caller_owned.process.stdout, b"out");
        assert_eq!(caller_owned.original_error.as_deref(), Some("save failed"));

        let mut pending = GuardedProbeReservation::acquire(
            u64::MAX - 4,
            "input".into(),
            "pair",
            "runner",
            256 * 1024,
        );
        pending.process_available = true;
        let mut failure = NativePairFailure::<rz_experiments::LockedIntegrationPairSpecV1> {
            cause: ArenaError::Io("save failed".into()),
            receipt: None,
            receipt_artifact: None,
            bundle: NativeRunBundle {
                owner: None,
                process: Some(caller_owned.process),
                receipt: None,
                receipt_artifact: None,
                original_error: Some("save failed".into()),
                guarded_preflight_failures: Vec::new(),
                probe_evidence: Some(pending),
                spawn_custody: crate::process::SpawnCustody::default(),
                physical_guard_required: true,
                #[cfg(target_os = "linux")]
                v6_supervisor: None,
            },
        };
        let refused = failure.with_guarded_evidence(|view| {
            assert_eq!(view.process.stdout, b"out");
            Err::<(), _>("storage unavailable")
        });
        assert_eq!(refused, Err("storage unavailable"));
        assert!(failure.process().is_some());
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = failure.with_guarded_evidence::<(), ()>(|_| panic!("save interrupted"));
        }));
        assert!(panic.is_err());
        assert!(
            failure
                .bundle
                .probe_evidence
                .as_ref()
                .unwrap()
                .persistence_failed
        );
        failure
            .bundle
            .spawn_custody
            .pending
            .store(true, Ordering::Release);
        let saved = failure.with_guarded_evidence(|view| {
            assert!(view.physical_or_process_completion_unknown);
            Ok::<_, ()>(view.process.stdout.len())
        });
        assert_eq!(saved, Ok(Some(3)));
        assert!(failure.bundle.unresolved());
        assert!(failure.take_guarded_evidence().is_none());
        // This state-only test did not spawn a child or construct an owner.
        failure
            .bundle
            .spawn_custody
            .pending
            .store(false, Ordering::Release);
        let evidence = failure
            .take_guarded_evidence()
            .expect("callback success did not release original evidence");
        assert_eq!(evidence.process.stdout, b"out");
    }
}
impl<S: NativeProviderDeclaration> Drop for NativeRunBundle<S> {
    fn drop(&mut self) {
        // A successful physical fence does not imply the raw output was saved.
        // Retain only the evidence here, so its recovery never owns a GPU or
        // keeps the native admission lease alive.
        if !self.unresolved() {
            if let Some(evidence) = self.take_evidence() {
                if let Err((error, evidence)) = retain_guarded_evidence(evidence) {
                    let count = GUARDED_EVIDENCE_FALLBACK_LOST.fetch_add(1, Ordering::Relaxed);
                    let mut losses = GUARDED_EVIDENCE_LOSSES
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    let index = (count % GUARDED_EVIDENCE_CAPACITY as u64) as usize;
                    losses[index] = Some(GuardedEvidenceLoss {
                        run_id: evidence.run_id,
                        input_sha256: evidence.input_sha256,
                        role: evidence.role,
                        probe: evidence.probe,
                        observed_retained_output_bytes: evidence.observed_retained_output_bytes,
                        reason: error,
                        interrupted_by_unwind: evidence.interrupted_by_unwind,
                    });
                    drop(losses);
                }
            }
            return;
        }
        let physical_lifetime_unknown = self.physical_guard_required
            && (self.spawn_custody.pending.load(Ordering::Acquire)
                || self.spawn_custody.child().is_some());
        // A reaped leader and a gone process group do not prove that no child
        // escaped into a new session. Keep the whole resource protected until
        // a supervisor-owned, independently verified tree fence exists.
        crate::native_launch::close_native_admission();
        let retained = Box::new(RetainedNativeScope {
            _owner: Box::new(self.owner.take().expect("native bundle owns its lease")),
            process: self.process.take(),
            spawned_child: self.spawn_custody.take_child(),
            last_spawn_pid: self.spawn_custody.last_spawn_pid,
            physical_lifetime_unknown,
            receipt: self.receipt.take().map(|r| Box::new(r) as Box<dyn Send>),
            original_error: self.original_error.take(),
            guarded_preflight_failures: std::mem::take(&mut self.guarded_preflight_failures),
            interrupted_by_unwind: std::thread::panicking(),
            guarded_probe_scope: self.probe_evidence.as_ref().map(|p| GuardedProbeScope {
                run_id: p.run_id,
                input_sha256: p.input_sha256.clone(),
                role: p.role,
                probe: p.probe,
            }),
            #[cfg(target_os = "linux")]
            _v6_supervisor: self.v6_supervisor.take(),
        });
        // The mutex protects only a short vector append; transient contention must not
        // turn an otherwise recoverable owner into an inaccessible leak.
        let mut slots = QUARANTINE.lock().unwrap_or_else(|e| e.into_inner());
        retain_in_slot(&mut slots, retained);
    }
}
struct RetainedNativeScope {
    _owner: Box<dyn Send>,
    process: Option<ProcessOutput>,
    spawned_child: Option<std::process::Child>,
    last_spawn_pid: Option<u32>,
    physical_lifetime_unknown: bool,
    receipt: Option<Box<dyn Send>>,
    original_error: Option<String>,
    guarded_preflight_failures: Vec<String>,
    interrupted_by_unwind: bool,
    guarded_probe_scope: Option<GuardedProbeScope>,
    #[cfg(target_os = "linux")]
    _v6_supervisor: Option<NativeV6Supervisor>,
}

#[derive(Clone, Debug, Serialize)]
pub struct GuardedProbeScope {
    pub run_id: u64,
    pub input_sha256: String,
    pub role: &'static str,
    pub probe: &'static str,
}
static QUARANTINE: Mutex<Vec<Box<RetainedNativeScope>>> = Mutex::new(Vec::new());
static QUARANTINE_FALLBACK: AtomicBool = AtomicBool::new(false);
fn retain_in_slot(slots: &mut Vec<Box<RetainedNativeScope>>, retained: Box<RetainedNativeScope>) {
    if slots.try_reserve(1).is_ok() {
        slots.push(retained);
    } else {
        // Drop cannot return an allocation error. Preserve the complete owner
        // rather than dropping live inputs; the fallback is observable.
        QUARANTINE_FALLBACK.store(true, Ordering::Release);
        std::mem::forget(retained);
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeQuarantineStatus {
    pub admission_closed: bool,
    pub slot_observed: bool,
    pub slot_busy: bool,
    pub bounded_fallback_retained: bool,
    /// At least one owner remains preserved without an enumerable handle; its
    /// identity and the number of such owners are unknown.
    pub unobservable_owner_fallback: bool,
    /// All scopes stay owned; the legacy detail fields describe the first.
    pub retained_scope_count: usize,
    pub process_id: Option<u32>,
    pub pending_leader_handle: Option<bool>,
    pub receipt_retained: bool,
    pub original_error_retained: bool,
    pub original_error: Option<String>,
    /// Separate from process-group state; absent for historical providers and
    /// never a claim that their physical GPU completion was observed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guarded_physical_lifetime_unknown: Option<bool>,
    /// An observed PID is not authority to signal or reap by number.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guarded_last_spawn_pid: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guarded_original_spawn_handle_retained: Option<bool>,
    /// Prior probe evidence is not attributed to a later spawned child.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guarded_previous_probe_pid: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub guarded_preflight_failures: Vec<String>,
    /// The child started but its local pipe buffers did not become a complete
    /// ProcessOutput. The original child handle may still be retained.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guarded_spawn_raw_evidence_incomplete: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guarded_probe_scope: Option<GuardedProbeScope>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guarded_interrupted_by_unwind: Option<bool>,
}

fn guarded_process_is_previous(
    physical_unknown: bool,
    current_child_held: bool,
    current_pid: Option<u32>,
    process_pid: Option<u32>,
) -> bool {
    physical_unknown && process_pid.is_some() && (current_child_held || process_pid != current_pid)
}
fn empty_quarantine_status() -> NativeQuarantineStatus {
    NativeQuarantineStatus {
        admission_closed: crate::native_admission_closed(),
        slot_observed: false,
        slot_busy: false,
        bounded_fallback_retained: QUARANTINE_FALLBACK.load(Ordering::Acquire),
        unobservable_owner_fallback: QUARANTINE_FALLBACK.load(Ordering::Acquire),
        retained_scope_count: 0,
        process_id: None,
        pending_leader_handle: None,
        receipt_retained: false,
        original_error_retained: false,
        original_error: None,
        guarded_physical_lifetime_unknown: None,
        guarded_last_spawn_pid: None,
        guarded_original_spawn_handle_retained: None,
        guarded_previous_probe_pid: None,
        guarded_preflight_failures: Vec::new(),
        guarded_spawn_raw_evidence_incomplete: None,
        guarded_probe_scope: None,
        guarded_interrupted_by_unwind: None,
    }
}

fn observe_quarantine_scope(
    slot: Option<&RetainedNativeScope>,
    status: &mut NativeQuarantineStatus,
) {
    status.slot_observed = true;
    if let Some(scope) = slot {
        let current_pid = scope
            .spawned_child
            .as_ref()
            .map(std::process::Child::id)
            .or(scope.last_spawn_pid);
        let previous_process = guarded_process_is_previous(
            scope.physical_lifetime_unknown,
            scope.spawned_child.is_some(),
            current_pid,
            scope.process.as_ref().map(|process| process.receipt.pid),
        );
        if previous_process {
            status.guarded_previous_probe_pid =
                scope.process.as_ref().map(|process| process.receipt.pid);
        } else {
            status.process_id = scope.process.as_ref().map(|process| process.receipt.pid);
            status.pending_leader_handle = scope
                .process
                .as_ref()
                .map(|process| process.pending_child.is_some());
        }
        status.receipt_retained = scope.receipt.is_some();
        status.original_error_retained = scope.original_error.is_some();
        status.original_error = scope.original_error.clone();
        status.guarded_preflight_failures = scope.guarded_preflight_failures.clone();
        status.guarded_probe_scope = scope.guarded_probe_scope.clone();
        if scope.physical_lifetime_unknown {
            status.guarded_interrupted_by_unwind = Some(scope.interrupted_by_unwind);
        }
        if scope.physical_lifetime_unknown {
            status.guarded_physical_lifetime_unknown = Some(true);
            status.guarded_spawn_raw_evidence_incomplete =
                Some(scope.spawned_child.is_some() || previous_process || scope.process.is_none());
            status.guarded_last_spawn_pid = current_pid;
            status.guarded_original_spawn_handle_retained = Some(
                scope.spawned_child.is_some()
                    || scope.process.as_ref().is_some_and(|process| {
                        Some(process.receipt.pid) == current_pid && process.pending_child.is_some()
                    }),
            );
        }
    }
}

/// Passive, nonblocking observation only. No numeric-PID signal, wait/reap,
/// release, deletion, retry or reopening of native admission is provided.
pub fn native_quarantine_status() -> NativeQuarantineStatus {
    let mut status = empty_quarantine_status();
    match QUARANTINE.try_lock() {
        Ok(slots) => {
            status.retained_scope_count = slots.len();
            observe_quarantine_scope(slots.first().map(Box::as_ref), &mut status);
        }
        Err(TryLockError::Poisoned(e)) => {
            let slots = e.into_inner();
            status.retained_scope_count = slots.len();
            observe_quarantine_scope(slots.first().map(Box::as_ref), &mut status);
        }
        Err(TryLockError::WouldBlock) => status.slot_busy = true,
    }
    status
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeQuarantineSnapshot {
    /// An owner is retained outside the enumerable list; its identity and
    /// count are unknown. Never interpret an empty list as no owners if true.
    pub unobservable_owner_fallback: bool,
    pub scopes: Vec<NativeQuarantineStatus>,
}

/// Snapshot every enumerable owner. `None` means contention or allocation
/// failure prevented a complete observation; it must not be treated as empty.
/// The fallback flag on each result reports only that an owner was retained
/// without an enumerable handle, not its identity or count.
pub fn native_quarantine_statuses() -> Option<NativeQuarantineSnapshot> {
    let slots = match QUARANTINE.try_lock() {
        Ok(slots) => slots,
        Err(TryLockError::Poisoned(e)) => e.into_inner(),
        Err(TryLockError::WouldBlock) => return None,
    };
    snapshot_retained_scopes(&slots, QUARANTINE_FALLBACK.load(Ordering::Acquire))
}

fn snapshot_retained_scopes(
    slots: &[Box<RetainedNativeScope>],
    fallback: bool,
) -> Option<NativeQuarantineSnapshot> {
    let mut statuses = Vec::new();
    statuses.try_reserve_exact(slots.len()).ok()?;
    for scope in slots.iter() {
        let mut status = empty_quarantine_status();
        status.retained_scope_count = slots.len();
        observe_quarantine_scope(Some(scope), &mut status);
        statuses.push(status);
    }
    Some(NativeQuarantineSnapshot {
        unobservable_owner_fallback: fallback,
        scopes: statuses,
    })
}

#[cfg(test)]
mod quarantine_observation_tests {
    use super::*;

    #[test]
    fn observes_each_owner_and_unenumerable_fallback_separately() {
        let owner = |pid, message: &str| {
            Box::new(RetainedNativeScope {
                _owner: Box::new(()),
                process: None,
                spawned_child: None,
                last_spawn_pid: Some(pid),
                physical_lifetime_unknown: true,
                receipt: None,
                original_error: Some(message.into()),
                guarded_preflight_failures: Vec::new(),
                interrupted_by_unwind: false,
                guarded_probe_scope: None,
                #[cfg(target_os = "linux")]
                _v6_supervisor: None,
            })
        };
        let slots = vec![owner(11, "first"), owner(22, "second")];
        let snapshot = snapshot_retained_scopes(&slots, true).unwrap();
        assert!(snapshot.unobservable_owner_fallback);
        assert_eq!(snapshot.scopes.len(), 2);
        assert_eq!(snapshot.scopes[0].guarded_last_spawn_pid, Some(11));
        assert_eq!(snapshot.scopes[1].guarded_last_spawn_pid, Some(22));
        assert_eq!(snapshot.scopes[0].original_error.as_deref(), Some("first"));
        assert_eq!(snapshot.scopes[1].original_error.as_deref(), Some("second"));
        let empty = snapshot_retained_scopes(&[], true).unwrap();
        assert!(empty.unobservable_owner_fallback);
        assert!(empty.scopes.is_empty());
    }
}

pub struct NativePairOutput<
    S: NativeProviderDeclaration = rz_experiments::LockedIntegrationPairSpecV1,
> {
    pub receipt: NativePairReceipt<S::Audit>,
    pub receipt_artifact: rz_experiments::ArtifactRef,
    bundle: NativeRunBundle<S>,
}
impl<S: NativeProviderDeclaration> NativePairOutput<S> {
    #[cfg(target_os = "linux")]
    pub(crate) fn v6_cancel_source_observation(
        &self,
    ) -> Option<&crate::pals_adaptive_launch::CancelSourceObservation> {
        self.bundle
            .v6_supervisor
            .as_ref()
            .map(|owner| &owner.cancel_source_observation)
    }
    /// Borrow the same V6 control observation before closing this owner.
    /// An ACK is not physical completion or a durable receipt.
    #[cfg(target_os = "linux")]
    pub(crate) fn v6_pre_signal_observation(
        &self,
    ) -> Option<&crate::process::PreSignalObservation> {
        self.bundle.v6_supervisor.as_ref().and_then(|owner| {
            owner
                .pre_signal_enabled
                .then_some(&owner.pre_signal_observation)
        })
    }
    pub fn process(&self) -> &ProcessOutput {
        self.bundle
            .process
            .as_ref()
            .expect("completed wrapper retains process evidence")
    }
    pub fn launch_owner(&self) -> &NativeLaunchOwner<S> {
        self.bundle
            .owner
            .as_ref()
            .expect("completed wrapper retains input owner")
    }

    /// Move the original, normally closed scope into a read-only archive and
    /// release only its exclusive execution lease. No serialized receipt can
    /// call this constructor. Unknown physical/process closure keeps `self`
    /// and its lease intact, preventing another pair from being admitted.
    #[allow(
        clippy::result_large_err,
        reason = "Refusal returns the original process/input/physical owner and lease without allocating on the closure failure path"
    )]
    pub fn into_closed_archive(mut self) -> Result<ClosedNativePairArchive<S>, Self> {
        let original = self.bundle.receipt.as_ref();
        let process = self.bundle.process.as_ref();
        let projections_match = original.is_some_and(|receipt| {
            matches!(
                (serde_json::to_vec(receipt), serde_json::to_vec(&self.receipt)),
                (Ok(a), Ok(b)) if a == b
            )
        }) && self.bundle.receipt_artifact.as_ref()
            == Some(&self.receipt_artifact);
        let closed = !self.bundle.unresolved()
            && projections_match
            && original.is_some_and(|receipt| {
                receipt.integration_checks_passed
                    && receipt.cleanup_verified
                    && !receipt.unresolved_owner_retained
                    && receipt.primary_error.is_none()
                    && receipt.provider_audit_error.is_none()
                    && receipt.provider_sessions.len()
                        == self.launch_owner().spec.expected_provider_sessions()
            })
            && process.is_some_and(|process| {
                process.receipt.stop == crate::ProcessStop::Exited
                    && process.receipt.exit_code == Some(0)
                    && process.receipt.exit_signal.is_none()
                    && process.receipt.group_cleanup == CleanupStatus::Gone
                    && process.pending_child.is_none()
                    && process.receipt.errors.is_empty()
                    && original.is_some_and(|receipt| {
                        matches!(
                            (
                                serde_json::to_vec(&receipt.process),
                                serde_json::to_vec(&process.receipt)
                            ),
                            (Ok(a), Ok(b)) if a == b
                        )
                    })
            });
        if !closed {
            return Err(self);
        }
        let owner = self.bundle.owner.take().expect("checked original owner");
        let NativeLaunchOwner {
            spec,
            _lease,
            #[cfg(target_os = "linux")]
            snapshot,
        } = owner;
        let archive = ClosedNativePairArchive {
            spec,
            #[cfg(target_os = "linux")]
            snapshot,
            process: self.bundle.process.take().expect("checked closed process"),
            receipt: self
                .bundle
                .receipt
                .take()
                .expect("checked original receipt"),
            receipt_artifact: self
                .bundle
                .receipt_artifact
                .take()
                .expect("checked original receipt artifact"),
        };
        drop(_lease);
        Ok(archive)
    }
}

/// Original process/input/receipt custody after normal closure. Its private
/// fields cannot be reconstituted from JSON or used to spawn another runner.
pub struct ClosedNativePairArchive<S: NativeProviderDeclaration> {
    spec: S,
    #[cfg(target_os = "linux")]
    snapshot: crate::native_launch::linux::Snapshot,
    process: ProcessOutput,
    receipt: NativePairReceipt<S::Audit>,
    receipt_artifact: rz_experiments::ArtifactRef,
}
impl<S: NativeProviderDeclaration> ClosedNativePairArchive<S> {
    pub fn spec(&self) -> &S {
        &self.spec
    }
    pub fn process(&self) -> &ProcessOutput {
        &self.process
    }
    pub fn receipt(&self) -> &NativePairReceipt<S::Audit> {
        &self.receipt
    }
    pub fn receipt_artifact(&self) -> &rz_experiments::ArtifactRef {
        &self.receipt_artifact
    }
    #[cfg(target_os = "linux")]
    pub(crate) fn output_root(&self) -> Option<&std::path::Path> {
        self.snapshot.path.parent()
    }
    #[cfg(target_os = "linux")]
    pub fn snapshots(&self) -> &[NativeSnapshotReceipt] {
        &self.snapshot.receipts
    }
    #[cfg(not(target_os = "linux"))]
    pub fn snapshots(&self) -> &[NativeSnapshotReceipt] {
        &[]
    }
}
impl<S: NativeProviderDeclaration> fmt::Debug for ClosedNativePairArchive<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClosedNativePairArchive")
            .field("input_sha256", &self.spec.input_sha256())
            .field("receipt_artifact", &self.receipt_artifact)
            .finish_non_exhaustive()
    }
}
impl<S: NativeProviderDeclaration> fmt::Debug for NativePairOutput<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativePairOutput")
            .field("receipt", &self.receipt)
            .finish_non_exhaustive()
    }
}
pub struct NativePairFailure<
    S: NativeProviderDeclaration = rz_experiments::LockedIntegrationPairSpecV1,
> {
    pub cause: ArenaError,
    pub receipt: Option<NativePairReceipt<S::Audit>>,
    pub receipt_artifact: Option<rz_experiments::ArtifactRef>,
    bundle: NativeRunBundle<S>,
}

/// Borrowed originals for a caller-controlled persistence operation. The
/// caller must not report the callback as successful before its own durability
/// requirements have been met. No data or GPU capability is copied here.
pub struct GuardedEvidenceView<'a, A> {
    pub run_id: u64,
    pub input_sha256: &'a str,
    pub role: &'static str,
    pub probe: &'static str,
    pub declared_output_limit_bytes: u64,
    pub process: &'a ProcessOutput,
    pub pair_receipt: Option<&'a NativePairReceipt<A>>,
    pub original_error: &'a ArenaError,
    pub failures: &'a [String],
    pub physical_or_process_completion_unknown: bool,
}

impl<S: NativeProviderDeclaration> NativePairFailure<S> {
    #[cfg(target_os = "linux")]
    pub(crate) fn v6_cancel_source_observation(
        &self,
    ) -> Option<&crate::pals_adaptive_launch::CancelSourceObservation> {
        self.bundle
            .v6_supervisor
            .as_ref()
            .map(|owner| &owner.cancel_source_observation)
    }
    /// The failure retains this same observation with the original owner.
    #[cfg(target_os = "linux")]
    pub(crate) fn v6_pre_signal_observation(
        &self,
    ) -> Option<&crate::process::PreSignalObservation> {
        self.bundle.v6_supervisor.as_ref().and_then(|owner| {
            owner
                .pre_signal_enabled
                .then_some(&owner.pre_signal_observation)
        })
    }
    /// Transfer original output and receipt when physical completion is known.
    /// The caller becomes responsible for durable storage of the returned data.
    pub fn take_guarded_evidence(&mut self) -> Option<GuardedPreflightEvidence> {
        self.bundle.take_evidence()
    }
    /// Let the original caller inspect or persist borrowed evidence. Callback
    /// success is not proof of durable storage and never releases the original.
    /// After independent storage verification the caller may explicitly take
    /// custody with `take_guarded_evidence` when physical completion is known.
    pub fn with_guarded_evidence<T, E>(
        &mut self,
        persist: impl FnOnce(GuardedEvidenceView<'_, S::Audit>) -> Result<T, E>,
    ) -> Result<Option<T>, E> {
        let Some(reservation) = self.bundle.probe_evidence.as_ref() else {
            return Ok(None);
        };
        if !reservation.persistence_failed || !reservation.process_available {
            return Ok(None);
        }
        let Some(process) = self.bundle.process.as_ref() else {
            return Ok(None);
        };
        let result = persist(GuardedEvidenceView {
            run_id: reservation.run_id,
            input_sha256: &reservation.input_sha256,
            role: reservation.role,
            probe: reservation.probe,
            declared_output_limit_bytes: reservation.output_limit_bytes,
            process,
            pair_receipt: self.bundle.receipt.as_ref(),
            original_error: &self.cause,
            failures: &self.bundle.guarded_preflight_failures,
            physical_or_process_completion_unknown: self.bundle.unresolved(),
        })?;
        // A callback result cannot prove durable storage or a physical fence.
        Ok(Some(result))
    }
    pub fn process(&self) -> Option<&ProcessOutput> {
        self.bundle.process.as_ref()
    }
    pub fn launch_owner(&self) -> Option<&NativeLaunchOwner<S>> {
        self.bundle.owner.as_ref()
    }
}
impl<S: NativeProviderDeclaration> fmt::Debug for NativePairFailure<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativePairFailure")
            .field("cause", &self.cause)
            .field("receipt", &self.receipt)
            .finish_non_exhaustive()
    }
}
impl<S: NativeProviderDeclaration> fmt::Display for NativePairFailure<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "native pair failed: {}; original process/input ownership retained={}",
            self.cause,
            self.bundle.owner.is_some()
        )
    }
}
impl<S: NativeProviderDeclaration> std::error::Error for NativePairFailure<S> {}

/// Consume exactly one prepared lease. Post-spawn failures retain the original
/// process output and every input/cwd pin, including when receipt saving fails.
pub fn run_native_pair(
    owner: NativeLaunchOwner,
    cancel: Option<&AtomicBool>,
) -> Result<NativePairOutput, Box<NativePairFailure>> {
    run_native_pair_for(owner, cancel)
}
pub(crate) fn run_native_pair_for<S: NativeProviderDeclaration>(
    owner: NativeLaunchOwner<S>,
    cancel: Option<&AtomicBool>,
) -> Result<NativePairOutput<S>, Box<NativePairFailure<S>>> {
    run_native_pair_inner(owner, cancel, None)
}

/// V6-only opt-in. The caller transfers its original execution/control owner;
/// no deadline is created from dispatch time and no physical fence is implied.
#[cfg(target_os = "linux")]
pub(crate) fn run_native_pair_for_v6_controlled(
    mut owner: NativeLaunchOwner<crate::pals_adaptive_launch::LockedPalsSystemArenaPairV6>,
    cancel: Option<&AtomicBool>,
    mut supervisor: NativeV6Supervisor,
) -> Result<
    NativePairOutput<crate::pals_adaptive_launch::LockedPalsSystemArenaPairV6>,
    Box<NativePairFailure<crate::pals_adaptive_launch::LockedPalsSystemArenaPairV6>>,
> {
    supervisor.original_pin_error = match supervisor.role_sockets.as_ref() {
        Some(sockets) => match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sockets.build_invocation(&owner, &supervisor.execution)
        })) {
            Ok(Ok(invocation)) => {
                owner.snapshot.invocation = invocation;
                None
            }
            Ok(Err(error)) => Some(error),
            Err(_) => Some(ArenaError::Io(
                "V6 original invocation preparation panicked".into(),
            )),
        },
        None => Some(ArenaError::Invalid(
            "V6 original role listeners unavailable".into(),
        )),
    };
    run_native_pair_inner(owner, cancel, Some(supervisor))
}

fn run_native_pair_inner<S: NativeProviderDeclaration>(
    owner: NativeLaunchOwner<S>,
    cancel: Option<&AtomicBool>,
    #[cfg(target_os = "linux")] supervisor: Option<NativeV6Supervisor>,
    #[cfg(not(target_os = "linux"))] _supervisor: Option<()>,
) -> Result<NativePairOutput<S>, Box<NativePairFailure<S>>> {
    // Existing V1/V5 and Rovent V3 declarations keep their established
    // behavior. PALS V6 must explicitly opt in before its launch guard opens.
    let physical_guard_required = owner.spec.requires_physical_lifetime_guard();
    let mut bundle = NativeRunBundle {
        owner: Some(owner),
        process: None,
        receipt: None,
        receipt_artifact: None,
        original_error: None,
        guarded_preflight_failures: Vec::new(),
        probe_evidence: None,
        spawn_custody: crate::process::SpawnCustody::default(),
        physical_guard_required,
        #[cfg(target_os = "linux")]
        v6_supervisor: supervisor,
    };
    #[cfg(target_os = "linux")]
    let result = match bundle
        .v6_supervisor
        .as_mut()
        .and_then(|supervisor| supervisor.original_pin_error.take())
    {
        Some(error) => Err(error),
        None => linux::run(&mut bundle, cancel),
    };
    #[cfg(not(target_os = "linux"))]
    let result: Result<(NativePairReceipt<S::Audit>, rz_experiments::ArtifactRef), ArenaError> = {
        let _ = cancel;
        Err(ArenaError::Invalid(format!(
            "{} NN integration runner requires Linux",
            bundle
                .owner
                .as_ref()
                .expect("prepared native owner")
                .spec
                .provider_name()
        )))
    };
    match result {
        Ok((receipt, receipt_artifact)) => {
            bundle.receipt = Some(receipt.clone());
            Ok(NativePairOutput {
                receipt,
                receipt_artifact,
                bundle,
            })
        }
        Err(cause) => {
            bundle.original_error = Some(cause.to_string());
            Err(Box::new(NativePairFailure {
                cause,
                receipt: bundle.receipt.clone(),
                receipt_artifact: bundle.receipt_artifact.clone(),
                bundle,
            }))
        }
    }
}

#[cfg(target_os = "linux")]
pub use linux::validate_native_provider_record_fields;

#[cfg(target_os = "linux")]
pub(crate) mod linux {
    use super::*;
    use crate::native_launch::{
        NATIVE_PAIR_METADATA_CAP,
        linux::{pin, verify_copy},
    };
    use crate::{
        GameSpec, PairSpec, PgnLimits, PgnOutcomePolicy, ProcessStop, audit_pair_pgn_for_spec,
        canonical_sha256,
    };
    use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::{Dir, OpenOptions, OpenOptionsExt};
    use nix::fcntl::{PosixFadviseAdvice, posix_fadvise};
    use rz_experiments::{
        ArtifactRef, CpuNativeLaunchSpecV1, NativeArtifactRole, NativeEngineRole, OutcomePolicy,
    };
    use serde_json::Value;
    use sha2::{Digest, Sha256};
    use std::{
        collections::{BTreeMap, BTreeSet},
        fs::File,
        io::Read,
        os::unix::fs::MetadataExt,
    };
    const PROVIDER_JSON_CAP: u64 = 256 * 1024;
    fn invalid(detail: &str) -> ArenaError {
        ArenaError::Integrity(detail.into())
    }
    fn artifact<S: NativeLaunchDeclaration>(
        owner: &NativeLaunchOwner<S>,
        name: &str,
        bytes: &[u8],
    ) -> ArtifactRef {
        ArtifactRef {
            path: format!("{}/{name}", owner.snapshot.output_directory),
            sha256: format!("{:x}", Sha256::digest(bytes)),
            bytes: bytes.len() as u64,
            source: format!(
                "RoveZero {} NN integration execution evidence",
                owner.spec.provider_name()
            ),
            license: "MIT execution evidence; external input rights remain separate".into(),
        }
    }
    fn put<S: NativeLaunchDeclaration>(
        owner: &NativeLaunchOwner<S>,
        name: &str,
        bytes: &[u8],
        cap: u64,
    ) -> Result<ArtifactRef, ArenaError> {
        if bytes.len() as u64 > cap {
            return Err(ArenaError::Budget(
                "native evidence exceeds reserved bytes".into(),
            ));
        }
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .mode(0o600);
        let mut file = owner
            .snapshot
            .directory
            .open_with(name, &options)
            .map_err(|_| {
                ArenaError::Io(
                    "cannot create exclusive native evidence; prior/partial bytes retained".into(),
                )
            })?;
        use std::io::Write;
        file.write_all(bytes)
            .and_then(|()| file.sync_all())
            .map_err(|_| {
                ArenaError::Io("cannot finish native evidence; partial bytes retained".into())
            })?;
        Ok(artifact(owner, name, bytes))
    }
    pub(crate) fn read_file(directory: &Dir, name: &str, cap: u64) -> Result<Vec<u8>, ArenaError> {
        let mut options = OpenOptions::new();
        options
            .read(true)
            .follow(FollowSymlinks::No)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        let file = directory
            .open_with(name, &options)
            .map_err(|_| {
                ArenaError::Io("native evidence file is missing or cannot be pinned".into())
            })?
            .into_std();
        read_bytes(file, cap)
    }
    fn read_bytes(mut file: File, cap: u64) -> Result<Vec<u8>, ArenaError> {
        use std::io::{Seek, SeekFrom};
        file.seek(SeekFrom::Start(0))
            .map_err(|_| invalid("cannot rewind pinned native evidence"))?;
        let metadata = file
            .metadata()
            .map_err(|_| invalid("native evidence metadata unavailable"))?;
        if !metadata.is_file() || metadata.len() > cap {
            return Err(ArenaError::Budget(
                "native evidence is not a bounded regular file".into(),
            ));
        }
        let limit = cap
            .checked_add(1)
            .ok_or_else(|| ArenaError::Budget("native read limit overflow".into()))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(
                usize::try_from(metadata.len())
                    .map_err(|_| ArenaError::Budget("native evidence cannot fit host".into()))?,
            )
            .map_err(|_| ArenaError::Budget("native evidence allocation failed".into()))?;
        file.take(limit)
            .read_to_end(&mut bytes)
            .map_err(|_| ArenaError::Io("cannot read pinned native evidence".into()))?;
        if bytes.len() as u64 > cap {
            return Err(ArenaError::Budget(
                "native evidence grew beyond limit".into(),
            ));
        }
        Ok(bytes)
    }
    fn verify_inputs<S: NativeLaunchDeclaration>(
        owner: &mut NativeLaunchOwner<S>,
        mut cache_hint: Option<&mut SnapshotCacheHint>,
    ) -> Result<(), ArenaError> {
        crate::native_launch::linux::verify_preflight_runtime_roots(&owner.snapshot)?;
        crate::native_launch::linux::verify_input_directories(
            &owner.snapshot.directory,
            &owner.snapshot.input_directory,
            &owner.snapshot.input_subdirectories,
        )?;
        for item in &mut owner.snapshot.pins {
            verify_copy(&mut item.file, &item.artifact)?;
            let relative = item
                .path
                .strip_prefix(&owner.snapshot.path)
                .map_err(|_| invalid("native pin escaped owned directory"))?;
            let mut options = OpenOptions::new();
            options
                .read(true)
                .follow(FollowSymlinks::No)
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
            let (directory, name) = crate::native_launch::linux::input_parent(
                &owner.snapshot.input_directory,
                &owner.snapshot.input_subdirectories,
                relative,
            )?;
            let path_file = directory
                .open_with(name, &options)
                .map_err(|_| invalid("native input pathname changed"))?
                .into_std();
            let held = item
                .file
                .metadata()
                .map_err(|_| invalid("native pin metadata missing"))?;
            let named = path_file
                .metadata()
                .map_err(|_| invalid("native named input metadata missing"))?;
            if (held.dev(), held.ino()) != (named.dev(), named.ino()) {
                return Err(invalid(
                    "native input pathname no longer names its owned inode",
                ));
            }
            // The held bytes and named inode have both passed their checks.
            // Callers exclude unresolved post-run owners from this advice.
            if let Some(status) = cache_hint.as_mut() {
                status.advise(&item.file, item.artifact.bytes);
            }
        }
        Ok(())
    }
    // Callback boundaries compare the owner-held capabilities without rehashing
    // a multi-gigabyte runtime per probe. Full byte verification remains at the
    // existing prelaunch/post-run boundaries; the profile loader hashes its own
    // bounded actual bytes inside the provider callback.
    fn verify_provider_input_names<S: NativeLaunchDeclaration>(
        owner: &NativeLaunchOwner<S>,
    ) -> Result<(), ArenaError> {
        crate::native_launch::linux::verify_input_directories(
            &owner.snapshot.directory,
            &owner.snapshot.input_directory,
            &owner.snapshot.input_subdirectories,
        )?;
        for item in &owner.snapshot.pins {
            let relative = item
                .path
                .strip_prefix(&owner.snapshot.path)
                .map_err(|_| invalid("provider input pin escaped owner root"))?;
            let (directory, name) = crate::native_launch::linux::input_parent(
                &owner.snapshot.input_directory,
                &owner.snapshot.input_subdirectories,
                relative,
            )?;
            let mut options = OpenOptions::new();
            options
                .read(true)
                .follow(FollowSymlinks::No)
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
            let named = directory
                .open_with(name, &options)
                .map_err(|_| invalid("provider input name is unavailable"))?
                .into_std()
                .metadata()
                .map_err(|_| invalid("provider named input metadata missing"))?;
            let held = item
                .file
                .metadata()
                .map_err(|_| invalid("provider input pin metadata missing"))?;
            if !held.is_file()
                || held.len() != item.artifact.bytes
                || held.mode() & 0o222 != 0
                || held.nlink() != 1
                || (held.dev(), held.ino(), held.uid(), held.mode(), held.len())
                    != (
                        named.dev(),
                        named.ino(),
                        named.uid(),
                        named.mode(),
                        named.len(),
                    )
            {
                return Err(invalid(
                    "provider input name differs from read-only owner pin",
                ));
            }
        }
        Ok(())
    }
    fn provider_input_views<S: NativeLaunchDeclaration>(
        owner: &NativeLaunchOwner<S>,
    ) -> Vec<NativeProviderInputView<'_>> {
        owner
            .snapshot
            .pins
            .iter()
            .map(|pin| NativeProviderInputView {
                artifact: &pin.artifact,
                path: &pin.path,
                file: &pin.file,
            })
            .collect()
    }
    // Only the launch owner's synced, verified private snapshots are eligible.
    // This is a best-effort kernel cache hint, not deletion, memory reclamation
    // evidence or permission to touch the source/C runtime cache/GPU buffers.
    pub(super) fn advise_input_cache(file: &File) -> Result<(), ArenaError> {
        posix_fadvise(file, 0, 0, PosixFadviseAdvice::POSIX_FADV_DONTNEED)
            .map_err(|error| ArenaError::Io(format!("native snapshot cache hint failed: {error}")))
    }

    // The caller has just verified these closed-writer private pins. Never
    // apply this hint to an original asset, the shared runtime or GPU memory.
    fn advise_verified_input_cache<S: NativeLaunchDeclaration>(
        owner: &NativeLaunchOwner<S>,
        phase: &str,
        game: Option<u8>,
    ) -> SnapshotCacheHint {
        let mut status = SnapshotCacheHint::new(phase, game);
        if owner.spec.advise_drop_input_cache() {
            for item in &owner.snapshot.pins {
                status.advise(&item.file, item.artifact.bytes);
            }
        }
        status.emit();
        status
    }

    #[cfg(test)]
    #[test]
    fn snapshot_cache_hint_preserves_readonly_pin_bytes_inode_and_cursor() {
        use std::io::{Seek, SeekFrom, Write};
        use std::os::unix::fs::{FileExt, PermissionsExt};

        let path =
            std::env::temp_dir().join(format!("rovezero-native-cache-hint-{}", std::process::id()));
        let mut writer = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        let bytes = vec![0x5a; 8 * 1024 * 1024 + 17];
        writer.write_all(&bytes).unwrap();
        writer.sync_all().unwrap();
        writer
            .set_permissions(std::fs::Permissions::from_mode(0o400))
            .unwrap();
        drop(writer);
        let mut pin = File::open(&path).unwrap();
        pin.seek(SeekFrom::Start(19)).unwrap();
        let before = pin.metadata().unwrap();
        let mut hint = SnapshotCacheHint::new("synthetic_verified_pin", None);
        hint.advise(&pin, 8 * 1024 * 1024 - 1);
        assert_eq!(hint.eligible_files, 0);
        hint.advise(&pin, before.len());
        assert_eq!((hint.eligible_files, hint.advised_files), (1, 1));
        assert!(hint.errors.is_empty());
        assert_eq!(pin.stream_position().unwrap(), 19);
        let after = std::fs::metadata(&path).unwrap();
        assert_eq!((before.dev(), before.ino()), (after.dev(), after.ino()));
        assert_eq!(before.len(), after.len());
        assert_eq!(after.permissions().mode() & 0o777, 0o400);
        let mut observed = vec![0; bytes.len()];
        pin.read_exact_at(&mut observed, 0).unwrap();
        assert_eq!(observed, bytes);
        drop(pin);
        std::fs::remove_file(path).unwrap();
        // Cache residency is deliberately not asserted: advice is not a seal.
    }

    #[cfg(test)]
    #[test]
    fn failed_cache_advice_is_recorded_and_unresolved_owner_is_ineligible() {
        let mut hint = SnapshotCacheHint::new("synthetic_failure", None);
        hint.record_advice(Err(ArenaError::Io("injected fadvise failure".into())));
        hint.record_advice(Ok(()));
        assert_eq!((hint.eligible_files, hint.advised_files), (2, 1));
        assert_eq!(hint.errors.len(), 1);
        assert!(hint.errors[0].contains("injected fadvise failure"));
        let encoded = serde_json::to_string(&hint).unwrap();
        assert!(encoded.contains("injected fadvise failure"));
        // The cleanup gate is false while a child/group is unresolved, even
        // in an experimental build. Advice does not release that owner's pins.
        assert!(SnapshotCacheHint::rolling(false, "postcheck_each_verified").is_none());
        assert_eq!(
            SnapshotCacheHint::rolling(true, "prelaunch_each_verified").is_some(),
            cfg!(feature = "experimental-snapshot-reclaim")
        );
    }
    fn pair<S: NativeLaunchDeclaration>(
        owner: &NativeLaunchOwner<S>,
    ) -> Result<PairSpec, ArenaError> {
        let input = owner.spec.view();
        let game = |ordinal: usize| -> Result<GameSpec, ArenaError> {
            let white = owner.spec.engine_view(input.white_order[ordinal])?;
            let black = owner.spec.engine_view(input.white_order[ordinal].other())?;
            Ok(GameSpec {
                id: format!("{}-game-{}", input.pair_id, ordinal + 1),
                white_engine: white.engine_id.into(),
                black_engine: black.engine_id.into(),
                engine_seeds: BTreeMap::new(),
                engine_slots: BTreeMap::new(),
            })
        };
        Ok(PairSpec {
            id: input.pair_id.into(),
            ordinal: 0,
            opening: input.opening.clone(),
            opening_input_sha256: canonical_sha256(input.opening)?,
            games: [game(0)?, game(1)?],
            execution_order: [0, 1],
        })
    }
    fn validate_preflight_provider_evidence(
        evidence: &[(String, Vec<u8>)],
    ) -> Result<(), ArenaError> {
        if evidence.len() > 4 {
            return Err(invalid("preflight provider evidence count exceeded"));
        }
        let mut names = BTreeSet::new();
        for (relative, bytes) in evidence {
            let components: Vec<_> = std::path::Path::new(relative).components().collect();
            if relative.len() > 256
                || components.len() != 2
                || components
                    .iter()
                    .any(|part| !matches!(part, std::path::Component::Normal(_)))
                || !names.insert(relative)
                || bytes.is_empty()
                || bytes.len() > 128 * 1024
            {
                return Err(invalid(
                    "preflight provider evidence topology/byte bound invalid",
                ));
            }
        }
        Ok(())
    }
    #[cfg(test)]
    #[test]
    fn preflight_provider_evidence_rejects_aliases_traversal_and_overflow() {
        let rows = vec![
            ("native-process-10/startup.json".into(), vec![1]),
            ("native-process-10/termination.json".into(), vec![2]),
        ];
        assert!(validate_preflight_provider_evidence(&[]).is_ok()); // historical default
        assert!(validate_preflight_provider_evidence(&rows).is_ok());
        let mut duplicate = rows.clone();
        duplicate.push(rows[0].clone());
        assert!(validate_preflight_provider_evidence(&duplicate).is_err());
        for name in [
            "../startup.json",
            "/root/startup.json",
            "native/../startup.json",
            "native/startup/nested.json",
        ] {
            assert!(validate_preflight_provider_evidence(&[(name.into(), vec![1])]).is_err());
        }
        assert!(validate_preflight_provider_evidence(&[(rows[0].0.clone(), vec![])]).is_err());
        assert!(
            validate_preflight_provider_evidence(&[(rows[0].0.clone(), vec![0; 128 * 1024 + 1])])
                .is_err()
        );
        let excessive: Vec<_> = (0..5)
            .map(|i| (format!("native-process-{i}/startup.json"), vec![1]))
            .collect();
        assert!(validate_preflight_provider_evidence(&excessive).is_err());
    }
    fn owned_role_runtime_directory<S: NativeProviderDeclaration>(
        owner: &NativeLaunchOwner<S>,
        index: usize,
    ) -> Result<cap_std::fs::Dir, ArenaError> {
        use cap_fs_ext::DirExt;
        use std::os::unix::fs::MetadataExt;
        let name = match index {
            0 => "baseline-runtime",
            1 => "candidate-runtime",
            _ => return Err(invalid("guarded role index is invalid")),
        };
        let pinned = owner.snapshot._runtime_root_pins[index]
            .try_clone()
            .map_err(|_| invalid("cannot clone owned role runtime pin"))?;
        let held = pinned
            .metadata()
            .map_err(|_| invalid("owned role runtime pin unavailable"))?;
        let named = owner
            .snapshot
            .directory
            .open_dir_nofollow(name)
            .map_err(|_| invalid("named role runtime directory unavailable"))?
            .into_std_file()
            .metadata()
            .map_err(|_| invalid("named role runtime metadata unavailable"))?;
        if !held.is_dir()
            || !named.is_dir()
            || (held.dev(), held.ino()) != (named.dev(), named.ino())
        {
            return Err(invalid(
                "role runtime name differs from its original owned pin",
            ));
        }
        Ok(cap_std::fs::Dir::from_std_file(pinned))
    }

    fn verify_guarded_preflight<S: NativeProviderDeclaration>(
        owner: &NativeLaunchOwner<S>,
        process: &ProcessOutput,
        role: NativeEngineRole,
        probe: NativePreflightProbe,
        pending: &AtomicBool,
        required: bool,
    ) -> Result<(), ArenaError> {
        if !required {
            return Ok(());
        }
        crate::native_launch::linux::verify_preflight_runtime_roots(&owner.snapshot)?;
        verify_provider_input_names(owner)?;
        let index = if role == NativeEngineRole::Baseline {
            0
        } else {
            1
        };
        let directory = match &owner.snapshot.preflight_runtime_roots[index] {
            Some(root) => root
                .directory
                .try_clone()
                .map_err(|_| invalid("guarded preflight runtime directory unavailable"))?,
            None => owned_role_runtime_directory(owner, index)?,
        };
        let pins = provider_input_views(owner);
        let inputs = NativeProviderInputContext {
            directory: &owner.snapshot.input_directory,
            pins: &pins,
        };
        owner
            .spec
            .verify_preflight_no_gpu(role, probe, process, &directory, &inputs)?;
        verify_provider_input_names(owner)?;
        if owner.snapshot.preflight_runtime_roots[index].is_none() {
            owned_role_runtime_directory(owner, index)?;
        }
        crate::native_launch::linux::verify_preflight_runtime_roots(&owner.snapshot)?;
        pending.store(false, Ordering::Release);
        Ok(())
    }

    fn verify_guarded_pair<S: NativeProviderDeclaration>(
        owner: &NativeLaunchOwner<S>,
        process: &ProcessOutput,
        pending: &AtomicBool,
        required: bool,
    ) -> Result<(), ArenaError> {
        if !required {
            return Ok(());
        }
        verify_provider_input_names(owner)?;
        let baseline = owned_role_runtime_directory(owner, 0)?;
        let candidate = owned_role_runtime_directory(owner, 1)?;
        let pins = provider_input_views(owner);
        let inputs = NativeProviderInputContext {
            directory: &owner.snapshot.input_directory,
            pins: &pins,
        };
        owner
            .spec
            .verify_pair_physical_completion(process, &baseline, &candidate, &inputs)?;
        verify_provider_input_names(owner)?;
        owned_role_runtime_directory(owner, 0)?;
        owned_role_runtime_directory(owner, 1)?;
        pending.store(false, Ordering::Release);
        Ok(())
    }

    fn persist_guarded_probe<S: NativeProviderDeclaration>(
        owner: &NativeLaunchOwner<S>,
        process: &ProcessOutput,
        tag: &str,
        stage: &str,
    ) -> Vec<ArenaError> {
        let mut failures = Vec::new();
        // Attempt each independent byte stream even if a previous write fails.
        for (suffix, bytes) in [
            ("stdout.log", &process.stdout),
            ("stderr.log", &process.stderr),
        ] {
            if let Err(error) = put(
                owner,
                &format!("external-{tag}-{stage}.{suffix}"),
                bytes,
                256 * 1024,
            ) {
                failures.push(error);
            }
        }
        match serde_json::to_vec_pretty(&process.receipt) {
            Ok(bytes) => {
                if let Err(error) = put(
                    owner,
                    &format!("external-{tag}-{stage}-process.v2.json"),
                    &bytes,
                    64 * 1024,
                ) {
                    failures.push(error);
                }
            }
            Err(_) => failures.push(invalid("preflight process receipt encoding failed")),
        }
        failures
    }

    fn preflight_endpoints<S: NativeProviderDeclaration>(
        bundle: &mut NativeRunBundle<S>,
        cancel: Option<&AtomicBool>,
        run_id: Option<u64>,
        v6_deadline: Option<std::time::Instant>,
    ) -> Result<(Vec<crate::ExternalUciPreflight>, Vec<ArtifactRef>), ArenaError> {
        let mut receipts = Vec::new();
        let mut provider_artifacts = Vec::new();
        for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
            let owner = bundle.owner.as_ref().expect("preflight owns snapshot");
            let Some(e) = owner.spec.engine_view(role)?.external.cloned() else {
                continue;
            };
            let binary = pin(&owner.snapshot.pins, &e.binary)?;
            // Option identification/readiness probes must not create game-only
            // provider records in the role runtime directory.
            let runtime_root = owner.snapshot.preflight_runtime_path(role);
            let args = crate::native_launch::linux::external_arguments(
                &owner.spec,
                role,
                &e,
                &owner.snapshot.pins,
                crate::native_launch::linux::ExternalArgumentMode::Preflight(&runtime_root),
            )?;
            let requested = owner
                .spec
                .external_options(role)?
                .iter()
                .map(|(k, v)| {
                    crate::external_uci::resolve_asset_tokens(v, &e, &owner.snapshot.pins)
                        .map(|v| (k.clone(), v))
                })
                .collect::<Result<BTreeMap<_, _>, _>>()?;
            let variables = e
                .environment
                .as_ref()
                .map(|environment| {
                    environment
                        .variables
                        .iter()
                        .map(|(name, value)| {
                            crate::external_uci::resolve_asset_tokens(
                                value,
                                &e,
                                &owner.snapshot.pins,
                            )
                            .map(|value| (name.clone(), value))
                        })
                        .collect::<Result<BTreeMap<_, _>, _>>()
                })
                .transpose()?;
            let launch = crate::engine_environment::prepare(
                binary,
                args,
                e.environment.as_ref(),
                variables.as_ref(),
                &owner.snapshot.pins,
            )?;
            let launch = crate::engine_environment::place(
                launch,
                owner.spec.match_execution(),
                role,
                e.environment.is_some(),
                &owner.snapshot.pins,
            )?;
            let mut limits = crate::ProcessLimits {
                wall_ms: clamp_original_wall(
                    e.handshake_timeout_ms,
                    earlier_deadline(owner.spec.original_wall_deadline()?, v6_deadline),
                    std::time::Instant::now(),
                )?,
                shutdown_grace_ms: owner.spec.view().timeouts.shutdown_ms,
                max_output_bytes: 256 * 1024,
                max_child_processes: owner.spec.view().budget.max_child_processes,
            };
            preflight_runtime_totals(owner)?;
            let tag = match role {
                NativeEngineRole::Baseline => "baseline",
                NativeEngineRole::Candidate => "candidate",
            };
            if bundle.physical_guard_required {
                bundle.probe_evidence = Some(GuardedProbeReservation::acquire(
                    run_id.expect("guarded run has evidence identity"),
                    owner.spec.input_sha256().to_string(),
                    tag,
                    "identification",
                    limits.max_output_bytes,
                ));
            }
            let process = crate::process::supervise_protocol_in_directory_with_spawn(
                &launch.program.file,
                &launch.arguments,
                &owner.snapshot.cwd,
                limits,
                cancel,
                &owner.snapshot.watch,
                b"uci\nquit\n",
                if bundle.physical_guard_required {
                    Some(&mut bundle.spawn_custody)
                } else {
                    None
                },
            )?;
            bundle.process = Some(process);
            if let Some(reservation) = bundle.probe_evidence.as_mut() {
                reservation.process_available = true;
            }
            let process = bundle.process.as_ref().expect("probe process retained");
            let root_recheck =
                crate::native_launch::linux::verify_preflight_runtime_roots(&owner.snapshot);
            // Save the actual bounded process receipt before the success
            // gate. Failure evidence changes no cleanup/admission requirement.
            let identification_process = process.receipt.clone();
            let root_recheck = if bundle.physical_guard_required {
                let physical = if root_recheck.is_ok() {
                    verify_guarded_preflight(
                        owner,
                        process,
                        role,
                        NativePreflightProbe::Identification,
                        &bundle.spawn_custody.pending,
                        true,
                    )
                } else {
                    Err(invalid("guarded preflight root binding was not verified"))
                };
                let mut persistence = persist_guarded_probe(owner, process, tag, "identification");
                if let Err(error) = &physical {
                    bundle
                        .guarded_preflight_failures
                        .push(format!("identification physical: {error}"));
                }
                for error in &persistence {
                    bundle
                        .guarded_preflight_failures
                        .push(format!("identification persistence: {error}"));
                }
                if persistence.is_empty() && physical.is_ok() && root_recheck.is_ok() {
                    bundle.probe_evidence.take();
                } else if let Some(reservation) = bundle.probe_evidence.as_mut() {
                    reservation.persistence_failed = !persistence.is_empty();
                }
                root_recheck?;
                physical?;
                if !persistence.is_empty() {
                    return Err(persistence.remove(0));
                }
                Ok(())
            } else {
                put(
                    owner,
                    &format!("external-{tag}-identification.stdout.log"),
                    &process.stdout,
                    256 * 1024,
                )?;
                put(
                    owner,
                    &format!("external-{tag}-identification.stderr.log"),
                    &process.stderr,
                    256 * 1024,
                )?;
                put(
                    owner,
                    &format!("external-{tag}-identification-process.v2.json"),
                    &serde_json::to_vec_pretty(&process.receipt)
                        .map_err(|_| invalid("preflight process receipt encoding failed"))?,
                    64 * 1024,
                )?;
                root_recheck
            };
            let clean = |p: &ProcessOutput| {
                p.receipt.stop == ProcessStop::Exited
                    && p.receipt.exit_code == Some(0)
                    && p.receipt.group_cleanup == CleanupStatus::Gone
                    && p.pending_child.is_none()
                    && p.receipt.errors.is_empty()
            };
            if !clean(process) {
                return Err(invalid(
                    "external identification failed process/cleanup gate",
                ));
            }
            root_recheck?;
            preflight_runtime_totals(owner)?;
            let advertisement = crate::parse_uci_advertisement(&process.stdout)?;
            crate::validate_uci_options(&advertisement, &e.expected_uci_name, &requested)?;
            let commands = crate::uci_preflight_commands(&requested, e.family == "stockfish")?;
            preflight_runtime_totals(owner)?;
            limits.wall_ms = clamp_original_wall(
                limits.wall_ms,
                earlier_deadline(owner.spec.original_wall_deadline()?, v6_deadline),
                std::time::Instant::now(),
            )?;
            if bundle.physical_guard_required {
                bundle.probe_evidence = Some(GuardedProbeReservation::acquire(
                    run_id.expect("guarded run has evidence identity"),
                    owner.spec.input_sha256().to_string(),
                    tag,
                    "readiness",
                    limits.max_output_bytes,
                ));
            }
            let process = crate::process::supervise_protocol_in_directory_with_spawn(
                &launch.program.file,
                &launch.arguments,
                &owner.snapshot.cwd,
                limits,
                cancel,
                &owner.snapshot.watch,
                &commands,
                if bundle.physical_guard_required {
                    Some(&mut bundle.spawn_custody)
                } else {
                    None
                },
            )?;
            bundle.process = Some(process);
            if let Some(reservation) = bundle.probe_evidence.as_mut() {
                reservation.process_available = true;
            }
            let process = bundle.process.as_ref().expect("readiness process retained");
            let root_recheck =
                crate::native_launch::linux::verify_preflight_runtime_roots(&owner.snapshot);
            let root_recheck = if bundle.physical_guard_required {
                let physical = if root_recheck.is_ok() {
                    verify_guarded_preflight(
                        owner,
                        process,
                        role,
                        NativePreflightProbe::Readiness,
                        &bundle.spawn_custody.pending,
                        true,
                    )
                } else {
                    Err(invalid("guarded preflight root binding was not verified"))
                };
                let mut persistence = persist_guarded_probe(owner, process, tag, "readiness");
                if let Err(error) = &physical {
                    bundle
                        .guarded_preflight_failures
                        .push(format!("readiness physical: {error}"));
                }
                for error in &persistence {
                    bundle
                        .guarded_preflight_failures
                        .push(format!("readiness persistence: {error}"));
                }
                if persistence.is_empty() && physical.is_ok() && root_recheck.is_ok() {
                    bundle.probe_evidence.take();
                } else if let Some(reservation) = bundle.probe_evidence.as_mut() {
                    reservation.persistence_failed = !persistence.is_empty();
                }
                root_recheck?;
                physical?;
                if !persistence.is_empty() {
                    return Err(persistence.remove(0));
                }
                Ok(())
            } else {
                put(
                    owner,
                    &format!("external-{tag}-readiness.stdout.log"),
                    &process.stdout,
                    256 * 1024,
                )?;
                put(
                    owner,
                    &format!("external-{tag}-readiness.stderr.log"),
                    &process.stderr,
                    256 * 1024,
                )?;
                put(
                    owner,
                    &format!("external-{tag}-readiness-process.v2.json"),
                    &serde_json::to_vec_pretty(&process.receipt)
                        .map_err(|_| invalid("preflight process receipt encoding failed"))?,
                    64 * 1024,
                )?;
                root_recheck
            };
            if !clean(process) {
                return Err(invalid(
                    "external readiness/stop/quit failed process/cleanup gate",
                ));
            }
            root_recheck?;
            preflight_runtime_totals(owner)?;
            if crate::parse_uci_advertisement(&process.stdout)? != advertisement {
                return Err(invalid(
                    "external advertisements differ across preflight sessions",
                ));
            }
            let output = std::str::from_utf8(&process.stdout)
                .map_err(|_| invalid("external readiness output is not UTF-8"))?;
            let state = rz_position::Position::startpos();
            let legal = state.legal_moves();
            let legal_bestmove = output
                .lines()
                .filter_map(|line| line.strip_prefix("bestmove "))
                .filter_map(|line| line.split_whitespace().next())
                .any(|value| legal.iter().any(|m| m.to_string() == value));
            if output.lines().filter(|line| *line == "readyok").count() != 2 || !legal_bestmove {
                return Err(invalid(
                    "external readiness or bounded search/stop witness missing",
                ));
            }
            let options = requested
                .into_iter()
                .map(|(name, value)| {
                    let advertised = advertisement.options[&name].clone();
                    (
                        name,
                        crate::ExternalOptionObservation {
                            requested: value,
                            advertised,
                            value_supported: true,
                            sent_to_preflight: true,
                            readiness_barrier_observed: true,
                            actual_value: None,
                        },
                    )
                })
                .collect();
            let receipt = crate::ExternalUciPreflight {
                schema_version: 2,
                engine_id: e.id.clone(),
                advertisement,
                options,
                environment: e
                    .environment
                    .as_ref()
                    .map(crate::engine_environment::observation),
                identification_process,
                readiness_process: process.receipt.clone(),
                stop_and_legal_bestmove_observed: true,
                compiler_information: output
                    .lines()
                    .filter(|line| {
                        line.contains("Compiled by")
                            || line.contains("Compilation settings")
                            || line.contains("using SSE")
                            || line.contains("using AVX")
                    })
                    .map(String::from)
                    .collect(),
                selected_isa: None,
                scope: "independent pinned-binary CPU UCI preflight; readyok is a barrier, not option readback or game-process attestation",
            };
            let bytes = serde_json::to_vec_pretty(&receipt)
                .map_err(|_| invalid("external preflight serialization failed"))?;
            put(
                owner,
                &format!("external-{tag}-preflight.v2.json"),
                &bytes,
                64 * 1024,
            )?;
            use cap_fs_ext::DirExt;
            let index = if role == NativeEngineRole::Baseline {
                0
            } else {
                1
            };
            let preflight_directory = match &owner.snapshot.preflight_runtime_roots[index] {
                Some(root) => root.directory.try_clone(),
                None => owner.snapshot.directory.open_dir_nofollow(if index == 0 {
                    "baseline-runtime"
                } else {
                    "candidate-runtime"
                }),
            }
            .map_err(|e| ArenaError::Io(e.to_string()))?;
            // The selected capability is owner-held; compare its path/name pins
            // immediately before and after provider consumption as well.
            crate::native_launch::linux::verify_preflight_runtime_roots(&owner.snapshot)?;
            verify_provider_input_names(owner)?;
            let pins = provider_input_views(owner);
            let inputs = NativeProviderInputContext {
                directory: &owner.snapshot.input_directory,
                pins: &pins,
            };
            let evidence = owner.spec.validate_external_preflight(
                role,
                &receipt,
                &preflight_directory,
                &inputs,
            )?;
            verify_provider_input_names(owner)?;
            let prefix = owner.snapshot.preflight_runtime_roots[index]
                .as_ref()
                .map_or(
                    if index == 0 {
                        "baseline-runtime"
                    } else {
                        "candidate-runtime"
                    },
                    |root| root.name,
                );
            validate_preflight_provider_evidence(&evidence)?;
            for (relative, bytes) in evidence {
                provider_artifacts.push(artifact(owner, &format!("{prefix}/{relative}"), &bytes));
            }
            crate::native_launch::linux::verify_preflight_runtime_roots(&owner.snapshot)?;
            preflight_runtime_totals(owner)?;
            receipts.push(receipt);
            bundle.process = None;
        }
        Ok((receipts, provider_artifacts))
    }
    pub(super) fn run<S: NativeProviderDeclaration>(
        bundle: &mut NativeRunBundle<S>,
        cancel: Option<&AtomicBool>,
    ) -> Result<(NativePairReceipt<S::Audit>, ArtifactRef), ArenaError> {
        let run_started = std::time::Instant::now();
        let (v6_window, v6_deadline, v6_grace) =
            if let Some(supervisor) = bundle.v6_supervisor.as_ref() {
                let owner = bundle.owner.as_ref().expect("prepared native owner");
                if !bundle.physical_guard_required
                    || owner.spec.input_sha256() != supervisor.execution.pair_sha256()
                {
                    return Err(invalid(
                        "V6 control requires the same guarded original pair",
                    ));
                }
                // A provider may specify a stricter limit; accept it without
                // extending the original run or its cleanup reserve.
                let (run_until, cleanup_until, grace) = v6_effective_deadlines(
                    supervisor.execution.run_until(),
                    supervisor.execution.cleanup_until(),
                    owner.spec.original_wall_deadline()?,
                )?;
                let window = crate::process::OriginalProcessWindow::new(
                    supervisor.execution.started(),
                    run_until,
                    cleanup_until,
                )?;
                (Some(window), Some(run_until), Some(grace))
            } else {
                (None, None, None)
            };
        let run_id = if bundle.physical_guard_required {
            Some(
                GUARDED_EVIDENCE_NEXT_RUN
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                    .map_err(|_| ArenaError::Budget("guarded_evidence_run_id_exhausted".into()))?,
            )
        } else {
            None
        };
        if v6_window.is_none() {
            let owner = bundle.owner.as_mut().expect("prepared native owner");
            owner.snapshot.limits.wall_ms = clamp_original_wall(
                owner.snapshot.limits.wall_ms,
                owner.spec.original_wall_deadline()?,
                std::time::Instant::now(),
            )?;
        }
        bundle
            .owner
            .as_ref()
            .expect("prepared native owner")
            .spec
            .validate_runtime_admission()?;
        let (external_preflight, preflight_artifacts) =
            preflight_endpoints(bundle, cancel, run_id, v6_deadline)?;
        let owner = bundle.owner.as_mut().expect("prepared native owner");
        crate::emit_native_phase("prelaunch_verification_started");
        let mut rolling = SnapshotCacheHint::rolling(
            owner.spec.advise_drop_input_cache(),
            "prelaunch_each_verified",
        );
        let checked = verify_inputs(owner, rolling.as_mut());
        if let Some(status) = &rolling {
            status.emit();
        }
        checked?;
        if let Some(supervisor) = bundle.v6_supervisor.as_ref() {
            supervisor
                .role_sockets
                .as_ref()
                .ok_or_else(|| {
                    ArenaError::Invalid("V6 original role listeners unavailable".into())
                })?
                .verify_listeners()?;
        }
        let mut cache_hints: Vec<_> = owner
            .snapshot
            .preparation_cache_hint
            .clone()
            .into_iter()
            .collect();
        cache_hints.extend(rolling);
        cache_hints.push(advise_verified_input_cache(owner, "before_launch", None));
        crate::emit_native_phase("prelaunch_verification_complete");
        let s = &owner.snapshot;
        let mut ready = crate::native_diagnostics::LogProbe::default();
        let mut on_stdout = |bytes: &[u8]| {
            for game in ready.ready_games(bytes) {
                cache_hints.push(advise_verified_input_cache(
                    owner,
                    "both_engines_ready",
                    Some(game),
                ));
            }
        };
        let mut limits = s.limits;
        if let Some(grace) = v6_grace {
            limits.shutdown_grace_ms = limits.shutdown_grace_ms.max(grace);
        }
        if v6_window.is_none() {
            limits.wall_ms = limits
                .wall_ms
                .checked_sub(u64::try_from(run_started.elapsed().as_millis()).unwrap_or(u64::MAX))
                .filter(|n| *n > 0)
                .ok_or_else(|| invalid("V2 prelaunch consumed whole pair runtime budget"))?;
            limits.wall_ms = clamp_original_wall(
                limits.wall_ms,
                owner.spec.original_wall_deadline()?,
                std::time::Instant::now(),
            )?;
        }
        if bundle.physical_guard_required {
            bundle.probe_evidence = Some(GuardedProbeReservation::acquire(
                run_id.expect("guarded run has evidence identity"),
                owner.spec.input_sha256().to_string(),
                "pair",
                "runner",
                limits.max_output_bytes,
            ));
        }
        let runner = &if let Some(execution) = owner.spec.match_execution() {
            crate::native_launch::linux::pin(&s.pins, &execution.executor)?
        } else {
            &s.pins[s.runner_index]
        }
        .file;
        let process =
            if let (Some(window), Some(supervisor)) = (v6_window, bundle.v6_supervisor.as_mut()) {
                let run_owner = Arc::clone(&supervisor.control);
                let cleanup_owner = Arc::clone(&supervisor.control);
                let cancel_owner = Arc::clone(&supervisor.control);
                let mut run_control = |custody: &crate::process::SpawnCustody, until| {
                    poll_v6_control(&run_owner, custody, until, false)
                };
                let mut cleanup_control = |custody: &crate::process::SpawnCustody, until| {
                    poll_v6_control(&cleanup_owner, custody, until, true)
                };
                let mut pre_signal_control = |custody: &crate::process::SpawnCustody, until| {
                    poll_v6_pre_signal(&cancel_owner, custody, until)
                };
                let pre_signal = supervisor
                    .pre_signal_enabled
                    .then_some(&mut pre_signal_control as crate::process::PreSignalControl<'_>);
                let pre_signal_observation = supervisor
                    .pre_signal_enabled
                    .then_some(&mut supervisor.pre_signal_observation);
                crate::process::supervise_tree_observed_with_v6_controls(
                    runner,
                    &s.invocation.args,
                    &s.cwd,
                    limits,
                    cancel,
                    &s.watch,
                    Some(&mut on_stdout),
                    Some(&mut bundle.spawn_custody),
                    Some(&mut run_control),
                    Some(&mut cleanup_control),
                    Some(window),
                    pre_signal,
                    pre_signal_observation,
                )?
            } else {
                crate::process::supervise_tree_observed_with_spawn(
                    runner,
                    &s.invocation.args,
                    &s.cwd,
                    limits,
                    cancel,
                    &s.watch,
                    Some(&mut on_stdout),
                    if bundle.physical_guard_required {
                        Some(&mut bundle.spawn_custody)
                    } else {
                        None
                    },
                )?
            };
        crate::native_diagnostics::finish_logs();
        crate::emit_native_phase("runner_supervision_finished");
        bundle.process = Some(process);
        if let Some(reservation) = bundle.probe_evidence.as_mut() {
            reservation.process_available = true;
        }
        let owner = bundle
            .owner
            .as_ref()
            .expect("native owner retained after spawn");
        let process = bundle
            .process
            .as_ref()
            .expect("native process output retained after spawn");
        // This independent evidence path runs even when the game fails for an
        // unrelated reason. Failure here preserves physical ownership but does
        // not prevent other available failure/PGN diagnostics being recorded.
        let physical_lifetime_error = verify_guarded_pair(
            owner,
            process,
            &bundle.spawn_custody.pending,
            bundle.physical_guard_required,
        )
        .err()
        .map(|error| error.to_string());
        let completed = process.receipt.stop == ProcessStop::Exited
            && process.receipt.exit_code == Some(0)
            && process.receipt.group_cleanup == CleanupStatus::Gone
            && process.pending_child.is_none()
            && process.receipt.errors.is_empty();
        let mut receipt = NativePairReceipt {
            receipt_version: owner.spec.receipt_version(),
            execution_ready: false,
            strength_eligible: false,
            validation_scope: owner.spec.scope().into(),
            input_sha256: owner.spec.input_sha256().into(),
            pair_id: owner.spec.view().pair_id.into(),
            contract_revision: owner.spec.contract_revision().into(),
            runner_source_commit: owner.spec.view().runner.source_commit.clone(),
            runner_binary_sha256: owner.spec.view().runner.binary.sha256.clone(),
            match_resources: owner.spec.match_execution().map(|e| e.plan()).transpose()?,
            ponder_enabled: owner.spec.match_execution().map(|e| e.ponder),
            process: process.receipt.clone(),
            snapshots: owner.snapshot.receipts.clone(),
            snapshot_cache_hints: cache_hints,
            artifacts: preflight_artifacts,
            provider_sessions: Vec::new(),
            physical_lifetime_error,
            external_preflight,
            external_execution: Vec::new(),
            environments: [NativeEngineRole::Baseline, NativeEngineRole::Candidate]
                .into_iter()
                .map(|role| owner.spec.engine_view(role))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .filter_map(|engine| {
                    engine.environment.map(|environment| {
                        (
                            engine.engine_id.to_owned(),
                            crate::engine_environment::observation(environment),
                        )
                    })
                })
                .collect(),
            external_audit_error: None,
            provider_audit_error: None,
            pgn_audit: None,
            pgn_audit_error: None,
            integration_checks_passed: false,
            scored_games: 0,
            clock_audit: None,
            clock_audit_error: None,
            failure_audit: None,
            failure_audit_error: None,
            incomplete_games: if process.receipt.stop == ProcessStop::Cancelled {
                2
            } else {
                0
            },
            primary_error: None,
            cleanup_verified: process.receipt.group_cleanup == CleanupStatus::Gone
                && process.pending_child.is_none(),
            unresolved_owner_retained: bundle.unresolved(),
            limitations: owner.snapshot.invocation.limitations.clone(),
        };
        // Capture the receipt before each fallible save so original process and
        // cleanup evidence still survive a disk/persistence failure.
        bundle.receipt = Some(receipt.clone());
        receipt.artifacts.push(put(
            owner,
            "stdout.log",
            &process.stdout,
            owner.snapshot.limits.max_output_bytes,
        )?);
        receipt.artifacts.push(put(
            owner,
            "stderr.log",
            &process.stderr,
            owner.snapshot.limits.max_output_bytes,
        )?);
        crate::emit_native_phase("runner_logs_saved");
        let pgn_cap = owner
            .spec
            .view()
            .budget
            .max_output_bytes
            .checked_sub(owner.spec.pair_metadata_cap())
            .and_then(|n| n.checked_sub(owner.snapshot.limits.max_output_bytes))
            .ok_or_else(|| ArenaError::Budget("native PGN reservation underflow".into()))?;
        let pgn_bytes = read_file(&owner.snapshot.directory, "match.pgn", pgn_cap);
        if let Ok(bytes) = &pgn_bytes {
            receipt.artifacts.push(artifact(owner, "match.pgn", bytes));
        }
        // Fastchess may persist config.json. Bound it together with the PGN;
        // absent config is permitted, but a non-regular/invalid existing file is not.
        match owner.snapshot.directory.symlink_metadata("config.json") {
            Ok(meta) => {
                if !meta.is_file() {
                    return Err(invalid("native runner config is not regular"));
                }
                let remaining =
                    pgn_cap.saturating_sub(pgn_bytes.as_ref().map_or(0, |b| b.len() as u64));
                let bytes = read_file(
                    &owner.snapshot.directory,
                    "config.json",
                    remaining.min(NATIVE_PAIR_METADATA_CAP),
                )?;
                receipt
                    .artifacts
                    .push(artifact(owner, "config.json", &bytes));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => return Err(ArenaError::Io("cannot inspect native runner config".into())),
        }
        if completed {
            match pgn_bytes
                .and_then(|bytes| {
                    String::from_utf8(bytes).map_err(|_| invalid("native match PGN is not UTF-8"))
                })
                .and_then(|pgn| {
                    audit_pair_pgn_for_spec(
                        &pair(owner)?,
                        &pgn,
                        PgnLimits {
                            max_bytes: usize::try_from(pgn_cap.min(crate::MAX_JSON_BYTES as u64))
                                .map_err(|_| {
                                ArenaError::Budget("native PGN cap cannot fit host".into())
                            })?,
                            max_plies: owner.spec.view().max_plies,
                        },
                        PgnOutcomePolicy {
                            engine_failure: OutcomePolicy::Loss,
                            max_plies_outcome: OutcomePolicy::Incomplete,
                            claim_policy: owner.spec.claim_policy(),
                            max_game_plies: owner.spec.view().max_plies,
                        },
                    )
                }) {
                Ok(audit) => {
                    receipt.incomplete_games = audit
                        .games
                        .iter()
                        .filter(|g| g.classification == "incomplete")
                        .count() as u32;
                    receipt.pgn_audit = Some(audit)
                }
                Err(error) => receipt.pgn_audit_error = Some(error.to_string()),
            }
            match audit_providers(
                owner,
                &process.stdout,
                &receipt.external_preflight,
                &mut receipt.artifacts,
            ) {
                Ok((sessions, pids)) => {
                    receipt.provider_sessions = sessions;
                    for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
                        if owner.spec.uses_provider_records(role)? {
                            continue;
                        }
                        if let Some(e) = owner.spec.engine_view(role)?.external {
                            let options = owner
                                .spec
                                .external_options(role)?
                                .iter()
                                .map(|(k, v)| {
                                    crate::external_uci::resolve_asset_tokens(
                                        v,
                                        e,
                                        &owner.snapshot.pins,
                                    )
                                    .map(|v| (k.clone(), v))
                                })
                                .collect::<Result<BTreeMap<_, _>, _>>()?;
                            match owner
                                .spec
                                .external_game_process_ids(role, &pids, &owner.snapshot.directory)
                                .and_then(|role_pids| {
                                    crate::external_uci::audit_external_game_protocol(
                                        &process.stdout,
                                        e,
                                        &options,
                                        &role_pids,
                                    )
                                }) {
                                Ok(a) => receipt.external_execution.push(a),
                                Err(e) => receipt.external_audit_error = Some(e.to_string()),
                            }
                        }
                    }
                }
                Err(error) => receipt.provider_audit_error = Some(error.to_string()),
            }
        } else {
            receipt.primary_error = Some(
                "native runner did not exit successfully with verified owned-group cleanup".into(),
            );
            receipt.pgn_audit_error =
                Some("not audited: process completion/cleanup gate failed".into());
            receipt.provider_audit_error =
                Some("not audited: partial provider evidence is not accepted".into());
            match owner
                .spec
                .audit_failure_trace(&process.stdout, &pair(owner)?)
            {
                Ok(audit) => receipt.failure_audit = audit,
                Err(error) => receipt.failure_audit_error = Some(error.to_string()),
            }
        }
        receipt.unresolved_owner_retained = bundle.unresolved();
        bundle.receipt = Some(receipt.clone());
        crate::emit_native_phase("postlaunch_verification_started");
        let mut rolling = SnapshotCacheHint::rolling(
            receipt.cleanup_verified && owner.spec.advise_drop_input_cache(),
            "postcheck_each_verified",
        );
        let checked = verify_inputs(
            bundle
                .owner
                .as_mut()
                .expect("native owner retained through postcheck"),
            rolling.as_mut(),
        );
        if let Some(status) = &rolling {
            status.emit();
        }
        receipt.snapshot_cache_hints.extend(rolling);
        bundle.receipt = Some(receipt.clone());
        checked?;
        crate::emit_native_phase("postlaunch_verification_complete");
        let owner = bundle
            .owner
            .as_ref()
            .expect("native owner retained through postcheck cache hint");
        if receipt.cleanup_verified && owner.spec.advise_drop_input_cache() {
            // Child reads and the final hash audit repopulate these pages.
            // Advise again only after owned cleanup and byte/identity recheck.
            receipt
                .snapshot_cache_hints
                .push(advise_verified_input_cache(owner, "after_postcheck", None));
            receipt.limitations.push(
                "private snapshot cache advice status recorded before launch, per-game both-ready and after owned cleanup/postcheck; kernel reclamation is not guaranteed".into(),
            );
        }
        receipt.integration_checks_passed = completed
            && !bundle.unresolved()
            && receipt.pgn_audit.is_some()
            && receipt.provider_audit_error.is_none()
            && receipt.external_audit_error.is_none()
            && receipt.provider_sessions.len() == owner.spec.expected_provider_sessions();
        let owner = bundle.owner.as_ref().expect("postchecked native owner");
        if let Some(audit) = &receipt.pgn_audit {
            match owner.spec.validate_clock_trace(&process.stdout, audit) {
                Ok(clock) => receipt.clock_audit = clock,
                Err(e) => {
                    receipt.clock_audit_error = Some(e.to_string());
                    receipt.integration_checks_passed = false;
                }
            }
        }
        if receipt.integration_checks_passed && receipt.clock_audit.is_some() {
            receipt.scored_games = receipt.pgn_audit.as_ref().map_or(0, |p| {
                p.games
                    .iter()
                    .filter(|g| {
                        matches!(
                            g.classification.as_str(),
                            "rules_terminal" | "accepted_claim" | "engine_loss"
                        )
                    })
                    .count() as u32
            });
        }
        if completed && !receipt.integration_checks_passed {
            receipt.primary_error =
                Some("native pair evidence audit rejected an integration gate".into());
        }
        bundle.receipt = Some(receipt.clone());
        let bytes = serde_json::to_vec_pretty(&receipt)
            .map_err(|_| invalid("cannot serialize bounded native receipt"))?;
        let artifact = put(
            bundle.owner.as_ref().expect("receipt retains owner"),
            bundle
                .owner
                .as_ref()
                .expect("receipt retains owner")
                .spec
                .receipt_filename(),
            &bytes,
            owner.spec.pair_metadata_cap(),
        )?;
        bundle.receipt_artifact = Some(artifact.clone());
        if let Some(reservation) = bundle.probe_evidence.as_mut() {
            reservation.persistence_failed = false;
        }
        if !bundle.unresolved() {
            bundle.probe_evidence.take();
        }
        crate::emit_native_phase("receipt_saved");
        // Keep receipts/PGN first. Process exit alone grants no NN-drain claim.
        // Each accepted provider session includes its matched physical drain.
        if !bundle.unresolved()
            && crate::native_retention::eligible_expected(
                owner.spec.expected_provider_sessions(),
                receipt.cleanup_verified,
                receipt.provider_audit_error.is_none(),
                receipt.provider_sessions.len(),
            )
        {
            crate::native_retention::retire(
                &mut bundle
                    .owner
                    .as_mut()
                    .expect("saved receipt retains owner")
                    .snapshot,
            )?;
            crate::emit_native_phase("private_input_retirement_complete");
        }
        if !receipt.integration_checks_passed {
            return Err(invalid(
                "native pair failed process, provider or Rules integration acceptance",
            ));
        }
        Ok((receipt, artifact))
    }

    pub(crate) fn json(bytes: &[u8]) -> Result<Value, ArenaError> {
        let text =
            std::str::from_utf8(bytes).map_err(|_| invalid("provider receipt is not UTF-8"))?;
        Ok(rz_experiments::decode_json::<Value>(text)?)
    }
    pub(crate) fn field<'a>(v: &'a Value, key: &str) -> Result<&'a Value, ArenaError> {
        v.get(key).ok_or_else(|| {
            invalid(&format!(
                "provider receipt is missing a required field: {key}"
            ))
        })
    }
    pub(crate) fn number(v: &Value, key: &str) -> Result<u64, ArenaError> {
        field(v, key)?
            .as_u64()
            .ok_or_else(|| invalid("provider receipt integer type differs"))
    }
    pub(crate) fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str, ArenaError> {
        field(v, key)?
            .as_str()
            .ok_or_else(|| invalid("provider receipt string type differs"))
    }
    pub(crate) fn equals(v: &Value, key: &str, expected: &str) -> Result<(), ArenaError> {
        if text(v, key)? != expected {
            Err(invalid("provider receipt locked string identity differs"))
        } else {
            Ok(())
        }
    }
    pub(crate) fn flag(v: &Value, key: &str, expected: bool) -> Result<(), ArenaError> {
        if field(v, key)?.as_bool() != Some(expected) {
            Err(invalid("provider receipt boolean profile differs"))
        } else {
            Ok(())
        }
    }
    pub(crate) fn n(v: &Value, key: &str, expected: u64) -> Result<(), ArenaError> {
        if number(v, key)? != expected {
            Err(invalid("provider receipt numeric profile differs"))
        } else {
            Ok(())
        }
    }
    pub(crate) fn none(v: &Value, key: &str) -> Result<(), ArenaError> {
        if !field(v, key)?.is_null() {
            Err(invalid(
                "provider receipt preserves a failure or unexpected retained state",
            ))
        } else {
            Ok(())
        }
    }
    pub(crate) fn keys(v: &Value, expected: &[&str]) -> Result<(), ArenaError> {
        let map = v
            .as_object()
            .ok_or_else(|| invalid("provider receipt object type differs"))?;
        if map.len() != expected.len() || expected.iter().any(|k| !map.contains_key(*k)) {
            Err(invalid(
                "provider receipt schema has unknown or missing fields",
            ))
        } else {
            Ok(())
        }
    }
    pub(crate) fn registry(v: &Value) -> Result<(), ArenaError> {
        keys(v, &["owner", "slot", "generation", "manifest_sha256"])?;
        if number(v, "owner")? == 0 {
            return Err(invalid("provider registry owner is zero"));
        }
        number(v, "slot")?;
        number(v, "generation")?;
        hash(text(v, "manifest_sha256")?)
    }
    pub(crate) fn hash(value: &str) -> Result<(), ArenaError> {
        if value.len() != 64
            || !value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            Err(invalid("provider digest is not canonical sha256"))
        } else {
            Ok(())
        }
    }
    fn epoch(v: &Value, expected: u64) -> Result<(), ArenaError> {
        keys(v, &["epoch", "sequence"])?;
        n(v, "epoch", expected)?;
        if number(v, "sequence")? == 0 {
            Err(invalid("provider execution/selection sequence is zero"))
        } else {
            Ok(())
        }
    }

    fn check_startup(
        startup: &Value,
        engine: &CpuNativeLaunchSpecV1,
        session: &str,
    ) -> Result<(), ArenaError> {
        keys(
            startup,
            &[
                "schema_version",
                "kind",
                "loaded",
                "process_id",
                "process_run_id",
                "process_epoch",
                "executable",
                "profile",
            ],
        )?;
        n(startup, "schema_version", 1)?;
        equals(startup, "kind", "startup")?;
        flag(startup, "loaded", true)?;
        let pid = number(startup, "process_id")?;
        if pid == 0 || pid > u32::MAX as u64 || session != format!("native-process-{pid}") {
            return Err(invalid(
                "provider session directory and process identity differ",
            ));
        }
        equals(startup, "process_run_id", session)?;
        if number(startup, "process_epoch")? == 0 {
            return Err(invalid("provider process epoch is zero"));
        }
        let executable = field(startup, "executable")?;
        keys(executable, &["sha256", "bytes", "identity_source"])?;
        equals(
            executable,
            "sha256",
            &engine.artifact(NativeArtifactRole::Binary)?.sha256,
        )?;
        n(
            executable,
            "bytes",
            engine.artifact(NativeArtifactRole::Binary)?.bytes,
        )?;
        equals(executable, "identity_source", "linux_proc_self_exe")?;
        let p = field(startup, "profile")?;
        keys(
            p,
            &[
                "source_weights_gzip_sha256",
                "source_weights_protobuf_sha256",
                "onnx_sha256",
                "onnx_bytes",
                "export_manifest_sha256",
                "converter_commit",
                "converter_binary_sha256",
                "ort_library_sha256",
                "ort_build_info_sha256",
                "ort_release",
                "backend_sha256",
                "model",
                "encoding",
                "action_map_sha256",
                "history_policy_sha256",
                "history_length",
                "contract_major",
                "contract_minor",
                "provider",
                "precision",
                "max_batch_items",
                "intra_threads",
                "max_workers",
                "full_steps",
                "min_steps",
                "max_steps",
                "require_full",
                "evaluation_mode",
                "history_fill",
            ],
        )?;
        for (key, role) in [
            (
                "source_weights_gzip_sha256",
                NativeArtifactRole::SourceWeights,
            ),
            ("onnx_sha256", NativeArtifactRole::Onnx),
            ("export_manifest_sha256", NativeArtifactRole::ExportManifest),
            ("ort_library_sha256", NativeArtifactRole::OrtLibrary),
        ] {
            equals(p, key, &engine.artifact(role)?.sha256)?;
        }
        n(
            p,
            "onnx_bytes",
            engine.artifact(NativeArtifactRole::Onnx)?.bytes,
        )?;
        equals(p, "backend_sha256", &engine.profile.expected_backend_sha256)?;
        for key in [
            "source_weights_protobuf_sha256",
            "converter_binary_sha256",
            "ort_build_info_sha256",
            "action_map_sha256",
            "history_policy_sha256",
        ] {
            hash(text(p, key)?)?;
        }
        equals(p, "ort_release", "1.22.0")?;
        equals(p, "provider", "cpu")?;
        equals(p, "precision", "fp32")?;
        equals(p, "evaluation_mode", "fresh")?;
        equals(p, "history_fill", "no")?;
        n(p, "contract_major", 0)?;
        n(p, "contract_minor", 1)?;
        n(p, "history_length", 8)?;
        for key in [
            "max_batch_items",
            "intra_threads",
            "max_workers",
            "full_steps",
            "min_steps",
            "max_steps",
        ] {
            n(p, key, 1)?;
        }
        flag(p, "require_full", true)?;
        let model = field(p, "model")?;
        let encoding = field(p, "encoding")?;
        registry(model)?;
        registry(encoding)?;
        equals(
            model,
            "manifest_sha256",
            &engine.artifact(NativeArtifactRole::ExportManifest)?.sha256,
        )?;
        equals(
            encoding,
            "manifest_sha256",
            &engine.profile.expected_encoding_sha256,
        )?;
        Ok(())
    }
    pub(crate) fn check_completion(c: &Value, startup: &Value) -> Result<(), ArenaError> {
        keys(
            c,
            &[
                "contract_major",
                "contract_minor",
                "request",
                "selection",
                "game",
                "root",
                "state_owner",
                "state_revision",
                "state_sha256",
                "legal_order_sha256",
                "input_sha256",
                "model",
                "encoding",
                "backend_sha256",
                "precision",
                "min_steps",
                "max_steps",
                "require_full",
                "execution",
                "actual",
            ],
        )?;
        let p = field(startup, "profile")?;
        let process_epoch = number(startup, "process_epoch")?;
        n(c, "contract_major", 0)?;
        n(c, "contract_minor", 1)?;
        equals(c, "backend_sha256", text(p, "backend_sha256")?)?;
        equals(c, "precision", "fp32")?;
        n(c, "min_steps", 1)?;
        n(c, "max_steps", 1)?;
        flag(c, "require_full", true)?;
        for key in ["request", "selection", "execution"] {
            epoch(field(c, key)?, process_epoch)?;
        }
        for key in ["game", "root", "state_owner", "state_revision"] {
            number(c, key)?;
        }
        for key in ["state_sha256", "legal_order_sha256", "input_sha256"] {
            hash(text(c, key)?)?;
        }
        if field(c, "model")? != field(p, "model")?
            || field(c, "encoding")? != field(p, "encoding")?
        {
            return Err(invalid(
                "completion registry differs from loaded startup registry",
            ));
        }
        let actual = field(c, "actual")?;
        keys(
            actual,
            &[
                "precision",
                "steps",
                "full",
                "backend_sha256",
                "execution",
                "provenance",
                "prior_execution",
            ],
        )?;
        equals(actual, "precision", "fp32")?;
        n(actual, "steps", 1)?;
        flag(actual, "full", true)?;
        equals(actual, "backend_sha256", text(p, "backend_sha256")?)?;
        equals(actual, "provenance", "computed")?;
        none(actual, "prior_execution")?;
        if field(actual, "execution")? != field(c, "execution")? {
            return Err(invalid(
                "actual computed execution differs from completed context",
            ));
        }
        Ok(())
    }
    fn check_termination(term: &Value, startup: &Value) -> Result<u64, ArenaError> {
        keys(
            term,
            &[
                "schema_version",
                "kind",
                "process_run_id",
                "startup",
                "run_succeeded",
                "actual_cpu_inference_observed",
                "physical_drain",
                "report",
                "original_service_failure",
                "collection_failure",
                "retained_owner_and_evidence",
            ],
        )?;
        n(term, "schema_version", 1)?;
        equals(term, "kind", "termination")?;
        if field(term, "startup")? != startup {
            return Err(invalid(
                "termination startup copy differs from actual startup record",
            ));
        }
        equals(term, "process_run_id", text(startup, "process_run_id")?)?;
        flag(term, "run_succeeded", true)?;
        flag(term, "actual_cpu_inference_observed", true)?;
        equals(term, "physical_drain", "confirmed")?;
        none(term, "original_service_failure")?;
        none(term, "collection_failure")?;
        flag(term, "retained_owner_and_evidence", false)?;
        let report = field(term, "report")?;
        keys(
            report,
            &[
                "model_manifest_sha256",
                "backend_sha256",
                "origin",
                "completed_by_runtime",
                "first_completed",
                "last_completed",
                "completed_scope",
                "canceled",
                "expired",
                "failures",
                "overflow",
                "boundary_error",
                "poison_error",
            ],
        )?;
        let p = field(startup, "profile")?;
        equals(
            report,
            "model_manifest_sha256",
            text(p, "export_manifest_sha256")?,
        )?;
        equals(report, "backend_sha256", text(p, "backend_sha256")?)?;
        equals(report, "origin", "cpu_onnx")?;
        equals(
            report,
            "completed_scope",
            "process_aggregate_normal_poll_first_last_not_full_root_or_physical_journal",
        )?;
        let count = number(report, "completed_by_runtime")?;
        if count == 0 {
            return Err(invalid(
                "loaded process has no D-accepted actual CPU completion",
            ));
        }
        check_completion(field(report, "first_completed")?, startup)?;
        check_completion(field(report, "last_completed")?, startup)?;
        if field(report, "failures")?
            .as_array()
            .is_none_or(|a| !a.is_empty())
        {
            return Err(invalid("native process retains failed inference receipts"));
        }
        for key in ["overflow", "boundary_error", "poison_error"] {
            none(report, key)?;
        }
        for key in ["canceled", "expired"] {
            let a = field(report, key)?;
            keys(a, &["count", "first", "last"])?;
            let count = number(a, "count")?;
            if (count == 0 && (!field(a, "first")?.is_null() || !field(a, "last")?.is_null()))
                || (count > 0 && (field(a, "first")?.is_null() || field(a, "last")?.is_null()))
            {
                return Err(invalid("native audit aggregate presence/count differs"));
            }
            if count > 0 {
                for end in ["first", "last"] {
                    let cause = field(a, end)?;
                    keys(
                        cause,
                        &[
                            "projection",
                            "projection_truncated",
                            "max_depth",
                            "max_nodes",
                            "projected_nodes",
                        ],
                    )?;
                    flag(cause, "projection_truncated", false)?;
                    n(cause, "max_depth", 8)?;
                    n(cause, "max_nodes", 128)?;
                    let nodes = number(cause, "projected_nodes")?;
                    if nodes == 0 || nodes > 128 {
                        return Err(invalid("native audit cause node bound differs"));
                    }
                    let projection = field(cause, "projection")?;
                    equals(projection, "type", "native_diagnostic")?;
                    equals(projection, "kind", "dispatch_refused")?;
                    let physical = field(field(projection, "failure")?, "cause")?;
                    none(physical, "backend")?;
                    let contract = field(field(physical, "contract")?, "cause")?;
                    equals(
                        contract,
                        "code",
                        if key == "canceled" {
                            "canceled"
                        } else {
                            "expired"
                        },
                    )?;
                }
            }
        }
        Ok(count)
    }
    /// Decode and compare bounded public wire fields only. Calling this with
    /// synthetic JSON is not evidence that a process or a neural backend ran.
    pub fn validate_native_provider_record_fields(
        startup_bytes: &[u8],
        termination_bytes: &[u8],
        engine: &CpuNativeLaunchSpecV1,
        session: &str,
    ) -> Result<NativeProviderSessionAudit, ArenaError> {
        if startup_bytes.len() as u64 > PROVIDER_JSON_CAP
            || termination_bytes.len() as u64 > PROVIDER_JSON_CAP
        {
            return Err(ArenaError::Budget(
                "provider wire record exceeds 256KiB".into(),
            ));
        }
        engine.validate()?;
        let startup = json(startup_bytes)?;
        let term = json(termination_bytes)?;
        check_startup(&startup, engine, session)?;
        let completed = check_termination(&term, &startup)?;
        Ok(NativeProviderSessionAudit {
            role: match engine.role {
                NativeEngineRole::Baseline => "baseline",
                NativeEngineRole::Candidate => "candidate",
            }
            .into(),
            process_id: u32::try_from(number(&startup, "process_id")?)
                .map_err(|_| invalid("provider PID exceeds native range"))?,
            process_run_id: session.into(),
            startup_sha256: format!("{:x}", Sha256::digest(startup_bytes)),
            termination_sha256: format!("{:x}", Sha256::digest(termination_bytes)),
            completed_by_runtime: completed,
            actual_cpu_inference_observed: true,
            physical_drain: "confirmed".into(),
        })
    }
    fn runtime_tree(
        directory: &Dir,
        depth: usize,
        files: &mut u32,
        dirs: &mut u32,
        bytes: &mut u64,
        budget: rz_experiments::NativeResourceBudgetV1,
    ) -> Result<(), ArenaError> {
        if depth > budget.max_runtime_depth as usize {
            return Err(ArenaError::Budget("native runtime depth exceeded".into()));
        }
        for entry in directory
            .entries()
            .map_err(|_| invalid("cannot inspect native runtime tree"))?
        {
            let entry = entry.map_err(|_| invalid("native runtime tree entry unavailable"))?;
            let name = entry.file_name();
            let metadata = directory
                .symlink_metadata(&name)
                .map_err(|_| invalid("native runtime tree metadata unavailable"))?;
            if metadata.file_type().is_symlink() {
                return Err(invalid("native runtime tree contains symlink"));
            }
            if metadata.is_dir() {
                *dirs = dirs.checked_add(1).ok_or_else(|| {
                    ArenaError::Budget("native runtime directory count overflow".into())
                })?;
                if files
                    .checked_add(*dirs)
                    .is_none_or(|n| n > budget.max_runtime_files)
                {
                    return Err(ArenaError::Budget(
                        "native runtime file+directory count exceeded".into(),
                    ));
                }
                runtime_tree(
                    &directory
                        .open_dir_nofollow(&name)
                        .map_err(|_| invalid("native runtime directory could not be pinned"))?,
                    depth + 1,
                    files,
                    dirs,
                    bytes,
                    budget,
                )?;
            } else if metadata.is_file() {
                *files = files.checked_add(1).ok_or_else(|| {
                    ArenaError::Budget("native runtime file count overflow".into())
                })?;
                if files
                    .checked_add(*dirs)
                    .is_none_or(|n| n > budget.max_runtime_files)
                {
                    return Err(ArenaError::Budget(
                        "native runtime file+directory count exceeded".into(),
                    ));
                }
                let mut options = OpenOptions::new();
                options
                    .read(true)
                    .follow(FollowSymlinks::No)
                    .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
                let file = directory
                    .open_with(&name, &options)
                    .map_err(|_| invalid("native runtime file could not be pinned"))?
                    .into_std();
                let metadata = file
                    .metadata()
                    .map_err(|_| invalid("native runtime pinned metadata unavailable"))?;
                if !metadata.is_file() || metadata.nlink() != 1 {
                    return Err(invalid("native runtime input is nonregular or hardlinked"));
                }
                *bytes = bytes
                    .checked_add(metadata.len())
                    .ok_or_else(|| ArenaError::Budget("native runtime byte sum overflow".into()))?;
                if *bytes > budget.max_runtime_bytes {
                    return Err(ArenaError::Budget(
                        "native runtime aggregate byte budget exceeded".into(),
                    ));
                }
            } else {
                return Err(invalid("native runtime tree contains nonregular entry"));
            }
        }
        Ok(())
    }
    /// Additional roots and all their children consume the existing shared
    /// runtime entry/byte limits. The periodic whole-attempt watch remains
    /// unchanged; this stage barrier does not create a kernel filesystem quota.
    fn preflight_runtime_totals<S: NativeLaunchDeclaration>(
        owner: &NativeLaunchOwner<S>,
    ) -> Result<(u32, u32, u64), ArenaError> {
        crate::native_launch::linux::verify_preflight_runtime_roots(&owner.snapshot)?;
        if owner
            .snapshot
            .preflight_runtime_roots
            .iter()
            .all(Option::is_none)
        {
            return Ok((0, 0, 0));
        }
        let mut files = 0u32;
        let mut dirs = owner
            .snapshot
            .preflight_runtime_roots
            .iter()
            .flatten()
            .count() as u32;
        let mut bytes = 0u64;
        if dirs > owner.spec.view().budget.max_runtime_files {
            return Err(ArenaError::Budget(
                "native preflight roots exceed runtime entry budget".into(),
            ));
        }
        for root in owner.snapshot.preflight_runtime_roots.iter().flatten() {
            runtime_tree(
                &root.directory,
                0,
                &mut files,
                &mut dirs,
                &mut bytes,
                owner.spec.view().budget,
            )?;
        }
        Ok((files, dirs, bytes))
    }
    #[cfg(test)]
    #[test]
    fn preflight_and_game_runtime_trees_consume_one_unchanged_aggregate() {
        let base = std::env::temp_dir().join(format!(
            "rovezero-preflight-runtime-quota-{}",
            std::process::id()
        ));
        std::fs::create_dir(&base).unwrap();
        for name in ["preflight", "game"] {
            std::fs::create_dir(base.join(name)).unwrap();
            std::fs::write(base.join(name).join("receipt.json"), b"abc").unwrap();
        }
        let directory = Dir::open_ambient_dir(&base, cap_std::ambient_authority()).unwrap();
        let preflight = directory.open_dir_nofollow("preflight").unwrap();
        let game = directory.open_dir_nofollow("game").unwrap();
        let budget = rz_experiments::NativeResourceBudgetV1 {
            max_input_bytes: 1,
            max_output_bytes: 1,
            max_runtime_bytes: 6,
            max_artifact_bytes: 8,
            max_child_processes: 1,
            max_runtime_files: 3,
            max_runtime_depth: 0,
            address_space_per_process_bytes: 0,
        };
        let scan = |budget| {
            // Only the additional preflight root consumes a new fixed runtime
            // directory entry. The historical game-root accounting is unchanged.
            let (mut files, mut dirs, mut bytes) = (0, 1, 0);
            runtime_tree(&preflight, 0, &mut files, &mut dirs, &mut bytes, budget)?;
            runtime_tree(&game, 0, &mut files, &mut dirs, &mut bytes, budget)?;
            Ok::<_, ArenaError>((files, dirs, bytes))
        };
        assert_eq!(scan(budget).unwrap(), (2, 1, 6));
        assert!(matches!(
            scan(rz_experiments::NativeResourceBudgetV1 {
                max_runtime_bytes: 5,
                ..budget
            }),
            Err(ArenaError::Budget(_))
        ));
        assert!(matches!(
            scan(rz_experiments::NativeResourceBudgetV1 {
                max_runtime_files: 2,
                ..budget
            }),
            Err(ArenaError::Budget(_))
        ));
        std::fs::create_dir(base.join("preflight/nested")).unwrap();
        assert!(matches!(scan(budget), Err(ArenaError::Budget(_))));
        std::fs::remove_dir(base.join("preflight/nested")).unwrap();
        std::os::unix::fs::symlink(base.join("game/receipt.json"), base.join("preflight/alias"))
            .unwrap();
        assert!(matches!(scan(budget), Err(ArenaError::Integrity(_))));
        std::fs::remove_file(base.join("preflight/alias")).unwrap();
        std::fs::hard_link(base.join("game/receipt.json"), base.join("preflight/alias")).unwrap();
        assert!(matches!(scan(budget), Err(ArenaError::Integrity(_))));
        drop(preflight);
        drop(game);
        drop(directory);
        std::fs::remove_dir_all(base).unwrap();
    }
    pub(super) fn verify_conversion_provenance(
        startup: &Value,
        manifest: &Value,
        batch_experiment: bool,
    ) -> Result<(), ArenaError> {
        // The validated launch declaration selects the wire. V4 wraps both B1
        // and B4 provider profiles; legacy CPU/CUDA records remain unwrapped.
        let provider = if batch_experiment {
            field(startup, "provider")?
        } else {
            startup
        };
        let profile = field(provider, "profile")?;
        for (observed, declared) in [
            ("source_weights_protobuf_sha256", "source_protobuf_sha256"),
            ("converter_commit", "converter_commit"),
            ("converter_binary_sha256", "converter_binary_sha256"),
        ] {
            equals(profile, observed, text(manifest, declared)?)?;
        }
        Ok(())
    }
    #[cfg(test)]
    #[test]
    fn export_provenance_uses_declared_batch_envelope_and_rejects_mismatch() {
        let manifest = serde_json::json!({
            "source_protobuf_sha256": "a".repeat(64),
            "converter_commit": "b".repeat(40),
            "converter_binary_sha256": "c".repeat(64)
        });
        let provider = serde_json::json!({"profile": {
            "source_weights_protobuf_sha256": manifest["source_protobuf_sha256"],
            "converter_commit": manifest["converter_commit"],
            "converter_binary_sha256": manifest["converter_binary_sha256"]
        }});
        assert!(verify_conversion_provenance(&provider, &manifest, false).is_ok());
        let wrapped = serde_json::json!({"provider": provider});
        assert!(verify_conversion_provenance(&wrapped, &manifest, true).is_ok());
        assert!(verify_conversion_provenance(&wrapped, &manifest, false).is_err());
        assert!(verify_conversion_provenance(&provider, &manifest, true).is_err());
        for field in [
            "source_weights_protobuf_sha256",
            "converter_commit",
            "converter_binary_sha256",
        ] {
            let mut wrong = wrapped.clone();
            wrong["provider"]["profile"][field] = serde_json::json!("0");
            assert!(verify_conversion_provenance(&wrong, &manifest, true).is_err());
        }
    }
    fn audit_providers<S: NativeProviderDeclaration>(
        owner: &NativeLaunchOwner<S>,
        stdout: &[u8],
        preflight: &[crate::ExternalUciPreflight],
        artifacts: &mut Vec<ArtifactRef>,
    ) -> Result<(Vec<S::Audit>, Vec<u32>), ArenaError> {
        let mut sessions = Vec::new();
        let mut ids = BTreeSet::new();
        let mut scanned = 0u32;
        let (mut tree_files, mut tree_dirs, mut tree_bytes) = preflight_runtime_totals(owner)?;
        for (role, name) in [
            (NativeEngineRole::Baseline, "baseline-runtime"),
            (NativeEngineRole::Candidate, "candidate-runtime"),
        ] {
            let engine = owner.spec.engine_view(role)?;
            if !owner.spec.uses_provider_records(role)? {
                continue;
            }
            let directory = owner
                .snapshot
                .directory
                .open_dir_nofollow(name)
                .map_err(|_| invalid("role runtime directory is unavailable"))?;
            runtime_tree(
                &directory,
                0,
                &mut tree_files,
                &mut tree_dirs,
                &mut tree_bytes,
                owner.spec.view().budget,
            )?;
            let mut session_names = Vec::new();
            for entry in directory
                .entries()
                .map_err(|_| invalid("cannot inspect native role runtime entries"))?
            {
                scanned = scanned
                    .checked_add(1)
                    .ok_or_else(|| ArenaError::Budget("native session scan overflow".into()))?;
                if scanned > owner.spec.view().budget.max_runtime_files {
                    return Err(ArenaError::Budget(
                        "native session scan entry bound exceeded".into(),
                    ));
                }
                let entry =
                    entry.map_err(|_| invalid("native role runtime entry is unavailable"))?;
                let os = entry.file_name();
                let text = os
                    .to_str()
                    .ok_or_else(|| invalid("native role runtime name is not UTF-8"))?;
                if text.starts_with("native-process-") {
                    session_names.push(text.to_owned());
                }
            }
            session_names.sort();
            if session_names.len() != 2 {
                return Err(invalid(
                    "each role requires exactly two fresh native process sessions",
                ));
            }
            for session in session_names {
                let dir = directory
                    .open_dir_nofollow(&session)
                    .map_err(|_| invalid("native session is not a real owned directory"))?;
                let startup_filename = owner.spec.role_startup_filename(role);
                let termination_filename = owner.spec.role_termination_filename(role);
                let startup_bytes = read_file(&dir, startup_filename, PROVIDER_JSON_CAP)?;
                let termination_bytes = read_file(&dir, termination_filename, PROVIDER_JSON_CAP)?;
                artifacts.push(artifact(
                    owner,
                    &format!("{name}/{session}/{startup_filename}"),
                    &startup_bytes,
                ));
                artifacts.push(artifact(
                    owner,
                    &format!("{name}/{session}/{termination_filename}"),
                    &termination_bytes,
                ));
                verify_provider_input_names(owner)?;
                let pins = provider_input_views(owner);
                let inputs = NativeProviderInputContext {
                    directory: &owner.snapshot.input_directory,
                    pins: &pins,
                };
                let (audit, pid) = owner.spec.validate_records_with_context(
                    role,
                    &startup_bytes,
                    &termination_bytes,
                    &session,
                    &inputs,
                    stdout,
                )?;
                verify_provider_input_names(owner)?;
                if !ids.insert(pid) {
                    return Err(invalid("native pair reused a process identity"));
                }
                for (relative, bytes) in
                    owner
                        .spec
                        .verify_session_evidence(role, pid, &startup_bytes, &directory)?
                {
                    artifacts.push(artifact(owner, &format!("{name}/{relative}"), &bytes));
                }
                // Compare conversion provenance with the exact copied export
                // manifest bytes rather than trusting additional receipt strings.
                if let Some(export) = owner.spec.provider_export_artifact(role)? {
                    let export = pin(&owner.snapshot.pins, export)?;
                    let manifest_bytes = read_bytes(
                        export
                            .file
                            .try_clone()
                            .map_err(|_| invalid("cannot clone export pin"))?,
                        64 * 1024,
                    )?;
                    owner.spec.validate_conversion_manifest(
                        role,
                        &startup_bytes,
                        &manifest_bytes,
                        engine.batch_experiment.is_some(),
                    )?;
                }
                sessions.push(audit);
            }
        }
        let expected_pids: Vec<_> = ids.into_iter().collect();
        owner
            .spec
            .validate_process_exit_trace(stdout, &expected_pids)?;
        let external = owner.spec.external_exit_ids(stdout, &expected_pids)?;
        verify_provider_input_names(owner)?;
        let pins = provider_input_views(owner);
        let inputs = NativeProviderInputContext {
            directory: &owner.snapshot.input_directory,
            pins: &pins,
        };
        owner.spec.validate_provider_session_set(
            &sessions,
            preflight,
            &owner.snapshot.directory,
            &inputs,
            stdout,
            artifacts,
        )?;
        verify_provider_input_names(owner)?;
        Ok((sessions, external))
    }
}
