//! Native UCI upper runner. It owns the same delegated resource scope before
//! the shared launch loads any model, and retains it after physical shutdown
//! or any failure. UCI parsing, native tickets and engine assembly stay in UCI.
use super::*;
#[cfg(target_os = "linux")]
use rz_experiments::NativeRuntimeSourceV6;
use rz_uci::rovent_bootstrap::launch::{
    RoventLaunchError, run_with_resource_admission, run_with_resource_admission_v1_v2,
};
use std::{
    io::{Read, Write},
    path::PathBuf,
    sync::{Mutex, OnceLock},
};

pub const ROVENT_RESOURCE_LAUNCH_PLAN: &str = "rz-rovent-native-resource-launch/1";
pub const ROVENT_RESOURCE_USAGE: &str = "rz-rovent-native [--resource-system-runtime-v3 [--custody-control]] --resource-plan=inputs/<plan.json> --resource-plan-sha256=<64 lowercase hex> --resource-plan-bytes=<positive bytes> <original Rovent source/config/FD arguments>";

/// Partial diagnostic projection only. The exact event remains in its owner;
/// unsupported or oversized details never become release authority.
fn observed_startup_failure(
    failure: &rz_search::pals::coordinator::RoleFailure,
    remaining: &mut usize,
) -> Value {
    use rz_search::pals::coordinator::RoleFailure;
    match failure {
        RoleFailure::Canceled => json!({"kind":"canceled"}),
        RoleFailure::Expired => json!({"kind":"expired"}),
        RoleFailure::InvalidOutput => json!({"kind":"invalid_output"}),
        RoleFailure::PhysicalUnknown => json!({"kind":"physical_unknown"}),
        RoleFailure::Backend(detail) => observed_failure_detail("backend", detail, remaining),
        RoleFailure::Unsupported(detail) => {
            observed_failure_detail("unsupported", detail, remaining)
        }
    }
}

fn observed_failure_detail(kind: &str, detail: &str, remaining: &mut usize) -> Value {
    if detail.len() > *remaining {
        return json!({"kind":kind, "detail_retained_by_owner":true});
    }
    *remaining -= detail.len();
    json!({"kind":kind, "detail":detail})
}
const MAX_PLAN_BYTES: u64 = 64 << 10;

/// Configuration only. The actual resource owner separately validates the
/// open cgroup descriptor, original lock, leaf, kernel controls and current PID.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RoventNativeResourceLaunchPlan {
    pub schema: String,
    pub delegated_cgroup_path: PathBuf,
    pub leaf_name: String,
    pub options: RoventNativeResourceOptions,
    pub profile: RoventNativeResourceProfile,
    /// A fresh directory below the original private working directory. Its
    /// ancestors must exist; this runner never reuses or deletes an output.
    pub evidence_directory: PathBuf,
}
impl RoventNativeResourceLaunchPlan {
    pub fn validate(&self) -> Result<(), ArenaError> {
        if self.schema != ROVENT_RESOURCE_LAUNCH_PLAN
            || !self.delegated_cgroup_path.is_absolute()
            || self
                .delegated_cgroup_path
                .components()
                .any(|component| !matches!(component, Component::RootDir | Component::Normal(_)))
            || self.delegated_cgroup_path.as_os_str().len() > 4096
            || !self.leaf_name.starts_with("rovent-")
            || !(8..=96).contains(&self.leaf_name.len())
            || !self
                .leaf_name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
            || !relative(&self.evidence_directory)
        {
            return Err(invalid("explicit bounded native resource launch plan"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RoventNativeResourcePlanPin {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}
impl RoventNativeResourcePlanPin {
    fn validate(&self) -> Result<(), ArenaError> {
        if !relative(&self.path)
            || !self.path.starts_with("inputs")
            || !(1..=MAX_PLAN_BYTES).contains(&self.bytes)
            || self.sha256.len() != 64
            || !self
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(invalid("original resource plan pin required"));
        }
        Ok(())
    }
}

pub enum RoventNativeResourceCli {
    Help,
    BuildCapabilities,
    SystemBuildCapabilities,
    Run {
        pin: RoventNativeResourcePlanPin,
        arguments: Vec<String>,
    },
    /// Explicit /3 original-lock + V6 system runtime path; never project a
    /// historical lock or infer this selection from an omitted CUDA bundle.
    RunSystemV3 {
        pin: RoventNativeResourcePlanPin,
        arguments: Vec<String>,
        custody_control: bool,
    },
}
pub fn parse_native_resource_cli(
    arguments: Vec<String>,
) -> Result<RoventNativeResourceCli, ArenaError> {
    if arguments == ["--help"] {
        return Ok(RoventNativeResourceCli::Help);
    }
    if arguments == ["--build-capabilities-json"] {
        return Ok(RoventNativeResourceCli::BuildCapabilities);
    }
    if arguments == ["--build-system-runtime-capabilities-json"] {
        return Ok(RoventNativeResourceCli::SystemBuildCapabilities);
    }
    let (mut path, mut sha256, mut bytes) = (None, None, None);
    let mut system_v3 = false;
    let mut custody_control = false;
    let mut original = Vec::new();
    for argument in arguments {
        if argument == "--resource-system-runtime-v3" {
            if std::mem::replace(&mut system_v3, true) {
                return Err(invalid("duplicate system runtime resource selection"));
            }
        } else if argument == "--custody-control" {
            if std::mem::replace(&mut custody_control, true) {
                return Err(invalid("duplicate direct custody control selection"));
            }
        } else if let Some(value) = argument.strip_prefix("--resource-plan=") {
            if path.replace(PathBuf::from(value)).is_some() {
                return Err(invalid("duplicate resource plan"));
            }
        } else if let Some(value) = argument.strip_prefix("--resource-plan-sha256=") {
            if sha256.replace(value.to_string()).is_some() {
                return Err(invalid("duplicate resource plan digest"));
            }
        } else if let Some(value) = argument.strip_prefix("--resource-plan-bytes=") {
            let value = value.parse().map_err(|_| invalid("resource plan bytes"))?;
            if bytes.replace(value).is_some() {
                return Err(invalid("duplicate resource plan bytes"));
            }
        } else {
            original.push(argument);
        }
    }
    let pin = RoventNativeResourcePlanPin {
        path: path.ok_or_else(|| invalid("resource plan path missing"))?,
        sha256: sha256.ok_or_else(|| invalid("resource plan digest missing"))?,
        bytes: bytes.ok_or_else(|| invalid("resource plan bytes missing"))?,
    };
    pin.validate()?;
    if custody_control && !system_v3 {
        return Err(invalid(
            "direct custody control requires explicit V3 selection",
        ));
    }
    Ok(if system_v3 {
        RoventNativeResourceCli::RunSystemV3 {
            pin,
            arguments: original,
            custody_control,
        }
    } else {
        RoventNativeResourceCli::Run {
            pin,
            arguments: original,
        }
    })
}

/// The success receipt also keeps the original scope/FDs alive. The returned
/// observation is a current snapshot, never post-process final peak evidence.
pub struct RoventNativeResourceRunReceipt {
    pub admission: RoventResourceObservation,
    pub finished_live: RoventResourceObservation,
    custody: Arc<Mutex<NativeResourceCustody>>,
}
impl RoventNativeResourceRunReceipt {
    pub fn observation(&self) -> Value {
        let evidence_directory = self
            .custody
            .lock()
            .ok()
            .map(|state| state.plan.evidence_directory.clone());
        json!({"schema":"rz-rovent-native-resource-run/1",
            "admission":self.admission,"finished_live":self.finished_live,
            "evidence_directory":evidence_directory,
            "original_resource_owner_retained":true,"controlled_comparison_accepted":false,
            "training_executed":false})
    }
    pub fn original_scope_matches(&self, lock: &Arc<RoventInputLock>) -> bool {
        self.custody.lock().ok().is_some_and(|state| {
            state
                .scope
                .as_ref()
                .is_some_and(|scope| scope.matches_input_lock(lock))
        })
    }
}

/// Versioned product receipt: V2 diagnostic persistence may be unavailable
/// even when the original resource policy was admitted and the product ran.
pub struct RoventNativeResourceRunReceiptV2 {
    pub admission: Option<RoventResourceObservation>,
    pub finished_live: Option<RoventResourceObservation>,
    custody: Arc<Mutex<NativeResourceCustody>>,
}
pub enum RoventNativeResourceRunReceiptVersioned {
    V1(RoventNativeResourceRunReceipt),
    V2(RoventNativeResourceRunReceiptV2),
}
impl RoventNativeResourceRunReceiptVersioned {
    pub fn observation(&self) -> Value {
        match self {
            Self::V1(receipt) => receipt.observation(),
            Self::V2(receipt) => match receipt.custody.lock() {
                Ok(state) => json!({"schema":"rz-rovent-native-resource-run/2",
                    "admission":receipt.admission,"finished_live":receipt.finished_live,
                    "admission_evidence_error":state.admission_evidence_error,
                    "finished_evidence_error":state.finished_evidence_error,
                    "finished_policy_observation_error":state.finished_live_error,
                    "evidence_directory":state.plan.evidence_directory,
                    "original_resource_owner_retained":true,
                    "controlled_comparison_accepted":false,"training_executed":false}),
                Err(_) => json!({"schema":"rz-rovent-native-resource-run/2",
                    "resource_state_poisoned":true,"original_resource_owner_retained":true}),
            },
        }
    }
}

/// V3 system execution retains the original resource owner and the exact
/// child observations. A live in-process resource readback is not a separate
/// post-exit process-tree or independent GPU completion observation.
pub struct RoventNativeSystemResourceRunReceiptV3 {
    pub admission: Option<RoventResourceObservation>,
    pub finished_live: Option<RoventResourceObservation>,
    pub admission_evidence_error: Option<String>,
    pub finished_evidence_error: Option<String>,
    pub finished_policy_observation_error: Option<String>,
    pub control_cleanup_error: Option<String>,
    startup: Option<rz_experiments::SystemRuntimeReceiptV6>,
    termination: Option<rz_experiments::SystemRuntimeReceiptV6>,
    system_runtime_evidence_error: Option<String>,
    system_lock_sha256: String,
    process_id: u32,
    process_generation: u64,
    product_finish: Option<rz_uci::rovent_bootstrap::launch::V3ProductFinishObservation>,
    custody: Arc<Mutex<NativeResourceCustody>>,
}
impl RoventNativeSystemResourceRunReceiptV3 {
    pub fn child_system_runtime_receipts_validated(&self) -> bool {
        self.startup.is_some()
            && self.termination.is_some()
            && self.system_runtime_evidence_error.is_none()
    }
    pub fn observation(&self) -> Value {
        json!({"schema":"rz-rovent-native-system-resource-run/3",
            "admission":self.admission,"finished_live":self.finished_live,
            "admission_evidence_error":self.admission_evidence_error,
            "finished_evidence_error":self.finished_evidence_error,
            "finished_policy_observation_error":self.finished_policy_observation_error,
            "control_cleanup_error":self.control_cleanup_error,
            "system_lock_sha256":self.system_lock_sha256,
            "process_id":self.process_id,"process_generation":self.process_generation,
            "product_finish":self.product_finish.as_ref().map(product_finish_json),
            "startup_provider_ready":self.startup.as_ref().map(|r|r.provider_ready),
            "child_terminal_physical_completion":self.termination.as_ref().map(|r|&r.physical_completion),
            "child_system_runtime_receipts_validated":self.child_system_runtime_receipts_validated(),
            "system_runtime_evidence_error":self.system_runtime_evidence_error,
            "original_resource_owner_retained":self.custody.lock().ok().is_some_and(|state|state.scope_v3.is_some()),
            "independent_post_exit_process_observation":false,
            "independent_gpu_fence_observation":false,
            "controlled_comparison_accepted":false,"training_executed":false})
    }
}

fn product_finish_json(
    observation: &rz_uci::rovent_bootstrap::launch::V3ProductFinishObservation,
) -> Value {
    json!({"schema":"rz-rovent-product-finish-observation/1",
        "process_id":observation.process_id,"process_epoch":observation.process_epoch,
        "input_lock_sha256":observation.input_lock_sha256,
        "state":"confirmed_by_original_product", "device_fence":false})
}

pub struct RoventNativeResourceRunFailure {
    pub detail: String,
    custody: Option<Arc<Mutex<NativeResourceCustody>>>,
    system_v3: bool,
    native_failure: Option<rz_uci::rovent_bootstrap::launch::RoventRetainingLaunchError>,
    native_recovery_attempts: Vec<rz_uci::rovent_bootstrap::launch::V3BeforeNativeScopeRecovery>,
    control_panic: Option<Box<dyn std::any::Any + Send>>,
    control_panic_may_be_native: bool,
    control_scope_marker: Option<Arc<OnceLock<rz_uci::rovent_bootstrap::launch::RoventV3ScopeKey>>>,
    pre_native_admission: Option<PreNativeAdmissionEvidence>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AdmissionCheckpointV3 {
    NotStarted,
    ProvenanceRejected,
    ResourcePrepareStarted,
}

/// This proves only that this controlled call returned before native scope
/// construction. It is not process exit, a device fence, or another owner’s
/// completion proof.
#[derive(Clone, Copy, serde::Serialize)]
struct PreNativeAdmissionEvidence {
    schema: &'static str,
    process_id: u32,
    process_generation: u64,
    reason: &'static str,
    native_scope_started: bool,
}

/// The actual native/cgroup failure and the exact endpoint created for the
/// same controlled run. Neither this wrapper nor a client request is a fence.
pub struct RoventDirectQuarantine {
    original_failure: RoventNativeResourceRunFailure,
    original_control: Arc<Mutex<super::direct_control::BoundDirectControl>>,
}

/// Opt-in direct control keeps pre-control failures distinct from a native
/// quarantine that necessarily owns the same bound endpoint.
pub enum RoventControlledRunFailure {
    BeforeControl(RoventNativeResourceRunFailure),
    Failed(RoventNativeResourceRunFailure),
    Quarantined(RoventDirectQuarantine),
}

enum V3InnerFailure {
    BeforeControl(RoventNativeResourceRunFailure),
    Failed(RoventNativeResourceRunFailure),
    Quarantined(RoventDirectQuarantine),
}

impl From<RoventNativeResourceRunFailure> for V3InnerFailure {
    fn from(failure: RoventNativeResourceRunFailure) -> Self {
        Self::Failed(failure)
    }
}

impl RoventDirectQuarantine {
    /// The process remains a visible, reconnectable foreground owner until a
    /// separately implemented same-owner completion path can release it.
    pub fn serve(mut self) -> ! {
        let key = self.original_failure.original_scope_key().cloned();
        let cleanup_until = key
            .as_ref()
            .and_then(|key| key.cleanup_until())
            .unwrap_or_else(std::time::Instant::now);
        let _ = writeln!(
            std::io::stderr().lock(),
            "{}",
            self.original_failure.observation()
        );
        let socket_identity = self
            .original_control
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .observe_bound_identity();
        let _ = writeln!(
            std::io::stderr().lock(),
            "{}",
            json!({
                "schema":"rz-rovent-direct-custody-quarantined/1",
                "process_id":std::process::id(), "physical_completion":"unknown",
                "reconnectable":"unknown", "socket_identity_verified":socket_identity.is_ok(),
                "socket_identity_error":socket_identity.err().map(|error| error.to_string()),
                "release_authorized":false,
            })
        );
        let mut last_error: Option<String> = None;
        let mut error_count = 0u64;
        loop {
            // The exact owner and listener stay alive even when one client
            // disconnects or requests malformed data. This is an explicit
            // registered quarantine, not a new native execution or a fence.
            let failure = &mut self.original_failure;
            let mut recover = |until| {
                let original_error = (failure.detail.len() <= super::MAX_EVIDENCE_BYTES)
                    .then(|| failure.detail.clone());
                let oversize_error_retained_by_owner = original_error.is_none();
                failure.try_recover_before_native_until(until).map(|result| {
                use rz_uci::rovent_bootstrap::launch::V3BeforeNativeScopeRecovery as Recovery;
                match result {
                    Recovery::Missing => json!({"state":"missing"}),
                    Recovery::Busy => json!({"state":"busy"}),
                    Recovery::Poisoned => json!({"state":"poisoned"}),
                    Recovery::Expired => json!({"state":"expired"}),
                    Recovery::WrongState => json!({"state":"wrong_state"}),
                    Recovery::WrongType => json!({"state":"wrong_type"}),
                    Recovery::Pending(error) => json!({"state":"pending", "error":format!("{error:?}")}),
                    Recovery::PhysicalPending(error) => json!({"state":"physical_reservation_pending", "error":format!("{error:?}")}),
                    Recovery::Panic(_) => json!({"state":"panic", "original_payload_retained":true}),
                    Recovery::LostRegistration => json!({"state":"registration_lost"}),
                    Recovery::InvariantViolation { pins, .. } => json!({"state":"invariant_violation", "original_owner_retained":true, "pins_retained":pins.len()}),
                    Recovery::Recovered { pending, pins } => {
                        use rz_uci::rovent_bootstrap::launch::V3FactoryOriginalEvent as Event;
                        let mut native = 0usize;
                        let mut reports = 0usize;
                        let mut snapshots = 0usize;
                        let mut panics = 0usize;
                        let mut opaque = 0usize;
                        let mut details_remaining = super::MAX_EVIDENCE_BYTES / 2;
                        let mut event_observations = Vec::new();
                        let max_observations = super::MAX_EVIDENCE_BYTES / std::mem::size_of::<Value>().max(1);
                        for event in pending.events() {
                            use rz_uci::pals_native::ticketed::TicketNativeEvent;
                            if event_observations.len() < max_observations {
                                let observation = match event {
                                    Event::Native(TicketNativeEvent::StartupFailed(failure)) => json!({"kind":"startup_failed", "failure":observed_startup_failure(failure, &mut details_remaining)}),
                                    Event::Native(TicketNativeEvent::Finished { physical_shutdown_confirmed }) => json!({"kind":"finished", "physical_shutdown_confirmed":physical_shutdown_confirmed}),
                                    Event::Native(_) => json!({"kind":"other_native_event_retained"}),
                                    Event::FinishReport(_) => json!({"kind":"finish_report_retained"}),
                                    Event::FinishReportSnapshot(_) => json!({"kind":"finish_report_snapshot_retained"}),
                                    Event::Panic(_) => json!({"kind":"panic_retained"}),
                                    Event::Opaque(_) => json!({"kind":"opaque_retained"}),
                                };
                                event_observations.push(observation);
                            }
                            match event {
                                Event::Native(_) => native += 1,
                                Event::FinishReport(_) => reports += 1,
                                Event::FinishReportSnapshot(_) => snapshots += 1,
                                Event::Panic(_) => panics += 1,
                                Event::Opaque(_) => opaque += 1,
                            }
                        }
                        json!({"state":"same_owner_recovered_and_retained",
                            "original_launch_error":original_error,
                            "oversize_error_retained_by_owner":oversize_error_retained_by_owner,
                            "before_native_proven":pending.before_native_proven(),
                            "original_failure_retained":pending.original_failure().is_some(),
                            "original_context_retained":pending.has_original_context(),
                            "factory_retained":pending.has_recovered_factory(),
                            "first_recovery_error":pending.first_recovery_error().map(|error| format!("{error:?}")),
                            "physical_reservation_pending":pending.physical_reservation_pending(),
                            "pins_retained":pins.len(),
                            "event_counts":{"native":native,"finish_report":reports,"finish_report_snapshot":snapshots,"panic":panics,"opaque":opaque},
                            "event_observations_omitted":pending.events().len().saturating_sub(event_observations.len()),
                            "event_observations":event_observations,
                            "complete_event_export":false, "release_authorized":false})
                    },
                }
                })
            };
            let result = self
                .original_control
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .observe_one(key.as_ref(), cleanup_until, &mut recover);
            match result {
                Err(error) => {
                    let detail = error.to_string();
                    if last_error.as_deref() == Some(detail.as_str()) {
                        error_count = error_count.saturating_add(1);
                    } else {
                        if let Some(previous) = last_error.replace(detail.clone()) {
                            let _ = writeln!(
                                std::io::stderr().lock(),
                                "direct custody prior observation error repeated {error_count} time(s): {previous}"
                            );
                        }
                        error_count = 1;
                        let _ = writeln!(
                            std::io::stderr().lock(),
                            "direct custody observation failed: {detail}"
                        );
                    }
                    std::thread::sleep(std::time::Duration::from_secs(1));
                }
                Ok(()) => {
                    if let Some(previous) = last_error.take() {
                        let _ = writeln!(
                            std::io::stderr().lock(),
                            "direct custody observation resumed after {error_count} error(s): {previous}"
                        );
                    }
                    error_count = 0;
                }
            }
        }
    }
}
impl std::fmt::Debug for RoventNativeResourceRunFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoventNativeResourceRunFailure")
            .field("detail", &self.detail)
            .field("original_custody_retained", &self.custody.is_some())
            .field("native_quarantined", &self.native_quarantined())
            .finish()
    }
}
impl std::fmt::Display for RoventNativeResourceRunFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.detail.fmt(f)
    }
}
impl std::error::Error for RoventNativeResourceRunFailure {}
impl RoventNativeResourceRunFailure {
    /// Retains the original native failure together with the resource scope.
    /// Merely inspecting this value is not a physical-completion proof.
    pub fn native_failure(
        &self,
    ) -> Option<&rz_uci::rovent_bootstrap::launch::RoventRetainingLaunchError> {
        self.native_failure.as_ref()
    }

    /// The caller must keep the entire failure, including its original
    /// resource owner, for as long as the native scope is quarantined.
    /// This observation itself does not authorize exit or resource release.
    pub fn native_quarantined(&self) -> bool {
        self.control_panic_may_be_native
            || matches!(
                self.native_failure.as_ref(),
                Some(
                    rz_uci::rovent_bootstrap::launch::RoventRetainingLaunchError::Quarantined { .. }
                )
            )
    }

    /// A panic payload is retained exactly; this boolean is not a fence.
    pub fn original_control_panic_retained(&self) -> bool {
        self.control_panic.is_some()
    }

    /// Exact scope of this attempt, when it was actually reserved. A key is
    /// an observation capability, never proof of completion or release.
    pub fn original_scope_key(
        &self,
    ) -> Option<&rz_uci::rovent_bootstrap::launch::RoventV3ScopeKey> {
        match self.native_failure.as_ref() {
            Some(rz_uci::rovent_bootstrap::launch::RoventRetainingLaunchError::Quarantined {
                owner,
                ..
            }) => Some(owner.scope_key()),
            _ => self
                .control_scope_marker
                .as_ref()
                .and_then(|marker| marker.get()),
        }
    }

    /// Move a quarantined run together with its original control. If any
    /// capability is missing the original failure is returned intact.
    pub fn into_direct_quarantine(self) -> Result<RoventDirectQuarantine, Self> {
        if !self.native_quarantined() {
            return Err(self);
        }
        let control = self.custody.as_ref().and_then(|owner| {
            owner
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .bound_control
                .clone()
        });
        match control {
            Some(original_control) => Ok(RoventDirectQuarantine {
                original_failure: self,
                original_control,
            }),
            None => Err(self),
        }
    }

    /// Original recovery results remain inspectable, including opaque owners.
    pub fn native_recovery_attempts(
        &self,
    ) -> &[rz_uci::rovent_bootstrap::launch::V3BeforeNativeScopeRecovery] {
        &self.native_recovery_attempts
    }

    /// Retry only the original pre-native reservation recovery, using its
    /// original cleanup deadline. All returned owners, events and failures
    /// remain in this failure value until its caller deliberately consumes it.
    /// No result here establishes completion of other native or GPU work.
    pub fn try_recover_before_native(
        &mut self,
    ) -> Option<&rz_uci::rovent_bootstrap::launch::V3BeforeNativeScopeRecovery> {
        let until = self.original_scope_key()?.cleanup_until()?;
        self.try_recover_before_native_until(until)
    }

    fn try_recover_before_native_until(
        &mut self,
        requested_until: std::time::Instant,
    ) -> Option<&rz_uci::rovent_bootstrap::launch::V3BeforeNativeScopeRecovery> {
        use rz_uci::rovent_bootstrap::launch::RoventRetainingLaunchError;
        use rz_uci::rovent_bootstrap::launch::V3BeforeNativeScopeRecovery as Recovery;
        // Terminal outcomes may carry original, non-serializable ownership.
        // Keep the first such result and never overwrite it on re-observation.
        if let Some(index) = self.native_recovery_attempts.iter().position(|result| {
            matches!(
                result,
                Recovery::Recovered { .. }
                    | Recovery::Panic(_)
                    | Recovery::InvariantViolation { .. }
                    | Recovery::LostRegistration
            )
        }) {
            return self.native_recovery_attempts.get(index);
        }
        let Some(RoventRetainingLaunchError::Quarantined { owner, .. }) =
            self.native_failure.as_ref()
        else {
            return None;
        };
        // The original scope validates its typed before-native state itself;
        // an error string is not a recovery authority.
        let deadline = owner.cleanup_until().min(requested_until);
        if std::time::Instant::now() >= deadline {
            return None;
        }
        let result = owner.scope_key().recover_before_native(deadline);
        // Non-owning typed observations have a finite domain. Coalesce only
        // exact repetitions so repeated requests cannot grow this owner
        // without bound; payload-bearing results are never coalesced.
        if let Some(index) =
            self.native_recovery_attempts
                .iter()
                .position(|previous| match (previous, &result) {
                    (Recovery::Missing, Recovery::Missing)
                    | (Recovery::Busy, Recovery::Busy)
                    | (Recovery::Poisoned, Recovery::Poisoned)
                    | (Recovery::Expired, Recovery::Expired)
                    | (Recovery::WrongState, Recovery::WrongState)
                    | (Recovery::WrongType, Recovery::WrongType) => true,
                    (Recovery::Pending(a), Recovery::Pending(b)) => a == b,
                    (Recovery::PhysicalPending(a), Recovery::PhysicalPending(b)) => a == b,
                    _ => false,
                })
        {
            return self.native_recovery_attempts.get(index);
        }
        self.native_recovery_attempts.push(result);
        self.native_recovery_attempts.last()
    }

    pub fn observation(&self) -> Value {
        let mut resource_state_poisoned = false;
        let held = self.custody.as_ref().map(|custody| match custody.lock() {
            Ok(owner) => owner,
            Err(error) => {
                resource_state_poisoned = true;
                error.into_inner()
            }
        });
        if self.system_v3 {
            let mut observation = json!({"schema":"rz-rovent-native-system-resource-run-failure/3","error":self.detail,
                "original_custody_retained":self.custody.is_some(),
                "resource_state_poisoned":resource_state_poisoned,
                "evidence_directory":held.as_ref().map(|state|&state.plan.evidence_directory),
                "resource_failure":held.as_ref().and_then(|state|state.failure_v3.as_ref().map(|failure|failure.observation())),
                "admission":held.as_ref().and_then(|state|state.admission.as_ref()),
                "admission_evidence_error":held.as_ref().and_then(|state|state.admission_evidence_error.as_ref()),
                "finished_live":held.as_ref().and_then(|state|state.finished_live.as_ref()),
                "finished_evidence_error":held.as_ref().and_then(|state|state.finished_evidence_error.as_ref()),
                "finished_policy_observation_error":held.as_ref().and_then(|state|state.finished_live_error.as_ref()),
                "control_cleanup_error":held.as_ref().and_then(|state|state.control_cleanup_error.as_ref()),
                "product_finish":self.original_scope_key().and_then(|key|key.product_finish_observation()).as_ref().map(product_finish_json),
                "session_evidence":self.original_scope_key().map(|key| match key.terminal_evidence() {
                    rz_uci::rovent_bootstrap::launch::V3ScopeTerminalEvidence::BeforeNative => "before_native",
                    rz_uci::rovent_bootstrap::launch::V3ScopeTerminalEvidence::SessionReleased => "session_released",
                    rz_uci::rovent_bootstrap::launch::V3ScopeTerminalEvidence::Unknown => "unknown",
                }).unwrap_or("scope_unavailable"),
                "control_panic_payload_retained":self.original_control_panic_retained(),
                "automatic_cleanup":false,"training_executed":false});
            if let Some(evidence) = self.pre_native_admission {
                observation["pre_native_admission"] = json!(evidence);
            }
            return observation;
        }
        json!({"schema":"rz-rovent-native-resource-run-failure/1","error":self.detail,
            "original_custody_retained":self.custody.is_some(),
            "resource_state_poisoned":resource_state_poisoned,
            "evidence_directory":held.as_ref().map(|state|&state.plan.evidence_directory),
            "resource_failure":held.as_ref().and_then(|state|state.failure.as_ref().map(|failure|failure.observation())),
            "admission":held.as_ref().and_then(|state|state.admission.as_ref()),
            "finished_live":held.as_ref().and_then(|state|state.finished_live.as_ref()),
            "finished_live_error":held.as_ref().and_then(|state|state.finished_live_error.as_ref()),
            "automatic_cleanup":false,"training_executed":false})
    }
}

pub enum RoventNativeResourceRunFailureVersioned {
    V1(RoventNativeResourceRunFailure),
    V2(RoventNativeResourceRunFailure),
}
impl RoventNativeResourceRunFailureVersioned {
    pub fn observation(&self) -> Value {
        match self {
            Self::V1(failure) => failure.observation(),
            Self::V2(failure) => {
                let mut poisoned = false;
                let held = failure
                    .custody
                    .as_ref()
                    .map(|custody| match custody.lock() {
                        Ok(owner) => owner,
                        Err(error) => {
                            poisoned = true;
                            error.into_inner()
                        }
                    });
                json!({"schema":"rz-rovent-native-resource-run-failure/2",
                    "error":failure.detail,
                    "original_custody_retained":failure.custody.is_some(),
                    "resource_state_poisoned":poisoned,
                    "resource_admission_not_started":if poisoned { None } else { held.as_ref().map(|state|state.no_admission_started()) },
                    "evidence_directory":held.as_ref().map(|state|&state.plan.evidence_directory),
                    "resource_failure":held.as_ref().and_then(|state|state.failure_v2.as_ref().map(|failure|failure.observation())),
                    "admission":held.as_ref().and_then(|state|state.admission.as_ref()),
                    "admission_evidence_error":held.as_ref().and_then(|state|state.admission_evidence_error.as_ref()),
                    "finished_live":held.as_ref().and_then(|state|state.finished_live.as_ref()),
                    "finished_evidence_error":held.as_ref().and_then(|state|state.finished_evidence_error.as_ref()),
                    "finished_policy_observation_error":held.as_ref().and_then(|state|state.finished_live_error.as_ref()),
                    "automatic_cleanup":false,"training_executed":false})
            }
        }
    }
}
impl std::fmt::Display for RoventNativeResourceRunFailureVersioned {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::V1(error) | Self::V2(error) => std::fmt::Display::fmt(error, formatter),
        }
    }
}

pub(crate) struct NativeResourceCustody {
    system_v3: bool,
    plan: RoventNativeResourceLaunchPlan,
    pin: RoventNativeResourcePlanPin,
    // Hold the exact plan source and output directory through every return.
    _plan_file: File,
    control: Option<Arc<Mutex<super::direct_control::DirectControl>>>,
    bound_control: Option<Arc<Mutex<super::direct_control::BoundDirectControl>>>,
    control_cleanup_error: Option<String>,
    directory: Dir,
    delegated: Option<File>,
    scope: Option<RoventNativeResourceScope>,
    failure: Option<Box<RoventNativeResourceFailure>>,
    scope_v2: Option<RoventNativeResourceScopeV2>,
    failure_v2: Option<Box<RoventNativeResourceFailureV2>>,
    scope_v3: Option<RoventNativeResourceScopeV3>,
    failure_v3: Option<Box<RoventNativeResourceFailureV3>>,
    admission: Option<RoventResourceObservation>,
    admission_evidence_error: Option<String>,
    finished_live: Option<RoventResourceObservation>,
    finished_evidence_error: Option<String>,
    finished_live_error: Option<String>,
    admission_attempted: bool,
    admission_checkpoint_v3: AdmissionCheckpointV3,
    admission_v2_succeeded: bool,
    admission_v3_succeeded: bool,
    finished_attempted: bool,
}
impl NativeResourceCustody {
    fn provenance_rejection_before_scope(
        &self,
        epoch: u64,
        before_control: bool,
        native_error: &rz_uci::rovent_bootstrap::launch::RoventRetainingLaunchError,
        control_panicked: bool,
    ) -> Option<PreNativeAdmissionEvidence> {
        use rz_uci::rovent_bootstrap::launch::RoventRetainingLaunchError;
        if !before_control
            || control_panicked
            || !matches!(native_error, RoventRetainingLaunchError::Ordinary(_))
        {
            return None;
        }
        if self.admission_checkpoint_v3 != AdmissionCheckpointV3::ProvenanceRejected
            || !self.system_v3
            || !self.admission_attempted
            || self.delegated.is_none()
            || self.control.is_some()
            || self.bound_control.is_some()
            || self.scope.is_some()
            || self.scope_v2.is_some()
            || self.scope_v3.is_some()
            || self.failure.is_some()
            || self.failure_v2.is_some()
            || self.failure_v3.is_some()
            || self.admission_v3_succeeded
        {
            return None;
        }
        Some(PreNativeAdmissionEvidence {
            schema: "rz-rovent-resource-pre-native-admission/1",
            process_id: std::process::id(),
            process_generation: epoch,
            reason: "resource_plan_provenance",
            native_scope_started: false,
        })
    }
    /// This resource owner never attempted admission or constructed a scope.
    /// It does not say anything about UCI, storage, or another process owner.
    pub(crate) fn no_admission_started(&self) -> bool {
        !self.admission_attempted
            && self.scope.is_none()
            && self.scope_v2.is_none()
            && self.scope_v3.is_none()
            && self.failure.is_none()
            && self.failure_v2.is_none()
            && self.failure_v3.is_none()
            && self.admission.is_none()
    }
    /// Retain and enter the same original /3 lock and delegated FD before any
    /// ORT session or GPU worker is created. No projection to /1 or /2 occurs.
    pub(crate) fn admit_v3(&mut self, lock: &Arc<RoventInputLockV3>) -> Result<(), ArenaError> {
        if self.admission_attempted {
            return Err(invalid("native resource callback replay"));
        }
        self.admission_attempted = true;
        let original: Vec<_> = lock
            .manifest
            .provenance
            .iter()
            .filter(|artifact| Path::new(&artifact.path) == self.pin.path)
            .collect();
        if original.len() != 1
            || original[0].sha256 != self.pin.sha256
            || original[0].bytes != self.pin.bytes
        {
            self.admission_checkpoint_v3 = AdmissionCheckpointV3::ProvenanceRejected;
            return Err(invalid(
                "resource plan is not the original unique /3 input-lock provenance file",
            ));
        }
        let delegated = self
            .delegated
            .take()
            .ok_or_else(|| invalid("original delegated descriptor missing"))?;
        self.admission_checkpoint_v3 = AdmissionCheckpointV3::ResourcePrepareStarted;
        match RoventNativeResourceScopeV3::prepare(
            Arc::clone(lock),
            delegated,
            &self.plan.delegated_cgroup_path,
            &self.plan.leaf_name,
            self.plan.options.clone(),
        ) {
            Ok(scope) => self.scope_v3 = Some(scope),
            Err(failure) => {
                let error = invalid(failure.to_string());
                self.failure_v3 = Some(failure);
                return Err(error);
            }
        }
        let snapshot = self
            .scope_v3
            .as_ref()
            .expect("stored original /3 scope")
            .observe();
        let (observation, evidence_error, policy) =
            persist_admission_observation(&snapshot, &self.directory, self.plan.profile);
        self.admission = observation;
        self.admission_evidence_error = evidence_error;
        policy?;
        self.admission_v3_succeeded = true;
        Ok(())
    }
    /// Admit the original /2 lock and exact delegated FD before native model
    /// loading. An error retains the same owner and any partially entered OS
    /// scope; it is not converted to /1 or reported as successful admission.
    pub(crate) fn admit_v2(&mut self, lock: &Arc<RoventInputLockV2>) -> Result<(), ArenaError> {
        if self.admission_attempted {
            return Err(invalid("native resource callback replay"));
        }
        self.admission_attempted = true;
        let original: Vec<_> = lock
            .manifest
            .provenance
            .iter()
            .filter(|artifact| Path::new(&artifact.path) == self.pin.path)
            .collect();
        if original.len() != 1
            || original[0].sha256 != self.pin.sha256
            || original[0].bytes != self.pin.bytes
        {
            return Err(invalid(
                "resource plan is not the original unique /2 input-lock provenance file",
            ));
        }
        let delegated = self
            .delegated
            .take()
            .ok_or_else(|| invalid("original delegated descriptor missing"))?;
        match RoventNativeResourceScopeV2::prepare(
            Arc::clone(lock),
            delegated,
            &self.plan.delegated_cgroup_path,
            &self.plan.leaf_name,
            self.plan.options.clone(),
        ) {
            Ok(scope) => self.scope_v2 = Some(scope),
            Err(failure) => {
                let error = invalid(failure.to_string());
                self.failure_v2 = Some(failure);
                return Err(error);
            }
        }
        let snapshot = self
            .scope_v2
            .as_ref()
            .expect("stored original /2 scope")
            .observe();
        // A failed evidence write is not a failure to apply or observe the
        // original resource policy. Keep it distinct while still withholding
        // V2 corpus eligibility when the required readback is absent.
        let (observation, evidence_error, policy) =
            persist_admission_observation(&snapshot, &self.directory, self.plan.profile);
        self.admission = observation;
        self.admission_evidence_error = evidence_error;
        policy?;
        self.admission_v2_succeeded = true;
        Ok(())
    }

    pub(crate) fn admit(&mut self, lock: &Arc<RoventInputLock>) -> Result<(), ArenaError> {
        if self.admission_attempted {
            return Err(invalid("native resource callback replay"));
        }
        self.admission_attempted = true;
        let original: Vec<_> = lock
            .manifest
            .provenance
            .iter()
            .filter(|artifact| Path::new(&artifact.path) == self.pin.path)
            .collect();
        if original.len() != 1
            || original[0].sha256 != self.pin.sha256
            || original[0].bytes != self.pin.bytes
        {
            return Err(invalid(
                "resource plan is not the original unique input-lock provenance file",
            ));
        }
        let delegated = self
            .delegated
            .take()
            .ok_or_else(|| invalid("original delegated descriptor missing"))?;
        match RoventNativeResourceScope::prepare(
            Arc::clone(lock),
            delegated,
            &self.plan.delegated_cgroup_path,
            &self.plan.leaf_name,
            self.plan.options.clone(),
        ) {
            Ok(scope) => self.scope = Some(scope),
            Err(failure) => {
                let error = invalid(failure.to_string());
                self.failure = Some(failure);
                return Err(error);
            }
        }
        let snapshot = self
            .scope
            .as_ref()
            .expect("stored original scope")
            .observe();
        // Preserve current actual controls before model loading. Finished-run
        // peaks and full comparison acceptance are checked only after launch.
        self.admission =
            Some(snapshot.write_evidence(&self.directory, Path::new("admission-readback.json"))?);
        snapshot.require_admission(self.plan.profile)
    }
    pub(crate) fn finish_live(&mut self) -> Result<(), ArenaError> {
        if self.finished_attempted {
            return Err(invalid("native resource finished observation replay"));
        }
        self.finished_attempted = true;
        if let Some(snapshot) = self
            .scope_v3
            .as_ref()
            .map(RoventNativeResourceScopeV3::observe)
            .or_else(|| {
                self.scope_v2
                    .as_ref()
                    .map(RoventNativeResourceScopeV2::observe)
            })
        {
            let (observation, evidence_error, policy) =
                persist_finished_observation(&snapshot, &self.directory, self.plan.profile);
            self.finished_live = observation;
            self.finished_evidence_error = evidence_error;
            // This is a post-execution observation/acceptance decision. A
            // missing file or unobserved final controls cannot rewrite the
            // already-completed engine result; retain the original reason.
            if let Err(error) = policy {
                self.finished_live_error = Some(error.to_string());
            }
            return Ok(());
        }
        let result: Result<(), ArenaError> = (|| {
            let snapshot = self.scope.as_ref().map(RoventNativeResourceScope::observe);
            if let Some(snapshot) = snapshot {
                self.finished_live =
                    Some(snapshot.write_evidence(
                        &self.directory,
                        Path::new("finished-live-readback.json"),
                    )?);
                snapshot.require_finished_profile(self.plan.profile)?;
            }
            Ok(())
        })();
        if let Err(error) = &result {
            self.finished_live_error = Some(error.to_string());
        }
        result
    }
    /// Only a successful readback for the same live /2 owner can be reported
    /// as resource admission. This still does not attest final peaks, native
    /// physical completion, teacher retirement or corpus publication.
    pub(crate) fn validate_success_v2(
        &self,
        lock: &Arc<RoventInputLockV2>,
    ) -> Result<(), ArenaError> {
        if !self.admission_attempted
            || !self.admission_v2_succeeded
            || !self.finished_attempted
            || self.scope.is_some()
            || self.failure.is_some()
            || self.failure_v2.is_some()
            || self.scope_v3.is_some()
            || self.failure_v3.is_some()
            || !self
                .scope_v2
                .as_ref()
                .is_some_and(|scope| scope.matches_input_lock(lock))
            || self.admission.is_none()
            || self.admission_evidence_error.is_some()
            || self.finished_live.is_none()
            || self.finished_evidence_error.is_some()
            || self.finished_live_error.is_some()
        {
            return Err(invalid(
                "original /2 resource scope and successful live readbacks required",
            ));
        }
        Ok(())
    }

    /// Require the same live /3 owner and actual admission. Receipt writes
    /// and post-run observation remain separately optional evidence.
    fn original_v3_scope(&self) -> Result<&RoventNativeResourceScopeV3, ArenaError> {
        if !self.admission_attempted
            || !self.admission_v3_succeeded
            || !self.finished_attempted
            || self.scope.is_some()
            || self.scope_v2.is_some()
            || self.failure.is_some()
            || self.failure_v2.is_some()
            || self.failure_v3.is_some()
        {
            return Err(invalid(
                "original /3 resource scope and actual admission required",
            ));
        }
        self.scope_v3
            .as_ref()
            .ok_or_else(|| invalid("original /3 resource scope missing"))
    }
    pub(crate) fn observation(&self) -> Value {
        if self.system_v3 {
            return json!({"schema":"rz-rovent-system-resource-custody/3","plan_pin":self.pin,
                "evidence_directory":self.plan.evidence_directory,"profile":self.plan.profile,
                "admission_attempted":self.admission_attempted,"finished_attempted":self.finished_attempted,
                "original_plan_descriptor_retained":true,
                "original_delegated_descriptor_retained":self.delegated.is_some()||self.scope_v3.is_some()||self.failure_v3.is_some(),
                "original_scope_retained":self.scope_v3.is_some(),"resource_failure":self.failure_v3.as_ref().map(|failure|failure.observation()),
                "admission":self.admission,"admission_evidence_error":self.admission_evidence_error,
                "finished_live":self.finished_live,"finished_evidence_error":self.finished_evidence_error,"finished_live_error":self.finished_live_error,
                "controlled_comparison_accepted":false,"automatic_cleanup":false});
        }
        json!({"schema":"rz-rovent-resource-custody/1","plan_pin":self.pin,
            "evidence_directory":self.plan.evidence_directory,"profile":self.plan.profile,
            "admission_attempted":self.admission_attempted,"finished_attempted":self.finished_attempted,
            "original_plan_descriptor_retained":true,"original_delegated_descriptor_retained":self.delegated.is_some()||self.scope.is_some()||self.failure.is_some()||self.scope_v2.is_some()||self.failure_v2.is_some(),
            "original_scope_retained":self.scope.is_some()||self.scope_v2.is_some(),"resource_failure":self.failure.as_ref().map(|failure|failure.observation()).or_else(||self.failure_v2.as_ref().map(|failure|failure.observation())),
            "admission":self.admission,"finished_live":self.finished_live,"finished_live_error":self.finished_live_error,
            "controlled_comparison_accepted":false,"automatic_cleanup":false})
    }

    /// Separate failure diagnostics, deliberately not part of the historical
    /// strict `/1` custody object used by N05 consumers.
    pub(crate) fn evidence_failure_observation(&self) -> Option<Value> {
        if self.admission_evidence_error.is_none() && self.finished_evidence_error.is_none() {
            return None;
        }
        Some(json!({
            "admission_evidence_error": self.admission_evidence_error,
            "finished_evidence_error": self.finished_evidence_error,
            "finished_policy_observation_error": self.finished_live_error,
            "admission_v2_succeeded": self.admission_v2_succeeded,
            "v2_corpus_eligibility": "unconfirmed",
        }))
    }
}

/// Persisting the readback is diagnostic; actual kernel-policy admission is
/// still checked even when the evidence file cannot be created.
fn persist_admission_observation(
    snapshot: &RoventNativeResourceSnapshot,
    directory: &Dir,
    profile: RoventNativeResourceProfile,
) -> (
    Option<RoventResourceObservation>,
    Option<String>,
    Result<(), ArenaError>,
) {
    let (observation, evidence_error) =
        match snapshot.write_evidence(directory, Path::new("admission-readback.json")) {
            Ok(observation) => (Some(observation), None),
            Err(error) => (None, Some(error.to_string())),
        };
    let policy = snapshot.require_admission(profile);
    (observation, evidence_error, policy)
}

fn persist_finished_observation(
    snapshot: &RoventNativeResourceSnapshot,
    directory: &Dir,
    profile: RoventNativeResourceProfile,
) -> (
    Option<RoventResourceObservation>,
    Option<String>,
    Result<(), ArenaError>,
) {
    let (observation, evidence_error) =
        match snapshot.write_evidence(directory, Path::new("finished-live-readback.json")) {
            Ok(observation) => (Some(observation), None),
            Err(error) => (None, Some(error.to_string())),
        };
    let policy = snapshot.require_finished_profile(profile);
    (observation, evidence_error, policy)
}

/// Bind the V3 child's receipt root to the already-owned resource evidence
/// directory.  A caller cannot redirect the system runtime receipt to an
/// unrelated path or silently reuse a different process generation.
fn bind_system_v3_arguments(
    arguments: &mut Vec<String>,
    evidence_directory: &Path,
) -> Result<u64, ArenaError> {
    if !relative(evidence_directory) {
        return Err(invalid(
            "original V3 receipt root must be a safe relative directory",
        ));
    }
    let path = evidence_directory
        .to_str()
        .ok_or_else(|| invalid("V3 receipt root UTF-8"))?;
    let mut epoch = None;
    let mut root_seen = false;
    for argument in arguments.iter() {
        if let Some(value) = argument.strip_prefix("--process-epoch=") {
            let value = value.parse::<u64>().map_err(invalid)?;
            if value == 0 || epoch.replace(value).is_some() {
                return Err(invalid("V3 requires one nonzero process epoch"));
            }
        } else if let Some(value) = argument.strip_prefix("--system-runtime-receipt-root=") {
            if root_seen || Path::new(value) != evidence_directory {
                return Err(invalid(
                    "V3 system receipt root differs from original resource owner",
                ));
            }
            root_seen = true;
        }
    }
    let epoch = epoch.ok_or_else(|| invalid("V3 process epoch missing"))?;
    if !root_seen {
        arguments.push(format!("--system-runtime-receipt-root={path}"));
    }
    Ok(epoch)
}

fn read_system_v3_receipt(
    directory: &Dir,
    name: &str,
    lock: &rz_experiments::SystemRuntimeLockV6,
    pid: u32,
    epoch: u64,
) -> Result<rz_experiments::SystemRuntimeReceiptV6, ArenaError> {
    let mut options = cap_std::fs::OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    let file = directory.open_with(name, &options).map_err(invalid)?;
    if !file.metadata().map_err(invalid)?.is_file() {
        return Err(invalid("V3 system receipt must be a regular file"));
    }
    let mut bytes = Vec::new();
    file.take((256 << 10) + 1)
        .read_to_end(&mut bytes)
        .map_err(invalid)?;
    if bytes.len() > 256 << 10 {
        return Err(invalid("V3 system receipt exceeds its byte budget"));
    }
    let text = std::str::from_utf8(&bytes).map_err(invalid)?;
    Ok(rz_experiments::SystemRuntimeReceiptV6::from_json(
        text, lock, pid, epoch,
    )?)
}

fn verify_system_v3_child_receipts(
    state: &NativeResourceCustody,
    epoch: u64,
) -> Result<
    (
        rz_experiments::SystemRuntimeReceiptV6,
        rz_experiments::SystemRuntimeReceiptV6,
    ),
    ArenaError,
> {
    let scope = state.original_v3_scope()?;
    let lock = &scope.input_lock().manifest.system_runtime_lock;
    let pid = std::process::id();
    let session = state
        .directory
        .open_dir_nofollow(format!("native-process-{pid}"))
        .map_err(|_| invalid("original V3 process receipt directory unavailable"))?;
    let startup =
        read_system_v3_receipt(&session, "system-runtime-startup.v6.json", lock, pid, epoch)?;
    let termination = read_system_v3_receipt(
        &session,
        "system-runtime-termination.v6.json",
        lock,
        pid,
        epoch,
    )?;
    validate_system_v3_receipt_pair(lock, pid, epoch, &startup, &termination)?;
    Ok((startup, termination))
}

/// Validate the original child's claims. This does not independently attest
/// device execution or elevate a CPU run to CUDA evidence.
fn validate_system_v3_receipt_pair(
    lock: &rz_experiments::SystemRuntimeLockV6,
    pid: u32,
    epoch: u64,
    startup: &rz_experiments::SystemRuntimeReceiptV6,
    termination: &rz_experiments::SystemRuntimeReceiptV6,
) -> Result<(), ArenaError> {
    use rz_experiments::{NativeRuntimeSourceV6, PhysicalCompletionV6, ProviderReadinessV6};
    startup.validate_against(lock, pid, epoch)?;
    if startup.provider_ready != ProviderReadinessV6::Ready
        || startup.physical_completion != PhysicalCompletionV6::Unknown
    {
        return Err(invalid(
            "V3 startup system runtime observation is incomplete",
        ));
    }
    match &lock.runtime {
        NativeRuntimeSourceV6::Cpu => {
            if startup.observed_identity_sha256.is_some() || !startup.observed_libraries.is_empty()
            {
                return Err(invalid(
                    "CPU startup cannot assert CUDA identity or libraries",
                ));
            }
        }
        NativeRuntimeSourceV6::SystemCuda(system) => {
            if startup.observed_identity_sha256.as_deref()
                != Some(system.system_identity_sha256.as_str())
                || !startup
                    .unregistered_native_mappings
                    .as_ref()
                    .is_some_and(Vec::is_empty)
            {
                return Err(invalid(
                    "V3 startup system runtime observation is incomplete",
                ));
            }
            system.validate_ready_observed(&startup.observed_libraries)?;
        }
    }
    termination.validate_success(lock, pid, epoch)?;
    if termination.physical_completion
        != (PhysicalCompletionV6::Confirmed {
            completion_id: format!("session_finish_confirmed/1:{pid}:{epoch}"),
        })
    {
        return Err(invalid(
            "V3 terminal receipt does not identify its original session completion",
        ));
    }
    Ok(())
}

/// The actual shared launch callback runs after its original lock/process
/// admission and before model/ORT/teacher creation. No UCI/search implementation
/// or private native authority is reconstructed by this upper runner.
pub fn run_native_with_resources(
    pin: RoventNativeResourcePlanPin,
    arguments: Vec<String>,
) -> Result<RoventNativeResourceRunReceipt, RoventNativeResourceRunFailure> {
    match run_native_with_resources_inner(pin, arguments, None)? {
        RoventNativeResourceRunReceiptVersioned::V1(receipt) => Ok(receipt),
        RoventNativeResourceRunReceiptVersioned::V2(_) => {
            unreachable!("V1-only resource callback cannot admit a V2 lock")
        }
    }
}

/// Native product entry with separately typed V1 and V2 resource admission.
pub fn run_native_with_resources_versioned(
    pin: RoventNativeResourcePlanPin,
    arguments: Vec<String>,
) -> Result<RoventNativeResourceRunReceiptVersioned, RoventNativeResourceRunFailureVersioned> {
    let selected_v2 = std::cell::Cell::new(false);
    run_native_with_resources_inner(pin, arguments, Some(&selected_v2)).map_err(|failure| {
        if selected_v2.get() {
            RoventNativeResourceRunFailureVersioned::V2(failure)
        } else {
            RoventNativeResourceRunFailureVersioned::V1(failure)
        }
    })
}

fn run_native_with_resources_inner(
    pin: RoventNativeResourcePlanPin,
    arguments: Vec<String>,
    selected_v2: Option<&std::cell::Cell<bool>>,
) -> Result<RoventNativeResourceRunReceiptVersioned, RoventNativeResourceRunFailure> {
    let prepare = || -> Result<NativeResourceCustody, ArenaError> {
        pin.validate()?;
        let root = Dir::open_ambient_dir(
            std::env::current_dir().map_err(invalid)?,
            cap_std::ambient_authority(),
        )
        .map_err(invalid)?;
        prepare_resource_custody(&root, pin)
    };
    let state = prepare().map_err(|error| RoventNativeResourceRunFailure {
        detail: error.to_string(),
        custody: None,
        system_v3: false,
        native_failure: None,
        native_recovery_attempts: Vec::new(),
        control_panic: None,
        control_panic_may_be_native: false,
        control_scope_marker: None,
        pre_native_admission: None,
    })?;
    let custody = Arc::new(Mutex::new(state));
    let callback_v1 = Arc::clone(&custody);
    let callback_v2 = Arc::clone(&custody);
    let admit_v1 = move |lock: &Arc<RoventInputLock>| {
        callback_v1
            .lock()
            .map_err(|_| RoventLaunchError {
                stage: "ResourceAdmission",
                detail: "original resource owner poisoned".into(),
            })?
            .admit(lock)
            .map_err(|error| RoventLaunchError {
                stage: "ResourceAdmission",
                detail: error.to_string(),
            })
    };
    let admit_v2 = move |lock: &Arc<RoventInputLockV2>| {
        callback_v2
            .lock()
            .map_err(|_| RoventLaunchError {
                stage: "ResourceAdmission",
                detail: "original resource owner poisoned".into(),
            })?
            .admit_v2(lock)
            .map_err(|error| RoventLaunchError {
                stage: "ResourceAdmission",
                detail: error.to_string(),
            })
    };
    let result = if let Some(selected) = selected_v2 {
        run_with_resource_admission_v1_v2(arguments, admit_v1, admit_v2, move || selected.set(true))
    } else {
        run_with_resource_admission(arguments, admit_v1)
    };
    let finish = custody
        .lock()
        .map_err(|_| invalid("original resource owner poisoned"))
        .and_then(|mut state| state.finish_live());
    let failure = |detail: String| RoventNativeResourceRunFailure {
        detail,
        custody: Some(Arc::clone(&custody)),
        system_v3: false,
        native_failure: None,
        native_recovery_attempts: Vec::new(),
        control_panic: None,
        control_panic_may_be_native: false,
        control_scope_marker: None,
        pre_native_admission: None,
    };
    if let Err(error) = result {
        let detail = match &finish {
            Ok(()) => error.to_string(),
            Err(resource_error) => {
                format!("{error}; resource observation also failed: {resource_error}")
            }
        };
        return Err(failure(detail));
    }
    finish.map_err(|error| failure(error.to_string()))?;
    let state = custody
        .lock()
        .map_err(|_| failure("original resource owner poisoned".into()))?;
    let admission = state.admission.clone();
    let finished_live = state.finished_live.clone();
    if state.admission_v2_succeeded {
        drop(state);
        return Ok(RoventNativeResourceRunReceiptVersioned::V2(
            RoventNativeResourceRunReceiptV2 {
                admission,
                finished_live,
                custody,
            },
        ));
    }
    let admission =
        admission.ok_or_else(|| failure("actual resource admission did not run".into()))?;
    let finished_live =
        finished_live.ok_or_else(|| failure("actual finished resource snapshot missing".into()))?;
    drop(state);
    Ok(RoventNativeResourceRunReceiptVersioned::V1(
        RoventNativeResourceRunReceipt {
            admission,
            finished_live,
            custody,
        },
    ))
}

/// Explicit /3 system CUDA launch under its original resource owner. It uses
/// the very same /3 lock Arc admitted by UCI, never a converted older lock.
/// Only child-owned session-finish evidence is accepted here: a separate outer
/// supervisor must still observe process exit and any independent GPU fence.
pub fn run_native_with_resources_v3(
    pin: RoventNativeResourcePlanPin,
    arguments: Vec<String>,
) -> Result<RoventNativeSystemResourceRunReceiptV3, RoventNativeResourceRunFailure> {
    run_native_with_resources_v3_inner(pin, arguments, false).map_err(|error| match error {
        V3InnerFailure::BeforeControl(failure) | V3InnerFailure::Failed(failure) => failure,
        V3InnerFailure::Quarantined(quarantine) => quarantine.original_failure,
    })
}

/// Explicit opt-in for a reconnectable owner endpoint prepared before native
/// entry. This does not change the legacy V3 or CPU resource requirements.
pub fn run_native_with_resources_v3_control(
    pin: RoventNativeResourcePlanPin,
    arguments: Vec<String>,
) -> Result<RoventNativeSystemResourceRunReceiptV3, RoventControlledRunFailure> {
    run_native_with_resources_v3_inner(pin, arguments, true).map_err(|error| match error {
        V3InnerFailure::BeforeControl(failure) => {
            RoventControlledRunFailure::BeforeControl(failure)
        }
        V3InnerFailure::Failed(failure) => RoventControlledRunFailure::Failed(failure),
        V3InnerFailure::Quarantined(quarantine) => {
            RoventControlledRunFailure::Quarantined(quarantine)
        }
    })
}

fn run_native_with_resources_v3_inner(
    pin: RoventNativeResourcePlanPin,
    mut arguments: Vec<String>,
    custody_control: bool,
) -> Result<RoventNativeSystemResourceRunReceiptV3, V3InnerFailure> {
    let prepare = || -> Result<NativeResourceCustody, ArenaError> {
        pin.validate()?;
        let root = Dir::open_ambient_dir(
            std::env::current_dir().map_err(invalid)?,
            cap_std::ambient_authority(),
        )
        .map_err(invalid)?;
        prepare_resource_custody(&root, pin)
    };
    let mut state = prepare().map_err(|error| RoventNativeResourceRunFailure {
        detail: error.to_string(),
        custody: None,
        system_v3: true,
        native_failure: None,
        native_recovery_attempts: Vec::new(),
        control_panic: None,
        control_panic_may_be_native: false,
        control_scope_marker: None,
        pre_native_admission: None,
    })?;
    state.system_v3 = true;
    let custody = Arc::new(Mutex::new(state));
    let failure = |detail: String| RoventNativeResourceRunFailure {
        detail,
        custody: Some(Arc::clone(&custody)),
        system_v3: true,
        native_failure: None,
        native_recovery_attempts: Vec::new(),
        control_panic: None,
        control_panic_may_be_native: false,
        control_scope_marker: None,
        pre_native_admission: None,
    };
    let epoch = {
        let state = custody
            .lock()
            .map_err(|_| failure("original resource owner poisoned".into()))?;
        bind_system_v3_arguments(&mut arguments, &state.plan.evidence_directory)
            .map_err(|error| failure(error.to_string()))?
    };
    let callback = Arc::clone(&custody);
    let control_slot: Arc<OnceLock<Arc<Mutex<super::direct_control::BoundDirectControl>>>> =
        Arc::new(OnceLock::new());
    let parent_stream = Arc::new(Mutex::new(None));
    let callback_parent_stream = Arc::clone(&parent_stream);
    let mut control_panic = None;
    let mut control_panic_may_be_native = false;
    let mut control_scope_marker = None;
    let mut before_control = false;
    let mut controlled_bound = None;
    let result = if custody_control {
        // Capture before the callback binds or publishes the endpoint.
        let original_parent_pid = nix::unistd::getppid().as_raw();
        use rz_uci::rovent_bootstrap::launch::{
            BeforeControlV3Outcome, ControlledV3Outcome, ControlledV3Result,
            RoventRetainingLaunchError,
        };
        let controlled =
            rz_uci::rovent_bootstrap::launch::run_with_v3_resource_admission_retaining_connected_control_slot(
                arguments,
                Arc::clone(&control_slot),
                parent_stream,
                move |lock, _root, _file, _startup_until, cleanup_until, directory, path, slot| {
                    let mut state = callback.lock().map_err(|_| RoventLaunchError {
                        stage: "ResourceAdmission",
                        detail: "original resource owner poisoned".into(),
                    })?;
                    state.admit_v3(lock).map_err(|error| RoventLaunchError {
                        stage: "ResourceAdmission",
                        detail: error.to_string(),
                    })?;
                    let directory = directory.ok_or_else(|| RoventLaunchError {
                        stage: "ControlAdmission",
                        detail: "original writer directory unavailable".into(),
                    })?;
                    let path = path.ok_or_else(|| RoventLaunchError {
                        stage: "ControlAdmission",
                        detail: "original writer route unavailable".into(),
                    })?;
                    let control = Arc::new(Mutex::new(
                        super::direct_control::DirectControl::prepare(
                            directory,
                            path,
                            cleanup_until,
                            std::time::Duration::from_millis(lock.manifest.cleanup_max_ms),
                        )
                        .map_err(|error| RoventLaunchError {
                            stage: "ControlAdmission",
                            detail: error.to_string(),
                        })?,
                    ));
                    // Retain the shell before bind, including on a bind failure/panic.
                    state.control = Some(Arc::clone(&control));
                    let mut shell = control.lock().unwrap_or_else(|error| error.into_inner());
                    shell.bind().map_err(|error| RoventLaunchError {
                        stage: "ControlAdmission",
                        detail: error.to_string(),
                    })?;
                    let control_path = shell.path();
                    let ordinal = match &lock.manifest.system_runtime_lock.runtime {
                        NativeRuntimeSourceV6::SystemCuda(_) => match &lock.manifest.engine.pc.device {
                            rz_experiments::RoventDevice::Cuda { ordinal, .. } => Some(*ordinal),
                            _ => return Err(RoventLaunchError { stage: "ControlAdmission", detail: "CUDA device declaration missing".into() }),
                        },
                        _ => None,
                    };
                    let _ = writeln!(
                        std::io::stderr().lock(),
                        "{}",
                        json!({"schema":"rz-rovent-direct-custody-startup/1",
                "process_id":std::process::id(),"process_epoch":epoch,
                "lock_sha256":lock.canonical_sha256,"cuda_ordinal":ordinal,
                "control_path":control_path,
                "physical_completion":"unknown_until_original_owner_confirms"})
                    );
                    drop(shell);
                    let bound = super::direct_control::BoundDirectControl::from_bound(control)
                        .map_err(|_original| RoventLaunchError {
                            stage: "ControlAdmission",
                            detail: "original listener not bound".into(),
                        })?;
                    let bound = Arc::new(Mutex::new(bound));
                    slot.set(Arc::clone(&bound))
                        .map_err(|_original| RoventLaunchError {
                            stage: "ControlAdmission",
                            detail: "original control slot already installed".into(),
                        })?;
                    state.bound_control = Some(bound);
                    let original_bound = Arc::clone(state.bound_control.as_ref().expect("bound control retained"));
                    drop(state);
                    if ordinal.is_some() {
                        let stream = original_bound.lock().map_err(|_| RoventLaunchError {
                            stage: "ControlAdmission", detail: "bound control owner poisoned".into(),
                        })?.accept_parent_until(_startup_until, original_parent_pid).map_err(|error| RoventLaunchError {
                            stage: "ControlAdmission", detail: error.to_string(),
                        })?;
                        let mut parent = callback_parent_stream.lock().map_err(|_| RoventLaunchError {
                            stage: "ControlAdmission", detail: "parent stream owner poisoned".into(),
                        })?;
                        if parent.replace(stream).is_some() {
                            return Err(RoventLaunchError { stage: "ControlAdmission", detail: "parent stream already installed".into() });
                        }
                    }
                    Ok(())
                },
            );
        let outcome = match controlled {
            ControlledV3Result::BeforeControl { outcome } => {
                before_control = true;
                match outcome {
                    BeforeControlV3Outcome::Error(error) => ControlledV3Outcome::Returned(Err(
                        RoventRetainingLaunchError::Ordinary(error),
                    )),
                    BeforeControlV3Outcome::Panic(payload) => ControlledV3Outcome::Panic(payload),
                }
            }
            ControlledV3Result::Controlled {
                original_control,
                scope_marker,
                outcome,
            } => {
                // Restore this exact Arc even when the callback unwound after
                // installing it but before updating the resource state.
                custody
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .bound_control = Some(Arc::clone(&original_control));
                controlled_bound = Some(original_control);
                control_panic_may_be_native = matches!(&outcome, ControlledV3Outcome::Panic(_))
                    && scope_marker.get().is_some();
                control_scope_marker = Some(scope_marker);
                outcome
            }
        };
        match outcome {
            ControlledV3Outcome::Returned(result) => result,
            ControlledV3Outcome::Panic(payload) => {
                control_panic = Some(payload);
                Err(RoventRetainingLaunchError::Ordinary(RoventLaunchError {
                    stage: "ControlPanic",
                    detail: "original control panic payload retained".into(),
                }))
            }
        }
    } else {
        rz_uci::rovent_bootstrap::launch::run_with_v3_resource_admission_retaining(
            arguments,
            move |lock, _original_root, _original_lock_file, _original_deadline| {
                callback
                    .lock()
                    .map_err(|_| RoventLaunchError {
                        stage: "ResourceAdmission",
                        detail: "original resource owner poisoned".into(),
                    })?
                    .admit_v3(lock)
                    .map_err(|error| RoventLaunchError {
                        stage: "ResourceAdmission",
                        detail: error.to_string(),
                    })
            },
        )
    };
    // A quarantined native owner may still be doing work.  Keep its resource
    // owner and do not publish an observation as a finished-run readback.
    let native_quarantined = control_panic_may_be_native
        || matches!(
            &result,
            Err(rz_uci::rovent_bootstrap::launch::RoventRetainingLaunchError::Quarantined { .. })
        );
    // This proves only the original session finished. Use it for the endpoint
    // cleanup window after a returned call, never to release quarantine or to
    // claim a product-wide/device-wide completion fence.
    let session_released = control_scope_marker
        .as_ref()
        .and_then(|marker| marker.get())
        .is_some_and(|key| {
            matches!(
                key.terminal_evidence(),
                rz_uci::rovent_bootstrap::launch::V3ScopeTerminalEvidence::SessionReleased
            )
        });
    if !native_quarantined && custody_control {
        let mut state = match custody.lock() {
            Ok(state) => state,
            Err(error) => {
                let mut state = error.into_inner();
                state.control_cleanup_error = Some("original resource state was poisoned".into());
                state
            }
        };
        if let Some(control) = state.control.as_ref() {
            let mut control = control.lock().unwrap_or_else(|error| error.into_inner());
            let result = if session_released && control_panic.is_none() {
                control.cleanup_after_session()
            } else {
                control.cleanup_confirmed()
            };
            drop(control);
            if let Err(error) = result {
                let previous = state.control_cleanup_error.take();
                state.control_cleanup_error = Some(match previous {
                    Some(previous) => format!("{previous}; {error}"),
                    None => error.to_string(),
                });
            }
        }
    }
    let finish = if native_quarantined || control_panic.is_some() {
        None
    } else {
        Some(
            custody
                .lock()
                .map_err(|_| invalid("original resource owner poisoned"))
                .and_then(|mut state| state.finish_live()),
        )
    };
    if let Err(native_error) = result {
        use rz_uci::rovent_bootstrap::launch::RoventRetainingLaunchError;
        let pre_native_admission = custody.try_lock().ok().and_then(|state| {
            state.provenance_rejection_before_scope(
                epoch,
                before_control,
                &native_error,
                control_panic.is_some(),
            )
        });
        let error = match &native_error {
            RoventRetainingLaunchError::Ordinary(error)
            | RoventRetainingLaunchError::Unstarted { error, .. }
            | RoventRetainingLaunchError::Settled { error, .. }
            | RoventRetainingLaunchError::Quarantined { error, .. } => error,
        };
        let detail = match &finish {
            None | Some(Ok(())) => error.to_string(),
            Some(Err(resource_error)) => {
                format!("{error}; resource observation also failed: {resource_error}")
            }
        };
        let mut retained = failure(detail);
        retained.pre_native_admission = pre_native_admission;
        retained.native_failure = Some(native_error);
        retained.control_panic = control_panic;
        retained.control_panic_may_be_native = control_panic_may_be_native;
        retained.control_scope_marker = control_scope_marker;
        if native_quarantined {
            if let Some(original_control) = controlled_bound {
                return Err(V3InnerFailure::Quarantined(RoventDirectQuarantine {
                    original_failure: retained,
                    original_control,
                }));
            }
        }
        return Err(if before_control {
            V3InnerFailure::BeforeControl(retained)
        } else {
            V3InnerFailure::Failed(retained)
        });
    }
    finish
        .expect("successful native execution is not quarantined")
        .map_err(|error| failure(error.to_string()))?;
    let state = custody
        .lock()
        .map_err(|_| failure("original resource owner poisoned".into()))?;
    let system_lock_sha256 = state
        .original_v3_scope()
        .map_err(|error| failure(error.to_string()))?
        .input_lock()
        .manifest
        .system_runtime_lock
        .canonical_sha256
        .clone();
    // A missing or malformed diagnostic is not an engine failure. Preserve
    // the original files and mark acceptance unknown rather than hiding it.
    let (startup, termination, system_runtime_evidence_error) =
        match verify_system_v3_child_receipts(&state, epoch) {
            Ok((startup, termination)) => (Some(startup), Some(termination), None),
            Err(error) => (None, None, Some(error.to_string())),
        };
    let admission = state.admission.clone();
    let finished_live = state.finished_live.clone();
    let admission_evidence_error = state.admission_evidence_error.clone();
    let finished_evidence_error = state.finished_evidence_error.clone();
    let finished_policy_observation_error = state.finished_live_error.clone();
    let control_cleanup_error = state.control_cleanup_error.clone();
    drop(state);
    Ok(RoventNativeSystemResourceRunReceiptV3 {
        admission,
        finished_live,
        admission_evidence_error,
        finished_evidence_error,
        finished_policy_observation_error,
        control_cleanup_error,
        startup,
        termination,
        system_runtime_evidence_error,
        system_lock_sha256,
        process_id: std::process::id(),
        process_generation: epoch,
        product_finish: control_scope_marker
            .as_ref()
            .and_then(|marker| marker.get())
            .and_then(|key| key.product_finish_observation()),
        custody,
    })
}

/// Shared pinned preparation for the direct native and Collection upper
/// callers. Only this original FD/readback owner implements resource lifetime.
pub(crate) fn prepare_resource_custody(
    root: &Dir,
    pin: RoventNativeResourcePlanPin,
) -> Result<NativeResourceCustody, ArenaError> {
    pin.validate()?;
    let (parent, leaf) = parent_and_leaf(root, &pin.path)?;
    let mut options = cap_std::fs::OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    let mut plan_file = parent.open_with(leaf, &options).map_err(invalid)?;
    if !plan_file.metadata().map_err(invalid)?.is_file() {
        return Err(invalid("resource plan must be a regular held file"));
    }
    let mut bytes = Vec::new();
    (&mut plan_file)
        .take(pin.bytes + 1)
        .read_to_end(&mut bytes)
        .map_err(invalid)?;
    if bytes.len() as u64 != pin.bytes || format!("{:x}", Sha256::digest(&bytes)) != pin.sha256 {
        return Err(invalid("original resource plan bytes differ"));
    }
    let plan: RoventNativeResourceLaunchPlan =
        crate::decode_json(std::str::from_utf8(&bytes).map_err(invalid)?)?;
    plan.validate()?;
    let (delegated, directory) = prepare_native_resource_paths(&plan, root)?;
    Ok(NativeResourceCustody {
        system_v3: false,
        plan,
        pin,
        _plan_file: plan_file.into_std(),
        control: None,
        bound_control: None,
        control_cleanup_error: None,
        directory,
        delegated: Some(delegated),
        scope: None,
        failure: None,
        scope_v2: None,
        failure_v2: None,
        scope_v3: None,
        failure_v3: None,
        admission: None,
        admission_evidence_error: None,
        finished_live: None,
        finished_evidence_error: None,
        finished_live_error: None,
        admission_attempted: false,
        admission_checkpoint_v3: AdmissionCheckpointV3::NotStarted,
        admission_v2_succeeded: false,
        admission_v3_succeeded: false,
        finished_attempted: false,
    })
}

/// Shared declaration/FD preparation only. Actual ownership, cgroup identity,
/// policy and self-entry are still validated by Scope::prepare with the SAME
/// launch lock in the pre-model callback. This never creates a native permit.
pub(crate) fn prepare_native_resource_paths(
    plan: &RoventNativeResourceLaunchPlan,
    original_private_root: &Dir,
) -> Result<(File, Dir), ArenaError> {
    plan.validate()?;
    let delegated = RoventTrainingResources::open_native_delegation(&plan.delegated_cgroup_path)?;
    let (parent, leaf) = parent_and_leaf(original_private_root, &plan.evidence_directory)?;
    parent.create_dir(leaf).map_err(invalid)?;
    let directory = parent.open_dir_nofollow(leaf).map_err(invalid)?;
    Ok((delegated, directory))
}

fn relative(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && path.as_os_str().len() <= 4096
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}
fn parent_and_leaf<'a>(
    root: &Dir,
    path: &'a Path,
) -> Result<(Dir, &'a std::ffi::OsStr), ArenaError> {
    if !relative(path) {
        return Err(invalid("bounded relative original path required"));
    }
    let mut parent = root.try_clone().map_err(invalid)?;
    let mut parts = path.components().peekable();
    while let Some(Component::Normal(part)) = parts.next() {
        if parts.peek().is_none() {
            return Ok((parent, part));
        }
        parent = parent.open_dir_nofollow(part).map_err(invalid)?;
    }
    Err(invalid("original basename missing"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provenance_rejection_pre_native_proof_requires_original_pre_scope_owner() {
        use rz_uci::rovent_bootstrap::launch::RoventRetainingLaunchError;
        // These descriptors are inert CPU mocks: this test never enters a
        // cgroup or starts a native runtime.
        let file = File::open("/dev/null").unwrap();
        let mut owner = NativeResourceCustody {
            system_v3: true,
            plan: RoventNativeResourceLaunchPlan {
                schema: ROVENT_RESOURCE_LAUNCH_PLAN.into(),
                delegated_cgroup_path: "/unused".into(),
                leaf_name: "unused".into(),
                options: RoventNativeResourceOptions {
                    pids_max: 1,
                    affinity: vec![0],
                    device: "unused".into(),
                },
                profile: RoventNativeResourceProfile::FunctionalValidation,
                evidence_directory: "unused".into(),
            },
            pin: RoventNativeResourcePlanPin {
                path: "inputs/mock.json".into(),
                sha256: "0".repeat(64),
                bytes: 1,
            },
            _plan_file: file.try_clone().unwrap(),
            directory: Dir::open_ambient_dir(std::env::temp_dir(), cap_std::ambient_authority())
                .unwrap(),
            delegated: Some(file),
            control: None,
            bound_control: None,
            control_cleanup_error: None,
            scope: None,
            failure: None,
            scope_v2: None,
            failure_v2: None,
            scope_v3: None,
            failure_v3: None,
            admission: None,
            admission_evidence_error: None,
            finished_live: None,
            finished_evidence_error: None,
            finished_live_error: None,
            admission_attempted: true,
            admission_checkpoint_v3: AdmissionCheckpointV3::ProvenanceRejected,
            admission_v2_succeeded: false,
            admission_v3_succeeded: false,
            finished_attempted: false,
        };
        let original_error = RoventRetainingLaunchError::Ordinary(RoventLaunchError {
            stage: "ResourceAdmission",
            detail: "original error".into(),
        });
        let evidence = owner
            .provenance_rejection_before_scope(37, true, &original_error, false)
            .unwrap();
        let json = serde_json::to_value(evidence).unwrap();
        assert_eq!(json["process_id"], std::process::id());
        assert_eq!(json["process_generation"], 37);
        assert_eq!(json["reason"], "resource_plan_provenance");
        assert_eq!(json["native_scope_started"], false);
        assert!(
            owner
                .provenance_rejection_before_scope(37, false, &original_error, false)
                .is_none()
        );
        assert!(
            owner
                .provenance_rejection_before_scope(37, true, &original_error, true)
                .is_none()
        );
        owner.admission_checkpoint_v3 = AdmissionCheckpointV3::ResourcePrepareStarted;
        assert!(
            owner
                .provenance_rejection_before_scope(37, true, &original_error, false)
                .is_none()
        );
        owner.admission_checkpoint_v3 = AdmissionCheckpointV3::ProvenanceRejected;
        owner.delegated = None;
        assert!(
            owner
                .provenance_rejection_before_scope(37, true, &original_error, false)
                .is_none()
        );
        let mut failure = RoventNativeResourceRunFailure {
            detail: "original error".into(),
            custody: None,
            system_v3: true,
            native_failure: None,
            native_recovery_attempts: Vec::new(),
            control_panic: None,
            control_panic_may_be_native: false,
            control_scope_marker: None,
            pre_native_admission: None,
        };
        assert!(failure.observation().get("pre_native_admission").is_none());
        failure.pre_native_admission = Some(evidence);
        assert_eq!(
            failure.observation()["pre_native_admission"]["process_generation"],
            37
        );
    }

    #[test]
    fn startup_failure_observation_preserves_supported_detail_or_retains_oversize() {
        use rz_search::pals::coordinator::RoleFailure;
        let failure = RoleFailure::Backend("original failure".into());
        let mut enough = "original failure".len();
        assert_eq!(
            observed_startup_failure(&failure, &mut enough),
            json!({"kind":"backend","detail":"original failure"})
        );
        assert_eq!(enough, 0);
        let mut insufficient = 0;
        assert_eq!(
            observed_startup_failure(&failure, &mut insufficient),
            json!({"kind":"backend","detail_retained_by_owner":true})
        );
        assert_eq!(
            observed_startup_failure(&RoleFailure::PhysicalUnknown, &mut insufficient),
            json!({"kind":"physical_unknown"})
        );
    }
    #[test]
    fn cpu_v3_receipts_require_original_child_ready_and_session_finish() {
        use rz_experiments::{
            NativeRuntimeSourceV6, PhysicalCompletionV6, ProviderReadinessV6,
            SYSTEM_RUNTIME_RECEIPT_V6_DOMAIN, SystemRuntimeLockV6, SystemRuntimeReceiptV6,
        };
        let lock = SystemRuntimeLockV6::new(NativeRuntimeSourceV6::Cpu).unwrap();
        let pid = std::process::id();
        let epoch = 7;
        let startup = SystemRuntimeReceiptV6 {
            domain: SYSTEM_RUNTIME_RECEIPT_V6_DOMAIN.into(),
            lock_sha256: lock.canonical_sha256.clone(),
            process_id: pid,
            process_generation: epoch,
            observed_identity_sha256: None,
            observed_libraries: Vec::new(),
            unregistered_native_mappings: None,
            provider_ready: ProviderReadinessV6::Ready,
            physical_completion: PhysicalCompletionV6::Unknown,
        };
        let mut termination = startup.clone();
        termination.physical_completion = PhysicalCompletionV6::Confirmed {
            completion_id: format!("session_finish_confirmed/1:{pid}:{epoch}"),
        };
        assert!(validate_system_v3_receipt_pair(&lock, pid, epoch, &startup, &termination).is_ok());
        assert!(
            validate_system_v3_receipt_pair(&lock, pid, epoch + 1, &startup, &termination).is_err()
        );
        let mut wrong_start = startup.clone();
        wrong_start.provider_ready = ProviderReadinessV6::Unknown;
        assert!(
            validate_system_v3_receipt_pair(&lock, pid, epoch, &wrong_start, &termination).is_err()
        );
        wrong_start = startup.clone();
        wrong_start.observed_identity_sha256 = Some("a".repeat(64));
        assert!(
            validate_system_v3_receipt_pair(&lock, pid, epoch, &wrong_start, &termination).is_err()
        );
        termination.physical_completion = PhysicalCompletionV6::Unknown;
        assert!(
            validate_system_v3_receipt_pair(&lock, pid, epoch, &startup, &termination).is_err()
        );
        termination.physical_completion = PhysicalCompletionV6::Confirmed {
            completion_id: "different".into(),
        };
        assert!(
            validate_system_v3_receipt_pair(&lock, pid, epoch, &startup, &termination).is_err()
        );
    }

    #[test]
    fn explicit_resource_cli_preserves_native_arguments_and_rejects_duplicate_or_escape() {
        let arguments = vec![
            "--resource-plan=inputs/resources.json".into(),
            format!("--resource-plan-sha256={}", "a".repeat(64)),
            "--resource-plan-bytes=42".into(),
            "--input-lock=inputs/lock.json".into(),
            "--native-assets=inputs/assets.json".into(),
        ];
        let RoventNativeResourceCli::Run {
            arguments: original,
            ..
        } = parse_native_resource_cli(arguments.clone()).unwrap()
        else {
            panic!("run expected")
        };
        assert_eq!(
            original,
            [
                "--input-lock=inputs/lock.json",
                "--native-assets=inputs/assets.json"
            ]
        );
        let mut duplicate = arguments.clone();
        duplicate.push("--resource-plan=inputs/second.json".into());
        assert!(parse_native_resource_cli(duplicate).is_err());
        let mut escape = arguments;
        escape[0] = "--resource-plan=inputs/../resources.json".into();
        assert!(parse_native_resource_cli(escape).is_err());
    }

    #[test]
    fn v3_resource_selection_and_receipt_root_are_explicit_and_bound() {
        let original = vec![
            "--resource-system-runtime-v3".into(),
            "--resource-plan=inputs/resources.json".into(),
            format!("--resource-plan-sha256={}", "a".repeat(64)),
            "--resource-plan-bytes=42".into(),
            "--input-lock=inputs/lock.json".into(),
            "--process-epoch=7".into(),
        ];
        let RoventNativeResourceCli::RunSystemV3 { mut arguments, .. } =
            parse_native_resource_cli(original.clone()).unwrap()
        else {
            panic!("explicit V3 selection expected")
        };
        assert_eq!(
            bind_system_v3_arguments(&mut arguments, Path::new("runs/one/resource")).unwrap(),
            7
        );
        assert!(arguments.contains(&"--system-runtime-receipt-root=runs/one/resource".to_owned()));
        let mut wrong = arguments.clone();
        wrong.push("--system-runtime-receipt-root=runs/other".into());
        assert!(bind_system_v3_arguments(&mut wrong, Path::new("runs/one/resource")).is_err());
        let mut duplicate = arguments;
        duplicate.push("--process-epoch=8".into());
        assert!(bind_system_v3_arguments(&mut duplicate, Path::new("runs/one/resource")).is_err());
        let mut duplicate_selection = original;
        duplicate_selection.push("--resource-system-runtime-v3".into());
        assert!(parse_native_resource_cli(duplicate_selection).is_err());
    }

    #[test]
    fn custody_control_is_explicit_v3_only() {
        let args = vec![
            "--resource-system-runtime-v3".to_owned(),
            "--custody-control".to_owned(),
            "--resource-plan=inputs/resources.json".to_owned(),
            format!("--resource-plan-sha256={}", "a".repeat(64)),
            "--resource-plan-bytes=42".to_owned(),
        ];
        let RoventNativeResourceCli::RunSystemV3 {
            custody_control,
            arguments,
            ..
        } = parse_native_resource_cli(args.clone()).unwrap()
        else {
            panic!("V3 selection expected")
        };
        assert!(custody_control);
        assert!(!arguments.iter().any(|value| value == "--custody-control"));
        let mut duplicate = args.clone();
        duplicate.push("--custody-control".to_owned());
        assert!(parse_native_resource_cli(duplicate).is_err());
        let ordinary = args
            .into_iter()
            .filter(|value| value != "--resource-system-runtime-v3")
            .collect();
        assert!(parse_native_resource_cli(ordinary).is_err());
    }

    #[test]
    fn readback_write_failures_do_not_skip_admission_or_finished_policy() {
        let path = std::env::temp_dir().join(format!(
            "rovent-v3-readback-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("admission-readback.json"), b"occupied").unwrap();
        let directory = Dir::open_ambient_dir(&path, cap_std::ambient_authority()).unwrap();
        let mut snapshot = RoventNativeResourceSnapshot {
            raw: json!({"fixture":"actual readback supplied by owner"}),
            applied: Some(true),
            threads: Some(1),
            limits: Some((1, 2)),
            inventory_complete: true,
        };
        let (observation, error, policy) = persist_admission_observation(
            &snapshot,
            &directory,
            RoventNativeResourceProfile::FunctionalValidation,
        );
        assert!(observation.is_none() && error.is_some() && policy.is_ok());
        snapshot.applied = Some(false);
        let (observation, error, policy) = persist_admission_observation(
            &snapshot,
            &directory,
            RoventNativeResourceProfile::FunctionalValidation,
        );
        assert!(observation.is_none() && error.is_some() && policy.is_err());
        std::fs::write(path.join("finished-live-readback.json"), b"occupied").unwrap();
        snapshot.applied = Some(true);
        let (observation, error, policy) = persist_finished_observation(
            &snapshot,
            &directory,
            RoventNativeResourceProfile::FunctionalValidation,
        );
        assert!(observation.is_none() && error.is_some() && policy.is_ok());
        snapshot.applied = Some(false);
        let (observation, error, policy) = persist_finished_observation(
            &snapshot,
            &directory,
            RoventNativeResourceProfile::FunctionalValidation,
        );
        assert!(observation.is_none() && error.is_some() && policy.is_err());
        drop(directory);
        std::fs::remove_file(path.join("admission-readback.json")).unwrap();
        std::fs::remove_file(path.join("finished-live-readback.json")).unwrap();
        std::fs::remove_dir(&path).unwrap();
    }
}
