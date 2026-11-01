//! Native UCI upper runner. It owns the same delegated resource scope before
//! the shared launch loads any model, and retains it after physical shutdown
//! or any failure. UCI parsing, native tickets and engine assembly stay in UCI.
use super::*;
use rz_uci::rovent_bootstrap::launch::{RoventLaunchError, run_with_resource_admission};
use std::{io::Read, path::PathBuf, sync::Mutex};

pub const ROVENT_RESOURCE_LAUNCH_PLAN: &str = "rz-rovent-native-resource-launch/1";
pub const ROVENT_RESOURCE_USAGE: &str = "rz-rovent-native [--resource-system-runtime-v3] --resource-plan=inputs/<plan.json> --resource-plan-sha256=<64 lowercase hex> --resource-plan-bytes=<positive bytes> <original Rovent source/config/FD arguments>";
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
    let mut original = Vec::new();
    for argument in arguments {
        if argument == "--resource-system-runtime-v3" {
            if std::mem::replace(&mut system_v3, true) {
                return Err(invalid("duplicate system runtime resource selection"));
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
    Ok(if system_v3 {
        RoventNativeResourceCli::RunSystemV3 {
            pin,
            arguments: original,
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

/// V3 system execution retains the original resource owner and the exact
/// child observations. A live in-process resource readback is not a separate
/// post-exit process-tree or independent GPU completion observation.
pub struct RoventNativeSystemResourceRunReceiptV3 {
    pub admission: Option<RoventResourceObservation>,
    pub finished_live: Option<RoventResourceObservation>,
    pub admission_evidence_error: Option<String>,
    pub finished_evidence_error: Option<String>,
    pub finished_policy_observation_error: Option<String>,
    startup: Option<rz_experiments::SystemRuntimeReceiptV6>,
    termination: Option<rz_experiments::SystemRuntimeReceiptV6>,
    system_runtime_evidence_error: Option<String>,
    system_lock_sha256: String,
    process_id: u32,
    process_generation: u64,
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
            "system_lock_sha256":self.system_lock_sha256,
            "process_id":self.process_id,"process_generation":self.process_generation,
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

pub struct RoventNativeResourceRunFailure {
    pub detail: String,
    custody: Option<Arc<Mutex<NativeResourceCustody>>>,
    system_v3: bool,
}
impl std::fmt::Debug for RoventNativeResourceRunFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoventNativeResourceRunFailure")
            .field("detail", &self.detail)
            .field("original_custody_retained", &self.custody.is_some())
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
    pub fn observation(&self) -> Value {
        let held = self
            .custody
            .as_ref()
            .and_then(|custody| custody.lock().ok());
        if self.system_v3 {
            return json!({"schema":"rz-rovent-native-system-resource-run-failure/3","error":self.detail,
                "original_custody_retained":self.custody.is_some(),
                "evidence_directory":held.as_ref().map(|state|&state.plan.evidence_directory),
                "resource_failure":held.as_ref().and_then(|state|state.failure_v3.as_ref().map(|failure|failure.observation())),
                "admission":held.as_ref().and_then(|state|state.admission.as_ref()),
                "admission_evidence_error":held.as_ref().and_then(|state|state.admission_evidence_error.as_ref()),
                "finished_live":held.as_ref().and_then(|state|state.finished_live.as_ref()),
                "finished_evidence_error":held.as_ref().and_then(|state|state.finished_evidence_error.as_ref()),
                "finished_policy_observation_error":held.as_ref().and_then(|state|state.finished_live_error.as_ref()),
                "automatic_cleanup":false,"training_executed":false});
        }
        json!({"schema":"rz-rovent-native-resource-run-failure/1","error":self.detail,
            "original_custody_retained":self.custody.is_some(),
            "evidence_directory":held.as_ref().map(|state|&state.plan.evidence_directory),
            "resource_failure":held.as_ref().and_then(|state|state.failure.as_ref().map(|failure|failure.observation())),
            "admission":held.as_ref().and_then(|state|state.admission.as_ref()),
            "finished_live":held.as_ref().and_then(|state|state.finished_live.as_ref()),
            "finished_live_error":held.as_ref().and_then(|state|state.finished_live_error.as_ref()),
            "automatic_cleanup":false,"training_executed":false})
    }
}

pub(crate) struct NativeResourceCustody {
    system_v3: bool,
    plan: RoventNativeResourceLaunchPlan,
    pin: RoventNativeResourcePlanPin,
    // Hold the exact plan source and output directory through every return.
    _plan_file: File,
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
    admission_v2_succeeded: bool,
    admission_v3_succeeded: bool,
    finished_attempted: bool,
}
impl NativeResourceCustody {
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
            return Err(invalid(
                "resource plan is not the original unique /3 input-lock provenance file",
            ));
        }
        let delegated = self
            .delegated
            .take()
            .ok_or_else(|| invalid("original delegated descriptor missing"))?;
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
            persist_v3_admission(&snapshot, &self.directory, self.plan.profile);
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
        self.admission =
            Some(snapshot.write_evidence(&self.directory, Path::new("admission-readback.json"))?);
        snapshot.require_admission(self.plan.profile)?;
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
        if let Some(scope) = &self.scope_v3 {
            let snapshot = scope.observe();
            match snapshot.write_evidence(&self.directory, Path::new("finished-live-readback.json"))
            {
                Ok(observation) => self.finished_live = Some(observation),
                Err(error) => self.finished_evidence_error = Some(error.to_string()),
            }
            // This is a post-execution observation/acceptance decision. A
            // missing file or unobserved final controls cannot rewrite the
            // already-completed engine result; retain the original reason.
            if let Err(error) = snapshot.require_finished_profile(self.plan.profile) {
                self.finished_live_error = Some(error.to_string());
            }
            return Ok(());
        }
        let result: Result<(), ArenaError> = (|| {
            let snapshot = self
                .scope
                .as_ref()
                .map(RoventNativeResourceScope::observe)
                .or_else(|| {
                    self.scope_v2
                        .as_ref()
                        .map(RoventNativeResourceScopeV2::observe)
                })
                .or_else(|| {
                    self.scope_v3
                        .as_ref()
                        .map(RoventNativeResourceScopeV3::observe)
                });
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
            || self.finished_live.is_none()
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
}

/// Persisting the readback is diagnostic; actual kernel-policy admission is
/// still checked even when the evidence file cannot be created.
fn persist_v3_admission(
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
    use rz_experiments::{NativeRuntimeSourceV6, PhysicalCompletionV6, ProviderReadinessV6};
    let scope = state.original_v3_scope()?;
    let lock = &scope.input_lock().manifest.system_runtime_lock;
    let system = match &lock.runtime {
        NativeRuntimeSourceV6::SystemCuda(system) => system,
        NativeRuntimeSourceV6::Cpu => {
            return Err(invalid("V3 system execution requires explicit system CUDA"));
        }
    };
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
    if startup.provider_ready != ProviderReadinessV6::Ready
        || startup.physical_completion != PhysicalCompletionV6::Unknown
        || startup.observed_identity_sha256.as_deref()
            != Some(system.system_identity_sha256.as_str())
        || !startup.unregistered_native_mappings.is_empty()
    {
        return Err(invalid(
            "V3 startup system runtime observation is incomplete",
        ));
    }
    system.validate_ready_observed(&startup.observed_libraries)?;
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
    Ok((startup, termination))
}

/// The actual shared launch callback runs after its original lock/process
/// admission and before model/ORT/teacher creation. No UCI/search implementation
/// or private native authority is reconstructed by this upper runner.
pub fn run_native_with_resources(
    pin: RoventNativeResourcePlanPin,
    arguments: Vec<String>,
) -> Result<RoventNativeResourceRunReceipt, RoventNativeResourceRunFailure> {
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
    })?;
    let custody = Arc::new(Mutex::new(state));
    let callback = Arc::clone(&custody);
    let result = run_with_resource_admission(arguments, move |lock| {
        callback
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
    });
    let finish = custody
        .lock()
        .map_err(|_| invalid("original resource owner poisoned"))
        .and_then(|mut state| state.finish_live());
    let failure = |detail: String| RoventNativeResourceRunFailure {
        detail,
        custody: Some(Arc::clone(&custody)),
        system_v3: false,
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
    let admission = state
        .admission
        .clone()
        .ok_or_else(|| failure("actual resource admission did not run".into()))?;
    let finished_live = state
        .finished_live
        .clone()
        .ok_or_else(|| failure("actual finished resource snapshot missing".into()))?;
    drop(state);
    Ok(RoventNativeResourceRunReceipt {
        admission,
        finished_live,
        custody,
    })
}

/// Explicit /3 system CUDA launch under its original resource owner. It uses
/// the very same /3 lock Arc admitted by UCI, never a converted older lock.
/// Only child-owned session-finish evidence is accepted here: a separate outer
/// supervisor must still observe process exit and any independent GPU fence.
pub fn run_native_with_resources_v3(
    pin: RoventNativeResourcePlanPin,
    mut arguments: Vec<String>,
) -> Result<RoventNativeSystemResourceRunReceiptV3, RoventNativeResourceRunFailure> {
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
    })?;
    state.system_v3 = true;
    let custody = Arc::new(Mutex::new(state));
    let failure = |detail: String| RoventNativeResourceRunFailure {
        detail,
        custody: Some(Arc::clone(&custody)),
        system_v3: true,
    };
    let epoch = {
        let state = custody
            .lock()
            .map_err(|_| failure("original resource owner poisoned".into()))?;
        bind_system_v3_arguments(&mut arguments, &state.plan.evidence_directory)
            .map_err(|error| failure(error.to_string()))?
    };
    let callback = Arc::clone(&custody);
    let result =
        rz_uci::rovent_bootstrap::launch::run_with_v3_resource_admission(arguments, move |lock| {
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
        });
    let finish = custody
        .lock()
        .map_err(|_| invalid("original resource owner poisoned"))
        .and_then(|mut state| state.finish_live());
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
    drop(state);
    Ok(RoventNativeSystemResourceRunReceiptV3 {
        admission,
        finished_live,
        admission_evidence_error,
        finished_evidence_error,
        finished_policy_observation_error,
        startup,
        termination,
        system_runtime_evidence_error,
        system_lock_sha256,
        process_id: std::process::id(),
        process_generation: epoch,
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
    fn v3_admission_preserves_write_failure_without_skipping_kernel_policy() {
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
        let (observation, error, policy) = persist_v3_admission(
            &snapshot,
            &directory,
            RoventNativeResourceProfile::FunctionalValidation,
        );
        assert!(observation.is_none() && error.is_some() && policy.is_ok());
        snapshot.applied = Some(false);
        let (observation, error, policy) = persist_v3_admission(
            &snapshot,
            &directory,
            RoventNativeResourceProfile::FunctionalValidation,
        );
        assert!(observation.is_none() && error.is_some() && policy.is_err());
        drop(directory);
        std::fs::remove_file(path.join("admission-readback.json")).unwrap();
        std::fs::remove_dir(&path).unwrap();
    }
}
