//! Native UCI upper runner. It owns the same delegated resource scope before
//! the shared launch loads any model, and retains it after physical shutdown
//! or any failure. UCI parsing, native tickets and engine assembly stay in UCI.
use super::*;
use rz_uci::rovent_bootstrap::launch::{RoventLaunchError, run_with_resource_admission};
use std::{io::Read, path::PathBuf, sync::Mutex};

pub const ROVENT_RESOURCE_LAUNCH_PLAN: &str = "rz-rovent-native-resource-launch/1";
pub const ROVENT_RESOURCE_USAGE: &str = "rz-rovent-native --resource-plan=inputs/<plan.json> --resource-plan-sha256=<64 lowercase hex> --resource-plan-bytes=<positive bytes> <original Rovent source/config/FD arguments>";
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
    Run {
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
    let (mut path, mut sha256, mut bytes) = (None, None, None);
    let mut original = Vec::new();
    for argument in arguments {
        if let Some(value) = argument.strip_prefix("--resource-plan=") {
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
    Ok(RoventNativeResourceCli::Run {
        pin,
        arguments: original,
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

pub struct RoventNativeResourceRunFailure {
    pub detail: String,
    custody: Option<Arc<Mutex<NativeResourceCustody>>>,
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
    admission: Option<RoventResourceObservation>,
    finished_live: Option<RoventResourceObservation>,
    finished_live_error: Option<String>,
    admission_attempted: bool,
    admission_v2_succeeded: bool,
    finished_attempted: bool,
}
impl NativeResourceCustody {
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
        let result: Result<(), ArenaError> = (|| {
            let snapshot = self
                .scope
                .as_ref()
                .map(RoventNativeResourceScope::observe)
                .or_else(|| {
                    self.scope_v2
                        .as_ref()
                        .map(RoventNativeResourceScopeV2::observe)
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
    pub(crate) fn observation(&self) -> Value {
        json!({"schema":"rz-rovent-resource-custody/1","plan_pin":self.pin,
            "evidence_directory":self.plan.evidence_directory,"profile":self.plan.profile,
            "admission_attempted":self.admission_attempted,"finished_attempted":self.finished_attempted,
            "original_plan_descriptor_retained":true,"original_delegated_descriptor_retained":self.delegated.is_some()||self.scope.is_some()||self.failure.is_some()||self.scope_v2.is_some()||self.failure_v2.is_some(),
            "original_scope_retained":self.scope.is_some()||self.scope_v2.is_some(),"resource_failure":self.failure.as_ref().map(|failure|failure.observation()).or_else(||self.failure_v2.as_ref().map(|failure|failure.observation())),
            "admission":self.admission,"finished_live":self.finished_live,"finished_live_error":self.finished_live_error,
            "controlled_comparison_accepted":false,"automatic_cleanup":false})
    }
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
        plan,
        pin,
        _plan_file: plan_file.into_std(),
        directory,
        delegated: Some(delegated),
        scope: None,
        failure: None,
        scope_v2: None,
        failure_v2: None,
        admission: None,
        finished_live: None,
        finished_live_error: None,
        admission_attempted: false,
        admission_v2_succeeded: false,
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
}
