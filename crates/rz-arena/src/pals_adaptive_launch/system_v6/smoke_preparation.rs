//! Non-executing, selected-role proxy smoke input preparation.
//!
//! This owns the normal admission lease and real readonly copies, but cannot
//! be converted into a paired `NativeLaunchOwner` or start a process. Observe
//! source files, Fastchess, opening and the other role are not read here.

use super::no_game_plan::{LockedNoGameSmokePlanV1, NoGameExecutionStartV1};
use crate::{
    ArenaError,
    native_launch::{
        self, NativeAdmissionLease, NativePreparationFailure, NativePreparationReceipt,
        NativeSnapshotReceipt, linux,
    },
};
use cap_fs_ext::DirExt;
use cap_std::fs::Dir;
use rz_experiments::{
    ArtifactRef, NativeEngineRole, NativeRuntimeSourceV6, PalsStartupAssetPurposeV5,
    PalsSystemSourcePolicyV6,
};
use std::{
    collections::BTreeMap,
    fs::{File, Permissions},
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    sync::{Arc, atomic::AtomicBool},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SmokePreparedSourceObservation {
    /// Strict input rows were copied and hash-checked; running source not proven.
    StrictInputsPinned,
    /// Observe files were intentionally not read; no conclusion follows.
    ObserveUnknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SmokePreparationUnobserved {
    Unknown,
}

/// Stages apply only to this selected NoGame preparation attempt. They do not
/// describe the result of an independent CPU or another engine execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NoGamePreparationStage {
    Admission,
    SelectedInputs,
    SourceAssets,
    OutputDirectory,
    SnapshotCopy,
    RuntimeDirectory,
}

pub(crate) struct NoGamePreparationFailure {
    pub(crate) stage: NoGamePreparationStage,
    pub(crate) original: NativePreparationFailure,
}

/// Retains the exact lease and pinned files. These fields provide no process
/// start, system provider, source mapping or physical-completion capability.
pub(crate) struct ProxyEntrySmokeSnapshotOwnerV1 {
    _lease: NativeAdmissionLease,
    pub(crate) execution: Arc<NoGameExecutionStartV1>,
    pub(super) binding_active: AtomicBool,
    pub(crate) plan: LockedNoGameSmokePlanV1,
    pub(crate) role: NativeEngineRole,
    pub(crate) directory: Dir,
    pub(crate) input_directory: Dir,
    pub(crate) input_subdirectories: BTreeMap<PathBuf, Dir>,
    pub(crate) cwd: File,
    pub(crate) path: PathBuf,
    pub(crate) runtime_directory: Dir,
    pub(crate) runtime_pin: File,
    pub(crate) pins: Vec<linux::InputPin>,
    pub(crate) receipts: Vec<NativeSnapshotReceipt>,
    pub(crate) source_observation: SmokePreparedSourceObservation,
    pub(crate) environment_application: SmokePreparationUnobserved,
    pub(crate) child_runtime_receipt: SmokePreparationUnobserved,
}

#[derive(Clone)]
struct PlannedInput {
    artifact: ArtifactRef,
    /// Relative to inputs, never a caller-controlled absolute destination.
    name: String,
    executable: bool,
}

fn invalid(message: &'static str) -> ArenaError {
    ArenaError::Invalid(message.into())
}
fn io(message: &'static str) -> ArenaError {
    ArenaError::Io(message.into())
}

fn add_input(
    inputs: &mut BTreeMap<String, PlannedInput>,
    artifact: &ArtifactRef,
    snapshot_path: &str,
    executable: bool,
) -> Result<(), ArenaError> {
    artifact.validate()?;
    let name = snapshot_path
        .strip_prefix("inputs/")
        .ok_or_else(|| invalid("smoke input requires registered inputs path"))?;
    if name.is_empty() {
        return Err(invalid("smoke input name missing"));
    }
    match inputs.get_mut(&artifact.path) {
        Some(previous) if previous.artifact == *artifact && previous.name == name => {
            previous.executable |= executable;
        }
        Some(_) => {
            return Err(ArenaError::Integrity(
                "smoke source identity or destination differs".into(),
            ));
        }
        None => {
            inputs.insert(
                artifact.path.clone(),
                PlannedInput {
                    artifact: artifact.clone(),
                    name: name.into(),
                    executable,
                },
            );
        }
    }
    Ok(())
}

fn selected_inputs(
    plan: &LockedNoGameSmokePlanV1,
) -> Result<(Vec<PlannedInput>, SmokePreparedSourceObservation), ArenaError> {
    let input = plan.input();
    let endpoint = &input.endpoint;
    endpoint.startup_lock.verify()?;
    let mut inputs = BTreeMap::new();
    add_input(
        &mut inputs,
        &endpoint.startup_lock_artifact,
        &endpoint.startup_snapshot_path,
        false,
    )?;
    add_input(
        &mut inputs,
        &endpoint.startup_lock.config.endpoint.binary,
        &endpoint.binary_snapshot_path,
        true,
    )?;
    add_input(
        &mut inputs,
        &input.proxy_binary,
        &input.proxy_snapshot_path,
        true,
    )?;
    for asset in &endpoint.startup_lock.config.assets {
        add_input(
            &mut inputs,
            &asset.artifact,
            &asset.snapshot_path,
            asset.purpose == PalsStartupAssetPurposeV5::ExternalBinary,
        )?;
    }
    let source_observation = match endpoint.startup_lock.config.source_policy {
        PalsSystemSourcePolicyV6::Strict => {
            for files in endpoint.source_files.values() {
                for file in files {
                    add_input(&mut inputs, &file.artifact, &file.snapshot_path, false)?;
                }
            }
            SmokePreparedSourceObservation::StrictInputsPinned
        }
        PalsSystemSourcePolicyV6::Observe => SmokePreparedSourceObservation::ObserveUnknown,
    };
    if let Some(environment) = &endpoint.environment {
        // Native preparation normally supplies a generated name when this
        // launcher has no explicit snapshot mapping. Preserve that behavior
        // in a smoke-specific namespace without reading another role.
        let name = if let Some(existing) = inputs.get(&environment.launcher.path) {
            format!("inputs/{}", existing.name)
        } else {
            // Pick a free generated basename rather than reserving a magic
            // path that might be a legitimate selected asset name.
            let index = (0..=inputs.len())
                .find(|index| {
                    let name = format!("smoke-environment-launcher-{index}");
                    !inputs.values().any(|item| item.name == name)
                })
                .ok_or_else(|| invalid("smoke launcher snapshot name unavailable"))?;
            format!("inputs/smoke-environment-launcher-{index}")
        };
        add_input(&mut inputs, &environment.launcher, &name, true)?;
    }
    let runtime = &endpoint.startup_lock.config.system_runtime_lock.runtime;
    runtime.validate()?;
    if let NativeRuntimeSourceV6::SystemCuda(system) = runtime {
        if system.platform != rz_experiments::SystemPlatformV6::Linux {
            return Err(invalid("smoke system CUDA requires Linux"));
        }
        if inputs.values().any(|input| {
            system.libraries.iter().any(|library| {
                input.artifact.bytes == library.bytes && input.artifact.sha256 == library.sha256
            })
        }) {
            return Err(invalid(
                "smoke system runtime libraries must not be snapshotted",
            ));
        }
    }
    let total = inputs
        .values()
        .try_fold(0u64, |total, input| total.checked_add(input.artifact.bytes))
        .ok_or_else(|| ArenaError::Budget("smoke input size overflow".into()))?;
    if total > input.budget.max_input_bytes {
        return Err(ArenaError::Budget("smoke input aggregate exceeded".into()));
    }
    let budget = &input.budget;
    let reserved = total
        .checked_add(budget.max_runtime_bytes)
        .and_then(|value| value.checked_add(budget.max_output_bytes))
        .ok_or_else(|| ArenaError::Budget("smoke artifact reservation overflow".into()))?;
    if reserved > budget.max_artifact_bytes || budget.max_runtime_files == 0 {
        return Err(ArenaError::Budget(
            "smoke declared artifact budget exceeded".into(),
        ));
    }
    let inputs: Vec<_> = inputs.into_values().collect();
    let names: Vec<_> = inputs.iter().map(|input| input.name.clone()).collect();
    linux::snapshot_parents(&names, usize::from(input.budget.max_runtime_depth))?;
    Ok((inputs, source_observation))
}

pub(super) fn validate_selection(plan: &LockedNoGameSmokePlanV1) -> Result<(), ArenaError> {
    selected_inputs(plan).map(|_| ())
}

/// Retain the original execution Arc and normal single-active admission lease.
/// Preparation does not prove a child receipt consumer or permit a run.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_proxy_entry_smoke_snapshot_v1(
    plan: &LockedNoGameSmokePlanV1,
    execution: &Arc<NoGameExecutionStartV1>,
    owner_id: [u8; 16],
    owner_epoch: u64,
    source_root: &Path,
    output_root: &Path,
    output_directory: &str,
) -> Result<ProxyEntrySmokeSnapshotOwnerV1, Box<NoGamePreparationFailure>> {
    let mut trace = NativePreparationReceipt {
        receipt_version: 1,
        policy_envelope: None,
        execution_ready: false,
        input_sha256: plan.sha256().into(),
        output_directory: if native_launch::safe_output_basename(output_directory) {
            output_directory.into()
        } else {
            "rejected-basename".into()
        },
        attempt_created: false,
        attempted_snapshot_relative_paths: Vec::new(),
        completed_snapshots: Vec::new(),
        child_spawned: false,
        writers_closed: true,
        input_pins_required: false,
        original_error: String::new(),
        subsequent_file_owner: "caller retains NoGame smoke preparation files; no child started"
            .into(),
        automatic_retry: false,
    };
    let mut retained_directory = None;
    let mut stage = NoGamePreparationStage::Admission;
    let result = (|| -> Result<_, ArenaError> {
        let lease = NativeAdmissionLease::acquire()?;
        if !native_launch::safe_output_basename(output_directory) {
            return Err(invalid("smoke attempt requires a safe unique basename"));
        }
        execution.require_running()?;
        if !execution.matches(plan, owner_id, owner_epoch) {
            return Err(invalid("smoke original execution identity differs"));
        }
        let role = plan.input().declared_role;
        stage = NoGamePreparationStage::SelectedInputs;
        let (inputs, source_observation) = selected_inputs(plan)?;
        let names: Vec<_> = inputs.iter().map(|item| item.name.clone()).collect();
        let parents =
            linux::snapshot_parents(&names, usize::from(plan.input().budget.max_runtime_depth))?;
        stage = NoGamePreparationStage::SourceAssets;
        let sources = inputs
            .iter()
            .map(|item| {
                execution.require_running()?;
                let file = item
                    .artifact
                    .open_verified(source_root, plan.input().budget.max_input_bytes)?;
                execution.require_running()?;
                Ok(file)
            })
            .collect::<Result<Vec<_>, ArenaError>>()?;
        execution.require_running()?;
        stage = NoGamePreparationStage::OutputDirectory;
        let root_path = linux::outside_git(output_root)?;
        let root = Dir::open_ambient_dir(&root_path, cap_std::ambient_authority())
            .map_err(|_| io("cannot pin smoke output root"))?;
        root.create_dir(output_directory)
            .map_err(|_| io("cannot create exclusive smoke attempt"))?;
        trace.attempt_created = true;
        let directory = root
            .open_dir_nofollow(output_directory)
            .map_err(|_| io("cannot pin smoke attempt"))?;
        retained_directory = Some(
            directory
                .try_clone()
                .map_err(|_| io("cannot retain smoke failure directory"))?,
        );
        linux::readable_directory_pin(&directory)?
            .set_permissions(Permissions::from_mode(0o700))
            .map_err(|_| io("cannot make smoke attempt private"))?;
        directory
            .create_dir("inputs")
            .map_err(|_| io("cannot create smoke inputs"))?;
        let input_directory = directory
            .open_dir_nofollow("inputs")
            .map_err(|_| io("cannot pin smoke inputs"))?;
        linux::readable_directory_pin(&input_directory)?
            .set_permissions(Permissions::from_mode(0o700))
            .map_err(|_| io("cannot make smoke inputs private"))?;
        let path = root_path.join(output_directory);
        let mut subdirectories = BTreeMap::<PathBuf, Dir>::new();
        for relative in parents {
            let parent_path = relative
                .parent()
                .ok_or_else(|| invalid("smoke input parent absent"))?;
            let parent = if parent_path.as_os_str().is_empty() {
                &input_directory
            } else {
                subdirectories
                    .get(parent_path)
                    .ok_or_else(|| invalid("smoke input parent not pinned"))?
            };
            let name = relative
                .file_name()
                .ok_or_else(|| invalid("smoke input name absent"))?;
            parent
                .create_dir(name)
                .map_err(|_| io("cannot create smoke input subdirectory"))?;
            let child = parent
                .open_dir_nofollow(name)
                .map_err(|_| io("cannot pin smoke input subdirectory"))?;
            linux::readable_directory_pin(&child)?
                .set_permissions(Permissions::from_mode(0o700))
                .map_err(|_| io("cannot make smoke input subdirectory private"))?;
            subdirectories.insert(relative, child);
        }
        let mut pins = Vec::new();
        let mut receipts = Vec::new();
        stage = NoGamePreparationStage::SnapshotCopy;
        for (item, source) in inputs.iter().zip(sources) {
            execution.require_running()?;
            let relative = PathBuf::from("inputs").join(&item.name);
            trace
                .attempted_snapshot_relative_paths
                .push(relative.to_string_lossy().into_owned());
            let (parent, name) = linux::input_parent(&input_directory, &subdirectories, &relative)?;
            let (pin, receipt) = linux::readonly_copy(
                &item.artifact,
                source,
                parent,
                name.to_str()
                    .ok_or_else(|| invalid("smoke input filename invalid"))?,
                path.join(&relative),
                item.executable,
                plan.input().budget.max_input_bytes,
            )?;
            let receipt = NativeSnapshotReceipt {
                snapshot_relative_path: relative.to_string_lossy().into_owned(),
                ..receipt
            };
            trace.completed_snapshots.push(receipt.clone());
            pins.push(pin);
            receipts.push(receipt);
            execution.require_running()?;
        }
        for child in subdirectories.values() {
            linux::readable_directory_pin(child)?
                .set_permissions(Permissions::from_mode(0o500))
                .map_err(|_| io("cannot close smoke input subdirectory"))?;
        }
        linux::readable_directory_pin(&input_directory)?
            .set_permissions(Permissions::from_mode(0o500))
            .map_err(|_| io("cannot close smoke input directory"))?;
        stage = NoGamePreparationStage::RuntimeDirectory;
        let runtime_name = &plan.input().endpoint.startup_lock.config.runtime_directory;
        directory
            .create_dir(runtime_name)
            .map_err(|_| io("cannot create smoke runtime root"))?;
        let runtime_directory = directory
            .open_dir_nofollow(runtime_name)
            .map_err(|_| io("cannot pin smoke runtime root"))?;
        let runtime_pin = linux::readable_directory_pin(&runtime_directory)?;
        runtime_pin
            .set_permissions(Permissions::from_mode(0o700))
            .map_err(|_| io("cannot make smoke runtime root private"))?;
        let cwd = directory
            .try_clone()
            .map_err(|_| io("cannot clone smoke cwd"))?
            .into_std_file();
        execution.require_running()?;
        Ok(ProxyEntrySmokeSnapshotOwnerV1 {
            _lease: lease,
            execution: Arc::clone(execution),
            binding_active: AtomicBool::new(false),
            plan: plan.clone(),
            role,
            directory,
            input_directory,
            input_subdirectories: subdirectories,
            cwd,
            path,
            runtime_directory,
            runtime_pin,
            pins,
            receipts,
            source_observation,
            environment_application: SmokePreparationUnobserved::Unknown,
            child_runtime_receipt: SmokePreparationUnobserved::Unknown,
        })
    })();
    match result {
        Ok(owner) => Ok(owner),
        Err(cause) => {
            trace.original_error = format!("NoGame {stage:?}: {cause}")
                .chars()
                .take(4096)
                .collect();
            let persisted = if trace.attempt_created {
                linux::save_preparation_failure(
                    retained_directory.as_ref(),
                    &trace,
                    "pals-proxy-entry-functional-smoke",
                )
            } else {
                Ok(None)
            };
            let (receipt_artifact, persistence_error) = match persisted {
                Ok(artifact) => (artifact, None),
                Err(error) => (None, Some(error)),
            };
            Err(Box::new(NoGamePreparationFailure {
                stage,
                original: NativePreparationFailure {
                    cause,
                    receipt: trace,
                    receipt_artifact,
                    persistence_error,
                },
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pals_adaptive_launch::system_v6::{
        no_game_plan::{
            NO_GAME_SMOKE_PLAN_DOMAIN, NoGameSmokeBudgetV1, NoGameSmokeEndpointV1,
            NoGameSmokePlanV1,
        },
        pair::tests::fixture,
    };

    #[test]
    fn smoke_plan_selects_role_inputs_without_fastchess_or_other_lock() {
        let mut pair = fixture();
        pair.budget.max_child_processes = 2;
        // Paired execution retains its five-process requirement; this test
        // registers only an independent direct-child NoGame plan.
        assert!(pair.clone().lock().is_err());
        let mut proxy = pair.runner.binary.clone();
        proxy.path = "test/independent-proxy".into();
        let plan = NoGameSmokePlanV1 {
            domain: NO_GAME_SMOKE_PLAN_DOMAIN.into(),
            declared_role: NativeEngineRole::Baseline,
            endpoint: NoGameSmokeEndpointV1::from(&pair.endpoints[0]),
            proxy_binary: proxy.clone(),
            proxy_snapshot_path: "inputs/test/independent-proxy".into(),
            budget: NoGameSmokeBudgetV1 {
                max_input_bytes: 1 << 30,
                max_runtime_bytes: 4 << 20,
                max_output_bytes: 4 << 20,
                max_artifact_bytes: (1 << 30) + (8 << 20),
                max_child_processes: 2,
                max_runtime_files: 16,
                max_runtime_depth: 4,
                address_space_per_process_bytes: 1 << 30,
            },
            paired_source_sha256: None,
        }
        .lock()
        .unwrap();
        let (inputs, source) = selected_inputs(&plan).unwrap();
        let paths: Vec<_> = inputs
            .iter()
            .map(|input| input.artifact.path.as_str())
            .collect();
        assert!(paths.contains(&proxy.path.as_str()));
        assert!(paths.contains(&pair.endpoints[0].startup_lock_artifact.path.as_str()));
        assert!(!paths.contains(&pair.endpoints[1].startup_lock_artifact.path.as_str()));
        assert!(!paths.contains(&pair.runner.binary.path.as_str()));
        assert!(!paths.contains(&pair.opening_artifact.path.as_str()));
        assert_eq!(plan.input().budget.max_child_processes, 2);
        assert_eq!(source, SmokePreparedSourceObservation::ObserveUnknown);
        let execution = NoGameExecutionStartV1::begin(&plan, [7; 16], 1).unwrap();
        assert!(execution.matches(&plan, [7; 16], 1));
        assert!(!execution.matches(&plan, [8; 16], 1));
        assert!(execution.started() < execution.run_until());
        assert!(execution.run_until() < execution.cleanup_until());
    }

    #[test]
    fn smoke_plan_rejects_contradictory_snapshot_mapping() {
        let pair = fixture().lock().unwrap();
        let artifact = &pair
            .endpoint(NativeEngineRole::Baseline)
            .startup_lock_artifact;
        let mut inputs = BTreeMap::new();
        add_input(&mut inputs, artifact, "inputs/a", false).unwrap();
        add_input(&mut inputs, artifact, "inputs/a", true).unwrap();
        assert!(inputs.values().next().unwrap().executable);
        assert!(add_input(&mut inputs, artifact, "inputs/b", false).is_err());
        assert_eq!(inputs.len(), 1);
    }
}
