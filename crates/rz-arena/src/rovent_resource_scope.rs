//! Native UCI/Collection resource custody reusing the trainer's original Linux
//! delegated-cgroup owner. No training, model, process spawn or global setting
//! is introduced here. The same-process upper runner enters before loading its
//! native model and keeps this owner across every success/failure return.
#![cfg(target_os = "linux")]

use crate::ArenaError;
use crate::rovent_training_process::{
    RoventNativeProcessEntry, RoventTrainingResourceFailure, RoventTrainingResourcePolicy,
    RoventTrainingResources,
};
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::Dir;
use rz_experiments::{
    ArtifactRef, RoventDevice, RoventInputLock, RoventInputLockV2, RoventObserved,
    RoventResourceObservation, RoventResourcePlan,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::{
    fs::File,
    io::Write,
    path::{Component, Path},
    sync::Arc,
};

pub const ROVENT_NATIVE_RESOURCE_EVIDENCE: &str = "rz-rovent-native-resource-readback/1";
// The schema names these local readback bytes; ArtifactRef.source identifies
// their public producer source, not a schema/domain or a remote evidence copy.
const ROVENT_NATIVE_RESOURCE_SOURCE: &str = "https://github.com/daejunnom/RoveZero";
const MAX_EVIDENCE_BYTES: usize = 4 << 20;

pub mod launch;

/// Declaration only. CPU/memory come from the same exact input lock; these
/// remaining physical bounds are explicit and are checked by the OS owner.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventNativeResourceOptions {
    pub pids_max: u32,
    pub affinity: Vec<u32>,
    pub device: String,
}

/// Both profiles require actual configured controls and a stable current tree
/// before model loading. Strict comparison separately requires complete plan
/// readback after execution; unobserved final peaks never prevent measurement.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RoventNativeResourceProfile {
    FunctionalValidation,
    StrictComparison,
}

pub struct RoventNativeResourceScope {
    lock: Arc<RoventInputLock>,
    resources: Arc<RoventTrainingResources>,
    entry: RoventNativeProcessEntry,
}

/// The same OS resource owner for a native /2 launch. The original typed
/// lock is retained rather than converting it into a /1 declaration.
pub struct RoventNativeResourceScopeV2 {
    lock: Arc<RoventInputLockV2>,
    resources: Arc<RoventTrainingResources>,
    entry: RoventNativeProcessEntry,
}

pub enum RoventNativeResourceFailureV2 {
    BeforePreparation {
        error: ArenaError,
        lock: Arc<RoventInputLockV2>,
        delegated: File,
    },
    Preparation {
        failure: Box<RoventTrainingResourceFailure>,
        lock: Arc<RoventInputLockV2>,
    },
    Entry {
        error: ArenaError,
        lock: Arc<RoventInputLockV2>,
        resources: Arc<RoventTrainingResources>,
    },
}

impl std::fmt::Debug for RoventNativeResourceFailureV2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BeforePreparation { error, .. } => {
                f.debug_tuple("BeforePreparation").field(error).finish()
            }
            Self::Preparation { failure, .. } => {
                f.debug_tuple("Preparation").field(failure).finish()
            }
            Self::Entry { error, .. } => f.debug_tuple("Entry").field(error).finish(),
        }
    }
}
impl std::fmt::Display for RoventNativeResourceFailureV2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BeforePreparation { error, .. } | Self::Entry { error, .. } => error.fmt(f),
            Self::Preparation { failure, .. } => failure.error.fmt(f),
        }
    }
}
impl std::error::Error for RoventNativeResourceFailureV2 {}
impl RoventNativeResourceFailureV2 {
    pub fn observation(&self) -> Value {
        match self {
            Self::BeforePreparation { error, lock, .. } => {
                json!({"schema":"rz-rovent-native-resource-failure/2",
                "stage":"before_preparation","input_lock_sha256":lock.canonical_sha256,"error":error.to_string(),"delegated_descriptor_retained":true,"automatic_cleanup":false})
            }
            Self::Preparation { failure, lock } => {
                json!({"schema":"rz-rovent-native-resource-failure/2",
                "stage":"preparation","input_lock_sha256":lock.canonical_sha256,"resource_failure":failure.observation(),"automatic_cleanup":false})
            }
            Self::Entry {
                error,
                lock,
                resources,
            } => json!({"schema":"rz-rovent-native-resource-failure/2",
                "stage":"self_entry_or_readback","input_lock_sha256":lock.canonical_sha256,"error":error.to_string(),
                "resource_policy":resources.policy(),"actual_partial_migration_must_be_preserved":true,"automatic_cleanup":false}),
        }
    }
}

impl RoventNativeResourceScopeV2 {
    pub fn prepare(
        lock: Arc<RoventInputLockV2>,
        delegated: File,
        delegated_path: &Path,
        leaf_name: &str,
        options: RoventNativeResourceOptions,
    ) -> Result<Self, Box<RoventNativeResourceFailureV2>> {
        let plan = (|| -> Result<_, ArenaError> {
            lock.validate()?;
            resource_policy(
                &lock.manifest.engine.pc.device,
                &lock.manifest.resources,
                options,
            )
        })();
        let policy = match plan {
            Ok(policy) => policy,
            Err(error) => {
                return Err(Box::new(RoventNativeResourceFailureV2::BeforePreparation {
                    error,
                    lock,
                    delegated,
                }));
            }
        };
        let resources =
            match RoventTrainingResources::prepare(delegated, delegated_path, leaf_name, policy) {
                Ok(resources) => Arc::new(resources),
                Err(failure) => {
                    return Err(Box::new(RoventNativeResourceFailureV2::Preparation {
                        failure,
                        lock,
                    }));
                }
            };
        let entry = match resources.enter_current_native_process() {
            Ok(entry) => entry,
            Err(error) => {
                return Err(Box::new(RoventNativeResourceFailureV2::Entry {
                    error,
                    lock,
                    resources,
                }));
            }
        };
        Ok(Self {
            lock,
            resources,
            entry,
        })
    }

    pub fn matches_input_lock(&self, lock: &Arc<RoventInputLockV2>) -> bool {
        Arc::ptr_eq(&self.lock, lock)
    }

    pub fn observe(&self) -> RoventNativeResourceSnapshot {
        observe_native(
            &self.resources,
            &self.entry,
            &self.lock.canonical_sha256,
            &self.lock.manifest.resources,
        )
    }
}

/// All actual descriptors survive partial preparation/entry failures. There
/// is no rollback, automatic migration, process kill or cgroup removal on Drop.
pub enum RoventNativeResourceFailure {
    BeforePreparation {
        error: ArenaError,
        lock: Arc<RoventInputLock>,
        delegated: File,
    },
    Preparation {
        failure: Box<RoventTrainingResourceFailure>,
        lock: Arc<RoventInputLock>,
    },
    Entry {
        error: ArenaError,
        lock: Arc<RoventInputLock>,
        resources: Arc<RoventTrainingResources>,
    },
}
impl std::fmt::Debug for RoventNativeResourceFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BeforePreparation { error, .. } => {
                f.debug_tuple("BeforePreparation").field(error).finish()
            }
            Self::Preparation { failure, .. } => {
                f.debug_tuple("Preparation").field(failure).finish()
            }
            Self::Entry { error, .. } => f.debug_tuple("Entry").field(error).finish(),
        }
    }
}
impl std::fmt::Display for RoventNativeResourceFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BeforePreparation { error, .. } | Self::Entry { error, .. } => error.fmt(f),
            Self::Preparation { failure, .. } => failure.error.fmt(f),
        }
    }
}
impl std::error::Error for RoventNativeResourceFailure {}
impl RoventNativeResourceFailure {
    pub fn observation(&self) -> Value {
        match self {
            Self::BeforePreparation { error, lock, .. } => {
                json!({"schema":"rz-rovent-native-resource-failure/1",
                "stage":"before_preparation","input_lock_sha256":lock.canonical_sha256,"error":error.to_string(),"delegated_descriptor_retained":true,"automatic_cleanup":false})
            }
            Self::Preparation { failure, lock } => {
                json!({"schema":"rz-rovent-native-resource-failure/1",
                "stage":"preparation","input_lock_sha256":lock.canonical_sha256,"resource_failure":failure.observation(),"automatic_cleanup":false})
            }
            Self::Entry {
                error,
                lock,
                resources,
            } => json!({"schema":"rz-rovent-native-resource-failure/1",
                "stage":"self_entry_or_readback","input_lock_sha256":lock.canonical_sha256,
                "error":error.to_string(),"resource_policy":resources.policy(),"actual_partial_migration_must_be_preserved":true,"automatic_cleanup":false}),
        }
    }
}

impl RoventNativeResourceScope {
    pub fn prepare(
        lock: Arc<RoventInputLock>,
        delegated: File,
        delegated_path: &Path,
        leaf_name: &str,
        options: RoventNativeResourceOptions,
    ) -> Result<Self, Box<RoventNativeResourceFailure>> {
        let policy = match policy(&lock, options) {
            Ok(policy) => policy,
            Err(error) => {
                return Err(Box::new(RoventNativeResourceFailure::BeforePreparation {
                    error,
                    lock,
                    delegated,
                }));
            }
        };
        let resources =
            match RoventTrainingResources::prepare(delegated, delegated_path, leaf_name, policy) {
                Ok(resources) => Arc::new(resources),
                Err(failure) => {
                    return Err(Box::new(RoventNativeResourceFailure::Preparation {
                        failure,
                        lock,
                    }));
                }
            };
        let entry = match resources.enter_current_native_process() {
            Ok(entry) => entry,
            Err(error) => {
                return Err(Box::new(RoventNativeResourceFailure::Entry {
                    error,
                    lock,
                    resources,
                }));
            }
        };
        Ok(Self {
            lock,
            resources,
            entry,
        })
    }

    pub fn input_lock(&self) -> &Arc<RoventInputLock> {
        &self.lock
    }

    /// No JSON observation can replace the source. The upper runner prepares
    /// this scope in the pre-model hook with the launch's exact Arc owner.
    pub fn matches_input_lock(&self, lock: &Arc<RoventInputLock>) -> bool {
        Arc::ptr_eq(&self.lock, lock)
    }

    /// Failed/racing probes remain explicit Unknown in the returned snapshot;
    /// the actual FD/process owner is always retained for subsequent cleanup.
    pub fn observe(&self) -> RoventNativeResourceSnapshot {
        observe_native(
            &self.resources,
            &self.entry,
            &self.lock.canonical_sha256,
            &self.lock.manifest.resources,
        )
    }
}

fn observe_native(
    resources: &RoventTrainingResources,
    entry: &RoventNativeProcessEntry,
    lock_sha256: &str,
    declared: &RoventResourcePlan,
) -> RoventNativeResourceSnapshot {
    let tree = resources.observe_native_process_tree(entry);
    let total = resources.observe_total_parent_scope();
    let startup = resources.native_startup_json();
    // A failed probe does not establish that controls were unapplied.
    // A leaf file alone also cannot establish application to this tree.
    let applied = (startup.is_ok() && tree.is_ok()).then_some(true);
    let threads = tree.as_ref().ok().map(|tree| tree.kernel_threads);
    let limits = tree.as_ref().ok().map(|tree| {
        (
            tree.readback.memory_high_bytes,
            tree.readback.memory_max_bytes,
        )
    });
    let inventory_complete = tree.is_ok();
    let raw = json!({
        "schema":ROVENT_NATIVE_RESOURCE_EVIDENCE,
        "input_lock_sha256":lock_sha256,
        "declared_resource_plan":declared,
        "actual_leaf_policy":resources.policy(),
        "leaf_readback":startup.as_ref().ok(),"leaf_readback_error":startup.err().map(|error|error.to_string()),
        "native_tree":tree.as_ref().ok(),"native_tree_error":tree.err().map(|error|error.to_string()),
        "held_common_ancestor":total.as_ref().ok(),"ancestor_probe_error":total.err().map(|error|error.to_string()),
        "full_quota_ancestry":"unknown","physical_topology":"unknown","actual_ort_thread_role_inventory":"unknown",
        "final_host_peak":"unknown","vram_peak":"unknown","pre_entry_memory_charge_migration":"unknown",
        "scope":"actual_configured_leaf_and_bounded_live_tree_only_not_controlled_comparison_or_final_peak",
        "automatic_cleanup":false
    });
    RoventNativeResourceSnapshot {
        raw,
        applied,
        threads,
        limits,
        inventory_complete,
    }
}

fn policy(
    lock: &RoventInputLock,
    options: RoventNativeResourceOptions,
) -> Result<RoventTrainingResourcePolicy, ArenaError> {
    lock.validate()?;
    resource_policy(
        &lock.manifest.engine.pc.device,
        &lock.manifest.resources,
        options,
    )
}

fn resource_policy(
    declared_device: &RoventDevice,
    plan: &RoventResourcePlan,
    options: RoventNativeResourceOptions,
) -> Result<RoventTrainingResourcePolicy, ArenaError> {
    let device = match declared_device {
        RoventDevice::Cpu => "cpu".into(),
        RoventDevice::Cuda { ordinal, .. } => format!("cuda:{ordinal}"),
    };
    if options.device != device || !options.affinity.windows(2).all(|pair| pair[0] < pair[1]) {
        return Err(invalid(
            "explicit native device/sorted affinity differs from the original plan",
        ));
    }
    let policy = RoventTrainingResourcePolicy {
        cpu_quota_millicpus: plan.cpu_quota_millicpus,
        memory_high_bytes: plan.memory_high_bytes,
        memory_max_bytes: plan.memory_max_bytes,
        pids_max: options.pids_max,
        affinity: options.affinity,
        device,
    };
    policy.validate()?;
    Ok(policy)
}

/// Actual readback projection, deliberately neither Deserialize nor a native
/// authority. It keeps live peak values in raw evidence without relabelling
/// them as the complete finished-run peak expected by the strict receipt.
pub struct RoventNativeResourceSnapshot {
    raw: Value,
    applied: Option<bool>,
    threads: Option<u64>,
    limits: Option<(u64, u64)>,
    inventory_complete: bool,
}
impl RoventNativeResourceSnapshot {
    pub fn evidence(&self) -> &Value {
        &self.raw
    }

    /// Start only from actual current kernel controls/tree/affinity. This gate
    /// does not require post-execution peaks or mistake a declaration for the
    /// current OS readback. Deserialized JSON cannot create this snapshot.
    pub fn require_admission(
        &self,
        _profile: RoventNativeResourceProfile,
    ) -> Result<(), ArenaError> {
        if self.applied != Some(true)
            || !self.inventory_complete
            || self.threads.is_none()
            || self.limits.is_none()
        {
            return Err(invalid(
                "actual native leaf/tree readback is unavailable; original custody retained",
            ));
        }
        Ok(())
    }

    /// This separate post-execution decision never gates initial measurement.
    /// Keep the actual execution evidence and owner when strict acceptance is
    /// held: the current same-process snapshot provides partial evidence only.
    pub fn require_finished_profile(
        &self,
        profile: RoventNativeResourceProfile,
    ) -> Result<(), ArenaError> {
        self.require_admission(profile)?;
        match profile {
            RoventNativeResourceProfile::FunctionalValidation => Ok(()),
            RoventNativeResourceProfile::StrictComparison => Err(invalid(
                "strict resource acceptance held after execution: complete quota ancestry, topology, ORT thread-role inventory, pre-entry memory charges and finished-run peaks are unknown",
            )),
        }
    }

    /// Original output-directory custody is supplied by the upper runner. The
    /// one new file is create-new/no-follow, synced and pinned to exactly these
    /// readback bytes. A failed write is preserved; no cleanup delete follows.
    pub fn write_evidence(
        &self,
        output_root: &Dir,
        relative_path: &Path,
    ) -> Result<RoventResourceObservation, ArenaError> {
        let bytes = serde_json::to_vec(&self.raw).map_err(|error| invalid(error.to_string()))?;
        if bytes.is_empty() || bytes.len() > MAX_EVIDENCE_BYTES {
            return Err(invalid("bounded native resource evidence capacity"));
        }
        let mut components = relative_path.components().peekable();
        let mut directory = output_root
            .try_clone()
            .map_err(|error| invalid(error.to_string()))?;
        let mut basename = None;
        while let Some(component) = components.next() {
            let Component::Normal(name) = component else {
                return Err(invalid("relative native resource evidence path required"));
            };
            if components.peek().is_none() {
                basename = Some(name);
            } else {
                directory = directory
                    .open_dir_nofollow(name)
                    .map_err(|error| invalid(error.to_string()))?;
            }
        }
        let basename =
            basename.ok_or_else(|| invalid("native resource evidence basename missing"))?;
        let path = relative_path
            .to_str()
            .ok_or_else(|| invalid("native resource evidence path UTF-8"))?
            .to_string();
        let artifact = ArtifactRef {
            path,
            sha256: format!("{:x}", Sha256::digest(&bytes)),
            bytes: bytes.len() as u64,
            source: ROVENT_NATIVE_RESOURCE_SOURCE.into(),
            license: "MIT".into(),
        };
        artifact.validate()?;
        let mut options = cap_std::fs::OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No);
        let mut file = directory
            .open_with(basename, &options)
            .map_err(|error| invalid(error.to_string()))?;
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|error| invalid(error.to_string()))?;
        directory
            .open(".")
            .and_then(|file| file.sync_all())
            .map_err(|error| invalid(error.to_string()))?;
        Ok(self.project_observation(artifact))
    }

    fn project_observation(&self, artifact: ArtifactRef) -> RoventResourceObservation {
        RoventResourceObservation {
            limits_applied: self.applied.map_or(RoventObserved::Unknown, |value| observed(value, "original_held_fresh_leaf_limit_readback")),
            // The held common ancestor is not complete quota ancestry. Its
            // declared number or hash must never stand in for that readback.
            process_tree_quota_millicpus: RoventObserved::Unknown,
            memory_high_bytes: self.limits.map_or(RoventObserved::Unknown, |(value,_)| observed(value,"configured_fresh_leaf_memory.high_readback_not_ancestor_or_pre_entry_charge_proof")),
            memory_max_bytes: self.limits.map_or(RoventObserved::Unknown, |(_,value)| observed(value,"configured_fresh_leaf_memory.max_readback_not_ancestor_or_pre_entry_charge_proof")),
            inventory_complete: if self.inventory_complete { observed(true,"stable_bounded_original_descendant_and_thread_snapshot_not_lifetime_inventory") } else { RoventObserved::Unknown },
            kernel_threads: self.threads.map_or(RoventObserved::Unknown, |count| observed(count,"actual_bounded_proc_thread_inventory")),
            verified_plan_readback_sha256: RoventObserved::Unknown,
            host_peak_bytes: RoventObserved::Unknown,
            vram_peak_bytes: RoventObserved::Unknown,
            evidence: artifact,
        }
    }
}

fn observed<T>(value: T, method: &str) -> RoventObserved<T> {
    RoventObserved::Observed {
        value,
        method: method.into(),
    }
}
fn invalid(message: impl std::fmt::Display) -> ArenaError {
    ArenaError::Invalid(format!("rovent_native_resources: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A projection fixture tests only Unknown handling, not kernel entry or
    /// any source/native authority. No public success constructor is added.
    fn fixture_snapshot() -> RoventNativeResourceSnapshot {
        RoventNativeResourceSnapshot {
            raw: json!({"schema":ROVENT_NATIVE_RESOURCE_EVIDENCE,"fixture":"projection_only_no_kernel_or_native_authority"}),
            applied: None,
            threads: None,
            limits: None,
            inventory_complete: false,
        }
    }
    fn fixture_artifact() -> ArtifactRef {
        let artifact = ArtifactRef {
            path: "fixture/resource-readback.json".into(),
            sha256: "a".repeat(64),
            bytes: 1,
            source: ROVENT_NATIVE_RESOURCE_SOURCE.into(),
            license: "MIT".into(),
        };
        artifact.validate().unwrap();
        artifact
    }

    #[test]
    fn written_evidence_validates_public_source_without_changing_readback_schema() {
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "rovezero-native-resource-evidence-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        let directory = Dir::open_ambient_dir(&root, cap_std::ambient_authority()).unwrap();
        // Only the exact create-new test file and its own empty directory may
        // be removed, including when an assertion unwinds. No recursive cleanup.
        struct Output {
            root: std::path::PathBuf,
            directory: Dir,
        }
        impl Drop for Output {
            fn drop(&mut self) {
                let _ = self.directory.remove_file("resource-readback.json");
                let _ = std::fs::remove_dir(&self.root);
            }
        }
        let output = Output { root, directory };
        let snapshot = fixture_snapshot();
        let observed = snapshot
            .write_evidence(&output.directory, Path::new("resource-readback.json"))
            .unwrap();
        observed.evidence.validate().unwrap();
        assert_eq!(observed.evidence.source, ROVENT_NATIVE_RESOURCE_SOURCE);
        let bytes = output.directory.read("resource-readback.json").unwrap();
        assert_eq!(observed.evidence.bytes, bytes.len() as u64);
        assert_eq!(
            observed.evidence.sha256,
            format!("{:x}", Sha256::digest(&bytes))
        );
        let raw: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(raw, snapshot.raw);
        assert_eq!(raw["schema"], ROVENT_NATIVE_RESOURCE_EVIDENCE);
        // Reproduce the original source/domain mixup through the real metadata
        // validator; accepting a local schema as a public source stays forbidden.
        let mut domain_as_source = observed.evidence.clone();
        domain_as_source.source = ROVENT_NATIVE_RESOURCE_EVIDENCE.into();
        assert!(domain_as_source.validate().is_err());
        assert_eq!(observed.limits_applied, RoventObserved::Unknown);
        assert_eq!(observed.inventory_complete, RoventObserved::Unknown);
        assert!(
            snapshot
                .require_admission(RoventNativeResourceProfile::FunctionalValidation)
                .is_err()
        );
        // Successful writing neither grants resource admission nor overwrites
        // existing evidence, even when the new bytes would be identical.
        assert!(
            snapshot
                .write_evidence(&output.directory, Path::new("resource-readback.json"))
                .is_err()
        );
    }

    #[test]
    fn failed_kernel_probes_remain_unknown_not_unapplied_or_empty() {
        let snapshot = fixture_snapshot();
        assert!(
            snapshot
                .require_admission(RoventNativeResourceProfile::FunctionalValidation)
                .is_err()
        );
        let observed = snapshot.project_observation(fixture_artifact());
        assert_eq!(observed.limits_applied, RoventObserved::Unknown);
        assert_eq!(observed.memory_high_bytes, RoventObserved::Unknown);
        assert_eq!(observed.memory_max_bytes, RoventObserved::Unknown);
        assert_eq!(observed.inventory_complete, RoventObserved::Unknown);
        assert_eq!(observed.kernel_threads, RoventObserved::Unknown);
    }

    #[test]
    fn partial_leaf_and_tree_observation_cannot_claim_full_resource_acceptance() {
        let mut partial = fixture_snapshot();
        partial.applied = Some(true);
        partial.threads = Some(3);
        partial.limits = Some((6 << 30, 12 << 30));
        partial.inventory_complete = true;
        assert!(
            partial
                .require_admission(RoventNativeResourceProfile::FunctionalValidation)
                .is_ok()
        );
        assert!(
            partial
                .require_admission(RoventNativeResourceProfile::StrictComparison)
                .is_ok()
        );
        assert!(
            partial
                .require_finished_profile(RoventNativeResourceProfile::StrictComparison)
                .is_err()
        );
        let observed = partial.project_observation(fixture_artifact());
        assert!(matches!(
            observed.limits_applied,
            RoventObserved::Observed { value: true, .. }
        ));
        assert!(
            matches!(observed.memory_high_bytes, RoventObserved::Observed { value, .. } if value == 6 << 30)
        );
        assert!(
            matches!(observed.memory_max_bytes, RoventObserved::Observed { value, .. } if value == 12 << 30)
        );
        assert!(matches!(
            observed.inventory_complete,
            RoventObserved::Observed { value: true, .. }
        ));
        assert!(matches!(
            observed.kernel_threads,
            RoventObserved::Observed { value: 3, .. }
        ));
        assert_eq!(
            observed.process_tree_quota_millicpus,
            RoventObserved::Unknown
        );
        assert_eq!(
            observed.verified_plan_readback_sha256,
            RoventObserved::Unknown
        );
        assert_eq!(observed.host_peak_bytes, RoventObserved::Unknown);
        assert_eq!(observed.vram_peak_bytes, RoventObserved::Unknown);
    }
}
