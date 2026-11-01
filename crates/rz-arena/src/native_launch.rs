//! Exclusive, source-verified inputs for the bounded CPU NN integration pair.
//!
//! Copies have new inodes and closed writers before publication. Read-only Unix
//! permissions are an ownership convention, not an immutable OS seal or a
//! same-UID sandbox. The parent exclusively controls the output tree and trusts
//! the pinned native programs not to chmod, replace or modify these inputs.

use crate::ArenaError;
#[cfg(target_os = "linux")]
use crate::FastchessInvocation;
use rz_experiments::{ArtifactRef, LockedIntegrationPairSpecV1};
#[cfg(target_os = "linux")]
use rz_experiments::{NativeArtifactRole, NativeEngineRole};
use serde::Serialize;
use std::{
    ffi::OsString,
    fmt,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

pub const NATIVE_PAIR_METADATA_CAP: u64 = 64 * 1024;
pub(crate) fn pair_stream_cap(output_bytes: u64, metadata_cap: u64) -> Result<u64, ArenaError> {
    output_bytes
        .checked_sub(metadata_cap)
        .and_then(|n| n.checked_div(2))
        .filter(|n| *n > 0 && metadata_cap > 0)
        .ok_or_else(|| {
            ArenaError::Budget(
                "native output budget needs receipt, PGN/config and stream reservations".into(),
            )
        })
}
const MAX_INNER_ARG_BYTES: usize = 16 * 1024;
static NATIVE_ACTIVE: AtomicBool = AtomicBool::new(false);
static NATIVE_ADMISSION_CLOSED: AtomicBool = AtomicBool::new(false);

/// Sealed executor views share file/process ownership, never provider profiles.
/// The CPU and CUDA declarations retain distinct public types and lock domains.
#[doc(hidden)]
pub trait NativeLaunchDeclaration: sealed::Sealed + Clone + fmt::Debug + Send + 'static {
    fn input_sha256(&self) -> &str;
    /// Optional policy provenance for preparation failures. Legacy formats
    /// omit this field entirely, retaining their original serialized bytes.
    fn preparation_policy_envelope(&self) -> Option<NativePreparationPolicyEnvelope> {
        None
    }
    fn view(&self) -> NativePairView<'_>;
    fn declared_inputs(&self) -> Vec<&ArtifactRef>;
    fn unique_bytes(&self) -> Result<u64, rz_experiments::ManifestError>;
    fn engine_view(
        &self,
        role: rz_experiments::NativeEngineRole,
    ) -> Result<NativeEngineView<'_>, rz_experiments::ManifestError>;
    fn provider_name(&self) -> &'static str;
    /// V6 declarations explicitly select the host runtime for each endpoint.
    /// Historical declarations return `None` and retain their original
    /// CPU/bundle semantics.  This declaration is not evidence of provider
    /// readiness, child mappings, or physical GPU completion.
    fn runtime_source_v6(
        &self,
        _role: rz_experiments::NativeEngineRole,
    ) -> Option<&rz_experiments::NativeRuntimeSourceV6> {
        None
    }
    /// A system CUDA endpoint may be admitted only when its launcher consumes
    /// the explicit V6 configuration and verifies the child-owned receipt.
    /// Merely supplying a runtime source does not enable this capability.
    fn verifies_system_runtime_receipt_v6(&self, _role: rz_experiments::NativeEngineRole) -> bool {
        false
    }
    /// V4 has independently typed model/resolver/recheck/stack selections.
    /// Historical declarations retain their original 32-token codec limit.
    fn argument_limit(&self) -> usize {
        32
    }
    /// Fixed receipt reservation selected before input copying or child spawn.
    /// Legacy declarations keep their original 64 KiB receipt and stream split.
    fn pair_metadata_cap(&self) -> u64 {
        NATIVE_PAIR_METADATA_CAP
    }
    /// Arguments needed only by the two game sessions, after option preflight.
    fn runtime_arguments(
        &self,
        _role: rz_experiments::NativeEngineRole,
        _runtime_root: &Path,
    ) -> Result<Vec<OsString>, ArenaError> {
        Ok(Vec::new())
    }
    /// Probe-only arguments must not create the two game sessions' evidence.
    fn preflight_arguments(
        &self,
        _role: rz_experiments::NativeEngineRole,
        _runtime_root: &Path,
    ) -> Result<Vec<OsString>, ArenaError> {
        Ok(Vec::new())
    }
    /// Default probes retain the historical role runtime path. A selected
    /// separate root is created and pinned by the Native owner, with a fixed
    /// role-specific sibling name; declarations cannot supply arbitrary paths.
    /// Its entries consume the existing runtime aggregate, never a new quota.
    fn uses_separate_preflight_runtime_root(
        &self,
        _role: rz_experiments::NativeEngineRole,
    ) -> Result<bool, ArenaError> {
        Ok(false)
    }
    fn amend_execution_limitations(&self, _limitations: &mut Vec<String>) {}
    /// A bounded path below `inputs`; native declarations keep their old names.
    fn snapshot_relative_path(&self, _artifact: &ArtifactRef) -> Option<String> {
        None
    }
    fn seed(&self) -> u64 {
        1
    }
    fn rove_tree_max_edges(&self) -> Option<u32> {
        None
    }
    fn match_execution(&self) -> Option<&rz_experiments::MatchExecutionV1> {
        None
    }
    fn external_options(
        &self,
        role: rz_experiments::NativeEngineRole,
    ) -> Result<std::collections::BTreeMap<String, String>, ArenaError> {
        let view = self.engine_view(role)?;
        let mut options = view
            .external
            .map(|e| e.requested_options.clone())
            .unwrap_or_default();
        if let Some(execution) = self.match_execution() {
            let plan = execution.plan()?;
            let allocation = plan
                .engines
                .iter()
                .find(|a| a.role == role)
                .expect("validated role");
            // These values are locked by the resource policy and checked against advertisement.
            options.insert("Ponder".into(), execution.ponder.to_string());
            options.insert("Threads".into(), allocation.threads.to_string());
        }
        Ok(options)
    }
    fn validate_execution(&self) -> Result<(), ArenaError> {
        Ok(())
    }
    fn advise_drop_input_cache(&self) -> bool {
        false
    }
    fn additional_manifest(&self) -> Option<&ArtifactRef> {
        None
    }
    fn validate_additional_manifest(&self, _bytes: &[u8]) -> Result<(), ArenaError> {
        Ok(())
    }
    fn pilot_cohort(&self) -> Option<&ArtifactRef> {
        None
    }
    fn validate_pilot_cohort(&self, _bytes: &[u8]) -> Result<(), ArenaError> {
        Ok(())
    }
}
pub(crate) mod sealed {
    pub trait Sealed {}
}
#[doc(hidden)]
pub struct NativeEngineView<'a> {
    pub engine_id: &'a str,
    pub artifacts: &'a [rz_experiments::NativeArtifactBinding],
    pub cuda_bundle: Option<&'a rz_experiments::CudaBundleBindingV1>,
    pub search: Option<rz_experiments::NativeCudaSearchV2>,
    pub batch_experiment: Option<usize>,
    pub external: Option<&'a rz_experiments::ExternalUciEndpointV2>,
    pub environment: Option<&'a rz_experiments::EngineEnvironmentV2>,
}
impl<'a> NativeEngineView<'a> {
    #[cfg(target_os = "linux")]
    fn executable_input(&self, artifact: &ArtifactRef) -> bool {
        self.artifact(NativeArtifactRole::Binary)
            .is_ok_and(|binary| binary == artifact)
            || self
                .environment
                .is_some_and(|environment| &environment.launcher == artifact)
    }

    pub fn artifact(
        &self,
        role: rz_experiments::NativeArtifactRole,
    ) -> Result<&'a ArtifactRef, ArenaError> {
        if let Some(external) = self.external {
            return if role == rz_experiments::NativeArtifactRole::Binary {
                Ok(&external.binary)
            } else {
                Err(ArenaError::Invalid(
                    "external UCI endpoint has no native NN role".into(),
                ))
            };
        }
        let mut matching = self.artifacts.iter().filter(|b| b.role == role);
        let first = matching
            .next()
            .ok_or_else(|| ArenaError::Integrity("native role lacks input".into()))?;
        if matching.next().is_some() {
            return Err(ArenaError::Integrity(
                "native input role is duplicated".into(),
            ));
        }
        Ok(&first.artifact)
    }
}

/// Check the explicit V6 selection before creating any private snapshot or
/// child.  A caller may not turn a system-runtime registration into another
/// copy of the registered system libraries.
#[cfg(target_os = "linux")]
fn validate_v6_runtime_snapshot_boundary(
    runtime: &rz_experiments::NativeRuntimeSourceV6,
    has_bundle: bool,
    has_external_launcher: bool,
    verifies_child_receipt: bool,
    inputs: &[&ArtifactRef],
) -> Result<(), ArenaError> {
    runtime.validate()?;
    if has_bundle {
        return Err(ArenaError::Invalid(
            "V6 runtime selection cannot inherit a CUDA bundle".into(),
        ));
    }
    if let rz_experiments::NativeRuntimeSourceV6::SystemCuda(system) = runtime {
        if system.platform != rz_experiments::SystemPlatformV6::Linux {
            return Err(ArenaError::Invalid(
                "V6 system CUDA platform differs from the Linux launcher".into(),
            ));
        }
        if !has_external_launcher || !verifies_child_receipt {
            return Err(ArenaError::Invalid(
                "V6 system CUDA requires an explicit child receipt consumer".into(),
            ));
        }
        if inputs.iter().any(|artifact| {
            system
                .libraries
                .iter()
                .any(|library| artifact.bytes == library.bytes && artifact.sha256 == library.sha256)
        }) {
            return Err(ArenaError::Invalid(
                "V6 system runtime libraries cannot be snapshotted as native inputs".into(),
            ));
        }
    }
    Ok(())
}
#[doc(hidden)]
pub struct NativePairView<'a> {
    pub pair_id: &'a str,
    pub white_order: [rz_experiments::NativeEngineRole; 2],
    pub opening: &'a rz_experiments::OpeningSpec,
    pub opening_artifact: &'a ArtifactRef,
    pub runner: &'a rz_experiments::ToolIdentity,
    pub clock: rz_experiments::NativePairClock,
    pub max_plies: u32,
    pub timeouts: rz_experiments::NativeTimeoutsV1,
    // Shared observation bounds are an executor view, not a CPU spec conversion.
    pub budget: rz_experiments::NativeResourceBudgetV1,
}
impl sealed::Sealed for LockedIntegrationPairSpecV1 {}
impl NativeLaunchDeclaration for LockedIntegrationPairSpecV1 {
    fn input_sha256(&self) -> &str {
        self.sha256()
    }
    fn view(&self) -> NativePairView<'_> {
        let p = self.input();
        NativePairView {
            pair_id: &p.pair_id,
            white_order: p.white_order,
            opening: &p.opening,
            opening_artifact: &p.opening_artifact,
            runner: &p.runner,
            clock: rz_experiments::NativePairClock::Movetime(p.clock),
            max_plies: p.max_plies,
            timeouts: p.timeouts,
            budget: p.budget,
        }
    }
    fn declared_inputs(&self) -> Vec<&ArtifactRef> {
        self.declared_artifacts()
    }
    fn unique_bytes(&self) -> Result<u64, rz_experiments::ManifestError> {
        self.unique_input_bytes()
    }
    fn engine_view(
        &self,
        role: rz_experiments::NativeEngineRole,
    ) -> Result<NativeEngineView<'_>, rz_experiments::ManifestError> {
        let engine = self.input().engine(role)?;
        Ok(NativeEngineView {
            engine_id: &engine.engine_id,
            artifacts: &engine.artifacts,
            cuda_bundle: None,
            search: None,
            batch_experiment: None,
            external: None,
            environment: None,
        })
    }
    fn provider_name(&self) -> &'static str {
        "CPU"
    }
}

/// One active lease bounds unresolved retention to one whole launch bundle.
pub(crate) struct NativeAdmissionLease {
    _private: (),
}
impl Drop for NativeAdmissionLease {
    fn drop(&mut self) {
        NATIVE_ACTIVE.store(false, Ordering::Release);
    }
}
impl NativeAdmissionLease {
    #[cfg(target_os = "linux")]
    fn acquire() -> Result<Self, ArenaError> {
        if NATIVE_ADMISSION_CLOSED.load(Ordering::Acquire) {
            return Err(ArenaError::Invalid(
                "native integration admission is closed".into(),
            ));
        }
        NATIVE_ACTIVE
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| {
                ArenaError::Budget("one native integration lease is already active".into())
            })?;
        let lease = Self { _private: () };
        if NATIVE_ADMISSION_CLOSED.load(Ordering::Acquire) {
            return Err(ArenaError::Invalid(
                "native integration admission closed during admission".into(),
            ));
        }
        Ok(lease)
    }
}
pub(crate) fn close_native_admission() {
    NATIVE_ADMISSION_CLOSED.store(true, Ordering::Release);
}
pub fn native_admission_closed() -> bool {
    NATIVE_ADMISSION_CLOSED.load(Ordering::Acquire)
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeSnapshotReceipt {
    pub artifact: ArtifactRef,
    pub snapshot_relative_path: String,
    pub bytes: u64,
    pub sha256: String,
    pub distinct_source_inode: bool,
    pub closed_writer_read_only: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativePreparationPolicyEnvelope {
    pub domain: String,
    pub policy: String,
    pub base_launch_sha256: String,
    pub execution_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativePreparationReceipt {
    pub receipt_version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub policy_envelope: Option<NativePreparationPolicyEnvelope>,
    pub execution_ready: bool,
    pub input_sha256: String,
    pub output_directory: String,
    pub attempt_created: bool,
    pub attempted_snapshot_relative_paths: Vec<String>,
    pub completed_snapshots: Vec<NativeSnapshotReceipt>,
    pub child_spawned: bool,
    pub writers_closed: bool,
    pub input_pins_required: bool,
    pub original_error: String,
    pub subsequent_file_owner: String,
    pub automatic_retry: bool,
}
pub struct NativePreparationFailure {
    pub cause: ArenaError,
    pub receipt: NativePreparationReceipt,
    pub receipt_artifact: Option<ArtifactRef>,
    pub persistence_error: Option<ArenaError>,
}
impl fmt::Debug for NativePreparationFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativePreparationFailure")
            .field("cause", &self.cause)
            .field("receipt", &self.receipt)
            .field("persistence_error", &self.persistence_error)
            .finish()
    }
}
impl fmt::Display for NativePreparationFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "native preparation failed before spawn: {}; attempt_created={}; evidence_save_failed={}",
            self.cause,
            self.receipt.attempt_created,
            self.persistence_error.is_some()
        )
    }
}
impl std::error::Error for NativePreparationFailure {}

/// Owns every input pin, the exact cwd and both runtime output roots. There is
/// deliberately no automatic file deletion. A pending run must retain this
/// owner, even after a terminal leader if its process group remains unverified.
pub struct NativeLaunchOwner<S: NativeLaunchDeclaration = LockedIntegrationPairSpecV1> {
    pub(crate) spec: S,
    pub(crate) _lease: NativeAdmissionLease,
    #[cfg(target_os = "linux")]
    pub(crate) snapshot: linux::Snapshot,
}
impl<S: NativeLaunchDeclaration> fmt::Debug for NativeLaunchOwner<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeLaunchOwner")
            .field("input_sha256", &self.spec.input_sha256())
            .finish_non_exhaustive()
    }
}
impl<S: NativeLaunchDeclaration> NativeLaunchOwner<S> {
    pub fn spec(&self) -> &S {
        &self.spec
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

/// Encode only the bounded subset supported by the pinned Fastchess
/// `argv_split` parser. This is a single `args=` outer argv element, not shell
/// quoting. Quotes and backslashes are rejected rather than ambiguously escaped.
pub fn encode_fastchess_native_args(tokens: &[OsString]) -> Result<OsString, ArenaError> {
    encode_fastchess_native_args_with_limit(tokens, 32)
}
pub fn encode_fastchess_native_args_with_limit(
    tokens: &[OsString],
    limit: usize,
) -> Result<OsString, ArenaError> {
    if !matches!(limit, 32 | 48) || tokens.is_empty() || tokens.len() > limit {
        return Err(ArenaError::Invalid(format!(
            "native engine arguments require 1..{limit} tokens and a registered 32/48-token limit"
        )));
    }
    let mut encoded = String::from("args=");
    for (index, token) in tokens.iter().enumerate() {
        let value = token
            .to_str()
            .ok_or_else(|| ArenaError::Invalid("native argument is not UTF-8".into()))?;
        if value.is_empty()
            || value.len() > 4096
            || value
                .chars()
                .any(|c| c.is_control() || matches!(c, '\"' | '\'' | '\\'))
        {
            return Err(ArenaError::Invalid(
                "native argument contains unsupported delimiters or length".into(),
            ));
        }
        if index != 0 {
            encoded.push(' ');
        }
        encoded.push('\"');
        encoded.push_str(value);
        encoded.push('\"');
        if encoded.len() > MAX_INNER_ARG_BYTES {
            return Err(ArenaError::Budget(
                "native inner argument byte limit exceeded".into(),
            ));
        }
    }
    Ok(encoded.into())
}

/// Verify declaration and source bytes, then create an exclusive outside-Git
/// attempt. `output_directory` is one safe basename; existing attempts are never
/// reused. No engine or runner is started by this function.
pub fn prepare_native_launch(
    spec: &LockedIntegrationPairSpecV1,
    source_root: &Path,
    output_root: &Path,
    output_directory: &str,
) -> Result<NativeLaunchOwner, Box<NativePreparationFailure>> {
    prepare_native_launch_for(spec, source_root, output_root, output_directory)
}
pub(crate) fn prepare_native_launch_for<S: NativeLaunchDeclaration>(
    spec: &S,
    source_root: &Path,
    output_root: &Path,
    output_directory: &str,
) -> Result<NativeLaunchOwner<S>, Box<NativePreparationFailure>> {
    let mut receipt=NativePreparationReceipt{receipt_version:1,policy_envelope:spec.preparation_policy_envelope(),execution_ready:false,input_sha256:spec.input_sha256().into(),
        output_directory:if safe_output_basename(output_directory){output_directory.into()}else{"rejected-basename".into()},
        attempt_created:false,attempted_snapshot_relative_paths:Vec::new(),completed_snapshots:Vec::new(),child_spawned:false,writers_closed:true,input_pins_required:false,original_error:String::new(),
        subsequent_file_owner:"caller retains exclusively owned outside-Git attempt; all writers are closed and no child was spawned, so input handles may be released; retained historical files are not a process-lifetime aggregate quota".into(),automatic_retry:false};
    #[cfg(target_os = "linux")]
    {
        let mut directory = None;
        let result: Result<NativeLaunchOwner<S>, ArenaError> = (|| {
            let lease = NativeAdmissionLease::acquire()?;
            let snapshot = linux::prepare(
                spec,
                source_root,
                output_root,
                output_directory,
                &mut receipt,
                &mut directory,
            )?;
            Ok(NativeLaunchOwner {
                spec: spec.clone(),
                _lease: lease,
                snapshot,
            })
        })();
        match result {
            Ok(owner) => Ok(owner),
            Err(cause) => {
                receipt.original_error = cause.to_string().chars().take(4096).collect();
                let persisted = if receipt.attempt_created {
                    linux::save_preparation_failure(
                        directory.as_ref(),
                        &receipt,
                        spec.provider_name(),
                    )
                } else {
                    Ok(None)
                };
                let (receipt_artifact, persistence_error) = match persisted {
                    Ok(artifact) => (artifact, None),
                    Err(error) => (None, Some(error)),
                };
                Err(Box::new(NativePreparationFailure {
                    cause,
                    receipt,
                    receipt_artifact,
                    persistence_error,
                }))
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (spec, source_root, output_root, output_directory);
        let cause = ArenaError::Invalid(format!(
            "{} NN integration runner currently requires Linux",
            spec.provider_name()
        ));
        receipt.original_error = cause.to_string();
        Err(Box::new(NativePreparationFailure {
            cause,
            receipt,
            receipt_artifact: None,
            persistence_error: None,
        }))
    }
}
fn safe_output_basename(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
}

#[cfg(target_os = "linux")]
pub(crate) mod linux {
    use super::*;
    use crate::{OwnedArtifactTreeWatch, ProcessLimits};
    use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::{Dir, OpenOptions, OpenOptionsExt};
    use sha2::{Digest, Sha256};
    use std::{
        collections::BTreeMap,
        fs::{File, Permissions},
        io::{Read, Seek, SeekFrom, Write},
        os::unix::fs::{MetadataExt, PermissionsExt},
        path::PathBuf,
    };

    // Only the syscall chunk changes. Source/copy hashes, full pin rechecks,
    // byte budgets, closed writers and inode/path ownership stay unchanged.
    // Keep the original 16KiB path as default until fixed-work E acceptance.
    const SNAPSHOT_IO_BYTES: usize = if cfg!(feature = "experimental-snapshot-io") {
        64 * 1024
    } else {
        16 * 1024
    };

    pub(crate) struct InputPin {
        pub artifact: ArtifactRef,
        pub path: PathBuf,
        pub file: File,
    }
    pub(crate) struct PreflightRuntimeRoot {
        pub name: &'static str,
        pub path: PathBuf,
        pub directory: Dir,
        pub file: File,
    }
    fn preflight_runtime_name(role: NativeEngineRole) -> &'static str {
        match role {
            NativeEngineRole::Baseline => "external-baseline-preflight-runtime",
            NativeEngineRole::Candidate => "external-candidate-preflight-runtime",
        }
    }
    fn create_preflight_runtime_root(
        parent: &Dir,
        parent_path: &Path,
        role: NativeEngineRole,
    ) -> Result<PreflightRuntimeRoot, ArenaError> {
        let name = preflight_runtime_name(role);
        parent
            .create_dir(name)
            .map_err(|_| io("cannot create exclusive native preflight runtime root"))?;
        let directory = parent
            .open_dir_nofollow(name)
            .map_err(|_| io("cannot pin native preflight runtime root"))?;
        let file = readable_directory_pin(&directory)?;
        file.set_permissions(Permissions::from_mode(0o700))
            .map_err(|_| io("cannot make native preflight runtime root private"))?;
        Ok(PreflightRuntimeRoot {
            name,
            path: parent_path.join(name),
            directory,
            file,
        })
    }
    pub(crate) struct Snapshot {
        pub directory: Dir,
        pub input_directory: Dir,
        pub cwd: File,
        pub path: PathBuf,
        pub output_directory: String,
        pub pins: Vec<InputPin>,
        pub receipts: Vec<NativeSnapshotReceipt>,
        pub runner_index: usize,
        pub invocation: FastchessInvocation,
        pub watch: OwnedArtifactTreeWatch,
        pub limits: ProcessLimits,
        pub _runtime_root_pins: [File; 2],
        pub preflight_runtime_roots: [Option<PreflightRuntimeRoot>; 2],
        pub input_subdirectories: BTreeMap<PathBuf, Dir>,
        pub preparation_cache_hint: Option<crate::SnapshotCacheHint>,
    }
    impl Snapshot {
        pub(crate) fn preflight_runtime_path(&self, role: NativeEngineRole) -> PathBuf {
            let index = match role {
                NativeEngineRole::Baseline => 0,
                NativeEngineRole::Candidate => 1,
            };
            self.preflight_runtime_roots[index].as_ref().map_or_else(
                || {
                    self.path.join(match role {
                        NativeEngineRole::Baseline => "baseline-runtime",
                        NativeEngineRole::Candidate => "candidate-runtime",
                    })
                },
                |root| root.path.clone(),
            )
        }
    }

    /// The optional roots retain both the directory capability and a readable
    /// permissions/inode pin. Reopening the direct name without following links
    /// binds the argv path to that retained inode before and after every probe.
    pub(crate) fn verify_preflight_runtime_roots(snapshot: &Snapshot) -> Result<(), ArenaError> {
        verify_preflight_runtime_root_pins(
            &snapshot.directory,
            &snapshot.path,
            &snapshot.preflight_runtime_roots,
        )
    }
    fn verify_preflight_runtime_root_pins(
        parent: &Dir,
        parent_path: &Path,
        roots: &[Option<PreflightRuntimeRoot>; 2],
    ) -> Result<(), ArenaError> {
        if roots.iter().all(Option::is_none) {
            return Ok(());
        }
        let mut inodes = std::collections::BTreeSet::new();
        // The crate forbids unsafe code. Read the effective UID from the same
        // trusted procfs self observation used for Linux arena admission.
        let mut status = Vec::new();
        File::open("/proc/self/status")
            .map_err(|_| io("cannot observe native preflight owner UID"))?
            .take(16 * 1024 + 1)
            .read_to_end(&mut status)
            .map_err(|_| io("cannot read native preflight owner UID"))?;
        if status.len() > 16 * 1024 {
            return Err(ArenaError::Budget(
                "native preflight owner observation exceeds 16KiB".into(),
            ));
        }
        let status = std::str::from_utf8(&status)
            .map_err(|_| io("native preflight owner UID observation is not UTF-8"))?;
        let mut uid_lines = status.lines().filter_map(|line| line.strip_prefix("Uid:"));
        let values = uid_lines
            .next()
            .ok_or_else(|| io("native preflight effective UID is unavailable"))?
            .split_whitespace()
            .map(str::parse::<u32>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| io("native preflight effective UID is invalid"))?;
        if values.len() != 4 || uid_lines.next().is_some() {
            return Err(ArenaError::Integrity(
                "native preflight UID observation is not one complete identity".into(),
            ));
        }
        let owner_uid = values[1];
        for (role, root) in [NativeEngineRole::Baseline, NativeEngineRole::Candidate]
            .into_iter()
            .zip(roots)
        {
            let Some(root) = root else { continue };
            let name = preflight_runtime_name(role);
            if root.name != name || root.path != parent_path.join(name) {
                return Err(ArenaError::Integrity(
                    "native preflight root role/name/path differs from the closed sibling plan"
                        .into(),
                ));
            }
            let named = parent.open_dir_nofollow(name).map_err(|_| {
                ArenaError::Integrity("native preflight root pathname changed".into())
            })?;
            let named_file = readable_directory_pin(&named)?;
            let capability_file = readable_directory_pin(&root.directory)?;
            let held = root
                .file
                .metadata()
                .map_err(|_| io("native preflight held root metadata unavailable"))?;
            let named = named_file
                .metadata()
                .map_err(|_| io("native preflight named root metadata unavailable"))?;
            let capability = capability_file
                .metadata()
                .map_err(|_| io("native preflight capability metadata unavailable"))?;
            for metadata in [&held, &named, &capability] {
                if !metadata.is_dir()
                    || metadata.uid() != owner_uid
                    || metadata.mode() & 0o7777 != 0o700
                {
                    return Err(ArenaError::Integrity(
                        "native preflight root is not a private owner directory".into(),
                    ));
                }
            }
            if (held.dev(), held.ino()) != (named.dev(), named.ino())
                || (held.dev(), held.ino()) != (capability.dev(), capability.ino())
                || !inodes.insert((held.dev(), held.ino()))
            {
                return Err(ArenaError::Integrity(
                    "native preflight root no longer names its distinct owned inode".into(),
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn pin<'a>(
        pins: &'a [InputPin],
        artifact: &ArtifactRef,
    ) -> Result<&'a InputPin, ArenaError> {
        pins.iter()
            .find(|pin| &pin.artifact == artifact)
            .ok_or_else(|| {
                ArenaError::Integrity("declared native input lacks its owned snapshot".into())
            })
    }
    fn io(detail: &'static str) -> ArenaError {
        ArenaError::Io(detail.into())
    }
    // cap_std's directory capability may use O_PATH. That descriptor can pin
    // cwd/path authority, but Linux fchmod rejects it with EBADF. Open a real
    // readable directory descriptor relative to the same capability instead.
    fn readable_directory_pin(directory: &Dir) -> Result<File, ArenaError> {
        let mut options = OpenOptions::new();
        options
            .read(true)
            .follow(FollowSymlinks::No)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_NONBLOCK);
        let file = directory
            .open_with(".", &options)
            .map_err(|_| io("cannot open readable native directory permissions pin"))?
            .into_std();
        if !file
            .metadata()
            .map_err(|_| io("cannot inspect readable native directory permissions pin"))?
            .is_dir()
        {
            return Err(ArenaError::Integrity(
                "native directory permissions pin is not a directory".into(),
            ));
        }
        Ok(file)
    }
    fn snapshot_path(relative: &str, max_depth: usize) -> Result<PathBuf, ArenaError> {
        if relative.is_empty()
            || relative.len() > 512
            || relative.split('/').count() > max_depth
            || relative.split('/').any(|part| {
                part.is_empty()
                    || part.len() > 128
                    || part.starts_with('.')
                    || !part.bytes().all(|byte| {
                        byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.')
                    })
            })
        {
            return Err(ArenaError::Invalid(
                "native snapshot path must have bounded safe relative components".into(),
            ));
        }
        Ok(PathBuf::from(relative))
    }
    fn snapshot_parents(
        names: &[String],
        max_depth: usize,
    ) -> Result<std::collections::BTreeSet<PathBuf>, ArenaError> {
        let mut destinations = std::collections::BTreeSet::new();
        let mut parents = std::collections::BTreeSet::new();
        for name in names {
            let relative = snapshot_path(name, max_depth)?;
            if !destinations.insert(relative.clone()) {
                return Err(ArenaError::Integrity(
                    "distinct native inputs share a snapshot destination".into(),
                ));
            }
            for parent in relative.ancestors().skip(1) {
                if !parent.as_os_str().is_empty() {
                    parents.insert(parent.to_path_buf());
                }
            }
        }
        if parents.iter().any(|parent| destinations.contains(parent)) {
            return Err(ArenaError::Integrity(
                "native snapshot file conflicts with an input directory".into(),
            ));
        }
        Ok(parents)
    }

    /// Recheck every named directory against its original no-follow capability.
    /// Checking ancestors first binds nested names to the same closed topology.
    pub(crate) fn verify_input_directories(
        output: &Dir,
        inputs: &Dir,
        subdirectories: &BTreeMap<PathBuf, Dir>,
    ) -> Result<(), ArenaError> {
        let compare = |parent: &Dir, name: &std::ffi::OsStr, held: &Dir| {
            let named = parent.open_dir_nofollow(name).map_err(|_| {
                ArenaError::Integrity("native input directory pathname changed".into())
            })?;
            let held_meta = readable_directory_pin(held)?
                .metadata()
                .map_err(|_| io("native input directory metadata unavailable"))?;
            let named_meta = readable_directory_pin(&named)?
                .metadata()
                .map_err(|_| io("native named directory metadata unavailable"))?;
            if (held_meta.dev(), held_meta.ino()) != (named_meta.dev(), named_meta.ino()) {
                return Err(ArenaError::Integrity(
                    "native input directory no longer names its owned inode".into(),
                ));
            }
            Ok(())
        };
        compare(output, std::ffi::OsStr::new("inputs"), inputs)?;
        for (relative, held) in subdirectories {
            let parent = relative.parent().ok_or_else(|| {
                ArenaError::Integrity("native input directory has no parent".into())
            })?;
            let parent = if parent.as_os_str().is_empty() {
                inputs
            } else {
                subdirectories.get(parent).ok_or_else(|| {
                    ArenaError::Integrity("native input directory parent is not retained".into())
                })?
            };
            compare(
                parent,
                relative.file_name().ok_or_else(|| {
                    ArenaError::Integrity("native input directory has no name".into())
                })?,
                held,
            )?;
        }
        Ok(())
    }

    pub(crate) fn input_parent<'a>(
        inputs: &'a Dir,
        subdirectories: &'a BTreeMap<PathBuf, Dir>,
        snapshot_relative: &'a Path,
    ) -> Result<(&'a Dir, &'a std::ffi::OsStr), ArenaError> {
        let relative = snapshot_relative.strip_prefix("inputs").map_err(|_| {
            ArenaError::Integrity("native input escaped closed snapshot topology".into())
        })?;
        let parent = relative
            .parent()
            .ok_or_else(|| ArenaError::Integrity("native input has no parent".into()))?;
        let directory = if parent.as_os_str().is_empty() {
            inputs
        } else {
            subdirectories.get(parent).ok_or_else(|| {
                ArenaError::Integrity("native input parent is outside closed topology".into())
            })?
        };
        let name = relative
            .file_name()
            .ok_or_else(|| ArenaError::Integrity("native input has no direct copy name".into()))?;
        Ok((directory, name))
    }
    fn outside_git(path: &Path) -> Result<PathBuf, ArenaError> {
        let metadata =
            std::fs::symlink_metadata(path).map_err(|_| io("native output root unavailable"))?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Err(ArenaError::Invalid(
                "native output root must be a real owned directory".into(),
            ));
        }
        let path = path
            .canonicalize()
            .map_err(|_| io("cannot resolve native output root"))?;
        for ancestor in path.ancestors() {
            match std::fs::symlink_metadata(ancestor.join(".git")) {
                Ok(_) => {
                    return Err(ArenaError::Invalid(
                        "native outputs must be outside Git checkouts".into(),
                    ));
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(_) => return Err(io("cannot check native output ownership boundary")),
            }
        }
        Ok(path)
    }
    pub(super) fn save_preparation_failure(
        directory: Option<&Dir>,
        receipt: &NativePreparationReceipt,
        provider: &str,
    ) -> Result<Option<ArtifactRef>, ArenaError> {
        let directory = directory.ok_or_else(|| {
            io("failed attempt could not retain a directory capability for evidence saving")
        })?;
        let bytes = serde_json::to_vec_pretty(receipt)
            .map_err(|_| io("cannot encode preparation failure receipt"))?;
        if bytes.len() as u64 > NATIVE_PAIR_METADATA_CAP {
            return Err(ArenaError::Budget(
                "preparation failure receipt exceeds 64KiB".into(),
            ));
        }
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .mode(0o600);
        let mut file = directory
            .open_with("native-preparation-failure.v1.json", &options)
            .map_err(|_| {
                io("cannot reserve preparation failure receipt; partial attempt preserved")
            })?;
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|_| {
                io("cannot finish preparation failure receipt; partial bytes preserved")
            })?;
        Ok(Some(ArtifactRef {
            path: format!(
                "{}/native-preparation-failure.v1.json",
                receipt.output_directory
            ),
            sha256: format!("{:x}", Sha256::digest(&bytes)),
            bytes: bytes.len() as u64,
            source: format!("RoveZero {provider} NN integration preparation evidence"),
            license: "MIT execution evidence; external input rights remain separate".into(),
        }))
    }
    pub(crate) fn readonly_copy(
        artifact: &ArtifactRef,
        mut source: File,
        directory: &Dir,
        relative: &str,
        absolute: PathBuf,
        executable: bool,
        max_bytes: u64,
    ) -> Result<(InputPin, NativeSnapshotReceipt), ArenaError> {
        // Every ancestor is already pinned by preparation. This operation may
        // create only one basename, never traverse a caller-supplied directory.
        snapshot_path(relative, 1)?;
        let source_metadata = source
            .metadata()
            .map_err(|_| io("cannot inspect verified source inode"))?;
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .mode(0o600);
        let mut writer = directory
            .open_with(relative, &options)
            .map_err(|_| io("cannot create exclusive native input copy"))?
            .into_std();
        let mut count = 0u64;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; SNAPSHOT_IO_BYTES];
        loop {
            let read = source
                .read(&mut buffer)
                .map_err(|_| io("cannot read verified native input"))?;
            if read == 0 {
                break;
            }
            count = count
                .checked_add(read as u64)
                .ok_or_else(|| ArenaError::Budget("native copy length overflow".into()))?;
            if count > artifact.bytes || count > max_bytes {
                return Err(ArenaError::Integrity(
                    "native input changed or exceeded copy budget".into(),
                ));
            }
            hasher.update(&buffer[..read]);
            writer
                .write_all(&buffer[..read])
                .map_err(|_| io("cannot write native input copy"))?;
        }
        if count != artifact.bytes || format!("{:x}", hasher.finalize()) != artifact.sha256 {
            return Err(ArenaError::Integrity(
                "native input changed while copying".into(),
            ));
        }
        writer
            .sync_all()
            .map_err(|_| io("cannot sync native input copy"))?;
        let copied_metadata = writer
            .metadata()
            .map_err(|_| io("cannot inspect native copy inode"))?;
        let distinct = (source_metadata.dev(), source_metadata.ino())
            != (copied_metadata.dev(), copied_metadata.ino());
        if !distinct {
            return Err(ArenaError::Integrity(
                "native copy shares source inode".into(),
            ));
        }
        drop(writer);
        let mut options = OpenOptions::new();
        options
            .read(true)
            .follow(FollowSymlinks::No)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        let mut file = directory
            .open_with(relative, &options)
            .map_err(|_| io("cannot pin native copy"))?
            .into_std();
        file.set_permissions(Permissions::from_mode(if executable {
            0o500
        } else {
            0o400
        }))
        .map_err(|_| io("cannot make native input copy read-only"))?;
        verify_copy(&mut file, artifact)?;
        Ok((
            InputPin {
                artifact: artifact.clone(),
                path: absolute,
                file,
            },
            NativeSnapshotReceipt {
                artifact: artifact.clone(),
                snapshot_relative_path: format!("inputs/{relative}"),
                bytes: artifact.bytes,
                sha256: artifact.sha256.clone(),
                distinct_source_inode: distinct,
                closed_writer_read_only: true,
            },
        ))
    }
    pub(crate) fn verify_copy(file: &mut File, artifact: &ArtifactRef) -> Result<(), ArenaError> {
        file.seek(SeekFrom::Start(0))
            .map_err(|_| io("cannot rewind native copy"))?;
        let mut hasher = Sha256::new();
        let mut count = 0u64;
        let mut buffer = [0u8; SNAPSHOT_IO_BYTES];
        loop {
            let read = file
                .read(&mut buffer)
                .map_err(|_| io("cannot recheck native copy"))?;
            if read == 0 {
                break;
            }
            count = count
                .checked_add(read as u64)
                .ok_or_else(|| ArenaError::Budget("native pin length overflow".into()))?;
            if count > artifact.bytes {
                return Err(ArenaError::Integrity("native copy grew".into()));
            }
            hasher.update(&buffer[..read]);
        }
        if count != artifact.bytes || format!("{:x}", hasher.finalize()) != artifact.sha256 {
            return Err(ArenaError::Integrity(
                "native copy digest or length differs".into(),
            ));
        }
        file.seek(SeekFrom::Start(0))
            .map_err(|_| io("cannot rewind checked native copy"))?;
        Ok(())
    }

    pub(crate) fn prepare<S: NativeLaunchDeclaration>(
        spec: &S,
        source_root: &Path,
        output_root: &Path,
        label: &str,
        trace: &mut NativePreparationReceipt,
        owned_directory: &mut Option<Dir>,
    ) -> Result<Snapshot, ArenaError> {
        if !safe_output_basename(label) {
            return Err(ArenaError::Invalid(
                "native attempt requires a safe unique basename".into(),
            ));
        }
        spec.validate_execution()?;
        let input = spec.view();
        let metadata_cap = spec.pair_metadata_cap();
        let stream_cap = pair_stream_cap(input.budget.max_output_bytes, metadata_cap)?;
        crate::validate_opening_artifact_for_spec(
            input.opening,
            input.max_plies,
            input.opening_artifact,
        )?;
        let total = spec.unique_bytes()?;
        if total > input.budget.max_input_bytes {
            return Err(ArenaError::Budget("native input aggregate exceeded".into()));
        }
        let separate_preflight = [
            spec.uses_separate_preflight_runtime_root(NativeEngineRole::Baseline)?,
            spec.uses_separate_preflight_runtime_root(NativeEngineRole::Candidate)?,
        ];
        let preflight_root_count = separate_preflight
            .iter()
            .filter(|selected| **selected)
            .count() as u32;
        if preflight_root_count > input.budget.max_runtime_files {
            return Err(ArenaError::Budget(
                "native preflight root reservation exceeds existing runtime entry budget".into(),
            ));
        }
        for (selected, role) in separate_preflight
            .into_iter()
            .zip([NativeEngineRole::Baseline, NativeEngineRole::Candidate])
        {
            if selected && spec.engine_view(role)?.external.is_none() {
                return Err(ArenaError::Integrity(
                    "a separate native preflight root requires an external UCI endpoint".into(),
                ));
            }
        }
        let declared_inputs = spec.declared_inputs();
        for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
            if let Some(runtime) = spec.runtime_source_v6(role) {
                let engine = spec.engine_view(role)?;
                validate_v6_runtime_snapshot_boundary(
                    runtime,
                    engine.cuda_bundle.is_some(),
                    engine.external.is_some(),
                    spec.verifies_system_runtime_receipt_v6(role),
                    &declared_inputs,
                )?;
            }
        }
        let mut unique = BTreeMap::new();
        for artifact in declared_inputs {
            if let Some(prior) = unique.insert(artifact.path.clone(), artifact.clone())
                && prior != *artifact
            {
                return Err(ArenaError::Integrity(
                    "same native source path has conflicting metadata".into(),
                ));
            }
        }
        let cuda_bundle = spec
            .engine_view(NativeEngineRole::Baseline)?
            .cuda_bundle
            .or(spec.engine_view(NativeEngineRole::Candidate)?.cuda_bundle);
        let mut names = Vec::new();
        for (index, artifact) in unique.values().enumerate() {
            let bundle_name = cuda_bundle.and_then(|bundle| {
                if &bundle.manifest == artifact {
                    Some("bundle.v1.json")
                } else {
                    bundle
                        .files
                        .iter()
                        .find(|entry| &entry.artifact == artifact)
                        .map(|entry| entry.filename.as_str())
                }
            });
            let legacy_name = bundle_name.map_or_else(
                || format!("pin-{index:02}"),
                |name| format!("cuda-bundle/{name}"),
            );
            let name = spec.snapshot_relative_path(artifact).unwrap_or(legacy_name);
            if bundle_name.is_some() && name != format!("cuda-bundle/{}", bundle_name.unwrap()) {
                return Err(ArenaError::Invalid(
                    "native snapshot override changes the declared CUDA bundle topology".into(),
                ));
            }
            if bundle_name.is_none() && name.starts_with("cuda-bundle/") {
                return Err(ArenaError::Invalid(
                    "native snapshot override uses the reserved CUDA bundle namespace".into(),
                ));
            }
            names.push(name);
        }
        let parents = snapshot_parents(&names, usize::from(input.budget.max_runtime_depth))?;
        // Verify every source before creating the attempt and keep those exact,
        // rewound handles for copying instead of reopening and hashing again.
        // The copy stream is still hashed, so same-inode writes fail closed.
        crate::emit_native_phase("source_verification_started");
        let sources = unique
            .values()
            .map(|artifact| artifact.open_verified(source_root, input.budget.max_input_bytes))
            .collect::<Result<Vec<_>, _>>()?;
        crate::emit_native_phase("source_verification_complete");
        let root_path = outside_git(output_root)?;
        let root = Dir::open_ambient_dir(&root_path, cap_std::ambient_authority())
            .map_err(|_| io("cannot pin native output root"))?;
        // Keep this attempt's explicit input/output bounds and overflow check.
        // When configured, the separately supervised advisory observer reports
        // total retained storage within its scope; absent, failed or partial
        // observations are not a claim about this root. Unrelated existing
        // outputs must not block this launch or force another recursive scan on
        // its critical path. Legacy profiles with their own registered strict
        // budget check it separately.
        let _attempt_reservation = total
            .checked_add(input.budget.max_output_bytes)
            .and_then(|n| n.checked_add(metadata_cap))
            .ok_or_else(|| io("native retention reservation overflow"))?;
        root.create_dir(label).map_err(|_| {
            io("cannot create exclusive native attempt; existing outputs preserved")
        })?;
        trace.attempt_created = true;
        let directory = root
            .open_dir_nofollow(label)
            .map_err(|_| io("cannot pin native attempt"))?;
        *owned_directory = Some(
            directory
                .try_clone()
                .map_err(|_| io("cannot retain preparation evidence directory"))?,
        );
        readable_directory_pin(&directory)?
            .set_permissions(Permissions::from_mode(0o700))
            .map_err(|_| io("cannot make native attempt private"))?;
        directory
            .create_dir("inputs")
            .map_err(|_| io("cannot create native input directory"))?;
        let inputs = directory
            .open_dir_nofollow("inputs")
            .map_err(|_| io("cannot pin native inputs directory"))?;
        readable_directory_pin(&inputs)?
            .set_permissions(Permissions::from_mode(0o700))
            .map_err(|_| io("cannot make native input directory private"))?;
        let path = root_path.join(label);
        let mut input_subdirectories: BTreeMap<PathBuf, Dir> = BTreeMap::new();
        for relative in parents {
            let parent = relative.parent().expect("planned relative parent");
            let parent = if parent.as_os_str().is_empty() {
                &inputs
            } else {
                input_subdirectories.get(parent).ok_or_else(|| {
                    ArenaError::Integrity("planned snapshot parent is absent".into())
                })?
            };
            let name = relative.file_name().expect("planned relative name");
            parent
                .create_dir(name)
                .map_err(|_| io("cannot create exclusive native input subdirectory"))?;
            let child = parent
                .open_dir_nofollow(name)
                .map_err(|_| io("cannot pin native input subdirectory"))?;
            readable_directory_pin(&child)?
                .set_permissions(Permissions::from_mode(0o700))
                .map_err(|_| io("cannot make native input subdirectory private"))?;
            input_subdirectories.insert(relative, child);
        }
        let mut pins = Vec::new();
        let mut receipts = Vec::new();
        let mut preparation_cache_hint = crate::SnapshotCacheHint::rolling(
            spec.advise_drop_input_cache(),
            "after_copy_each_verified",
        );
        crate::emit_native_phase("snapshot_copy_started");
        for ((artifact, source), name) in unique.values().zip(sources).zip(names) {
            trace
                .attempted_snapshot_relative_paths
                .push(format!("inputs/{name}"));
            let executable = artifact == &input.runner.binary
                || spec
                    .match_execution()
                    .is_some_and(|e| &e.executor == artifact)
                || [NativeEngineRole::Baseline, NativeEngineRole::Candidate]
                    .into_iter()
                    .any(|role| {
                        spec.engine_view(role)
                            .is_ok_and(|e| e.executable_input(artifact))
                    });
            let snapshot_relative = PathBuf::from("inputs").join(&name);
            let destination = input_parent(&inputs, &input_subdirectories, &snapshot_relative)?;
            let copied = readonly_copy(
                artifact,
                source,
                destination.0,
                destination
                    .1
                    .to_str()
                    .expect("validated UTF-8 snapshot name"),
                path.join("inputs").join(&name),
                executable,
                input.budget.max_input_bytes,
            );
            if copied.is_err()
                && let Some(status) = &preparation_cache_hint
            {
                status.emit();
            }
            let (pin, receipt) = copied?;
            // readonly_copy has synced and closed its writer, established a
            // distinct inode and fully reverified the read-only private pin.
            // Never advise the source handle, runtime cache or a GPU buffer.
            if let Some(status) = &mut preparation_cache_hint {
                status.advise(&pin.file, artifact.bytes);
            }
            let receipt = NativeSnapshotReceipt {
                snapshot_relative_path: format!("inputs/{name}"),
                ..receipt
            };
            trace.completed_snapshots.push(receipt.clone());
            pins.push(pin);
            receipts.push(receipt);
        }
        crate::emit_native_phase("snapshot_copy_complete");
        if let Some(status) = &preparation_cache_hint {
            status.emit();
        }
        // Additional metadata is read only after all complete private copies
        // exist. A malformed manifest cannot produce a native executable owner.
        for (is_cohort, manifest) in [
            (false, spec.additional_manifest()),
            (true, spec.pilot_cohort()),
        ] {
            let Some(manifest) = manifest else { continue };
            let manifest = pin(&pins, manifest)?;
            let mut file = manifest
                .file
                .try_clone()
                .map_err(|_| io("cannot clone CUDA bundle manifest pin"))?;
            file.seek(SeekFrom::Start(0))
                .map_err(|_| io("cannot rewind CUDA bundle manifest pin"))?;
            let mut bytes = Vec::new();
            file.take(64 * 1024 + 1)
                .read_to_end(&mut bytes)
                .map_err(|_| io("cannot read CUDA bundle manifest pin"))?;
            if bytes.len() > 64 * 1024 {
                return Err(ArenaError::Budget(
                    "CUDA bundle manifest exceeds 64KiB".into(),
                ));
            }
            if is_cohort {
                spec.validate_pilot_cohort(&bytes)?;
            } else {
                spec.validate_additional_manifest(&bytes)?;
            }
        }
        for child in input_subdirectories.values() {
            readable_directory_pin(child)?
                .set_permissions(Permissions::from_mode(0o500))
                .map_err(|_| io("cannot close native input subdirectory to writes"))?;
        }
        readable_directory_pin(&inputs)?
            .set_permissions(Permissions::from_mode(0o500))
            .map_err(|_| io("cannot close native inputs directory to writes"))?;
        let runner_index = pins
            .iter()
            .position(|p| p.artifact == input.runner.binary)
            .ok_or_else(|| ArenaError::Integrity("runner snapshot absent".into()))?;
        let runtime_roots = [
            path.join("baseline-runtime"),
            path.join("candidate-runtime"),
        ];
        let mut runtime_root_pins = Vec::new();
        for name in ["baseline-runtime", "candidate-runtime"] {
            directory
                .create_dir(name)
                .map_err(|_| io("cannot create private role runtime root"))?;
            let role_directory = directory
                .open_dir_nofollow(name)
                .map_err(|_| io("cannot pin role runtime root"))?;
            let pin = readable_directory_pin(&role_directory)?;
            pin.set_permissions(Permissions::from_mode(0o700))
                .map_err(|_| io("cannot make role runtime root private"))?;
            runtime_root_pins.push(pin);
        }
        let mut preflight_runtime_roots: [Option<PreflightRuntimeRoot>; 2] = [None, None];
        for (index, role) in [NativeEngineRole::Baseline, NativeEngineRole::Candidate]
            .into_iter()
            .enumerate()
        {
            if !separate_preflight[index] {
                continue;
            }
            preflight_runtime_roots[index] =
                Some(create_preflight_runtime_root(&directory, &path, role)?);
        }
        verify_preflight_runtime_root_pins(&directory, &path, &preflight_runtime_roots)?;
        let cwd = directory
            .try_clone()
            .map_err(|_| io("cannot clone native cwd"))?
            .into_std_file();
        let invocation = build_invocation(spec, &pins, &path, &runtime_roots)?;
        let watch_bytes = total
            .checked_add(input.budget.max_runtime_bytes)
            .and_then(|n| n.checked_add(input.budget.max_output_bytes))
            .ok_or_else(|| ArenaError::Budget("native watched byte overflow".into()))?;
        if watch_bytes > input.budget.max_artifact_bytes {
            return Err(ArenaError::Budget(
                "native artifact budget needs input/runtime/output reservations".into(),
            ));
        }
        let max_files = usize::try_from(input.budget.max_runtime_files)
            .map_err(|_| ArenaError::Budget("native file count cannot fit host".into()))?
            .checked_add(pins.len())
            .and_then(|n| n.checked_add(input_subdirectories.len()))
            .and_then(|n| n.checked_add(8))
            .ok_or_else(|| ArenaError::Budget("native file count overflow".into()))?;
        Ok(Snapshot {
            directory,
            input_directory: inputs,
            cwd,
            path,
            output_directory: label.into(),
            pins,
            receipts,
            runner_index,
            invocation,
            watch: OwnedArtifactTreeWatch {
                max_total_bytes: watch_bytes,
                max_file_bytes: input
                    .budget
                    .max_input_bytes
                    .max(input.budget.max_runtime_bytes)
                    .max(input.budget.max_output_bytes),
                max_files,
                max_depth: usize::from(input.budget.max_runtime_depth)
                    .checked_add(1)
                    .ok_or_else(|| ArenaError::Budget("native tree depth overflow".into()))?,
            },
            limits: ProcessLimits {
                wall_ms: input.timeouts.runtime_ms,
                shutdown_grace_ms: input.timeouts.shutdown_ms,
                max_output_bytes: stream_cap,
                max_child_processes: input.budget.max_child_processes,
            },
            _runtime_root_pins: runtime_root_pins.try_into().map_err(|_| {
                ArenaError::Integrity("native runtime root pin count differs".into())
            })?,
            preflight_runtime_roots,
            input_subdirectories,
            preparation_cache_hint,
        })
    }

    fn key_path(key: &str, path: &Path) -> Result<OsString, ArenaError> {
        let value = path
            .to_str()
            .ok_or_else(|| ArenaError::Invalid("native path is not UTF-8".into()))?;
        if !path.is_absolute() || value.len() > 4096 || value.chars().any(char::is_control) {
            return Err(ArenaError::Invalid(
                "native path must be a bounded absolute path".into(),
            ));
        }
        Ok(format!("{key}{value}").into())
    }
    pub(crate) enum ExternalArgumentMode<'a> {
        Preflight(&'a Path),
        Runtime(&'a Path),
    }
    pub(crate) fn external_arguments<S: NativeLaunchDeclaration>(
        spec: &S,
        role: NativeEngineRole,
        external: &rz_experiments::ExternalUciEndpointV2,
        pins: &[InputPin],
        mode: ExternalArgumentMode<'_>,
    ) -> Result<Vec<OsString>, ArenaError> {
        let mut tokens = external
            .arguments
            .iter()
            .map(|arg| {
                crate::external_uci::resolve_asset_tokens(arg, external, pins).map(OsString::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        tokens.extend(match mode {
            ExternalArgumentMode::Preflight(root) => spec.preflight_arguments(role, root)?,
            ExternalArgumentMode::Runtime(root) => spec.runtime_arguments(role, root)?,
        });
        Ok(tokens)
    }
    fn validate_native_runner(
        runner: &rz_experiments::ToolIdentity,
        clock: rz_experiments::NativePairClock,
    ) -> Result<(), ArenaError> {
        let reap_patch = runner.dirty_patch.as_ref().is_some_and(|patch| {
            patch.sha256 == crate::pals_launch::PALS_CLOCK_REAP_PATCH_SHA256
                && patch.bytes == crate::pals_launch::PALS_CLOCK_REAP_PATCH_BYTES
        });
        // The audited clock+reap release uses a separately pinned short version.
        // Keep historical clean movetime and clock-only declarations unchanged;
        // both registered reap builds must match their version/hash/bytes together.
        let supported_version = if reap_patch {
            runner.dirty
                && matches!(clock, rz_experiments::NativePairClock::Game(_))
                && crate::pals_launch::registered_reap_runner(
                    &runner.version,
                    &runner.binary.sha256,
                    runner.binary.bytes,
                )
        } else {
            runner.version == crate::FASTCHESS_VERSION
        };
        if runner.source_url != crate::FASTCHESS_SOURCE_URL
            || runner.source_commit != crate::FASTCHESS_SOURCE_COMMIT
            || !supported_version
            || (matches!(clock, rz_experiments::NativePairClock::Movetime(_))
                && (runner.dirty || runner.dirty_patch.is_some()))
        {
            return Err(ArenaError::Invalid(
                "unsupported or dirty native integration Fastchess".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn build_invocation<S: NativeLaunchDeclaration>(
        spec: &S,
        pins: &[InputPin],
        cwd: &Path,
        roots: &[PathBuf; 2],
    ) -> Result<FastchessInvocation, ArenaError> {
        let input = spec.view();
        validate_native_runner(input.runner, input.clock)?;
        let prefix = u32::try_from(input.opening.moves.len())
            .map_err(|_| ArenaError::Budget("native opening ply overflow".into()))?;
        let remaining = input
            .max_plies
            .checked_sub(prefix)
            .filter(|n| *n > 0 && *n % 2 == 0)
            .ok_or_else(|| {
                ArenaError::Invalid(
                    "native pair requires positive even searched plies after opening".into(),
                )
            })?;
        let mut args = Vec::new();
        // Fastchess and its engine children inherit E's cleared environment.
        // Resolve C's owned cache in the parent and pass only its explicit root;
        // do not propagate HOME, loader variables or other ambient settings.
        let runtime_cache = rz_eval::runtime_pin::RuntimeCache::for_user().map_err(|error| {
            ArenaError::Io(format!("native runtime cache preparation:{:?}", error.kind))
        })?;
        for role in input.white_order {
            let engine = spec.engine_view(role)?;
            let binary = pin(pins, engine.artifact(NativeArtifactRole::Binary)?)?;
            if let Some(external) = engine.external {
                let root = &roots[match role {
                    NativeEngineRole::Baseline => 0,
                    NativeEngineRole::Candidate => 1,
                }];
                let tokens = external_arguments(
                    spec,
                    role,
                    external,
                    pins,
                    ExternalArgumentMode::Runtime(root),
                )?;
                let variables = external
                    .environment
                    .as_ref()
                    .map(|environment| {
                        environment
                            .variables
                            .iter()
                            .map(|(name, value)| {
                                crate::external_uci::resolve_asset_tokens(value, external, pins)
                                    .map(|value| (name.clone(), value))
                            })
                            .collect::<Result<std::collections::BTreeMap<_, _>, _>>()
                    })
                    .transpose()?;
                let launch = crate::engine_environment::prepare(
                    binary,
                    tokens,
                    engine.environment,
                    variables.as_ref(),
                    pins,
                )?;
                let launch = crate::engine_environment::place(
                    launch,
                    spec.match_execution(),
                    role,
                    engine.environment.is_some(),
                    pins,
                )?;
                args.extend([
                    OsString::from("-engine"),
                    key_path("cmd=", &launch.program.path)?,
                    key_path("dir=", cwd)?,
                    format!("name={}", engine.engine_id).into(),
                ]);
                if !launch.arguments.is_empty() {
                    args.push(encode_fastchess_native_args_with_limit(
                        &launch.arguments,
                        spec.argument_limit(),
                    )?);
                }
                for (name, value) in spec.external_options(role)? {
                    let value = crate::external_uci::resolve_asset_tokens(&value, external, pins)?;
                    args.push(format!("option.{name}={value}").into());
                }
                continue;
            }
            let root = &roots[match role {
                NativeEngineRole::Baseline => 0,
                NativeEngineRole::Candidate => 1,
            }];
            let mut tokens = vec![
                OsString::from(if engine.cuda_bundle.is_some() {
                    "--onnx-cuda"
                } else {
                    "--onnx-cpu"
                }),
                OsString::from(if engine.batch_experiment.is_some() {
                    "--batch-attestation"
                } else {
                    "--attestation"
                }),
            ];
            if let Some(width) = engine.batch_experiment {
                tokens.push(format!("--experimental-batch={width}").into());
            }
            for (flag, kind) in [
                ("--source-weights=", NativeArtifactRole::SourceWeights),
                ("--onnx-model=", NativeArtifactRole::Onnx),
                ("--export-manifest=", NativeArtifactRole::ExportManifest),
                ("--ort-library=", NativeArtifactRole::OrtLibrary),
            ] {
                tokens.push(key_path(flag, &pin(pins, engine.artifact(kind)?)?.path)?);
            }
            tokens.push(
                format!(
                    "--manifest-sha256={}",
                    engine.artifact(NativeArtifactRole::ExportManifest)?.sha256
                )
                .into(),
            );
            tokens.push(
                format!(
                    "--ort-sha256={}",
                    engine.artifact(NativeArtifactRole::OrtLibrary)?.sha256
                )
                .into(),
            );
            tokens.push(key_path("--output-root=", root)?);
            tokens.push(key_path("--runtime-cache-root=", runtime_cache.root())?);
            if let Some(bundle) = engine.cuda_bundle {
                tokens.push(key_path(
                    "--cuda-bundle=",
                    &pin(pins, &bundle.manifest)?.path,
                )?);
                tokens.push(format!("--cuda-bundle-sha256={}", bundle.manifest.sha256).into());
            }
            if let Some(search) = engine.search {
                tokens.push(format!("--search-simulations={}", search.simulations).into());
                tokens.push(format!("--final-selection={}", search.final_selection.cli()).into());
            }
            if let Some(max_edges) = spec.rove_tree_max_edges() {
                tokens.push(format!("--search-max-edges={max_edges}").into());
            }
            let launch =
                crate::engine_environment::prepare(binary, tokens, engine.environment, None, pins)?;
            let launch = crate::engine_environment::place(
                launch,
                spec.match_execution(),
                role,
                engine.environment.is_some(),
                pins,
            )?;
            args.extend([
                OsString::from("-engine"),
                key_path("cmd=", &launch.program.path)?,
                key_path("dir=", cwd)?,
                format!("name={}", engine.engine_id).into(),
                encode_fastchess_native_args_with_limit(&launch.arguments, spec.argument_limit())?,
            ]);
            if let Some(execution) = spec.match_execution() {
                args.push(format!("option.Ponder={}", execution.ponder).into());
            }
        }
        let opening = &pin(pins, input.opening_artifact)?.path;
        args.extend([
            "-each".into(),
            "proto=uci".into(),
            match input.clock {
                rz_experiments::NativePairClock::Movetime(c) => {
                    format!("st={}.{:03}", c.movetime_ms / 1000, c.movetime_ms % 1000)
                }
                rz_experiments::NativePairClock::Game(c) => format!(
                    "tc={}.{:03}+{}.{:03}",
                    c.base_ms / 1000,
                    c.base_ms % 1000,
                    c.increment_ms / 1000,
                    c.increment_ms % 1000
                ),
            }
            .into(),
            "restart=on".into(),
            "timemargin=0".into(),
            "-openings".into(),
            key_path("file=", opening)?,
            "format=pgn".into(),
            "order=sequential".into(),
            format!("plies={prefix}").into(),
            "start=1".into(),
            "-rounds".into(),
            "1".into(),
            "-games".into(),
            "2".into(),
            "-repeat".into(),
            "-concurrency".into(),
            "1".into(),
            "-srand".into(),
            spec.seed().to_string().into(),
            "-maxmoves".into(),
            (remaining / 2).to_string().into(),
            "-pgnout".into(),
            key_path("file=", &cwd.join("match.pgn"))?,
            "append=false".into(),
            "notation=uci".into(),
            "min=false".into(),
            "timeleft=true".into(),
            "latency=true".into(),
            "-autosaveinterval".into(),
            "0".into(),
            "-ratinginterval".into(),
            "0".into(),
            "-startup-ms".into(),
            input.timeouts.startup_ms.to_string().into(),
            "-ping-ms".into(),
            input.timeouts.handshake_ms.to_string().into(),
            "-ucinewgame-ms".into(),
            input.timeouts.drain_ms.to_string().into(),
            "-log".into(),
            "file=/proc/self/fd/1".into(),
            "append=true".into(),
            "level=trace".into(),
            "realtime=true".into(),
            "engine=true".into(),
        ]);
        // Fastchess strict stops on a time-loss WARN before persisting PGN.
        // A pilot retains both losses before E stops the next pair. Legacy
        // integration callers preserve their original strict argument order.
        if matches!(input.clock, rz_experiments::NativePairClock::Movetime(_)) {
            let index = args
                .iter()
                .position(|arg| arg == "-log")
                .expect("closed log argument");
            args.insert(index, "-strict".into());
        }
        if let Some(execution) = spec.match_execution() {
            let plan = execution.plan()?;
            let cpus = plan
                .runner_cpu_ids
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let mut placed = vec![
                "engine-exec".into(),
                cpus.into(),
                "-".into(),
                pin(pins, &input.runner.binary)?.path.as_os_str().to_owned(),
            ];
            placed.extend(args);
            args = placed;
        }
        let mut limitations=[
            "NN integration only; execution_ready=false; strength_eligible=false; same weights and search in both roles",
            "private input copies use separate inodes and closed writers; Unix readonly is an ownership convention, not a same-UID sandbox or immutable seal",
            "trusted pinned programs and parent exclusively own snapshots and runtime output ancestors throughout execution and pending retention",
            "Fastchess movetime and its internal 100ms read margin do not attest the shared whole-engine clock contract",
            "Fastchess automatic draw rules differ from explicit A claims; PGN audit is independent and maxmoves draws remain incomplete",
            "owned-tree/process snapshots are observed limits, not aggregate RAM or kernel disk quotas; escaped/transient children are outside the process-group guarantee",
            "per-process address-space declaration requires separately recorded inherited enforcement; this library does not install RAM/CPU affinity limits",
            "process cleanup alone is not physical NN drain; matched bounded B startup/final provider records are required separately",
            "runtime libraries use C's external immutable cache: at most 4 entries/8GiB per cache slot, separate from this attempt's watched artifact budget; cache reuse is not inference-cache provenance",
            "native engine binaries must accept the explicit runtime-cache-root option; historical binaries/launchers remain reproducible at their original source pins",
        ].map(String::from).to_vec();
        limitations[0] = format!(
            "{} NN integration only; execution_ready=false; strength_eligible=false; same weights and search in both roles",
            spec.provider_name()
        );
        if matches!(input.clock, rz_experiments::NativePairClock::Game(_)) {
            let baseline = spec.engine_view(NativeEngineRole::Baseline)?;
            let candidate = spec.engine_view(NativeEngineRole::Candidate)?;
            limitations[0] = if baseline.external.is_some() || candidate.external.is_some() {
                "V2 external UCI paired pilot only; distinct engine resources/statistics; no Elo or model promotion"
            } else if candidate.batch_experiment.is_some() {
                "CUDA S batch pilot only; execution_ready=false; strength_eligible=false; same weights/precision/PUCT/final visits; batch width and scheduling differ"
            } else if baseline.search==candidate.search {
                "CUDA B1 A/A whole-clock memory check; execution_ready=false; strength_eligible=false; same weights/runtime/binary/search"
            } else {
                "CUDA S0/S1 pilot only; execution_ready=false; strength_eligible=false; same weights/runtime/binary; only final selection differs"
            }.into();
            limitations[3] = "Pinned clock patch measures position transmission through bestmove with steady_clock, charges partial milliseconds and earns increment only after a timely move; the 100ms read margin is not chess time".into();
            limitations[4] = "Fastchess automatic draw claims require independent A current-position evidence; cutoff remains Incomplete, engine failures remain Loss and stop a broken pilot gate".into();
        }
        if spec.provider_name() == "CUDA" {
            limitations.push("CUDA device0/FP32/TF32off/selected arena is requested admission metadata, not measured VRAM, aggregate GPU allocation or a kernel-enforced hard cap".into());
        }
        if let Some(execution) = spec.match_execution() {
            limitations[0] = "V2 locked CPU/GPU/Hybrid paired execution; execution_ready=false; strength_eligible=false; comparisons follow the declared model/search/resource identities".into();
            limitations[6] = "same-PID executor installs and reads back runner/engine CPU affinity; aggregate RAM still requires recorded inherited enforcement".into();
            limitations.push(format!("Ponder={}; sharing={:?}; CUDA visibility is trusted-engine placement; GPU memory reservations and compute sharing are not hard allocator/compute quotas", execution.ponder, execution.sharing));
        }
        spec.amend_execution_limitations(&mut limitations);
        Ok(FastchessInvocation { args, limitations })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn v6_system_runtime_needs_child_receipt_and_never_snapshots_libraries() {
            use rz_experiments::{
                NativeRuntimeSourceV6, SystemLibrarySpecV6, SystemPlatformV6, SystemRuntimeSpecV6,
            };
            let library = |role: &str, inode: u64, hash: char| SystemLibrarySpecV6 {
                role: role.into(),
                requested_path: format!("/system/{role}.so"),
                canonical_path: format!("/system/{role}.so"),
                version: None,
                sha256: hash.to_string().repeat(64),
                bytes: 100 + inode,
                device: 1,
                inode,
                required_mapping: true,
            };
            let source = NativeRuntimeSourceV6::SystemCuda(SystemRuntimeSpecV6 {
                platform: SystemPlatformV6::Linux,
                libraries: vec![
                    library("ort-core", 1, 'a'),
                    library("ort-provider-cuda", 2, 'b'),
                ],
                system_identity_sha256: "c".repeat(64),
                cuda_device_id: 0,
                cuda_arena_bytes: 1024,
                startup_ms: 100,
                drain_ms: 100,
                capture_max_libraries: 2,
                capture_max_file_bytes: 1024,
                capture_max_total_bytes: 2048,
                capture_timeout_ms: 100,
            });
            let model = ArtifactRef {
                path: "model.onnx".into(),
                sha256: "d".repeat(64),
                bytes: 900,
                source: "fixture".into(),
                license: "fixture".into(),
            };
            let copied_provider = ArtifactRef {
                path: "copied-provider.so".into(),
                sha256: "b".repeat(64),
                bytes: 102,
                source: "fixture".into(),
                license: "fixture".into(),
            };
            assert!(
                validate_v6_runtime_snapshot_boundary(&source, false, true, true, &[&model])
                    .is_ok()
            );
            assert!(
                validate_v6_runtime_snapshot_boundary(&source, true, true, true, &[&model])
                    .is_err()
            );
            assert!(
                validate_v6_runtime_snapshot_boundary(&source, false, false, true, &[&model])
                    .is_err()
            );
            assert!(
                validate_v6_runtime_snapshot_boundary(&source, false, true, false, &[&model])
                    .is_err()
            );
            assert!(
                validate_v6_runtime_snapshot_boundary(
                    &source,
                    false,
                    true,
                    true,
                    &[&model, &copied_provider]
                )
                .is_err()
            );
            assert!(
                validate_v6_runtime_snapshot_boundary(
                    &NativeRuntimeSourceV6::Cpu,
                    false,
                    false,
                    false,
                    &[&model]
                )
                .is_ok()
            );
            assert!(
                validate_v6_runtime_snapshot_boundary(
                    &NativeRuntimeSourceV6::Cpu,
                    true,
                    false,
                    false,
                    &[&model]
                )
                .is_err()
            );
        }

        #[test]
        fn native_runner_registration_preserves_clean_movetime_and_pairs_reap_pins() {
            use crate::pals_launch::{
                PALS_CLOCK_REAP_COLAB_RUNNER_BYTES, PALS_CLOCK_REAP_COLAB_RUNNER_SHA256,
                PALS_CLOCK_REAP_COLAB_RUNNER_VERSION, PALS_CLOCK_REAP_PATCH_BYTES,
                PALS_CLOCK_REAP_PATCH_SHA256, PALS_CLOCK_REAP_RUNNER_BYTES,
                PALS_CLOCK_REAP_RUNNER_SHA256,
            };
            let artifact = |path: &str, sha: &str, bytes| ArtifactRef {
                path: path.into(),
                sha256: sha.into(),
                bytes,
                source: crate::FASTCHESS_SOURCE_URL.into(),
                license: "GPL-3.0".into(),
            };
            let mut runner = rz_experiments::ToolIdentity {
                version: crate::FASTCHESS_VERSION.into(),
                source_url: crate::FASTCHESS_SOURCE_URL.into(),
                source_commit: crate::FASTCHESS_SOURCE_COMMIT.into(),
                binary: artifact("runner/fastchess", &"a".repeat(64), 32),
                dirty: false,
                dirty_patch: None,
                build_mode: "release".into(),
                compiler: "synthetic-metadata-only".into(),
                target: "x86_64-unknown-linux-gnu".into(),
                isa: "x86-64".into(),
            };
            let movetime =
                rz_experiments::NativePairClock::Movetime(rz_experiments::NativeMovetimeV1 {
                    movetime_ms: 10,
                });
            let game = rz_experiments::NativePairClock::Game(rz_experiments::NativeGameClockV3 {
                base_ms: 120_000,
                increment_ms: 1_000,
            });
            assert!(validate_native_runner(&runner, movetime).is_ok());
            runner.dirty = true;
            assert!(validate_native_runner(&runner, movetime).is_err());
            runner.dirty_patch = Some(artifact(
                "runner/clock.patch",
                rz_experiments::FASTCHESS_CLOCK_PATCH_SHA256,
                rz_experiments::FASTCHESS_CLOCK_PATCH_BYTES,
            ));
            assert!(validate_native_runner(&runner, game).is_ok());
            runner.dirty_patch = Some(artifact(
                "runner/clock-reap.patch",
                PALS_CLOCK_REAP_PATCH_SHA256,
                PALS_CLOCK_REAP_PATCH_BYTES,
            ));
            let versions = [
                crate::FASTCHESS_VERSION,
                PALS_CLOCK_REAP_COLAB_RUNNER_VERSION,
            ];
            let hashes = [
                PALS_CLOCK_REAP_RUNNER_SHA256,
                PALS_CLOCK_REAP_COLAB_RUNNER_SHA256,
            ];
            let sizes = [
                PALS_CLOCK_REAP_RUNNER_BYTES,
                PALS_CLOCK_REAP_COLAB_RUNNER_BYTES,
            ];
            for (v, version) in versions.iter().enumerate() {
                for (h, hash) in hashes.iter().enumerate() {
                    for (b, bytes) in sizes.iter().enumerate() {
                        runner.version = (*version).into();
                        runner.binary.sha256 = (*hash).into();
                        runner.binary.bytes = *bytes;
                        assert_eq!(
                            validate_native_runner(&runner, game).is_ok(),
                            v == h && h == b
                        );
                        assert!(validate_native_runner(&runner, movetime).is_err());
                    }
                }
            }
            assert!(validate_native_runner(&runner, game).is_ok());
            for field in ["source", "commit", "patch", "dirty", "version"] {
                let mut invalid = runner.clone();
                match field {
                    "source" => invalid.source_url.push_str("/unsupported"),
                    "commit" => invalid.source_commit = "0".repeat(40),
                    "patch" => invalid.dirty_patch.as_mut().unwrap().bytes -= 1,
                    "dirty" => invalid.dirty = false,
                    "version" => invalid.version.push_str(" unsupported"),
                    _ => unreachable!(),
                }
                assert!(validate_native_runner(&invalid, game).is_err(), "{field}");
            }
        }

        #[test]
        fn v4_argument_ceiling_is_explicit_and_preserves_legacy_32() {
            let args = vec![OsString::from("--bounded-fixture=value"); 48];
            assert!(encode_fastchess_native_args(&args[..32]).is_ok());
            assert!(encode_fastchess_native_args(&args[..33]).is_err());
            assert!(encode_fastchess_native_args_with_limit(&args, 48).is_ok());
            assert!(encode_fastchess_native_args_with_limit(&args, 47).is_err());
            let mut excessive = args;
            excessive.push("--extra=value".into());
            assert!(encode_fastchess_native_args_with_limit(&excessive, 48).is_err());
        }

        #[derive(Clone, Debug)]
        struct RuntimeArgumentsFixture;
        impl sealed::Sealed for RuntimeArgumentsFixture {}
        impl NativeLaunchDeclaration for RuntimeArgumentsFixture {
            fn input_sha256(&self) -> &str {
                "unused-runtime-arguments-fixture"
            }
            fn view(&self) -> NativePairView<'_> {
                panic!("argv fixture does not own a pair")
            }
            fn declared_inputs(&self) -> Vec<&ArtifactRef> {
                Vec::new()
            }
            fn unique_bytes(&self) -> Result<u64, rz_experiments::ManifestError> {
                Ok(0)
            }
            fn engine_view(
                &self,
                _role: NativeEngineRole,
            ) -> Result<NativeEngineView<'_>, rz_experiments::ManifestError> {
                panic!("argv fixture does not resolve an engine")
            }
            fn provider_name(&self) -> &'static str {
                "synthetic"
            }
            fn runtime_arguments(
                &self,
                role: NativeEngineRole,
                root: &Path,
            ) -> Result<Vec<OsString>, ArenaError> {
                Ok(vec![
                    format!("--fixture-role={role:?}").into(),
                    key_path("--fixture-runtime-root=", root)?,
                ])
            }
            fn preflight_arguments(
                &self,
                _role: NativeEngineRole,
                root: &Path,
            ) -> Result<Vec<OsString>, ArenaError> {
                Ok(vec![key_path("--fixture-preflight-cache-root=", root)?])
            }
        }
        impl crate::native_runner::NativeProviderDeclaration for RuntimeArgumentsFixture {
            type Audit = serde_json::Value;
            fn scope(&self) -> &'static str {
                "synthetic-admission-fixture"
            }
            fn receipt_filename(&self) -> &'static str {
                "unused-admission-receipt.json"
            }
            fn startup_filename(&self) -> &'static str {
                "unused-startup.json"
            }
            fn termination_filename(&self) -> &'static str {
                "unused-termination.json"
            }
            fn validate_runtime_admission(&self) -> Result<(), ArenaError> {
                Err(ArenaError::Budget(
                    "injected identical-resource admission failure".into(),
                ))
            }
            fn validate_records(
                &self,
                _role: NativeEngineRole,
                _startup: &[u8],
                _termination: &[u8],
                _session: &str,
            ) -> Result<(Self::Audit, u32), ArenaError> {
                panic!("rejected admission must never inspect provider records")
            }
        }

        #[test]
        fn preparation_failure_policy_envelope_is_saved_and_legacy_bytes_stay_identical() {
            use serde::Serialize;
            #[derive(Clone, Debug)]
            struct PreparationFixture {
                input: String,
                policy: Option<NativePreparationPolicyEnvelope>,
            }
            impl sealed::Sealed for PreparationFixture {}
            impl NativeLaunchDeclaration for PreparationFixture {
                fn input_sha256(&self) -> &str { &self.input }
                fn preparation_policy_envelope(&self) -> Option<NativePreparationPolicyEnvelope> {
                    self.policy.clone()
                }
                fn view(&self) -> NativePairView<'_> { panic!("rejected basename has no pair") }
                fn declared_inputs(&self) -> Vec<&ArtifactRef> { Vec::new() }
                fn unique_bytes(&self) -> Result<u64, rz_experiments::ManifestError> { Ok(0) }
                fn engine_view(&self, _: NativeEngineRole) -> Result<NativeEngineView<'_>, rz_experiments::ManifestError> {
                    panic!("rejected basename has no engine")
                }
                fn provider_name(&self) -> &'static str { "fixture" }
            }
            // This explicit legacy representation is the pre-envelope wire
            // format and compares actual serialized bytes, including field order.
            #[derive(Serialize)]
            struct LegacyPreparation<'a> {
                receipt_version: u32,
                execution_ready: bool,
                input_sha256: &'a str,
                output_directory: &'a str,
                attempt_created: bool,
                attempted_snapshot_relative_paths: &'a [String],
                completed_snapshots: &'a [NativeSnapshotReceipt],
                child_spawned: bool,
                writers_closed: bool,
                input_pins_required: bool,
                original_error: &'a str,
                subsequent_file_owner: &'a str,
                automatic_retry: bool,
            }
            let base_sha = "a".repeat(64);
            let execution_sha = "b".repeat(64);
            let strict = PreparationFixture { input: base_sha.clone(), policy: None };
            let advisory = PreparationFixture {
                input: execution_sha.clone(),
                policy: Some(NativePreparationPolicyEnvelope {
                    domain: "rz-pals-preparation-storage-advisory-v1/1".into(),
                    policy: "advisory".into(),
                    base_launch_sha256: base_sha.clone(),
                    execution_sha256: execution_sha.clone(),
                }),
            };
            let fail = |fixture: &PreparationFixture| {
                prepare_native_launch_for(fixture, Path::new("/"), Path::new("/"), "../invalid")
                    .err().expect("invalid basename cannot start a child")
            };
            let strict_failure = fail(&strict);
            let advisory_failure = fail(&advisory);
            assert!(!strict_failure.receipt.child_spawned && !advisory_failure.receipt.child_spawned);
            let r = &strict_failure.receipt;
            assert!(r.policy_envelope.is_none());
            let old = LegacyPreparation {
                receipt_version: r.receipt_version, execution_ready: r.execution_ready,
                input_sha256: &r.input_sha256, output_directory: &r.output_directory,
                attempt_created: r.attempt_created,
                attempted_snapshot_relative_paths: &r.attempted_snapshot_relative_paths,
                completed_snapshots: &r.completed_snapshots, child_spawned: r.child_spawned,
                writers_closed: r.writers_closed, input_pins_required: r.input_pins_required,
                original_error: &r.original_error, subsequent_file_owner: &r.subsequent_file_owner,
                automatic_retry: r.automatic_retry,
            };
            assert_eq!(serde_json::to_vec(&r).unwrap(), serde_json::to_vec(&old).unwrap());
            let policy = advisory_failure.receipt.policy_envelope.as_ref().unwrap();
            assert_eq!(policy.domain, "rz-pals-preparation-storage-advisory-v1/1");
            assert_eq!(policy.base_launch_sha256, base_sha);
            assert_eq!(policy.execution_sha256, execution_sha);
            assert_eq!(advisory_failure.receipt.input_sha256, execution_sha);

            struct OwnedFixture(std::path::PathBuf);
            impl Drop for OwnedFixture {
                fn drop(&mut self) {
                    let _ = std::fs::remove_file(self.0.join("native-preparation-failure.v1.json"));
                    let _ = std::fs::remove_dir(&self.0);
                }
            }
            let root = OwnedFixture(std::env::temp_dir().join(format!(
                "rz-preparation-policy-{}-{}", std::process::id(),
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
            )));
            std::fs::create_dir(&root.0).unwrap();
            let directory = Dir::open_ambient_dir(&root.0, cap_std::ambient_authority()).unwrap();
            let artifact = save_preparation_failure(Some(&directory), &advisory_failure.receipt, "fixture")
                .unwrap().expect("fixture directory permits evidence saving");
            let bytes = std::fs::read(root.0.join("native-preparation-failure.v1.json")).unwrap();
            assert_eq!(bytes, serde_json::to_vec_pretty(&advisory_failure.receipt).unwrap());
            assert_eq!(artifact.bytes, bytes.len() as u64);
            assert_eq!(artifact.sha256, format!("{:x}", Sha256::digest(&bytes)));
        }

        #[test]
        fn preflight_runtime_roots_are_exclusive_private_fixed_role_pins() {
            let base = std::env::temp_dir().join(format!(
                "rovezero-preflight-root-pins-{}",
                std::process::id()
            ));
            std::fs::create_dir(&base).unwrap();
            let directory = Dir::open_ambient_dir(&base, cap_std::ambient_authority()).unwrap();
            assert!(
                !RuntimeArgumentsFixture
                    .uses_separate_preflight_runtime_root(NativeEngineRole::Baseline)
                    .unwrap()
            );
            assert!(
                !RuntimeArgumentsFixture
                    .uses_separate_preflight_runtime_root(NativeEngineRole::Candidate)
                    .unwrap()
            );
            let mut roots = [
                Some(
                    create_preflight_runtime_root(&directory, &base, NativeEngineRole::Baseline)
                        .unwrap(),
                ),
                Some(
                    create_preflight_runtime_root(&directory, &base, NativeEngineRole::Candidate)
                        .unwrap(),
                ),
            ];
            verify_preflight_runtime_root_pins(&directory, &base, &roots).unwrap();
            assert!(
                create_preflight_runtime_root(&directory, &base, NativeEngineRole::Baseline)
                    .is_err()
            );
            assert_ne!(
                roots[0].as_ref().unwrap().file.metadata().unwrap().ino(),
                roots[1].as_ref().unwrap().file.metadata().unwrap().ino()
            );
            let original_name = roots[0].as_ref().unwrap().name;
            for invalid in [
                "inputs",
                "baseline-runtime",
                "../escape",
                "nested/root",
                "external-candidate-preflight-runtime",
            ] {
                roots[0].as_mut().unwrap().name = invalid;
                assert!(verify_preflight_runtime_root_pins(&directory, &base, &roots).is_err());
            }
            roots[0].as_mut().unwrap().name = original_name;
            roots[0]
                .as_ref()
                .unwrap()
                .file
                .set_permissions(Permissions::from_mode(0o755))
                .unwrap();
            assert!(verify_preflight_runtime_root_pins(&directory, &base, &roots).is_err());
            roots[0]
                .as_ref()
                .unwrap()
                .file
                .set_permissions(Permissions::from_mode(0o700))
                .unwrap();
            verify_preflight_runtime_root_pins(&directory, &base, &roots).unwrap();
            drop(roots);
            drop(directory);
            std::fs::remove_dir_all(base).unwrap();
        }
        #[test]
        fn preflight_runtime_root_recheck_rejects_symlink_and_replaced_inode() {
            let base = std::env::temp_dir().join(format!(
                "rovezero-preflight-root-replace-{}",
                std::process::id()
            ));
            std::fs::create_dir(&base).unwrap();
            let directory = Dir::open_ambient_dir(&base, cap_std::ambient_authority()).unwrap();
            let roots = [
                Some(
                    create_preflight_runtime_root(&directory, &base, NativeEngineRole::Baseline)
                        .unwrap(),
                ),
                None,
            ];
            let path = roots[0].as_ref().unwrap().path.clone();
            let old = base.join("retained-original");
            std::fs::rename(&path, &old).unwrap();
            std::os::unix::fs::symlink(&old, &path).unwrap();
            assert!(verify_preflight_runtime_root_pins(&directory, &base, &roots).is_err());
            std::fs::remove_file(&path).unwrap();
            std::fs::create_dir(&path).unwrap();
            std::fs::set_permissions(&path, Permissions::from_mode(0o700)).unwrap();
            assert!(verify_preflight_runtime_root_pins(&directory, &base, &roots).is_err());
            std::fs::remove_dir(&path).unwrap();
            std::fs::rename(&old, &path).unwrap();
            verify_preflight_runtime_root_pins(&directory, &base, &roots).unwrap();
            drop(roots);
            drop(directory);
            std::fs::remove_dir_all(base).unwrap();
        }
        #[test]
        fn runtime_admission_failure_precedes_preflight_and_retains_original_owner() {
            let base = std::env::temp_dir()
                .join(format!("rovezero-runtime-admission-{}", std::process::id()));
            std::fs::create_dir(&base).expect("exclusive admission fixture");
            std::fs::create_dir(base.join("inputs")).unwrap();
            let directory = Dir::open_ambient_dir(&base, cap_std::ambient_authority()).unwrap();
            let input_directory = directory.open_dir_nofollow("inputs").unwrap();
            let owner = NativeLaunchOwner {
                spec: RuntimeArgumentsFixture,
                _lease: NativeAdmissionLease::acquire().unwrap(),
                snapshot: Snapshot {
                    cwd: directory.try_clone().unwrap().into_std_file(),
                    _runtime_root_pins: [
                        directory.try_clone().unwrap().into_std_file(),
                        directory.try_clone().unwrap().into_std_file(),
                    ],
                    preflight_runtime_roots: [None, None],
                    directory,
                    input_directory,
                    path: base.clone(),
                    output_directory: "synthetic-admission-only".into(),
                    pins: Vec::new(),
                    receipts: Vec::new(),
                    // No runner exists: reaching supervision would be a bug.
                    runner_index: usize::MAX,
                    invocation: FastchessInvocation {
                        args: Vec::new(),
                        limitations: Vec::new(),
                    },
                    watch: OwnedArtifactTreeWatch {
                        max_total_bytes: 1,
                        max_file_bytes: 1,
                        max_files: 1,
                        max_depth: 1,
                    },
                    limits: ProcessLimits {
                        wall_ms: 100,
                        shutdown_grace_ms: 10,
                        max_output_bytes: 1,
                        max_child_processes: 1,
                    },
                    input_subdirectories: BTreeMap::new(),
                    preparation_cache_hint: None,
                },
            };
            let failure = crate::native_runner::run_native_pair_for(owner, None).unwrap_err();
            assert!(matches!(&failure.cause, ArenaError::Budget(detail)
                if detail == "injected identical-resource admission failure"));
            assert!(failure.process().is_none());
            assert!(failure.receipt.is_none() && failure.receipt_artifact.is_none());
            assert_eq!(failure.launch_owner().unwrap().snapshot.path, base);
            drop(failure);
            assert!(
                base.join("inputs").is_dir(),
                "failure must preserve the owned attempt"
            );
            // Synthetic pre-spawn owner only; no runner/model/process exists.
            std::fs::remove_dir_all(base).unwrap();
        }

        #[test]
        fn external_runtime_arguments_are_absent_from_option_preflight() {
            let binary = ArtifactRef {
                path: "synthetic-engine".into(),
                sha256: "a".repeat(64),
                bytes: 1,
                source: "https://example.org/argv-hook-fixture".into(),
                license: "Synthetic argv fixture only".into(),
            };
            let mut external = crate::stockfish19_endpoint(binary, NativeEngineRole::Candidate);
            external.arguments = vec!["--fixture-engine-mode".into()];
            let declaration = RuntimeArgumentsFixture;
            let preflight = external_arguments(
                &declaration,
                NativeEngineRole::Candidate,
                &external,
                &[],
                ExternalArgumentMode::Preflight(Path::new("/owned/attempt/candidate-runtime")),
            )
            .unwrap();
            assert_eq!(
                preflight,
                vec![
                    OsString::from("--fixture-engine-mode"),
                    OsString::from(
                        "--fixture-preflight-cache-root=/owned/attempt/candidate-runtime"
                    ),
                ]
            );
            let runtime = external_arguments(
                &declaration,
                NativeEngineRole::Candidate,
                &external,
                &[],
                ExternalArgumentMode::Runtime(Path::new("/owned/attempt/candidate-runtime")),
            )
            .unwrap();
            assert_eq!(
                runtime,
                vec![
                    OsString::from("--fixture-engine-mode"),
                    OsString::from("--fixture-role=Candidate"),
                    OsString::from("--fixture-runtime-root=/owned/attempt/candidate-runtime"),
                ]
            );
            assert!(encode_fastchess_native_args(&runtime).is_ok());
        }

        #[test]
        fn snapshot_topology_rejects_aliases_duplicates_prefixes_and_excess_depth() {
            let names = vec![
                "pin-00".into(),
                "cuda-bundle/libcudart.so.12".into(),
                "pals-0123456789abcdef/export.json".into(),
                "pals-0123456789abcdef/public_memory.onnx".into(),
            ];
            let parents = snapshot_parents(&names, 4).unwrap();
            assert_eq!(
                parents,
                [
                    PathBuf::from("cuda-bundle"),
                    PathBuf::from("pals-0123456789abcdef")
                ]
                .into_iter()
                .collect()
            );
            for invalid in [
                "",
                "/absolute",
                "../parent",
                "a/../file",
                "a/./file",
                "a//file",
                "a/file/",
                "a\\file",
                "C:/file",
                "a/\nfile",
                "a/'file",
                "a/\"file",
                "a/b/c/d/file",
            ] {
                assert!(
                    snapshot_parents(&[invalid.into()], 4).is_err(),
                    "accepted {invalid:?}"
                );
            }
            for invalid in [vec!["same", "same"], vec!["folder", "folder/model.onnx"]] {
                assert!(
                    snapshot_parents(
                        &invalid.into_iter().map(String::from).collect::<Vec<_>>(),
                        4
                    )
                    .is_err()
                );
            }
        }

        #[test]
        fn preserved_export_topology_rechecks_directory_identity_and_rejects_symlinks() {
            let base = std::env::temp_dir()
                .join(format!("rovezero-export-topology-{}", std::process::id()));
            std::fs::create_dir(&base).expect("exclusive topology fixture");
            std::fs::create_dir(base.join("inputs")).unwrap();
            std::fs::create_dir(base.join("inputs/pals-export")).unwrap();
            let directory = Dir::open_ambient_dir(&base, cap_std::ambient_authority()).unwrap();
            let inputs = directory.open_dir_nofollow("inputs").unwrap();
            let child = inputs.open_dir_nofollow("pals-export").unwrap();
            let subdirectories = [(PathBuf::from("pals-export"), child)]
                .into_iter()
                .collect();
            let bytes = b"synthetic graph bytes";
            let source_path = base.join("source");
            std::fs::write(&source_path, bytes).unwrap();
            let artifact = ArtifactRef {
                path: "public_memory.onnx".into(),
                sha256: format!("{:x}", Sha256::digest(bytes)),
                bytes: bytes.len() as u64,
                source: "https://example.org/export-topology-fixture".into(),
                license: "Synthetic ownership fixture only".into(),
            };
            let relative = Path::new("inputs/pals-export/public_memory.onnx");
            let (parent, name) = input_parent(&inputs, &subdirectories, relative).unwrap();
            let (pin, _) = readonly_copy(
                &artifact,
                File::open(&source_path).unwrap(),
                parent,
                name.to_str().unwrap(),
                base.join(relative),
                false,
                4096,
            )
            .unwrap();
            verify_input_directories(&directory, &inputs, &subdirectories).unwrap();
            assert_eq!(std::fs::read(&pin.path).unwrap(), bytes);
            assert!(
                input_parent(&inputs, &subdirectories, Path::new("inputs/unowned/file")).is_err()
            );
            assert!(
                readonly_copy(
                    &artifact,
                    File::open(&source_path).unwrap(),
                    &inputs,
                    "pals-export/alias",
                    base.join("inputs/pals-export/alias"),
                    false,
                    4096
                )
                .is_err()
            );
            std::fs::rename(base.join("inputs/pals-export"), base.join("old-export")).unwrap();
            std::os::unix::fs::symlink(base.join("old-export"), base.join("inputs/pals-export"))
                .unwrap();
            assert!(verify_input_directories(&directory, &inputs, &subdirectories).is_err());
            std::fs::remove_file(base.join("inputs/pals-export")).unwrap();
            std::fs::create_dir(base.join("inputs/pals-export")).unwrap();
            assert!(verify_input_directories(&directory, &inputs, &subdirectories).is_err());
            assert_eq!(
                std::fs::read(base.join("old-export/public_memory.onnx")).unwrap(),
                bytes
            );
            drop(pin);
            // Exclusive synthetic fixture, no engine/session or model owner.
            std::fs::remove_dir_all(base).unwrap();
        }

        #[test]
        fn engine_and_environment_launcher_are_executable_readonly_private_snapshots() {
            let base = std::env::temp_dir()
                .join(format!("rovezero-environment-pin-{}", std::process::id()));
            std::fs::create_dir(&base).expect("exclusive environment pin fixture");
            let source = base.join("source");
            let output = base.join("output");
            std::fs::create_dir(&source).unwrap();
            std::fs::create_dir(&output).unwrap();
            let make = |name: &str| {
                let bytes = format!("synthetic {name}; not a native execution proof");
                std::fs::write(source.join(name), &bytes).unwrap();
                ArtifactRef {
                    path: name.into(),
                    sha256: format!("{:x}", Sha256::digest(bytes.as_bytes())),
                    bytes: bytes.len() as u64,
                    source: "https://example.org/environment-pin-fixture".into(),
                    license: "Synthetic ownership fixture only".into(),
                }
            };
            let binary = make("engine");
            let launcher = make("launcher");
            let asset = make("asset");
            let mut external =
                crate::stockfish19_endpoint(binary.clone(), NativeEngineRole::Candidate);
            external.assets.push(asset.clone());
            external.environment = Some(rz_experiments::EngineEnvironmentV2 {
                launcher: launcher.clone(),
                variables: std::collections::BTreeMap::new(),
            });
            let view = NativeEngineView {
                engine_id: &external.id,
                artifacts: &[],
                cuda_bundle: None,
                search: None,
                batch_experiment: None,
                external: Some(&external),
                environment: external.environment.as_ref(),
            };
            let directory = Dir::open_ambient_dir(&output, cap_std::ambient_authority()).unwrap();
            for (artifact, mode) in [(&binary, 0o500), (&launcher, 0o500), (&asset, 0o400)] {
                let (pin, receipt) = readonly_copy(
                    artifact,
                    artifact.open_verified(&source, 4096).unwrap(),
                    &directory,
                    &artifact.path,
                    output.join(&artifact.path),
                    view.executable_input(artifact),
                    4096,
                )
                .unwrap();
                assert_eq!(
                    pin.file.metadata().unwrap().permissions().mode() & 0o777,
                    mode
                );
                assert!(receipt.distinct_source_inode && receipt.closed_writer_read_only);
                assert_eq!(
                    std::fs::read(&pin.path).unwrap(),
                    std::fs::read(source.join(&artifact.path)).unwrap()
                );
            }
            drop(directory);
            // Only this exclusive synthetic tree; no processes or model buffers exist.
            std::fs::remove_dir_all(base).unwrap();
        }

        #[test]
        fn snapshot_stream_checks_tail_corruption_growth_and_truncation_across_chunks() {
            let base = std::env::temp_dir().join(format!(
                "rovezero-snapshot-stream-boundary-{}",
                std::process::id()
            ));
            std::fs::create_dir(&base).expect("exclusive synthetic stream fixture");
            let source = base.join("source");
            let output = base.join("output");
            std::fs::create_dir(&source).unwrap();
            std::fs::create_dir(&output).unwrap();
            let bytes: Vec<u8> = (0..(2 * 64 * 1024 + 17)).map(|i| (i % 251) as u8).collect();
            let artifact = ArtifactRef {
                path: "input".into(),
                sha256: format!("{:x}", Sha256::digest(&bytes)),
                bytes: bytes.len() as u64,
                source: "https://example.org/synthetic-stream-boundary".into(),
                license: "Synthetic ownership fixture only".into(),
            };
            let directory = Dir::open_ambient_dir(&output, cap_std::ambient_authority()).unwrap();
            for case in ["valid", "tail", "growth", "truncation"] {
                std::fs::write(source.join("input"), &bytes).unwrap();
                let verified = artifact.open_verified(&source, artifact.bytes).unwrap();
                match case {
                    "tail" => {
                        let mut changed = bytes.clone();
                        *changed.last_mut().unwrap() ^= 1;
                        std::fs::write(source.join("input"), changed).unwrap();
                    }
                    "growth" => {
                        let mut writer = std::fs::OpenOptions::new()
                            .append(true)
                            .open(source.join("input"))
                            .unwrap();
                        writer.write_all(&[1]).unwrap();
                    }
                    "truncation" => {
                        std::fs::OpenOptions::new()
                            .write(true)
                            .open(source.join("input"))
                            .unwrap()
                            .set_len(artifact.bytes - 1)
                            .unwrap();
                    }
                    _ => {}
                }
                let copied = readonly_copy(
                    &artifact,
                    verified,
                    &directory,
                    case,
                    output.join(case),
                    false,
                    artifact.bytes,
                );
                if case == "valid" {
                    let (mut pin, receipt) = copied.unwrap();
                    assert!(receipt.distinct_source_inode && receipt.closed_writer_read_only);
                    assert_eq!(std::fs::read(&pin.path).unwrap(), bytes);
                    assert_eq!(pin.file.stream_position().unwrap(), 0);
                    verify_copy(&mut pin.file, &artifact).unwrap();
                    assert_eq!(pin.file.stream_position().unwrap(), 0);
                    std::fs::set_permissions(&pin.path, Permissions::from_mode(0o600)).unwrap();
                    let mut changed = bytes.clone();
                    *changed.last_mut().unwrap() ^= 1;
                    std::fs::write(&pin.path, changed).unwrap();
                    assert!(matches!(
                        verify_copy(&mut pin.file, &artifact),
                        Err(ArenaError::Integrity(_))
                    ));
                } else {
                    assert!(matches!(copied, Err(ArenaError::Integrity(_))));
                    assert!(
                        output.join(case).exists(),
                        "failed partial copy must remain"
                    );
                }
            }
            drop(directory);
            // This exclusive synthetic tree never owned native children/NN.
            std::fs::remove_dir_all(base).unwrap();
        }

        #[test]
        fn verified_source_pin_survives_path_replacement_and_rejects_same_inode_writes() {
            let base = std::env::temp_dir()
                .join(format!("rovezero-verified-copy-pin-{}", std::process::id()));
            std::fs::create_dir(&base).expect("exclusive synthetic test directory");
            let source = base.join("source");
            let output = base.join("output");
            std::fs::create_dir(&source).unwrap();
            std::fs::create_dir(&output).unwrap();
            std::fs::write(source.join("input"), b"original").unwrap();
            std::fs::hard_link(source.join("input"), source.join("alias")).unwrap();
            let artifact = ArtifactRef {
                path: "input".into(),
                sha256: format!("{:x}", Sha256::digest(b"original")),
                bytes: 8,
                source: "https://example.org/synthetic-copy-pin".into(),
                license: "Synthetic ownership fixture only".into(),
            };
            let original = artifact.open_verified(&source, 8).unwrap();
            let mutation_probe = artifact.open_verified(&source, 8).unwrap();
            std::fs::rename(source.join("input"), source.join("renamed")).unwrap();
            std::fs::write(source.join("input"), b"replaced").unwrap();
            let directory = Dir::open_ambient_dir(&output, cap_std::ambient_authority()).unwrap();
            let (pin, receipt) = readonly_copy(
                &artifact,
                original,
                &directory,
                "valid",
                output.join("valid"),
                false,
                8,
            )
            .unwrap();
            assert_eq!(std::fs::read(&pin.path).unwrap(), b"original");
            assert!(receipt.distinct_source_inode && receipt.closed_writer_read_only);
            drop(pin);

            // Identical length through another name of the pinned inode must
            // still fail the copy-stream digest, without publishing an InputPin.
            std::fs::write(source.join("alias"), b"mutated!").unwrap();
            let result = readonly_copy(
                &artifact,
                mutation_probe,
                &directory,
                "rejected",
                output.join("rejected"),
                false,
                8,
            );
            assert!(matches!(result, Err(ArenaError::Integrity(_))));
            assert_eq!(std::fs::read(output.join("valid")).unwrap(), b"original");
            drop(directory);
            // This exclusive synthetic tree never owned a child or NN runtime.
            std::fs::remove_dir_all(base).unwrap();
        }
    }
}
