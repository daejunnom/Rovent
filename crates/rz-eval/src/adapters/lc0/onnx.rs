//! Explicit FP32 ONNX Runtime backend. No implicit runtime download or CPU fallback.
//!
//! Successful `run` is synchronous and requires exclusive access. A CUDA Run
//! error does not prove device completion: the session and input are quarantined.
//! Runtime must retain request leases until a physical Ready, independently of
//! logical cancellation or deadline rejection.

use crate::asset::{
    self, AssetMetadata, AssetProfile, MaiaAsset, INPUT_NAME, MLH_NAME, POLICY_NAME, WDL_NAME,
};
use crate::error::{BackendError, CauseCode, FailureKind as K, FailureStage as S};
use crate::runtime_pin::RuntimeLibraryPin;
use crate::{output, RawOutput};
use ort::execution_providers::{
    ArenaExtendStrategy, CPUExecutionProvider, CUDAExecutionProvider, ExecutionProvider,
};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::tensor::TensorElementType;
use ort::value::{Tensor, ValueType};
use rz_encoding::classical::{EncodedInput, INPUT_VALUES};
use rz_encoding::POLICY_SIZE;
#[cfg(target_os = "linux")]
pub use rz_native_loader::system::{
    SystemCaptureLimits, SystemLibraryRequest, SystemMappingAvailability, SystemRuntime,
};
use rz_native_loader::{LibrarySet, LoadError, ProcessLibrarySet};
pub use rz_native_loader::{
    NativeLoadingProfile, NativeMappingObservation, NVIDIA_LOAD_ORDER, ORT_LIBRARY_NAMES,
};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[cfg(feature = "experimental-io-binding")]
#[path = "onnx_io.rs"]
mod io;
const MAX_BATCH: usize = 16;

/// A successful normal-operation check may leave native mapping evidence
/// unknown. Only `Observed` represents a complete mapping observation; neither
/// variant is a CUDA completion fence or proof of a successful GPU Run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaRuntimeMappingEvidence {
    Observed,
    Unknown,
}

/// CPU wall-clock boundaries. Run includes kernels and synchronization;
/// these are not GPU event times. None means that transfer is inside Run.
#[derive(Clone, Copy, Debug, Default)]
pub struct IoTimings {
    pub host_stage: std::time::Duration,
    pub transfer_in: Option<std::time::Duration>,
    pub run: std::time::Duration,
    pub output_fence: Option<std::time::Duration>,
    pub transfer_out: Option<std::time::Duration>,
    pub own_outputs: std::time::Duration,
}

/// Independent, explicit experiments; compiling them does not enable them.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExecutionExperiments {
    pub reuse_buffers: bool,
    pub io_binding: bool,
    pub cuda_graph: bool,
    /// CUDA's implicit CPU allocator only; never enables CPU node fallback.
    pub disable_cuda_cpu_arena: bool,
    /// Independent derived-model trial. Its native session owns the backing bytes.
    pub zero_copy_ort: bool,
    /// Independent ORT trial with native copying and post-commit Rust buffer release.
    pub copy_ort_model: bool,
}
impl ExecutionExperiments {
    pub fn uses_ort_model(self) -> bool {
        self.zero_copy_ort || self.copy_ort_model
    }
}

/// Source-thread host intervals. NativeInvocation includes synchronous ORT Run,
/// not separately measured GPU kernels or H2D/D2H. Failed CUDA Run is unfenced.
#[derive(Default, Debug)]
pub struct NativeRunTimings {
    pub preparation: Option<(Instant, Instant)>,
    pub invocation: Option<(Instant, Instant)>,
    pub output: Option<(Instant, Instant)>,
    pub completion_attested: bool,
}
/// Input staging plus native and owned output copies. Excludes caller features,
/// model/session/ORT activation workspace: bootstrap must budget those too.
pub const IO_BYTES_PER_ITEM: usize = (INPUT_VALUES + 2 * (POLICY_SIZE + 3)) * 4;

/// Declaration identity only. This is not evidence of a loaded provider or Run.
/// Bootstrap and the offline batch preparer use exactly the same codec.
pub fn declared_backend_identity(
    runtime: [u8; 32],
    asset_manifest: [u8; 32],
    bundle: Option<[u8; 32]>,
    config: &BackendConfig,
) -> [u8; 32] {
    let mut profile = format!(
        "rz-maia-ort-v1;ort=1.22.0;wrapper=2.0.0-rc.10;runtime={runtime:?};asset={asset_manifest:?};provider={:?};threads={};batch={};fp32;tf32=0;opt=1;sync;full=1;temp=1;sum=1e-5",
        config.provider, config.intra_threads, config.max_batch
    );
    if let Some(bundle) = bundle {
        profile.push_str(&format!(
            ";cuda-bundle-v1={bundle:?};loader=linux-exact-global-v1"
        ));
    }
    if matches!(config.provider, Provider::Cuda { .. }) {
        profile.push_str(";cuda-arena-extend=same-as-requested-v1");
    }
    let experiments = config.experiments;
    if experiments.reuse_buffers || experiments.io_binding || experiments.cuda_graph {
        // Keep the original buffer-only codec. Binding now owns its copy
        // sessions and never acquires duplicate native output handles.
        let copy = if experiments.io_binding {
            "synchronous-owned-onnx-identity-v2"
        } else {
            "synchronous-ort-identity"
        };
        profile.push_str(&format!(
            ";execution-experiment-v1=ExecutionExperiments {{ reuse_buffers: {}, io_binding: {}, cuda_graph: {} }};copy={copy}",
            experiments.reuse_buffers, experiments.io_binding, experiments.cuda_graph
        ));
    }
    if experiments.disable_cuda_cpu_arena {
        profile.push_str(";cuda-cpu-arena=disabled-v1");
    }
    if experiments.zero_copy_ort {
        profile.push_str(";ort-flatbuffer=owned-direct-initializers-v1");
    }
    if experiments.copy_ort_model {
        profile.push_str(";ort-flatbuffer=native-copy-serialized-release-v1");
    }
    asset::sha256(profile.as_bytes())
}

#[derive(Clone)]
pub struct OrtRuntime {
    owner: RuntimeOwner,
    build_info: String,
    cuda_libraries: Option<ProcessLibrarySet>,
    ort_libraries: Option<Arc<Vec<rz_native_loader::OwnedLibrary>>>,
    loading_profile: Option<NativeLoadingProfile>,
}

#[derive(Clone)]
enum RuntimeOwner {
    Copy(RuntimeLibraryPin),
    #[cfg(target_os = "linux")]
    System {
        runtime: SystemRuntime,
        core_digest: [u8; 32],
    },
}

impl RuntimeOwner {
    fn same_copy(&self, pin: &RuntimeLibraryPin) -> bool {
        match self {
            Self::Copy(current) => {
                current.path() == pin.path()
                    && current.binary_digest() == pin.binary_digest()
                    && current.bundle_digest() == pin.bundle_digest()
            }
            #[cfg(target_os = "linux")]
            Self::System { .. } => false,
        }
    }

    fn binary_digest(&self) -> [u8; 32] {
        match self {
            Self::Copy(pin) => pin.binary_digest(),
            #[cfg(target_os = "linux")]
            Self::System { core_digest, .. } => *core_digest,
        }
    }

    fn legacy_bundle_digest(&self) -> Option<[u8; 32]> {
        match self {
            Self::Copy(pin) => pin.bundle_digest(),
            #[cfg(target_os = "linux")]
            Self::System { .. } => None,
        }
    }

    fn same_owner(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Copy(left), Self::Copy(right)) => {
                self.same_copy(right) && right.path() == left.path()
            }
            #[cfg(target_os = "linux")]
            (Self::System { runtime: left, .. }, Self::System { runtime: right, .. }) => {
                left.identity_sha256() == right.identity_sha256()
            }
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }
}

impl std::fmt::Debug for OrtRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrtRuntime")
            .field("runtime_binary_digest", &self.owner.binary_digest())
            .field("build_info", &self.build_info)
            .field("cuda_bundle", &self.owner.legacy_bundle_digest())
            .field("native_loading_profile", &self.loading_profile)
            .finish()
    }
}

struct RuntimeLatch {
    // ORT may retain the native library even if later initialization fails.
    // Preserve its bootstrap capability on both success and latched failure.
    owner: RuntimeOwner,
    loading_profile: Option<NativeLoadingProfile>,
    result: Result<OrtRuntime, BackendError>,
}

static RUNTIME: Mutex<Option<RuntimeLatch>> = Mutex::new(None);

#[cfg(target_os = "linux")]
fn system_runtime_unavailable() -> BackendError {
    BackendError::new(
        K::BackendUnavailable,
        S::Backend,
        "system ORT initialization failed; install compatible native dependencies",
    )
}

/// Keep the callback adjacent to the actual native invocation. In particular,
/// a non-UTF8 path rejects before notifying the original owner.
#[cfg(target_os = "linux")]
fn with_system_ort_entry<T>(
    path: Option<&str>,
    before_native: impl FnOnce(),
    initialize: impl FnOnce(&str) -> Result<T, BackendError>,
) -> Result<T, BackendError> {
    let path = path.ok_or_else(system_runtime_unavailable)?;
    before_native();
    initialize(path)
}

#[cfg(target_os = "linux")]
fn with_checked_system_ort_entry<T>(
    path: Option<&str>,
    before_native: impl FnOnce() -> Result<(), BackendError>,
    entered: &std::cell::Cell<bool>,
    initialize: impl FnOnce(&str) -> Result<T, BackendError>,
) -> Result<T, BackendError> {
    with_system_ort_entry(
        path,
        || {},
        |path| {
            before_native()?;
            entered.set(true);
            initialize(path)
        },
    )
}

#[cfg(target_os = "linux")]
fn should_latch_system_attempt(entered: bool, latch_before_entry_failure: bool) -> bool {
    entered || latch_before_entry_failure
}

impl OrtRuntime {
    /// Bootstrap must be the sole initializer of the process-global `ort`
    /// library. Refuse prior initialization, path changes and failed retries.
    /// Requires the bootstrap-owned, verified and durably pinned copy. Mutable
    /// caller paths cannot authorize native loading. CUDA additionally requires
    /// the complete pinned bundle; its dependency preload never changes PATH or
    /// LD_LIBRARY_PATH. No dependencies are downloaded by this API.
    /// A hash is identity, not a sandbox for executing native code.
    pub fn load(pin: &RuntimeLibraryPin) -> Result<Self, BackendError> {
        Self::load_with_cuda_loading_profile(pin, NativeLoadingProfile::EagerCuda12Cudnn9V1)
    }

    /// Explicit diagnostic profile selection before any native constructor.
    /// Existing CPU/LC0 callers keep load() and eager16. A different profile
    /// cannot reuse an already initialized process, even with identical files.
    pub fn load_with_cuda_loading_profile(
        pin: &RuntimeLibraryPin,
        requested_profile: NativeLoadingProfile,
    ) -> Result<Self, BackendError> {
        let loading_profile = cuda_loading_profile(pin.bundle_digest(), requested_profile)?;
        let unavailable = || {
            BackendError::new(
                K::BackendUnavailable,
                S::Backend,
                "ORT initialization failed; native dependencies must be provided explicitly",
            )
        };
        let mut guard = RUNTIME.lock().map_err(|error| {
            unavailable().with_external_cause(CauseCode::RuntimeInitialize, &error)
        })?;
        if let Some(latch) = guard.as_mut() {
            if !latch.owner.same_copy(pin) {
                return Err(BackendError::new(
                    K::IdentityMismatch,
                    S::Backend,
                    "ORT library is process-global and already pinned",
                ));
            }
            if latch.loading_profile != loading_profile {
                let failure = BackendError::new(
                    K::IdentityMismatch,
                    S::Backend,
                    "ORT process native loading policy cannot be changed",
                );
                if latch.result.is_ok() {
                    latch.result = Err(failure);
                }
                return Err(latch.result.as_ref().unwrap_err().clone());
            }
            let runtime = latch.result.as_ref().map_err(Clone::clone)?;
            return if runtime.owner.same_copy(pin) {
                let runtime = runtime.clone();
                drop(guard);
                if runtime.bundle_digest().is_some() {
                    runtime.verify_cuda_mappings(false)?;
                }
                Ok(runtime)
            } else {
                Err(BackendError::new(
                    K::IdentityMismatch,
                    S::Backend,
                    "ORT library is process-global and already pinned",
                ))
            };
        }
        // ort rc.10 panics on dlopen / API-version failure; translate that narrow
        // bootstrap boundary and latch failure. Native crashes are not recoverable.
        let result = std::panic::catch_unwind(|| {
            let path_text = pin.path().to_str().ok_or_else(unavailable)?;
            let ort_libraries = if pin.bundle_digest().is_some() {
                Some(Arc::new(
                    pin.try_clone_ort_pins()?
                        .into_iter()
                        .map(|library| rz_native_loader::OwnedLibrary {
                            path: library.path,
                            file: library.file,
                            digest: library.digest,
                            bytes: library.bytes,
                        })
                        .collect::<Vec<_>>(),
                ))
            } else {
                None
            };
            if let Some(libraries) = &ort_libraries {
                // First CUDA profile is closed to the selected GPU wheel's
                // exact bytes and digests. Validate its declarations before any
                // NVIDIA constructor can run, using C's verified copied pins.
                rz_native_loader::validate_runtime_profile(libraries).map_err(loader_error)?;
            }
            let cuda_libraries = if pin.bundle_digest().is_some() {
                let dependencies = pin
                    .try_clone_nvidia_pins()?
                    .into_iter()
                    .map(|library| rz_native_loader::OwnedLibrary {
                        path: library.path,
                        file: library.file,
                        digest: library.digest,
                        bytes: library.bytes,
                    })
                    .collect();
                Some(
                    LibrarySet::load_with_profile(
                        dependencies,
                        loading_profile.expect("CUDA has an explicit loading profile"),
                    )
                    .map_err(loader_error)?,
                )
            } else {
                None
            };
            if !ort::init_from(path_text)
                .with_name("RoveZero-C")
                .commit()
                .map_err(|error| {
                    unavailable().with_ort_cause(CauseCode::RuntimeInitialize, error)
                })?
            {
                return Err(BackendError::new(
                    K::IdentityMismatch,
                    S::Backend,
                    "ORT was initialized outside bootstrap",
                ));
            }
            let info = ort::info();
            // Official wheels use git-branch=HEAD; the tag's source commit is
            // authoritative, while the bootstrap-supplied digest pins the build.
            if !info.contains("git-commit-id=f217402897,")
                && !info.contains("git-commit-id=f217402897f40ebba457e2421bc0a4702771968e,")
            {
                return Err(BackendError::new(
                    K::BackendUnavailable,
                    S::Backend,
                    "this profile requires ORT release 1.22.0",
                ));
            }
            Ok(Self {
                owner: RuntimeOwner::Copy(pin.clone()),
                build_info: info.to_owned(),
                cuda_libraries,
                ort_libraries,
                loading_profile,
            })
        })
        .unwrap_or_else(|payload| {
            let cause = CauseCode::RuntimePanic;
            let error = if let Some(message) = payload.downcast_ref::<&str>() {
                unavailable().with_external_cause(cause, message)
            } else if let Some(message) = payload.downcast_ref::<String>() {
                unavailable().with_external_cause(cause, message)
            } else {
                unavailable().with_external_cause(cause, &"non-string native initialization panic")
            };
            Err(error)
        });
        *guard = Some(RuntimeLatch {
            owner: RuntimeOwner::Copy(pin.clone()),
            loading_profile,
            result: result.clone(),
        });
        result
    }

    /// Initialize ORT from an explicitly captured system installation without
    /// copying CUDA or ORT files into an engine-owned runtime directory.
    ///
    /// `ort-core` and `ort-provider-cuda` must be declared and required. The
    /// provider and GPU are not considered operational until session creation,
    /// a real CUDA Run and the existing physical-completion checks succeed.
    /// ORT is loaded by its captured canonical system path so provider discovery
    /// can use that installation's layout. A concurrent package replacement
    /// can race native loading: the origin audit detects a mismatched mapping
    /// afterwards but cannot undo native constructors that have already run.
    /// Provider readiness and physical completion require separate real checks.
    #[cfg(target_os = "linux")]
    pub fn load_system_cuda(runtime: &SystemRuntime) -> Result<Self, BackendError> {
        Self::load_system_cuda_with_entry(runtime, || {})
    }

    /// Reports this attempt's first ORT native entry to its original owner.
    /// Pure admission failures and reuse of the process latch do not call the
    /// callback. Neither callback nor latch reuse attests GPU completion or
    /// says whether earlier work in the process has completed.
    ///
    /// The callback executes immediately before `ort::init_from` and within
    /// the same unwind boundary as initialization. If it panics, initialization
    /// is not attempted; the original owner and typed panic error remain in
    /// the process latch. Callers must separately preserve their own custody.
    #[cfg(target_os = "linux")]
    pub fn load_system_cuda_with_entry(
        runtime: &SystemRuntime,
        before_native: impl FnOnce(),
    ) -> Result<Self, BackendError> {
        Self::load_system_cuda_with_entry_impl(
            runtime,
            || {
                before_native();
                Ok(())
            },
            true,
        )
    }

    /// Check the caller's live capability immediately before first native
    /// initialization. Rejection or unwind before entry does not install a
    /// process latch. On same-owner latch reuse this hook is not called: the
    /// caller must independently recheck its parent and scope in that branch.
    /// The check runs under the runtime mutex and must not reenter this loader.
    #[cfg(target_os = "linux")]
    pub fn load_system_cuda_with_checked_entry(
        runtime: &SystemRuntime,
        before_native: impl FnOnce() -> Result<(), BackendError>,
    ) -> Result<Self, BackendError> {
        Self::load_system_cuda_with_entry_impl(runtime, before_native, false)
    }

    #[cfg(target_os = "linux")]
    fn load_system_cuda_with_entry_impl(
        runtime: &SystemRuntime,
        before_native: impl FnOnce() -> Result<(), BackendError>,
        latch_before_entry_failure: bool,
    ) -> Result<Self, BackendError> {
        let unavailable = system_runtime_unavailable;
        let libraries = runtime.libraries();
        let core = libraries
            .iter()
            .find(|item| item.role == "ort-core" && item.required_mapping)
            .ok_or_else(|| {
                BackendError::new(
                    K::InvalidInput,
                    S::Admission,
                    "system CUDA requires a required ort-core role",
                )
            })?;
        if !libraries
            .iter()
            .any(|item| item.role == "ort-provider-cuda" && item.required_mapping)
        {
            return Err(BackendError::new(
                K::InvalidInput,
                S::Admission,
                "system CUDA requires a required ort-provider-cuda role",
            ));
        }
        let owner = RuntimeOwner::System {
            runtime: runtime.clone(),
            core_digest: core.sha256,
        };
        let mut guard = RUNTIME.lock().map_err(|error| {
            unavailable().with_external_cause(CauseCode::RuntimeInitialize, &error)
        })?;
        if let Some(latch) = guard.as_ref() {
            if !latch.owner.same_owner(&owner) || latch.loading_profile.is_some() {
                return Err(BackendError::new(
                    K::IdentityMismatch,
                    S::Backend,
                    "ORT process runtime owner cannot be changed",
                ));
            }
            let result = latch.result.clone();
            drop(guard);
            let loaded = result?;
            loaded.verify_cuda_mappings(false)?;
            return Ok(loaded);
        }
        // This is a new load. A source-path encoding error precedes native
        // entry and must not install a latch for an unattempted initialization.
        // Existing latch reuse retains its historical validation order.
        let core_path = core.canonical_path.to_str().ok_or_else(unavailable)?;
        let entered = std::cell::Cell::new(false);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_checked_system_ort_entry(Some(core_path), before_native, &entered, |path| {
                if !ort::init_from(path)
                    .with_name("RoveZero-C-system")
                    .commit()
                    .map_err(|error| {
                        unavailable().with_ort_cause(CauseCode::RuntimeInitialize, error)
                    })?
                {
                    return Err(BackendError::new(
                        K::IdentityMismatch,
                        S::Backend,
                        "ORT was initialized outside bootstrap",
                    ));
                }
                let info = ort::info();
                // A build commit is an observation, not a compatibility gate.
                let observation = runtime.observe_mappings().map_err(loader_error)?;
                if observation.availability == SystemMappingAvailability::Observed
                    && !observation
                        .mapped_roles
                        .iter()
                        .any(|role| role == "ort-core")
                {
                    return Err(BackendError::new(
                        K::IdentityMismatch,
                        S::Backend,
                        "system ORT core mapping does not match the captured runtime",
                    ));
                }
                Ok(Self {
                    owner: owner.clone(),
                    build_info: info.to_owned(),
                    cuda_libraries: None,
                    ort_libraries: None,
                    loading_profile: None,
                })
            })
        }))
        .unwrap_or_else(|payload| {
            let error = if let Some(message) = payload.downcast_ref::<&str>() {
                unavailable().with_external_cause(CauseCode::RuntimePanic, message)
            } else if let Some(message) = payload.downcast_ref::<String>() {
                unavailable().with_external_cause(CauseCode::RuntimePanic, message)
            } else {
                unavailable().with_external_cause(
                    CauseCode::RuntimePanic,
                    &"non-string native initialization panic",
                )
            };
            Err(error)
        });
        if !should_latch_system_attempt(entered.get(), latch_before_entry_failure) {
            return result;
        }
        *guard = Some(RuntimeLatch {
            owner,
            loading_profile: None,
            result: result.clone(),
        });
        result
    }

    pub fn binary_digest(&self) -> [u8; 32] {
        self.owner.binary_digest()
    }
    pub fn build_info(&self) -> &str {
        &self.build_info
    }

    pub fn bundle_digest(&self) -> Option<[u8; 32]> {
        self.owner.legacy_bundle_digest()
    }

    /// Actual CUDA runtime identity; system identity is never presented as a
    /// historical bundle digest. This value alone does not prove a GPU Run.
    pub fn cuda_runtime_identity(&self) -> Option<[u8; 32]> {
        match &self.owner {
            RuntimeOwner::Copy(pin) => pin.bundle_digest(),
            #[cfg(target_os = "linux")]
            RuntimeOwner::System { runtime, .. } => Some(runtime.identity_sha256()),
        }
    }

    pub fn is_cuda(&self) -> bool {
        self.cuda_runtime_identity().is_some()
    }

    pub fn is_system_cuda(&self) -> bool {
        match &self.owner {
            RuntimeOwner::Copy(_) => false,
            #[cfg(target_os = "linux")]
            RuntimeOwner::System { .. } => true,
        }
    }

    /// Access the same pinned owner for the new system-specific observation
    /// schema. Mapping observation remains separate from GPU completion.
    #[cfg(target_os = "linux")]
    pub fn system_runtime(&self) -> Option<&SystemRuntime> {
        match &self.owner {
            RuntimeOwner::System { runtime, .. } => Some(runtime),
            RuntimeOwner::Copy(_) => None,
        }
    }

    pub fn native_loading_profile(&self) -> Option<NativeLoadingProfile> {
        self.loading_profile
    }

    pub fn native_loading_profile_digest(&self) -> Option<[u8; 32]> {
        self.loading_profile
            .map(|profile| asset::sha256(profile.canonical_descriptor().as_bytes()))
    }

    /// Actual mapping origin audit, sampled by explicit acceptance callers.
    /// It shares the immutable failure latch with normal verification and does
    /// not initialize a provider or replace a physical completion fence.
    pub fn cuda_mapping_observation(
        &self,
        include_runtime: bool,
    ) -> Result<NativeMappingObservation, BackendError> {
        self.audit_cuda_mappings(include_runtime, true, true)
            .map(|(observed, _)| observed.expect("explicit origin observation was requested"))
    }

    /// Canonical descriptors of the actually retained runtime bundle. This is
    /// loaded pin metadata, not an unchecked bootstrap JSON declaration.
    pub fn bundle_files(&self) -> Option<&[crate::runtime_pin::RuntimeBundleFile]> {
        match &self.owner {
            RuntimeOwner::Copy(pin) => pin.bundle_files(),
            #[cfg(target_os = "linux")]
            RuntimeOwner::System { .. } => None,
        }
    }

    /// Registered manifest semantics of the retained pins, including aliases.
    pub fn bundle_spec(&self) -> Option<crate::runtime_pin::CudaRuntimeBundleSpec> {
        match &self.owner {
            RuntimeOwner::Copy(pin) => pin.bundle_spec(),
            #[cfg(target_os = "linux")]
            RuntimeOwner::System { .. } => None,
        }
    }

    /// Audit the explicitly preloaded NVIDIA dependencies before constructing a
    /// CUDA session. ORT provider images may load lazily during the first CUDA
    /// Run; callers must use the full audit after that Run and its completion
    /// fence succeed. This does not initialize a provider or prove completion.
    pub fn verify_cuda_dependencies(&self) -> Result<(), BackendError> {
        self.verify_cuda_mappings(false)
    }

    /// Recheck all resident bundle images after later numerical/worker calls,
    /// before accepting their final GPU report. This is an origin audit, not a
    /// completion fence, VRAM measurement or device-drain operation.
    pub fn verify_cuda_runtime_mappings(&self) -> Result<(), BackendError> {
        self.audit_cuda_mappings(true, false, true).map(|_| ())
    }

    /// Check for observed CUDA mapping conflicts during normal execution.
    /// Missing `/proc` evidence is explicitly `Unknown`; callers must not
    /// promote that result to a strict mapping witness or physical completion.
    pub fn observe_cuda_runtime_mappings(
        &self,
    ) -> Result<CudaRuntimeMappingEvidence, BackendError> {
        self.audit_cuda_mappings(true, false, false)
            .map(|(_, evidence)| evidence)
    }

    fn verify_cuda_mappings(&self, include_runtime: bool) -> Result<(), BackendError> {
        self.audit_cuda_mappings(include_runtime, false, false)
            .map(|_| ())
    }

    fn audit_cuda_mappings(
        &self,
        include_runtime: bool,
        observe: bool,
        strict: bool,
    ) -> Result<(Option<NativeMappingObservation>, CudaRuntimeMappingEvidence), BackendError> {
        #[cfg(not(target_os = "linux"))]
        let _ = strict;
        // Existing clones must observe the first latched failure too. A later
        // filesystem/map change cannot turn a poisoned process runtime into a
        // valid constructor input, or replace its original typed cause.
        let mut guard = RUNTIME.lock().map_err(|error| {
            BackendError::new(
                K::BackendUnavailable,
                S::Backend,
                "CUDA runtime latch is poisoned",
            )
            .with_external_cause(CauseCode::RuntimeInitialize, &error)
        })?;
        if let Some(latch) = guard.as_ref() {
            if !latch.owner.same_owner(&self.owner) || latch.loading_profile != self.loading_profile
            {
                return Err(BackendError::new(
                    K::IdentityMismatch,
                    S::Backend,
                    "CUDA audit differs from the process-global runtime bundle",
                ));
            }
            latch.result.as_ref().map_err(Clone::clone)?;
        }
        #[cfg(target_os = "linux")]
        if let RuntimeOwner::System { runtime, .. } = &self.owner {
            // The legacy observation schema describes the fixed bundle and
            // cannot truthfully represent a system installation.
            if observe {
                return Err(BackendError::new(
                    K::UnsupportedModel,
                    S::Backend,
                    "legacy CUDA bundle observation does not represent system runtime mappings",
                ));
            }
            let observed = if strict && include_runtime {
                runtime.verify_mappings()
            } else {
                runtime.observe_mappings()
            };
            if let Err(error) = &observed {
                if matches!(
                    error.kind,
                    rz_native_loader::LoadErrorKind::ObservationUnavailable
                        | rz_native_loader::LoadErrorKind::ForeignMapping
                ) {
                    // Missing evidence or an extra unrelated dependency can
                    // fail strict proof without poisoning normal execution.
                    return Err(loader_error(error.clone()));
                }
            }
            let result = observed.map_err(loader_error).and_then(|observation| {
                classify_system_mapping(&observation, include_runtime)
                    .map(|evidence| (None, evidence))
            });
            if let Err(failure) = &result {
                if let Some(latch) = guard.as_mut() {
                    if latch.owner.same_owner(&self.owner) && latch.result.is_ok() {
                        latch.result = Err(failure.clone());
                    }
                }
            }
            return result;
        }
        let libraries = self.cuda_libraries.as_ref().ok_or_else(|| {
            BackendError::new(
                K::IdentityMismatch,
                S::Backend,
                "CUDA requires a pinned runtime bundle",
            )
        })?;
        // Normal verification does not allocate receipt/name vectors on the
        // existing per-Run path. Only explicit acceptance observations do so.
        let verified = libraries
            .verify_mappings()
            .and_then(|()| {
                if include_runtime {
                    // All three ORT images become mandatory after the actual
                    // CUDA probe. Keep the existing separate NVIDIA audit
                    // before this full runtime-origin audit.
                    let ort = self
                        .ort_libraries
                        .as_ref()
                        .expect("CUDA runtime owns its three pinned ORT images");
                    if observe {
                        libraries.runtime_mapping_observation(ort).map(Some)
                    } else {
                        libraries.verify_runtime_mappings(ort).map(|()| None)
                    }
                } else if observe {
                    libraries.mapping_observation().map(Some)
                } else {
                    Ok(None)
                }
            })
            .map_err(loader_error);
        if let Err(failure) = &verified {
            // An origin audit cannot be retried into success with the same
            // process-global ORT session. Keep the first typed source and pins.
            if let Some(latch) = guard.as_mut() {
                if latch.owner.same_owner(&self.owner)
                    && latch.loading_profile == self.loading_profile
                    && latch.result.is_ok()
                {
                    latch.result = Err(failure.clone());
                }
            }
        }
        verified.map(|observation| (observation, CudaRuntimeMappingEvidence::Observed))
    }
}

#[cfg(target_os = "linux")]
fn classify_system_mapping(
    observation: &rz_native_loader::system::SystemMappingObservation,
    include_runtime: bool,
) -> Result<CudaRuntimeMappingEvidence, BackendError> {
    // A missing required role is incomplete evidence, not an observed
    // identity conflict. SystemRuntime::observe_mappings already reports
    // actual path/inode/lifetime conflicts as errors before this projection.
    if observation.availability == SystemMappingAvailability::Observed
        && observation
            .mapped_roles
            .iter()
            .any(|role| role == "ort-core")
        && (!include_runtime || observation.missing_required_roles.is_empty())
        && observation.preexisting_mapping_observed
        && observation.preexisting_native_mappings.is_empty()
        && observation.unregistered_native_mappings.is_empty()
    {
        Ok(CudaRuntimeMappingEvidence::Observed)
    } else {
        Ok(CudaRuntimeMappingEvidence::Unknown)
    }
}

#[cfg(all(test, target_os = "linux"))]
mod system_mapping_evidence_tests {
    use super::*;
    use rz_native_loader::system::SystemMappingObservation;

    #[test]
    fn unavailable_and_missing_roles_remain_unknown() {
        let mut observation = SystemMappingObservation {
            identity_sha256: [0; 32],
            availability: SystemMappingAvailability::Unavailable,
            preexisting_mapping_observed: false,
            preexisting_native_mappings: vec![],
            mapped_roles: vec![],
            missing_required_roles: vec![],
            missing_optional_roles: vec![],
            unregistered_native_mappings: vec![],
            gpu_provider_verified: false,
            gpu_completion_verified: false,
        };
        assert_eq!(
            classify_system_mapping(&observation, true).unwrap(),
            CudaRuntimeMappingEvidence::Unknown
        );
        observation.availability = SystemMappingAvailability::Observed;
        // An observed map that does not yet show ORT core is incomplete; an
        // identity conflict is reported separately by the system loader.
        observation.preexisting_mapping_observed = true;
        assert_eq!(
            classify_system_mapping(&observation, false).unwrap(),
            CudaRuntimeMappingEvidence::Unknown
        );
        observation.mapped_roles.push("ort-core".into());
        observation.preexisting_mapping_observed = false;
        assert_eq!(
            classify_system_mapping(&observation, true).unwrap(),
            CudaRuntimeMappingEvidence::Unknown
        );
        observation.preexisting_mapping_observed = true;
        assert_eq!(
            classify_system_mapping(&observation, true).unwrap(),
            CudaRuntimeMappingEvidence::Observed
        );
        observation
            .unregistered_native_mappings
            .push("foreign.so".into());
        assert_eq!(
            classify_system_mapping(&observation, true).unwrap(),
            CudaRuntimeMappingEvidence::Unknown
        );
        observation.unregistered_native_mappings.clear();
        observation
            .missing_required_roles
            .push("ort-provider-cuda".into());
        assert_eq!(
            classify_system_mapping(&observation, true).unwrap(),
            CudaRuntimeMappingEvidence::Unknown
        );
    }
}

fn cuda_loading_profile(
    bundle: Option<[u8; 32]>,
    requested: NativeLoadingProfile,
) -> Result<Option<NativeLoadingProfile>, BackendError> {
    if bundle.is_some() {
        Ok(Some(requested))
    } else if requested == NativeLoadingProfile::EagerCuda12Cudnn9V1 {
        Ok(None)
    } else {
        Err(BackendError::new(
            K::IdentityMismatch,
            S::Backend,
            "experimental native loading profile requires a complete CUDA bundle",
        ))
    }
}

#[cfg(test)]
mod native_loading_profile_tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn checked_system_entry_distinguishes_rejection_panic_and_entered_failure() {
        let entered = std::cell::Cell::new(false);
        let called = std::cell::Cell::new(false);
        let rejected = with_checked_system_ort_entry(
            Some("/not-opened"),
            || Err(BackendError::new(K::InvalidInput, S::Admission, "denied")),
            &entered,
            |_| {
                called.set(true);
                Ok(())
            },
        );
        assert!(rejected.is_err());
        assert!(!entered.get() && !called.get());
        assert!(!should_latch_system_attempt(entered.get(), false));
        assert!(should_latch_system_attempt(entered.get(), true));

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = with_checked_system_ort_entry(
                Some("/not-opened"),
                || panic!("admission panic"),
                &entered,
                |_| {
                    called.set(true);
                    Ok(())
                },
            );
        }));
        assert!(panic.is_err());
        assert!(!entered.get() && !called.get());
        assert!(!should_latch_system_attempt(entered.get(), false));

        let native_error = with_checked_system_ort_entry(
            Some("/not-opened"),
            || Ok(()),
            &entered,
            |_| {
                Err::<(), _>(BackendError::new(
                    K::BackendFailure,
                    S::Backend,
                    "native failure",
                ))
            },
        );
        assert!(native_error.is_err() && entered.get());
        assert!(should_latch_system_attempt(entered.get(), false));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn system_ort_entry_rejects_bad_path_before_native_callback() {
        let entered = std::cell::Cell::new(false);
        let called = std::cell::Cell::new(false);
        let result = with_system_ort_entry(
            None,
            || entered.set(true),
            |_| {
                called.set(true);
                Ok(())
            },
        );
        assert_eq!(result.unwrap_err().kind, K::BackendUnavailable);
        assert!(!entered.get());
        assert!(!called.get());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn system_ort_entry_runs_owner_hook_before_native_invocation() {
        let entered = std::cell::Cell::new(false);
        let output = with_system_ort_entry(
            Some("/not-opened-in-this-test"),
            || entered.set(true),
            |path| {
                assert!(entered.get());
                Ok(path.len())
            },
        )
        .unwrap();
        assert_eq!(output, "/not-opened-in-this-test".len());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn system_ort_entry_hook_panic_does_not_call_initializer() {
        let called = std::cell::Cell::new(false);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = with_system_ort_entry(
                Some("/not-opened-in-this-test"),
                || panic!("owner hook failure"),
                |_| {
                    called.set(true);
                    Ok(())
                },
            );
        }));
        assert!(panic.is_err());
        assert!(!called.get());
    }

    #[test]
    fn cpu_bootstrap_keeps_no_native_profile_and_rejects_explicit_shim() {
        assert_eq!(
            cuda_loading_profile(None, NativeLoadingProfile::EagerCuda12Cudnn9V1).unwrap(),
            None
        );
        assert!(cuda_loading_profile(None, NativeLoadingProfile::CuDnnShimLazyV1).is_err());
        assert_eq!(
            cuda_loading_profile(Some([7; 32]), NativeLoadingProfile::CuDnnShimLazyV1).unwrap(),
            Some(NativeLoadingProfile::CuDnnShimLazyV1)
        );
        assert_ne!(
            asset::sha256(
                NativeLoadingProfile::EagerCuda12Cudnn9V1
                    .canonical_descriptor()
                    .as_bytes()
            ),
            asset::sha256(
                NativeLoadingProfile::CuDnnShimLazyV1
                    .canonical_descriptor()
                    .as_bytes()
            )
        );
    }
}

fn loader_error(error: LoadError) -> BackendError {
    let failure = BackendError::new(
        K::BackendUnavailable,
        S::Backend,
        "pinned CUDA dependency loading or mapping audit failed",
    );
    let mut failure = if let Some(message) = error.diagnostic() {
        failure
            .with_external_cause(CauseCode::RuntimeLibraryLoad, &message)
            .with_diagnostic(error.cause_code(), message)
    } else {
        failure
            .with_external_cause(CauseCode::RuntimeLibraryLoad, &error)
            .with_diagnostic(error.cause_code(), error.detail)
    };
    // The loader has already bounded its original native text. Preserve that
    // truncation even when its 1024-byte prefix fits C's second boundary exactly.
    if error.diagnostic_truncated() {
        if let Some(cause) = failure.cause.as_mut() {
            cause.truncated = true;
        }
        if let Some(native) = failure.native.as_mut() {
            native.truncated = true;
        }
    }
    failure
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Provider {
    Cpu,
    /// The arena cap is NOT a total VRAM cap. External process/device monitoring
    /// must account for cuDNN workspace, drivers and other resident sessions.
    Cuda {
        device_id: i32,
        arena_bytes: usize,
    },
}

#[derive(Clone, Debug)]
pub struct BackendConfig {
    pub experiments: ExecutionExperiments,
    pub provider: Provider,
    pub max_batch: usize,
    pub host_io_bytes: usize,
    pub intra_threads: usize,
    /// Required for CUDA's one-shot actual-node-placement probe. Kept on disk.
    pub profiling_prefix: Option<PathBuf>,
}

impl BackendConfig {
    /// Native D02 v1 has no raw-hit, batch or experimental I/O timeline schema.
    pub fn source_profile_supported(&self) -> bool {
        self.max_batch == 1 && self.experiments == ExecutionExperiments::default()
    }

    pub fn cpu() -> Self {
        Self {
            experiments: ExecutionExperiments::default(),
            provider: Provider::Cpu,
            max_batch: 16,
            host_io_bytes: 16 * IO_BYTES_PER_ITEM,
            intra_threads: 1,
            profiling_prefix: None,
        }
    }

    pub fn validate(&self) -> Result<(), BackendError> {
        if (self.experiments.reuse_buffers && !cfg!(feature = "experimental-io-buffers"))
            || (self.experiments.disable_cuda_cpu_arena
                && (!cfg!(feature = "experimental-ort-cpu-arena")
                    || !matches!(self.provider, Provider::Cuda { .. })))
            || (self.experiments.uses_ort_model()
                && (!cfg!(feature = "experimental-ort-model")
                    || !matches!(self.provider, Provider::Cuda { .. })
                    || (self.experiments.zero_copy_ort && self.experiments.copy_ort_model)
                    || self.experiments.io_binding
                    || self.experiments.cuda_graph
                    || self.experiments.reuse_buffers
                    || self.experiments.disable_cuda_cpu_arena))
            || (self.experiments.io_binding && !cfg!(feature = "experimental-io-binding"))
            || (self.experiments.cuda_graph && !cfg!(feature = "experimental-cuda-graph"))
            || (self.experiments.cuda_graph
                && (!self.experiments.io_binding
                    || !matches!(self.provider, Provider::Cuda { .. })
                    || self.max_batch != 1))
        {
            return Err(BackendError::new(
                K::UnsupportedModel,
                S::Admission,
                "execution experiment unsupported or CUDA Graph lacks fixed CUDA B1 binding",
            ));
        }
        if self.experiments.cuda_graph {
            // The pinned ORT 1.22.0/cuDNN profile passed numeric checks, then
            // aborted in a native exit handler (also reproduced under gdb).
            // A Run/output fence cannot attest process teardown. Keep this
            // failed experiment unavailable rather than hiding it by retaining
            // sessions forever, exiting without destructors, or disabling Graph.
            return Err(BackendError::new(
                K::BackendUnavailable,
                S::Admission,
                "CUDA Graph blocked: pinned runtime failed normal native-exit acceptance",
            ));
        }
        if self.max_batch == 0
            || self.max_batch > MAX_BATCH
            || self.intra_threads == 0
            || self.intra_threads > 4
            || self.host_io_bytes < self.max_batch * IO_BYTES_PER_ITEM
        {
            return Err(BackendError::new(
                K::ResourceExhausted,
                S::Admission,
                "invalid finite batch/thread/IO budget",
            ));
        }
        if let Provider::Cuda {
            device_id,
            arena_bytes,
        } = self.provider
        {
            if device_id < 0
                || arena_bytes == 0
                || !self
                    .profiling_prefix
                    .as_ref()
                    .is_some_and(|prefix| prefix.is_absolute() && prefix.file_name().is_some())
            {
                return Err(BackendError::new(
                    K::InvalidInput,
                    S::Admission,
                    "CUDA requires device, arena cap and absolute profile prefix",
                ));
            }
        } else if self.profiling_prefix.is_some() {
            return Err(BackendError::new(
                K::InvalidInput,
                S::Admission,
                "profiling is limited to the one-shot CUDA placement probe",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct CudaEvidence {
    pub profile_path: PathBuf,
    pub profile_sha256: [u8; 32],
    pub executed_cuda_nodes: usize,
}

/// Host startup intervals, separate from numerical output and provider claims.
#[derive(Clone, Copy, Debug, Default)]
pub struct StartupTimings {
    pub backend_setup: std::time::Duration,
    pub model_commit: std::time::Duration,
    pub cuda_probe_validation: Option<std::time::Duration>,
}

/// Synchronous startup boundaries for opt-in, bounded host-memory research.
/// These are observation points, not phase peaks or GPU completion fences.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StartupPhase {
    SerializedModelReady,
    SessionCommittedBeforeSerializedRelease,
    SerializedModelReleased,
    ProviderReady,
    OrtModelRetained,
    SessionCommittedWithOrtModelRetained,
}

// Native must be destroyed before the bytes it references. Keeping both in one
// private owner also pins the model on early errors, Run quarantine and unwinds.
// No native session handle can escape this owner. Rust drops fields in order.
struct SessionStorage<N = Session, M = Vec<u8>> {
    native: N,
    retained_model: Option<M>,
}
impl std::ops::Deref for SessionStorage {
    type Target = Session;
    fn deref(&self) -> &Session {
        &self.native
    }
}
impl std::ops::DerefMut for SessionStorage {
    fn deref_mut(&mut self) -> &mut Session {
        &mut self.native
    }
}

enum VerifiedModel<'a> {
    BorrowedOnnx(&'a [u8]),
    OwnedOnnx(Vec<u8>),
    #[cfg(feature = "experimental-ort-model")]
    OwnedOrt {
        bytes: Vec<u8>,
        manifest_digest: [u8; 32],
    },
}
impl AsRef<[u8]> for VerifiedModel<'_> {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::BorrowedOnnx(bytes) => bytes,
            Self::OwnedOnnx(bytes) => bytes,
            #[cfg(feature = "experimental-ort-model")]
            Self::OwnedOrt { bytes, .. } => bytes,
        }
    }
}

/// Status of this exact opt-in startup, not a device-wide completion claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Lc0StartupState {
    BeforeSessionEntry,
    NativeEnteredUnknown,
    Ready,
    Transferred,
}

#[derive(Default)]
struct Lc0StartupSlots {
    session: Option<SessionStorage>,
    backend: Option<OnnxBackend>,
}

fn enter_lc0_session(
    state: &mut Lc0StartupState,
    before_native: &mut impl FnMut() -> Result<(), BackendError>,
) -> Result<(), BackendError> {
    before_native()?;
    *state = Lc0StartupState::NativeEnteredUnknown;
    Ok(())
}

fn take_ready_pair<A, B>(
    state: &mut Lc0StartupState,
    metadata: &mut Option<A>,
    backend: &mut Option<B>,
) -> Option<(A, B)> {
    if *state != Lc0StartupState::Ready || metadata.is_none() || backend.is_none() {
        return None;
    }
    let metadata = metadata.take().expect("checked metadata");
    let backend = backend.take().expect("checked backend");
    *state = Lc0StartupState::Transferred;
    Some((metadata, backend))
}

/// Caller-owned custody for one LC0 startup attempt.  The original serialized
/// model, runtime and any session/backend returned by ORT stay in this owner on
/// errors and unwinds.  An entered but failed attempt is not a completion
/// fence, including when ORT never returned a partial session to Rust.
pub struct Lc0StartupCustody {
    runtime: Option<OrtRuntime>,
    metadata: Option<AssetMetadata>,
    serialized: Option<Vec<u8>>,
    slots: Lc0StartupSlots,
    state: Lc0StartupState,
    attempted: bool,
}

impl Lc0StartupCustody {
    pub fn new(runtime: OrtRuntime, asset: MaiaAsset) -> Self {
        let (metadata, bytes) = asset.into_parts();
        Self {
            runtime: Some(runtime),
            metadata: Some(metadata),
            serialized: Some(bytes),
            slots: Lc0StartupSlots::default(),
            state: Lc0StartupState::BeforeSessionEntry,
            attempted: false,
        }
    }
    pub fn state(&self) -> Lc0StartupState {
        self.state
    }
    pub fn metadata(&self) -> Option<&AssetMetadata> {
        self.metadata.as_ref()
    }

    /// Callback is invoked immediately before the first session-builder entry,
    /// after admission. It must enforce the caller's original parent scope.
    pub fn load(
        &mut self,
        config: BackendConfig,
        mut before_native: impl FnMut() -> Result<(), BackendError>,
    ) -> Result<(), BackendError> {
        if self.attempted {
            return Err(BackendError::new(
                K::InvalidInput,
                S::Admission,
                "LC0 startup owner has already been used",
            ));
        }
        self.attempted = true;
        let runtime = self.runtime.as_ref().expect("original runtime");
        let bytes = self.serialized.as_deref().expect("original model");
        let state = &mut self.state;
        let mut entry = || enter_lc0_session(state, &mut before_native);
        let backend = OnnxBackend::load_model(
            runtime,
            self.metadata.as_ref().expect("original metadata"),
            VerifiedModel::BorrowedOnnx(bytes),
            config,
            &mut |_| {},
            None,
            Some(&mut self.slots),
            Some(&mut entry),
        )?;
        self.slots.backend = Some(backend);
        self.serialized = None;
        self.state = Lc0StartupState::Ready;
        Ok(())
    }
    pub fn take_backend(&mut self) -> Option<OnnxBackend> {
        if self.state != Lc0StartupState::Ready {
            return None;
        }
        let backend = self.slots.backend.take()?;
        self.state = Lc0StartupState::Transferred;
        Some(backend)
    }

    /// Transfer the exact original metadata and completed backend together.
    /// A failed or incomplete startup leaves every custody field untouched.
    pub fn take_ready(&mut self) -> Option<(AssetMetadata, OnnxBackend)> {
        take_ready_pair(&mut self.state, &mut self.metadata, &mut self.slots.backend)
    }
}

impl Drop for Lc0StartupCustody {
    fn drop(&mut self) {
        if self.state == Lc0StartupState::NativeEnteredUnknown {
            // No destructor or owner disconnect can manufacture a fence.
            std::mem::forget((
                self.runtime.take(),
                self.serialized.take(),
                self.slots.session.take(),
                self.slots.backend.take(),
            ));
        }
    }
}

struct SynchronousRunTracker {
    clean: bool,
    completed: u64,
}
impl SynchronousRunTracker {
    fn new() -> Self {
        Self {
            clean: true,
            completed: 0,
        }
    }
    fn enter(&mut self) -> bool {
        std::mem::replace(&mut self.clean, false)
    }
    fn returned(&mut self, was_clean: bool, success: bool) {
        let next = if success {
            self.completed.checked_add(1)
        } else {
            None
        };
        if let Some(count) = next {
            self.completed = count;
        }
        self.clean = was_clean && next.is_some();
    }
}

pub struct OnnxBackend {
    session: Option<SessionStorage>,
    // Kept outside the Run stack so worker quarantine also pins tensor storage
    // if a Rust wrapper unexpectedly unwinds before attesting completion.
    active_input: Option<Tensor<f32>>,
    #[cfg(feature = "experimental-io-buffers")]
    spare_input: Option<Tensor<f32>>,
    raw_pool: Vec<RawOutput>,
    #[cfg(feature = "experimental-io-binding")]
    bound: Option<io::BoundBuffers>,
    /// Successful synchronous binding runs. This is not proof of GPU replay.
    bound_runs: u64,
    timings: Option<IoTimings>,
    startup_timings: StartupTimings,
    config: BackendConfig,
    identity: [u8; 32],
    asset_identity: [u8; 32],
    asset_profile: AssetProfile,
    cuda_evidence: Option<CudaEvidence>,
    quarantine_cause: Option<BackendError>,
    // The loaded CUDA session can re-audit its actual nineteen-library maps
    // without accepting a caller-selected runtime or private mutable pathname.
    cuda_runtime: Option<OrtRuntime>,
    synchronous_identity: Arc<()>,
    synchronous_runs: SynchronousRunTracker,
    startup_complete: bool,
}

impl Drop for OnnxBackend {
    fn drop(&mut self) {
        if matches!(self.config.provider, Provider::Cuda { .. })
            && (self.quarantine_cause.is_some() || self.active_input.is_some())
        {
            // A returned error or wrapper unwind is not a CUDA fence. The
            // process owns these native allocations until exit, even when a
            // direct caller drops the backend outside SingleWorker.
            std::mem::forget((self.session.take(), self.active_input.take()));
            #[cfg(feature = "experimental-io-binding")]
            std::mem::forget(self.bound.take());
        }
    }
}

impl OnnxBackend {
    pub fn load(
        runtime: &OrtRuntime,
        asset: &MaiaAsset,
        config: BackendConfig,
    ) -> Result<Self, BackendError> {
        Self::load_model(
            runtime,
            asset.metadata(),
            VerifiedModel::BorrowedOnnx(asset.onnx_bytes()),
            config,
            &mut |_| {},
            None,
            None,
            None,
        )
    }

    /// Consumes the verified serialized bytes, releasing that host allocation
    /// immediately after ORT creates its session, before CUDA warm placement.
    /// Metadata remains typed; a consumed model cannot masquerade as empty bytes.
    pub fn load_owned(
        runtime: &OrtRuntime,
        asset: MaiaAsset,
        config: BackendConfig,
    ) -> Result<(AssetMetadata, Self), BackendError> {
        Self::load_owned_observed(runtime, asset, config, |_| {})
    }

    /// Opt-in startup observations only. The caller receives no model storage,
    /// tensors or session handles. The normal constructor uses a no-op observer.
    pub fn load_owned_observed(
        runtime: &OrtRuntime,
        asset: MaiaAsset,
        config: BackendConfig,
        mut observe: impl FnMut(StartupPhase),
    ) -> Result<(AssetMetadata, Self), BackendError> {
        let (metadata, bytes) = asset.into_parts();
        let backend = Self::load_model(
            runtime,
            &metadata,
            VerifiedModel::OwnedOnnx(bytes),
            config,
            &mut observe,
            None,
            None,
            None,
        )?;
        Ok((metadata, backend))
    }

    #[cfg(feature = "experimental-ort-model")]
    pub fn load_owned_ort_observed(
        runtime: &OrtRuntime,
        asset: crate::ort_model::OwnedOrtAsset,
        config: BackendConfig,
        mut observe: impl FnMut(StartupPhase),
    ) -> Result<(AssetMetadata, Self), BackendError> {
        if !config.experiments.uses_ort_model()
            || asset::parse_sha256(&asset.manifest.runtime_core_sha256)? != runtime.binary_digest()
            || Some(asset::parse_sha256(&asset.manifest.runtime_bundle_sha256)?)
                != runtime.bundle_digest()
        {
            return Err(BackendError::new(
                K::IdentityMismatch,
                S::Admission,
                "derived ORT requires explicit mode and its exact export runtime",
            ));
        }
        let backend = Self::load_model(
            runtime,
            &asset.source,
            VerifiedModel::OwnedOrt {
                bytes: asset.bytes,
                manifest_digest: asset.manifest_digest,
            },
            config,
            &mut observe,
            None,
            None,
            None,
        )?;
        Ok((asset.source, backend))
    }

    /// Separate conversion, never part of a benchmark's measured workload.
    /// The caller owns a fresh regular output FD; ORT writes that exact inode.
    #[cfg(all(feature = "experimental-ort-model", target_os = "linux"))]
    pub fn export_owned_ort(
        runtime: &OrtRuntime,
        asset: MaiaAsset,
        config: BackendConfig,
        output: &std::fs::File,
    ) -> Result<(AssetMetadata, Self), BackendError> {
        let metadata = output
            .metadata()
            .map_err(|_| BackendError::new(K::Io, S::Asset, "cannot inspect ORT export FD"))?;
        if !metadata.is_file()
            || metadata.len() != 0
            || !matches!(config.provider, Provider::Cuda { .. })
            || config.experiments != ExecutionExperiments::default()
            || asset.profile() != AssetProfile::Bt4It332
        {
            return Err(BackendError::new(
                K::InvalidInput,
                S::Admission,
                "ORT export requires fresh FD, BT4 and baseline CUDA",
            ));
        }
        let (metadata, bytes) = asset.into_parts();
        let backend = Self::load_model(
            runtime,
            &metadata,
            VerifiedModel::OwnedOnnx(bytes),
            config,
            &mut |_| {},
            Some(output),
            None,
            None,
        )?;
        Ok((metadata, backend))
    }

    // The optional hooks carry distinct startup ownership and observation
    // roles; keeping them explicit preserves the existing call boundaries.
    #[allow(clippy::too_many_arguments)]
    fn load_model(
        runtime: &OrtRuntime,
        asset: &AssetMetadata,
        model: VerifiedModel<'_>,
        config: BackendConfig,
        observe: &mut impl FnMut(StartupPhase),
        export: Option<&std::fs::File>,
        custody: Option<&mut Lc0StartupSlots>,
        before_session: Option<&mut dyn FnMut() -> Result<(), BackendError>>,
    ) -> Result<Self, BackendError> {
        let setup_started = Instant::now();
        config.validate()?;
        let is_ort = match &model {
            #[cfg(feature = "experimental-ort-model")]
            VerifiedModel::OwnedOrt { .. } => true,
            _ => false,
        };
        if config.experiments.uses_ort_model() != is_ort {
            return Err(BackendError::new(
                K::InvalidInput,
                S::Admission,
                "model storage and explicit ORT mode differ",
            ));
        }
        match (config.provider, runtime.cuda_runtime_identity()) {
            (Provider::Cuda { .. }, None) => {
                return Err(BackendError::new(
                    K::IdentityMismatch,
                    S::Backend,
                    "CUDA requires an explicitly captured CUDA runtime; CPU-only pin is not sufficient",
                ));
            }
            (Provider::Cpu, Some(_)) => {
                return Err(BackendError::new(
                    K::IdentityMismatch,
                    S::Backend,
                    "CUDA runtime bundle cannot be admitted as a CPU runtime",
                ));
            }
            _ => {}
        }
        if matches!(config.provider, Provider::Cuda { .. }) {
            runtime.verify_cuda_mappings(false)?;
        }
        let setup_error = |code, error: ort::Error| {
            BackendError::new(
                K::BackendUnavailable,
                S::Backend,
                "ORT session/provider setup failed",
            )
            .with_ort_cause(code, error)
        };
        if let Some(before_session) = before_session {
            before_session()?;
        }
        let mut builder = Session::builder()
            .map_err(|error| setup_error(CauseCode::SessionBuilder, error))?
            .with_no_environment_execution_providers()
            .map_err(|error| setup_error(CauseCode::SessionConfiguration, error))?
            .with_intra_threads(config.intra_threads)
            .map_err(|error| setup_error(CauseCode::SessionConfiguration, error))?
            .with_inter_threads(1)
            .map_err(|error| setup_error(CauseCode::SessionConfiguration, error))?
            .with_parallel_execution(false)
            .map_err(|error| setup_error(CauseCode::SessionConfiguration, error))?
            .with_memory_pattern(false)
            .map_err(|error| setup_error(CauseCode::SessionConfiguration, error))?
            .with_intra_op_spinning(false)
            .map_err(|error| setup_error(CauseCode::SessionConfiguration, error))?
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .map_err(|error| setup_error(CauseCode::SessionConfiguration, error))?;
        match config.provider {
            Provider::Cpu => {
                builder = builder
                    .with_execution_providers([CPUExecutionProvider::default()
                        .with_arena_allocator(false)
                        .build()
                        .error_on_failure()])
                    .map_err(|error| setup_error(CauseCode::ProviderRegistration, error))?;
            }
            Provider::Cuda {
                device_id,
                arena_bytes,
            } => {
                if config.experiments.disable_cuda_cpu_arena {
                    // In ort rc.10 this changes DisableCpuMemArena on the session
                    // options; it does not append an explicit CPU provider. ORT
                    // 1.22 still creates its implicit CPU allocator, while the
                    // disable_cpu_ep_fallback check and CUDA placement remain.
                    builder = builder
                        .with_execution_providers([CPUExecutionProvider::default()
                            .with_arena_allocator(false)
                            .build()
                            .error_on_failure()])
                        .map_err(|error| setup_error(CauseCode::SessionConfiguration, error))?;
                }
                let cuda = CUDAExecutionProvider::default();
                if !cuda
                    .is_available()
                    .map_err(|error| setup_error(CauseCode::ProviderQuery, error))?
                {
                    return Err(BackendError::new(
                        K::BackendUnavailable,
                        S::Backend,
                        "CUDA EP is unavailable; CPU fallback is forbidden",
                    ));
                }
                builder = builder
                    .with_config_entry("session.disable_cpu_ep_fallback", "1")
                    .map_err(|error| setup_error(CauseCode::SessionConfiguration, error))?
                    .with_execution_providers([cuda
                        .with_device_id(device_id)
                        .with_memory_limit(arena_bytes)
                        // Avoid power-of-two pool growth crowding another resident
                        // engine. This changes allocation, not tensors or kernels.
                        .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
                        .with_tf32(false)
                        .with_conv_max_workspace(false)
                        .with_cuda_graph(config.experiments.cuda_graph)
                        .build()
                        .error_on_failure()])
                    .map_err(|error| setup_error(CauseCode::ProviderRegistration, error))?;
            }
        }
        if let Some(prefix) = &config.profiling_prefix {
            builder = builder
                .with_profiling(prefix)
                .map_err(|error| setup_error(CauseCode::ProfilingStart, error))?;
        }
        if let Some(_output) = export {
            #[cfg(all(feature = "experimental-ort-model", target_os = "linux"))]
            {
                use std::os::fd::AsRawFd;
                builder = builder
                    .with_config_entry("session.save_model_format", "ORT")
                    .map_err(|error| setup_error(CauseCode::SessionConfiguration, error))?
                    .with_optimized_model_path(format!("/proc/self/fd/{}", _output.as_raw_fd()))
                    .map_err(|error| setup_error(CauseCode::SessionConfiguration, error))?;
            }
            #[cfg(not(all(feature = "experimental-ort-model", target_os = "linux")))]
            return Err(BackendError::new(
                K::UnsupportedModel,
                S::Admission,
                "ORT export requires its feature on Linux",
            ));
        }
        let commit_started = Instant::now();
        let backend_setup = commit_started.duration_since(setup_started);
        let model_error = |error| {
            BackendError::new(
                K::BackendUnavailable,
                S::Backend,
                "ORT could not load the verified model with the requested provider",
            )
            .with_ort_cause(CauseCode::ModelLoad, error)
        };
        let (session, derived_manifest): (_, Option<[u8; 32]>) = match model {
            #[cfg(feature = "experimental-ort-model")]
            VerifiedModel::OwnedOrt {
                bytes,
                manifest_digest,
            } => {
                let direct = config.experiments.zero_copy_ort;
                builder = builder
                    .with_config_entry("session.load_model_format", "ORT")
                    .map_err(|error| setup_error(CauseCode::SessionConfiguration, error))?
                    .with_config_entry(
                        "session.use_ort_model_bytes_directly",
                        if direct { "1" } else { "0" },
                    )
                    .map_err(|error| setup_error(CauseCode::SessionConfiguration, error))?
                    .with_config_entry(
                        "session.use_ort_model_bytes_for_initializers",
                        if direct { "1" } else { "0" },
                    )
                    .map_err(|error| setup_error(CauseCode::SessionConfiguration, error))?;
                let storage = if direct {
                    // Same two documented options as ort rc.10's borrowed
                    // InMemorySession. Vec stays private and immutable until
                    // after native destruction, including errors and quarantine.
                    observe(StartupPhase::OrtModelRetained);
                    let native = builder.commit_from_memory(&bytes).map_err(model_error)?;
                    let storage = SessionStorage {
                        native,
                        retained_model: Some(bytes),
                    };
                    observe(StartupPhase::SessionCommittedWithOrtModelRetained);
                    storage
                } else {
                    // Both direct-reference options are explicitly off. Normal
                    // commit_from_memory owns its native copy; reuse the existing
                    // success/error buffer-release boundary, never shorten the
                    // lifetime of a direct-reference session's backing bytes.
                    SessionStorage {
                        native: commit_verified_model(
                            bytes,
                            |bytes| builder.commit_from_memory(bytes).map_err(model_error),
                            observe,
                        )?,
                        retained_model: None,
                    }
                };
                (storage, Some(manifest_digest))
            }
            model => (
                SessionStorage {
                    native: commit_verified_model(
                        model,
                        |bytes| builder.commit_from_memory(bytes).map_err(model_error),
                        observe,
                    )?,
                    retained_model: None,
                },
                None,
            ),
        };
        let model_commit = commit_started.elapsed();
        let mut local_slots = Lc0StartupSlots::default();
        let slots = custody.unwrap_or(&mut local_slots);
        slots.session = Some(session);
        validate_interface(
            slots.session.as_ref().expect("returned session"),
            asset.profile(),
        )?;
        // This is a versioned C backend identity, not a new global wire codec.
        let mut identity = declared_backend_identity(
            runtime.binary_digest(),
            asset.manifest_digest(),
            runtime.bundle_digest(),
            &config,
        );
        if runtime.is_system_cuda() {
            let mut identified = b"rz-maia-system-runtime-backend-v1\0".to_vec();
            identified.extend_from_slice(&identity);
            identified.extend_from_slice(
                &runtime
                    .cuda_runtime_identity()
                    .expect("system CUDA identity"),
            );
            identity = asset::sha256(&identified);
        }
        if let Some(digest) = derived_manifest {
            let mut identified = b"rz-derived-ort-backend-v1\0".to_vec();
            identified.extend_from_slice(&identity);
            identified.extend_from_slice(&digest);
            identity = asset::sha256(&identified);
        }
        let cuda_runtime =
            matches!(config.provider, Provider::Cuda { .. }).then(|| runtime.clone());
        slots.backend = Some(Self {
            session: slots.session.take(),
            active_input: None,
            #[cfg(feature = "experimental-io-buffers")]
            spare_input: None,
            raw_pool: Vec::new(),
            #[cfg(feature = "experimental-io-binding")]
            bound: None,
            bound_runs: 0,
            timings: None,
            startup_timings: StartupTimings {
                backend_setup,
                model_commit,
                cuda_probe_validation: None,
            },
            config,
            identity,
            asset_identity: asset.manifest_digest(),
            asset_profile: asset.profile(),
            cuda_evidence: None,
            quarantine_cause: None,
            cuda_runtime,
            synchronous_identity: Arc::new(()),
            synchronous_runs: SynchronousRunTracker::new(),
            startup_complete: false,
        });
        let result = slots.backend.as_mut().expect("installed LC0 backend");
        if matches!(result.config.provider, Provider::Cuda { .. }) {
            let probe_started = Instant::now();
            runtime.verify_cuda_mappings(false)?;
            // A registered provider alone proves nothing. Synchronous physical
            // execution plus ORT kernel placement is required before returning.
            let probe = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                result.run_values(&[&vec![0.0; INPUT_VALUES]])
            }));
            match probe {
                Ok(completed) => {
                    completed?;
                }
                Err(payload) => {
                    // Bootstrap has no runtime lease yet. Retain this session and
                    // active tensor if an unexpected unwind leaves completion unknown.
                    let error = BackendError::new(
                        K::BackendFailure,
                        S::Backend,
                        "CUDA probe unwound; session quarantined until process exit",
                    );
                    let message = payload
                        .downcast_ref::<&str>()
                        .copied()
                        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                        .unwrap_or("non-string CUDA probe panic payload");
                    let error = error
                        .with_external_cause(CauseCode::RuntimePanic, &message)
                        .with_diagnostic("CudaProbePanic", message);
                    result.quarantine_cause = Some(error.clone());
                    return Err(error);
                }
            }
            runtime.verify_cuda_mappings(true)?;
            let path = result
                .session
                .as_mut()
                .ok_or_else(|| {
                    BackendError::new(
                        K::BackendFailure,
                        S::Backend,
                        "CUDA probe session is missing",
                    )
                })?
                .end_profiling()
                .map_err(|error| setup_error(CauseCode::ProfilingFinish, error))?;
            let profile_path = Path::new(&path);
            let profile_parent = result
                .config
                .profiling_prefix
                .as_ref()
                .and_then(|prefix| prefix.parent())
                .ok_or_else(|| {
                    BackendError::new(
                        K::InvalidInput,
                        S::Backend,
                        "CUDA profiling parent is missing",
                    )
                })?;
            let named = std::fs::symlink_metadata(profile_path).map_err(|error| {
                BackendError::new(K::Io, S::Backend, "cannot inspect CUDA placement profile")
                    .with_external_cause(CauseCode::ProfileParse, &error)
            })?;
            if !profile_path.is_absolute()
                || !named.is_file()
                || named.file_type().is_symlink()
                || profile_path
                    .parent()
                    .and_then(|parent| parent.canonicalize().ok())
                    != Some(profile_parent.canonicalize().map_err(|error| {
                        BackendError::new(
                            K::Io,
                            S::Backend,
                            "cannot verify CUDA placement profile owner",
                        )
                        .with_external_cause(CauseCode::ProfileParse, &error)
                    })?)
            {
                return Err(BackendError::new(
                    K::IdentityMismatch,
                    S::Backend,
                    "CUDA placement profile is outside its explicit namespace",
                ));
            }
            let bytes = asset::read_bounded(Path::new(&path), 4 * 1024 * 1024)?;
            let executed_cuda_nodes = verify_cuda_profile(&bytes)?;
            result.cuda_evidence = Some(CudaEvidence {
                profile_path: path.into(),
                profile_sha256: asset::sha256(&bytes),
                executed_cuda_nodes,
            });
            result.startup_timings.cuda_probe_validation = Some(probe_started.elapsed());
        }
        observe(StartupPhase::ProviderReady);
        result.startup_complete = true;
        Ok(slots.backend.take().expect("completed LC0 backend"))
    }

    /// Owned model storage, not total ORT or GPU memory.
    pub fn retained_model_bytes(&self) -> usize {
        self.session
            .as_ref()
            .and_then(|s| s.retained_model.as_ref())
            .map_or(0, Vec::len)
    }

    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
    pub fn asset_identity(&self) -> [u8; 32] {
        self.asset_identity
    }
    pub fn asset_profile(&self) -> AssetProfile {
        self.asset_profile
    }
    pub fn config(&self) -> &BackendConfig {
        &self.config
    }
    pub fn cuda_evidence(&self) -> Option<&CudaEvidence> {
        self.cuda_evidence.as_ref()
    }

    /// Identity of this exact session owner; it is not a completion fence.
    pub fn synchronous_owner_identity(&self) -> Arc<()> {
        Arc::clone(&self.synchronous_identity)
    }

    /// Same-owner normal session release only. Zero completed Runs is not GPU
    /// execution evidence, and this proof is not a device-wide fence.
    pub fn synchronous_shutdown_proof(&self) -> Option<crate::worker::SynchronousOwnerProof> {
        if !self.startup_complete
            || !self.synchronous_runs.clean
            || self.session.is_none()
            || self.active_input.is_some()
            || self.quarantine_cause.is_some()
        {
            return None;
        }
        Some(crate::worker::SynchronousOwnerProof::new(
            Arc::clone(&self.synchronous_identity),
            self.synchronous_runs.completed,
        ))
    }

    /// Actual loaded runtime metadata, not a caller-issued provider claim.
    /// CPU identity and its single-library loading path remain unchanged.
    pub fn runtime_bundle_digest(&self) -> Option<[u8; 32]> {
        self.cuda_runtime
            .as_ref()
            .and_then(OrtRuntime::bundle_digest)
    }

    /// System-runtime identity, separate from historical bundled identity.
    /// This is not a provider-ready or physical-completion witness.
    pub fn system_cuda_runtime_identity(&self) -> Option<[u8; 32]> {
        self.cuda_runtime
            .as_ref()
            .filter(|runtime| runtime.is_system_cuda())
            .and_then(OrtRuntime::cuda_runtime_identity)
    }

    /// Re-audits the complete CUDA bundle owned by this loaded session. An
    /// uncertain Run remains quarantined and never becomes a successful audit.
    pub fn verify_cuda_runtime_mappings(&self) -> Result<(), BackendError> {
        self.check_cuda_mapping_lifetime()?;
        self.cuda_runtime
            .as_ref()
            .ok_or_else(|| {
                BackendError::new(
                    K::IdentityMismatch,
                    S::Backend,
                    "loaded session has no CUDA runtime bundle",
                )
            })?
            .verify_cuda_runtime_mappings()
    }

    /// Check known mapping conflicts for ordinary system-runtime operation.
    /// Unknown origin evidence remains distinct from a strict audit witness.
    pub fn observe_cuda_runtime_mappings(
        &self,
    ) -> Result<CudaRuntimeMappingEvidence, BackendError> {
        self.check_cuda_mapping_lifetime()?;
        self.cuda_runtime
            .as_ref()
            .ok_or_else(|| {
                BackendError::new(
                    K::IdentityMismatch,
                    S::Backend,
                    "loaded session has no CUDA runtime bundle",
                )
            })?
            .observe_cuda_runtime_mappings()
    }

    fn check_cuda_mapping_lifetime(&self) -> Result<(), BackendError> {
        if let Some(cause) = &self.quarantine_cause {
            return Err(cause.clone());
        }
        if self.active_input.is_some() {
            return Err(BackendError::new(
                K::BackendUnavailable,
                S::Backend,
                "CUDA physical completion is unconfirmed; mapped-image audit cannot release it",
            ));
        }
        Ok(())
    }

    pub fn has_unconfirmed_physical_completion(&self) -> bool {
        self.quarantine_cause.is_some() || self.active_input.is_some()
    }

    /// Original bounded cause of a CUDA execution with unconfirmed physical
    /// completion. Presence forbids PhysicalReady and reservation release.
    pub fn physical_quarantine_cause(&self) -> Option<&BackendError> {
        self.quarantine_cause.as_ref()
    }

    pub fn run(&mut self, inputs: &[&EncodedInput]) -> Result<Vec<RawOutput>, BackendError> {
        if inputs.len() > self.config.max_batch {
            return Err(BackendError::new(
                K::ResourceExhausted,
                S::Admission,
                "batch exceeds configured limit",
            ));
        }
        self.run_values(
            &inputs
                .iter()
                .map(|input| input.values())
                .collect::<Vec<_>>(),
        )
    }

    /// Ownership must return before any raw buffer is reused. Retained outputs
    /// in another caller are never modified. The physical worker returns these
    /// after converting to independently owned legal policy/WDL.
    pub fn recycle_outputs(&mut self, outputs: Vec<RawOutput>) {
        if !self.config.experiments.reuse_buffers {
            return;
        }
        for raw in outputs {
            if self.raw_pool.len() < self.config.max_batch
                && raw.policy_logits.capacity() == POLICY_SIZE
                && raw.wdl.capacity() == 3
            {
                self.raw_pool.push(raw);
            }
        }
    }
    pub fn last_io_timings(&self) -> Option<IoTimings> {
        self.timings
    }
    pub fn startup_timings(&self) -> StartupTimings {
        self.startup_timings
    }
    pub fn binding_runs(&self) -> u64 {
        self.bound_runs
    }
    #[cfg(feature = "experimental-io-binding")]
    fn release_completed_input(&mut self) {
        #[cfg(feature = "experimental-io-buffers")]
        if self.config.experiments.reuse_buffers {
            self.spare_input = self.active_input.take();
        }
        self.active_input = None;
    }

    /// Dense entry point for independent reference fixtures, still fully checked.
    pub fn run_values(&mut self, inputs: &[&[f32]]) -> Result<Vec<RawOutput>, BackendError> {
        self.run_values_profiled(inputs, None)
    }

    pub fn run_profiled(
        &mut self,
        inputs: &[&EncodedInput],
    ) -> (Result<Vec<RawOutput>, BackendError>, NativeRunTimings) {
        let mut timings = NativeRunTimings::default();
        if !self.config.source_profile_supported() {
            return (
                Err(BackendError::new(
                    K::UnsupportedModel,
                    S::Admission,
                    "source profiling requires default B1 native execution",
                )),
                timings,
            );
        }
        if inputs.len() > self.config.max_batch {
            return (
                Err(BackendError::new(
                    K::ResourceExhausted,
                    S::Admission,
                    "batch exceeds configured limit",
                )),
                timings,
            );
        }
        let values = inputs
            .iter()
            .map(|input| input.values())
            .collect::<Vec<_>>();
        let result = self.run_values_profiled(&values, Some(&mut timings));
        (result, timings)
    }

    fn run_values_profiled(
        &mut self,
        inputs: &[&[f32]],
        mut timings: Option<&mut NativeRunTimings>,
    ) -> Result<Vec<RawOutput>, BackendError> {
        let preparation_start = timings.as_ref().map(|_| Instant::now());
        if self.active_input.is_some() {
            return Err(BackendError::new(
                K::BackendUnavailable,
                S::Backend,
                "previous run did not attest physical completion",
            ));
        }
        if inputs.is_empty() || inputs.len() > self.config.max_batch {
            return Err(BackendError::new(
                K::ResourceExhausted,
                S::Admission,
                "empty or oversized batch",
            ));
        }
        if inputs
            .iter()
            .any(|input| input.len() != INPUT_VALUES || input.iter().any(|v| !v.is_finite()))
        {
            return Err(BackendError::new(
                K::InvalidInput,
                S::Admission,
                "input must be finite FP32 [112,8,8]",
            ));
        }
        let measured = self.config.experiments != ExecutionExperiments::default();
        let stage_started = measured.then(std::time::Instant::now);
        self.timings = measured.then(IoTimings::default);
        #[cfg(feature = "experimental-io-buffers")]
        let spare = if self.config.experiments.reuse_buffers {
            self.spare_input
                .take()
                .filter(|tensor| tensor.shape().as_ref() == [inputs.len() as i64, 112, 8, 8])
        } else {
            None
        };
        #[cfg(not(feature = "experimental-io-buffers"))]
        let spare: Option<Tensor<f32>> = None;
        let tensor = match spare {
            Some(mut tensor) => {
                let (_, values) = tensor.try_extract_tensor_mut::<f32>().map_err(|error| {
                    BackendError::new(
                        K::BackendFailure,
                        S::Backend,
                        "cannot write exclusive input buffer",
                    )
                    .with_ort_cause(CauseCode::TensorCreate, error)
                })?;
                for (destination, input) in values.chunks_exact_mut(INPUT_VALUES).zip(inputs) {
                    destination.copy_from_slice(input);
                }
                tensor
            }
            None => {
                let mut dense = Vec::new();
                dense
                    .try_reserve_exact(inputs.len() * INPUT_VALUES)
                    .map_err(|error| {
                        BackendError::new(
                            K::ResourceExhausted,
                            S::Admission,
                            "input staging allocation failed",
                        )
                        .with_external_cause(CauseCode::InputAllocation, &error)
                    })?;
                for input in inputs {
                    dense.extend_from_slice(input);
                }
                Tensor::from_array(([inputs.len(), 112, 8, 8], dense)).map_err(|error| {
                    BackendError::new(
                        K::BackendFailure,
                        S::Backend,
                        "cannot create ORT input tensor",
                    )
                    .with_ort_cause(CauseCode::TensorCreate, error)
                })?
            }
        };
        self.active_input = Some(tensor);
        if let (Some(timing), Some(started)) = (&mut self.timings, stage_started) {
            timing.host_stage = started.elapsed();
        }

        #[cfg(feature = "experimental-io-binding")]
        if self.config.experiments.io_binding {
            if self
                .bound
                .as_ref()
                .is_none_or(|buffers| buffers.batch != inputs.len())
            {
                // One shape slot, changed only after the preceding physical fence.
                let buffers = match io::BoundBuffers::new(
                    self.session.as_ref().expect("loaded session"),
                    self.config.provider,
                    inputs.len(),
                ) {
                    Ok(buffers) => buffers,
                    Err(error) => {
                        let failure = BackendError::new(
                            K::BackendFailure,
                            S::Backend,
                            "fixed binding allocation failed",
                        )
                        .with_ort_cause(CauseCode::TensorCreate, error);
                        if matches!(self.config.provider, Provider::Cuda { .. }) {
                            self.quarantine_cause = Some(failure.clone());
                        } else {
                            self.active_input = None;
                        }
                        return Err(failure);
                    }
                };
                self.bound = Some(buffers);
            }
            let was_clean = self.synchronous_runs.enter();
            let result = self.bound.as_mut().expect("fixed binding").run(
                self.session.as_mut().expect("loaded session"),
                self.active_input.as_ref().expect("input pin"),
                &mut self.raw_pool,
                self.config.experiments.reuse_buffers,
                self.timings.as_mut(),
            );
            self.synchronous_runs.returned(
                was_clean,
                matches!(&result, Ok(_) | Err(io::BoundFailure::Output(_))),
            );
            match result {
                Ok(outputs) => {
                    self.bound_runs = self.bound_runs.saturating_add(1);
                    self.release_completed_input();
                    return Ok(outputs);
                }
                Err(io::BoundFailure::Native(error)) => {
                    let failure = BackendError::new(
                        K::BackendFailure,
                        S::Backend,
                        "binding run/copy/fence failed",
                    )
                    .with_ort_cause(CauseCode::OrtRun, error);
                    if matches!(self.config.provider, Provider::Cuda { .. }) {
                        self.quarantine_cause = Some(failure.clone());
                    } else {
                        self.active_input = None;
                    }
                    return Err(failure);
                }
                Err(io::BoundFailure::Output(error)) => {
                    self.release_completed_input();
                    return Err(error);
                }
            }
        }
        // No RunOptions enabling asynchronous EP execution or terminate-on-cancel.
        // Only a successful CUDA Run attests the synchronous device fence.
        // Arbitrary CUDA errors retain the input/session instead of granting
        // physical Ready. CPU errors keep the existing completed-error behavior.
        let run_started = measured.then(std::time::Instant::now);
        let invocation_start = timings.as_ref().map(|_| Instant::now());
        if let (Some(start), Some(end), Some(timing)) =
            (preparation_start, invocation_start, timings.as_deref_mut())
        {
            timing.preparation = Some((start, end));
        }
        let was_clean = self.synchronous_runs.enter();
        let run_result = self
            .session
            .as_mut()
            .ok_or_else(|| {
                BackendError::new(K::BackendFailure, S::Backend, "ORT session is missing")
            })?
            .run(ort::inputs![INPUT_NAME => self.active_input.as_ref()
            .ok_or(BackendError::new(K::BackendFailure, S::Backend, "input pin is missing"))?]);
        self.synchronous_runs
            .returned(was_clean, run_result.is_ok());
        if let (Some(timing), Some(started)) = (&mut self.timings, run_started) {
            timing.run = started.elapsed();
        }
        let invocation_end = timings.as_ref().map(|_| Instant::now());
        if let (Some(start), Some(end), Some(timing)) =
            (invocation_start, invocation_end, timings.as_deref_mut())
        {
            timing.invocation = Some((start, end));
            timing.completion_attested = run_result.is_ok();
        }
        let outputs = match run_result {
            Ok(outputs) => outputs,
            Err(error) => {
                let failure =
                    BackendError::new(K::BackendFailure, S::Backend, "synchronous ORT Run failed")
                        .with_ort_cause(CauseCode::OrtRun, error);
                if matches!(self.config.provider, Provider::Cuda { .. }) {
                    self.quarantine_cause = Some(failure.clone());
                } else {
                    self.active_input = None;
                }
                return Err(failure);
            }
        };
        #[cfg(feature = "experimental-io-buffers")]
        if self.config.experiments.reuse_buffers {
            self.spare_input = self.active_input.take();
        }
        self.active_input = None;
        let malformed = || {
            BackendError::new(
                K::NumericalFailure,
                S::Output,
                "ORT output shape/dtype differs",
            )
        };
        let (policy_shape, policy) = outputs
            .get(POLICY_NAME)
            .ok_or_else(malformed)?
            .try_extract_tensor::<f32>()
            .map_err(|error| malformed().with_ort_cause(CauseCode::PolicyExtract, error))?;
        let (wdl_shape, wdl) = outputs
            .get(WDL_NAME)
            .ok_or_else(malformed)?
            .try_extract_tensor::<f32>()
            .map_err(|error| malformed().with_ort_cause(CauseCode::WdlExtract, error))?;
        if policy_shape.as_ref() != [inputs.len() as i64, POLICY_SIZE as i64]
            || wdl_shape.as_ref() != [inputs.len() as i64, 3]
        {
            return Err(malformed());
        }
        let own_started = measured.then(std::time::Instant::now);
        let result = pack_outputs(
            policy,
            wdl,
            inputs.len(),
            &mut self.raw_pool,
            self.config.experiments.reuse_buffers,
        );
        if let (Some(timing), Some(started)) = (&mut self.timings, own_started) {
            timing.own_outputs = started.elapsed();
        }
        if let (Some(start), Some(timing)) = (invocation_end, timings) {
            timing.output = Some((start, Instant::now()));
        }
        result
    }
}

impl crate::worker::synchronous_owner_sealed::Sealed for OnnxBackend {}
impl crate::worker::SynchronousPhysicalOwner for OnnxBackend {
    fn synchronous_owner_identity(&self) -> Arc<()> {
        OnnxBackend::synchronous_owner_identity(self)
    }
}

// Normal commit_from_memory creates an owned ORT session. This scope retains
// serialized bytes through the call and releases an owned buffer on both
// success and error, before interface checks or physical CUDA warm placement.
fn commit_verified_model<T>(
    model: impl AsRef<[u8]>,
    commit: impl FnOnce(&[u8]) -> Result<T, BackendError>,
    observe: &mut impl FnMut(StartupPhase),
) -> Result<T, BackendError> {
    observe(StartupPhase::SerializedModelReady);
    let result = commit(model.as_ref());
    if result.is_ok() {
        observe(StartupPhase::SessionCommittedBeforeSerializedRelease);
    }
    drop(model);
    observe(StartupPhase::SerializedModelReleased);
    result
}

fn pack_outputs(
    policy: &[f32],
    wdl: &[f32],
    batch: usize,
    pool: &mut Vec<RawOutput>,
    reuse: bool,
) -> Result<Vec<RawOutput>, BackendError> {
    if policy.len() != batch * POLICY_SIZE || wdl.len() != batch * 3 {
        return Err(BackendError::new(
            K::NumericalFailure,
            S::Output,
            "bound head shape differs",
        ));
    }
    let mut result = Vec::with_capacity(batch);
    for (policy, wdl) in policy.chunks_exact(POLICY_SIZE).zip(wdl.chunks_exact(3)) {
        let mut raw = if reuse { pool.pop() } else { None }.unwrap_or_else(|| RawOutput {
            policy_logits: Vec::with_capacity(POLICY_SIZE),
            wdl: Vec::with_capacity(3),
        });
        raw.policy_logits.clear();
        raw.policy_logits.extend_from_slice(policy);
        raw.wdl.clear();
        raw.wdl.extend_from_slice(wdl);
        output::validate_maia(&raw, &[0]).map_err(|error| {
            BackendError::new(
                K::NumericalFailure,
                S::Output,
                "nonfinite or inadmissible model output",
            )
            .with_output_cause(&error)
        })?;
        result.push(raw);
    }
    Ok(result)
}

fn validate_interface(session: &Session, profile: AssetProfile) -> Result<(), BackendError> {
    let valid = |value: &ValueType, expected: &[i64]| {
        matches!(value,
        ValueType::Tensor { ty: TensorElementType::Float32, shape, .. } if shape.as_ref() == expected)
    };
    if session.inputs.len() != 1
        || session.outputs.len() != 2 + usize::from(profile.has_moves_left_head())
        || session.inputs[0].name != INPUT_NAME
        || !valid(&session.inputs[0].input_type, &[-1, 112, 8, 8])
        || ![(POLICY_NAME, 1858), (WDL_NAME, 3)]
            .iter()
            .all(|(name, size)| {
                session
                    .outputs
                    .iter()
                    .any(|out| out.name == *name && valid(&out.output_type, &[-1, *size]))
            })
        || (profile.has_moves_left_head()
            && !session
                .outputs
                .iter()
                .any(|out| out.name == MLH_NAME && valid(&out.output_type, &[-1, 1])))
    {
        return Err(BackendError::new(
            K::UnsupportedModel,
            S::Asset,
            "expected selected dynamic-batch FP32 input/logits/WDL/optional MLH interface",
        ));
    }
    Ok(())
}

/// Strict evidence check; an empty/registration-only profile never passes.
pub fn verify_cuda_profile(bytes: &[u8]) -> Result<usize, BackendError> {
    let fail = || {
        BackendError::new(
            K::BackendUnavailable,
            S::Backend,
            "profile does not prove exclusive CUDA node execution",
        )
    };
    if bytes.len() > 4 * 1024 * 1024 {
        return Err(fail());
    }
    let events: Vec<serde_json::Value> = serde_json::from_slice(bytes)
        .map_err(|error| fail().with_external_cause(CauseCode::ProfileParse, &error))?;
    let mut count = 0;
    for event in events {
        if event.get("cat").and_then(|v| v.as_str()) != Some("Node") {
            continue;
        }
        let name = event
            .get("name")
            .and_then(|value| value.as_str())
            .ok_or_else(fail)?;
        if name.ends_with("_fence_before") || name.ends_with("_fence_after") {
            // ORT fence records are synchronization metadata, not node kernels.
            // They need not identify an EP, but a supplied EP must still be CUDA.
            if let Some(provider) = event.pointer("/args/provider") {
                if provider.as_str() != Some("CUDAExecutionProvider") {
                    return Err(fail());
                }
            }
            continue;
        }
        // Unknown Node records cannot be silently excluded from placement proof.
        // Actual kernel events require an explicit, correctly typed CUDA EP.
        if !name.ends_with("_kernel_time")
            || event
                .pointer("/args/provider")
                .and_then(|value| value.as_str())
                != Some("CUDAExecutionProvider")
        {
            return Err(fail());
        }
        count += 1;
    }
    if count == 0 {
        return Err(fail());
    }
    Ok(count)
}

#[cfg(test)]
mod buffer_tests {
    use super::*;
    #[test]
    fn ready_pair_transfer_is_atomic_and_preserves_owner_values() {
        let mut state = Lc0StartupState::NativeEnteredUnknown;
        let mut metadata = Some(Box::new(7_u8));
        let mut backend = Some(Box::new(9_u8));
        let original_metadata = &**metadata.as_ref().unwrap() as *const u8;
        let original_backend = &**backend.as_ref().unwrap() as *const u8;
        assert!(take_ready_pair(&mut state, &mut metadata, &mut backend).is_none());
        assert!(metadata.is_some() && backend.is_some());
        state = Lc0StartupState::Ready;
        let (moved_metadata, moved_backend) =
            take_ready_pair(&mut state, &mut metadata, &mut backend).unwrap();
        assert_eq!(&*moved_metadata as *const u8, original_metadata);
        assert_eq!(&*moved_backend as *const u8, original_backend);
        assert_eq!(state, Lc0StartupState::Transferred);
        assert!(metadata.is_none() && backend.is_none());
        assert!(take_ready_pair(&mut state, &mut metadata, &mut backend).is_none());

        let mut state = Lc0StartupState::Ready;
        let mut metadata = Some(7_u8);
        let mut backend: Option<u8> = None;
        assert!(take_ready_pair(&mut state, &mut metadata, &mut backend).is_none());
        assert_eq!(state, Lc0StartupState::Ready);
        assert_eq!(metadata, Some(7));
    }
    #[test]
    fn rejected_or_unwound_admission_does_not_claim_native_entry() {
        let mut state = Lc0StartupState::BeforeSessionEntry;
        let result = enter_lc0_session(&mut state, &mut || {
            Err(BackendError::new(K::InvalidInput, S::Admission, "rejected"))
        });
        assert!(result.is_err());
        assert_eq!(state, Lc0StartupState::BeforeSessionEntry);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = enter_lc0_session(&mut state, &mut || panic!("admission panic"));
        }));
        assert!(result.is_err());
        assert_eq!(state, Lc0StartupState::BeforeSessionEntry);

        enter_lc0_session(&mut state, &mut || Ok(())).expect("admitted");
        assert_eq!(state, Lc0StartupState::NativeEnteredUnknown);
    }

    #[test]
    fn synchronous_tracker_requires_success_and_never_clears_taint() {
        let mut tracker = SynchronousRunTracker::new();
        assert!(tracker.clean);
        let entry = tracker.enter();
        assert!(!tracker.clean);
        tracker.returned(entry, true);
        assert!(tracker.clean);
        assert_eq!(tracker.completed, 1);
        let entry = tracker.enter();
        tracker.returned(entry, false);
        assert!(!tracker.clean);
        let entry = tracker.enter();
        tracker.returned(entry, true);
        assert!(!tracker.clean);
        assert_eq!(tracker.completed, 2);
    }

    #[test]
    fn synchronous_tracker_unwind_and_count_overflow_remain_dirty() {
        let mut tracker = SynchronousRunTracker::new();
        let _ = tracker.enter(); // an unwind skips returned()
        assert!(!tracker.clean);
        let mut tracker = SynchronousRunTracker {
            clean: true,
            completed: u64::MAX,
        };
        let entry = tracker.enter();
        tracker.returned(entry, true);
        assert!(!tracker.clean);
        assert_eq!(tracker.completed, u64::MAX);
    }
    #[test]
    fn native_storage_drops_before_borrowed_model_on_success_error_and_unwind() {
        use std::sync::atomic::{AtomicBool, Ordering};
        struct Native(Arc<AtomicBool>);
        impl Drop for Native {
            fn drop(&mut self) {
                assert!(!self.0.load(Ordering::SeqCst));
            }
        }
        struct Model(Arc<AtomicBool>);
        impl Drop for Model {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        for mode in 0..3 {
            let dropped = Arc::new(AtomicBool::new(false));
            let outcome = std::panic::catch_unwind(|| {
                let storage = SessionStorage {
                    native: Native(Arc::clone(&dropped)),
                    retained_model: Some(Model(Arc::clone(&dropped))),
                };
                if mode == 1 {
                    return Err::<(), _>("postcommit validation failed");
                }
                if mode == 2 {
                    panic!("probe unwound");
                }
                drop(storage);
                Ok(())
            });
            assert_eq!(outcome.is_err(), mode == 2);
            assert!(dropped.load(Ordering::SeqCst));
        }
    }
    #[test]
    fn ort_mode_is_explicit_cuda_only_and_keeps_legacy_b1_identity() {
        let mut config = BackendConfig::cpu();
        config.experiments.zero_copy_ort = true;
        assert!(config.validate().is_err());
        config.provider = Provider::Cuda {
            device_id: 0,
            arena_bytes: 3 * 1024 * 1024 * 1024,
        };
        config.profiling_prefix = Some(std::env::temp_dir().join("ort-mode-test/placement"));
        assert_eq!(
            config.validate().is_ok(),
            cfg!(feature = "experimental-ort-model")
        );
        config.experiments.io_binding = true;
        assert!(config.validate().is_err());
        config.experiments.io_binding = false;
        let direct = declared_backend_identity([0; 32], [0; 32], Some([0; 32]), &config);
        config.experiments.zero_copy_ort = false;
        let baseline = declared_backend_identity([0; 32], [0; 32], Some([0; 32]), &config);
        assert_ne!(direct, baseline);
        config.experiments.copy_ort_model = true;
        assert_eq!(
            config.validate().is_ok(),
            cfg!(feature = "experimental-ort-model")
        );
        let copied = declared_backend_identity([0; 32], [0; 32], Some([0; 32]), &config);
        assert_ne!(copied, baseline);
        assert_ne!(copied, direct);
        config.experiments.zero_copy_ort = true;
        assert!(
            config.validate().is_err(),
            "direct and copied are mutually exclusive"
        );
        config.experiments.zero_copy_ort = false;
        config.provider = Provider::Cpu;
        assert!(config.validate().is_err());
    }
    #[test]
    fn extracted_identity_codec_preserves_published_b1_format() {
        let mut config = BackendConfig::cpu();
        config.max_batch = 1;
        config.provider = Provider::Cuda {
            device_id: 0,
            arena_bytes: 3 * 1024 * 1024 * 1024,
        };
        assert_eq!(
            declared_backend_identity([0; 32], [0; 32], Some([0; 32]), &config),
            asset::parse_sha256("0a142d7a64dada4119e49e53015f55427ecca05cdae69b6deb9eb35af56b6644")
                .unwrap()
        );
        let b1 = declared_backend_identity([0; 32], [0; 32], Some([0; 32]), &config);
        config.max_batch = 4;
        assert_ne!(
            b1,
            declared_backend_identity([0; 32], [0; 32], Some([0; 32]), &config)
        );
    }
    #[test]
    fn serialized_model_owner_is_released_before_postcommit_work_on_success_and_error() {
        use std::sync::atomic::{AtomicBool, Ordering};
        struct ModelBytes {
            dropped: Arc<AtomicBool>,
            bytes: Vec<u8>,
        }
        impl AsRef<[u8]> for ModelBytes {
            fn as_ref(&self) -> &[u8] {
                &self.bytes
            }
        }
        impl Drop for ModelBytes {
            fn drop(&mut self) {
                self.dropped.store(true, Ordering::Release);
            }
        }
        for succeeds in [true, false] {
            let dropped = Arc::new(AtomicBool::new(false));
            let model = ModelBytes {
                dropped: Arc::clone(&dropped),
                bytes: vec![1, 2, 3, 4],
            };
            let mut phases = Vec::new();
            let result = commit_verified_model(
                model,
                |bytes| {
                    assert!(!dropped.load(Ordering::Acquire));
                    assert_eq!(bytes, &[1, 2, 3, 4]);
                    if succeeds {
                        Ok(7)
                    } else {
                        Err(BackendError::new(
                            K::BackendUnavailable,
                            S::Backend,
                            "fixture commit failure",
                        ))
                    }
                },
                &mut |phase| {
                    assert_eq!(
                        dropped.load(Ordering::Acquire),
                        phase == StartupPhase::SerializedModelReleased
                    );
                    phases.push(phase);
                },
            );
            assert_eq!(result.is_ok(), succeeds);
            let expected = if succeeds {
                vec![
                    StartupPhase::SerializedModelReady,
                    StartupPhase::SessionCommittedBeforeSerializedRelease,
                    StartupPhase::SerializedModelReleased,
                ]
            } else {
                vec![
                    StartupPhase::SerializedModelReady,
                    StartupPhase::SerializedModelReleased,
                ]
            };
            assert_eq!(phases, expected);
            assert!(
                dropped.load(Ordering::Acquire),
                "serialized bytes overlap postcommit work"
            );
        }
        // Existing callers may keep a borrowed model for independent checks.
        let borrowed = vec![3, 2, 1];
        assert_eq!(
            commit_verified_model(&borrowed, |bytes| Ok(bytes.len()), &mut |_| {}).unwrap(),
            3
        );
        assert_eq!(borrowed, [3, 2, 1]);
    }

    #[test]
    fn raw_pool_reuses_returned_ownership_without_mutating_retained_outputs() {
        let policy = vec![0.0; POLICY_SIZE];
        let wdl = [0.5, 0.3, 0.2];
        let mut pool = Vec::new();
        let retained = pack_outputs(&policy, &wdl, 1, &mut pool, true)
            .unwrap()
            .remove(0);
        let returned = pack_outputs(&policy, &wdl, 1, &mut pool, true)
            .unwrap()
            .remove(0);
        let pointer = returned.policy_logits.as_ptr();
        pool.push(returned);
        let changed = vec![1.0; POLICY_SIZE];
        let next = pack_outputs(&changed, &wdl, 1, &mut pool, true)
            .unwrap()
            .remove(0);
        assert_eq!(next.policy_logits.as_ptr(), pointer);
        assert!(retained.policy_logits.iter().all(|value| *value == 0.0));
        assert!(next.policy_logits.iter().all(|value| *value == 1.0));
        let mut invalid = changed;
        invalid[POLICY_SIZE - 1] = f32::NAN;
        assert!(pack_outputs(&invalid, &wdl, 1, &mut pool, true).is_err());
        assert!(pack_outputs(&[], &wdl, 1, &mut pool, true).is_err());
    }
}
