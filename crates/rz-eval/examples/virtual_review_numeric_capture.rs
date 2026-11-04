//! Actual Fresh B1 ORT raw capture; numerical admission is a separate reader.
//! This executable does not label tensor fixtures as Rules/legal-game evidence.
//! Every CUDA graph requires the explicit V3 source/placement policy. Required
//! capture failures preserve pending recovery files. A separate post-join policy
//! reporter failure preserves published raw data but remains a partial command.
use rz_eval::asset;
use rz_eval::error::BackendError;
use rz_eval::onnx::{NativeLoadingProfile, OrtRuntime, Provider};
#[cfg(target_os = "linux")]
use rz_eval::onnx::{SystemCaptureLimits, SystemLibraryRequest, SystemRuntime};
use rz_eval::pals_model::virtual_review::{
    self as model, PalsVirtualReviewInput, ReviewExecutionIdentity,
};
use rz_eval::pals_model::{PalsCandidateToken, PalsModelProfile, PalsRawOutput, PalsRole};
#[cfg(target_os = "linux")]
use rz_eval::pals_onnx::virtual_review::SystemPlacementObservation;
use rz_eval::pals_onnx::virtual_review::{
    PalsVirtualReviewOnnxBackend, VirtualReviewExecutionMode, VirtualReviewInvocation,
    VirtualReviewNativeCommand as Command, VirtualReviewNativeResult as Outcome,
    validate_virtual_review_export_manifest,
};
#[cfg(target_os = "linux")]
use rz_eval::pals_onnx::virtual_review::{
    VirtualReviewArtifactKind, VirtualReviewExportManifestSummary, VirtualReviewHeldFileSources,
    VirtualReviewHeldGraph, VirtualReviewProfileDirectory, VirtualReviewStartupCustody,
};
use rz_eval::pals_onnx::{
    PalsBackendStats, PalsCuDnnConvSearch, PalsCudaControlPolicy, PalsOnnxConfig,
};
use rz_eval::runtime_pin::RuntimeCache;
use rz_eval::worker::{PhysicalPoll, SingleWorker};
#[cfg(target_os = "linux")]
use rz_eval::worker::{PhysicalRecoveryKey, PhysicalRecoveryReservation, SessionReleased};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    task::Poll,
    time::{Duration, Instant},
};
#[cfg(target_os = "linux")]
#[path = "virtual_review_numeric_capture/owned_worker.rs"]
mod owned_worker;
#[path = "virtual_review_numeric_capture/tensors.rs"]
mod tensors;
type Result<T> = std::result::Result<T, Box<dyn Error>>;
type LegacyWorker = SingleWorker<Command, std::result::Result<Outcome, BackendError>>;

enum Worker {
    Legacy(LegacyWorker),
    #[cfg(target_os = "linux")]
    System(SystemCaptureWorker),
}

enum PreparedRuntime {
    Legacy(OrtRuntime),
    #[cfg(target_os = "linux")]
    System(SystemRuntime),
}
const LIMIT: usize = 4 * 1024 * 1024;
const POLICY_LIMIT: usize = 16 * 1024;
// This bound is the registered logical raw numerical payload total. The exact
// recovery staging owner adds one fixed header per file and publication keeps
// at most one extra 4MiB payload. Separately bounded profile/placement evidence
// is not relabeled as numerical payload or unobserved disk use.
const RAW_TOTAL: u64 = 64 * 1024 * 1024;
const PENDING_HEADER: &[u8] = b"RZ_PENDING_PROFILE_AND_JOIN\n";

/// serde's typed unknown-field validation alone does not reject duplicate
/// object keys inside BTreeMaps or payloads. Validate every JSON object before
/// constructing the typed plan/config; the caller has already bounded bytes.
fn decode_unique<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    struct Unique(serde_json::Value);
    impl<'de> Deserialize<'de> for Unique {
        fn deserialize<D: serde::Deserializer<'de>>(
            decoder: D,
        ) -> std::result::Result<Self, D::Error> {
            struct Visitor;
            impl<'de> serde::de::Visitor<'de> for Visitor {
                type Value = Unique;
                fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    f.write_str("bounded JSON without duplicate object keys")
                }
                fn visit_bool<E: serde::de::Error>(
                    self,
                    v: bool,
                ) -> std::result::Result<Unique, E> {
                    Ok(Unique(v.into()))
                }
                fn visit_i64<E: serde::de::Error>(self, v: i64) -> std::result::Result<Unique, E> {
                    Ok(Unique(v.into()))
                }
                fn visit_u64<E: serde::de::Error>(self, v: u64) -> std::result::Result<Unique, E> {
                    Ok(Unique(v.into()))
                }
                fn visit_f64<E: serde::de::Error>(self, v: f64) -> std::result::Result<Unique, E> {
                    serde_json::Number::from_f64(v)
                        .map(|n| Unique(n.into()))
                        .ok_or_else(|| E::custom("nonfinite JSON number"))
                }
                fn visit_str<E: serde::de::Error>(self, v: &str) -> std::result::Result<Unique, E> {
                    Ok(Unique(v.into()))
                }
                fn visit_string<E: serde::de::Error>(
                    self,
                    v: String,
                ) -> std::result::Result<Unique, E> {
                    Ok(Unique(v.into()))
                }
                fn visit_unit<E: serde::de::Error>(self) -> std::result::Result<Unique, E> {
                    Ok(Unique(serde_json::Value::Null))
                }
                fn visit_seq<A: serde::de::SeqAccess<'de>>(
                    self,
                    mut seq: A,
                ) -> std::result::Result<Unique, A::Error> {
                    let mut values = Vec::new();
                    while let Some(v) = seq.next_element::<Unique>()? {
                        values.push(v.0);
                    }
                    Ok(Unique(values.into()))
                }
                fn visit_map<A: serde::de::MapAccess<'de>>(
                    self,
                    mut map: A,
                ) -> std::result::Result<Unique, A::Error> {
                    let mut values = serde_json::Map::new();
                    while let Some(key) = map.next_key::<String>()? {
                        if values.contains_key(&key) {
                            return Err(serde::de::Error::custom("duplicate JSON object key"));
                        }
                        values.insert(key, map.next_value::<Unique>()?.0);
                    }
                    Ok(Unique(values.into()))
                }
            }
            decoder.deserialize_any(Visitor)
        }
    }
    let unique: Unique = serde_json::from_slice(bytes)?;
    Ok(serde_json::from_value(unique.0)?)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Bindings {
    model_profile: String,
    model_schema: String,
    model_semantics: String,
    encoding_schema: String,
    model_semantic_sha256: String,
    encoding_sha256: String,
    model_epoch: u64,
    weights_sha256: String,
    export_manifest_sha256: String,
    graphs: BTreeMap<String, String>,
    runtime_library_sha256: String,
    #[serde(default)]
    cuda_libraries: BTreeSet<String>,
    cuda_manifest_sha256: Option<String>,
    #[serde(default)]
    system_cuda_config_sha256: Option<String>,
    #[serde(default)]
    system_cuda_libraries: Option<Vec<SystemLibraryBinding>>,
    backend: String,
    precision: String,
    tf32: bool,
    source_components: BTreeMap<String, String>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SystemLibraryBinding {
    role: String,
    sha256: String,
    bytes: u64,
    required_mapping: bool,
}
impl Serialize for Bindings {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("model_profile", &self.model_profile)?;
        map.serialize_entry("model_schema", &self.model_schema)?;
        map.serialize_entry("model_semantics", &self.model_semantics)?;
        map.serialize_entry("encoding_schema", &self.encoding_schema)?;
        map.serialize_entry("model_semantic_sha256", &self.model_semantic_sha256)?;
        map.serialize_entry("encoding_sha256", &self.encoding_sha256)?;
        map.serialize_entry("model_epoch", &self.model_epoch)?;
        map.serialize_entry("weights_sha256", &self.weights_sha256)?;
        map.serialize_entry("export_manifest_sha256", &self.export_manifest_sha256)?;
        map.serialize_entry("graphs", &self.graphs)?;
        map.serialize_entry("runtime_library_sha256", &self.runtime_library_sha256)?;
        if self.backend == "ort_cuda_system" {
            map.serialize_entry("system_cuda_config_sha256", &self.system_cuda_config_sha256)?;
            map.serialize_entry("system_cuda_libraries", &self.system_cuda_libraries)?;
        } else {
            map.serialize_entry("cuda_libraries", &self.cuda_libraries)?;
            map.serialize_entry("cuda_manifest_sha256", &self.cuda_manifest_sha256)?;
        }
        map.serialize_entry("backend", &self.backend)?;
        map.serialize_entry("precision", &self.precision)?;
        map.serialize_entry("tf32", &self.tf32)?;
        map.serialize_entry("source_components", &self.source_components)?;
        map.end()
    }
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(
    tag = "model_input_kind",
    content = "input",
    rename_all = "snake_case",
    deny_unknown_fields
)]
enum Input {
    VirtualReview(PalsVirtualReviewInput),
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    input: Input,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Group {
    id: String,
    role: PalsRole,
    line_bucket: usize,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    domain: String,
    schema_version: u32,
    invocation: String,
    batch: u32,
    bindings: Bindings,
    groups: Vec<Group>,
}
#[derive(Clone, Debug, Deserialize)]
#[cfg_attr(
    not(target_os = "linux"),
    allow(
        dead_code,
        reason = "Original directory FD handoff is refused outside Linux"
    )
)]
#[serde(deny_unknown_fields)]
struct OriginalEmptyNativeOutputRoot {
    schema: String,
    device: u64,
    inode: u64,
    uid: u32,
    gid: u32,
}

struct CaptureOutputRoot {
    path: PathBuf,
    admission: Option<OriginalEmptyNativeOutputRoot>,
    #[cfg(target_os = "linux")]
    original: Option<fs::File>,
}
impl CaptureOutputRoot {
    fn verify(&self) -> Result<()> {
        let Some(admission) = &self.admission else {
            return Ok(());
        };
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::fs::MetadataExt;
            let original = self
                .original
                .as_ref()
                .ok_or("original output directory FD absent")?;
            let held = original.metadata()?;
            let named = fs::symlink_metadata(&self.path)?;
            if admission.schema != "rz-pals-original-empty-native-output-root/1"
                || admission.inode == 0
                || !held.is_dir()
                || !named.is_dir()
                || named.file_type().is_symlink()
                || (held.dev(), held.ino(), held.uid(), held.gid())
                    != (
                        admission.device,
                        admission.inode,
                        admission.uid,
                        admission.gid,
                    )
                || (named.dev(), named.ino(), named.uid(), named.gid())
                    != (
                        admission.device,
                        admission.inode,
                        admission.uid,
                        admission.gid,
                    )
                || outside_git(&self.path)? != self.path
            {
                return Err(
                    "original admitted output directory inode/owner/ancestry changed".into(),
                );
            }
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = admission;
            Err("original existing output directory admission requires Linux".into())
        }
    }
}
fn capture_output_root(
    path: &Path,
    admission: Option<&OriginalEmptyNativeOutputRoot>,
) -> Result<CaptureOutputRoot> {
    let parent = outside_git(path.parent().ok_or("output parent absent")?)?;
    let root = parent.join(path.file_name().ok_or("output root absent")?);
    if admission.is_none() {
        // Preserve the independent CLI's original create-new behavior.
        fs::create_dir(&root)?;
        return Ok(CaptureOutputRoot {
            path: root,
            admission: None,
            #[cfg(target_os = "linux")]
            original: None,
        });
    }
    #[cfg(target_os = "linux")]
    {
        // Ancestry and leaf-link refusal precede opening; the held and named
        // inode are checked again after opening and after worker join.
        if outside_git(&root)? != root {
            return Err("original output directory must use its canonical path".into());
        }
        let owner = CaptureOutputRoot {
            original: Some(fs::File::open(&root)?),
            path: root,
            admission: admission.cloned(),
        };
        owner.verify()?;
        if fs::read_dir(&owner.path)?.next().transpose()?.is_some() {
            return Err("original admitted output directory must be empty".into());
        }
        owner.verify()?;
        Ok(owner)
    }
    #[cfg(not(target_os = "linux"))]
    {
        Err("original existing output directory admission requires Linux".into())
    }
}

/// Prepare the exact manifest, graph and optional profile directory owners
/// before entering ORT. Native loading will re-check bytes from these FDs.
#[cfg(target_os = "linux")]
fn system_startup_custody(
    manifest_path: &Path,
    summary: &VirtualReviewExportManifestSummary,
    output: &CaptureOutputRoot,
    with_control: bool,
) -> Result<VirtualReviewStartupCustody> {
    use std::os::unix::fs::OpenOptionsExt;
    use std::path::Component;

    const O_NOFOLLOW: i32 = 0x20000;
    const O_NONBLOCK: i32 = 0x800;
    const O_DIRECTORY: i32 = 0x10000;

    let open_file = |path: &Path| -> Result<fs::File> {
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(O_NOFOLLOW | O_NONBLOCK)
            .open(path)?;
        if !file.metadata()?.is_file() {
            return Err("system capture source is not a regular file".into());
        }
        Ok(file)
    };
    let manifest = open_file(manifest_path)?;
    let parent = manifest_path.parent().ok_or("manifest parent absent")?;
    let mut graphs = Vec::new();
    graphs.try_reserve_exact(summary.graphs.len())?;
    for graph in &summary.graphs {
        let mut components = Path::new(&graph.file).components();
        if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
            return Err("registered graph path is not a single filename".into());
        }
        graphs.push(VirtualReviewHeldGraph {
            role: graph.role.clone(),
            registered_file: graph.file.clone(),
            sha256: graph.sha256,
            file: open_file(&parent.join(&graph.file))?,
        });
    }
    let profile = if with_control {
        output.verify()?;
        let path = output.path.join("profiles");
        fs::create_dir(&path)?;
        let dir = OpenOptions::new()
            .read(true)
            .custom_flags(O_NOFOLLOW | O_DIRECTORY)
            .open(&path)?;
        output.verify()?;
        Some(VirtualReviewProfileDirectory::from_owned_directory(dir)?)
    } else {
        None
    };
    let sources = VirtualReviewHeldFileSources::new(manifest, graphs);
    Ok(VirtualReviewStartupCustody::new(
        std::sync::Arc::new(sources),
        profile,
    ))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    case_plan: PathBuf,
    case_plan_sha256: String,
    export_manifest: PathBuf,
    runtime_library: PathBuf,
    runtime_cache: PathBuf,
    output_root: PathBuf,
    #[serde(default)]
    output_root_admission: Option<OriginalEmptyNativeOutputRoot>,
    provider: String,
    /// Omission retains the original CUDA EP option without calling a setter.
    /// This is execution-policy provenance, outside the unchanged raw V5 ABI.
    #[serde(default)]
    cudnn_conv_search: Option<String>,
    cuda_bundle: Option<PathBuf>,
    cuda_inventory: Option<PathBuf>,
    cuda_inventory_sha256: Option<String>,
    #[serde(default)]
    system_cuda_config: Option<PathBuf>,
    #[serde(default)]
    system_cuda_config_sha256: Option<String>,
    /// Separate, explicitly pinned V3 static graph inventory. Its presence
    /// admits only the exact metadata/control CPU nodes checked at session
    /// initialization; it is not a runtime or physical completion witness.
    #[serde(default)]
    system_cuda_inventory: Option<PathBuf>,
    #[serde(default)]
    system_cuda_inventory_sha256: Option<String>,
    gpu_device_id: i32,
    gpu_arena_bytes: usize,
    deadline_seconds: u64,
    cleanup_seconds: u64,
}
impl Config {
    /// Pure selection before asset/native I/O. The loader repeats this fence
    /// against the actual retained profile and parsed V3 inventory.
    fn execution_config(&self) -> Result<PalsOnnxConfig> {
        let mut selected = PalsOnnxConfig::cpu();
        selected.intra_threads = 1;
        selected.cache_public_memory = false;
        let profile = match self.provider.as_str() {
            "cpu" => None,
            "cuda-system" => None,
            "cuda-control" => Some(NativeLoadingProfile::EagerCuda12Cudnn9V1),
            "cuda-control-shim" => Some(NativeLoadingProfile::CuDnnShimLazyV1),
            _ => {
                return Err("explicit cpu/cuda-control/cuda-control-shim provider required".into());
            }
        };
        if profile.is_some() || self.provider == "cuda-system" {
            selected.provider = Provider::Cuda {
                device_id: self.gpu_device_id,
                arena_bytes: self.gpu_arena_bytes,
            };
        }
        selected.cudnn_conv_search = match self.cudnn_conv_search.as_deref() {
            None | Some("legacy_implicit_exhaustive") => {
                PalsCuDnnConvSearch::LegacyImplicitExhaustive
            }
            Some("experimental_default") => PalsCuDnnConvSearch::ExperimentalDefault,
            _ => return Err("unsupported capture cuDNN convolution search selection".into()),
        };
        if selected.cudnn_conv_search == PalsCuDnnConvSearch::ExperimentalDefault {
            if self.cuda_bundle.is_none() || self.cuda_inventory.is_none() {
                return Err(
                    "experimental DEFAULT requires explicit V3 inventory and bundle".into(),
                );
            }
            pin(self
                .cuda_inventory_sha256
                .as_deref()
                .ok_or("experimental DEFAULT requires explicit V3 inventory SHA")?)?;
        }
        if self.provider == "cuda-system" {
            if self.cuda_bundle.is_some()
                || self.cuda_inventory.is_some()
                || self.cuda_inventory_sha256.is_some()
                || self.system_cuda_config.is_none()
                || self.system_cuda_config_sha256.is_none()
            {
                return Err(
                    "system CUDA requires its own config and no legacy bundle/control".into(),
                );
            }
            match (
                &self.system_cuda_inventory,
                &self.system_cuda_inventory_sha256,
            ) {
                (None, None) => {}
                (Some(path), Some(digest)) if path.is_absolute() => {
                    pin(digest)?;
                }
                _ => {
                    return Err("system CUDA control requires an absolute inventory and SHA".into());
                }
            }
            selected.validate_system_execution_selection()?;
        } else {
            if self.system_cuda_config.is_some()
                || self.system_cuda_config_sha256.is_some()
                || self.system_cuda_inventory.is_some()
                || self.system_cuda_inventory_sha256.is_some()
            {
                return Err("system config is only supported by cuda-system".into());
            }
            selected.validate_execution_selection(profile, profile.is_some())?;
        }
        Ok(selected)
    }
}

/// Mapping unavailability is a typed observation, not an inferred kernel
/// result. A verified witness remains scoped to this owner and profile window.
#[cfg(target_os = "linux")]
fn system_placement_observation_json(
    observation: &SystemPlacementObservation,
) -> serde_json::Value {
    match observation {
        SystemPlacementObservation::Verified(witness) => serde_json::json!({
            "status":"verified", "witness":witness,
            "physical_completion":"requires_same_native_owner_completion"
        }),
        SystemPlacementObservation::MappingUnavailable => serde_json::json!({
            "status":"mapping_unavailable", "gpu_kernel_placement":"unknown",
            "physical_completion":"unknown"
        }),
        SystemPlacementObservation::UnregisteredOrPreexistingMappings => serde_json::json!({
            "status":"unregistered_or_preexisting_mappings", "gpu_kernel_placement":"unknown",
            "physical_completion":"unknown"
        }),
    }
}

#[cfg(target_os = "linux")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SystemCudaConfig {
    max_libraries: usize,
    max_file_bytes: u64,
    max_total_bytes: u64,
    max_capture_ms: u64,
    libraries: Vec<SystemCudaLibrary>,
}
#[cfg(target_os = "linux")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SystemCudaLibrary {
    role: String,
    path: PathBuf,
    required_mapping: bool,
}

#[derive(Serialize)]
struct SessionPolicyObservation {
    role: String,
    provider: &'static str,
    intra_threads: usize,
    inter_threads: usize,
    intra_op_spinning: bool,
    inter_op_spinning: bool,
    parallel_execution: bool,
    cuda_provider_available: Option<bool>,
    evidence_scope: &'static str,
    coverage_scope: &'static str,
    evidence_method: &'static str,
}

/// Copied from the successfully loaded backend before it moves to the sole
/// physical worker. This metadata grants no completion or process-exit proof.
#[derive(Serialize)]
struct LoadedPolicyObservation {
    requested_cudnn_conv_search: Option<String>,
    backend_config_cudnn_conv_search: &'static str,
    cudnn_conv_algorithm_search_request: &'static str,
    cudnn_setting_evidence: &'static str,
    public_memory: &'static str,
    native_loading_profile: Option<&'static str>,
    applied_session_settings: Vec<SessionPolicyObservation>,
}

#[derive(Serialize)]
struct JoinedPolicyObservation {
    domain: &'static str,
    scope: &'static str,
    loaded: LoadedPolicyObservation,
    captured_physical_completions_known: bool,
    native_worker_joined: bool,
    actual_cudnn_plan: &'static str,
    tf32_actual: &'static str,
    numerical_acceptance: bool,
    performance_improvement_claim: bool,
    process_exit_evidence: &'static str,
    cuda_profiling_window: Option<CudaProfilingWindow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    original_system_session_release: Option<SystemSessionReleaseObservation>,
}

#[derive(Serialize)]
struct SystemSessionReleaseObservation {
    completed_synchronous_runs: u64,
    scope: &'static str,
    process_exit: &'static str,
    device_wide_gpu_fence: &'static str,
}

#[derive(Serialize)]
struct CudaProfilingWindow {
    scope: &'static str,
    actual_cases: Vec<ProfiledCase>,
    completed_nn_inputs: u64,
    remaining_cases_profiled: bool,
}

#[derive(Serialize)]
struct ProfiledCase {
    group: String,
    case: String,
    role: PalsRole,
    line_bucket: usize,
    records: usize,
    anchors: usize,
}

fn loaded_policy_observation(
    backend: &PalsVirtualReviewOnnxBackend,
    requested: Option<&str>,
    explicit_control: bool,
) -> Result<LoadedPolicyObservation> {
    let config = backend.config();
    if backend.system_cuda_runtime_selected() {
        config.validate_system_execution_selection()?;
    } else {
        config.validate_execution_selection(backend.native_loading_profile(), explicit_control)?;
    }
    if config.device_public_memory {
        return Err("raw capture requires the original host-public execution path".into());
    }
    let (selection, option_request) = match config.cudnn_conv_search {
        PalsCuDnnConvSearch::LegacyImplicitExhaustive => {
            ("legacy_implicit_exhaustive", "no_setter_ep_default")
        }
        PalsCuDnnConvSearch::ExperimentalDefault => ("experimental_default", "DEFAULT"),
        PalsCuDnnConvSearch::ExperimentalDefaultDeviceWarm => {
            return Err("device Warm is outside Fresh capture selection".into());
        }
    };
    let applied_session_settings = backend
        .applied_session_settings()
        .iter()
        .map(|settings| SessionPolicyObservation {
            role: settings.role().to_owned(),
            provider: match settings.provider() {
                Provider::Cpu => "CPUExecutionProvider",
                Provider::Cuda { .. } => "CUDAExecutionProvider",
            },
            intra_threads: settings.intra_threads(),
            inter_threads: settings.inter_threads(),
            intra_op_spinning: settings.intra_op_spinning(),
            inter_op_spinning: settings.inter_op_spinning(),
            parallel_execution: settings.parallel_execution(),
            cuda_provider_available: settings.cuda_provider_available(),
            evidence_scope: settings.evidence_scope(),
            coverage_scope: settings.coverage_scope(),
            evidence_method: settings.evidence_method(),
        })
        .collect::<Vec<_>>();
    if applied_session_settings.is_empty() || applied_session_settings.len() > 3 {
        return Err("loaded capture model session count differs".into());
    }
    Ok(LoadedPolicyObservation {
        requested_cudnn_conv_search: requested.map(str::to_owned),
        backend_config_cudnn_conv_search: selection,
        cudnn_conv_algorithm_search_request: option_request,
        cudnn_setting_evidence: "successful_loader_config_not_observed_cudnn_plan",
        public_memory: "host",
        native_loading_profile: backend.native_loading_profile().map(|p| p.identifier()),
        applied_session_settings,
    })
}
#[derive(Serialize)]
struct RawCase {
    id: String,
    input: Input,
    input_tensors: Vec<tensors::TensorCapture>,
    input_tensor_sha256: String,
    candidate_order: Vec<PalsCandidateToken>,
    raw_output: PalsRawOutput,
    physical_completion_id: Option<String>,
}
#[derive(Serialize)]
struct RawRun {
    schema_version: u32,
    domain: &'static str,
    run_id: String,
    producer: &'static str,
    producer_source_component: String,
    bindings: Bindings,
    invocation: &'static str,
    batch: u32,
    observed_provider: Option<&'static str>,
    completed_nn_inputs: u64,
    public_cache_hits: u64,
    physical_completion_known: bool,
    cases: Vec<RawCase>,
}
#[derive(Debug, Default, Eq, PartialEq, Serialize)]
struct StageWork {
    admitted_role_requests: u64,
    public_cache_hits: u64,
    public_cache_misses: u64,
    public_nn_runs_attempted: u64,
    public_nn_runs_completed: u64,
    public_nn_runs_failed_known: u64,
    role_nn_runs_attempted: u64,
    role_nn_runs_completed: u64,
    role_nn_runs_failed_known: u64,
    validated_public_outputs: u64,
    validated_role_outputs: u64,
    completed_nn_inputs: u64,
}

impl StageWork {
    fn checked_add(&mut self, other: Self) -> Result<()> {
        let add = |a: u64, b: u64| {
            a.checked_add(b)
                .ok_or_else(|| Box::<dyn Error>::from("native stage delta sum overflow"))
        };
        let next = Self {
            admitted_role_requests: add(self.admitted_role_requests, other.admitted_role_requests)?,
            public_cache_hits: add(self.public_cache_hits, other.public_cache_hits)?,
            public_cache_misses: add(self.public_cache_misses, other.public_cache_misses)?,
            public_nn_runs_attempted: add(
                self.public_nn_runs_attempted,
                other.public_nn_runs_attempted,
            )?,
            public_nn_runs_completed: add(
                self.public_nn_runs_completed,
                other.public_nn_runs_completed,
            )?,
            public_nn_runs_failed_known: add(
                self.public_nn_runs_failed_known,
                other.public_nn_runs_failed_known,
            )?,
            role_nn_runs_attempted: add(self.role_nn_runs_attempted, other.role_nn_runs_attempted)?,
            role_nn_runs_completed: add(self.role_nn_runs_completed, other.role_nn_runs_completed)?,
            role_nn_runs_failed_known: add(
                self.role_nn_runs_failed_known,
                other.role_nn_runs_failed_known,
            )?,
            validated_public_outputs: add(
                self.validated_public_outputs,
                other.validated_public_outputs,
            )?,
            validated_role_outputs: add(self.validated_role_outputs, other.validated_role_outputs)?,
            completed_nn_inputs: add(self.completed_nn_inputs, other.completed_nn_inputs)?,
        };
        *self = next;
        Ok(())
    }

    fn validate_fresh_count(&self, count: usize) -> Result<()> {
        let n = count as u64;
        if [
            self.admitted_role_requests,
            self.public_cache_misses,
            self.public_nn_runs_attempted,
            self.public_nn_runs_completed,
            self.role_nn_runs_attempted,
            self.role_nn_runs_completed,
            self.validated_public_outputs,
            self.validated_role_outputs,
        ]
        .iter()
        .any(|v| *v != n)
            || self.public_cache_hits != 0
            || self.public_nn_runs_failed_known != 0
            || self.role_nn_runs_failed_known != 0
            || self.completed_nn_inputs != 2 * n
        {
            return Err(
                "Fresh cache-off public/private completed work does not match captured cases"
                    .into(),
            );
        }
        Ok(())
    }
}

fn stage_delta(
    before: &PalsBackendStats,
    after: &PalsBackendStats,
    count: usize,
) -> Result<StageWork> {
    before.validate()?;
    after.validate()?;
    let delta = |a: u64, b: u64| {
        a.checked_sub(b)
            .ok_or_else(|| Box::<dyn Error>::from("native stage counter regressed"))
    };
    let s = StageWork {
        admitted_role_requests: delta(after.admitted_role_requests, before.admitted_role_requests)?,
        public_cache_hits: delta(after.public_cache_hits, before.public_cache_hits)?,
        public_cache_misses: delta(after.public_cache_misses, before.public_cache_misses)?,
        public_nn_runs_attempted: delta(
            after.public_nn_runs_attempted,
            before.public_nn_runs_attempted,
        )?,
        public_nn_runs_completed: delta(
            after.public_nn_runs_completed,
            before.public_nn_runs_completed,
        )?,
        public_nn_runs_failed_known: delta(
            after.public_nn_runs_failed_known,
            before.public_nn_runs_failed_known,
        )?,
        role_nn_runs_attempted: delta(after.role_nn_runs_attempted, before.role_nn_runs_attempted)?,
        role_nn_runs_completed: delta(after.role_nn_runs_completed, before.role_nn_runs_completed)?,
        role_nn_runs_failed_known: delta(
            after.role_nn_runs_failed_known,
            before.role_nn_runs_failed_known,
        )?,
        validated_public_outputs: delta(
            after.validated_public_outputs,
            before.validated_public_outputs,
        )?,
        validated_role_outputs: delta(after.validated_role_outputs, before.validated_role_outputs)?,
        completed_nn_inputs: delta(after.completed_nn_inputs, before.completed_nn_inputs)?,
    };
    s.validate_fresh_count(count)?;
    Ok(s)
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CaptureSlot {
    group: usize,
    case: usize,
}

struct CaptureOrder {
    slots: Vec<CaptureSlot>,
    profile_cases: usize,
}

fn capture_order(groups: &[Group], cuda: bool, collection: bool) -> Result<CaptureOrder> {
    let mut slots = Vec::new();
    if cuda {
        let roles = [PalsRole::Proposer, PalsRole::Critic, PalsRole::Validator];
        for role in &roles[..if collection { 3 } else { 2 }] {
            // Use original cases only. Prefer a nonempty record branch, short
            // physical L, and all available anchors before ending profiles.
            // Kernel/provider acceptance remains the original backend's job.
            let slot = groups
                .iter()
                .enumerate()
                .filter(|(_, group)| group.role == *role)
                .flat_map(|(group_index, group)| {
                    group
                        .cases
                        .iter()
                        .enumerate()
                        .map(move |(case_index, case)| {
                            let Input::VirtualReview(input) = &case.input;
                            (
                                (
                                    input.records.is_empty(),
                                    group.line_bucket,
                                    std::cmp::Reverse(input.anchors.len()),
                                    group_index,
                                    case_index,
                                ),
                                CaptureSlot {
                                    group: group_index,
                                    case: case_index,
                                },
                            )
                        })
                })
                .min_by_key(|(key, _)| *key)
                .map(|(_, slot)| slot)
                .ok_or("capture plan lacks an original case for a registered role")?;
            slots.push(slot);
        }
    }
    let profile_cases = slots.len();
    let selected = slots.iter().copied().collect::<BTreeSet<_>>();
    for (group, item) in groups.iter().enumerate() {
        for case in 0..item.cases.len() {
            let slot = CaptureSlot { group, case };
            if !selected.contains(&slot) {
                slots.push(slot);
            }
        }
    }
    Ok(CaptureOrder {
        slots,
        profile_cases,
    })
}

struct GroupCapture {
    id: String,
    remaining: Vec<Option<Case>>,
    captured: Vec<Option<RawCase>>,
    stage: StageWork,
}

impl From<Group> for GroupCapture {
    fn from(group: Group) -> Self {
        Self {
            id: group.id,
            captured: (0..group.cases.len()).map(|_| None).collect(),
            remaining: group.cases.into_iter().map(Some).collect(),
            stage: StageWork::default(),
        }
    }
}

fn capture_case(
    worker: &mut Worker,
    group: &mut GroupCapture,
    case_index: usize,
    epoch: [u8; 32],
    backend_digest: [u8; 32],
    completion: &mut u64,
    until: Instant,
) -> Result<u64> {
    let case = group
        .remaining
        .get_mut(case_index)
        .and_then(Option::take)
        .ok_or("capture case was already executed or is outside the original plan")?;
    if group.captured[case_index].is_some() {
        return Err("capture case has an earlier physical completion".into());
    }
    let Input::VirtualReview(input) = &case.input;
    let input_tensors = tensors::capture(input)?;
    let input_tensor_sha256 = tensors::digest(&input_tensors, input.role)?;
    let identity = identity(input, epoch, backend_digest)?;
    // These are adjacent snapshots of this actual invocation, never synthetic
    // group boundaries spanning another group's interleaved work.
    let before = stats(worker, until)?;
    let output = match command(
        worker,
        Command::Evaluate(VirtualReviewInvocation {
            request_history: identity.rules_history,
            identity,
            input: input.clone(),
        }),
        until,
    )? {
        Outcome::Evaluation(raw) => raw,
        _ => return Err("incorrect physical evaluation completion".into()),
    };
    let after = stats(worker, until)?;
    let delta = stage_delta(&before, &after, 1)?;
    let actual_nn_inputs = delta.completed_nn_inputs;
    group.stage.checked_add(delta)?;
    *completion = completion
        .checked_add(1)
        .ok_or("physical completion sequence overflow")?;
    let candidate_order = input.candidates.clone();
    group.captured[case_index] = Some(RawCase {
        id: case.id,
        input: case.input,
        candidate_order,
        input_tensors,
        input_tensor_sha256,
        raw_output: output,
        physical_completion_id: Some(format!("native-{}-{}", group.id, *completion)),
    });
    Ok(actual_nn_inputs)
}

fn hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn pin(value: &str) -> Result<[u8; 32]> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("lowercase SHA-256 required".into());
    }
    Ok(asset::parse_sha256(value)?)
}
fn id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_-.".contains(&b))
}
/// One shared source registry binds this executable, its numerical reader and
/// the independent reference. No local framing/catalog can diverge from V5.
fn source_pins() -> Result<BTreeMap<String, String>> {
    let registered = rz_experiments::compiled_source_components_v5()?;
    [
        "model_native",
        "model_encoding",
        "nn_runtime",
        "model_reference",
        "model_export",
    ]
    .into_iter()
    .map(|name| {
        let pin = registered
            .get(name)
            .ok_or("required common source component absent")?;
        Ok((name.to_string(), pin.clone()))
    })
    .collect()
}
fn outside_git(path: &Path) -> Result<PathBuf> {
    if !path.is_absolute() {
        return Err("capture paths must be absolute".into());
    }
    for ancestor in path.ancestors() {
        if fs::symlink_metadata(ancestor)?.file_type().is_symlink() {
            return Err("capture owner must not traverse original symlink ancestors".into());
        }
    }
    let resolved = path.canonicalize()?;
    for ancestor in resolved.ancestors() {
        if ancestor.join(".git").exists()
            || fs::symlink_metadata(ancestor)?.file_type().is_symlink()
        {
            return Err("capture owner must be outside Git with no symlink ancestors".into());
        }
    }
    Ok(resolved)
}
fn write_exclusive(path: &Path, value: &impl Serialize) -> Result<u64> {
    write_exclusive_bounded(path, value, LIMIT)
}
fn write_exclusive_bounded(path: &Path, value: &impl Serialize, limit: usize) -> Result<u64> {
    let bytes = serde_json::to_vec(value)?;
    if bytes.is_empty() || bytes.len() > limit {
        return Err("capture observation exceeds its registered file limit".into());
    }
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(bytes.len() as u64)
}

fn publish_system_placement_witness(
    root: &Path,
    name: &str,
    observation: &serde_json::Value,
) -> Result<Option<String>> {
    if observation.get("status").and_then(|value| value.as_str()) != Some("verified") {
        return Ok(None);
    }
    let witness = observation
        .get("witness")
        .ok_or("verified system V3 placement lacks its witness")?;
    let bytes = serde_json::to_vec(witness)?;
    // The placement witness can include exact approved transfer evidence and
    // is not the small execution-policy sidecar. Use the same existing bound
    // as other full placement evidence instead of the smaller sidecar limit.
    write_exclusive_bounded(&root.join(name), witness, LIMIT)?;
    Ok(Some(hex(asset::sha256(&bytes))))
}

/// Save the actual first P/C observation immediately as explicitly partial
/// evidence. Later case failure or worker cleanup failure must not erase it.
/// This file neither certifies the capture as complete nor proves GPU/process
/// completion; the joined final observation uses a separate schema and path.
fn publish_initial_system_placement_partial(
    root: &Path,
    plan_sha256: &str,
    inventory_sha256: &str,
    manifest_sha256: &str,
    observation: &serde_json::Value,
) -> Result<u64> {
    let witness_sha = observation
        .get("witness")
        .map(serde_json::to_vec)
        .transpose()?
        .map(|bytes| hex(asset::sha256(&bytes)));
    write_exclusive_bounded(
        &root.join("initial-system-cuda-placement.partial.json"),
        &serde_json::json!({
            "schema":"rovezero.virtual-system-cuda-placement-partial.v1",
            "status":"partial_observation",
            "stage":"first_registered_role_coverage",
            "plan_sha256":plan_sha256,
            "inventory_sha256":inventory_sha256,
            "manifest_sha256":manifest_sha256,
            "observation":observation,
            "embedded_witness_sha256":witness_sha,
            "capture_completion":"not_established",
            "physical_completion":"not_attested",
            "process_exit":"not_observed"
        }),
        LIMIT,
    )
}
fn write_pending(path: &Path, value: &impl Serialize, remaining: u64) -> Result<u64> {
    let bytes = serde_json::to_vec(value)?;
    if bytes.is_empty() || bytes.len() > LIMIT || bytes.len() as u64 > remaining {
        return Err("pending payload exceeds registered file/total limit before creation".into());
    }
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    // A pending file is deliberately not a valid raw-receipt JSON document.
    // The independent reader cannot accept it before profile audit and join.
    file.write_all(PENDING_HEADER)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(bytes.len() as u64)
}
fn publish_pending(pending: &Path, final_path: &Path) -> Result<()> {
    let bytes = asset::read_bounded(pending, LIMIT + PENDING_HEADER.len())?;
    let raw = bytes
        .strip_prefix(PENDING_HEADER)
        .ok_or("pending capture framing differs")?;
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(final_path)?;
    file.write_all(raw)?;
    file.sync_all()?;
    drop(file);
    let actual = asset::read_bounded(final_path, LIMIT)?;
    if actual != raw {
        return Err("published raw receipt integrity differs; retaining pending".into());
    }
    // Only this exact create-new staging file is removed after full recovery
    // bytes are verified. Failed captures retain all original pending files.
    fs::remove_file(pending)?;
    Ok(())
}

enum PostJoinCapturePublication {
    Complete,
    DiagnosticFailure(Box<dyn Error>),
}

/// Called only after the original runtime/placement checks and worker join.
/// A separate policy reporter cannot prevent publishing the unchanged numerical
/// payload. Its failure still makes this command partial, not fully successful.
fn publish_joined_capture(
    publish_raw: impl FnOnce() -> Result<()>,
    publish_policy: impl FnOnce() -> Result<u64>,
) -> Result<PostJoinCapturePublication> {
    publish_raw()?;
    Ok(match publish_policy() {
        Ok(_) => PostJoinCapturePublication::Complete,
        Err(error) => PostJoinCapturePublication::DiagnosticFailure(error),
    })
}

#[cfg(target_os = "linux")]
struct SystemCaptureWorker {
    worker: Option<owned_worker::Worker>,
    reservation: Option<PhysicalRecoveryReservation>,
    recovery_key: Option<PhysicalRecoveryKey>,
    pending: Option<owned_worker::Lease>,
    late_completion: Option<std::result::Result<owned_worker::CaptureReply, BackendError>>,
    release: Option<SessionReleased>,
}

#[cfg(target_os = "linux")]
#[derive(Debug)]
struct SystemPhysicalUnknown {
    detail: &'static str,
    cause: Option<BackendError>,
}

#[cfg(target_os = "linux")]
impl std::fmt::Display for SystemPhysicalUnknown {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.detail)?;
        if let Some(cause) = &self.cause {
            write!(formatter, ": {cause}")?;
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
impl Error for SystemPhysicalUnknown {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.cause
            .as_ref()
            .map(|cause| cause as &(dyn Error + 'static))
    }
}

#[cfg(target_os = "linux")]
#[derive(Debug)]
struct CaptureAndCleanupFailure {
    primary_stage: &'static str,
    primary: Box<dyn Error>,
    cleanup: Box<dyn Error>,
}

#[cfg(target_os = "linux")]
impl std::fmt::Display for CaptureAndCleanupFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "system {} failed: {}; cleanup: {}",
            self.primary_stage, self.primary, self.cleanup
        )
    }
}

#[cfg(target_os = "linux")]
impl Error for CaptureAndCleanupFailure {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.primary.as_ref())
    }
}

#[cfg(target_os = "linux")]
impl SystemCaptureWorker {
    fn wait(
        &mut self,
        mut lease: owned_worker::Lease,
        until: Instant,
    ) -> Result<owned_worker::CaptureReply> {
        loop {
            match lease.poll() {
                PhysicalPoll::Ready(result) => return Ok(result?),
                PhysicalPoll::Quarantined => {
                    let cause = match lease.quarantine_cause() {
                        Ok(cause) => cause,
                        Err(cause) => Some(cause),
                    };
                    self.pending = Some(lease);
                    return Err(Box::new(SystemPhysicalUnknown {
                        detail: "system capture physical completion unknown; original lease retained",
                        cause,
                    }));
                }
                PhysicalPoll::Consumed => {
                    self.pending = Some(lease);
                    return Err(Box::new(SystemPhysicalUnknown {
                        detail: "system capture completion consumed twice; original lease retained",
                        cause: None,
                    }));
                }
                PhysicalPoll::Pending if Instant::now() < until => {
                    std::thread::sleep(Duration::from_millis(1));
                }
                PhysicalPoll::Pending => {
                    self.pending = Some(lease);
                    return Err(Box::new(SystemPhysicalUnknown {
                        detail: "system capture deadline reached; original physical lease retained",
                        cause: None,
                    }));
                }
            }
        }
    }

    fn start(&mut self, until: Instant) -> Result<owned_worker::Started> {
        if Instant::now() >= until {
            return Err("capture window expired before native startup dispatch".into());
        }
        let lease =
            owned_worker::submit_start(self.worker.as_mut().ok_or("system worker absent")?)?;
        match self.wait(lease, until)? {
            owned_worker::CaptureReply::Started(started) => Ok(started),
            owned_worker::CaptureReply::Native(_) => {
                Err("system startup returned a native command result".into())
            }
        }
    }

    fn command(&mut self, request: Command, until: Instant) -> Result<Outcome> {
        if self.pending.is_some() {
            return Err("system capture has an unresolved original physical lease".into());
        }
        if Instant::now() >= until {
            return Err("capture window expired before native dispatch".into());
        }
        let lease = owned_worker::submit_native(
            self.worker.as_mut().ok_or("system worker absent")?,
            request,
        )?;
        match self.wait(lease, until)? {
            owned_worker::CaptureReply::Native(outcome) => Ok(outcome),
            owned_worker::CaptureReply::Started(_) => {
                Err("native command returned a startup result".into())
            }
        }
    }

    fn finish(&mut self, until: Instant) -> Result<()> {
        loop {
            if let Some(lease) = self.pending.as_mut() {
                match lease.poll() {
                    PhysicalPoll::Ready(result) => {
                        // Preserve a late result separately; it does not turn
                        // the earlier deadline or request error into success.
                        self.late_completion = Some(result);
                        self.pending.take();
                    }
                    PhysicalPoll::Quarantined => {
                        let cause = match lease.quarantine_cause() {
                            Ok(cause) => cause,
                            Err(cause) => Some(cause),
                        };
                        return Err(Box::new(SystemPhysicalUnknown {
                            detail: "system capture is quarantined; original lease and owner retained",
                            cause,
                        }));
                    }
                    PhysicalPoll::Consumed => {
                        return Err(Box::new(SystemPhysicalUnknown {
                            detail: "system capture late completion was already consumed",
                            cause: None,
                        }));
                    }
                    PhysicalPoll::Pending => {}
                }
            }
            let worker = self.worker.as_mut().ok_or("system worker absent")?;
            match worker.try_shutdown() {
                Poll::Ready(Ok(())) if self.pending.is_none() => {
                    self.release = Some(worker.session_release().ok_or(
                        "system worker joined without original owner session-release proof",
                    )?);
                    return Ok(());
                }
                Poll::Ready(Ok(())) => {
                    return Err(
                        "system worker joined while original lease remained unresolved".into(),
                    );
                }
                Poll::Ready(Err(error)) => return Err(Box::new(error)),
                Poll::Pending if Instant::now() < until => {
                    std::thread::sleep(Duration::from_millis(1));
                }
                Poll::Pending => {
                    return Err("system worker did not join within cleanup deadline; original owner retained".into());
                }
            }
        }
    }
}

#[cfg(target_os = "linux")]
impl Drop for SystemCaptureWorker {
    fn drop(&mut self) {
        if self.release.is_none() {
            // Keep the exact worker, request lease, reservation and key alive
            // for the remaining process lifetime when no same-owner session
            // release was proven. This does not establish external persistence
            // after process exit or a device-wide GPU fence.
            if let Some(lease) = self.pending.take() {
                std::mem::forget(lease);
            }
            if let Some(worker) = self.worker.take() {
                std::mem::forget(worker);
            }
            if let Some(reservation) = self.reservation.take() {
                std::mem::forget(reservation);
            }
            if let Some(key) = self.recovery_key.take() {
                std::mem::forget(key);
            }
            if let Some(late) = self.late_completion.take() {
                std::mem::forget(late);
            }
        }
    }
}

fn command(worker: &mut Worker, command: Command, until: Instant) -> Result<Outcome> {
    match worker {
        Worker::Legacy(worker) => legacy_command(worker, command, until),
        #[cfg(target_os = "linux")]
        Worker::System(worker) => worker.command(command, until),
    }
}

fn legacy_command(worker: &mut LegacyWorker, command: Command, until: Instant) -> Result<Outcome> {
    if Instant::now() >= until {
        return Err("capture window expired before native dispatch".into());
    }
    let mut lease = worker.submit(command)?;
    loop {
        match lease.poll() {
            PhysicalPoll::Ready(result) => return Ok(result?),
            PhysicalPoll::Quarantined => {
                let cause = lease.quarantine_cause()?;
                std::mem::forget(lease);
                return Err(format!("physical completion unknown: {cause:?}").into());
            }
            PhysicalPoll::Consumed => return Err("completion consumed twice".into()),
            PhysicalPoll::Pending if Instant::now() < until => {
                std::thread::sleep(Duration::from_millis(1))
            }
            PhysicalPoll::Pending => {
                std::mem::forget(lease);
                return Err("finite capture deadline reached; physical owners retained".into());
            }
        }
    }
}
fn stats(worker: &mut Worker, until: Instant) -> Result<PalsBackendStats> {
    match command(worker, Command::SnapshotStats, until)? {
        Outcome::Stats(s) => Ok(s),
        _ => Err("incorrect stats completion".into()),
    }
}
fn finish(worker: &mut Worker, until: Instant) -> Result<()> {
    match worker {
        Worker::Legacy(worker) => finish_legacy(worker, until),
        #[cfg(target_os = "linux")]
        Worker::System(worker) => worker.finish(until),
    }
}

fn finish_legacy(worker: &mut LegacyWorker, until: Instant) -> Result<()> {
    loop {
        match worker.try_shutdown() {
            Poll::Ready(result) => return Ok(result?),
            Poll::Pending if Instant::now() < until => std::thread::sleep(Duration::from_millis(1)),
            Poll::Pending => {
                return Err("native worker did not join within cleanup deadline".into());
            }
        }
    }
}
fn identity(
    input: &PalsVirtualReviewInput,
    epoch: [u8; 32],
    backend: [u8; 32],
) -> Result<ReviewExecutionIdentity> {
    // Numerical tensor fixtures are not Rules states. These explicit fixture
    // namespaces prevent cache aliasing and grant no Rules/history authority.
    let anchors = input
        .anchors
        .iter()
        .map(|anchor| {
            let bytes = serde_json::to_vec(anchor)?;
            let mut hash = Sha256::new();
            hash.update(b"rz-pals-numeric-fixture-anchor-no-rules-history-claim/1");
            hash.update(bytes);
            Ok(hash.finalize().into())
        })
        .collect::<Result<Vec<[u8; 32]>>>()?;
    Ok(ReviewExecutionIdentity {
        model_epoch: epoch,
        backend,
        fp32: true,
        rules_history: anchors[0],
        anchor_history_seals: anchors,
    })
}
fn execute(config: Config) -> Result<()> {
    let selected = config.execution_config()?;
    let is_cuda = matches!(selected.provider, Provider::Cuda { .. });
    let system_cuda = config.provider == "cuda-system";
    let system_control = system_cuda && config.system_cuda_inventory.is_some();
    for path in [
        &config.case_plan,
        &config.export_manifest,
        &config.runtime_library,
        &config.runtime_cache,
        &config.output_root,
    ]
    .into_iter()
    .chain(config.cuda_bundle.as_ref())
    .chain(config.cuda_inventory.as_ref())
    .chain(config.system_cuda_config.as_ref())
    .chain(config.system_cuda_inventory.as_ref())
    {
        if !path.is_absolute() {
            return Err("all configured capture asset/output paths must be absolute".into());
        }
    }
    if !(1..=7200).contains(&config.deadline_seconds) || !(1..=30).contains(&config.cleanup_seconds)
    {
        return Err("capture requires registered finite execution/cleanup windows".into());
    }
    let until = Instant::now() + Duration::from_secs(config.deadline_seconds);
    let plan_bytes = asset::read_bounded(&config.case_plan, LIMIT)?;
    if asset::sha256(&plan_bytes) != pin(&config.case_plan_sha256)? {
        return Err("actual case plan SHA differs".into());
    }
    let raw_plan: serde_json::Value = decode_unique(&plan_bytes)?;
    let binding_keys = raw_plan
        .get("bindings")
        .and_then(|v| v.as_object())
        .ok_or("plan bindings missing")?;
    if system_cuda {
        if binding_keys.contains_key("cuda_manifest_sha256")
            || binding_keys.contains_key("cuda_libraries")
            || !binding_keys.contains_key("system_cuda_config_sha256")
            || !binding_keys.contains_key("system_cuda_libraries")
        {
            return Err("system plan must use only system CUDA bindings".into());
        }
    } else if !binding_keys.contains_key("cuda_manifest_sha256")
        || !binding_keys.contains_key("cuda_libraries")
        || binding_keys.contains_key("system_cuda_config_sha256")
        || binding_keys.contains_key("system_cuda_libraries")
    {
        return Err("V5 plan must retain legacy binding shape".into());
    }
    let plan: Plan = serde_json::from_value(raw_plan)?;
    if plan.domain
        != if system_cuda {
            "rz-pals-numerical-case-plan-v6/1"
        } else {
            "rz-pals-numerical-case-plan-v5/1"
        }
        || plan.schema_version != if system_cuda { 6 } else { 5 }
        || plan.invocation != "fresh"
        || plan.batch != 1
        || plan.groups.is_empty()
        || plan.groups.len() > 18
        || plan.bindings.source_components != source_pins()?
    {
        return Err("exact Fresh B1 plan/source registration required".into());
    }
    let mut ids = BTreeSet::new();
    let mut cases = 0usize;
    for group in &plan.groups {
        if !id(&group.id)
            || !id(&format!("native-{}", group.id))
            || !ids.insert(group.id.clone())
            || !model::LENGTH_BUCKETS.contains(&group.line_bucket)
            || group.cases.is_empty()
            || group.cases.len() > 10
        {
            return Err("bounded unique role/bucket group required".into());
        }
        for case in &group.cases {
            let Input::VirtualReview(input) = &case.input;
            let prepared = input.prepare()?;
            cases += 1;
            if !id(&case.id)
                || !ids.insert(case.id.clone())
                || input.role != group.role
                || prepared.length_bucket != group.line_bucket
                || input.candidates.is_empty()
                || ![0.0, 1.0].contains(&input.anchors[0].metadata[0])
                || cases > 256
            {
                return Err("case role/bucket/order capacity differs".into());
            }
            let captured = tensors::capture(input)?;
            tensors::digest(&captured, input.role)?;
        }
    }
    let b = &plan.bindings;
    if b.model_epoch == 0 {
        return Err(
            "declared model epoch must be nonzero; actual identity is the pinned checkpoint SHA"
                .into(),
        );
    }
    if b.model_profile != model::PROFILE
        || b.model_schema != model::MODEL_SCHEMA
        || b.model_semantics != model::MODEL_SEMANTICS
        || b.encoding_schema != model::ENCODING
        || b.precision != "fp32"
        || b.tf32
        || b.backend
            != if system_cuda {
                "ort_cuda_system"
            } else if is_cuda {
                "ort_cuda"
            } else {
                "ort_cpu"
            }
        || b.model_semantic_sha256
            != hex(PalsModelProfile::VirtualReviewV1.registered_semantics_sha256())
        || b.encoding_sha256 != hex(model::registered_encoding_sha256())
    {
        return Err("registered model/precision/backend semantics differ".into());
    }
    for value in [
        &b.weights_sha256,
        &b.export_manifest_sha256,
        &b.runtime_library_sha256,
        &b.encoding_sha256,
    ] {
        pin(value)?;
    }
    let mode = if b.graphs.contains_key("validator") {
        VirtualReviewExecutionMode::Collection
    } else {
        VirtualReviewExecutionMode::Deployment
    };
    let roles = plan
        .groups
        .iter()
        .map(|g| format!("{:?}", g.role))
        .collect::<BTreeSet<_>>();
    let expected = BTreeSet::from_iter(
        ["Proposer".to_string(), "Critic".to_string()]
            .into_iter()
            .chain(
                (mode == VirtualReviewExecutionMode::Collection).then_some("Validator".to_string()),
            ),
    );
    if roles != expected {
        return Err("capture plan omits or adds an actual registered role".into());
    }
    let order = capture_order(
        &plan.groups,
        is_cuda && (!system_cuda || system_control),
        mode == VirtualReviewExecutionMode::Collection,
    )?;
    let mut profiling_window = (is_cuda && (!system_cuda || system_control)).then(|| CudaProfilingWindow {
        scope: "original planned cases only; profiles close at first registered role coverage, not full-run kernel coverage",
        actual_cases: order.slots[..order.profile_cases].iter().map(|slot| {
            let group = &plan.groups[slot.group];
            let case = &group.cases[slot.case];
            let Input::VirtualReview(input) = &case.input;
            ProfiledCase {
                group: group.id.clone(),
                case: case.id.clone(),
                role: input.role,
                line_bucket: group.line_bucket,
                records: input.records.len(),
                anchors: input.anchors.len(),
            }
        }).collect(),
        completed_nn_inputs: 0,
        remaining_cases_profiled: false,
    });
    let manifest_bytes = asset::read_bounded(&config.export_manifest, 256 * 1024)?;
    let summary = validate_virtual_review_export_manifest(&manifest_bytes, mode)?;
    let graph_pins = summary
        .graphs
        .iter()
        .map(|g| (g.role.clone(), hex(g.sha256)))
        .collect::<BTreeMap<_, _>>();
    if hex(summary.manifest_sha256()) != b.export_manifest_sha256
        || hex(summary.checkpoint_sha256) != b.weights_sha256
        || graph_pins != b.graphs
    {
        return Err("actual strict export/checkpoint/graph/encoding pins differ".into());
    }
    if is_cuda && !system_cuda {
        return Err("legacy CUDA bundle execution disabled; use a V6 system CUDA plan".into());
    }
    if is_cuda && std::env::var("NVIDIA_TF32_OVERRIDE").ok().as_deref() != Some("0") {
        return Err("CUDA capture requires NVIDIA_TF32_OVERRIDE=0".into());
    }
    if !system_cuda {
        if config.cuda_bundle.is_some()
            || config.cuda_inventory.is_some()
            || config.cuda_inventory_sha256.is_some()
            || b.cuda_manifest_sha256.is_some()
            || !b.cuda_libraries.is_empty()
        {
            return Err("CPU cannot carry CUDA assets".into());
        }
    }
    let control = match (
        &config.system_cuda_inventory,
        &config.system_cuda_inventory_sha256,
    ) {
        (Some(path), Some(digest)) => Some(PalsCudaControlPolicy::from_virtual_review_inventory(
            path, digest,
        )?),
        (None, None) => None,
        _ => return Err("system CUDA control inventory selection is incomplete".into()),
    };
    let runtime = if system_cuda {
        #[cfg(target_os = "linux")]
        {
            let bytes = asset::read_bounded(
                config
                    .system_cuda_config
                    .as_ref()
                    .ok_or("system config missing")?,
                64 * 1024,
            )?;
            let config_sha = hex(asset::sha256(&bytes));
            if Some(&config_sha) != config.system_cuda_config_sha256.as_ref()
                || Some(&config_sha) != b.system_cuda_config_sha256.as_ref()
            {
                return Err("system config differs from pinned plan".into());
            }
            let system: SystemCudaConfig = decode_unique(&bytes)?;
            if system.max_capture_ms == 0
                || system
                    .libraries
                    .iter()
                    .any(|entry| !entry.path.is_absolute())
            {
                return Err("system library paths and finite capture limit required".into());
            }
            let owner = SystemRuntime::capture(
                system
                    .libraries
                    .into_iter()
                    .map(|entry| SystemLibraryRequest {
                        role: entry.role,
                        path: entry.path,
                        required_mapping: entry.required_mapping,
                    })
                    .collect(),
                SystemCaptureLimits {
                    max_libraries: system.max_libraries,
                    max_file_bytes: system.max_file_bytes,
                    max_total_bytes: system.max_total_bytes,
                    max_elapsed: Duration::from_millis(system.max_capture_ms),
                },
            )?;
            let mut observed = owner
                .libraries()
                .iter()
                .map(|entry| SystemLibraryBinding {
                    role: entry.role.clone(),
                    sha256: hex(entry.sha256),
                    bytes: entry.bytes,
                    required_mapping: entry.required_mapping,
                })
                .collect::<Vec<_>>();
            observed.sort_by(|a, b| a.role.cmp(&b.role));
            let expected = b
                .system_cuda_libraries
                .as_ref()
                .ok_or("system roster missing")?;
            if expected != &observed || observed.windows(2).any(|w| w[0].role >= w[1].role) {
                return Err("captured system library roster differs from plan".into());
            }
            let core = owner
                .libraries()
                .into_iter()
                .find(|entry| entry.role == "ort-core" && entry.required_mapping)
                .ok_or("required ort-core missing")?;
            if hex(core.sha256) != b.runtime_library_sha256
                || fs::canonicalize(&config.runtime_library)? != core.canonical_path
            {
                return Err("system ORT core differs from plan".into());
            }
            if Instant::now() >= until {
                return Err("capture window expired before system runtime load".into());
            }
            // Merely capture system FDs here. The first ORT entry occurs on
            // the retained worker after its original custody is registered.
            PreparedRuntime::System(owner)
        }
        #[cfg(not(target_os = "linux"))]
        {
            return Err("system CUDA capture requires Linux".into());
        }
    } else {
        let cache = RuntimeCache::open(&config.runtime_cache)?;
        let runtime_pin = cache.library(&config.runtime_library, &b.runtime_library_sha256)?;
        if Instant::now() >= until {
            return Err("capture window expired before runtime load".into());
        }
        PreparedRuntime::Legacy(OrtRuntime::load(&runtime_pin)?)
    };
    if system_cuda {
        selected.validate_system_execution_selection()?;
    }
    let output_owner =
        capture_output_root(&config.output_root, config.output_root_admission.as_ref())?;
    let root = &output_owner.path;
    if Instant::now() >= until {
        return Err("capture window expired before model/session load".into());
    }
    let (mut worker, epoch, backend_digest, policy_observation) = match runtime {
        PreparedRuntime::Legacy(runtime) => {
            if hex(runtime.binary_digest()) != b.runtime_library_sha256 {
                return Err("actual loaded ORT pin differs".into());
            }
            selected.validate_execution_selection(
                runtime.native_loading_profile(),
                control.is_some(),
            )?;
            let backend = match (mode, control) {
                (VirtualReviewExecutionMode::Collection, Some(policy)) => {
                    PalsVirtualReviewOnnxBackend::load_collection_with_cuda_control_policy(
                        &config.export_manifest,
                        &b.export_manifest_sha256,
                        runtime,
                        selected,
                        policy,
                        &root.join("profiles"),
                    )?
                }
                (_, Some(policy)) => PalsVirtualReviewOnnxBackend::load_with_cuda_control_policy(
                    &config.export_manifest,
                    &b.export_manifest_sha256,
                    runtime,
                    selected,
                    policy,
                    &root.join("profiles"),
                )?,
                (VirtualReviewExecutionMode::Collection, None) => {
                    PalsVirtualReviewOnnxBackend::load_collection(
                        &config.export_manifest,
                        &b.export_manifest_sha256,
                        runtime,
                        selected,
                    )?
                }
                (_, None) => PalsVirtualReviewOnnxBackend::load(
                    &config.export_manifest,
                    &b.export_manifest_sha256,
                    runtime,
                    selected,
                )?,
            };
            let epoch = backend.model_epoch();
            let backend_digest = backend.backend_digest();
            let observation = loaded_policy_observation(
                &backend,
                config.cudnn_conv_search.as_deref(),
                is_cuda && !system_cuda,
            )?;
            (
                Worker::Legacy(backend.into_worker()?),
                epoch,
                backend_digest,
                observation,
            )
        }
        #[cfg(target_os = "linux")]
        PreparedRuntime::System(runtime) => {
            let custody = system_startup_custody(
                &config.export_manifest,
                &summary,
                &output_owner,
                system_control,
            )?;
            let (mut reservation, key) = PhysicalRecoveryReservation::reserve()?;
            let mut original = Some(custody);
            let startup = owned_worker::Startup {
                runtime,
                expected_runtime_sha256: pin(&b.runtime_library_sha256)?,
                expected_manifest_sha256: b.export_manifest_sha256.clone(),
                config: selected,
                mode,
                kind: VirtualReviewArtifactKind::Fresh,
                policy: control,
                requested_cudnn_conv_search: config.cudnn_conv_search.clone(),
                until,
            };
            let physical = match owned_worker::spawn(&mut original, &mut reservation, startup) {
                Ok(worker) => worker,
                Err(error) => {
                    // Spawn failure is before worker/native entry; only the
                    // unpublished original reservation may be canceled after
                    // the exact custody was returned to its original slot.
                    if original.is_some() {
                        if let Err((owner, cancel_error)) = reservation.try_cancel_unpublished() {
                            let _ = writeln!(
                                std::io::stderr().lock(),
                                "RZ_NUMERIC_UNPUBLISHED_RESERVATION_RETAINED={cancel_error:?}"
                            );
                            std::mem::forget(owner);
                        }
                    } else {
                        let _ = writeln!(
                            std::io::stderr().lock(),
                            "RZ_NUMERIC_RESERVATION_RETAINED={{\"reason\":\"original_custody_unavailable\"}}"
                        );
                        std::mem::forget(reservation);
                    }
                    return Err(Box::new(error));
                }
            };
            let mut owner = SystemCaptureWorker {
                worker: Some(physical),
                reservation: Some(reservation),
                recovery_key: Some(key),
                pending: None,
                late_completion: None,
                release: None,
            };
            let started = match owner.start(until) {
                Ok(started) => started,
                Err(startup) => {
                    let cleanup_until =
                        Instant::now() + Duration::from_secs(config.cleanup_seconds);
                    return match owner.finish(cleanup_until) {
                        Ok(()) => Err(startup),
                        Err(cleanup) => Err(Box::new(CaptureAndCleanupFailure {
                            primary_stage: "startup",
                            primary: startup,
                            cleanup,
                        })),
                    };
                }
            };
            (
                Worker::System(owner),
                started.epoch,
                started.backend_digest,
                started.policy_observation,
            )
        }
    };
    let mut system_mapping = None;
    #[cfg(target_os = "linux")]
    let mut system_placement: Option<(serde_json::Value, serde_json::Value)> = None;
    #[cfg(not(target_os = "linux"))]
    let system_placement: Option<(serde_json::Value, serde_json::Value)> = None;
    #[cfg(target_os = "linux")]
    let mut initial_partial_failure: Option<String> = None;
    #[cfg(not(target_os = "linux"))]
    let initial_partial_failure: Option<String> = None;
    let work = (|| -> Result<Vec<(PathBuf, PathBuf)>> {
        let mut outputs = (0..plan.groups.len()).map(|_| None).collect::<Vec<_>>();
        let mut bytes = 0u64;
        let mut completion = 0u64;
        let mut stages = BTreeMap::new();
        let mut groups = plan
            .groups
            .into_iter()
            .map(GroupCapture::from)
            .collect::<Vec<_>>();
        let mut first_placement = None;
        #[cfg(target_os = "linux")]
        let mut first_system_placement = None;
        for (index, slot) in order.slots.iter().enumerate() {
            let actual_nn_inputs = capture_case(
                &mut worker,
                &mut groups[slot.group],
                slot.case,
                epoch,
                backend_digest,
                &mut completion,
                until,
            )?;
            if index < order.profile_cases {
                let window = profiling_window
                    .as_mut()
                    .ok_or("CUDA profile window absent")?;
                window.completed_nn_inputs = window
                    .completed_nn_inputs
                    .checked_add(actual_nn_inputs)
                    .ok_or("profile window NN input overflow")?;
            }
            if is_cuda && !system_cuda && index + 1 == order.profile_cases {
                first_placement = Some(
                    match command(&mut worker, Command::VerifyVirtualCudaPlacement, until)? {
                        Outcome::VirtualCudaPlacementVerified(witness) => witness,
                        _ => return Err("actual initial V3 placement witness absent".into()),
                    },
                );
            }
            #[cfg(target_os = "linux")]
            if system_control && index + 1 == order.profile_cases {
                let first_observation = match command(
                    &mut worker,
                    Command::ObserveSystemVirtualCudaPlacement,
                    until,
                )? {
                    Outcome::SystemVirtualCudaPlacementObserved(observation) => {
                        system_placement_observation_json(&observation)
                    }
                    _ => return Err("system V3 placement observation absent".into()),
                };
                let saved = output_owner.verify().and_then(|()| {
                    publish_initial_system_placement_partial(
                        root,
                        &config.case_plan_sha256,
                        config
                            .system_cuda_inventory_sha256
                            .as_deref()
                            .ok_or("system CUDA inventory digest absent")?,
                        &b.export_manifest_sha256,
                        &first_observation,
                    )
                    .map(|_| ())
                });
                if let Err(error) = saved {
                    let detail = error.to_string().chars().take(1024).collect::<String>();
                    let _ = writeln!(
                        std::io::stderr().lock(),
                        "RZ_NUMERIC_INITIAL_PLACEMENT_PARTIAL_FAILURE={}",
                        serde_json::json!({"detail":detail,"capture_completion":"not_established", "physical_completion":"unknown"})
                    );
                    initial_partial_failure = Some(detail);
                }
                first_system_placement = Some(first_observation);
            }
            let group = &mut groups[slot.group];
            if group.remaining.iter().any(Option::is_some) {
                continue;
            }
            // Publish each completed group's pending bytes immediately. Only
            // its early representative remains held while another group runs;
            // do not accumulate all raw tensors until the complete plan ends.
            let captured = group
                .captured
                .iter_mut()
                .map(Option::take)
                .collect::<Option<Vec<_>>>()
                .ok_or("original capture case lacks a physical completion")?;
            let stage = std::mem::take(&mut group.stage);
            stage.validate_fresh_count(captured.len())?;
            let completed_nn_inputs = stage.completed_nn_inputs;
            let public_cache_hits = stage.public_cache_hits;
            stages.insert(group.id.clone(), stage);
            let raw = RawRun {
                schema_version: if system_cuda { 6 } else { 5 },
                domain: if system_cuda {
                    "rz-pals-numerical-raw-run-v6/1"
                } else {
                    "rz-pals-numerical-raw-run-v5/1"
                },
                run_id: format!("native-{}", group.id),
                producer: "rust_ort_native",
                producer_source_component: b.source_components["model_native"].clone(),
                bindings: b.clone(),
                invocation: "fresh",
                batch: 1,
                observed_provider: Some(if is_cuda {
                    "CUDAExecutionProvider"
                } else {
                    "CPUExecutionProvider"
                }),
                completed_nn_inputs,
                public_cache_hits,
                physical_completion_known: true,
                cases: captured,
            };
            let pending = root.join(format!("native-{}.pending.json", group.id));
            let final_path = root.join(format!("native-{}.json", group.id));
            bytes = bytes
                .checked_add(write_pending(&pending, &raw, RAW_TOTAL - bytes)?)
                .filter(|n| *n <= RAW_TOTAL)
                .ok_or("raw collection exceeds 64MiB")?;
            if outputs[slot.group].replace((pending, final_path)).is_some() {
                return Err("original capture group was published twice".into());
            }
        }
        if completion != order.slots.len() as u64
            || groups
                .iter()
                .any(|group| group.remaining.iter().any(Option::is_some))
        {
            return Err("physical capture count differs from original plan".into());
        }
        let mut outputs = outputs
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or("original capture plan has an unpublished group")?;
        if system_cuda {
            match command(&mut worker, Command::ObserveRuntimeMapping, until)? {
                Outcome::RuntimeMappingObserved {
                    mapping,
                    system_placement,
                } => {
                    system_mapping = Some(serde_json::json!({
                        "schema":"rovezero.virtual-system-mapping-observation.v1",
                        "mapping":format!("{mapping:?}"), "placement_prerequisites":format!("{system_placement:?}"),
                        "gpu_kernel_placement":"unknown", "process_exit":"requires_external_supervisor"
                    }));
                }
                _ => return Err("system runtime mapping observation unavailable".into()),
            }
            #[cfg(target_os = "linux")]
            if system_control {
                let final_observation = match command(
                    &mut worker,
                    Command::ObserveSystemVirtualCudaPlacement,
                    until,
                )? {
                    Outcome::SystemVirtualCudaPlacementObserved(observation) => {
                        system_placement_observation_json(&observation)
                    }
                    _ => return Err("final system V3 placement observation absent".into()),
                };
                system_placement = Some((
                    first_system_placement.ok_or("first system V3 placement observation absent")?,
                    final_observation,
                ));
            }
        } else {
            match command(&mut worker, Command::VerifyRuntime, until)? {
                Outcome::RuntimeVerified => {}
                _ => return Err("runtime audit incomplete".into()),
            }
        }
        if is_cuda && !system_cuda {
            match command(&mut worker, Command::VerifyVirtualCudaPlacement, until)? {
                Outcome::VirtualCudaPlacementVerified(witness) => {
                    if first_placement.as_ref() != Some(&witness) {
                        return Err(
                            "final V3 placement differs from original profiled witness".into()
                        );
                    }
                    write_exclusive(&root.join("actual-virtual-cuda-placement.json"), &witness)?;
                }
                _ => return Err("actual V3 placement witness absent".into()),
            }
        }
        let stage_pending = root.join("actual-stage-counters.pending.json");
        write_pending(&stage_pending, &stages, LIMIT as u64)?;
        outputs.push((stage_pending, root.join("actual-stage-counters.json")));
        Ok(outputs)
    })();
    let joined = finish(
        &mut worker,
        Instant::now() + Duration::from_secs(config.cleanup_seconds),
    );
    let outputs = match (work, joined) {
        (Ok(outputs), Ok(())) => outputs,
        (Err(work), Err(join)) => {
            #[cfg(target_os = "linux")]
            if system_cuda {
                return Err(Box::new(CaptureAndCleanupFailure {
                    primary_stage: "capture",
                    primary: work,
                    cleanup: join,
                }));
            }
            return Err(format!(
                "capture failed: {work}; physical cleanup failed: {join}; pending files retained"
            )
            .into());
        }
        (Err(work), Ok(())) => {
            #[cfg(target_os = "linux")]
            if system_cuda {
                let _ = writeln!(
                    std::io::stderr().lock(),
                    "RZ_NUMERIC_SYSTEM_WORKER_JOINED_AFTER_FAILURE={{\"scope\":\"same_original_native_session\"}}"
                );
                return Err(work);
            }
            return Err(format!(
                "capture failed: {work}; native worker joined; pending files retained"
            )
            .into());
        }
        (Ok(_), Err(join)) => {
            return Err(format!("physical cleanup failed: {join}; pending files retained").into());
        }
    };
    let original_system_session_release = match &worker {
        Worker::Legacy(_) => None,
        #[cfg(target_os = "linux")]
        Worker::System(owner) => owner
            .release
            .map(|release| SystemSessionReleaseObservation {
                completed_synchronous_runs: release.completed_synchronous_runs,
                scope: "same_original_native_session_after_successful_worker_join",
                process_exit: "requires_external_supervisor",
                device_wide_gpu_fence: "not_established",
            }),
    };
    output_owner.verify()?;
    let publication = publish_joined_capture(
        || {
            for (pending, final_path) in outputs {
                if final_path.exists() {
                    return Err("refusing final receipt overwrite".into());
                }
                publish_pending(&pending, &final_path)?;
            }
            output_owner.verify()
        },
        || {
            // This small observation is separate from the unchanged numerical
            // V5 payload. Required runtime/placement checks and worker join have
            // already succeeded; consumers requiring this sidecar must still
            // reject its absence or a partial write under their own contract.
            if let Some(observation) = &system_mapping {
                write_exclusive_bounded(
                    &root.join("system-runtime-mapping.json"),
                    observation,
                    POLICY_LIMIT,
                )?;
            }
            if let Some((first, final_observation)) = &system_placement {
                // Preserve the actual typed witness in its own unchanged
                // representation. Unknown mappings have no invented witness.
                let first_witness_sha = publish_system_placement_witness(
                    root,
                    "initial-virtual-system-cuda-placement.json",
                    first,
                )?;
                let witness_sha = publish_system_placement_witness(
                    root,
                    "actual-virtual-system-cuda-placement.json",
                    final_observation,
                )?;
                write_exclusive_bounded(
                    &root.join("system-cuda-placement-observation.json"),
                    &serde_json::json!({
                        "schema":"rovezero.virtual-system-cuda-placement-observation.v1",
                        "plan_sha256":config.case_plan_sha256,
                        "inventory_sha256":config.system_cuda_inventory_sha256,
                        "manifest_sha256":b.export_manifest_sha256,
                        "first_registered_role_coverage_status":first.get("status"),
                        "initial_witness_sha256":first_witness_sha,
                        "initial_partial_observation":if initial_partial_failure.is_some() {"write_failed"} else {"written"},
                        "initial_partial_error":initial_partial_failure,
                        "final_status":final_observation.get("status"),
                        "witness_sha256":witness_sha,
                        "profile_scope":"original planned cases through first registered role coverage",
                        "physical_completion":"requires_same_native_owner_completion",
                        "process_exit":"requires_external_supervisor"
                    }),
                    POLICY_LIMIT,
                )?;
            }
            write_exclusive_bounded(
                &root.join("actual-cudnn-execution-policy.json"),
                &JoinedPolicyObservation {
                    domain: "rz-pals-numerical-capture-execution-policy/1",
                    scope: "loaded_request_and_session_settings_not_selected_cudnn_plan",
                    loaded: policy_observation,
                    captured_physical_completions_known: true,
                    native_worker_joined: true,
                    actual_cudnn_plan: "unknown",
                    tf32_actual: "unknown",
                    numerical_acceptance: false,
                    performance_improvement_claim: false,
                    process_exit_evidence: "requires_external_process_owner_exit_observation",
                    cuda_profiling_window: profiling_window,
                    original_system_session_release,
                },
                POLICY_LIMIT,
            )
        },
    )?;
    if let PostJoinCapturePublication::DiagnosticFailure(error) = &publication {
        // Preserve the reporter failure before the final required owner check,
        // so a later owner error remains primary without hiding this secondary.
        let detail = error.to_string().chars().take(1024).collect::<String>();
        let _ = writeln!(
            std::io::stderr().lock(),
            "RZ_NUMERIC_CAPTURE_DIAGNOSTIC_FAILURE={}",
            serde_json::json!({"stage":"post_join_policy_observation","detail":detail,
                "raw_payload_published":true,"full_capture_success":false})
        );
    }
    output_owner.verify()?;
    if let PostJoinCapturePublication::DiagnosticFailure(error) = publication {
        println!(
            "RZ_NUMERIC_CAPTURE_STATUS={{\"status\":\"captured_with_diagnostic_failure\",\"numerical_acceptance\":false,\"rules_legality_certified\":false,\"native_worker_joined\":true,\"raw_payload_published\":true,\"diagnostics_complete\":false,\"diagnostic_failure\":\"post_join_policy_observation\"}}"
        );
        return Err(error);
    }
    println!(
        "RZ_NUMERIC_CAPTURE_STATUS={{\"status\":\"captured\",\"numerical_acceptance\":false,\"rules_legality_certified\":false,\"native_worker_joined\":true}}"
    );
    Ok(())
}
fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args == ["--source-pins"] {
        println!("{}", serde_json::to_string(&source_pins()?)?);
        return Ok(());
    }
    if args == ["--model-registry"] {
        // Static registry methods only: no asset, runtime or model is opened.
        // The consumer must separately join the actual build and source bytes.
        println!(
            "{}",
            serde_json::json!({
                "schema": "rz-pals-report-model-registry-query/1",
                "model_profile": model::PROFILE,
                "model_schema": model::MODEL_SCHEMA,
                "model_semantics": model::MODEL_SEMANTICS,
                "model_semantic_sha256": hex(PalsModelProfile::VirtualReviewV1.registered_semantics_sha256()),
                "encoding_schema": model::ENCODING,
                "encoding_sha256": hex(model::registered_encoding_sha256()),
                "registry_methods": [
                    "PalsModelProfile::VirtualReviewV1.registered_semantics_sha256",
                    "pals_model::virtual_review::registered_encoding_sha256"
                ],
                "source_provenance": "caller_must_join_actual_path_dependency_bytes_to_registered_source",
                "model_initialized": false,
                "ort_loaded": false,
                "neural_executed": false,
                "training_executed": false,
                "numerical_authority_issued": false
            })
        );
        return Ok(());
    }
    if args.len() != 1 {
        return Err(
            "usage: virtual_review_numeric_capture --source-pins | --model-registry | ABSOLUTE_CAPTURE_CONFIG.json"
                .into(),
        );
    }
    let path = Path::new(&args[0]);
    if !path.is_absolute() {
        return Err("capture config must be absolute".into());
    }
    let bytes = asset::read_bounded(path, 64 * 1024)?;
    let config: Config = decode_unique(&bytes)?;
    execute(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joined_capture_policy_failure_keeps_the_required_raw_publication() {
        let calls = std::cell::RefCell::new(Vec::new());
        let result = publish_joined_capture(
            || {
                calls.borrow_mut().push("raw");
                Ok(())
            },
            || {
                calls.borrow_mut().push("policy");
                Err("policy reporter unavailable".into())
            },
        )
        .unwrap();
        assert_eq!(*calls.borrow(), ["raw", "policy"]);
        let PostJoinCapturePublication::DiagnosticFailure(error) = result else {
            panic!("reporter failure must not become complete capture success");
        };
        assert_eq!(error.to_string(), "policy reporter unavailable");
    }

    #[test]
    fn joined_capture_required_raw_failure_does_not_run_the_policy_reporter() {
        let policy_called = std::cell::Cell::new(false);
        let result = publish_joined_capture(
            || Err("required raw integrity failure".into()),
            || {
                policy_called.set(true);
                Ok(1)
            },
        );
        assert_eq!(
            result.err().unwrap().to_string(),
            "required raw integrity failure"
        );
        assert!(!policy_called.get());
    }

    #[test]
    fn joined_capture_success_publishes_raw_before_the_separate_policy() {
        let calls = std::cell::RefCell::new(Vec::new());
        let result = publish_joined_capture(
            || {
                calls.borrow_mut().push("raw");
                Ok(())
            },
            || {
                calls.borrow_mut().push("policy");
                Ok(1)
            },
        )
        .unwrap();
        assert!(matches!(result, PostJoinCapturePublication::Complete));
        assert_eq!(*calls.borrow(), ["raw", "policy"]);
    }

    fn schedule_group(role: PalsRole, bucket: usize) -> Group {
        let role_name = format!("{role:?}");
        let cases = (0..if bucket == 8 { 10 } else { 8 })
            .map(|index| {
                let token = PalsCandidateToken {
                    from: 48,
                    to: 56,
                    promotion: 1,
                };
                let anchors = if index == 1 { 1 } else { 4 };
                let input = PalsVirtualReviewInput {
                    schema: model::INPUT_SCHEMA.into(),
                    role,
                    anchors: (0..anchors)
                        .map(|_| model::ReviewAnchorInput {
                            board: vec![0; 64],
                            metadata: [0.0; 16],
                        })
                        .collect(),
                    records: if index == 0 {
                        Vec::new()
                    } else {
                        vec![model::ReviewRecordInput {
                            kind: model::ReviewRecordKind::Counterexample,
                            perspective: 1,
                            completed_iteration: true,
                            rules_terminal: false,
                            line: vec![token],
                            anchor: 0,
                            parent: None,
                            supersedes: None,
                        }]
                    },
                    query: model::ReviewQueryInput {
                        purpose: if role == PalsRole::Validator {
                            model::ReviewQueryPurpose::TaskPolicy
                        } else {
                            model::ReviewQueryPurpose::ReplyPolicy
                        },
                        perspective: 1,
                        prefix: Vec::new(),
                        proposal: vec![token; bucket],
                        counter: Vec::new(),
                        counter_present: false,
                        anchors: [0; 3],
                    },
                    candidates: vec![token],
                    divergences: Vec::new(),
                    token_budget: 512,
                };
                Case {
                    id: format!("{role_name}_L{bucket}_{index}"),
                    input: Input::VirtualReview(input),
                }
            })
            .collect();
        Group {
            id: format!("{role_name}_L{bucket}"),
            role,
            line_bucket: bucket,
            cases,
        }
    }

    fn cumulative_stats(calls: u64) -> PalsBackendStats {
        PalsBackendStats {
            admitted_role_requests: calls,
            public_cache_misses: calls,
            public_nn_runs_attempted: calls,
            public_nn_runs_completed: calls,
            role_nn_runs_attempted: calls,
            role_nn_runs_completed: calls,
            validated_public_outputs: calls,
            validated_role_outputs: calls,
            completed_nn_inputs: 2 * calls,
            ..PalsBackendStats::default()
        }
    }

    #[test]
    fn short_original_profile_prefix_covers_roles_without_repeating_or_reordering_raw_cases() {
        for collection in [false, true] {
            let roles = [PalsRole::Proposer, PalsRole::Critic, PalsRole::Validator];
            let count = if collection { 3 } else { 2 };
            let groups = roles[..count]
                .iter()
                .flat_map(|role| {
                    model::LENGTH_BUCKETS
                        .iter()
                        .map(move |bucket| schedule_group(*role, *bucket))
                })
                .collect::<Vec<_>>();
            let expected = groups
                .iter()
                .map(|group| {
                    group
                        .cases
                        .iter()
                        .map(|case| case.id.clone())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let schedule = capture_order(&groups, true, collection).unwrap();
            assert_eq!(schedule.profile_cases, count);
            assert_eq!(schedule.slots.len(), 50 * count);
            assert_eq!(
                schedule
                    .slots
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>()
                    .len(),
                50 * count
            );
            for (slot, role) in schedule.slots[..count].iter().zip(&roles[..count]) {
                let group = &groups[slot.group];
                let Input::VirtualReview(input) = &group.cases[slot.case].input;
                assert_eq!(&group.role, role);
                assert_eq!(group.line_bucket, 8);
                assert_eq!(slot.case, 2);
                assert!(!input.records.is_empty());
                assert_eq!(input.anchors.len(), 4);
            }
            let mut actual = groups
                .into_iter()
                .map(GroupCapture::from)
                .collect::<Vec<_>>();
            let mut restored = expected
                .iter()
                .map(|cases| vec![None; cases.len()])
                .collect::<Vec<_>>();
            let mut profile_nn_inputs = 0;
            for (sequence, slot) in schedule.slots.iter().enumerate() {
                let case = actual[slot.group].remaining[slot.case]
                    .take()
                    .expect("each original case once");
                restored[slot.group][slot.case] = Some(case.id);
                let delta = stage_delta(
                    &cumulative_stats(sequence as u64),
                    &cumulative_stats(sequence as u64 + 1),
                    1,
                )
                .unwrap();
                if sequence < schedule.profile_cases {
                    profile_nn_inputs += delta.completed_nn_inputs;
                }
                actual[slot.group].stage.checked_add(delta).unwrap();
            }
            assert_eq!(profile_nn_inputs, 2 * count as u64);
            for ((group, restored), expected) in actual.iter().zip(restored).zip(expected) {
                assert!(group.remaining.iter().all(Option::is_none));
                assert_eq!(
                    restored.into_iter().collect::<Option<Vec<_>>>().unwrap(),
                    expected
                );
                group
                    .stage
                    .validate_fresh_count(group.remaining.len())
                    .unwrap();
            }
        }
    }

    #[test]
    fn cpu_capture_order_is_unchanged_and_missing_cuda_role_is_explicitly_refused() {
        let groups = vec![
            schedule_group(PalsRole::Critic, 16),
            schedule_group(PalsRole::Proposer, 8),
        ];
        let schedule = capture_order(&groups, false, false).unwrap();
        assert_eq!(schedule.profile_cases, 0);
        assert_eq!(
            schedule.slots,
            (0..8)
                .map(|case| CaptureSlot { group: 0, case })
                .chain((0..10).map(|case| CaptureSlot { group: 1, case }))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            capture_order(&groups, true, true)
                .err()
                .unwrap()
                .to_string(),
            "capture plan lacks an original case for a registered role"
        );
    }

    #[test]
    fn interleaved_work_requires_adjacent_actual_deltas_and_checked_group_sums() {
        assert_eq!(
            stage_delta(&cumulative_stats(0), &cumulative_stats(2), 1)
                .unwrap_err()
                .to_string(),
            "Fresh cache-off public/private completed work does not match captured cases"
        );
        assert_eq!(
            stage_delta(&cumulative_stats(2), &cumulative_stats(1), 1)
                .unwrap_err()
                .to_string(),
            "native stage counter regressed"
        );
        let mut sum = StageWork {
            public_cache_hits: u64::MAX,
            ..StageWork::default()
        };
        let before = serde_json::to_value(&sum).unwrap();
        assert_eq!(
            sum.checked_add(StageWork {
                admitted_role_requests: 1,
                public_cache_hits: 1,
                ..StageWork::default()
            })
            .unwrap_err()
            .to_string(),
            "native stage delta sum overflow"
        );
        assert_eq!(serde_json::to_value(&sum).unwrap(), before);
    }

    #[cfg(target_os = "linux")]
    struct OwnedOutputRootTestDirectory {
        path: PathBuf,
        parent: PathBuf,
        device: u64,
        inode: u64,
        parent_device: u64,
        parent_inode: u64,
    }
    #[cfg(target_os = "linux")]
    impl OwnedOutputRootTestDirectory {
        fn new() -> Self {
            use std::os::unix::fs::{DirBuilderExt, MetadataExt};
            use std::sync::atomic::{AtomicU64, Ordering};
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let parent = std::env::temp_dir().canonicalize().unwrap();
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = parent.join(format!(
                "rz-g0-native-output-root-test-{}-{}-{nonce}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::DirBuilder::new()
                .mode(0o700)
                .create(&path)
                .unwrap();
            let identity = std::fs::symlink_metadata(&path).unwrap();
            let parent_identity = std::fs::symlink_metadata(&parent).unwrap();
            Self {
                path,
                parent,
                device: identity.dev(),
                inode: identity.ino(),
                parent_device: parent_identity.dev(),
                parent_inode: parent_identity.ino(),
            }
        }
    }
    #[cfg(target_os = "linux")]
    impl Drop for OwnedOutputRootTestDirectory {
        fn drop(&mut self) {
            use std::os::unix::fs::MetadataExt;
            let (Ok(identity), Ok(parent_identity)) = (
                std::fs::symlink_metadata(&self.path),
                std::fs::symlink_metadata(&self.parent),
            ) else {
                return;
            };
            if self.path.is_absolute()
                && self.path.parent() == Some(self.parent.as_path())
                && self
                    .path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|name| name.starts_with("rz-g0-native-output-root-test-"))
                && identity.is_dir()
                && !identity.file_type().is_symlink()
                && (identity.dev(), identity.ino()) == (self.device, self.inode)
                && parent_identity.is_dir()
                && !parent_identity.file_type().is_symlink()
                && (parent_identity.dev(), parent_identity.ino())
                    == (self.parent_device, self.parent_inode)
            {
                // Only the create-new directory this fixture owns is removed.
                // A replaced root or ancestor is preserved instead of followed.
                let _ = std::fs::remove_dir_all(&self.path);
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn output_admission(path: &Path) -> OriginalEmptyNativeOutputRoot {
        use std::os::unix::fs::MetadataExt;
        let metadata = fs::metadata(path).unwrap();
        OriginalEmptyNativeOutputRoot {
            schema: "rz-pals-original-empty-native-output-root/1".into(),
            device: metadata.dev(),
            inode: metadata.ino(),
            uid: metadata.uid(),
            gid: metadata.gid(),
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn output_root_none_creates_new_but_still_refuses_an_existing_empty_directory() {
        use std::os::unix::fs::MetadataExt;
        let fixture = OwnedOutputRootTestDirectory::new();
        let path = fixture.path.join("native");
        let owner = capture_output_root(&path, None).unwrap();
        assert_eq!(owner.path, path);
        assert!(owner.admission.is_none());
        let inode = fs::metadata(&path).unwrap().ino();
        owner.verify().unwrap();
        drop(owner);
        let error = capture_output_root(&path, None)
            .err()
            .expect("existing root must remain refused");
        assert_eq!(
            error.downcast_ref::<std::io::Error>().unwrap().kind(),
            std::io::ErrorKind::AlreadyExists
        );
        assert_eq!(fs::metadata(&path).unwrap().ino(), inode);
        assert!(fs::read_dir(&path).unwrap().next().is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn output_root_original_empty_admission_retains_the_same_inode_through_publication() {
        use std::os::unix::fs::MetadataExt;
        let fixture = OwnedOutputRootTestDirectory::new();
        let path = fixture.path.join("native");
        fs::create_dir(&path).unwrap();
        let proof = output_admission(&path);
        let owner = capture_output_root(&path, Some(&proof)).unwrap();
        assert_eq!(
            owner.original.as_ref().unwrap().metadata().unwrap().ino(),
            proof.inode
        );
        assert_eq!(fs::metadata(&path).unwrap().ino(), proof.inode);
        // Once admitted, the capture may publish its own output; final verify
        // checks original identity, not an incorrectly repeated emptiness test.
        fs::write(path.join("owned-output.json"), b"owned capture output").unwrap();
        owner.verify().unwrap();
        assert_eq!(
            fs::read(path.join("owned-output.json")).unwrap(),
            b"owned capture output"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn output_root_original_admission_refuses_nonempty_stale_schema_and_owner_without_removal() {
        let fixture = OwnedOutputRootTestDirectory::new();
        let path = fixture.path.join("native");
        fs::create_dir(&path).unwrap();
        let proof = output_admission(&path);
        let preserved = path.join("preserved.txt");
        fs::write(&preserved, b"do not delete").unwrap();
        assert!(capture_output_root(&path, Some(&proof)).is_err());
        assert_eq!(fs::read(&preserved).unwrap(), b"do not delete");
        // Remove only this fixture's explicit marker to test each independent
        // admission failure against the same still-existing empty directory.
        fs::remove_file(&preserved).unwrap();
        let mut bad_schema = proof.clone();
        bad_schema.schema.push_str("-unknown");
        let mut stale_inode = proof.clone();
        stale_inode.inode = stale_inode.inode.wrapping_add(1);
        let mut wrong_uid = proof.clone();
        wrong_uid.uid = wrong_uid.uid.wrapping_add(1);
        let mut wrong_gid = proof.clone();
        wrong_gid.gid = wrong_gid.gid.wrapping_add(1);
        for invalid in [bad_schema, stale_inode, wrong_uid, wrong_gid] {
            assert!(capture_output_root(&path, Some(&invalid)).is_err());
            assert!(path.is_dir());
            assert!(fs::read_dir(&path).unwrap().next().is_none());
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn output_root_original_admission_refuses_symlink_leaf_and_ancestor() {
        use std::os::unix::fs::symlink;
        let fixture = OwnedOutputRootTestDirectory::new();
        let parent = fixture.path.join("actual");
        fs::create_dir(&parent).unwrap();
        let path = parent.join("native");
        fs::create_dir(&path).unwrap();
        let proof = output_admission(&path);
        let linked_leaf = fixture.path.join("linked-native");
        symlink(&path, &linked_leaf).unwrap();
        assert!(capture_output_root(&linked_leaf, Some(&proof)).is_err());
        let linked_parent = fixture.path.join("linked-parent");
        symlink(&parent, &linked_parent).unwrap();
        assert!(capture_output_root(&linked_parent.join("native"), Some(&proof)).is_err());
        assert!(fs::read_dir(&path).unwrap().next().is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn output_root_replacement_after_admission_is_refused_and_original_files_survive() {
        use std::os::unix::fs::MetadataExt;
        let fixture = OwnedOutputRootTestDirectory::new();
        let path = fixture.path.join("native");
        let moved = fixture.path.join("original-native");
        fs::create_dir(&path).unwrap();
        let proof = output_admission(&path);
        let owner = capture_output_root(&path, Some(&proof)).unwrap();
        fs::write(path.join("preserved.txt"), b"original evidence").unwrap();
        fs::rename(&path, &moved).unwrap();
        fs::create_dir(&path).unwrap();
        fs::write(path.join("replacement.txt"), b"replacement must survive").unwrap();
        assert!(owner.verify().is_err());
        assert_eq!(
            owner.original.as_ref().unwrap().metadata().unwrap().ino(),
            proof.inode
        );
        assert_eq!(fs::metadata(&moved).unwrap().ino(), proof.inode);
        assert_ne!(fs::metadata(&path).unwrap().ino(), proof.inode);
        assert_eq!(
            fs::read(moved.join("preserved.txt")).unwrap(),
            b"original evidence"
        );
        assert_eq!(
            fs::read(path.join("replacement.txt")).unwrap(),
            b"replacement must survive"
        );
    }

    fn selection_config(provider: &str) -> Config {
        Config {
            case_plan: PathBuf::new(),
            case_plan_sha256: String::new(),
            export_manifest: PathBuf::new(),
            runtime_library: PathBuf::new(),
            runtime_cache: PathBuf::new(),
            output_root: PathBuf::new(),
            output_root_admission: None,
            provider: provider.to_owned(),
            cudnn_conv_search: None,
            cuda_bundle: Some(PathBuf::new()),
            cuda_inventory: Some(PathBuf::new()),
            cuda_inventory_sha256: Some("a".repeat(64)),
            system_cuda_config: None,
            system_cuda_config_sha256: None,
            system_cuda_inventory: None,
            system_cuda_inventory_sha256: None,
            gpu_device_id: 0,
            gpu_arena_bytes: 1024 * 1024,
            deadline_seconds: 1,
            cleanup_seconds: 1,
        }
    }

    #[test]
    fn system_cuda_requires_own_config_and_refuses_legacy_inputs() {
        let mut config = selection_config("cuda-system");
        config.system_cuda_config = Some(PathBuf::from("/tmp/system.json"));
        config.system_cuda_config_sha256 = Some("a".repeat(64));
        assert!(config.execution_config().is_err());
        config.cuda_bundle = None;
        config.cuda_inventory = None;
        config.cuda_inventory_sha256 = None;
        assert!(config.execution_config().is_ok());
        config.system_cuda_inventory = Some(PathBuf::from("/tmp/inventory.json"));
        assert!(config.execution_config().is_err());
        config.system_cuda_inventory_sha256 = Some("a".repeat(64));
        assert!(config.execution_config().is_ok());
        config.system_cuda_inventory = Some(PathBuf::from("relative-inventory.json"));
        assert!(config.execution_config().is_err());
        config.system_cuda_inventory = Some(PathBuf::from("/tmp/inventory.json"));
        config.system_cuda_inventory_sha256 = Some("not-a-pin".into());
        assert!(config.execution_config().is_err());
        config.system_cuda_inventory = None;
        config.system_cuda_inventory_sha256 = None;
        config.cudnn_conv_search = Some("experimental_default".into());
        assert!(config.execution_config().is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn system_placement_mapping_unknown_is_not_a_witness_or_completion() {
        for (observation, expected) in [
            (
                SystemPlacementObservation::MappingUnavailable,
                "mapping_unavailable",
            ),
            (
                SystemPlacementObservation::UnregisteredOrPreexistingMappings,
                "unregistered_or_preexisting_mappings",
            ),
        ] {
            let value = system_placement_observation_json(&observation);
            assert_eq!(value["status"], expected);
            assert_eq!(value["gpu_kernel_placement"], "unknown");
            assert_eq!(value["physical_completion"], "unknown");
            assert!(value.get("witness").is_none());
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn first_system_placement_partial_survives_later_failure_without_completion_claim() {
        let output = OwnedOutputRootTestDirectory::new();
        let observation =
            system_placement_observation_json(&SystemPlacementObservation::MappingUnavailable);
        publish_initial_system_placement_partial(
            &output.path,
            "plan",
            "inventory",
            "manifest",
            &observation,
        )
        .unwrap();

        // A later case can fail before joined/final publication. The initial
        // on-disk evidence must still be readable and remain explicitly partial.
        let later: Result<()> = Err("later case failed".into());
        assert!(later.is_err());
        let path = output
            .path
            .join("initial-system-cuda-placement.partial.json");
        let value: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert_eq!(value["status"], "partial_observation");
        assert_eq!(value["observation"]["status"], "mapping_unavailable");
        assert_eq!(value["capture_completion"], "not_established");
        assert_eq!(value["physical_completion"], "not_attested");
        assert_eq!(value["process_exit"], "not_observed");
        assert!(value["embedded_witness_sha256"].is_null());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn full_system_placement_witness_can_exceed_small_policy_sidecar_limit() {
        let output = OwnedOutputRootTestDirectory::new();
        // Model a placement witness with enough transfer evidence to exceed
        // the small policy sidecar, while remaining within the placement bound.
        let witness = serde_json::json!({"transfer_evidence": "x".repeat(POLICY_LIMIT)});
        let witness_bytes = serde_json::to_vec(&witness).unwrap();
        assert!(witness_bytes.len() > POLICY_LIMIT);
        assert!(witness_bytes.len() < LIMIT);
        let observation = serde_json::json!({"status":"verified", "witness":witness});
        let expected_sha = hex(asset::sha256(&witness_bytes));

        let actual_sha =
            publish_system_placement_witness(&output.path, "witness.json", &observation).unwrap();
        assert_eq!(actual_sha.as_deref(), Some(expected_sha.as_str()));
        assert_eq!(
            fs::read(output.path.join("witness.json")).unwrap(),
            witness_bytes
        );

        publish_initial_system_placement_partial(
            &output.path,
            "plan",
            "inventory",
            "manifest",
            &observation,
        )
        .unwrap();
        let partial: serde_json::Value = serde_json::from_slice(
            &fs::read(
                output
                    .path
                    .join("initial-system-cuda-placement.partial.json"),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(partial["embedded_witness_sha256"], expected_sha);
        assert_eq!(partial["physical_completion"], "not_attested");
    }

    #[test]
    fn omitted_capture_conv_selection_preserves_original_provider_modes() {
        for provider in ["cpu", "cuda-control", "cuda-control-shim"] {
            let mut config = selection_config(provider);
            assert_eq!(
                config.execution_config().unwrap().cudnn_conv_search,
                PalsCuDnnConvSearch::LegacyImplicitExhaustive,
            );
            config.cudnn_conv_search = Some("legacy_implicit_exhaustive".into());
            assert_eq!(
                config.execution_config().unwrap().cudnn_conv_search,
                PalsCuDnnConvSearch::LegacyImplicitExhaustive,
            );
        }
        let config: Config = decode_unique(
            br#"{
            "case_plan":"plan","case_plan_sha256":"",
            "export_manifest":"export","runtime_library":"runtime",
            "runtime_cache":"cache","output_root":"output","provider":"cpu",
            "cuda_bundle":null,"cuda_inventory":null,"cuda_inventory_sha256":null,
            "gpu_device_id":0,"gpu_arena_bytes":0,
            "deadline_seconds":1,"cleanup_seconds":1
        }"#,
        )
        .unwrap();
        assert!(config.cudnn_conv_search.is_none());
        assert!(config.output_root_admission.is_none());
        assert_eq!(
            config.execution_config().unwrap().cudnn_conv_search,
            PalsCuDnnConvSearch::LegacyImplicitExhaustive,
        );
    }

    #[test]
    fn capture_default_requires_exact_eager_v3_control_host_selection() {
        let mut config = selection_config("cuda-control");
        config.cudnn_conv_search = Some("experimental_default".into());
        let selected = config.execution_config().unwrap();
        assert_eq!(
            selected.cudnn_conv_search,
            PalsCuDnnConvSearch::ExperimentalDefault
        );
        assert!(!selected.device_public_memory);
        assert!(!selected.cache_public_memory);
        assert_eq!(selected.intra_threads, 1);
        for provider in [
            "cpu",
            "cuda-control-shim",
            "cuda",
            "cuda-control-conv-default",
        ] {
            config.provider = provider.into();
            assert!(config.execution_config().is_err());
        }
        config.provider = "cuda-control".into();
        config.cuda_inventory = None;
        assert!(config.execution_config().is_err());
        config.cuda_inventory = Some(PathBuf::new());
        config.cuda_bundle = None;
        assert!(config.execution_config().is_err());
        config.cuda_bundle = Some(PathBuf::new());
        config.cuda_inventory_sha256 = None;
        assert!(config.execution_config().is_err());
        config.cuda_inventory_sha256 = Some("not-a-pin".into());
        assert!(config.execution_config().is_err());
        config.cuda_inventory_sha256 = Some("a".repeat(64));
        config.gpu_device_id = -1;
        assert!(config.execution_config().is_err());
    }

    #[test]
    fn capture_conv_literals_do_not_admit_device_warm_or_fallback() {
        for provider in ["cpu", "cuda-control", "cuda-control-shim"] {
            let mut config = selection_config(provider);
            for value in [
                "DEFAULT",
                "experimental_default_device_warm",
                "",
                "exhaustive",
            ] {
                config.cudnn_conv_search = Some(value.into());
                assert!(config.execution_config().is_err());
            }
        }
    }

    #[test]
    fn json_duplicates_and_unknown_config_fields_fail_before_state_construction() {
        assert!(
            decode_unique::<serde_json::Value>(br#"{"sources":{"native":"a","native":"b"}}"#)
                .is_err()
        );
        assert!(decode_unique::<serde_json::Value>(br#"{"outer":[{"x":1,"x":2}]}"#).is_err());
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Fields {
            n: u64,
        }
        assert_eq!(decode_unique::<Fields>(br#"{"n":7}"#).unwrap().n, 7);
        assert!(decode_unique::<Fields>(br#"{"n":7,"secret":0}"#).is_err());
    }
    #[test]
    fn source_registry_selects_exact_registered_capture_components() {
        let actual = source_pins().unwrap();
        let common = rz_experiments::compiled_source_components_v5().unwrap();
        assert_eq!(actual.len(), 5);
        assert!(
            actual
                .iter()
                .all(|(name, digest)| common.get(name) == Some(digest))
        );
        assert!(!id("unsafe:ads"));
    }
}
