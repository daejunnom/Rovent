//! One V5 native factory on the ticket owner. No ambient model/provider choices.
//!
//! Original export bytes and graph descriptors are read from the child's exact
//! verified FD owner. The same owner stays inside the physical backend, including
//! quarantine. Startup/session metadata is distinct from first-Run placement,
//! numerical acceptance, resource readback and training/content provenance.
use crate::pals_adaptive_startup::{PinnedPalsStartupV5, PinnedStartupAssetV5};
use crate::pals_native::ticketed::NativeVirtualReviewTicketBackend;
use crate::pals_native::virtual_encoding::validate_pals_virtual_rules_descriptor_bytes;
#[cfg(target_os = "linux")]
use cap_fs_ext::DirExt;
use rz_eval::asset::parse_sha256;
use rz_eval::error::{BackendError, FailureKind as K, FailureStage as S};
use rz_eval::onnx::{NativeLoadingProfile, OrtRuntime, Provider};
use rz_eval::pals_model::{PalsModelProfile, virtual_review};
use rz_eval::pals_onnx::virtual_review::{
    PalsVirtualReviewOnnxBackend, VirtualReviewArtifactKind, VirtualReviewExecutionMode,
    VirtualReviewOwnedSources, VirtualReviewProfileDirectory,
    validate_virtual_review_export_manifest_kind,
};
use rz_eval::pals_onnx::{
    PalsCuDnnConvSearch, PalsCudaControlPolicy, PalsGraphPlacement, PalsOnnxConfig,
    PalsSessionResidency,
};
use rz_eval::runtime_pin::{CudaRuntimeBundleSpec, RuntimeLibraryPin};
use rz_experiments::{
    PalsArtifactDomainV5, PalsEngineModeV5, PalsModelBackendV3,
    PalsStartupAssetPurposeV5 as Purpose, PalsStartupContentAdmissionV5,
    verify_compiled_source_components_v5,
};
use rz_runtime::contracts::{ContractClock, ContractSystemClock};
use rz_search::pals::coordinator::RoleFailure;
use serde::Serialize;
use sha2::{Digest as _, Sha256};
use std::{
    fs::File,
    path::{Component, Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

/// Optional read-only tensor/export intake on the actual loaded native owner.
/// This runs during explicit startup, before the worker can admit any Run.
/// It grants no numeric, training, or placement acceptance by itself.
pub type NativeVirtualReviewIntakeV5 =
    Box<dyn FnOnce(&PalsVirtualReviewOnnxBackend, Instant) -> Result<(), BackendError> + Send>;

pub struct NativeVirtualReviewStartupResourcesV5 {
    /// Deep producer-metadata and independent raw-numeric admission issued
    /// from this endpoint's actual retained descriptors, never a passing flag.
    pub content_admission: PalsStartupContentAdmissionV5,
    /// The bootstrap/cache verified native copy already owns its durable pins.
    /// OrtRuntime retains this capability for the entire process once loaded.
    pub runtime_pin: RuntimeLibraryPin,
    /// New exclusive child path beneath the existing endpoint runtime owner.
    /// The native physical owner creates it once; callers must not precreate it.
    /// CUDA requires it; CPU cannot silently enable a CUDA profile directory.
    pub cuda_profile_root: Option<PathBuf>,
    /// Admission deadline, not a promise that native constructors can be killed.
    pub until: Instant,
    pub content_intake: Option<NativeVirtualReviewIntakeV5>,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeVirtualReviewStartupSnapshotV5 {
    pub schema: &'static str,
    pub process_epoch: u64,
    pub model_profile: String,
    pub model_semantic_sha256: [u8; 32],
    pub encoding_sha256: [u8; 32],
    /// The full registered native component, distinct from the narrow model boundary.
    pub model_implementation_sha256: [u8; 32],
    pub model_boundary_source_sha256: [u8; 32],
    pub content_admission_config_sha256: [u8; 32],
    pub content_admission_semantic_lock_sha256: [u8; 32],
    /// Reused independent math evidence; not this session's first Run.
    pub numerical_acceptance_sha256: [u8; 32],
    /// Startup's pure metadata gate never decodes checkpoint tensors/optimizer.
    pub tensor_optimizer_content_verified: bool,
    pub registered_model_epoch: u64,
    /// Loaded immutable checkpoint hash; separate from the registered epoch.
    pub loaded_weights_sha256: [u8; 32],
    pub export_manifest_sha256: [u8; 32],
    pub backend_sha256: [u8; 32],
    pub runtime_library_sha256: [u8; 32],
    pub runtime_bundle_sha256: Option<[u8; 32]>,
    pub configured_provider: &'static str,
    pub configured_intra_threads: usize,
    pub configured_inter_threads: usize,
    pub configured_spinning: bool,
    pub configured_tf32: bool,
    pub configured_cache_public_memory: bool,
    pub configured_approximate_warm: bool,
    pub validator_loaded: bool,
    pub physical_workers_created: usize,
    pub session_residency: PalsSessionResidency,
    /// Actual model-load assignment gate, not a Run/kernel/physical fence.
    pub cuda_initialization: Vec<PalsGraphPlacement>,
    pub neural_service_executed: bool,
}
pub struct LoadedNativeVirtualReviewStartupV5 {
    pub backend: NativeVirtualReviewTicketBackend,
    pub snapshot: NativeVirtualReviewStartupSnapshotV5,
}
impl NativeVirtualReviewStartupSnapshotV5 {
    /// Exact producer mapping; the ABI registry and the ticket encoding key
    /// have different domains. Manifest model epoch is also not a cache epoch.
    pub fn adaptive_model_namespace(
        &self,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::AdaptiveModelNamespace,
        rz_search::pals::engine::RoleError,
    > {
        if self.model_profile != "virtual_review_v1" || self.configured_tf32 {
            return Err(rz_search::pals::engine::RoleError::InvalidOutput);
        }
        rz_search::pals::engine::adaptive_port::AdaptiveModelNamespace::new(
            self.model_value_identity()?,
            rz_contracts::Digest(self.loaded_weights_sha256),
            rz_contracts::Digest(Sha256::digest(virtual_review::ENCODING.as_bytes()).into()),
            rz_contracts::PrecisionProfile::Fp32,
            1, // NativeVirtualReviewTicketBackend's frozen representation epoch.
        )
    }
    /// Namespace mirror for Search, made from the actual loaded export and
    /// native backend. It grants no readiness, WDL, ticket or seed authority.
    pub fn role_model_identity(&self) -> String {
        format!(
            "pals-native-virtual-v3:{}:{}:{}",
            hex(self.export_manifest_sha256),
            hex(self.loaded_weights_sha256),
            hex(self.backend_sha256),
        )
    }
    pub fn model_value_identity(
        &self,
    ) -> Result<rz_search::pals::value::ModelValueIdentity, rz_search::pals::engine::RoleError>
    {
        let identity = rz_search::pals::value::ModelValueIdentity {
            semantics: rz_search::pals::value::MODEL_WDL_VALUE_SEMANTICS.into(),
            model: self.role_model_identity(),
            encoding: format!("{}:{}", virtual_review::ENCODING, hex(self.encoding_sha256)),
            precision: "fp32_tf32_off".into(),
            model_epoch: self.loaded_weights_sha256,
        };
        identity.validate()?;
        Ok(identity)
    }
}

struct PinnedVirtualReviewSourcesV5(Arc<PinnedPalsStartupV5>);
impl VirtualReviewOwnedSources for PinnedVirtualReviewSourcesV5 {
    fn read_manifest(&self, maximum: usize) -> Result<Vec<u8>, BackendError> {
        read_pinned(
            self.0.asset(Purpose::ExportManifest).map_err(transport)?,
            maximum,
        )
    }
    fn read_graph(
        &self,
        role: &str,
        registered_file: &str,
        expected: [u8; 32],
        maximum: usize,
    ) -> Result<Vec<u8>, BackendError> {
        let purpose = graph_purpose(role)?;
        let graph = self.0.asset(purpose).map_err(transport)?;
        let manifest = self.0.asset(Purpose::ExportManifest).map_err(transport)?;
        exact_graph_snapshot_join(
            &manifest.declaration.snapshot_path,
            &graph.declaration.snapshot_path,
            registered_file,
        )?;
        if parse_sha256(&graph.declaration.artifact.sha256)? != expected {
            return Err(fail(
                K::IdentityMismatch,
                "graph declaration differs from export",
            ));
        }
        read_pinned(graph, maximum)
    }
}

/// Invoke this once inside TicketedNativeService's owner factory. No model,
/// encoder, forward, archive or pipe work belongs on the manager thread.
pub fn load_native_virtual_review_startup_v5(
    owner: Arc<PinnedPalsStartupV5>,
    clock: ContractSystemClock,
    resources: NativeVirtualReviewStartupResourcesV5,
) -> Result<LoadedNativeVirtualReviewStartupV5, RoleFailure> {
    load_inner(owner, clock, resources).map_err(|error| RoleFailure::Backend(error.to_string()))
}
fn load_inner(
    owner: Arc<PinnedPalsStartupV5>,
    clock: ContractSystemClock,
    resources: NativeVirtualReviewStartupResourcesV5,
) -> Result<LoadedNativeVirtualReviewStartupV5, BackendError> {
    let started = Instant::now();
    if owner.metadata_preflight {
        return Err(fail(
            K::InvalidInput,
            "metadata preflight cannot create a native model owner",
        ));
    }
    owner
        .config
        .validate_against(&owner.lock)
        .map_err(transport)?;
    verify_compiled_source_components_v5(&owner.config.source_components).map_err(transport)?;
    let canonical = owner
        .config
        .canonical_bytes(&owner.lock)
        .map_err(transport)?;
    if hex(Sha256::digest(&canonical).into()) != owner.config_sha256
        || owner.assets().len() != owner.config.assets.len()
        || owner
            .assets()
            .iter()
            .zip(&owner.config.assets)
            .any(|(p, a)| &p.declaration != a)
    {
        return Err(fail(
            K::IdentityMismatch,
            "pinned startup declarations changed after admission",
        ));
    }
    let owned_assets = owner
        .assets()
        .iter()
        .map(|asset| (&asset.declaration, &asset.file))
        .collect::<Vec<_>>();
    resources
        .content_admission
        .verify_binding(&owner.config, &owner.lock, &owned_assets)
        .map_err(transport)?;
    drop(owned_assets);
    check_until(resources.until)?;
    let endpoint = &owner.config.endpoint;
    let model = &endpoint.model;
    let native = &owner.config.native_options;
    let declared_until = started
        .checked_add(Duration::from_millis(native.startup_ms))
        .ok_or_else(|| fail(K::ResourceExhausted, "startup time declaration overflow"))?;
    let resources = NativeVirtualReviewStartupResourcesV5 {
        until: resources.until.min(declared_until),
        ..resources
    };
    if model.profile != virtual_review::PROFILE
        || model.semantic_sha256
            != hex(PalsModelProfile::VirtualReviewV1.registered_semantics_sha256())
        || model.encoding_sha256 != hex(virtual_review::registered_encoding_sha256())
        || owner.config.source_components.get("model_native") != Some(&model.implementation_sha256)
    {
        return Err(fail(
            K::UnsupportedModel,
            "startup requires exact registered V3 model and tensor ABI",
        ));
    }
    if endpoint.resources.ort_intra_threads != 1
        || endpoint.resources.ort_inter_threads != 1
        || endpoint.resources.ort_spinning
        || native.device_public_memory
    {
        return Err(fail(
            K::UnsupportedModel,
            "V3 startup requires registered host memory and ORT 1/1/spinning-off",
        ));
    }
    let mode = match endpoint.mode {
        PalsEngineModeV5::Deployment => VirtualReviewExecutionMode::Deployment,
        PalsEngineModeV5::Collection => VirtualReviewExecutionMode::Collection,
    };
    let kind = if native.approximate_warm {
        VirtualReviewArtifactKind::ApproxWarm
    } else {
        VirtualReviewArtifactKind::Fresh
    };
    if native.approximate_warm && mode == VirtualReviewExecutionMode::Collection {
        return Err(fail(
            K::UnsupportedModel,
            "registered V3 ApproxWarm artifact is V-free; Collection requires its Fresh V graph",
        ));
    }
    let sources = Arc::new(PinnedVirtualReviewSourcesV5(Arc::clone(&owner)));
    let manifest_raw = sources.read_manifest(256 * 1024)?;
    let manifest = validate_virtual_review_export_manifest_kind(&manifest_raw, mode, kind)?;
    let rules_descriptor = validate_pals_virtual_rules_descriptor_bytes(&read_pinned(
        owner.asset(Purpose::RulesDescriptor).map_err(transport)?,
        64 * 1024,
    )?)?;
    if manifest.checkpoint_sha256 != parse_sha256(&model.weights.sha256)?
        || manifest.validator_present != (endpoint.mode == PalsEngineModeV5::Collection)
        || manifest.model_profile != PalsModelProfile::VirtualReviewV1
        || manifest.rules_input_semantic_sha256
            != crate::pals_native::virtual_encoding::pals_virtual_rules_semantic_digest()
        || rules_descriptor.config != manifest.config
        || rules_descriptor.semantic_sha256 != manifest.rules_input_semantic_sha256
        || rules_descriptor.encoder_source_sha256 != manifest.rules_encoder_source_sha256
        || rules_descriptor.descriptor_sha256() != manifest.rules_profile_descriptor_sha256
    {
        return Err(fail(
            K::IdentityMismatch,
            "actual export and Rules descriptor differ from registered weights/roles/compiled semantics",
        ));
    }
    match model.artifact_domain {
        PalsArtifactDomainV5::Untrained if manifest.trained || manifest.training_steps != 0 => {
            return Err(fail(
                K::IdentityMismatch,
                "untrained startup cannot carry optimizer claims",
            ));
        }
        PalsArtifactDomainV5::RealGeneration
            if !manifest.trained || manifest.training_steps != 5000 =>
        {
            return Err(fail(
                K::IdentityMismatch,
                "generation startup lacks the registered complete 5000-update export structure",
            ));
        }
        PalsArtifactDomainV5::FrozenPcCollection => {
            let frozen = resources
                .content_admission
                .frozen_pc_collection()
                .ok_or_else(|| {
                    fail(
                        K::IdentityMismatch,
                        "frozen P/C lacks original descriptor-bound content admission",
                    )
                })?;
            if endpoint.mode != PalsEngineModeV5::Collection
                || !manifest.trained
                || manifest.training_steps != 4000
                || native.approximate_warm
                || frozen.checkpoint_sha256() != model.weights.sha256
                || frozen.model_epoch() != model.epoch
                || !manifest.validator_present
            {
                return Err(fail(
                    K::IdentityMismatch,
                    "frozen P/C loaded export differs from its exact 4000-update Collection namespace",
                ));
            }
        }
        PalsArtifactDomainV5::Diagnostic => {
            return Err(fail(
                K::UnsupportedModel,
                "diagnostic assets cannot enter the actual V5 factory",
            ));
        }
        _ => {}
    }
    for graph in &manifest.graphs {
        let asset = owner
            .asset(graph_purpose(&graph.role)?)
            .map_err(transport)?;
        exact_graph_snapshot_join(
            &owner
                .asset(Purpose::ExportManifest)
                .map_err(transport)?
                .declaration
                .snapshot_path,
            &asset.declaration.snapshot_path,
            &graph.file,
        )?;
        if parse_sha256(&asset.declaration.artifact.sha256)? != graph.sha256 {
            return Err(fail(
                K::IdentityMismatch,
                "declared graph pin differs before session loading",
            ));
        }
    }
    drop(manifest_raw);
    let runtime_asset = owner.asset(Purpose::RuntimeLibrary).map_err(transport)?;
    if resources.runtime_pin.binary_digest()
        != parse_sha256(&runtime_asset.declaration.artifact.sha256)?
    {
        return Err(fail(
            K::IdentityMismatch,
            "retained runtime binary differs from startup asset",
        ));
    }
    let cudnn = match native.cudnn_conv_search.as_str() {
        "legacy_implicit_exhaustive" => PalsCuDnnConvSearch::LegacyImplicitExhaustive,
        "experimental_default" => PalsCuDnnConvSearch::ExperimentalDefault,
        _ => {
            return Err(fail(
                K::UnsupportedModel,
                "unregistered host-public cuDNN option",
            ));
        }
    };
    let (provider, loading) = match model.backend {
        PalsModelBackendV3::OrtCpu => {
            if resources.runtime_pin.bundle_digest().is_some()
                || native.native_loading_profile.is_some()
                || native.cuda_device_id.is_some()
                || native.cuda_arena_bytes.is_some()
                || resources.cuda_profile_root.is_some()
                || cudnn != PalsCuDnnConvSearch::LegacyImplicitExhaustive
            {
                return Err(fail(
                    K::IdentityMismatch,
                    "CPU startup cannot select CUDA options or a native bundle",
                ));
            }
            (Provider::Cpu, None)
        }
        PalsModelBackendV3::OrtCuda => {
            let profile = parse_loading_profile(native.native_loading_profile.as_deref())?;
            let device = native
                .cuda_device_id
                .ok_or_else(|| fail(K::InvalidInput, "CUDA device is not registered"))?;
            let arena = usize::try_from(
                native
                    .cuda_arena_bytes
                    .ok_or_else(|| fail(K::InvalidInput, "CUDA arena is not registered"))?,
            )
            .map_err(|_| {
                fail(
                    K::ResourceExhausted,
                    "CUDA arena exceeds host address space",
                )
            })?;
            verify_runtime_bundle(&owner, &resources.runtime_pin)?;
            (
                Provider::Cuda {
                    device_id: device,
                    arena_bytes: arena,
                },
                Some(profile),
            )
        }
        PalsModelBackendV3::RustCpu | PalsModelBackendV3::DeterministicMock => {
            return Err(fail(
                K::UnsupportedModel,
                "native V3 factory cannot substitute a CPU/mock implementation for ORT",
            ));
        }
    };
    let configuration = PalsOnnxConfig {
        provider,
        intra_threads: 1,
        cache_public_memory: native.cache_public_memory,
        device_public_memory: native.device_public_memory,
        cudnn_conv_search: cudnn,
    };
    configuration.validate_execution_selection(loading, loading.is_some())?;
    let control = if loading.is_some() {
        let asset = owner
            .asset(Purpose::CudaControlInventory)
            .map_err(transport)?;
        let inventory = read_pinned(asset, 4 * 1024 * 1024)?;
        let policy = PalsCudaControlPolicy::from_virtual_review_inventory_bytes(
            &inventory,
            &asset.declaration.artifact.sha256,
        )?;
        let root = resources
            .cuda_profile_root
            .as_ref()
            .ok_or_else(|| fail(K::InvalidInput, "CUDA profile owner is missing"))?;
        let directory = create_owned_profile_directory(&owner, root)?;
        Some((policy, directory))
    } else {
        None
    };
    check_until(resources.until)?;
    let runtime = match loading {
        Some(profile) => {
            OrtRuntime::load_with_cuda_loading_profile(&resources.runtime_pin, profile)?
        }
        None => OrtRuntime::load(&resources.runtime_pin)?,
    };
    check_until(resources.until)?;
    let backend = PalsVirtualReviewOnnxBackend::load_owned_sources_with_profile(
        sources,
        &owner
            .asset(Purpose::ExportManifest)
            .map_err(transport)?
            .declaration
            .artifact
            .sha256,
        runtime,
        configuration,
        mode,
        kind,
        control,
    )?;
    // A constructor cannot be asynchronously freed or retried upon expiry.
    // If this bound expired, no worker/admission is published; native session
    // destruction occurs on this same owner and OrtRuntime retains its pins.
    check_until(resources.until)?;
    let snapshot = NativeVirtualReviewStartupSnapshotV5 {
        schema: "rz-pals-native-virtual-startup-snapshot-v5/1",
        process_epoch: clock.domain().0.0,
        model_profile: backend.model_config().profile.as_str().into(),
        model_semantic_sha256: backend.model_config().profile.registered_semantics_sha256(),
        encoding_sha256: virtual_review::registered_encoding_sha256(),
        model_implementation_sha256: parse_sha256(&model.implementation_sha256)?,
        model_boundary_source_sha256: rz_eval::pals_model::compiled_virtual_model_boundary_sha256(),
        content_admission_config_sha256: parse_sha256(resources.content_admission.config_sha256())?,
        content_admission_semantic_lock_sha256: parse_sha256(
            resources.content_admission.semantic_lock_sha256(),
        )?,
        numerical_acceptance_sha256: parse_sha256(
            &resources.content_admission.numerical().acceptance.sha256,
        )?,
        tensor_optimizer_content_verified: resources
            .content_admission
            .tensor_optimizer_content_verified(),
        registered_model_epoch: model.epoch,
        loaded_weights_sha256: backend.model_epoch(),
        export_manifest_sha256: backend.manifest_digest(),
        backend_sha256: backend.backend_digest(),
        runtime_library_sha256: backend.runtime_binary_digest(),
        runtime_bundle_sha256: backend.runtime_bundle_digest(),
        configured_provider: match provider {
            Provider::Cpu => "CPUExecutionProvider",
            Provider::Cuda { .. } => "CUDAExecutionProvider",
        },
        configured_intra_threads: 1,
        configured_inter_threads: 1,
        configured_spinning: false,
        configured_tf32: false,
        configured_cache_public_memory: configuration.cache_public_memory,
        configured_approximate_warm: backend.approximate_warm_capable(),
        validator_loaded: backend.validator_present(),
        physical_workers_created: 1,
        session_residency: backend.residency().clone(),
        cuda_initialization: backend.startup_placement(),
        neural_service_executed: false,
    };
    if let Some(intake) = resources.content_intake {
        intake(&backend, resources.until)?;
        check_until(resources.until)?;
    }
    let backend = NativeVirtualReviewTicketBackend::new(
        backend,
        clock,
        Duration::from_millis(native.drain_ms),
    )
    .map_err(|_| fail(K::BackendFailure, "V3 ticket owner construction failed"))?;
    Ok(LoadedNativeVirtualReviewStartupV5 { backend, snapshot })
}

fn verify_runtime_bundle(
    owner: &PinnedPalsStartupV5,
    pin: &RuntimeLibraryPin,
) -> Result<(), BackendError> {
    let asset = owner.asset(Purpose::CudaManifest).map_err(transport)?;
    let raw = read_pinned(asset, 64 * 1024)?;
    let declared = CudaRuntimeBundleSpec::from_json(
        std::str::from_utf8(&raw)
            .map_err(|_| fail(K::InvalidInput, "CUDA bundle must be UTF-8"))?,
    )?;
    if pin.bundle_spec().as_ref() != Some(&declared)
        || pin.bundle_digest() != Some(declared.digest()?)
    {
        return Err(fail(
            K::IdentityMismatch,
            "retained bundle semantics differ from its pinned declaration",
        ));
    }
    for library in &declared.files {
        let matches = owner
            .assets()
            .iter()
            .filter(|a| {
                matches!(
                    a.declaration.purpose,
                    Purpose::RuntimeLibrary | Purpose::CudaLibrary
                ) && Path::new(&a.declaration.snapshot_path)
                    .file_name()
                    .and_then(|v| v.to_str())
                    == Some(library.filename.as_str())
            })
            .collect::<Vec<_>>();
        if matches.len() != 1
            || matches[0].declaration.artifact.sha256 != library.sha256
            || matches[0].declaration.artifact.bytes != library.bytes
        {
            return Err(fail(
                K::IdentityMismatch,
                "retained CUDA library roster differs from exact startup assets",
            ));
        }
    }
    if owner
        .assets()
        .iter()
        .filter(|a| {
            matches!(
                a.declaration.purpose,
                Purpose::RuntimeLibrary | Purpose::CudaLibrary
            )
        })
        .count()
        != declared.files.len()
    {
        return Err(fail(
            K::IdentityMismatch,
            "startup CUDA asset roster has an unregistered extra library",
        ));
    }
    Ok(())
}
fn parse_loading_profile(value: Option<&str>) -> Result<NativeLoadingProfile, BackendError> {
    [
        NativeLoadingProfile::EagerCuda12Cudnn9V1,
        NativeLoadingProfile::CuDnnShimLazyV1,
    ]
    .into_iter()
    .find(|p| Some(p.identifier()) == value)
    .ok_or_else(|| {
        fail(
            K::UnsupportedModel,
            "exact native loading profile is required",
        )
    })
}
pub(crate) fn graph_purpose(role: &str) -> Result<Purpose, BackendError> {
    match role {
        "public" => Ok(Purpose::PublicGraph),
        "shared_pc" | "shared_warm" => Ok(Purpose::SharedPcGraph),
        "validator" => Ok(Purpose::ValidatorGraph),
        _ => Err(fail(K::UnsupportedModel, "unregistered V3 graph role")),
    }
}
fn exact_graph_snapshot_join(manifest: &str, graph: &str, file: &str) -> Result<(), BackendError> {
    if !matches!(
        Path::new(file).components().collect::<Vec<_>>().as_slice(),
        [Component::Normal(_)]
    ) || Path::new(file).file_name().and_then(|n| n.to_str()) != Some(file)
        || Path::new(manifest)
            .parent()
            .map(|p| p.join(file))
            .as_deref()
            != Some(Path::new(graph))
    {
        return Err(fail(
            K::IdentityMismatch,
            "private graph path differs from the original manifest parent and basename",
        ));
    }
    Ok(())
}
/// Parse a naming declaration, not a filesystem capability. Only this basename
/// is subsequently used beneath the already held endpoint runtime descriptor.
fn profile_child_name(
    private_cwd: &Path,
    runtime_directory: &str,
    requested: &Path,
) -> Result<String, BackendError> {
    let runtime = private_cwd.join(runtime_directory);
    let name = requested
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| {
            fail(
                K::InvalidInput,
                "CUDA profile path requires a new named child",
            )
        })?;
    if !private_cwd.is_absolute()
        || !requested.is_absolute()
        || requested
            .components()
            .any(|c| matches!(c, Component::ParentDir | Component::CurDir))
        || requested.parent() != Some(runtime.as_path())
        || requested
            .to_str()
            .is_none_or(|s| s.split(['/', '\\']).any(|p| matches!(p, "." | "..")))
        || name.is_empty()
        || name.len() > 96
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        || !matches!(
            Path::new(name).components().collect::<Vec<_>>().as_slice(),
            [Component::Normal(_)]
        )
    {
        return Err(fail(
            K::InvalidInput,
            "CUDA profile must be a direct child of the endpoint runtime",
        ));
    }
    Ok(name.to_owned())
}
#[cfg(target_os = "linux")]
fn create_profile_in_runtime(
    runtime: &cap_std::fs::Dir,
    name: &str,
) -> Result<VirtualReviewProfileDirectory, BackendError> {
    // Exact exclusive mkdir and no-follow directory open use the same held
    // runtime capability. No ambient pathname is reopened after validation.
    runtime.create_dir(name).map_err(|_| {
        fail(
            K::InvalidInput,
            "CUDA profile child must be new and exclusive",
        )
    })?;
    let profile = runtime.open_dir_nofollow(name).map_err(|_| {
        fail(
            K::InvalidInput,
            "CUDA profile child could not be pinned without links",
        )
    })?;
    VirtualReviewProfileDirectory::from_owned_directory(profile.into_std_file())
}
fn create_owned_profile_directory(
    owner: &PinnedPalsStartupV5,
    requested: &Path,
) -> Result<VirtualReviewProfileDirectory, BackendError> {
    let name = profile_child_name(
        owner.private_cwd(),
        &owner.config.runtime_directory,
        requested,
    )?;
    #[cfg(target_os = "linux")]
    {
        let runtime = owner.runtime_directory_capability().map_err(transport)?;
        create_profile_in_runtime(&runtime, &name)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = name;
        Err(fail(
            K::UnsupportedModel,
            "V3 CUDA profile directory capability requires Linux; no ordinary-path substitution",
        ))
    }
}
fn read_pinned(asset: &PinnedStartupAssetV5, maximum: usize) -> Result<Vec<u8>, BackendError> {
    let before = asset
        .file
        .metadata()
        .map_err(|_| fail(K::InvalidInput, "pinned asset metadata unavailable"))?;
    let length = usize::try_from(asset.declaration.artifact.bytes)
        .map_err(|_| fail(K::ResourceExhausted, "pinned asset length overflow"))?;
    if !before.is_file() || length == 0 || length > maximum || before.len() != length as u64 {
        return Err(fail(
            K::ResourceExhausted,
            "pinned asset exceeds the exact bounded input",
        ));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| fail(K::ResourceExhausted, "pinned input allocation failed"))?;
    bytes.resize(length, 0);
    #[cfg(windows)]
    let reader =
        rz_native_loader::pinned_reader::independent_read_handle(&asset.file).map_err(|error| {
            let mut failure = BackendError::new(
                K::InvalidInput,
                S::Admission,
                "pinned input reader unavailable",
            )
            .with_external_cause(rz_eval::error::CauseCode::ModelLoad, &error);
            failure.native = Some(Box::new(rz_eval::error::NativeDiagnostic {
                code: "PinnedReadHandle".into(),
                message: format!("raw_os_error={:?}", error.raw_os_error()),
                truncated: false,
            }));
            failure
        })?;
    #[cfg(windows)]
    let read_file = &reader;
    #[cfg(not(windows))]
    let read_file = &asset.file;
    let mut consumed = 0;
    while consumed < length {
        let count = positional_read(read_file, &mut bytes[consumed..], consumed as u64)
            .map_err(|_| fail(K::InvalidInput, "pinned input positional read failed"))?;
        if count == 0 {
            return Err(fail(
                K::IdentityMismatch,
                "pinned input ended before declared length",
            ));
        }
        consumed += count;
    }
    let after = asset
        .file
        .metadata()
        .map_err(|_| fail(K::InvalidInput, "pinned input metadata disappeared"))?;
    if before.len() != after.len()
        || before.modified().ok() != after.modified().ok()
        || <[u8; 32]>::from(Sha256::digest(&bytes))
            != parse_sha256(&asset.declaration.artifact.sha256)?
    {
        return Err(fail(
            K::IdentityMismatch,
            "pinned input changed during read",
        ));
    }
    Ok(bytes)
}
#[cfg(unix)]
fn positional_read(file: &File, bytes: &mut [u8], at: u64) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(bytes, at)
}
#[cfg(windows)]
fn positional_read(file: &File, bytes: &mut [u8], at: u64) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_read(bytes, at)
}
#[cfg(not(any(unix, windows)))]
fn positional_read(_: &File, _: &mut [u8], _: u64) -> std::io::Result<usize> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "pinned positional reads are unsupported on this OS",
    ))
}
fn check_until(until: Instant) -> Result<(), BackendError> {
    if Instant::now() < until {
        Ok(())
    } else {
        Err(fail(
            K::BackendFailure,
            "native startup admission window ended",
        ))
    }
}
fn hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn fail(kind: K, message: &'static str) -> BackendError {
    BackendError::new(kind, S::Admission, message)
}
fn transport(_: rz_experiments::ManifestError) -> BackendError {
    fail(
        K::IdentityMismatch,
        "typed pinned startup declaration failed validation",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom, Write};
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);

    struct FixtureDirectory {
        path: PathBuf,
    }
    impl FixtureDirectory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "rz-v5-startup-source-{}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                NEXT.fetch_add(1, Ordering::Relaxed),
            ));
            std::fs::create_dir(&path).unwrap();
            Self { path }
        }
    }
    impl Drop for FixtureDirectory {
        fn drop(&mut self) {
            // Remove only explicitly owned fixture leaves, without recursion.
            let _ = std::fs::remove_file(self.path.join("fixture.bin"));
            let _ = std::fs::remove_file(self.path.join("candidate-runtime"));
            let _ = std::fs::remove_dir(self.path.join("candidate-runtime/existing"));
            let _ = std::fs::remove_dir(self.path.join("candidate-runtime/profiles-v3"));
            let _ = std::fs::remove_dir(self.path.join("candidate-runtime"));
            let _ = std::fs::remove_dir(self.path.join("moved-runtime/profiles-v3"));
            let _ = std::fs::remove_dir(self.path.join("moved-runtime"));
            let _ = std::fs::remove_dir(&self.path);
        }
    }

    #[test]
    fn private_graph_name_and_parent_are_both_required() {
        exact_graph_snapshot_join(
            "candidate-model/export.json",
            "candidate-model/public.onnx",
            "public.onnx",
        )
        .unwrap();
        for (graph, name) in [
            ("other/public.onnx", "public.onnx"),
            ("candidate-model/public.onnx", "../public.onnx"),
            ("candidate-model/public.onnx", "."),
            ("candidate-model/public.onnx", "nested/public.onnx"),
        ] {
            assert_eq!(
                exact_graph_snapshot_join("candidate-model/export.json", graph, name)
                    .unwrap_err()
                    .kind,
                K::IdentityMismatch
            );
        }
    }

    #[test]
    fn pinned_positional_reads_preserve_cursor_and_repeat_the_exact_identity_check() {
        let directory = FixtureDirectory::new();
        let path = directory.path.join("fixture.bin");
        let bytes = b"readonly source fixture";
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(bytes).unwrap();
        file.seek(SeekFrom::Start(5)).unwrap();
        let mut asset = PinnedStartupAssetV5 {
            declaration: rz_experiments::PalsStartupAssetV5 {
                artifact: rz_experiments::ArtifactRef {
                    path: "logical/fixture.bin".into(),
                    sha256: hex(Sha256::digest(bytes).into()),
                    bytes: bytes.len() as u64,
                    source: "independent source fixture".into(),
                    license: "MIT".into(),
                },
                snapshot_path: "candidate-model/fixture.bin".into(),
                purpose: Purpose::PublicGraph,
            },
            file,
        };
        assert_eq!(read_pinned(&asset, bytes.len()).unwrap(), bytes);
        assert_eq!(asset.file.stream_position().unwrap(), 5);
        std::thread::scope(|scope| {
            let first = scope.spawn(|| read_pinned(&asset, bytes.len()));
            let second = scope.spawn(|| read_pinned(&asset, bytes.len()));
            assert_eq!(first.join().unwrap().unwrap(), bytes);
            assert_eq!(second.join().unwrap().unwrap(), bytes);
        });
        assert_eq!(asset.file.stream_position().unwrap(), 5);
        assert_eq!(
            read_pinned(&asset, bytes.len() - 1).unwrap_err().kind,
            K::ResourceExhausted
        );
        asset.declaration.artifact.sha256 = "11".repeat(32);
        assert_eq!(
            read_pinned(&asset, bytes.len()).unwrap_err().kind,
            K::IdentityMismatch
        );
        assert_eq!(asset.file.stream_position().unwrap(), 5);
    }

    #[test]
    fn profile_naming_requires_one_bounded_child_without_interpreting_paths() {
        let directory = FixtureDirectory::new();
        let runtime = directory.path.join("candidate-runtime");
        std::fs::create_dir(&runtime).unwrap();
        let child = runtime.join("profiles-v3");
        assert_eq!(
            profile_child_name(&directory.path, "candidate-runtime", &child).unwrap(),
            "profiles-v3"
        );
        for path in [
            directory.path.join("profiles-v3"),
            runtime.join("sub/../../outside"),
            runtime.join("../outside"),
            runtime.join("nested/profiles-v3"),
            runtime.join("./profiles-v3"),
            runtime.join("name:with:extra:meaning"),
            runtime.join("x".repeat(97)),
        ] {
            assert!(profile_child_name(&directory.path, "candidate-runtime", &path).is_err());
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn held_runtime_profile_creation_survives_parent_path_replacement() {
        let owner = FixtureDirectory::new();
        let outside = FixtureDirectory::new();
        let path = owner.path.join("candidate-runtime");
        std::fs::create_dir(&path).unwrap();
        let parent =
            cap_std::fs::Dir::open_ambient_dir(&owner.path, cap_std::ambient_authority()).unwrap();
        let runtime = parent.open_dir_nofollow("candidate-runtime").unwrap();
        std::fs::rename(&path, owner.path.join("moved-runtime")).unwrap();
        std::os::unix::fs::symlink(&outside.path, owner.path.join("candidate-runtime")).unwrap();
        assert!(parent.open_dir_nofollow("candidate-runtime").is_err());
        let profile = create_profile_in_runtime(&runtime, "profiles-v3").unwrap();
        assert!(profile.path().starts_with("/proc/self/fd"));
        assert!(owner.path.join("moved-runtime/profiles-v3").is_dir());
        assert!(!outside.path.join("profiles-v3").exists());
        assert!(create_profile_in_runtime(&runtime, "profiles-v3").is_err());
    }
}
