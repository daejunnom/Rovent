//! Explicit, single-session Maia ONNX composition with provider-specific guards.
//!
//! Assets and the native runtime are loaded once, before protocol service. A
//! search receives only a lightweight adapter to the process-owned physical
//! worker. Normal cancellation audits retain counts and first/last receipts;
//! they are deliberately not a complete request journal. Fatal native causes
//! remain owned alongside B's common-contract failures, including after quit.

#[cfg(feature = "onnx-cuda")]
use crate::native_cuda_attestation::CudaProfileV1;
use crate::{
    engine::{
        EngineError, EvaluatorFactory, EvaluatorProfile, ManagedEvaluator, OwnerRegistry,
        ProcessClock, SearchAuthority,
    },
    native_attestation::CpuProfileV1,
};
use rz_contracts::*;
use rz_encoding::classical::HistoryFill;
use rz_eval::{
    asset::{self, AssetMetadata, MaiaAsset},
    contracts::{
        ClassicalProjection, HOST_BYTES_PER_ITEM, MaiaBinding, PhysicalFailure, encoding_manifest,
    },
    error::{BackendError, CauseCode, FailureKind, FailureStage},
    native_runtime_bridge::{
        NATIVE_RUNTIME_OVERHEAD_BYTES, NativeDiagnosticBatch, NativeDiagnosticKind,
        NativeDiagnosticReceipt, NativeRuntimeBackend, NativeWorkerOrigin, NativeWorkerOwner,
    },
    onnx::{BackendConfig, IO_BYTES_PER_ITEM, OnnxBackend, OrtRuntime},
    runtime_pin::RuntimeCache,
};
#[cfg(all(feature = "onnx-cuda", target_os = "linux"))]
use rz_eval::{
    onnx::Provider,
    runtime_pin::{CudaRuntimeBundleSpec, RuntimeBundleFileRole},
};
use rz_position::contracts::RulesState;
use rz_runtime::{
    Clock, DrainState, Limits, Resources,
    contracts::{ContractClock, ContractEvaluator, ContractsAdapter, SharedScope},
};
use rz_telemetry::source::{SourceJournal, SourceStage};
use std::{
    fmt, fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard, TryLockError},
    task::Poll,
    thread,
    time::{Duration, Instant},
};

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cuda", target_os = "linux"))]
#[allow(dead_code)]
mod system_cuda;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cuda", target_os = "linux"))]
pub use system_cuda::{
    Lc0ParentListener, Lc0PendingSupervisor, Lc0ReceiptOwner, Lc0SupervisedSession,
    parent_preflight as lc0_parent_preflight,
};

const MAX_PATH_BYTES: usize = 4096;
const MAX_FATAL_RECEIPTS: usize = 32;
const NATIVE_DIAGNOSTIC_CAPACITY: usize = 32;
const FINAL_COLLECTION_LIMIT: Duration = Duration::from_secs(2);
pub const CUDA_ARENA_BYTES: usize = 1024 * 1024 * 1024;
pub const MAX_NATIVE_SIMULATIONS: u64 = 4096;
pub const MAX_NATIVE_TREE_EDGES: usize = 4_194_304;
const MAX_NATIVE_EVALUATIONS: u64 = MAX_NATIVE_SIMULATIONS + 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeProvider {
    Cpu,
    Cuda,
}

fn error(code: ErrorCode, stage: Stage, detail: &'static str) -> ContractError {
    ContractError::new(code, stage, detail)
}

/// Private local paths are never rendered by Debug, Display, or parser errors.
#[derive(Clone)]
pub struct NativeConfig {
    execution_experiments: rz_eval::onnx::ExecutionExperiments,
    raw_cache: bool,
    parallelism: usize,
    search_simulations: u64,
    search_max_edges: Option<usize>,
    final_move_policy: rz_search::tree::FinalMovePolicy,
    source_weights: PathBuf,
    onnx_model: PathBuf,
    export_manifest: PathBuf,
    manifest_sha256: [u8; 32],
    ort_library: PathBuf,
    ort_sha256: [u8; 32],
    output_root: PathBuf,
    runtime_cache_root: Option<PathBuf>,
    attestation: bool,
    batch_attestation: bool,
    profiling: bool,
    provider: NativeProvider,
    cuda_bundle: Option<PathBuf>,
    cuda_bundle_sha256: Option<[u8; 32]>,
    system_cuda_lock: Option<PathBuf>,
    system_cuda_lock_sha256: Option<[u8; 32]>,
}
impl fmt::Debug for NativeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeConfig")
            .field("manifest_sha256", &self.manifest_sha256)
            .field("ort_sha256", &self.ort_sha256)
            .field("provider", &self.provider)
            .field(
                "experiments",
                &(u8::from(self.execution_experiments.reuse_buffers)
                    | (u8::from(self.execution_experiments.io_binding) << 1)
                    | (u8::from(self.execution_experiments.cuda_graph) << 2)
                    | (u8::from(self.raw_cache) << 3)),
            )
            .field("batch", &self.parallelism)
            .field("search_simulations", &self.search_simulations)
            .field("search_max_edges", &self.search_max_edges)
            .field("final", &self.final_move_policy)
            .field(
                "bundle",
                &(self.cuda_bundle.is_some(), &self.cuda_bundle_sha256),
            )
            .field("system_cuda_lock", &self.system_cuda_lock.is_some())
            .field("system_cuda_lock_sha256", &self.system_cuda_lock_sha256)
            .finish_non_exhaustive()
    }
}
impl NativeConfig {
    pub fn parse(
        arguments: impl IntoIterator<Item = String>,
    ) -> Result<Self, NativeBootstrapError> {
        // Arena forwards this one diagnostic switch explicitly; ordinary
        // launches keep profiling off and no arbitrary environment is inherited.
        Self::parse_with_source_profile(
            arguments,
            std::env::var_os("RZ_NATIVE_SOURCE_PROFILE").as_deref()
                == Some(std::ffi::OsStr::new("1")),
        )
    }
    fn parse_with_source_profile(
        arguments: impl IntoIterator<Item = String>,
        source_profile: bool,
    ) -> Result<Self, NativeBootstrapError> {
        let mut selected_adapter = false;
        let mut filtered = Vec::new();
        for argument in arguments {
            if let Some(adapter) = argument.strip_prefix("--model-adapter=") {
                if selected_adapter || adapter != "lc0" {
                    return Err(NativeBootstrapError::Contract(error(
                        ErrorCode::UnsupportedContract,
                        Stage::Admission,
                        "unsupported or duplicate native model adapter",
                    )));
                }
                selected_adapter = true;
            } else {
                filtered.push(argument);
            }
        }
        let arguments = filtered;
        let mut execution_experiments = rz_eval::onnx::ExecutionExperiments::default();
        let mut raw_cache = false;
        let mut parallelism = None;
        let mut search_simulations = None;
        let mut search_max_edges = None;
        let mut final_move_policy = None;
        let mut provider = None;
        let mut source_weights = None;
        let mut onnx_model = None;
        let mut export_manifest = None;
        let mut manifest_sha256 = None;
        let mut ort_library = None;
        let mut ort_sha256 = None;
        let mut output_root = None;
        let mut runtime_cache_root = None;
        let mut attestation = false;
        let mut batch_attestation = false;
        let mut profiling = false;
        let mut cuda_bundle = None;
        let mut cuda_bundle_sha256 = None;
        let mut system_cuda_lock = None;
        let mut system_cuda_lock_sha256 = None;
        for argument in arguments {
            let experiment = match argument.as_str() {
                "--experimental-io-buffers" => Some(&mut execution_experiments.reuse_buffers),
                "--experimental-io-binding" => Some(&mut execution_experiments.io_binding),
                "--experimental-cuda-graph" => Some(&mut execution_experiments.cuda_graph),
                "--experimental-raw-cache" => Some(&mut raw_cache),
                _ => None,
            };
            if let Some(enabled) = experiment {
                if std::mem::replace(enabled, true) {
                    return Err(NativeBootstrapError::Config(
                        "duplicate execution experiment",
                    ));
                }
                continue;
            }
            if argument == "--profile" {
                if std::mem::replace(&mut profiling, true) {
                    return Err(NativeBootstrapError::Config(
                        "duplicate native profile flag",
                    ));
                }
                continue;
            }
            if argument == "--batch-attestation" {
                if std::mem::replace(&mut batch_attestation, true) {
                    return Err(NativeBootstrapError::Config("duplicate batch attestation"));
                }
                continue;
            }
            if argument == "--attestation" {
                if std::mem::replace(&mut attestation, true) {
                    return Err(NativeBootstrapError::Config(
                        "duplicate native attestation flag",
                    ));
                }
                continue;
            }
            if argument == "--onnx-cpu" || argument == "--onnx-cuda" {
                let selected = if argument == "--onnx-cpu" {
                    NativeProvider::Cpu
                } else {
                    NativeProvider::Cuda
                };
                if provider.replace(selected).is_some() {
                    return Err(NativeBootstrapError::Config(
                        "duplicate native provider flag",
                    ));
                }
                continue;
            }
            let (name, value) = argument
                .split_once('=')
                .ok_or(NativeBootstrapError::Config(
                    "native CPU arguments require named asset/hash values",
                ))?;
            match name {
                "--final-selection" => {
                    let selected = match value {
                        "visits" => rz_search::tree::FinalMovePolicy::Visits,
                        "exact-terminal" => rz_search::tree::FinalMovePolicy::ExactTerminal,
                        _ => {
                            return Err(NativeBootstrapError::Config(
                                "unsupported final selection policy",
                            ));
                        }
                    };
                    if final_move_policy.replace(selected).is_some() {
                        return Err(NativeBootstrapError::Config(
                            "duplicate final selection policy",
                        ));
                    }
                }
                "--search-simulations" => {
                    if search_simulations.is_some() {
                        return Err(NativeBootstrapError::Config(
                            "duplicate search simulation limit",
                        ));
                    }
                    let limit = value.parse::<u64>().map_err(|_| {
                        NativeBootstrapError::Config("invalid search simulation limit")
                    })?;
                    if !(1..=MAX_NATIVE_SIMULATIONS).contains(&limit) {
                        return Err(NativeBootstrapError::Config(
                            "native search simulation limit must be 1..=4096",
                        ));
                    }
                    search_simulations = Some(limit);
                }
                "--search-max-edges" => {
                    if search_max_edges.is_some() {
                        return Err(NativeBootstrapError::Config("duplicate search edge limit"));
                    }
                    let limit = value
                        .parse::<usize>()
                        .map_err(|_| NativeBootstrapError::Config("invalid search edge limit"))?;
                    if !(1..=MAX_NATIVE_TREE_EDGES).contains(&limit) {
                        return Err(NativeBootstrapError::Config(
                            "native search edge limit must be 1..=4194304",
                        ));
                    }
                    search_max_edges = Some(limit);
                }
                "--experimental-batch" => {
                    if parallelism.is_some() {
                        return Err(NativeBootstrapError::Config("duplicate batch width"));
                    }
                    parallelism = Some(
                        value
                            .parse::<usize>()
                            .map_err(|_| NativeBootstrapError::Config("invalid batch width"))?,
                    );
                }
                "--source-weights" => set_path(&mut source_weights, value)?,
                "--onnx-model" => set_path(&mut onnx_model, value)?,
                "--export-manifest" => set_path(&mut export_manifest, value)?,
                "--manifest-sha256" => set_hash(&mut manifest_sha256, value)?,
                "--ort-library" => set_path(&mut ort_library, value)?,
                "--ort-sha256" => set_hash(&mut ort_sha256, value)?,
                "--output-root" => set_path(&mut output_root, value)?,
                "--runtime-cache-root" => set_path(&mut runtime_cache_root, value)?,
                "--cuda-bundle" => set_path(&mut cuda_bundle, value)?,
                "--cuda-bundle-sha256" => set_hash(&mut cuda_bundle_sha256, value)?,
                "--system-cuda-lock" => set_path(&mut system_cuda_lock, value)?,
                "--system-cuda-lock-sha256" => set_hash(&mut system_cuda_lock_sha256, value)?,
                _ => {
                    return Err(NativeBootstrapError::Config(
                        "unsupported or mixed native provider argument",
                    ));
                }
            }
        }
        let provider = provider.ok_or(NativeBootstrapError::Config(
            "explicit --onnx-cpu or --onnx-cuda selection is required",
        ))?;
        let system_requested = system_cuda_lock.is_some() || system_cuda_lock_sha256.is_some();
        if system_requested
            && (provider != NativeProvider::Cuda
                || system_cuda_lock.is_none()
                || system_cuda_lock_sha256.is_none()
                || cuda_bundle.is_some()
                || cuda_bundle_sha256.is_some()
                || runtime_cache_root.is_some()
                || attestation
                || batch_attestation
                || !cfg!(all(
                    feature = "pals-adaptive",
                    feature = "onnx-cuda",
                    target_os = "linux"
                )))
        {
            return Err(NativeBootstrapError::Config(
                "system CUDA requires its lock and SHA, Linux system support and no bundle, runtime cache or legacy attestation",
            ));
        }
        match provider {
            NativeProvider::Cpu if cuda_bundle.is_some() || cuda_bundle_sha256.is_some() => {
                return Err(NativeBootstrapError::Config(
                    "CPU provider rejects CUDA bundle arguments",
                ));
            }
            NativeProvider::Cuda
                if !system_requested && (cuda_bundle.is_none() || cuda_bundle_sha256.is_none()) =>
            {
                return Err(NativeBootstrapError::Config(
                    "CUDA provider requires the bundle file and its explicit SHA256",
                ));
            }
            _ => {}
        }
        let parallelism = parallelism.unwrap_or(1);
        profiling |= source_profile;
        // D02 v1 records one fresh execution per accepted request. Raw hits,
        // batching and experimental I/O need distinct provenance/timing schemas.
        if profiling
            && (raw_cache
                || parallelism != 1
                || execution_experiments != rz_eval::onnx::ExecutionExperiments::default())
        {
            return Err(NativeBootstrapError::Config(
                "native source profile v1 requires default B1 execution without raw cache",
            ));
        }
        if !(1..=16).contains(&parallelism)
            || (parallelism > 1 && !cfg!(feature = "experimental-batch"))
            || (parallelism > 1 && (attestation || execution_experiments.cuda_graph))
        {
            return Err(NativeBootstrapError::Config(
                "experimental batch requires compiled width 1..=16 and excludes CUDA Graph/B1 attestation",
            ));
        }
        if (execution_experiments.reuse_buffers && !cfg!(feature = "experimental-io-buffers"))
            || (execution_experiments.io_binding && !cfg!(feature = "experimental-io-binding"))
            || (execution_experiments.cuda_graph
                && (!cfg!(feature = "experimental-cuda-graph")
                    || !execution_experiments.io_binding
                    || provider != NativeProvider::Cuda))
            || (raw_cache && !cfg!(feature = "experimental-raw-cache"))
            || (attestation
                && (raw_cache
                    || execution_experiments != rz_eval::onnx::ExecutionExperiments::default()))
        {
            return Err(NativeBootstrapError::Config(
                "unsupported experiment or Computed-only V1 attestation requested for experimental execution",
            ));
        }
        let missing = || {
            NativeBootstrapError::Config(
                "native CPU requires all asset, manifest, runtime hash and private output-root arguments",
            )
        };
        if batch_attestation
            && (!cfg!(feature = "experimental-batch")
                || provider != NativeProvider::Cuda
                || attestation
                || profiling
                || raw_cache
                || execution_experiments != rz_eval::onnx::ExecutionExperiments::default()
                || final_move_policy.unwrap_or_default()
                    != rz_search::tree::FinalMovePolicy::Visits
                || search_simulations != Some(4096))
        {
            return Err(NativeBootstrapError::Config(
                "batch attestation requires explicit CUDA/batch feature, visits/4096 and no other experiment or V1 attestation",
            ));
        }
        Ok(Self {
            execution_experiments,
            raw_cache,
            parallelism,
            search_simulations: search_simulations.unwrap_or(128),
            search_max_edges,
            final_move_policy: final_move_policy.unwrap_or_default(),
            source_weights: source_weights.ok_or_else(missing)?,
            onnx_model: onnx_model.ok_or_else(missing)?,
            export_manifest: export_manifest.ok_or_else(missing)?,
            manifest_sha256: manifest_sha256.ok_or_else(missing)?,
            ort_library: ort_library.ok_or_else(missing)?,
            ort_sha256: ort_sha256.ok_or_else(missing)?,
            output_root: output_root.ok_or_else(missing)?,
            runtime_cache_root,
            attestation,
            batch_attestation,
            profiling,
            provider,
            cuda_bundle,
            cuda_bundle_sha256,
            system_cuda_lock,
            system_cuda_lock_sha256,
        })
    }
    pub fn attestation_requested(&self) -> bool {
        self.attestation
    }
    pub fn batch_attestation_requested(&self) -> bool {
        self.batch_attestation
    }
    pub fn max_batch(&self) -> usize {
        self.parallelism
    }
    pub fn profiling_requested(&self) -> bool {
        self.profiling
    }
    #[cfg(feature = "onnx-cuda")]
    fn cuda_attestation_width(&self) -> Option<usize> {
        // Only the separate batch writer may describe a multi-item provider.
        // CLI admission keeps legacy --attestation closed to B1 and excludes
        // buffer/binding/Graph changes from either receipt path.
        (self.provider == NativeProvider::Cuda
            && (self.parallelism == 1 || self.batch_attestation)
            && self.execution_experiments == rz_eval::onnx::ExecutionExperiments::default())
        .then_some(self.parallelism)
    }
    pub fn engine_settings(&self) -> crate::engine::EngineSettings {
        let mut settings = crate::engine::EngineSettings {
            max_workers: 1,
            ..crate::engine::EngineSettings::default()
        };
        settings.search.max_simulations = self.search_simulations;
        if let Some(max_edges) = self.search_max_edges {
            settings.tree.max_edges = max_edges;
        }
        settings.final_move_policy = self.final_move_policy;
        settings
    }
    fn source_journal(
        &self,
    ) -> Result<Option<SourceJournal<CompletionContext>>, NativeBootstrapError> {
        if self.profiling {
            SourceJournal::try_new(8192)
                .map(Some)
                .map_err(NativeBootstrapError::Config)
        } else {
            Ok(None)
        }
    }
    pub fn provider(&self) -> NativeProvider {
        self.provider
    }
    pub(crate) fn attestation_output_root(&self) -> Result<PathBuf, NativeBootstrapError> {
        private_output_root(&self.output_root).map_err(Into::into)
    }
}
fn set_path(slot: &mut Option<PathBuf>, value: &str) -> Result<(), NativeBootstrapError> {
    if slot.is_some() {
        return Err(NativeBootstrapError::Config(
            "duplicate native asset argument",
        ));
    }
    if value.is_empty() || value.len() > MAX_PATH_BYTES || value.chars().any(char::is_control) {
        return Err(NativeBootstrapError::Config(
            "empty, oversized or invalid native asset path",
        ));
    }
    *slot = Some(PathBuf::from(value));
    Ok(())
}
fn set_hash(slot: &mut Option<[u8; 32]>, value: &str) -> Result<(), NativeBootstrapError> {
    if slot.is_some() {
        return Err(NativeBootstrapError::Config(
            "duplicate native hash argument",
        ));
    }
    *slot = Some(asset::parse_sha256(value).map_err(|_| {
        NativeBootstrapError::Config(
            "native hash must contain exactly 64 lowercase hexadecimal characters",
        )
    })?);
    Ok(())
}

#[derive(Debug)]
pub enum NativeBootstrapError {
    Config(&'static str),
    Backend(Box<BackendError>),
    Physical(Box<PhysicalFailure>),
    Contract(ContractError),
}
impl fmt::Display for NativeBootstrapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(detail) => f.write_str(detail),
            Self::Backend(error) => fmt::Display::fmt(error, f),
            Self::Physical(error) => match &error.backend {
                Some(backend) => fmt::Display::fmt(backend, f),
                None => fmt::Display::fmt(&error.contract, f),
            },
            Self::Contract(error) => fmt::Display::fmt(error, f),
        }
    }
}
impl std::error::Error for NativeBootstrapError {}
impl From<BackendError> for NativeBootstrapError {
    fn from(error: BackendError) -> Self {
        Self::Backend(Box::new(error))
    }
}
impl From<PhysicalFailure> for NativeBootstrapError {
    fn from(error: PhysicalFailure) -> Self {
        Self::Physical(Box::new(error))
    }
}
impl From<ContractError> for NativeBootstrapError {
    fn from(error: ContractError) -> Self {
        Self::Contract(error)
    }
}

/// A bounded aggregate: middle normal audit receipts are not a request journal.
#[derive(Debug, Default)]
pub struct NativeAuditAggregate {
    pub count: u64,
    pub first: Option<NativeDiagnosticReceipt>,
    pub last: Option<NativeDiagnosticReceipt>,
}
impl NativeAuditAggregate {
    fn accept(&mut self, receipt: NativeDiagnosticReceipt) -> Option<NativeDiagnosticReceipt> {
        let Some(count) = self.count.checked_add(1) else {
            return Some(receipt);
        };
        if self.first.is_none() {
            self.first = Some(receipt.clone());
        }
        self.last = Some(receipt);
        self.count = count;
        None
    }
}

/// Typed process evidence. Canceled/expired prelaunch audits are count+first/last
/// aggregates, not full request provenance. Fatal receipts preserve C's bounded
/// original BackendError; ordinary Display never renders its raw native text.
#[derive(Debug)]
pub struct NativeRunReport {
    pub model_manifest: Digest,
    pub backend: Digest,
    pub origin: NativeWorkerOrigin,
    /// Completed results accepted by D, not total physical native invocations.
    pub completed_by_runtime: u64,
    /// Experimental raw reuse is separate from physical Computed evidence.
    #[cfg(feature = "experimental-raw-cache")]
    pub raw_cache_completions: NativeSearchAggregate,
    #[cfg(feature = "experimental-raw-cache")]
    pub raw_cache_root_initializations: NativeSearchAggregate,
    #[cfg(feature = "experimental-raw-cache")]
    pub raw_cache_non_root_backups: NativeSearchAggregate,
    pub first_completed: Option<NativeCompletedReceipt>,
    pub last_completed: Option<NativeCompletedReceipt>,
    /// Metadata observed only after B's unchanged final tree guard committed.
    pub search_root_initializations: NativeSearchAggregate,
    pub search_non_root_backups: NativeSearchAggregate,
    pub observations: NativeObservationReport,
    pub canceled: NativeAuditAggregate,
    pub expired: NativeAuditAggregate,
    pub failures: Vec<NativeDiagnosticReceipt>,
    pub overflow: Option<NativeDiagnosticReceipt>,
    pub boundary_error: Option<ContractError>,
    pub poison_error: Option<ContractError>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeCompletedReceipt {
    pub context: CompletionContext,
    pub actual: ActualCompute,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeSearchConsumedReceipt {
    pub completed: NativeCompletedReceipt,
    pub traversed_edges: usize,
}
#[derive(Debug, Default)]
pub struct NativeSearchAggregate {
    pub count: u64,
    pub first: Option<NativeSearchConsumedReceipt>,
    pub last: Option<NativeSearchConsumedReceipt>,
}
#[cfg(feature = "experimental-raw-cache")]
fn valid_raw_hit(
    context: CompletionContext,
    actual: ActualCompute,
    report: &NativeRunReport,
) -> bool {
    actual.precision == PrecisionProfile::Fp32
        && actual.steps == 1
        && actual.full
        && actual.backend == report.backend
        && actual.execution.is_none()
        && context.execution.is_none()
        && context.request.backend == report.backend
        && context.request.model.manifest == report.model_manifest
        && matches!(actual.provenance, CacheProvenance::RawEvalHit { source_execution: Some(source) } if source.epoch == context.request.request.epoch)
}
#[cfg(feature = "experimental-raw-cache")]
fn record_raw_receipt(
    aggregate: &mut NativeSearchAggregate,
    context: CompletionContext,
    actual: ActualCompute,
    traversed_edges: usize,
) -> Result<(), ContractError> {
    let receipt = NativeSearchConsumedReceipt {
        completed: NativeCompletedReceipt { context, actual },
        traversed_edges,
    };
    aggregate.first.get_or_insert(receipt);
    aggregate.last = Some(receipt);
    aggregate.count = aggregate.count.checked_add(1).ok_or_else(|| {
        error(
            ErrorCode::ResourceExhausted,
            Stage::Output,
            "native raw-cache receipt count exhausted",
        )
    })?;
    Ok(())
}
#[derive(Debug, Default)]
pub struct NativeObservationReport {
    pub scheduler_events: u64,
    pub scheduler_dropped: u64,
    pub scheduler_counter_overflow: bool,
    pub delivery_events: u64,
    pub delivery_dropped: u64,
    pub delivery_counter_overflow: bool,
    pub drain_discarded_results: u64,
}
impl NativeRunReport {
    fn new(profile: &EvaluatorProfile, origin: NativeWorkerOrigin) -> Result<Self, ContractError> {
        let mut failures = Vec::new();
        failures
            .try_reserve_exact(MAX_FATAL_RECEIPTS)
            .map_err(|_| {
                error(
                    ErrorCode::ResourceExhausted,
                    Stage::Admission,
                    "cannot reserve bounded native evidence store",
                )
            })?;
        Ok(Self {
            model_manifest: profile.model.handle().manifest,
            backend: profile.backend,
            origin,
            completed_by_runtime: 0,
            #[cfg(feature = "experimental-raw-cache")]
            raw_cache_completions: NativeSearchAggregate::default(),
            #[cfg(feature = "experimental-raw-cache")]
            raw_cache_root_initializations: NativeSearchAggregate::default(),
            #[cfg(feature = "experimental-raw-cache")]
            raw_cache_non_root_backups: NativeSearchAggregate::default(),
            first_completed: None,
            last_completed: None,
            search_root_initializations: NativeSearchAggregate::default(),
            search_non_root_backups: NativeSearchAggregate::default(),
            observations: NativeObservationReport::default(),
            canceled: NativeAuditAggregate::default(),
            expired: NativeAuditAggregate::default(),
            failures,
            overflow: None,
            boundary_error: None,
            poison_error: None,
        })
    }
    pub fn has_failure(&self) -> bool {
        !self.failures.is_empty()
            || self.overflow.is_some()
            || self.boundary_error.is_some()
            || self.poison_error.is_some()
    }
    fn remaining_receipts(&self) -> usize {
        if self.overflow.is_some() {
            0
        } else {
            MAX_FATAL_RECEIPTS + 1 - self.failures.len()
        }
    }
    fn accept(&mut self, receipt: NativeDiagnosticReceipt) -> Option<ContractError> {
        let normal = receipt.kind == NativeDiagnosticKind::DispatchRefused
            && receipt.failure.backend.is_none();
        let aggregate = match (normal, receipt.failure.contract.code) {
            (true, ErrorCode::Canceled) => Some(&mut self.canceled),
            (true, ErrorCode::Expired) => Some(&mut self.expired),
            _ => None,
        };
        if let Some(aggregate) = aggregate {
            if let Some(receipt) = aggregate.accept(receipt) {
                let overflow = error(
                    ErrorCode::ResourceExhausted,
                    Stage::Output,
                    "native normal-audit count exhausted; original overflow receipt retained",
                );
                self.overflow = Some(receipt);
                self.boundary_error.get_or_insert(overflow);
                return Some(overflow);
            }
            return None;
        }
        let failure = receipt.failure.contract;
        if self.failures.len() < MAX_FATAL_RECEIPTS {
            self.failures.push(receipt);
        } else {
            debug_assert!(
                self.overflow.is_none(),
                "collector takes only remaining receipt slots"
            );
            self.overflow = Some(receipt);
            self.boundary_error.get_or_insert(error(
                ErrorCode::ResourceExhausted,
                Stage::Output,
                "native fatal evidence limit reached; first overflow original retained",
            ));
        }
        Some(failure)
    }
}
impl fmt::Display for NativeRunReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let provider = match self.origin {
            NativeWorkerOrigin::CpuOnnx => "ONNX CPU evidence",
            NativeWorkerOrigin::CudaOnnx => "ONNX CUDA evidence",
            NativeWorkerOrigin::Injected => "ONNX injected evidence",
        };
        write!(
            f,
            "{provider}: origin={:?} completed_by_runtime={} (fresh fullsteps=1 FP32 Computed; physical invocation count is separate), canceled={} expired={} (count+first/last; no full request journal), fatal={} overflow={} boundary={} poison={}",
            self.origin,
            self.completed_by_runtime,
            self.canceled.count,
            self.expired.count,
            self.failures.len(),
            self.overflow.is_some(),
            self.boundary_error.is_some(),
            self.poison_error.is_some()
        )?;
        if let Some(first) = self.first_completed {
            write!(
                f,
                "; first_completed={:?}/{:?}/{:?}/game={:?}/root={:?}",
                first.context.execution,
                first.context.request.request,
                first.context.request.selection,
                first.context.request.game,
                first.context.request.root
            )?;
        }
        if let Some(last) = self.last_completed {
            write!(
                f,
                "; last_completed={:?}/{:?}/{:?}/game={:?}/root={:?}",
                last.context.execution,
                last.context.request.request,
                last.context.request.selection,
                last.context.request.game,
                last.context.request.root
            )?;
        }
        if let Some(receipt) = self.failures.first().or(self.overflow.as_ref()) {
            write!(f, "; first fatal {:?} {:?}: ", receipt.ticket, receipt.kind)?;
            match &receipt.failure.backend {
                Some(backend) => fmt::Display::fmt(backend, f)?,
                None => fmt::Display::fmt(&receipt.failure.contract, f)?,
            }
        }
        Ok(())
    }
}

struct EvidenceState {
    report: Option<NativeRunReport>,
    closed: Option<ContractError>,
}
/// Retained on final collection timeout, so concurrent late receipts remain
/// owned and may be inspected explicitly instead of disappearing with an error.
#[derive(Clone)]
pub struct NativeEvidenceHandle(Arc<Mutex<EvidenceState>>);
impl NativeEvidenceHandle {
    fn record_search(
        &self,
        evaluation: &rz_search::contracts::AcceptedEvaluation,
        traversed_edges: usize,
    ) -> Result<(), ContractError> {
        let Some(mut state) = self.try_state()? else {
            return Err(error(
                ErrorCode::ResourceExhausted,
                Stage::Output,
                "native search evidence consumer busy",
            ));
        };
        let report = state.report.as_mut().ok_or_else(|| {
            error(
                ErrorCode::Canceled,
                Stage::Output,
                "native search evidence already transferred",
            )
        })?;
        if report.origin == NativeWorkerOrigin::Injected {
            return Ok(());
        }
        let context = evaluation.context;
        let actual = evaluation.actual;
        #[cfg(feature = "experimental-raw-cache")]
        if valid_raw_hit(context, actual, report) {
            let aggregate = if traversed_edges == 0 {
                &mut report.raw_cache_root_initializations
            } else {
                &mut report.raw_cache_non_root_backups
            };
            return record_raw_receipt(aggregate, context, actual, traversed_edges);
        }
        if actual.precision != PrecisionProfile::Fp32
            || actual.steps != 1
            || !actual.full
            || actual.provenance != CacheProvenance::Computed
            || actual.backend != report.backend
            || actual.execution.is_none()
            || context.execution != actual.execution
            || context.request.backend != report.backend
            || context.request.model.manifest != report.model_manifest
        {
            let boundary = error(
                ErrorCode::IdentityMismatch,
                Stage::Backup,
                "accepted native search metadata differs from the verified provider profile",
            );
            report.boundary_error.get_or_insert(boundary);
            state.closed.get_or_insert(boundary);
            return Err(boundary);
        }
        let aggregate = if traversed_edges == 0 {
            &mut report.search_root_initializations
        } else {
            &mut report.search_non_root_backups
        };
        let receipt = NativeSearchConsumedReceipt {
            completed: NativeCompletedReceipt { context, actual },
            traversed_edges,
        };
        aggregate.first.get_or_insert(receipt);
        aggregate.last = Some(receipt);
        match aggregate.count.checked_add(1) {
            Some(count) => {
                aggregate.count = count;
                Ok(())
            }
            None => {
                let boundary = error(
                    ErrorCode::ResourceExhausted,
                    Stage::Output,
                    "native search count exhausted; last original metadata retained",
                );
                report.boundary_error.get_or_insert(boundary);
                state.closed.get_or_insert(boundary);
                Err(boundary)
            }
        }
    }
    fn record_observations(&self, delta: NativeObservationReport) -> Result<(), ContractError> {
        let Some(mut state) = self.try_state()? else {
            return Err(error(
                ErrorCode::ResourceExhausted,
                Stage::Output,
                "native observation evidence consumer busy",
            ));
        };
        let report = state.report.as_mut().ok_or_else(|| {
            error(
                ErrorCode::Canceled,
                Stage::Output,
                "native observation evidence already transferred",
            )
        })?;
        let observations = &mut report.observations;
        let mut overflow = delta.scheduler_counter_overflow || delta.delivery_counter_overflow;
        for (counter, delta) in [
            (&mut observations.scheduler_events, delta.scheduler_events),
            (&mut observations.scheduler_dropped, delta.scheduler_dropped),
            (&mut observations.delivery_events, delta.delivery_events),
            (&mut observations.delivery_dropped, delta.delivery_dropped),
            (
                &mut observations.drain_discarded_results,
                delta.drain_discarded_results,
            ),
        ] {
            match counter.checked_add(delta) {
                Some(value) => *counter = value,
                None => {
                    *counter = u64::MAX;
                    overflow = true;
                }
            }
        }
        observations.scheduler_counter_overflow |= delta.scheduler_counter_overflow;
        observations.delivery_counter_overflow |= delta.delivery_counter_overflow || overflow;
        if overflow {
            let boundary = error(
                ErrorCode::ResourceExhausted,
                Stage::Output,
                "native observation counter overflow; journal is incomplete",
            );
            report.boundary_error.get_or_insert(boundary);
            state.closed.get_or_insert(boundary);
            return Err(boundary);
        }
        Ok(())
    }
    fn try_state(&self) -> Result<Option<MutexGuard<'_, EvidenceState>>, ContractError> {
        match self.0.try_lock() {
            Ok(state) => Ok(Some(state)),
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Poisoned(poisoned)) => {
                let mut state = poisoned.into_inner();
                let boundary = error(
                    ErrorCode::BackendFailure,
                    Stage::Output,
                    "native evidence mutex poisoned; admission closed and originals retained",
                );
                state.closed.get_or_insert(boundary);
                if let Some(report) = state.report.as_mut() {
                    report.poison_error.get_or_insert(boundary);
                }
                Ok(Some(state))
            }
        }
    }
    pub fn try_take_report(&self) -> Result<Option<NativeRunReport>, ContractError> {
        Ok(self.try_state()?.and_then(|mut state| state.report.take()))
    }
    fn admission_error(&self) -> Result<Option<ContractError>, ContractError> {
        self.try_state()?.map(|state| state.closed).ok_or_else(|| {
            error(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "native evidence consumer busy; request not admitted",
            )
        })
    }
    fn record_completed(&self, output: &EvalOutput) -> Result<(), ContractError> {
        let Some(mut state) = self.try_state()? else {
            return Err(error(
                ErrorCode::ResourceExhausted,
                Stage::Output,
                "native completion evidence consumer busy",
            ));
        };
        let report = state.report.as_mut().ok_or_else(|| {
            error(
                ErrorCode::Canceled,
                Stage::Output,
                "native completion evidence already transferred",
            )
        })?;
        if !matches!(
            report.origin,
            NativeWorkerOrigin::CpuOnnx | NativeWorkerOrigin::CudaOnnx
        ) {
            return Ok(());
        }
        #[cfg(feature = "experimental-raw-cache")]
        {
            let context = CompletionContext {
                request: output.context,
                execution: output.actual.execution,
            };
            if valid_raw_hit(context, output.actual, report) {
                return record_raw_receipt(
                    &mut report.raw_cache_completions,
                    context,
                    output.actual,
                    0,
                );
            }
        }
        if output.actual.precision != PrecisionProfile::Fp32
            || output.actual.steps != 1
            || !output.actual.full
            || output.actual.provenance != CacheProvenance::Computed
            || output.actual.backend != report.backend
            || output.actual.execution.is_none()
        {
            let boundary = error(
                ErrorCode::UnsupportedContract,
                Stage::Output,
                "D-completed native output differs from fresh single-step provider profile",
            );
            report.boundary_error.get_or_insert(boundary);
            state.closed.get_or_insert(boundary);
            return Err(boundary);
        }
        let receipt = NativeCompletedReceipt {
            context: CompletionContext {
                request: output.context,
                execution: output.actual.execution,
            },
            actual: output.actual,
        };
        if report.first_completed.is_none() {
            report.first_completed = Some(receipt);
        }
        report.last_completed = Some(receipt);
        let count = report.completed_by_runtime.checked_add(1).ok_or_else(|| {
            error(
                ErrorCode::ResourceExhausted,
                Stage::Output,
                "native completed-by-runtime count exhausted; last original receipt retained",
            )
        });
        match count {
            Ok(count) => {
                report.completed_by_runtime = count;
                Ok(())
            }
            Err(boundary) => {
                report.boundary_error.get_or_insert(boundary);
                state.closed.get_or_insert(boundary);
                Err(boundary)
            }
        }
    }
    fn collect(&self, owner: &NativeWorkerOwner) -> Result<bool, ContractError> {
        let Some(mut state) = self.try_state()? else {
            return Ok(false);
        };
        if let Some(closed) = state.closed {
            owner.try_close_admission(closed)?;
        }
        let Some(report) = state.report.as_mut() else {
            return Err(error(
                ErrorCode::Canceled,
                Stage::Output,
                "native evidence report already transferred; original owner retained",
            ));
        };
        // One transfer per poll keeps the consumer's first overflow original
        // exclusive. After overflow no further original is removed from C.
        let batch = match owner.try_take_diagnostics(report.remaining_receipts().min(1)) {
            Ok(batch) => batch,
            Err(boundary) => {
                report.boundary_error.get_or_insert(boundary);
                state.closed.get_or_insert(boundary);
                owner.try_close_admission(boundary)?;
                return Err(boundary);
            }
        };
        let Some(NativeDiagnosticBatch {
            entries,
            boundary_error,
            poison_error,
        }) = batch
        else {
            return Ok(false);
        };
        let mut fatal = None;
        for receipt in entries {
            if let Some(error) = report.accept(receipt) {
                fatal.get_or_insert(error);
            }
        }
        if let Some(boundary) = boundary_error {
            report.boundary_error.get_or_insert(boundary);
            fatal.get_or_insert(boundary);
        }
        if let Some(poison) = poison_error {
            report.poison_error.get_or_insert(poison);
            fatal.get_or_insert(poison);
        }
        if let Some(fatal) = fatal {
            state.closed.get_or_insert(fatal);
        }
        if let Some(closed) = state.closed {
            owner.try_close_admission(closed)?;
        }
        Ok(true)
    }
}

/// Both service and native evidence survive compound failure. If collection
/// could not finish within its bound, the original owner and shared evidence
/// capability are retained rather than claiming that physical work drained.
pub struct NativeRunError {
    pub service: Option<Box<EngineError>>,
    pub report: Option<Box<NativeRunReport>>,
    pub collection_error: Option<ContractError>,
    /// Process-worker teardown has its own original contract/native cause. A
    /// request-drain result must not replace failure to destroy/join the session.
    pub worker_shutdown_error: Option<Box<PhysicalFailure>>,
    /// Final loaded-image audit is independent of physical drain. Keep its
    /// bounded original backend cause even when collection is also incomplete.
    pub runtime_mapping_error: Option<Box<BackendError>>,
    pub retained_evidence: NativeEvidenceHandle,
    retained_owner: NativeWorkerOwner,
}
impl NativeRunError {
    pub fn retained_owner(&self) -> &NativeWorkerOwner {
        &self.retained_owner
    }
    pub fn worker_shutdown_failure(&self) -> Option<&PhysicalFailure> {
        self.worker_shutdown_error.as_deref()
    }
}
impl fmt::Debug for NativeRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeRunError")
            .field("service", &self.service)
            .field("report", &self.report)
            .field("collection_error", &self.collection_error)
            .field("worker_shutdown_error", &self.worker_shutdown_error)
            .finish_non_exhaustive()
    }
}
impl fmt::Display for NativeRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(service) = &self.service {
            write!(f, "protocol service failed: {service}; ")?;
        }
        if let Some(report) = &self.report {
            fmt::Display::fmt(report, f)?;
        } else {
            f.write_str("native evidence remains owned pending bounded collection")?;
        }
        if let Some(error) = &self.collection_error {
            write!(f, "; collection: {error}")?;
        }
        if let Some(error) = &self.worker_shutdown_error {
            write!(f, "; process worker shutdown: {error}")?;
        }
        if let Some(error) = &self.runtime_mapping_error {
            write!(f, "; final runtime mapping audit: {error}")?;
        }
        Ok(())
    }
}
impl std::error::Error for NativeRunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.worker_shutdown_failure()
            .map(|error| error as &(dyn std::error::Error + 'static))
    }
}

/// One verified session/worker for the executable process. No inference-count
/// lifetime quota: identity ledgers are constant-space and every root has B's
/// own finite 129-request/execution range. Fatal errors close future admission.
struct NativeSessionFactory {
    owner: NativeWorkerOwner,
    profile: EvaluatorProfile,
    evidence: NativeEvidenceHandle,
    _runtime: Option<OrtRuntime>,
    loaded_profile: Option<CpuProfileV1>,
    #[cfg(feature = "onnx-cuda")]
    loaded_cuda_profile: Option<CudaProfileV1>,
}

impl NativeSessionFactory {
    fn load_cpu(
        owners: &OwnerRegistry,
        config: &NativeConfig,
    ) -> Result<Self, NativeBootstrapError> {
        if config.provider != NativeProvider::Cpu {
            return Err(NativeBootstrapError::Config(
                "CPU factory requires the explicit CPU provider",
            ));
        }
        let (asset, runtime, backend, projection) = Self::load_parts(owners, config)?;
        let loaded_profile = if (config.parallelism == 1 || config.batch_attestation)
            && config.execution_experiments == rz_eval::onnx::ExecutionExperiments::default()
        {
            Some(CpuProfileV1::from_loaded(
                &asset,
                &runtime,
                &backend,
                projection.model(),
            )?)
        } else {
            None
        };
        #[cfg(feature = "experimental-batch")]
        let owner = if config.parallelism > 1 {
            NativeWorkerOwner::from_onnx_batched(backend, projection, NATIVE_DIAGNOSTIC_CAPACITY)?
        } else {
            NativeWorkerOwner::from_onnx_profiled(
                backend,
                projection,
                NATIVE_DIAGNOSTIC_CAPACITY,
                config.source_journal()?,
            )?
        };
        #[cfg(not(feature = "experimental-batch"))]
        let owner = NativeWorkerOwner::from_onnx_profiled(
            backend,
            projection,
            NATIVE_DIAGNOSTIC_CAPACITY,
            config.source_journal()?,
        )?;
        let mut factory = Self::from_owner(owner, Some(runtime))?;
        factory.loaded_profile = loaded_profile;
        Ok(factory)
    }
    fn load_parts(
        owners: &OwnerRegistry,
        config: &NativeConfig,
    ) -> Result<(AssetMetadata, OrtRuntime, OnnxBackend, ClassicalProjection), NativeBootstrapError>
    {
        if config.system_cuda_lock.is_some() {
            // The system loader and worker support this runtime identity, but
            // this legacy entry point has no original native-entry/failure
            // custody yet. Reject before opening assets or creating an output
            // until that owner can retain unknown physical completion.
            return Err(NativeBootstrapError::Config(
                "system CUDA native entry requires lifecycle custody; no runtime was started",
            ));
        }
        let parts_started = Instant::now();
        let asset = MaiaAsset::load(
            &config.source_weights,
            &config.onnx_model,
            &config.export_manifest,
        )?;
        if asset.manifest_digest() != config.manifest_sha256 {
            return Err(BackendError::new(
                FailureKind::IdentityMismatch,
                FailureStage::Asset,
                "export manifest file digest differs from explicit bootstrap pin",
            )
            .into());
        }
        let asset_verified = Instant::now();
        let output_root = private_output_root(&config.output_root)?;
        let runtime_cache = config
            .runtime_cache_root
            .as_deref()
            .map_or_else(RuntimeCache::for_user, RuntimeCache::open)?;
        let expected_ort: String = config
            .ort_sha256
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let mut backend_config = BackendConfig::cpu();
        backend_config.experiments = config.execution_experiments;
        backend_config.max_batch = config.parallelism;
        backend_config.intra_threads = 1;
        backend_config.host_io_bytes = config.parallelism * IO_BYTES_PER_ITEM;
        let pin = match config.provider {
            NativeProvider::Cpu => runtime_cache.library(&config.ort_library, &expected_ort)?,
            NativeProvider::Cuda => {
                #[cfg(all(feature = "onnx-cuda", target_os = "linux"))]
                {
                    let path = config
                        .cuda_bundle
                        .as_ref()
                        .ok_or(NativeBootstrapError::Config("CUDA bundle path is absent"))?;
                    let bytes = asset::read_bounded(path, 64 * 1024)?;
                    if Some(asset::sha256(&bytes)) != config.cuda_bundle_sha256 {
                        return Err(NativeBootstrapError::Config(
                            "CUDA bundle file differs from the explicit SHA256 pin",
                        ));
                    }
                    let text = std::str::from_utf8(&bytes).map_err(|_| {
                        NativeBootstrapError::Config("CUDA bundle must be bounded UTF-8 JSON")
                    })?;
                    let spec = CudaRuntimeBundleSpec::from_json(text)?;
                    let core = spec
                        .files
                        .iter()
                        .find(|file| file.role == RuntimeBundleFileRole::Core)
                        .ok_or(NativeBootstrapError::Config(
                            "CUDA bundle lacks its core declaration",
                        ))?;
                    if config
                        .ort_library
                        .file_name()
                        .and_then(|name| name.to_str())
                        != Some(core.filename.as_str())
                        || core.sha256 != expected_ort
                    {
                        return Err(NativeBootstrapError::Config(
                            "explicit ORT core name/hash differs from the CUDA bundle",
                        ));
                    }
                    let profile_directory =
                        output_root.join(format!("native-cuda-placement-{}", std::process::id()));
                    use std::os::unix::fs::DirBuilderExt;
                    fs::DirBuilder::new()
                        .mode(0o700)
                        .create(&profile_directory)
                        .map_err(|_| {
                            NativeBootstrapError::Config(
                                "cannot reserve a fresh private CUDA placement directory",
                            )
                        })?;
                    backend_config.provider = Provider::Cuda {
                        device_id: 0,
                        arena_bytes: asset.profile().cuda_arena_bytes(),
                    };
                    backend_config.profiling_prefix = Some(profile_directory.join("placement"));
                    runtime_cache.cuda_bundle(
                        config
                            .ort_library
                            .parent()
                            .ok_or(NativeBootstrapError::Config(
                                "CUDA core lacks a source parent",
                            ))?,
                        &spec,
                    )?
                }
                #[cfg(not(all(feature = "onnx-cuda", target_os = "linux")))]
                {
                    return Err(NativeBootstrapError::Config(
                        "explicit CUDA native startup requires the onnx-cuda feature on Linux; no provider was started",
                    ));
                }
            }
        };
        let runtime_pinned = Instant::now();
        let runtime = OrtRuntime::load(&pin)?;
        let runtime_loaded = Instant::now();
        let (asset, backend) = OnnxBackend::load_owned(&runtime, asset, backend_config)?;
        let backend_loaded = Instant::now();
        let backend_timings = backend.startup_timings();
        // Storage evidence is a separate sidecar, not a change to E's closed
        // inference/attestation schema or backend identity. No private paths.
        let storage_receipt = serde_json::json!({
            "schema": 1,
            "runtime_storage": pin.storage(),
            "runtime_sha256": expected_ort,
            "runtime_bundle_sha256": runtime.bundle_digest(),
            "native_startup_loaded": true,
            "startup_host_timing": {
                "schema": 1,
                "scope": "model_runtime_load_parts_before_profile_and_worker",
                "asset_verification_ns": asset_verified.duration_since(parts_started).as_nanos(),
                "runtime_pin_ns": runtime_pinned.duration_since(asset_verified).as_nanos(),
                "runtime_load_ns": runtime_loaded.duration_since(runtime_pinned).as_nanos(),
                "backend_load_ns": backend_loaded.duration_since(runtime_loaded).as_nanos(),
                "backend_setup_ns": backend_timings.backend_setup.as_nanos(),
                "ort_model_commit_ns": backend_timings.model_commit.as_nanos(),
                "cuda_probe_validation_ns": backend_timings.cuda_probe_validation.map(|time| time.as_nanos()),
                "total_load_parts_ns": backend_loaded.duration_since(parts_started).as_nanos(),
            },
        });
        use std::io::Write;
        let mut receipt = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output_root.join(format!(
                "native-runtime-storage-{}.json",
                std::process::id()
            )))
            .map_err(|cause| {
                BackendError::new(
                    FailureKind::Io,
                    FailureStage::Backend,
                    "cannot create native runtime storage receipt",
                )
                .with_external_cause(CauseCode::RuntimePath, &cause)
            })?;
        receipt
            .write_all(&serde_json::to_vec_pretty(&storage_receipt).map_err(|_| {
                NativeBootstrapError::Config("cannot serialize runtime storage receipt")
            })?)
            .map_err(|cause| {
                BackendError::new(
                    FailureKind::Io,
                    FailureStage::Backend,
                    "cannot write native runtime storage receipt",
                )
                .with_external_cause(CauseCode::RuntimePath, &cause)
            })?;
        let fill = HistoryFill::No;
        let encoding = EncodingHandle {
            owner: owners.allocate()?,
            slot: 0,
            generation: SlotGeneration(1),
            manifest: encoding_manifest(fill),
        };
        let model = ModelHandle {
            owner: owners.allocate()?,
            slot: 0,
            generation: SlotGeneration(1),
            manifest: Digest(asset.manifest_digest()),
        };
        let projection =
            ClassicalProjection::new(MaiaBinding::for_backend(&backend, model, encoding, fill)?);
        #[cfg(feature = "experimental-raw-cache")]
        if config.raw_cache {
            projection.configure_raw_cache(rz_eval::raw_cache::RawCacheLimits::default())?;
        }
        Ok((asset, runtime, backend, projection))
    }
    #[cfg(feature = "onnx-cuda")]
    fn load_cuda(
        owners: &OwnerRegistry,
        config: &NativeConfig,
    ) -> Result<Self, NativeBootstrapError> {
        if !cfg!(target_os = "linux") {
            return Err(NativeBootstrapError::Config(
                "CUDA native startup is supported only on Linux; no assets or provider were loaded",
            ));
        }
        if config.provider != NativeProvider::Cuda {
            return Err(NativeBootstrapError::Config(
                "CUDA factory requires the explicit CUDA provider",
            ));
        }
        let (asset, runtime, backend, projection) = Self::load_parts(owners, config)?;
        let loaded_profile = if let Some(width) = config.cuda_attestation_width() {
            Some(CudaProfileV1::from_loaded_limit(
                &asset,
                &runtime,
                &backend,
                projection.model(),
                config
                    .cuda_bundle_sha256
                    .ok_or(NativeBootstrapError::Config(
                        "CUDA bundle file SHA256 is absent",
                    ))?,
                width,
            )?)
        } else {
            None
        };
        #[cfg(feature = "experimental-batch")]
        let owner = if config.parallelism > 1 {
            NativeWorkerOwner::from_cuda_onnx_batched(
                backend,
                projection,
                NATIVE_DIAGNOSTIC_CAPACITY,
            )?
        } else {
            NativeWorkerOwner::from_cuda_onnx_profiled(
                backend,
                projection,
                NATIVE_DIAGNOSTIC_CAPACITY,
                config.source_journal()?,
            )?
        };
        #[cfg(not(feature = "experimental-batch"))]
        let owner = NativeWorkerOwner::from_cuda_onnx_profiled(
            backend,
            projection,
            NATIVE_DIAGNOSTIC_CAPACITY,
            config.source_journal()?,
        )?;
        let mut factory = Self::from_owner(owner, Some(runtime))?;
        #[cfg(feature = "experimental-batch")]
        if config.batch_attestation {
            factory.owner.enable_batch_journal()?;
        }
        factory.loaded_cuda_profile = loaded_profile;
        Ok(factory)
    }
    fn from_owner(
        owner: NativeWorkerOwner,
        runtime: Option<OrtRuntime>,
    ) -> Result<Self, ContractError> {
        let profile = EvaluatorProfile {
            model: Arc::clone(owner.projection().model()),
            precision: PrecisionProfile::Fp32,
            backend: owner.projection().backend(),
            compute: ComputeBudget {
                min_steps: 1,
                max_steps: 1,
                require_full: true,
            },
            bytes: ByteBudget {
                host: HOST_BYTES_PER_ITEM,
                // The frozen Rules input is host-owned. C reserves CUDA arena
                // admission separately in its execution additional_resources.
                device: 0,
                pinned: 0,
            },
        };
        let evidence = NativeEvidenceHandle(Arc::new(Mutex::new(EvidenceState {
            report: Some(NativeRunReport::new(&profile, owner.origin())?),
            closed: None,
        })));
        Ok(Self {
            owner,
            profile,
            evidence,
            _runtime: runtime,
            loaded_profile: None,
            #[cfg(feature = "onnx-cuda")]
            loaded_cuda_profile: None,
        })
    }
    pub(crate) fn attestation_profile(&self) -> Result<&CpuProfileV1, ContractError> {
        if self.owner.origin() != NativeWorkerOrigin::CpuOnnx || self._runtime.is_none() {
            return Err(error(
                ErrorCode::UnsupportedContract,
                Stage::Output,
                "injected owner cannot issue native CPU provider evidence",
            ));
        }
        self.loaded_profile.as_ref().ok_or_else(|| {
            error(
                ErrorCode::IdentityMismatch,
                Stage::Output,
                "loaded CPU provider snapshot is unavailable",
            )
        })
    }
    /// Called for serve success and failure. Request drain and process-worker
    /// teardown share one deadline. A busy owner or pending reaper never permits
    /// report transfer, and a bounded failure keeps the original capabilities.
    pub fn finish(
        &self,
        served: Result<(), EngineError>,
    ) -> Result<NativeRunReport, NativeRunError> {
        self.finish_until(served, Instant::now() + FINAL_COLLECTION_LIMIT)
    }
    fn finish_until(
        &self,
        served: Result<(), EngineError>,
        until: Instant,
    ) -> Result<NativeRunReport, NativeRunError> {
        let mut service = served.err();
        self.finish_until_with_mapping(&mut service, until, false).0
    }

    // System runtime mappings are a separate observation, not a reason to
    // convert unavailable /proc evidence into a physical completion failure.
    // The legacy bundled route keeps its strict mapping audit unchanged.
    fn finish_until_with_mapping(
        &self,
        service: &mut Option<EngineError>,
        until: Instant,
        system_observation: bool,
    ) -> (
        Result<NativeRunReport, NativeRunError>,
        Option<rz_eval::onnx::CudaRuntimeMappingEvidence>,
    ) {
        let mut collection_error = None;
        let mut worker_shutdown_error = None;
        let mut worker_joined = false;
        loop {
            // Request terminal admission closure before other drain waits. A
            // busy owner lock remains Pending until C linearizes that closure;
            // it keeps active-lease polling available and starts its reaper only
            // once the physical lease has drained.
            if !worker_joined && worker_shutdown_error.is_none() {
                match self.owner.try_shutdown() {
                    Poll::Ready(Ok(())) => worker_joined = true,
                    Poll::Ready(Err(failure)) => {
                        worker_shutdown_error = Some(Box::new(failure));
                    }
                    Poll::Pending => {}
                }
            }
            if Instant::now() >= until {
                collection_error = Some(error(
                    ErrorCode::Expired,
                    Stage::Output,
                    "native final collection or worker join deadline reached; owner and evidence retained",
                ));
                break;
            }
            match confirm_collected_drain(
                || self.evidence.collect(&self.owner),
                || self.owner.try_status(),
            ) {
                Ok(true) if worker_joined => break,
                Ok(true) => {
                    if let Some(failure) = &worker_shutdown_error {
                        collection_error = Some(failure.contract);
                        break;
                    }
                }
                Ok(false) => {}
                Err(error) => {
                    collection_error = Some(error);
                    break;
                }
            }
            thread::sleep(Duration::from_millis(1));
        }
        let mut mapping_observation = None;
        let runtime_mapping_error =
            if collection_error.is_none() && self.owner.origin() == NativeWorkerOrigin::CudaOnnx {
                if system_observation {
                    match self._runtime.as_ref() {
                        Some(runtime) if runtime.is_system_cuda() => {
                            match runtime.observe_cuda_runtime_mappings() {
                                Ok(observation) => {
                                    mapping_observation = Some(observation);
                                    None
                                }
                                Err(error) => Some(Box::new(error)),
                            }
                        }
                        _ => Some(Box::new(BackendError::new(
                            FailureKind::IdentityMismatch,
                            FailureStage::Backend,
                            "original system CUDA runtime is unavailable during final observation",
                        ))),
                    }
                } else {
                    self._runtime
                        .as_ref()
                        .and_then(|runtime| runtime.verify_cuda_runtime_mappings().err())
                        .map(Box::new)
                }
            } else {
                None
            };
        // If physical work, destructor/join or final collection is unconfirmed,
        // leave the report in its shared store for the retained owner/collector.
        let report = if collection_error.is_none() {
            match self.evidence.try_take_report() {
                Ok(report) => report,
                Err(error) => {
                    collection_error.get_or_insert(error);
                    None
                }
            }
        } else {
            None
        };
        if service.is_none()
            && collection_error.is_none()
            && worker_shutdown_error.is_none()
            && runtime_mapping_error.is_none()
            && report.as_ref().is_some_and(|report| !report.has_failure())
        {
            return (
                Ok(report.expect("checked report presence")),
                mapping_observation,
            );
        }
        (
            Err(NativeRunError {
                report: report.map(Box::new),
                collection_error,
                worker_shutdown_error,
                runtime_mapping_error,
                retained_evidence: self.evidence.clone(),
                retained_owner: self.owner.clone(),
                // Move the original service failure only after the other
                // report fields have been prepared. The caller retains it if
                // those preparations unwind.
                service: service.take().map(Box::new),
            }),
            mapping_observation,
        )
    }
}

/// CPU construction and CPU V1 evidence remain guarded independently of CUDA.
pub struct NativeCpuFactory {
    inner: NativeSessionFactory,
}
impl NativeCpuFactory {
    #[cfg(feature = "experimental-raw-cache")]
    pub fn configure_raw_cache(
        &self,
        limits: rz_eval::raw_cache::RawCacheLimits,
    ) -> Result<(), ContractError> {
        self.inner.owner.projection().configure_raw_cache(limits)
    }
    pub fn load(
        owners: &OwnerRegistry,
        config: &NativeConfig,
    ) -> Result<Self, NativeBootstrapError> {
        Ok(Self {
            inner: NativeSessionFactory::load_cpu(owners, config)?,
        })
    }
    #[cfg(test)]
    fn from_owner(
        owner: NativeWorkerOwner,
        runtime: Option<OrtRuntime>,
    ) -> Result<Self, ContractError> {
        if owner.origin() == NativeWorkerOrigin::CudaOnnx
            || owner.admission_policy()
                != rz_eval::native_runtime_bridge::NativeAdmissionPolicy::Cpu
        {
            return Err(error(
                ErrorCode::UnsupportedContract,
                Stage::Admission,
                "CPU factory rejects CUDA owner",
            ));
        }
        Ok(Self {
            inner: NativeSessionFactory::from_owner(owner, runtime)?,
        })
    }
    pub(crate) fn attestation_profile(&self) -> Result<&CpuProfileV1, ContractError> {
        self.inner.attestation_profile()
    }
    pub fn finish(
        &self,
        served: Result<(), EngineError>,
    ) -> Result<NativeRunReport, NativeRunError> {
        self.inner.finish(served)
    }
}

#[cfg(feature = "onnx-cuda")]
pub struct NativeCudaFactory {
    inner: NativeSessionFactory,
}
#[cfg(feature = "onnx-cuda")]
impl NativeCudaFactory {
    /// Uses only the same worker/runtime already held by the original system
    /// custody. It does not start native execution or transfer the custody;
    /// callers must retain that owner through shutdown and session release.
    #[cfg(all(feature = "pals-adaptive", target_os = "linux"))]
    #[allow(dead_code)]
    fn from_system_custody(
        custody: &system_cuda::Lc0SystemStartupCustody,
    ) -> Result<Self, NativeBootstrapError> {
        let owner = custody
            .worker()
            .cloned()
            .ok_or(NativeBootstrapError::Config(
                "original system CUDA worker is not available",
            ))?;
        let runtime = custody
            .runtime()
            .cloned()
            .ok_or(NativeBootstrapError::Config(
                "original system CUDA runtime is not available",
            ))?;
        if owner.origin() != NativeWorkerOrigin::CudaOnnx
            || !runtime.is_system_cuda()
            || !owner.system_cuda_metadata().is_some_and(|metadata| {
                Some(metadata.system_runtime_identity.0) == runtime.cuda_runtime_identity()
            })
        {
            return Err(NativeBootstrapError::Config(
                "original system CUDA owner identity differs",
            ));
        }
        Ok(Self {
            inner: NativeSessionFactory::from_owner(owner, Some(runtime))?,
        })
    }

    #[cfg(feature = "experimental-batch")]
    pub fn batch_journal(
        &self,
        transfer: bool,
    ) -> Option<rz_eval::batch_journal::BatchJournalReceipt> {
        self.inner.owner.batch_journal_receipt(transfer)
    }
    #[cfg(feature = "experimental-raw-cache")]
    pub fn configure_raw_cache(
        &self,
        limits: rz_eval::raw_cache::RawCacheLimits,
    ) -> Result<(), ContractError> {
        self.inner.owner.projection().configure_raw_cache(limits)
    }
    pub fn load(
        owners: &OwnerRegistry,
        config: &NativeConfig,
    ) -> Result<Self, NativeBootstrapError> {
        Ok(Self {
            inner: NativeSessionFactory::load_cuda(owners, config)?,
        })
    }
    pub(crate) fn attestation_profile(&self) -> Result<&CudaProfileV1, ContractError> {
        if self.inner.owner.origin() != NativeWorkerOrigin::CudaOnnx
            || self.inner._runtime.is_none()
        {
            return Err(error(
                ErrorCode::UnsupportedContract,
                Stage::Output,
                "only an actually loaded CUDA owner can issue provider evidence",
            ));
        }
        self.inner.loaded_cuda_profile.as_ref().ok_or_else(|| {
            error(
                ErrorCode::IdentityMismatch,
                Stage::Output,
                "loaded CUDA provider snapshot is unavailable",
            )
        })
    }
    pub fn finish(
        &self,
        served: Result<(), EngineError>,
    ) -> Result<NativeRunReport, NativeRunError> {
        self.inner.finish(served)
    }
}

/// Status observation may itself discover owner poison. Collect after each
/// final idle observation before accepting the report, and confirm the same
/// physical idle condition again. A busy lock remains bounded by finish().
fn confirm_collected_drain(
    mut collect: impl FnMut() -> Result<bool, ContractError>,
    mut status: impl FnMut() -> Result<
        Option<rz_eval::native_runtime_bridge::NativeOwnerStatus>,
        ContractError,
    >,
) -> Result<bool, ContractError> {
    let idle = |value: Option<rz_eval::native_runtime_bridge::NativeOwnerStatus>| {
        value.is_some_and(|value| {
            !value.active && value.reserved_diagnostics == 0 && value.occupied_diagnostics == 0
        })
    };
    if !collect()? || !idle(status()?) || !collect()? || !idle(status()?) {
        return Ok(false);
    }
    // The second status can also be the first observer of poison; preserve its
    // original receipt even when a previous normal close set admission_error.
    collect()
}

fn private_output_root(path: &Path) -> Result<PathBuf, BackendError> {
    let boundary = |detail| BackendError::new(FailureKind::Io, FailureStage::Asset, detail);
    let metadata = fs::symlink_metadata(path).map_err(|cause| {
        boundary("native output root is unavailable")
            .with_external_cause(CauseCode::RuntimePath, &cause)
    })?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(boundary(
            "native output root must be an existing real private directory",
        ));
    }
    let canonical = path.canonicalize().map_err(|cause| {
        boundary("cannot resolve native output root")
            .with_external_cause(CauseCode::RuntimePath, &cause)
    })?;
    for ancestor in canonical.ancestors() {
        match fs::symlink_metadata(ancestor.join(".git")) {
            Ok(_) => return Err(boundary("native output root must be outside Git checkouts")),
            Err(cause) if cause.kind() == std::io::ErrorKind::NotFound => {}
            Err(cause) => {
                return Err(
                    boundary("cannot verify native output-root ownership boundary")
                        .with_external_cause(CauseCode::RuntimePath, &cause),
                );
            }
        }
    }
    Ok(canonical)
}

#[derive(Clone)]
struct RuntimeClock(ProcessClock);
impl Clock for RuntimeClock {
    type Tick = MonotonicTick;
    fn now(&self) -> MonotonicTick {
        self.0.now().unwrap_or(MonotonicTick(u64::MAX))
    }
    fn elapsed(&self, earlier: MonotonicTick, later: MonotonicTick) -> Duration {
        Duration::from_nanos(later.0.saturating_sub(earlier.0))
    }
}
impl ContractClock for RuntimeClock {
    fn domain(&self) -> ClockDomain {
        self.0.domain()
    }
}
impl EvaluatorFactory for NativeSessionFactory {
    #[cfg(feature = "experimental-batch")]
    fn parallelism(&self) -> usize {
        self.owner.max_batch()
    }

    #[cfg(feature = "experimental-notify")]
    fn completion_signal(&self) -> Option<rz_runtime::CompletionSignal> {
        Some(self.owner.completion_signal())
    }

    fn reset_game(&self) -> Result<(), ContractError> {
        #[cfg(feature = "experimental-raw-cache")]
        self.owner.projection().clear_raw_cache()?;
        Ok(())
    }
    fn source_trace(&self) -> Option<SourceJournal<CompletionContext>> {
        self.owner.source_trace().cloned()
    }
    fn observe_search_acceptance(
        &self,
        evaluation: &rz_search::contracts::AcceptedEvaluation,
        traversed_edges: usize,
    ) -> Result<(), ContractError> {
        let observed = self.evidence.record_search(evaluation, traversed_edges);
        #[cfg(feature = "experimental-batch")]
        let observed = observed.and_then(|()| self.owner.batch_consume(evaluation.context));
        if let Err(error) = observed {
            let _ = self.owner.try_close_admission(error);
        }
        observed
    }
    fn profile(&self) -> EvaluatorProfile {
        self.profile.clone()
    }
    fn input_key(&self, state: &RulesState, legal: &[Move]) -> Result<EvalInputKey, ContractError> {
        self.owner.projection().input_key(state, legal)
    }
    fn create(
        &self,
        clock: ProcessClock,
        authority: SearchAuthority,
    ) -> Result<Box<dyn ManagedEvaluator>, ContractError> {
        self.evidence.collect(&self.owner)?;
        if let Some(error) = self.evidence.admission_error()? {
            return Err(error);
        }
        let high_water = clock.reserve_executions(MAX_NATIVE_EVALUATIONS)?;
        let runtime_clock = RuntimeClock(clock.clone());
        let scope = SharedScope::new(authority.current_scope()?);
        let adapter = ContractsAdapter::with_execution_high_water(
            scope.clone(),
            runtime_clock.clone(),
            self.owner.max_batch(),
            high_water,
        )?;
        #[cfg(feature = "experimental-batch")]
        let adapter = if self.owner.max_batch() > 1 {
            adapter.with_mixed_legal_batching()
        } else {
            adapter
        };
        let backend = NativeRuntimeBackend::new(self.owner.clone(), runtime_clock)?;
        let evaluator = ContractEvaluator::new(
            adapter,
            backend,
            Limits {
                max_requests: self.owner.max_batch(),
                max_batch_items: self.owner.max_batch(),
                max_executions: 1,
                max_batch_wait: if self.owner.max_batch() > 1 {
                    Duration::from_micros(200)
                } else {
                    Duration::ZERO
                },
                max_queue_age: Duration::from_secs(30),
                deadline_reserve: Duration::ZERO,
                memory: Resources {
                    host_bytes: self.owner.max_batch() as u64
                        * (HOST_BYTES_PER_ITEM + NATIVE_RUNTIME_OVERHEAD_BYTES),
                    device_bytes: self
                        .owner
                        .admission_policy()
                        .execution_resources()
                        .device_bytes
                        .saturating_mul(self.owner.max_batch() as u64),
                    pinned_bytes: self
                        .owner
                        .admission_policy()
                        .execution_resources()
                        .pinned_bytes,
                },
            },
            256,
        )?;
        #[cfg(feature = "experimental-raw-cache")]
        let evaluator = {
            let mut evaluator = evaluator;
            evaluator.set_raw_reuse(Box::new(self.owner.projection().raw_cache_provider()));
            evaluator
        };
        let mut other_pending = std::collections::VecDeque::new();
        other_pending
            .try_reserve(self.owner.max_batch().saturating_sub(1))
            .map_err(|_| {
                error(
                    ErrorCode::ResourceExhausted,
                    Stage::Admission,
                    "native pending context allocation failed",
                )
            })?;
        Ok(Box::new(NativeRuntime {
            evaluator,
            scope,
            authority,
            clock,
            owner: self.owner.clone(),
            evidence: self.evidence.clone(),
            submissions: 0,
            pending: None,
            other_pending,
            observation_error: None,
        }))
    }
}

macro_rules! delegate_native_factory {
    ($factory:ty) => {
        impl EvaluatorFactory for $factory {
            #[cfg(feature = "experimental-batch")]
            fn parallelism(&self) -> usize {
                self.inner.parallelism()
            }

            #[cfg(feature = "experimental-notify")]
            fn completion_signal(&self) -> Option<rz_runtime::CompletionSignal> {
                self.inner.completion_signal()
            }

            fn reset_game(&self) -> Result<(), ContractError> {
                self.inner.reset_game()
            }
            fn source_trace(&self) -> Option<SourceJournal<CompletionContext>> {
                self.inner.source_trace()
            }
            fn observe_search_acceptance(
                &self,
                evaluation: &rz_search::contracts::AcceptedEvaluation,
                traversed_edges: usize,
            ) -> Result<(), ContractError> {
                self.inner
                    .observe_search_acceptance(evaluation, traversed_edges)
            }
            fn profile(&self) -> EvaluatorProfile {
                self.inner.profile()
            }
            fn input_key(
                &self,
                state: &RulesState,
                legal: &[Move],
            ) -> Result<EvalInputKey, ContractError> {
                self.inner.input_key(state, legal)
            }
            fn create(
                &self,
                clock: ProcessClock,
                authority: SearchAuthority,
            ) -> Result<Box<dyn ManagedEvaluator>, ContractError> {
                self.inner.create(clock, authority)
            }
        }
    };
}
delegate_native_factory!(NativeCpuFactory);
#[cfg(feature = "onnx-cuda")]
delegate_native_factory!(NativeCudaFactory);

struct NativeRuntime {
    evaluator: ContractEvaluator<RulesState, NativeRuntimeBackend<RuntimeClock>, RuntimeClock>,
    scope: SharedScope,
    authority: SearchAuthority,
    clock: ProcessClock,
    owner: NativeWorkerOwner,
    evidence: NativeEvidenceHandle,
    submissions: u64,
    pending: Option<EvalContext>,
    other_pending: std::collections::VecDeque<EvalContext>,
    observation_error: Option<ContractError>,
}
impl NativeRuntime {
    #[cfg(feature = "experimental-batch")]
    fn record_batch_result(
        &self,
        result: &EvalResult,
        discarded: bool,
    ) -> Result<(), ContractError> {
        let (request, status) = match result {
            EvalResult::Completed(o) => (o.context.request, if discarded { 6 } else { 1 }),
            EvalResult::Canceled(c) => (c.request.request, 2),
            EvalResult::Expired(c) => (c.request.request, 3),
            EvalResult::Stale(c) => (c.request.request, 4),
            EvalResult::Failed(f) => (f.context.request.request, 5),
        };
        self.owner.batch_logical(request, status, false)
    }
    fn remove_pending(&mut self, context: EvalContext) {
        if self.pending == Some(context) {
            self.pending = self.other_pending.pop_front();
        } else if let Some(index) = self.other_pending.iter().position(|c| *c == context) {
            self.other_pending.remove(index);
        }
    }
    fn collect_observations(&mut self, drain_discarded: u64) -> Result<(), ContractError> {
        let scheduler = self.evaluator.take_observations();
        let delivery = self.evaluator.take_delivery_observations();
        if let Some(trace) = self.owner.source_trace() {
            trace.note_external_loss(scheduler.dropped, scheduler.counter_overflow);
            trace.note_external_loss(delivery.dropped, delivery.counter_overflow);
            let source_clock = self.clock.b_clock();
            for event in &scheduler.events {
                use rz_runtime::ObservationKind as O;
                let (request, execution, stage, start, succeeded) = match event.kind {
                    O::Admitted { request } => {
                        (Some(request), None, SourceStage::Admitted, event.at, true)
                    }
                    O::RequestDispatched {
                        request,
                        execution,
                        dispatch_started_at,
                    } => (
                        Some(request),
                        Some(execution),
                        SourceStage::DispatchStarted,
                        dispatch_started_at,
                        true,
                    ),
                    O::Dispatched {
                        execution,
                        dispatch_started_at,
                        ..
                    } => (
                        None,
                        Some(execution),
                        SourceStage::DispatchFinished,
                        dispatch_started_at,
                        true,
                    ),
                    O::PhysicalCompleted { execution } => (
                        None,
                        Some(execution),
                        SourceStage::PhysicalReadyObserved,
                        event.at,
                        true,
                    ),
                    O::ValidationStarted { request, execution } => (
                        Some(request),
                        Some(execution),
                        SourceStage::ValidationStarted,
                        event.at,
                        true,
                    ),
                    O::ValidationFinished {
                        request,
                        execution,
                        valid,
                    } => (
                        Some(request),
                        Some(execution),
                        SourceStage::ValidationFinished,
                        event.at,
                        valid,
                    ),
                    O::Finished { request, kind, .. } => (
                        Some(request),
                        None,
                        SourceStage::LogicalFinished,
                        event.at,
                        kind == rz_telemetry::FinishKind::Completed,
                    ),
                    O::Rejected { .. } | O::ReservationChanged => continue,
                };
                let key = self
                    .pending
                    .filter(|context| request.is_none_or(|id| id == context.request))
                    .map(|request| CompletionContext { request, execution });
                match (
                    source_clock.instant_at(start),
                    source_clock.instant_at(event.at),
                ) {
                    (Ok(start), Ok(end)) => trace.record(
                        key,
                        stage,
                        start,
                        if stage == SourceStage::DispatchStarted {
                            start
                        } else {
                            end
                        },
                        succeeded,
                    ),
                    _ => trace.note_invalid_interval(),
                }
            }
            for event in &delivery.events {
                use rz_runtime::contracts::DeliveryObservationKind as O;
                if event.clock != self.clock.domain() {
                    trace.note_invalid_interval();
                    continue;
                }
                let (stage, succeeded) = match event.kind {
                    O::Accepted => (SourceStage::DeliveryAccepted, true),
                    O::Rejected { .. } => (SourceStage::DeliveryRejected, false),
                    O::Terminal { .. } | O::MailboxDisconnected { .. } => {
                        (SourceStage::DeliveryTerminal, false)
                    }
                };
                match source_clock.instant_at(event.at) {
                    Ok(at) => trace.record(Some(event.context), stage, at, at, succeeded),
                    Err(_) => trace.note_invalid_interval(),
                }
            }
        }
        self.evidence.record_observations(NativeObservationReport {
            scheduler_events: scheduler.events.len() as u64,
            scheduler_dropped: scheduler.dropped,
            scheduler_counter_overflow: scheduler.counter_overflow,
            delivery_events: delivery.events.len() as u64,
            delivery_dropped: delivery.dropped,
            delivery_counter_overflow: delivery.counter_overflow,
            drain_discarded_results: drain_discarded,
        })
    }
    fn refresh(&self) -> Result<(), ContractError> {
        self.scope.update(self.authority.current_scope()?);
        Ok(())
    }
    fn drain(&mut self, until: Instant) -> Result<(), ContractError> {
        // Closing D admission never revokes queued UCI natural-completion output.
        self.evaluator.begin_shutdown(self.clock.deadline(until)?)?;
        loop {
            self.refresh()?;
            let mut discarded = 0_u64;
            while let Some(result) = self.evaluator.poll() {
                #[cfg(feature = "experimental-batch")]
                self.record_batch_result(&result, true)?;
                #[cfg(not(feature = "experimental-batch"))]
                let _ = result;
                discarded += 1;
            }
            self.collect_observations(discarded)?;
            self.evidence.collect(&self.owner)?;
            let snapshot = self.evaluator.shutdown_snapshot();
            match snapshot.drain {
                DrainState::Drained if snapshot.state.reserved_requests == 0 => return Ok(()),
                DrainState::TimedOut { .. } => {
                    return Err(error(
                        ErrorCode::BackendFailure,
                        Stage::Backend,
                        "native physical drain timed out; lease remains pinned",
                    ));
                }
                _ if Instant::now() >= until => {
                    return Err(error(
                        ErrorCode::Expired,
                        Stage::Backend,
                        "native shutdown deadline reached; completion unconfirmed",
                    ));
                }
                _ => thread::sleep(Duration::from_millis(1)),
            }
        }
    }
}
impl Evaluator<RulesState> for NativeRuntime {
    fn submit(&mut self, request: Arc<EvalRequest<RulesState>>) -> Result<(), ContractError> {
        self.refresh()?;
        self.evidence.collect(&self.owner)?;
        if let Some(error) = self.evidence.admission_error()? {
            return Err(error);
        }
        if self.submissions >= MAX_NATIVE_EVALUATIONS {
            return Err(error(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "per-root native execution range limit reached",
            ));
        }
        self.submissions += 1;
        if usize::from(self.pending.is_some()) + self.other_pending.len() >= self.owner.max_batch()
        {
            return Err(error(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "native pending limit reached",
            ));
        }
        let context = request.context();
        #[cfg(feature = "experimental-batch")]
        self.owner.batch_register(context)?;
        let result = self.evaluator.submit(request);
        #[cfg(feature = "experimental-batch")]
        if result.is_err() {
            self.owner.batch_logical(context.request, 5, false)?;
        }
        if result.is_ok() {
            if let Some(previous) = self.pending.replace(context) {
                self.other_pending.push_back(previous);
            }
        }
        if let Err(error) = self.collect_observations(0) {
            self.observation_error.get_or_insert(error);
            let _ = self.owner.try_close_admission(error);
        }
        if result.is_err() {
            self.pending = None;
        }
        // A passive observation failure must not turn an already admitted
        // request into submit Err. Its original admission remains owned by D;
        // poll/shutdown publish the observation failure and drain the lease.
        result
    }
    fn poll(&mut self) -> Option<EvalResult> {
        if let Some(error) = self.observation_error
            && let Some(request) = self
                .pending
                .take()
                .or_else(|| self.other_pending.pop_front())
        {
            return Some(EvalResult::Failed(EvalFailure {
                context: CompletionContext {
                    request,
                    execution: None,
                },
                error,
                recovery: RecoveryOutcome::Failed,
            }));
        }
        if let Err(error) = self.refresh() {
            self.authority.cancel();
            return self
                .pending
                .take()
                .or_else(|| self.other_pending.pop_front())
                .map(|request| {
                    EvalResult::Failed(EvalFailure {
                        context: CompletionContext {
                            request,
                            execution: None,
                        },
                        error,
                        recovery: RecoveryOutcome::NotAttempted,
                    })
                });
        }
        let result = self.evaluator.poll();
        let collection = self.evidence.collect(&self.owner);
        if let Err(error) = self.collect_observations(0) {
            self.observation_error.get_or_insert(error);
            let _ = self.owner.try_close_admission(error);
        }
        if let Some(EvalResult::Completed(output)) = &result
            && let Err(error) = self.evidence.record_completed(output)
        {
            self.remove_pending(output.context);
            return Some(EvalResult::Failed(EvalFailure {
                context: CompletionContext {
                    request: output.context,
                    execution: output.actual.execution,
                },
                error,
                recovery: RecoveryOutcome::Failed,
            }));
        }
        if let Some(result) = result {
            #[cfg(feature = "experimental-batch")]
            if let Err(error) = self.record_batch_result(&result, false) {
                self.observation_error.get_or_insert(error);
                let _ = self.owner.try_close_admission(error);
            }
            let context = match &result {
                EvalResult::Completed(output) => output.context,
                EvalResult::Canceled(c) | EvalResult::Expired(c) | EvalResult::Stale(c) => {
                    c.request
                }
                EvalResult::Failed(f) => f.context.request,
            };
            self.remove_pending(context);
            return Some(result);
        }
        if let Err(error) = collection {
            return self
                .pending
                .take()
                .or_else(|| self.other_pending.pop_front())
                .map(|request| {
                    EvalResult::Failed(EvalFailure {
                        context: CompletionContext {
                            request,
                            execution: None,
                        },
                        error,
                        recovery: RecoveryOutcome::NotAttempted,
                    })
                });
        }
        None
    }
    fn cancel(&mut self, request: RequestId) -> Result<(), ContractError> {
        #[cfg(feature = "experimental-batch")]
        self.owner.batch_logical(request, 0, true)?;
        self.evaluator.cancel(request)
    }
}
impl ManagedEvaluator for NativeRuntime {
    #[cfg(feature = "experimental-notify")]
    fn wake_after(&self) -> Option<Duration> {
        let state = self.evaluator.state();
        // Queued batch wait must expire even when no worker can notify yet.
        // During physical work, its completion signal owns the wakeup.
        (state.queued > 0 && state.executions == 0).then_some(Duration::from_micros(200))
    }
    fn shutdown(&mut self, until: Instant) -> Result<(), ContractError> {
        let drained = self.drain(until);
        let collected = self.evidence.collect(&self.owner);
        // Common cleanup failure is retained by B; C's original cause remains in
        // the native evidence owner even when both operations fail.
        drained
            .and(collected.map(|_| ()))
            .and(self.observation_error.take().map_or(Ok(()), Err))
    }
}

#[cfg(test)]
#[path = "native_bootstrap_tests.rs"]
mod tests;

#[cfg(test)]
mod physical_owner_tests {
    use super::*;
    use crate::{
        EngineIdentity, Event, PositionPort, PositionSpec,
        engine::{self, RulesUciPort},
    };
    use rz_eval::{RawOutput, contracts::PreparedBatch, worker::SingleWorker};
    use std::{
        io::Write,
        sync::{
            atomic::{AtomicBool, AtomicUsize, Ordering},
            mpsc,
        },
    };

    const WAIT: Duration = Duration::from_secs(5);

    #[test]
    fn final_idle_observation_collects_new_poison_even_after_prior_normal_close() {
        use std::cell::Cell;
        let observed = Cell::new(0_u8);
        let pending_poison = Cell::new(false);
        let preserved_poison = Cell::new(false);
        let normal_close = error(
            ErrorCode::Canceled,
            Stage::Admission,
            "normal logical close",
        );
        let confirmed = confirm_collected_drain(
            || {
                if pending_poison.replace(false) {
                    preserved_poison.set(true);
                }
                Ok(true)
            },
            || {
                observed.set(observed.get() + 1);
                if observed.get() == 2 {
                    pending_poison.set(true);
                }
                Ok(Some(rz_eval::native_runtime_bridge::NativeOwnerStatus {
                    active: false,
                    admission_error: Some(normal_close),
                    occupied_diagnostics: 0,
                    reserved_diagnostics: 0,
                }))
            },
        )
        .unwrap();
        assert!(confirmed);
        assert_eq!(observed.get(), 2);
        assert!(
            preserved_poison.get(),
            "a final status-created poison must reach the collector despite unchanged admission_error"
        );
        assert!(!pending_poison.get());
        assert!(!confirm_collected_drain(|| Ok(true), || Ok(None)).unwrap());
        let original = error(
            ErrorCode::BackendFailure,
            Stage::Output,
            "collection failed",
        );
        assert_eq!(
            confirm_collected_drain(
                || Err(original),
                || panic!("failed collection cannot authorize a status check")
            ),
            Err(original)
        );
    }

    #[test]
    fn injected_physical_factory_cannot_issue_native_provider_startup_attestation() {
        let owners = OwnerRegistry::default();
        let worker = SingleWorker::spawn(|batch: &PreparedBatch<RulesState>| {
            batch
                .requests()
                .iter()
                .map(|request| request.physical_output(&raw(), batch.execution()))
                .collect()
        })
        .unwrap();
        let owner = NativeWorkerOwner::from_worker(worker, projection(&owners), 1).unwrap();
        let factory = NativeCpuFactory::from_owner(owner, None).unwrap();
        let clock = ProcessClock::new(ProcessEpoch(304));
        let denied =
            crate::native_attestation::StartupReceiptV1::capture(&factory, &clock).unwrap_err();
        assert!(denied.to_string().contains("actually loaded CPU ONNX"));
        let report = factory.finish(Ok(())).unwrap();
        assert_eq!(report.origin, NativeWorkerOrigin::Injected);
        assert_eq!(report.completed_by_runtime, 0);
    }

    struct SessionDropProbe {
        dropped: Arc<AtomicBool>,
        gate: Option<(mpsc::SyncSender<()>, mpsc::Receiver<()>)>,
        panic_on_drop: bool,
    }
    impl Drop for SessionDropProbe {
        fn drop(&mut self) {
            if let Some((entered, release)) = &self.gate {
                let _ = entered.try_send(());
                assert!(
                    release.recv_timeout(WAIT).is_ok(),
                    "bounded injected session destructor release"
                );
            }
            assert!(
                !self.panic_on_drop,
                "private injected session destructor panic"
            );
            self.dropped.store(true, Ordering::Release);
        }
    }
    struct DestructorRelease(Option<mpsc::SyncSender<()>>);
    impl DestructorRelease {
        fn release(&mut self) {
            if let Some(release) = self.0.take() {
                let _ = release.try_send(());
            }
        }
    }
    impl Drop for DestructorRelease {
        fn drop(&mut self) {
            self.release();
        }
    }
    fn factory_with_drop_probe(
        owners: &OwnerRegistry,
        probe: SessionDropProbe,
    ) -> NativeCpuFactory {
        let worker = SingleWorker::spawn(move |batch: &PreparedBatch<RulesState>| {
            // Keep the probe in the same long-lived closure as an ONNX session.
            let _session = &probe;
            batch
                .requests()
                .iter()
                .map(|request| request.physical_output(&raw(), batch.execution()))
                .collect()
        })
        .unwrap();
        let owner = NativeWorkerOwner::from_worker(worker, projection(owners), 1).unwrap();
        NativeCpuFactory::from_owner(owner, None).unwrap()
    }

    #[test]
    fn final_report_waits_for_idle_session_destruction_and_actual_worker_join() {
        let owners = OwnerRegistry::default();
        let dropped = Arc::new(AtomicBool::new(false));
        let factory = factory_with_drop_probe(
            &owners,
            SessionDropProbe {
                dropped: Arc::clone(&dropped),
                gate: None,
                panic_on_drop: false,
            },
        );
        assert!(!dropped.load(Ordering::Acquire));
        let report = factory.finish(Ok(())).unwrap();
        assert!(
            dropped.load(Ordering::Acquire),
            "owner idle alone cannot authorize the final report before session drop"
        );
        assert_eq!(report.origin, NativeWorkerOrigin::Injected);
        assert_eq!(report.completed_by_runtime, 0);
        assert!(!report.has_failure());
    }

    #[test]
    fn blocked_session_destructor_keeps_service_source_and_evidence_until_join() {
        let owners = OwnerRegistry::default();
        let dropped = Arc::new(AtomicBool::new(false));
        let (entered, observing) = mpsc::sync_channel(1);
        let (release, waiting) = mpsc::sync_channel(1);
        let mut release = DestructorRelease(Some(release));
        let factory = factory_with_drop_probe(
            &owners,
            SessionDropProbe {
                dropped: Arc::clone(&dropped),
                gate: Some((entered, waiting)),
                panic_on_drop: false,
            },
        );
        // Establish the destructor gate before starting the short deadline;
        // thread scheduling time is not the behavior under test.
        assert!(matches!(factory.inner.owner.try_shutdown(), Poll::Pending));
        observing.recv_timeout(WAIT).unwrap();
        let service = error(
            ErrorCode::BackendFailure,
            Stage::Output,
            "independent injected protocol failure",
        );
        // The production wrapper always uses FINAL_COLLECTION_LIMIT. This
        // shorter injected deadline exercises the same bounded finish path.
        let failed = factory
            .inner
            .finish_until(
                Err(EngineError::Contract(service)),
                Instant::now() + Duration::from_millis(100),
            )
            .unwrap_err();
        assert!(!dropped.load(Ordering::Acquire));
        assert_eq!(failed.collection_error.unwrap().code, ErrorCode::Expired);
        assert!(matches!(
            failed.service.as_deref(),
            Some(EngineError::Contract(original)) if *original == service
        ));
        assert!(failed.report.is_none());
        assert!(
            failed
                .retained_evidence
                .try_state()
                .unwrap()
                .unwrap()
                .report
                .is_some()
        );
        assert!(failed.worker_shutdown_failure().is_none());
        release.release();
        let report = factory.finish(Ok(())).unwrap();
        assert!(dropped.load(Ordering::Acquire));
        assert_eq!(report.origin, NativeWorkerOrigin::Injected);
        assert!(!report.has_failure());
    }

    #[test]
    fn session_destructor_panic_keeps_typed_original_and_cannot_transfer_healthy_report() {
        let owners = OwnerRegistry::default();
        let dropped = Arc::new(AtomicBool::new(false));
        let factory = factory_with_drop_probe(
            &owners,
            SessionDropProbe {
                dropped: Arc::clone(&dropped),
                gate: None,
                panic_on_drop: true,
            },
        );
        let failed = factory.finish(Ok(())).unwrap_err();
        assert!(!dropped.load(Ordering::Acquire));
        let original = failed.worker_shutdown_failure().unwrap();
        assert_eq!(failed.collection_error, Some(original.contract));
        assert_eq!(original.contract.code, ErrorCode::BackendFailure);
        assert_eq!(
            original.backend.as_ref().unwrap().cause.unwrap().code,
            CauseCode::RuntimePanic
        );
        assert!(
            std::error::Error::source(&failed)
                .unwrap()
                .downcast_ref::<PhysicalFailure>()
                .is_some()
        );
        assert!(failed.report.is_none());
        assert!(
            failed
                .retained_evidence
                .try_state()
                .unwrap()
                .unwrap()
                .report
                .is_some()
        );
        assert!(!failed.to_string().contains("private injected"));
        assert!(!format!("{failed:?}").contains("private injected"));
    }

    #[test]
    fn quarantined_execution_keeps_original_pinned_evidence_and_cannot_finish_successfully() {
        let owners = Arc::new(OwnerRegistry::default());
        let source = BackendError::new(
            FailureKind::BackendFailure,
            FailureStage::Backend,
            "injected execution completion is unknown",
        )
        .with_external_cause(CauseCode::OrtRun, &"private-quarantine-cause");
        let expected = source.clone();
        let worker = SingleWorker::spawn_with_outcome(move |_: &PreparedBatch<RulesState>| {
            rz_eval::worker::PhysicalRun::Quarantined(source.clone())
        })
        .unwrap();
        let owner = NativeWorkerOwner::from_worker(worker, projection(&owners), 1).unwrap();
        let factory = NativeCpuFactory::from_owner(owner, None).unwrap();
        let clock = ProcessClock::new(ProcessEpoch(306));
        let request = request(&factory, &owners, &clock, 1);
        let context = request.context();
        let mut runtime = factory.create(clock, authority(context)).unwrap();
        runtime.submit(request).unwrap();
        let until = Instant::now() + WAIT;
        loop {
            assert!(
                !matches!(runtime.poll(), Some(EvalResult::Completed(_))),
                "quarantine cannot return a completed evaluation"
            );
            let observed = factory
                .inner
                .evidence
                .try_state()
                .unwrap()
                .is_some_and(|state| {
                    state.report.as_ref().is_some_and(|report| {
                        report
                            .failures
                            .iter()
                            .any(|receipt| receipt.kind == NativeDiagnosticKind::Quarantined)
                    })
                });
            if observed {
                break;
            }
            assert!(
                Instant::now() < until,
                "bounded original quarantine receipt"
            );
            thread::yield_now();
        }
        let failed = factory
            .inner
            .finish_until(Ok(()), Instant::now() + Duration::from_millis(100))
            .unwrap_err();
        assert_eq!(failed.collection_error.unwrap().code, ErrorCode::Expired);
        assert!(failed.report.is_none());
        assert!(failed.worker_shutdown_failure().is_none());
        let state = failed.retained_evidence.try_state().unwrap().unwrap();
        let report = state.report.as_ref().unwrap();
        assert_eq!(report.origin, NativeWorkerOrigin::Injected);
        assert_eq!(report.completed_by_runtime, 0);
        assert_eq!(report.failures.len(), 1);
        let receipt = &report.failures[0];
        assert_eq!(receipt.kind, NativeDiagnosticKind::Quarantined);
        assert_eq!(receipt.context.request, context);
        assert!(receipt.context.execution.is_some());
        assert_eq!(receipt.failure.backend.as_ref(), Some(&expected));
        assert!(report.has_failure());
    }

    #[test]
    fn per_root_runtime_shutdown_preserves_the_session_until_process_finish() {
        let owners = Arc::new(OwnerRegistry::default());
        let dropped = Arc::new(AtomicBool::new(false));
        let factory = factory_with_drop_probe(
            &owners,
            SessionDropProbe {
                dropped: Arc::clone(&dropped),
                gate: None,
                panic_on_drop: false,
            },
        );
        let clock = ProcessClock::new(ProcessEpoch(305));
        for sequence in 1..=2 {
            let request = request(&factory, &owners, &clock, sequence);
            let context = request.context();
            let mut runtime = factory.create(clock.clone(), authority(context)).unwrap();
            runtime.submit(request).unwrap();
            let until = Instant::now() + WAIT;
            let result = loop {
                if let Some(result) = runtime.poll() {
                    break result;
                }
                assert!(Instant::now() < until, "bounded injected root completion");
                thread::yield_now();
            };
            let EvalResult::Completed(output) = result else {
                panic!("the reusable injected session must complete both roots");
            };
            assert_eq!(output.context, context);
            runtime.shutdown(Instant::now() + WAIT).unwrap();
            assert!(!dropped.load(Ordering::Acquire));
        }
        let report = factory.finish(Ok(())).unwrap();
        assert!(dropped.load(Ordering::Acquire));
        assert_eq!(report.origin, NativeWorkerOrigin::Injected);
        assert_eq!(report.completed_by_runtime, 0);
    }

    fn projection(owners: &OwnerRegistry) -> ClassicalProjection {
        ClassicalProjection::new(
            MaiaBinding::new(
                ModelHandle {
                    owner: owners.allocate().unwrap(),
                    slot: 0,
                    generation: SlotGeneration(1),
                    manifest: Digest([51; 32]),
                },
                EncodingHandle {
                    owner: owners.allocate().unwrap(),
                    slot: 0,
                    generation: SlotGeneration(1),
                    manifest: encoding_manifest(HistoryFill::No),
                },
                HistoryFill::No,
                Digest([52; 32]),
                1,
            )
            .unwrap(),
        )
    }
    fn raw() -> RawOutput {
        RawOutput {
            policy_logits: vec![0.0; rz_encoding::POLICY_SIZE],
            wdl: vec![0.4, 0.3, 0.3],
        }
    }
    fn request(
        factory: &NativeCpuFactory,
        owners: &Arc<OwnerRegistry>,
        clock: &ProcessClock,
        sequence: u64,
    ) -> Arc<EvalRequest<RulesState>> {
        let position =
            RulesUciPort::new(Arc::clone(owners), rz_position::PositionLimits::default())
                .prepare(&PositionSpec::default())
                .unwrap()
                .snapshot;
        let profile = factory.profile();
        let context = EvalContext {
            revision: CONTRACT_REVISION,
            request: RequestId::new(clock.epoch(), sequence),
            selection: SelectionId::new(clock.epoch(), sequence),
            game: GameGeneration(1),
            root: RootGeneration(1),
            state: position.state().snapshot().identity(),
            legal_order: position.state().legal_moves().order(),
            input: factory
                .input_key(
                    position.state().rules(),
                    position.state().legal_moves().moves(),
                )
                .unwrap(),
            model: profile.model.handle(),
            encoding: profile.model.encoding().handle,
            precision: profile.precision,
            compute: profile.compute,
            backend: profile.backend,
        };
        Arc::new(
            EvalRequest::try_new(
                context,
                position.state().snapshot().clone(),
                position.state().legal_moves().clone(),
                profile.model,
                clock.deadline(Instant::now() + WAIT).unwrap(),
                CancelToken::new(),
                profile.bytes,
            )
            .unwrap(),
        )
    }
    fn authority(context: EvalContext) -> SearchAuthority {
        SearchAuthority::isolated(AcceptanceScope {
            game: context.game,
            root: context.root,
            model: context.model,
            encoding: context.encoding,
            backend: context.backend,
        })
    }

    #[cfg(all(feature = "experimental-batch", feature = "experimental-notify"))]
    #[test]
    fn queued_native_batch_timer_reaches_worker_without_an_initial_completion_signal() {
        let owners = Arc::new(OwnerRegistry::default());
        let original = projection(&owners);
        let projection = ClassicalProjection::new(
            MaiaBinding::new(
                original.model().handle(),
                original.model().encoding().handle,
                HistoryFill::No,
                original.backend(),
                4,
            )
            .unwrap(),
        );
        let calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = Arc::clone(&calls);
        let worker = SingleWorker::spawn(move |batch: &PreparedBatch<RulesState>| {
            observed_calls.fetch_add(1, Ordering::AcqRel);
            batch
                .requests()
                .iter()
                .map(|request| request.physical_output(&raw(), batch.execution()))
                .collect()
        })
        .unwrap();
        let owner = NativeWorkerOwner::from_worker_batched(
            worker,
            projection,
            4,
            rz_eval::native_runtime_bridge::NativeAdmissionPolicy::Cpu,
            4,
        )
        .unwrap();
        let factory = NativeCpuFactory::from_owner(owner, None).unwrap();
        let signal = factory.completion_signal().unwrap();
        let clock = ProcessClock::new(ProcessEpoch(305));
        let request = request(&factory, &owners, &clock, 1);
        let context = request.context();
        let mut runtime = factory.create(clock, authority(context)).unwrap();
        runtime.submit(request).unwrap();
        assert_eq!(calls.load(Ordering::Acquire), 0);
        assert_eq!(runtime.wake_after(), Some(Duration::from_micros(200)));
        let until = Instant::now() + WAIT;
        let result = loop {
            // Use the same predicate/sequence/timer order as the UCI owner.
            let observed = signal.version();
            if let Some(result) = runtime.poll() {
                break result;
            }
            assert!(Instant::now() < until, "bounded native batch wakeup");
            signal.wait_changed(
                observed,
                runtime
                    .wake_after()
                    .unwrap_or(WAIT)
                    .min(until.saturating_duration_since(Instant::now())),
            );
        };
        let EvalResult::Completed(output) = result else {
            panic!("queued batch must complete after its timer and worker signal");
        };
        assert_eq!(output.context, context);
        assert_eq!(calls.load(Ordering::Acquire), 1);
        assert_eq!(runtime.wake_after(), None);
        runtime.shutdown(Instant::now() + WAIT).unwrap();
        factory.finish(Ok(())).unwrap();
    }

    #[cfg(feature = "experimental-batch")]
    #[test]
    fn batch_journal_links_four_requests_one_physical_lease_and_committed_consumption() {
        let owners = Arc::new(OwnerRegistry::default());
        let original = projection(&owners);
        let projection = ClassicalProjection::new(
            MaiaBinding::new(
                original.model().handle(),
                original.model().encoding().handle,
                HistoryFill::No,
                original.backend(),
                4,
            )
            .unwrap(),
        );
        let worker = SingleWorker::spawn(move |batch: &PreparedBatch<RulesState>| {
            batch
                .requests()
                .iter()
                .map(|r| r.physical_output(&raw(), batch.execution()))
                .collect()
        })
        .unwrap();
        let owner = NativeWorkerOwner::from_worker_batched(
            worker,
            projection,
            4,
            rz_eval::native_runtime_bridge::NativeAdmissionPolicy::Cpu,
            4,
        )
        .unwrap();
        owner.enable_batch_journal().unwrap();
        let journal_owner = owner.clone();
        let factory = NativeCpuFactory::from_owner(owner, None).unwrap();
        let clock = ProcessClock::new(ProcessEpoch(306));
        let requests: Vec<_> = (1..=4)
            .map(|sequence| request(&factory, &owners, &clock, sequence))
            .collect();
        let mut runtime = factory
            .create(clock, authority(requests[0].context()))
            .unwrap();
        for request in requests {
            runtime.submit(request).unwrap();
        }
        let until = Instant::now() + WAIT;
        let mut delivered = 0;
        while delivered < 4 {
            assert!(Instant::now() < until);
            if let Some(result) = runtime.poll() {
                let EvalResult::Completed(output) = result else {
                    panic!("mock batch completion must remain typed");
                };
                let accepted = rz_search::contracts::AcceptedEvaluation {
                    context: CompletionContext {
                        request: output.context,
                        execution: output.actual.execution,
                    },
                    actual: output.actual,
                };
                let mut wrong_root = accepted.context;
                wrong_root.request.root = RootGeneration(2);
                assert!(journal_owner.batch_consume(wrong_root).is_err());
                factory.observe_search_acceptance(&accepted, 1).unwrap();
                assert!(journal_owner.batch_consume(accepted.context).is_err()); // duplicate backup evidence is rejected
                delivered += 1;
            } else {
                thread::yield_now();
            }
        }
        runtime.shutdown(Instant::now() + WAIT).unwrap();
        factory.finish(Ok(())).unwrap();
        let receipt = journal_owner.batch_journal_receipt(true).unwrap();
        let audit = rz_eval::batch_journal::audit_batch_journal(&receipt, true).unwrap();
        assert_eq!(
            (
                audit.physical_dispatches,
                audit.physical_completed,
                audit.completed_nn_items,
                audit.consumed_evaluations
            ),
            (1, 1, 4, 4)
        );
        assert_eq!(audit.completed_batch_distribution[3], 1);
    }

    #[test]
    fn native_physical_failure_keeps_original_cause_and_service_cleanup_independently() {
        let owners = Arc::new(OwnerRegistry::default());
        let source = BackendError::new(
            FailureKind::BackendFailure,
            FailureStage::Backend,
            "injected physical failure",
        )
        .with_external_cause(CauseCode::OrtRun, &"private-native-cause")
        .with_diagnostic("InjectedNative", "private-native-cause");
        let expected = source.clone();
        let worker = SingleWorker::spawn(move |_: &PreparedBatch<RulesState>| {
            Err(PhysicalFailure::from(source.clone()))
        })
        .unwrap();
        let owner = NativeWorkerOwner::from_worker(worker, projection(&owners), 1).unwrap();
        let factory = NativeCpuFactory::from_owner(owner, None).unwrap();
        let clock = ProcessClock::new(ProcessEpoch(301));
        let request = request(&factory, &owners, &clock, 1);
        let context = request.context();
        let mut runtime = factory.create(clock.clone(), authority(context)).unwrap();
        runtime.submit(request).unwrap();
        let until = Instant::now() + WAIT;
        let result = loop {
            if let Some(result) = runtime.poll() {
                break result;
            }
            assert!(
                Instant::now() < until,
                "bounded physical failure publication"
            );
            thread::yield_now();
        };
        let EvalResult::Failed(failure) = result else {
            panic!("native physical failure cannot become a completed evaluation");
        };
        assert_eq!(failure.context.request, context);
        assert!(failure.context.execution.is_some());
        runtime.shutdown(Instant::now() + WAIT).unwrap();
        assert_eq!(
            factory
                .create(clock, authority(context))
                .err()
                .unwrap()
                .code,
            ErrorCode::BackendFailure
        );
        let cleanup = error(
            ErrorCode::Expired,
            Stage::Backend,
            "independent injected outer cleanup failure",
        );
        let failed = factory
            .finish(Err(EngineError::Contract(cleanup)))
            .unwrap_err();
        assert!(
            matches!(failed.service.as_deref(), Some(EngineError::Contract(actual)) if *actual == cleanup)
        );
        let report = failed.report.as_ref().unwrap();
        assert_eq!(report.origin, NativeWorkerOrigin::Injected);
        assert_eq!(report.completed_by_runtime, 0);
        assert_eq!(report.failures.len(), 1);
        let receipt = &report.failures[0];
        assert_eq!(receipt.kind, NativeDiagnosticKind::PhysicalFailure);
        assert_eq!(receipt.context, failure.context);
        assert_eq!(receipt.failure.backend.as_ref(), Some(&expected));
        assert!(!failed.to_string().contains("private-native-cause"));
        assert!(!format!("{failed:?}").contains("private-native-cause"));
    }

    #[test]
    fn periodic_consumer_accepts_normal_cancel_audits_without_lifetime_exhaustion() {
        use rz_runtime::{Backend, contracts::RuntimeRequest};
        let owners = Arc::new(OwnerRegistry::default());
        let worker = SingleWorker::spawn(
            |_: &PreparedBatch<RulesState>| -> Result<Vec<EvalOutput>, PhysicalFailure> {
                panic!("a prelaunch canceled request cannot run native code")
            },
        )
        .unwrap();
        let owner = NativeWorkerOwner::from_worker(worker, projection(&owners), 1).unwrap();
        let factory = NativeCpuFactory::from_owner(owner, None).unwrap();
        let clock = ProcessClock::new(ProcessEpoch(302));
        let mut backend =
            NativeRuntimeBackend::new(factory.inner.owner.clone(), RuntimeClock(clock.clone()))
                .unwrap();
        for sequence in 1..=64 {
            let eval = request(&factory, &owners, &clock, sequence);
            eval.cancel_token().cancel();
            let refused = backend.dispatch(
                &ExecutionId::new(clock.epoch(), sequence),
                &[Arc::new(RuntimeRequest::new(eval))],
            );
            assert_eq!(refused.err().unwrap().code, ErrorCode::Canceled);
            assert!(
                factory
                    .inner
                    .evidence
                    .collect(&factory.inner.owner)
                    .unwrap()
            );
        }
        let report = factory.finish(Ok(())).unwrap();
        assert_eq!(report.canceled.count, 64);
        assert_eq!(report.canceled.first.as_ref().unwrap().ticket.1.sequence, 1);
        assert_eq!(report.canceled.last.as_ref().unwrap().ticket.1.sequence, 64);
        assert!(report.failures.is_empty());
        assert!(!report.has_failure());
    }

    #[derive(Clone, Default)]
    struct Capture(Arc<Mutex<Vec<u8>>>);
    impl Write for Capture {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            let mut output = self.0.lock().unwrap();
            if output.len() + bytes.len() > 16 * 1024 {
                return Err(std::io::Error::other("bounded test output exhausted"));
            }
            output.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl Capture {
        fn text(&self) -> String {
            String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
        }
    }
    struct CountingFactory {
        native: Arc<NativeCpuFactory>,
        created: AtomicUsize,
    }
    impl EvaluatorFactory for CountingFactory {
        fn profile(&self) -> EvaluatorProfile {
            self.native.profile()
        }
        fn input_key(
            &self,
            state: &RulesState,
            legal: &[Move],
        ) -> Result<EvalInputKey, ContractError> {
            self.native.input_key(state, legal)
        }
        fn create(
            &self,
            clock: ProcessClock,
            authority: SearchAuthority,
        ) -> Result<Box<dyn ManagedEvaluator>, ContractError> {
            self.created.fetch_add(1, Ordering::AcqRel);
            self.native.create(clock, authority)
        }
    }
    struct ServiceGuard {
        events: mpsc::SyncSender<Event>,
        release: Option<mpsc::SyncSender<()>>,
        service: Option<thread::JoinHandle<Result<(), EngineError>>>,
    }
    impl Drop for ServiceGuard {
        fn drop(&mut self) {
            if let Some(release) = self.release.take() {
                let _ = release.try_send(());
            }
            let _ = self.events.try_send(Event::Line("quit".into()));
        }
    }

    fn assert_blocked_physical_owner_active(owner: &NativeWorkerOwner, stage: &'static str) {
        let until = Instant::now() + WAIT;
        loop {
            match owner.try_status() {
                Ok(Some(status)) => {
                    assert!(
                        status.active,
                        "{stage}: blocked physical owner is inactive: {status:?}"
                    );
                    return;
                }
                Err(error) => panic!("{stage}: native owner status query failed: {error:?}"),
                Ok(None) => {
                    // None means owner-lock contention, not an inactive worker.
                    // Retry the actual observation without claiming synchronization.
                    assert!(
                        Instant::now() < until,
                        "{stage}: native owner status remained busy until the bounded observation deadline"
                    );
                    thread::yield_now();
                }
            }
        }
    }

    #[test]
    fn root_replacement_while_the_single_physical_worker_is_blocked_returns_current_legal_fallback_once()
     {
        blocked_native_output_boundary(BlockedBoundary::RootReplacement);
    }

    #[test]
    fn stop_output_waits_for_native_lease_release_and_input_owner_remains_responsive() {
        blocked_native_output_boundary(BlockedBoundary::Stop);
    }

    #[test]
    fn deadline_output_waits_for_native_lease_release_without_admitting_late_backup() {
        blocked_native_output_boundary(BlockedBoundary::Deadline);
    }

    #[derive(Clone, Copy)]
    enum BlockedBoundary {
        RootReplacement,
        Stop,
        Deadline,
    }

    fn blocked_native_output_boundary(boundary: BlockedBoundary) {
        let owners = Arc::new(OwnerRegistry::default());
        let (entered, entering) = mpsc::sync_channel(1);
        let (release, released) = mpsc::sync_channel(1);
        let worker = SingleWorker::spawn(move |batch: &PreparedBatch<RulesState>| {
            entered.send(()).unwrap();
            if released.recv_timeout(WAIT).is_err() {
                return Err(BackendError::new(
                    FailureKind::BackendFailure,
                    FailureStage::Backend,
                    "blocked injected worker release timed out",
                )
                .into());
            }
            batch
                .requests()
                .iter()
                .map(|item| item.physical_output(&raw(), batch.execution()))
                .collect()
        })
        .unwrap();
        let owner = NativeWorkerOwner::from_worker(worker, projection(&owners), 1).unwrap();
        let native = Arc::new(NativeCpuFactory::from_owner(owner, None).unwrap());
        let factory = Arc::new(CountingFactory {
            native: Arc::clone(&native),
            created: AtomicUsize::new(0),
        });
        let process = engine::EngineProcess::new(
            factory.clone(),
            owners,
            ProcessClock::new(ProcessEpoch(303)),
        )
        .with_identity(EngineIdentity {
            name: "RoveZero injected single physical worker fixture".into(),
            author: "RoveZero tests".into(),
        });
        let settings = engine::EngineSettings {
            max_workers: 1,
            ..engine::EngineSettings::default()
        };
        let (events, receiver) = engine::event_channel();
        let sender = events.clone();
        let mut protocol = Capture::default();
        let protocol_view = protocol.clone();
        let mut diagnostics = Capture::default();
        let diagnostics_view = diagnostics.clone();
        let service = thread::spawn(move || {
            engine::serve(
                receiver,
                sender,
                &mut protocol,
                &mut diagnostics,
                process,
                settings,
            )
        });
        let mut guard = ServiceGuard {
            events,
            release: Some(release),
            service: Some(service),
        };
        let go = if matches!(boundary, BlockedBoundary::Deadline) {
            "go nodes 128 movetime 300"
        } else {
            "go nodes 128 movetime 30000"
        };
        for line in ["uci", "position startpos", go] {
            guard.events.send(Event::Line(line.into())).unwrap();
        }
        entering
            .recv_timeout(WAIT)
            .expect("first root reached the actual physical worker");
        assert_blocked_physical_owner_active(&native.inner.owner, "after physical worker entry");
        match boundary {
            BlockedBoundary::RootReplacement => {
                for line in ["position startpos moves e2e4", "go nodes 1 movetime 30000"] {
                    guard.events.send(Event::Line(line.into())).unwrap();
                }
            }
            BlockedBoundary::Stop => {
                for _ in 0..2 {
                    guard.events.send(Event::Line("stop".into())).unwrap();
                }
            }
            BlockedBoundary::Deadline => {
                let until = Instant::now() + WAIT;
                while !diagnostics_view.text().contains("Expired") {
                    assert!(
                        Instant::now() < until,
                        "independent hard deadline must close admission"
                    );
                    thread::sleep(Duration::from_millis(1));
                }
            }
        }
        guard.events.send(Event::Line("isready".into())).unwrap();
        let until = Instant::now() + WAIT;
        while !protocol_view.text().lines().any(|line| line == "readyok") {
            assert!(
                Instant::now() < until,
                "input owner stays responsive during physical drain"
            );
            thread::sleep(Duration::from_millis(1));
        }
        assert!(
            !protocol_view
                .text()
                .lines()
                .any(|line| line.starts_with("bestmove "))
        );
        assert_blocked_physical_owner_active(
            &native.inner.owner,
            "before current-root output release",
        );
        guard.release.take().unwrap().send(()).unwrap();
        let until = Instant::now() + WAIT;
        let bestmove = loop {
            if let Some(line) = protocol_view
                .text()
                .lines()
                .find(|line| line.starts_with("bestmove "))
            {
                break line.to_owned();
            }
            assert!(
                Instant::now() < until,
                "current root received a bounded legal fallback"
            );
            thread::sleep(Duration::from_millis(1));
        };
        let mut position = rz_position::Position::startpos();
        if matches!(boundary, BlockedBoundary::RootReplacement) {
            position.make_uci("e2e4").unwrap();
        }
        let movement =
            rz_position::BoardMove::from_uci(bestmove.strip_prefix("bestmove ").unwrap()).unwrap();
        assert!(position.legal_moves().contains(&movement));
        assert_eq!(
            factory.created.load(Ordering::Acquire),
            1,
            "busy root fallback cannot reload/create another provider session"
        );
        assert!(
            !native.inner.owner.try_status().unwrap().unwrap().active,
            "bestmove cannot acknowledge an active physical worker"
        );
        guard.events.send(Event::Line("quit".into())).unwrap();
        let until = Instant::now() + WAIT;
        while !guard.service.as_ref().unwrap().is_finished() {
            assert!(
                Instant::now() < until,
                "bounded quit and old-root physical drain"
            );
            thread::sleep(Duration::from_millis(1));
        }
        let served = guard.service.take().unwrap().join().unwrap();
        let report = native.finish(served).unwrap();
        assert_eq!(report.origin, NativeWorkerOrigin::Injected);
        assert_eq!(
            report.completed_by_runtime, 0,
            "an injected worker is not native ONNX execution evidence"
        );
        assert_eq!(
            protocol_view
                .text()
                .lines()
                .filter(|line| line.starts_with("bestmove "))
                .count(),
            1
        );
        if matches!(boundary, BlockedBoundary::RootReplacement) {
            assert!(diagnostics_view.text().contains("WorkerLimit"));
        }
        assert_eq!(report.search_root_initializations.count, 0);
        assert_eq!(report.search_non_root_backups.count, 0);
        assert!(!report.has_failure());
    }
}
