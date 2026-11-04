//! Bounded public receipts for the explicitly selected CPU ONNX process.
//!
//! Loading a session is separate from observing an accepted inference result.
//! These receipts disclose identities and typed causes, never local asset paths,
//! export commands, native error strings, or a complete request journal.

pub(crate) mod causes;
pub use causes::{CauseNodeV1, CauseReceiptV1};

use crate::{
    engine::ProcessClock,
    native_bootstrap::{
        NativeBootstrapError, NativeCompletedReceipt, NativeConfig, NativeCpuFactory,
        NativeRunError, NativeRunReport,
    },
};

#[cfg(test)]
use cap_std::{ambient_authority, fs::Dir};
use rz_contracts::*;
use rz_eval::{
    asset::AssetMetadata,
    native_runtime_bridge::NativeWorkerOrigin,
    onnx::{OnnxBackend, OrtRuntime, Provider},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::{fmt, io};

pub const STARTUP_FILE: &str = "native-cpu-startup.v1.json";
pub const TERMINATION_FILE: &str = "native-cpu-termination.v1.json";
pub const SCHEMA_VERSION: u32 = 1;
pub const MAX_RECEIPT_BYTES: usize = 256 * 1024;

pub(crate) fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|value| format!("{value:02x}")).collect()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RegistryIdentityV1 {
    pub owner: u64,
    pub slot: u64,
    pub generation: u64,
    pub manifest_sha256: String,
}
impl From<ModelHandle> for RegistryIdentityV1 {
    fn from(value: ModelHandle) -> Self {
        Self {
            owner: value.owner.0,
            slot: value.slot,
            generation: value.generation.0,
            manifest_sha256: hex(&value.manifest.0),
        }
    }
}
impl From<EncodingHandle> for RegistryIdentityV1 {
    fn from(value: EncodingHandle) -> Self {
        Self {
            owner: value.owner.0,
            slot: value.slot,
            generation: value.generation.0,
            manifest_sha256: hex(&value.manifest.0),
        }
    }
}

/// Captured exclusively after C has validated the asset, runtime, and session.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CpuProfileV1 {
    pub source_weights_gzip_sha256: String,
    pub source_weights_protobuf_sha256: String,
    pub onnx_sha256: String,
    pub onnx_bytes: usize,
    pub export_manifest_sha256: String,
    pub converter_commit: String,
    pub converter_binary_sha256: String,
    pub ort_library_sha256: String,
    pub ort_build_info_sha256: String,
    pub ort_release: String,
    pub backend_sha256: String,
    pub model: RegistryIdentityV1,
    pub encoding: RegistryIdentityV1,
    pub action_map_sha256: String,
    pub history_policy_sha256: String,
    pub history_length: u16,
    pub contract_major: u16,
    pub contract_minor: u16,
    pub provider: String,
    pub precision: String,
    pub max_batch_items: usize,
    pub intra_threads: usize,
    pub max_workers: usize,
    pub full_steps: u32,
    pub min_steps: u32,
    pub max_steps: u32,
    pub require_full: bool,
    pub evaluation_mode: String,
    pub history_fill: String,
}
impl CpuProfileV1 {
    pub(crate) fn from_loaded(
        asset: &AssetMetadata,
        runtime: &OrtRuntime,
        backend: &OnnxBackend,
        model: &ModelDescriptor,
    ) -> Result<Self, NativeBootstrapError> {
        let config = backend.config();
        if config.provider != Provider::Cpu
            || config.max_batch != 1
            || config.intra_threads != 1
            || backend.asset_identity() != asset.manifest_digest()
            || model.handle().manifest.0 != asset.manifest_digest()
            || !model.supports(PrecisionProfile::Fp32)
            || model.full_steps() != 1
            || model.max_batch_items() != 1
        {
            return Err(NativeBootstrapError::Config(
                "loaded provider differs from the CPU attestation profile",
            ));
        }
        let manifest = asset.manifest();
        let encoding = model.encoding();
        Ok(Self {
            source_weights_gzip_sha256: manifest.source_gzip_sha256.clone(),
            source_weights_protobuf_sha256: manifest.source_protobuf_sha256.clone(),
            onnx_sha256: manifest.onnx_sha256.clone(),
            onnx_bytes: manifest.onnx_bytes,
            export_manifest_sha256: hex(&asset.manifest_digest()),
            converter_commit: manifest.converter_commit.clone(),
            converter_binary_sha256: manifest.converter_binary_sha256.clone(),
            ort_library_sha256: hex(&runtime.binary_digest()),
            ort_build_info_sha256: hex(&Sha256::digest(runtime.build_info().as_bytes()).into()),
            ort_release: "1.22.0".into(),
            backend_sha256: hex(&backend.identity()),
            model: model.handle().into(),
            encoding: encoding.handle.into(),
            action_map_sha256: hex(&encoding.action_map.0),
            history_policy_sha256: hex(&encoding.history_policy.0),
            history_length: encoding.history_length,
            contract_major: CONTRACT_REVISION.major,
            contract_minor: CONTRACT_REVISION.minor,
            provider: "cpu".into(),
            precision: "fp32".into(),
            max_batch_items: config.max_batch,
            intra_threads: config.intra_threads,
            max_workers: 1,
            full_steps: model.full_steps(),
            min_steps: 1,
            max_steps: 1,
            require_full: true,
            evaluation_mode: "fresh".into(),
            history_fill: "no".into(),
        })
    }
}

pub use crate::process_receipts::{ExecutableIdentitySourceV1, ExecutableIdentityV1};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StartupReceiptV1 {
    pub schema_version: u32,
    pub kind: String,
    /// Loaded means a verified CPU session, not an executed inference probe.
    pub loaded: bool,
    pub process_id: u32,
    pub process_run_id: String,
    pub process_epoch: u64,
    pub executable: ExecutableIdentityV1,
    pub profile: CpuProfileV1,
}
impl StartupReceiptV1 {
    pub fn capture(
        factory: &NativeCpuFactory,
        clock: &ProcessClock,
    ) -> Result<Self, AttestationError> {
        let profile = factory.attestation_profile().map_err(|error| {
            AttestationError::contract(
                "only an actually loaded CPU ONNX factory can issue startup evidence",
                error,
            )
        })?;
        Ok(Self {
            schema_version: SCHEMA_VERSION,
            kind: "startup".into(),
            loaded: true,
            process_id: std::process::id(),
            process_run_id: process_run_id(),
            process_epoch: clock.domain().0.0,
            executable: ExecutableIdentityV1::observe()?,
            profile: profile.clone(),
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EpochSequenceV1 {
    pub epoch: u64,
    pub sequence: u64,
}
fn request_id(value: RequestId) -> EpochSequenceV1 {
    EpochSequenceV1 {
        epoch: value.epoch.0,
        sequence: value.sequence,
    }
}
fn execution_id(value: ExecutionId) -> EpochSequenceV1 {
    EpochSequenceV1 {
        epoch: value.epoch.0,
        sequence: value.sequence,
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionReceiptV1 {
    pub contract_major: u16,
    pub contract_minor: u16,
    pub request: EpochSequenceV1,
    pub selection: EpochSequenceV1,
    pub game: u64,
    pub root: u64,
    pub state_owner: u64,
    pub state_revision: u64,
    pub state_sha256: String,
    pub legal_order_sha256: String,
    pub input_sha256: String,
    pub model: RegistryIdentityV1,
    pub encoding: RegistryIdentityV1,
    pub backend_sha256: String,
    pub precision: String,
    pub min_steps: u32,
    pub max_steps: u32,
    pub require_full: bool,
    pub execution: Option<EpochSequenceV1>,
    pub actual: Option<ActualComputeReceiptV1>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ActualComputeReceiptV1 {
    pub precision: String,
    pub steps: u32,
    pub full: bool,
    pub backend_sha256: String,
    pub execution: Option<EpochSequenceV1>,
    pub provenance: String,
    pub prior_execution: Option<EpochSequenceV1>,
}
fn precision(value: PrecisionProfile) -> String {
    match value {
        PrecisionProfile::Fp32 => "fp32".into(),
        PrecisionProfile::Fp16 => "fp16".into(),
        PrecisionProfile::Bf16 => "bf16".into(),
        PrecisionProfile::Quantized(digest) => format!("quantized:{}", hex(&digest.0)),
    }
}
impl CompletionReceiptV1 {
    pub(crate) fn from_context(context: CompletionContext) -> Self {
        let request = context.request;
        Self {
            contract_major: request.revision.major,
            contract_minor: request.revision.minor,
            request: request_id(request.request),
            selection: EpochSequenceV1 {
                epoch: request.selection.epoch.0,
                sequence: request.selection.sequence,
            },
            game: request.game.0,
            root: request.root.0,
            state_owner: request.state.owner.0,
            state_revision: request.state.revision.0,
            state_sha256: hex(&request.state.semantic.0),
            legal_order_sha256: hex(&request.legal_order.0.0),
            input_sha256: hex(&request.input.0.0),
            model: request.model.into(),
            encoding: request.encoding.into(),
            backend_sha256: hex(&request.backend.0),
            precision: precision(request.precision),
            min_steps: request.compute.min_steps,
            max_steps: request.compute.max_steps,
            require_full: request.compute.require_full,
            execution: context.execution.map(execution_id),
            actual: None,
        }
    }
    pub(crate) fn completed(receipt: NativeCompletedReceipt) -> Self {
        let mut result = Self::from_context(receipt.context);
        let (provenance, prior) = match receipt.actual.provenance {
            CacheProvenance::Computed => ("computed", None),
            CacheProvenance::ExactFeatureReuse => ("exact_feature_reuse", None),
            CacheProvenance::RawEvalHit { source_execution } => ("raw_eval_hit", source_execution),
        };
        result.actual = Some(ActualComputeReceiptV1 {
            precision: precision(receipt.actual.precision),
            steps: receipt.actual.steps,
            full: receipt.actual.full,
            backend_sha256: hex(&receipt.actual.backend.0),
            execution: receipt.actual.execution.map(execution_id),
            provenance: provenance.into(),
            prior_execution: prior.map(execution_id),
        });
        result
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AuditAggregateV1 {
    pub count: u64,
    pub first: Option<causes::CauseReceiptV1>,
    pub last: Option<causes::CauseReceiptV1>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunEvidenceV1 {
    pub model_manifest_sha256: String,
    pub backend_sha256: String,
    pub origin: String,
    /// Process-level D acceptances observed on B's normal poll path. This is
    /// neither physical invocation count nor proof that every root used NN.
    pub completed_by_runtime: u64,
    pub first_completed: Option<CompletionReceiptV1>,
    pub last_completed: Option<CompletionReceiptV1>,
    pub completed_scope: String,
    pub canceled: AuditAggregateV1,
    pub expired: AuditAggregateV1,
    pub failures: Vec<causes::CauseReceiptV1>,
    pub overflow: Option<causes::CauseReceiptV1>,
    pub boundary_error: Option<causes::CauseReceiptV1>,
    pub poison_error: Option<causes::CauseReceiptV1>,
}
impl TryFrom<&NativeRunReport> for RunEvidenceV1 {
    type Error = ContractError;
    fn try_from(report: &NativeRunReport) -> Result<Self, Self::Error> {
        #[cfg(feature = "experimental-raw-cache")]
        if report.raw_cache_completions.count != 0 {
            return Err(ContractError::new(
                ErrorCode::UnsupportedContract,
                Stage::Output,
                "experimental raw-cache evidence requires the typed report; baseline V1 is Computed-only",
            ));
        }
        if report.origin == NativeWorkerOrigin::CudaOnnx {
            return Err(ContractError::new(
                ErrorCode::UnsupportedContract,
                Stage::Output,
                "CUDA evidence cannot be published as CPU V1",
            ));
        }
        let aggregate = |value: &crate::native_bootstrap::NativeAuditAggregate| AuditAggregateV1 {
            count: value.count,
            first: value.first.as_ref().map(causes::native),
            last: value.last.as_ref().map(causes::native),
        };
        Ok(Self {
            model_manifest_sha256: hex(&report.model_manifest.0),
            backend_sha256: hex(&report.backend.0),
            origin: match report.origin {
                NativeWorkerOrigin::CpuOnnx => "cpu_onnx",
                NativeWorkerOrigin::Injected => "injected",
                NativeWorkerOrigin::CudaOnnx => {
                    return Err(ContractError::new(
                        ErrorCode::UnsupportedContract,
                        Stage::Output,
                        "CUDA evidence cannot be published as CPU V1",
                    ));
                }
            }
            .into(),
            completed_by_runtime: report.completed_by_runtime,
            first_completed: report.first_completed.map(CompletionReceiptV1::completed),
            last_completed: report.last_completed.map(CompletionReceiptV1::completed),
            completed_scope:
                "process_aggregate_normal_poll_first_last_not_full_root_or_physical_journal".into(),
            canceled: aggregate(&report.canceled),
            expired: aggregate(&report.expired),
            failures: report.failures.iter().map(causes::native).collect(),
            overflow: report.overflow.as_ref().map(causes::native),
            boundary_error: report.boundary_error.as_ref().map(causes::contract),
            poison_error: report.poison_error.as_ref().map(causes::contract),
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TerminationReceiptV1 {
    pub schema_version: u32,
    pub kind: String,
    pub process_run_id: String,
    pub startup: StartupReceiptV1,
    pub run_succeeded: bool,
    /// Process-level observation, not proof for every root or bestmove.
    pub actual_cpu_inference_observed: bool,
    pub physical_drain: String,
    pub report: Option<RunEvidenceV1>,
    pub original_service_failure: Option<causes::CauseReceiptV1>,
    pub collection_failure: Option<causes::CauseReceiptV1>,
    pub retained_owner_and_evidence: bool,
}
impl TerminationReceiptV1 {
    pub fn from_result(
        startup: &StartupReceiptV1,
        result: &Result<NativeRunReport, NativeRunError>,
    ) -> Result<Self, AttestationError> {
        let (report, service, collection, drain, retained) = match result {
            Ok(report) => (Some(report), None, None, "confirmed", false),
            Err(error) => (
                error.report.as_deref(),
                error.service.as_deref(),
                error.collection_error.as_ref(),
                if error.report.is_some() && error.collection_error.is_none() {
                    "confirmed"
                } else {
                    "unconfirmed"
                },
                true,
            ),
        };
        let wire_report = report
            .map(RunEvidenceV1::try_from)
            .transpose()
            .map_err(|error| {
                AttestationError::contract("CPU V1 refuses a different native origin", error)
            })?;
        Ok(Self {
            schema_version: SCHEMA_VERSION,
            kind: "termination".into(),
            process_run_id: startup.process_run_id.clone(),
            startup: startup.clone(),
            run_succeeded: result.is_ok(),
            actual_cpu_inference_observed: report
                .is_some_and(|report| actual_cpu_observed(startup, report)),
            physical_drain: drain.into(),
            report: wire_report,
            original_service_failure: service
                .map(causes::engine)
                .transpose()
                .map_err(AttestationError::from)?,
            collection_failure: collection.map(causes::contract),
            retained_owner_and_evidence: retained,
        })
    }
}

fn actual_cpu_observed(startup: &StartupReceiptV1, report: &NativeRunReport) -> bool {
    let valid = |value: NativeCompletedReceipt| {
        let actual = value.actual;
        let request = value.context.request;
        let registry = |value: ModelHandle, expected: &RegistryIdentityV1| {
            value.owner.0 == expected.owner
                && value.slot == expected.slot
                && value.generation.0 == expected.generation
                && hex(&value.manifest.0) == expected.manifest_sha256
        };
        let encoding = request.encoding;
        value.context.execution == actual.execution
            && actual.execution.is_some()
            && actual.precision == PrecisionProfile::Fp32
            && actual.steps == 1
            && actual.full
            && actual.provenance == CacheProvenance::Computed
            && hex(&actual.backend.0) == startup.profile.backend_sha256
            && request.backend == actual.backend
            && request.precision == PrecisionProfile::Fp32
            && request.compute
                == ComputeBudget {
                    min_steps: 1,
                    max_steps: 1,
                    require_full: true,
                }
            && request.revision == CONTRACT_REVISION
            && request.request.epoch.0 == startup.process_epoch
            && request.selection.epoch == request.request.epoch
            && actual
                .execution
                .is_some_and(|value| value.epoch == request.request.epoch)
            && registry(request.model, &startup.profile.model)
            && encoding.owner.0 == startup.profile.encoding.owner
            && encoding.slot == startup.profile.encoding.slot
            && encoding.generation.0 == startup.profile.encoding.generation
            && hex(&encoding.manifest.0) == startup.profile.encoding.manifest_sha256
    };
    startup.loaded
        && startup.profile.provider == "cpu"
        && startup.profile.precision == "fp32"
        && report.origin == NativeWorkerOrigin::CpuOnnx
        && report.completed_by_runtime > 0
        && hex(&report.model_manifest.0) == startup.profile.export_manifest_sha256
        && hex(&report.backend.0) == startup.profile.backend_sha256
        && report.first_completed.is_some_and(valid)
        && report.last_completed.is_some_and(valid)
}

pub(crate) fn bounded_json(value: &impl Serialize) -> Result<Vec<u8>, AttestationError> {
    crate::process_receipts::bounded_json(value).map_err(Into::into)
}
/// The caller must exclusively own the existing private output directory and
/// prevent outside mutation during this process. Both names are fixed; opened
/// file handles and the directory capability survive later pathname changes.
/// Empty or partially written files after failure are incomplete receipts.
pub struct ReceiptWriter(crate::process_receipts::ProcessReceiptWriter);
impl ReceiptWriter {
    pub fn open(config: &NativeConfig) -> Result<Self, AttestationError> {
        if config.provider() != crate::native_bootstrap::NativeProvider::Cpu {
            return Err(AttestationError::boundary(
                "CPU V1 writer requires the CPU provider",
            ));
        }
        Self::open_named(config, STARTUP_FILE, TERMINATION_FILE)
    }
    pub(crate) fn open_named(
        config: &NativeConfig,
        startup_name: &'static str,
        termination_name: &'static str,
    ) -> Result<Self, AttestationError> {
        let root = config.attestation_output_root().map_err(|error| {
            AttestationError::bootstrap("private attestation output root is invalid", error)
        })?;
        Self::open_named_root(&root, startup_name, termination_name)
    }
    pub(crate) fn open_named_root(
        root: &std::path::Path,
        startup_name: &'static str,
        termination_name: &'static str,
    ) -> Result<Self, AttestationError> {
        Ok(Self(
            crate::process_receipts::ProcessReceiptWriter::open_named_root(
                root,
                startup_name,
                termination_name,
            )?,
        ))
    }
    #[cfg(test)]
    fn from_directory(directory: Dir) -> Result<Self, AttestationError> {
        Ok(Self(
            crate::process_receipts::ProcessReceiptWriter::from_directory_named(
                directory,
                STARTUP_FILE,
                TERMINATION_FILE,
            )?,
        ))
    }
    pub fn startup(&mut self, receipt: &StartupReceiptV1) -> Result<(), AttestationError> {
        self.publish_startup(receipt)
    }
    pub(crate) fn publish_startup(
        &mut self,
        receipt: &impl Serialize,
    ) -> Result<(), AttestationError> {
        self.0.publish_startup(receipt).map_err(Into::into)
    }
    pub(crate) fn publish_auxiliary(
        &self,
        name: &'static str,
        receipt: &impl Serialize,
    ) -> Result<(), AttestationError> {
        self.0.publish_auxiliary(name, receipt).map_err(Into::into)
    }
    #[cfg(feature = "experimental-batch")]
    pub(crate) fn publish_large_auxiliary(
        &self,
        name: &'static str,
        receipt: &impl Serialize,
        cap: u64,
    ) -> Result<(String, u64), AttestationError> {
        self.0
            .publish_large_auxiliary(name, receipt, cap)
            .map_err(Into::into)
    }
    pub fn termination(&mut self, receipt: &TerminationReceiptV1) -> Result<(), AttestationError> {
        self.publish_termination(receipt)
    }
    pub(crate) fn publish_termination(
        &mut self,
        receipt: &impl Serialize,
    ) -> Result<(), AttestationError> {
        self.0.publish_termination(receipt).map_err(Into::into)
    }
}
pub(crate) fn process_run_id() -> String {
    format!("native-process-{}", std::process::id())
}

/// Public error metadata is static or typed; the native run error remains owned
/// when publication and protocol/drain failure happen together.
pub struct AttestationError {
    pub stage: &'static str,
    pub io_kind: Option<io::ErrorKind>,
    pub io_error: Option<io::Error>,
    pub bootstrap_error: Option<Box<NativeBootstrapError>>,
    pub contract_error: Option<ContractError>,
    pub native_run: Option<Box<NativeRunError>>,
    pub native_report: Option<Box<NativeRunReport>>,
}
impl From<crate::process_receipts::ProcessReceiptError> for AttestationError {
    fn from(error: crate::process_receipts::ProcessReceiptError) -> Self {
        Self {
            stage: error.stage,
            io_kind: error.io_kind,
            io_error: error.io_error,
            ..Self::boundary(error.stage)
        }
    }
}
impl AttestationError {
    pub(crate) fn boundary(stage: &'static str) -> Self {
        Self {
            stage,
            io_kind: None,
            io_error: None,
            bootstrap_error: None,
            contract_error: None,
            native_run: None,
            native_report: None,
        }
    }
    pub(crate) fn io(stage: &'static str, error: io::Error) -> Self {
        Self {
            io_kind: Some(error.kind()),
            io_error: Some(error),
            ..Self::boundary(stage)
        }
    }
    fn bootstrap(stage: &'static str, error: NativeBootstrapError) -> Self {
        Self {
            bootstrap_error: Some(Box::new(error)),
            ..Self::boundary(stage)
        }
    }
    pub(crate) fn contract(stage: &'static str, error: ContractError) -> Self {
        Self {
            contract_error: Some(error),
            ..Self::boundary(stage)
        }
    }
    pub fn retaining(mut self, result: Result<NativeRunReport, NativeRunError>) -> Self {
        match result {
            Ok(report) => self.native_report = Some(Box::new(report)),
            Err(error) => self.native_run = Some(Box::new(error)),
        }
        self
    }
}
impl fmt::Debug for AttestationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AttestationError")
            .field("stage", &self.stage)
            .field("io_kind", &self.io_kind)
            .field("native_run_retained", &self.native_run.is_some())
            .finish()
    }
}
impl fmt::Display for AttestationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "native CPU attestation failed: {} ({:?}); native failure retained={}",
            self.stage,
            self.io_kind,
            self.native_run.is_some()
        )
    }
}
impl std::error::Error for AttestationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    static DIRECTORY_IDS: AtomicU64 = AtomicU64::new(0);
    fn directory() -> (PathBuf, Dir) {
        let id = DIRECTORY_IDS.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("rz-native-attestation-{}-{id}", std::process::id()));
        fs::create_dir(&path).unwrap();
        let directory = Dir::open_ambient_dir(&path, ambient_authority()).unwrap();
        (path, directory)
    }

    #[test]
    fn auxiliary_receipt_requires_startup_and_refuses_overwrite() {
        let (path, directory) = directory();
        let mut writer = ReceiptWriter::from_directory(directory).unwrap();
        assert!(writer.publish_auxiliary("search.json", &"config").is_err());
        writer.publish_startup(&"startup").unwrap();
        writer.publish_auxiliary("search.json", &"config").unwrap();
        let original = fs::read(path.join("search.json")).unwrap();
        assert!(
            writer
                .publish_auxiliary("search.json", &"replacement")
                .is_err()
        );
        assert_eq!(fs::read(path.join("search.json")).unwrap(), original);
        drop(writer);
        for name in [STARTUP_FILE, TERMINATION_FILE, "search.json"] {
            fs::remove_file(path.join(name)).unwrap();
        }
        fs::remove_dir(path).unwrap();
    }

    #[test]
    fn receipt_byte_budget_rejects_oversized_payload_before_publication() {
        let rejected = bounded_json(&"x".repeat(MAX_RECEIPT_BYTES)).unwrap_err();
        assert_eq!(rejected.stage, "cannot encode bounded public receipt");
        assert!(bounded_json(&vec!["startup", "termination"]).unwrap().len() < MAX_RECEIPT_BYTES);
    }

    #[test]
    fn existing_receipt_is_rejected_without_overwrite_or_new_final_file() {
        let (path, directory) = directory();
        fs::write(path.join(STARTUP_FILE), b"original receipt").unwrap();
        let rejected = ReceiptWriter::from_directory(directory).err().unwrap();
        assert_eq!(rejected.io_kind, Some(io::ErrorKind::AlreadyExists));
        assert_eq!(
            fs::read(path.join(STARTUP_FILE)).unwrap(),
            b"original receipt"
        );
        assert!(!path.join(TERMINATION_FILE).exists());
        fs::remove_file(path.join(STARTUP_FILE)).unwrap();
        fs::remove_dir(path).unwrap();
    }

    #[test]
    fn process_slot_reuse_is_rejected_and_public_errors_never_echo_its_path() {
        let (path, directory) = directory();
        drop(directory);
        let private_marker = path.to_string_lossy().into_owned();
        let config = NativeConfig::parse(vec![
            "--onnx-cpu".into(),
            "--attestation".into(),
            "--source-weights=unused".into(),
            "--onnx-model=unused".into(),
            "--export-manifest=unused".into(),
            format!("--manifest-sha256={}", "ab".repeat(32)),
            "--ort-library=unused".into(),
            format!("--ort-sha256={}", "cd".repeat(32)),
            format!("--output-root={private_marker}"),
        ])
        .unwrap();
        let writer = ReceiptWriter::open(&config).unwrap();
        let rejected = ReceiptWriter::open(&config).err().unwrap();
        assert_eq!(rejected.stage, "reserve fresh process receipt directory");
        assert_eq!(rejected.io_kind, Some(io::ErrorKind::AlreadyExists));
        assert!(!format!("{rejected:?} {rejected}").contains(&private_marker));
        let slot = path.join(process_run_id());
        assert!(slot.join(STARTUP_FILE).is_file());
        assert!(slot.join(TERMINATION_FILE).is_file());
        drop(writer);
        fs::remove_file(slot.join(STARTUP_FILE)).unwrap();
        fs::remove_file(slot.join(TERMINATION_FILE)).unwrap();
        fs::remove_dir(slot).unwrap();
        fs::remove_dir(path).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn receipt_symlink_is_rejected_and_target_bytes_are_preserved() {
        let (path, directory) = directory();
        fs::write(path.join("owned-target"), b"target must survive").unwrap();
        std::os::unix::fs::symlink("owned-target", path.join(STARTUP_FILE)).unwrap();
        assert!(ReceiptWriter::from_directory(directory).is_err());
        assert_eq!(
            fs::read(path.join("owned-target")).unwrap(),
            b"target must survive"
        );
        fs::remove_file(path.join(STARTUP_FILE)).unwrap();
        fs::remove_file(path.join("owned-target")).unwrap();
        fs::remove_dir(path).unwrap();
    }
}
