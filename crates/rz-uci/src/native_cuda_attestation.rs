//! Additive CUDA receipts. CPU V1 keeps its original provider meaning.
//! Counts are bounded process aggregates, never physical invocation totals or
//! proof that every root/bestmove used a neural evaluation.

use crate::{
    engine::ProcessClock,
    native_attestation::{
        self, AttestationError, AuditAggregateV1, CauseReceiptV1, CompletionReceiptV1,
        ExecutableIdentityV1, ReceiptWriter, RegistryIdentityV1, causes,
    },
    native_bootstrap::{
        NativeBootstrapError, NativeCompletedReceipt, NativeConfig, NativeCudaFactory,
        NativeObservationReport, NativeProvider, NativeRunError, NativeRunReport,
        NativeSearchAggregate, NativeSearchConsumedReceipt,
    },
};
use rz_contracts::*;
use rz_eval::{
    asset::AssetMetadata,
    native_runtime_bridge::NativeWorkerOrigin,
    onnx::{OnnxBackend, OrtRuntime, Provider},
    runtime_pin::RuntimeBundleFileRole,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

pub const STARTUP_FILE: &str = "native-cuda-startup.v1.json";
pub const TERMINATION_FILE: &str = "native-cuda-termination.v1.json";
pub const SCHEMA_VERSION: u32 = 1;
pub const SEARCH_FILE: &str = "native-cuda-search-config.v1.json";

/// Separate additive search record; provider V1's closed field set is retained.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CudaSearchReceiptV1 {
    pub schema_version: u32,
    pub kind: String,
    pub process_run_id: String,
    pub startup_sha256: String,
    pub simulations: u64,
    pub final_selection: String,
    pub policy_temperature_milli: u32,
    pub raw_cache: bool,
    pub batch_size: u32,
    pub search_workers: usize,
    pub output_margin_ms: u64,
    pub drain_margin_ms: u64,
    pub shutdown_ms: u64,
    pub max_nodes: usize,
    pub max_edges: usize,
    pub max_depth: usize,
    pub scope: String,
}
impl CudaSearchReceiptV1 {
    pub fn capture(
        startup: &CudaStartupReceiptV1,
        settings: &crate::engine::EngineSettings,
    ) -> Result<Self, AttestationError> {
        let startup_bytes = native_attestation::bounded_json(startup)?;
        let milliseconds = |duration: std::time::Duration| {
            u64::try_from(duration.as_millis())
                .map_err(|_| AttestationError::boundary("search duration exceeds receipt range"))
        };
        Ok(Self {
            schema_version: 1,
            kind: "search_config".into(),
            process_run_id: startup.process_run_id.clone(),
            startup_sha256: native_attestation::hex(&Sha256::digest(startup_bytes).into()),
            simulations: settings.search.max_simulations,
            final_selection: match settings.final_move_policy {
                rz_search::tree::FinalMovePolicy::Visits => "visits",
                rz_search::tree::FinalMovePolicy::ExactTerminal => "exact-terminal",
            }
            .into(),
            policy_temperature_milli: 1000,
            raw_cache: false,
            batch_size: 1,
            search_workers: settings.max_workers,
            output_margin_ms: milliseconds(settings.search.time_config.output_margin)?,
            drain_margin_ms: milliseconds(settings.search.time_config.drain_margin)?,
            shutdown_ms: milliseconds(settings.shutdown_limit)?,
            max_nodes: settings.tree.max_nodes,
            max_edges: settings.tree.max_edges,
            max_depth: settings.tree.max_depth,
            scope: "startup_config_bound_to_served_settings_not_per_move_timing_or_strength".into(),
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CudaBundleFileV1 {
    pub role: String,
    pub filename: String,
    pub bytes: u64,
    pub sha256: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SessionAdmissionV1 {
    pub host_bytes: u64,
    pub device_bytes: u64,
    pub pinned_bytes: u64,
    pub scope: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CudaProfileV1 {
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
    pub device_id: i32,
    pub arena_bytes: u64,
    pub tf32: bool,
    pub runtime_bundle_manifest_sha256: String,
    pub runtime_bundle_sha256: String,
    pub runtime_bundle_files: Vec<CudaBundleFileV1>,
    pub cuda_profile_sha256: String,
    pub executed_cuda_nodes: usize,
    pub runtime_mapping_verified: bool,
    pub session_resident_admission: SessionAdmissionV1,
}
impl CudaProfileV1 {
    pub(crate) fn from_loaded_limit(
        asset: &AssetMetadata,
        runtime: &OrtRuntime,
        backend: &OnnxBackend,
        model: &ModelDescriptor,
        bundle_file_sha: [u8; 32],
        max_batch: usize,
    ) -> Result<Self, NativeBootstrapError> {
        let config = backend.config();
        if config.provider
            != (Provider::Cuda {
                device_id: 0,
                arena_bytes: asset.profile().cuda_arena_bytes(),
            })
            || config.max_batch != max_batch
            || !(1..=16).contains(&max_batch)
            || config.intra_threads != 1
            || backend.asset_identity() != asset.manifest_digest()
            || model.handle().manifest.0 != asset.manifest_digest()
            || !model.supports(PrecisionProfile::Fp32)
            || model.full_steps() != 1
            || model.max_batch_items() != max_batch
            || model.encoding().handle.manifest
                != rz_eval::contracts::encoding_manifest(rz_encoding::classical::HistoryFill::No)
        {
            return Err(NativeBootstrapError::Config(
                "loaded provider differs from the fixed CUDA attestation profile",
            ));
        }
        runtime.verify_cuda_runtime_mappings()?;
        let bundle_digest = runtime.bundle_digest().ok_or(NativeBootstrapError::Config(
            "loaded CUDA runtime has no pinned bundle identity",
        ))?;
        let files = runtime.bundle_files().ok_or(NativeBootstrapError::Config(
            "loaded CUDA runtime has no exact pinned bundle files",
        ))?;
        // Keep the loader-alias declaration in the digest. Reconstructing a
        // schema-1 spec from filenames alone would discard schema-2 identity.
        let spec = runtime.bundle_spec().ok_or(NativeBootstrapError::Config(
            "loaded CUDA runtime has no pinned bundle specification",
        ))?;
        if files.len() != 19
            || spec.digest()? != bundle_digest
            || backend.runtime_bundle_digest() != Some(bundle_digest)
        {
            return Err(NativeBootstrapError::Config(
                "loaded CUDA runtime differs from the exact nineteen-file bundle",
            ));
        }
        let probe = backend
            .cuda_evidence()
            .filter(|probe| probe.executed_cuda_nodes > 0)
            .ok_or(NativeBootstrapError::Config(
                "loaded CUDA session has no successful placement probe",
            ))?;
        let manifest = asset.manifest();
        let encoding = model.encoding();
        let hex = native_attestation::hex;
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
            provider: "cuda".into(),
            precision: "fp32".into(),
            max_batch_items: max_batch,
            intra_threads: 1,
            max_workers: 1,
            full_steps: 1,
            min_steps: 1,
            max_steps: 1,
            require_full: true,
            evaluation_mode: "fresh".into(),
            history_fill: "no".into(),
            device_id: 0,
            arena_bytes: asset.profile().cuda_arena_bytes() as u64,
            tf32: false,
            runtime_bundle_manifest_sha256: hex(&bundle_file_sha),
            runtime_bundle_sha256: hex(&bundle_digest),
            runtime_bundle_files: files
                .iter()
                .map(|file| CudaBundleFileV1 {
                    role: match file.role {
                        RuntimeBundleFileRole::Core => "core",
                        RuntimeBundleFileRole::ProvidersShared => "providers_shared",
                        RuntimeBundleFileRole::ProvidersCuda => "providers_cuda",
                        RuntimeBundleFileRole::NvidiaDependency => "nvidia_dependency",
                    }
                    .into(),
                    filename: file.filename.clone(),
                    bytes: file.bytes,
                    sha256: file.sha256.clone(),
                })
                .collect(),
            cuda_profile_sha256: hex(&probe.profile_sha256),
            executed_cuda_nodes: probe.executed_cuda_nodes,
            runtime_mapping_verified: true,
            session_resident_admission: SessionAdmissionV1 {
                host_bytes: 0,
                device_bytes: asset.profile().cuda_arena_bytes() as u64,
                pinned_bytes: 0,
                scope: "declaration_not_measured_vram_or_hardcap".into(),
            },
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CudaStartupReceiptV1 {
    pub schema_version: u32,
    pub kind: String,
    pub loaded: bool,
    pub process_id: u32,
    pub process_run_id: String,
    pub process_epoch: u64,
    pub executable: ExecutableIdentityV1,
    pub profile: CudaProfileV1,
}
impl CudaStartupReceiptV1 {
    pub fn capture(
        factory: &NativeCudaFactory,
        clock: &ProcessClock,
    ) -> Result<Self, AttestationError> {
        let profile = factory.attestation_profile().map_err(|error| {
            AttestationError::contract(
                "only an actually loaded CUDA owner can issue startup evidence",
                error,
            )
        })?;
        Ok(Self {
            schema_version: SCHEMA_VERSION,
            kind: "startup".into(),
            loaded: true,
            process_id: std::process::id(),
            process_run_id: native_attestation::process_run_id(),
            process_epoch: clock.domain().0.0,
            executable: ExecutableIdentityV1::observe()?,
            profile: profile.clone(),
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchConsumedReceiptV1 {
    pub completed: CompletionReceiptV1,
    pub traversed_edges: usize,
}
impl From<NativeSearchConsumedReceipt> for SearchConsumedReceiptV1 {
    fn from(value: NativeSearchConsumedReceipt) -> Self {
        Self {
            completed: CompletionReceiptV1::completed(value.completed),
            traversed_edges: value.traversed_edges,
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchAggregateV1 {
    pub count: u64,
    pub first: Option<SearchConsumedReceiptV1>,
    pub last: Option<SearchConsumedReceiptV1>,
}
impl From<&NativeSearchAggregate> for SearchAggregateV1 {
    fn from(value: &NativeSearchAggregate) -> Self {
        Self {
            count: value.count,
            first: value.first.map(Into::into),
            last: value.last.map(Into::into),
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationReportV1 {
    pub scheduler_events: u64,
    pub scheduler_dropped: u64,
    pub scheduler_counter_overflow: bool,
    pub delivery_events: u64,
    pub delivery_dropped: u64,
    pub delivery_counter_overflow: bool,
    pub drain_discarded_results: u64,
    pub scope: String,
}
impl From<&NativeObservationReport> for ObservationReportV1 {
    fn from(value: &NativeObservationReport) -> Self {
        Self {
            scheduler_events: value.scheduler_events,
            scheduler_dropped: value.scheduler_dropped,
            scheduler_counter_overflow: value.scheduler_counter_overflow,
            delivery_events: value.delivery_events,
            delivery_dropped: value.delivery_dropped,
            delivery_counter_overflow: value.delivery_counter_overflow,
            drain_discarded_results: value.drain_discarded_results,
            scope: "drained_event_counts_and_loss_not_full_journal_or_physical_inference_count"
                .into(),
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CudaRunEvidenceV1 {
    pub model_manifest_sha256: String,
    pub backend_sha256: String,
    pub origin: String,
    pub completed_by_runtime: u64,
    pub first_completed: Option<CompletionReceiptV1>,
    pub last_completed: Option<CompletionReceiptV1>,
    pub completed_scope: String,
    pub search_root_initializations: SearchAggregateV1,
    pub search_non_root_backups: SearchAggregateV1,
    pub search_scope: String,
    pub observations: ObservationReportV1,
    pub canceled: AuditAggregateV1,
    pub expired: AuditAggregateV1,
    pub failures: Vec<CauseReceiptV1>,
    pub overflow: Option<CauseReceiptV1>,
    pub boundary_error: Option<CauseReceiptV1>,
    pub poison_error: Option<CauseReceiptV1>,
}
impl TryFrom<&NativeRunReport> for CudaRunEvidenceV1 {
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
        if report.origin != NativeWorkerOrigin::CudaOnnx {
            return Err(ContractError::new(
                ErrorCode::UnsupportedContract,
                Stage::Output,
                "CUDA V1 rejects CPU or injected evidence",
            ));
        }
        let audit = |value: &crate::native_bootstrap::NativeAuditAggregate| AuditAggregateV1 {
            count: value.count,
            first: value.first.as_ref().map(causes::native),
            last: value.last.as_ref().map(causes::native),
        };
        Ok(Self {
            model_manifest_sha256: native_attestation::hex(&report.model_manifest.0), backend_sha256: native_attestation::hex(&report.backend.0),
            origin: "cuda_onnx".into(), completed_by_runtime: report.completed_by_runtime,
            first_completed: report.first_completed.map(CompletionReceiptV1::completed), last_completed: report.last_completed.map(CompletionReceiptV1::completed),
            completed_scope: "process_aggregate_normal_poll_first_last_not_full_root_or_physical_journal".into(),
            search_root_initializations: (&report.search_root_initializations).into(), search_non_root_backups: (&report.search_non_root_backups).into(),
            search_scope: "unchanged_final_tree_guard_accepted_count_first_last_not_every_root_or_bestmove_proof".into(),
            observations: (&report.observations).into(), canceled: audit(&report.canceled), expired: audit(&report.expired),
            failures: report.failures.iter().map(causes::native).collect(), overflow: report.overflow.as_ref().map(causes::native),
            boundary_error: report.boundary_error.as_ref().map(causes::contract), poison_error: report.poison_error.as_ref().map(causes::contract),
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CudaTerminationReceiptV1 {
    pub schema_version: u32,
    pub kind: String,
    pub process_run_id: String,
    pub startup: CudaStartupReceiptV1,
    pub run_succeeded: bool,
    pub actual_cuda_inference_observed: bool,
    pub actual_rules_search_backup_observed: bool,
    pub physical_drain: String,
    pub report: Option<CudaRunEvidenceV1>,
    pub original_service_failure: Option<CauseReceiptV1>,
    pub collection_failure: Option<CauseReceiptV1>,
    pub runtime_mapping_failure: Option<CauseReceiptV1>,
    pub retained_owner_and_evidence: bool,
}
impl CudaTerminationReceiptV1 {
    pub fn from_result(
        startup: &CudaStartupReceiptV1,
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
            .map(CudaRunEvidenceV1::try_from)
            .transpose()
            .map_err(|error| {
                AttestationError::contract("CUDA V1 refuses a different native origin", error)
            })?;
        let actual_cuda = report.is_some_and(|report| actual_cuda_observed(startup, report));
        let backup = report.is_some_and(|report| {
            report.search_non_root_backups.count > 0
                && report.search_non_root_backups.first.is_some_and(|receipt| {
                    receipt.traversed_edges > 0 && valid_completed(startup, receipt.completed)
                })
                && report.search_non_root_backups.last.is_some_and(|receipt| {
                    receipt.traversed_edges > 0 && valid_completed(startup, receipt.completed)
                })
        });
        Ok(Self {
            schema_version: SCHEMA_VERSION,
            kind: "termination".into(),
            process_run_id: startup.process_run_id.clone(),
            startup: startup.clone(),
            run_succeeded: result.is_ok(),
            actual_cuda_inference_observed: actual_cuda,
            actual_rules_search_backup_observed: actual_cuda && backup,
            physical_drain: drain.into(),
            report: wire_report,
            original_service_failure: service
                .map(causes::engine)
                .transpose()
                .map_err(AttestationError::from)?,
            collection_failure: collection.map(causes::contract),
            runtime_mapping_failure: result
                .as_ref()
                .err()
                .and_then(|error| error.runtime_mapping_error.as_deref())
                .map(causes::backend),
            retained_owner_and_evidence: retained,
        })
    }
}
fn valid_completed(startup: &CudaStartupReceiptV1, receipt: NativeCompletedReceipt) -> bool {
    let request = receipt.context.request;
    let actual = receipt.actual;
    let hex = native_attestation::hex;
    let model = request.model;
    let encoding = request.encoding;
    actual.execution.is_some()
        && receipt.context.execution == actual.execution
        && actual.precision == PrecisionProfile::Fp32
        && actual.steps == 1
        && actual.full
        && actual.provenance == CacheProvenance::Computed
        && hex(&actual.backend.0) == startup.profile.backend_sha256
        && request.backend == actual.backend
        && request.precision == PrecisionProfile::Fp32
        && request.compute
            == (ComputeBudget {
                min_steps: 1,
                max_steps: 1,
                require_full: true,
            })
        && request.revision == CONTRACT_REVISION
        && request.request.epoch.0 == startup.process_epoch
        && request.selection.epoch == request.request.epoch
        && actual
            .execution
            .is_some_and(|execution| execution.epoch == request.request.epoch)
        && model.owner.0 == startup.profile.model.owner
        && model.slot == startup.profile.model.slot
        && model.generation.0 == startup.profile.model.generation
        && hex(&model.manifest.0) == startup.profile.model.manifest_sha256
        && encoding.owner.0 == startup.profile.encoding.owner
        && encoding.slot == startup.profile.encoding.slot
        && encoding.generation.0 == startup.profile.encoding.generation
        && hex(&encoding.manifest.0) == startup.profile.encoding.manifest_sha256
}
fn actual_cuda_observed(startup: &CudaStartupReceiptV1, report: &NativeRunReport) -> bool {
    startup.loaded
        && startup.profile.provider == "cuda"
        && startup.profile.precision == "fp32"
        && report.origin == NativeWorkerOrigin::CudaOnnx
        && startup.profile.runtime_mapping_verified
        && startup.profile.executed_cuda_nodes > 0
        && report.completed_by_runtime > 0
        && native_attestation::hex(&report.model_manifest.0)
            == startup.profile.export_manifest_sha256
        && native_attestation::hex(&report.backend.0) == startup.profile.backend_sha256
        && report
            .first_completed
            .is_some_and(|receipt| valid_completed(startup, receipt))
        && report
            .last_completed
            .is_some_and(|receipt| valid_completed(startup, receipt))
}

pub struct CudaReceiptWriter(ReceiptWriter);
impl CudaReceiptWriter {
    pub fn open(config: &NativeConfig) -> Result<Self, AttestationError> {
        if config.provider() != NativeProvider::Cuda {
            return Err(AttestationError::boundary(
                "CUDA V1 writer requires the CUDA provider",
            ));
        }
        Ok(Self(ReceiptWriter::open_named(
            config,
            STARTUP_FILE,
            TERMINATION_FILE,
        )?))
    }
    pub fn startup(&mut self, receipt: &CudaStartupReceiptV1) -> Result<(), AttestationError> {
        self.0.publish_startup(receipt)
    }
    pub fn search_config(&self, receipt: &CudaSearchReceiptV1) -> Result<(), AttestationError> {
        self.0.publish_auxiliary(SEARCH_FILE, receipt)
    }
    pub fn termination(
        &mut self,
        receipt: &CudaTerminationReceiptV1,
    ) -> Result<(), AttestationError> {
        self.0.publish_termination(receipt)
    }
}
