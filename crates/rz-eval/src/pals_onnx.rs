//! Frozen, P/C-only PALS ONNX sessions. Chess authority remains with Rules.
//!
//! A synchronous successful Run is a physical completion fence. Failed CUDA
//! execution permanently quarantines this owner and its active native values.
//! Logical cancellation belongs to the existing runtime and never frees these
//! buffers. Public memory is exact and role-neutral; role latents start fresh.
use crate::asset::{self, parse_sha256};
use crate::error::{BackendError, CauseCode, FailureKind as K, FailureStage as S};
use crate::onnx::{NativeLoadingProfile, NativeMappingObservation, OrtRuntime, Provider};
use crate::pals_model::{
    PalsModelConfig, PalsModelInput, PalsModelProfile, PalsRawOutput, PalsRole,
    PreparedPalsFullLineTensors, PreparedPalsTensors, MAX_LINE_PLY, PALS_ENCODING_SCHEMA,
    PALS_MODEL_SCHEMA, QUERY_LINE_COUNT, V_TASK_NAMES,
};
use crate::worker::{PhysicalRun, SingleWorker};
use ort::execution_providers::cuda::CuDNNConvAlgorithmSearch;
use ort::execution_providers::{
    ArenaExtendStrategy, CPUExecutionProvider, CUDAExecutionProvider, ExecutionProvider,
};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::tensor::TensorElementType;
use ort::value::{Tensor, ValueType};
use rz_contracts::{Digest, PrecisionProfile};
use rz_runtime::pals::{
    MemoryBank, MemoryBankSnapshot, MemoryKey, MemoryPin, PublicPageKey, PublicPageKind,
};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Component, Path};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::time::Instant;
mod cuda_control;
mod device_packing_admission;
mod device_packing_assets;
mod device_packing_graph;
mod device_pages_plan;
mod public_pages;
pub mod virtual_review;
mod warm;
pub use cuda_control::{
    PalsControlTransfer, PalsControlTransferKind, PalsCudaControlPolicy, PalsCudaPlacementWitness,
    PalsGraphOptimization, PalsGraphPlacement, PalsKernelWitness, VirtualReviewControlLayout,
    VirtualReviewCudaPlacementWitness,
};
pub use device_packing_admission::{
    DevicePackingAdmissionError, DevicePackingDeclaredResources, DevicePackingResourceDeclaration,
    PackingArtifactPart, PackingArtifactRegistration, PackingNativeVerification,
    PackingVerificationScope, RegisteredPackingArtifactBytes, RegisteredPackingBytePin,
    DEVICE_PACKING_DOMAIN, DEVICE_PACKING_GRAPH_FILE, DEVICE_PACKING_MANIFEST_FILE,
    DEVICE_PACKING_MAX_GRAPH_BYTES, DEVICE_PACKING_MAX_MANIFEST_BYTES,
    DEVICE_PACKING_MAX_NODE_PAYLOAD_SUM, DEVICE_PACKING_SCHEMA,
};
pub use device_packing_assets::{load_checked_fixed_packing_graph, DevicePackingAssetError};
pub use device_packing_graph::{
    CheckedFixedPackingGraph, PackingGraphBodyError, PackingGraphBodyInspection,
    PackingGraphBodyVerification, PackingGraphDataType, PackingGraphDimension,
    PackingGraphInspectionBudget, PackingGraphName, PackingGraphNodeDescriptor, PackingGraphShape,
    PackingGraphTensorDescriptor, PACKING_GRAPH_BODY_INSPECTOR_VERSION,
    PACKING_GRAPH_MAX_INSPECTION_HOST_BYTES, PACKING_GRAPH_MAX_WIRE_FIELDS,
};
#[cfg(feature = "experimental-io-binding")]
pub use device_pages_plan::native::{
    resident_cuda_minimum_metadata_host_bytes, CudaRecordPageSnapshot, CudaRecordPageStats,
};
pub use device_pages_plan::{
    device_packing_maximum_node_payload_sum, CpuOwnedPublicBacking, DevicePageDomain,
    DevicePageError, DevicePageInvocationDeclaration, DevicePageNamespace, DevicePagePayload,
    DevicePagePlan, DevicePageReservation, DevicePagesLimits, DevicePagesRegistry,
    DeviceProjectionOffset, DevicePublicBacking, DevicePublicBlock, DevicePublicBlockDescriptor,
    DEVICE_PAGE_RECORD_CAPACITY,
};
pub use public_pages::{HostRecordPagePolicy, HostRecordPageSnapshot, HostRecordPageStats};
#[cfg(feature = "experimental-io-binding")]
pub use warm::cuda::{
    cuda_private_warm_implementation_digest, PalsCudaWarmCapability, PalsCudaWarmExecutionBinding,
    PalsCudaWarmInput, PalsCudaWarmInvocationObservation, PalsCudaWarmInvocationPurpose,
    PalsCudaWarmInvocationState, PalsCudaWarmObservationHandle, PalsCudaWarmObservationSnapshot,
    PalsCudaWarmObservationStatus, PalsCudaWarmOwnerObservation, PalsCudaWarmRuntimeLimits,
    PRIVATE_CUDA_QUERY_SEMANTICS, PRIVATE_CUDA_WARM_EXECUTION_DOMAIN,
    PRIVATE_CUDA_WARM_GRAPH_SEMANTICS, PRIVATE_CUDA_WARM_MEASUREMENT_CONTRACT,
    PRIVATE_CUDA_WARM_SCHEMA,
};
pub use warm::{
    PalsWarmCapability, PalsWarmInput, FROZEN_QUERY_SEMANTICS_V1, PRIVATE_WARM_GRAPH_SEMANTICS,
    PRIVATE_WARM_SCHEMA,
};

/// Implementation provenance only. Changing this source digest does not
/// change the public input, model epoch or encoding semantic namespace.
pub fn host_record_page_implementation_digest() -> [u8; 32] {
    use sha2::{Digest as _, Sha256};
    static DIGEST: std::sync::OnceLock<[u8; 32]> = std::sync::OnceLock::new();
    *DIGEST.get_or_init(|| {
        let mut digest = Sha256::new();
        digest.update(b"rz-pals-host-record-pages-implementation/1");
        for source in [
            include_bytes!("pals_onnx/public_pages.rs").as_slice(),
            include_bytes!("pals_model.rs").as_slice(),
            include_bytes!("pals_model/full_line.rs").as_slice(),
            include_bytes!("pals_onnx.rs").as_slice(),
            include_bytes!("../../rz-runtime/src/pals.rs").as_slice(),
        ] {
            digest.update((source.len() as u64).to_le_bytes());
            digest.update(source);
        }
        digest.finalize().into()
    })
}

/// Explicit resident implementation provenance, separate from model/input
/// semantics and from native Run, placement or physical-completion evidence.
#[cfg(feature = "experimental-io-binding")]
pub fn cuda_record_pages_implementation_digest() -> [u8; 32] {
    use sha2::{Digest as _, Sha256};
    static DIGEST: std::sync::OnceLock<[u8; 32]> = std::sync::OnceLock::new();
    *DIGEST.get_or_init(|| {
        let mut digest = Sha256::new();
        digest.update(b"rz-pals-cuda-record-pages-implementation/1");
        for source in [
            include_bytes!("pals_onnx/device_pages_plan/native.rs").as_slice(),
            include_bytes!("pals_onnx/device_pages_plan.rs").as_slice(),
            include_bytes!("pals_onnx/device_packing_admission.rs").as_slice(),
            include_bytes!("pals_onnx/device_packing_graph.rs").as_slice(),
            include_bytes!("pals_onnx/device_packing_assets.rs").as_slice(),
            include_bytes!("pals_onnx/cuda_control.rs").as_slice(),
            include_bytes!("pals_device_resources.rs").as_slice(),
            include_bytes!("pals_model.rs").as_slice(),
            include_bytes!("pals_model/full_line.rs").as_slice(),
            include_bytes!("pals_onnx.rs").as_slice(),
            include_bytes!("../../rz-runtime/src/pals.rs").as_slice(),
            include_bytes!("../../rz-native-loader/src/ort_binding.rs").as_slice(),
        ] {
            digest.update((source.len() as u64).to_le_bytes());
            digest.update(source);
        }
        digest.finalize().into()
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HostRecordPageObservationBoundary {
    BeforeWorker,
    Evaluate,
    NewGame,
    SnapshotStats,
    #[cfg(feature = "experimental-io-binding")]
    SnapshotCudaRecordPages,
    VerifyRuntime,
    VerifyCudaPlacement,
    ObserveRuntimeMappings,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HostRecordPageObservationOutcome {
    BeforeWorker,
    ReturnedOk,
    ReturnedError,
    PhysicalCompletionUnknown,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HostRecordPageObservationStatus {
    Available,
    Unavailable,
    Contended,
    Poisoned,
    OrdinalExhausted,
}
/// A command-return observation, never a new physical completion fence.
#[derive(Clone, Debug)]
pub struct HostRecordPageObservation {
    pub command_ordinal: u64,
    pub boundary: HostRecordPageObservationBoundary,
    pub outcome: HostRecordPageObservationOutcome,
    pub snapshot: HostRecordPageSnapshot,
}
#[derive(Clone, Debug)]
pub struct HostRecordPageObservationSnapshot {
    pub status: HostRecordPageObservationStatus,
    pub attempted_command_ordinal: u64,
    /// May be the last earlier observation on failure; its ordinal and status
    /// must be preserved instead of reporting invented current/zero counters.
    pub latest: Option<HostRecordPageObservation>,
}
struct HostRecordPageObservationLedger {
    latest: Mutex<Option<HostRecordPageObservation>>,
    ordinal: AtomicU64,
    contended: AtomicBool,
    poisoned: AtomicBool,
    ordinal_exhausted: AtomicBool,
}
/// Bounded metadata only. This handle owns no session, native value, page pin
/// or worker. The backend remains with its physical owner through shutdown or
/// quarantine. try_lock keeps observations from delaying the native worker.
#[derive(Clone)]
pub struct HostRecordPageObservationHandle(Arc<HostRecordPageObservationLedger>);
pub type HostRecordPageObservedWorker = (
    SingleWorker<PalsNativeCommand, Result<PalsNativeResult, BackendError>>,
    HostRecordPageObservationHandle,
);
impl HostRecordPageObservationHandle {
    fn new(initial: HostRecordPageSnapshot) -> Self {
        Self(Arc::new(HostRecordPageObservationLedger {
            latest: Mutex::new(Some(HostRecordPageObservation {
                command_ordinal: 0,
                boundary: HostRecordPageObservationBoundary::BeforeWorker,
                outcome: HostRecordPageObservationOutcome::BeforeWorker,
                snapshot: initial,
            })),
            ordinal: AtomicU64::new(0),
            contended: AtomicBool::new(false),
            poisoned: AtomicBool::new(false),
            ordinal_exhausted: AtomicBool::new(false),
        }))
    }
    fn record(
        &self,
        boundary: HostRecordPageObservationBoundary,
        outcome: HostRecordPageObservationOutcome,
        snapshot: Option<HostRecordPageSnapshot>,
    ) {
        let ordinal = match self
            .0
            .ordinal
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
        {
            Ok(previous) => previous + 1,
            Err(_) => {
                self.0.ordinal_exhausted.store(true, Ordering::Release);
                return;
            }
        };
        match self.0.latest.try_lock() {
            Ok(mut latest) => {
                *latest = snapshot.map(|snapshot| HostRecordPageObservation {
                    command_ordinal: ordinal,
                    boundary,
                    outcome,
                    snapshot,
                });
            }
            Err(TryLockError::WouldBlock) => self.0.contended.store(true, Ordering::Release),
            Err(TryLockError::Poisoned(_)) => self.0.poisoned.store(true, Ordering::Release),
        }
    }
    pub fn snapshot(&self) -> HostRecordPageObservationSnapshot {
        let ordinal = self.0.ordinal.load(Ordering::Acquire);
        let (mut status, latest) = match self.0.latest.try_lock() {
            Ok(latest) => (HostRecordPageObservationStatus::Available, latest.clone()),
            Err(TryLockError::WouldBlock) => (HostRecordPageObservationStatus::Contended, None),
            Err(TryLockError::Poisoned(error)) => (
                HostRecordPageObservationStatus::Poisoned,
                error.into_inner().clone(),
            ),
        };
        if self.0.ordinal_exhausted.load(Ordering::Acquire) {
            status = HostRecordPageObservationStatus::OrdinalExhausted;
        } else if self.0.poisoned.load(Ordering::Acquire) {
            status = HostRecordPageObservationStatus::Poisoned;
        } else if self.0.contended.load(Ordering::Acquire)
            || latest
                .as_ref()
                .is_some_and(|entry| entry.command_ordinal != ordinal)
        {
            status = HostRecordPageObservationStatus::Contended;
        } else if latest.is_none() {
            status = HostRecordPageObservationStatus::Unavailable;
        }
        HostRecordPageObservationSnapshot {
            status,
            attempted_command_ordinal: ordinal,
            latest,
        }
    }
}

const MAX_MANIFEST_BYTES: usize = 128 * 1024;
const MAX_GRAPH_BYTES: usize = 256 * 1024 * 1024;
const MAX_MODEL_BYTES: u64 = 512 * 1024 * 1024;
const MAX_STARTUP_STAGE_EVENTS: usize = 64;
const MAX_STARTUP_STAGE_REQUESTS: u8 = 2;
const MAX_STARTUP_INPUT_RECORDS: usize = 128;

/// Copies of input metadata used by an existing startup invocation. Neither a
/// prepared value nor a returned stage event is a physical-completion witness.
#[derive(Clone, Debug, Serialize)]
pub struct PalsStartupTensorObservation {
    pub element_type: &'static str,
    pub shape: Vec<i64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PalsStartupInputCompletionBinding {
    pub command_ordinal: usize,
    pub backend_owner_id: u64,
    pub retained_input_key: [u8; 32],
    pub physical_completion_confirmed: bool,
    pub raw_decode_succeeded: bool,
    pub inventory_sha256: [u8; 32],
    pub public_profile_sha256: [u8; 32],
    pub shared_pc_profile_sha256: [u8; 32],
}

#[derive(Clone, Debug, Serialize)]
pub struct PalsStartupInputObservation {
    pub request_ordinal: u8,
    pub role: PalsRole,
    pub backend_owner_id: Option<u64>,
    pub input_key: [u8; 32],
    pub prepared_input_key: [u8; 32],
    pub public_memory_key: [u8; 32],
    pub logical_records: usize,
    pub prepared_records: usize,
    pub record_mask: Option<Vec<bool>>,
    pub board: Option<PalsStartupTensorObservation>,
    pub records: Option<PalsStartupTensorObservation>,
    pub record_mask_tensor: Option<PalsStartupTensorObservation>,
    pub memory_key: Option<PalsStartupTensorObservation>,
    pub memory_value: Option<PalsStartupTensorObservation>,
    pub memory_mask: Option<PalsStartupTensorObservation>,
    pub public_cache_hit: Option<bool>,
    pub public_run_returned_ok: Option<bool>,
    pub private_run_returned_ok: Option<bool>,
    pub role_evaluation_returned_ok: Option<bool>,
    /// Filled only by Native after matching its retained command, Ready and
    /// raw decode to this owner/input and the same-worker profile ACK.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_binding: Option<PalsStartupInputCompletionBinding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_binding_error: Option<&'static str>,
}

#[derive(Clone, Copy)]
struct StartupTensorMetadata {
    element_type: &'static str,
    rank: usize,
    shape: [i64; 4],
}
impl StartupTensorMetadata {
    #[cfg(feature = "experimental-io-binding")]
    fn observe<T: ort::tensor::PrimitiveTensorElementType + std::fmt::Debug>(
        value: &Tensor<T>,
    ) -> Option<Self> {
        let ValueType::Tensor { ty, shape, .. } = value.dtype() else {
            return None;
        };
        let element_type = match ty {
            TensorElementType::Float32 => "float32",
            TensorElementType::Int64 => "int64",
            TensorElementType::Bool => "bool",
            _ => return None,
        };
        if shape.len() > 4 {
            return None;
        }
        let mut observed = [0; 4];
        observed[..shape.len()].copy_from_slice(shape.as_ref());
        Some(Self {
            element_type,
            rank: shape.len(),
            shape: observed,
        })
    }
    fn snapshot(self) -> PalsStartupTensorObservation {
        PalsStartupTensorObservation {
            element_type: self.element_type,
            shape: self.shape[..self.rank].to_vec(),
        }
    }
}

#[derive(Clone, Copy)]
struct StartupInputMetadata {
    request_ordinal: u8,
    role: PalsRole,
    backend_owner_id: Option<u64>,
    input_key: [u8; 32],
    prepared_input_key: [u8; 32],
    public_memory_key: [u8; 32],
    logical_records: usize,
    prepared_records: usize,
    record_mask: Option<[bool; MAX_STARTUP_INPUT_RECORDS]>,
    board: Option<StartupTensorMetadata>,
    records: Option<StartupTensorMetadata>,
    record_mask_tensor: Option<StartupTensorMetadata>,
    memory_key: Option<StartupTensorMetadata>,
    memory_value: Option<StartupTensorMetadata>,
    memory_mask: Option<StartupTensorMetadata>,
    public_cache_hit: Option<bool>,
    public_run_returned_ok: Option<bool>,
    private_run_returned_ok: Option<bool>,
    role_evaluation_returned_ok: Option<bool>,
}
#[cfg(feature = "experimental-io-binding")]
struct StartupPreparedInputIdentity {
    input_key: [u8; 32],
    public_memory_key: [u8; 32],
    records: usize,
}
impl StartupInputMetadata {
    fn snapshot(self) -> PalsStartupInputObservation {
        PalsStartupInputObservation {
            request_ordinal: self.request_ordinal,
            role: self.role,
            backend_owner_id: self.backend_owner_id,
            input_key: self.input_key,
            prepared_input_key: self.prepared_input_key,
            public_memory_key: self.public_memory_key,
            logical_records: self.logical_records,
            prepared_records: self.prepared_records,
            record_mask: self
                .record_mask
                .filter(|_| self.prepared_records <= MAX_STARTUP_INPUT_RECORDS)
                .map(|mask| mask[..self.prepared_records].to_vec()),
            board: self.board.map(StartupTensorMetadata::snapshot),
            records: self.records.map(StartupTensorMetadata::snapshot),
            record_mask_tensor: self.record_mask_tensor.map(StartupTensorMetadata::snapshot),
            memory_key: self.memory_key.map(StartupTensorMetadata::snapshot),
            memory_value: self.memory_value.map(StartupTensorMetadata::snapshot),
            memory_mask: self.memory_mask.map(StartupTensorMetadata::snapshot),
            public_cache_hit: self.public_cache_hit,
            public_run_returned_ok: self.public_run_returned_ok,
            private_run_returned_ok: self.private_run_returned_ok,
            role_evaluation_returned_ok: self.role_evaluation_returned_ok,
            completion_binding: None,
            completion_binding_error: None,
        }
    }
}

/// Diagnostic stages only: a returned event is not an independent physical
/// fence, native-input count, readiness proof or permission to release owners.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PalsStartupBackendStage {
    RoleEvaluation,
    InputPreparation,
    PublicCacheHit,
    PublicRun,
    PublicOutputPreparation,
    PrivateRun,
    PrivateOutputValidation,
    FirstRuntimeOriginAudit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PalsStartupStageBoundary {
    Entered,
    ReturnedOk,
    ReturnedError,
    Observed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct PalsStartupStageEvent {
    /// Ordinal of the actual admitted startup Evaluate, not a NN input count.
    pub request_ordinal: u8,
    pub role: PalsRole,
    pub stage: PalsStartupBackendStage,
    pub boundary: PalsStartupStageBoundary,
    pub elapsed_ns: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PalsStartupSnapshotStatus {
    Available,
    Contended,
    Poisoned,
}

#[derive(Clone, Debug, Serialize)]
pub struct PalsStartupStageSnapshot {
    pub schema: &'static str,
    /// Exactly the Instant supplied by the caller to start(), also suitable for
    /// the UCI command timeline. No wall-clock or inferred loading timestamp.
    pub clock_scope: &'static str,
    pub snapshot_status: PalsStartupSnapshotStatus,
    /// None means the diagnostic lock prevented observing the ledger origin.
    pub capture_started: Option<bool>,
    pub capture_closed: bool,
    pub snapshot_elapsed_ns: Option<u64>,
    pub captured_requests: u8,
    pub max_requests: u8,
    pub max_events: usize,
    pub overflow: bool,
    pub recording_contended: bool,
    pub recording_poisoned: bool,
    pub events: Vec<PalsStartupStageEvent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_observations: Option<Vec<PalsStartupInputObservation>>,
}

struct StartupStageLedger {
    origin: Option<Instant>,
    events: [Option<PalsStartupStageEvent>; MAX_STARTUP_STAGE_EVENTS],
    len: usize,
    inputs: [Option<StartupInputMetadata>; MAX_STARTUP_STAGE_REQUESTS as usize],
}

struct StartupStageShared {
    start_attempted: AtomicBool,
    active: AtomicBool,
    input_observations_enabled: AtomicBool,
    requests: AtomicU8,
    overflow: AtomicBool,
    recording_contended: AtomicBool,
    recording_poisoned: AtomicBool,
    ledger: Mutex<StartupStageLedger>,
}

/// An opt-in metadata-only handle, retained outside the physical worker. It
/// owns no session, tensor, file descriptor or physical lease. Both recording
/// and snapshots use try_lock, so missing diagnostic evidence cannot change a
/// native result or turn a still-live invocation into physical completion.
#[derive(Clone)]
pub struct PalsStartupStageProbe {
    shared: Arc<StartupStageShared>,
}

impl PalsStartupStageProbe {
    fn new() -> Self {
        Self {
            shared: Arc::new(StartupStageShared {
                start_attempted: AtomicBool::new(false),
                active: AtomicBool::new(false),
                input_observations_enabled: AtomicBool::new(false),
                requests: AtomicU8::new(0),
                overflow: AtomicBool::new(false),
                recording_contended: AtomicBool::new(false),
                recording_poisoned: AtomicBool::new(false),
                ledger: Mutex::new(StartupStageLedger {
                    origin: None,
                    events: [None; MAX_STARTUP_STAGE_EVENTS],
                    len: 0,
                    inputs: [None; MAX_STARTUP_STAGE_REQUESTS as usize],
                }),
            }),
        }
    }

    /// Arm once immediately before the first startup Evaluate, using the same
    /// monotonic origin as the caller's command observations. Never a retry.
    pub fn start(&self, origin: Instant) -> bool {
        self.start_inner(origin, false)
    }
    /// Explicit input diagnostics only; default startup wire and invocations
    /// keep the metadata channel disabled. This neither admits nor runs work.
    pub fn start_with_input_observations(&self, origin: Instant) -> bool {
        self.start_inner(origin, true)
    }
    fn start_inner(&self, origin: Instant, inputs: bool) -> bool {
        if self
            .shared
            .start_attempted
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }
        match self.shared.ledger.try_lock() {
            Ok(mut ledger) => {
                ledger.origin = Some(origin);
                self.shared
                    .input_observations_enabled
                    .store(inputs, Ordering::Release);
                self.shared.active.store(true, Ordering::Release);
                true
            }
            Err(TryLockError::WouldBlock) => {
                self.shared
                    .recording_contended
                    .store(true, Ordering::Release);
                false
            }
            Err(TryLockError::Poisoned(_)) => {
                self.shared
                    .recording_poisoned
                    .store(true, Ordering::Release);
                false
            }
        }
    }

    fn closed(&self) -> bool {
        self.shared.start_attempted.load(Ordering::Acquire)
            && !self.shared.active.load(Ordering::Acquire)
    }

    fn begin_role(&self, role: PalsRole) -> Option<StartupRoleTrace> {
        if !self.shared.active.load(Ordering::Acquire) {
            return None;
        }
        let ordinal = self.shared.requests.fetch_add(1, Ordering::AcqRel) + 1;
        if ordinal > MAX_STARTUP_STAGE_REQUESTS {
            self.shared.active.store(false, Ordering::Release);
            return None;
        }
        let trace = StartupRoleTrace {
            probe: self.clone(),
            ordinal,
            role,
        };
        trace.record(
            PalsStartupBackendStage::RoleEvaluation,
            PalsStartupStageBoundary::Entered,
        );
        Some(trace)
    }

    /// Freeze a partial or successful startup observation without waiting for
    /// the worker. The stop flag does not cancel native work or release pins.
    pub fn snapshot_and_stop(&self) -> PalsStartupStageSnapshot {
        // Closing a dormant diagnostic is final too; it must not arm a later
        // capture after the caller has already published its frozen view.
        self.shared.start_attempted.store(true, Ordering::Release);
        self.shared.active.store(false, Ordering::Release);
        let mut snapshot = PalsStartupStageSnapshot {
            schema: "rovezero.pals-startup-backend-stages.v1",
            clock_scope: "caller_supplied_monotonic_startup_origin",
            snapshot_status: PalsStartupSnapshotStatus::Available,
            capture_started: None,
            capture_closed: true,
            snapshot_elapsed_ns: None,
            captured_requests: self
                .shared
                .requests
                .load(Ordering::Acquire)
                .min(MAX_STARTUP_STAGE_REQUESTS),
            max_requests: MAX_STARTUP_STAGE_REQUESTS,
            max_events: MAX_STARTUP_STAGE_EVENTS,
            overflow: self.shared.overflow.load(Ordering::Acquire),
            recording_contended: self.shared.recording_contended.load(Ordering::Acquire),
            recording_poisoned: self.shared.recording_poisoned.load(Ordering::Acquire),
            events: Vec::new(),
            input_observations: self
                .shared
                .input_observations_enabled
                .load(Ordering::Acquire)
                .then(Vec::new),
        };
        // Copy the fixed metadata array while locked; allocate the serialized
        // view only after releasing it. No IO or native observation here.
        let copied = match self.shared.ledger.try_lock() {
            Ok(ledger) => Some((ledger.origin, ledger.events, ledger.len, ledger.inputs)),
            Err(TryLockError::WouldBlock) => {
                snapshot.snapshot_status = PalsStartupSnapshotStatus::Contended;
                None
            }
            Err(TryLockError::Poisoned(_)) => {
                snapshot.snapshot_status = PalsStartupSnapshotStatus::Poisoned;
                None
            }
        };
        if let Some((origin, events, len, inputs)) = copied {
            snapshot.capture_started = Some(origin.is_some());
            snapshot.snapshot_elapsed_ns = origin.and_then(elapsed_ns);
            snapshot.events = events.into_iter().take(len).flatten().collect();
            if snapshot.input_observations.is_some() {
                snapshot.input_observations = Some(
                    inputs
                        .into_iter()
                        .flatten()
                        .map(StartupInputMetadata::snapshot)
                        .collect(),
                );
            }
        }
        snapshot
    }
}

struct StartupRoleTrace {
    probe: PalsStartupStageProbe,
    ordinal: u8,
    role: PalsRole,
}

impl StartupRoleTrace {
    fn inputs_enabled(&self) -> bool {
        self.probe.shared.active.load(Ordering::Acquire)
            && self
                .probe
                .shared
                .input_observations_enabled
                .load(Ordering::Acquire)
    }
    fn update_input(&self, update: impl FnOnce(&mut Option<StartupInputMetadata>)) {
        if !self.inputs_enabled() {
            return;
        }
        match self.probe.shared.ledger.try_lock() {
            Ok(mut ledger) => {
                if self.inputs_enabled() {
                    update(&mut ledger.inputs[usize::from(self.ordinal - 1)]);
                }
            }
            Err(TryLockError::WouldBlock) => self
                .probe
                .shared
                .recording_contended
                .store(true, Ordering::Release),
            Err(TryLockError::Poisoned(_)) => self
                .probe
                .shared
                .recording_poisoned
                .store(true, Ordering::Release),
        }
    }
    #[cfg(feature = "experimental-io-binding")]
    fn observe_prepared_input(
        &self,
        owner: Option<u64>,
        input_key: [u8; 32],
        input: &PalsModelInput,
        prepared: StartupPreparedInputIdentity,
        active: &ActiveInputs,
    ) {
        if !self.inputs_enabled() {
            return;
        }
        // record_mask is the owned CPU input created from the actual prepared
        // vector. No device value, K/V payload or additional tensor is copied.
        let record_mask = active
            .record_mask
            .try_extract_tensor::<bool>()
            .ok()
            .filter(|(_, values)| {
                values.len() == prepared.records && values.len() <= MAX_STARTUP_INPUT_RECORDS
            })
            .map(|(_, values)| {
                let mut copied = [false; MAX_STARTUP_INPUT_RECORDS];
                copied[..values.len()].copy_from_slice(values);
                copied
            });
        let observed = StartupInputMetadata {
            request_ordinal: self.ordinal,
            role: input.role,
            backend_owner_id: owner,
            input_key,
            prepared_input_key: prepared.input_key,
            public_memory_key: prepared.public_memory_key,
            logical_records: input.records.len(),
            prepared_records: prepared.records,
            record_mask,
            board: StartupTensorMetadata::observe(&active.board),
            records: StartupTensorMetadata::observe(&active.records),
            record_mask_tensor: StartupTensorMetadata::observe(&active.record_mask),
            memory_key: None,
            memory_value: None,
            memory_mask: None,
            public_cache_hit: None,
            public_run_returned_ok: None,
            private_run_returned_ok: None,
            role_evaluation_returned_ok: None,
        };
        self.update_input(|slot| {
            if slot.is_none() {
                *slot = Some(observed);
            }
        });
    }
    #[cfg(feature = "experimental-io-binding")]
    fn observe_memory(&self, memory: &device::DeviceMemory, cached: bool) {
        if !self.inputs_enabled() {
            return;
        }
        let key = StartupTensorMetadata::observe(&memory.memory_key);
        let value = StartupTensorMetadata::observe(&memory.memory_value);
        let mask = StartupTensorMetadata::observe(&memory.mask);
        self.update_input(|slot| {
            if let Some(input) = slot {
                input.memory_key = key;
                input.memory_value = value;
                input.memory_mask = mask;
                input.public_cache_hit = Some(cached);
            }
        });
    }
    fn record(&self, stage: PalsStartupBackendStage, boundary: PalsStartupStageBoundary) {
        if !self.probe.shared.active.load(Ordering::Acquire) {
            return;
        }
        match self.probe.shared.ledger.try_lock() {
            Ok(mut ledger) => {
                if !self.probe.shared.active.load(Ordering::Acquire) {
                    return;
                }
                if ledger.len == MAX_STARTUP_STAGE_EVENTS {
                    self.probe.shared.overflow.store(true, Ordering::Release);
                    return;
                }
                let Some(elapsed_ns) = ledger.origin.and_then(elapsed_ns) else {
                    return;
                };
                let index = ledger.len;
                ledger.events[index] = Some(PalsStartupStageEvent {
                    request_ordinal: self.ordinal,
                    role: self.role,
                    stage,
                    boundary,
                    elapsed_ns,
                });
                ledger.len += 1;
            }
            Err(TryLockError::WouldBlock) => self
                .probe
                .shared
                .recording_contended
                .store(true, Ordering::Release),
            Err(TryLockError::Poisoned(_)) => self
                .probe
                .shared
                .recording_poisoned
                .store(true, Ordering::Release),
        }
    }
    fn returned(&self, stage: PalsStartupBackendStage, success: bool) {
        self.update_input(|slot| {
            if let Some(input) = slot {
                match stage {
                    PalsStartupBackendStage::PublicRun => {
                        input.public_run_returned_ok = Some(success)
                    }
                    PalsStartupBackendStage::PrivateRun => {
                        input.private_run_returned_ok = Some(success)
                    }
                    PalsStartupBackendStage::RoleEvaluation => {
                        input.role_evaluation_returned_ok = Some(success)
                    }
                    _ => {}
                }
            }
        });
        self.record(
            stage,
            if success {
                PalsStartupStageBoundary::ReturnedOk
            } else {
                PalsStartupStageBoundary::ReturnedError
            },
        );
    }
    fn finish(&self, success: bool) {
        self.returned(PalsStartupBackendStage::RoleEvaluation, success);
        if self.ordinal == MAX_STARTUP_STAGE_REQUESTS {
            self.probe.shared.active.store(false, Ordering::Release);
        }
    }
}

fn elapsed_ns(origin: Instant) -> Option<u64> {
    u64::try_from(Instant::now().saturating_duration_since(origin).as_nanos()).ok()
}

fn startup_enter(trace: Option<&StartupRoleTrace>, stage: PalsStartupBackendStage) {
    if let Some(trace) = trace {
        trace.record(stage, PalsStartupStageBoundary::Entered);
    }
}

fn startup_return(trace: Option<&StartupRoleTrace>, stage: PalsStartupBackendStage, success: bool) {
    if let Some(trace) = trace {
        trace.returned(stage, success);
    }
}

/// Explicit diagnostic request only; neither variant observes a selected plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PalsCuDnnConvSearch {
    /// Preserve the CUDA EP's implicit EXHAUSTIVE option: no setter is called.
    LegacyImplicitExhaustive,
    /// A separately registered host-public CUDA control diagnostic variable.
    ExperimentalDefault,
    /// Separate opt-in device-public CUDA Warm diagnostic; never host admission.
    ExperimentalDefaultDeviceWarm,
}

/// Loaded execution configuration only. No selected plan or native exit proof.
#[derive(Clone, Debug, Serialize)]
pub struct PalsCuDnnExecutionDiagnostic {
    pub descriptor: serde_json::Value,
    pub descriptor_sha256: [u8; 32],
    pub descriptor_encoding: &'static str,
    pub actual_cudnn_plan: &'static str,
    pub numerical_note: &'static str,
    pub tf32_actual: &'static str,
    pub tf32_retry: &'static str,
    pub production_default_changed: bool,
    pub warm_success_claim: bool,
    pub pilot_success_claim: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct PalsOnnxConfig {
    pub provider: Provider,
    pub intra_threads: usize,
    /// The first exact cache has one bounded public-memory entry. Set false for
    /// independent fresh-versus-cached correctness comparisons.
    pub cache_public_memory: bool,
    /// Device-resident public K/V has a separate explicit admission. The
    /// feature only compiles its code; selecting CPU never enables it.
    pub device_public_memory: bool,
    pub cudnn_conv_search: PalsCuDnnConvSearch,
}
impl PalsOnnxConfig {
    pub fn cpu() -> Self {
        Self {
            provider: Provider::Cpu,
            intra_threads: 2,
            cache_public_memory: true,
            device_public_memory: false,
            cudnn_conv_search: PalsCuDnnConvSearch::LegacyImplicitExhaustive,
        }
    }
    fn validate(&self) -> Result<(), BackendError> {
        if !(1..=2).contains(&self.intra_threads) {
            return Err(fail(
                K::InvalidInput,
                S::Admission,
                "PALS CPU thread declaration must be 1..=2",
            ));
        }
        if let Provider::Cuda {
            device_id,
            arena_bytes,
        } = self.provider
        {
            if device_id < 0 || arena_bytes == 0 || arena_bytes > 6 * 1024 * 1024 * 1024 {
                return Err(fail(
                    K::InvalidInput,
                    S::Admission,
                    "PALS CUDA device/memory declaration is invalid",
                ));
            }
        }
        if self.device_public_memory
            && (!cfg!(feature = "experimental-io-binding")
                || !matches!(self.provider, Provider::Cuda { .. }))
        {
            return Err(fail(
                K::BackendUnavailable,
                S::Admission,
                "device PALS memory requires explicit CUDA and the I/O binding feature",
            ));
        }
        if self.cudnn_conv_search == PalsCuDnnConvSearch::ExperimentalDefault
            && (!matches!(self.provider, Provider::Cuda { .. }) || self.device_public_memory)
        {
            return Err(fail(
                K::BackendUnavailable,
                S::Admission,
                "experimental cuDNN DEFAULT requires CUDA with host public memory",
            ));
        }
        if self.cudnn_conv_search == PalsCuDnnConvSearch::ExperimentalDefaultDeviceWarm
            && (!matches!(self.provider, Provider::Cuda { .. })
                || !self.device_public_memory
                || !cfg!(feature = "experimental-io-binding"))
        {
            return Err(fail(
                K::BackendUnavailable,
                S::Admission,
                "DEFAULT device-Warm requires CUDA device public memory and I/O binding",
            ));
        }
        Ok(())
    }
    /// Pure preflight for callers before runtime/model I/O. The loader repeats
    /// this with the actual retained runtime profile before reading the model.
    pub fn validate_execution_selection(
        self,
        native_loading: Option<NativeLoadingProfile>,
        explicit_control: bool,
    ) -> Result<(), BackendError> {
        self.validate()?;
        if self.cudnn_conv_search == PalsCuDnnConvSearch::ExperimentalDefaultDeviceWarm {
            return Err(fail(
                K::BackendUnavailable,
                S::Admission,
                "DEFAULT device-Warm requires the separately checked CUDA Warm loader",
            ));
        }
        if self.cudnn_conv_search == PalsCuDnnConvSearch::ExperimentalDefault
            && (!matches!(self.provider, Provider::Cuda { .. })
                || self.device_public_memory
                || !explicit_control
                || native_loading != Some(NativeLoadingProfile::EagerCuda12Cudnn9V1))
        {
            return Err(fail(
                K::BackendUnavailable,
                S::Admission,
                "experimental cuDNN DEFAULT requires explicit eager CUDA control with host public memory",
            ));
        }
        validate_native_loading_path(native_loading, self, explicit_control)
    }
    /// Separate pure selection fence. The Warm loader also checks its actual
    /// loaded export/capability; this request alone grants no Warm authority.
    pub fn validate_cuda_warm_execution_selection(
        self,
        native_loading: Option<NativeLoadingProfile>,
        explicit_control: bool,
    ) -> Result<(), BackendError> {
        self.validate()?;
        if self.cudnn_conv_search == PalsCuDnnConvSearch::ExperimentalDefault {
            return Err(fail(
                K::BackendUnavailable,
                S::Admission,
                "host DEFAULT cannot select the CUDA Warm loader",
            ));
        }
        if self.cudnn_conv_search == PalsCuDnnConvSearch::ExperimentalDefaultDeviceWarm
            && (!explicit_control
                || native_loading != Some(NativeLoadingProfile::EagerCuda12Cudnn9V1))
        {
            return Err(fail(
                K::BackendUnavailable,
                S::Admission,
                "DEFAULT device-Warm requires explicit eager inventory control",
            ));
        }
        validate_native_loading_path(native_loading, self, explicit_control)
    }
}

fn validate_native_loading_path(
    profile: Option<NativeLoadingProfile>,
    config: PalsOnnxConfig,
    explicit_control: bool,
) -> Result<(), BackendError> {
    if profile == Some(NativeLoadingProfile::CuDnnShimLazyV1)
        && (!explicit_control
            || !matches!(config.provider, Provider::Cuda { .. })
            || config.device_public_memory)
    {
        return Err(fail(
            K::BackendUnavailable,
            S::Admission,
            "experimental PALS shim loading requires explicit CUDA control with host public memory",
        ));
    }
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExportManifest {
    schema: String,
    #[serde(default)]
    model_semantics: Option<String>,
    #[serde(default)]
    model_profile: Option<PalsModelProfile>,
    #[serde(default)]
    encoding_schema: Option<String>,
    #[serde(default)]
    layout: Option<String>,
    #[serde(default)]
    layout_revision: Option<u32>,
    #[serde(default)]
    role_batching: Option<String>,
    #[serde(default)]
    rules_input_profile: Option<String>,
    #[serde(default)]
    rules_input_semantic_sha256: Option<String>,
    #[serde(default)]
    rules_encoder_source_sha256: Option<String>,
    #[serde(default)]
    rules_profile_descriptor_sha256: Option<String>,
    #[serde(default)]
    rules_profile_canonical_sha256: Option<String>,
    #[serde(default)]
    rules_input_declaration: Option<serde_json::Value>,
    #[serde(default)]
    learned_input_compatibility: Option<String>,
    #[serde(default)]
    reader_initializer_bank: Option<ReaderInitializerBank>,
    config: PalsModelConfig,
    checkpoint_sha256: String,
    trained: bool,
    training_steps: u64,
    roles: Vec<String>,
    validator_present: bool,
    graphs: Vec<GraphManifest>,
    runtime: RuntimeManifest,
    candidate_promotion: BTreeMap<String, String>,
    wdl_perspective: String,
    task_names: Vec<String>,
    numeric_status: String,
    cuda_status: String,
}
/// Serialized ONNX initializer ownership only. Native optimization can still
/// allocate transformed/prepacked copies and is not measured by this audit.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReaderInitializerBank {
    pub scope: String,
    pub layout_revision: u32,
    pub shared_parameters: Vec<SharedParameter>,
    pub unique_shared_parameters: usize,
    pub shared_weight_bytes: u64,
    pub if_routes: usize,
    pub branch_local_initializers: usize,
    pub branch_operations: Vec<BranchOperations>,
    pub inactive_private_execution: String,
    pub ort_prepack_copies: String,
    pub device_residency_sharing: String,
}
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SharedParameter {
    pub parameter: String,
    pub initializer: String,
    pub bytes: u64,
    pub sha256: String,
    pub source_parameter_sha256: String,
    pub shape: Vec<usize>,
}
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BranchOperations {
    pub route: String,
    pub branch: String,
    pub nodes: usize,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Layout {
    SeparatePc,
    SharedPcIf,
}
impl ExportManifest {
    fn resolved_layout(&self) -> Result<Layout, BackendError> {
        let profile = self.config.profile;
        if !profile.is_legacy()
            && (self.model_semantics.as_deref() != Some(profile.model_semantics())
                || self.model_profile != Some(profile)
                || self.encoding_schema.as_deref() != Some(profile.encoding_schema()))
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "PALS V2 model/profile/encoding declaration differs",
            ));
        }
        if profile.is_legacy()
            && (self.model_profile.is_some_and(|p| !p.is_legacy())
                || self
                    .encoding_schema
                    .as_deref()
                    .is_some_and(|s| s != PALS_ENCODING_SCHEMA))
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "legacy PALS export cannot declare V2 input semantics",
            ));
        }
        match (
            self.schema.as_str(),
            self.layout.as_deref(),
            self.model_semantics.as_deref(),
        ) {
            (PALS_MODEL_SCHEMA, None | Some("separate_pc"), semantics)
                if (semantics == Some(profile.model_semantics())
                    || (semantics.is_none() && profile.is_legacy()))
                    && self.reader_initializer_bank.is_none() =>
            {
                Ok(Layout::SeparatePc)
            }
            ("rovezero.pals-model.v2", Some("shared_pc_if"), Some(semantics))
                if semantics == profile.model_semantics()
                    && self.layout_revision == Some(1)
                    && self.role_batching.as_deref()
                        == Some("one_scalar_role_per_physical_batch") =>
            {
                Ok(Layout::SharedPcIf)
            }
            _ => Err(fail(
                K::UnsupportedModel,
                S::Asset,
                "unsupported explicit PALS graph layout/semantics",
            )),
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeManifest {
    onnxruntime: String,
    rust_ort: String,
    compatibility: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphManifest {
    file: String,
    sha256: String,
    role: String,
    inputs: Vec<TensorManifest>,
    outputs: Vec<TensorManifest>,
    opset: u32,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorManifest {
    name: String,
    dtype: String,
    shape: Vec<serde_json::Value>,
}

#[derive(Clone, Copy)]
struct Interface {
    name: &'static str,
    dtype: TensorElementType,
    shape: &'static [i64],
}
fn public_inputs() -> Vec<Interface> {
    vec![
        spec("board", TensorElementType::Int64, &[-1, 64]),
        spec("metadata", TensorElementType::Float32, &[-1, 16]),
        spec("records", TensorElementType::Float32, &[-1, -1, 16]),
        spec("record_mask", TensorElementType::Bool, &[-1, -1]),
    ]
}
fn public_inputs_for(profile: PalsModelProfile) -> Vec<Interface> {
    let mut inputs = public_inputs();
    if profile.uses_full_line() {
        inputs.extend([
            spec(
                "record_line_tokens",
                TensorElementType::Int64,
                &[-1, -1, 256, 3],
            ),
            spec("record_line_mask", TensorElementType::Bool, &[-1, -1, 256]),
        ]);
    }
    inputs
}
fn memory_outputs() -> Vec<Interface> {
    vec![
        spec("memory_key", TensorElementType::Float32, &[-1, 2, -1, 64]),
        spec("memory_value", TensorElementType::Float32, &[-1, 2, -1, 64]),
        spec("memory_mask", TensorElementType::Bool, &[-1, -1]),
    ]
}
fn role_inputs(critic: bool) -> Vec<Interface> {
    let mut result = memory_outputs();
    result.extend([
        spec("candidates", TensorElementType::Int64, &[-1, -1, 3]),
        spec("candidate_mask", TensorElementType::Bool, &[-1, -1]),
        spec("query", TensorElementType::Float32, &[-1, 16]),
    ]);
    if critic {
        result.extend([
            spec(
                "divergence_features",
                TensorElementType::Float32,
                &[-1, -1, 8],
            ),
            spec("divergence_mask", TensorElementType::Bool, &[-1, -1]),
        ]);
    }
    result
}
fn role_inputs_for(critic: bool, profile: PalsModelProfile) -> Vec<Interface> {
    let mut inputs = role_inputs(critic);
    if profile.uses_full_line() {
        inputs.extend([
            spec(
                "query_line_tokens",
                TensorElementType::Int64,
                &[-1, 3, 256, 3],
            ),
            spec("query_line_mask", TensorElementType::Bool, &[-1, 3, 256]),
            spec("record_relations", TensorElementType::Int64, &[-1, -1, 2]),
        ]);
    }
    inputs
}
fn role_outputs(critic: bool) -> Vec<Interface> {
    let mut result = vec![
        spec("candidate_logits", TensorElementType::Float32, &[-1, -1]),
        spec("wdl_logits", TensorElementType::Float32, &[-1, 3]),
        spec("private_latent", TensorElementType::Float32, &[-1, 16, 384]),
    ];
    if critic {
        result.push(spec(
            "divergence_logits",
            TensorElementType::Float32,
            &[-1, -1],
        ));
    }
    result
}
fn shared_inputs() -> Vec<Interface> {
    let mut result = vec![spec("role_is_critic", TensorElementType::Bool, &[])];
    result.extend(role_inputs(true));
    result
}
fn shared_inputs_for(profile: PalsModelProfile) -> Vec<Interface> {
    let mut inputs = vec![spec("role_is_critic", TensorElementType::Bool, &[])];
    inputs.extend(role_inputs_for(true, profile));
    inputs
}
fn shared_outputs() -> Vec<Interface> {
    let mut result = role_outputs(true);
    result.push(spec("is_critic", TensorElementType::Bool, &[]));
    result
}
const fn spec(name: &'static str, dtype: TensorElementType, shape: &'static [i64]) -> Interface {
    Interface { name, dtype, shape }
}
fn fail(kind: K, stage: S, detail: &'static str) -> BackendError {
    BackendError::new(kind, stage, detail)
}
fn native(code: CauseCode, detail: &'static str, error: ort::Error) -> BackendError {
    fail(K::BackendFailure, S::Backend, detail).with_ort_cause(code, error)
}
fn model_input(error: impl std::fmt::Display) -> BackendError {
    fail(
        K::InvalidInput,
        S::Admission,
        "PALS model request violates its bounded tensor contract",
    )
    .with_external_cause(CauseCode::OutputValidation, &error)
}
fn validate_role_tag(shape: &[i64], values: &[bool], critic: bool) -> Result<(), BackendError> {
    if !shape.is_empty() || values != [critic] {
        return Err(fail(
            K::IdentityMismatch,
            S::Output,
            "shared P/C output role tag differs from admitted request",
        ));
    }
    Ok(())
}
#[derive(Default)]
enum CudaMappingAudit {
    #[default]
    AwaitingFirstRun,
    Confirmed,
    Rejected(BackendError),
}
impl CudaMappingAudit {
    fn allow_run(&self) -> Result<(), BackendError> {
        match self {
            Self::Rejected(cause) => Err(cause.clone()),
            _ => Ok(()),
        }
    }
    fn after_run(
        &mut self,
        completed: bool,
        full_audit: impl FnOnce() -> Result<(), BackendError>,
    ) -> Result<(), BackendError> {
        self.allow_run()?;
        if !completed || matches!(self, Self::Confirmed) {
            return Ok(());
        }
        match full_audit() {
            Ok(()) => {
                *self = Self::Confirmed;
                Ok(())
            }
            Err(mut failure) => {
                failure.detail = "PALS CUDA full mapping audit failed after completed native Run";
                *self = Self::Rejected(failure.clone());
                Err(failure)
            }
        }
    }
    fn final_audit(
        &mut self,
        full_audit: impl FnOnce() -> Result<(), BackendError>,
    ) -> Result<(), BackendError> {
        self.allow_run()?;
        if !matches!(self, Self::Confirmed) {
            return Err(fail(
                K::BackendUnavailable,
                S::Backend,
                "PALS full CUDA audit requires a completed native Run",
            ));
        }
        match full_audit() {
            Ok(()) => Ok(()),
            Err(mut failure) => {
                failure.detail =
                    "PALS CUDA final full mapping audit failed after completed native Run";
                *self = Self::Rejected(failure.clone());
                Err(failure)
            }
        }
    }
}

fn validate_declared_interface(
    values: &[TensorManifest],
    expected: &[Interface],
) -> Result<(), BackendError> {
    if values.len() != expected.len() {
        return Err(fail(
            K::UnsupportedModel,
            S::Asset,
            "PALS graph descriptor interface differs",
        ));
    }
    for expected in expected {
        let matching: Vec<_> = values.iter().filter(|v| v.name == expected.name).collect();
        if matching.len() != 1 {
            return Err(fail(
                K::UnsupportedModel,
                S::Asset,
                "PALS graph descriptor names differ",
            ));
        }
        let value = matching[0];
        let dtype = match expected.dtype {
            TensorElementType::Float32 => "FLOAT",
            TensorElementType::Int64 => "INT64",
            TensorElementType::Bool => "BOOL",
            _ => unreachable!(),
        };
        if value.dtype != dtype
            || value.shape.len() != expected.shape.len()
            || !value
                .shape
                .iter()
                .zip(expected.shape)
                .all(|(actual, expected)| {
                    if *expected == -1 {
                        actual
                            .as_str()
                            .is_some_and(|s| !s.is_empty() && s.len() <= 64)
                    } else {
                        actual.as_i64() == Some(*expected)
                    }
                })
        {
            return Err(fail(
                K::UnsupportedModel,
                S::Asset,
                "PALS descriptor tensor dtype/shape differs",
            ));
        }
    }
    Ok(())
}
fn validate_manifest(manifest: &ExportManifest) -> Result<[u8; 32], BackendError> {
    manifest.config.validate().map_err(model_input)?;
    let layout = manifest.resolved_layout()?;
    match (
        &manifest.rules_input_profile,
        &manifest.rules_input_semantic_sha256,
    ) {
        (None, None) if layout == Layout::SeparatePc => {}
        (Some(profile), Some(digest))
            if profile
                == if manifest.config.profile.uses_full_line() {
                    "rz-pals-rules-full-line-v2"
                } else {
                    "rz-pals-rules-fields-v1"
                } =>
        {
            parse_sha256(digest)?;
        }
        _ => {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "PALS Rules semantic profile/digest is missing or invalid",
            ));
        }
    }
    for digest in [
        &manifest.rules_encoder_source_sha256,
        &manifest.rules_profile_descriptor_sha256,
        &manifest.rules_profile_canonical_sha256,
    ]
    .into_iter()
    .flatten()
    {
        parse_sha256(digest)?;
    }
    if let Some(declaration) = &manifest.rules_input_declaration {
        if declaration
            .get("rules_input_profile")
            .and_then(serde_json::Value::as_str)
            != manifest.rules_input_profile.as_deref()
            || declaration
                .get("rules_input_semantic_sha256")
                .and_then(serde_json::Value::as_str)
                != manifest.rules_input_semantic_sha256.as_deref()
            || declaration
                .get("rules_encoder_source_sha256")
                .and_then(serde_json::Value::as_str)
                != manifest.rules_encoder_source_sha256.as_deref()
            || declaration.get("config")
                != Some(&serde_json::to_value(&manifest.config).map_err(model_input)?)
            || manifest.learned_input_compatibility.as_deref()
                != Some("unverified_declaration_only")
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "PALS Rules declaration contradicts registered tensor semantics",
            ));
        }
    }
    if layout == Layout::SharedPcIf {
        if manifest.rules_encoder_source_sha256.is_none() {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "shared P/C artifact has no Rules encoder provenance digest",
            ));
        }
        validate_reader_bank(manifest.reader_initializer_bank.as_ref().ok_or_else(|| {
            fail(
                K::IdentityMismatch,
                S::Asset,
                "shared P/C artifact has no initializer ownership audit",
            )
        })?)?;
    }
    let promotion = ["none", "queen", "rook", "bishop", "knight"]
        .iter()
        .enumerate()
        .map(|(i, name)| (i.to_string(), name.to_string()))
        .collect::<BTreeMap<_, _>>();
    let mut roles = manifest.roles.clone();
    roles.sort();
    if manifest.validator_present
        || roles != ["critic", "proposer"]
        || manifest.graphs.len() != if layout == Layout::SharedPcIf { 2 } else { 3 }
        || manifest.wdl_perspective != "input_side_to_move"
        || !manifest
            .task_names
            .iter()
            .map(String::as_str)
            .eq(V_TASK_NAMES)
        || manifest.candidate_promotion != promotion
        || manifest.runtime.onnxruntime != "1.22.0"
        || manifest.runtime.rust_ort != "2.0.0-rc.10"
        || manifest.runtime.compatibility != "requires_actual_numeric_check"
        || manifest.numeric_status != "not_run"
        || manifest.cuda_status != "not_run"
        || (manifest.trained && manifest.training_steps == 0)
        || (!manifest.trained && manifest.training_steps != 0)
    {
        return Err(fail(
            K::UnsupportedModel,
            S::Asset,
            "expected registered P/C-only FP32 PALS export; V is not a product role",
        ));
    }
    let expected_roles: &[&str] = if layout == Layout::SharedPcIf {
        &["public", "shared_pc"]
    } else {
        &["public", "proposer", "critic"]
    };
    for role in expected_roles.iter().copied() {
        let graphs: Vec<_> = manifest.graphs.iter().filter(|g| g.role == role).collect();
        if graphs.len() != 1 {
            return Err(fail(
                K::UnsupportedModel,
                S::Asset,
                "PALS public/P/C graph set differs",
            ));
        }
        let graph = graphs[0];
        let expected_file = if role == "public" {
            "public_memory.onnx".to_owned()
        } else if role == "shared_pc" {
            "shared_pc_if.onnx".to_owned()
        } else {
            format!("role_{role}.onnx")
        };
        if graph.file != expected_file
            || graph.opset != 17
            || !matches!(
                Path::new(&graph.file)
                    .components()
                    .collect::<Vec<_>>()
                    .as_slice(),
                [Component::Normal(_)]
            )
        {
            return Err(fail(
                K::UnsupportedModel,
                S::Asset,
                "PALS graph file/opset is unsupported",
            ));
        }
        parse_sha256(&graph.sha256)?;
        let critic = role == "critic";
        validate_declared_interface(
            &graph.inputs,
            &if role == "public" {
                public_inputs_for(manifest.config.profile)
            } else if role == "shared_pc" {
                shared_inputs_for(manifest.config.profile)
            } else {
                role_inputs_for(critic, manifest.config.profile)
            },
        )?;
        validate_declared_interface(
            &graph.outputs,
            &if role == "public" {
                memory_outputs()
            } else if role == "shared_pc" {
                shared_outputs()
            } else {
                role_outputs(critic)
            },
        )?;
    }
    parse_sha256(&manifest.checkpoint_sha256)
}
fn validate_reader_bank(bank: &ReaderInitializerBank) -> Result<(), BackendError> {
    let mut parameters = std::collections::BTreeSet::new();
    let mut initializers = std::collections::BTreeSet::new();
    let mut total = 0_u64;
    for entry in &bank.shared_parameters {
        let elements = entry
            .shape
            .iter()
            .try_fold(1_u64, |n, dim| n.checked_mul(*dim as u64));
        if entry.parameter.is_empty()
            || entry.initializer.is_empty()
            || entry.parameter.len() > 256
            || entry.initializer.len() > 256
            || entry.shape.len() > 8
            || entry.shape.contains(&0)
            || elements.and_then(|n| n.checked_mul(4)) != Some(entry.bytes)
            || !parameters.insert(entry.parameter.as_str())
            || !initializers.insert(entry.initializer.as_str())
        {
            return Err(fail(
                K::UnsupportedModel,
                S::Asset,
                "shared initializer names/shapes/counts differ",
            ));
        }
        parse_sha256(&entry.sha256)?;
        parse_sha256(&entry.source_parameter_sha256)?;
        total = total.checked_add(entry.bytes).ok_or_else(|| {
            fail(
                K::ResourceExhausted,
                S::Asset,
                "shared initializer budget overflow",
            )
        })?;
    }
    let mut routes = BTreeMap::<&str, std::collections::BTreeSet<&str>>::new();
    for branch in &bank.branch_operations {
        if branch.route.is_empty()
            || branch.route.len() > 128
            || branch.nodes == 0
            || !matches!(branch.branch.as_str(), "then_branch" | "else_branch")
            || !routes
                .entry(&branch.route)
                .or_default()
                .insert(&branch.branch)
        {
            return Err(fail(
                K::UnsupportedModel,
                S::Asset,
                "shared If branch ownership audit differs",
            ));
        }
    }
    if bank.scope != "onnx_serialized_initializers_only"
        || bank.layout_revision != 1
        || bank.shared_parameters.is_empty()
        || bank.unique_shared_parameters != parameters.len()
        || total != bank.shared_weight_bytes
        || total > MAX_MODEL_BYTES
        || bank.if_routes != 6
        || routes.len() != bank.if_routes
        || routes.values().any(|branches| branches.len() != 2)
        || bank.branch_local_initializers != 0
        || bank.inactive_private_execution != "requires_runtime_profile"
        || bank.ort_prepack_copies != "unknown"
        || bank.device_residency_sharing != "unknown"
    {
        return Err(fail(
            K::UnsupportedModel,
            S::Asset,
            "shared P/C initializer audit is unsupported",
        ));
    }
    Ok(())
}
fn validate_session(
    session: &Session,
    graph: &GraphManifest,
    manifest: &ExportManifest,
) -> Result<(), BackendError> {
    let inputs = if graph.role == "public" {
        public_inputs_for(manifest.config.profile)
    } else if graph.role == "shared_pc" {
        shared_inputs_for(manifest.config.profile)
    } else {
        role_inputs_for(graph.role == "critic", manifest.config.profile)
    };
    let outputs = if graph.role == "public" {
        memory_outputs()
    } else if graph.role == "shared_pc" {
        shared_outputs()
    } else {
        role_outputs(graph.role == "critic")
    };
    let valid = |name: &str, ty: &ValueType, expected: &[Interface]| {
        expected.iter().any(|expected| {
        name == expected.name && matches!(ty, ValueType::Tensor { ty, shape, .. } if *ty == expected.dtype && shape.as_ref() == expected.shape)
    })
    };
    if session.inputs.len() != inputs.len()
        || session.outputs.len() != outputs.len()
        || !session
            .inputs
            .iter()
            .all(|v| valid(&v.name, &v.input_type, &inputs))
        || !session
            .outputs
            .iter()
            .all(|v| valid(&v.name, &v.output_type, &outputs))
    {
        return Err(fail(
            K::UnsupportedModel,
            S::Asset,
            "loaded PALS ONNX tensor interface differs",
        ));
    }
    let metadata = session
        .metadata()
        .map_err(|e| native(CauseCode::ModelLoad, "cannot inspect PALS ONNX metadata", e))?;
    let steps = manifest.training_steps.to_string();
    for (key, value) in [
        ("schema", manifest.schema.as_str()),
        ("role", graph.role.as_str()),
        ("checkpoint_sha256", manifest.checkpoint_sha256.as_str()),
        ("trained", if manifest.trained { "true" } else { "false" }),
        ("training_steps", steps.as_str()),
        ("precision", "fp32"),
        ("expected_ort", "1.22.0"),
        ("public_kv", "shared_role_neutral"),
    ] {
        if metadata
            .custom(key)
            .map_err(|e| native(CauseCode::ModelLoad, "cannot read PALS ONNX metadata", e))?
            .as_deref()
            != Some(value)
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "PALS ONNX metadata disagrees with registered export",
            ));
        }
    }
    let shared = manifest.resolved_layout()? == Layout::SharedPcIf;
    if !manifest.config.profile.is_legacy() {
        for (key, expected) in [
            ("model_profile", manifest.config.profile.as_str()),
            ("encoding_schema", manifest.config.profile.encoding_schema()),
        ] {
            if metadata
                .custom(key)
                .map_err(|e| {
                    native(
                        CauseCode::ModelLoad,
                        "cannot read PALS V2 profile metadata",
                        e,
                    )
                })?
                .as_deref()
                != Some(expected)
            {
                return Err(fail(
                    K::IdentityMismatch,
                    S::Asset,
                    "PALS graph V2 profile metadata differs",
                ));
            }
        }
    }
    for (key, expected) in [
        ("model_semantics", manifest.model_semantics.as_deref()),
        ("layout", manifest.layout.as_deref()),
        (
            "rules_input_profile",
            manifest.rules_input_profile.as_deref(),
        ),
        (
            "rules_input_semantic_sha256",
            manifest.rules_input_semantic_sha256.as_deref(),
        ),
        (
            "rules_encoder_source_sha256",
            manifest.rules_encoder_source_sha256.as_deref(),
        ),
    ] {
        if let Some(expected) = expected {
            if metadata
                .custom(key)
                .map_err(|e| {
                    native(
                        CauseCode::ModelLoad,
                        "cannot read PALS semantic metadata",
                        e,
                    )
                })?
                .as_deref()
                != Some(expected)
            {
                return Err(fail(
                    K::IdentityMismatch,
                    S::Asset,
                    "PALS graph semantic metadata disagrees with export",
                ));
            }
        }
    }
    if shared
        && metadata
            .custom("layout_revision")
            .map_err(|e| native(CauseCode::ModelLoad, "cannot read PALS layout revision", e))?
            .as_deref()
            != Some("1")
    {
        return Err(fail(
            K::IdentityMismatch,
            S::Asset,
            "PALS graph layout revision differs",
        ));
    }
    Ok(())
}
fn session_optimization(config: PalsOnnxConfig, explicit_control: bool) -> PalsGraphOptimization {
    if explicit_control && matches!(config.provider, Provider::Cuda { .. }) {
        PalsGraphOptimization::Disable
    } else {
        PalsGraphOptimization::Level1
    }
}
/// 원래 모델 SessionBuilder의 setter, EP 등록, session commit이 성공한 설정이다.
/// OS thread 생성 수·helper affinity·CUDA kernel placement·첫 NN 완료를
/// 관측한 값이 아니다. 공개 생성자는 없으며 성공한 loader만 발급한다.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PalsAppliedSessionSettings {
    role: String,
    provider: Provider,
    intra_threads: usize,
    inter_threads: usize,
    intra_op_spinning: bool,
    inter_op_spinning: bool,
    parallel_execution: bool,
    cuda_provider_available: Option<bool>,
}
impl PalsAppliedSessionSettings {
    pub fn role(&self) -> &str {
        &self.role
    }
    pub fn provider(&self) -> Provider {
        self.provider
    }
    pub fn intra_threads(&self) -> usize {
        self.intra_threads
    }
    pub fn inter_threads(&self) -> usize {
        self.inter_threads
    }
    pub fn intra_op_spinning(&self) -> bool {
        self.intra_op_spinning
    }
    pub fn inter_op_spinning(&self) -> bool {
        self.inter_op_spinning
    }
    pub fn parallel_execution(&self) -> bool {
        self.parallel_execution
    }
    /// CPU는 해당 CUDA 가용성 조회를 하지 않았으므로 None이다.
    pub fn cuda_provider_available(&self) -> Option<bool> {
        self.cuda_provider_available
    }
    pub fn evidence_scope(&self) -> &'static str {
        "session_builder_accepted_settings"
    }
    /// 별도 weight-free packing session이나 프로세스 전체 native inventory는 제외한다.
    pub fn coverage_scope(&self) -> &'static str {
        "loaded_model_sessions_only"
    }
    pub fn evidence_method(&self) -> &'static str {
        "setters_execution_provider_registration_and_session_commit_success"
    }
}
fn load_session(
    bytes: Vec<u8>,
    config: PalsOnnxConfig,
    role: &str,
    audit: Option<(&PalsCudaControlPolicy, &Path)>,
) -> Result<
    (
        Session,
        Option<PalsGraphPlacement>,
        PalsAppliedSessionSettings,
    ),
    BackendError,
> {
    let configure = |e| {
        native(
            CauseCode::SessionConfiguration,
            "PALS ORT session configuration failed",
            e,
        )
    };
    let mut builder = Session::builder()
        .map_err(configure)?
        .with_no_environment_execution_providers()
        .map_err(configure)?
        .with_intra_threads(config.intra_threads)
        .map_err(configure)?
        .with_inter_threads(1)
        .map_err(configure)?
        .with_parallel_execution(false)
        .map_err(configure)?
        .with_intra_op_spinning(false)
        .map_err(configure)?
        .with_inter_op_spinning(false)
        .map_err(configure)?
        .with_memory_pattern(false)
        .map_err(configure)?
        // Explicit control v2 is separately attested. Disabling graph rewrites
        // keeps named source metadata chains; strict CUDA and CPU retain L1.
        .with_optimization_level(match session_optimization(config, audit.is_some()) {
            PalsGraphOptimization::Disable => GraphOptimizationLevel::Disable,
            PalsGraphOptimization::Level1 => GraphOptimizationLevel::Level1,
        })
        .map_err(configure)?;
    let mut cuda_provider_available = None;
    builder = match config.provider {
        Provider::Cpu => builder
            .with_execution_providers([CPUExecutionProvider::default()
                .with_arena_allocator(false)
                .build()
                .error_on_failure()])
            .map_err(configure)?,
        Provider::Cuda {
            device_id,
            arena_bytes,
        } => {
            let cuda = match config.cudnn_conv_search {
                PalsCuDnnConvSearch::LegacyImplicitExhaustive => CUDAExecutionProvider::default(),
                PalsCuDnnConvSearch::ExperimentalDefault
                | PalsCuDnnConvSearch::ExperimentalDefaultDeviceWarm => {
                    CUDAExecutionProvider::default()
                        .with_conv_algorithm_search(CuDNNConvAlgorithmSearch::Default)
                }
            };
            if !cuda.is_available().map_err(configure)? {
                return Err(fail(
                    K::BackendUnavailable,
                    S::Backend,
                    "PALS CUDA provider unavailable; CPU fallback forbidden",
                ));
            }
            cuda_provider_available = Some(true);
            builder
                // Strict is unchanged unless the caller registered an exact
                // metadata-only policy. Its complete init gate runs before Run.
                .with_config_entry(
                    "session.disable_cpu_ep_fallback",
                    if audit.is_some() { "0" } else { "1" },
                )
                .map_err(configure)?
                .with_execution_providers([cuda
                    .with_device_id(device_id)
                    .with_memory_limit(arena_bytes)
                    .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
                    .with_tf32(false)
                    .with_cuda_graph(false)
                    .build()
                    .error_on_failure()])
                .map_err(configure)?
        }
    };
    let placement_log = if let Some((policy, root)) = audit {
        let (configured, log) =
            policy.configure(builder, role, &root.join(format!("pals-{role}-kernels")))?;
        builder = configured;
        Some(log)
    } else {
        None
    };
    // Default native copying; no direct-reference initializer option is enabled.
    // Serialized owned bytes are freed immediately after successful/failed load.
    let loaded = builder.commit_from_memory(&bytes);
    drop(bytes);
    let placement = if let (Some((policy, _)), Some(log)) = (audit, &placement_log) {
        loaded
            .is_ok()
            .then(|| policy.verify_initialization(role, log))
    } else {
        None
    };
    let recorded = if let (Some((policy, root)), Some(log)) = (audit, &placement_log) {
        if policy.virtual_review_layout().is_some() {
            policy.record_initial_log_virtual(
                log,
                &root.join(format!("{role}-initial-placement.json")),
                role,
                loaded.is_ok(),
                placement.as_ref(),
            )
        } else {
            cuda_control::record_initial_log(
                log,
                &root.join(format!("{role}-initial-placement.json")),
                role,
                loaded.is_ok(),
                placement.as_ref(),
            )
        }
    } else {
        Ok(())
    };
    // Preserve the primary native initialization failure; successful sessions
    // require both a saved bounded log and the independent pre-Run gate.
    let session =
        loaded.map_err(|e| native(CauseCode::ModelLoad, "PALS ONNX loading failed", e))?;
    // Recording is attempted even for a rejected native/policy initialization.
    // Preserve that primary error; a successful gate cannot bypass failed IO.
    let placement = placement.transpose()?;
    recorded?;
    // 위 setter/EP 등록/commit과 모든 기존 초기화 검사가 성공한 뒤에만 발급한다.
    let applied = PalsAppliedSessionSettings {
        role: role.to_owned(),
        provider: config.provider,
        intra_threads: config.intra_threads,
        inter_threads: 1,
        intra_op_spinning: false,
        inter_op_spinning: false,
        parallel_execution: false,
        cuda_provider_available,
    };
    Ok((session, placement, applied))
}

struct ActiveInputs {
    /// Entire known prepared backing capacities, before ownership moves into
    /// OrtValues. Native session/allocator overhead remains unobserved.
    host_bytes: u64,
    // ONNX If consumes its scalar condition on CPU, including CUDA sessions.
    role_is_critic: Tensor<bool>,
    board: Tensor<i64>,
    metadata: Tensor<f32>,
    records: Tensor<f32>,
    record_mask: Tensor<bool>,
    candidates: Tensor<i64>,
    candidate_mask: Tensor<bool>,
    query: Tensor<f32>,
    divergences: Tensor<f32>,
    divergence_mask: Tensor<bool>,
    private_warm: Option<warm::ActiveWarmInputs>,
    full_line: Option<ActiveFullLineInputs>,
}
struct ActiveFullLineInputs {
    record_line_tokens: Tensor<i64>,
    record_line_mask: Tensor<bool>,
    query_line_tokens: Tensor<i64>,
    query_line_mask: Tensor<bool>,
    record_relations: Tensor<i64>,
}
impl ActiveFullLineInputs {
    fn new(value: PreparedPalsFullLineTensors, records: usize) -> Result<Self, BackendError> {
        let error = |e| {
            native(
                CauseCode::TensorCreate,
                "cannot own full-line PALS input tensor",
                e,
            )
        };
        Ok(Self {
            record_line_tokens: Tensor::from_array((
                [1, records, MAX_LINE_PLY, 3],
                value.record_line_tokens,
            ))
            .map_err(error)?,
            record_line_mask: Tensor::from_array((
                [1, records, MAX_LINE_PLY],
                value.record_line_mask,
            ))
            .map_err(error)?,
            query_line_tokens: Tensor::from_array((
                [1, QUERY_LINE_COUNT, MAX_LINE_PLY, 3],
                value.query_line_tokens,
            ))
            .map_err(error)?,
            query_line_mask: Tensor::from_array((
                [1, QUERY_LINE_COUNT, MAX_LINE_PLY],
                value.query_line_mask,
            ))
            .map_err(error)?,
            record_relations: Tensor::from_array(([1, records, 2], value.record_relations))
                .map_err(error)?,
        })
    }
}
impl ActiveInputs {
    fn new(
        value: PreparedPalsTensors,
        critic: bool,
        private: Option<&PalsWarmInput>,
    ) -> Result<Self, BackendError> {
        let tensor_error = |e| native(CauseCode::TensorCreate, "cannot own PALS input tensor", e);
        let r = value.record_mask.len();
        let c = value.candidate_mask.len();
        let d = value.divergence_mask.len();
        let host_bytes = (std::mem::size_of::<Self>()
            + (value.board.capacity() + value.candidates.capacity()) * 8
            + (value.metadata.capacity()
                + value.records.capacity()
                + value.query.capacity()
                + value.divergences.capacity())
                * 4
            + value.record_mask.capacity()
            + value.candidate_mask.capacity()
            + value.divergence_mask.capacity()
            + 1
            + value
                .full_line
                .as_ref()
                .map_or(0, PreparedPalsFullLineTensors::owned_bytes))
            as u64;
        let private_warm = private.map(PalsWarmInput::prepare).transpose()?;
        let host_bytes = host_bytes
            .checked_add(private_warm.as_ref().map_or(0, |v| v.host_bytes))
            .ok_or_else(|| {
                fail(
                    K::ResourceExhausted,
                    S::Admission,
                    "PALS Warm input byte overflow",
                )
            })?;
        Ok(Self {
            host_bytes,
            role_is_critic: Tensor::from_array((Vec::<usize>::new(), vec![critic]))
                .map_err(tensor_error)?,
            board: Tensor::from_array(([1, 64], value.board)).map_err(tensor_error)?,
            metadata: Tensor::from_array(([1, 16], value.metadata)).map_err(tensor_error)?,
            records: Tensor::from_array(([1, r, 16], value.records)).map_err(tensor_error)?,
            record_mask: Tensor::from_array(([1, r], value.record_mask)).map_err(tensor_error)?,
            candidates: Tensor::from_array(([1, c, 3], value.candidates)).map_err(tensor_error)?,
            candidate_mask: Tensor::from_array(([1, c], value.candidate_mask))
                .map_err(tensor_error)?,
            query: Tensor::from_array(([1, 16], value.query)).map_err(tensor_error)?,
            divergences: Tensor::from_array(([1, d, 8], value.divergences))
                .map_err(tensor_error)?,
            divergence_mask: Tensor::from_array(([1, d], value.divergence_mask))
                .map_err(tensor_error)?,
            private_warm,
            full_line: value
                .full_line
                .map(|lines| ActiveFullLineInputs::new(lines, r))
                .transpose()?,
        })
    }
}
struct PublicMemory {
    tokens: usize,
    memory_key: Tensor<f32>,
    memory_value: Tensor<f32>,
    mask: Tensor<bool>,
}
fn public_owner_bytes(key_capacity: usize, value_capacity: usize, mask_capacity: usize) -> u64 {
    // Charge all three backing owners, never a short view of a larger K/V.
    // Native OrtValue/allocator overhead is not an observed heap/VRAM peak.
    ((key_capacity + value_capacity) * std::mem::size_of::<f32>()
        + mask_capacity * std::mem::size_of::<bool>()
        + std::mem::size_of::<PublicMemory>()) as u64
}
fn public_bank_error(error: impl std::fmt::Display) -> BackendError {
    fail(
        K::ResourceExhausted,
        S::Backend,
        "PALS bounded host public-memory reservation or pin release failed",
    )
    .with_external_cause(CauseCode::InputAllocation, &error)
}
/// A copied diagnostic witness after physical completion, never a product
/// private latent or a second inference. Device-only K/V has no host witness.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PalsPublicMemoryWitness {
    pub tokens: usize,
    pub memory_key: Vec<f32>,
    pub memory_value: Vec<f32>,
    pub memory_mask: Vec<bool>,
}

/// Cache reset is a physical-worker command so a completed NewGame response
/// proves the old cache was retired after the preceding native execution.
/// It neither reloads weights nor invokes a neural graph.
// Exactly one exclusive physical worker owns at most one in-flight command.
// Keep its bounded input inline rather than adding a Box allocation to every
// role request merely to match the zero-payload control variants' size.
#[allow(clippy::large_enum_variant)]
pub enum PalsNativeCommand {
    Evaluate(PalsModelInput),
    /// Explicit V4 evidence from the same invocation, without a second Run or
    /// Stats command. The legacy Evaluate result and accounting stay unchanged.
    EvaluateWithEvidence(PalsModelInput),
    /// Separate CPU-only Warm graph domain; a Fresh mode payload is also explicit.
    EvaluatePrivateWarm(PalsWarmInput),
    /// CUDA Warm has its own capability and physical owner domain.
    #[cfg(feature = "experimental-io-binding")]
    EvaluateCudaWarm(PalsCudaWarmInput),
    NewGame,
    /// NN-zero reset to an explicit logical target. Repeated logical new games
    /// may skip generations; the exclusive backend accepts only increasing ones.
    ResetTo(u64),
    SnapshotStats,
    /// Metadata from this exclusive worker's actual resident owner. This command
    /// neither creates a native Run nor converts unknown completion to a fence.
    #[cfg(feature = "experimental-io-binding")]
    SnapshotCudaRecordPages,
    VerifyRuntime,
    VerifyCudaPlacement,
    ObserveRuntimeMappings,
}
pub enum PalsNativeResult {
    Evaluation(PalsRawOutput),
    EvaluationWithEvidence {
        /// A known returned failure retains its original BackendError. CUDA
        /// unknown completion still takes the outer quarantine path instead.
        outcome: Result<PalsRawOutput, BackendError>,
        actual_completed_nn_inputs_delta: Option<u64>,
        /// Completeness of measurement only; this is not output acceptance.
        complete: bool,
        /// 같은 단일 physical owner의 원본 호출에서 관측한 단계별 계수.
        /// 알려지지 않은 완료나 검증되지 않은 역할 capability는 None이다.
        owner_accounting: Option<PalsNativeOwnerAccounting>,
    },
    NewGame,
    ResetTo {
        game_generation: u64,
    },
    Stats(PalsBackendStats),
    #[cfg(feature = "experimental-io-binding")]
    CudaRecordPagesObserved(Box<CudaRecordPageSnapshot>),
    RuntimeVerified,
    CudaPlacementVerified(Box<PalsCudaPlacementWitness>),
    RuntimeMappingsObserved(Box<PalsNativeMappingWitness>),
}

fn host_observation_outcome(
    result: &Result<PalsNativeResult, BackendError>,
    physical_completion_unknown: bool,
) -> HostRecordPageObservationOutcome {
    if physical_completion_unknown {
        HostRecordPageObservationOutcome::PhysicalCompletionUnknown
    } else if matches!(
        result,
        Err(_)
            | Ok(PalsNativeResult::EvaluationWithEvidence {
                outcome: Err(_),
                ..
            })
    ) {
        HostRecordPageObservationOutcome::ReturnedError
    } else {
        HostRecordPageObservationOutcome::ReturnedOk
    }
}

impl PalsNativeCommand {
    /// Stable diagnostic label matched in the enum's defining feature domain.
    /// Downstream feature unification must not hide an enabled control variant.
    pub fn diagnostic_kind(&self) -> &'static str {
        match self {
            Self::Evaluate(input) if input.role == PalsRole::Critic => "critic_evaluation",
            Self::Evaluate(_) => "proposer_evaluation",
            Self::EvaluateWithEvidence(input) if input.role == PalsRole::Critic => {
                "critic_evaluation_with_evidence"
            }
            Self::EvaluateWithEvidence(_) => "proposer_evaluation_with_evidence",
            Self::EvaluatePrivateWarm(_) => "private_warm_evaluation",
            #[cfg(feature = "experimental-io-binding")]
            Self::EvaluateCudaWarm(_) => "cuda_warm_evaluation",
            Self::VerifyRuntime => "runtime_origin_audit",
            Self::ObserveRuntimeMappings => "runtime_loading_mapping",
            Self::VerifyCudaPlacement => "cuda_placement_audit",
            Self::NewGame => "new_game_reset",
            Self::ResetTo(_) => "explicit_game_reset",
            Self::SnapshotStats => "backend_stats",
            #[cfg(feature = "experimental-io-binding")]
            Self::SnapshotCudaRecordPages => "cuda_record_pages_observation",
        }
    }
}
impl PalsNativeResult {
    /// Stable result diagnostic; a label is not a physical-completion witness.
    pub fn diagnostic_kind(&self) -> &'static str {
        match self {
            Self::Evaluation(_) => "evaluation",
            Self::EvaluationWithEvidence { .. } => "evaluation_with_evidence",
            Self::NewGame => "new_game",
            Self::ResetTo { .. } => "reset_to_game",
            Self::Stats(_) => "stats",
            Self::RuntimeVerified => "runtime_verified",
            Self::CudaPlacementVerified(_) => "cuda_placement_verified",
            Self::RuntimeMappingsObserved(_) => "runtime_mappings_observed",
            #[cfg(feature = "experimental-io-binding")]
            Self::CudaRecordPagesObserved(_) => "cuda_record_pages_observed",
        }
    }
}

/// NN-zero metadata observation from the exclusive physical worker. A full
/// origin audit requires an already completed CUDA Run; it is not a kernel,
/// VRAM, device-drain, or successful-process-exit witness.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PalsNativeMappingWitness {
    pub schema: &'static str,
    pub runtime_sha256: [u8; 32],
    pub runtime_bundle_sha256: [u8; 32],
    pub loading_profile: &'static str,
    pub loading_profile_sha256: [u8; 32],
    pub scope: &'static str,
    pub declared_nvidia_files: usize,
    pub required_nvidia_files: Vec<String>,
    pub mapped_nvidia_files: Vec<String>,
    pub deferred_nvidia_not_mapped: Vec<String>,
    pub mapped_ort_files: Vec<String>,
}

fn native_mapping_witness(
    observed: NativeMappingObservation,
    profile: NativeLoadingProfile,
    runtime_sha256: [u8; 32],
    runtime_bundle_sha256: [u8; 32],
    loading_profile_sha256: [u8; 32],
) -> Result<PalsNativeMappingWitness, BackendError> {
    let expected_roots: Vec<_> = profile
        .eager_indices()
        .iter()
        .map(|&index| rz_native_loader::NVIDIA_LOAD_ORDER[index])
        .collect();
    let expected_mapped: Vec<_> = rz_native_loader::NVIDIA_LOAD_ORDER
        .iter()
        .copied()
        .filter(|name| observed.mapped_nvidia_files.contains(name))
        .collect();
    let expected_absent: Vec<_> = rz_native_loader::NVIDIA_LOAD_ORDER
        .iter()
        .copied()
        .filter(|name| !expected_roots.contains(name) && !expected_mapped.contains(name))
        .collect();
    if observed.profile != profile
        || observed.declared_nvidia_files != rz_native_loader::NVIDIA_LOAD_ORDER.len()
        || observed.required_nvidia_files != expected_roots
        || observed.mapped_nvidia_files != expected_mapped
        || expected_roots
            .iter()
            .any(|name| !expected_mapped.contains(name))
        || observed.deferred_nvidia_not_mapped != expected_absent
        || observed.mapped_ort_files.as_deref()
            != Some(rz_native_loader::ORT_LIBRARY_NAMES.as_slice())
        || loading_profile_sha256 != asset::sha256(profile.canonical_descriptor().as_bytes())
    {
        return Err(fail(
            K::IdentityMismatch,
            S::Backend,
            "full PALS runtime observation differs from the admitted loading profile",
        ));
    }
    let owned = |names: Vec<&'static str>| names.into_iter().map(str::to_owned).collect();
    Ok(PalsNativeMappingWitness {
        schema: "rovezero.pals-native-mapping-witness.v1",
        runtime_sha256,
        runtime_bundle_sha256,
        loading_profile: profile.identifier(),
        loading_profile_sha256,
        scope: "exclusive_physical_worker_full_runtime_origin",
        declared_nvidia_files: observed.declared_nvidia_files,
        required_nvidia_files: owned(observed.required_nvidia_files),
        mapped_nvidia_files: owned(observed.mapped_nvidia_files),
        deferred_nvidia_not_mapped: owned(observed.deferred_nvidia_not_mapped),
        mapped_ort_files: owned(
            observed
                .mapped_ort_files
                .expect("full audit checked ORT images"),
        ),
    })
}
/// Native graph evidence, cumulative across games for one frozen backend.
/// These are typed counters; a caller must choose its own receipt schema.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PalsBackendStats {
    pub admitted_role_requests: u64,
    pub public_cache_hits: u64,
    pub public_cache_misses: u64,
    pub public_nn_runs_attempted: u64,
    pub public_nn_runs_completed: u64,
    pub public_nn_runs_failed_known: u64,
    pub role_nn_runs_attempted: u64,
    pub role_nn_runs_completed: u64,
    pub role_nn_runs_failed_known: u64,
    /// Physical B1 inputs from both public and role graphs that returned and
    /// completed their physical fence; output validation is a separate count.
    pub completed_nn_inputs: u64,
    /// Full shape/finite/mask acceptance in the host-cache path. Device K/V
    /// currently has no independent host witness, so this stays zero there.
    pub validated_public_outputs: u64,
    pub validated_role_outputs: u64,
    pub new_game_resets: u64,
    pub live_public_cache_entries: u64,
}
impl PalsBackendStats {
    pub fn validate(&self) -> Result<(), BackendError> {
        if self.live_public_cache_entries > 1
            || self
                .public_nn_runs_completed
                .checked_add(self.public_nn_runs_failed_known)
                .is_none_or(|n| n > self.public_nn_runs_attempted)
            || self
                .role_nn_runs_completed
                .checked_add(self.role_nn_runs_failed_known)
                .is_none_or(|n| n > self.role_nn_runs_attempted)
            || self
                .public_nn_runs_completed
                .checked_add(self.role_nn_runs_completed)
                != Some(self.completed_nn_inputs)
            || self.validated_public_outputs > self.public_nn_runs_completed
            || self.validated_role_outputs > self.role_nn_runs_completed
            || self.role_nn_runs_attempted > self.admitted_role_requests
            || self
                .public_cache_hits
                .checked_add(self.public_cache_misses)
                .is_none_or(|n| n > self.admitted_role_requests)
        {
            return Err(fail(
                K::BackendFailure,
                S::Backend,
                "PALS native graph counters violate completion accounting",
            ));
        }
        Ok(())
    }
}

/// Differences of the actual public and role completion counters. Saturated,
/// decreasing, inconsistent or physically unknown counters yield no delta.
/// This neither calls a graph nor acknowledges a physical completion fence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PalsNNInputEvidence {
    pub actual_completed_nn_inputs_delta: Option<u64>,
    pub complete: bool,
}
impl PalsNNInputEvidence {
    pub fn between(
        before: &PalsBackendStats,
        after: &PalsBackendStats,
        physical_completion_known: bool,
    ) -> Self {
        let delta = (|| {
            if !physical_completion_known || before.validate().is_err() || after.validate().is_err()
            {
                return None;
            }
            let cumulative = |stats: &PalsBackendStats| {
                [
                    stats.admitted_role_requests,
                    stats.public_cache_hits,
                    stats.public_cache_misses,
                    stats.public_nn_runs_attempted,
                    stats.public_nn_runs_completed,
                    stats.public_nn_runs_failed_known,
                    stats.role_nn_runs_attempted,
                    stats.role_nn_runs_completed,
                    stats.role_nn_runs_failed_known,
                    stats.completed_nn_inputs,
                    stats.validated_public_outputs,
                    stats.validated_role_outputs,
                    stats.new_game_resets,
                ]
            };
            if cumulative(before)
                .into_iter()
                .zip(cumulative(after))
                .any(|(old, new)| {
                    old == u64::MAX || new == u64::MAX || new.checked_sub(old).is_none()
                })
            {
                return None;
            }
            let public = after
                .public_nn_runs_completed
                .checked_sub(before.public_nn_runs_completed)?;
            let role = after
                .role_nn_runs_completed
                .checked_sub(before.role_nn_runs_completed)?;
            let inputs = after
                .completed_nn_inputs
                .checked_sub(before.completed_nn_inputs)?;
            (public.checked_add(role)? == inputs).then_some(inputs)
        })();
        Self {
            actual_completed_nn_inputs_delta: delta,
            complete: delta.is_some(),
        }
    }
}

/// 한 원본 호출의 cache/public/private 실행 계수. 생성자는 공개하지 않는다.
/// 단일 physical worker가 같은 owner의 호출 전후를 비교한 경우에만 발급하며,
/// 출력 수락·검색 소비·프로세스 전체 V 부재를 증명하지 않는다.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PalsNativeOwnerAccounting {
    public_cache_hits: u64,
    public_cache_misses: u64,
    public_nn_runs_attempted: u64,
    public_nn_runs_completed: u64,
    public_nn_runs_failed_known: u64,
    role_nn_runs_attempted: u64,
    role_nn_runs_completed: u64,
    role_nn_runs_failed_known: u64,
    validator_nn_runs_attempted: u64,
    validator_nn_runs_completed: u64,
    validator_nn_runs_failed_known: u64,
}
impl PalsNativeOwnerAccounting {
    fn between(
        before: &PalsBackendStats,
        after: &PalsBackendStats,
        physical_completion_known: bool,
        validator_absence_verified: bool,
    ) -> Option<Self> {
        if !validator_absence_verified
            || !PalsNNInputEvidence::between(before, after, physical_completion_known).complete
        {
            return None;
        }
        let admitted = after
            .admitted_role_requests
            .checked_sub(before.admitted_role_requests)?;
        let value = Self {
            public_cache_hits: after
                .public_cache_hits
                .checked_sub(before.public_cache_hits)?,
            public_cache_misses: after
                .public_cache_misses
                .checked_sub(before.public_cache_misses)?,
            public_nn_runs_attempted: after
                .public_nn_runs_attempted
                .checked_sub(before.public_nn_runs_attempted)?,
            public_nn_runs_completed: after
                .public_nn_runs_completed
                .checked_sub(before.public_nn_runs_completed)?,
            public_nn_runs_failed_known: after
                .public_nn_runs_failed_known
                .checked_sub(before.public_nn_runs_failed_known)?,
            role_nn_runs_attempted: after
                .role_nn_runs_attempted
                .checked_sub(before.role_nn_runs_attempted)?,
            role_nn_runs_completed: after
                .role_nn_runs_completed
                .checked_sub(before.role_nn_runs_completed)?,
            role_nn_runs_failed_known: after
                .role_nn_runs_failed_known
                .checked_sub(before.role_nn_runs_failed_known)?,
            // 로드된 manifest와 실제 session 집합으로 확인한 owner capability다.
            // 요청 role을 보고 V 실행 수를 추정하지 않는다.
            validator_nn_runs_attempted: 0,
            validator_nn_runs_completed: 0,
            validator_nn_runs_failed_known: 0,
        };
        if admitted > 1
            || after.new_game_resets != before.new_game_resets
            || value
                .public_cache_hits
                .checked_add(value.public_cache_misses)?
                > admitted
            || value.role_nn_runs_attempted > admitted
            || value
                .public_nn_runs_completed
                .checked_add(value.public_nn_runs_failed_known)?
                != value.public_nn_runs_attempted
            || value
                .role_nn_runs_completed
                .checked_add(value.role_nn_runs_failed_known)?
                != value.role_nn_runs_attempted
        {
            return None;
        }
        // record-page 경로는 한 요청 안에서 여러 public graph를 실행할 수 있다.
        // 실제 public run 계수를 B1 요청 수 하나로 축소하지 않는다.
        Some(value)
    }
    pub fn public_cache_hits(&self) -> u64 {
        self.public_cache_hits
    }
    pub fn public_cache_misses(&self) -> u64 {
        self.public_cache_misses
    }
    pub fn public_nn_runs_attempted(&self) -> u64 {
        self.public_nn_runs_attempted
    }
    pub fn public_nn_runs_completed(&self) -> u64 {
        self.public_nn_runs_completed
    }
    pub fn public_nn_runs_failed_known(&self) -> u64 {
        self.public_nn_runs_failed_known
    }
    pub fn role_nn_runs_attempted(&self) -> u64 {
        self.role_nn_runs_attempted
    }
    pub fn role_nn_runs_completed(&self) -> u64 {
        self.role_nn_runs_completed
    }
    pub fn role_nn_runs_failed_known(&self) -> u64 {
        self.role_nn_runs_failed_known
    }
    pub fn validator_nn_runs_attempted(&self) -> u64 {
        self.validator_nn_runs_attempted
    }
    pub fn validator_nn_runs_completed(&self) -> u64 {
        self.validator_nn_runs_completed
    }
    pub fn validator_nn_runs_failed_known(&self) -> u64 {
        self.validator_nn_runs_failed_known
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PalsGraphIdentity {
    pub role: String,
    pub sha256: [u8; 32],
    /// Actual hash-verified serialized bytes read into native session loading;
    /// this is not an estimate of optimized native/session/VRAM residency.
    pub serialized_bytes: u64,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PalsSessionResidency {
    /// Loaded model graphs only. Weight-free auxiliary graphs are reported in
    /// their separate owner snapshot, never folded into model weight identity.
    pub graphs: Vec<PalsGraphIdentity>,
    /// Loaded model sessions. A selected resident owner separately reports its
    /// actual packing Session; callers must include it in total-session budgets.
    pub native_sessions: usize,
    pub layout: String,
    pub reader_initializer_bank: Option<ReaderInitializerBank>,
    /// v1 has independent P/C sessions (Some(false)). v2 has one serialized
    /// reader bank, but optimized native/prepacked storage sharing is unknown
    /// (None). This field never attests actual GPU/VRAM residency.
    pub role_reader_weights_shared: Option<bool>,
    pub native_resident_parameter_bytes: Option<u64>,
    pub vram_peak_bytes: Option<u64>,
}
/// Explicit resident-page registration, before the backend enters its worker.
/// Model, graph, encoding, runtime and game identities are derived from this
/// backend; neither a caller Session nor a placement declaration is accepted.
#[cfg(feature = "experimental-io-binding")]
pub struct PalsCudaRecordPageRegistration {
    pub checked_graph: CheckedFixedPackingGraph,
    pub process_epoch: rz_contracts::ProcessEpoch,
    pub frozen_epoch: u64,
    pub limits: DevicePagesLimits,
    pub invocation: DevicePageInvocationDeclaration,
    pub profile_root: std::path::PathBuf,
}
/// A session's execution weight epoch never changes. It has exactly one cache
/// slot, one active input slot and one exclusive caller/physical worker.
pub struct PalsOnnxBackend {
    public: Option<Session>,
    proposer: Option<Session>,
    critic: Option<Session>,
    shared_pc: Option<Session>,
    // 검증된 manifest와 실제 로드된 session 집합에서만 얻는 불변 capability.
    // 일반 PALS 모델이 항상 P/C-only라고 가정하지 않는다.
    validator_absence_verified: bool,
    applied_session_settings: Arc<[PalsAppliedSessionSettings]>,
    layout: Layout,
    private_warm: Option<PalsWarmCapability>,
    #[cfg(feature = "experimental-io-binding")]
    private_cuda_warm: Option<PalsCudaWarmCapability>,
    runtime: OrtRuntime,
    // Registered descriptor sources outlive their sessions and profile I/O.
    // Unknown physical completion retains this same owner with native values.
    owned_sources: Option<Arc<dyn PalsOwnedSources>>,
    config: PalsOnnxConfig,
    model_config: PalsModelConfig,
    epoch: [u8; 32],
    manifest_digest: [u8; 32],
    trained: bool,
    rules_input_profile: Option<String>,
    rules_input_semantic_sha256: Option<[u8; 32]>,
    residency: PalsSessionResidency,
    stats: PalsBackendStats,
    cuda_mapping_audit: RefCell<CudaMappingAudit>,
    cuda_control_audit: Option<CudaControlAudit>,
    // The bank keeps one immutable whole-input owner. Only the current physical
    // invocation holds an external pin; idle retained cache pages are unpinned.
    // Option permits preserving the complete bank on unknown native completion.
    memory: Option<MemoryBank<PublicMemory>>,
    cached_memory_key: Option<MemoryKey>,
    active_memory: Option<MemoryPin<PublicMemory>>,
    record_pages: Option<public_pages::HostRecordPages>,
    game_generation: u64,
    active: Option<ActiveInputs>,
    #[cfg(feature = "experimental-io-binding")]
    device_memory: Option<device::DeviceMemory>,
    #[cfg(feature = "experimental-io-binding")]
    device_role: Option<device::DeviceRole>,
    #[cfg(feature = "experimental-io-binding")]
    device_warm_role: Option<warm::cuda::DeviceWarmRole>,
    #[cfg(feature = "experimental-io-binding")]
    cuda_warm_observer: Option<PalsCudaWarmObservationHandle>,
    #[cfg(feature = "experimental-io-binding")]
    cuda_record_pages: Option<device_pages_plan::native::CudaRecordPages>,
    quarantine: Option<BackendError>,
    startup_stage_probe: Option<PalsStartupStageProbe>,
    pub public_encodes: u64,
    pub public_cache_hits: u64,
}
struct CudaControlAudit {
    policy: PalsCudaControlPolicy,
    profile_root: std::path::PathBuf,
    initialization: Vec<PalsGraphPlacement>,
    proposer_seen: bool,
    critic_seen: bool,
    witness: Option<PalsCudaPlacementWitness>,
    rejected: Option<BackendError>,
}
/// Immutable artifact access owned by a native startup capability. A callback
/// must not infer graph identity from a role name or replace a missing graph.
/// Returning bytes grants no acceptance: native loading independently repeats
/// bounded manifest validation and each complete graph hash before ORT sees it.
pub trait PalsOwnedSources: Send + Sync {
    fn read_manifest(&self, maximum: usize) -> Result<Vec<u8>, BackendError>;
    fn read_graph(
        &self,
        role: &str,
        registered_file: &str,
        expected_sha256: [u8; 32],
        maximum: usize,
    ) -> Result<Vec<u8>, BackendError>;
}
// Historical path loaders preserve their ordinary-path access and validation;
// only the new registered startup loader supplies retained descriptor sources.
struct PathPalsSources<'a> {
    path: &'a Path,
}
impl PalsOwnedSources for PathPalsSources<'_> {
    fn read_manifest(&self, maximum: usize) -> Result<Vec<u8>, BackendError> {
        asset::read_bounded(self.path, maximum)
    }
    fn read_graph(
        &self,
        _role: &str,
        registered_file: &str,
        _expected_sha256: [u8; 32],
        maximum: usize,
    ) -> Result<Vec<u8>, BackendError> {
        let parent = self.path.parent().ok_or_else(|| {
            fail(
                K::InvalidInput,
                S::Asset,
                "PALS export manifest has no parent",
            )
        })?;
        asset::read_bounded(&parent.join(registered_file), maximum)
    }
}
impl PalsOnnxBackend {
    /// A startup owner may retain its original read-only descriptors instead of
    /// reopening ambient model paths. These callbacks supply bytes only: the
    /// existing manifest, digest, shape, provider and session validators below
    /// remain the native admission authority.
    pub fn load_owned_sources(
        sources: Arc<dyn PalsOwnedSources>,
        expected_manifest_sha256: &str,
        runtime: OrtRuntime,
        config: PalsOnnxConfig,
    ) -> Result<Self, BackendError> {
        let mut backend = Self::load_sources_inner(
            sources.as_ref(),
            expected_manifest_sha256,
            runtime,
            config,
            None,
        )?;
        backend.owned_sources = Some(sources);
        Ok(backend)
    }
    /// The profile owner is a retained directory descriptor, rather than a
    /// checked-then-reopened path. It must outlive this backend, including any
    /// physical-completion quarantine, as part of the startup source owner.
    pub fn load_owned_sources_with_cuda_control_policy(
        sources: Arc<dyn PalsOwnedSources>,
        expected_manifest_sha256: &str,
        runtime: OrtRuntime,
        config: PalsOnnxConfig,
        policy: PalsCudaControlPolicy,
        profile: &virtual_review::VirtualReviewProfileDirectory,
    ) -> Result<Self, BackendError> {
        if config.cudnn_conv_search != PalsCuDnnConvSearch::LegacyImplicitExhaustive {
            config.validate_execution_selection(runtime.native_loading_profile(), true)?;
        }
        if !matches!(config.provider, Provider::Cuda { .. }) || config.device_public_memory {
            return Err(fail(
                K::BackendUnavailable,
                S::Admission,
                "PALS owned metadata-control requires explicit CUDA with host public memory",
            ));
        }
        profile.validate_live()?;
        let mut backend = Self::load_sources_inner(
            sources.as_ref(),
            expected_manifest_sha256,
            runtime,
            config,
            Some((policy, profile.path().to_owned())),
        )?;
        backend.owned_sources = Some(sources);
        Ok(backend)
    }
    pub fn load(
        path: &Path,
        expected_manifest_sha256: &str,
        runtime: OrtRuntime,
        config: PalsOnnxConfig,
    ) -> Result<Self, BackendError> {
        Self::load_inner(path, expected_manifest_sha256, runtime, config, None)
    }
    /// Optional, separately registered CUDA mode. CPU neural fallback remains
    /// forbidden by the complete pre-Run exact-flow placement gate.
    pub fn load_with_cuda_control_policy(
        path: &Path,
        expected_manifest_sha256: &str,
        runtime: OrtRuntime,
        config: PalsOnnxConfig,
        policy: PalsCudaControlPolicy,
        profile_root: &Path,
    ) -> Result<Self, BackendError> {
        if config.cudnn_conv_search != PalsCuDnnConvSearch::LegacyImplicitExhaustive {
            config.validate_execution_selection(runtime.native_loading_profile(), true)?;
        }
        if !matches!(config.provider, Provider::Cuda { .. }) || config.device_public_memory {
            return Err(fail(
                K::BackendUnavailable,
                S::Admission,
                "PALS metadata-control v1 requires explicit CUDA with host public memory",
            ));
        }
        let root = cuda_control::owned_profile_root(profile_root)?;
        Self::load_inner(
            path,
            expected_manifest_sha256,
            runtime,
            config,
            Some((policy, root)),
        )
    }
    fn load_inner(
        path: &Path,
        expected_manifest_sha256: &str,
        runtime: OrtRuntime,
        config: PalsOnnxConfig,
        audit: Option<(PalsCudaControlPolicy, std::path::PathBuf)>,
    ) -> Result<Self, BackendError> {
        config.validate_execution_selection(runtime.native_loading_profile(), audit.is_some())?;
        if !path.is_absolute() {
            return Err(fail(
                K::InvalidInput,
                S::Asset,
                "PALS manifest path must be absolute",
            ));
        }
        let sources = PathPalsSources { path };
        Self::load_sources_inner(&sources, expected_manifest_sha256, runtime, config, audit)
    }
    fn load_sources_inner(
        sources: &dyn PalsOwnedSources,
        expected_manifest_sha256: &str,
        runtime: OrtRuntime,
        config: PalsOnnxConfig,
        audit: Option<(PalsCudaControlPolicy, std::path::PathBuf)>,
    ) -> Result<Self, BackendError> {
        config.validate_execution_selection(runtime.native_loading_profile(), audit.is_some())?;
        let bytes = sources.read_manifest(MAX_MANIFEST_BYTES)?;
        if bytes.len() > MAX_MANIFEST_BYTES {
            return Err(fail(
                K::ResourceExhausted,
                S::Asset,
                "PALS manifest source exceeded its bound",
            ));
        }
        let manifest_digest = asset::sha256(&bytes);
        if manifest_digest != parse_sha256(expected_manifest_sha256)? {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "PALS export manifest hash mismatch",
            ));
        }
        let manifest: ExportManifest = serde_json::from_slice(&bytes).map_err(|e| {
            fail(
                K::UnsupportedModel,
                S::Asset,
                "cannot decode PALS export manifest",
            )
            .with_external_cause(CauseCode::ModelLoad, &e)
        })?;
        let epoch = validate_manifest(&manifest)?;
        let layout = manifest.resolved_layout()?;
        let rules_input_semantic_sha256 = manifest
            .rules_input_semantic_sha256
            .as_deref()
            .map(parse_sha256)
            .transpose()?;
        match (config.provider, runtime.bundle_digest()) {
            (Provider::Cpu, Some(_)) | (Provider::Cuda { .. }, None) => {
                return Err(fail(
                    K::IdentityMismatch,
                    S::Backend,
                    "PALS provider and verified native runtime bundle differ",
                ));
            }
            (Provider::Cuda { .. }, Some(_)) => {
                runtime.verify_cuda_dependencies().map_err(|mut failure| {
                    failure.detail =
                        "PALS CUDA dependency mapping audit failed before session creation";
                    failure
                })?
            }
            _ => {}
        }
        let mut sessions = BTreeMap::new();
        let mut applied_session_settings = Vec::with_capacity(manifest.graphs.len());
        let mut graph_identities = Vec::with_capacity(3);
        let mut total = 0_u64;
        let mut placements = Vec::new();
        for graph in &manifest.graphs {
            if let Some((policy, _)) = &audit {
                policy.validate_pins(manifest_digest, graph)?;
            }
            let expected_graph_sha256 = parse_sha256(&graph.sha256)?;
            let graph_bytes = sources.read_graph(
                &graph.role,
                &graph.file,
                expected_graph_sha256,
                MAX_GRAPH_BYTES,
            )?;
            if graph_bytes.len() > MAX_GRAPH_BYTES {
                return Err(fail(
                    K::ResourceExhausted,
                    S::Asset,
                    "PALS graph source exceeded its bound",
                ));
            }
            total = total.checked_add(graph_bytes.len() as u64).ok_or_else(|| {
                fail(
                    K::ResourceExhausted,
                    S::Asset,
                    "PALS graph byte budget overflow",
                )
            })?;
            if total > MAX_MODEL_BYTES {
                return Err(fail(
                    K::ResourceExhausted,
                    S::Asset,
                    "PALS export exceeds total serialized model budget",
                ));
            }
            let graph_digest = asset::sha256(&graph_bytes);
            if graph_digest != expected_graph_sha256 {
                return Err(fail(
                    K::IdentityMismatch,
                    S::Asset,
                    "PALS ONNX graph hash mismatch",
                ));
            }
            graph_identities.push(PalsGraphIdentity {
                role: graph.role.clone(),
                sha256: graph_digest,
                serialized_bytes: graph_bytes.len() as u64,
            });
            let (session, placement, applied) = load_session(
                graph_bytes,
                config,
                &graph.role,
                audit
                    .as_ref()
                    .map(|(policy, root)| (policy, root.as_path())),
            )
            .map_err(|mut failure| {
                // Bounded role/phase context identifies partial-constructor
                // failure without retaining graph bytes or retrying native init.
                // Preserve the exact pre-Run policy/configuration diagnostic;
                // only a native model-load failure receives constructor context.
                if failure
                    .cause
                    .is_some_and(|cause| cause.code == CauseCode::ModelLoad)
                {
                    failure.detail = match graph.role.as_str() {
                        "public" => "PALS public graph session creation failed",
                        "shared_pc" => "PALS shared P/C graph session creation failed",
                        "proposer" => "PALS proposer graph session creation failed",
                        "critic" => "PALS critic graph session creation failed",
                        _ => "PALS graph session creation failed",
                    };
                }
                failure
            })?;
            if let Some(placement) = placement {
                placements.push(placement);
            }
            validate_session(&session, graph, &manifest)?;
            applied_session_settings.push(applied);
            sessions.insert(graph.role.clone(), session);
        }
        let max_tokens = manifest
            .config
            .public_memory_tokens(manifest.config.max_records)
            .map_err(model_input)?;
        let memory = MemoryBank::new(
            1,
            public_owner_bytes(2 * max_tokens * 64, 2 * max_tokens * 64, max_tokens),
        )
        .map_err(public_bank_error)?;
        let validator_absence_verified = !manifest.validator_present
            && !sessions.contains_key("validator")
            && sessions.contains_key("public")
            && match layout {
                Layout::SharedPcIf => sessions.len() == 2 && sessions.contains_key("shared_pc"),
                Layout::SeparatePc => {
                    sessions.len() == 3
                        && sessions.contains_key("proposer")
                        && sessions.contains_key("critic")
                }
            };
        Ok(Self {
            public: sessions.remove("public"),
            proposer: sessions.remove("proposer"),
            critic: sessions.remove("critic"),
            shared_pc: sessions.remove("shared_pc"),
            validator_absence_verified,
            applied_session_settings: applied_session_settings.into(),
            layout,
            private_warm: None,
            #[cfg(feature = "experimental-io-binding")]
            private_cuda_warm: None,
            runtime,
            owned_sources: None,
            config,
            model_config: manifest.config,
            epoch,
            manifest_digest,
            trained: manifest.trained,
            rules_input_profile: manifest.rules_input_profile,
            rules_input_semantic_sha256,
            residency: PalsSessionResidency {
                graphs: graph_identities,
                native_sessions: if layout == Layout::SharedPcIf { 2 } else { 3 },
                layout: if layout == Layout::SharedPcIf {
                    "shared_pc_if"
                } else {
                    "separate_pc"
                }
                .into(),
                reader_initializer_bank: manifest.reader_initializer_bank,
                role_reader_weights_shared: if layout == Layout::SharedPcIf {
                    None
                } else {
                    Some(false)
                },
                native_resident_parameter_bytes: None,
                vram_peak_bytes: None,
            },
            stats: PalsBackendStats::default(),
            cuda_mapping_audit: RefCell::default(),
            cuda_control_audit: audit.map(|(policy, profile_root)| CudaControlAudit {
                policy,
                profile_root,
                initialization: placements,
                proposer_seen: false,
                critic_seen: false,
                witness: None,
                rejected: None,
            }),
            memory: Some(memory),
            cached_memory_key: None,
            active_memory: None,
            record_pages: None,
            game_generation: 0,
            active: None,
            quarantine: None,
            startup_stage_probe: None,
            public_encodes: 0,
            public_cache_hits: 0,
            #[cfg(feature = "experimental-io-binding")]
            device_memory: None,
            #[cfg(feature = "experimental-io-binding")]
            device_role: None,
            #[cfg(feature = "experimental-io-binding")]
            device_warm_role: None,
            #[cfg(feature = "experimental-io-binding")]
            cuda_warm_observer: None,
            #[cfg(feature = "experimental-io-binding")]
            cuda_record_pages: None,
        })
    }
    #[cfg(feature = "experimental-io-binding")]
    pub fn install_registered_cuda_record_pages(
        &mut self,
        registration: PalsCudaRecordPageRegistration,
    ) -> Result<(), BackendError> {
        if !self.model_config.profile.is_legacy() {
            return Err(fail(
                K::UnsupportedModel,
                S::Admission,
                "registered CUDA packing pages require their legacy graph input ABI",
            ));
        }
        let device_id = match self.config.provider {
            Provider::Cuda { device_id, .. } if device_id >= 0 => device_id as u32,
            _ => {
                return Err(fail(
                    K::BackendUnavailable,
                    S::Admission,
                    "resident PALS pages require an explicit CUDA device",
                ));
            }
        };
        let public_graph = self
            .residency
            .graphs
            .iter()
            .find(|graph| graph.role == "public")
            .ok_or_else(|| {
                fail(
                    K::IdentityMismatch,
                    S::Admission,
                    "resident PALS pages have no loaded public graph identity",
                )
            })?
            .sha256;
        let rules_semantic = self.rules_input_semantic_sha256.ok_or_else(|| {
            fail(
                K::IdentityMismatch,
                S::Admission,
                "resident PALS pages require the loaded Rules semantic input identity",
            )
        })?;
        let encoding = {
            use sha2::Digest as _;
            let mut digest = sha2::Sha256::new();
            digest.update(PALS_ENCODING_SCHEMA);
            digest.update(rules_semantic);
            digest.finalize().into()
        };
        let namespace = DevicePageNamespace {
            process_epoch: registration.process_epoch,
            model_manifest: self.manifest_digest,
            model_epoch: self.epoch,
            public_graph,
            encoding,
            frozen_epoch: registration.frozen_epoch,
            game_generation: self.game_generation,
            domain: DevicePageDomain::Cuda {
                device_id,
                runtime_sha256: self.runtime.bundle_digest().ok_or_else(|| {
                    fail(
                        K::IdentityMismatch,
                        S::Admission,
                        "resident PALS pages require the actual CUDA runtime bundle identity",
                    )
                })?,
            },
        };
        self.install_cuda_record_pages(
            registration.checked_graph,
            namespace,
            registration.limits,
            registration.invocation,
            registration.profile_root,
        )
    }
    /// Observation of this owner only, never host K/V or a new fence.
    #[cfg(feature = "experimental-io-binding")]
    pub fn cuda_record_page_snapshot(
        &self,
    ) -> Result<Option<CudaRecordPageSnapshot>, BackendError> {
        self.cuda_record_pages
            .as_ref()
            .map(device_pages_plan::native::CudaRecordPages::snapshot)
            .transpose()
    }
    fn has_active_physical_invocation(&self) -> bool {
        if self.active.is_some() || self.active_memory.is_some() {
            return true;
        }
        #[cfg(feature = "experimental-io-binding")]
        if self.device_warm_role.is_some()
            || self
                .cuda_warm_observer
                .as_ref()
                .is_some_and(PalsCudaWarmObservationHandle::has_unfinished_invocation)
        {
            return true;
        }
        #[cfg(feature = "experimental-io-binding")]
        if self
            .cuda_record_pages
            .as_ref()
            .is_some_and(device_pages_plan::native::CudaRecordPages::has_active_invocation)
        {
            return true;
        }
        false
    }
    pub fn model_epoch(&self) -> [u8; 32] {
        self.epoch
    }
    pub fn model_config(&self) -> &PalsModelConfig {
        &self.model_config
    }
    pub fn manifest_digest(&self) -> [u8; 32] {
        self.manifest_digest
    }
    pub fn runtime_binary_digest(&self) -> [u8; 32] {
        self.runtime.binary_digest()
    }
    pub fn runtime_bundle_digest(&self) -> Option<[u8; 32]> {
        self.runtime.bundle_digest()
    }
    pub fn native_loading_profile(&self) -> Option<NativeLoadingProfile> {
        self.runtime.native_loading_profile()
    }
    pub fn native_loading_profile_digest(&self) -> Option<[u8; 32]> {
        self.runtime.native_loading_profile_digest()
    }
    /// Export provenance declaration only. Loading a checkpoint cannot attest
    /// that the declared optimizer steps were actually executed.
    pub fn is_trained(&self) -> bool {
        self.trained
    }
    pub fn residency(&self) -> &PalsSessionResidency {
        &self.residency
    }
    pub fn rules_input_profile(&self) -> Option<&str> {
        self.rules_input_profile.as_deref()
    }
    pub fn rules_input_semantic_sha256(&self) -> Option<[u8; 32]> {
        self.rules_input_semantic_sha256
    }
    pub fn public_memory_witness(&self) -> Result<Option<PalsPublicMemoryWitness>, BackendError> {
        if let Some(cause) = &self.quarantine {
            return Err(cause.clone());
        }
        if self.config.device_public_memory {
            return Ok(None);
        }
        #[cfg(feature = "experimental-io-binding")]
        if self.cuda_record_pages.is_some() {
            return Ok(None);
        }
        self.record_pages
            .as_ref()
            .and_then(|pages| pages.joined())
            .or_else(|| {
                self.cached_memory_key.and_then(|key| {
                    self.memory
                        .as_ref()
                        .expect("public bank installed")
                        .get(key)
                })
            })
            .map(|memory| {
                let copy = |tensor: &Tensor<f32>| -> Result<Vec<f32>, BackendError> {
                    let (shape, values) = tensor.try_extract_tensor::<f32>().map_err(|e| {
                        native(
                            CauseCode::PolicyExtract,
                            "cannot read completed public-memory witness",
                            e,
                        )
                    })?;
                    if shape.as_ref() != [1, 2, memory.tokens as i64, 64]
                        || values.iter().any(|value| !value.is_finite())
                    {
                        return Err(fail(
                            K::NumericalFailure,
                            S::Output,
                            "completed public-memory witness shape/finite failure",
                        ));
                    }
                    Ok(values.to_vec())
                };
                let (shape, values) = memory.mask.try_extract_tensor::<bool>().map_err(|e| {
                    native(
                        CauseCode::PolicyExtract,
                        "cannot read completed public-memory mask witness",
                        e,
                    )
                })?;
                if shape.as_ref() != [1, memory.tokens as i64] {
                    return Err(fail(
                        K::NumericalFailure,
                        S::Output,
                        "completed public-memory mask witness shape differs",
                    ));
                }
                Ok(PalsPublicMemoryWitness {
                    tokens: memory.tokens,
                    memory_key: copy(&memory.memory_key)?,
                    memory_value: copy(&memory.memory_value)?,
                    memory_mask: values.to_vec(),
                })
            })
            .transpose()
    }
    pub fn quarantine_cause(&self) -> Option<&BackendError> {
        if let Some(cause) = &self.quarantine {
            return Some(cause);
        }
        #[cfg(feature = "experimental-io-binding")]
        if let Some(cause) = self
            .cuda_record_pages
            .as_ref()
            .and_then(device_pages_plan::native::CudaRecordPages::quarantine_cause)
        {
            return Some(cause);
        }
        None
    }
    /// Bounded host page ownership evidence. This is neither an ORT allocator
    /// observation nor a VRAM peak. A quarantine may retain actual active pins.
    pub fn host_public_page_snapshot(&self) -> MemoryBankSnapshot {
        if let Some(pages) = &self.record_pages {
            return pages.bank_snapshot();
        }
        self.memory
            .as_ref()
            .expect("public bank installed")
            .snapshot()
    }
    fn public_page_key(&self, content: [u8; 32]) -> MemoryKey {
        MemoryKey::PublicPage(PublicPageKey {
            kind: PublicPageKind::WholeInput,
            content: Digest(content),
            model: Digest(self.manifest_digest),
            encoding: Digest(asset::sha256(
                self.model_config.profile.encoding_schema().as_bytes(),
            )),
            precision: PrecisionProfile::Fp32,
            // No numeric mutable epoch is declared: the actual frozen 32-byte
            // checkpoint epoch is already part of the exact content digest.
            frozen_epoch: 0,
            game_generation: self.game_generation,
        })
    }
    pub fn clear_public_memory(&mut self) -> Result<(), BackendError> {
        if let Some(cause) = &self.quarantine {
            return Err(cause.clone());
        }
        let next_generation = self.game_generation.checked_add(1).ok_or_else(|| {
            fail(
                K::ResourceExhausted,
                S::Backend,
                "PALS game generation exhausted",
            )
        })?;
        self.clear_public_memory_to(next_generation)
    }
    /// Actual exclusive-worker reset target, separate from logical UCI changes.
    /// Refusal leaves the physical owner/cache namespace unchanged.
    pub fn clear_public_memory_to(&mut self, next_generation: u64) -> Result<(), BackendError> {
        if let Some(cause) = &self.quarantine {
            return Err(cause.clone());
        }
        if self.has_active_physical_invocation() {
            return Err(fail(
                K::BackendFailure,
                S::Backend,
                "PALS NewGame requires the completed physical invocation boundary",
            ));
        }
        if next_generation <= self.game_generation {
            return Err(fail(
                K::InvalidInput,
                S::Admission,
                "PALS reset target must strictly increase the physical game generation",
            ));
        }
        #[cfg(feature = "experimental-io-binding")]
        if let Some(pages) = &mut self.cuda_record_pages {
            // The resident owner checks its actual invocation/fence state before
            // the logical generation or any independent host bank is reset.
            pages.clear_for_new_game(next_generation)?;
        }
        // Refused clear is all-or-nothing. No logical reset releases a pin.
        self.memory
            .as_mut()
            .expect("public bank installed")
            .clear()
            .map_err(public_bank_error)?;
        if let Some(pages) = &mut self.record_pages {
            pages.clear()?;
        }
        self.cached_memory_key = None;
        #[cfg(feature = "experimental-io-binding")]
        {
            self.device_role = None;
            self.device_memory = None;
        }
        self.game_generation = next_generation;
        self.stats.new_game_resets = self.stats.new_game_resets.saturating_add(1);
        Ok(())
    }
    pub fn snapshot_stats(&self) -> Result<PalsBackendStats, BackendError> {
        if let Some(cause) = &self.quarantine {
            return Err(cause.clone());
        }
        if self.has_active_physical_invocation() {
            return Err(fail(
                K::BackendFailure,
                S::Backend,
                "PALS stats require the physical invocation boundary",
            ));
        }
        let mut stats = self.stats.clone();
        // Preserve the existing whole-input cache counter. Independent host
        // pages have their own snapshot rather than pretending to be one entry.
        stats.live_public_cache_entries =
            self.memory.as_ref().expect("public bank installed").len() as u64;
        #[cfg(feature = "experimental-io-binding")]
        if self.device_memory.is_some() {
            stats.live_public_cache_entries += 1;
        }
        stats.validate()?;
        Ok(stats)
    }
    pub fn config(&self) -> PalsOnnxConfig {
        self.config
    }
    /// 성공한 session 생성에서 받은 불변 원본 설정 기록. 요청 config와 구분한다.
    pub fn applied_session_settings(&self) -> Arc<[PalsAppliedSessionSettings]> {
        Arc::clone(&self.applied_session_settings)
    }
    pub fn cudnn_conv_search_diagnostic(
        &self,
    ) -> Result<Option<PalsCuDnnExecutionDiagnostic>, BackendError> {
        let selection = self.config.cudnn_conv_search;
        if selection == PalsCuDnnConvSearch::LegacyImplicitExhaustive {
            return Ok(None);
        }
        let control = self.cuda_control_inventory_digest().is_some();
        let selected_profile = match selection {
            PalsCuDnnConvSearch::ExperimentalDefault => {
                self.config
                    .validate_execution_selection(self.native_loading_profile(), control)?;
                "default-host-control-v1"
            }
            PalsCuDnnConvSearch::ExperimentalDefaultDeviceWarm => {
                self.config.validate_cuda_warm_execution_selection(
                    self.native_loading_profile(),
                    control,
                )?;
                #[cfg(feature = "experimental-io-binding")]
                if self.private_cuda_warm_capability().is_none() {
                    return Err(fail(
                        K::IdentityMismatch,
                        S::Admission,
                        "DEFAULT device-Warm requires an actual loaded Warm export capability",
                    ));
                }
                "default-device-warm-v1"
            }
            PalsCuDnnConvSearch::LegacyImplicitExhaustive => unreachable!(),
        };
        let Provider::Cuda {
            device_id,
            arena_bytes,
        } = self.config.provider
        else {
            return Err(fail(
                K::BackendUnavailable,
                S::Admission,
                "DEFAULT diagnostic requires actual CUDA provider",
            ));
        };
        let hex = |bytes: &[u8; 32]| bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
        let environment = match std::env::var_os("NVIDIA_TF32_OVERRIDE") {
            None => "absent",
            Some(v) if v == "0" => "0",
            Some(v) if v == "1" => "1",
            Some(_) => "unrecognized",
        };
        let descriptor = serde_json::json!({
            "schema":"rz-pals-cudnn-default-loaded-execution/1",
            "scope":"loaded_backend_request_not_selected_plan_or_process_exit",
            "selected_profile":selected_profile,
            "manifest_sha256":hex(&self.manifest_digest()),"model_epoch":hex(&self.model_epoch()),
            "backend_source_sha256":hex(&asset::sha256(include_bytes!("pals_onnx.rs"))),
            "runtime_sha256":hex(&self.runtime_binary_digest()),
            "runtime_bundle_sha256":self.runtime_bundle_digest().as_ref().map(hex),
            "control_inventory_sha256":self.cuda_control_inventory_digest().as_ref().map(hex),
            "provider":"CUDAExecutionProvider","device_id":device_id,"session_arena_bytes":arena_bytes,
            "intra_threads":self.config.intra_threads,"inter_threads":1,"parallel_execution":false,
            "cache_public_memory":self.config.cache_public_memory,
            "public_memory":if self.config.device_public_memory {"device"} else {"host"},
            "native_loading_profile":"eager-cuda12-cudnn9-v1","graph_optimization":self.graph_optimization(),
            "cudnn_conv_algo_search_requested":"DEFAULT","tf32_requested":false,"cuda_graph_requested":false,
            "nvidia_tf32_override_environment_request":environment
        });
        let bytes = serde_json::to_vec(&descriptor).map_err(|_| {
            fail(
                K::BackendFailure,
                S::Backend,
                "cannot serialize bounded DEFAULT execution descriptor",
            )
        })?;
        Ok(Some(PalsCuDnnExecutionDiagnostic {
            descriptor,
            descriptor_sha256: asset::sha256(&bytes),
            descriptor_encoding: "serde_json_utf8_fixed_descriptor_object",
            actual_cudnn_plan: "unknown",
            numerical_note: "unknown",
            tf32_actual: "unknown",
            tf32_retry: "unknown",
            production_default_changed: false,
            warm_success_claim: false,
            pilot_success_claim: false,
        }))
    }
    pub fn graph_optimization(&self) -> PalsGraphOptimization {
        session_optimization(self.config, self.cuda_control_audit.is_some())
    }
    pub fn verify_runtime(&self) -> Result<(), BackendError> {
        if matches!(self.config.provider, Provider::Cuda { .. }) {
            self.cuda_mapping_audit
                .borrow_mut()
                .final_audit(|| self.runtime.verify_cuda_runtime_mappings())?;
        }
        Ok(())
    }

    /// Same immutable failure latch as VerifyRuntime, returned as a separately
    /// acknowledged NN-zero control operation. Unknown physical ownership or
    /// an uncompleted first Run can never produce a success witness.
    pub fn runtime_mapping_witness(&self) -> Result<PalsNativeMappingWitness, BackendError> {
        if let Some(cause) = &self.quarantine {
            return Err(cause.clone());
        }
        if self.has_active_physical_invocation() {
            return Err(fail(
                K::BackendFailure,
                S::Backend,
                "PALS runtime observation requires physical idle",
            ));
        }
        if !matches!(self.config.provider, Provider::Cuda { .. }) {
            return Err(fail(
                K::BackendUnavailable,
                S::Backend,
                "PALS full runtime mapping observation requires explicit CUDA",
            ));
        }
        let profile = self.native_loading_profile().ok_or_else(|| {
            fail(
                K::IdentityMismatch,
                S::Backend,
                "PALS native loading profile absent",
            )
        })?;
        let bundle = self.runtime_bundle_digest().ok_or_else(|| {
            fail(
                K::IdentityMismatch,
                S::Backend,
                "PALS native runtime bundle absent",
            )
        })?;
        let digest = self.native_loading_profile_digest().ok_or_else(|| {
            fail(
                K::IdentityMismatch,
                S::Backend,
                "PALS native loading identity absent",
            )
        })?;
        let mut witness = None;
        self.cuda_mapping_audit.borrow_mut().final_audit(|| {
            let observed = self.runtime.cuda_mapping_observation(true)?;
            witness = Some(native_mapping_witness(
                observed,
                profile,
                self.runtime_binary_digest(),
                bundle,
                digest,
            )?);
            Ok(())
        })?;
        Ok(witness.expect("successful full audit produced an origin witness"))
    }
    pub fn cuda_placement_witness(&self) -> Option<&PalsCudaPlacementWitness> {
        self.cuda_control_audit
            .as_ref()
            .and_then(|audit| audit.witness.as_ref())
    }
    pub fn cuda_control_inventory_digest(&self) -> Option<[u8; 32]> {
        self.cuda_control_audit
            .as_ref()
            .map(|audit| audit.policy.inventory)
    }
    /// NN-zero control operation after both physically completed, validated
    /// startup P/C queries. Origin validation remains a separate control ACK.
    pub fn verify_cuda_placement(&mut self) -> Result<PalsCudaPlacementWitness, BackendError> {
        if let Some(cause) = &self.quarantine {
            return Err(cause.clone());
        }
        if self.has_active_physical_invocation() {
            return Err(fail(
                K::BackendFailure,
                S::Backend,
                "PALS CUDA placement requires physical idle",
            ));
        }
        let audit = self.cuda_control_audit.as_mut().ok_or_else(|| {
            fail(
                K::BackendUnavailable,
                S::Backend,
                "PALS CUDA placement has no separately registered explicit control policy",
            )
        })?;
        if let Some(cause) = &audit.rejected {
            return Err(cause.clone());
        }
        if let Some(witness) = &audit.witness {
            return Ok(witness.clone());
        }
        let verified = (|| {
            if !audit.proposer_seen || !audit.critic_seen {
                return Err(fail(
                    K::BackendUnavailable,
                    S::Backend,
                    "PALS CUDA witness requires actual validated P and C startup queries",
                ));
            }
            let public = cuda_control::finish_profile(
                self.public.as_mut().expect("public session"),
                &audit.profile_root,
            )?;
            let shared = cuda_control::finish_profile(
                self.shared_pc.as_mut().ok_or_else(|| {
                    fail(
                        K::UnsupportedModel,
                        S::Backend,
                        "PALS metadata-control supports shared P/C sessions only",
                    )
                })?,
                &audit.profile_root,
            )?;
            audit.policy.verify_profiles(
                &public,
                &shared,
                audit.initialization.clone(),
                self.runtime.binary_digest(),
                self.runtime.bundle_digest().ok_or_else(|| {
                    fail(
                        K::IdentityMismatch,
                        S::Backend,
                        "PALS CUDA runtime bundle absent",
                    )
                })?,
            )
        })();
        match verified {
            Ok(witness) => {
                audit.witness = Some(witness.clone());
                Ok(witness)
            }
            Err(error) => {
                audit.rejected = Some(error.clone());
                Err(error)
            }
        }
    }
    pub fn worker(
        mut self,
    ) -> Result<SingleWorker<PalsModelInput, Result<PalsRawOutput, BackendError>>, BackendError>
    {
        SingleWorker::spawn_with_outcome(move |input| {
            let result = self.run(input);
            match self.quarantine.as_ref() {
                Some(cause) => PhysicalRun::Quarantined(cause.clone()),
                None => PhysicalRun::Complete(result),
            }
        })
    }
    /// Explicit startup diagnostics only. Obtain the handle before moving this
    /// backend into the physical worker; the caller arms it with its startup
    /// clock and freezes it on startup success/failure. Default runs allocate
    /// no ledger and ordinary game invocations never capture stage events.
    pub fn enable_startup_stage_probe(&mut self) -> PalsStartupStageProbe {
        self.startup_stage_probe
            .get_or_insert_with(PalsStartupStageProbe::new)
            .clone()
    }
    pub fn controlled_worker(
        self,
    ) -> Result<SingleWorker<PalsNativeCommand, Result<PalsNativeResult, BackendError>>, BackendError>
    {
        self.controlled_worker_inner(None)
    }
    /// Opt-in checked metadata channel. Capture occurs on actual command
    /// return, including a latched unknown-completion failure, before its
    /// physical outcome is published. A diagnostic failure never creates a
    /// successful result or releases the backend's quarantined native owners.
    pub fn controlled_worker_with_host_record_page_observations(
        self,
    ) -> Result<HostRecordPageObservedWorker, BackendError> {
        let initial = self.host_record_page_snapshot().ok_or_else(|| {
            fail(
                K::InvalidInput,
                S::Admission,
                "host record page observation requires an enabled page policy",
            )
        })?;
        let handle = HostRecordPageObservationHandle::new(initial);
        let worker = self.controlled_worker_inner(Some(handle.clone()))?;
        Ok((worker, handle))
    }
    fn controlled_worker_inner(
        mut self,
        observer: Option<HostRecordPageObservationHandle>,
    ) -> Result<SingleWorker<PalsNativeCommand, Result<PalsNativeResult, BackendError>>, BackendError>
    {
        SingleWorker::spawn_with_outcome(move |command| {
            let boundary = match &command {
                PalsNativeCommand::Evaluate(_)
                | PalsNativeCommand::EvaluateWithEvidence(_)
                | PalsNativeCommand::EvaluatePrivateWarm(_) => {
                    HostRecordPageObservationBoundary::Evaluate
                }
                #[cfg(feature = "experimental-io-binding")]
                PalsNativeCommand::EvaluateCudaWarm(_) => {
                    HostRecordPageObservationBoundary::Evaluate
                }
                PalsNativeCommand::NewGame | PalsNativeCommand::ResetTo(_) => {
                    HostRecordPageObservationBoundary::NewGame
                }
                PalsNativeCommand::SnapshotStats => {
                    HostRecordPageObservationBoundary::SnapshotStats
                }
                #[cfg(feature = "experimental-io-binding")]
                PalsNativeCommand::SnapshotCudaRecordPages => {
                    HostRecordPageObservationBoundary::SnapshotCudaRecordPages
                }
                PalsNativeCommand::VerifyRuntime => {
                    HostRecordPageObservationBoundary::VerifyRuntime
                }
                PalsNativeCommand::VerifyCudaPlacement => {
                    HostRecordPageObservationBoundary::VerifyCudaPlacement
                }
                PalsNativeCommand::ObserveRuntimeMappings => {
                    HostRecordPageObservationBoundary::ObserveRuntimeMappings
                }
            };
            let result = match command {
                PalsNativeCommand::Evaluate(input) => {
                    self.run(input).map(PalsNativeResult::Evaluation)
                }
                PalsNativeCommand::EvaluateWithEvidence(input) => {
                    let before = self.stats.clone();
                    let outcome = self.run(input);
                    let evidence = PalsNNInputEvidence::between(
                        &before,
                        &self.stats,
                        self.quarantine_cause().is_none(),
                    );
                    let owner_accounting = PalsNativeOwnerAccounting::between(
                        &before,
                        &self.stats,
                        self.quarantine_cause().is_none(),
                        self.validator_absence_verified,
                    );
                    match self.quarantine_cause() {
                        Some(cause) => Err(cause.clone()),
                        None => Ok(PalsNativeResult::EvaluationWithEvidence {
                            outcome,
                            actual_completed_nn_inputs_delta: evidence
                                .actual_completed_nn_inputs_delta,
                            complete: evidence.complete,
                            owner_accounting,
                        }),
                    }
                }
                PalsNativeCommand::EvaluatePrivateWarm(input) => self
                    .run_private_warm(input)
                    .map(PalsNativeResult::Evaluation),
                #[cfg(feature = "experimental-io-binding")]
                PalsNativeCommand::EvaluateCudaWarm(input) => self
                    .run_cuda_private_warm(input)
                    .map(PalsNativeResult::Evaluation),
                PalsNativeCommand::NewGame => self
                    .clear_public_memory()
                    .map(|()| PalsNativeResult::NewGame),
                PalsNativeCommand::ResetTo(target) => {
                    self.clear_public_memory_to(*target)
                        .map(|()| PalsNativeResult::ResetTo {
                            game_generation: *target,
                        })
                }
                PalsNativeCommand::SnapshotStats => {
                    self.snapshot_stats().map(PalsNativeResult::Stats)
                }
                #[cfg(feature = "experimental-io-binding")]
                PalsNativeCommand::SnapshotCudaRecordPages => self
                    .cuda_record_page_snapshot()
                    .and_then(|snapshot| {
                        snapshot.ok_or_else(|| {
                            fail(
                                K::InvalidInput,
                                S::Admission,
                                "CUDA resident observation requires an installed owner",
                            )
                        })
                    })
                    .map(|snapshot| PalsNativeResult::CudaRecordPagesObserved(Box::new(snapshot))),
                PalsNativeCommand::VerifyRuntime => self
                    .verify_runtime()
                    .map(|()| PalsNativeResult::RuntimeVerified),
                PalsNativeCommand::VerifyCudaPlacement => self
                    .verify_cuda_placement()
                    .map(|witness| PalsNativeResult::CudaPlacementVerified(Box::new(witness))),
                PalsNativeCommand::ObserveRuntimeMappings => self
                    .runtime_mapping_witness()
                    .map(|witness| PalsNativeResult::RuntimeMappingsObserved(Box::new(witness))),
            };
            if let Some(observer) = &observer {
                let outcome = host_observation_outcome(&result, self.quarantine.is_some());
                observer.record(boundary, outcome, self.host_record_page_snapshot());
            }
            match self.quarantine.as_ref() {
                Some(cause) => PhysicalRun::Quarantined(cause.clone()),
                None => PhysicalRun::Complete(result),
            }
        })
    }
    pub fn run(&mut self, input: &PalsModelInput) -> Result<PalsRawOutput, BackendError> {
        if self
            .startup_stage_probe
            .as_ref()
            .is_some_and(PalsStartupStageProbe::closed)
        {
            self.startup_stage_probe = None;
        }
        let trace = self
            .startup_stage_probe
            .as_ref()
            .and_then(|probe| probe.begin_role(input.role));
        let result = self.run_inner(input, trace.as_ref(), None);
        if let Some(trace) = trace {
            trace.finish(result.is_ok());
            if trace.probe.closed() {
                self.startup_stage_probe = None;
            }
        }
        result
    }
    fn run_inner(
        &mut self,
        input: &PalsModelInput,
        trace: Option<&StartupRoleTrace>,
        private: Option<&PalsWarmInput>,
    ) -> Result<PalsRawOutput, BackendError> {
        if let Some(cause) = &self.quarantine {
            return Err(cause.clone());
        }
        #[cfg(feature = "experimental-io-binding")]
        if self.private_cuda_warm.is_some() {
            return Err(fail(
                K::UnsupportedModel,
                S::Admission,
                "explicit CUDA Warm graph requires its registered capability payload",
            ));
        }
        match (&self.private_warm, private) {
            (None, None) => {}
            (Some(capability), Some(input)) => input.validate_for(capability)?,
            _ => {
                return Err(fail(
                    K::UnsupportedModel,
                    S::Admission,
                    "PALS legacy and explicit private Warm graph invocation domains differ",
                ));
            }
        }
        self.cuda_mapping_audit.borrow().allow_run()?;
        if let Some(cause) = self
            .cuda_control_audit
            .as_ref()
            .and_then(|audit| audit.rejected.as_ref())
        {
            return Err(cause.clone());
        }
        input.validate(&self.model_config).map_err(model_input)?;
        if input.model_epoch != self.epoch {
            return Err(fail(
                K::IdentityMismatch,
                S::Admission,
                "PALS request weight epoch differs from frozen session",
            ));
        }
        if input.role == PalsRole::Validator {
            return Err(fail(
                K::UnsupportedModel,
                S::Admission,
                "validator inference is absent from P/C product export",
            ));
        }
        #[cfg(feature = "experimental-io-binding")]
        if self.cuda_record_pages.is_some() {
            let result = self.run_cuda_record_pages(input, trace);
            return self.finish_role_run(input, trace, result);
        }
        startup_enter(trace, PalsStartupBackendStage::InputPreparation);
        let prepared = input
            .prepare_tensors(&self.model_config)
            .map_err(model_input)
            .inspect_err(|_| {
                startup_return(trace, PalsStartupBackendStage::InputPreparation, false);
            })?;
        let key = prepared.public_memory_key;
        self.active = Some(
            ActiveInputs::new(prepared, input.role == PalsRole::Critic, private).inspect_err(
                |_| {
                    startup_return(trace, PalsStartupBackendStage::InputPreparation, false);
                },
            )?,
        );
        startup_return(trace, PalsStartupBackendStage::InputPreparation, true);
        self.stats.admitted_role_requests = self.stats.admitted_role_requests.saturating_add(1);
        let result = {
            #[cfg(feature = "experimental-io-binding")]
            if self.config.device_public_memory {
                self.run_device(input, key, trace)
            } else {
                self.run_active(input, key, trace)
            }
            #[cfg(not(feature = "experimental-io-binding"))]
            self.run_active(input, key, trace)
        };
        if self.quarantine.is_none() {
            self.active = None;
            // Both native Runs are synchronously fenced, or the CPU failure is
            // known complete. CUDA unknown retains this pin and its owner bank.
            self.active_memory = None;
            if let Some(pages) = &mut self.record_pages {
                pages.release_completed();
            }
            if !self.config.cache_public_memory {
                self.memory
                    .as_mut()
                    .expect("public bank installed")
                    .clear()
                    .map_err(public_bank_error)?;
                self.cached_memory_key = None;
                #[cfg(feature = "experimental-io-binding")]
                {
                    self.device_role = None;
                    self.device_memory = None;
                }
            }
        }
        self.finish_role_run(input, trace, result)
    }
    /// The resident and legacy paths share the same post-fence origin audit and
    /// actual successful role observations. This helper creates no new fence.
    fn finish_role_run(
        &mut self,
        input: &PalsModelInput,
        trace: Option<&StartupRoleTrace>,
        result: Result<PalsRawOutput, BackendError>,
    ) -> Result<PalsRawOutput, BackendError> {
        if matches!(self.config.provider, Provider::Cuda { .. }) {
            // `run_active`'s synchronous Run or `run_device`'s output sync has
            // physically completed. ORT provider images can now be required;
            // this one-time audit must never preempt session creation, and is
            // not a per-node mapping scan. A failed origin audit is latched but
            // does not mislabel this already-fenced execution as unknown.
            self.cuda_mapping_audit
                .borrow_mut()
                .after_run(result.is_ok(), || {
                    startup_enter(trace, PalsStartupBackendStage::FirstRuntimeOriginAudit);
                    let verified = self.runtime.verify_cuda_runtime_mappings();
                    startup_return(
                        trace,
                        PalsStartupBackendStage::FirstRuntimeOriginAudit,
                        verified.is_ok(),
                    );
                    verified
                })?;
        }
        if result.is_ok() {
            if let Some(audit) = &mut self.cuda_control_audit {
                audit.proposer_seen |= input.role == PalsRole::Proposer;
                audit.critic_seen |= input.role == PalsRole::Critic;
            }
        }
        result
    }
    #[cfg(feature = "experimental-io-binding")]
    fn run_device(
        &mut self,
        input: &PalsModelInput,
        key: [u8; 32],
        trace: Option<&StartupRoleTrace>,
    ) -> Result<PalsRawOutput, BackendError> {
        let device_id = match self.config.provider {
            Provider::Cuda { device_id, .. } => device_id,
            _ => unreachable!("admission validates explicit CUDA"),
        };
        let cached = self.config.cache_public_memory
            && self.device_memory.as_ref().is_some_and(|m| m.key == key);
        if cached {
            if let Some(trace) = trace {
                trace.record(
                    PalsStartupBackendStage::PublicCacheHit,
                    PalsStartupStageBoundary::Observed,
                );
            }
            self.public_cache_hits = self.public_cache_hits.saturating_add(1);
            self.stats.public_cache_hits = self.stats.public_cache_hits.saturating_add(1);
        } else {
            self.stats.public_cache_misses = self.stats.public_cache_misses.saturating_add(1);
            let tokens = self
                .model_config
                .public_memory_tokens(input.records.len())
                .map_err(model_input)?;
            // Creation performs allocations/bind-output ownership only. Install
            // it before input binding can enqueue a native transfer.
            self.device_memory = Some(
                device::DeviceMemory::new(
                    self.public.as_ref().expect("public session"),
                    key,
                    tokens,
                    device_id,
                )
                .map_err(|e| {
                    native(
                        CauseCode::TensorCreate,
                        "cannot allocate bounded PALS device memory",
                        e,
                    )
                })?,
            );
            startup_enter(trace, PalsStartupBackendStage::PublicRun);
            let result = self
                .device_memory
                .as_mut()
                .expect("device memory installed")
                .run(
                    self.public.as_mut().expect("public session"),
                    self.active.as_ref().expect("physical inputs pinned"),
                    &mut self.stats,
                );
            startup_return(trace, PalsStartupBackendStage::PublicRun, result.is_ok());
            if let Err(error) = result {
                let failure = native(
                    CauseCode::OrtRun,
                    "PALS public device binding/copy/fence failed",
                    error,
                );
                self.quarantine = Some(failure.clone());
                return Err(failure);
            }
            startup_enter(trace, PalsStartupBackendStage::PublicOutputPreparation);
            if let Err(error) = self
                .device_memory
                .as_ref()
                .expect("device memory completed")
                .validate_mask(input.records.len())
            {
                // Successful Run/fence permits discarding an invalid cache,
                // never admitting its same key as a later successful hit.
                self.device_memory = None;
                return Err(error);
            }
            self.public_encodes = self.public_encodes.saturating_add(1);
            startup_return(
                trace,
                PalsStartupBackendStage::PublicOutputPreparation,
                true,
            );
        }
        let critic = input.role == PalsRole::Critic;
        let shared = self.layout == Layout::SharedPcIf;
        let session = if shared {
            self.shared_pc.as_ref().expect("shared P/C session")
        } else if critic {
            self.critic.as_ref().expect("critic session")
        } else {
            self.proposer.as_ref().expect("proposer session")
        };
        self.device_role = Some(
            device::DeviceRole::new(
                session,
                input.candidates.len().max(1),
                input.divergence_features.len().max(1),
                critic,
                shared,
            )
            .map_err(|e| {
                native(
                    CauseCode::TensorCreate,
                    "cannot allocate bounded PALS private outputs",
                    e,
                )
            })?,
        );
        let session = if shared {
            self.shared_pc.as_mut().expect("shared P/C session")
        } else if critic {
            self.critic.as_mut().expect("critic session")
        } else {
            self.proposer.as_mut().expect("proposer session")
        };
        startup_enter(trace, PalsStartupBackendStage::PrivateRun);
        let result = self
            .device_role
            .as_mut()
            .expect("private outputs installed")
            .run(
                session,
                self.active.as_ref().expect("physical inputs pinned"),
                self.device_memory.as_ref().expect("public memory pinned"),
                &mut self.stats,
            );
        startup_return(trace, PalsStartupBackendStage::PrivateRun, result.is_ok());
        if let Err(error) = result {
            let failure = native(
                CauseCode::OrtRun,
                "PALS private device binding/copy/fence failed",
                error,
            );
            self.quarantine = Some(failure.clone());
            return Err(failure);
        }
        startup_enter(trace, PalsStartupBackendStage::PrivateOutputValidation);
        let output = self
            .device_role
            .as_ref()
            .expect("private outputs completed")
            .extract(input)?;
        output
            .decode(input, &self.model_config)
            .map_err(model_input)?;
        self.stats.validated_role_outputs = self.stats.validated_role_outputs.saturating_add(1);
        // No private role latent is fed into another invocation. After the
        // physical fence the private binding's aliases can be released.
        self.device_role = None;
        startup_return(
            trace,
            PalsStartupBackendStage::PrivateOutputValidation,
            true,
        );
        Ok(output)
    }
    fn run_active(
        &mut self,
        input: &PalsModelInput,
        key: [u8; 32],
        trace: Option<&StartupRoleTrace>,
    ) -> Result<PalsRawOutput, BackendError> {
        if self.record_pages.is_some() {
            self.prepare_record_pages(input, trace)?;
        } else {
            let page_key = self.public_page_key(key);
            self.active_memory = self
                .config
                .cache_public_memory
                .then(|| {
                    self.memory
                        .as_mut()
                        .expect("public bank installed")
                        .acquire(page_key)
                })
                .flatten();
            let cached = self.active_memory.is_some();
            if cached {
                if let Some(trace) = trace {
                    trace.record(
                        PalsStartupBackendStage::PublicCacheHit,
                        PalsStartupStageBoundary::Observed,
                    );
                }
                self.public_cache_hits = self.public_cache_hits.saturating_add(1);
                self.stats.public_cache_hits = self.stats.public_cache_hits.saturating_add(1);
            } else {
                self.stats.public_cache_misses = self.stats.public_cache_misses.saturating_add(1);
                let active = self.active.as_ref().expect("owned input installed");
                self.stats.public_nn_runs_attempted =
                    self.stats.public_nn_runs_attempted.saturating_add(1);
                startup_enter(trace, PalsStartupBackendStage::PublicRun);
                let mut values = ort::inputs!["board" => &active.board, "metadata" => &active.metadata, "records" => &active.records, "record_mask" => &active.record_mask];
                if let Some(lines) = &active.full_line {
                    values.extend(ort::inputs!["record_line_tokens" => &lines.record_line_tokens, "record_line_mask" => &lines.record_line_mask]);
                }
                let result = self
                    .public
                    .as_mut()
                    .expect("loaded public session")
                    .run(values);
                startup_return(trace, PalsStartupBackendStage::PublicRun, result.is_ok());
                let outputs = match result {
                    Ok(outputs) => outputs,
                    Err(error) => {
                        let failure = native(
                            CauseCode::OrtRun,
                            "PALS public-memory native Run failed",
                            error,
                        );
                        if matches!(self.config.provider, Provider::Cuda { .. }) {
                            self.quarantine = Some(failure.clone());
                        } else {
                            self.stats.public_nn_runs_failed_known =
                                self.stats.public_nn_runs_failed_known.saturating_add(1);
                        }
                        return Err(failure);
                    }
                };
                self.stats.public_nn_runs_completed =
                    self.stats.public_nn_runs_completed.saturating_add(1);
                self.stats.completed_nn_inputs = self.stats.completed_nn_inputs.saturating_add(1);
                startup_enter(trace, PalsStartupBackendStage::PublicOutputPreparation);
                let tokens = self
                    .model_config
                    .public_memory_tokens(input.records.len())
                    .map_err(model_input)?;
                let extract = |name: &str| -> Result<Vec<f32>, BackendError> {
                    let (shape, values) =
                        outputs[name].try_extract_tensor::<f32>().map_err(|e| {
                            native(
                                CauseCode::PolicyExtract,
                                "PALS public-memory extraction failed",
                                e,
                            )
                        })?;
                    if shape.as_ref() != [1, 2, tokens as i64, 64]
                        || values.iter().any(|v| !v.is_finite())
                    {
                        return Err(fail(
                            K::NumericalFailure,
                            S::Output,
                            "PALS public-memory output shape/finite check failed",
                        ));
                    }
                    Ok(values.to_vec())
                };
                let key_values = extract("memory_key")?;
                let key_capacity = key_values.capacity();
                let memory_key =
                    Tensor::from_array(([1, 2, tokens, 64], key_values)).map_err(|e| {
                        native(
                            CauseCode::TensorCreate,
                            "PALS public-memory ownership failed",
                            e,
                        )
                    })?;
                let value_values = extract("memory_value")?;
                let value_capacity = value_values.capacity();
                let memory_value =
                    Tensor::from_array(([1, 2, tokens, 64], value_values)).map_err(|e| {
                        native(
                            CauseCode::TensorCreate,
                            "PALS public-memory ownership failed",
                            e,
                        )
                    })?;
                let (shape, values) = outputs["memory_mask"]
                    .try_extract_tensor::<bool>()
                    .map_err(|e| {
                        native(
                            CauseCode::PolicyExtract,
                            "PALS public-memory mask extraction failed",
                            e,
                        )
                    })?;
                if shape.as_ref() != [1, tokens as i64]
                    || values[..66].iter().any(|v| !*v)
                    || values[66..].iter().filter(|v| **v).count() != input.records.len()
                {
                    return Err(fail(
                        K::NumericalFailure,
                        S::Output,
                        "PALS public-memory mask differs from admitted records",
                    ));
                }
                let mask_values = values.to_vec();
                let mask_capacity = mask_values.capacity();
                let mask = Tensor::from_array(([1, tokens], mask_values)).map_err(|e| {
                    native(
                        CauseCode::TensorCreate,
                        "PALS public-memory mask ownership failed",
                        e,
                    )
                })?;
                drop(outputs);
                let memory = PublicMemory {
                    tokens,
                    memory_key,
                    memory_value,
                    mask,
                };
                self.active_memory = Some(
                    self.memory
                        .as_mut()
                        .expect("public bank installed")
                        .insert(
                            page_key,
                            memory,
                            public_owner_bytes(key_capacity, value_capacity, mask_capacity),
                        )
                        .map_err(public_bank_error)?,
                );
                self.cached_memory_key = Some(page_key);
                self.public_encodes = self.public_encodes.saturating_add(1);
                self.stats.validated_public_outputs =
                    self.stats.validated_public_outputs.saturating_add(1);
                startup_return(
                    trace,
                    PalsStartupBackendStage::PublicOutputPreparation,
                    true,
                );
            }
        }
        let active = self.active.as_ref().expect("owned inputs remain installed");
        let memory: &PublicMemory = if let Some(pages) = &self.record_pages {
            pages
                .joined()
                .expect("page join installed after successful fence")
        } else {
            self.active_memory
                .as_ref()
                .expect("public memory installed after successful fence")
        };
        debug_assert_eq!(
            memory.tokens,
            self.model_config
                .public_memory_tokens(input.records.len())
                .expect("validated")
        );
        let critic = input.role == PalsRole::Critic;
        let shared = self.layout == Layout::SharedPcIf;
        self.stats.role_nn_runs_attempted = self.stats.role_nn_runs_attempted.saturating_add(1);
        startup_enter(trace, PalsStartupBackendStage::PrivateRun);
        let mut values = ort::inputs!["memory_key" => &memory.memory_key, "memory_value" => &memory.memory_value, "memory_mask" => &memory.mask, "candidates" => &active.candidates, "candidate_mask" => &active.candidate_mask, "query" => &active.query];
        if critic || shared {
            values.extend(ort::inputs!["divergence_features" => &active.divergences, "divergence_mask" => &active.divergence_mask]);
        }
        if shared {
            values.extend(ort::inputs!["role_is_critic" => &active.role_is_critic]);
        }
        if let Some(lines) = &active.full_line {
            values.extend(ort::inputs!["query_line_tokens" => &lines.query_line_tokens, "query_line_mask" => &lines.query_line_mask, "record_relations" => &lines.record_relations]);
        }
        if let Some(private) = &active.private_warm {
            values.extend(ort::inputs!["initial_latent" => &private.initial_latent, "warm_start" => &private.warm_start]);
        }
        let session = if shared {
            self.shared_pc.as_mut().expect("loaded shared P/C session")
        } else if critic {
            self.critic.as_mut().expect("loaded critic session")
        } else {
            self.proposer.as_mut().expect("loaded proposer session")
        };
        let result = session.run(values);
        startup_return(trace, PalsStartupBackendStage::PrivateRun, result.is_ok());
        let outputs = match result {
            Ok(outputs) => outputs,
            Err(error) => {
                let failure = native(
                    CauseCode::OrtRun,
                    "PALS private-role native Run failed",
                    error,
                );
                if matches!(self.config.provider, Provider::Cuda { .. }) {
                    self.quarantine = Some(failure.clone());
                } else {
                    self.stats.role_nn_runs_failed_known =
                        self.stats.role_nn_runs_failed_known.saturating_add(1);
                }
                return Err(failure);
            }
        };
        self.stats.role_nn_runs_completed = self.stats.role_nn_runs_completed.saturating_add(1);
        self.stats.completed_nn_inputs = self.stats.completed_nn_inputs.saturating_add(1);
        startup_enter(trace, PalsStartupBackendStage::PrivateOutputValidation);
        let extract = |name: &str, expected: &[i64]| -> Result<Vec<f32>, BackendError> {
            let (shape, values) = outputs[name].try_extract_tensor::<f32>().map_err(|e| {
                native(
                    CauseCode::PolicyExtract,
                    "PALS role output extraction failed",
                    e,
                )
            })?;
            if shape.as_ref() != expected || values.iter().any(|v| !v.is_finite()) {
                return Err(fail(
                    K::NumericalFailure,
                    S::Output,
                    "PALS role output shape/finite check failed",
                ));
            }
            Ok(values.to_vec())
        };
        let mut candidate_logits = extract(
            "candidate_logits",
            &[1, input.candidates.len().max(1) as i64],
        )?;
        candidate_logits.truncate(input.candidates.len());
        let wdl_logits = extract("wdl_logits", &[1, 3])?
            .try_into()
            .map_err(|_| fail(K::NumericalFailure, S::Output, "PALS WDL shape failed"))?;
        let private_latent = extract("private_latent", &[1, 16, 384])?;
        if shared {
            let (shape, tag) = outputs["is_critic"]
                .try_extract_tensor::<bool>()
                .map_err(|e| {
                    native(
                        CauseCode::PolicyExtract,
                        "PALS role tag extraction failed",
                        e,
                    )
                })?;
            validate_role_tag(shape.as_ref(), tag, critic)?;
        }
        let divergence_logits = if critic {
            let mut values = extract(
                "divergence_logits",
                &[1, input.divergence_features.len().max(1) as i64],
            )?;
            values.truncate(input.divergence_features.len());
            Some(values)
        } else {
            if shared {
                let values = extract(
                    "divergence_logits",
                    &[1, input.divergence_features.len().max(1) as i64],
                )?;
                if values.iter().any(|value| *value != 0.) {
                    return Err(fail(
                        K::NumericalFailure,
                        S::Output,
                        "shared proposer emitted nonzero critic output",
                    ));
                }
            }
            None
        };
        let output = PalsRawOutput {
            candidate_logits,
            wdl_logits,
            private_latent,
            divergence_logits,
            task_logits: None,
        };
        output
            .decode(input, &self.model_config)
            .map_err(model_input)?;
        self.stats.validated_role_outputs = self.stats.validated_role_outputs.saturating_add(1);
        startup_return(
            trace,
            PalsStartupBackendStage::PrivateOutputValidation,
            true,
        );
        Ok(output)
    }
}

impl Drop for PalsOnnxBackend {
    fn drop(&mut self) {
        let completion_unknown = self.quarantine.is_some();
        #[cfg(feature = "experimental-io-binding")]
        let completion_unknown = completion_unknown
            || self.cuda_record_pages.as_ref().is_some_and(
                device_pages_plan::native::CudaRecordPages::physical_completion_unknown,
            )
            || self
                .cuda_warm_observer
                .as_ref()
                .is_some_and(PalsCudaWarmObservationHandle::has_unfinished_invocation);
        if completion_unknown {
            #[cfg(feature = "experimental-io-binding")]
            if let Some(observer) = &self.cuda_warm_observer {
                observer.backend_retained_unknown();
            }
            // The worker normally retains the entire closure on quarantine. A
            // direct caller may drop its owner: retain the same resources rather
            // than using Drop as an unproven CUDA completion acknowledgement.
            for session in [
                self.public.take(),
                self.proposer.take(),
                self.critic.take(),
                self.shared_pc.take(),
            ]
            .into_iter()
            .flatten()
            {
                std::mem::forget(session);
            }
            if let Some(sources) = self.owned_sources.take() {
                std::mem::forget(sources);
            }
            if let Some(active) = self.active.take() {
                std::mem::forget(active);
            }
            if let Some(memory) = self.memory.take() {
                std::mem::forget(memory);
            }
            if let Some(pages) = self.record_pages.take() {
                // Retain independent cache owners, actual pins, unfinished join
                // backing and subset OrtValues together on unknown completion.
                std::mem::forget(pages);
            }
            if let Some(pin) = self.active_memory.take() {
                std::mem::forget(pin);
            }
            #[cfg(feature = "experimental-io-binding")]
            {
                if let Some(pages) = self.cuda_record_pages.take() {
                    // Includes packing session, pinned public blocks, pending
                    // transfers, joined outputs and all dependent allocators.
                    std::mem::forget(pages);
                }
                if let Some(role) = self.device_role.take() {
                    std::mem::forget(role);
                }
                if let Some(role) = self.device_warm_role.take() {
                    std::mem::forget(role);
                }
                if let Some(memory) = self.device_memory.take() {
                    std::mem::forget(memory);
                }
            }
        } else {
            // Allocator-backed values and bindings are destroyed before their
            // sessions on the physically complete normal shutdown path.
            #[cfg(feature = "experimental-io-binding")]
            {
                self.device_role.take();
                self.device_warm_role.take();
                self.cuda_record_pages.take();
                self.device_memory.take();
            }
            self.active_memory.take();
            self.memory.take();
            self.record_pages.take();
            self.active.take();
            #[cfg(feature = "experimental-io-binding")]
            if let Some(observer) = &self.cuda_warm_observer {
                // Publish this selected observer's terminal owner event only
                // after dependent values and all actual model sessions drop.
                // Runtime/worker/bank ownership is joined separately by Native.
                self.public.take();
                self.proposer.take();
                self.critic.take();
                self.shared_pc.take();
                observer.backend_dropped_known();
            }
        }
    }
}

#[cfg(feature = "experimental-io-binding")]
mod device {
    use super::*;
    use ort::io_binding::IoBinding;
    use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
    use rz_native_loader::ort_binding::run_fixed_binding;
    fn alias<T: ort::tensor::PrimitiveTensorElementType + std::fmt::Debug>(
        tensor: &Tensor<T>,
    ) -> ort::Result<Tensor<T>> {
        // Upgrading an owned view retains the same OrtValue, unlike Clone's
        // device copy. The alias remains inside this exclusive physical owner.
        tensor
            .view()
            .try_upgrade()
            .map_err(|_| ort::Error::new("PALS owned tensor cannot upgrade its view"))
    }
    pub(super) struct DeviceMemory {
        binding: IoBinding,
        pub key: [u8; 32],
        tokens: usize,
        pub(super) memory_key: Tensor<f32>,
        pub(super) memory_value: Tensor<f32>,
        pub(super) mask: Tensor<bool>,
        // rc.10 allocated tensors retain callbacks rather than this Allocator;
        // keep it last, after all bindings and dependent tensors.
        _allocator: Allocator,
        // Last: all native binding/value handles above have really dropped
        // before the metadata guard records their payload release.
        _warm_observation: Option<warm::cuda::CudaWarmDeviceAllocationGuard>,
    }
    impl DeviceMemory {
        pub fn new(
            session: &Session,
            key: [u8; 32],
            tokens: usize,
            device: i32,
        ) -> ort::Result<Self> {
            Self::new_inner(session, key, tokens, device, None)
        }
        pub(super) fn new_with_warm_observation(
            session: &Session,
            key: [u8; 32],
            tokens: usize,
            device: i32,
            observer: Option<&PalsCudaWarmObservationHandle>,
        ) -> Result<Self, BackendError> {
            let bytes = u64::try_from(tokens)
                .ok()
                .and_then(|n| n.checked_mul(2 * 64 * 4))
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(|| {
                    fail(
                        K::ResourceExhausted,
                        S::Admission,
                        "CUDA Warm bounded K/V allocation byte count overflows",
                    )
                })?;
            let observation = observer
                .map(|observer| observer.reserve_device_payload(bytes))
                .transpose()?;
            Self::new_inner(session, key, tokens, device, observation).map_err(|error| {
                native(
                    CauseCode::TensorCreate,
                    "cannot allocate CUDA Warm public K/V owner",
                    error,
                )
            })
        }
        fn new_inner(
            session: &Session,
            key: [u8; 32],
            tokens: usize,
            device: i32,
            observation: Option<warm::cuda::CudaWarmDeviceAllocationGuard>,
        ) -> ort::Result<Self> {
            let allocator = Allocator::new(
                session,
                MemoryInfo::new(
                    AllocationDevice::CUDA,
                    device,
                    AllocatorType::Device,
                    MemoryType::Default,
                )?,
            )?;
            let binding = session.create_binding()?;
            // Stage into options already owned by the aggregate, so a later
            // allocation failure cannot drop its allocator before values.
            let mut owned = DeviceAllocation {
                binding: Some(binding),
                key: None,
                value: None,
                allocator,
                observation,
            };
            owned.key = Some(Tensor::new(&owned.allocator, [1, 2, tokens, 64])?);
            if let Some(observation) = &mut owned.observation {
                // Bytes of the actual successfully created typed CUDA Tensor,
                // not manifest syntax or a claim about allocator/VRAM size.
                observation.tensor_created((tokens as u64) * 2 * 64 * 4);
            }
            owned.value = Some(Tensor::new(&owned.allocator, [1, 2, tokens, 64])?);
            if let Some(observation) = &mut owned.observation {
                observation.tensor_created((tokens as u64) * 2 * 64 * 4);
            }
            let mask = Tensor::new(&Allocator::default(), [1, tokens])?;
            let binding = owned
                .binding
                .as_mut()
                .expect("binding owned before tensors");
            binding.bind_output(
                "memory_key",
                alias(owned.key.as_ref().expect("key allocated"))?,
            )?;
            binding.bind_output(
                "memory_value",
                alias(owned.value.as_ref().expect("value allocated"))?,
            )?;
            binding.bind_output("memory_mask", alias(&mask)?)?;
            Ok(Self {
                binding: owned.binding.take().expect("binding installed"),
                key,
                tokens,
                memory_key: owned.key.take().expect("key allocated"),
                memory_value: owned.value.take().expect("value allocated"),
                mask,
                _allocator: owned.allocator,
                _warm_observation: owned.observation.take(),
            })
        }
        pub fn run(
            &mut self,
            session: &mut Session,
            active: &ActiveInputs,
            stats: &mut PalsBackendStats,
        ) -> ort::Result<()> {
            self.binding.bind_input("board", &active.board)?;
            self.binding.bind_input("metadata", &active.metadata)?;
            self.binding.bind_input("records", &active.records)?;
            self.binding
                .bind_input("record_mask", &active.record_mask)?;
            if let Some(lines) = &active.full_line {
                self.binding
                    .bind_input("record_line_tokens", &lines.record_line_tokens)?;
                self.binding
                    .bind_input("record_line_mask", &lines.record_line_mask)?;
            }
            self.binding.synchronize_inputs()?;
            stats.public_nn_runs_attempted = stats.public_nn_runs_attempted.saturating_add(1);
            run_fixed_binding(session, &self.binding)?;
            self.binding.synchronize_outputs()?;
            stats.public_nn_runs_completed = stats.public_nn_runs_completed.saturating_add(1);
            stats.completed_nn_inputs = stats.completed_nn_inputs.saturating_add(1);
            self.binding.clear_inputs();
            Ok(())
        }
        pub fn validate_mask(&self, records: usize) -> Result<(), BackendError> {
            let (shape, mask) = self.mask.try_extract_tensor::<bool>().map_err(|e| {
                native(
                    CauseCode::PolicyExtract,
                    "cannot extract completed PALS mask",
                    e,
                )
            })?;
            if shape.as_ref() != [1, self.tokens as i64]
                || mask[..66].iter().any(|v| !*v)
                || mask[66..].iter().filter(|v| **v).count() != records
            {
                return Err(fail(
                    K::NumericalFailure,
                    S::Output,
                    "bound PALS memory mask differs",
                ));
            }
            Ok(())
        }
    }
    struct DeviceAllocation {
        binding: Option<IoBinding>,
        key: Option<Tensor<f32>>,
        value: Option<Tensor<f32>>,
        allocator: Allocator,
        // If a later allocation/bind fails, partial native values are dropped
        // before this last guard records the actual successful allocations.
        observation: Option<warm::cuda::CudaWarmDeviceAllocationGuard>,
    }
    pub(super) struct DeviceRole {
        pub(super) binding: IoBinding,
        candidate: Tensor<f32>,
        wdl: Tensor<f32>,
        latent: Tensor<f32>,
        divergence: Option<Tensor<f32>>,
        is_critic: Option<Tensor<bool>>,
    }
    impl DeviceRole {
        pub fn new(
            session: &Session,
            candidates: usize,
            divergences: usize,
            critic: bool,
            shared: bool,
        ) -> ort::Result<Self> {
            let mut binding = session.create_binding()?;
            let cpu = Allocator::default();
            let candidate = Tensor::new(&cpu, [1, candidates])?;
            let wdl = Tensor::new(&cpu, [1_usize, 3])?;
            let latent = Tensor::new(&cpu, [1_usize, 16, 384])?;
            let divergence = (critic || shared)
                .then(|| Tensor::new(&cpu, [1, divergences]))
                .transpose()?;
            let is_critic = shared
                .then(|| Tensor::new(&cpu, Vec::<usize>::new()))
                .transpose()?;
            binding.bind_output("candidate_logits", alias(&candidate)?)?;
            binding.bind_output("wdl_logits", alias(&wdl)?)?;
            binding.bind_output("private_latent", alias(&latent)?)?;
            if let Some(value) = &divergence {
                binding.bind_output("divergence_logits", alias(value)?)?;
            }
            if let Some(value) = &is_critic {
                binding.bind_output("is_critic", alias(value)?)?;
            }
            Ok(Self {
                binding,
                candidate,
                wdl,
                latent,
                divergence,
                is_critic,
            })
        }
        pub fn run(
            &mut self,
            session: &mut Session,
            active: &ActiveInputs,
            memory: &DeviceMemory,
            stats: &mut PalsBackendStats,
        ) -> ort::Result<()> {
            self.run_joined(
                session,
                active,
                &memory.memory_key,
                &memory.memory_value,
                &memory.mask,
                stats,
            )
        }
        /// Borrow physically completed joined K/V inside the exclusive owner.
        /// This method does not certify, cache or manufacture a completion fence.
        pub fn run_joined(
            &mut self,
            session: &mut Session,
            active: &ActiveInputs,
            memory_key: &Tensor<f32>,
            memory_value: &Tensor<f32>,
            mask: &Tensor<bool>,
            stats: &mut PalsBackendStats,
        ) -> ort::Result<()> {
            if self.is_critic.is_some() {
                // CUDA If's condition has OrtMemTypeCPUInput. This is an
                // owned CPU bool scalar, never an uploaded device condition.
                self.binding
                    .bind_input("role_is_critic", &active.role_is_critic)?;
            }
            self.binding.bind_input("memory_key", memory_key)?;
            self.binding.bind_input("memory_value", memory_value)?;
            self.binding.bind_input("memory_mask", mask)?;
            self.binding.bind_input("candidates", &active.candidates)?;
            self.binding
                .bind_input("candidate_mask", &active.candidate_mask)?;
            self.binding.bind_input("query", &active.query)?;
            if let Some(lines) = &active.full_line {
                self.binding
                    .bind_input("query_line_tokens", &lines.query_line_tokens)?;
                self.binding
                    .bind_input("query_line_mask", &lines.query_line_mask)?;
                self.binding
                    .bind_input("record_relations", &lines.record_relations)?;
            }
            if self.divergence.is_some() {
                self.binding
                    .bind_input("divergence_features", &active.divergences)?;
                self.binding
                    .bind_input("divergence_mask", &active.divergence_mask)?;
            }
            self.binding.synchronize_inputs()?;
            stats.role_nn_runs_attempted = stats.role_nn_runs_attempted.saturating_add(1);
            run_fixed_binding(session, &self.binding)?;
            self.binding.synchronize_outputs()?;
            stats.role_nn_runs_completed = stats.role_nn_runs_completed.saturating_add(1);
            stats.completed_nn_inputs = stats.completed_nn_inputs.saturating_add(1);
            self.binding.clear_inputs();
            Ok(())
        }
        pub fn extract(&self, input: &PalsModelInput) -> Result<PalsRawOutput, BackendError> {
            let extract =
                |tensor: &Tensor<f32>, expected: &[i64]| -> Result<Vec<f32>, BackendError> {
                    let (shape, values) = tensor.try_extract_tensor::<f32>().map_err(|e| {
                        native(
                            CauseCode::PolicyExtract,
                            "PALS bound private output extraction failed",
                            e,
                        )
                    })?;
                    if shape.as_ref() != expected || values.iter().any(|v| !v.is_finite()) {
                        return Err(fail(
                            K::NumericalFailure,
                            S::Output,
                            "bound PALS private output differs",
                        ));
                    }
                    Ok(values.to_vec())
                };
            let mut candidate_logits =
                extract(&self.candidate, &[1, input.candidates.len().max(1) as i64])?;
            candidate_logits.truncate(input.candidates.len());
            let mut divergence_logits = self
                .divergence
                .as_ref()
                .map(|v| {
                    let mut result =
                        extract(v, &[1, input.divergence_features.len().max(1) as i64])?;
                    result.truncate(input.divergence_features.len());
                    Ok(result)
                })
                .transpose()?;
            if let Some(tag) = &self.is_critic {
                let (shape, values) = tag.try_extract_tensor::<bool>().map_err(|e| {
                    native(
                        CauseCode::PolicyExtract,
                        "bound shared role tag extraction failed",
                        e,
                    )
                })?;
                validate_role_tag(shape.as_ref(), values, input.role == PalsRole::Critic)?;
                if input.role == PalsRole::Proposer {
                    if divergence_logits
                        .as_ref()
                        .expect("shared divergence output")
                        .iter()
                        .any(|value| *value != 0.)
                    {
                        return Err(fail(
                            K::NumericalFailure,
                            S::Output,
                            "bound shared proposer emitted nonzero critic output",
                        ));
                    }
                    // Validate the padded raw slot too, before semantic trim.
                    let padded = extract(
                        self.divergence.as_ref().expect("shared divergence output"),
                        &[1, input.divergence_features.len().max(1) as i64],
                    )?;
                    if padded.iter().any(|value| *value != 0.) {
                        return Err(fail(
                            K::NumericalFailure,
                            S::Output,
                            "bound shared proposer padded critic output is nonzero",
                        ));
                    }
                    divergence_logits = None;
                }
            }
            Ok(PalsRawOutput {
                candidate_logits,
                wdl_logits: extract(&self.wdl, &[1, 3])?.try_into().expect("fixed WDL"),
                private_latent: extract(&self.latent, &[1, 16, 384])?,
                divergence_logits,
                task_logits: None,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn host_page_observation_fixture() -> HostRecordPageSnapshot {
        HostRecordPageSnapshot {
            policy: HostRecordPagePolicy::for_registered_graph([4; 32]),
            bank: MemoryBankSnapshot::default(),
            stats: HostRecordPageStats::default(),
            active_pin_count: 0,
            retained_join_bytes: 0,
            active_subset_bytes: 0,
            active_join_backing_bytes: 0,
            active_full_input_bytes: 0,
            transient_reservation_bytes: 0,
            quarantined: false,
        }
    }
    #[test]
    fn host_page_observation_keeps_actual_command_unknown_and_stale_status() {
        let initial = host_page_observation_fixture();
        let handle = HostRecordPageObservationHandle::new(initial.clone());
        let observation = handle.snapshot();
        assert_eq!(
            observation.status,
            HostRecordPageObservationStatus::Available
        );
        assert_eq!(
            observation.latest.unwrap().boundary,
            HostRecordPageObservationBoundary::BeforeWorker
        );
        let mut unknown = initial;
        unknown.quarantined = true;
        unknown.active_pin_count = 2;
        unknown.active_full_input_bytes = 123;
        handle.record(
            HostRecordPageObservationBoundary::Evaluate,
            HostRecordPageObservationOutcome::PhysicalCompletionUnknown,
            Some(unknown),
        );
        let captured = handle.snapshot();
        let latest = captured.latest.unwrap();
        assert_eq!(captured.attempted_command_ordinal, 1);
        assert_eq!(latest.command_ordinal, 1);
        assert_eq!(
            latest.outcome,
            HostRecordPageObservationOutcome::PhysicalCompletionUnknown
        );
        assert_eq!(latest.snapshot.active_full_input_bytes, 123);
        assert!(latest.snapshot.quarantined);
        let held = handle.0.latest.lock().unwrap();
        handle.record(
            HostRecordPageObservationBoundary::SnapshotStats,
            HostRecordPageObservationOutcome::ReturnedError,
            Some(host_page_observation_fixture()),
        );
        drop(held);
        let stale = handle.snapshot();
        assert_eq!(stale.status, HostRecordPageObservationStatus::Contended);
        assert_eq!(stale.attempted_command_ordinal, 2);
        assert_eq!(stale.latest.unwrap().command_ordinal, 1);
    }
    #[test]
    fn host_page_observation_poison_missing_and_ordinal_exhaustion_are_explicit() {
        let missing = HostRecordPageObservationHandle::new(host_page_observation_fixture());
        missing.record(
            HostRecordPageObservationBoundary::NewGame,
            HostRecordPageObservationOutcome::ReturnedError,
            None,
        );
        assert_eq!(
            missing.snapshot().status,
            HostRecordPageObservationStatus::Unavailable
        );
        assert!(missing.snapshot().latest.is_none());
        let exhausted = HostRecordPageObservationHandle::new(host_page_observation_fixture());
        exhausted.0.ordinal.store(u64::MAX, Ordering::Release);
        exhausted.record(
            HostRecordPageObservationBoundary::NewGame,
            HostRecordPageObservationOutcome::ReturnedOk,
            Some(host_page_observation_fixture()),
        );
        assert_eq!(
            exhausted.snapshot().status,
            HostRecordPageObservationStatus::OrdinalExhausted
        );
        let poisoned = HostRecordPageObservationHandle::new(host_page_observation_fixture());
        let other = poisoned.clone();
        assert!(std::thread::spawn(move || {
            let _guard = other.0.latest.lock().unwrap();
            panic!("poison metadata-only host page observation fixture");
        })
        .join()
        .is_err());
        let snapshot = poisoned.snapshot();
        assert_eq!(snapshot.status, HostRecordPageObservationStatus::Poisoned);
        assert!(snapshot.latest.is_some());
    }

    #[test]
    fn startup_input_metadata_is_opt_in_bounded_and_never_a_completion_ack() {
        fn metadata(trace: &StartupRoleTrace, input: &PalsModelInput) -> StartupInputMetadata {
            let prepared = input.prepare_tensors(&PalsModelConfig::baseline()).unwrap();
            let mut mask = [false; MAX_STARTUP_INPUT_RECORDS];
            mask[..prepared.record_mask.len()].copy_from_slice(&prepared.record_mask);
            StartupInputMetadata {
                request_ordinal: trace.ordinal,
                role: input.role,
                backend_owner_id: Some(7),
                input_key: prepared.input_key,
                prepared_input_key: prepared.input_key,
                public_memory_key: prepared.public_memory_key,
                logical_records: input.records.len(),
                prepared_records: prepared.record_mask.len(),
                record_mask: Some(mask),
                board: None,
                records: None,
                record_mask_tensor: None,
                memory_key: None,
                memory_value: None,
                memory_mask: None,
                public_cache_hit: None,
                public_run_returned_ok: None,
                private_run_returned_ok: None,
                role_evaluation_returned_ok: None,
            }
        }
        let legacy = PalsStartupStageProbe::new();
        assert!(legacy.start(Instant::now()));
        let trace = legacy.begin_role(PalsRole::Proposer).unwrap();
        trace.update_input(|slot| *slot = Some(metadata(&trace, &input())));
        let frozen = legacy.snapshot_and_stop();
        assert!(frozen.input_observations.is_none());
        assert!(serde_json::to_value(frozen)
            .unwrap()
            .get("input_observations")
            .is_none());

        let probe = PalsStartupStageProbe::new();
        assert!(probe.start_with_input_observations(Instant::now()));
        let mut empty = input();
        empty.records.clear();
        empty.required_critical_records.clear();
        let first = probe.begin_role(empty.role).unwrap();
        first.update_input(|slot| *slot = Some(metadata(&first, &empty)));
        first.returned(PalsStartupBackendStage::PublicRun, true);
        first.returned(PalsStartupBackendStage::PrivateRun, true);
        first.finish(true);
        let mut present = input();
        present.role = PalsRole::Critic;
        let second = probe.begin_role(present.role).unwrap();
        second.update_input(|slot| *slot = Some(metadata(&second, &present)));
        second.finish(false);
        assert!(probe.begin_role(PalsRole::Proposer).is_none());
        let frozen = probe.snapshot_and_stop();
        let observations = frozen.input_observations.as_ref().unwrap();
        assert_eq!(observations.len(), 2);
        assert_eq!(observations[0].logical_records, 0);
        assert_eq!(observations[1].logical_records, 1);
        assert_eq!(
            observations[0].prepared_records,
            observations[1].prepared_records
        );
        assert_eq!(observations[0].record_mask, Some(vec![false]));
        assert_eq!(observations[1].record_mask, Some(vec![true]));
        assert_ne!(observations[0].input_key, observations[1].input_key);
        assert_eq!(observations[0].role_evaluation_returned_ok, Some(true));
        assert_eq!(observations[1].role_evaluation_returned_ok, Some(false));
        assert!(observations
            .iter()
            .all(|input| input.completion_binding.is_none()));
        assert!(!probe.start_with_input_observations(Instant::now()));
        first.update_input(|slot| *slot = None);
        assert_eq!(
            probe.snapshot_and_stop().input_observations.unwrap().len(),
            2
        );

        let contended = PalsStartupStageProbe::new();
        assert!(contended.start_with_input_observations(Instant::now()));
        let trace = contended.begin_role(empty.role).unwrap();
        let held = contended.shared.ledger.lock().unwrap();
        trace.update_input(|slot| *slot = Some(metadata(&trace, &empty)));
        drop(held);
        let frozen = contended.snapshot_and_stop();
        assert!(frozen.recording_contended);
        assert!(frozen.input_observations.unwrap().is_empty());
    }
    #[test]
    fn startup_stage_probe_preserves_partial_stage_and_shared_clock_without_retry() {
        let dormant = PalsStartupStageProbe::new();
        let closed = dormant.snapshot_and_stop();
        assert_eq!(closed.capture_started, Some(false));
        assert!(closed.capture_closed);
        assert!(closed.events.is_empty());
        assert!(!dormant.start(Instant::now()));
        assert!(dormant.begin_role(PalsRole::Proposer).is_none());

        let probe = PalsStartupStageProbe::new();
        assert!(probe.begin_role(PalsRole::Proposer).is_none());
        let origin = Instant::now();
        assert!(probe.start(origin));
        let trace = probe.begin_role(PalsRole::Proposer).unwrap();
        startup_enter(Some(&trace), PalsStartupBackendStage::PublicRun);
        let observed = probe.snapshot_and_stop();
        assert_eq!(
            observed.snapshot_status,
            PalsStartupSnapshotStatus::Available
        );
        assert_eq!(observed.capture_started, Some(true));
        assert!(observed.capture_closed);
        assert_eq!(observed.captured_requests, 1);
        assert_eq!(observed.events.len(), 2);
        assert_eq!(observed.events[1].stage, PalsStartupBackendStage::PublicRun);
        assert_eq!(
            observed.events[1].boundary,
            PalsStartupStageBoundary::Entered
        );
        assert_eq!(observed.events[1].role, PalsRole::Proposer);
        assert_eq!(observed.events[1].request_ordinal, 1);
        assert!(observed
            .events
            .windows(2)
            .all(|pair| pair[0].elapsed_ns <= pair[1].elapsed_ns));
        assert!(observed
            .events
            .iter()
            .all(|event| event.elapsed_ns <= observed.snapshot_elapsed_ns.unwrap()));
        // A late native return cannot mutate the already frozen startup view,
        // arm a new capture, or be mistaken for an observed physical fence.
        startup_return(Some(&trace), PalsStartupBackendStage::PublicRun, true);
        trace.finish(true);
        assert_eq!(probe.snapshot_and_stop().events, observed.events);
        assert!(!probe.start(Instant::now()));
        assert!(probe.begin_role(PalsRole::Critic).is_none());
    }

    #[test]
    fn startup_stage_probe_bounds_storage_and_detaches_after_two_actual_roles() {
        let probe = PalsStartupStageProbe::new();
        assert!(probe.start(Instant::now()));
        let proposer = probe.begin_role(PalsRole::Proposer).unwrap();
        proposer.finish(true);
        let critic = probe.begin_role(PalsRole::Critic).unwrap();
        critic.finish(true);
        assert!(probe.closed());
        assert!(probe.begin_role(PalsRole::Validator).is_none());
        let captured = probe.snapshot_and_stop();
        assert_eq!(captured.captured_requests, 2);
        assert_eq!(captured.events.len(), 4);
        assert_eq!(captured.events[2].role, PalsRole::Critic);
        assert_eq!(captured.events[2].request_ordinal, 2);
        assert!(!captured.overflow);

        let probe = PalsStartupStageProbe::new();
        assert!(probe.start(Instant::now()));
        let trace = probe.begin_role(PalsRole::Proposer).unwrap();
        for _ in 0..MAX_STARTUP_STAGE_EVENTS + 10 {
            trace.record(
                PalsStartupBackendStage::PublicCacheHit,
                PalsStartupStageBoundary::Observed,
            );
        }
        let captured = probe.snapshot_and_stop();
        assert_eq!(captured.events.len(), MAX_STARTUP_STAGE_EVENTS);
        assert!(captured.overflow);
        assert_eq!(
            captured.events[0].stage,
            PalsStartupBackendStage::RoleEvaluation
        );
        assert_eq!(
            captured.events[0].boundary,
            PalsStartupStageBoundary::Entered
        );
    }

    #[test]
    fn startup_stage_probe_contention_does_not_block_physical_worker_completion() {
        use crate::worker::PhysicalPoll;
        use std::time::Duration;

        let probe = PalsStartupStageProbe::new();
        assert!(probe.start(Instant::now()));
        let held = probe.shared.ledger.lock().unwrap();
        let worker_probe = probe.clone();
        let mut worker = SingleWorker::spawn(move |input: &u8| {
            let trace = worker_probe.begin_role(PalsRole::Proposer).unwrap();
            startup_enter(Some(&trace), PalsStartupBackendStage::PublicRun);
            trace.finish(true);
            *input
        })
        .unwrap();
        let mut lease = worker.submit(42).unwrap();
        let until = Instant::now() + Duration::from_secs(2);
        loop {
            match lease.poll() {
                PhysicalPoll::Ready(value) => {
                    assert_eq!(value, 42);
                    break;
                }
                PhysicalPoll::Pending if Instant::now() < until => std::thread::yield_now(),
                _ => panic!("diagnostic contention changed the completed mock physical result"),
            }
        }
        let missing = probe.snapshot_and_stop();
        assert_eq!(
            missing.snapshot_status,
            PalsStartupSnapshotStatus::Contended
        );
        assert_eq!(missing.capture_started, None);
        assert!(missing.recording_contended);
        assert!(missing.events.is_empty());
        drop(held);
        loop {
            match worker.try_shutdown() {
                std::task::Poll::Ready(result) => {
                    result.unwrap();
                    break;
                }
                std::task::Poll::Pending if Instant::now() < until => std::thread::yield_now(),
                _ => panic!("completed mock worker did not join within the finite fixture window"),
            }
        }
        let observed = probe.snapshot_and_stop();
        assert_eq!(
            observed.snapshot_status,
            PalsStartupSnapshotStatus::Available
        );
        assert!(observed.events.is_empty());
        assert!(observed.recording_contended);
    }

    #[test]
    fn startup_stage_probe_poison_is_explicit_and_does_not_rearm() {
        let probe = PalsStartupStageProbe::new();
        let other = probe.clone();
        assert!(std::thread::spawn(move || {
            let _held = other.shared.ledger.lock().unwrap();
            panic!("poison metadata-only startup diagnostic fixture");
        })
        .join()
        .is_err());
        assert!(!probe.start(Instant::now()));
        assert!(probe.begin_role(PalsRole::Proposer).is_none());
        let snapshot = probe.snapshot_and_stop();
        assert_eq!(
            snapshot.snapshot_status,
            PalsStartupSnapshotStatus::Poisoned
        );
        assert!(snapshot.recording_poisoned);
        assert!(snapshot.events.is_empty());
        assert!(!probe.start(Instant::now()));
    }

    #[test]
    fn shim_product_admission_requires_explicit_host_cuda_control() {
        let cpu = PalsOnnxConfig::cpu();
        let shim = Some(NativeLoadingProfile::CuDnnShimLazyV1);
        validate_native_loading_path(None, cpu, false).unwrap();
        assert!(validate_native_loading_path(shim, cpu, true).is_err());
        let mut cuda = cpu;
        cuda.provider = Provider::Cuda {
            device_id: 0,
            arena_bytes: 2 * 1024 * 1024 * 1024,
        };
        validate_native_loading_path(Some(NativeLoadingProfile::EagerCuda12Cudnn9V1), cuda, false)
            .unwrap();
        assert!(validate_native_loading_path(shim, cuda, false).is_err());
        validate_native_loading_path(shim, cuda, true).unwrap();
        cuda.device_public_memory = true;
        assert!(validate_native_loading_path(shim, cuda, true).is_err());
    }

    #[test]
    fn conv_default_admission_requires_eager_host_cuda_control() {
        let eager = Some(NativeLoadingProfile::EagerCuda12Cudnn9V1);
        let mut config = PalsOnnxConfig::cpu();
        assert_eq!(
            config.cudnn_conv_search,
            PalsCuDnnConvSearch::LegacyImplicitExhaustive
        );
        config.cudnn_conv_search = PalsCuDnnConvSearch::ExperimentalDefault;
        assert!(config.validate().is_err());
        assert!(config.validate_execution_selection(eager, true).is_err());
        config.provider = Provider::Cuda {
            device_id: 0,
            arena_bytes: 2 * 1024 * 1024 * 1024,
        };
        assert!(config.validate_execution_selection(eager, false).is_err());
        assert!(config.validate_execution_selection(None, true).is_err());
        assert!(config
            .validate_execution_selection(Some(NativeLoadingProfile::CuDnnShimLazyV1), true)
            .is_err());
        config.validate_execution_selection(eager, true).unwrap();
        config.device_public_memory = true;
        assert!(config.validate().is_err());
        assert!(config.validate_execution_selection(eager, true).is_err());
    }

    #[test]
    fn conv_default_device_warm_is_a_separate_admission() {
        let eager = Some(NativeLoadingProfile::EagerCuda12Cudnn9V1);
        let mut config = PalsOnnxConfig::cpu();
        config.cudnn_conv_search = PalsCuDnnConvSearch::ExperimentalDefaultDeviceWarm;
        assert!(config
            .validate_cuda_warm_execution_selection(eager, true)
            .is_err());
        config.provider = Provider::Cuda {
            device_id: 0,
            arena_bytes: 256 * 1024 * 1024,
        };
        assert!(config
            .validate_cuda_warm_execution_selection(eager, true)
            .is_err());
        config.device_public_memory = true;
        assert!(config.validate_execution_selection(eager, true).is_err());
        assert!(config
            .validate_cuda_warm_execution_selection(eager, false)
            .is_err());
        assert!(config
            .validate_cuda_warm_execution_selection(
                Some(NativeLoadingProfile::CuDnnShimLazyV1),
                true
            )
            .is_err());
        assert!(config
            .validate_cuda_warm_execution_selection(None, true)
            .is_err());
        assert_eq!(
            config
                .validate_cuda_warm_execution_selection(eager, true)
                .is_ok(),
            cfg!(feature = "experimental-io-binding")
        );
        config.cudnn_conv_search = PalsCuDnnConvSearch::ExperimentalDefault;
        assert!(config
            .validate_cuda_warm_execution_selection(eager, true)
            .is_err());
    }

    fn shim_observation() -> NativeMappingObservation {
        let profile = NativeLoadingProfile::CuDnnShimLazyV1;
        let required_nvidia_files: Vec<_> = profile
            .eager_indices()
            .iter()
            .map(|&index| rz_native_loader::NVIDIA_LOAD_ORDER[index])
            .collect();
        NativeMappingObservation {
            profile,
            declared_nvidia_files: 16,
            mapped_nvidia_files: required_nvidia_files.clone(),
            required_nvidia_files,
            deferred_nvidia_not_mapped: rz_native_loader::NVIDIA_LOAD_ORDER[8..15].to_vec(),
            mapped_ort_files: Some(rz_native_loader::ORT_LIBRARY_NAMES.to_vec()),
        }
    }

    #[test]
    fn full_mapping_witness_preserves_actual_partial_residency_and_profile() {
        let profile = NativeLoadingProfile::CuDnnShimLazyV1;
        let digest = asset::sha256(profile.canonical_descriptor().as_bytes());
        let witness =
            native_mapping_witness(shim_observation(), profile, [7; 32], [8; 32], digest).unwrap();
        assert_eq!(witness.loading_profile, "experimental-cudnn-shim-lazy-v1");
        assert_eq!(witness.mapped_nvidia_files.len(), 9);
        assert_eq!(witness.deferred_nvidia_not_mapped.len(), 7);
        assert_eq!(witness.mapped_ort_files.len(), 3);
        assert_eq!(witness.loading_profile_sha256, digest);
        let serialized = serde_json::to_value(&witness).unwrap();
        assert_eq!(
            serialized["scope"],
            "exclusive_physical_worker_full_runtime_origin"
        );
        assert_eq!(
            serialized["schema"],
            "rovezero.pals-native-mapping-witness.v1"
        );
    }

    #[test]
    fn full_mapping_witness_rejects_wrong_scope_profile_and_missing_root() {
        let profile = NativeLoadingProfile::CuDnnShimLazyV1;
        let digest = asset::sha256(profile.canonical_descriptor().as_bytes());
        for invalid in 0..7 {
            let mut observed = shim_observation();
            match invalid {
                0 => observed.mapped_ort_files = None,
                1 => {
                    observed.mapped_ort_files.as_mut().unwrap().pop();
                }
                2 => observed.profile = NativeLoadingProfile::EagerCuda12Cudnn9V1,
                3 => {
                    observed.mapped_nvidia_files.pop();
                }
                4 => {
                    observed.required_nvidia_files.pop();
                }
                5 => observed.mapped_nvidia_files.push("libcudnn.so.8"),
                _ => {
                    observed.deferred_nvidia_not_mapped.pop();
                }
            }
            assert!(native_mapping_witness(observed, profile, [7; 32], [8; 32], digest).is_err());
        }
        assert!(
            native_mapping_witness(shim_observation(), profile, [7; 32], [8; 32], [0; 32]).is_err()
        );
    }

    #[test]
    fn explicit_cuda_control_disables_rewrites_without_changing_cpu_or_strict_cuda() {
        let cpu = PalsOnnxConfig::cpu();
        assert_eq!(
            session_optimization(cpu, false),
            PalsGraphOptimization::Level1
        );
        assert_eq!(
            session_optimization(cpu, true),
            PalsGraphOptimization::Level1
        );
        let mut cuda = cpu;
        cuda.provider = Provider::Cuda {
            device_id: 0,
            arena_bytes: 2 * 1024 * 1024 * 1024,
        };
        assert_eq!(
            session_optimization(cuda, false),
            PalsGraphOptimization::Level1
        );
        assert_eq!(
            session_optimization(cuda, true),
            PalsGraphOptimization::Disable
        );
        assert_eq!(
            serde_json::to_value(PalsGraphOptimization::Disable).unwrap(),
            "disable"
        );
    }
    use crate::pals_model::{PalsCandidateToken, PalsRecordToken};
    fn input() -> PalsModelInput {
        PalsModelInput {
            role: PalsRole::Proposer,
            board: vec![0; 64],
            metadata: [0.; 16],
            records: vec![PalsRecordToken {
                record_id: 1,
                revision: 2,
                critical: true,
                features: [0.; 16],
            }],
            required_critical_records: vec![1],
            candidates: vec![PalsCandidateToken {
                from: 12,
                to: 28,
                promotion: 0,
            }],
            divergence_features: vec![],
            query: [0.; 16],
            situation_revision: 3,
            model_epoch: [4; 32],
            history_digest: [5; 32],
            full_line: None,
        }
    }
    #[test]
    fn v2_manifest_requires_complete_profile_identity_and_exact_tensor_abi() {
        use crate::pals_model::PALS_MODEL_SEMANTICS_V2;
        let mut m = manifest();
        m.config = PalsModelConfig::full_line_interaction_v2();
        assert!(validate_manifest(&m).is_err());
        m.model_profile = Some(m.config.profile);
        m.model_semantics = Some(PALS_MODEL_SEMANTICS_V2.to_owned());
        m.encoding_schema = Some(m.config.profile.encoding_schema().to_owned());
        assert!(validate_manifest(&m).is_err()); // Legacy tensor ABI is insufficient.
        for graph in &mut m.graphs {
            let inputs = if graph.role == "public" {
                public_inputs_for(m.config.profile)
            } else {
                role_inputs_for(graph.role == "critic", m.config.profile)
            };
            graph.inputs = inputs
                .into_iter()
                .map(|v| TensorManifest {
                    name: v.name.to_owned(),
                    dtype: match v.dtype {
                        TensorElementType::Float32 => "FLOAT",
                        TensorElementType::Int64 => "INT64",
                        TensorElementType::Bool => "BOOL",
                        _ => unreachable!(),
                    }
                    .to_owned(),
                    shape: v
                        .shape
                        .iter()
                        .enumerate()
                        .map(|(axis, &dim)| {
                            if dim == -1 {
                                serde_json::json!(format!("dynamic_{axis}"))
                            } else {
                                serde_json::json!(dim)
                            }
                        })
                        .collect(),
                })
                .collect();
        }
        validate_manifest(&m).unwrap();
        m.model_profile = Some(PalsModelProfile::FullLineV2);
        assert!(validate_manifest(&m).is_err());
        m.model_profile = Some(m.config.profile);
        m.graphs[0]
            .inputs
            .iter_mut()
            .find(|v| v.name == "record_line_tokens")
            .unwrap()
            .shape[2] = 255.into();
        assert!(validate_manifest(&m).is_err());
    }
    #[test]
    fn public_cache_role_neutral_private_query_and_candidates_are_not_memory() {
        let a = input();
        let mut b = a.clone();
        b.role = PalsRole::Critic;
        b.divergence_features.push([1.; 8]);
        b.query[0] = 9.;
        b.candidates.clear();
        let key = |input: &PalsModelInput| {
            input
                .public_memory_key(&PalsModelConfig::baseline())
                .unwrap()
        };
        assert_eq!(key(&a), key(&b));
        b.records[0].revision += 1;
        assert_ne!(key(&a), key(&b));
        b = a.clone();
        b.model_epoch[0] += 1;
        assert_ne!(key(&a), key(&b));
        b = a.clone();
        b.metadata[0] = -0.;
        assert_ne!(key(&a), key(&b));
    }
    #[test]
    fn descriptor_rejects_fixed_shapes_and_wrong_dtype() {
        let expected = [spec("board", TensorElementType::Int64, &[-1, 64])];
        let mut fields = vec![TensorManifest {
            name: "board".into(),
            dtype: "INT64".into(),
            shape: vec!["batch".into(), 64.into()],
        }];
        validate_declared_interface(&fields, &expected).unwrap();
        fields[0].shape[0] = 1.into();
        assert!(validate_declared_interface(&fields, &expected).is_err());
        fields[0].shape[0] = "batch".into();
        fields[0].dtype = "FLOAT".into();
        assert!(validate_declared_interface(&fields, &expected).is_err());
    }

    fn manifest() -> ExportManifest {
        let tensor = |v: Interface| {
            serde_json::json!({
                "name":v.name,
                "dtype":match v.dtype { TensorElementType::Float32 => "FLOAT", TensorElementType::Int64 => "INT64", TensorElementType::Bool => "BOOL", _ => unreachable!() },
                "shape":v.shape.iter().enumerate().map(|(axis, dim)| if *dim == -1 { serde_json::json!(format!("dynamic_{axis}")) } else { serde_json::json!(dim) }).collect::<Vec<_>>()
            })
        };
        let graphs = ["public", "proposer", "critic"].map(|role| serde_json::json!({
            "role":role,"file":if role == "public" { "public_memory.onnx".to_owned() } else {format!("role_{role}.onnx")},
            "sha256":"01".repeat(32),"opset":17,
            "inputs":if role == "public" {public_inputs()} else {role_inputs(role=="critic")}.into_iter().map(tensor).collect::<Vec<_>>(),
            "outputs":if role == "public" {memory_outputs()} else {role_outputs(role=="critic")}.into_iter().map(tensor).collect::<Vec<_>>()
        }));
        serde_json::from_value(serde_json::json!({
            "schema":PALS_MODEL_SCHEMA,"config":PalsModelConfig::baseline(),"checkpoint_sha256":"02".repeat(32),
            "trained":false,"training_steps":0,"roles":["proposer","critic"],"validator_present":false,"graphs":graphs,
            "runtime":{"onnxruntime":"1.22.0","rust_ort":"2.0.0-rc.10","compatibility":"requires_actual_numeric_check"},
            "candidate_promotion":{"0":"none","1":"queen","2":"rook","3":"bishop","4":"knight"},
            "wdl_perspective":"input_side_to_move","task_names":V_TASK_NAMES,"numeric_status":"not_run","cuda_status":"not_run"
        })).unwrap()
    }
    #[test]
    fn product_manifest_rejects_v_duplicate_roles_and_path_escape() {
        validate_manifest(&manifest()).unwrap();
        let mut m = manifest();
        m.validator_present = true;
        assert!(validate_manifest(&m).is_err());
        m = manifest();
        m.roles.push("validator".into());
        assert!(validate_manifest(&m).is_err());
        m = manifest();
        m.graphs[1].role = "critic".into();
        assert!(validate_manifest(&m).is_err());
        m = manifest();
        m.graphs[0].file = "../public_memory.onnx".into();
        assert!(validate_manifest(&m).is_err());
        m = manifest();
        m.task_names.swap(0, 1);
        assert!(validate_manifest(&m).is_err());
    }
    #[test]
    fn execution_capability_never_changes_cuda_request_to_cpu() {
        PalsOnnxConfig::cpu().validate().unwrap();
        let mut config = PalsOnnxConfig::cpu();
        config.device_public_memory = true;
        assert!(config.validate().is_err());
        config.device_public_memory = false;
        config.provider = Provider::Cuda {
            device_id: 0,
            arena_bytes: 0,
        };
        assert!(config.validate().is_err());
        let mut m = manifest();
        m.trained = true;
        assert!(validate_manifest(&m).is_err());
    }
    #[test]
    fn shared_layout_requires_semantic_pin_audit_scalar_and_role_fence() {
        let mut m = manifest();
        m.schema = "rovezero.pals-model.v2".into();
        m.model_semantics = Some(PALS_MODEL_SCHEMA.into());
        m.layout = Some("shared_pc_if".into());
        m.layout_revision = Some(1);
        m.role_batching = Some("one_scalar_role_per_physical_batch".into());
        m.rules_input_profile = Some("rz-pals-rules-fields-v1".into());
        m.rules_input_semantic_sha256 = Some("03".repeat(32));
        m.rules_encoder_source_sha256 = Some("06".repeat(32));
        m.graphs.truncate(1);
        let tensors = |values: Vec<Interface>| {
            values
                .into_iter()
                .map(|v| TensorManifest {
                    name: v.name.into(),
                    dtype: match v.dtype {
                        TensorElementType::Float32 => "FLOAT",
                        TensorElementType::Int64 => "INT64",
                        TensorElementType::Bool => "BOOL",
                        _ => unreachable!(),
                    }
                    .into(),
                    shape: v
                        .shape
                        .iter()
                        .enumerate()
                        .map(|(axis, dim)| {
                            if *dim == -1 {
                                serde_json::json!(format!("dynamic_{axis}"))
                            } else {
                                serde_json::json!(dim)
                            }
                        })
                        .collect(),
                })
                .collect()
        };
        m.graphs.push(GraphManifest {
            file: "shared_pc_if.onnx".into(),
            sha256: "04".repeat(32),
            role: "shared_pc".into(),
            inputs: tensors(shared_inputs()),
            outputs: tensors(shared_outputs()),
            opset: 17,
        });
        m.reader_initializer_bank = Some(ReaderInitializerBank {
            scope: "onnx_serialized_initializers_only".into(),
            layout_revision: 1,
            shared_parameters: vec![SharedParameter {
                parameter: "reader.weight".into(),
                initializer: "reader_weight".into(),
                bytes: 4,
                sha256: "05".repeat(32),
                source_parameter_sha256: "05".repeat(32),
                shape: vec![1],
            }],
            unique_shared_parameters: 1,
            shared_weight_bytes: 4,
            if_routes: 6,
            branch_local_initializers: 0,
            branch_operations: (0..6)
                .flat_map(|route| {
                    ["then_branch", "else_branch"].map(|branch| BranchOperations {
                        route: format!("route_{route}"),
                        branch: branch.into(),
                        nodes: 1,
                    })
                })
                .collect(),
            inactive_private_execution: "requires_runtime_profile".into(),
            ort_prepack_copies: "unknown".into(),
            device_residency_sharing: "unknown".into(),
        });
        validate_manifest(&m).unwrap();
        m.rules_encoder_source_sha256 = None;
        assert!(validate_manifest(&m).is_err());
        m.rules_encoder_source_sha256 = Some("06".repeat(32));
        m.rules_input_semantic_sha256 = None;
        assert!(validate_manifest(&m).is_err());
        m.rules_input_semantic_sha256 = Some("03".repeat(32));
        m.graphs[1].inputs[0].shape.push(1.into());
        assert!(validate_manifest(&m).is_err());
        m.graphs[1].inputs[0].shape.clear();
        m.reader_initializer_bank
            .as_mut()
            .unwrap()
            .shared_weight_bytes = 8;
        assert!(validate_manifest(&m).is_err());
        validate_role_tag(&[], &[true], true).unwrap();
        assert!(validate_role_tag(&[1], &[true], true).is_err());
        assert!(validate_role_tag(&[], &[false], true).is_err());
    }
    #[test]
    fn native_stats_separate_requests_graphs_inputs_and_validated_outputs() {
        let mut stats = PalsBackendStats {
            admitted_role_requests: 2,
            public_cache_hits: 1,
            public_cache_misses: 1,
            public_nn_runs_attempted: 1,
            public_nn_runs_completed: 1,
            role_nn_runs_attempted: 2,
            role_nn_runs_completed: 2,
            completed_nn_inputs: 3,
            validated_public_outputs: 1,
            validated_role_outputs: 2,
            new_game_resets: 2,
            live_public_cache_entries: 1,
            ..Default::default()
        };
        stats.validate().unwrap();
        stats.completed_nn_inputs = 2;
        assert!(stats.validate().is_err());
        stats.completed_nn_inputs = 3;
        stats.validated_role_outputs = 3;
        assert!(stats.validate().is_err());
        stats.validated_role_outputs = 2;
        stats.live_public_cache_entries = 2;
        assert!(stats.validate().is_err());
    }
    #[test]
    fn host_page_observation_preserves_known_role_failure_inside_evidence() {
        let error = fail(K::NumericalFailure, S::Output, "known role output failure");
        let legacy = Err(error.clone());
        let evidence = Ok(PalsNativeResult::EvaluationWithEvidence {
            outcome: Err(error),
            actual_completed_nn_inputs_delta: Some(1),
            complete: true,
            owner_accounting: None,
        });
        for result in [&legacy, &evidence] {
            assert_eq!(
                host_observation_outcome(result, false),
                HostRecordPageObservationOutcome::ReturnedError
            );
            assert_eq!(
                host_observation_outcome(result, true),
                HostRecordPageObservationOutcome::PhysicalCompletionUnknown
            );
        }
        assert_eq!(
            host_observation_outcome(&Ok(PalsNativeResult::NewGame), false),
            HostRecordPageObservationOutcome::ReturnedOk
        );
    }

    #[test]
    fn native_nn_evidence_counts_public_miss_and_cache_hit_without_another_command() {
        let before = PalsBackendStats::default();
        let miss = PalsBackendStats {
            admitted_role_requests: 1,
            public_cache_misses: 1,
            public_nn_runs_attempted: 1,
            public_nn_runs_completed: 1,
            role_nn_runs_attempted: 1,
            role_nn_runs_completed: 1,
            completed_nn_inputs: 2,
            ..Default::default()
        };
        assert_eq!(
            PalsNNInputEvidence::between(&before, &miss, true),
            PalsNNInputEvidence {
                actual_completed_nn_inputs_delta: Some(2),
                complete: true,
            }
        );
        let hit = PalsBackendStats {
            admitted_role_requests: 2,
            public_cache_hits: 1,
            live_public_cache_entries: 1,
            role_nn_runs_attempted: 2,
            role_nn_runs_completed: 2,
            completed_nn_inputs: 3,
            ..miss.clone()
        };
        assert_eq!(
            PalsNNInputEvidence::between(&miss, &hit, true),
            PalsNNInputEvidence {
                actual_completed_nn_inputs_delta: Some(1),
                complete: true,
            }
        );
        // A cache gauge can retire while cumulative NN evidence stays exact.
        let retired = PalsBackendStats {
            live_public_cache_entries: 0,
            ..hit.clone()
        };
        assert_eq!(
            PalsNNInputEvidence::between(&hit, &retired, true).actual_completed_nn_inputs_delta,
            Some(0)
        );
    }
    #[test]
    fn native_nn_evidence_known_failure_preserves_partial_graph_completion() {
        let failed = PalsBackendStats {
            admitted_role_requests: 1,
            public_cache_misses: 1,
            public_nn_runs_attempted: 1,
            public_nn_runs_completed: 1,
            role_nn_runs_attempted: 1,
            role_nn_runs_failed_known: 1,
            completed_nn_inputs: 1,
            ..Default::default()
        };
        assert_eq!(
            PalsNNInputEvidence::between(&PalsBackendStats::default(), &failed, true),
            PalsNNInputEvidence {
                actual_completed_nn_inputs_delta: Some(1),
                complete: true
            }
        );
        assert_eq!(
            PalsNNInputEvidence::between(&PalsBackendStats::default(), &failed, false),
            PalsNNInputEvidence {
                actual_completed_nn_inputs_delta: None,
                complete: false
            }
        );
    }
    #[test]
    fn native_nn_evidence_rejects_saturation_regression_and_inconsistent_totals() {
        let valid = PalsBackendStats {
            admitted_role_requests: 1,
            public_cache_misses: 1,
            public_nn_runs_attempted: 1,
            public_nn_runs_completed: 1,
            role_nn_runs_attempted: 1,
            role_nn_runs_completed: 1,
            completed_nn_inputs: 2,
            ..Default::default()
        };
        let mut bad = valid.clone();
        bad.role_nn_runs_attempted = u64::MAX;
        assert!(!PalsNNInputEvidence::between(&valid, &bad, true).complete);
        bad = valid.clone();
        bad.public_nn_runs_attempted = u64::MAX;
        assert!(!PalsNNInputEvidence::between(&bad, &bad, true).complete);
        bad = valid.clone();
        bad.completed_nn_inputs = 1;
        assert!(!PalsNNInputEvidence::between(&valid, &bad, true).complete);
        assert!(!PalsNNInputEvidence::between(&valid, &PalsBackendStats::default(), true).complete);
        bad = valid.clone();
        bad.admitted_role_requests = 2;
        bad.role_nn_runs_attempted = 2;
        bad.role_nn_runs_failed_known = 1;
        // Both frames separately validate; a failed-known counter regression
        // still makes the per-invocation stage history incomplete.
        let regressed_failure = PalsBackendStats {
            role_nn_runs_completed: 2,
            role_nn_runs_failed_known: 0,
            completed_nn_inputs: 3,
            ..bad.clone()
        };
        bad.validate().unwrap();
        regressed_failure.validate().unwrap();
        assert!(!PalsNNInputEvidence::between(&bad, &regressed_failure, true).complete);
    }
    #[test]
    fn native_owner_accounting_retains_exact_miss_hit_and_private_counts() {
        let before = PalsBackendStats::default();
        let miss = PalsBackendStats {
            admitted_role_requests: 1,
            public_cache_misses: 1,
            public_nn_runs_attempted: 1,
            public_nn_runs_completed: 1,
            role_nn_runs_attempted: 1,
            role_nn_runs_completed: 1,
            completed_nn_inputs: 2,
            ..Default::default()
        };
        let actual = PalsNativeOwnerAccounting::between(&before, &miss, true, true).unwrap();
        assert_eq!(actual.public_cache_hits(), 0);
        assert_eq!(actual.public_cache_misses(), 1);
        assert_eq!(actual.public_nn_runs_attempted(), 1);
        assert_eq!(actual.public_nn_runs_completed(), 1);
        assert_eq!(actual.role_nn_runs_attempted(), 1);
        assert_eq!(actual.role_nn_runs_completed(), 1);
        assert_eq!(actual.validator_nn_runs_attempted(), 0);
        assert_eq!(actual.validator_nn_runs_completed(), 0);
        let hit = PalsBackendStats {
            admitted_role_requests: 2,
            public_cache_hits: 1,
            role_nn_runs_attempted: 2,
            role_nn_runs_completed: 2,
            completed_nn_inputs: 3,
            ..miss.clone()
        };
        let actual = PalsNativeOwnerAccounting::between(&miss, &hit, true, true).unwrap();
        assert_eq!(actual.public_cache_hits(), 1);
        assert_eq!(actual.public_cache_misses(), 0);
        assert_eq!(actual.public_nn_runs_attempted(), 0);
        assert_eq!(actual.public_nn_runs_completed(), 0);
        assert_eq!(actual.role_nn_runs_completed(), 1);
        assert_eq!(actual.validator_nn_runs_failed_known(), 0);
    }
    #[test]
    fn native_owner_accounting_preserves_known_partial_failure_and_public_pages() {
        let before = PalsBackendStats::default();
        let failure = PalsBackendStats {
            admitted_role_requests: 1,
            public_cache_misses: 1,
            public_nn_runs_attempted: 1,
            public_nn_runs_completed: 1,
            role_nn_runs_attempted: 1,
            role_nn_runs_failed_known: 1,
            completed_nn_inputs: 1,
            ..Default::default()
        };
        let actual = PalsNativeOwnerAccounting::between(&before, &failure, true, true).unwrap();
        assert_eq!(actual.public_nn_runs_completed(), 1);
        assert_eq!(actual.role_nn_runs_completed(), 0);
        assert_eq!(actual.role_nn_runs_failed_known(), 1);
        let pages_failure = PalsBackendStats {
            public_nn_runs_attempted: 3,
            public_nn_runs_completed: 2,
            public_nn_runs_failed_known: 1,
            role_nn_runs_attempted: 0,
            role_nn_runs_failed_known: 0,
            completed_nn_inputs: 2,
            ..failure
        };
        let actual =
            PalsNativeOwnerAccounting::between(&before, &pages_failure, true, true).unwrap();
        assert_eq!(actual.public_nn_runs_attempted(), 3);
        assert_eq!(actual.public_nn_runs_completed(), 2);
        assert_eq!(actual.public_nn_runs_failed_known(), 1);
        assert_eq!(actual.role_nn_runs_attempted(), 0);
    }
    #[test]
    fn native_owner_accounting_requires_loaded_capability_and_known_completion() {
        let empty = PalsBackendStats::default();
        // 시작 전 거부된 요청도 실제 로드된 P/C owner가 반환했다면 known zero다.
        let no_run = PalsNativeOwnerAccounting::between(&empty, &empty, true, true).unwrap();
        assert_eq!(no_run.public_nn_runs_attempted(), 0);
        assert_eq!(no_run.role_nn_runs_attempted(), 0);
        assert!(PalsNativeOwnerAccounting::between(&empty, &empty, false, true).is_none());
        // 일반 native PALS/V capability를 확인하지 못했다면 V=0을 발급하지 않는다.
        assert!(PalsNativeOwnerAccounting::between(&empty, &empty, true, false).is_none());
    }
    #[test]
    fn native_owner_accounting_rejects_saturation_regression_and_mixed_calls() {
        let empty = PalsBackendStats::default();
        let mut invalid = PalsBackendStats {
            admitted_role_requests: 2,
            public_cache_hits: 2,
            ..Default::default()
        };
        assert!(PalsNativeOwnerAccounting::between(&empty, &invalid, true, true).is_none());
        invalid.admitted_role_requests = u64::MAX;
        assert!(PalsNativeOwnerAccounting::between(&empty, &invalid, true, true).is_none());
        let previous = PalsBackendStats {
            admitted_role_requests: 1,
            public_cache_hits: 1,
            ..Default::default()
        };
        assert!(PalsNativeOwnerAccounting::between(&previous, &empty, true, true).is_none());
        let reset = PalsBackendStats {
            new_game_resets: 1,
            ..Default::default()
        };
        assert!(PalsNativeOwnerAccounting::between(&empty, &reset, true, true).is_none());
        let unfinished = PalsBackendStats {
            admitted_role_requests: 1,
            role_nn_runs_attempted: 1,
            ..Default::default()
        };
        // 완료/known-failure 계수가 없는 attempted work를 known zero로 숨기지 않는다.
        assert!(PalsNativeOwnerAccounting::between(&empty, &unfinished, true, true).is_none());
    }
    #[test]
    fn cuda_full_audit_waits_for_completed_run_and_latches_origin_failure() {
        let mut phase = CudaMappingAudit::default();
        phase.allow_run().unwrap();
        phase
            .after_run(false, || {
                panic!("provider images cannot be required before a completed Run")
            })
            .unwrap();
        assert!(matches!(phase, CudaMappingAudit::AwaitingFirstRun));
        let calls = std::cell::Cell::new(0);
        phase
            .after_run(true, || {
                calls.set(calls.get() + 1);
                Ok(())
            })
            .unwrap();
        phase
            .after_run(true, || panic!("full origin audit is not a per-node scan"))
            .unwrap();
        assert_eq!(calls.get(), 1);
        assert!(matches!(phase, CudaMappingAudit::Confirmed));
        let mut rejected = CudaMappingAudit::default();
        let failure = rejected
            .after_run(true, || {
                Err(fail(
                    K::BackendUnavailable,
                    S::Backend,
                    "missing provider mapping",
                ))
            })
            .unwrap_err();
        assert_eq!(
            failure.detail,
            "PALS CUDA full mapping audit failed after completed native Run"
        );
        assert_eq!(rejected.allow_run().unwrap_err(), failure);
        assert_eq!(
            rejected
                .after_run(true, || panic!(
                    "failed origin audit cannot be retried into success"
                ))
                .unwrap_err(),
            failure
        );
        let mut final_rejected = CudaMappingAudit::default();
        assert!(final_rejected
            .final_audit(|| panic!(
                "final audit cannot require lazy provider images before first Run"
            ))
            .is_err());
        final_rejected.after_run(true, || Ok(())).unwrap();
        let failure = final_rejected
            .final_audit(|| {
                Err(fail(
                    K::BackendUnavailable,
                    S::Backend,
                    "late mapping origin changed",
                ))
            })
            .unwrap_err();
        assert_eq!(
            failure.detail,
            "PALS CUDA final full mapping audit failed after completed native Run"
        );
        assert_eq!(final_rejected.allow_run().unwrap_err(), failure);
        assert_eq!(
            final_rejected
                .after_run(true, || panic!(
                    "no new Run may be admitted after final origin failure"
                ))
                .unwrap_err(),
            failure
        );
        assert_eq!(
            final_rejected
                .final_audit(|| panic!("final origin failure cannot be retried into success"))
                .unwrap_err(),
            failure
        );
    }
}
