//! Original LC0 system-CUDA startup custody, before product integration.
//!
//! This owner alone is not a supervisor, a device fence, or permission to
//! terminate the process. A caller must retain it together with the original
//! authenticated parent and must not replace an unknown physical result with
//! process exit. The legacy CLI intentionally rejects this path until that
//! caller and the normal shutdown/receipt path are connected.

use super::{
    NATIVE_DIAGNOSTIC_CAPACITY, NativeBootstrapError, NativeConfig, NativeCudaFactory,
    NativeRunError, NativeRunReport,
};
use crate::engine::{EngineError, OwnerRegistry};
use crate::pals_adaptive_bootstrap::runtime_assets::system_cuda::SystemCudaStartup;
use crate::rovent_bootstrap::launch::{ConnectedParent, SystemReceiptWriter};
use rz_contracts::{
    CompletionContext, ContractError, Digest, EncodingHandle, ErrorCode, ModelHandle,
    SlotGeneration, Stage,
};
use rz_encoding::classical::HistoryFill;
use rz_eval::{
    asset::{AssetMetadata, MaiaAsset},
    contracts::{ClassicalProjection, MaiaBinding, PhysicalFailure, encoding_manifest},
    error::{BackendError, FailureKind, FailureStage},
    native_runtime_bridge::NativeWorkerOwner,
    onnx::{
        BackendConfig, CudaRuntimeMappingEvidence, ExecutionExperiments, IO_BYTES_PER_ITEM,
        Lc0StartupCustody, Lc0StartupState, OnnxBackend, OrtRuntime, Provider, SystemRuntime,
    },
    worker::{PhysicalRecoveryKey, PhysicalRecoveryReservation, SessionReleased},
};
use rz_experiments::{
    NativeRuntimeSourceV6, PhysicalCompletionV6, ProviderReadinessV6, SystemRuntimeLockV6,
    SystemRuntimeReceiptV6,
};
use rz_telemetry::source::SourceJournal;
use std::{
    io,
    os::unix::net::UnixStream,
    sync::{Arc, atomic::AtomicBool},
    task::Poll,
    time::Instant,
};

mod control;
mod endpoint;
mod facade;
mod profile;
mod projection;
pub use facade::{
    Lc0AsyncAdmission, Lc0AsyncFinishStatus, Lc0AsyncInputs, Lc0ParentListener, Lc0PendingFailure,
    Lc0PendingFailureCause, Lc0PendingSupervisor, Lc0ReceiptOwner, Lc0SupervisedSession,
    parent_preflight,
};
pub use projection::{
    Lc0ObservedPublication, Lc0ObservedRelease, Lc0ObservedService, Lc0OriginalSessionObservation,
};

/// A parent PID captured before the caller publishes any child endpoint. The
/// accepted stream is bound to this exact parent and the V6 runtime lock;
/// admission is neither device ownership nor a physical completion witness.
pub(super) struct Lc0PendingParent {
    original_parent_pid: i32,
    process_epoch: u64,
    runtime_lock_sha256: String,
    cuda_ordinal: i32,
}

impl Lc0PendingParent {
    fn parent_pid(&self) -> i32 {
        self.original_parent_pid
    }

    pub(super) fn capture(
        process_epoch: u64,
        lock: &SystemRuntimeLockV6,
    ) -> Result<Self, BackendError> {
        lock.verify().map_err(|_| {
            BackendError::new(
                FailureKind::IdentityMismatch,
                FailureStage::Admission,
                "LC0 system CUDA lock is invalid",
            )
        })?;
        let spec = match &lock.runtime {
            NativeRuntimeSourceV6::SystemCuda(spec) if process_epoch != 0 => spec,
            _ => {
                return Err(BackendError::new(
                    FailureKind::InvalidInput,
                    FailureStage::Admission,
                    "LC0 parent admission requires a system lock and process epoch",
                ));
            }
        };
        let original_parent_pid = ConnectedParent::capture_original_parent().map_err(|_| {
            BackendError::new(
                FailureKind::BackendUnavailable,
                FailureStage::Admission,
                "LC0 original supervisor PID unavailable",
            )
        })?;
        Ok(Self {
            original_parent_pid,
            process_epoch,
            runtime_lock_sha256: lock.canonical_sha256.clone(),
            cuda_ordinal: spec.cuda_device_id,
        })
    }

    pub(super) fn authenticate(
        &self,
        stream: UnixStream,
        until: Instant,
    ) -> io::Result<Arc<ConnectedParent>> {
        ConnectedParent::authenticate(
            stream,
            std::process::id(),
            self.original_parent_pid,
            self.process_epoch,
            &self.runtime_lock_sha256,
            self.cuda_ordinal,
            until,
        )
        .map(Arc::new)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Lc0SystemStartupPhase {
    BeforeOrtCall,
    /// Includes a reused process latch: absence of a new entry callback does
    /// not prove that previous native work never entered or has completed.
    OrtCallUnknown,
    OrtReady,
    SessionEntryUnknown,
    BackendReady,
    WorkerOwned,
    SessionReleased,
}

fn matches_registered_provider(provider: Provider, device_id: i32, arena_bytes: u64) -> bool {
    matches!(
        provider,
        Provider::Cuda { device_id: actual, arena_bytes: arena }
            if actual == device_id && u64::try_from(arena) == Ok(arena_bytes)
    )
}

/// One original owner must be registered with the live supervisor before
/// calling `load`. No state in this object establishes physical completion.
pub(super) struct Lc0SystemStartupCustody {
    asset: Option<MaiaAsset>,
    system: Option<SystemRuntime>,
    runtime: Option<OrtRuntime>,
    startup: Option<SystemCudaStartup>,
    model: Option<Lc0StartupCustody>,
    ready: Option<Lc0ReadyObjects>,
    worker: Option<NativeWorkerOwner>,
    reservation: Option<PhysicalRecoveryReservation>,
    recovery_key: Option<PhysicalRecoveryKey>,
    session_release: Option<SessionReleased>,
    parent: Option<Arc<ConnectedParent>>,
    process_epoch: u64,
    execution_experiments: ExecutionExperiments,
    raw_cache: bool,
    phase: Lc0SystemStartupPhase,
    ort_entry_observed: bool,
    attempted: bool,
}

/// Same-owner intermediate state. A retained worker must consume the actual
/// backend through a recovery reservation; borrowing these fields is not a
/// physical completion witness or a successful worker handoff.
struct Lc0ReadyObjects {
    metadata: AssetMetadata,
    backend: Option<OnnxBackend>,
}

/// Holds the original startup custody beside the factory that uses its worker.
/// This is an internal construction boundary, not a supervisor or an executable
/// launch path. The caller must retain this object and its process on unknown
/// completion, including unwinds; dropping it is not a physical fence.
pub(super) struct Lc0SystemSessionOwner {
    custody: Lc0SystemStartupCustody,
    factory: Option<Arc<NativeCudaFactory>>,
    owner_token: Arc<()>,
}

/// The three observations have distinct meanings. In particular, a mapping
/// `Unknown`, a service failure, and an unconfirmed physical release cannot
/// be substituted for one another or authorize the supervisor to exit.
pub(super) struct Lc0SystemFinish {
    owner_token: Arc<()>,
    pub(super) run: Result<NativeRunReport, NativeRunError>,
    pub(super) mapping: Option<CudaRuntimeMappingEvidence>,
    // None differs from a same-worker release poll returning Pending; neither
    // establishes GPU-wide physical completion.
    pub(super) session_release: Option<Poll<Result<SessionReleased, PhysicalFailure>>>,
}

impl Lc0SystemSessionOwner {
    pub(super) fn new(custody: Lc0SystemStartupCustody) -> Self {
        Self {
            custody,
            factory: None,
            owner_token: Arc::new(()),
        }
    }

    pub(super) fn custody_mut(&mut self) -> &mut Lc0SystemStartupCustody {
        &mut self.custody
    }

    /// Build the factory from the same authenticated custody and reservation.
    /// On any error, including one after worker creation, `self` retains the
    /// original custody so the caller can preserve or drain it. This operation
    /// is intentionally separate from the guarded legacy executable path.
    pub(super) fn connect(
        &mut self,
        owners: &OwnerRegistry,
        config: &NativeConfig,
    ) -> Result<(), NativeBootstrapError> {
        if config.parallelism != 1
            || config.execution_experiments != self.custody.execution_experiments
            || config.raw_cache != self.custody.raw_cache
        {
            return Err(NativeBootstrapError::Config(
                "LC0 system session options differ from the original B1 custody",
            ));
        }
        if self.factory.is_some() {
            return Err(NativeBootstrapError::Config(
                "LC0 system session already connected",
            ));
        }
        match self.custody.phase() {
            Lc0SystemStartupPhase::BackendReady => {
                // Validate tracing before moving the backend out of startup.
                let trace = config.source_journal()?;
                if self.custody.ready_parts().is_none() {
                    self.custody.prepare_handoff()?;
                }
                let projection = self.custody.build_projection(owners)?;
                self.custody.start_retained_worker(
                    projection,
                    NATIVE_DIAGNOSTIC_CAPACITY,
                    trace,
                )?;
            }
            // Factory construction may fail after the original worker was
            // installed. Retry only the factory step, never native startup.
            Lc0SystemStartupPhase::WorkerOwned => {}
            _ => {
                return Err(NativeBootstrapError::Config(
                    "LC0 system startup is not ready for session connection",
                ));
            }
        }
        self.factory = Some(Arc::new(NativeCudaFactory::from_system_custody(
            &self.custody,
        )?));
        Ok(())
    }

    /// A returned factory is not a completion witness and does not authorize
    /// process exit. The caller must retain this owner while any clone is live.
    pub(super) fn factory(&self) -> Option<&Arc<NativeCudaFactory>> {
        self.factory.as_ref()
    }

    /// Bounded collection on the original factory. The caller still owns
    /// `self` on every outcome and must retain the authenticated process if
    /// session release is pending or unknown. This is not a supervisor.
    pub(super) fn finish_until(
        &mut self,
        service: &mut Option<EngineError>,
        until: Instant,
    ) -> Result<Lc0SystemFinish, NativeBootstrapError> {
        let factory = self.factory.as_ref().ok_or(NativeBootstrapError::Config(
            "LC0 system session is not connected",
        ))?;
        // Retain the capability before the report takes the original service
        // error; no fallible owner operation follows that ownership transfer.
        let owner_token = Arc::clone(&self.owner_token);
        let (run, mapping) = factory
            .inner
            .finish_until_with_mapping(service, until, true);
        // The caller must retain this report before invoking any further
        // worker operation that can unwind; it may contain the original UCI
        // service error and evidence owner.
        Ok(Lc0SystemFinish {
            owner_token,
            run,
            mapping,
            session_release: None,
        })
    }

    /// Only the same worker's session release; this does not attest that the
    /// protocol service or report collection succeeded.
    pub(super) fn try_release(&mut self) -> Poll<Result<SessionReleased, PhysicalFailure>> {
        self.custody.try_finish()
    }

    /// Observe the original runtime without converting unavailable mapping
    /// evidence into a conflict or a successful strict-attestation witness.
    /// The result is independent of the worker's physical completion.
    pub(super) fn observe_mappings(&self) -> Result<CudaRuntimeMappingEvidence, BackendError> {
        if !matches!(
            self.custody.phase(),
            Lc0SystemStartupPhase::WorkerOwned | Lc0SystemStartupPhase::SessionReleased
        ) {
            return Err(BackendError::new(
                FailureKind::InvalidInput,
                FailureStage::Admission,
                "LC0 system worker is not ready for mapping observation",
            ));
        }
        self.custody
            .runtime()
            .ok_or_else(|| {
                BackendError::new(
                    FailureKind::BackendUnavailable,
                    FailureStage::Admission,
                    "LC0 original system runtime is unavailable",
                )
            })?
            .observe_cuda_runtime_mappings()
    }

    /// A partial V6 claim from the original custody. It deliberately leaves
    /// physical completion unknown; a same-worker release is not a GPU fence.
    pub(super) fn runtime_receipt(
        &self,
        lock: &SystemRuntimeLockV6,
    ) -> Result<SystemRuntimeReceiptV6, BackendError> {
        self.custody.runtime_receipt(lock)
    }

    /// Publish to the same no-follow, create-new private directory writer used
    /// by the existing native supervisor. Failure to write is not a physical
    /// completion result, and the caller must retain this owner on failure.
    pub(super) fn publish_runtime_receipt(
        &self,
        writer: &SystemReceiptWriter,
        lock: &SystemRuntimeLockV6,
        finish: bool,
    ) -> Result<(), String> {
        let receipt = self
            .runtime_receipt(lock)
            .map_err(|error| error.to_string())?;
        let name = if finish {
            "lc0-system-runtime-finish.v6.json"
        } else {
            "lc0-system-runtime-startup.v6.json"
        };
        writer.write(name, &receipt, lock)
    }

    /// This observation is separately scoped from the V6 runtime receipt and
    /// binds to the exact owner that produced `finish`. The same-worker release
    /// remains a session observation, never a device-wide completion claim.
    pub(super) fn publish_finish_observation(
        &self,
        writer: &SystemReceiptWriter,
        lock: &SystemRuntimeLockV6,
        finish: &Lc0SystemFinish,
    ) -> Result<(), String> {
        if !Arc::ptr_eq(&self.owner_token, &finish.owner_token) {
            return Err("LC0 finish belongs to a different owner".into());
        }
        let spec = match &lock.runtime {
            NativeRuntimeSourceV6::SystemCuda(spec) if lock.verify().is_ok() => spec,
            _ => return Err("LC0 finish requires a valid system CUDA lock".into()),
        };
        if !self.custody.parent.as_ref().is_some_and(|parent| {
            parent.matches(
                self.custody.process_epoch,
                &lock.canonical_sha256,
                spec.cuda_device_id,
            )
        }) {
            return Err("LC0 finish lock differs from the original owner".into());
        }
        let (session_status, completed_runs) = match &finish.session_release {
            None => ("unknown", None),
            Some(Poll::Pending) => ("pending", None),
            Some(Poll::Ready(Err(_))) => ("failed", None),
            Some(Poll::Ready(Ok(release))) => {
                ("released", Some(release.completed_synchronous_runs))
            }
        };
        let mapping = match finish.mapping {
            Some(CudaRuntimeMappingEvidence::Observed) => "observed",
            Some(CudaRuntimeMappingEvidence::Unknown) => "unknown",
            None if finish
                .run
                .as_ref()
                .err()
                .is_some_and(|error| error.runtime_mapping_error.is_some()) =>
            {
                "failed"
            }
            None => "not_observed",
        };
        let failure = finish.run.as_ref().err();
        let startup_probe = self.custody.worker.as_ref().and_then(|worker| {
            worker.system_cuda_metadata().map(|metadata| {
                serde_json::json!({
                    "executed_cuda_nodes": metadata.executed_cuda_nodes,
                    "synchronous_run_returned": metadata.warm_probe_synchronous_run_returned,
                    "scope": "startup probe only; not evidence for later requests"
                })
            })
        });
        let observation = serde_json::json!({
            "domain": "rz-lc0-system-runtime-finish-observation/1",
            "process_id": std::process::id(),
            "process_epoch": self.custody.process_epoch,
            "lock_sha256": lock.canonical_sha256,
            "run_status": if failure.is_some() { "failed" } else { "completed" },
            "service_error": failure.is_some_and(|error| error.service.is_some()),
            "collection_error": failure.is_some_and(|error| error.collection_error.is_some()),
            "worker_shutdown_error": failure.is_some_and(|error| error.worker_shutdown_error.is_some()),
            "runtime_mapping_observation": mapping,
            "startup_cuda_probe": startup_probe,
            "session_release": { "status": session_status, "completed_synchronous_runs": completed_runs },
            "physical_completion": "unknown",
            "scope": "original process-local LC0 session, not device-wide completion or numerical acceptance"
        });
        let bytes = serde_json::to_vec(&observation).map_err(|error| error.to_string())?;
        writer.write_bytes("lc0-system-finish-observation.v1.json", &bytes)
    }
}

impl Lc0SystemStartupCustody {
    /// LC0's own supervisor must authenticate directly against the V6 system
    /// runtime lock digest. A Rovent V3 parent bound to an outer input lock is
    /// a different authority and cannot be silently substituted here.
    pub(super) fn new(
        asset: MaiaAsset,
        parent: Arc<ConnectedParent>,
        process_epoch: u64,
        lock: &SystemRuntimeLockV6,
        config: &NativeConfig,
    ) -> Result<Self, BackendError> {
        lock.verify().map_err(|_| {
            BackendError::new(
                FailureKind::IdentityMismatch,
                FailureStage::Admission,
                "LC0 system CUDA lock is invalid",
            )
        })?;
        let spec = match &lock.runtime {
            NativeRuntimeSourceV6::SystemCuda(spec) => spec,
            NativeRuntimeSourceV6::Cpu => {
                return Err(BackendError::new(
                    FailureKind::InvalidInput,
                    FailureStage::Admission,
                    "LC0 system CUDA requires a system CUDA runtime lock",
                ));
            }
        };
        let lock_path = config.system_cuda_lock.as_ref().ok_or_else(|| {
            BackendError::new(
                FailureKind::InvalidInput,
                FailureStage::Admission,
                "LC0 system CUDA lock path is absent",
            )
        })?;
        let lock_bytes = rz_eval::asset::read_bounded(lock_path, 256 * 1024)?;
        if Some(rz_eval::asset::sha256(&lock_bytes)) != config.system_cuda_lock_sha256 {
            return Err(BackendError::new(
                FailureKind::IdentityMismatch,
                FailureStage::Admission,
                "LC0 system CUDA lock file digest differs",
            ));
        }
        let lock_text = std::str::from_utf8(&lock_bytes).map_err(|_| {
            BackendError::new(
                FailureKind::InvalidInput,
                FailureStage::Admission,
                "LC0 system CUDA lock file is not UTF-8",
            )
        })?;
        let file_lock = SystemRuntimeLockV6::from_json(lock_text).map_err(|_| {
            BackendError::new(
                FailureKind::IdentityMismatch,
                FailureStage::Admission,
                "LC0 system CUDA lock file is invalid",
            )
        })?;
        let core = spec.libraries.iter().find(|entry| entry.role == "ort-core");
        let expected_core = config
            .ort_sha256
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let selected_core = std::fs::canonicalize(&config.ort_library).map_err(|_| {
            BackendError::new(
                FailureKind::InvalidInput,
                FailureStage::Admission,
                "LC0 system CUDA ORT core path cannot be resolved",
            )
        })?;
        if file_lock.canonical_sha256 != lock.canonical_sha256
            || asset.manifest_digest() != config.manifest_sha256
            || u64::try_from(asset.profile().cuda_arena_bytes()) != Ok(spec.cuda_arena_bytes)
            || !core.is_some_and(|entry| {
                entry.sha256 == expected_core
                    && selected_core == std::path::Path::new(&entry.canonical_path)
            })
        {
            return Err(BackendError::new(
                FailureKind::IdentityMismatch,
                FailureStage::Admission,
                "LC0 asset, ORT core or system runtime declaration differs",
            ));
        }
        if process_epoch == 0
            || !parent.matches(process_epoch, &lock.canonical_sha256, spec.cuda_device_id)
        {
            return Err(BackendError::new(
                FailureKind::IdentityMismatch,
                FailureStage::Admission,
                "LC0 supervisor is not bound to the V6 system runtime lock",
            ));
        }
        if config.parallelism != 1 {
            return Err(BackendError::new(
                FailureKind::InvalidInput,
                FailureStage::Admission,
                "LC0 system worker currently supports B1 only",
            ));
        }
        // Reserve before the first possible native entry. This reserves a
        // process-local recovery slot, not a GPU or an external supervisor.
        let (reservation, recovery_key) = PhysicalRecoveryReservation::reserve()?;
        Ok(Self {
            asset: Some(asset),
            system: None,
            runtime: None,
            startup: None,
            model: None,
            ready: None,
            worker: None,
            reservation: Some(reservation),
            recovery_key: Some(recovery_key),
            session_release: None,
            parent: Some(parent),
            process_epoch,
            execution_experiments: config.execution_experiments,
            raw_cache: config.raw_cache,
            phase: Lc0SystemStartupPhase::BeforeOrtCall,
            ort_entry_observed: false,
            attempted: false,
        })
    }

    pub(super) fn phase(&self) -> Lc0SystemStartupPhase {
        self.phase
    }

    pub(super) fn ort_entry_observed(&self) -> bool {
        self.ort_entry_observed
    }

    /// Construct a process-bound partial observation from actual custody,
    /// never from the declaration alone. Failure to observe libraries stays
    /// an error and does not authorize owner release or imply GPU failure.
    pub(super) fn runtime_receipt(
        &self,
        lock: &SystemRuntimeLockV6,
    ) -> Result<SystemRuntimeReceiptV6, BackendError> {
        lock.verify().map_err(|_| {
            BackendError::new(
                FailureKind::IdentityMismatch,
                FailureStage::Admission,
                "LC0 system receipt lock is invalid",
            )
        })?;
        let spec = match &lock.runtime {
            NativeRuntimeSourceV6::SystemCuda(spec) => spec,
            NativeRuntimeSourceV6::Cpu => {
                return Err(BackendError::new(
                    FailureKind::IdentityMismatch,
                    FailureStage::Admission,
                    "LC0 system receipt requires a CUDA lock",
                ));
            }
        };
        if !self.parent.as_ref().is_some_and(|parent| {
            parent.matches(
                self.process_epoch,
                &lock.canonical_sha256,
                spec.cuda_device_id,
            )
        }) {
            return Err(BackendError::new(
                FailureKind::IdentityMismatch,
                FailureStage::Admission,
                "LC0 system receipt owner differs from the original parent",
            ));
        }
        let ready = matches!(
            self.phase,
            Lc0SystemStartupPhase::BackendReady
                | Lc0SystemStartupPhase::WorkerOwned
                | Lc0SystemStartupPhase::SessionReleased
        );
        let (observed_libraries, unregistered_native_mappings) = match &self.startup {
            Some(startup) if ready => {
                let observed = startup.observe_ready_mappings()?;
                (observed.libraries, observed.unregistered_native_mappings)
            }
            Some(startup) => (startup.observe()?, None),
            None => (Vec::new(), None),
        };
        let observed_identity_sha256 = self.runtime.as_ref().and_then(|runtime| {
            runtime.cuda_runtime_identity().map(|digest| {
                use std::fmt::Write;
                let mut text = String::with_capacity(64);
                for byte in digest {
                    write!(&mut text, "{byte:02x}").expect("format byte into String");
                }
                text
            })
        });
        let receipt = SystemRuntimeReceiptV6 {
            domain: rz_experiments::SYSTEM_RUNTIME_RECEIPT_V6_DOMAIN.into(),
            lock_sha256: lock.canonical_sha256.clone(),
            process_id: std::process::id(),
            process_generation: self.process_epoch,
            observed_identity_sha256,
            observed_libraries,
            unregistered_native_mappings,
            provider_ready: if ready {
                ProviderReadinessV6::Ready
            } else {
                ProviderReadinessV6::Unknown
            },
            physical_completion: PhysicalCompletionV6::Unknown,
        };
        receipt
            .validate_against(lock, std::process::id(), self.process_epoch)
            .map_err(|_| {
                BackendError::new(
                    FailureKind::IdentityMismatch,
                    FailureStage::Admission,
                    "LC0 system receipt does not match its original lock",
                )
            })?;
        Ok(receipt)
    }

    /// The already-owned authenticated parent channel is checked immediately
    /// before each native boundary; it is not a physical completion fence.
    /// Errors and unwinds leave all returned native objects in this owner.
    pub(super) fn load(
        &mut self,
        lock: &SystemRuntimeLockV6,
        until: Instant,
        cancelled: &AtomicBool,
        config: BackendConfig,
    ) -> Result<(), BackendError> {
        if self.attempted {
            return Err(BackendError::new(
                FailureKind::InvalidInput,
                FailureStage::Admission,
                "LC0 system startup owner already used",
            ));
        }
        self.attempted = true;
        let spec = match &lock.runtime {
            NativeRuntimeSourceV6::SystemCuda(spec) => spec,
            NativeRuntimeSourceV6::Cpu => {
                return Err(BackendError::new(
                    FailureKind::InvalidInput,
                    FailureStage::Admission,
                    "LC0 system CUDA requires a system CUDA runtime lock",
                ));
            }
        };
        if !matches_registered_provider(config.provider, spec.cuda_device_id, spec.cuda_arena_bytes)
            || config.experiments != self.execution_experiments
            || config.max_batch != 1
            || config.intra_threads != 1
            || config.host_io_bytes != IO_BYTES_PER_ITEM
        {
            return Err(BackendError::new(
                FailureKind::IdentityMismatch,
                FailureStage::Admission,
                "LC0 backend provider, device, arena or B1 execution options differ",
            ));
        }
        let parent = self.parent.as_ref().expect("original parent retained");
        if self.process_epoch == 0
            || !parent.matches(
                self.process_epoch,
                &lock.canonical_sha256,
                spec.cuda_device_id,
            )
        {
            return Err(BackendError::new(
                FailureKind::IdentityMismatch,
                FailureStage::Admission,
                "LC0 original supervisor and system CUDA lock differ",
            ));
        }
        let check_parent = || {
            parent.ensure_live().map_err(|error| {
                let detail = match error.kind() {
                    std::io::ErrorKind::PermissionDenied => "LC0 supervisor identity changed",
                    std::io::ErrorKind::UnexpectedEof => "LC0 supervisor disconnected",
                    std::io::ErrorKind::InvalidData => "LC0 supervisor channel had unexpected data",
                    _ => "LC0 supervisor channel I/O failed",
                };
                BackendError::new(
                    FailureKind::BackendUnavailable,
                    FailureStage::Admission,
                    detail,
                )
            })
        };
        let system_slot = &mut self.system;
        let runtime_slot = &mut self.runtime;
        let phase = &mut self.phase;
        let entry_observed = &mut self.ort_entry_observed;
        let startup =
            SystemCudaStartup::capture_and_load_until_with(lock, until, cancelled, |system| {
                *system_slot = Some(system.clone());
                check_parent()?;
                *phase = Lc0SystemStartupPhase::OrtCallUnknown;
                let mut rejected_before_entry = false;
                let runtime = OrtRuntime::load_system_cuda_with_checked_entry(system, || {
                    if let Err(error) = check_parent() {
                        rejected_before_entry = true;
                        return Err(error);
                    }
                    *entry_observed = true;
                    Ok(())
                });
                if rejected_before_entry {
                    // The checked callback ran on a new latch and rejected
                    // before native entry. This says nothing about work from
                    // other owners or earlier processes.
                    *phase = Lc0SystemStartupPhase::BeforeOrtCall;
                }
                let runtime = runtime?;
                // Save before the outer helper's identity/deadline checks.
                *runtime_slot = Some(runtime.clone());
                Ok(runtime)
            })?;
        self.startup = Some(startup);
        self.phase = Lc0SystemStartupPhase::OrtReady;
        let runtime = self.runtime.as_ref().expect("captured ORT runtime").clone();
        let asset = self.asset.take().expect("original LC0 asset");
        self.model = Some(Lc0StartupCustody::new(runtime, asset));
        let phase = &mut self.phase;
        self.model
            .as_mut()
            .expect("original LC0 startup owner")
            .load(config, || {
                check_parent()?;
                *phase = Lc0SystemStartupPhase::SessionEntryUnknown;
                Ok(())
            })?;
        self.phase = Lc0SystemStartupPhase::BackendReady;
        Ok(())
    }

    pub(super) fn model_state(&self) -> Option<Lc0StartupState> {
        self.model.as_ref().map(Lc0StartupCustody::state)
    }

    /// Move the original pair only after a successful startup. Both remain in
    /// this owner if later projection or worker admission fails.
    pub(super) fn prepare_handoff(&mut self) -> Result<(), BackendError> {
        if self.phase != Lc0SystemStartupPhase::BackendReady || self.ready.is_some() {
            return Err(BackendError::new(
                FailureKind::InvalidInput,
                FailureStage::Admission,
                "LC0 system startup is not ready for the original handoff",
            ));
        }
        let (metadata, backend) = self
            .model
            .as_mut()
            .and_then(Lc0StartupCustody::take_ready)
            .ok_or_else(|| {
                BackendError::new(
                    FailureKind::BackendUnavailable,
                    FailureStage::Admission,
                    "LC0 original model and backend are not both ready",
                )
            })?;
        self.ready = Some(Lc0ReadyObjects {
            metadata,
            backend: Some(backend),
        });
        Ok(())
    }

    /// Projection can inspect the original pair without gaining the ability
    /// to take and accidentally drop the physical backend. Worker transfer
    /// needs a separate reservation-bound operation on this owner.
    pub(super) fn ready_parts(&self) -> Option<(&AssetMetadata, &OnnxBackend)> {
        self.ready.as_ref().and_then(|ready| {
            ready
                .backend
                .as_ref()
                .map(|backend| (&ready.metadata, backend))
        })
    }

    /// Build the existing LC0 model/encoding projection without dispatching a
    /// neural request or transferring the physical backend.
    pub(super) fn build_projection(
        &self,
        owners: &OwnerRegistry,
    ) -> Result<ClassicalProjection, ContractError> {
        let (metadata, backend) = self.ready_parts().ok_or_else(|| {
            ContractError::new(
                ErrorCode::InvalidInput,
                Stage::Admission,
                "LC0 original metadata and backend are not ready",
            )
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
            manifest: Digest(metadata.manifest_digest()),
        };
        let projection =
            ClassicalProjection::new(MaiaBinding::for_backend(backend, model, encoding, fill)?);
        #[cfg(feature = "experimental-raw-cache")]
        if self.raw_cache {
            projection.configure_raw_cache(rz_eval::raw_cache::RawCacheLimits::default())?;
        }
        Ok(projection)
    }

    /// Transfer only through the original reservation-aware factory. On any
    /// admission/spawn failure the factory returns with backend and reservation
    /// still in their original slots; success installs the same worker here.
    pub(super) fn start_retained_worker(
        &mut self,
        projection: ClassicalProjection,
        diagnostic_capacity: usize,
        trace: Option<SourceJournal<CompletionContext>>,
    ) -> Result<(), PhysicalFailure> {
        if self.phase != Lc0SystemStartupPhase::BackendReady || self.worker.is_some() {
            return Err(BackendError::new(
                FailureKind::InvalidInput,
                FailureStage::Admission,
                "LC0 system backend is not ready for worker transfer",
            )
            .into());
        }
        self.parent
            .as_ref()
            .expect("original parent retained")
            .ensure_live()
            .map_err(|_| {
                BackendError::new(
                    FailureKind::BackendUnavailable,
                    FailureStage::Admission,
                    "LC0 original supervisor unavailable before worker transfer",
                )
            })?;
        let ready = self.ready.as_mut().ok_or_else(|| {
            BackendError::new(
                FailureKind::InvalidInput,
                FailureStage::Admission,
                "LC0 original handoff is absent",
            )
        })?;
        let reservation = self.reservation.as_mut().ok_or_else(|| {
            BackendError::new(
                FailureKind::InvalidInput,
                FailureStage::Admission,
                "LC0 original physical reservation is absent",
            )
        })?;
        let worker = NativeWorkerOwner::from_system_cuda_onnx_retained_profiled(
            &mut ready.backend,
            reservation,
            projection,
            diagnostic_capacity,
            trace,
        )?;
        self.worker = Some(worker);
        self.phase = Lc0SystemStartupPhase::WorkerOwned;
        Ok(())
    }

    pub(super) fn worker(&self) -> Option<&NativeWorkerOwner> {
        self.worker.as_ref()
    }

    pub(super) fn runtime(&self) -> Option<&OrtRuntime> {
        self.runtime.as_ref()
    }

    pub(super) fn recovery_key(&self) -> Option<&PhysicalRecoveryKey> {
        self.recovery_key.as_ref()
    }

    /// A future supervisor may use this distinct stream for its bounded
    /// control protocol. The original admission stream remains liveness-only.
    /// This authenticates the original parent; it neither starts that protocol
    /// nor changes the custody or completion state.
    pub(super) fn authenticate_control(
        &self,
        stream: UnixStream,
        until: Instant,
    ) -> io::Result<UnixStream> {
        self.parent
            .as_ref()
            .ok_or_else(|| io::Error::other("LC0 original parent is unavailable"))?
            .authenticate_separate_control(stream, until)
    }

    /// Normal same-worker release only. It is not a device-wide GPU fence and
    /// cannot convert a failed or unknown physical execution into success.
    pub(super) fn try_finish(&mut self) -> Poll<Result<SessionReleased, PhysicalFailure>> {
        if self.phase == Lc0SystemStartupPhase::SessionReleased {
            return Poll::Ready(Ok(self
                .session_release
                .expect("released phase has evidence")));
        }
        if self.phase != Lc0SystemStartupPhase::WorkerOwned {
            return Poll::Ready(Err(BackendError::new(
                FailureKind::InvalidInput,
                FailureStage::Admission,
                "LC0 original worker is not active",
            )
            .into()));
        }
        let worker = self.worker.as_ref().expect("worker-owned phase");
        match worker.try_shutdown() {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
            Poll::Ready(Ok(())) => {}
        }
        match worker.try_session_release() {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(error)) => Poll::Ready(Err(error)),
            Poll::Ready(Ok(None)) => Poll::Ready(Err(BackendError::new(
                FailureKind::BackendUnavailable,
                FailureStage::Backend,
                "LC0 same-worker session release is unknown",
            )
            .into())),
            Poll::Ready(Ok(Some(release))) => {
                self.session_release = Some(release);
                self.phase = Lc0SystemStartupPhase::SessionReleased;
                Poll::Ready(Ok(release))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_must_match_original_system_device_and_arena() {
        let actual = Provider::Cuda {
            device_id: 1,
            arena_bytes: 4096,
        };
        assert!(matches_registered_provider(actual, 1, 4096));
        assert!(!matches_registered_provider(actual, 0, 4096));
        assert!(!matches_registered_provider(actual, 1, 8192));
        assert!(!matches_registered_provider(Provider::Cpu, 1, 4096));
    }
}

impl Drop for Lc0SystemStartupCustody {
    fn drop(&mut self) {
        if !matches!(
            self.phase,
            Lc0SystemStartupPhase::BeforeOrtCall | Lc0SystemStartupPhase::SessionReleased
        ) {
            // Defensive in-process quarantine only: the real caller still
            // needs to retain the process and original parent on unknown.
            std::mem::forget((
                self.asset.take(),
                self.system.take(),
                self.runtime.take(),
                self.startup.take(),
                self.model.take(),
                self.ready.take(),
                self.parent.take(),
                self.worker.take(),
                self.reservation.take(),
                self.recovery_key.take(),
            ));
        }
    }
}
