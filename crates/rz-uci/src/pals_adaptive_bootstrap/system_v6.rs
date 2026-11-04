//! Separate PALS V6 metadata preflight; it does not construct a native worker.
use super::*;
use crate::pals_adaptive_startup::{PalsSystemStartupOptionsV6, PinnedPalsSystemStartupV6};

#[cfg(all(target_os = "linux", feature = "onnx-cuda"))]
mod before_native_recovery;
#[cfg(all(target_os = "linux", feature = "onnx-cuda"))]
mod cpu_prepare;
#[cfg(all(target_os = "linux", feature = "onnx-cuda"))]
mod driver_assemble;
#[cfg(all(target_os = "linux", feature = "onnx-cuda"))]
mod driver_facade;
#[cfg(all(target_os = "linux", feature = "onnx-cuda"))]
mod driver_lifecycle;
#[cfg(all(target_os = "linux", feature = "onnx-cuda"))]
mod driver_prepare;
#[cfg(all(target_os = "linux", feature = "onnx-cuda"))]
mod failed_startup_drain;
#[cfg(all(target_os = "linux", feature = "onnx-cuda"))]
mod pipeline;

/// The receipt directory is derived from the exact held startup root, never
/// from a caller-provided pathname or a different supervisor's writer.
#[cfg(all(target_os = "linux", feature = "onnx-cuda"))]
struct PalsSystemReceiptOwnerV6 {
    owner: Arc<PinnedPalsSystemStartupV6>,
    parent: Arc<crate::rovent_bootstrap::launch::ConnectedParent>,
    writer: crate::rovent_bootstrap::launch::SystemReceiptWriter,
}

/// The supervisor receives this same capability from the factory; absent or
/// disconnected delivery remains unknown and does not attest GPU completion.
#[cfg(all(target_os = "linux", feature = "onnx-cuda"))]
struct PalsSystemReceiptPublicationV6 {
    owner: Option<PalsSystemReceiptOwnerV6>,
    startup_write: Result<(), String>,
    execution_source_write: Result<(), String>,
}

#[cfg(all(target_os = "linux", feature = "onnx-cuda"))]
struct PalsSystemNativeStartV6 {
    service: Option<ActualService>,
    receipt_publication: mpsc::Receiver<PalsSystemReceiptPublicationV6>,
    startup_snapshot: mpsc::Receiver<
        Result<
            crate::pals_native::startup_system_v6::PalsSystemNativeSnapshotV6,
            rz_eval::error::BackendError,
        >,
    >,
}

/// One original startup scope. Once the native worker is spawned this value
/// must stay with the live process until the same worker's physical outcome
/// is resolved; dropping it or observing an EOF is not a completion fence.
#[cfg(all(target_os = "linux", feature = "onnx-cuda"))]
#[must_use]
struct PalsSystemStartupCustodyV6 {
    cpu: cpu_prepare::PalsSystemCpuPreparationV6,
    parent: Arc<crate::rovent_bootstrap::launch::ConnectedParent>,
    admission: Arc<crate::rovent_bootstrap::launch::PalsPostImageAdmissionV6>,
    native: PalsSystemNativeStartV6,
}

/// Return the original owned values on both pre-entry failure and failed
/// thread creation. In particular, never discard the latter's recovery key
/// or its potentially unstarted thread in an error conversion.
#[cfg(all(target_os = "linux", feature = "onnx-cuda"))]
#[allow(dead_code)]
enum PalsSystemStartupRejectionV6 {
    BeforeNative {
        cpu: cpu_prepare::PalsSystemCpuPreparationV6,
        parent: Arc<crate::rovent_bootstrap::launch::ConnectedParent>,
        admission: Arc<crate::rovent_bootstrap::launch::PalsPostImageAdmissionV6>,
        observer: Option<Box<dyn VirtualReviewCollectionObserver>>,
        error: std::io::Error,
    },
    NativeSpawn {
        cpu: cpu_prepare::PalsSystemCpuPreparationV6,
        parent: Arc<crate::rovent_bootstrap::launch::ConnectedParent>,
        admission: Arc<crate::rovent_bootstrap::launch::PalsPostImageAdmissionV6>,
        failure: crate::pals_native::ticketed::FactoryStartupSpawnRejection<
            crate::pals_native::startup_system_v6::PalsSystemFactoryCustodyV6,
        >,
    },
}

/// Join the already prepared CPU and native factory to the same owner and
/// original admission. This is still an internal handoff; the executable and
/// arena guards must stay active until its supervisor and shutdown consumer
/// retain every branch, including unwinds and unknown physical completion.
#[cfg(all(target_os = "linux", feature = "onnx-cuda"))]
#[allow(dead_code)]
fn start_owned_system_native_v6(
    cpu: cpu_prepare::PalsSystemCpuPreparationV6,
    parent: Arc<crate::rovent_bootstrap::launch::ConnectedParent>,
    admission: Arc<crate::rovent_bootstrap::launch::PalsPostImageAdmissionV6>,
    observer: Option<Box<dyn VirtualReviewCollectionObserver>>,
) -> Result<PalsSystemStartupCustodyV6, PalsSystemStartupRejectionV6> {
    if let Err(error) = cpu
        .original_source
        .check_native_admission(&parent, &admission)
    {
        return Err(PalsSystemStartupRejectionV6::BeforeNative {
            cpu,
            parent,
            admission,
            observer,
            error,
        });
    }
    let clock = cpu.clock.runtime_clock();
    match spawn_system_native_v6(
        Arc::clone(&cpu.original_source),
        Arc::clone(&parent),
        Arc::clone(&admission),
        clock,
        observer,
    ) {
        Ok(native) => Ok(PalsSystemStartupCustodyV6 {
            cpu,
            parent,
            admission,
            native,
        }),
        Err(failure) => Err(PalsSystemStartupRejectionV6::NativeSpawn {
            cpu,
            parent,
            admission,
            failure,
        }),
    }
}

#[cfg(all(target_os = "linux", feature = "onnx-cuda"))]
impl PalsSystemReceiptOwnerV6 {
    fn open(
        owner: Arc<PinnedPalsSystemStartupV6>,
        parent: Arc<crate::rovent_bootstrap::launch::ConnectedParent>,
    ) -> Result<Self, String> {
        use rz_experiments::NativeRuntimeSourceV6;
        owner
            .verify_unchanged()
            .map_err(|error| error.to_string())?;
        let lock = owner.lock();
        let NativeRuntimeSourceV6::SystemCuda(runtime) = &lock.config.system_runtime_lock.runtime
        else {
            return Err("PALS system receipt requires CUDA".into());
        };
        if !parent.matches(
            lock.config.process_generation,
            &lock.canonical_sha256,
            runtime.cuda_device_id,
        ) {
            return Err("PALS receipt supervisor differs from original owner".into());
        }
        parent.ensure_live().map_err(|error| error.to_string())?;
        let directory = owner
            .runtime_directory_capability()
            .map_err(|error| error.to_string())?;
        let writer = crate::rovent_bootstrap::launch::SystemReceiptWriter::open(
            &directory,
            std::path::Path::new(""),
        )
        .map_err(|error| error.to_string())?;
        Ok(Self {
            owner,
            parent,
            writer,
        })
    }

    fn write_startup(
        &self,
        owner: &Arc<PinnedPalsSystemStartupV6>,
        parent: &Arc<crate::rovent_bootstrap::launch::ConnectedParent>,
        receipt: &rz_experiments::SystemRuntimeReceiptV6,
    ) -> Result<(), String> {
        if !Arc::ptr_eq(&self.owner, owner) || !Arc::ptr_eq(&self.parent, parent) {
            return Err("PALS receipt owner or supervisor changed".into());
        }
        self.parent
            .ensure_live()
            .map_err(|error| error.to_string())?;
        self.writer.write(
            "system-runtime-startup.v6.json",
            receipt,
            &self.owner.lock().config.system_runtime_lock,
        )
    }

    /// A separate observation does not change the older runtime receipt or
    /// claim equivalence of the entire workspace or supervisor executable.
    fn write_execution_source(&self) -> Result<(), String> {
        use crate::pals_adaptive_startup::{
            PALS_SYSTEM_V6_EXECUTION_SOURCE_COMPONENT, PalsSystemSourceObservationV6,
            compiled_pals_system_v6_execution_source_digest,
        };
        self.parent
            .ensure_live()
            .map_err(|error| error.to_string())?;
        let lock = self.owner.lock();
        let observation = match self.owner.system_v6_execution_source() {
            PalsSystemSourceObservationV6::Verified => "verified",
            PalsSystemSourceObservationV6::Missing => "unknown",
            PalsSystemSourceObservationV6::Mismatch => "mismatch",
        };
        let value = serde_json::json!({
            "domain": "rz-pals-system-v6-execution-source-observation/1",
            "scope": "rz-pals-system-v6-execution-source/1",
            "process_id": std::process::id(),
            "process_generation": lock.config.process_generation,
            "lock_sha256": lock.canonical_sha256,
            "source_policy": lock.config.source_policy,
            "declared_digest": lock.config.source_components.get(PALS_SYSTEM_V6_EXECUTION_SOURCE_COMPONENT),
            "compiled_digest": compiled_pals_system_v6_execution_source_digest(),
            "observation": observation,
            "parent_executable_verified": false,
            "gpu_physical_completion": "unknown",
        });
        let bytes = serde_json::to_vec(&value).map_err(|error| error.to_string())?;
        self.writer
            .write_bytes("pals-execution-source.v6.json", &bytes)
    }
}

/// Construct the native worker with the original system CUDA factory custody.
/// This is an internal handoff for a future V6 supervisor, not an executable
/// launch path: the caller still must verify and retain its OS resource owner,
/// retain this service and parent on failure, consume the V6
/// startup/termination records, and resolve physical completion before exit.
/// Neither `run_cli` nor the V6 arena enables this path yet.
#[cfg(all(target_os = "linux", feature = "onnx-cuda"))]
#[allow(dead_code)]
fn spawn_system_native_v6(
    owner: Arc<PinnedPalsSystemStartupV6>,
    parent: Arc<crate::rovent_bootstrap::launch::ConnectedParent>,
    admission: Arc<crate::rovent_bootstrap::launch::PalsPostImageAdmissionV6>,
    clock: rz_runtime::contracts::ContractSystemClock,
    observer: Option<Box<dyn VirtualReviewCollectionObserver>>,
) -> Result<
    PalsSystemNativeStartV6,
    crate::pals_native::ticketed::FactoryStartupSpawnRejection<
        crate::pals_native::startup_system_v6::PalsSystemFactoryCustodyV6,
    >,
> {
    use crate::pals_native::startup_system_v6::PalsSystemFactoryCustodyV6;
    use rz_search::pals::coordinator::RoleFailure;

    // The validated V6 lock carries the same V5 bounded queue declarations.
    // Host byte caps match the existing PALS ticket service profile. They are
    // resource limits, not measurements or proof of actual OS quota settings.
    let endpoint = &owner.lock().config.endpoint;
    let limits = TicketServiceLimits {
        review_token_budget: endpoint.context.token_budget as u16,
        ..TicketServiceLimits::default()
    };
    let pending_capacity = endpoint.coordinator.slices.role_tickets_max as usize;
    let command_capacity = endpoint.coordinator.slices.worker_commands_max as usize;
    let receipt_owner = Arc::clone(&owner);
    let custody = PalsSystemFactoryCustodyV6::new(owner, observer, Arc::clone(&parent), admission);
    let (receipt_sender, receipt_publication) = mpsc::sync_channel(1);
    let (snapshot_sender, startup_snapshot) = mpsc::sync_channel(1);
    let service = TicketedNativeService::spawn_with_startup_custody(
        custody,
        move |factory| {
            PalsSystemFactoryCustodyV6::initialize_runtime(factory, &parent)
                .map_err(|error| RoleFailure::Backend(error.to_string()))?;
            PalsSystemFactoryCustodyV6::initialize_sessions(factory, &parent)
                .map_err(|error| RoleFailure::Backend(error.to_string()))?;
            let snapshot = factory.parts().0.startup_snapshot();
            PalsSystemFactoryCustodyV6::prepare_ticket(factory, clock)?;
            if snapshot_sender.try_send(snapshot).is_err() {
                eprintln!("PALS system startup snapshot publication unavailable");
            }
            let publication = match PalsSystemReceiptOwnerV6::open(
                Arc::clone(&receipt_owner),
                Arc::clone(&parent),
            ) {
                Ok(writer) => {
                    let execution_source_write = writer.write_execution_source();
                    let startup_write = factory
                        .parts()
                        .0
                        .startup_runtime_report()
                        .map_err(|error| error.to_string())
                        .and_then(|report| writer.write_startup(&receipt_owner, &parent, &report));
                    PalsSystemReceiptPublicationV6 {
                        owner: Some(writer),
                        startup_write,
                        execution_source_write,
                    }
                }
                Err(error) => PalsSystemReceiptPublicationV6 {
                    owner: None,
                    startup_write: Err(error.clone()),
                    execution_source_write: Err(error),
                },
            };
            // Never block a native owner on a diagnostic consumer. The
            // caller must retain this channel and distinguish Empty and
            // Disconnected from a successfully published startup record.
            if receipt_sender.try_send(publication).is_err() {
                eprintln!("PALS system receipt publication unavailable");
            }
            factory
                .parts()
                .0
                .take_prepared_ticket()
                .map(ActualNativeBackendV5::from)
                .map(ProjectedNativeBackend::new)
                .ok_or(RoleFailure::InvalidOutput)
        },
        limits,
        pending_capacity,
        command_capacity,
    )?;
    Ok(PalsSystemNativeStartV6 {
        service: Some(service),
        receipt_publication,
        startup_snapshot,
    })
}

/// This executes before any lock, source asset, runtime or model is opened.
/// The response is an observation of this binary, not external provenance.
pub(super) fn source_inventory() -> Result<(), AdaptiveBootstrapError> {
    use crate::pals_adaptive_startup::{
        PALS_SYSTEM_V6_EXECUTION_SOURCE_COMPONENT, PALS_SYSTEM_V6_SOURCE_COMPONENT,
        compiled_pals_system_v6_execution_source_digest, compiled_pals_system_v6_source_digest,
    };
    let mut components = compiled_source_components_v5()
        .map_err(|error| failure("SystemV6SourceInventory", &error))?;
    components.insert(
        PALS_SYSTEM_V6_SOURCE_COMPONENT.to_owned(),
        compiled_pals_system_v6_source_digest(),
    );
    components.insert(
        PALS_SYSTEM_V6_EXECUTION_SOURCE_COMPONENT.to_owned(),
        compiled_pals_system_v6_execution_source_digest(),
    );
    let raw_commit = env!("RZ_PALS_COMPILED_SOURCE_COMMIT");
    let commit = (raw_commit.len() == 40
        && raw_commit
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && raw_commit.bytes().any(|byte| byte != b'0'))
    .then_some(raw_commit);
    let output = serde_json::json!({
        "domain": "rz-pals-system-compiled-source-v6/1",
        "source_commit": commit,
        "source_commit_observation": if commit.is_some() { "declared_by_build" } else { "unknown" },
        "source_components": components,
        "source_component_scope": "v5-registry-plus-pals-system-v6-boundary-and-execution",
        "mapped_binary_verified": false,
        "provider_initialized": false,
        "neural_execution": false,
        "training_execution": false
    });
    println!(
        "{}",
        serde_json::to_string(&output)
            .map_err(|error| failure("SystemV6SourceInventory", &error))?
    );
    Ok(())
}

pub(super) fn run_cli(arguments: Vec<String>) -> Result<(), AdaptiveBootstrapError> {
    let started = Instant::now();
    let options = PalsSystemStartupOptionsV6::parse(&arguments)
        .map_err(|error| failure("SystemV6Arguments", &error))?;
    if !options.preflight {
        // Do not disguise an unimplemented native factory as a functional
        // engine or fall back to V5's CUDA bundle startup.
        return Err(boundary("SystemV6NativeFactoryUnsupported"));
    }
    let until = started
        .checked_add(Duration::from_millis(options.startup_ms))
        .ok_or_else(|| boundary("SystemV6StartupDeadline"))?;
    let cancelled = Arc::new(AtomicBool::new(false));
    let cwd = std::env::current_dir().map_err(|error| failure("WorkingDirectory", &error))?;
    let owner = PinnedPalsSystemStartupV6::open(
        &options.lock_path,
        &options.lock_sha256,
        options.startup_ms,
        &cwd,
        started,
        until,
        Arc::clone(&cancelled),
    )
    .map_err(|error| failure("SystemV6PinnedStartup", &error))?;
    owner
        .verify_unchanged()
        .map_err(|error| failure("SystemV6SourceOwner", &error))?;
    // Use this executable's actual compiled coordinator identity, never a
    // fabricated digest or the optional claimed source identity in the lock.
    let components = compiled_source_components_v5()
        .map_err(|error| failure("SystemV6SourceInventory", &error))?;
    let coordinator = components
        .get("coordinator")
        .ok_or_else(|| boundary("SystemV6CoordinatorIdentity"))?;
    let implementation = Digest(
        parse_sha256(coordinator)
            .map_err(|error| failure("SystemV6CoordinatorIdentity", &error))?,
    );
    owner
        .verify_unchanged()
        .map_err(|error| failure("SystemV6SourceOwner", &error))?;
    owner
        .publish_preflight()
        .map_err(|error| failure("SystemV6MetadataReceipt", &error))?;
    let clock = ProcessClock::new(ProcessEpoch(owner.lock().config.process_generation));
    let identity = EngineIdentity {
        name: format!("RoveZero PALS adaptive {}", owner.lock().config.endpoint.id),
        author: "RoveZero contributors".into(),
    };
    let mut settings = EngineSettings {
        max_workers: 1,
        shutdown_limit: Duration::from_millis(owner.lock().config.native_options.drain_ms),
        ..EngineSettings::default()
    };
    settings.search.untimed_limit =
        Duration::from_millis(owner.lock().config.window.wall_time_max_ms);
    let result = protocol::serve_driver(
        Arc::new(protocol::RulesOnlyPreflightDriver {
            identity,
            implementation,
        }),
        clock,
        Arc::new(OwnerRegistry::default()),
        settings,
    )
    .map_err(|error| failure("SystemV6MetadataProtocol", &error));
    // Keep every original source FD and the private-root capability alive
    // until the process has actually finished serving its preflight protocol.
    drop(owner);
    result
}
