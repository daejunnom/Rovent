//! Separate PALS V6 metadata preflight; it does not construct a native worker.
use super::*;
use crate::pals_adaptive_startup::{PalsSystemStartupOptionsV6, PinnedPalsSystemStartupV6};

/// Construct the native worker with the original system CUDA factory custody.
/// This is an internal handoff for a future V6 supervisor, not an executable
/// launch path: the caller still must derive the limits from its own sealed
/// resource plan, retain this service and parent on failure, consume the V6
/// startup/termination records, and resolve physical completion before exit.
/// Neither `run_cli` nor the V6 arena enables this path yet.
#[cfg(all(target_os = "linux", feature = "onnx-cuda"))]
#[allow(dead_code)]
fn spawn_system_native_v6(
    owner: Arc<PinnedPalsSystemStartupV6>,
    parent: Arc<crate::rovent_bootstrap::launch::ConnectedParent>,
    clock: rz_runtime::contracts::ContractSystemClock,
    observer: Option<Box<dyn VirtualReviewCollectionObserver>>,
    receipt_writer: Option<Arc<crate::rovent_bootstrap::launch::SystemReceiptWriter>>,
    limits: TicketServiceLimits,
    pending_capacity: usize,
    command_capacity: usize,
) -> Result<
    ActualService,
    crate::pals_native::ticketed::FactoryStartupSpawnRejection<
        crate::pals_native::startup_system_v6::PalsSystemFactoryCustodyV6,
    >,
> {
    use crate::pals_native::startup_system_v6::PalsSystemFactoryCustodyV6;
    use rz_search::pals::coordinator::RoleFailure;

    let runtime_lock = owner.lock().config.system_runtime_lock.clone();
    let custody = PalsSystemFactoryCustodyV6::new(owner, observer, Arc::clone(&parent));
    TicketedNativeService::spawn_with_startup_custody(
        custody,
        move |factory| {
            PalsSystemFactoryCustodyV6::initialize_runtime(factory, &parent)
                .map_err(|error| RoleFailure::Backend(error.to_string()))?;
            PalsSystemFactoryCustodyV6::initialize_sessions(factory, &parent)
                .map_err(|error| RoleFailure::Backend(error.to_string()))?;
            PalsSystemFactoryCustodyV6::prepare_ticket(factory, clock)?;
            if let Some(writer) = receipt_writer.as_ref() {
                let report = factory.parts().0.startup_runtime_report();
                let result = report
                    .map_err(|error| error.to_string())
                    .and_then(|report| {
                        writer.write("system-runtime-startup.v6.json", &report, &runtime_lock)
                    });
                if let Err(error) = result {
                    // Evidence publication is separate from the original
                    // worker's physical lifetime and functional outcome.
                    eprintln!("PALS system startup receipt warning: {error}");
                }
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
    )
}

/// This executes before any lock, source asset, runtime or model is opened.
/// The response is an observation of this binary, not external provenance.
pub(super) fn source_inventory() -> Result<(), AdaptiveBootstrapError> {
    use crate::pals_adaptive_startup::{
        PALS_SYSTEM_V6_SOURCE_COMPONENT, compiled_pals_system_v6_source_digest,
    };
    let mut components = compiled_source_components_v5()
        .map_err(|error| failure("SystemV6SourceInventory", &error))?;
    components.insert(
        PALS_SYSTEM_V6_SOURCE_COMPONENT.to_owned(),
        compiled_pals_system_v6_source_digest(),
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
        "source_component_scope": "v5-registry-plus-pals-system-v6-boundary",
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
