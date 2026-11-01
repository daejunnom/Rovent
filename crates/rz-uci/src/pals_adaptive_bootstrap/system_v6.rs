//! Separate PALS V6 metadata preflight; it does not construct a native worker.
use super::*;
use crate::pals_adaptive_startup::{PalsSystemStartupOptionsV6, PinnedPalsSystemStartupV6};

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
    let cancelled = AtomicBool::new(false);
    let cwd = std::env::current_dir().map_err(|error| failure("WorkingDirectory", &error))?;
    let owner = PinnedPalsSystemStartupV6::open(
        &options.lock_path,
        &options.lock_sha256,
        options.startup_ms,
        &cwd,
        until,
        &cancelled,
    )
    .map_err(|error| failure("SystemV6PinnedStartup", &error))?;
    owner
        .verify_unchanged(until, &cancelled)
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
        .verify_unchanged(until, &cancelled)
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
