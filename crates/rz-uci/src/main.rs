use rz_contracts::ProcessEpoch;
use rz_uci::search_driver::SearchSessionDriver;
#[cfg(feature = "onnx-cpu")]
use rz_uci::{
    EngineIdentity,
    engine::EvaluatorFactory,
    native_attestation::{ReceiptWriter, StartupReceiptV1, TerminationReceiptV1},
    native_bootstrap::{NativeConfig, NativeCpuFactory, NativeProvider},
    native_profile::{ProfileWriter, ProfiledWrite},
};
use rz_uci::{
    bootstrap::CpuMockFactory,
    engine::{self, EngineProcess, EngineSettings, OwnerRegistry, ProcessClock},
    forward_lines,
};
#[cfg(all(feature = "onnx-cuda", target_os = "linux"))]
use rz_uci::{
    native_bootstrap::NativeCudaFactory,
    native_cuda_attestation::{CudaReceiptWriter, CudaStartupReceiptV1, CudaTerminationReceiptV1},
};
use std::{io, sync::Arc, thread, time::Duration};

fn main() {
    if let Err(error) = run() {
        eprintln!("RoveZero: {error}");
        std::process::exit(2);
    }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments
        .iter()
        .any(|argument| argument == "--build-capabilities-json")
    {
        if arguments.len() != 1 {
            return Err("--build-capabilities-json must be the only argument; no runtime, model or engine was started".into());
        }
        return print_build_capabilities();
    }
    let work_receipts = SearchWorkOptions::take(&mut arguments)?;
    let mut search = None;
    arguments.retain(|argument| {
        if let Some(value) = argument.strip_prefix("--search=") {
            if search.is_some() {
                search = Some("duplicate".to_owned());
            } else {
                search = Some(value.to_owned());
            }
            false
        } else {
            true
        }
    });
    match search.as_deref().unwrap_or("puct") {
        "puct" => {}
        "cpu" => return run_own_cpu(arguments, work_receipts),
        "pals" => return run_pals(arguments, work_receipts),
        _ => return Err("unsupported or duplicate --search; expected puct, cpu, or pals".into()),
    }
    if work_receipts.is_some() {
        return Err("search work receipts require the CPU or PALS driver; legacy PUCT retains its existing receipt path".into());
    }
    if arguments
        .iter()
        .any(|argument| argument == "--onnx-cpu" || argument == "--onnx-cuda")
    {
        return run_native(arguments);
    }
    run_mock(arguments)
}

fn print_build_capabilities() -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    // This branch runs before every provider/model/driver parser. These are
    // compile-time facts and the current no-load state, never runtime/provider
    // availability or an inference-admission/placement witness. Semver and
    // Rust target constants are JSON-safe without the optional serde feature.
    writeln!(
        io::stdout().lock(),
        "{{\"schema\":\"rz-uci-build-capability/1\",\"target_os\":\"{}\",\"target_arch\":\"{}\",\"package_version\":\"{}\",\"compile_features\":{{\"onnx_cpu\":{},\"onnx_cuda\":{},\"experimental_io_binding\":{}}},\"cuda_path_compiled\":{},\"native_runtime_loaded\":false,\"native_model_loaded\":false}}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        env!("CARGO_PKG_VERSION"),
        cfg!(feature = "onnx-cpu"),
        cfg!(feature = "onnx-cuda"),
        cfg!(feature = "experimental-io-binding"),
        cfg!(all(feature = "onnx-cuda", target_os = "linux")),
    )?;
    Ok(())
}

fn run_pals(
    arguments: Vec<String>,
    work_receipts: Option<SearchWorkOptions>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut model = None;
    let mut rounds = None;
    let mut cpu_nodes = None;
    let mut cpu_depth = None;
    let mut situations = None;
    let mut post_repair_recheck = None;
    let mut resolver_policy = None;
    let mut cpu_resume = None;
    let mut arena_wire = None;
    let mut archive_root = None;
    let mut archive_repository = None;
    let mut archive_limits = PalsArchiveLimitOptions::default();
    let mut native = PalsNativeOptions::default();
    for argument in arguments {
        if archive_limits.take(&argument)? {
            continue;
        }
        if native.cuda_record_pages.take(&argument)? {
            continue;
        }
        if let Some(value) = argument.strip_prefix("--pals-cuda-warm-max-leases=") {
            if native
                .cuda_warm_max_leases
                .replace(value.parse()?)
                .is_some()
            {
                return Err("duplicate CUDA Warm lease bound".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-warm-device-bytes-max=") {
            if native
                .cuda_warm_device_bytes_max
                .replace(value.parse()?)
                .is_some()
            {
                return Err("duplicate CUDA Warm owned-payload byte bound".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-model=") {
            if model.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS model".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-max-rounds=") {
            if rounds.replace(value.parse::<u64>()?).is_some() {
                return Err("duplicate PALS round limit".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-max-cpu-nodes=") {
            if cpu_nodes.replace(value.parse::<u64>()?).is_some() {
                return Err("duplicate PALS CPU node limit".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cpu-depth=") {
            if cpu_depth.replace(value.parse::<u16>()?).is_some() {
                return Err("duplicate PALS CPU depth".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-post-repair-recheck=") {
            if post_repair_recheck.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS post-Repair recheck policy".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-model-profile=") {
            if native.model_profile.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS model profile".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-resolver-policy=") {
            if resolver_policy.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS resolver policy".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cpu-resume=") {
            if cpu_resume.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS CPU resume policy".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-arena-wire=") {
            if arena_wire.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS arena wire".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-archive-root=") {
            if archive_root.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS archive root".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-archive-repository=") {
            if archive_repository.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS archive repository boundary".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cpu-checker=") {
            if native.checker.selection.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS CPU checker selection".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cpu-profile=") {
            if native.checker.profile.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS CPU checker profile".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cpu-profile-sha256=") {
            if native
                .checker
                .profile_hash
                .replace(value.to_owned())
                .is_some()
            {
                return Err("duplicate PALS CPU checker profile digest".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-max-situations=") {
            if situations.replace(value.parse::<usize>()?).is_some() {
                return Err("duplicate PALS situation capacity".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-export-manifest=") {
            if native.manifest.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS export manifest".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-export-sha256=") {
            if native.manifest_hash.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS export hash".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-runtime-path=") {
            if native.runtime.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS runtime path".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-runtime-sha256=") {
            if native.runtime_hash.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS runtime hash".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-runtime-cache-root=") {
            if native
                .runtime_cache_root
                .replace(value.to_owned())
                .is_some()
            {
                return Err("duplicate PALS runtime cache root".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-output-root=") {
            if native.output_root.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS output root".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-launch-sha256=") {
            if native.launch_hash.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS launch digest".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-endpoint-id=") {
            if native.endpoint_id.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS endpoint ID".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-provider=") {
            if native.provider.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS provider".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-bundle=") {
            if native.cuda_bundle.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS CUDA bundle".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-bundle-sha256=") {
            if native.cuda_bundle_hash.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS CUDA bundle hash".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-device=") {
            if native.cuda_device.replace(value.parse::<i32>()?).is_some() {
                return Err("duplicate PALS CUDA device".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-session-arena-bytes=") {
            if native
                .cuda_session_arena
                .replace(value.parse::<u64>()?)
                .is_some()
            {
                return Err("duplicate PALS CUDA session allocator limit".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-host-record-pages=") {
            if native
                .host_record_pages
                .replace(value.parse::<bool>()?)
                .is_some()
            {
                return Err("duplicate PALS host record page option".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-private-warm=") {
            if native
                .private_warm
                .replace(value.parse::<bool>()?)
                .is_some()
            {
                return Err("duplicate PALS private warm option".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-private-warm=") {
            if native
                .cuda_private_warm
                .replace(value.parse::<bool>()?)
                .is_some()
            {
                return Err("duplicate PALS CUDA private warm option".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-device-public-memory=") {
            if native
                .device_public_memory
                .replace(value.parse::<bool>()?)
                .is_some()
            {
                return Err("duplicate PALS device public memory option".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-control-mode=") {
            if native.cuda_control_mode.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS CUDA control mode".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-control-inventory=") {
            if native
                .cuda_control_inventory
                .replace(value.to_owned())
                .is_some()
            {
                return Err("duplicate PALS CUDA control inventory".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-control-inventory-sha256=") {
            if native
                .cuda_control_inventory_hash
                .replace(value.to_owned())
                .is_some()
            {
                return Err("duplicate PALS CUDA control inventory hash".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-profile-root=") {
            if native.cuda_profile_root.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS CUDA profile root".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-profile-parent=") {
            if native
                .cuda_profile_parent
                .replace(value.to_owned())
                .is_some()
            {
                return Err("duplicate PALS CUDA profile parent".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-startup-probe-timeout-ms=") {
            let value = value
                .parse::<u64>()
                .map_err(|_| "invalid PALS startup probe timeout")?;
            if !(1..=180_000).contains(&value) {
                return Err("PALS startup probe timeout must be 1..180000 milliseconds".into());
            }
            if native.startup_probe_timeout_ms.replace(value).is_some() {
                return Err("duplicate PALS startup probe timeout".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-shutdown-timeout-ms=") {
            take_pals_shutdown_timeout(&mut native, value)?;
        } else if let Some(value) = argument.strip_prefix("--pals-cudnn-conv-search=") {
            if native.cudnn_conv_search.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS cuDNN convolution diagnostic selection".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-loading-profile=") {
            if native
                .cuda_loading_profile
                .replace(value.to_owned())
                .is_some()
            {
                return Err("duplicate PALS CUDA loading profile".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-loading-profile-sha256=") {
            if native
                .cuda_loading_profile_hash
                .replace(value.to_owned())
                .is_some()
            {
                return Err("duplicate PALS CUDA loading profile SHA-256".into());
            }
        } else {
            return Err("PALS requires an explicit model and finite PALS flags; LC0/ORT arguments are not implicitly reused".into());
        }
    }
    let max_cpu_nodes = cpu_nodes.unwrap_or(100_000);
    let rounds = rounds.unwrap_or(16);
    let cpu_depth = cpu_depth.unwrap_or(2);
    let config = rz_search::pals::engine::PalsConfig {
        max_nodes: situations.unwrap_or(4096),
        ..Default::default()
    };
    config.validate()?;
    let external_checker = native.checker.is_external()?;
    let selected_resolver = pals_resolver_policy(resolver_policy.as_deref(), external_checker)?;
    let selected_resume = match cpu_resume.as_deref() {
        None | Some("completed-iteration") => rz_search::cpu::CpuResumePolicy::CompletedIteration,
        Some("paused-stack") => rz_search::cpu::CpuResumePolicy::PausedStack,
        _ => return Err("PALS CPU resume requires completed-iteration or paused-stack".into()),
    };
    if external_checker && selected_resume == rz_search::cpu::CpuResumePolicy::PausedStack {
        return Err(
            "PALS PausedStack requires an owned CPU checker; no foreign helper was started".into(),
        );
    }
    if external_checker && cpu_resume.is_some() {
        return Err("PALS explicit native CPU resume policy requires an owned checker; no foreign helper was started".into());
    }
    let v4 = pals_arena_wire_v4(arena_wire.as_deref())?;
    let archive = pals_archive_selection(archive_root, archive_repository, archive_limits, v4)?;
    native.validate_warm_observation_selection(v4)?;
    let cuda_record_pages =
        native
            .cuda_record_pages
            .validate(model.as_deref(), &native, external_checker)?;
    // Close unsupported combinations before feature dispatch, native asset
    // admission, model loading or a foreign checker lifecycle can begin.
    let refinement_policy =
        pals_refinement_policy(post_repair_recheck.as_deref(), external_checker)?;
    if refinement_policy != rz_search::pals::engine::PostRepairRecheckPolicy::Disabled
        && !refinement_policy.uses_frozen_model_wdl()
        && selected_resolver != rz_search::pals::engine::ResolverPolicy::OwnRawRestricted
    {
        return Err("PALS post-Repair recheck v1 requires the own raw resolver; no checker profile or model was loaded".into());
    }
    if external_checker && model.as_deref() != Some("onnx") {
        return Err("external PALS CPU checker requires the explicit contextual-WDL ONNX model; the legal-order mock has no model-value endpoint".into());
    }
    if rounds == 0
        || rounds > 1_000_000
        || max_cpu_nodes == 0
        || cpu_depth == 0
        || cpu_depth > rz_search::cpu::CpuConfig::default().max_depth
    {
        return Err(
            "PALS round/CPU work limits must be finite and within the registered profile".into(),
        );
    }
    if model.as_deref() == Some("onnx") {
        pals_model_profile(native.model_profile.as_deref())?;
        if work_receipts.is_some() {
            return Err("native PALS uses pals-output-root/launch-sha256/endpoint-id for one combined native/search receipt".into());
        }
        return run_native_pals(
            native,
            cuda_record_pages,
            config,
            rounds,
            max_cpu_nodes,
            cpu_depth,
            refinement_policy,
            selected_resolver,
            selected_resume,
            resolver_policy.is_some(),
            v4,
            archive,
        );
    }
    if model.as_deref() != Some("legal-order-mock") {
        return Err("PALS requires explicit --pals-model=legal-order-mock or onnx; no model fallback was started".into());
    }
    if native.any() {
        return Err(
            "PALS mock selection cannot accept neural assets or a provider declaration".into(),
        );
    }
    if selected_resolver != rz_search::pals::engine::ResolverPolicy::OwnRawRestricted
        || refinement_policy.uses_frozen_model_wdl()
    {
        return Err("model WDL policies require a frozen ONNX value endpoint; the legal-order mock has no model value".into());
    }
    let archive_startup_deadline = match archive.as_ref().and_then(|archive| archive.runtime_limits)
    {
        Some(limits) => Some(
            std::time::Instant::now()
                .checked_add(std::time::Duration::from_millis(
                    limits.load_deadline_max_ms,
                ))
                .ok_or("PALS archive startup deadline outside finite clock domain")?,
        ),
        None => None,
    };
    let archive_startup_cancel = std::sync::atomic::AtomicBool::new(false);
    let driver = Arc::new(
        if !v4
            && selected_resume == rz_search::cpu::CpuResumePolicy::CompletedIteration
            && resolver_policy.is_none()
        {
            rz_uci::search_driver::PalsSessionDriver::new_with_refinement_policy(
                config,
                rz_search::pals::engine::LegalOrderRoleMock,
                rz_search::cpu::CpuConfig::default(),
                rounds,
                max_cpu_nodes,
                cpu_depth,
                rz_uci::EngineIdentity {
                    name: "RoveZero PALS explicit legal-order CPU mock + own CPU_R".into(),
                    author: "RoveZero contributors".into(),
                },
                refinement_policy,
            )?
        } else {
            let cpu = rz_search::cpu::CpuEngine::with_resume_policy(
                rz_search::cpu::CpuConfig::default(),
                selected_resume,
            )?;
            let checker = rz_search::cpu_checker::OwnedCpuChecker::new(cpu)?;
            rz_uci::search_driver::PalsSessionDriver::new_with_boxed_checker_and_policies(
                config,
                rz_search::pals::engine::LegalOrderRoleMock,
                Box::new(checker),
                rounds,
                max_cpu_nodes,
                cpu_depth,
                rz_uci::EngineIdentity {
                    name: "RoveZero PALS explicit legal-order CPU mock + own CPU_R".into(),
                    author: "RoveZero contributors".into(),
                },
                selected_resolver,
                refinement_policy,
            )?
        },
    );
    if let Some(archive) = archive {
        driver.enable_archive(archive.config)?;
        if let Some(limits) = archive.runtime_limits {
            driver.set_archive_runtime_limits(limits)?;
            driver.observe_archive_startup(
                archive_startup_deadline.ok_or("PALS archive startup deadline missing")?,
                &archive_startup_cancel,
            )?;
        }
    }
    #[cfg(feature = "search-work-receipts")]
    if v4 {
        driver.enable_followup_execution(None, None)?;
    }
    let work_receipts = if v4 {
        Some(
            work_receipts
                .ok_or("PALS V4 mock wire requires explicit search work output/launch/endpoint")?
                .with_pals_followup(&driver)?,
        )
    } else {
        work_receipts
    };
    let mut settings = EngineSettings::default();
    settings.search.max_simulations = max_cpu_nodes;
    serve_search_process(driver, settings, work_receipts)?;
    Ok(())
}

fn pals_refinement_policy(
    selected: Option<&str>,
    external_checker: bool,
) -> Result<rz_search::pals::engine::PostRepairRecheckPolicy, Box<dyn std::error::Error>> {
    use rz_search::pals::engine::PostRepairRecheckPolicy;
    let policy = match selected {
        None => PostRepairRecheckPolicy::Disabled,
        Some("same-repaired-line-once-v1") => PostRepairRecheckPolicy::SameRepairedLineOnceV1,
        Some("actual-opponent-continuation-v1") => PostRepairRecheckPolicy::ActualOpponentContinuationV1,
        Some("frozen-model-wdl-v2") => PostRepairRecheckPolicy::FrozenModelWdlV2,
        Some("iterative-frozen-model-wdl-v2") => PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
        Some(_) => return Err("PALS post-Repair recheck requires an exact explicit same-repaired-line-once-v1, actual-opponent-continuation-v1, frozen-model-wdl-v2, or iterative-frozen-model-wdl-v2 lane; omission retains disabled".into()),
    };
    if external_checker
        && policy != PostRepairRecheckPolicy::Disabled
        && !policy.uses_frozen_model_wdl()
    {
        return Err("PALS post-Repair recheck v1 supports own CPU/Rules evidence only; no external checker or model was started".into());
    }
    Ok(policy)
}
fn pals_resolver_policy(
    selected: Option<&str>,
    external: bool,
) -> Result<rz_search::pals::engine::ResolverPolicy, Box<dyn std::error::Error>> {
    use rz_search::pals::engine::ResolverPolicy;
    let policy = match selected {
        None if external => ResolverPolicy::ModelWdlRestricted,
        None | Some("own-raw-restricted") => ResolverPolicy::OwnRawRestricted,
        Some("model-wdl-restricted") => ResolverPolicy::ModelWdlRestricted,
        _ => {
            return Err("PALS resolver requires own-raw-restricted or model-wdl-restricted".into());
        }
    };
    if external && policy == ResolverPolicy::OwnRawRestricted {
        return Err(
            "PALS own raw resolver requires an owned checker; no profile or model was loaded"
                .into(),
        );
    }
    Ok(policy)
}
fn pals_model_profile(
    selected: Option<&str>,
) -> Result<rz_eval::pals_model::PalsModelProfile, Box<dyn std::error::Error>> {
    use rz_eval::pals_model::PalsModelProfile;
    match selected {
        None | Some("full_line_interaction_v2") => Ok(PalsModelProfile::FullLineInteractionV2),
        Some("legacy_summary_v1") => Ok(PalsModelProfile::LegacySummaryV1),
        Some("full_line_v2") => Ok(PalsModelProfile::FullLineV2),
        Some("interaction_head_v2") => Ok(PalsModelProfile::InteractionHeadV2),
        _ => Err("unsupported PALS model profile".into()),
    }
}
fn pals_arena_wire_v4(selected: Option<&str>) -> Result<bool, Box<dyn std::error::Error>> {
    let v4 = match selected {
        None => false,
        Some("v4") => true,
        _ => return Err("PALS arena wire supports only explicit v4".into()),
    };
    if v4 && !cfg!(feature = "search-work-receipts") {
        return Err("PALS V4 wire requires search-work-receipts".into());
    }
    Ok(v4)
}
fn pals_archive_config(
    root: Option<String>,
    repository: Option<String>,
) -> Result<Option<rz_search::pals::store::ArchiveConfig>, Box<dyn std::error::Error>> {
    match (root, repository) {
        (None, None) => Ok(None),
        (Some(root), Some(repository)) => {
            let root = std::path::PathBuf::from(root);
            let repository = std::path::PathBuf::from(repository);
            if !root.is_absolute() || !repository.is_absolute() {
                return Err("PALS archive root and repository boundary must be absolute".into());
            }
            Ok(Some(rz_search::pals::store::ArchiveConfig::new(
                root, repository,
            )))
        }
        _ => Err("PALS archive requires both its absolute root and repository boundary".into()),
    }
}

#[derive(Default)]
struct PalsArchiveLimitOptions {
    game: Option<u64>,
    global: Option<u64>,
    index_entries: Option<u64>,
    index_bytes: Option<u64>,
    load_bytes: Option<u64>,
    load_ms: Option<u64>,
}
impl PalsArchiveLimitOptions {
    fn take(&mut self, argument: &str) -> Result<bool, Box<dyn std::error::Error>> {
        for (prefix, slot) in [
            ("--pals-archive-game-bytes-max=", &mut self.game),
            ("--pals-archive-global-bytes-max=", &mut self.global),
            ("--pals-archive-index-entries-max=", &mut self.index_entries),
            ("--pals-archive-index-bytes-max=", &mut self.index_bytes),
            ("--pals-archive-load-bytes-max=", &mut self.load_bytes),
            ("--pals-archive-load-deadline-max-ms=", &mut self.load_ms),
        ] {
            if let Some(value) = argument.strip_prefix(prefix) {
                if slot.replace(value.parse()?).is_some() {
                    return Err("duplicate archive runtime bound".into());
                }
                return Ok(true);
            }
        }
        Ok(false)
    }
    fn any(&self) -> bool {
        [
            self.game,
            self.global,
            self.index_entries,
            self.index_bytes,
            self.load_bytes,
            self.load_ms,
        ]
        .iter()
        .any(Option::is_some)
    }
}
struct PalsArchiveSelection {
    config: rz_search::pals::store::ArchiveConfig,
    runtime_limits: Option<rz_search::pals::store::ArchiveRuntimeLimits>,
}
fn pals_archive_selection(
    root: Option<String>,
    repository: Option<String>,
    limits: PalsArchiveLimitOptions,
    v4: bool,
) -> Result<Option<PalsArchiveSelection>, Box<dyn std::error::Error>> {
    if !v4 {
        if limits.any() {
            return Err("archive runtime bounds require explicit V4 wire".into());
        }
        return Ok(
            pals_archive_config(root, repository)?.map(|config| PalsArchiveSelection {
                config,
                runtime_limits: None,
            }),
        );
    }
    if root.is_none() && repository.is_none() && !limits.any() {
        return Ok(None);
    }
    let [
        Some(game),
        Some(global),
        Some(entries),
        Some(index),
        Some(load),
        Some(ms),
    ] = [
        limits.game,
        limits.global,
        limits.index_entries,
        limits.index_bytes,
        limits.load_bytes,
        limits.load_ms,
    ]
    else {
        return Err(
            "V4 archive requires all six explicit runtime bounds and both absolute roots".into(),
        );
    };
    let (Some(root), Some(repository)) = (root, repository) else {
        return Err("V4 archive requires both absolute roots".into());
    };
    let root = std::path::PathBuf::from(root);
    let repository = std::path::PathBuf::from(repository);
    if !root.is_absolute()
        || !repository.is_absolute()
        || game == 0
        || game > 256 * 1024 * 1024
        || global < game
        || global > 4 * 1024 * 1024 * 1024
        || entries == 0
        || entries > u32::MAX as u64
        || index == 0
        || index > 16 * 1024 * 1024
        || load == 0
        || load > game.min(16 * 1024 * 1024)
        || ms == 0
        || ms > 180_000
    {
        return Err("V4 archive runtime bounds are outside the actual owner contract".into());
    }
    let config =
        rz_search::pals::store::ArchiveConfig::with_quotas(root, repository, game, global)?;
    let runtime_limits = rz_search::pals::store::ArchiveRuntimeLimits {
        game_bytes_max: game,
        global_bytes_max: global,
        index_entries_max: u32::try_from(entries)?,
        index_bytes_max: index,
        load_bytes_max: load,
        load_deadline_max_ms: ms,
        record_payload_bytes_max: config.record_bytes,
        max_load_pins: u32::try_from(config.max_load_pins)?,
    };
    Ok(Some(PalsArchiveSelection {
        config,
        runtime_limits: Some(runtime_limits),
    }))
}

#[derive(Default)]
struct PalsNativeOptions {
    model_profile: Option<String>,
    checker: PalsCheckerOptions,
    manifest: Option<String>,
    manifest_hash: Option<String>,
    runtime: Option<String>,
    runtime_hash: Option<String>,
    runtime_cache_root: Option<String>,
    provider: Option<String>,
    output_root: Option<String>,
    launch_hash: Option<String>,
    endpoint_id: Option<String>,
    cuda_bundle: Option<String>,
    cuda_bundle_hash: Option<String>,
    cuda_device: Option<i32>,
    cuda_session_arena: Option<u64>,
    host_record_pages: Option<bool>,
    private_warm: Option<bool>,
    cuda_private_warm: Option<bool>,
    cuda_warm_max_leases: Option<u32>,
    cuda_warm_device_bytes_max: Option<u64>,
    device_public_memory: Option<bool>,
    cuda_control_mode: Option<String>,
    cuda_control_inventory: Option<String>,
    cuda_control_inventory_hash: Option<String>,
    cuda_profile_root: Option<String>,
    cuda_profile_parent: Option<String>,
    cuda_loading_profile: Option<String>,
    cuda_loading_profile_hash: Option<String>,
    cudnn_conv_search: Option<String>,
    startup_probe_timeout_ms: Option<u64>,
    shutdown_timeout_ms: Option<u64>,
    cuda_record_pages: PalsCudaRecordPagesOptions,
}

const PALS_CUDA_RECORD_PAGES_RESOURCE_JSON_LIMIT: usize = 8192;

#[derive(Default)]
struct PalsCudaRecordPagesOptions {
    mode: Option<String>,
    manifest: Option<String>,
    manifest_hash: Option<String>,
    graph: Option<String>,
    graph_hash: Option<String>,
    resources: Option<String>,
}

#[cfg(all(feature = "onnx-cpu", feature = "experimental-io-binding"))]
#[derive(Debug)]
struct PalsCudaRecordPagesSelection {
    manifest: std::path::PathBuf,
    manifest_hash: [u8; 32],
    graph: std::path::PathBuf,
    graph_hash: [u8; 32],
    resources: rz_eval::pals_device_resources::NativeCudaRecordPagesResourceInput,
}
#[cfg(not(all(feature = "onnx-cpu", feature = "experimental-io-binding")))]
type PalsCudaRecordPagesSelection = ();

impl PalsCudaRecordPagesOptions {
    fn any(&self) -> bool {
        self.mode.is_some()
            || self.manifest.is_some()
            || self.manifest_hash.is_some()
            || self.graph.is_some()
            || self.graph_hash.is_some()
            || self.resources.is_some()
    }

    fn take(&mut self, argument: &str) -> Result<bool, Box<dyn std::error::Error>> {
        let (slot, value, name) = if let Some(value) =
            argument.strip_prefix("--pals-cuda-record-pages=")
        {
            (&mut self.mode, value, "mode")
        } else if let Some(value) = argument.strip_prefix("--pals-packing-manifest=") {
            (&mut self.manifest, value, "manifest")
        } else if let Some(value) = argument.strip_prefix("--pals-packing-manifest-sha256=") {
            (&mut self.manifest_hash, value, "manifest SHA-256")
        } else if let Some(value) = argument.strip_prefix("--pals-packing-graph=") {
            (&mut self.graph, value, "graph")
        } else if let Some(value) = argument.strip_prefix("--pals-packing-graph-sha256=") {
            (&mut self.graph_hash, value, "graph SHA-256")
        } else if let Some(value) = argument.strip_prefix("--pals-cuda-record-pages-resources=") {
            (&mut self.resources, value, "resources")
        } else {
            return Ok(false);
        };
        if slot.is_some() {
            return Err(format!("duplicate PALS CUDA record pages {name}").into());
        }
        if name == "resources" && value.len() > PALS_CUDA_RECORD_PAGES_RESOURCE_JSON_LIMIT {
            return Err("PALS CUDA record pages inline resource JSON exceeds 8192 UTF-8 bytes; no asset or native loading was started".into());
        }
        *slot = Some(value.to_owned());
        Ok(true)
    }

    /// Pure admission selection: no checker, file, cache, runtime or model is opened.
    fn validate(
        &self,
        model: Option<&str>,
        native: &PalsNativeOptions,
        external_checker: bool,
    ) -> Result<Option<PalsCudaRecordPagesSelection>, Box<dyn std::error::Error>> {
        if !self.any() {
            return Ok(None);
        }
        let (mode, manifest, manifest_hash, graph, graph_hash, resources) = match (
            self.mode.as_deref(), self.manifest.as_deref(), self.manifest_hash.as_deref(),
            self.graph.as_deref(), self.graph_hash.as_deref(), self.resources.as_deref(),
        ) {
            (Some(mode), Some(manifest), Some(manifest_hash), Some(graph), Some(graph_hash), Some(resources)) =>
                (mode, manifest, manifest_hash, graph, graph_hash, resources),
            _ => return Err("PALS CUDA record pages requires all six explicit mode/packing-manifest/pin/packing-graph/pin/inline-resources flags together; no loading was started".into()),
        };
        if mode != "registered-packing-v1" {
            return Err("unsupported PALS CUDA record pages mode; expected registered-packing-v1; no loading was started".into());
        }
        if model != Some("onnx") || native.provider.as_deref() != Some("cuda") {
            return Err("PALS CUDA record pages requires explicit ONNX/CUDA selection; no CPU or model fallback was started".into());
        }
        // Warm/host declarations remain separate opt-ins. Arena's explicit
        // legacy device=false preserves host control and does not select a lane.
        if native.private_warm.is_some()
            || native.host_record_pages.is_some()
            || native.device_public_memory.unwrap_or(false)
        {
            return Err("PALS CUDA record pages cannot mix private warm, host record pages or legacy device public memory flags; no loading was started".into());
        }
        if external_checker {
            return Err("PALS CUDA record pages supports own CPU only; external checker/helper selection was refused before profile or model loading".into());
        }
        let manifest = pals_packing_absolute_path(manifest)?;
        let graph = pals_packing_absolute_path(graph)?;
        let manifest_hash = pals_packing_sha256(manifest_hash)?;
        let graph_hash = pals_packing_sha256(graph_hash)?;
        if resources.len() > PALS_CUDA_RECORD_PAGES_RESOURCE_JSON_LIMIT {
            return Err("PALS CUDA record pages inline resource JSON exceeds 8192 UTF-8 bytes; no loading was started".into());
        }
        if native.cuda_control_mode.as_deref() != Some("inventory-v2")
            || native.cuda_control_inventory.is_none()
            || native.cuda_control_inventory_hash.is_none()
            || native.cuda_profile_root.is_some() == native.cuda_profile_parent.is_some()
        {
            return Err("PALS CUDA record pages requires complete inventory-v2 control and exactly one absolute profile root/parent; no loading was started".into());
        }
        pals_packing_absolute_path(
            native
                .cuda_control_inventory
                .as_deref()
                .ok_or("missing PALS CUDA control inventory")?,
        )?;
        pals_packing_sha256(
            native
                .cuda_control_inventory_hash
                .as_deref()
                .ok_or("missing PALS CUDA control inventory SHA-256")?,
        )?;
        let profile = pals_packing_absolute_path(
            native
                .cuda_profile_root
                .as_deref()
                .or(native.cuda_profile_parent.as_deref())
                .ok_or("missing PALS CUDA profile root/parent")?,
        )?;
        if native.cuda_profile_root.is_some() && profile.file_name().is_none() {
            return Err("PALS CUDA record pages requires a named exclusive profile root; no loading was started".into());
        }
        if !cfg!(all(
            feature = "onnx-cpu",
            feature = "onnx-cuda",
            feature = "experimental-io-binding",
            target_os = "linux"
        )) {
            return Err("PALS CUDA record pages requires onnx-cpu, onnx-cuda and experimental-io-binding on Linux; no provider or model fallback was started".into());
        }
        #[cfg(all(feature = "onnx-cpu", feature = "experimental-io-binding"))]
        {
            Ok(Some(PalsCudaRecordPagesSelection {
                manifest,
                manifest_hash,
                graph,
                graph_hash,
                resources: pals_cuda_record_pages_resources(resources)?,
            }))
        }
        #[cfg(not(all(feature = "onnx-cpu", feature = "experimental-io-binding")))]
        {
            let _ = (manifest, manifest_hash, graph, graph_hash);
            Err("PALS CUDA record pages support is not compiled; no loading was started".into())
        }
    }
}

fn pals_packing_absolute_path(
    value: &str,
) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let path = std::path::PathBuf::from(value);
    if !path.is_absolute()
        || path.components().any(|part| {
            matches!(
                part,
                std::path::Component::ParentDir | std::path::Component::CurDir
            )
        })
    {
        return Err("PALS packing paths must be absolute without parent/current-directory components; no loading was started".into());
    }
    Ok(path)
}

fn pals_packing_sha256(value: &str) -> Result<[u8; 32], Box<dyn std::error::Error>> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("PALS packing SHA-256 pins must be exactly 64 lowercase hexadecimal bytes; no loading was started".into());
    }
    let nibble = |b: u8| {
        if b.is_ascii_digit() {
            b - b'0'
        } else {
            b - b'a' + 10
        }
    };
    let mut digest = [0; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        digest[index] = (nibble(pair[0]) << 4) | nibble(pair[1]);
    }
    Ok(digest)
}

#[cfg(all(feature = "onnx-cpu", feature = "experimental-io-binding"))]
fn pals_cuda_record_pages_resources(
    value: &str,
) -> Result<
    rz_eval::pals_device_resources::NativeCudaRecordPagesResourceInput,
    Box<dyn std::error::Error>,
> {
    if value.len() > PALS_CUDA_RECORD_PAGES_RESOURCE_JSON_LIMIT {
        return Err("PALS CUDA record pages inline resource JSON exceeds 8192 UTF-8 bytes; no loading was started".into());
    }
    // The owner DTO rejects unknown, duplicate, absent and inconsistent fields.
    // This parser grants neither a namespace nor any native/device capability.
    serde_json::from_str(value).map_err(|error| {
        format!(
            "invalid PALS CUDA record pages resource declaration: {error}; no loading was started"
        )
        .into()
    })
}

#[cfg(all(feature = "onnx-cpu", feature = "experimental-io-binding"))]
fn pals_cuda_record_pages_profile_root(
    control: &std::path::Path,
) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let mut name = control
        .file_name()
        .ok_or("PALS CUDA record pages requires a named control profile root")?
        .to_os_string();
    name.push("-registered-packing-v1");
    Ok(control
        .parent()
        .ok_or("PALS CUDA control profile parent is absent")?
        .join(name))
}

#[cfg(test)]
mod pals_cuda_record_page_option_tests {
    use super::{
        PalsCudaRecordPagesOptions, PalsNativeOptions, pals_packing_absolute_path,
        pals_packing_sha256, run_pals,
    };

    fn flags() -> Vec<String> {
        let parent = std::env::temp_dir();
        vec![
            "--pals-cuda-record-pages=registered-packing-v1".into(),
            format!(
                "--pals-packing-manifest={}",
                parent.join("unopened-packing-manifest.json").display()
            ),
            format!("--pals-packing-manifest-sha256={}", "0".repeat(64)),
            format!(
                "--pals-packing-graph={}",
                parent.join("unopened-packing-graph.onnx").display()
            ),
            format!("--pals-packing-graph-sha256={}", "1".repeat(64)),
            "--pals-cuda-record-pages-resources={}".into(),
        ]
    }

    #[test]
    fn omission_preserves_legacy_selection_and_native_any() {
        let mut native = PalsNativeOptions::default();
        assert!(!native.any());
        for (model, provider) in [
            (None, None),
            (Some("legal-order-mock"), None),
            (Some("onnx"), Some("cpu")),
            (Some("onnx"), Some("cuda")),
        ] {
            native.provider = provider.map(str::to_owned);
            assert!(
                native
                    .cuda_record_pages
                    .validate(model, &native, false)
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[test]
    fn every_new_flag_is_native_and_duplicates_fail_before_dispatch() {
        for flag in flags() {
            let mut native = PalsNativeOptions::default();
            assert!(native.cuda_record_pages.take(&flag).unwrap());
            assert!(native.any());
            assert!(
                native
                    .cuda_record_pages
                    .take(&flag)
                    .unwrap_err()
                    .to_string()
                    .contains("duplicate PALS CUDA record pages")
            );
            let error = run_pals(vec![flag.clone(), flag], None).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("duplicate PALS CUDA record pages")
            );
        }
        for flag in [
            "--pals-cuda-record-pages",
            "--pals-packing-graph",
            "--pals-cuda-record-pages-unknown=value",
        ] {
            assert!(!PalsCudaRecordPagesOptions::default().take(flag).unwrap());
            assert!(run_pals(vec![flag.into()], None).is_err());
        }
    }

    #[test]
    fn partial_and_unknown_selections_refuse_unopened_assets() {
        for omitted in 0..6 {
            let mut args = flags();
            args.remove(omitted);
            args.extend(["--pals-model=onnx".into(), "--pals-provider=cuda".into()]);
            assert!(
                run_pals(args, None)
                    .unwrap_err()
                    .to_string()
                    .contains("all six explicit")
            );
        }
        for mode in [
            "",
            "true",
            "false",
            "registered-packing-v2",
            "REGISTERED-PACKING-V1",
        ] {
            let mut args = flags();
            args[0] = format!("--pals-cuda-record-pages={mode}");
            args.extend(["--pals-model=onnx".into(), "--pals-provider=cuda".into()]);
            assert!(
                run_pals(args, None)
                    .unwrap_err()
                    .to_string()
                    .contains("unsupported PALS CUDA record pages mode")
            );
        }
    }

    #[test]
    fn cpu_mock_old_lanes_and_external_helper_refuse_before_loading() {
        for (model, provider) in [("onnx", "cpu"), ("legal-order-mock", "cuda")] {
            let mut args = flags();
            args.extend([
                format!("--pals-model={model}"),
                format!("--pals-provider={provider}"),
            ]);
            assert!(
                run_pals(args, None)
                    .unwrap_err()
                    .to_string()
                    .contains("explicit ONNX/CUDA selection")
            );
        }
        for option in ["private-warm", "host-record-pages"] {
            for value in ["true", "false"] {
                let mut args = flags();
                args.extend([
                    "--pals-model=onnx".into(),
                    "--pals-provider=cuda".into(),
                    format!("--pals-{option}={value}"),
                ]);
                assert!(
                    run_pals(args, None)
                        .unwrap_err()
                        .to_string()
                        .contains("cannot mix")
                );
            }
        }
        for (value, expected) in [
            ("true", "cannot mix"),
            ("false", "complete inventory-v2 control"),
        ] {
            let mut args = flags();
            args.extend([
                "--pals-model=onnx".into(),
                "--pals-provider=cuda".into(),
                format!("--pals-device-public-memory={value}"),
            ]);
            // False reaches the next pure gate; true refuses before any gate
            // could open the intentionally unavailable packing assets.
            assert!(
                run_pals(args, None)
                    .unwrap_err()
                    .to_string()
                    .contains(expected)
            );
        }
        let mut args = flags();
        args.extend([
            "--pals-model=onnx".into(),
            "--pals-provider=cuda".into(),
            "--pals-cpu-checker=external-uci".into(),
            "--pals-cpu-profile=/intentionally-unopened-helper-profile.json".into(),
            "--pals-cpu-profile-sha256=unread-fixture-pin".into(),
        ]);
        assert!(
            run_pals(args, None)
                .unwrap_err()
                .to_string()
                .contains("supports own CPU only")
        );
    }

    #[test]
    fn incomplete_control_selection_is_refused_before_inventory_loading() {
        let parent = std::env::temp_dir();
        let inventory = format!(
            "--pals-cuda-control-inventory={}",
            parent.join("unopened-inventory.json").display()
        );
        let pin = format!("--pals-cuda-control-inventory-sha256={}", "0".repeat(64));
        let profile = format!(
            "--pals-cuda-profile-root={}",
            parent.join("unopened-profile-owner").display()
        );
        for control in [
            vec![],
            vec![
                "--pals-cuda-control-mode=inventory-v2".into(),
                inventory.clone(),
                profile.clone(),
            ],
            vec![
                "--pals-cuda-control-mode=inventory-v2".into(),
                pin.clone(),
                profile.clone(),
            ],
            vec![
                "--pals-cuda-control-mode=inventory-v2".into(),
                inventory.clone(),
                pin.clone(),
            ],
            vec![
                "--pals-cuda-control-mode=unknown".into(),
                inventory.clone(),
                pin.clone(),
                profile.clone(),
            ],
            vec![
                "--pals-cuda-control-mode=inventory-v2".into(),
                inventory,
                pin,
                profile,
                format!("--pals-cuda-profile-parent={}", parent.display()),
            ],
        ] {
            let mut args = flags();
            args.extend(["--pals-model=onnx".into(), "--pals-provider=cuda".into()]);
            args.extend(control);
            assert!(
                run_pals(args, None)
                    .unwrap_err()
                    .to_string()
                    .contains("complete inventory-v2 control")
            );
        }
    }

    #[cfg(not(all(
        feature = "onnx-cpu",
        feature = "onnx-cuda",
        feature = "experimental-io-binding",
        target_os = "linux"
    )))]
    #[test]
    fn unsupported_build_refuses_complete_selection_before_any_asset_load() {
        let parent = std::env::temp_dir();
        let mut args = flags();
        args.extend([
            "--pals-model=onnx".into(),
            "--pals-provider=cuda".into(),
            "--pals-cuda-control-mode=inventory-v2".into(),
            format!(
                "--pals-cuda-control-inventory={}",
                parent.join("unopened-inventory.json").display()
            ),
            format!("--pals-cuda-control-inventory-sha256={}", "0".repeat(64)),
            format!(
                "--pals-cuda-profile-root={}",
                parent.join("unopened-profile-owner").display()
            ),
        ]);
        assert!(
            run_pals(args, None)
                .unwrap_err()
                .to_string()
                .contains("experimental-io-binding on Linux")
        );
    }

    #[test]
    fn paths_pins_and_inline_json_byte_limit_are_pure_checks() {
        assert!(pals_packing_absolute_path("").is_err());
        assert!(pals_packing_absolute_path("relative/graph.onnx").is_err());
        assert!(
            pals_packing_absolute_path(
                &std::env::temp_dir().join("../graph.onnx").to_string_lossy()
            )
            .is_err()
        );
        assert!(pals_packing_sha256(&"01".repeat(32)).is_ok());
        for pin in [
            "0".repeat(63),
            "0".repeat(65),
            "A".repeat(64),
            "g".repeat(64),
            format!(" {}", "0".repeat(64)),
        ] {
            assert!(pals_packing_sha256(&pin).is_err());
        }
        let oversized = format!("--pals-cuda-record-pages-resources={}", " ".repeat(8193));
        assert!(
            PalsCudaRecordPagesOptions::default()
                .take(&oversized)
                .unwrap_err()
                .to_string()
                .contains("8192 UTF-8 bytes")
        );
        let non_ascii = format!("--pals-cuda-record-pages-resources={}", "가".repeat(2731));
        assert!(
            PalsCudaRecordPagesOptions::default()
                .take(&non_ascii)
                .is_err()
        );
    }

    #[cfg(all(feature = "onnx-cpu", feature = "experimental-io-binding"))]
    #[test]
    fn closed_resource_dto_rejects_missing_unknown_duplicate_zero_and_overflow() {
        use super::pals_cuda_record_pages_resources;
        let value = concat!(
            "{\"limits\":{\"max_blocks\":1,\"max_bank_whole_payload_bytes\":16384,\"max_registry_entries\":128,",
            "\"max_container_host_bytes\":65536,\"max_invocation_host_bytes\":8388608,\"max_invocation_device_bytes\":16777216},",
            "\"invocation\":{\"public_session_bytes\":128,\"packing_session_bytes\":128,\"private_session_bytes\":128,",
            "\"private_output_payload\":{\"host\":64,\"device\":64},\"original_input_and_transfer_payload\":{\"host\":64,\"device\":64},",
            "\"additional_owner_metadata_payload\":{\"host\":65536,\"device\":0}},",
            "\"packing_artifact\":{\"packing_session_bytes\":128,\"additional_owner_metadata_host_bytes\":65536,",
            "\"max_owned_artifact_host_bytes\":1048576,\"max_declared_host_bytes\":4194304,\"max_declared_device_bytes\":4194304},",
            "\"graph_inspection\":{\"max_inspection_host_bytes\":1024,\"max_wire_fields\":1024}}"
        );
        assert!(pals_cuda_record_pages_resources(value).is_ok());
        assert!(
            pals_cuda_record_pages_resources(&format!("{value}{}", " ".repeat(8192 - value.len())))
                .is_ok()
        );
        for invalid in [
            value.replacen('{', "{\"unknown\":0,", 1),
            value.replace("\"max_blocks\":1", "\"max_blocks\":1,\"max_blocks\":1"),
            value.replace("\"public_session_bytes\":128,", ""),
            value.replace(
                "\"packing_session_bytes\":128",
                "\"packing_session_bytes\":0",
            ),
            value.replace(
                "\"public_session_bytes\":128",
                "\"public_session_bytes\":18446744073709551615",
            ),
            format!("{value}{}", " ".repeat(8193 - value.len())),
        ] {
            assert!(pals_cuda_record_pages_resources(&invalid).is_err());
        }
    }

    #[cfg(all(feature = "onnx-cpu", feature = "experimental-io-binding"))]
    #[test]
    fn resident_profile_is_a_separate_named_sibling_without_creation() {
        let control = std::env::temp_dir().join("uncreated-control-owner");
        let resident = super::pals_cuda_record_pages_profile_root(&control).unwrap();
        assert_eq!(resident.parent(), control.parent());
        assert_ne!(resident, control);
        assert_eq!(
            resident.file_name().unwrap(),
            "uncreated-control-owner-registered-packing-v1"
        );
    }
}
#[derive(Default)]
struct PalsCheckerOptions {
    selection: Option<String>,
    profile: Option<String>,
    profile_hash: Option<String>,
}
impl PalsCheckerOptions {
    fn is_external(&self) -> Result<bool, Box<dyn std::error::Error>> {
        match (
            self.selection.as_deref().unwrap_or("own"),
            &self.profile,
            &self.profile_hash,
        ) {
            ("own", None, None) => Ok(false),
            ("external-uci", Some(path), Some(hash)) if !path.is_empty() && !hash.is_empty() => Ok(true),
            _ => Err("PALS CPU checker expects own without a profile, or external-uci with both absolute profile path and SHA-256; no fallback was started".into()),
        }
    }
    #[cfg(feature = "onnx-cpu")]
    fn load(
        &self,
    ) -> Result<
        Option<rz_uci::pals_checker_profile::LoadedPalsCheckerProfile>,
        Box<dyn std::error::Error>,
    > {
        if !self.is_external()? {
            return Ok(None);
        }
        Ok(Some(
            rz_uci::pals_checker_profile::PalsCheckerProfile::load(
                std::path::Path::new(self.profile.as_deref().ok_or("missing checker profile")?),
                self.profile_hash
                    .as_deref()
                    .ok_or("missing checker profile pin")?,
            )?,
        ))
    }
}
#[cfg(test)]
mod pals_checker_option_tests {
    use super::{
        PalsCheckerOptions, pals_model_profile, pals_refinement_policy, pals_resolver_policy,
        run_pals,
    };

    #[test]
    fn followup_selections_are_explicit_and_reject_own_raw_foreign_before_loading() {
        use rz_eval::pals_model::PalsModelProfile;
        use rz_search::pals::engine::{PostRepairRecheckPolicy, ResolverPolicy};
        assert_eq!(
            pals_model_profile(None).unwrap(),
            PalsModelProfile::FullLineInteractionV2
        );
        assert_eq!(
            pals_model_profile(Some("legacy_summary_v1")).unwrap(),
            PalsModelProfile::LegacySummaryV1
        );
        assert_eq!(
            pals_resolver_policy(Some("model-wdl-restricted"), false).unwrap(),
            ResolverPolicy::ModelWdlRestricted
        );
        assert_eq!(
            pals_refinement_policy(Some("frozen-model-wdl-v2"), true).unwrap(),
            PostRepairRecheckPolicy::FrozenModelWdlV2
        );
        assert_eq!(
            pals_refinement_policy(Some("iterative-frozen-model-wdl-v2"), true).unwrap(),
            PostRepairRecheckPolicy::IterativeFrozenModelWdlV2
        );
        for profile in ["legacy_summary_v1", "full_line_interaction_v2"] {
            let failure = run_pals(
                vec![
                    "--pals-model=onnx".into(),
                    format!("--pals-model-profile={profile}"),
                    "--pals-cpu-checker=external-uci".into(),
                    "--pals-cpu-profile=/intentionally-unopened-helper-profile.json".into(),
                    "--pals-cpu-profile-sha256=unread-fixture-pin".into(),
                    "--pals-resolver-policy=own-raw-restricted".into(),
                ],
                None,
            )
            .unwrap_err();
            assert!(
                failure
                    .to_string()
                    .contains("no profile or model was loaded")
            );
        }
        let failure = run_pals(
            vec![
                "--pals-model=onnx".into(),
                "--pals-cpu-checker=external-uci".into(),
                "--pals-cpu-profile=/intentionally-unopened-helper-profile.json".into(),
                "--pals-cpu-profile-sha256=unread-fixture-pin".into(),
                "--pals-cpu-resume=paused-stack".into(),
            ],
            None,
        )
        .unwrap_err();
        assert!(
            failure
                .to_string()
                .contains("PausedStack requires an owned CPU checker")
        );
        let failure = run_pals(
            vec![
                "--pals-model=onnx".into(),
                "--pals-resolver-policy=model-wdl-restricted".into(),
                "--pals-post-repair-recheck=same-repaired-line-once-v1".into(),
            ],
            None,
        )
        .unwrap_err();
        assert!(
            failure
                .to_string()
                .contains("no checker profile or model was loaded")
        );
    }

    #[test]
    fn post_repair_policy_is_an_exact_opt_in_with_legacy_omission() {
        use rz_search::pals::engine::PostRepairRecheckPolicy;
        assert_eq!(
            pals_refinement_policy(None, false).unwrap(),
            PostRepairRecheckPolicy::Disabled
        );
        assert_eq!(
            pals_refinement_policy(None, true).unwrap(),
            PostRepairRecheckPolicy::Disabled
        );
        assert_eq!(
            pals_refinement_policy(Some("same-repaired-line-once-v1"), false).unwrap(),
            PostRepairRecheckPolicy::SameRepairedLineOnceV1,
        );
        for value in [
            "",
            "disabled",
            "same_repaired_line_once_v1",
            "same-repaired-line-once-v2",
            "SAME-REPAIRED-LINE-ONCE-V1",
        ] {
            assert!(
                pals_refinement_policy(Some(value), false).is_err(),
                "{value}"
            );
        }
    }

    #[test]
    fn duplicate_or_malformed_post_repair_flags_never_reach_a_factory() {
        let flag = "--pals-post-repair-recheck=same-repaired-line-once-v1";
        let duplicate = run_pals(vec![flag.into(), flag.into()], None).unwrap_err();
        assert!(duplicate.to_string().contains("duplicate PALS post-Repair"));
        for flag in [
            "--pals-post-repair-recheck",
            "--pals-post-repair-recheck=unknown",
        ] {
            assert!(run_pals(vec![flag.into()], None).is_err());
        }
    }

    #[test]
    fn actual_opponent_continuation_is_explicit_and_does_not_alias_suffix_replay() {
        use rz_search::pals::engine::PostRepairRecheckPolicy;
        assert_eq!(
            pals_refinement_policy(Some("actual-opponent-continuation-v1"), false).unwrap(),
            PostRepairRecheckPolicy::ActualOpponentContinuationV1
        );
        assert!(pals_refinement_policy(Some("actual-opponent-continuation-v1"), true).is_err());
        for value in [
            "actual_opponent_continuation_v1",
            "actual-opponent-continuation-v2",
            "ACTUAL-OPPONENT-CONTINUATION-V1",
        ] {
            assert!(pals_refinement_policy(Some(value), false).is_err());
        }
        let duplicate = run_pals(
            vec![
                "--pals-post-repair-recheck=same-repaired-line-once-v1".into(),
                "--pals-post-repair-recheck=actual-opponent-continuation-v1".into(),
            ],
            None,
        )
        .unwrap_err();
        assert!(duplicate.to_string().contains("duplicate PALS post-Repair"));
    }

    #[test]
    fn post_repair_external_checker_is_rejected_before_profile_or_model_loading() {
        let failure = run_pals(
            vec![
                "--pals-model=onnx".into(),
                "--pals-cpu-checker=external-uci".into(),
                "--pals-cpu-profile=/intentionally-unavailable-profile.json".into(),
                "--pals-cpu-profile-sha256=unread-fixture-pin".into(),
                "--pals-post-repair-recheck=same-repaired-line-once-v1".into(),
            ],
            None,
        )
        .unwrap_err();
        assert!(failure.to_string().contains("own CPU/Rules evidence only"));
    }

    #[test]
    fn helper_selection_requires_an_explicit_complete_profile_without_fallback() {
        assert!(!PalsCheckerOptions::default().is_external().unwrap());
        for (selection, path, hash, accepted) in [
            ("own", None, None, true),
            ("own", Some("/profile.json"), Some("pin"), false),
            ("external-uci", None, None, false),
            ("external-uci", Some("/profile.json"), None, false),
            ("external-uci", None, Some("pin"), false),
            ("external-uci", Some(""), Some("pin"), false),
            ("external-uci", Some("/profile.json"), Some("pin"), true),
            ("unknown", None, None, false),
        ] {
            let options = PalsCheckerOptions {
                selection: Some(selection.into()),
                profile: path.map(str::to_owned),
                profile_hash: hash.map(str::to_owned),
            };
            assert_eq!(options.is_external().is_ok(), accepted, "{selection}");
        }
        // This parser only selects a branch. Absolute paths, byte/canonical pin,
        // assets and supported options are independently checked by the loader.
    }
}
impl PalsNativeOptions {
    fn validate_warm_observation_selection(
        &self,
        v4: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let selected = self.cuda_private_warm.unwrap_or(false);
        match (self.cuda_warm_max_leases,self.cuda_warm_device_bytes_max) {
            (None,None) if !v4 || !selected=>Ok(()),
            (Some(1),Some(bytes)) if bytes>0 && v4 && selected=>Ok(()),
            _=>Err("V4 CUDA Warm requires both explicit observation bounds (one lease and positive owned device bytes); bounds are forbidden when the lane is disabled".into()),
        }
    }
    #[cfg(all(feature = "onnx-cpu", feature = "experimental-io-binding"))]
    fn warm_observation_limits(
        &self,
    ) -> Option<rz_uci::pals_native::NativeCudaWarmObservationLimits> {
        self.cuda_warm_max_leases
            .zip(self.cuda_warm_device_bytes_max)
            .map(|(max_leases, device_bytes_max)| {
                rz_uci::pals_native::NativeCudaWarmObservationLimits {
                    max_leases,
                    device_bytes_max,
                }
            })
    }
    fn any(&self) -> bool {
        self.model_profile.is_some()
            || self.manifest.is_some()
            || self.manifest_hash.is_some()
            || self.runtime.is_some()
            || self.runtime_hash.is_some()
            || self.runtime_cache_root.is_some()
            || self.provider.is_some()
            || self.output_root.is_some()
            || self.launch_hash.is_some()
            || self.endpoint_id.is_some()
            || self.cuda_bundle.is_some()
            || self.cuda_bundle_hash.is_some()
            || self.cuda_device.is_some()
            || self.cuda_session_arena.is_some()
            || self.device_public_memory.is_some()
            || self.host_record_pages.is_some()
            || self.private_warm.is_some()
            || self.cuda_private_warm.is_some()
            || self.cuda_warm_max_leases.is_some()
            || self.cuda_warm_device_bytes_max.is_some()
            || self.cuda_control_mode.is_some()
            || self.cuda_control_inventory.is_some()
            || self.cuda_control_inventory_hash.is_some()
            || self.cuda_profile_root.is_some()
            || self.cuda_profile_parent.is_some()
            || self.cuda_loading_profile.is_some()
            || self.cuda_loading_profile_hash.is_some()
            || self.cudnn_conv_search.is_some()
            || self.startup_probe_timeout_ms.is_some()
            || self.shutdown_timeout_ms.is_some()
            || self.cuda_record_pages.any()
    }
}

fn take_pals_shutdown_timeout(
    native: &mut PalsNativeOptions,
    value: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let value = value
        .parse::<u64>()
        .map_err(|_| "invalid PALS shutdown timeout")?;
    if !(1..=30_000).contains(&value) {
        return Err("PALS shutdown timeout must be 1..30000 milliseconds".into());
    }
    if native.shutdown_timeout_ms.is_some() {
        return Err("duplicate PALS shutdown timeout".into());
    }
    native.shutdown_timeout_ms = Some(value);
    Ok(())
}

#[cfg(test)]
mod pals_shutdown_timeout_tests {
    use super::*;

    #[test]
    fn explicit_native_cleanup_budget_is_finite_and_duplicate_safe() {
        let mut native = PalsNativeOptions::default();
        for invalid in ["0", "30001", "-1", "invalid", "18446744073709551616"] {
            assert!(take_pals_shutdown_timeout(&mut native, invalid).is_err());
            assert!(native.shutdown_timeout_ms.is_none());
        }
        take_pals_shutdown_timeout(&mut native, "15000").unwrap();
        assert!(native.any());
        assert_eq!(native.shutdown_timeout_ms, Some(15_000));
        assert!(take_pals_shutdown_timeout(&mut native, "1").is_err());
        assert_eq!(native.shutdown_timeout_ms, Some(15_000));
    }
}

#[cfg(not(feature = "onnx-cpu"))]
#[allow(clippy::too_many_arguments)]
fn run_native_pals(
    _native: PalsNativeOptions,
    _cuda_record_pages: Option<PalsCudaRecordPagesSelection>,
    _config: rz_search::pals::engine::PalsConfig,
    _rounds: u64,
    _cpu_nodes: u64,
    _cpu_depth: u16,
    _refinement_policy: rz_search::pals::engine::PostRepairRecheckPolicy,
    _resolver_policy: rz_search::pals::engine::ResolverPolicy,
    _resume_policy: rz_search::cpu::CpuResumePolicy,
    _explicit_resolver: bool,
    _v4: bool,
    _archive: Option<PalsArchiveSelection>,
) -> Result<(), Box<dyn std::error::Error>> {
    Err(
        "PALS ONNX startup requires feature onnx-cpu; no provider or model fallback was started"
            .into(),
    )
}

#[cfg(feature = "onnx-cpu")]
#[allow(clippy::too_many_arguments)]
fn run_native_pals(
    native: PalsNativeOptions,
    cuda_record_pages: Option<PalsCudaRecordPagesSelection>,
    config: rz_search::pals::engine::PalsConfig,
    rounds: u64,
    cpu_nodes: u64,
    cpu_depth: u16,
    refinement_policy: rz_search::pals::engine::PostRepairRecheckPolicy,
    resolver_policy: rz_search::pals::engine::ResolverPolicy,
    resume_policy: rz_search::cpu::CpuResumePolicy,
    explicit_resolver: bool,
    v4: bool,
    archive: Option<PalsArchiveSelection>,
) -> Result<(), Box<dyn std::error::Error>> {
    // PALS owns a distinct manifest and tensors. LC0 NativeConfig and its
    // attestation cannot authorize or describe this model.
    let expected_profile = pals_model_profile(native.model_profile.as_deref())?;
    let followup_lane = v4
        || expected_profile != rz_eval::pals_model::PalsModelProfile::LegacySummaryV1
        || explicit_resolver
        || resume_policy == rz_search::cpu::CpuResumePolicy::PausedStack
        || refinement_policy.uses_frozen_model_wdl();
    // Selection-only validation precedes helper metadata, model and runtime I/O.
    let cudnn_conv_search = pals_cudnn_conv_selection(&native, expected_profile, v4)?;
    let checker_profile = native.checker.load()?;
    let checker = checker_profile
        .as_ref()
        .map(|profile| profile.create_checker())
        .transpose()?;
    let checker_handshake_timeout = checker_profile
        .as_ref()
        .map(|profile| profile.config().map(|config| config.handshake_timeout))
        .transpose()?
        .unwrap_or(Duration::ZERO); // unused for the synchronous own checker
    let provider = match native.provider.as_deref() {
        Some("cpu") => "cpu",
        Some("cuda") if cfg!(all(feature = "onnx-cuda", target_os = "linux")) => "cuda",
        Some("cuda") => {
            return Err(
                "PALS CUDA requires onnx-cuda on Linux; no CPU fallback was started".into(),
            );
        }
        _ => {
            return Err(
                "PALS requires explicit --pals-provider=cpu or cuda; no fallback was started"
                    .into(),
            );
        }
    };
    if provider == "cpu"
        && (native.cuda_bundle.is_some()
            || native.cuda_bundle_hash.is_some()
            || native.cuda_device.is_some()
            || native.cuda_session_arena.is_some()
            || native.device_public_memory.unwrap_or(false)
            || native.cuda_control_mode.is_some()
            || native.cuda_control_inventory.is_some()
            || native.cuda_control_inventory_hash.is_some()
            || native.cuda_profile_root.is_some()
            || native.cuda_profile_parent.is_some()
            || native.cuda_loading_profile.is_some()
            || native.cuda_loading_profile_hash.is_some()
            || native.startup_probe_timeout_ms.is_some())
    {
        return Err("PALS CPU selection cannot accept CUDA bundle/device/allocator options".into());
    }
    if native.device_public_memory.unwrap_or(false) && !cfg!(feature = "experimental-io-binding") {
        return Err("PALS device public memory requires explicit experimental-io-binding; no binding fallback was started".into());
    }
    if native.host_record_pages.unwrap_or(false) && native.device_public_memory.unwrap_or(false) {
        return Err("PALS host record pages and device public memory are mutually exclusive; no fallback was started".into());
    }
    if native.private_warm.unwrap_or(false)
        && (provider != "cpu" || native.device_public_memory.unwrap_or(false))
    {
        return Err("PALS private warm requires a separately pinned CPU/host warm export; CUDA/device/V are unsupported".into());
    }
    let cuda_warm = native.cuda_private_warm.unwrap_or(false);
    if cuda_warm
        && (provider != "cuda"
            || !expected_profile.uses_full_line()
            || !native.device_public_memory.unwrap_or(false)
            || native.private_warm.unwrap_or(false)
            || native.host_record_pages.unwrap_or(false)
            || native.cuda_record_pages.any())
    {
        return Err("PALS CUDA Warm requires explicit CUDA/device public memory, its own export, and no CPU Warm/host pages/resident packing selection".into());
    }
    if cuda_warm && !cfg!(feature = "experimental-io-binding") {
        return Err(
            "PALS CUDA Warm requires experimental-io-binding; no fallback was started".into(),
        );
    }
    #[cfg(feature = "experimental-io-binding")]
    let warm_observation_limits = native.warm_observation_limits();
    let loading_profile = pals_cuda_loading_profile(
        native.cuda_loading_profile.as_deref(),
        native.cuda_loading_profile_hash.as_deref(),
        provider,
        native.cuda_control_mode.as_deref() == Some("inventory-v2"),
        native.device_public_memory.unwrap_or(false),
    )?;
    let profile_root = pals_profile_root(native.cuda_profile_root, native.cuda_profile_parent)?;
    let cuda_control = match (
        native.cuda_control_mode,
        native.cuda_control_inventory,
        native.cuda_control_inventory_hash,
        profile_root,
    ) {
        (None, None, None, None) => None,
        (Some(mode), Some(inventory), Some(hash), Some(profile))
            if mode == "inventory-v2" && provider == "cuda" =>
        {
            if native.device_public_memory.unwrap_or(false) && !cuda_warm {
                return Err("PALS inventory-v2 control mode requires host public memory; no device-binding fallback was started".into());
            }
            let inventory = std::path::PathBuf::from(inventory);
            if !inventory.is_absolute() || !profile.is_absolute() {
                return Err("PALS control inventory and profile root must be absolute".into());
            }
            rz_eval::asset::parse_sha256(&hash)?;
            let policy = rz_eval::pals_onnx::PalsCudaControlPolicy::from_inventory(&inventory, &hash)?;
            Some((policy, profile))
        }
        _ => return Err("PALS CUDA control mode requires inventory-v2, inventory path, inventory SHA-256 and a fresh owned profile root or exclusive process-profile parent together; strict mode is the default".into()),
    };
    let receipt_config =
        match (native.output_root, native.launch_hash, native.endpoint_id) {
            (None, None, None) => None,
            (Some(root), Some(launch), Some(endpoint)) => {
                let root = std::path::PathBuf::from(root);
                if !root.is_absolute() {
                    return Err("PALS output root must be absolute".into());
                }
                rz_eval::asset::parse_sha256(&launch)?;
                Some((root, launch, endpoint))
            }
            _ => return Err(
                "PALS receipt output requires output root, launch digest and endpoint ID together"
                    .into(),
            ),
        };
    if checker_profile.is_some() && receipt_config.is_none() {
        return Err("external PALS checker requires the combined output root, launch digest and endpoint ID for independent owner evidence".into());
    }
    let manifest =
        std::path::PathBuf::from(native.manifest.ok_or("PALS export manifest is required")?);
    let runtime_path =
        std::path::PathBuf::from(native.runtime.ok_or("PALS runtime path is required")?);
    if !manifest.is_absolute() || !runtime_path.is_absolute() {
        return Err("PALS model/runtime paths must be absolute".into());
    }
    let manifest_hash = native
        .manifest_hash
        .ok_or("PALS export SHA-256 is required")?;
    let runtime_hash = native
        .runtime_hash
        .ok_or("PALS runtime SHA-256 is required")?;
    rz_eval::asset::parse_sha256(&manifest_hash)?;
    rz_eval::asset::parse_sha256(&runtime_hash)?;
    let cache = match native.runtime_cache_root {
        Some(path) => {
            let path = std::path::PathBuf::from(path);
            if !path.is_absolute() {
                return Err("PALS runtime cache root must be absolute".into());
            }
            rz_eval::runtime_pin::RuntimeCache::open(&path)?
        }
        None => rz_eval::runtime_pin::RuntimeCache::for_user()?,
    };
    let mut backend_config = rz_eval::pals_onnx::PalsOnnxConfig::cpu();
    backend_config.cudnn_conv_search = cudnn_conv_search;
    let pin = if provider == "cpu" {
        cache.library(&runtime_path, &runtime_hash)?
    } else {
        let bundle_path =
            std::path::PathBuf::from(native.cuda_bundle.ok_or("PALS CUDA bundle is required")?);
        if !bundle_path.is_absolute() {
            return Err("PALS CUDA bundle path must be absolute".into());
        }
        let bundle_hash = native
            .cuda_bundle_hash
            .ok_or("PALS CUDA bundle SHA-256 is required")?;
        let expected_bundle = rz_eval::asset::parse_sha256(&bundle_hash)?;
        let bytes = rz_eval::asset::read_bounded(&bundle_path, 64 * 1024)?;
        if rz_eval::asset::sha256(&bytes) != expected_bundle {
            return Err("PALS CUDA bundle descriptor differs from its SHA-256 pin".into());
        }
        let spec =
            rz_eval::runtime_pin::CudaRuntimeBundleSpec::from_json(std::str::from_utf8(&bytes)?)?;
        let core = spec
            .files
            .iter()
            .find(|entry| entry.role == rz_eval::runtime_pin::RuntimeBundleFileRole::Core)
            .ok_or("PALS CUDA bundle core is absent")?;
        if runtime_path.file_name().and_then(|name| name.to_str()) != Some(core.filename.as_str())
            || core.sha256 != runtime_hash
        {
            return Err("PALS explicit ORT core filename/hash differs from CUDA bundle".into());
        }
        let device_id = native
            .cuda_device
            .ok_or("PALS CUDA device ID is required")?;
        let arena = native
            .cuda_session_arena
            .ok_or("PALS per-session CUDA allocator limit is required")?;
        if device_id != 0 || arena != 2 * 1024 * 1024 * 1024 {
            return Err("first PALS CUDA profile requires device 0 and 2GiB per-session allocator declaration; this is not a measured peak".into());
        }
        backend_config.provider = rz_eval::onnx::Provider::Cuda {
            device_id,
            arena_bytes: usize::try_from(arena)?,
        };
        backend_config.device_public_memory = native.device_public_memory.unwrap_or(false);
        cache.cuda_bundle(
            runtime_path
                .parent()
                .ok_or("PALS CUDA core parent is absent")?,
            &spec,
        )?
    };
    let mut settings = EngineSettings::default();
    settings.search.max_simulations = cpu_nodes;
    if let Some(milliseconds) = native.shutdown_timeout_ms {
        // The arena's finite cleanup reservation must reach the actual native
        // owner. Final origin/Stats controls still require their physical ack;
        // expiry retains unknown ownership and never fabricates a fence.
        settings.shutdown_limit = Duration::from_millis(milliseconds);
    }
    let owner_options = rz_uci::pals_native::NativeOwnerOptions {
        drain_limit: settings.shutdown_limit,
        host_record_pages: native.host_record_pages.unwrap_or(false).then_some(
            rz_uci::pals_native::NativeHostRecordPageLimits {
                max_page_entries: 257,
                max_page_bytes: 2 * 1024 * 1024,
                max_transient_bytes: 2 * 1024 * 1024,
            },
        ),
    };
    #[cfg(feature = "experimental-io-binding")]
    let resident_registration = match cuda_record_pages {
        Some(selection) => {
            let (_, control_profile) = cuda_control
                .as_ref()
                .ok_or("PALS CUDA record pages requires the selected inventory control owner")?;
            let profile = pals_cuda_record_pages_profile_root(control_profile)?;
            let (limits, invocation, declaration, inspection_budget) = selection.resources.validated_parts()
                .map_err(|error| format!("invalid PALS CUDA record pages resource declaration: {error}; no native loading was started"))?;
            // One guarded reader/validator owns the pinned bytes. No ORT model
            // path reread, static CLI process or duplicate graph parser is used.
            let checked_graph = rz_eval::pals_onnx::load_checked_fixed_packing_graph(
                &selection.manifest,
                &selection.graph,
                selection.manifest_hash,
                selection.graph_hash,
                declaration,
                inspection_budget,
            )?;
            Some(
                rz_uci::pals_native::NativeCudaRecordPagesRegistration::for_deployment(
                    checked_graph,
                    limits,
                    invocation,
                    profile,
                ),
            )
        }
        None => None,
    };
    #[cfg(not(feature = "experimental-io-binding"))]
    let _ = cuda_record_pages;
    // Observe actual runtime/model factory cost independently of the probe
    // deadline. Earlier pin/cache preparation belongs to overall CLI cost.
    let model_loading_started = std::time::Instant::now();
    let mut model = match cuda_control {
        #[cfg(feature = "experimental-io-binding")]
        Some((policy, profile)) if cuda_warm => {
            let runtime = match loading_profile {
                Some(loading) => {
                    rz_eval::onnx::OrtRuntime::load_with_cuda_loading_profile(&pin, loading)?
                }
                None => rz_eval::onnx::OrtRuntime::load(&pin)?,
            };
            rz_uci::pals_native::NativeRoleModel::load_with_runtime_and_cuda_private_warm_and_observations(&manifest, &manifest_hash, runtime, backend_config, owner_options, Some((policy, profile.as_path())),warm_observation_limits)?
        }
        Some((policy, profile)) => {
            let runtime = match loading_profile {
                Some(loading) => {
                    rz_eval::onnx::OrtRuntime::load_with_cuda_loading_profile(&pin, loading)?
                }
                None => rz_eval::onnx::OrtRuntime::load(&pin)?,
            };
            #[cfg(feature = "experimental-io-binding")]
            {
                match resident_registration {
                    Some(registration) => rz_uci::pals_native::NativeRoleModel::load_with_runtime_and_registered_cuda_record_pages(
                        &manifest, &manifest_hash, runtime, backend_config, policy,
                        &profile, (owner_options, registration),
                    )?,
                    None => rz_uci::pals_native::NativeRoleModel::load_with_runtime_and_cuda_control_policy_with_options(
                        &manifest, &manifest_hash, runtime, backend_config, policy,
                        &profile, owner_options,
                    )?,
                }
            }
            #[cfg(not(feature = "experimental-io-binding"))]
            {
                rz_uci::pals_native::NativeRoleModel::load_with_runtime_and_cuda_control_policy_with_options(
                    &manifest, &manifest_hash, runtime, backend_config, policy,
                    &profile, owner_options,
                )?
            }
        }
        None if native.private_warm.unwrap_or(false) => {
            rz_uci::pals_native::NativeRoleModel::load_pinned_cpu_private_warm_with_options(
                &manifest,
                &manifest_hash,
                &pin,
                backend_config,
                owner_options,
            )?
        }
        #[cfg(feature = "experimental-io-binding")]
        None if cuda_warm => {
            let runtime = rz_eval::onnx::OrtRuntime::load(&pin)?;
            rz_uci::pals_native::NativeRoleModel::load_with_runtime_and_cuda_private_warm_and_observations(&manifest, &manifest_hash, runtime, backend_config, owner_options, None,warm_observation_limits)?
        }
        None => rz_uci::pals_native::NativeRoleModel::load_pinned_with_options(
            &manifest,
            &manifest_hash,
            &pin,
            backend_config,
            owner_options,
        )?,
    };
    let model_loading_elapsed = model_loading_started.elapsed();
    let finish = model.finish_handle();
    let mut original_startup_deadline = None;
    let archive_startup_cancel = std::sync::atomic::AtomicBool::new(false);
    let startup = (|| {
        if model.model_config().profile != expected_profile {
            return Err(rz_search::pals::engine::RoleError::Backend(
                "loaded PALS model profile differs from explicit/default selection".into(),
            ));
        }
        if v4
            && (archive.is_some()
                || resume_policy == rz_search::cpu::CpuResumePolicy::PausedStack
                || refinement_policy.uses_frozen_model_wdl()
                || cuda_warm)
        {
            model.enable_followup_execution_evidence()?;
        }
        let budget = model.configure_startup_probe_timeout(native.startup_probe_timeout_ms)?;
        model.observe_startup_loading(model_loading_elapsed)?;
        let deadline = std::time::Instant::now()
            .checked_add(budget)
            .ok_or_else(|| {
                rz_search::pals::engine::RoleError::Backend(
                    "PALS startup deadline outside finite clock domain".into(),
                )
            })?;
        original_startup_deadline = Some(deadline);
        model.prepare_startup(deadline)
    })();
    if let Err(primary) = startup {
        let mut publication = None;
        let mut failure_writer = match receipt_config.as_ref() {
            Some((root, launch, endpoint)) => {
                match open_pals_receipt(root, endpoint, launch, &runtime_hash, v4, None) {
                    Ok(mut writer) => match writer.failed_startup(finish.receipt(), &primary, None)
                    {
                        Ok(()) => Some(writer),
                        Err(error) => {
                            publication = Some(error);
                            None
                        }
                    },
                    Err(error) => {
                        publication = Some(error);
                        None
                    }
                }
            }
            None => None,
        };
        let cleanup = finish
            .finish(std::time::Instant::now() + settings.shutdown_limit)
            .err();
        if let Some(writer) = failure_writer.as_mut() {
            publication = writer.termination(finish.receipt(), false, None).err();
        }
        return Err(Box::new(PalsFinishError {
            primary: Some(Box::new(primary)),
            cleanup,
            checker_cleanup: None,
            followup_cleanup: None,
            checker_observation: None,
            publication,
            work_observation: None,
        }));
    }
    let identity = rz_uci::EngineIdentity {
        name: format!(
            "RoveZero PALS P/C ONNX {} + {} CPU_R",
            if provider == "cpu" { "CPU" } else { "CUDA" },
            if checker.is_some() {
                "external UCI"
            } else {
                "own"
            },
        ),
        author: "RoveZero contributors".into(),
    };
    // The helper is unstarted through every fallible constructor. No foreign
    // process can disappear into a constructor Drop without a shutdown receipt.
    let model_source = model.source_identity();
    let constructed = if followup_lane {
        match checker {
            Some(checker) => {
                rz_uci::search_driver::PalsSessionDriver::new_with_boxed_checker_and_policies(
                    config,
                    model,
                    Box::new(checker),
                    rounds,
                    cpu_nodes,
                    cpu_depth,
                    identity,
                    resolver_policy,
                    refinement_policy,
                )
            }
            None => rz_uci::search_driver::PalsSessionDriver::new_with_policies(
                config,
                model,
                rz_search::cpu::CpuConfig::default(),
                rounds,
                cpu_nodes,
                cpu_depth,
                identity,
                resolver_policy,
                refinement_policy,
                resume_policy,
            ),
        }
    } else {
        match checker {
            Some(checker) => {
                rz_uci::search_driver::PalsSessionDriver::new_with_checker_and_refinement_policy(
                    config,
                    model,
                    checker,
                    rounds,
                    cpu_nodes,
                    cpu_depth,
                    identity,
                    refinement_policy,
                )
            }
            None => rz_uci::search_driver::PalsSessionDriver::new_with_refinement_policy(
                config,
                model,
                rz_search::cpu::CpuConfig::default(),
                rounds,
                cpu_nodes,
                cpu_depth,
                identity,
                refinement_policy,
            ),
        }
    };
    let (driver, followup) = match constructed.and_then(|driver| {
        if let Some(archive) = archive {
            driver.enable_archive(archive.config)?;
            if let Some(limits) = archive.runtime_limits {
                driver.set_archive_runtime_limits(limits)?;
                driver.observe_archive_startup(
                    original_startup_deadline.ok_or_else(|| {
                        rz_uci::search_driver::SearchSessionFailure {
                            physical_completion:
                                rz_uci::search_driver::DriverPhysicalCompletion::Confirmed,
                            code: "PalsArchiveStartup",
                            detail: "original startup deadline missing".into(),
                        }
                    })?,
                    &archive_startup_cancel,
                )?;
            }
        }
        if v4 {
            #[cfg(feature = "experimental-io-binding")]
            let cuda_owner =
                cuda_warm.then(rz_eval::pals_onnx::cuda_private_warm_implementation_digest);
            #[cfg(not(feature = "experimental-io-binding"))]
            let cuda_owner = None;
            driver.enable_followup_execution(Some(finish.receipt().process_epoch), cuda_owner)?;
            driver.capture_warm_followup(finish.cuda_warm_followup_observation())?;
        }
        let followup = if v4 {
            Some(
                rz_uci::pals_attestation::PalsFollowupMarkerV4::capture_native(
                    &model_source,
                    &driver,
                )?,
            )
        } else {
            None
        };
        Ok((driver, followup))
    }) {
        Ok((driver, followup)) => (Arc::new(driver), followup),
        Err(primary) => {
            let cleanup = finish
                .finish(std::time::Instant::now() + settings.shutdown_limit)
                .err();
            return Err(Box::new(PalsFinishError {
                primary: Some(Box::new(primary)),
                cleanup,
                checker_cleanup: None,
                followup_cleanup: None,
                checker_observation: None,
                publication: None,
                work_observation: None,
            }));
        }
    };
    if let Some(profile) = checker_profile.as_ref() {
        let control = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let startup = driver.start_checker(
            std::time::Instant::now() + checker_handshake_timeout,
            control,
        );
        if let Err(primary) = startup {
            // Both owners are attempted independently. A failed helper start is
            // not a Native startup failure and does not invent a ready helper.
            let startup_lifecycle = driver.followup_lifecycle();
            let (checker_cleanup, cleanup, followup_cleanup) =
                finish_failed_pals_owners(&driver, &finish, settings.shutdown_limit, v4);
            let observed_work = driver.work_receipt();
            let observed_checker =
                rz_uci::pals_attestation::checker::PalsCheckerProcessReceipt::observe(
                    &driver,
                    profile.file_sha256(),
                    profile.canonical_sha256(),
                );
            let mut publication = None;
            if let Some((root, launch, endpoint)) = receipt_config.as_ref() {
                let result = (|| {
                    let mut writer = open_pals_receipt(
                        root,
                        endpoint,
                        launch,
                        &runtime_hash,
                        v4,
                        followup.clone(),
                    )?;
                    if let Ok(evidence) = &observed_checker {
                        writer.observe_checker(evidence.clone())?;
                    }
                    writer.set_followup_lifecycle(
                        startup_lifecycle.as_ref().ok().cloned().flatten(),
                    )?;
                    writer.startup(
                        finish.receipt(),
                        observed_work.as_ref().ok().cloned().flatten(),
                    )?;
                    writer.set_followup_lifecycle(driver.followup_lifecycle().ok().flatten())?;
                    writer.termination(
                        finish.receipt(),
                        false,
                        observed_work.as_ref().ok().cloned().flatten(),
                    )
                })();
                publication = result.err();
            }
            return Err(Box::new(PalsFinishError {
                primary: Some(Box::new(primary)),
                cleanup,
                checker_cleanup,
                followup_cleanup,
                checker_observation: observed_checker.err(),
                publication,
                work_observation: observed_work.err(),
            }));
        }
    }
    let startup_checker = checker_profile
        .as_ref()
        .map(|profile| {
            rz_uci::pals_attestation::checker::PalsCheckerProcessReceipt::observe(
                &driver,
                profile.file_sha256(),
                profile.canonical_sha256(),
            )
        })
        .transpose();
    let startup_checker = match startup_checker {
        Ok(evidence) => evidence,
        Err(checker_observation) => {
            let (checker_cleanup, cleanup, followup_cleanup) =
                finish_failed_pals_owners(&driver, &finish, settings.shutdown_limit, v4);
            return Err(Box::new(PalsFinishError {
                primary: None,
                cleanup,
                checker_cleanup,
                followup_cleanup,
                checker_observation: Some(checker_observation),
                publication: None,
                work_observation: None,
            }));
        }
    };
    let startup_work = match driver.work_receipt() {
        Ok(work) => work,
        Err(work_observation) => {
            let (checker_cleanup, cleanup, followup_cleanup) =
                finish_failed_pals_owners(&driver, &finish, settings.shutdown_limit, v4);
            return Err(Box::new(PalsFinishError {
                primary: None,
                cleanup,
                checker_cleanup,
                followup_cleanup,
                checker_observation: None,
                publication: None,
                work_observation: Some(work_observation),
            }));
        }
    };
    let startup_lifecycle = match driver.followup_lifecycle() {
        Ok(value) => value,
        Err(work_observation) => {
            let (checker_cleanup, cleanup, followup_cleanup) =
                finish_failed_pals_owners(&driver, &finish, settings.shutdown_limit, v4);
            return Err(Box::new(PalsFinishError {
                primary: None,
                cleanup,
                checker_cleanup,
                followup_cleanup,
                checker_observation: None,
                publication: None,
                work_observation: Some(work_observation),
            }));
        }
    };
    let mut receipts = match receipt_config {
        Some((root, launch, endpoint)) => {
            let opened = open_pals_receipt(
                &root,
                &endpoint,
                &launch,
                &runtime_hash,
                v4,
                followup.clone(),
            );
            match opened {
                Ok(mut writer) => match (|| {
                    writer.set_followup_lifecycle(startup_lifecycle.clone())?;
                    if let Some(evidence) = startup_checker.as_ref() {
                        writer.observe_checker(evidence.clone())?;
                    }
                    writer.startup(finish.receipt(), startup_work.clone())
                })() {
                    Ok(()) => Some(writer),
                    Err(publication) => {
                        let (checker_cleanup, cleanup, followup_cleanup) =
                            finish_failed_pals_owners(
                                &driver,
                                &finish,
                                settings.shutdown_limit,
                                v4,
                            );
                        return Err(Box::new(PalsFinishError {
                            primary: None,
                            cleanup,
                            checker_cleanup,
                            followup_cleanup,
                            checker_observation: None,
                            publication: Some(publication),
                            work_observation: None,
                        }));
                    }
                },
                Err(publication) => {
                    let (checker_cleanup, cleanup, followup_cleanup) =
                        finish_failed_pals_owners(&driver, &finish, settings.shutdown_limit, v4);
                    return Err(Box::new(PalsFinishError {
                        primary: None,
                        cleanup,
                        checker_cleanup,
                        followup_cleanup,
                        checker_observation: None,
                        publication: Some(publication),
                        work_observation: None,
                    }));
                }
            }
        }
        None => None,
    };
    let owners = Arc::new(OwnerRegistry::default());
    let clock = ProcessClock::new(ProcessEpoch(1));
    let process = EngineProcess::with_search_driver(driver.clone(), owners, clock);
    let served = serve_process(process, settings);
    let owner_until = std::time::Instant::now() + settings.shutdown_limit;
    let (checker_finished, finished) = finish_pals_owners_before(
        owner_until,
        |until| driver.finish_checker(until),
        |until| finish.finish(until),
    );
    let native_final = finish.receipt();
    let native_fenced = finished.is_ok()
        && native_final.physical_shutdown_confirmed
        && native_final.physical_runs_in_flight == 0
        && !native_final.quarantined;
    let mut followup_failure = if v4 {
        driver
            .finish_followup_owners(owner_until, native_fenced)
            .err()
    } else {
        None
    };
    if v4 {
        followup_failure = followup_failure.or(driver
            .seal_followup_lifecycle(
                native_fenced,
                served.is_ok(),
                finish.cuda_warm_followup_observation(),
            )
            .err());
    }
    let lifecycle = driver.followup_lifecycle();
    if let Err(error) = &lifecycle {
        followup_failure.get_or_insert(*error);
    }
    let observed_work = driver.work_receipt();
    let observed_checker = checker_profile
        .as_ref()
        .map(|profile| {
            rz_uci::pals_attestation::checker::PalsCheckerProcessReceipt::observe(
                &driver,
                profile.file_sha256(),
                profile.canonical_sha256(),
            )
        })
        .transpose();
    let publication = receipts.as_mut().and_then(|writer| {
        if let Err(error) =
            writer.set_followup_lifecycle(lifecycle.as_ref().ok().cloned().flatten())
        {
            return Some(error);
        }
        if let Ok(Some(evidence)) = &observed_checker {
            if let Err(error) = writer.observe_checker(evidence.clone()) {
                return Some(error);
            }
        }
        writer
            .termination(
                finish.receipt(),
                served.is_ok()
                    && finished.is_ok()
                    && checker_finished.is_ok()
                    && followup_failure.is_none()
                    && observed_work.is_ok()
                    && observed_checker.is_ok(),
                observed_work.as_ref().ok().cloned().flatten(),
            )
            .err()
    });
    if publication.is_some()
        || observed_work.is_err()
        || checker_finished.is_err()
        || observed_checker.is_err()
        || followup_failure.is_some()
    {
        return Err(Box::new(PalsFinishError {
            primary: served
                .err()
                .map(|error| Box::new(error) as Box<dyn std::error::Error>),
            cleanup: finished.err(),
            checker_cleanup: checker_finished.err(),
            checker_observation: observed_checker.err(),
            publication,
            work_observation: observed_work.err(),
            followup_cleanup: followup_failure,
        }));
    }
    match (served, finished) {
        (Ok(()), Ok(receipt)) => {
            eprintln!(
                "PALS native role shutdown: completed={} delivered={} consumed={} canceled={} expired={} failed={} invalid={} in_flight={} joined={} buffers_released={} quarantined={}",
                receipt.completed_role_inputs,
                receipt.delivered_role_inputs,
                receipt.search_consumed_role_inputs,
                receipt.canceled_requests,
                receipt.expired_requests,
                receipt.failed_physical_role_calls,
                receipt.invalid_role_outputs,
                receipt.physical_runs_in_flight,
                receipt.physical_shutdown_confirmed,
                receipt.native_buffers_released,
                receipt.quarantined,
            );
            Ok(())
        }
        (Err(primary), Ok(_)) => Err(Box::new(primary)),
        (primary, Err(cleanup)) => Err(Box::new(PalsFinishError {
            primary: primary
                .err()
                .map(|error| Box::new(error) as Box<dyn std::error::Error>),
            cleanup: Some(cleanup),
            checker_cleanup: None,
            followup_cleanup: None,
            checker_observation: None,
            publication: None,
            work_observation: None,
        })),
    }
}

#[cfg(feature = "onnx-cpu")]
fn pals_cudnn_conv_selection(
    native: &PalsNativeOptions,
    profile: rz_eval::pals_model::PalsModelProfile,
    v4: bool,
) -> Result<rz_eval::pals_onnx::PalsCuDnnConvSearch, &'static str> {
    use rz_eval::pals_onnx::PalsCuDnnConvSearch as Selection;
    let Some(selected) = native.cudnn_conv_search.as_deref() else {
        return Ok(Selection::LegacyImplicitExhaustive);
    };
    if !cfg!(all(feature = "onnx-cuda", target_os = "linux"))
        || native.provider.as_deref() != Some("cuda")
        || native.cuda_control_mode.as_deref() != Some("inventory-v2")
        || native.cuda_control_inventory.is_none()
        || native.cuda_control_inventory_hash.is_none()
        || (native.cuda_profile_root.is_none() && native.cuda_profile_parent.is_none())
        || native.cuda_loading_profile.is_some()
        || native.cuda_loading_profile_hash.is_some()
        || native.private_warm.unwrap_or(false)
        || native.host_record_pages.unwrap_or(false)
        || native.cuda_record_pages.any()
        || !v4
        || native.output_root.is_none()
        || native.launch_hash.is_none()
        || native.endpoint_id.is_none()
    {
        return Err(
            "PALS DEFAULT diagnostic requires explicit eager CUDA inventory control, V4 receipts and no CPU Warm/pages/packing selection",
        );
    }
    match selected {
        "default-host-control-v1"
            if !native.device_public_memory.unwrap_or(false)
                && !native.cuda_private_warm.unwrap_or(false) =>
        {
            Ok(Selection::ExperimentalDefault)
        }
        "default-device-warm-v1"
            if cfg!(feature = "experimental-io-binding")
                && native.device_public_memory.unwrap_or(false)
                && native.cuda_private_warm.unwrap_or(false)
                && profile.uses_full_line() =>
        {
            Ok(Selection::ExperimentalDefaultDeviceWarm)
        }
        _ => Err(
            "PALS cuDNN diagnostic must exactly match host control or separately pinned device-Warm selection",
        ),
    }
}

#[cfg(feature = "onnx-cpu")]
fn open_pals_receipt(
    root: &std::path::Path,
    endpoint: &str,
    launch: &str,
    runtime: &str,
    v4: bool,
    marker: Option<rz_uci::pals_attestation::PalsFollowupMarkerV4>,
) -> Result<
    rz_uci::pals_attestation::PalsReceiptWriter,
    rz_uci::process_receipts::ProcessReceiptError,
> {
    if v4 {
        rz_uci::pals_attestation::PalsReceiptWriter::open_v4(
            root, endpoint, launch, runtime, marker,
        )
    } else {
        rz_uci::pals_attestation::PalsReceiptWriter::open(root, endpoint, launch, runtime)
    }
}

/// Both owners are attempted independently against the same overall deadline.
/// A slow helper must not grant the Native owner a fresh cleanup window.
#[cfg(all(test, feature = "onnx-cpu"))]
fn finish_pals_owners_within<C, N>(
    limit: Duration,
    checker: impl FnOnce(std::time::Instant) -> C,
    native: impl FnOnce(std::time::Instant) -> N,
) -> (C, N) {
    let now = std::time::Instant::now();
    let until = now.checked_add(limit).unwrap_or(now);
    finish_pals_owners_before(until, checker, native)
}
#[cfg(any(feature = "onnx-cpu", test))]
fn finish_pals_owners_before<C, N>(
    until: std::time::Instant,
    checker: impl FnOnce(std::time::Instant) -> C,
    native: impl FnOnce(std::time::Instant) -> N,
) -> (C, N) {
    let checker_result = checker(until);
    let native_result = native(until);
    (checker_result, native_result)
}

/// Startup/publication failures still close the same actual owners. Keep the
/// independent followup failure alongside the original failure, and retain an
/// incomplete lifecycle when protocol output was never retired.
#[cfg(feature = "onnx-cpu")]
fn finish_failed_pals_owners(
    driver: &rz_uci::search_driver::PalsSessionDriver<rz_uci::pals_native::NativeRoleModel>,
    finish: &rz_uci::pals_native::NativeRoleFinishHandle,
    limit: Duration,
    v4: bool,
) -> (
    Option<rz_uci::search_driver::SearchSessionFailure>,
    Option<rz_search::pals::engine::RoleError>,
    Option<rz_contracts::ContractError>,
) {
    let now = std::time::Instant::now();
    let until = now.checked_add(limit).unwrap_or(now);
    let (checker, native) = finish_pals_owners_before(
        until,
        |deadline| driver.finish_checker(deadline),
        |deadline| finish.finish(deadline),
    );
    let receipt = finish.receipt();
    let native_fenced = native.is_ok()
        && receipt.physical_shutdown_confirmed
        && receipt.physical_runs_in_flight == 0
        && !receipt.quarantined;
    let followup = if v4 {
        let closed = driver.finish_followup_owners(until, native_fenced).err();
        let sealed = driver
            .seal_followup_lifecycle(
                native_fenced,
                false,
                finish.cuda_warm_followup_observation(),
            )
            .err();
        closed.or(sealed)
    } else {
        None
    };
    (checker.err(), native.err(), followup)
}

#[cfg(feature = "onnx-cpu")]
fn pals_cuda_loading_profile(
    profile: Option<&str>,
    expected_hash: Option<&str>,
    provider: &str,
    inventory_control: bool,
    device_public_memory: bool,
) -> Result<Option<rz_eval::onnx::NativeLoadingProfile>, Box<dyn std::error::Error>> {
    match (profile, expected_hash) {
        (None, None) => Ok(None),
        (Some("experimental-cudnn-shim-lazy-v1"), Some(hash))
            if provider == "cuda" && inventory_control && !device_public_memory =>
        {
            let profile = rz_eval::onnx::NativeLoadingProfile::CuDnnShimLazyV1;
            if rz_eval::asset::parse_sha256(hash)?
                != rz_eval::asset::sha256(profile.canonical_descriptor().as_bytes())
            {
                return Err("PALS CUDA loading profile canonical SHA-256 differs; no native loading was started".into());
            }
            Ok(Some(profile))
        }
        _ => Err("PALS experimental CUDA loading profile and canonical SHA-256 must be selected together with CUDA inventory-v2 host control; no loading-profile or CPU fallback was started".into()),
    }
}

#[cfg(feature = "onnx-cpu")]
fn pals_profile_root(
    root: Option<String>,
    parent: Option<String>,
) -> Result<Option<std::path::PathBuf>, Box<dyn std::error::Error>> {
    let path = match (root, parent) {
        (None, None) => None,
        (Some(root), None) => Some(std::path::PathBuf::from(root)),
        (None, Some(parent)) => {
            let parent = std::path::PathBuf::from(parent);
            if !parent.is_absolute() {
                return Err("PALS CUDA profile parent must be absolute".into());
            }
            let metadata = std::fs::symlink_metadata(&parent)?;
            if !metadata.is_dir() || metadata.file_type().is_symlink() {
                return Err("PALS CUDA profile parent must be an existing owned directory".into());
            }
            // Fastchess uses the same launch arguments across games. Derive a
            // fresh process-owned child here, then leave exclusive creation to
            // the backend: a reused PID or nonce never authorizes overwriting.
            let nonce = u64::try_from(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_nanos(),
            )?;
            Some(parent.join(format!(
                "pals-cuda-profile-{}-{nonce:016x}",
                std::process::id()
            )))
        }
        (Some(_), Some(_)) => {
            return Err("PALS CUDA profile root and parent are mutually exclusive".into());
        }
    };
    Ok(path)
}

#[cfg(all(test, feature = "onnx-cpu"))]
mod pals_profile_tests {
    use super::{finish_pals_owners_within, pals_cuda_loading_profile, pals_profile_root};

    #[test]
    fn conv_default_profiles_are_explicit_and_closed_before_io() {
        use super::{PalsNativeOptions, pals_cudnn_conv_selection};
        use rz_eval::{pals_model::PalsModelProfile, pals_onnx::PalsCuDnnConvSearch as Selection};
        let profile = PalsModelProfile::FullLineInteractionV2;
        let mut native = PalsNativeOptions::default();
        assert_eq!(
            pals_cudnn_conv_selection(&native, profile, false).unwrap(),
            Selection::LegacyImplicitExhaustive
        );
        native.cudnn_conv_search = Some("default-host-control-v1".into());
        assert!(pals_cudnn_conv_selection(&native, profile, true).is_err());
        native.provider = Some("cuda".into());
        native.cuda_control_mode = Some("inventory-v2".into());
        native.cuda_control_inventory = Some("not-read".into());
        native.cuda_control_inventory_hash = Some("not-read".into());
        native.cuda_profile_root = Some("not-created".into());
        native.output_root = Some("not-created".into());
        native.launch_hash = Some("not-read".into());
        native.endpoint_id = Some("diagnostic".into());
        assert_eq!(
            pals_cudnn_conv_selection(&native, profile, true).is_ok(),
            cfg!(all(feature = "onnx-cuda", target_os = "linux"))
        );
        assert!(pals_cudnn_conv_selection(&native, profile, false).is_err());
        native.cudnn_conv_search = Some("default-device-warm-v1".into());
        assert!(pals_cudnn_conv_selection(&native, profile, true).is_err());
        native.device_public_memory = Some(true);
        native.cuda_private_warm = Some(true);
        assert_eq!(
            pals_cudnn_conv_selection(&native, profile, true).is_ok(),
            cfg!(all(
                feature = "onnx-cuda",
                feature = "experimental-io-binding",
                target_os = "linux"
            ))
        );
        native.cuda_loading_profile = Some("cudnn-shim-lazy-v1".into());
        assert!(pals_cudnn_conv_selection(&native, profile, true).is_err());
        native.cuda_loading_profile = None;
        native.provider = Some("cpu".into());
        assert!(pals_cudnn_conv_selection(&native, profile, true).is_err());
    }

    #[test]
    fn both_owner_cleanup_attempts_share_one_deadline_even_after_helper_failure() {
        use std::cell::RefCell;
        use std::time::{Duration, Instant};
        for limit in [Duration::from_secs(1), Duration::ZERO] {
            let observed = RefCell::new(Vec::new());
            let started = Instant::now();
            let (checker, native) = finish_pals_owners_within(
                limit,
                |until| {
                    observed.borrow_mut().push(("checker", until));
                    Err::<(), _>("helper cleanup failed")
                },
                |until| {
                    observed.borrow_mut().push(("native", until));
                    Err::<(), _>("native cleanup failed")
                },
            );
            assert_eq!(checker, Err("helper cleanup failed"));
            assert_eq!(native, Err("native cleanup failed"));
            let observed = observed.into_inner();
            assert_eq!(observed.len(), 2);
            assert_eq!(observed[0].0, "checker");
            assert_eq!(observed[1].0, "native");
            assert_eq!(observed[0].1, observed[1].1);
            assert!(observed[0].1 >= started);
            assert!(observed[0].1 <= Instant::now() + limit);
            if limit.is_zero() {
                assert!(observed[1].1 <= Instant::now());
            }
        }
    }

    #[test]
    fn loading_profile_is_an_explicit_canonical_host_control_selection() {
        const SHIM: &str = "experimental-cudnn-shim-lazy-v1";
        const SHA: &str = "3084f27e678d6a6750713265bfb921e90687f199fb71484ff2c475000982ee6d";
        assert_eq!(
            pals_cuda_loading_profile(None, None, "cpu", false, false).unwrap(),
            None
        );
        assert_eq!(
            pals_cuda_loading_profile(None, None, "cuda", false, false).unwrap(),
            None
        );
        assert_eq!(
            pals_cuda_loading_profile(Some(SHIM), Some(SHA), "cuda", true, false).unwrap(),
            Some(rz_eval::onnx::NativeLoadingProfile::CuDnnShimLazyV1)
        );
        // These are pure option checks. No cache, file, model or native runtime
        // is opened, including the unsupported device/public-memory case.
        for (profile, digest, provider, control, device) in [
            (Some(SHIM), None, "cuda", true, false),
            (None, Some(SHA), "cuda", true, false),
            (Some(SHIM), Some(SHA), "cpu", true, false),
            (Some(SHIM), Some(SHA), "cuda", false, false),
            (Some(SHIM), Some(SHA), "cuda", true, true),
            (
                Some("eager-cuda12-cudnn9-v1"),
                Some(SHA),
                "cuda",
                true,
                false,
            ),
            (Some(SHIM), Some("00"), "cuda", true, false),
            (
                Some(SHIM),
                Some("0000000000000000000000000000000000000000000000000000000000000000"),
                "cuda",
                true,
                false,
            ),
        ] {
            assert!(pals_cuda_loading_profile(profile, digest, provider, control, device).is_err());
        }
    }

    #[test]
    fn exact_root_and_absent_control_preserve_existing_arguments() {
        assert_eq!(pals_profile_root(None, None).unwrap(), None);
        let exact = std::env::temp_dir().join("pals-explicit-profile-root");
        assert_eq!(
            pals_profile_root(Some(exact.to_string_lossy().into_owned()), None).unwrap(),
            Some(exact)
        );
        assert!(pals_profile_root(Some("root".into()), Some("parent".into())).is_err());
    }

    #[test]
    fn process_profile_parent_derives_but_does_not_create_a_child() {
        let parent = std::env::temp_dir();
        let child = pals_profile_root(None, Some(parent.to_string_lossy().into_owned()))
            .unwrap()
            .unwrap();
        assert_eq!(child.parent(), Some(parent.as_path()));
        assert!(
            child
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(&format!("pals-cuda-profile-{}-", std::process::id()))
        );
        assert!(!child.exists());
        assert!(pals_profile_root(None, Some("relative-profile-parent".into())).is_err());
    }
}

#[cfg(feature = "onnx-cpu")]
#[derive(Debug)]
struct PalsFinishError {
    primary: Option<Box<dyn std::error::Error>>,
    cleanup: Option<rz_search::pals::engine::RoleError>,
    checker_cleanup: Option<rz_uci::search_driver::SearchSessionFailure>,
    followup_cleanup: Option<rz_contracts::ContractError>,
    checker_observation: Option<rz_uci::search_driver::SearchSessionFailure>,
    publication: Option<rz_uci::process_receipts::ProcessReceiptError>,
    work_observation: Option<rz_contracts::ContractError>,
}
#[cfg(feature = "onnx-cpu")]
impl std::fmt::Display for PalsFinishError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(primary) = &self.primary {
            write!(formatter, "{primary}; ")?;
        }
        if let Some(cleanup) = &self.cleanup {
            write!(formatter, "PALS physical shutdown failed: {cleanup}; ")?;
        }
        if let Some(cleanup) = &self.checker_cleanup {
            write!(formatter, "PALS CPU helper shutdown failed: {cleanup}; ")?;
        }
        if let Some(cleanup) = &self.followup_cleanup {
            write!(formatter, "PALS followup cleanup failed: {cleanup}; ")?;
        }
        if let Some(observation) = &self.checker_observation {
            write!(
                formatter,
                "PALS CPU helper observation failed: {observation}; "
            )?;
        }
        if let Some(publication) = &self.publication {
            write!(formatter, "PALS receipt publication failed: {publication}")?;
        }
        if let Some(observation) = &self.work_observation {
            write!(formatter, "PALS work observation failed: {observation}")?;
        }
        Ok(())
    }
}
#[cfg(feature = "onnx-cpu")]
impl std::error::Error for PalsFinishError {}

fn run_own_cpu(
    arguments: Vec<String>,
    work_receipts: Option<SearchWorkOptions>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut max_depth = None;
    let mut max_nodes = None;
    let mut tt_entries = None;
    let mut arena_wire = None;
    for argument in arguments {
        if let Some(value) = argument.strip_prefix("--cpu-max-depth=") {
            if max_depth.replace(value.parse::<u16>()?).is_some() {
                return Err("duplicate CPU depth limit".into());
            }
        } else if let Some(value) = argument.strip_prefix("--cpu-max-nodes=") {
            if max_nodes.replace(value.parse::<u64>()?).is_some() {
                return Err("duplicate CPU node limit".into());
            }
        } else if let Some(value) = argument.strip_prefix("--cpu-tt-entries=") {
            if tt_entries.replace(value.parse::<usize>()?).is_some() {
                return Err("duplicate CPU TT limit".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-arena-wire=") {
            if arena_wire.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS arena wire".into());
            }
        } else {
            return Err("CPU search only accepts --cpu-max-depth, --cpu-max-nodes, --cpu-tt-entries and an explicit PALS arena wire; neural provider/model flags are inapplicable".into());
        }
    }
    let work_receipts = if pals_arena_wire_v4(arena_wire.as_deref())? {
        Some(
            work_receipts
                .ok_or("V4 OwnCpu wire requires explicit search work output/launch/endpoint")?
                .with_v4()?,
        )
    } else {
        work_receipts
    };
    let max_nodes = max_nodes.unwrap_or(100_000);
    // Explicit finite limits: default profile is own bootstrap CPU_R, not NNUE.
    let config = rz_search::cpu::CpuConfig {
        max_depth: max_depth.unwrap_or(8),
        tt_entries: tt_entries.unwrap_or(65_536),
        ..Default::default()
    };
    let driver = Arc::new(rz_uci::search_driver::CpuSessionDriver::new(
        config, max_nodes,
    )?);
    let mut settings = EngineSettings::default();
    // Legacy field is the UCI work admission cap. CPU interprets go nodes as
    // actual CPU nodes; no completed simulation or NN receipt is fabricated.
    settings.search.max_simulations = max_nodes;
    serve_search_process(driver, settings, work_receipts)?;
    Ok(())
}

struct SearchWorkOptions {
    root: std::path::PathBuf,
    launch_sha256: String,
    endpoint_id: String,
    #[cfg(feature = "search-work-receipts")]
    v4: bool,
    #[cfg(feature = "search-work-receipts")]
    followup: Option<rz_uci::pals_attestation::PalsFollowupMarkerV4>,
}
impl SearchWorkOptions {
    fn with_pals_followup<M: rz_search::pals::engine::RoleModel + 'static>(
        self,
        driver: &rz_uci::search_driver::PalsSessionDriver<M>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        #[cfg(feature = "search-work-receipts")]
        {
            let mut receipt = self;
            receipt.v4 = true;
            receipt.followup =
                Some(rz_uci::pals_attestation::PalsFollowupMarkerV4::capture_search(driver)?);
            Ok(receipt)
        }
        #[cfg(not(feature = "search-work-receipts"))]
        {
            let _ = (self, driver);
            Err("PALS V4 mock wire requires search-work-receipts".into())
        }
    }
    fn with_v4(self) -> Result<Self, Box<dyn std::error::Error>> {
        #[cfg(feature = "search-work-receipts")]
        {
            let mut receipt = self;
            receipt.v4 = true;
            Ok(receipt)
        }
        #[cfg(not(feature = "search-work-receipts"))]
        {
            let _ = self;
            Err("V4 wire requires search-work-receipts".into())
        }
    }
    fn take(arguments: &mut Vec<String>) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        let mut root = None;
        let mut launch = None;
        let mut endpoint = None;
        let mut duplicate = false;
        arguments.retain(|argument| {
            let replaced = if let Some(value) = argument.strip_prefix("--search-work-output-root=")
            {
                root.replace(value.to_owned())
            } else if let Some(value) = argument.strip_prefix("--search-work-launch-sha256=") {
                launch.replace(value.to_owned())
            } else if let Some(value) = argument.strip_prefix("--search-work-endpoint-id=") {
                endpoint.replace(value.to_owned())
            } else {
                return true;
            };
            duplicate |= replaced.is_some();
            false
        });
        if duplicate {
            return Err("duplicate search work receipt argument".into());
        }
        match (root, launch, endpoint) {
            (None, None, None) => Ok(None),
            (Some(root), Some(launch_sha256), Some(endpoint_id)) => {
                let root = std::path::PathBuf::from(root);
                if !root.is_absolute() {
                    return Err("search work output root must be absolute".into());
                }
                rz_eval::asset::parse_sha256(&launch_sha256)?;
                #[cfg(not(feature = "search-work-receipts"))]
                {
                    let _ = (root, launch_sha256, endpoint_id);
                    Err("unsupported feature: search work receipts require feature search-work-receipts; no driver was started".into())
                }
                #[cfg(feature = "search-work-receipts")]
                {
                    Ok(Some(Self {
                        root,
                        launch_sha256,
                        endpoint_id,
                        v4: false,
                        followup: None,
                    }))
                }
            }
            _ => Err(
                "search work output requires root, launch digest and endpoint ID together".into(),
            ),
        }
    }
}

fn serve_search_process(
    driver: Arc<dyn SearchSessionDriver>,
    settings: EngineSettings,
    receipt: Option<SearchWorkOptions>,
) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "search-work-receipts")]
    let mut receipts = match receipt {
        Some(receipt) => {
            let mut writer = match (receipt.v4, receipt.followup) {
                (true, Some(marker)) => rz_uci::pals_attestation::SearchWorkReceiptWriter::open_v4(
                    &receipt.root,
                    &receipt.endpoint_id,
                    &receipt.launch_sha256,
                    marker,
                )?,
                (true, None) => {
                    rz_uci::pals_attestation::SearchWorkReceiptWriter::open_v4_without_pals(
                        &receipt.root,
                        &receipt.endpoint_id,
                        &receipt.launch_sha256,
                    )?
                }
                (false, _) => rz_uci::pals_attestation::SearchWorkReceiptWriter::open(
                    &receipt.root,
                    &receipt.endpoint_id,
                    &receipt.launch_sha256,
                )?,
            };
            writer.set_followup_lifecycle(driver.followup_lifecycle()?)?;
            writer.startup(driver.work_receipt()?)?;
            Some(writer)
        }
        None => None,
    };
    #[cfg(not(feature = "search-work-receipts"))]
    if receipt.is_some() {
        return Err("unsupported feature: search-work-receipts".into());
    }
    let process = EngineProcess::with_search_driver(
        Arc::clone(&driver),
        Arc::new(OwnerRegistry::default()),
        ProcessClock::new(ProcessEpoch(1)),
    );
    let served = serve_process(process, settings);
    #[cfg(feature = "search-work-receipts")]
    {
        let observed = driver.work_receipt();
        let followup_failure = driver.seal_followup_lifecycle(served.is_ok()).err();
        let lifecycle = driver.followup_lifecycle();
        let publication = receipts.as_mut().and_then(|writer| {
            if let Err(error) =
                writer.set_followup_lifecycle(lifecycle.as_ref().ok().cloned().flatten())
            {
                return Some(error);
            }
            writer
                .termination(
                    observed.as_ref().ok().cloned().flatten(),
                    served.is_ok() && followup_failure.is_none() && lifecycle.is_ok(),
                )
                .err()
        });
        if observed.is_err()
            || publication.is_some()
            || followup_failure.is_some()
            || lifecycle.is_err()
        {
            return Err(Box::new(SearchWorkFinishError {
                primary: served.err(),
                observation: observed.err().or(followup_failure).or(lifecycle.err()),
                publication,
            }));
        }
    }
    served?;
    Ok(())
}
#[cfg(feature = "search-work-receipts")]
#[derive(Debug)]
struct SearchWorkFinishError {
    primary: Option<engine::EngineError>,
    observation: Option<rz_contracts::ContractError>,
    publication: Option<rz_uci::process_receipts::ProcessReceiptError>,
}
#[cfg(feature = "search-work-receipts")]
impl std::fmt::Display for SearchWorkFinishError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "search finish failed: primary={:?}, observation={:?}, publication={:?}",
            self.primary, self.observation, self.publication
        )
    }
}
#[cfg(feature = "search-work-receipts")]
impl std::error::Error for SearchWorkFinishError {}

fn run_mock(arguments: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut explicit_mock = false;
    let mut model_adapter = None;
    let mut delay = Duration::ZERO;
    for argument in arguments {
        if argument == "--cpu-mock" {
            explicit_mock = true;
        } else if let Some(value) = argument.strip_prefix("--model-adapter=") {
            if model_adapter.replace(value.to_owned()).is_some() {
                return Err("duplicate model adapter".into());
            }
        } else if let Some(value) = argument.strip_prefix("--mock-delay-ms=") {
            delay = Duration::from_millis(value.parse()?);
        } else {
            return Err(
                "unsupported argument; supported flags: --cpu-mock [--mock-delay-ms=0..1000]"
                    .into(),
            );
        }
    }
    if !explicit_mock {
        return Err("this executable requires --cpu-mock; no neural backend is enabled".into());
    }
    let owners = Arc::new(OwnerRegistry::default());
    let clock = ProcessClock::new(ProcessEpoch(1)); // Issued once for this executable process.
    match model_adapter.as_deref().unwrap_or("lc0") {
        "lc0" => {
            let factory = Arc::new(CpuMockFactory::new(&owners, delay)?);
            serve_process(
                EngineProcess::new(factory, owners, clock),
                EngineSettings::default(),
            )?;
        }
        "entity-candidate-mock" => {
            if !delay.is_zero() {
                return Err("entity mock does not support scripted delay".into());
            }
            let factory = Arc::new(rz_uci::bootstrap::EntityMockFactory::new(&owners)?);
            let process = EngineProcess::new(factory.clone(), owners, clock).with_identity(
                rz_uci::EngineIdentity {
                    name: "RoveZero entity candidate CPU mock".into(),
                    author: "RoveZero contributors".into(),
                },
            );
            let served = serve_process(process, EngineSettings::default());
            let finished = factory.finish(std::time::Instant::now() + Duration::from_secs(2));
            served?;
            finished?;
        }
        _ => return Err("unsupported model adapter".into()),
    }
    Ok(())
}

#[cfg(not(feature = "onnx-cpu"))]
fn run_native(arguments: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    // Do not inspect or echo native asset arguments when this provider is absent.
    if arguments.iter().any(|argument| argument == "--onnx-cuda") {
        return Err("unsupported feature: --onnx-cuda requires build feature onnx-cuda on Linux; no provider was started".into());
    }
    Err(
        "unsupported feature: --onnx-cpu requires build feature onnx-cpu; no provider was started"
            .into(),
    )
}

#[cfg(feature = "onnx-cpu")]
fn run_native(arguments: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    // NativeConfig owns exact flag/asset/hash validation and bounded public errors.
    let config = NativeConfig::parse(arguments)?;
    match config.provider() {
        NativeProvider::Cpu => run_cpu(config),
        NativeProvider::Cuda => run_cuda(config),
    }
}
#[cfg(feature = "onnx-cpu")]
fn run_cpu(config: NativeConfig) -> Result<(), Box<dyn std::error::Error>> {
    let owners = Arc::new(OwnerRegistry::default());
    let clock = ProcessClock::new(ProcessEpoch(1));
    let factory = Arc::new(NativeCpuFactory::load(&owners, &config)?);
    let trace = factory.source_trace();
    let mut profile_writer = if config.profiling_requested() {
        Some(ProfileWriter::open(&config)?)
    } else {
        None
    };
    let mut receipts = if config.attestation_requested() {
        Some(ReceiptWriter::open(&config)?)
    } else {
        None
    };
    let startup = if let Some(writer) = receipts.as_mut() {
        let startup = StartupReceiptV1::capture(&factory, &clock)?;
        writer.startup(&startup)?;
        Some(startup)
    } else {
        None
    };
    let process =
        EngineProcess::new(factory.clone(), owners, clock).with_identity(EngineIdentity {
            name: "RoveZero LC0 weights ONNX CPU integration".into(),
            author: "RoveZero contributors".into(),
        });
    let settings = config.engine_settings();
    // Retain the native owner outside EngineProcess. Report/drain acceptance runs
    // for both successful protocol service and EngineError; neither implies GPU support.
    let served = serve_native_process(process, settings, trace.clone());
    let finished = factory.finish(served);
    if let (Some(writer), Some(startup)) = (receipts.as_mut(), startup.as_ref()) {
        let receipt = match TerminationReceiptV1::from_result(startup, &finished) {
            Ok(receipt) => receipt,
            Err(error) => return Err(Box::new(error.retaining(finished))),
        };
        if let Err(error) = writer.termination(&receipt) {
            // Publication must not replace an original service/drain failure.
            return Err(Box::new(error.retaining(finished)));
        }
    }
    if let (Some(writer), Some(trace)) = (profile_writer.as_mut(), trace.as_ref())
        && let Err(error) = writer.publish(trace, &finished)
    {
        return Err(Box::new(error.retaining(finished)));
    }
    let report = finished?;
    eprintln!("{report}");
    Ok(())
}

#[cfg(all(
    feature = "onnx-cpu",
    not(all(feature = "onnx-cuda", target_os = "linux"))
))]
fn run_cuda(_config: NativeConfig) -> Result<(), Box<dyn std::error::Error>> {
    Err("explicit CUDA native startup requires the onnx-cuda feature on Linux; no provider was started".into())
}
#[cfg(all(feature = "onnx-cuda", target_os = "linux"))]
fn run_cuda(config: NativeConfig) -> Result<(), Box<dyn std::error::Error>> {
    let owners = Arc::new(OwnerRegistry::default());
    let clock = ProcessClock::new(ProcessEpoch(1));
    let factory = Arc::new(NativeCudaFactory::load(&owners, &config)?);
    let trace = factory.source_trace();
    #[cfg(feature = "experimental-batch")]
    let mut batch_writer = if config.batch_attestation_requested() {
        Some(rz_uci::native_batch_attestation::BatchReceiptWriter::open(
            &config,
        )?)
    } else {
        None
    };
    #[cfg(feature = "experimental-batch")]
    let batch_startup = if let Some(writer) = batch_writer.as_mut() {
        Some(writer.startup(
            CudaStartupReceiptV1::capture(&factory, &clock)?,
            &config.engine_settings(),
        )?)
    } else {
        None
    };
    let mut profile_writer = if config.profiling_requested() {
        Some(ProfileWriter::open(&config)?)
    } else {
        None
    };
    let mut receipts = if config.attestation_requested() {
        Some(CudaReceiptWriter::open(&config)?)
    } else {
        None
    };
    let startup = if let Some(writer) = receipts.as_mut() {
        let startup = CudaStartupReceiptV1::capture(&factory, &clock)?;
        writer.startup(&startup)?;
        // Capture the exact settings subsequently served. Experimental paths
        // do not issue this fixed-profile attestation (factory guard above).
        let search = rz_uci::native_cuda_attestation::CudaSearchReceiptV1::capture(
            &startup,
            &config.engine_settings(),
        )?;
        writer.search_config(&search)?;
        Some(startup)
    } else {
        None
    };
    let process =
        EngineProcess::new(factory.clone(), owners, clock).with_identity(EngineIdentity {
            name: "RoveZero LC0 weights ONNX CUDA integration".into(),
            author: "RoveZero contributors".into(),
        });
    let served = serve_native_process(process, config.engine_settings(), trace.clone());
    let finished = factory.finish(served);
    #[cfg(feature = "experimental-batch")]
    if let (Some(writer), Some(startup)) = (batch_writer.as_mut(), batch_startup.as_ref()) {
        if let Err(error) = writer.termination(startup, &factory, &finished) {
            return Err(Box::new(error.retaining(finished)));
        }
    }
    if let (Some(writer), Some(startup)) = (receipts.as_mut(), startup.as_ref()) {
        let receipt = match CudaTerminationReceiptV1::from_result(startup, &finished) {
            Ok(receipt) => receipt,
            Err(error) => return Err(Box::new(error.retaining(finished))),
        };
        if let Err(error) = writer.termination(&receipt) {
            return Err(Box::new(error.retaining(finished)));
        }
    }
    if let (Some(writer), Some(trace)) = (profile_writer.as_mut(), trace.as_ref())
        && let Err(error) = writer.publish(trace, &finished)
    {
        return Err(Box::new(error.retaining(finished)));
    }
    let report = finished?;
    eprintln!("{report}");
    Ok(())
}

fn serve_process(
    process: EngineProcess,
    settings: EngineSettings,
) -> Result<(), engine::EngineError> {
    let (sender, events) = engine::event_channel();
    let input = sender.clone();
    // A blocked OS stdin read is owned by this independent producer. It cannot
    // delay stop/search/drain, and process exit terminates it after quit.
    thread::spawn(move || {
        let _ = forward_lines(
            &mut io::stdin().lock(),
            &input,
            settings.parser.max_line_bytes,
        );
    });
    engine::serve(
        events,
        sender,
        &mut io::stdout().lock(),
        &mut io::stderr().lock(),
        process,
        settings,
    )
}

#[cfg(feature = "onnx-cpu")]
fn serve_native_process(
    process: EngineProcess,
    settings: EngineSettings,
    trace: Option<rz_telemetry::source::SourceJournal<rz_contracts::CompletionContext>>,
) -> Result<(), engine::EngineError> {
    let (sender, events) = engine::event_channel();
    let input = sender.clone();
    thread::spawn(move || {
        let _ = forward_lines(
            &mut io::stdin().lock(),
            &input,
            settings.parser.max_line_bytes,
        );
    });
    engine::serve(
        events,
        sender,
        &mut ProfiledWrite::new(io::stdout().lock(), trace),
        &mut io::stderr().lock(),
        process,
        settings,
    )
}
