//! Explicit bounded Fresh GPU lifetime validation; not a production manager.
use super::*;
use rz_eval::pals_onnx::virtual_review::native_validation_diagnostics::GpuLifecycleTrace;

#[path = "gpu_numerical_prerequisite.rs"]
mod numerical_prerequisite;
use numerical_prerequisite::ContentBoundValidation;

const VALIDATION_DOMAIN: &str = "rz-pals-v3-typed-gpu-inflight-plan/1";

fn cases(
    service: &mut VirtualReviewRoleTicketService,
    config: &Config,
    plan: &Plan,
    trace: &GpuLifecycleTrace,
    clock: &ContractSystemClock,
    metadata: &Value,
    until: Instant,
) -> Result<()> {
    let root = root_position(plan)?;
    let mut stores = PalsStores::new(StoreLimits::default());
    let situation = stores.focus_actual_moves(root.snapshot())?;
    let mut index = ReviewContextIndex::default();
    let authority = SearchAuthority {
        epoch: clock.domain().0,
        game: GameGeneration(1),
        root: RootGeneration(1),
        implementation: Digest(Sha256::digest(include_bytes!("gpu_inflight_validation.rs")).into()),
    };
    let mut registry = checked(Registry::new_with_capacity(
        authority,
        512,
        CoordinatorPolicy::Fixed,
        1,
        1,
    ))?;
    let epoch = parse_digest(
        metadata["model_epoch"]
            .as_str()
            .ok_or("loaded epoch absent")?,
    )?;
    let backend = parse_digest(
        metadata["backend"]
            .as_str()
            .ok_or("loaded backend absent")?,
    )?;
    // Exactly the preregistered twelve P/C/L groups; no retry-until-success loop.
    for group in &plan.groups {
        if Instant::now() >= until {
            return Err("original validation deadline reached".into());
        }
        let q = question(
            &root, &stores, situation, &mut index, group, authority, clock, until,
        )?;
        let view = Arc::clone(&q.view);
        let (input, identity) =
            rz_uci::pals_native::virtual_encoding::prepare_selected_virtual_review_input(
                &q.query.position,
                &q.query.context,
                &view,
                epoch,
                backend,
            )?;
        let expected_input = input.fresh_key(&identity)?.0;
        let id = service
            .prepare(q)
            .map_err(|(e, _)| format!("prepare rejected: {e:?}"))?;
        let seal = match next(service, until)? {
            TicketNativeEvent::Prepared(s) if s.preparation == id => s,
            e => return Err(unexpected(&e).into()),
        };
        if seal.representation.input.0 != expected_input
            || seal.representation.model.0 != identity.model_epoch
            || seal.view.view_digest.0 != view.seal_digest()
        {
            return Err("actual prepared Fresh seal differs".into());
        }
        let ticket = checked(registry.admit(
            RoleSubmission {
                question: seal.question,
                view: seal.view,
                representation: seal.representation,
                role: seal.role,
                original_deadline: seal.original_deadline,
            },
            view,
            clock.domain(),
            clock.now(),
        ))?;
        if registry.take_dispatch().is_none_or(|d| d.ticket != ticket) {
            return Err("original dispatch absent".into());
        }
        checked(service.execute(id, ticket))?;
        let mut canceled = false;
        let (completion, observation_failures) = loop {
            // Observation only: never gate, sleep or suspend the native owner.
            if !canceled && trace.has_open_ort_run(&ticket) {
                checked(service.cancel(id))?;
                canceled = true;
            }
            if let Some(event) = service.poll() {
                match event {
                    TicketNativeEvent::Started(t) if t == ticket => {
                        checked(registry.physical_started(&t))?
                    }
                    TicketNativeEvent::OwnerRunStarted { ticket: t, .. } if t == ticket => {}
                    TicketNativeEvent::Completed {
                        completion,
                        observation_failures,
                        ..
                    } if completion.ticket == ticket => {
                        break (completion, observation_failures);
                    }
                    e => return Err(unexpected(&e).into()),
                }
            }
            if Instant::now() >= until {
                return Err("original ticket deadline; no follow-up permission".into());
            }
            std::thread::yield_now();
        };
        if completion.physical != PhysicalRoleCompletion::Confirmed {
            if let Err(rejected) = registry.complete(completion) {
                std::mem::forget(rejected);
            }
            std::mem::forget(registry);
            return Err("physical completion unknown; original owner retained".into());
        }
        if !observation_failures.is_empty() {
            let error = format!("native observation failure: {:?}", observation_failures);
            std::mem::forget(registry);
            return Err(error.into());
        }
        checked(registry.complete(completion).map_err(|e| e.error))?;
        let ready = registry
            .take_ready(clock.domain(), clock.now())
            .ok_or("original delivery absent")?;
        if ready.ticket != ticket {
            return Err("different original delivery".into());
        }
        // No P/C output is accepted into search or a Warm seed in this profile.
        checked(service.discard(id, ticket))?;
        match next(service, until)? {
            TicketNativeEvent::Retired {
                ticket: t,
                accepted: false,
                result: Ok(()),
                rejected_permit: None,
                observation_failures,
                ..
            } if t == ticket && observation_failures.is_empty() => {}
            e => return Err(unexpected(&e).into()),
        }
        checked(registry.discard_delivered(&ticket))?;
    }
    checked(service.begin_root_quiesce(until))?;
    match next(service, until)? {
        TicketNativeEvent::RootQuiesced(fence)
            if fence.authority() == authority && fence.permits_reuse() => {}
        e => return Err(unexpected(&e).into()),
    }
    let _ = config; // All asset/execution settings are the original loaded config.
    Ok(())
}

fn execute_validation(input: ContentBoundValidation) -> Result<()> {
    let started = Instant::now();
    let config = input.execution.clone();
    if !(1..=600).contains(&config.deadline_seconds)
        || config.cleanup_seconds != 30
        || config.process_epoch == 0
        || !matches!(
            config.provider.as_str(),
            "cuda-control" | "cuda-control-shim"
        )
    {
        return Err(
            "explicit CUDA Fresh validation; finite <=600+30 and nonzero epoch required".into(),
        );
    }
    if config.source_components != rz_experiments::compiled_source_components_v5()? {
        return Err("actual compiled source components differ".into());
    }
    for path in [
        &config.plan,
        &config.export_manifest,
        &config.runtime_library,
        &config.runtime_cache,
        &config.output_root,
    ]
    .into_iter()
    .chain(config.cuda_bundle.as_ref())
    .chain(config.cuda_inventory.as_ref())
    {
        if !path.is_absolute() {
            return Err("original absolute paths required".into());
        }
    }
    for hash in [
        &config.plan_sha256,
        &config.export_manifest_sha256,
        &config.checkpoint_sha256,
        &config.runtime_library_sha256,
    ] {
        parse_digest(hash)?;
    }
    let plan_raw = asset::read_bounded(&config.plan, LIMIT)?;
    if sha(&plan_raw) != config.plan_sha256 {
        return Err("actual plan SHA differs".into());
    }
    let plan: Plan = decode_json(&plan_raw)?;
    validate_plan_domain(&plan, VALIDATION_DOMAIN)?;
    let manifest = asset::read_bounded(&config.export_manifest, 256 * 1024)?;
    let summary = validate_virtual_review_export_manifest_kind(
        &manifest,
        VirtualReviewExecutionMode::Deployment,
        VirtualReviewArtifactKind::Fresh,
    )?;
    if hex(summary.manifest_sha256()) != config.export_manifest_sha256
        || hex(summary.checkpoint_sha256) != config.checkpoint_sha256
    {
        return Err("actual V-free Fresh export/checkpoint differs".into());
    }
    let until = started + Duration::from_secs(config.deadline_seconds);
    let cleanup_until = until + Duration::from_secs(config.cleanup_seconds);
    // Prior independent current-executor math; never this run's future receipt.
    let mut prerequisite = Some(Arc::new(input.prepare(until)?));
    let prerequisite_observation = prerequisite
        .as_ref()
        .ok_or("prerequisite absent")?
        .observation
        .clone();
    let factory_prerequisite = Arc::clone(prerequisite.as_ref().ok_or("prerequisite absent")?);
    let trace = GpuLifecycleTrace::new();
    let mut owner = Output::create(&config.output_root)?;
    let loaded = Arc::new(Mutex::new(None));
    let publish = Arc::clone(&loaded);
    let factory_config = config.clone();
    let factory_trace = trace.clone();
    let profiles = owner.path.join("profiles");
    let mut service = checked(VirtualReviewRoleTicketService::spawn_with_capacity(
        move || {
            let build = || -> Result<NativeVirtualReviewTicketBackend> {
                if Instant::now() >= until {
                    return Err("original deadline before load".into());
                }
                factory_prerequisite.recheck(until)?;
                let model_sources: Arc<
                    dyn rz_eval::pals_onnx::virtual_review::VirtualReviewOwnedSources,
                > = factory_prerequisite.clone();
                let control_metadata = factory_prerequisite.cuda_metadata()?;
                let mut backend = load_kind_from_sources(
                    &factory_config,
                    &profiles,
                    VirtualReviewArtifactKind::Fresh,
                    Some(model_sources),
                    Some(control_metadata),
                )?;
                backend.enable_gpu_lifecycle_validation(factory_trace)?;
                let metadata = json!({"model_epoch":hex(backend.model_epoch()),"backend":hex(backend.backend_digest())});
                let clock = ContractSystemClock::with_origin(
                    ProcessEpoch(factory_config.process_epoch),
                    started,
                );
                let result = checked(NativeVirtualReviewTicketBackend::new(
                    backend,
                    clock,
                    Duration::from_secs(factory_config.cleanup_seconds),
                ))?;
                *publish.lock().map_err(|_| "loaded metadata poisoned")? = Some(metadata);
                Ok(result)
            };
            build().map_err(|e| RoleFailure::Backend(e.to_string()))
        },
        TicketServiceLimits::default(),
        1,
        1,
    ))?;
    let mut startup_failed = false;
    // Install before any preparation; exact same Arc reaches both owner layers.
    let mut work = (|| -> Result<()> {
        checked(service.enable_gpu_lifecycle_validation(trace.clone()))?;
        let clock = match next(&service, until)? {
            TicketNativeEvent::Ready { clock } => clock,
            TicketNativeEvent::StartupFailed(failure) => {
                startup_failed = true;
                return Err(format!("original startup failure: {failure:?}").into());
            }
            e => return Err(unexpected(&e).into()),
        };
        let metadata = loaded
            .lock()
            .map_err(|_| "loaded metadata poisoned")?
            .clone()
            .ok_or("loaded identity absent")?;
        cases(
            &mut service,
            &config,
            &plan,
            &trace,
            &clock,
            &metadata,
            until,
        )
    })();
    // Work primary is preserved; every path after owner construction drains once.
    service.begin_shutdown(cleanup_until);
    let mut joined = false;
    let mut cleanup_error = None;
    let mut finished_observed = false;
    while Instant::now() < cleanup_until {
        while let Some(event) = service.poll() {
            if matches!(event, TicketNativeEvent::Finished { .. }) {
                finished_observed = true;
            }
        }
        // StartupFailed's original path has no Finished event. Join that actual
        // terminated thread without synthesizing a physical drain observation.
        if finished_observed || startup_failed {
            match service.try_join() {
                Ok(true) => {
                    joined = true;
                    break;
                }
                Ok(false) => {}
                Err(e) => {
                    cleanup_error = Some(format!("{e:?}"));
                    break;
                }
            }
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    while service.poll().is_some() {}
    let mut prerequisite_error = None;
    if joined && !service.is_quarantined() {
        if let Err(error) = prerequisite
            .as_ref()
            .ok_or("prerequisite absent")?
            .recheck(cleanup_until)
        {
            prerequisite_error = Some(error.to_string());
            if work.is_ok() {
                work = Err(error);
            } else {
                eprintln!("secondary prerequisite recheck failure: {error}");
            }
        }
    }
    let receipt = trace.receipt();
    let pass = work.is_ok()
        && joined
        && cleanup_error.is_none()
        && !service.is_quarantined()
        && receipt["status"] == "Observed";
    let prerequisite_retained_until_process_exit = !pass;
    if prerequisite_retained_until_process_exit {
        if let Some(held) = prerequisite.take() {
            std::mem::forget(held);
        }
    }
    let report=owner.write("gpu-inflight-validation.json",&json!({"schema":"rz-explicit-gpu-inflight-validation-run/2",
        "source_components":config.source_components,"weights":config.checkpoint_sha256,"export":config.export_manifest_sha256,
        "provider":config.provider,"cudnn_conv_search":config.cudnn_conv_search,"runtime":config.runtime_library_sha256,
        "validation_option":"--gpu-inflight-validation=host-ort-v1","plan_sha256":config.plan_sha256,
        "status":if pass{"Observed"}else{"FailedOrNotObserved"},"work_primary":work.as_ref().err().map(|e|e.to_string()),
        "cleanup_error":cleanup_error,"joined":joined,"quarantined":service.is_quarantined(),"trace":receipt,
        "whole_product_UCI_acceptance":false,"numerical_acceptance":false,"performance_evidence":false,
        "prerequisite_content_admission":prerequisite_observation,"prerequisite_recheck_error":prerequisite_error,
        "prerequisite_retained_until_original_process_exit":prerequisite_retained_until_process_exit,
        "process_exit_evidence":"requires_original_external_owner_exit_EOF_and_cgroup_observation"}));
    if let Err(error) = work {
        if let Err(secondary) = report {
            eprintln!("secondary GPU validation receipt failure: {secondary}");
        }
        if let Some(secondary) = cleanup_error {
            eprintln!("secondary GPU validation cleanup failure: {secondary}");
        }
        return Err(error);
    }
    if let Err(error) = report {
        if let Some(held) = prerequisite.take() {
            std::mem::forget(held);
        }
        return Err(error);
    }
    if !pass {
        return Err(
            "required actual host-CUDA-Run cancel/late/fence/drain case not observed or incomplete"
                .into(),
        );
    }
    Ok(())
}

pub fn main() {
    let args: Vec<_> = std::env::args().collect();
    let result = (|| -> Result<()> {
        if args.len() != 3 || args[1] != "--gpu-inflight-validation=host-ort-v1" {
            return Err("explicit --gpu-inflight-validation=host-ort-v1 CONFIG required; defaults remain off".into());
        }
        let bytes = asset::read_bounded(Path::new(&args[2]), LIMIT)?;
        execute_validation(decode_json(&bytes)?)
    })();
    if let Err(error) = result {
        eprintln!("GPU validation failed: {error}");
        std::process::exit(2);
    }
}
