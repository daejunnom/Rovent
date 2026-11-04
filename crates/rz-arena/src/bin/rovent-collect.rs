//! 원 source/native/teacher owner를 같은 Collection caller로 연결하는 진입점.
//! 실제 ELF pin과 프로세스/자원 경계는 공유 UCI launch가 별도로 검사한다.
#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
struct LockV2Cli {
    mode: String,
    plan: rz_arena::rovent_collect::RoventPublicationFile,
    input: std::path::PathBuf,
    output: std::path::PathBuf,
    worker_ms: u64,
    cleanup_ms: u64,
    sealed_teacher_slots: Vec<usize>,
    speculative_teacher: bool,
    partial_spec: Option<rz_arena::rovent_collect::RoventPublicationFile>,
    partial_game_limits: Option<(u64, u64)>,
    original: Vec<String>,
}

#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
fn parse_lock_v2(arguments: Vec<String>) -> Result<Option<LockV2Cli>, Box<dyn std::error::Error>> {
    if !arguments.iter().any(|arg| arg.starts_with("--v2-")) {
        return Ok(None);
    }
    let mut fields = std::collections::BTreeMap::new();
    let mut original = Vec::new();
    for arg in arguments {
        if arg.starts_with("--v2-") {
            let (key, value) = arg.split_once('=').ok_or("V2 options require =value")?;
            if !matches!(
                key,
                "--v2-mode"
                    | "--v2-plan"
                    | "--v2-plan-sha256"
                    | "--v2-plan-bytes"
                    | "--v2-input-dir"
                    | "--v2-output-dir"
                    | "--v2-worker-timeout-ms"
                    | "--v2-cleanup-timeout-ms"
                    | "--v2-sealed-teacher-slots"
                    | "--v2-speculative-teacher"
                    | "--v2-partial-spec"
                    | "--v2-partial-spec-sha256"
                    | "--v2-partial-spec-bytes"
                    | "--v2-partial-game-bytes"
                    | "--v2-partial-game-index-bytes"
            ) || value.is_empty()
                || fields.insert(key.to_owned(), value.to_owned()).is_some()
            {
                return Err("unknown, empty or duplicate V2 option".into());
            }
        } else if arg.starts_with("--collection-plan")
            || arg.starts_with("--utility-cohort-plan")
            || arg.starts_with("--utility-intervention-plan")
            || arg.starts_with("--prepare-utility-cohort-pins")
        {
            return Err("legacy and V2 collection selectors cannot be mixed".into());
        } else {
            original.push(arg);
        }
    }
    let mut take = |key: &str| fields.remove(key).ok_or_else(|| format!("missing {key}"));
    let mode = take("--v2-mode")?;
    if !matches!(mode.as_str(), "pc" | "own-nnuec") {
        return Err("unsupported V2 collection mode".into());
    }
    let plan = rz_arena::rovent_collect::RoventPublicationFile {
        path: take("--v2-plan")?,
        sha256: take("--v2-plan-sha256")?,
        bytes: take("--v2-plan-bytes")?.parse()?,
    };
    let input = std::path::PathBuf::from(take("--v2-input-dir")?);
    let output = std::path::PathBuf::from(take("--v2-output-dir")?);
    let worker_ms = take("--v2-worker-timeout-ms")?.parse::<u64>()?;
    let cleanup_ms = take("--v2-cleanup-timeout-ms")?.parse::<u64>()?;
    let sealed_teacher_slots = match fields.remove("--v2-sealed-teacher-slots") {
        None => Vec::new(),
        Some(value) => {
            let mut slots = value
                .split(',')
                .map(str::parse::<usize>)
                .collect::<Result<Vec<_>, _>>()?;
            if slots.is_empty() || slots.len() > 16 {
                return Err("sealed teacher slot count exceeds limit".into());
            }
            slots.sort_unstable();
            slots.dedup();
            slots
        }
    };
    let speculative_teacher = match fields.remove("--v2-speculative-teacher").as_deref() {
        None | Some("false") => false,
        Some("true") => true,
        _ => return Err("speculative teacher must be true or false".into()),
    };
    let partial_fields = (
        fields.remove("--v2-partial-spec"),
        fields.remove("--v2-partial-spec-sha256"),
        fields.remove("--v2-partial-spec-bytes"),
    );
    let partial_spec = match partial_fields {
        (None, None, None) => None,
        (Some(path), Some(sha256), Some(bytes)) if mode == "pc" => {
            Some(rz_arena::rovent_collect::RoventPublicationFile {
                path,
                sha256,
                bytes: bytes.parse()?,
            })
        }
        _ => {
            return Err(
                "partial collection requires all three pinned spec fields and pc mode".into(),
            );
        }
    };
    let partial_game_limits = match (
        fields.remove("--v2-partial-game-bytes"),
        fields.remove("--v2-partial-game-index-bytes"),
    ) {
        (None, None) => None,
        (Some(total), Some(index)) if partial_spec.is_some() => {
            let total = total.parse::<u64>()?;
            let index = index.parse::<u64>()?;
            if total == 0 || index == 0 {
                return Err("game partial requires positive byte limits".into());
            }
            Some((total, index))
        }
        _ => return Err("game partial requires both limits and a pinned partial spec".into()),
    };
    if !input.is_absolute() || !output.is_absolute() || worker_ms == 0 || cleanup_ms < worker_ms {
        return Err(
            "absolute input/output and finite ordered worker/cleanup timeouts required".into(),
        );
    }
    Ok(Some(LockV2Cli {
        mode,
        plan,
        input,
        output,
        worker_ms,
        cleanup_ms,
        sealed_teacher_slots,
        speculative_teacher,
        partial_spec,
        partial_game_limits,
        original,
    }))
}

#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum V2CliOutcome {
    StrictComplete,
    PartialReceiptWritten,
    PartialGameIndexWritten,
    Incomplete,
}

#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
fn run_lock_v2(
    cli: LockV2Cli,
) -> Result<(serde_json::Value, V2CliOutcome), Box<dyn std::error::Error>> {
    use rz_arena::rovent_collect::launch::{
        load_rovent_collection_lock_v2_plan, load_rovent_partial_collection_spec,
        load_verified_partial_game_inputs, run_rovent_own_nnuec_collection_lock_v2,
        run_rovent_pc_collection_lock_v2_with_speculative_teacher,
        run_rovent_pc_collection_lock_v2_with_verified_partial_game,
    };
    use std::time::{Duration, Instant};
    let start = Instant::now();
    let worker_until = start
        .checked_add(Duration::from_millis(cli.worker_ms))
        .ok_or("worker deadline overflow")?;
    let cleanup_until = start
        .checked_add(Duration::from_millis(cli.cleanup_ms))
        .ok_or("cleanup deadline overflow")?;
    let directory = cap_std::fs::Dir::open_ambient_dir(&cli.input, cap_std::ambient_authority())?;
    let mut verified_game =
        if let Some((max_total_bytes, index_max_bytes)) = cli.partial_game_limits {
            Some(load_verified_partial_game_inputs(
                &directory,
                &cli.plan,
                cli.partial_spec
                    .as_ref()
                    .ok_or("partial game spec missing")?,
                max_total_bytes,
                index_max_bytes,
            )?)
        } else {
            None
        };
    let (plan, partial_spec) = if let Some(inputs) = verified_game.as_ref() {
        (inputs.plan().clone(), Some(inputs.spec().clone()))
    } else {
        (
            load_rovent_collection_lock_v2_plan(&directory, &cli.plan)?,
            cli.partial_spec
                .as_ref()
                .map(|pin| load_rovent_partial_collection_spec(&directory, pin, &cli.plan.sha256))
                .transpose()?,
        )
    };
    if plan.output_directory != cli.output {
        return Err("V2 output directory differs from pinned plan".into());
    }
    if !valid_v2_root_capacity(&cli.mode, plan.maximum_roots) {
        return Err("V2 collection root capacity must be between 1 and 64".into());
    }
    let root_capacity = plan.maximum_roots;
    let publication_timeout_ms = plan.publication_timeout_ms;
    let speculative_requested = cli.speculative_teacher;
    let speculative_effective = speculative_requested && cli.mode == "pc";
    let sealed_requested = cli.sealed_teacher_slots.clone();
    let sealed_effective: Vec<_> = sealed_requested
        .iter()
        .copied()
        .filter(|&index| cli.mode == "pc" && index < plan.teacher_sources.len())
        .collect();
    let sealed_fallback: Vec<_> = sealed_requested
        .iter()
        .copied()
        .filter(|index| !sealed_effective.contains(index))
        .collect();
    let (observation, accepted, publication, publication_error) = match cli.mode.as_str() {
        "pc" => {
            let mut run = match verified_game.take() {
                Some(inputs) => run_rovent_pc_collection_lock_v2_with_verified_partial_game(
                    cli.original,
                    inputs,
                    &directory,
                    worker_until,
                    cleanup_until,
                    sealed_effective.clone(),
                    speculative_effective,
                )?,
                None => run_rovent_pc_collection_lock_v2_with_speculative_teacher(
                    cli.original,
                    plan,
                    &directory,
                    worker_until,
                    cleanup_until,
                    sealed_effective.clone(),
                    speculative_effective,
                )?,
            };
            while !run.poll_worker() && Instant::now() < cleanup_until {
                std::thread::sleep(
                    cleanup_until
                        .saturating_duration_since(Instant::now())
                        .min(Duration::from_millis(10)),
                );
            }
            let until = Instant::now()
                .checked_add(Duration::from_millis(publication_timeout_ms))
                .ok_or("publication deadline overflow")?
                .min(cleanup_until);
            let (publication, publication_error) = match partial_spec.as_ref() {
                Some(spec) => {
                    let result = match cli.partial_game_limits {
                        Some(_) => run.collect_partial_settled_game(until),
                        None => run.collect_partial_single_root(spec, &cli.plan.sha256, until),
                    };
                    match result {
                        Ok(()) => ("partial", None),
                        Err(error) => ("partial", Some(error.to_string())),
                    }
                }
                None => match run.publish_single_root(until) {
                    Ok(true) => ("published", None),
                    Ok(false) => ("partial", None),
                    Err(error) => ("partial", Some(error.to_string())),
                },
            };
            let failure_evidence = if publication != "published" {
                match run.write_failure_evidence(publication_error.as_deref()) {
                    Ok(complete) => {
                        serde_json::json!({"status":if complete { "written" } else { "partial" }, "path":"pc-failure-evidence/receipt.json"})
                    }
                    Err(error) => {
                        serde_json::json!({"status":"unavailable", "error":error.to_string()})
                    }
                }
            } else {
                serde_json::json!({"status":"not_needed"})
            };
            let mut observation = run.observation();
            if let Some(object) = observation.as_object_mut() {
                object.insert("failure_evidence".into(), failure_evidence);
            }
            let accepted = run.closed_owner().is_ok()
                && observation
                    .get("native_error")
                    .is_some_and(serde_json::Value::is_null)
                && observation
                    .get("resource_error")
                    .is_some_and(serde_json::Value::is_null);
            (observation, accepted, publication, publication_error)
        }
        "own-nnuec" => {
            let mut run = run_rovent_own_nnuec_collection_lock_v2(
                cli.original,
                plan,
                &directory,
                worker_until,
                cleanup_until,
            )?;
            while !run.poll_worker() && Instant::now() < cleanup_until {
                std::thread::sleep(
                    cleanup_until
                        .saturating_duration_since(Instant::now())
                        .min(Duration::from_millis(10)),
                );
            }
            let observation = run.observation();
            let accepted = run.closed_owner().is_ok()
                && observation
                    .get("native_error")
                    .is_some_and(serde_json::Value::is_null)
                && observation
                    .get("resource_error")
                    .is_some_and(serde_json::Value::is_null);
            let publication = observation
                .get("publisher_status")
                .and_then(serde_json::Value::as_str)
                .map(|value| match value {
                    "published"
                        if observation
                            .get("corpus_published")
                            .and_then(serde_json::Value::as_bool)
                            == Some(true) =>
                    {
                        "published"
                    }
                    "published" | "partial" => "partial",
                    _ => "unknown_or_not_started",
                })
                .unwrap_or("unknown_or_not_started");
            let error = observation
                .get("publisher_error")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned);
            (observation, accepted, publication, error)
        }
        _ => return Err("unsupported V2 collection mode".into()),
    };
    // `run_*` succeeds only after the native completion boundary; `closed_owner`
    // checks the original joined worker and handoff, and publication is the
    // actual durable corpus result. Process-tree completion and enforcement
    // of the native wall deadline remain the external supervisor's evidence.
    let in_process_complete = accepted && publication == "published" && publication_error.is_none();
    let partial_receipt_written = partial_spec.is_some() && publication_error.is_none();
    let mut result = serde_json::json!({"schema":"rz-rovent-collection-cli-v2-observation/1", "mode":cli.mode,
        "root_capacity":root_capacity,
        "input_directory":cli.input, "output_directory":cli.output, "worker_timeout_ms":cli.worker_ms,
        "cleanup_timeout_ms":cli.cleanup_ms, "runner_observation":observation,
        "owner_inspection_available":accepted, "publication":publication, "publication_error":publication_error, "training_executed":false,
        "owner_custody":"process_lifetime_only", "durable_recovery":"unavailable",
        "native_execution_deadline":"external_supervisor_required",
        "cli_acceptance":if in_process_complete { "complete_in_process" } else { "incomplete" }});
    if let Some(pin) = cli.partial_spec.as_ref() {
        result
            .as_object_mut()
            .expect("CLI observation is an object")
            .insert(
                "partial_collection_selection".into(),
            serde_json::json!({"policy":if cli.partial_game_limits.is_some() { "partial_game_v2" } else { "partial_policy_unavailable_v1" }, "spec":pin,
                "receipt_status":if partial_receipt_written { "written" } else { "unavailable" },
                "receipt_path":if partial_receipt_written {
                    Some(if cli.partial_game_limits.is_some() { "partial-game-v2.json" } else { "partial_collection/partial_collection.json" })
                } else { None },
                "training_eligible":false,"physical_completion":"not_attested"}),
            );
    }
    if speculative_requested {
        result.as_object_mut().expect("CLI observation is an object").insert(
            "speculative_teacher_selection".into(),
            serde_json::json!({"requested":true,"selected_for_pc":speculative_effective,
                "status":if speculative_effective { "selected_for_pc" } else { "unsupported_for_own" },
                "scope":"optional_collection_only"}),
        );
    }
    if !sealed_requested.is_empty() {
        result.as_object_mut().expect("CLI observation is an object").insert(
            "sealed_teacher_selection".into(),
            serde_json::json!({"requested":sealed_requested,"effective":sealed_effective,
                "unsupported_or_invalid_fallback":sealed_fallback,"scope":"optional_preparation_only"}),
        );
    }
    let outcome = if in_process_complete {
        V2CliOutcome::StrictComplete
    } else if partial_receipt_written && cli.partial_game_limits.is_some() {
        V2CliOutcome::PartialGameIndexWritten
    } else if partial_receipt_written {
        V2CliOutcome::PartialReceiptWritten
    } else {
        V2CliOutcome::Incomplete
    };
    Ok((result, outcome))
}

/// This is a capacity, not a claim that that many roots were admitted,
/// processed or published. The original worker and owner report those states.
#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
fn valid_v2_root_capacity(mode: &str, maximum_roots: u16) -> bool {
    matches!(mode, "pc" | "own-nnuec") && (1..=64).contains(&maximum_roots)
}

#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
fn main() -> std::process::ExitCode {
    use rz_arena::rovent_collect::launch::{
        ROVENT_COLLECTION_USAGE, RoventCollectionCliParse, parse_rovent_collection_cli,
        prepare_rovent_utility_cohort_pins_from_cli, run_rovent_collection_from_cli,
        run_rovent_utility_cohort_from_cli, run_rovent_utility_intervention_from_cli,
    };
    use std::io::Write;
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match parse_lock_v2(arguments.clone()) {
        Ok(Some(cli)) => {
            let _ = writeln!(
                std::io::stderr().lock(),
                "V2 worker/cleanup timeouts do not limit the native UCI loop; an external finite process supervisor is required"
            );
            return match run_lock_v2(cli) {
                Ok((observation, outcome)) => {
                    if writeln!(std::io::stderr().lock(), "{observation}").is_err() {
                        std::process::ExitCode::FAILURE
                    } else {
                        match outcome {
                            V2CliOutcome::StrictComplete => std::process::ExitCode::SUCCESS,
                            V2CliOutcome::PartialReceiptWritten => std::process::ExitCode::from(
                                rz_arena::rovent_collect::launch::partial_followup_v1::PARTIAL_COLLECTION_EXIT_CODE as u8,
                            ),
                            V2CliOutcome::PartialGameIndexWritten => std::process::ExitCode::from(
                                rz_arena::rovent_collect::launch::partial_game_v2::GAME_PARTIAL_EXIT_CODE,
                            ),
                            V2CliOutcome::Incomplete => std::process::ExitCode::FAILURE,
                        }
                    }
                }
                Err(error) => {
                    let detail: String = error
                        .to_string()
                        .chars()
                        .take(2048)
                        .map(|c| if c.is_control() { ' ' } else { c })
                        .collect();
                    let _ = writeln!(
                        std::io::stderr().lock(),
                        "rovent-collect V2 halted: {detail}"
                    );
                    std::process::ExitCode::FAILURE
                }
            };
        }
        Err(error) => {
            let _ = writeln!(
                std::io::stderr().lock(),
                "rovent-collect V2 configuration: {error}"
            );
            return std::process::ExitCode::FAILURE;
        }
        Ok(None) => {}
    }
    let run = || -> Result<Option<serde_json::Value>, Box<dyn std::error::Error>> {
        match parse_rovent_collection_cli(arguments)? {
            RoventCollectionCliParse::Help => {
                writeln!(std::io::stdout().lock(), "{ROVENT_COLLECTION_USAGE}")?;
                writeln!(
                    std::io::stdout().lock(),
                    "V2: --v2-mode=pc|own-nnuec --v2-plan=inputs/<plan.json> --v2-plan-sha256=<sha256> --v2-plan-bytes=<bytes> --v2-input-dir=<absolute> --v2-output-dir=<absolute> --v2-worker-timeout-ms=<ms> --v2-cleanup-timeout-ms=<ms> [--v2-partial-spec=<path> --v2-partial-spec-sha256=<sha256> --v2-partial-spec-bytes=<bytes>] <original source/config/FD arguments>"
                )?;
                writeln!(
                    std::io::stdout().lock(),
                    "선택형 cohort: --utility-cohort-plan=inputs/<plan.json> --utility-cohort-plan-sha256=<SHA> --utility-cohort-plan-bytes=<bytes> <원 source/config 인수>"
                )?;
                writeln!(
                    std::io::stdout().lock(),
                    "읽기 전용 예상 pin 준비: --prepare-utility-cohort-pins=inputs/<plan.json> --prepare-utility-cohort-pins-sha256=<SHA> --prepare-utility-cohort-pins-bytes=<bytes>"
                )?;
                writeln!(
                    std::io::stdout().lock(),
                    "독립 root 개입 비교 /2: --utility-intervention-plan=inputs/<plan.json> --utility-intervention-plan-sha256=<SHA> --utility-intervention-plan-bytes=<bytes> <원 source/config 인수>"
                )?;
                Ok(None)
            }
            RoventCollectionCliParse::PrepareCohortPins { plan } => {
                Ok(Some(prepare_rovent_utility_cohort_pins_from_cli(&plan)?))
            }
            RoventCollectionCliParse::Cohort { plan, arguments } => {
                let receipt = run_rovent_utility_cohort_from_cli(&plan, arguments)?;
                let resources = receipt.resource_observations()?;
                let collection = receipt.collection_observations()?;
                Ok(Some(
                    serde_json::json!({"schema":"rz-rovent-utility-cohort-launch-result/1",
                    "publication":receipt.publication,"durable_ack":receipt.durable_ack,
                    "root":receipt.output_directory,"trials":receipt.trials,"pairs":receipt.pairs,
                    "resource":resources,"collection":collection,"training_executed":false}),
                ))
            }
            RoventCollectionCliParse::Intervention { plan, arguments } => {
                let receipt = run_rovent_utility_intervention_from_cli(&plan, arguments)?;
                if let Some(warning) = &receipt.diagnostic_warning {
                    // Ancillary diagnostics are not cohort publication or ACK.
                    // Preserve the warning without changing the /1 stdout schema.
                    let _ = writeln!(
                        std::io::stderr().lock(),
                        "intervention diagnostic warning: {warning}"
                    );
                }
                let resources = receipt.resource_observations()?;
                let collection = receipt.collection_observations()?;
                Ok(Some(serde_json::json!({
                    "schema":"rz-rovent-utility-intervention-launch-result/1",
                    "publication":receipt.publication,"durable_ack":receipt.durable_ack,
                    "root":receipt.output_directory,"trials":receipt.trials,"pairs":receipt.pairs,
                    "resource":resources,"collection":collection,"training_executed":false
                })))
            }
            RoventCollectionCliParse::Run { plan, arguments } => {
                let receipt = run_rovent_collection_from_cli(&plan, arguments)?;
                let resource = receipt.resource_observation()?;
                let r_data = receipt
                    .r_datasets
                    .iter()
                    .map(|dataset| dataset.observation())
                    .collect::<Vec<_>>();
                Ok(Some(
                    serde_json::json!({"schema":"rz-rovent-collection-launch-result/1",
                    "publication":receipt.publication_file,"root":receipt.output_directory,
                    "roots":receipt.roots,"records":receipt.records,
                    "teacher_failure_count":receipt.teacher_failure_count,
                    "historical_data":receipt.historical_dataset.observation(),
                    "resource":resource,"r_data":r_data,"r_sidecars":receipt.r_sidecars,
                    "utility_root_local_files":receipt.utility_root_local_files,
                    "training_executed":false}),
                ))
            }
        }
    };
    match run() {
        Ok(result) => {
            if let Some(result) = result
                && writeln!(std::io::stderr().lock(), "{result}").is_err()
            {
                return std::process::ExitCode::FAILURE;
            }
            std::process::ExitCode::SUCCESS
        }
        Err(error) => {
            let detail: String = error
                .to_string()
                .chars()
                .take(2048)
                .map(|value| if value.is_control() { ' ' } else { value })
                .collect();
            let _ = writeln!(std::io::stderr().lock(), "rovent-collect halted: {detail}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(all(test, target_os = "linux", feature = "pals-collection-onnx"))]
mod v2_cli_tests {
    use super::{parse_lock_v2, valid_v2_root_capacity};

    #[test]
    fn v2_root_capacity_matches_both_production_supervisors() {
        for mode in ["pc", "own-nnuec"] {
            for count in [1, 2, 64] {
                assert!(valid_v2_root_capacity(mode, count));
            }
            for count in [0, 65] {
                assert!(!valid_v2_root_capacity(mode, count));
            }
        }
        assert!(!valid_v2_root_capacity("unknown", 1));
    }

    fn arguments() -> Vec<String> {
        [
            "--v2-mode=pc",
            "--v2-plan=inputs/plan.json",
            "--v2-plan-sha256=unused-at-parse",
            "--v2-plan-bytes=42",
            "--v2-input-dir=/tmp/input",
            "--v2-output-dir=/tmp/output",
            "--v2-worker-timeout-ms=100",
            "--v2-cleanup-timeout-ms=200",
            "--process-epoch=1",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }

    #[test]
    fn v2_selector_preserves_native_arguments_and_rejects_mixed_or_invalid_modes() {
        let result = parse_lock_v2(arguments()).unwrap().unwrap();
        assert_eq!(result.mode, "pc");
        assert_eq!(result.original, ["--process-epoch=1"]);
        assert!(result.sealed_teacher_slots.is_empty());
        assert!(!result.speculative_teacher);
        assert!(result.partial_spec.is_none());
        let mut mixed = arguments();
        mixed.push("--collection-plan=inputs/old.json".into());
        assert!(parse_lock_v2(mixed).is_err());
        let mut unknown = arguments();
        unknown[0] = "--v2-mode=own-board".into();
        assert!(parse_lock_v2(unknown).is_err());
        let mut duplicate = arguments();
        duplicate.push("--v2-mode=own-nnuec".into());
        assert!(parse_lock_v2(duplicate).is_err());
        assert!(
            parse_lock_v2(vec!["--collection-plan=inputs/old.json".into()])
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn partial_policy_requires_an_explicit_complete_pin_and_pc_mode() {
        let mut selected = arguments();
        selected.extend([
            "--v2-partial-spec=inputs/partial.json".to_owned(),
            "--v2-partial-spec-sha256=unused-at-parse".to_owned(),
            "--v2-partial-spec-bytes=123".to_owned(),
        ]);
        assert!(
            parse_lock_v2(selected.clone())
                .unwrap()
                .unwrap()
                .partial_spec
                .is_some()
        );
        selected[0] = "--v2-mode=own-nnuec".into();
        assert!(parse_lock_v2(selected).is_err());
        let mut missing = arguments();
        missing.push("--v2-partial-spec=inputs/partial.json".into());
        assert!(parse_lock_v2(missing).is_err());
    }

    #[test]
    fn game_partial_requires_explicit_partial_spec_and_both_finite_limits() {
        let mut selected = arguments();
        selected.extend([
            "--v2-partial-spec=inputs/partial.json".to_owned(),
            "--v2-partial-spec-sha256=unused-at-parse".to_owned(),
            "--v2-partial-spec-bytes=123".to_owned(),
            "--v2-partial-game-bytes=4096".to_owned(),
            "--v2-partial-game-index-bytes=1024".to_owned(),
        ]);
        assert_eq!(
            parse_lock_v2(selected.clone())
                .unwrap()
                .unwrap()
                .partial_game_limits,
            Some((4096, 1024))
        );
        selected.pop();
        assert!(parse_lock_v2(selected).is_err());
        let mut no_spec = arguments();
        no_spec.extend([
            "--v2-partial-game-bytes=4096".into(),
            "--v2-partial-game-index-bytes=1024".into(),
        ]);
        assert!(parse_lock_v2(no_spec).is_err());
    }

    #[test]
    fn sealed_teacher_selection_is_explicit_and_keeps_native_arguments() {
        let mut input = arguments();
        input.push("--v2-sealed-teacher-slots=2,0,2".into());
        let result = parse_lock_v2(input).unwrap().unwrap();
        assert_eq!(result.sealed_teacher_slots, [0, 2]);
        assert_eq!(result.original, ["--process-epoch=1"]);
        let mut invalid = arguments();
        invalid.push("--v2-sealed-teacher-slots=0,no".into());
        assert!(parse_lock_v2(invalid).is_err());
    }

    #[test]
    fn speculative_teacher_is_explicit_and_keeps_native_arguments() {
        let mut selected = arguments();
        selected.push("--v2-speculative-teacher=true".into());
        let selected = parse_lock_v2(selected).unwrap().unwrap();
        assert!(selected.speculative_teacher);
        assert_eq!(selected.original, ["--process-epoch=1"]);
        let mut disabled = arguments();
        disabled.push("--v2-speculative-teacher=false".into());
        assert!(
            !parse_lock_v2(disabled)
                .unwrap()
                .unwrap()
                .speculative_teacher
        );
        let mut invalid = arguments();
        invalid.push("--v2-speculative-teacher=yes".into());
        assert!(parse_lock_v2(invalid).is_err());
    }

    #[test]
    fn v2_requires_absolute_directories_and_ordered_nonzero_deadlines() {
        let mut relative = arguments();
        relative[4] = "--v2-input-dir=relative".into();
        assert!(parse_lock_v2(relative).is_err());
        let mut deadlines = arguments();
        deadlines[7] = "--v2-cleanup-timeout-ms=99".into();
        assert!(parse_lock_v2(deadlines).is_err());
    }
}
#[cfg(not(all(target_os = "linux", feature = "pals-collection-onnx")))]
fn main() -> std::process::ExitCode {
    use std::io::Write;
    let _ = writeln!(
        std::io::stderr().lock(),
        "rovent-collect requires Linux and pals-collection-onnx"
    );
    std::process::ExitCode::FAILURE
}
