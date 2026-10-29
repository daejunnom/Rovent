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
        original,
    }))
}

#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
fn run_lock_v2(cli: LockV2Cli) -> Result<(serde_json::Value, bool), Box<dyn std::error::Error>> {
    use rz_arena::rovent_collect::launch::{
        load_rovent_collection_lock_v2_plan, run_rovent_own_nnuec_collection_lock_v2,
        run_rovent_pc_collection_lock_v2,
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
    let plan = load_rovent_collection_lock_v2_plan(&directory, &cli.plan)?;
    if plan.output_directory != cli.output {
        return Err("V2 output directory differs from pinned plan".into());
    }
    if plan.maximum_roots != 1 {
        return Err("V2 runner supports exactly one root".into());
    }
    let publication_timeout_ms = plan.publication_timeout_ms;
    let (observation, accepted, publication, publication_error) = match cli.mode.as_str() {
        "pc" => {
            let mut run = run_rovent_pc_collection_lock_v2(
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
            let until = Instant::now().checked_add(Duration::from_millis(publication_timeout_ms))
                .ok_or("publication deadline overflow")?.min(cleanup_until);
            let result = run.publish_single_root(until);
            let (publication, publication_error) = match result {
                Ok(true) => ("published", None),
                Ok(false) => ("partial", None),
                Err(error) => ("partial", Some(error.to_string())),
            };
            let observation = run.observation();
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
            let publication = observation.get("publisher_status").and_then(serde_json::Value::as_str)
                .map(|value| match value {
                    "published" if observation.get("corpus_published").and_then(serde_json::Value::as_bool) == Some(true) => "published",
                    "published" | "partial" => "partial", _ => "unknown_or_not_started" })
                .unwrap_or("unknown_or_not_started");
            let error = observation.get("publisher_error").and_then(serde_json::Value::as_str).map(str::to_owned);
            (observation, accepted, publication, error)
        }
        _ => return Err("unsupported V2 collection mode".into()),
    };
    Ok((
        serde_json::json!({"schema":"rz-rovent-collection-cli-v2-observation/1", "mode":cli.mode,
        "input_directory":cli.input, "output_directory":cli.output, "worker_timeout_ms":cli.worker_ms,
        "cleanup_timeout_ms":cli.cleanup_ms, "runner_observation":observation,
        "owner_inspection_available":accepted, "publication":publication, "publication_error":publication_error, "training_executed":false,
        "owner_custody":"process_lifetime_only", "durable_recovery":"unavailable",
        "native_execution_deadline":"external_supervisor_required", "cli_acceptance":"incomplete"}),
        false,
    ))
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
                Ok((observation, accepted)) => {
                    if writeln!(std::io::stderr().lock(), "{observation}").is_err() || !accepted {
                        std::process::ExitCode::FAILURE
                    } else {
                        std::process::ExitCode::SUCCESS
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
                    "V2: --v2-mode=pc|own-nnuec --v2-plan=inputs/<plan.json> --v2-plan-sha256=<sha256> --v2-plan-bytes=<bytes> --v2-input-dir=<absolute> --v2-output-dir=<absolute> --v2-worker-timeout-ms=<ms> --v2-cleanup-timeout-ms=<ms> <original source/config/FD arguments>"
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
    use super::parse_lock_v2;

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
