use rz_arena::{
    ArenaPlan, Event, Ledger, LedgerLimits, PlanLimits, decode_json, prepare_native_launch,
    run_fixture_pair, run_native_pair,
};
#[cfg(feature = "native-cuda")]
use rz_arena::{prepare_native_cuda_launch, run_native_cuda_pair};
#[cfg(feature = "native-cuda")]
use rz_experiments::{
    CudaIntegrationPairSpecV1, CudaIntegrationPairSpecV2, CudaSearchPilotPairSpecV3,
    LockedCudaIntegrationPairSpecV1, LockedCudaIntegrationPairSpecV2,
    LockedCudaSearchPilotPairSpecV3, MAX_CUDA_NATIVE_LAUNCH_JSON_BYTES,
};
use rz_experiments::{
    IntegrationPairSpecV1, LockedIntegrationPairSpecV1, LockedManifest, MAX_MANIFEST_BYTES,
    MAX_NATIVE_LAUNCH_JSON_BYTES,
};
use std::env;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const JSON_LIMIT: u64 = MAX_MANIFEST_BYTES as u64;
const USAGE: &str = "Usage:\n  rz-arena plan LOCKED OUTPUT --max-pairs N --max-plan-bytes N\n  rz-arena ledger-init PLAN OUTPUT --max-pairs N --max-plan-bytes N --max-events N --max-ledger-bytes N\n  rz-arena ledger-append PLAN LEDGER EVENT OUTPUT --max-pairs N --max-plan-bytes N --max-events N --max-ledger-bytes N\n  rz-arena audit PLAN LEDGER --max-pairs N --max-plan-bytes N --max-events N --max-ledger-bytes N [--expected-tip SHA]\n  rz-arena fixture-pair PLAN ARTIFACT_ROOT NEW_OUTPUT_BASENAME --max-pairs N --max-plan-bytes N\n  rz-arena native-lock INPUT OUTPUT\n  rz-arena native-pair LOCKED ARTIFACT_ROOT OUTPUT_ROOT NEW_OUTPUT_BASENAME\n\nAll bounds are required and positive. fixture-pair is a synthetic smoke only; execution_ready=false. native-lock checks declarations for one CPU NN integration pair, with required locked budgets and strength_eligible=false. native-pair requires Linux and exclusively owned outside-Git roots; it audits actual CPU NN evidence and process cleanup within the locked budgets.";

#[derive(Clone, Copy)]
enum Operation {
    Plan,
    LedgerInit,
    LedgerAppend,
    Audit,
    FixturePair,
}

struct Bounds {
    max_pairs: u64,
    max_plan_bytes: usize,
    max_events: Option<u64>,
    max_ledger_bytes: Option<u64>,
}

impl Bounds {
    fn plan_limits(&self) -> PlanLimits {
        PlanLimits {
            max_pairs: self.max_pairs,
            max_json_bytes: self.max_plan_bytes,
        }
    }

    fn ledger_limits(&self, plan: &ArenaPlan) -> Result<LedgerLimits, String> {
        let max_events = self
            .max_events
            .ok_or_else(|| "--max-events is required".to_string())?;
        let max_bytes = self
            .max_ledger_bytes
            .ok_or_else(|| "--max-ledger-bytes is required".to_string())?;
        if max_bytes > plan.manifest().input().budget.max_output_bytes {
            return Err("--max-ledger-bytes exceeds the locked manifest output budget".to_string());
        }
        Ok(LedgerLimits {
            max_events,
            max_bytes,
        })
    }
}

enum Command {
    Help,
    NativeLock {
        input: PathBuf,
        output: PathBuf,
    },
    NativePair {
        locked: PathBuf,
        artifact_root: PathBuf,
        output_root: PathBuf,
        output_directory: OsString,
    },
    #[cfg(feature = "native-cuda")]
    NativeCudaLock {
        input: PathBuf,
        output: PathBuf,
        version: u32,
    },
    #[cfg(feature = "native-cuda")]
    NativeCudaPair {
        version: u32,
        locked: PathBuf,
        artifact_root: PathBuf,
        output_root: PathBuf,
        output_directory: OsString,
    },
    Execute {
        operation: Operation,
        paths: Vec<PathBuf>,
        bounds: Bounds,
        expected_tip: Option<String>,
    },
}

fn main() -> ExitCode {
    rz_arena::emit_native_phase("cli_started");
    match parse_command(env::args_os().skip(1).collect()).and_then(execute) {
        Ok(output) => {
            println!("{output}");
            rz_arena::emit_native_phase("cli_result_emitted");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("rz-arena: {error}");
            rz_arena::emit_native_phase("cli_failed");
            ExitCode::from(2)
        }
    }
}

fn parse_command(args: Vec<OsString>) -> Result<Command, String> {
    let command = args.first().and_then(|arg| arg.to_str());
    if matches!(command, Some("--help" | "-h")) && args.len() == 1 {
        return Ok(Command::Help);
    }
    if command == Some("native-lock") {
        if args.len() != 3 {
            return Err("native-lock requires exactly INPUT OUTPUT".to_string());
        }
        return Ok(Command::NativeLock {
            input: PathBuf::from(&args[1]),
            output: PathBuf::from(&args[2]),
        });
    }
    if command == Some("native-pair") {
        if args.len() != 5 {
            return Err(
                "native-pair requires exactly LOCKED ARTIFACT_ROOT OUTPUT_ROOT NEW_OUTPUT_BASENAME"
                    .to_string(),
            );
        }
        return Ok(Command::NativePair {
            locked: PathBuf::from(&args[1]),
            artifact_root: PathBuf::from(&args[2]),
            output_root: PathBuf::from(&args[3]),
            output_directory: args[4].clone(),
        });
    }
    #[cfg(feature = "native-cuda")]
    if matches!(
        command,
        Some(
            "native-cuda-lock"
                | "native-cuda-v2-lock"
                | "native-cuda-pilot-lock"
                | "native-cuda-batch-lock"
                | "native-cuda-b1-clock-lock"
        )
    ) {
        if args.len() != 3 {
            return Err("native-cuda-lock requires exactly INPUT OUTPUT".into());
        }
        return Ok(Command::NativeCudaLock {
            version: if command == Some("native-cuda-b1-clock-lock") {
                5
            } else if command == Some("native-cuda-batch-lock") {
                4
            } else if command == Some("native-cuda-lock") {
                1
            } else if command == Some("native-cuda-pilot-lock") {
                3
            } else {
                2
            },
            input: PathBuf::from(&args[1]),
            output: PathBuf::from(&args[2]),
        });
    }
    #[cfg(feature = "native-cuda")]
    if matches!(
        command,
        Some(
            "native-cuda-pair"
                | "native-cuda-v2-pair"
                | "native-cuda-pilot-pair"
                | "native-cuda-batch-pair"
                | "native-cuda-b1-clock-pair"
        )
    ) {
        if args.len() != 5 {
            return Err("native-cuda-pair requires exactly LOCKED ARTIFACT_ROOT OUTPUT_ROOT NEW_OUTPUT_BASENAME".into());
        }
        return Ok(Command::NativeCudaPair {
            version: if command == Some("native-cuda-b1-clock-pair") {
                5
            } else if command == Some("native-cuda-batch-pair") {
                4
            } else if command == Some("native-cuda-pair") {
                1
            } else if command == Some("native-cuda-pilot-pair") {
                3
            } else {
                2
            },
            locked: PathBuf::from(&args[1]),
            artifact_root: PathBuf::from(&args[2]),
            output_root: PathBuf::from(&args[3]),
            output_directory: args[4].clone(),
        });
    }
    let (operation, path_count, ledger_bounds) = match command {
        Some("plan") => (Operation::Plan, 2, false),
        Some("ledger-init") => (Operation::LedgerInit, 2, true),
        Some("ledger-append") => (Operation::LedgerAppend, 4, true),
        Some("audit") => (Operation::Audit, 2, true),
        Some("fixture-pair") => (Operation::FixturePair, 3, false),
        _ => return Err(format!("invalid or missing command\n{USAGE}")),
    };
    if args.len() < path_count + 1 {
        return Err(format!("missing command paths\n{USAGE}"));
    }
    let mut max_pairs = None;
    let mut max_plan_bytes = None;
    let mut max_events = None;
    let mut max_ledger_bytes = None;
    let mut expected_tip = None;
    let mut index = path_count + 1;
    while index < args.len() {
        let value = args
            .get(index + 1)
            .ok_or_else(|| "bound option requires a value".to_string())?;
        if args[index] == "--expected-tip" {
            if !matches!(operation, Operation::Audit) {
                return Err("--expected-tip is only supported by audit".to_string());
            }
            if expected_tip.is_some() {
                return Err("duplicate --expected-tip option".to_string());
            }
            let tip = value
                .to_str()
                .filter(|text| {
                    text.len() == 64
                        && text
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                })
                .ok_or_else(|| {
                    "--expected-tip must be 64 lowercase hexadecimal digits".to_string()
                })?;
            expected_tip = Some(tip.to_string());
            index += 2;
            continue;
        }
        let target = match args[index].to_str() {
            Some("--max-pairs") => &mut max_pairs,
            Some("--max-plan-bytes") => &mut max_plan_bytes,
            Some("--max-events") if ledger_bounds => &mut max_events,
            Some("--max-ledger-bytes") if ledger_bounds => &mut max_ledger_bytes,
            _ => return Err("unknown bound option".to_string()),
        };
        if target.is_some() {
            return Err("duplicate bound option".to_string());
        }
        *target = Some(
            value
                .to_str()
                .and_then(|text| text.parse::<u64>().ok())
                .filter(|number| *number > 0)
                .ok_or_else(|| "all bounds must be positive u64 values".to_string())?,
        );
        index += 2;
    }
    let max_pairs = max_pairs.ok_or_else(|| "--max-pairs is required".to_string())?;
    let max_plan_bytes =
        max_plan_bytes.ok_or_else(|| "--max-plan-bytes is required".to_string())?;
    if max_plan_bytes > JSON_LIMIT {
        return Err("--max-plan-bytes cannot exceed 4 MiB".to_string());
    }
    if ledger_bounds && (max_events.is_none() || max_ledger_bytes.is_none()) {
        return Err("--max-events and --max-ledger-bytes are required".to_string());
    }
    let max_plan_bytes = usize::try_from(max_plan_bytes)
        .map_err(|_| "plan byte bound is unsupported on this platform".to_string())?;
    Ok(Command::Execute {
        operation,
        paths: args[1..=path_count].iter().map(PathBuf::from).collect(),
        bounds: Bounds {
            max_pairs,
            max_plan_bytes,
            max_events,
            max_ledger_bytes,
        },
        expected_tip,
    })
}

fn execute(command: Command) -> Result<String, String> {
    let command = match command {
        #[cfg(feature = "native-cuda")]
        Command::NativeCudaLock {
            input,
            output,
            version,
        } => {
            let limit = MAX_CUDA_NATIVE_LAUNCH_JSON_BYTES as u64;
            let input = read_text(&input, limit)?;
            let json = if version == 5 {
                rz_experiments::CudaB1ClockPairSpecV5::from_json(&input)
                    .and_then(|s| s.lock())
                    .and_then(|s| s.to_json())
            } else if version == 4 {
                rz_experiments::CudaBatchPilotPairSpecV4::from_json(&input)
                    .and_then(|s| s.lock())
                    .and_then(|s| s.to_json())
            } else if version == 1 {
                CudaIntegrationPairSpecV1::from_json(&input)
                    .and_then(|s| s.lock())
                    .and_then(|s| s.to_json())
            } else if version == 3 {
                CudaSearchPilotPairSpecV3::from_json(&input)
                    .and_then(|s| s.lock())
                    .and_then(|s| s.to_json())
            } else {
                CudaIntegrationPairSpecV2::from_json(&input)
                    .and_then(|s| s.lock())
                    .and_then(|s| s.to_json())
            }
            .map_err(|e| e.to_string())?;
            write_new_file(&output, json.as_bytes(), limit)?;
            return Ok(success(if version == 3 {
                "native CUDA search pilot declarations locked; execution not verified"
            } else {
                "native CUDA integration declarations locked; execution not verified"
            }));
        }
        #[cfg(feature = "native-cuda")]
        Command::NativeCudaPair {
            version,
            locked,
            artifact_root,
            output_root,
            output_directory,
        } => {
            let input = read_text(&locked, MAX_CUDA_NATIVE_LAUNCH_JSON_BYTES as u64)?;
            return if version == 5 {
                execute_cuda_pair(
                    rz_experiments::LockedCudaB1ClockPairSpecV5::from_json(&input)
                        .map_err(|e| e.to_string())?,
                    artifact_root,
                    output_root,
                    output_directory,
                )
            } else if version == 4 {
                execute_cuda_pair(
                    rz_experiments::LockedCudaBatchPilotPairSpecV4::from_json(&input)
                        .map_err(|e| e.to_string())?,
                    artifact_root,
                    output_root,
                    output_directory,
                )
            } else if version == 1 {
                execute_cuda_pair(
                    LockedCudaIntegrationPairSpecV1::from_json(&input)
                        .map_err(|e| e.to_string())?,
                    artifact_root,
                    output_root,
                    output_directory,
                )
            } else if version == 3 {
                execute_cuda_pair(
                    LockedCudaSearchPilotPairSpecV3::from_json(&input)
                        .map_err(|e| e.to_string())?,
                    artifact_root,
                    output_root,
                    output_directory,
                )
            } else {
                execute_cuda_pair(
                    LockedCudaIntegrationPairSpecV2::from_json(&input)
                        .map_err(|e| e.to_string())?,
                    artifact_root,
                    output_root,
                    output_directory,
                )
            };
        }
        Command::NativeLock { input, output } => {
            let limit = MAX_NATIVE_LAUNCH_JSON_BYTES as u64;
            let spec = IntegrationPairSpecV1::from_json(&read_text(&input, limit)?)
                .map_err(|error| error.to_string())?;
            let locked = spec.lock().map_err(|error| error.to_string())?;
            let json = locked.to_json().map_err(|error| error.to_string())?;
            write_new_file(&output, json.as_bytes(), limit)?;
            return Ok(success(
                "native integration declarations locked; execution not verified",
            ));
        }
        Command::NativePair {
            locked,
            artifact_root,
            output_root,
            output_directory,
        } => {
            let spec = LockedIntegrationPairSpecV1::from_json(&read_text(
                &locked,
                MAX_NATIVE_LAUNCH_JSON_BYTES as u64,
            )?)
            .map_err(|error| error.to_string())?;
            let output_directory = output_directory
                .to_str()
                .ok_or_else(|| "output basename must be ASCII".to_string())?;
            let cancelled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            #[cfg(unix)]
            {
                signal_hook::flag::register(signal_hook::consts::SIGINT, cancelled.clone())
                    .map_err(|_| "cannot install SIGINT cancellation".to_string())?;
                signal_hook::flag::register(signal_hook::consts::SIGTERM, cancelled.clone())
                    .map_err(|_| "cannot install SIGTERM cancellation".to_string())?;
            }
            let owner = prepare_native_launch(
                &spec,
                &artifact_root,
                &output_root,
                output_directory,
            )
            .map_err(|failure| {
                if let Ok(receipt) = serde_json::to_string(&serde_json::json!({
                    "preparation": failure.receipt,
                    "receipt_artifact": failure.receipt_artifact,
                    "persistence_error": failure.persistence_error.as_ref().map(ToString::to_string),
                })) {
                    eprintln!("native_preparation_failure_receipt={receipt}");
                }
                failure.to_string()
            })?;
            rz_arena::emit_native_phase("preparation_complete");
            let output = run_native_pair(owner, Some(&cancelled)).map_err(|failure| {
                if let Ok(receipt) = serde_json::to_string(&serde_json::json!({
                    "receipt": failure.receipt,
                    "receipt_artifact": failure.receipt_artifact,
                    "process": failure.process().map(|process| &process.receipt),
                    "cause": failure.cause.to_string(),
                    "ownership_scope": "unverified process ownership is retained for this CLI process lifetime; files are preserved",
                })) {
                    eprintln!("native_pair_failure_receipt={receipt}");
                }
                failure.to_string()
            })?;
            return serde_json::to_string_pretty(&serde_json::json!({
                "execution_ready": false,
                "strength_eligible": false,
                "validation_scope": output.receipt.validation_scope,
                "integration_checks_passed": output.receipt.integration_checks_passed,
                "receipt": output.receipt_artifact,
                "provider_sessions": output.receipt.provider_sessions,
                "scored_games": output.receipt.scored_games,
                "incomplete_games": output.receipt.incomplete_games,
                "process_cleanup": output.receipt.process.group_cleanup,
            }))
            .map_err(|error| error.to_string());
        }
        other => other,
    };
    let Command::Execute {
        operation,
        paths,
        bounds,
        expected_tip,
    } = command
    else {
        #[cfg(feature = "native-cuda")]
        return Ok(format!(
            "{USAGE}\n\nWith native-cuda feature (historical bundle profiles):\n  rz-arena native-cuda-lock INPUT OUTPUT\n  rz-arena native-cuda-pair LOCKED ARTIFACT_ROOT OUTPUT_ROOT NEW_OUTPUT_BASENAME\n  rz-arena native-cuda-v2-lock INPUT OUTPUT\n  rz-arena native-cuda-v2-pair LOCKED ARTIFACT_ROOT OUTPUT_ROOT NEW_OUTPUT_BASENAME\nV2 is BT4-only A/A with an explicit search profile; V1 limits and domain remain closed. Historical lock formats remain readable, but bundle-based execution and runtime copies are disabled. System CUDA uses a separate runtime profile and requires its own execution and physical-completion evidence."
        ) + "\nV3 search pilot commands:\n  rz-arena native-cuda-pilot-lock INPUT OUTPUT\n  rz-arena native-cuda-pilot-pair LOCKED ARTIFACT_ROOT OUTPUT_ROOT NEW_OUTPUT_BASENAME\nV3 retains the separate BT4 S0/S1 protocol and paired cohort.\nV4 batch pilot (native-cuda-batch feature):\n  rz-arena native-cuda-batch-lock INPUT OUTPUT\n  rz-arena native-cuda-batch-pair LOCKED ARTIFACT_ROOT OUTPUT_ROOT NEW_OUTPUT_BASENAME\nV5 B1 visits A/A memory check:\n  rz-arena native-cuda-b1-clock-lock INPUT OUTPUT\n  rz-arena native-cuda-b1-clock-pair LOCKED ARTIFACT_ROOT OUTPUT_ROOT NEW_OUTPUT_BASENAME\nV4/V5 fix 120+1 Fischer, max256 ply, 15min pair wall and 30s cleanup. Pilot receipts confer no promotion or Elo authority.");
        #[cfg(not(feature = "native-cuda"))]
        return Ok(USAGE.to_string());
    };
    if matches!(operation, Operation::Plan) {
        let manifest = LockedManifest::from_json(&read_text(&paths[0], JSON_LIMIT)?)
            .map_err(|error| error.to_string())?;
        let plan =
            ArenaPlan::build(&manifest, bounds.plan_limits()).map_err(|error| error.to_string())?;
        let json = plan.to_json().map_err(|error| error.to_string())?;
        write_new_file(&paths[1], json.as_bytes(), bounds.max_plan_bytes as u64)?;
        return Ok(success("plan created"));
    }
    let plan = ArenaPlan::from_json(
        &read_text(&paths[0], bounds.max_plan_bytes as u64)?,
        bounds.plan_limits(),
    )
    .map_err(|error| error.to_string())?;
    if matches!(operation, Operation::FixturePair) {
        let output_name = paths[2]
            .to_str()
            .ok_or_else(|| "output basename must be ASCII".to_string())?;
        let cancelled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        #[cfg(unix)]
        {
            signal_hook::flag::register(signal_hook::consts::SIGINT, cancelled.clone())
                .map_err(|_| "cannot install SIGINT cancellation".to_string())?;
            signal_hook::flag::register(signal_hook::consts::SIGTERM, cancelled.clone())
                .map_err(|_| "cannot install SIGTERM cancellation".to_string())?;
        }
        let result =
            run_fixture_pair(&plan, &paths[1], output_name, Some(&cancelled)).map_err(|error| {
                if let rz_arena::ArenaError::Execution(failure) = &error {
                    // Last available evidence channel when output storage fails.
                    if let Ok(receipt) = serde_json::to_string(&failure.process.receipt) {
                        eprintln!("process_failure_receipt={receipt}");
                    }
                }
                error.to_string()
            })?;
        if result.receipt.process.group_cleanup != rz_arena::CleanupStatus::Gone {
            return Err(format!(
                "fixture cleanup is unverified (pid={}); receipt and excluded ledger preserved; CLI releases pending ownership on exit",
                result.receipt.process.pid
            ));
        }
        if result.receipt.pgn_audit.is_none() {
            return Err(format!(
                "fixture run failed ({:?}); receipt {} and excluded ledger preserved",
                result.receipt.process.stop, result.receipt_artifact.path
            ));
        }
        return serde_json::to_string_pretty(&serde_json::json!({
            "execution_ready": false,
            "validation_scope": result.receipt.validation_scope,
            "receipt": result.receipt_artifact,
            "ledger_tip_sha256": result.ledger_tip_sha256,
            "summary": result.summary,
            "process_cleanup": result.receipt.process.group_cleanup,
        }))
        .map_err(|error| error.to_string());
    }
    let ledger_limits = bounds.ledger_limits(&plan)?;
    let max_ledger_bytes = ledger_limits.max_bytes;
    match operation {
        Operation::LedgerInit => {
            let ledger = Ledger::new(&plan, ledger_limits).map_err(|error| error.to_string())?;
            let jsonl = ledger.to_jsonl().map_err(|error| error.to_string())?;
            write_new_file(&paths[1], jsonl.as_bytes(), max_ledger_bytes)?;
            Ok(success("ledger initialized"))
        }
        Operation::LedgerAppend => {
            let mut ledger = Ledger::from_jsonl(
                &plan,
                &read_text(&paths[1], max_ledger_bytes)?,
                ledger_limits,
            )
            .map_err(|error| error.to_string())?;
            let event: Event = decode_json(&read_text(&paths[2], JSON_LIMIT)?)
                .map_err(|error| error.to_string())?;
            ledger.append(event).map_err(|error| error.to_string())?;
            let jsonl = ledger.to_jsonl().map_err(|error| error.to_string())?;
            write_new_file(&paths[3], jsonl.as_bytes(), max_ledger_bytes)?;
            Ok(success("ledger event appended"))
        }
        Operation::Audit => {
            let ledger = Ledger::from_jsonl(
                &plan,
                &read_text(&paths[1], max_ledger_bytes)?,
                ledger_limits,
            )
            .map_err(|error| error.to_string())?;
            if let Some(tip) = expected_tip {
                ledger.verify_tip(&tip).map_err(|error| error.to_string())?;
            }
            let summary = ledger.summary().map_err(|error| error.to_string())?;
            let output = serde_json::json!({
                "execution_ready": false,
                "validation_scope": "structural_only",
                "tip_sha256": ledger.tip_sha256(),
                "summary": summary,
            });
            serde_json::to_string_pretty(&output).map_err(|error| error.to_string())
        }
        Operation::Plan => unreachable!("plan operation is handled above"),
        Operation::FixturePair => unreachable!("fixture pair is handled above"),
    }
}

fn success(message: &str) -> String {
    format!("{message}; validation_scope=structural_only; execution_ready=false")
}

fn read_text(path: &Path, max_bytes: u64) -> Result<String, String> {
    let metadata =
        fs::symlink_metadata(path).map_err(|error| format!("cannot inspect input: {error}"))?;
    if !metadata.is_file() {
        return Err("input must be a regular file".to_string());
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let file = options
        .open(path)
        .map_err(|error| format!("cannot open input: {error}"))?;
    let metadata = file
        .metadata()
        .map_err(|error| format!("cannot inspect opened input: {error}"))?;
    if !metadata.is_file() {
        return Err("input must be a regular file".to_string());
    }
    if metadata.len() > max_bytes {
        return Err(format!("input exceeds its {max_bytes}-byte bound"));
    }
    let read_limit = max_bytes
        .checked_add(1)
        .ok_or_else(|| "input byte bound is too large for a sentinel read".to_string())?;
    let mut bytes = Vec::new();
    file.take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read input: {error}"))?;
    if bytes.len() as u64 > max_bytes {
        return Err(format!("input exceeds its {max_bytes}-byte bound"));
    }
    String::from_utf8(bytes).map_err(|_| "input is not valid UTF-8".to_string())
}

fn write_new_file(path: &Path, bytes: &[u8], max_bytes: u64) -> Result<(), String> {
    if bytes.len() as u64 > max_bytes {
        return Err(format!("output exceeds its {max_bytes}-byte bound"));
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create output (existing files are preserved): {error}"))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|error| {
            format!("cannot write output; partial output preserved for inspection: {error}")
        })
}

#[cfg(feature = "native-cuda")]
fn execute_cuda_pair<P: rz_experiments::CudaLaunchProfile>(
    spec: rz_experiments::LockedCudaIntegrationPairSpec<P>,
    artifact_root: PathBuf,
    output_root: PathBuf,
    output_directory: OsString,
) -> Result<String, String> {
    if P::VERSION == 4 && !cfg!(feature = "native-cuda-batch") {
        return Err("native-cuda-batch feature is required before preparing a V4 launch".into());
    }
    let name = output_directory
        .to_str()
        .ok_or_else(|| "output basename must be ASCII".to_string())?;
    let cancelled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    #[cfg(unix)]
    {
        signal_hook::flag::register(signal_hook::consts::SIGINT, cancelled.clone())
            .map_err(|_| "cannot install SIGINT cancellation".to_string())?;
        signal_hook::flag::register(signal_hook::consts::SIGTERM, cancelled.clone())
            .map_err(|_| "cannot install SIGTERM cancellation".to_string())?;
    }
    let owner=prepare_native_cuda_launch(&spec,&artifact_root,&output_root,name).map_err(|failure| {
                if let Ok(receipt)=serde_json::to_string(&serde_json::json!({"preparation":failure.receipt,"receipt_artifact":failure.receipt_artifact,
                    "persistence_error":failure.persistence_error.as_ref().map(ToString::to_string)})) {
                    eprintln!("native_cuda_preparation_failure_receipt={receipt}");
                }
                failure.to_string()
            })?;
    rz_arena::emit_native_phase("preparation_complete");
    let output=run_native_cuda_pair(owner,Some(&cancelled)).map_err(|failure| {
                if let Ok(receipt)=serde_json::to_string(&serde_json::json!({"receipt":failure.receipt,"receipt_artifact":failure.receipt_artifact,
                    "process":failure.process().map(|p|&p.receipt),"cause":failure.cause.to_string(),
                    "ownership_scope":"unverified process ownership is retained for this CLI process lifetime; files are preserved"})) {
                    eprintln!("native_cuda_pair_failure_receipt={receipt}");
                }
                failure.to_string()
            })?;
    serde_json::to_string_pretty(&serde_json::json!({"execution_ready":false,"strength_eligible":false,
                "validation_scope":output.receipt.validation_scope,"integration_checks_passed":output.receipt.integration_checks_passed,
                "receipt":output.receipt_artifact,"provider_sessions":output.receipt.provider_sessions,"scored_games":output.receipt.scored_games,
                "incomplete_games":output.receipt.incomplete_games,"process_cleanup":output.receipt.process.group_cleanup}))
                .map_err(|e|e.to_string())
}
