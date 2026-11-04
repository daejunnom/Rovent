//! Finite external UCI probe without RoveZero model inference or training.
//! The external engine may evaluate its embedded NNUE. Run only under the
//! existing outer process-tree/resource supervisor; unknown join is not clean.

#[cfg(target_os = "linux")]
mod linux_probe {
    use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt, OpenOptionsSyncExt};
    use cap_std::{
        ambient_authority,
        fs::{Dir, OpenOptions},
    };
    use rz_contracts::pals::SearchAuthority;
    use rz_contracts::pals_adaptive::{CpuWorkerId, ReviewViewSeal};
    use rz_contracts::rovent::{
        ChessPly, QuestionEnvelope, QuestionPurpose, ReciprocalQuestionId, SearchRecipient,
    };
    use rz_contracts::{
        ClockDomain, Deadline, Digest, ExecutionId, GameGeneration, MonotonicTick, OwnerId,
        ProcessEpoch, RootGeneration,
    };
    use rz_experiments::RoventInputLock;
    use rz_position::{Position, contracts::ContractPosition};
    use rz_runtime::adaptive::ControlMailbox;
    use rz_search::external_cpu::worker::ExternalWorkerClock;
    use rz_search::external_cpu::{
        ExternalGoConstraint, ProfiledExternalJob, ProfiledExternalScope,
    };
    use rz_uci::rovent_external_launch::{PreparedRoventExternalLaunch, RoventExternalSlot};
    use rz_uci::{Command, ParserLimits, PositionBase};
    use serde::{Deserialize, Serialize};
    use serde_json::{Value, json};
    use sha2::{Digest as _, Sha256};
    use std::{
        ffi::OsString,
        fs::File,
        io::{Read, Write},
        path::{Component, Path, PathBuf},
        sync::atomic::AtomicBool,
        time::{Duration, Instant},
    };

    #[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    enum Scenario {
        Natural,
        CancelAfterProgress,
        YieldAfterProgress,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Request {
        schema: String,
        input_lock: PathBuf,
        slot: RoventExternalSlot,
        profile: PathBuf,
        binary: PathBuf,
        working_directory: PathBuf,
        position_command: String,
        constraint: ExternalGoConstraint,
        soft_quantum_ms: u64,
        maximum_wall_ms: u64,
        cleanup_ms: u64,
        process_epoch: u64,
        scenario: Scenario,
    }

    fn validate_request(request: &Request) -> Result<(), String> {
        if request.schema != "rz-rovent-external-probe-request/1"
            || request.process_epoch == 0
            || !(1000..=60_000).contains(&request.maximum_wall_ms)
            || !(500..=30_000).contains(&request.cleanup_ms)
            || request.maximum_wall_ms <= request.cleanup_ms + 1000
            || !(100..=10_000).contains(&request.soft_quantum_ms)
            || request.position_command.len() > 16_384
        {
            return Err("request_schema_or_finite_limits".into());
        }
        Ok(())
    }

    fn parent(path: &Path) -> Result<(Dir, OsString), String> {
        if !path.is_absolute() {
            return Err("absolute_input_output_path_required".into());
        }
        let components = path.components().collect::<Vec<_>>();
        let Some(Component::Normal(name)) = components.last() else {
            return Err("regular_file_name_required".into());
        };
        let mut directory = Dir::open_ambient_dir("/", ambient_authority())
            .map_err(|_| "root_capability_unavailable")?;
        for part in &components[..components.len() - 1] {
            match part {
                Component::RootDir => {}
                Component::Normal(part) => {
                    directory = directory
                        .open_dir_nofollow(part)
                        .map_err(|_| "input_output_parent_missing_or_linked")?
                }
                _ => return Err("normal_path_components_required".into()),
            }
        }
        Ok((directory, name.to_os_string()))
    }

    fn open_regular(path: &Path) -> Result<File, String> {
        let (directory, name) = parent(path)?;
        let mut options = OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No).nonblock(true);
        let file = directory
            .open_with(name, &options)
            .map_err(|_| "asset_missing_or_linked")?
            .into_std();
        if !file
            .metadata()
            .map_err(|_| "asset_metadata_unavailable")?
            .is_file()
        {
            return Err("asset_not_regular".into());
        }
        Ok(file)
    }

    fn read_bounded(path: &Path, cap: u64) -> Result<Vec<u8>, String> {
        use std::os::unix::fs::MetadataExt;
        let mut file = open_regular(path)?;
        let before = file.metadata().map_err(|_| "input_metadata_unavailable")?;
        if before.len() == 0 || before.len() > cap {
            return Err("input_byte_bounds".into());
        }
        let mut bytes = Vec::with_capacity(before.len() as usize);
        (&mut file)
            .take(cap + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| "bounded_input_read_failed")?;
        let after = file.metadata().map_err(|_| "input_recheck_failed")?;
        if bytes.len() as u64 != before.len()
            || after.len() != before.len()
            || before.dev() != after.dev()
            || before.ino() != after.ino()
            || before.mtime() != after.mtime()
            || before.mtime_nsec() != after.mtime_nsec()
            || before.ctime() != after.ctime()
            || before.ctime_nsec() != after.ctime_nsec()
        {
            return Err("input_changed_during_read".into());
        }
        Ok(bytes)
    }

    fn directory(path: &Path) -> Result<File, String> {
        if !path.is_absolute() {
            return Err("absolute_working_directory_required".into());
        }
        let mut directory = Dir::open_ambient_dir("/", ambient_authority())
            .map_err(|_| "root_capability_unavailable")?;
        for part in path.components() {
            match part {
                Component::RootDir => {}
                Component::Normal(part) => {
                    directory = directory
                        .open_dir_nofollow(part)
                        .map_err(|_| "working_directory_missing_or_linked")?
                }
                _ => return Err("normal_working_directory_components_required".into()),
            }
        }
        Ok(directory.into_std_file())
    }

    fn parse_digest(value: &str) -> Result<Digest, String> {
        let mut bytes = [0; 32];
        if value.len() != 64 {
            return Err("digest_length".into());
        }
        for (at, pair) in value.as_bytes().chunks_exact(2).enumerate() {
            let digit = |b: u8| match b {
                b'0'..=b'9' => Some(b - b'0'),
                b'a'..=b'f' => Some(b - b'a' + 10),
                _ => None,
            };
            bytes[at] = digit(pair[0])
                .zip(digit(pair[1]))
                .map(|(a, b)| a * 16 + b)
                .ok_or("digest_encoding")?;
        }
        Ok(Digest(bytes))
    }

    fn position(command: &str) -> Result<Position, String> {
        let Command::Position(spec) = rz_uci::parse(command, ParserLimits::default())
            .map_err(|_| "invalid_position_command")?
        else {
            return Err("position_command_required".into());
        };
        let mut position = match spec.base {
            PositionBase::StartPos => Position::startpos(),
            PositionBase::Fen(fen) => Position::from_fen(&fen).map_err(|_| "invalid_fen")?,
        };
        for text in spec.moves {
            position
                .make_move(rz_position::BoardMove::from_uci(&text).map_err(|_| "invalid_move")?)
                .map_err(|_| "illegal_position_move")?;
        }
        Ok(position)
    }

    fn run(request: Request) -> Result<Value, String> {
        validate_request(&request)?;
        let started = Instant::now();
        let whole = started
            .checked_add(Duration::from_millis(request.maximum_wall_ms))
            .ok_or("whole_deadline_overflow")?;
        let admission = whole
            .checked_sub(Duration::from_millis(request.cleanup_ms))
            .ok_or("cleanup_deadline_overflow")?;
        let lock_bytes = read_bounded(&request.input_lock, 512 * 1024)?;
        let lock = RoventInputLock::from_json(
            std::str::from_utf8(&lock_bytes).map_err(|_| "lock_not_utf8")?,
        )
        .map_err(|_| "invalid_input_lock")?;
        if request.maximum_wall_ms > lock.manifest.wall_time_max_ms
            || request.cleanup_ms > lock.manifest.cleanup_max_ms
        {
            return Err("probe_window_exceeds_input_lock".into());
        }
        let profile_bytes = read_bounded(&request.profile, 64 * 1024)?;
        let launch = PreparedRoventExternalLaunch::from_lock(&lock, request.slot, &profile_bytes)
            .map_err(|e| format!("profile_admission:{e:?}"))?;
        launch
            .require_query(request.constraint, false)
            .map_err(|e| format!("query_admission:{e:?}"))?;
        if launch.profile().max_output_bytes > 256 * 1024 {
            return Err("probe_requires_explicit_output_cap_at_most_256kib".into());
        }
        let position = position(&request.position_command)?;
        let state = ContractPosition::new(OwnerId(1), position.clone())
            .export()
            .map_err(|_| "rules_export_failed")?;
        let anchor = state.snapshot().identity();
        let clock = ExternalWorkerClock {
            domain: ClockDomain(ProcessEpoch(request.process_epoch)),
            origin: started,
        };
        // The lock digest namespaces this diagnostic root. This does not claim
        // source/binary verification for the complete RoveZero engine.
        let authority = SearchAuthority {
            epoch: clock.domain.0,
            game: GameGeneration(1),
            root: RootGeneration(1),
            implementation: parse_digest(&lock.canonical_sha256)?,
        };
        let cancel = AtomicBool::new(false);
        let mut owner = launch
            .spawn_pinned(
                open_regular(&request.binary)?,
                directory(&request.working_directory)?,
                CpuWorkerId(1),
                clock,
                authority,
                admission,
                &cancel,
            )
            .map_err(|e| format!("spawn_admission:{e:?}"))?;
        let signal = owner.worker().signal();
        let mut startup_summary = Value::Null;
        let mut completion_summary = Value::Null;
        let mut control_sent = false;
        let mut retained_dispatch = None;
        let result = (|| -> Result<(), String> {
            loop {
                let version = signal.version();
                if let Some(startup) = owner
                    .verified_startup()
                    .map_err(|e| format!("startup:{e:?}"))?
                {
                    let process = startup.process_identity();
                    startup_summary = json!({"profile_canonical_sha256": startup.canonical_sha256(), "capability_sha256": startup.capability_sha256(), "input_lock_sha256": startup.input_lock_sha256(), "uci_name": startup.observed_uci().name, "uci_author": startup.observed_uci().author, "pid": process.pid, "process_group": process.process_group, "proc_start_ticks": process.proc_start_ticks, "options_application_observed": startup.options_application_observed(), "model_loading_observed": startup.model_loading_observed()});
                    break;
                }
                if Instant::now() >= admission || owner.try_join().is_some() {
                    return Err("startup_not_available_before_deadline".into());
                }
                signal.wait_changed(version, Duration::from_millis(20));
            }
            let question = QuestionEnvelope {
                id: ReciprocalQuestionId {
                    authority,
                    sequence: 1,
                },
                parent: None,
                sender: SearchRecipient::Proposer,
                recipient: SearchRecipient::Researcher,
                purpose: QuestionPurpose::TacticalReview,
                anchor,
                anchor_root_ply: ChessPly(0),
                original_first_move: None,
                prefix_digest: Digest(Sha256::digest(request.position_command.as_bytes()).into()),
                line_digest: None,
                line_offset: ChessPly(0),
                question_digest: Digest(
                    Sha256::digest(
                        serde_json::to_vec(&request.constraint)
                            .map_err(|_| "constraint_encoding")?,
                    )
                    .into(),
                ),
                view: ReviewViewSeal {
                    authority,
                    view_sequence: 1,
                    context_policy: Digest([1; 32]),
                    view_digest: Digest([2; 32]),
                    root_state: anchor,
                    actual_tokens: 1,
                    selected_records: 0,
                    anchors: 1,
                    length_bucket: 8,
                },
                evidence_revision: 0,
                root_budget: parse_digest(&lock.manifest.coordinator.original_root_budget_sha256)?,
                original_deadline: Deadline {
                    clock: clock.domain,
                    at: MonotonicTick(
                        u64::try_from(admission.duration_since(started).as_nanos())
                            .map_err(|_| "deadline_tick_overflow")?,
                    ),
                },
            };
            let scope = ProfiledExternalScope {
                question: question.id,
                execution: ExecutionId {
                    epoch: authority.epoch,
                    sequence: 1,
                },
                owner: CpuWorkerId(1),
            };
            let control = ControlMailbox::new(scope, signal.clone());
            let dispatch = owner
                .submit(ProfiledExternalJob {
                    question,
                    mode: lock.manifest.engine.mode,
                    execution: scope.execution,
                    owner: scope.owner,
                    position,
                    root_moves: None,
                    constraint: request.constraint,
                    soft_quantum: Duration::from_millis(request.soft_quantum_ms),
                    context_token_budget: 512,
                    control: control.clone(),
                })
                .map_err(|e| format!("submit:{e:?}"))?;
            retained_dispatch = Some(dispatch.clone());
            loop {
                let version = signal.version();
                if let Some(completion) = owner.worker().take_completion() {
                    let original_terminal_debug = format!("{completion:?}");
                    completion_summary =
                        json!({"original_terminal_debug": original_terminal_debug});
                    owner
                        .validate_completion(&completion, &dispatch)
                        .map_err(|e| format!("completion_provenance:{e:?}"))?;
                    let controls = completion.controls();
                    let timing = completion.timing();
                    let millis = |time: Option<Instant>| {
                        time.map(|at| at.duration_since(started).as_millis())
                    };
                    completion_summary = json!({"go_issue": format!("{:?}", completion.go_issue()), "controls": {"cancel_sequence": controls.cancel_sequence, "cancel_ack": controls.cancel_ack, "yield_sequence": controls.yield_sequence, "yield_ack": controls.yield_ack, "terminal": controls.terminal}, "timing_ms": {"admitted": millis(timing.admitted_at), "run_started": millis(timing.run_started_at), "go_issued": millis(timing.go_issued_at), "stop_sent": millis(timing.stop_sent_at), "bestmove_drained": millis(timing.bestmove_drained_at), "ready_completed": millis(timing.ready_completed_at), "returned": millis(timing.returned_at)}});
                    completion_summary["original_terminal_debug"] = json!(original_terminal_debug);
                    match completion.result() {
                        Ok(outcome) => {
                            let r = &outcome.report;
                            completion_summary["result"] = json!({"request_id":r.raw.request_id, "requested":r.requested, "issued":r.issued, "issued_go":r.issued_go, "bestmove":r.raw.best_move.map(|m|m.to_string()), "reported_depth":r.raw.reported_depth, "requested_depth":r.raw.requested_depth, "reported_nodes":r.raw.work.nodes, "node_overshoot":r.reported_node_overshoot, "completion":format!("{:?}",r.raw.completion), "raw_score":format!("{:?}",r.raw.score), "strict_native_nodes":r.strict_native_nodes, "readiness_barrier_completed":r.readiness_barrier_completed, "conditional_policy_target_known":outcome.conditional_policy_target_known});
                        }
                        Err(error) => return Err(format!("actual_query:{error:?}")),
                    }
                    if request.scenario != Scenario::Natural
                        && (!control_sent || (controls.cancel_ack == 0 && controls.yield_ack == 0))
                    {
                        return Err("requested_control_not_observed".into());
                    }
                    break;
                }
                while let Some((_sequence, progress)) = owner.worker().take_progress() {
                    if !control_sent && progress.scope == scope {
                        match request.scenario {
                            Scenario::Natural => {}
                            Scenario::CancelAfterProgress => {
                                control
                                    .request_cancel(1)
                                    .map_err(|_| "cancel_request_rejected")?;
                                control_sent = true;
                            }
                            Scenario::YieldAfterProgress => {
                                control
                                    .request_yield(1)
                                    .map_err(|_| "yield_request_rejected")?;
                                control_sent = true;
                            }
                        }
                    }
                }
                if Instant::now() >= admission {
                    return Err("query_deadline_before_terminal".into());
                }
                signal.wait_changed(version, Duration::from_millis(20));
            }
            Ok(())
        })();
        owner.worker().request_shutdown(whole);
        let mut joined = owner.worker().shutdown_snapshot();
        let mut late_terminals = Vec::new();
        while joined.is_none() && Instant::now() < whole {
            let version = signal.version();
            while let Some(completion) = owner.worker().take_completion() {
                let validation = retained_dispatch
                    .as_ref()
                    .map(|dispatch| owner.validate_completion(&completion, dispatch));
                late_terminals.push(json!({"original_terminal_debug":format!("{completion:?}"), "dispatch_validation":format!("{validation:?}")}));
            }
            joined = owner.try_join();
            if joined.is_none() {
                signal.wait_changed(version, Duration::from_millis(20));
            }
        }
        while let Some(completion) = owner.worker().take_completion() {
            let validation = retained_dispatch
                .as_ref()
                .map(|dispatch| owner.validate_completion(&completion, dispatch));
            late_terminals.push(json!({"original_terminal_debug":format!("{completion:?}"), "dispatch_validation":format!("{validation:?}")}));
        }
        let shutdown = match &joined {
            Some(Ok(s)) => {
                json!({"cleanup_complete":s.cleanup_complete, "quarantined":s.quarantined, "ownership_lost":s.ownership_lost, "stdout_drained":s.stdout_drained, "stderr_drained":s.stderr_drained, "exit_observed":s.exit_observed, "exit_code":s.exit_code, "exit_signal":s.exit_signal, "stop_sent":s.stop_sent, "quit_sent":s.quit_sent})
            }
            Some(Err(e)) => json!({"error":format!("{e:?}")}),
            None => Value::Null,
        };
        let clean = joined.as_ref().is_some_and(|r| {
            r.as_ref().is_ok_and(|s| {
                s.cleanup_complete
                    && !s.quarantined
                    && !s.ownership_lost
                    && s.stdout_drained
                    && s.stderr_drained
                    && s.exit_observed
            })
        });
        let diagnostics = owner.try_take_diagnostics_after_join();
        let transcript = diagnostics.map(|d| json!({"stdout_sha256":format!("{:x}",Sha256::digest(d.stdout())), "stderr_sha256":format!("{:x}",Sha256::digest(d.stderr())), "stdout_original_bytes":d.stdout(), "stderr_original_bytes":d.stderr()}));
        let occupancy = owner.worker().occupancy();
        let drained = occupancy.commands == 0
            && occupancy.reliable_queued == 0
            && occupancy.reliable_reserved == 0;
        Ok(
            json!({"schema":"rz-rovent-external-probe-receipt/1", "status":if result.is_ok() && clean && drained {"PASS_EXTERNAL_PROBE_ONLY"}else{"FAILED_OR_UNKNOWN"}, "error":result.err(), "input_lock_sha256":lock.canonical_sha256, "scenario":request.scenario, "control_request_sent":control_sent, "startup":startup_summary, "startup_original_observation_debug":owner.worker().startup_report().map(|startup|format!("{startup:?}")), "completion":completion_summary, "late_terminals":late_terminals, "shutdown":shutdown, "thread_join_observed":joined.is_some(), "terminal_obligations_drained":drained, "original_transcript":transcript, "wall_time_ms":started.elapsed().as_millis(), "resources_applied":"unknown", "performance_eligible":false, "training_performed":false, "full_engine_accepted":false, "requires_outer_process_tree_supervisor":true}),
        )
    }

    pub fn main() -> std::process::ExitCode {
        let args = std::env::args_os().skip(1).collect::<Vec<_>>();
        if args.len() != 2 {
            eprintln!("usage: rovent_external_probe REQUEST_JSON ABSOLUTE_RECEIPT_JSON");
            return std::process::ExitCode::from(2);
        }
        let value = read_bounded(Path::new(&args[0]), 64 * 1024)
            .and_then(|bytes| {
                serde_json::from_slice::<Request>(&bytes)
                    .map_err(|_| "invalid_probe_request_json".into())
            })
            .and_then(run);
        let receipt = match value {
            Ok(receipt) => receipt,
            Err(error) => {
                json!({"schema":"rz-rovent-external-probe-receipt/1", "status":"NOT_STARTED_OR_FAILED_ADMISSION", "error":error, "asset_availability":"unknown", "performance_eligible":false, "training_performed":false, "full_engine_accepted":false})
            }
        };
        let success = receipt["status"] == "PASS_EXTERNAL_PROBE_ONLY";
        let write = (|| -> Result<(), String> {
            // Compact arrays avoid per-byte indentation amplification while
            // retaining the original admitted 256 KiB process transcript.
            let bytes = serde_json::to_vec(&receipt).map_err(|_| "receipt_encoding_failed")?;
            if bytes.len() > 2 * 1024 * 1024 {
                return Err("receipt_output_cap".into());
            }
            let (parent, name) = parent(Path::new(&args[1]))?;
            let mut options = OpenOptions::new();
            options
                .write(true)
                .create_new(true)
                .follow(FollowSymlinks::No);
            let mut file = parent
                .open_with(name, &options)
                .map_err(|_| "receipt_exists_or_unavailable")?;
            file.write_all(&bytes).map_err(|_| "receipt_write_failed")?;
            file.sync_all().map_err(|_| "receipt_sync_failed")?;
            Ok(())
        })();
        if let Err(error) = write {
            eprintln!("{error}");
            return std::process::ExitCode::from(2);
        }
        if success {
            std::process::ExitCode::SUCCESS
        } else {
            std::process::ExitCode::from(1)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn probe_bounds_and_schema_are_checked_before_any_assets_or_process() {
            let mut request = Request {
                schema: "rz-rovent-external-probe-request/1".into(),
                input_lock: "/unavailable/lock.json".into(),
                slot: RoventExternalSlot::Researcher,
                profile: "/unavailable/profile.json".into(),
                binary: "/unavailable/engine".into(),
                working_directory: "/unavailable".into(),
                position_command: "position startpos".into(),
                constraint: ExternalGoConstraint::Nodes { reported_nodes: 32 },
                soft_quantum_ms: 100,
                maximum_wall_ms: 10_000,
                cleanup_ms: 1000,
                process_epoch: 1,
                scenario: Scenario::Natural,
            };
            assert!(validate_request(&request).is_ok());
            request.maximum_wall_ms = 60_001;
            assert!(validate_request(&request).is_err());
            request.maximum_wall_ms = 10_000;
            request.process_epoch = 0;
            assert!(validate_request(&request).is_err());
        }

        #[test]
        fn maximum_admitted_raw_transcript_fits_the_compact_receipt_budget() {
            let receipt = json!({"stdout_original_bytes":vec![255_u8;256*1024],"stderr_original_bytes":Vec::<u8>::new()});
            let bytes = serde_json::to_vec(&receipt).unwrap();
            assert!(bytes.len() < 2 * 1024 * 1024);
            let decoded: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(
                decoded["stdout_original_bytes"].as_array().unwrap().len(),
                256 * 1024
            );
            assert_eq!(decoded["stdout_original_bytes"][256 * 1024 - 1], 255);
        }
    }
}

#[cfg(target_os = "linux")]
fn main() -> std::process::ExitCode {
    linux_probe::main()
}
#[cfg(not(target_os = "linux"))]
fn main() -> std::process::ExitCode {
    eprintln!("Rovent external probe requires Linux native ELF");
    std::process::ExitCode::from(2)
}
