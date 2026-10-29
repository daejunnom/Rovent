//! Opt-in integration check using the real pinned V2 CPU launch and owner.
//! Run only under the project's managed process/resource supervisor. The JSON
//! config path comes from ROVENT_V2_CPU_CONTENTION_CONFIG and supplies all
//! source/asset pins and launch arguments; no machine-specific pins are here.
#![cfg(all(
    target_os = "linux",
    feature = "pals-collection-onnx",
    feature = "test-fixtures"
))]

use cap_std::{ambient_authority, fs::Dir};
use rz_arena::rovent_collect::launch::{
    load_rovent_collection_lock_v2_plan, run_rovent_pc_collection_lock_v2_with_test_owner,
};
use rz_arena::rovent_collect::{RoventCollectionOwnerLockV2, RoventPublicationFile};
use serde::Deserialize;
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::Path,
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    input_directory: String,
    working_directory: String,
    stdout_report: String,
    stderr_report: String,
    plan_pin: RoventPublicationFile,
    engine_arguments: Vec<String>,
    timeout_seconds: u64,
}

const CHILD: &str = "ROVENT_V2_CPU_CONTENTION_CHILD";
const CONFIG: &str = "ROVENT_V2_CPU_CONTENTION_CONFIG";
const DONE: &str = "ROVENT_V2_CPU_CONTENTION_NATIVE_COMPLETED";

#[derive(Clone, Copy)]
enum Event {
    UciOk,
    ReadyOk,
    NativeCompleted,
    BestMove,
    OutputFailure,
}

fn scan_output(
    mut source: impl Read,
    mut report: File,
    stdout: bool,
    sender: mpsc::SyncSender<Event>,
    failed: Arc<AtomicBool>,
) {
    const MAX_LINE: usize = 64 * 1024;
    const MAX_TOTAL: usize = 8 * 1024 * 1024;
    let mut line = Vec::new();
    let mut total = 0usize;
    let mut chunk = [0u8; 4096];
    let send = |event| {
        if matches!(event, Event::OutputFailure) {
            failed.store(true, Ordering::Release);
        }
        if sender.try_send(event).is_err() {
            failed.store(true, Ordering::Release);
        }
    };
    loop {
        let count = match source.read(&mut chunk) {
            Ok(0) => break,
            Ok(count) => count,
            Err(_) => {
                send(Event::OutputFailure);
                break;
            }
        };
        total += count;
        if total > MAX_TOTAL {
            send(Event::OutputFailure);
            break;
        }
        if report.write_all(&chunk[..count]).is_err() {
            send(Event::OutputFailure);
            break;
        }
        for byte in &chunk[..count] {
            if *byte != b'\n' {
                if line.len() >= MAX_LINE {
                    send(Event::OutputFailure);
                    return;
                }
                line.push(*byte);
                continue;
            }
            let value = line.strip_suffix(b"\r").unwrap_or(&line);
            if stdout && value == b"uciok" {
                send(Event::UciOk);
            }
            if stdout && value == b"readyok" {
                send(Event::ReadyOk);
            }
            if stdout && value.starts_with(b"bestmove ") {
                send(Event::BestMove);
            }
            if !stdout
                && value
                    .windows(DONE.len())
                    .any(|window| window == DONE.as_bytes())
            {
                send(Event::NativeCompleted);
            }
            line.clear();
        }
    }
}

struct ChildGuard(Child, bool);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        if !self.1 {
            // The child inherits the outer managed group. The outer supervisor
            // owns group drain; this guard stops and reaps only its own child.
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}

#[test]
#[ignore = "requires explicitly registered V2 CPU assets and a managed finite execution scope"]
fn real_v2_capture_contention() {
    let config_path =
        std::env::var_os(CONFIG).expect("set the externally registered CPU contention config path");
    assert!(
        Path::new(&config_path).is_absolute(),
        "test config path must be absolute"
    );
    let config: Config =
        serde_json::from_slice(&std::fs::read(config_path).expect("read CPU contention config"))
            .expect("parse CPU contention config");
    assert!(
        (10..=180).contains(&config.timeout_seconds),
        "bounded test timeout required"
    );
    assert!(
        Path::new(&config.input_directory).is_absolute()
            && Path::new(&config.working_directory).is_absolute(),
        "registered input and working directories must be absolute"
    );
    assert!(
        Path::new(&config.stdout_report).is_absolute()
            && Path::new(&config.stderr_report).is_absolute()
            && config.stdout_report != config.stderr_report,
        "distinct managed output report paths required"
    );
    if std::env::var_os(CHILD).is_some() {
        run_child(config);
        return;
    }
    let deadline = Instant::now() + Duration::from_secs(config.timeout_seconds);
    let stdout_report = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&config.stdout_report)
        .expect("create registered stdout report");
    let stderr_report = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&config.stderr_report)
        .expect("create registered stderr report");
    let child = Command::new(std::env::current_exe().expect("test executable"))
        .args([
            "--exact",
            "real_v2_capture_contention",
            "--ignored",
            "--nocapture",
        ])
        .current_dir(&config.working_directory)
        .env(CHILD, "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start bounded integration child");
    let mut child = ChildGuard(child, false);
    let mut stdin = child.0.stdin.take().expect("child stdin");
    let stdout = child.0.stdout.take().expect("child stdout");
    let stderr = child.0.stderr.take().expect("child stderr");
    let (sender, receiver) = mpsc::sync_channel(16);
    let stdout_sender = sender.clone();
    let output_failed = Arc::new(AtomicBool::new(false));
    let stdout_failed = Arc::clone(&output_failed);
    let stderr_failed = Arc::clone(&output_failed);
    let (closed_tx, closed_rx) = mpsc::sync_channel(2);
    let stdout_closed = closed_tx.clone();
    std::thread::spawn(move || {
        scan_output(stdout, stdout_report, true, stdout_sender, stdout_failed);
        let _ = stdout_closed.send(());
    });
    std::thread::spawn(move || {
        scan_output(stderr, stderr_report, false, sender, stderr_failed);
        let _ = closed_tx.send(());
    });
    writeln!(stdin, "uci").expect("send UCI handshake");
    stdin.flush().expect("flush UCI handshake");
    assert!(
        matches!(
            receiver.recv_timeout(deadline.saturating_duration_since(Instant::now())),
            Ok(Event::UciOk)
        ),
        "uciok was not observed"
    );
    writeln!(stdin, "isready").expect("send readiness request");
    stdin.flush().expect("flush readiness request");
    assert!(
        matches!(
            receiver.recv_timeout(deadline.saturating_duration_since(Instant::now())),
            Ok(Event::ReadyOk)
        ),
        "readyok was not observed"
    );
    writeln!(stdin, "position startpos\ngo movetime 60000").expect("send bounded UCI search");
    stdin.flush().expect("flush UCI search");
    let mut native = false;
    let mut bestmove = false;
    while Instant::now() < deadline && !(native && bestmove) {
        match receiver.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(Event::NativeCompleted) => {
                native = true;
                let _ = writeln!(stdin, "stop");
                let _ = stdin.flush();
            }
            Ok(Event::BestMove) => bestmove = true,
            Ok(Event::UciOk | Event::ReadyOk) => {}
            Ok(Event::OutputFailure) => break,
            Err(_) => break,
        }
    }
    if native && bestmove {
        let _ = writeln!(stdin, "quit");
        let _ = stdin.flush();
    }
    drop(stdin);
    let exit = loop {
        if let Some(status) = child.0.try_wait().expect("observe integration child") {
            child.1 = true;
            break Some(status);
        }
        if Instant::now() >= deadline {
            break None;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let reports_closed = (0..2).all(|_| {
        closed_rx
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .is_ok()
    });
    assert!(
        native,
        "actual staged ticket did not reach confirmed native completion"
    );
    assert!(bestmove, "actual UCI bestmove was not observed");
    assert!(
        reports_closed,
        "bounded stdout/stderr report writers did not close"
    );
    assert!(
        !output_failed.load(Ordering::Acquire),
        "bounded output reader or event channel failed"
    );
    assert!(
        exit.is_some_and(|status| status.success()),
        "integration child failed or timed out"
    );
}

fn run_child(config: Config) {
    let input = Dir::open_ambient_dir(&config.input_directory, ambient_authority())
        .expect("open registered input directory");
    let plan = load_rovent_collection_lock_v2_plan(&input, &config.plan_pin)
        .expect("verify registered V2 plan");
    let deadline = Instant::now() + Duration::from_secs(config.timeout_seconds.saturating_sub(2));
    let (observed_tx, observed_rx) = mpsc::sync_channel(1);
    let run = run_rovent_pc_collection_lock_v2_with_test_owner(
        config.engine_arguments,
        plan,
        &input,
        deadline,
        deadline,
        move |owner| {
            let owner = owner.clone();
            let (armed_tx, armed_rx) = mpsc::sync_channel(1);
            std::thread::spawn(move || {
                let result = exercise_real_contention(&owner, deadline, armed_tx);
                let _ = observed_tx.send(result);
            });
            armed_rx
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .map_err(|error| {
                    rz_arena::ArenaError::Invalid(format!(
                        "test registration lock not armed: {error}"
                    ))
                })?;
            Ok(())
        },
    )
    .expect("run original V2 collection launch");
    let result = observed_rx
        .recv_timeout(deadline.saturating_duration_since(Instant::now()))
        .expect("contention observer finished");
    result.expect("actual contention and completion checks");
    // Output is an observation only; retain the original run owner until after
    // the checks and do not treat this as publication or whole-plan success.
    let observation = run.observation();
    assert!(
        observation
            .get("native_error")
            .is_some_and(serde_json::Value::is_null),
        "V2 launch reported a native failure"
    );
}

fn exercise_real_contention(
    owner: &RoventCollectionOwnerLockV2,
    deadline: Instant,
    armed: mpsc::SyncSender<()>,
) -> Result<(), String> {
    let original = owner
        .test_with_capture_registration_held(|| {
            armed.send(()).map_err(|error| error.to_string())?;
            loop {
                if Instant::now() >= deadline {
                    return Err("no staged input before deadline".to_owned());
                }
                let staged = owner.test_staged_snapshot().map_err(|e| e.to_string())?;
                if let Some((ticket, sequence)) = staged.first().copied() {
                    std::thread::sleep(Duration::from_millis(20));
                    if owner
                        .test_staged_sequence(&ticket)
                        .map_err(|e| e.to_string())?
                        != Some(sequence)
                    {
                        return Err("original staged sequence changed while mutex was held".into());
                    }
                    if owner
                        .test_completion(&ticket)
                        .map_err(|e| e.to_string())?
                        .is_some()
                    {
                        return Err(
                            "native completion observed before registration lock release".into(),
                        );
                    }
                    if owner
                        .test_native_started(&ticket)
                        .map_err(|e| e.to_string())?
                    {
                        return Err("native start observed before registration lock release".into());
                    }
                    return Ok((ticket, sequence));
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        })
        .map_err(|e| e.to_string())??;
    loop {
        if Instant::now() >= deadline {
            return Err("native completion not observed before deadline".into());
        }
        if let Some((sequence, physical, successful)) = owner
            .test_completion(&original.0)
            .map_err(|e| e.to_string())?
        {
            if sequence != original.1 || !physical || !successful {
                return Err("original reservation changed or native completion failed or is physically unknown".into());
            }
            eprintln!("{DONE}");
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}
