//! CPU-only checks for the opt-in NoGame retained process owner.
//! These tests do not establish native model or GPU completion.

use super::smoke_reobserve::{self, ChildObservation, PipeObservation, SmokeReobserveError};
use super::*;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

const MODE: &str = ".rz-retained-test-mode";
const READY: &str = ".rz-retained-test-ready";
const RELEASE: &str = ".rz-retained-test-release";
const CHILD: &str = "process::smoke_reobserve_tests::retained_mock_child";

// Never drop an unresolved test child or its original pipes on a failure.
fn preserve_unknown(owner: SpawnCustody) {
    static UNKNOWN: OnceLock<Mutex<Vec<SpawnCustody>>> = OnceLock::new();
    UNKNOWN
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .push(owner);
}

struct TestOwner {
    custody: Option<SpawnCustody>,
    root: PathBuf,
}

impl TestOwner {
    fn new(mode: &[u8]) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "rz-retained-owner-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join(MODE), mode).unwrap();
        Self {
            custody: Some(SpawnCustody::default()),
            root,
        }
    }
}

impl Drop for TestOwner {
    fn drop(&mut self) {
        let Some(mut owner) = self.custody.take() else {
            return;
        };
        let mut reaped = owner.child.is_none();
        if let Some(child) = owner.child.as_mut() {
            // This fixture alone owns its direct child; it creates no children.
            // Do not infer a process group or GPU completion from this cleanup.
            match child.try_wait() {
                Ok(Some(_)) => reaped = true,
                Ok(None) => {
                    if child.kill().is_ok() {
                        let end = Instant::now() + Duration::from_secs(1);
                        while Instant::now() < end {
                            match child.try_wait() {
                                Ok(Some(_)) => {
                                    reaped = true;
                                    break;
                                }
                                Ok(None) => std::thread::sleep(Duration::from_millis(5)),
                                Err(_) => break,
                            }
                        }
                    }
                }
                Err(_) => {}
            }
        }
        if reaped {
            let _ = std::fs::remove_file(self.root.join(RELEASE));
            let _ = std::fs::remove_file(self.root.join(READY));
            let _ = std::fs::remove_file(self.root.join(MODE));
            let _ = std::fs::remove_dir(&self.root);
        } else {
            preserve_unknown(owner);
        }
    }
}

#[test]
fn retained_mock_child() {
    let Ok(mode) = std::fs::read(MODE) else {
        return;
    };
    let mut input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();
    let mut line = String::new();
    input.read_line(&mut line).unwrap();
    assert_eq!(line, "uci\n");
    output.write_all(b"\nuciok\n").unwrap();
    output.flush().unwrap();
    line.clear();
    input.read_line(&mut line).unwrap();
    assert_eq!(line, "isready\n");
    if mode == b"cap" {
        for _ in 0..1024 {
            output.write_all(&[b'x'; 1024]).unwrap();
            output.flush().unwrap();
        }
    } else {
        output.write_all(b"info string retained-stdout\n").unwrap();
        output.flush().unwrap();
        let mut error = std::io::stderr().lock();
        error.write_all(b"retained-stderr\n").unwrap();
        error.flush().unwrap();
        std::fs::write(READY, b"ready").unwrap();
        let end = Instant::now() + Duration::from_secs(30);
        while Instant::now() < end && !std::path::Path::new(RELEASE).is_file() {
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

fn limits(cap: u64) -> ProcessLimits {
    ProcessLimits {
        wall_ms: 5000,
        shutdown_grace_ms: 250,
        max_output_bytes: cap,
        max_child_processes: 1,
    }
}

fn watch() -> OwnedArtifactTreeWatch {
    OwnedArtifactTreeWatch {
        max_total_bytes: 4096,
        max_file_bytes: 1024,
        max_files: 4,
        max_depth: 0,
    }
}

fn window(limits: ProcessLimits) -> OriginalProcessWindow {
    let now = Instant::now();
    OriginalProcessWindow::new(
        now,
        now + Duration::from_millis(limits.wall_ms),
        now + Duration::from_millis(limits.wall_ms + 2 * limits.shutdown_grace_ms),
    )
    .unwrap()
}

#[test]
fn retaining_cancel_keeps_original_child_pipes_and_nonreaping_observations() {
    let mut owner = TestOwner::new(b"cancel");
    let program = File::open(std::env::current_exe().unwrap()).unwrap();
    let directory = File::open(&owner.root).unwrap();
    let ready = owner.root.join(READY);
    let limits = limits(256 * 1024);
    let original = window(limits);
    let mut observation = OriginalSmokeUciObservation::new(original);
    let cancel = AtomicBool::new(false);
    let mut control_pid = None;
    let mut callback_pid = None;
    let mut control = |custody: &SpawnCustody, _: Instant| {
        if ready.is_file() {
            control_pid = custody.child().map(|child| child.id());
            cancel.store(true, Ordering::Release);
        }
        Ok(())
    };
    let mut ack = |custody: &SpawnCustody, until: Instant| {
        assert!(Instant::now() < until);
        callback_pid = custody.child().map(|child| child.id());
        Ok(PreSignalProgress::Acknowledged)
    };
    let outcome = supervise_smoke_uci_retaining_in_original_window(
        &program,
        &["--exact".into(), CHILD.into(), "--nocapture".into()],
        &directory,
        limits,
        Some(&cancel),
        &watch(),
        original,
        owner.custody.as_mut().unwrap(),
        &mut control,
        None,
        Some(&mut ack),
        &mut observation,
    )
    .unwrap();
    let receipt = match outcome {
        RetainingSmokeOutcome::Retained(receipt) => receipt,
        RetainingSmokeOutcome::Finished(_) => panic!("sleeping child unexpectedly finished"),
    };
    assert_eq!(receipt.stop, ProcessStop::Cancelled);
    assert_eq!(receipt.group_cleanup, CleanupStatus::Unverified);
    assert_eq!(control_pid, Some(receipt.pid));
    assert_eq!(callback_pid, control_pid);
    assert!(observation.pre_signal().acknowledged_at.is_some());
    assert!(observation.uci_sent() && observation.uciok_observed() && observation.isready_sent());
    assert!(!observation.readyok_observed() && !observation.quit_sent());
    assert!(
        observation
            .buffered_stdout()
            .windows(b"info string retained-stdout\n".len())
            .any(|part| part == b"info string retained-stdout\n")
    );
    assert!(
        observation
            .buffered_stderr()
            .windows(b"retained-stderr\n".len())
            .any(|part| part == b"retained-stderr\n")
    );
    let custody = owner.custody.as_mut().unwrap();
    assert_eq!(custody.child().map(|child| child.id()), Some(receipt.pid));
    assert!(
        custody
            .retained_smoke
            .as_ref()
            .is_some_and(|pipes| pipes.stdin.is_some()
                && pipes.stdout.is_some()
                && pipes.stderr.is_some())
    );
    let first = smoke_reobserve::observe_once(
        custody,
        &mut observation,
        Instant::now() + Duration::from_millis(100),
    )
    .unwrap();
    let second = smoke_reobserve::observe_once(
        custody,
        &mut observation,
        Instant::now() + Duration::from_millis(100),
    )
    .unwrap();
    assert_eq!(first.child, ChildObservation::Alive);
    assert_eq!(second.child, ChildObservation::Alive);
    assert_eq!(custody.child().map(|child| child.id()), Some(receipt.pid));
    // A test-only marker lets the same child exit without a new UCI command.
    // Repeated terminal waitid observations must not reap the original child.
    std::fs::write(owner.root.join(RELEASE), b"release").unwrap();
    let end = Instant::now() + Duration::from_secs(2);
    let terminal = loop {
        let tick = smoke_reobserve::observe_once(custody, &mut observation, end).unwrap();
        if matches!(tick.child, ChildObservation::Exited(_)) {
            break tick.child;
        }
        assert!(
            Instant::now() < end,
            "mock child did not exit within test bound"
        );
        std::thread::sleep(Duration::from_millis(5));
    };
    let again = smoke_reobserve::observe_once(
        custody,
        &mut observation,
        Instant::now() + Duration::from_millis(100),
    )
    .unwrap();
    assert_eq!(again.child, terminal);
    assert_eq!(custody.child().map(|child| child.id()), Some(receipt.pid));
    let mut unrelated = OriginalSmokeUciObservation::new(original);
    assert_eq!(
        smoke_reobserve::observe_once(
            custody,
            &mut unrelated,
            Instant::now() + Duration::from_millis(100)
        ),
        Err(SmokeReobserveError::NotRetained)
    );
    // ACK is the test callback's observation; it is not a GPU fence or IPC acceptance proof.
}

#[test]
fn retaining_output_limit_keeps_original_bounded_bytes_and_child() {
    let mut owner = TestOwner::new(b"cap");
    let program = File::open(std::env::current_exe().unwrap()).unwrap();
    let directory = File::open(&owner.root).unwrap();
    let limits = limits(512);
    let original = window(limits);
    let mut observation = OriginalSmokeUciObservation::new(original);
    let mut control = |_: &SpawnCustody, _: Instant| Ok(());
    let outcome = supervise_smoke_uci_retaining_in_original_window(
        &program,
        &["--exact".into(), CHILD.into(), "--nocapture".into()],
        &directory,
        limits,
        None,
        &watch(),
        original,
        owner.custody.as_mut().unwrap(),
        &mut control,
        None,
        None,
        &mut observation,
    )
    .unwrap();
    let receipt = match outcome {
        RetainingSmokeOutcome::Retained(receipt) => receipt,
        RetainingSmokeOutcome::Finished(_) => panic!("flooding child unexpectedly finished"),
    };
    assert_eq!(receipt.stop, ProcessStop::OutputLimit);
    let total = observation.buffered_stdout().len() + observation.buffered_stderr().len();
    assert!(total <= limits.max_output_bytes as usize);
    let custody = owner.custody.as_mut().unwrap();
    assert_eq!(custody.child().map(|child| child.id()), Some(receipt.pid));
    let readback = smoke_reobserve::observe_once(
        custody,
        &mut observation,
        Instant::now() + Duration::from_millis(100),
    )
    .unwrap();
    assert!(
        matches!(readback.stdout, PipeObservation::OutputCap)
            || matches!(readback.stderr, PipeObservation::OutputCap)
    );
    assert!(
        observation.buffered_stdout().len() + observation.buffered_stderr().len()
            <= limits.max_output_bytes as usize
    );
}
