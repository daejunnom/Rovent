//! The ordinary UCI owner is shared by V5 native and labelled metadata probes.
//! A probe obtains legal stop moves from Rules and constructs no CPU engine.
use crate::engine::{self, EngineProcess, EngineSettings, OwnerRegistry, ProcessClock};
use crate::search_driver::{
    DriverWork, SearchKind, SearchSessionContext, SearchSessionDriver, SearchSessionFailure,
    SearchSessionReport,
};
use crate::{EngineIdentity, forward_lines};
use rz_contracts::Digest;
use rz_position::{BoardMove, Position};
use std::{
    io,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

pub fn serve_driver(
    driver: Arc<dyn SearchSessionDriver>,
    clock: ProcessClock,
    owners: Arc<OwnerRegistry>,
    settings: EngineSettings,
) -> Result<(), engine::EngineError> {
    let process = EngineProcess::with_search_driver(Arc::clone(&driver), owners, clock);
    let (sender, events) = engine::event_channel();
    let input = sender.clone();
    // Only stdin may block here. Search, stop and physical drain have independent
    // owners; EOF/quit is processed by the existing bounded protocol engine.
    thread::Builder::new()
        .name("pals-adaptive-stdin".into())
        .spawn(move || {
            let _ = forward_lines(
                &mut io::stdin().lock(),
                &input,
                settings.parser.max_line_bytes,
            );
        })
        .map_err(|_| stdin_spawn_failure(&driver, settings.shutdown_limit))?;
    engine::serve(
        events,
        sender,
        &mut io::stdout().lock(),
        // Keep only the stderr handle for the session. A search worker can
        // emit its lifecycle row before its Complete event, so retaining
        // StderrLock here would block the worker until UCI shutdown.
        &mut io::stderr(),
        process,
        settings,
    )
}

/// No engine event loop or search can have started on this path. Try to close
/// the original driver within its registered shutdown limit; retain the spawn
/// failure as the primary error and never promote unknown completion to success.
fn stdin_spawn_failure(
    driver: &Arc<dyn SearchSessionDriver>,
    shutdown_limit: Duration,
) -> engine::EngineError {
    use crate::search_driver::DriverPhysicalCompletion;
    let primary = engine::EngineError::Contract(rz_contracts::ContractError::new(
        rz_contracts::ErrorCode::BackendFailure,
        rz_contracts::Stage::Admission,
        "adaptive stdin owner thread creation failed",
    ));
    let cleanup = match Instant::now().checked_add(shutdown_limit) {
        None => Some(engine::EngineError::DrainTimeout),
        Some(until) => match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            driver.finish_session(until)
        })) {
            Ok(Ok(Some(DriverPhysicalCompletion::Unknown))) => {
                Some(engine::EngineError::SearchDriverClose(
                    SearchSessionFailure::physical_completion_unknown(),
                ))
            }
            Ok(Err(error)) => Some(engine::EngineError::SearchDriverClose(error)),
            Err(_) => Some(engine::EngineError::WorkerPanic),
            Ok(Ok(_)) => None,
        },
    };
    match cleanup {
        Some(cleanup) => engine::EngineError::Cleanup {
            primary: Box::new(primary),
            cleanup: vec![cleanup],
        },
        None => primary,
    }
}

pub struct RulesOnlyPreflightDriver {
    pub identity: EngineIdentity,
    pub implementation: Digest,
}
impl SearchSessionDriver for RulesOnlyPreflightDriver {
    fn kind(&self) -> SearchKind {
        SearchKind::Pals
    }
    fn identity(&self) -> EngineIdentity {
        self.identity.clone()
    }
    fn implementation(&self) -> Digest {
        self.implementation
    }
    fn run(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        _: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        if !context
            .accepts()
            .map_err(|e| SearchSessionFailure::debug("PreflightAuthority", &e))?
        {
            return Err(SearchSessionFailure::debug(
                "PreflightExpired",
                &"Rules-only probe revoked",
            ));
        }
        let best_move = position.legal_moves().into_iter().next();
        Ok(SearchSessionReport {
            mate_strategy: None,
            best_move,
            work: DriverWork::Pals {
                rounds: 0,
                cpu_nodes: 0,
                completed_tasks: 0,
                consumed_role_outputs: 0,
                retained_situations: 0,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{BufRead, BufReader, Read, Write},
        process::{Child, Command, Stdio},
        sync::{
            atomic::{AtomicUsize, Ordering},
            mpsc::{self, Receiver},
        },
        time::{Duration, Instant},
    };

    const WORKER_MARKER: &str = "rules-only-worker-stderr-before-report";
    const CHILD_TEST: &str = "pals_adaptive_bootstrap::protocol::tests::worker_stderr_stdio_child";

    #[test]
    fn failed_stdin_spawn_closes_the_original_driver_and_preserves_unknown() {
        use crate::search_driver::DriverPhysicalCompletion;
        use std::sync::Mutex;

        struct ClosingDriver {
            result: DriverPhysicalCompletion,
            calls: AtomicUsize,
            deadline: Mutex<Option<Instant>>,
        }
        impl SearchSessionDriver for ClosingDriver {
            fn kind(&self) -> SearchKind {
                SearchKind::Pals
            }
            fn identity(&self) -> EngineIdentity {
                EngineIdentity {
                    name: "early-close fixture".into(),
                    author: "RoveZero".into(),
                }
            }
            fn implementation(&self) -> Digest {
                Digest([7; 32])
            }
            fn run(
                &self,
                _: &Position,
                _: &SearchSessionContext,
                _: &mut dyn FnMut(BoardMove),
            ) -> Result<SearchSessionReport, SearchSessionFailure> {
                unreachable!("stdin failure happens before UCI search")
            }
            fn finish_session(
                &self,
                until: Instant,
            ) -> Result<Option<DriverPhysicalCompletion>, SearchSessionFailure> {
                self.calls.fetch_add(1, Ordering::AcqRel);
                *self.deadline.lock().unwrap() = Some(until);
                Ok(Some(self.result))
            }
        }
        for expected in [
            DriverPhysicalCompletion::Confirmed,
            DriverPhysicalCompletion::Unknown,
        ] {
            let original = Arc::new(ClosingDriver {
                result: expected,
                calls: AtomicUsize::new(0),
                deadline: Mutex::new(None),
            });
            let driver: Arc<dyn SearchSessionDriver> = original.clone();
            let before = Instant::now();
            let error = stdin_spawn_failure(&driver, Duration::from_secs(2));
            let after = Instant::now();
            let deadline = original.deadline.lock().unwrap().unwrap();
            assert!(deadline >= before + Duration::from_secs(2));
            assert!(deadline <= after + Duration::from_secs(2));
            assert_eq!(original.calls.load(Ordering::Acquire), 1);
            match expected {
                DriverPhysicalCompletion::Confirmed => {
                    assert!(matches!(error, engine::EngineError::Contract(_)))
                }
                DriverPhysicalCompletion::Unknown => assert!(matches!(error,
                    engine::EngineError::Cleanup { primary, cleanup }
                        if matches!(primary.as_ref(), engine::EngineError::Contract(_))
                            && matches!(cleanup.as_slice(), [engine::EngineError::SearchDriverClose(_)]))),
            }
        }
    }

    // This is a Rules-only fixture, not a native/lifecycle receipt producer.
    // The real worker writes to the same process-wide stderr that the ordinary
    // UCI owner uses, then returns through actual WorkerCompletion/Complete.
    struct StderrWorkerDriver {
        inner: RulesOnlyPreflightDriver,
        writes: Arc<AtomicUsize>,
        closes: Arc<AtomicUsize>,
    }
    impl SearchSessionDriver for StderrWorkerDriver {
        fn kind(&self) -> SearchKind {
            self.inner.kind()
        }
        fn identity(&self) -> EngineIdentity {
            self.inner.identity()
        }
        fn implementation(&self) -> Digest {
            self.inner.implementation()
        }
        fn run(
            &self,
            position: &Position,
            context: &SearchSessionContext,
            progress: &mut dyn FnMut(BoardMove),
        ) -> Result<SearchSessionReport, SearchSessionFailure> {
            writeln!(io::stderr().lock(), "{WORKER_MARKER}")
                .map_err(|error| SearchSessionFailure::debug("FixtureStderr", &error))?;
            self.writes.fetch_add(1, Ordering::AcqRel);
            self.inner.run(position, context, progress)
        }
        fn finish_session(
            &self,
            _: Instant,
        ) -> Result<Option<crate::search_driver::DriverPhysicalCompletion>, SearchSessionFailure>
        {
            self.closes.fetch_add(1, Ordering::AcqRel);
            // No original native physical receipt exists in this mock fixture.
            Ok(None)
        }
    }

    #[test]
    #[ignore = "private stdio child, selected explicitly by the bounded parent test"]
    fn worker_stderr_stdio_child() {
        let writes = Arc::new(AtomicUsize::new(0));
        let closes = Arc::new(AtomicUsize::new(0));
        let driver = StderrWorkerDriver {
            inner: RulesOnlyPreflightDriver {
                identity: EngineIdentity {
                    name: "Rules-only stderr worker fixture".into(),
                    author: "RoveZero".into(),
                },
                implementation: Digest([0x37; 32]),
            },
            writes: Arc::clone(&writes),
            closes: Arc::clone(&closes),
        };
        serve_driver(
            Arc::new(driver),
            ProcessClock::new(rz_contracts::ProcessEpoch(327)),
            Arc::new(OwnerRegistry::default()),
            EngineSettings {
                shutdown_limit: Duration::from_secs(2),
                ..EngineSettings::default()
            },
        )
        .expect("actual shared stdio owner and worker must close");
        assert_eq!(writes.load(Ordering::Acquire), 1);
        assert_eq!(closes.load(Ordering::Acquire), 1);
    }

    struct StdioChild {
        process: Child,
        lines: Receiver<String>,
        stderr: Option<thread::JoinHandle<Vec<u8>>>,
    }
    impl Drop for StdioChild {
        fn drop(&mut self) {
            // This child has no model, external engine or subprocess. A failed
            // assertion must still reap all of its in-process fixture threads.
            if self.process.try_wait().ok().flatten().is_none() {
                let _ = self.process.kill();
                let _ = self.process.wait();
            }
        }
    }
    impl StdioChild {
        fn send(&mut self, command: &str) {
            let input = self.process.stdin.as_mut().expect("owned fixture stdin");
            writeln!(input, "{command}").unwrap();
            input.flush().unwrap();
        }
        fn wait_line(&self, prefix: &str) -> String {
            let until = Instant::now() + Duration::from_secs(5);
            loop {
                let line = self
                    .lines
                    .recv_timeout(until.saturating_duration_since(Instant::now()))
                    .expect("actual stdio/worker notification must precede quit");
                assert!(line.len() <= 4096, "fixture stdout line bound");
                if line.starts_with(prefix) {
                    return line;
                }
            }
        }
    }

    #[test]
    fn shared_stdio_worker_stderr_reaches_bestmove_before_quit() {
        let mut process = Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", CHILD_TEST, "--nocapture"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let output = process.stdout.take().unwrap();
        let (sender, lines) = mpsc::sync_channel(32);
        thread::spawn(move || {
            for line in BufReader::new(output.take(64 * 1024)).lines().take(128) {
                let Ok(line) = line else { break };
                if line.len() > 4096 || sender.send(line).is_err() {
                    break;
                }
            }
        });
        let error = process.stderr.take().unwrap();
        let stderr = thread::spawn(move || {
            let mut bytes = Vec::new();
            error.take(16 * 1024 + 1).read_to_end(&mut bytes).unwrap();
            bytes
        });
        let mut child = StdioChild {
            process,
            lines,
            stderr: Some(stderr),
        };
        child.send("uci");
        child.wait_line("uciok");
        child.send("isready");
        child.wait_line("readyok");
        child.send("position startpos");
        child.send("go nodes 1");
        // Real completion must be delivered while the stdio owner is still
        // serving. Sending quit first would conceal the original lock cycle.
        let bestmove = child.wait_line("bestmove ");
        let movement = BoardMove::from_uci(bestmove.trim_start_matches("bestmove ")).unwrap();
        assert!(
            Position::startpos()
                .ordered_legal_moves()
                .moves()
                .contains(&movement)
        );
        child.send("quit");
        drop(child.process.stdin.take());
        let until = Instant::now() + Duration::from_secs(5);
        let status = loop {
            if let Some(status) = child.process.try_wait().unwrap() {
                break status;
            }
            assert!(
                Instant::now() < until,
                "original fixture process exit bound"
            );
            thread::sleep(Duration::from_millis(1));
        };
        assert!(status.success());
        let stderr = child.stderr.take().unwrap().join().unwrap();
        assert!(stderr.len() <= 16 * 1024, "fixture stderr bound");
        let text = String::from_utf8(stderr).unwrap();
        assert_eq!(
            text.lines().filter(|line| *line == WORKER_MARKER).count(),
            1
        );
        assert!(!text.contains("DrainTimeout"));
        assert!(!text.contains("PhysicalFenceFailure"));
    }
}
