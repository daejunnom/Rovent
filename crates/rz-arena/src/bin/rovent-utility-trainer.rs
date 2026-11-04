//! Utility-only product trainer entry point. The library owns original cohort,
//! source/runtime, resource, tensor and physical-retirement admission.

#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
fn main() -> std::process::ExitCode {
    // Parsing, cancellation-hook registration and admission all use this one
    // original clock. An error never retries with a fresh execution allowance.
    let original_started = std::time::Instant::now();
    match supported::run(original_started) {
        Ok(true) => std::process::ExitCode::SUCCESS,
        Ok(false) => std::process::ExitCode::FAILURE,
        Err(error) => {
            use std::io::Write;
            let detail: String = error
                .to_string()
                .chars()
                .take(2048)
                .map(|character| {
                    if character.is_control() {
                        ' '
                    } else {
                        character
                    }
                })
                .collect();
            let _ = writeln!(
                std::io::stderr().lock(),
                "rovent-utility-trainer halted: {detail}"
            );
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(not(all(target_os = "linux", feature = "pals-collection-onnx")))]
fn main() -> std::process::ExitCode {
    use std::io::Write;
    let _ = writeln!(
        std::io::stderr().lock(),
        "rovent-utility-trainer unsupported: requires Linux and the pals-collection-onnx feature"
    );
    std::process::ExitCode::FAILURE
}

#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
mod supported {
    use rz_arena::rovent_training_process::utility::{
        cli::{ROVENT_UTILITY_CLI_USAGE, RoventUtilityCliParse, parse_rovent_utility_cli},
        runtime::run_rovent_utility_from_cli,
    };
    use std::{
        error::Error,
        io::{self, Write},
        sync::{Arc, atomic::AtomicBool},
        time::Instant,
    };

    const MAX_REPORT_BYTES: u64 = 64 << 10;

    /// Only cancellation is published by signal handlers. The runtime closes
    /// admission, drains its physical work and joins its own child processes.
    struct CancellationHooks(Vec<signal_hook::SigId>);
    impl CancellationHooks {
        fn register(cancel: &Arc<AtomicBool>) -> io::Result<Self> {
            let mut hooks = Self(Vec::with_capacity(2));
            for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
                hooks
                    .0
                    .push(signal_hook::flag::register(signal, Arc::clone(cancel))?);
            }
            Ok(hooks)
        }
    }
    impl Drop for CancellationHooks {
        fn drop(&mut self) {
            for hook in self.0.drain(..) {
                signal_hook::low_level::unregister(hook);
            }
        }
    }

    pub(super) fn run(original_started: Instant) -> Result<bool, Box<dyn Error>> {
        let cancel = Arc::new(AtomicBool::new(false));
        let _hooks = CancellationHooks::register(&cancel)?;
        let command = match parse_rovent_utility_cli(std::env::args_os().skip(1))? {
            RoventUtilityCliParse::Help => {
                let mut stdout = io::stdout().lock();
                stdout.write_all(ROVENT_UTILITY_CLI_USAGE.as_bytes())?;
                stdout.write_all(b"\n")?;
                stdout.flush()?;
                return Ok(true);
            }
            RoventUtilityCliParse::Command(command) => command,
        };
        let maximum_report_bytes = command.scope.max_stdio_bytes.min(MAX_REPORT_BYTES);
        let request = command.anchor_original_window(original_started)?;
        request.ensure_admission_open()?;
        let outcome = run_rovent_utility_from_cli(request, cancel)?;

        // The concrete owner has finished bounded cleanup/retirement or kept
        // sticky quarantine custody before returning. Buffer the complete small
        // redacted observation so overflow cannot print partial JSON success.
        let mut report = BoundedReport {
            bytes: Vec::new(),
            maximum: maximum_report_bytes,
        };
        serde_json::to_writer(&mut report, &outcome.observation())?;
        report.write_all(b"\n")?;
        let mut stdout = io::stdout().lock();
        stdout.write_all(&report.bytes)?;
        stdout.flush()?;
        // Preserved post-spawn failures may be Result::Ok. Their typed outcome
        // still determines a nonzero process exit; no execution is repeated.
        Ok(outcome.success())
    }

    struct BoundedReport {
        bytes: Vec<u8>,
        maximum: u64,
    }
    impl Write for BoundedReport {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            let next = u64::try_from(self.bytes.len())
                .ok()
                .and_then(|current| current.checked_add(bytes.len() as u64))
                .filter(|next| *next <= self.maximum)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::FileTooLarge, "utility CLI report limit")
                })?;
            self.bytes
                .try_reserve(bytes.len())
                .map_err(|_| io::Error::other("utility CLI report allocation failed"))?;
            self.bytes.extend_from_slice(bytes);
            debug_assert_eq!(next, self.bytes.len() as u64);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
}
