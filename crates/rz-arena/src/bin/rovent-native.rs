//! Actual native UCI entry with the shared delegated Linux resource owner.
#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
fn main() -> std::process::ExitCode {
    use rz_arena::rovent_resource_scope::launch::{
        ROVENT_RESOURCE_USAGE, RoventControlledRunFailure, RoventNativeResourceCli,
        parse_native_resource_cli, run_native_with_resources_v3,
        run_native_with_resources_v3_control, run_native_with_resources_versioned,
    };
    use std::io::Write;
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    // Optional diagnostic FD must be sealed before argument parsing, provider
    // initialization, or any product thread/child can be created. Failure is
    // diagnostic only; the later observer gate remains disabled.
    // SAFETY: at this process entry no product thread or Rust FD owner has
    // been created. A caller opting into this path must supply a descriptor
    // inherited exclusively for this child, as declared by the CLI option.
    unsafe { rz_uci::rovent_bootstrap::launch::seal_go_observer_at_process_entry(&arguments) };
    let result = match parse_native_resource_cli(arguments) {
        Ok(RoventNativeResourceCli::Help) => {
            writeln!(std::io::stdout().lock(), "{ROVENT_RESOURCE_USAGE}")
                .map_err(|error| error.to_string())
        }
        Ok(RoventNativeResourceCli::BuildCapabilities) => {
            rz_uci::rovent_bootstrap::launch::run(vec!["--build-capabilities-json".into()])
                .map_err(|error| error.to_string())
        }
        Ok(RoventNativeResourceCli::SystemBuildCapabilities) => {
            rz_uci::rovent_bootstrap::launch::run(vec![
                "--build-system-runtime-capabilities-json".into(),
            ])
            .map_err(|error| error.to_string())
        }
        Ok(RoventNativeResourceCli::Run { pin, arguments }) => {
            match run_native_with_resources_versioned(pin, arguments) {
                Ok(receipt) => writeln!(std::io::stderr().lock(), "{}", receipt.observation())
                    .map_err(|error| error.to_string()),
                Err(failure) => {
                    let _ = writeln!(std::io::stderr().lock(), "{}", failure.observation());
                    Err(failure.to_string())
                }
            }
        }
        Ok(RoventNativeResourceCli::RunSystemV3 {
            pin,
            arguments,
            custody_control,
        }) => {
            if custody_control {
                match run_native_with_resources_v3_control(pin, arguments) {
                    Ok(receipt) => writeln!(std::io::stderr().lock(), "{}", receipt.observation())
                        .map_err(|error| error.to_string()),
                    Err(RoventControlledRunFailure::Quarantined(owner)) => owner.serve(),
                    Err(RoventControlledRunFailure::BeforeControl(failure))
                    | Err(RoventControlledRunFailure::Failed(failure)) => {
                        let _ = writeln!(std::io::stderr().lock(), "{}", failure.observation());
                        Err(failure.to_string())
                    }
                }
            } else {
                match run_native_with_resources_v3(pin, arguments) {
                    Ok(receipt) => writeln!(std::io::stderr().lock(), "{}", receipt.observation())
                        .map_err(|error| error.to_string()),
                    Err(failure) => {
                        let _ = writeln!(std::io::stderr().lock(), "{}", failure.observation());
                        Err(failure.to_string())
                    }
                }
            }
        }
        Err(error) => Err(error.to_string()),
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            let detail: String = error
                .chars()
                .take(2048)
                .map(|c| if c.is_control() { ' ' } else { c })
                .collect();
            let _ = writeln!(
                std::io::stderr().lock(),
                "rovent native resource launch halted: {detail}"
            );
            std::process::ExitCode::FAILURE
        }
    }
}
#[cfg(not(all(target_os = "linux", feature = "pals-collection-onnx")))]
fn main() -> std::process::ExitCode {
    eprintln!("rovent native resource launch requires Linux and pals-collection-onnx");
    std::process::ExitCode::FAILURE
}
