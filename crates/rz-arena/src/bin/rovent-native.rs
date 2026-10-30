//! Actual native UCI entry with the shared delegated Linux resource owner.
#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
fn main() -> std::process::ExitCode {
    use rz_arena::rovent_resource_scope::launch::{
        ROVENT_RESOURCE_USAGE, RoventNativeResourceCli, parse_native_resource_cli,
        run_native_with_resources,
    };
    use std::io::Write;
    let result = match parse_native_resource_cli(std::env::args().skip(1).collect()) {
        Ok(RoventNativeResourceCli::Help) => {
            writeln!(std::io::stdout().lock(), "{ROVENT_RESOURCE_USAGE}")
                .map_err(|error| error.to_string())
        }
        Ok(RoventNativeResourceCli::BuildCapabilities) => {
            rz_uci::rovent_bootstrap::launch::run(vec!["--build-capabilities-json".into()])
                .map_err(|error| error.to_string())
        }
        Ok(RoventNativeResourceCli::Run { pin, arguments }) => {
            match run_native_with_resources(pin, arguments) {
                Ok(receipt) => writeln!(std::io::stderr().lock(), "{}", receipt.observation())
                    .map_err(|error| error.to_string()),
                Err(failure) => {
                    let _ = writeln!(std::io::stderr().lock(), "{}", failure.observation());
                    Err(failure.to_string())
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
