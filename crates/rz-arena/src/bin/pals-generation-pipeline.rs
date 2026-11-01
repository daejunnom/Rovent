//! Explicit actual generation pipeline. Parsing and the default invocation
//! never execute collection, optimizer updates, or evaluation.
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(series) => {
            // Keep the original series, failed-pair custody and pipeline alive
            // through report serialization and publication. JSON is readback,
            // never a replacement for the typed evaluation owner.
            let completed = series.report().status
                == rz_arena::pals_adaptive_launch::PalsEvaluationSeriesStatusV5::Completed;
            match serde_json::to_value(series.report()) {
                Ok(report) => {
                    println!("{report}");
                    if completed {
                        ExitCode::SUCCESS
                    } else {
                        ExitCode::FAILURE
                    }
                }
                Err(error) => {
                    eprintln!("generation-pipeline report serialization failed: {error}");
                    ExitCode::FAILURE
                }
            }
        }
        Err(error) => {
            eprintln!("generation-pipeline halted: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run()
-> Result<rz_arena::pals_adaptive_launch::PalsEvaluationSeriesV5, Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if !matches!(arguments.len(), 5 | 7)
        || arguments[0] != "--execute"
        || arguments[1] != "--declaration"
        || arguments[3] != "--declaration-sha256"
        || (arguments.len() == 7
            && (arguments[5] != "--storage-policy" || arguments[6] != "advisory"))
    {
        return Err("usage: rz-pals-generation-pipeline --execute --declaration /absolute/prepared-declaration.json --declaration-sha256 <sha256> [--storage-policy advisory]; execution requires all three explicit flags".into());
    }
    if !std::path::Path::new(&arguments[2]).is_absolute()
        || arguments[4].len() != 64
        || !arguments[4].bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("generation-pipeline requires an absolute declaration path and a 64-digit hexadecimal SHA-256".into());
    }

    #[cfg(not(target_os = "linux"))]
    return Err("actual generation pipeline requires Linux original process transport".into());

    #[cfg(target_os = "linux")]
    {
        use std::path::Path;
        use std::sync::{Arc, atomic::AtomicBool};

        let cancel = Arc::new(AtomicBool::new(false));
        let mut hooks = Vec::new();
        for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
            match signal_hook::flag::register(signal, Arc::clone(&cancel)) {
                Ok(id) => hooks.push(id),
                Err(error) => {
                    for id in hooks {
                        signal_hook::low_level::unregister(id);
                    }
                    return Err(error.into());
                }
            }
        }

        let storage_policy = if arguments.len() == 7 {
            rz_arena::pals_adaptive_launch::PalsOwnedStoragePolicy::AdvisoryV1
        } else {
            rz_arena::pals_adaptive_launch::PalsOwnedStoragePolicy::StrictV5
        };
        let result = rz_arena::pals_collect_v3::run_generation_and_evaluation_owned_with_storage_policy_v3(
            Path::new(&arguments[2]),
            &arguments[4],
            Arc::clone(&cancel),
            storage_policy,
        )
        .map_err(Into::into);

        for id in hooks {
            signal_hook::low_level::unregister(id);
        }
        result
    }
}
