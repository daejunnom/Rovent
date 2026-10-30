//! One real game and one real action context, retained as non-training data.
use std::process::ExitCode;

#[cfg(target_os = "linux")]
fn run() -> Result<(), String> {
    use rz_arena::pals_collect_v3::{
        DiagnosticGameFailureV1, run_action_diagnostic_owned_v3,
        run_diagnostic_game_from_declaration_v1,
    };
    use std::{path::Path, sync::{Arc, atomic::AtomicBool}, time::{Duration, Instant}};

    let args: Vec<_> = std::env::args().skip(1).collect();
    let names = ["--selfplay-declaration", "--selfplay-sha256", "--action-declaration",
        "--action-sha256", "--decision-index", "--game-and-root-ms", "--action-wall-ms",
        "--max-root-read-bytes"];
    if args.len() != names.len() * 2 || names.iter().enumerate().any(|(i, name)| args[2 * i] != *name) {
        return Err("usage: rz-pals-action-diagnostic --selfplay-declaration <absolute> --selfplay-sha256 <sha256> --action-declaration <absolute> --action-sha256 <sha256> --decision-index <integer> --game-and-root-ms <integer> --action-wall-ms <integer> --max-root-read-bytes <integer>".into());
    }
    let decision = args[9].parse::<usize>().map_err(|_| "invalid decision index")?;
    let game_ms = args[11].parse::<u64>().map_err(|_| "invalid game window")?;
    let action_ms = args[13].parse::<u64>().map_err(|_| "invalid action window")?;
    let read_bytes = args[15].parse::<u64>().map_err(|_| "invalid root read limit")?;
    if !(1..=7_200_000).contains(&game_ms) || !(1..=3_600_000).contains(&action_ms)
        || read_bytes == 0 || read_bytes > 4 * 1024 * 1024
        || !Path::new(&args[1]).is_absolute() || !Path::new(&args[5]).is_absolute() {
        return Err("diagnostic paths or finite resource bounds are invalid".into());
    }
    let cancel = Arc::new(AtomicBool::new(false));
    let mut hooks = Vec::new();
    for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
        match signal_hook::flag::register(signal, Arc::clone(&cancel)) {
            Ok(id) => hooks.push(id),
            Err(error) => {
                for id in hooks { signal_hook::low_level::unregister(id); }
                return Err(format!("signal registration: {error}"));
            }
        }
    }
    let result = (|| {
        let until = Instant::now().checked_add(Duration::from_millis(game_ms)).ok_or("game deadline overflow")?;
        let game = run_diagnostic_game_from_declaration_v1(
            Path::new(&args[1]), &args[3], decision, until, &cancel, read_bytes,
        ).map_err(|failure| match failure {
            DiagnosticGameFailureV1::Preparation(error) => format!("game preparation: {error}"),
            DiagnosticGameFailureV1::Game(failure) => format!("game: {}", failure.error),
            DiagnosticGameFailureV1::RootSelection { error, .. } => format!("actual root selection: {error}"),
        })?;
        let action_until = Instant::now().checked_add(Duration::from_millis(action_ms)).ok_or("action deadline overflow")?;
        let action = match run_action_diagnostic_owned_v3(
            Path::new(&args[5]), &args[7], Arc::clone(game.archive()), decision,
            Arc::clone(&cancel), Duration::from_millis(action_ms),
        ) {
            Ok(action) => action,
            Err(original) => {
                // Preserve the actual failed intake/process and partial
                // owners until process exit, including any unresolved child.
                let message = original.to_string();
                let _retained_until_exit = Box::leak(original);
                return Err(message);
            }
        };
        println!("{}", serde_json::json!({
            "schema":"rz-pals-actual-game-action-diagnostic/1",
            "source_game_count":1, "root_read_bytes":game.root_read_bytes(),
            "action":action.status(), "training_eligible":false, "optimizer_updates":0,
            "diagnostic_identity": diagnostic_identity(action_until, &cancel),
        }));
        Ok(())
    })();
    for id in hooks { signal_hook::low_level::unregister(id); }
    result
}

/// Readback only; inability to observe it must not change a completed game or
/// substitute for the independent production source inventory.
#[cfg(target_os = "linux")]
fn diagnostic_identity(until: std::time::Instant, cancel: &std::sync::atomic::AtomicBool) -> serde_json::Value {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let binary = (|| -> std::io::Result<String> {
        let path = std::env::current_exe()?;
        let mut file = std::fs::File::open(path)?;
        let before = file.metadata()?;
        if before.len() == 0 || before.len() > 512 * 1024 * 1024 {
            return Err(std::io::Error::other("diagnostic executable outside read bound"));
        }
        let mut hasher = Sha256::new();
        let mut bytes = [0u8; 64 * 1024];
        loop {
            if std::time::Instant::now() >= until || cancel.load(std::sync::atomic::Ordering::Acquire) {
                return Err(std::io::Error::other("diagnostic identity window ended"));
            }
            let n = file.read(&mut bytes)?;
            if n == 0 { break; }
            hasher.update(&bytes[..n]);
        }
        let after = file.metadata()?;
        if before.len() != after.len() || before.modified().ok() != after.modified().ok() {
            return Err(std::io::Error::other("diagnostic executable changed while reading"));
        }
        Ok(format!("{:x}", hasher.finalize()))
    })().ok();
    serde_json::json!({
        "declared_source_commit": option_env!("RZ_PALS_SOURCE_COMMIT"),
        "executable_sha256": binary,
        "observation_scope": "separate diagnostic readback; not the production source inventory",
    })
}

#[cfg(not(target_os = "linux"))]
fn run() -> Result<(), String> { Err("actual diagnostic requires Linux original process transport".into()) }

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => { eprintln!("pals-action-diagnostic failed: {error}"); ExitCode::FAILURE }
    }
}
