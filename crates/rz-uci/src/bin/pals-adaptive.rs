//! Explicit V5 route. Legacy rz-uci flags and startup remain independently usable.
fn main() {
    // This is a self-observation at Rust main entry, not the OS spawn time.
    let main_entry = std::time::Instant::now();
    if let Err(error) = rz_uci::pals_adaptive_bootstrap::run_cli_from_main_entry(
        std::env::args().skip(1).collect(),
        main_entry,
    ) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
