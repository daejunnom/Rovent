//! Thin native Rovent UCI entry; source and resource admission live in the
//! shared bootstrap rather than a second protocol or runtime implementation.

fn main() {
    if let Err(error) = rz_uci::rovent_bootstrap::launch::run(std::env::args().skip(1).collect()) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
