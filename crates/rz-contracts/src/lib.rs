//! Coordinator-owned revision 0.1. No position, search, backend or wire dependency.
//! Concrete adapters must attest Rules legality, immutable snapshots and registry
//! liveness. These types validate the boundary; they do not implement chess,
//! scheduling, exactly-once finalization or GPU completion.
#![forbid(unsafe_code)]

mod evaluation;
pub mod pals;
pub mod pals_adaptive;
mod primitives;
pub mod rovent;
mod state;

pub use evaluation::*;
pub use primitives::*;
pub use state::*;
