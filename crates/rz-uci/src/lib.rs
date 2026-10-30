//! B-owned UCI text and session control. Rules and Runtime are injected adapters.
//!
//! This crate deliberately does not define shared evaluation IDs or implement
//! chess rules. A [`PositionPort`] must validate complete move traces and supply
//! the root's ordered legal moves before a [`Session`] can start searching.

#![forbid(unsafe_code)]

pub mod bootstrap;
pub mod bridge;
pub mod contracts;
pub mod engine;
#[cfg(feature = "onnx-cpu")]
pub mod native_attestation;
#[cfg(all(feature = "onnx-cuda", feature = "experimental-batch"))]
pub mod native_batch_attestation;
#[cfg(feature = "onnx-cpu")]
pub mod native_bootstrap;
#[cfg(feature = "onnx-cuda")]
pub mod native_cuda_attestation;
#[cfg(feature = "onnx-cpu")]
pub mod native_profile;
pub mod pals_adaptive;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub mod pals_adaptive_bootstrap;
#[cfg(feature = "pals-adaptive")]
pub mod pals_adaptive_sources;
#[cfg(feature = "pals-adaptive")]
pub mod pals_adaptive_startup;
#[cfg(feature = "search-work-receipts")]
pub mod pals_attestation;
#[cfg(feature = "search-work-receipts")]
pub mod pals_checker_profile;
#[cfg(feature = "search-work-receipts")]
pub mod pals_cpu_task;
pub mod pals_native;
pub mod parser;
#[cfg(feature = "search-work-receipts")]
pub mod process_receipts;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub mod rovent_bootstrap;
pub mod rovent_driver;
#[cfg(feature = "pals-adaptive")]
pub mod rovent_external_launch;
pub mod rovent_ponder;
#[cfg(feature = "pals-adaptive")]
pub mod rovent_root_admission;
pub mod runner;
pub mod search_driver;
pub mod session;

pub use parser::{Command, GoLimits, ParseError, ParserLimits, PositionBase, PositionSpec, parse};
pub use runner::{
    Event, ServeError, ServeFailure, forward_lines, handle_event, serve_events,
    serve_events_with_handler,
};
pub use session::{
    CancelReason, Diagnostic, Effect, EngineIdentity, OptionKind, OptionSpec, OptionValue,
    PositionPort, PreparedPosition, SearchCompletion, SearchTicket, Session, SessionError,
    SessionResult,
};
