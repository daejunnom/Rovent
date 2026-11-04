//! E-owned input planning and event bookkeeping, not an execution permit.
//! Rules reconstruction, runner receipts and physical fairness require adapters.
#![forbid(unsafe_code)]

mod contract;
mod engine_environment;
mod external_uci;
mod fastchess;
mod ledger;
mod match_execution;
mod model_endpoints;
#[cfg(feature = "native-cuda")]
mod native_cuda;
mod native_diagnostics;
mod native_exit;
mod native_launch;
mod native_pilot;
#[cfg(target_os = "linux")]
mod native_retention;
mod native_runner;
pub mod pals_adaptive_launch;
pub mod pals_collect;
pub mod pals_collect_v3;
pub mod pals_launch;
#[cfg(feature = "pals-collection-onnx")]
pub mod pals_replay;
mod pgn;
mod plan;
mod process;
#[cfg(feature = "pals-collection-onnx")]
pub mod rovent_collect;
#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
pub mod rovent_resource_scope;
#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
pub mod rovent_training_process;
mod runner;

pub use contract::*;
pub use engine_environment::EngineEnvironmentObservation;
pub use external_uci::*;
pub use fastchess::*;
pub use ledger::*;
pub use match_execution::*;
pub use model_endpoints::*;
#[cfg(feature = "native-cuda")]
pub use native_cuda::*;
pub use native_diagnostics::emit_native_phase;
#[cfg(not(feature = "native-cuda"))]
pub use native_exit::validate_native_cuda_process_exit_trace;
pub use native_launch::*;
pub use native_pilot::*;
pub use native_runner::*;
pub use pgn::*;
pub use plan::*;
pub use process::*;
pub use runner::*;

use rz_experiments::ManifestError;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fmt;

pub const MAX_JSON_BYTES: usize = rz_experiments::MAX_MANIFEST_BYTES;

#[derive(Debug)]
pub enum ArenaError {
    Execution(Box<FixtureExecutionFailure>),
    Contract(rz_contracts::ContractError),
    Manifest(ManifestError),
    Invalid(String),
    Integrity(String),
    Budget(String),
    Io(String),
    HardwareProbe {
        cause: String,
        process: Box<ProcessOutput>,
    },
}

impl fmt::Display for ArenaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Execution(failure) => write!(
                f,
                "fixture evidence persistence failed: {} (pid={}, cleanup={:?})",
                failure.cause, failure.process.receipt.pid, failure.process.receipt.group_cleanup
            ),
            Self::Contract(error) => error.fmt(f),
            Self::Manifest(error) => error.fmt(f),
            Self::Invalid(reason) => write!(f, "arena validation: {reason}"),
            Self::Integrity(reason) => write!(f, "arena integrity: {reason}"),
            Self::Budget(reason) => write!(f, "arena budget: {reason}"),
            Self::Io(reason) => write!(f, "arena IO: {reason}"),
            Self::HardwareProbe { cause, process } => write!(
                f,
                "arena hardware probe: {cause}; pid={} cleanup={:?}",
                process.receipt.pid, process.receipt.group_cleanup
            ),
        }
    }
}

impl std::error::Error for ArenaError {}

impl From<rz_contracts::ContractError> for ArenaError {
    fn from(error: rz_contracts::ContractError) -> Self {
        Self::Contract(error)
    }
}

impl From<ManifestError> for ArenaError {
    fn from(error: ManifestError) -> Self {
        Self::Manifest(error)
    }
}

/// Strict bounded JSON shared with E01; semantic validation belongs to the reader.
pub fn decode_json<T: serde::de::DeserializeOwned>(input: &str) -> Result<T, ArenaError> {
    rz_experiments::decode_json(input).map_err(ArenaError::Manifest)
}

/// Canonical sorted compact UTF-8 JSON SHA-256 for versioned E02 formats.
/// Persisted schemas use integer numeric fields only. Include a domain/version
/// in the hashed payload; exclude its own digest field.
pub fn canonical_sha256<T: Serialize>(value: &T) -> Result<String, ArenaError> {
    let mut value =
        serde_json::to_value(value).map_err(|error| ArenaError::Integrity(error.to_string()))?;
    value.sort_all_objects();
    let bytes =
        serde_json::to_vec(&value).map_err(|error| ArenaError::Integrity(error.to_string()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
