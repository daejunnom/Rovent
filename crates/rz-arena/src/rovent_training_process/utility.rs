//! Utility-only training declarations and original process ownership.
//!
//! The utility learner has its own preparation, cohort evidence, checkpoint
//! and execution namespaces. Passive declarations do not mint a comparison,
//! optimizer update, learned model, or past native execution capability.
#![cfg(target_os = "linux")]

#[macro_use]
mod registration;
mod ancestry;
mod checkpoint;
pub mod cli;
mod cost;
mod dataset;
mod execution;
mod execution_v2;
mod intervention;
mod intervention_checkpoint;
mod intervention_dataset;
mod intervention_execution;
mod intervention_window;
mod ledger;
mod numerical_parts;
mod registration_cohort;
pub mod runtime;
mod runtime_family;
mod window_v2;

pub use ancestry::{RoventUtilityInterventionAncestorsRejected, VerifiedUtilityAncestors};
pub use checkpoint::*;
pub use cli::{
    ROVENT_UTILITY_CLI_USAGE, RoventUtilityCliAction, RoventUtilityCliAdmissionRequest,
    RoventUtilityCliCommand, RoventUtilityCliParse, RoventUtilityCliScopeRequest,
    parse_rovent_utility_cli,
};
pub use cost::*;
pub use dataset::{
    ROVENT_UTILITY_CONTEXT_SOURCE, ROVENT_UTILITY_DATASET, ROVENT_UTILITY_POOL,
    VerifiedUtilityDataset,
};
pub use execution::*;
pub use execution_v2::*;
pub use intervention::*;
pub use intervention_checkpoint::{
    PublishedUtilityCheckpointArchive as PublishedUtilityInterventionCheckpointArchive,
    UtilityCheckpointArchiveFailure as UtilityInterventionCheckpointArchiveFailure,
    UtilityCheckpointReadFailure as UtilityInterventionCheckpointReadFailure,
    VerifiedUtilityCheckpointSource as VerifiedUtilityInterventionCheckpointSource,
    VerifiedUtilityOldRunReclamation as VerifiedUtilityInterventionOldRunReclamation,
};
pub use intervention_dataset::*;
pub use intervention_execution::*;
pub use intervention_window::*;
pub use registration::*;
pub use registration_cohort::*;
pub use runtime::{
    RoventUtilityTrainingOutcome, compiled_rovent_utility_trainer_source_sha256,
    run_rovent_utility_from_cli,
};
pub use window_v2::*;

#[cfg(test)]
mod wire_tests;
#[cfg(test)]
mod wire_v2_tests;
