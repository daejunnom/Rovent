//! Exact intervention-only window declarations. No old window is filled,
//! retagged or admitted by this parser; physical execution remains separate.

use super::execution::{
    ROVENT_UTILITY_MAX_CLEANUP_NS, ROVENT_UTILITY_MAX_READ_BYTES, ROVENT_UTILITY_MAX_UPDATES,
    ROVENT_UTILITY_MAX_WALL_NS, execution_identifier,
};
use super::intervention::validate_utility_intervention_preparation;
use crate::{ArenaError, rovent_training_process::MAX_TRAINER_OUTPUT_BYTES};
use rz_experiments::pals_data_v3::{generation_digest, generation_sha};
use serde::de::{MapAccess, Visitor, value::MapAccessDeserializer};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const ROVENT_UTILITY_INTERVENTION_WINDOW_DOMAIN: &str = "rz-rovent-utility-train-window/3";

fn check(condition: bool, reason: &str) -> Result<(), ArenaError> {
    if condition {
        Ok(())
    } else {
        Err(ArenaError::Integrity(format!(
            "intervention window: {reason}"
        )))
    }
}

utility_object! {
/// The same 23 bounded window quantities with a distinct explicit family.
/// This is a declaration; no original clock, lease or checkpoint is issued.
pub struct RoventUtilityInterventionTrainWindow {
    pub schema: String,
    pub window_id: String,
    pub run_id: String,
    pub preparation_sha256: String,
    pub dataset_sha256: String,
    pub device: String,
    pub precision: String,
    pub tf32: bool,
    pub deterministic: bool,
    pub cpu_threads: u32,
    pub interop_threads: u32,
    pub training_seed: u32,
    pub microbatch: u32,
    pub accumulation: u32,
    pub max_optimizer_updates: u64,
    pub checkpoint_every_updates: u64,
    pub max_wall_time_ns: u64,
    pub cleanup_time_ns: u64,
    pub max_read_bytes: u64,
    pub max_output_bytes: u64,
    pub max_groups: u32,
    pub max_contexts: u32,
    pub checkpoint_reserve_ns: u64,
}
}
impl RoventUtilityInterventionTrainWindow {
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            self.schema == ROVENT_UTILITY_INTERVENTION_WINDOW_DOMAIN
                && execution_identifier(&self.window_id)
                && execution_identifier(&self.run_id)
                && generation_sha(&self.preparation_sha256)
                && generation_sha(&self.dataset_sha256),
            "explicit intervention window/preparation/dataset family required",
        )?;
        check(
            super::window_v2::canonical_device(&self.device)
                && self.precision == "fp32"
                && !self.tf32
                && self.deterministic
                && (1..=256).contains(&self.cpu_threads)
                && (1..=256).contains(&self.interop_threads)
                && self.microbatch == 1
                && self.accumulation == 8,
            "original deterministic FP32 utility numerical/thread recipe required",
        )?;
        check(
            self.max_optimizer_updates <= ROVENT_UTILITY_MAX_UPDATES
                && (1..=ROVENT_UTILITY_MAX_UPDATES).contains(&self.checkpoint_every_updates)
                && (1..=ROVENT_UTILITY_MAX_WALL_NS).contains(&self.max_wall_time_ns)
                && (1..=ROVENT_UTILITY_MAX_CLEANUP_NS).contains(&self.cleanup_time_ns)
                && (1..=ROVENT_UTILITY_MAX_READ_BYTES).contains(&self.max_read_bytes)
                && (1..=MAX_TRAINER_OUTPUT_BYTES).contains(&self.max_output_bytes)
                && self.max_groups > 0
                && self.max_contexts > 0
                && self.checkpoint_reserve_ns > 0
                && self.checkpoint_reserve_ns < self.max_wall_time_ns,
            "finite original update/group/context/read/output/checkpoint/time limits required",
        )
    }
    pub fn canonical_sha256(&self) -> Result<String, ArenaError> {
        self.validate()?;
        generation_digest(ROVENT_UTILITY_INTERVENTION_WINDOW_DOMAIN, self)
            .map_err(ArenaError::Manifest)
    }
    /// The common machine invokes this typed method only for this new lane.
    /// It consumes exact9 metadata without creating or reusing old PREP2.
    pub fn join_cohort_preparation(&self, preparation: &Value) -> Result<(), ArenaError> {
        self.validate()?;
        let raw =
            serde_json::to_string(preparation).map_err(|e| ArenaError::Integrity(e.to_string()))?;
        let preparation = validate_utility_intervention_preparation(&raw)?;
        check(
            self.preparation_sha256 == preparation.preparation_sha256,
            "intervention window changed its original exact9 preparation",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn window() -> RoventUtilityInterventionTrainWindow {
        RoventUtilityInterventionTrainWindow {
            schema: ROVENT_UTILITY_INTERVENTION_WINDOW_DOMAIN.into(),
            window_id: "window".into(),
            run_id: "run".into(),
            preparation_sha256: "1".repeat(64),
            dataset_sha256: "2".repeat(64),
            device: "cpu".into(),
            precision: "fp32".into(),
            tf32: false,
            deterministic: true,
            cpu_threads: 2,
            interop_threads: 1,
            training_seed: 31,
            microbatch: 1,
            accumulation: 8,
            max_optimizer_updates: 0,
            checkpoint_every_updates: 250,
            max_wall_time_ns: 1_000_000,
            cleanup_time_ns: 100_000,
            max_read_bytes: 1 << 20,
            max_output_bytes: 1 << 20,
            max_groups: 1,
            max_contexts: 1,
            checkpoint_reserve_ns: 100_000,
        }
    }
    #[test]
    fn explicit_intervention_window_is_not_a_legacy_window() {
        let value = window();
        value.validate().unwrap();
        let raw = serde_json::to_value(&value).unwrap();
        assert_eq!(raw.as_object().unwrap().len(), 23);
        let legacy: super::super::window_v2::RoventUtilityTrainWindowV2 =
            serde_json::from_value(raw.clone()).unwrap();
        assert!(legacy.validate().is_err());
        let mut changed = value.clone();
        changed.schema = super::super::window_v2::ROVENT_UTILITY_WINDOW_DOMAIN_V2.into();
        assert!(changed.validate().is_err());
        assert_ne!(
            value.canonical_sha256().unwrap(),
            generation_digest(
                super::super::window_v2::ROVENT_UTILITY_WINDOW_DOMAIN_V2,
                &raw
            )
            .unwrap()
        );
    }
}
