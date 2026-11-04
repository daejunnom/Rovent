//! Utility window /2 declarations with an explicit checkpoint time reserve.
//!
//! This is a distinct 23-field codec. The original window /1 is neither filled
//! with a default reserve nor retagged to validate this window. The reserve is
//! an admission boundary declaration, not a claim that checkpoint I/O, process
//! cleanup or physical completion has already happened.
#![cfg(target_os = "linux")]

use super::execution::{
    ROVENT_UTILITY_MAX_CLEANUP_NS, ROVENT_UTILITY_MAX_READ_BYTES, ROVENT_UTILITY_MAX_UPDATES,
    ROVENT_UTILITY_MAX_WALL_NS, execution_identifier,
};
use super::registration_cohort::validate_utility_cohort_preparation;
use crate::ArenaError;
use crate::rovent_training_process::MAX_TRAINER_OUTPUT_BYTES;
use rz_experiments::pals_data_v3::{generation_digest, generation_sha};
use serde::de::{MapAccess, Visitor, value::MapAccessDeserializer};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const ROVENT_UTILITY_WINDOW_DOMAIN_V2: &str = "rz-rovent-utility-train-window/2";

fn invalid(reason: impl Into<String>) -> ArenaError {
    ArenaError::Integrity(format!(
        "Rovent utility window /2 metadata: {}",
        reason.into()
    ))
}
fn check(condition: bool, reason: &str) -> Result<(), ArenaError> {
    if condition {
        Ok(())
    } else {
        Err(invalid(reason))
    }
}
pub(super) fn canonical_device(value: &str) -> bool {
    value == "cpu"
        || value.strip_prefix("cuda:").is_some_and(|index| {
            !index.is_empty()
                && index.len() <= 2
                && index.bytes().all(|byte| byte.is_ascii_digit())
                && index
                    .parse::<u32>()
                    .is_ok_and(|number| number < 64 && number.to_string() == index)
        })
}

utility_object! {
/// Required window /2 roster: the original 22 fields plus the original caller's
/// checkpoint_reserve_ns. Zero optimizer updates still permits a finite export
/// or inspection request without claiming a completed optimizer update.
pub struct RoventUtilityTrainWindowV2 {
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
impl RoventUtilityTrainWindowV2 {
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            self.schema == ROVENT_UTILITY_WINDOW_DOMAIN_V2
                && execution_identifier(&self.window_id)
                && execution_identifier(&self.run_id)
                && generation_sha(&self.preparation_sha256)
                && generation_sha(&self.dataset_sha256),
            "explicit /2 window/run/preparation/dataset metadata required",
        )?;
        check(
            canonical_device(&self.device)
                && self.precision == "fp32"
                && !self.tf32
                && self.deterministic
                && (1..=256).contains(&self.cpu_threads)
                && (1..=256).contains(&self.interop_threads)
                && self.microbatch == 1
                && self.accumulation == 8,
            "deterministic FP32 utility window, microbatch 1 and accumulation 8 required",
        )?;
        check(
            self.max_optimizer_updates <= ROVENT_UTILITY_MAX_UPDATES
                && (1..=ROVENT_UTILITY_MAX_UPDATES).contains(&self.checkpoint_every_updates)
                && (1..=ROVENT_UTILITY_MAX_WALL_NS).contains(&self.max_wall_time_ns)
                && (1..=ROVENT_UTILITY_MAX_CLEANUP_NS).contains(&self.cleanup_time_ns)
                && (1..=ROVENT_UTILITY_MAX_READ_BYTES).contains(&self.max_read_bytes)
                && (1..=MAX_TRAINER_OUTPUT_BYTES).contains(&self.max_output_bytes)
                && self.max_groups > 0
                && self.max_contexts > 0,
            "finite original utility update/group/context/time/read/output limits required",
        )?;
        check(
            self.checkpoint_reserve_ns > 0 && self.checkpoint_reserve_ns < self.max_wall_time_ns,
            "positive checkpoint reserve strictly below the original wall limit required",
        )
    }
    pub fn canonical_sha256(&self) -> Result<String, ArenaError> {
        self.validate()?;
        generation_digest(ROVENT_UTILITY_WINDOW_DOMAIN_V2, self).map_err(ArenaError::Manifest)
    }
    /// Metadata identity only. Actual cohort intake still requires the original
    /// held publication, complete ancestry and source/Rules/cost evidence.
    pub fn join_cohort_preparation(&self, preparation: &Value) -> Result<(), ArenaError> {
        self.validate()?;
        check(
            self.preparation_sha256 == validate_utility_cohort_preparation(preparation)?,
            "window /2 changed its original cohort PREP2 metadata seal",
        )
    }
}

#[cfg(test)]
mod tests {
    //! Source fixtures only: no files, clocks, processes or tensor execution.
    use super::*;
    use serde_json::json;

    fn window() -> RoventUtilityTrainWindowV2 {
        RoventUtilityTrainWindowV2 {
            schema: ROVENT_UTILITY_WINDOW_DOMAIN_V2.into(),
            window_id: "utility-window-v2".into(),
            run_id: "utility-run".into(),
            preparation_sha256: "12".repeat(32),
            dataset_sha256: "13".repeat(32),
            device: "cpu".into(),
            precision: "fp32".into(),
            tf32: false,
            deterministic: true,
            cpu_threads: 2,
            interop_threads: 1,
            training_seed: 31,
            microbatch: 1,
            accumulation: 8,
            max_optimizer_updates: 4,
            checkpoint_every_updates: 2,
            max_wall_time_ns: 60_000_000_000,
            cleanup_time_ns: 30_000_000_000,
            max_read_bytes: 1 << 20,
            max_output_bytes: 1 << 20,
            max_groups: 16,
            max_contexts: 128,
            checkpoint_reserve_ns: 10_000_000_000,
        }
    }

    #[test]
    fn exact_23_field_roundtrip_preserves_required_checkpoint_reserve() {
        let original = window();
        original.validate().unwrap();
        let value = serde_json::to_value(&original).unwrap();
        assert_eq!(value.as_object().unwrap().len(), 23);
        let decoded: RoventUtilityTrainWindowV2 = serde_json::from_value(value).unwrap();
        assert_eq!(decoded, original);
        assert_eq!(
            decoded.canonical_sha256().unwrap(),
            original.canonical_sha256().unwrap()
        );
        let mut export_window = decoded;
        export_window.max_optimizer_updates = 0;
        export_window.validate().unwrap();
    }

    #[test]
    fn reserve_must_be_positive_below_the_original_wall_limit_without_overflow() {
        let mut value = window();
        for reserve in [0, value.max_wall_time_ns, u64::MAX] {
            value.checkpoint_reserve_ns = reserve;
            assert!(value.validate().is_err());
            assert!(value.canonical_sha256().is_err());
        }
        value.checkpoint_reserve_ns = 1;
        value.validate().unwrap();
        value.checkpoint_reserve_ns = value.max_wall_time_ns - 1;
        value.validate().unwrap();
        value.max_wall_time_ns = u64::MAX;
        assert!(value.validate().is_err());
    }

    #[test]
    fn missing_boolean_negative_and_overflow_reserves_are_not_defaulted_or_coerced() {
        let original = serde_json::to_value(window()).unwrap();
        let mut missing = original.clone();
        missing
            .as_object_mut()
            .unwrap()
            .remove("checkpoint_reserve_ns");
        assert!(serde_json::from_value::<RoventUtilityTrainWindowV2>(missing).is_err());
        for invalid in [json!(true), json!(-1), json!("1000"), Value::Null] {
            let mut changed = original.clone();
            changed["checkpoint_reserve_ns"] = invalid;
            assert!(serde_json::from_value::<RoventUtilityTrainWindowV2>(changed).is_err());
        }
        let text = serde_json::to_string(&window()).unwrap();
        let overflow = text.replace(
            "\"checkpoint_reserve_ns\":10000000000",
            "\"checkpoint_reserve_ns\":18446744073709551616",
        );
        assert!(serde_json::from_str::<RoventUtilityTrainWindowV2>(&overflow).is_err());
    }

    #[test]
    fn old_version_is_rejected_without_filling_or_retagging_its_window() {
        let mut changed = window();
        changed.schema = super::super::execution::ROVENT_UTILITY_WINDOW_DOMAIN.into();
        assert!(changed.validate().is_err());
        assert!(changed.canonical_sha256().is_err());
        let mut original_v1_roster = serde_json::to_value(changed).unwrap();
        original_v1_roster
            .as_object_mut()
            .unwrap()
            .remove("checkpoint_reserve_ns");
        assert!(serde_json::from_value::<RoventUtilityTrainWindowV2>(original_v1_roster).is_err());
    }

    #[test]
    fn duplicate_unknown_and_sequence_windows_are_rejected_at_the_new_decoder() {
        let text = serde_json::to_string(&window()).unwrap();
        let duplicate = text.replacen("{", "{\"checkpoint_reserve_ns\":1,", 1);
        assert!(serde_json::from_str::<RoventUtilityTrainWindowV2>(&duplicate).is_err());
        let mut unknown = serde_json::to_value(window()).unwrap();
        unknown["derived_cutoff_ns"] = json!(50_000_000_000_u64);
        assert!(serde_json::from_value::<RoventUtilityTrainWindowV2>(unknown).is_err());
        let object = serde_json::to_value(window()).unwrap();
        let sequence = object
            .as_object()
            .unwrap()
            .values()
            .cloned()
            .collect::<Vec<_>>();
        assert!(serde_json::from_value::<RoventUtilityTrainWindowV2>(json!(sequence)).is_err());
    }

    #[test]
    fn full_reserve_and_new_domain_both_participate_in_the_canonical_seal() {
        let original = window();
        let canonical = original.canonical_sha256().unwrap();
        assert_eq!(
            canonical,
            generation_digest(ROVENT_UTILITY_WINDOW_DOMAIN_V2, &original).unwrap()
        );
        assert_ne!(
            canonical,
            generation_digest(
                super::super::execution::ROVENT_UTILITY_WINDOW_DOMAIN,
                &original
            )
            .unwrap()
        );
        let mut changed = original.clone();
        changed.checkpoint_reserve_ns += 1;
        assert_ne!(canonical, changed.canonical_sha256().unwrap());
        assert!(
            original
                .join_cohort_preparation(&json!({
                    "schema":super::super::registration::ROVENT_UTILITY_PREPARATION_DOMAIN
                }))
                .is_err()
        );
    }
}
