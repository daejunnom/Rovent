//! Unregistered R auxiliary metadata draft. No dataset decoder, optimizer,
//! original update owner, or execution/Verified factory is provided here.
//!
//! Existing /1 and disabled /2 codecs/profiles are deliberately untouched.
//! Consumers must opt in only for Enabled; presence round-tripping itself
//! must not re-sign an old window or add a default auxiliary object.
use super::{ArenaError, check, generation_identifier, generation_sha, invalid};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

pub const ROVENT_R_AUXILIARY_RECIPE: &str = "rz-rovent-r-auxiliary-recipe/1";
pub const ROVENT_R_AUXILIARY_COUNTER_PROJECTION: &str =
    "rz-rovent-r-auxiliary-update-counter-projection/1";
pub const ROVENT_R_AUXILIARY_NORMALIZATION: &str = "valid_exact_trajectory_rows";
pub const ROVENT_R_AUXILIARY_SUPERVISION: &str = "r_auxiliary_only_preserve_pc_evidence";
/// Source draft bound, matched with F's new stdlib helper. Registration and
/// the real R dataset/actual-update consumers remain a separate source unit.
pub const MAX_R_AUXILIARY_ROWS_PER_PC_UPDATE: u16 = 64;

/// Field absence is represented only by the enclosing object's serde default;
/// an explicit JSON null is never deserialized into Absent.
///
/// The containing field must use both `#[serde(default)]` and
/// `#[serde(skip_serializing_if = "RoventRAuxiliaryPresence::is_absent")]`.
/// Serializing a standalone Absent fails rather than silently changing it to
/// null. Null serializes as JSON null and Enabled delegates to its exact DTO.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum RoventRAuxiliaryPresence<T> {
    #[default]
    Absent,
    Null,
    Enabled(T),
}
impl<T> RoventRAuxiliaryPresence<T> {
    pub fn is_absent(&self) -> bool {
        matches!(self, Self::Absent)
    }
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
    pub fn enabled(&self) -> Option<&T> {
        match self {
            Self::Enabled(value) => Some(value),
            _ => None,
        }
    }
    pub fn state(&self) -> RoventRAuxiliaryPresenceState {
        match self {
            Self::Absent => RoventRAuxiliaryPresenceState::Absent,
            Self::Null => RoventRAuxiliaryPresenceState::Null,
            Self::Enabled(_) => RoventRAuxiliaryPresenceState::Enabled,
        }
    }
}
impl<T: Serialize> Serialize for RoventRAuxiliaryPresence<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Absent => Err(serde::ser::Error::custom(
                "absent R auxiliary must be skipped by the containing field",
            )),
            Self::Null => serializer.serialize_none(),
            Self::Enabled(value) => value.serialize(serializer),
        }
    }
}
impl<'de, T: Deserialize<'de>> Deserialize<'de> for RoventRAuxiliaryPresence<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(match Option::<T>::deserialize(deserializer)? {
            Some(value) => Self::Enabled(value),
            None => Self::Null,
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoventRAuxiliaryPresenceState {
    Absent,
    Null,
    Enabled,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRAuxiliaryRecipe {
    pub schema: String,
    /// Raw published JSON file SHA, never its parsed canonical manifest seal.
    pub manifest_file_sha256: String,
    /// Raw published JSON file SHA, never its parsed canonical index seal.
    pub index_file_sha256: String,
    pub sampling_seed: u64,
    pub rows_per_pc_update: u16,
    pub weight: f64,
    pub normalization: String,
    /// A name to join later with recipe.parameter_ownership["nnuec"] and the
    /// real membership owner, distinct from shared/P/C/V owners. Shape alone
    /// proves neither that this owner exists nor that it is exclusive.
    pub optimizer_owner: String,
    pub supervision_source: String,
}
impl RoventRAuxiliaryRecipe {
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            self.schema == ROVENT_R_AUXILIARY_RECIPE
                && generation_sha(&self.manifest_file_sha256)
                && generation_sha(&self.index_file_sha256)
                && self.sampling_seed <= u64::from(u32::MAX)
                && (1..=MAX_R_AUXILIARY_ROWS_PER_PC_UPDATE).contains(&self.rows_per_pc_update)
                && self.weight.is_finite()
                && self.weight > 0.0
                && self.normalization == ROVENT_R_AUXILIARY_NORMALIZATION
                && generation_identifier(&self.optimizer_owner)
                && self.supervision_source == ROVENT_R_AUXILIARY_SUPERVISION,
            "explicit finite raw-file-pinned R auxiliary metadata required",
        )
    }
    /// The same normalized raw recipe shape used by window /2's existing
    /// coefficient-f64-bits-v1 codec: only weight becomes its BE f64 hex16.
    /// No separate coefficient schema/hash domain or profile authority exists.
    pub fn coefficient_projection(&self) -> Result<Value, ArenaError> {
        self.validate()?;
        let mut normalized =
            serde_json::to_value(self).map_err(|error| invalid(error.to_string()))?;
        normalized["weight"] = Value::String(format!("{:016x}", self.weight.to_bits()));
        Ok(normalized)
    }
}
impl RoventRAuxiliaryPresence<RoventRAuxiliaryRecipe> {
    /// Absent and explicit null retain their distinct wire presence, but both
    /// return no enabled projection. Existing disabled /1 and /2 profile
    /// construction must remain on its original byte path.
    pub fn enabled_coefficient_projection(&self) -> Result<Option<Value>, ArenaError> {
        self.enabled()
            .map(RoventRAuxiliaryRecipe::coefficient_projection)
            .transpose()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoventRAuxiliaryUpdateRole {
    Proposer,
    Critic,
    Validator,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoventRAuxiliaryUpdateCompletion {
    Completed,
    NotCompleted,
    Unknown,
}

/// Plain observations supplied by a future actual update owner. These fields
/// do not prove execution, gradients, row validity, or parameter membership.
/// All NNUEC counters below describe ONLY the new R auxiliary stage. The
/// existing disabled P/C nnuec_wdl objective/optimizer behavior is preserved
/// and is neither disabled nor counted by this projection.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRAuxiliaryUpdateObservation {
    pub role: RoventRAuxiliaryUpdateRole,
    pub actual_role_update: RoventRAuxiliaryUpdateCompletion,
    pub pc_valid_rows: u64,
    pub r_valid_exact_trajectory_rows: u64,
    pub r_auxiliary_nnuec_optimizer_steps: u64,
    pub r_auxiliary_nnuec_scheduler_steps: u64,
    /// Completed private-optimizer moment-step events, not a claim that every
    /// parameter tensor was independently inspected by this metadata helper.
    pub r_auxiliary_nnuec_moments_steps: u64,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RoventRAuxiliaryUpdateCounterProjection {
    pub schema: String,
    pub auxiliary_presence: RoventRAuxiliaryPresenceState,
    pub role: RoventRAuxiliaryUpdateRole,
    pub completed_pc_updates: u64,
    pub r_valid_exact_trajectory_rows: u64,
    pub expected_r_auxiliary_nnuec_steps: u64,
    pub r_auxiliary_nnuec_optimizer_steps: u64,
    pub r_auxiliary_nnuec_scheduler_steps: u64,
    pub r_auxiliary_nnuec_moments_steps: u64,
    pub observations_only: bool,
    pub execution_authorized: bool,
}
/// Pure consistency projection. It neither observes nor advances an optimizer.
/// Unknown completion is rejected, never converted into a zero-work success.
/// A successful completed P/C boundary with enabled positive-valid R must
/// report exactly one R-stage optimizer/scheduler/moments step. Failure or
/// partial work belongs in an unknown/partial-tail record, never this success.
/// Zero here means zero R-stage steps, not zero existing P/C nnuec_wdl steps.
pub fn project_rovent_r_auxiliary_update(
    auxiliary: &RoventRAuxiliaryPresence<RoventRAuxiliaryRecipe>,
    observed: &RoventRAuxiliaryUpdateObservation,
) -> Result<RoventRAuxiliaryUpdateCounterProjection, ArenaError> {
    let recipe = auxiliary.enabled();
    if let Some(recipe) = recipe {
        recipe.validate()?;
    }
    check(
        observed.actual_role_update != RoventRAuxiliaryUpdateCompletion::Unknown,
        "unknown actual update completion cannot produce known counter projection",
    )?;
    let pc_role = observed.role != RoventRAuxiliaryUpdateRole::Validator;
    let pc_completed =
        pc_role && observed.actual_role_update == RoventRAuxiliaryUpdateCompletion::Completed;
    check(
        !pc_completed || observed.pc_valid_rows > 0,
        "all-masked P/C observations cannot create a completed update",
    )?;
    check(
        pc_role || observed.pc_valid_rows == 0,
        "V observations cannot claim P/C target rows",
    )?;
    check(
        pc_completed || observed.r_valid_exact_trajectory_rows == 0,
        "R auxiliary may be consumed only at a completed actual P/C boundary",
    )?;
    let row_limit = recipe.map_or(0_u64, |r| u64::from(r.rows_per_pc_update));
    check(
        observed.r_valid_exact_trajectory_rows <= row_limit,
        "disabled/V/out-of-range R auxiliary target observations",
    )?;
    let expected_steps =
        u64::from(pc_completed && observed.r_valid_exact_trajectory_rows > 0 && recipe.is_some());
    check(
        observed.r_auxiliary_nnuec_optimizer_steps == expected_steps
            && observed.r_auxiliary_nnuec_scheduler_steps == expected_steps
            && observed.r_auxiliary_nnuec_moments_steps == expected_steps,
        "successful R auxiliary optimizer/scheduler/moments must match the exact expected zero/one boundary step",
    )?;
    Ok(RoventRAuxiliaryUpdateCounterProjection {
        schema: ROVENT_R_AUXILIARY_COUNTER_PROJECTION.into(),
        auxiliary_presence: auxiliary.state(),
        role: observed.role,
        completed_pc_updates: u64::from(pc_completed),
        r_valid_exact_trajectory_rows: observed.r_valid_exact_trajectory_rows,
        expected_r_auxiliary_nnuec_steps: expected_steps,
        r_auxiliary_nnuec_optimizer_steps: observed.r_auxiliary_nnuec_optimizer_steps,
        r_auxiliary_nnuec_scheduler_steps: observed.r_auxiliary_nnuec_scheduler_steps,
        r_auxiliary_nnuec_moments_steps: observed.r_auxiliary_nnuec_moments_steps,
        observations_only: true,
        execution_authorized: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn recipe() -> RoventRAuxiliaryRecipe {
        RoventRAuxiliaryRecipe {
            schema: ROVENT_R_AUXILIARY_RECIPE.into(),
            manifest_file_sha256: "a".repeat(64),
            index_file_sha256: "b".repeat(64),
            sampling_seed: 29,
            rows_per_pc_update: 4,
            weight: 0.125,
            normalization: ROVENT_R_AUXILIARY_NORMALIZATION.into(),
            optimizer_owner: "nnuec-private-owner".into(),
            supervision_source: ROVENT_R_AUXILIARY_SUPERVISION.into(),
        }
    }
    fn observed() -> RoventRAuxiliaryUpdateObservation {
        RoventRAuxiliaryUpdateObservation {
            role: RoventRAuxiliaryUpdateRole::Proposer,
            actual_role_update: RoventRAuxiliaryUpdateCompletion::Completed,
            pc_valid_rows: 8,
            r_valid_exact_trajectory_rows: 0,
            r_auxiliary_nnuec_optimizer_steps: 0,
            r_auxiliary_nnuec_scheduler_steps: 0,
            r_auxiliary_nnuec_moments_steps: 0,
        }
    }
    #[derive(Debug, Default, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Wire {
        #[serde(default, skip_serializing_if = "RoventRAuxiliaryPresence::is_absent")]
        r_auxiliary: RoventRAuxiliaryPresence<RoventRAuxiliaryRecipe>,
    }
    #[test]
    fn absence_null_and_enabled_roundtrip_without_default_object() {
        let missing: Wire = serde_json::from_value(json!({})).unwrap();
        assert!(missing.r_auxiliary.is_absent());
        assert_eq!(serde_json::to_value(&missing).unwrap(), json!({}));
        let null: Wire = serde_json::from_value(json!({"r_auxiliary":null})).unwrap();
        assert!(null.r_auxiliary.is_null());
        assert_eq!(
            serde_json::to_value(&null).unwrap(),
            json!({"r_auxiliary":null})
        );
        let enabled = json!({"r_auxiliary":recipe()});
        let decoded: Wire = serde_json::from_value(enabled.clone()).unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), enabled);
        assert!(serde_json::to_value(RoventRAuxiliaryPresence::<u8>::Absent).is_err());
    }
    #[test]
    fn recipe_has_raw_file_pin_names_and_strict_finite_bounds() {
        recipe().validate().unwrap();
        let mut value = serde_json::to_value(recipe()).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("manifest_file_sha256");
        value["canonical_manifest_sha256"] = json!("a".repeat(64));
        assert!(serde_json::from_value::<RoventRAuxiliaryRecipe>(value).is_err());
        for rows in [0, 65] {
            let mut r = recipe();
            r.rows_per_pc_update = rows;
            assert!(r.validate().is_err());
        }
        for weight in [0.0, -0.0, -1.0, f64::INFINITY, f64::NAN] {
            let mut r = recipe();
            r.weight = weight;
            assert!(r.validate().is_err());
        }
        let mut r = recipe();
        r.sampling_seed = u64::from(u32::MAX) + 1;
        assert!(r.validate().is_err());
    }
    #[test]
    fn enabled_coefficient_bits_are_exact_and_disabled_profiles_have_no_projection() {
        let r = recipe();
        let mut expected = serde_json::to_value(&r).unwrap();
        expected["weight"] = json!("3fc0000000000000");
        let normalized = r.coefficient_projection().unwrap();
        assert_eq!(normalized, expected);
        assert_eq!(normalized["schema"], json!(ROVENT_R_AUXILIARY_RECIPE));
        assert!(normalized.get("weight_f64_bits").is_none());
        let empty: RoventRAuxiliaryPresence<RoventRAuxiliaryRecipe> =
            RoventRAuxiliaryPresence::Absent;
        assert!(empty.enabled_coefficient_projection().unwrap().is_none());
        let null: RoventRAuxiliaryPresence<RoventRAuxiliaryRecipe> = RoventRAuxiliaryPresence::Null;
        assert!(null.enabled_coefficient_projection().unwrap().is_none());
    }
    #[test]
    fn zero_valid_r_preserves_pc_update_without_r_auxiliary_nnuec_step() {
        let enabled = RoventRAuxiliaryPresence::Enabled(recipe());
        let mut row = observed();
        let projection = project_rovent_r_auxiliary_update(&enabled, &row).unwrap();
        assert_eq!(projection.completed_pc_updates, 1);
        assert_eq!(projection.expected_r_auxiliary_nnuec_steps, 0);
        row.r_auxiliary_nnuec_optimizer_steps = 1;
        row.r_auxiliary_nnuec_scheduler_steps = 1;
        row.r_auxiliary_nnuec_moments_steps = 1;
        assert!(project_rovent_r_auxiliary_update(&enabled, &row).is_err());
    }
    #[test]
    fn valid_r_requires_exactly_one_successful_r_auxiliary_step_at_pc_boundary() {
        let enabled = RoventRAuxiliaryPresence::Enabled(recipe());
        let mut row = observed();
        row.r_valid_exact_trajectory_rows = 2;
        assert!(project_rovent_r_auxiliary_update(&enabled, &row).is_err());
        row.r_auxiliary_nnuec_optimizer_steps = 1;
        row.r_auxiliary_nnuec_scheduler_steps = 1;
        row.r_auxiliary_nnuec_moments_steps = 1;
        assert_eq!(
            project_rovent_r_auxiliary_update(&enabled, &row)
                .unwrap()
                .expected_r_auxiliary_nnuec_steps,
            1
        );
        row.r_auxiliary_nnuec_moments_steps = 0;
        assert!(project_rovent_r_auxiliary_update(&enabled, &row).is_err());
        row.r_auxiliary_nnuec_optimizer_steps = 2;
        row.r_auxiliary_nnuec_scheduler_steps = 2;
        row.r_auxiliary_nnuec_moments_steps = 2;
        assert!(project_rovent_r_auxiliary_update(&enabled, &row).is_err());
    }
    #[test]
    fn v_unknown_partial_or_all_masked_rows_cannot_create_r_updates() {
        let enabled = RoventRAuxiliaryPresence::Enabled(recipe());
        let mut row = observed();
        row.role = RoventRAuxiliaryUpdateRole::Validator;
        row.pc_valid_rows = 0;
        assert_eq!(
            project_rovent_r_auxiliary_update(&enabled, &row)
                .unwrap()
                .completed_pc_updates,
            0
        );
        row.r_valid_exact_trajectory_rows = 1;
        assert!(project_rovent_r_auxiliary_update(&enabled, &row).is_err());
        row = observed();
        row.actual_role_update = RoventRAuxiliaryUpdateCompletion::Unknown;
        assert!(project_rovent_r_auxiliary_update(&enabled, &row).is_err());
        row = observed();
        row.actual_role_update = RoventRAuxiliaryUpdateCompletion::NotCompleted;
        row.r_valid_exact_trajectory_rows = 1;
        assert!(project_rovent_r_auxiliary_update(&enabled, &row).is_err());
        row = observed();
        row.pc_valid_rows = 0;
        assert!(project_rovent_r_auxiliary_update(&enabled, &row).is_err());
    }
    #[test]
    fn disabled_auxiliary_does_not_change_pc_accounting_or_enable_r_rows() {
        for disabled in [
            RoventRAuxiliaryPresence::Absent,
            RoventRAuxiliaryPresence::Null,
        ] {
            let mut row = observed();
            let result = project_rovent_r_auxiliary_update(&disabled, &row).unwrap();
            assert_eq!(result.completed_pc_updates, 1);
            assert_eq!(result.expected_r_auxiliary_nnuec_steps, 0);
            // Existing P/C nnuec_wdl counters are outside this R-stage DTO.
            assert!(
                serde_json::to_value(&result)
                    .unwrap()
                    .get("nnuec_optimizer_steps")
                    .is_none()
            );
            assert!(!result.execution_authorized);
            assert!(result.observations_only);
            row.r_valid_exact_trajectory_rows = 1;
            assert!(project_rovent_r_auxiliary_update(&disabled, &row).is_err());
        }
    }
}
