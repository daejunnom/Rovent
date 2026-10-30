//! Validated numerical metadata shared by explicitly separate learner lanes.
//!
//! No constructor emits another preparation envelope. The original frozen
//! declarations are read without changing their bytes, family or eligibility.
//! These accessors are not tensor membership, FD, peer or optimizer authority.

use super::{
    RoventUtilityBaseConfig, RoventUtilityCohortSourceRegistration,
    RoventUtilityInterventionTrainingPreparation, RoventUtilityParameterSpec,
    RoventUtilityTrainingRecipe, validate_utility_cohort_preparation,
};
use crate::ArenaError;
use rz_experiments::pals_data_v3::generation_digest;
use rz_experiments::rovent_utility_intervention_preparation::ROVENT_UTILITY_INTERVENTION_PREPARATION_DOMAIN;
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UtilityNumericalFamily {
    CohortV1,
    InterventionV2,
}

/// Closed metadata accessors only. Fields are private and every constructor
/// invokes the original typed numerical validators plus its own exact family
/// parser. Raw source files and the original preparation remain separately
/// held by the actual dataset/runtime/checkpoint owner.
#[derive(Clone)]
pub(super) struct ValidatedUtilityNumericalParts {
    optimizer_owner: String,
    registration: RoventUtilityCohortSourceRegistration,
    recipe_sha256: String,
    source_registration_sha256: String,
    parameter_spec_sha256: String,
    preparation_sha256: String,
}
impl ValidatedUtilityNumericalParts {
    pub(super) fn from_cohort(value: &Value) -> Result<Self, ArenaError> {
        let preparation_sha256 = validate_utility_cohort_preparation(value)?;
        let recipe = serde_json::from_value(value["recipe"].clone())
            .map_err(|e| ArenaError::Integrity(format!("original utility recipe: {e}")))?;
        let registration = serde_json::from_value(value["registration"].clone())
            .map_err(|e| ArenaError::Integrity(format!("original utility registration: {e}")))?;
        let parameter_spec = serde_json::from_value(value["parameter_spec"].clone())
            .map_err(|e| ArenaError::Integrity(format!("original utility parameter spec: {e}")))?;
        Self::validated(
            UtilityNumericalFamily::CohortV1,
            &recipe,
            registration,
            &parameter_spec,
            preparation_sha256,
        )
    }
    pub(super) fn from_intervention(value: &Value) -> Result<Self, ArenaError> {
        let preparation: RoventUtilityInterventionTrainingPreparation =
            serde_json::from_value(value.clone()).map_err(|e| {
                ArenaError::Integrity(format!("exact intervention preparation: {e}"))
            })?;
        Self::from_intervention_typed(&preparation)
    }
    pub(super) fn from_intervention_typed(
        value: &RoventUtilityInterventionTrainingPreparation,
    ) -> Result<Self, ArenaError> {
        value.validate().map_err(ArenaError::Manifest)?;
        Self::validated(
            UtilityNumericalFamily::InterventionV2,
            &value.recipe,
            value.registration.clone(),
            &value.parameter_spec,
            value.preparation_sha256.clone(),
        )
    }
    fn validated(
        family: UtilityNumericalFamily,
        recipe: &RoventUtilityTrainingRecipe,
        registration: RoventUtilityCohortSourceRegistration,
        parameter_spec: &RoventUtilityParameterSpec,
        preparation_sha256: String,
    ) -> Result<Self, ArenaError> {
        registration.validate(recipe, parameter_spec)?;
        let recipe_sha256 = recipe.canonical_sha256()?;
        let parameter_spec_sha256 = parameter_spec
            .canonical_sha256(&registration.frozen_base.config, &recipe.optimizer_owner)?;
        let source_registration_sha256 = match family {
            UtilityNumericalFamily::CohortV1 => {
                registration.canonical_sha256(recipe, parameter_spec)?
            }
            UtilityNumericalFamily::InterventionV2 => generation_digest(
                ROVENT_UTILITY_INTERVENTION_PREPARATION_DOMAIN,
                &json!({"kind":"source_registration", "value":registration}),
            )
            .map_err(ArenaError::Manifest)?,
        };
        // The exact preparation remains with the dataset/runtime owner. Only
        // values used after validation are retained here: family selects the
        // source seal above, and the full recipe/spec need not be cloned into
        // every later ledger metadata view.
        Ok(Self {
            optimizer_owner: recipe.optimizer_owner.clone(),
            registration,
            recipe_sha256,
            source_registration_sha256,
            parameter_spec_sha256,
            preparation_sha256,
        })
    }
    pub(super) fn registration(&self) -> &RoventUtilityCohortSourceRegistration {
        &self.registration
    }
    pub(super) fn recipe_sha256(&self) -> &str {
        &self.recipe_sha256
    }
    pub(super) fn source_registration_sha256(&self) -> &str {
        &self.source_registration_sha256
    }
    pub(super) fn parameter_spec_sha256(&self) -> &str {
        &self.parameter_spec_sha256
    }
    pub(super) fn preparation_sha256(&self) -> &str {
        &self.preparation_sha256
    }
    pub(super) fn optimizer_owner(&self) -> &str {
        &self.optimizer_owner
    }
    pub(super) fn frozen_base_config(&self) -> &RoventUtilityBaseConfig {
        &self.registration.frozen_base.config
    }
    pub(super) fn frozen_base_weights_sha256(&self) -> &str {
        &self.registration.frozen_base.weights_sha256
    }
    pub(super) fn initial_utility_weights_sha256(&self) -> &str {
        &self.registration.utility.initial_weights_sha256
    }
}
