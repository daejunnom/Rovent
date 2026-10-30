//! Explicit cohort preparation V2 metadata. Root-local V1 cost observations
//! cannot be retagged, filled, or resigned into this family by a decoder.
//! A valid declaration is still not an actual cohort, known comparison,
//! physical-release observation, frozen tensor, execution or learned capability.
use super::registration::{
    ROVENT_UTILITY_BASE_SCHEMA, ROVENT_UTILITY_BASE_SEMANTICS, ROVENT_UTILITY_COST_OBJECTIVE,
    ROVENT_UTILITY_MAX_PREPARATION_BYTES, ROVENT_UTILITY_SCHEMA, ROVENT_UTILITY_SEMANTICS,
    RoventUtilityFrozenBase, RoventUtilityInitialParameters, RoventUtilityParameterSpec,
    RoventUtilityTrainingRecipe,
};
use crate::ArenaError;
use rz_experiments::pals_data_v3::{generation_digest, generation_sha};
use serde::de::{MapAccess, Visitor, value::MapAccessDeserializer};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub const ROVENT_UTILITY_COHORT_PREPARATION_DOMAIN: &str =
    "rz-rovent-utility-training-preparation/2";
pub const ROVENT_UTILITY_COHORT_COST_SCOPE: &str = "cohort_route_with_bounded_materialization_and_physical_pin_zero_excluding_durable_dataset_io_v1";

fn invalid(reason: impl Into<String>) -> ArenaError {
    ArenaError::Integrity(format!("Rovent utility cohort metadata: {}", reason.into()))
}
fn check(condition: bool, reason: &str) -> Result<(), ArenaError> {
    if condition {
        Ok(())
    } else {
        Err(invalid(reason))
    }
}
fn bounded_sha256(value: &impl Serialize) -> Result<String, ArenaError> {
    let mut envelope = serde_json::to_value((ROVENT_UTILITY_COHORT_PREPARATION_DOMAIN, value))
        .map_err(|error| invalid(error.to_string()))?;
    envelope.sort_all_objects();
    check(
        serde_json::to_vec(&envelope)
            .map_err(|error| invalid(error.to_string()))?
            .len()
            <= ROVENT_UTILITY_MAX_PREPARATION_BYTES,
        "cohort metadata digest envelope exceeds 64KiB",
    )?;
    generation_digest(ROVENT_UTILITY_COHORT_PREPARATION_DOMAIN, value).map_err(ArenaError::Manifest)
}

utility_object! {
/// Same eight source field names, with a separate exact cost scope and source
/// registration digest family. Semantic base parameter identities are distinct
/// from the parent's raw frozen-weight FilePin and current physical ownership.
pub struct RoventUtilityCohortSourceRegistration {
    pub collection_source_sha256: String,
    pub trainer_source_sha256: String,
    pub cost_epoch_sha256: String,
    pub resource_profile_sha256: String,
    pub cost_scope: String,
    pub objective: String,
    pub frozen_base: RoventUtilityFrozenBase,
    pub utility: RoventUtilityInitialParameters,
}
}
impl RoventUtilityCohortSourceRegistration {
    pub fn validate(
        &self,
        recipe: &RoventUtilityTrainingRecipe,
        spec: &RoventUtilityParameterSpec,
    ) -> Result<(), ArenaError> {
        recipe.validate()?;
        check(
            [
                &self.collection_source_sha256,
                &self.trainer_source_sha256,
                &self.cost_epoch_sha256,
                &self.resource_profile_sha256,
                &self.frozen_base.weights_sha256,
                &self.utility.initial_weights_sha256,
                &self.utility.base_weights_sha256,
                &self.utility.parameter_spec_sha256,
            ]
            .into_iter()
            .all(|pin| generation_sha(pin))
                && self.cost_scope == ROVENT_UTILITY_COHORT_COST_SCOPE
                && self.objective == ROVENT_UTILITY_COST_OBJECTIVE
                && self.frozen_base.schema == ROVENT_UTILITY_BASE_SCHEMA
                && self.frozen_base.semantics == ROVENT_UTILITY_BASE_SEMANTICS
                && self.utility.schema == ROVENT_UTILITY_SCHEMA
                && self.utility.semantics == ROVENT_UTILITY_SEMANTICS,
            "separate explicit cohort source/cost/resource and existing numerical semantics required",
        )?;
        spec.validate(&self.frozen_base.config, &recipe.optimizer_owner)?;
        check(
            self.utility.base_weights_sha256 == self.frozen_base.weights_sha256
                && self.utility.parameter_spec_sha256
                    == spec.canonical_sha256(&self.frozen_base.config, &recipe.optimizer_owner)?,
            "cohort utility changed semantic frozen-base identity or original parameter mapping",
        )
    }
    pub fn canonical_sha256(
        &self,
        recipe: &RoventUtilityTrainingRecipe,
        spec: &RoventUtilityParameterSpec,
    ) -> Result<String, ArenaError> {
        self.validate(recipe, spec)?;
        bounded_sha256(&json!({"kind":"source_registration", "value":self}))
    }
}

/// F's prepare_cohort_training metadata result only. Recipe and parameter-spec
/// seals retain their original V1 framing. Source registration and the complete
/// preparation use V2; no original V1 input is converted by this function.
pub fn prepare_utility_cohort_training(
    recipe: &RoventUtilityTrainingRecipe,
    registration: &RoventUtilityCohortSourceRegistration,
    spec: &RoventUtilityParameterSpec,
) -> Result<Value, ArenaError> {
    registration.validate(recipe, spec)?;
    let mut result = json!({
        "schema":ROVENT_UTILITY_COHORT_PREPARATION_DOMAIN,"status":"prepared_metadata_only",
        "recipe":recipe,"recipe_sha256":recipe.canonical_sha256()?,
        "registration":registration,"registration_sha256":registration.canonical_sha256(recipe,spec)?,
        "parameter_spec":spec,"parameter_spec_sha256":spec.canonical_sha256(&registration.frozen_base.config,&recipe.optimizer_owner)?,
        "optimizer_owner":recipe.optimizer_owner,"prepared_artifacts":["checkpoint_metadata","export_metadata"],
        "tensor_membership_verified":false,"source_authority_verified":false,
        "optimizer_state_present":false,"checkpoint_tensor_present":false,"export_tensor_present":false,
        "performed_optimizer_updates":0,"execution_authorized":false
    });
    let sha256 = bounded_sha256(&result)?;
    result["preparation_sha256"] = json!(sha256);
    check(
        serde_json::to_vec(&result)
            .map_err(|error| invalid(error.to_string()))?
            .len()
            <= ROVENT_UTILITY_MAX_PREPARATION_BYTES,
        "complete cohort preparation exceeds 64KiB",
    )?;
    Ok(result)
}

/// Exact V2 parser only. Supplying V1 metadata, a new scope string alone or
/// success flags cannot establish cohort lineage or a known utility pair.
pub fn validate_utility_cohort_preparation(preparation: &Value) -> Result<String, ArenaError> {
    let object = preparation
        .as_object()
        .ok_or_else(|| invalid("cohort preparation object required"))?;
    check(
        object.get("schema").and_then(Value::as_str)
            == Some(ROVENT_UTILITY_COHORT_PREPARATION_DOMAIN),
        "explicit cohort preparation V2 required",
    )?;
    let field = |name: &str| {
        object
            .get(name)
            .cloned()
            .ok_or_else(|| invalid(format!("missing cohort preparation {name}")))
    };
    let recipe: RoventUtilityTrainingRecipe =
        serde_json::from_value(field("recipe")?).map_err(|error| invalid(error.to_string()))?;
    let source: RoventUtilityCohortSourceRegistration =
        serde_json::from_value(field("registration")?)
            .map_err(|error| invalid(error.to_string()))?;
    let spec: RoventUtilityParameterSpec = serde_json::from_value(field("parameter_spec")?)
        .map_err(|error| invalid(error.to_string()))?;
    let expected = prepare_utility_cohort_training(&recipe, &source, &spec)?;
    check(
        *preparation == expected,
        "cohort preparation differs from its complete original V2 metadata result",
    )?;
    expected["preparation_sha256"]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| invalid("cohort preparation canonical SHA unavailable"))
}

#[cfg(test)]
mod tests {
    //! Pure versioned metadata fixtures, not evidence of actual cohort action,
    //! dataset I/O exclusion, physical pin zero, tensors or training.
    use super::super::registration::{
        ROVENT_UTILITY_PREPARATION_DOMAIN, RoventUtilitySourceRegistration,
    };
    use super::*;

    fn metadata() -> (
        RoventUtilityTrainingRecipe,
        RoventUtilityCohortSourceRegistration,
        RoventUtilityParameterSpec,
    ) {
        let recipe = RoventUtilityTrainingRecipe::fixed("utility-owner").unwrap();
        let config = json!({"board_width":2,"line_width":3,"context_width":4,"local_width":5,
            "hidden_width":6,"causal_window":3,"max_lines":64,"max_plies":256});
        let spec: RoventUtilityParameterSpec = serde_json::from_value(json!({
            "hidden_width":2,"input_width":17,"optimizer_owner":"utility-owner","frozen_components":["nnuec_base","p","c"],
            "wire_to_parameter":{"hidden_weights":"hidden_weights","hidden_bias":"hidden_bias","output_weights":"output_weights","output_bias":"output_bias"},
            "parameters":{
                "hidden_weights":{"checkpoint_path":"parameters.hidden_weights","shape":[2,17],"dtype":"float32","planned_requires_grad":true},
                "hidden_bias":{"checkpoint_path":"parameters.hidden_bias","shape":[2],"dtype":"float32","planned_requires_grad":true},
                "output_weights":{"checkpoint_path":"parameters.output_weights","shape":[7,2],"dtype":"float32","planned_requires_grad":true},
                "output_bias":{"checkpoint_path":"parameters.output_bias","shape":[7],"dtype":"float32","planned_requires_grad":true}
            }
        })).unwrap();
        let base_config = serde_json::from_value(config.clone()).unwrap();
        let source = serde_json::from_value(json!({
            "collection_source_sha256":"10".repeat(32),"trainer_source_sha256":"32".repeat(32),
            "cost_epoch_sha256":"50".repeat(32),"resource_profile_sha256":"40".repeat(32),
            "cost_scope":ROVENT_UTILITY_COHORT_COST_SCOPE,"objective":"exclusive_wall_nanoseconds",
            "frozen_base":{"schema":ROVENT_UTILITY_BASE_SCHEMA,"semantics":ROVENT_UTILITY_BASE_SEMANTICS,
                "weights_sha256":"20".repeat(32),"config":config},
            "utility":{"schema":ROVENT_UTILITY_SCHEMA,"semantics":ROVENT_UTILITY_SEMANTICS,
                "initial_weights_sha256":"30".repeat(32),"base_weights_sha256":"20".repeat(32),
                "parameter_spec_sha256":spec.canonical_sha256(&base_config,&recipe.optimizer_owner).unwrap()}
        })).unwrap();
        (recipe, source, spec)
    }

    #[test]
    fn cohort_preserves_recipe_and_parameter_domains_but_seals_new_source_and_preparation() {
        let (recipe, source, spec) = metadata();
        let result = prepare_utility_cohort_training(&recipe, &source, &spec).unwrap();
        assert_eq!(result["schema"], ROVENT_UTILITY_COHORT_PREPARATION_DOMAIN);
        assert_eq!(result["recipe_sha256"], recipe.canonical_sha256().unwrap());
        assert_eq!(
            result["parameter_spec_sha256"],
            spec.canonical_sha256(&source.frozen_base.config, &recipe.optimizer_owner)
                .unwrap()
        );
        assert_eq!(
            source.canonical_sha256(&recipe, &spec).unwrap(),
            generation_digest(
                ROVENT_UTILITY_COHORT_PREPARATION_DOMAIN,
                &json!({"kind":"source_registration","value":source})
            )
            .unwrap()
        );
        assert_ne!(
            result["parameter_spec_sha256"],
            generation_digest(
                ROVENT_UTILITY_COHORT_PREPARATION_DOMAIN,
                &json!({"kind":"utility_parameter_spec","value":spec})
            )
            .unwrap()
        );
        assert_eq!(
            validate_utility_cohort_preparation(&result).unwrap(),
            result["preparation_sha256"].as_str().unwrap()
        );
        assert_eq!(
            serde_json::to_value(&source)
                .unwrap()
                .as_object()
                .unwrap()
                .len(),
            8
        );
        assert_eq!(
            serde_json::from_value::<RoventUtilityCohortSourceRegistration>(
                serde_json::to_value(&source).unwrap()
            )
            .unwrap(),
            source
        );
    }

    #[test]
    fn original_v1_scope_domain_or_success_flags_do_not_enter_cohort_metadata() {
        let (recipe, source, spec) = metadata();
        let original = prepare_utility_cohort_training(&recipe, &source, &spec).unwrap();
        let mut changed = original.clone();
        changed["schema"] = json!(ROVENT_UTILITY_PREPARATION_DOMAIN);
        assert!(validate_utility_cohort_preparation(&changed).is_err());
        changed = original.clone();
        changed["registration"]["cost_scope"] =
            json!(super::super::registration::ROVENT_UTILITY_COST_SCOPE);
        assert!(validate_utility_cohort_preparation(&changed).is_err());
        changed = original.clone();
        changed["source_authority_verified"] = json!(true);
        assert!(validate_utility_cohort_preparation(&changed).is_err());
        changed = original.clone();
        changed["performed_optimizer_updates"] = json!(1);
        assert!(validate_utility_cohort_preparation(&changed).is_err());
        changed = original;
        changed["registration_sha256"] = json!("ff".repeat(32));
        assert!(validate_utility_cohort_preparation(&changed).is_err());
        let legacy: RoventUtilitySourceRegistration =
            serde_json::from_value(serde_json::to_value(source).unwrap()).unwrap();
        assert!(legacy.validate(&recipe, &spec).is_err());
    }

    #[test]
    fn cohort_source_keeps_strict_object_fields_and_independent_pins() {
        let (recipe, source, spec) = metadata();
        let mut changed = source.clone();
        changed.trainer_source_sha256 = source.collection_source_sha256.clone();
        changed.validate(&recipe, &spec).unwrap();
        changed = source.clone();
        changed.utility.parameter_spec_sha256 = "ff".repeat(32);
        assert!(changed.validate(&recipe, &spec).is_err());
        let mut raw = serde_json::to_value(source).unwrap();
        raw["actual_cohort_verified"] = json!(true);
        assert!(serde_json::from_value::<RoventUtilityCohortSourceRegistration>(raw).is_err());
        assert!(
            serde_json::from_value::<RoventUtilityCohortSourceRegistration>(json!([])).is_err()
        );
    }
}
