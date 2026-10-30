//! No-execution bridge to the original numerical metadata validators.
//!
//! This does not enter the utility runtime, inspect model tensors or issue an
//! accepted historical source. Intervention collection remains separately
//! verified by its original file/ACK/native owners.

use super::{
    RoventUtilityCohortSourceRegistration, RoventUtilityParameterSpec, RoventUtilityTrainingRecipe,
};
use crate::ArenaError;
use rz_experiments::ManifestError;
use rz_experiments::rovent_utility_intervention_preparation::{
    RoventUtilityInterventionParts, RoventUtilityInterventionPreparation,
    RoventUtilityInterventionSource,
};

pub type RoventUtilityInterventionTrainingPreparation = RoventUtilityInterventionPreparation<
    RoventUtilityTrainingRecipe,
    RoventUtilityCohortSourceRegistration,
    RoventUtilityParameterSpec,
>;

impl RoventUtilityInterventionParts<RoventUtilityTrainingRecipe, RoventUtilityParameterSpec>
    for RoventUtilityCohortSourceRegistration
{
    fn validate_parts(
        &self,
        recipe: &RoventUtilityTrainingRecipe,
        parameter_spec: &RoventUtilityParameterSpec,
    ) -> Result<(), ManifestError> {
        // This original validator already calls recipe.validate(), validates
        // the exact source/cost domain, checks the frozen base configuration,
        // and joins the original parameter-spec seal and optimizer owner.
        self.validate(recipe, parameter_spec).map_err(|error| {
            ManifestError::Integrity(format!("original utility metadata: {error}"))
        })
    }
}

/// Prepare exact metadata only. The original numerical declarations remain
/// their original types and domains; there is no automatic V1/V2 conversion.
pub fn prepare_utility_intervention_training(
    recipe: &RoventUtilityTrainingRecipe,
    registration: &RoventUtilityCohortSourceRegistration,
    parameter_spec: &RoventUtilityParameterSpec,
    intervention_source: RoventUtilityInterventionSource,
) -> Result<RoventUtilityInterventionTrainingPreparation, ArenaError> {
    RoventUtilityInterventionTrainingPreparation::prepare(
        recipe.clone(),
        registration.clone(),
        parameter_spec.clone(),
        intervention_source,
    )
    .map_err(ArenaError::Manifest)
}

/// Strict production metadata entry. It uses the concrete original recipe,
/// registration and spec bridge above rather than an unvalidated Value body.
/// Success still grants no execution, source, target or optimizer authority.
pub fn validate_utility_intervention_preparation(
    raw: &str,
) -> Result<RoventUtilityInterventionTrainingPreparation, ArenaError> {
    RoventUtilityInterventionTrainingPreparation::from_json(raw).map_err(ArenaError::Manifest)
}

#[cfg(test)]
mod tests {
    use super::super::{
        prepare_utility_cohort_training, validate_utility_cohort_preparation,
        validate_utility_training_preparation,
    };
    use super::*;
    use serde_json::json;

    fn prepared() -> RoventUtilityInterventionTrainingPreparation {
        // Original pure metadata only, not a loaded source, tensor or corpus.
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
        let registration: RoventUtilityCohortSourceRegistration = serde_json::from_value(json!({
            "collection_source_sha256":"10".repeat(32),"trainer_source_sha256":"32".repeat(32),
            "cost_epoch_sha256":"50".repeat(32),"resource_profile_sha256":"40".repeat(32),
            "cost_scope":super::super::ROVENT_UTILITY_COHORT_COST_SCOPE,"objective":"exclusive_wall_nanoseconds",
            "frozen_base":{"schema":super::super::ROVENT_UTILITY_BASE_SCHEMA,"semantics":super::super::ROVENT_UTILITY_BASE_SEMANTICS,
                "weights_sha256":"20".repeat(32),"config":config},
            "utility":{"schema":super::super::ROVENT_UTILITY_SCHEMA,"semantics":super::super::ROVENT_UTILITY_SEMANTICS,
                "initial_weights_sha256":"30".repeat(32),"base_weights_sha256":"20".repeat(32),
                "parameter_spec_sha256":spec.canonical_sha256(&base_config,&recipe.optimizer_owner).unwrap()}
        })).unwrap();
        let pc = |role: &str| {
            json!({"role":role,
            "artifact":{"path":format!("{role}.onnx"),"sha256":"4".repeat(64),"bytes":1024,
                "source":"https://example.test/model","license":"MIT"},
            "model_semantics_sha256":"5".repeat(64),"model_identity_sha256":"6".repeat(64),"encoding_sha256":"7".repeat(64)})
        };
        let source = serde_json::from_value(json!({
            "cohort_id":"1".repeat(64),"profile_sha256":"2".repeat(64),
            "publication":{"sha256":"3".repeat(64),"bytes":4096},"durable_ack":{"sha256":"4".repeat(64),"bytes":512},
            "neutral_input_sha256":"5".repeat(64),"neutral_target_sha256":"6".repeat(64),
            "neutral_context_sha256":"7".repeat(64),"root_rules_sha256":"8".repeat(64),
            "role_queries":[
                {"trial_sequence":1,"recipient":"route_p","question_sha256":"9".repeat(64),"view_sha256":"a".repeat(64),"full_view_sha256":"b".repeat(64)},
                {"trial_sequence":2,"recipient":"route_c","question_sha256":"c".repeat(64),"view_sha256":"d".repeat(64),"full_view_sha256":"e".repeat(64)}],
            "frozen_pc_sources":[pc("proposer"),pc("critic")],"ancestor_datasets":[],
            "known_pair_sha256":["f".repeat(64)],"all_pair_sha256":["f".repeat(64)]
        })).unwrap();
        prepare_utility_intervention_training(&recipe, &registration, &spec, source).unwrap()
    }

    #[test]
    fn original_typed_parts_roundtrip_without_new_execution_rights() {
        let value = prepared();
        let raw = String::from_utf8(value.canonical_bytes().unwrap()).unwrap();
        assert_eq!(
            validate_utility_intervention_preparation(&raw).unwrap(),
            value
        );
        assert!(!value.execution_authorized);
        assert_eq!(value.performed_optimizer_updates, 0);
        let legacy_cohort = prepare_utility_cohort_training(
            &value.recipe,
            &value.registration,
            &value.parameter_spec,
        )
        .unwrap();
        assert!(validate_utility_intervention_preparation(&legacy_cohort.to_string()).is_err());
        let current = serde_json::to_value(&value).unwrap();
        assert!(validate_utility_cohort_preparation(&current).is_err());
        assert!(validate_utility_training_preparation(&current).is_err());
    }

    #[test]
    fn original_parameter_and_source_validator_is_required_by_production_entry() {
        let value = prepared();
        let mut registration = value.registration.clone();
        registration.utility.base_weights_sha256 = "0".repeat(64);
        assert!(
            prepare_utility_intervention_training(
                &value.recipe,
                &registration,
                &value.parameter_spec,
                value.intervention_source.clone()
            )
            .is_err()
        );
        let mut spec = value.parameter_spec.clone();
        spec.optimizer_owner = "another-owner".into();
        assert!(
            prepare_utility_intervention_training(
                &value.recipe,
                &value.registration,
                &spec,
                value.intervention_source
            )
            .is_err()
        );
    }

    /// Deliberately ignored unless root registers an explicit managed output
    /// path for cross-language codec checking. This is a metadata fixture, not
    /// an actual intervention, native source, known target or training run.
    #[test]
    #[ignore = "requires an explicit managed cross-language fixture output path"]
    fn write_intervention_preparation_wire_fixture() {
        use std::io::Write;
        let path = std::env::var_os("RZ_ROVENT_UTILITY_INTERVENTION_WIRE_FIXTURE")
            .map(std::path::PathBuf::from)
            .expect("explicit managed output path");
        assert!(path.is_absolute());
        let value = prepared();
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        file.write_all(&value.canonical_bytes().unwrap()).unwrap();
        file.sync_all().unwrap();
    }
}
