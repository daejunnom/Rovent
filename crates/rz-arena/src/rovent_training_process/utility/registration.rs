//! N05 utility metadata, separate from the existing P/C/V and utility V1 codecs.
//!
//! These declarations mirror `rovent_utility_training.py`. Parsing, validating,
//! or hashing them does not verify a tensor, frozen state, source owner, actual
//! intake, optimizer step, checkpoint, or learned export. The process owner must
//! join these independent metadata pins to its original retained inputs.
use crate::ArenaError;
use rz_experiments::pals_data_v3::{generation_digest, generation_identifier, generation_sha};
use serde::de::{MapAccess, Visitor, value::MapAccessDeserializer};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeSet;

// A derived struct can also accept a sequence. F accepts dictionaries only,
// including nested objects. Keep duplicate/missing/unknown-field rejection in
// the derived private map decoder without converting through a lossy Value.
macro_rules! utility_object {
    ($(#[$meta:meta])* pub struct $name:ident {
        $($(#[$field_meta:meta])* pub $field:ident: $ty:ty,)*
    }) => {
        $(#[$meta])*
        #[derive(Clone, Debug, Eq, PartialEq, Serialize)]
        pub struct $name { $($(#[$field_meta])* pub $field: $ty,)* }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct Fields { $($(#[$field_meta])* $field: $ty,)* }
                struct Object;
                impl<'input> Visitor<'input> for Object {
                    type Value = Fields;
                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        formatter.write_str("an exact utility metadata object")
                    }
                    fn visit_map<M: MapAccess<'input>>(self, map: M) -> Result<Fields, M::Error> {
                        Fields::deserialize(MapAccessDeserializer::new(map))
                    }
                }
                let fields = deserializer.deserialize_map(Object)?;
                Ok(Self { $($field: fields.$field,)* })
            }
        }
    };
}

pub const ROVENT_UTILITY_RECIPE_DOMAIN: &str = "rz-rovent-utility-training-recipe/1";
pub const ROVENT_UTILITY_PREPARATION_DOMAIN: &str = "rz-rovent-utility-training-preparation/1";
pub const ROVENT_UTILITY_RECIPE_VERSION: &str = "actual-route-known-pairwise-logistic/1";
pub const ROVENT_UTILITY_BASE_SCHEMA: &str = "rz-nnuec-fp32-v2";
pub const ROVENT_UTILITY_BASE_SEMANTICS: &str =
    "board800-exactb192-causal-line-kind4-directedpair-originfirst-exactc256-fp32-v2";
pub const ROVENT_UTILITY_SCHEMA: &str = "rz-nnuec-utility-fp32-v1";
pub const ROVENT_UTILITY_SEMANTICS: &str = "nnuec-v2-targetbundle-risk-route-shadow-fp32-v1";
pub const ROVENT_UTILITY_COST_SCOPE: &str =
    "exclusive_submitted_global_target_route_and_descendants_with_required_drain";
pub const ROVENT_UTILITY_COST_OBJECTIVE: &str = "exclusive_wall_nanoseconds";
pub const ROVENT_UTILITY_MAX_PREPARATION_BYTES: usize = 64 * 1024;
pub const ROVENT_UTILITY_MAX_CONTEXTS: u64 = 8192;
pub const ROVENT_UTILITY_MAX_PAIR_ENTRIES_PER_CONTEXT: u64 = 64;
pub const ROVENT_UTILITY_PARAMETER_FIELDS: [&str; 4] = [
    "hidden_weights",
    "hidden_bias",
    "output_weights",
    "output_bias",
];
pub const ROVENT_UTILITY_FROZEN_COMPONENTS: [&str; 3] = ["nnuec_base", "p", "c"];

fn invalid(reason: impl Into<String>) -> ArenaError {
    ArenaError::Integrity(format!("Rovent utility metadata: {}", reason.into()))
}
fn check(condition: bool, reason: &str) -> Result<(), ArenaError> {
    if condition {
        Ok(())
    } else {
        Err(invalid(reason))
    }
}
fn identifier(value: &str) -> bool {
    generation_identifier(value) && !value.trim().is_empty()
}
fn exact_strings(actual: &[String], expected: &[&str]) -> bool {
    actual
        .iter()
        .map(String::as_str)
        .eq(expected.iter().copied())
}

/// F limits the sorted UTF-8 compact `[domain, value]` envelope as well as the
/// metadata itself. Only integer/string/boolean/container JSON is used here;
/// this is not a claim of cross-language arbitrary-float canonicalization.
fn bounded_sha256(domain: &str, value: &impl Serialize) -> Result<String, ArenaError> {
    let mut envelope =
        serde_json::to_value((domain, value)).map_err(|error| invalid(error.to_string()))?;
    envelope.sort_all_objects();
    let bytes = serde_json::to_vec(&envelope).map_err(|error| invalid(error.to_string()))?;
    check(
        bytes.len() <= ROVENT_UTILITY_MAX_PREPARATION_BYTES,
        "canonical metadata envelope exceeds 64KiB",
    )?;
    generation_digest(domain, value).map_err(ArenaError::Manifest)
}

utility_object! {
/// The eight original recipe fields. Decimal coefficients intentionally remain
/// strings; the existing training-profile f64-bits codec does not apply.
pub struct RoventUtilityTrainingRecipe {
    pub schema: String,
    pub version: String,
    pub loss: RoventUtilityLossRecipe,
    pub optimizer: RoventUtilityOptimizerRecipe,
    pub optimizer_owner: String,
    pub active_parameter_names: Vec<String>,
    pub frozen_components: Vec<String>,
    pub limits: RoventUtilityPreparationLimits,
}
}
utility_object! {
pub struct RoventUtilityLossRecipe {
    pub kind: String,
    pub route_actions: Vec<u8>,
    pub normalization: String,
    pub weight: String,
}
}
utility_object! {
pub struct RoventUtilityOptimizerRecipe {
    pub kind: String,
    pub learning_rate: String,
    pub betas: Vec<String>,
    pub epsilon: String,
    pub weight_decay: String,
    pub gradient_clip_norm: String,
    pub schedule: String,
}
}
utility_object! {
pub struct RoventUtilityPreparationLimits {
    pub max_contexts: u64,
    pub max_pair_entries_per_context: u64,
    pub max_preparation_bytes: u64,
}
}
impl RoventUtilityTrainingRecipe {
    pub fn fixed(optimizer_owner: impl Into<String>) -> Result<Self, ArenaError> {
        let result = Self {
            schema: ROVENT_UTILITY_RECIPE_DOMAIN.into(),
            version: ROVENT_UTILITY_RECIPE_VERSION.into(),
            loss: RoventUtilityLossRecipe {
                kind: "pairwise_logistic".into(),
                route_actions: vec![4, 5, 6],
                normalization: "mean_unique_known_pairs".into(),
                weight: "1.0".into(),
            },
            optimizer: RoventUtilityOptimizerRecipe {
                kind: "adamw".into(),
                learning_rate: "0.0001".into(),
                betas: vec!["0.9".into(), "0.999".into()],
                epsilon: "0.00000001".into(),
                weight_decay: "0.01".into(),
                gradient_clip_norm: "1.0".into(),
                schedule: "constant".into(),
            },
            optimizer_owner: optimizer_owner.into(),
            active_parameter_names: ROVENT_UTILITY_PARAMETER_FIELDS
                .iter()
                .map(|name| (*name).into())
                .collect(),
            frozen_components: ROVENT_UTILITY_FROZEN_COMPONENTS
                .iter()
                .map(|name| (*name).into())
                .collect(),
            limits: RoventUtilityPreparationLimits {
                max_contexts: ROVENT_UTILITY_MAX_CONTEXTS,
                max_pair_entries_per_context: ROVENT_UTILITY_MAX_PAIR_ENTRIES_PER_CONTEXT,
                max_preparation_bytes: ROVENT_UTILITY_MAX_PREPARATION_BYTES as u64,
            },
        };
        result.validate()?;
        Ok(result)
    }
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            identifier(&self.optimizer_owner),
            "bounded nonblank optimizer owner",
        )?;
        check(
            self.schema == ROVENT_UTILITY_RECIPE_DOMAIN
                && self.version == ROVENT_UTILITY_RECIPE_VERSION
                && self.loss.kind == "pairwise_logistic"
                && self.loss.route_actions == [4, 5, 6]
                && self.loss.normalization == "mean_unique_known_pairs"
                && self.loss.weight == "1.0"
                && self.optimizer.kind == "adamw"
                && self.optimizer.learning_rate == "0.0001"
                && exact_strings(&self.optimizer.betas, &["0.9", "0.999"])
                && self.optimizer.epsilon == "0.00000001"
                && self.optimizer.weight_decay == "0.01"
                && self.optimizer.gradient_clip_norm == "1.0"
                && self.optimizer.schedule == "constant"
                && exact_strings(
                    &self.active_parameter_names,
                    &ROVENT_UTILITY_PARAMETER_FIELDS,
                )
                && exact_strings(&self.frozen_components, &ROVENT_UTILITY_FROZEN_COMPONENTS)
                && self.limits.max_contexts == ROVENT_UTILITY_MAX_CONTEXTS
                && self.limits.max_pair_entries_per_context
                    == ROVENT_UTILITY_MAX_PAIR_ENTRIES_PER_CONTEXT
                && self.limits.max_preparation_bytes == ROVENT_UTILITY_MAX_PREPARATION_BYTES as u64,
            "recipe differs from the registered fixed utility recipe",
        )
    }
    pub fn canonical_sha256(&self) -> Result<String, ArenaError> {
        self.validate()?;
        bounded_sha256(ROVENT_UTILITY_RECIPE_DOMAIN, self)
    }
}

utility_object! {
/// Eight actual frozen-base configuration integers, not loaded weights.
pub struct RoventUtilityBaseConfig {
    pub board_width: u32,
    pub line_width: u32,
    pub context_width: u32,
    pub local_width: u32,
    pub hidden_width: u32,
    pub causal_window: u32,
    pub max_lines: u32,
    pub max_plies: u32,
}
}
impl RoventUtilityBaseConfig {
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            [
                self.board_width,
                self.line_width,
                self.context_width,
                self.local_width,
            ]
            .into_iter()
            .all(|width| (1..=256).contains(&width))
                && (1..=64).contains(&self.hidden_width)
                && matches!(self.causal_window, 3 | 5)
                && (1..=128).contains(&self.max_lines)
                && (1..=256).contains(&self.max_plies),
            "bounded frozen V2 base configuration required",
        )
    }
    pub fn utility_input_width(&self) -> Result<u32, ArenaError> {
        self.validate()?;
        // Each positive width is at most 256, so this sum cannot overflow u32.
        Ok(2 * self.board_width + 2 * self.context_width + self.local_width)
    }
}

utility_object! {
pub struct RoventUtilityParameterNames {
    pub hidden_weights: String,
    pub hidden_bias: String,
    pub output_weights: String,
    pub output_bias: String,
}
}
impl RoventUtilityParameterNames {
    fn values(&self) -> [&str; 4] {
        [
            &self.hidden_weights,
            &self.hidden_bias,
            &self.output_weights,
            &self.output_bias,
        ]
    }
}
utility_object! {
pub struct RoventUtilityParameter {
    pub checkpoint_path: String,
    pub shape: Vec<u32>,
    pub dtype: String,
    pub planned_requires_grad: bool,
}
}
utility_object! {
pub struct RoventUtilityParameters {
    pub hidden_weights: RoventUtilityParameter,
    pub hidden_bias: RoventUtilityParameter,
    pub output_weights: RoventUtilityParameter,
    pub output_bias: RoventUtilityParameter,
}
}
impl RoventUtilityParameters {
    fn values(&self) -> [&RoventUtilityParameter; 4] {
        [
            &self.hidden_weights,
            &self.hidden_bias,
            &self.output_weights,
            &self.output_bias,
        ]
    }
}

utility_object! {
/// Six original membership fields. A wire-to-parameter name is a declaration,
/// not observed tensor membership. The actual owner must check named tensors,
/// storage aliasing, dtype, requires_grad, frozen components, and optimizer
/// membership again; a metadata name never proves those properties.
pub struct RoventUtilityParameterSpec {
    pub hidden_width: u32,
    pub input_width: u32,
    pub wire_to_parameter: RoventUtilityParameterNames,
    pub parameters: RoventUtilityParameters,
    pub optimizer_owner: String,
    pub frozen_components: Vec<String>,
}
}
impl RoventUtilityParameterSpec {
    pub fn validate(
        &self,
        frozen_base_config: &RoventUtilityBaseConfig,
        optimizer_owner: &str,
    ) -> Result<(), ArenaError> {
        let width = frozen_base_config.utility_input_width()?;
        check(
            identifier(optimizer_owner)
                && self.optimizer_owner == optimizer_owner
                && (1..=64).contains(&self.hidden_width)
                && self.input_width == width
                && exact_strings(&self.frozen_components, &ROVENT_UTILITY_FROZEN_COMPONENTS),
            "utility H/I, optimizer owner and frozen components must match",
        )?;
        let names = self.wire_to_parameter.values();
        check(
            names.into_iter().all(|name| {
                identifier(name)
                    && !matches!(
                        name.split('.').next().unwrap_or_default(),
                        "base" | "nnuec_base" | "p" | "c"
                    )
            }) && names.into_iter().collect::<BTreeSet<_>>().len() == 4,
            "four distinct non-frozen utility parameter names required",
        )?;
        let expected_shapes = [
            vec![self.hidden_width, width],
            vec![self.hidden_width],
            vec![7, self.hidden_width],
            vec![7],
        ];
        for ((name, parameter), shape) in ROVENT_UTILITY_PARAMETER_FIELDS
            .iter()
            .zip(self.parameters.values())
            .zip(expected_shapes)
        {
            check(
                parameter.checkpoint_path == format!("parameters.{name}")
                    && parameter.shape == shape
                    && parameter.dtype == "float32"
                    && parameter.planned_requires_grad,
                "only four exact FP32 private utility arrays may be declared",
            )?;
        }
        Ok(())
    }
    pub fn canonical_sha256(
        &self,
        frozen_base_config: &RoventUtilityBaseConfig,
        optimizer_owner: &str,
    ) -> Result<String, ArenaError> {
        self.validate(frozen_base_config, optimizer_owner)?;
        bounded_sha256(
            ROVENT_UTILITY_PREPARATION_DOMAIN,
            &json!({"kind":"utility_parameter_spec", "value":self}),
        )
    }
}

utility_object! {
pub struct RoventUtilityFrozenBase {
    pub schema: String,
    pub semantics: String,
    /// Original semantic parameter identity, separate from the base file's raw
    /// SHA and from the Python mapping observation.
    pub weights_sha256: String,
    pub config: RoventUtilityBaseConfig,
}
}
utility_object! {
pub struct RoventUtilityInitialParameters {
    pub schema: String,
    pub semantics: String,
    pub initial_weights_sha256: String,
    pub base_weights_sha256: String,
    /// Canonical parameter-spec metadata SHA, separate from semantic weights
    /// and the original raw asset-file pin.
    pub parameter_spec_sha256: String,
}
}

utility_object! {
/// The eight F source fields only. Collection/trainer source pins have separate
/// roles even if their byte values happen to agree. Cost/resource pins are SHA
/// strings, never a legacy integer epoch, current runtime, or execution lease.
/// Actual frozen P/C pins belong to the parent's separate execution envelope.
pub struct RoventUtilitySourceRegistration {
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
impl RoventUtilitySourceRegistration {
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
                && self.cost_scope == ROVENT_UTILITY_COST_SCOPE
                && self.objective == ROVENT_UTILITY_COST_OBJECTIVE
                && self.frozen_base.schema == ROVENT_UTILITY_BASE_SCHEMA
                && self.frozen_base.semantics == ROVENT_UTILITY_BASE_SEMANTICS
                && self.utility.schema == ROVENT_UTILITY_SCHEMA
                && self.utility.semantics == ROVENT_UTILITY_SEMANTICS,
            "explicit source/cost/resource pins and existing base/utility semantics required",
        )?;
        spec.validate(&self.frozen_base.config, &recipe.optimizer_owner)?;
        check(
            self.utility.base_weights_sha256 == self.frozen_base.weights_sha256
                && self.utility.parameter_spec_sha256
                    == spec.canonical_sha256(&self.frozen_base.config, &recipe.optimizer_owner)?,
            "initial utility must join the original frozen-base semantic identity and parameter spec",
        )
    }
    pub fn canonical_sha256(
        &self,
        recipe: &RoventUtilityTrainingRecipe,
        spec: &RoventUtilityParameterSpec,
    ) -> Result<String, ArenaError> {
        self.validate(recipe, spec)?;
        bounded_sha256(
            ROVENT_UTILITY_PREPARATION_DOMAIN,
            &json!({"kind":"source_registration", "value":self}),
        )
    }
}

#[cfg(test)]
mod tests {
    //! Pure metadata only. These fixtures do not construct models, execute an
    //! optimizer, read an asset, or issue source/native/learned capabilities.
    use super::*;
    use serde_json::Value;

    fn config() -> RoventUtilityBaseConfig {
        RoventUtilityBaseConfig {
            board_width: 2,
            line_width: 3,
            context_width: 4,
            local_width: 5,
            hidden_width: 6,
            causal_window: 3,
            max_lines: 64,
            max_plies: 256,
        }
    }
    fn spec() -> RoventUtilityParameterSpec {
        let parameter = |name: &str, shape: Vec<u32>| RoventUtilityParameter {
            checkpoint_path: format!("parameters.{name}"),
            shape,
            dtype: "float32".into(),
            planned_requires_grad: true,
        };
        RoventUtilityParameterSpec {
            hidden_width: 2,
            input_width: 17,
            wire_to_parameter: RoventUtilityParameterNames {
                hidden_weights: "hidden_weights".into(),
                hidden_bias: "hidden_bias".into(),
                output_weights: "output_weights".into(),
                output_bias: "output_bias".into(),
            },
            parameters: RoventUtilityParameters {
                hidden_weights: parameter("hidden_weights", vec![2, 17]),
                hidden_bias: parameter("hidden_bias", vec![2]),
                output_weights: parameter("output_weights", vec![7, 2]),
                output_bias: parameter("output_bias", vec![7]),
            },
            optimizer_owner: "utility-adamw-owner".into(),
            frozen_components: vec!["nnuec_base".into(), "p".into(), "c".into()],
        }
    }
    fn source() -> RoventUtilitySourceRegistration {
        let spec = spec();
        RoventUtilitySourceRegistration {
            collection_source_sha256: "10".repeat(32),
            trainer_source_sha256: "11".repeat(32),
            cost_epoch_sha256: "50".repeat(32),
            resource_profile_sha256: "40".repeat(32),
            cost_scope: ROVENT_UTILITY_COST_SCOPE.into(),
            objective: ROVENT_UTILITY_COST_OBJECTIVE.into(),
            frozen_base: RoventUtilityFrozenBase {
                schema: ROVENT_UTILITY_BASE_SCHEMA.into(),
                semantics: ROVENT_UTILITY_BASE_SEMANTICS.into(),
                weights_sha256: "20".repeat(32),
                config: config(),
            },
            utility: RoventUtilityInitialParameters {
                schema: ROVENT_UTILITY_SCHEMA.into(),
                semantics: ROVENT_UTILITY_SEMANTICS.into(),
                initial_weights_sha256: "30".repeat(32),
                base_weights_sha256: "20".repeat(32),
                parameter_spec_sha256: spec
                    .canonical_sha256(&config(), &spec.optimizer_owner)
                    .unwrap(),
            },
        }
    }
    fn recipe() -> RoventUtilityTrainingRecipe {
        RoventUtilityTrainingRecipe::fixed("utility-adamw-owner").unwrap()
    }

    #[test]
    fn decimal_recipe_and_original_source_fields_roundtrip_without_authority() {
        let recipe = recipe();
        let source = source();
        let spec = spec();
        source.validate(&recipe, &spec).unwrap();
        let raw = serde_json::to_string(&recipe).unwrap();
        assert_eq!(
            serde_json::from_str::<RoventUtilityTrainingRecipe>(&raw).unwrap(),
            recipe
        );
        let value = serde_json::to_value(&recipe).unwrap();
        assert_eq!(value.as_object().unwrap().len(), 8);
        assert_eq!(value["optimizer"]["learning_rate"], "0.0001");
        assert_eq!(value["loss"]["weight"], "1.0");
        let raw = serde_json::to_string(&source).unwrap();
        assert_eq!(
            serde_json::from_str::<RoventUtilitySourceRegistration>(&raw).unwrap(),
            source
        );
        let value = serde_json::to_value(&source).unwrap();
        assert_eq!(value.as_object().unwrap().len(), 8);
        assert!(value.get("source_sha256").is_none());
        assert!(value.get("cost_epoch").is_none());
        assert!(value.get("execution_authorized").is_none());
        let value = serde_json::to_value(&spec).unwrap();
        assert_eq!(value.as_object().unwrap().len(), 6);
        assert_eq!(
            serde_json::from_value::<RoventUtilityParameterSpec>(value).unwrap(),
            spec
        );
    }

    #[test]
    fn numeric_alias_unknown_field_and_changed_fixed_recipe_are_rejected() {
        let base = serde_json::to_value(recipe()).unwrap();
        for (path, replacement) in [
            (vec!["optimizer", "learning_rate"], json!(0.0001)),
            (vec!["loss", "weight"], json!(1)),
            (vec!["loss", "route_actions"], json!([4.0, 5, 6])),
            (vec!["limits", "max_contexts"], json!(8192.0)),
        ] {
            let mut value = base.clone();
            value[path[0]][path[1]] = replacement;
            assert!(serde_json::from_value::<RoventUtilityTrainingRecipe>(value).is_err());
        }
        let mut changed = recipe();
        changed.optimizer.learning_rate = "1e-4".into();
        assert!(changed.validate().is_err());
        changed = recipe();
        changed.active_parameter_names.swap(0, 1);
        assert!(changed.validate().is_err());
        changed = recipe();
        changed.optimizer_owner = " \t ".into();
        assert!(changed.validate().is_err());
        let mut value = base;
        value["runtime"] = json!({});
        assert!(serde_json::from_value::<RoventUtilityTrainingRecipe>(value).is_err());
    }

    #[test]
    fn every_metadata_object_rejects_sequence_and_duplicate_wire_forms() {
        let recipe_value = serde_json::to_value(recipe()).unwrap();
        let sequence = json!([
            recipe_value["schema"],
            recipe_value["version"],
            recipe_value["loss"],
            recipe_value["optimizer"],
            recipe_value["optimizer_owner"],
            recipe_value["active_parameter_names"],
            recipe_value["frozen_components"],
            recipe_value["limits"]
        ]);
        assert!(serde_json::from_value::<RoventUtilityTrainingRecipe>(sequence).is_err());
        let mut value = recipe_value;
        value["loss"] = json!([
            "pairwise_logistic",
            [4, 5, 6],
            "mean_unique_known_pairs",
            "1.0"
        ]);
        assert!(serde_json::from_value::<RoventUtilityTrainingRecipe>(value).is_err());
        let mut value = serde_json::to_value(spec()).unwrap();
        value["wire_to_parameter"] = json!([
            "hidden_weights",
            "hidden_bias",
            "output_weights",
            "output_bias"
        ]);
        assert!(serde_json::from_value::<RoventUtilityParameterSpec>(value).is_err());
        let mut value = serde_json::to_value(spec()).unwrap();
        value["parameters"]["hidden_weights"] =
            json!(["parameters.hidden_weights", [2, 17], "float32", true]);
        assert!(serde_json::from_value::<RoventUtilityParameterSpec>(value).is_err());
        let mut value = serde_json::to_value(source()).unwrap();
        value["frozen_base"]["config"] = json!([2, 3, 4, 5, 6, 3, 64, 256]);
        assert!(serde_json::from_value::<RoventUtilitySourceRegistration>(value).is_err());
        let raw = serde_json::to_string(&recipe()).unwrap();
        let duplicate = raw.replacen("\"schema\":", "\"schema\":\"duplicate\",\"schema\":", 1);
        assert!(serde_json::from_str::<RoventUtilityTrainingRecipe>(&duplicate).is_err());
    }

    #[test]
    fn unicode_owner_and_parameter_names_follow_the_original_utf8_contract() {
        let owner = "효용-소유자";
        let recipe = RoventUtilityTrainingRecipe::fixed(owner).unwrap();
        let mut spec = spec();
        spec.optimizer_owner = owner.into();
        spec.wire_to_parameter.hidden_weights = "효용.은닉_가중치".into();
        spec.validate(&config(), owner).unwrap();
        let mut source = source();
        source.utility.parameter_spec_sha256 = spec.canonical_sha256(&config(), owner).unwrap();
        source.validate(&recipe, &spec).unwrap();
        let raw = serde_json::to_string(&spec).unwrap();
        assert!(raw.contains("효용.은닉_가중치"));
        assert!(!raw.contains("\\u"));
        assert!(RoventUtilityTrainingRecipe::fixed("가".repeat(86)).is_err());
        assert!(RoventUtilityTrainingRecipe::fixed("owner\u{0085}").is_err());
        assert!(RoventUtilityTrainingRecipe::fixed(" \u{2003} ").is_err());
    }

    #[test]
    fn parameter_spec_rejects_frozen_alias_wrong_shape_dtype_and_owner() {
        let original = spec();
        original.validate(&config(), "utility-adamw-owner").unwrap();
        for name in ["base.hidden_weights", "nnuec_base.x", "p.x", "c.x"] {
            let mut changed = original.clone();
            changed.wire_to_parameter.hidden_weights = name.into();
            assert!(changed.validate(&config(), "utility-adamw-owner").is_err());
        }
        let mut changed = original.clone();
        changed.wire_to_parameter.hidden_bias = changed.wire_to_parameter.hidden_weights.clone();
        assert!(changed.validate(&config(), "utility-adamw-owner").is_err());
        changed = original.clone();
        changed.parameters.hidden_weights.shape = vec![2, 16];
        assert!(changed.validate(&config(), "utility-adamw-owner").is_err());
        changed = original.clone();
        changed.parameters.hidden_weights.dtype = "float16".into();
        assert!(changed.validate(&config(), "utility-adamw-owner").is_err());
        changed = original.clone();
        changed.parameters.output_bias.planned_requires_grad = false;
        assert!(changed.validate(&config(), "utility-adamw-owner").is_err());
        assert!(original.validate(&config(), "pc-shared-owner").is_err());
        let mut config = config();
        config.board_width = 3;
        assert!(original.validate(&config, "utility-adamw-owner").is_err());
    }

    #[test]
    fn original_source_cost_and_parameter_mapping_have_independent_seals() {
        let recipe = recipe();
        let spec = spec();
        let source = source();
        let original = source.canonical_sha256(&recipe, &spec).unwrap();
        for field in [
            "collection_source_sha256",
            "trainer_source_sha256",
            "cost_epoch_sha256",
            "resource_profile_sha256",
        ] {
            let mut changed = serde_json::to_value(&source).unwrap();
            changed[field] = json!("99".repeat(32));
            let changed =
                serde_json::from_value::<RoventUtilitySourceRegistration>(changed).unwrap();
            assert_ne!(changed.canonical_sha256(&recipe, &spec).unwrap(), original);
        }
        let mut same_pin = source.clone();
        same_pin.trainer_source_sha256 = same_pin.collection_source_sha256.clone();
        same_pin.validate(&recipe, &spec).unwrap();
        let mut invalid = source.clone();
        invalid.utility.base_weights_sha256 = "99".repeat(32);
        assert!(invalid.validate(&recipe, &spec).is_err());
        invalid = source.clone();
        invalid.utility.parameter_spec_sha256 = "99".repeat(32);
        assert!(invalid.validate(&recipe, &spec).is_err());
        let mut value = serde_json::to_value(&source).unwrap();
        value["cost_epoch_sha256"] = json!(3);
        assert!(serde_json::from_value::<RoventUtilitySourceRegistration>(value).is_err());
        let mut value = serde_json::to_value(&source).unwrap();
        value["source_sha256"] = value["collection_source_sha256"].clone();
        assert!(serde_json::from_value::<RoventUtilitySourceRegistration>(value).is_err());
    }

    #[test]
    fn canonical_metadata_uses_f_original_domain_envelopes() {
        let recipe = recipe();
        let spec = spec();
        let source = source();
        let expected_recipe: Value = json!({
            "schema":"rz-rovent-utility-training-recipe/1",
            "version":"actual-route-known-pairwise-logistic/1",
            "loss":{"kind":"pairwise_logistic", "route_actions":[4,5,6],
                "normalization":"mean_unique_known_pairs", "weight":"1.0"},
            "optimizer":{"kind":"adamw", "learning_rate":"0.0001", "betas":["0.9","0.999"],
                "epsilon":"0.00000001", "weight_decay":"0.01", "gradient_clip_norm":"1.0", "schedule":"constant"},
            "optimizer_owner":"utility-adamw-owner",
            "active_parameter_names":["hidden_weights","hidden_bias","output_weights","output_bias"],
            "frozen_components":["nnuec_base","p","c"],
            "limits":{"max_contexts":8192, "max_pair_entries_per_context":64, "max_preparation_bytes":65536}
        });
        assert_eq!(serde_json::to_value(&recipe).unwrap(), expected_recipe);
        assert_eq!(
            recipe.canonical_sha256().unwrap(),
            generation_digest(ROVENT_UTILITY_RECIPE_DOMAIN, &expected_recipe).unwrap()
        );
        assert_eq!(
            spec.canonical_sha256(&config(), &recipe.optimizer_owner)
                .unwrap(),
            generation_digest(
                ROVENT_UTILITY_PREPARATION_DOMAIN,
                &json!({"kind":"utility_parameter_spec", "value":spec})
            )
            .unwrap()
        );
        assert_eq!(
            source.canonical_sha256(&recipe, &spec).unwrap(),
            generation_digest(
                ROVENT_UTILITY_PREPARATION_DOMAIN,
                &json!({"kind":"source_registration", "value":source})
            )
            .unwrap()
        );
        assert!(
            bounded_sha256(
                ROVENT_UTILITY_PREPARATION_DOMAIN,
                &"x".repeat(ROVENT_UTILITY_MAX_PREPARATION_BYTES)
            )
            .is_err()
        );
    }
}
