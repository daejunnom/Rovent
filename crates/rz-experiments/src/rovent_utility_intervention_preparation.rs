//! Closed, metadata-only preparation for the explicit intervention family.
//!
//! The numerical recipe, registration and parameter specification remain in
//! their original owner crate. A typed bridge must invoke those original
//! validators; decoding this declaration never creates source authority,
//! tensor membership, a known target, an execution lease or an optimizer step.

use crate::pals_data_v3::{generation_digest, generation_sha};
use crate::{ArtifactRef, ManifestError, decode_json};
use serde::de::{DeserializeOwned, MapAccess, Visitor, value::MapAccessDeserializer};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use std::marker::PhantomData;

pub const ROVENT_UTILITY_INTERVENTION_PREPARATION_DOMAIN: &str =
    "rz-rovent-utility-intervention-preparation/1";
pub const ROVENT_UTILITY_INTERVENTION_PREPARATION_MAX_BYTES: usize = 64 << 10;

fn invalid(reason: &str) -> ManifestError {
    ManifestError::Integrity(format!("Rovent intervention preparation: {reason}"))
}
fn require(condition: bool, reason: &str) -> Result<(), ManifestError> {
    if condition {
        Ok(())
    } else {
        Err(invalid(reason))
    }
}

/// Enforce dictionary objects even where serde's derived struct decoder would
/// otherwise accept a positional array. Duplicate keys are rejected by the
/// outer decode_json traversal and by each typed map decoder.
fn object<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct Object<T>(PhantomData<T>);
    impl<'de, T: Deserialize<'de>> Visitor<'de> for Object<T> {
        type Value = T;
        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("an exact intervention metadata object")
        }
        fn visit_map<M: MapAccess<'de>>(self, map: M) -> Result<T, M::Error> {
            T::deserialize(MapAccessDeserializer::new(map))
        }
    }
    deserializer.deserialize_map(Object(PhantomData))
}

macro_rules! metadata_object {
    ($(#[$meta:meta])* pub struct $name:ident {
        $($(#[$field_meta:meta])* pub $field:ident: $ty:ty,)*
    }) => {
        $(#[$meta])*
        #[derive(Clone, Debug, Eq, PartialEq, Serialize)]
        pub struct $name { $($(#[$field_meta])* pub $field: $ty,)* }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct Fields { $($(#[$field_meta])* $field: $ty,)* }
                let fields: Fields = object(deserializer)?;
                Ok(Self { $($field: fields.$field,)* })
            }
        }
    };
}

metadata_object! {
/// Raw publication and durable-ACK file identities are separate. No filesystem
/// path, FD verification or completion capability is inferred from these pins.
pub struct RoventInterventionRawPin {
    pub sha256: String,
    pub bytes: u64,
}
}
impl RoventInterventionRawPin {
    fn validate(&self) -> Result<(), ManifestError> {
        require(
            generation_sha(&self.sha256)
                && (1..=ROVENT_UTILITY_INTERVENTION_PREPARATION_MAX_BYTES as u64)
                    .contains(&self.bytes),
            "positive bounded original publication/ACK pin required",
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum RoventInterventionRecipient {
    #[serde(rename = "route_p")]
    Proposer,
    #[serde(rename = "route_c")]
    Critic,
    #[serde(rename = "route_r")]
    Researcher,
}

metadata_object! {
pub struct RoventInterventionRoleQuery {
    pub trial_sequence: u64,
    pub recipient: RoventInterventionRecipient,
    pub question_sha256: String,
    pub view_sha256: String,
    pub full_view_sha256: String,
}
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoventInterventionFrozenPcRole {
    Proposer,
    Critic,
}

metadata_object! {
pub struct RoventInterventionFrozenPcSource {
    pub role: RoventInterventionFrozenPcRole,
    #[serde(deserialize_with = "object")]
    pub artifact: ArtifactRef,
    pub model_semantics_sha256: String,
    pub model_identity_sha256: String,
    pub encoding_sha256: String,
}
}

metadata_object! {
pub struct RoventInterventionCorpusAncestorPins {
    pub manifest_file_sha256: String,
    pub index_file_sha256: String,
    pub canonical_manifest_sha256: String,
    pub canonical_index_sha256: String,
}
}
metadata_object! {
pub struct RoventInterventionUtilityAncestorPins {
    pub dataset_sha256: String,
    pub pool_sha256: String,
    pub preparation_sha256: String,
    pub context_sha256: String,
}
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RoventInterventionAncestorSource {
    PcLegacy {
        publication_sha256: String,
        metadata: RoventInterventionCorpusAncestorPins,
    },
    PcSubmittedInputV1 {
        publication_sha256: String,
        metadata: RoventInterventionCorpusAncestorPins,
    },
    RAuxiliary {
        publication_sha256: String,
        metadata: RoventInterventionCorpusAncestorPins,
    },
    UtilityCohort {
        publication_sha256: String,
        metadata: RoventInterventionUtilityAncestorPins,
    },
    /// Separate /2 source family; never relabel an intervention as cohort /1.
    UtilityIntervention {
        publication_sha256: String,
        metadata: RoventInterventionUtilityAncestorPins,
    },
}
impl RoventInterventionAncestorSource {
    pub fn publication_sha256(&self) -> &str {
        match self {
            Self::PcLegacy {
                publication_sha256, ..
            }
            | Self::PcSubmittedInputV1 {
                publication_sha256, ..
            }
            | Self::RAuxiliary {
                publication_sha256, ..
            }
            | Self::UtilityCohort {
                publication_sha256, ..
            }
            | Self::UtilityIntervention {
                publication_sha256, ..
            } => publication_sha256,
        }
    }
    fn validate(&self) -> Result<(), ManifestError> {
        let pins = match self {
            Self::PcLegacy { metadata, .. }
            | Self::PcSubmittedInputV1 { metadata, .. }
            | Self::RAuxiliary { metadata, .. } => [
                &metadata.manifest_file_sha256,
                &metadata.index_file_sha256,
                &metadata.canonical_manifest_sha256,
                &metadata.canonical_index_sha256,
            ],
            Self::UtilityCohort { metadata, .. } | Self::UtilityIntervention { metadata, .. } => [
                &metadata.dataset_sha256,
                &metadata.pool_sha256,
                &metadata.preparation_sha256,
                &metadata.context_sha256,
            ],
        };
        require(
            generation_sha(self.publication_sha256())
                && pins.into_iter().all(|s| generation_sha(s)),
            "exact original ancestor source pins required",
        )
    }
}

metadata_object! {
/// Historical source declarations only. The owning reader must separately
/// verify original files, actual neutral/role queries, Rules, target/support,
/// frozen model sources and every ancestor's original historical owner.
pub struct RoventUtilityInterventionSource {
    pub cohort_id: String,
    pub profile_sha256: String,
    pub publication: RoventInterventionRawPin,
    pub durable_ack: RoventInterventionRawPin,
    pub neutral_input_sha256: String,
    pub neutral_target_sha256: String,
    pub neutral_context_sha256: String,
    pub root_rules_sha256: String,
    pub role_queries: Vec<RoventInterventionRoleQuery>,
    pub frozen_pc_sources: Vec<RoventInterventionFrozenPcSource>,
    pub ancestor_datasets: Vec<RoventInterventionAncestorSource>,
    pub known_pair_sha256: Vec<String>,
    pub all_pair_sha256: Vec<String>,
}
}
impl RoventUtilityInterventionSource {
    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            [
                &self.cohort_id,
                &self.profile_sha256,
                &self.neutral_input_sha256,
                &self.neutral_target_sha256,
                &self.neutral_context_sha256,
                &self.root_rules_sha256,
            ]
            .into_iter()
            .all(|s| generation_sha(s)),
            "invalid intervention source SHA",
        )?;
        self.publication.validate()?;
        self.durable_ack.validate()?;
        require(
            (2..=3).contains(&self.role_queries.len()),
            "two or three original role queries required",
        )?;
        let mut recipients = BTreeSet::new();
        for (index, query) in self.role_queries.iter().enumerate() {
            require(
                query.trial_sequence == index as u64 + 1
                    && recipients.insert(query.recipient)
                    && [
                        &query.question_sha256,
                        &query.view_sha256,
                        &query.full_view_sha256,
                    ]
                    .into_iter()
                    .all(|s| generation_sha(s)),
                "role query sequence/order/recipient/source mismatch",
            )?;
        }
        require(
            self.frozen_pc_sources.iter().map(|s| s.role).eq([
                RoventInterventionFrozenPcRole::Proposer,
                RoventInterventionFrozenPcRole::Critic,
            ]),
            "original proposer/critic frozen source order required",
        )?;
        for source in &self.frozen_pc_sources {
            source.artifact.validate()?;
            require(
                [
                    &source.model_semantics_sha256,
                    &source.model_identity_sha256,
                    &source.encoding_sha256,
                ]
                .into_iter()
                .all(|s| generation_sha(s)),
                "invalid frozen P/C source namespace",
            )?;
        }
        require(
            self.ancestor_datasets.len() <= 256,
            "ancestor source bound exceeded",
        )?;
        let mut publications = BTreeSet::new();
        for source in &self.ancestor_datasets {
            source.validate()?;
            require(
                source.publication_sha256() != self.publication.sha256
                    && publications.insert(source.publication_sha256()),
                "duplicate or current original ancestor publication",
            )?;
        }
        // Keep original publication order: these are complete pair-seal
        // rosters, not aggregate hashes or sorted replacement lists.
        for seals in [&self.known_pair_sha256, &self.all_pair_sha256] {
            require(
                seals.len() <= 3
                    && seals.iter().all(|seal| generation_sha(seal))
                    && seals.iter().collect::<BTreeSet<_>>().len() == seals.len(),
                "bounded unique original pair-seal lists required",
            )?;
        }
        require(
            self.known_pair_sha256
                .iter()
                .all(|seal| self.all_pair_sha256.contains(seal))
                && self.all_pair_sha256.len()
                    <= self.role_queries.len() * (self.role_queries.len() - 1) / 2,
            "known pair seals must retain their original complete pair coverage",
        )?;
        Ok(())
    }
}

/// Implemented by the original typed registration owner. No default, Value
/// fallback or permissive implementation exists in this crate. Implementations
/// must call the original recipe/registration/parameter-spec validators.
pub trait RoventUtilityInterventionParts<R, P> {
    fn validate_parts(&self, recipe: &R, parameter_spec: &P) -> Result<(), ManifestError>;
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RoventUtilityInterventionPreparation<R, S, P> {
    pub schema: String,
    pub status: String,
    pub recipe: R,
    pub registration: S,
    pub parameter_spec: P,
    pub intervention_source: RoventUtilityInterventionSource,
    pub preparation_sha256: String,
    pub execution_authorized: bool,
    pub performed_optimizer_updates: u64,
}
impl<'de, R, S, P> Deserialize<'de> for RoventUtilityInterventionPreparation<R, S, P>
where
    R: Deserialize<'de>,
    S: Deserialize<'de>,
    P: Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(
            deny_unknown_fields,
            bound(deserialize = "R: Deserialize<'de>, S: Deserialize<'de>, P: Deserialize<'de>")
        )]
        struct Fields<R, S, P> {
            schema: String,
            status: String,
            #[serde(deserialize_with = "object")]
            recipe: R,
            #[serde(deserialize_with = "object")]
            registration: S,
            #[serde(deserialize_with = "object")]
            parameter_spec: P,
            intervention_source: RoventUtilityInterventionSource,
            preparation_sha256: String,
            execution_authorized: bool,
            performed_optimizer_updates: u64,
        }
        let f: Fields<R, S, P> = object(deserializer)?;
        Ok(Self {
            schema: f.schema,
            status: f.status,
            recipe: f.recipe,
            registration: f.registration,
            parameter_spec: f.parameter_spec,
            intervention_source: f.intervention_source,
            preparation_sha256: f.preparation_sha256,
            execution_authorized: f.execution_authorized,
            performed_optimizer_updates: f.performed_optimizer_updates,
        })
    }
}

impl<R: Serialize, S: Serialize + RoventUtilityInterventionParts<R, P>, P: Serialize>
    RoventUtilityInterventionPreparation<R, S, P>
{
    pub fn prepare(
        recipe: R,
        registration: S,
        parameter_spec: P,
        intervention_source: RoventUtilityInterventionSource,
    ) -> Result<Self, ManifestError> {
        let mut result = Self {
            schema: ROVENT_UTILITY_INTERVENTION_PREPARATION_DOMAIN.into(),
            status: "prepared_metadata_only".into(),
            recipe,
            registration,
            parameter_spec,
            intervention_source,
            preparation_sha256: String::new(),
            execution_authorized: false,
            performed_optimizer_updates: 0,
        };
        result.validate_body()?;
        result.preparation_sha256 = result.unsigned_sha256()?;
        result.validate()?;
        Ok(result)
    }
    fn validate_body(&self) -> Result<(), ManifestError> {
        require(
            self.schema == ROVENT_UTILITY_INTERVENTION_PREPARATION_DOMAIN
                && self.status == "prepared_metadata_only"
                && !self.execution_authorized
                && self.performed_optimizer_updates == 0,
            "only explicit no-execution intervention preparation is permitted",
        )?;
        self.registration
            .validate_parts(&self.recipe, &self.parameter_spec)?;
        self.intervention_source.validate()
    }
    fn unsigned(&self) -> Result<Value, ManifestError> {
        let mut value =
            serde_json::to_value(self).map_err(|e| ManifestError::Parse(e.to_string()))?;
        value
            .as_object_mut()
            .ok_or_else(|| invalid("preparation object required"))?
            .remove("preparation_sha256");
        Ok(value)
    }
    fn unsigned_sha256(&self) -> Result<String, ManifestError> {
        let value = self.unsigned()?;
        let envelope =
            serde_json::to_value((ROVENT_UTILITY_INTERVENTION_PREPARATION_DOMAIN, &value))
                .map_err(|e| ManifestError::Parse(e.to_string()))?;
        bounded_canonical(envelope)?;
        generation_digest(ROVENT_UTILITY_INTERVENTION_PREPARATION_DOMAIN, &value)
    }
    pub fn validate(&self) -> Result<(), ManifestError> {
        self.validate_body()?;
        require(
            generation_sha(&self.preparation_sha256)
                && self.preparation_sha256 == self.unsigned_sha256()?,
            "preparation SHA differs from its exact unsigned eight-field body",
        )?;
        bounded_canonical(
            serde_json::to_value(self).map_err(|e| ManifestError::Parse(e.to_string()))?,
        )?;
        Ok(())
    }
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        self.validate()?;
        bounded_canonical(
            serde_json::to_value(self).map_err(|e| ManifestError::Parse(e.to_string()))?,
        )
    }
    /// The preparation seal uses sorted compact UTF-8 JSON [domain, unsigned
    /// eight-field body]. It is not the raw file SHA or a native NUL framing.
    pub fn canonical_sha256(&self) -> Result<String, ManifestError> {
        self.validate()?;
        self.unsigned_sha256()
    }
}
impl<
    R: Serialize + DeserializeOwned,
    S: Serialize + DeserializeOwned + RoventUtilityInterventionParts<R, P>,
    P: Serialize + DeserializeOwned,
> RoventUtilityInterventionPreparation<R, S, P>
{
    pub fn from_json(raw: &str) -> Result<Self, ManifestError> {
        require(
            raw.len() <= ROVENT_UTILITY_INTERVENTION_PREPARATION_MAX_BYTES,
            "preparation exceeds 64 KiB",
        )?;
        let result: Self = decode_json(raw)?;
        result.validate()?;
        Ok(result)
    }
}

fn bounded_canonical(mut value: Value) -> Result<Vec<u8>, ManifestError> {
    fn integer_metadata(value: &Value) -> bool {
        match value {
            Value::Number(n) => n.is_i64() || n.is_u64(),
            Value::Array(values) => values.iter().all(integer_metadata),
            Value::Object(values) => values.values().all(integer_metadata),
            _ => true,
        }
    }
    require(
        integer_metadata(&value),
        "metadata decimals must retain their original string representation",
    )?;
    value.sort_all_objects();
    let bytes = serde_json::to_vec(&value).map_err(|e| ManifestError::Parse(e.to_string()))?;
    require(
        bytes.len() <= ROVENT_UTILITY_INTERVENTION_PREPARATION_MAX_BYTES,
        "canonical metadata exceeds 64 KiB",
    )?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    //! Codec-only fixture parts. Production uses the Arena bridge to the
    //! original recipe/source/parameter validators, never these test types.
    use super::*;
    use serde_json::json;

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Part {
        original: u64,
    }
    impl RoventUtilityInterventionParts<Part, Part> for Part {
        fn validate_parts(&self, recipe: &Part, spec: &Part) -> Result<(), ManifestError> {
            require(
                self.original == 2 && recipe.original == 1 && spec.original == 3,
                "test original typed parts mismatch",
            )
        }
    }
    type Preparation = RoventUtilityInterventionPreparation<Part, Part, Part>;

    fn source() -> RoventUtilityInterventionSource {
        let pc = |role: &str| {
            json!({
                "role": role,
                "artifact": {"path": format!("{role}.onnx"), "sha256": "4".repeat(64),
                    "bytes": 1024, "source": "https://example.test/model", "license": "MIT"},
                "model_semantics_sha256": "5".repeat(64), "model_identity_sha256": "6".repeat(64),
                "encoding_sha256": "7".repeat(64),
            })
        };
        serde_json::from_value(json!({
            "cohort_id": "1".repeat(64), "profile_sha256": "2".repeat(64),
            "publication": {"sha256": "3".repeat(64), "bytes": 4096},
            "durable_ack": {"sha256": "4".repeat(64), "bytes": 512},
            "neutral_input_sha256": "5".repeat(64), "neutral_target_sha256": "6".repeat(64),
            "neutral_context_sha256": "7".repeat(64), "root_rules_sha256": "8".repeat(64),
            "role_queries": [
                {"trial_sequence": 1, "recipient": "route_p", "question_sha256": "9".repeat(64),
                    "view_sha256": "a".repeat(64), "full_view_sha256": "b".repeat(64)},
                {"trial_sequence": 2, "recipient": "route_c", "question_sha256": "c".repeat(64),
                    "view_sha256": "d".repeat(64), "full_view_sha256": "e".repeat(64)}],
            "frozen_pc_sources": [pc("proposer"), pc("critic")],
            "ancestor_datasets": [{"kind": "r_auxiliary", "publication_sha256": "a".repeat(64),
                "metadata": {"manifest_file_sha256": "b".repeat(64), "index_file_sha256": "c".repeat(64),
                    "canonical_manifest_sha256": "d".repeat(64), "canonical_index_sha256": "e".repeat(64)}}],
            "known_pair_sha256": ["f".repeat(64)], "all_pair_sha256": ["f".repeat(64)],
        })).unwrap()
    }
    fn prepared() -> Preparation {
        Preparation::prepare(
            Part { original: 1 },
            Part { original: 2 },
            Part { original: 3 },
            source(),
        )
        .unwrap()
    }

    #[test]
    fn exact_nine_fields_use_unsigned_eight_field_json_envelope() {
        let prepared = prepared();
        let canonical = prepared.canonical_bytes().unwrap();
        assert_eq!(
            Preparation::from_json(std::str::from_utf8(&canonical).unwrap()).unwrap(),
            prepared
        );
        let mut unsigned = serde_json::to_value(&prepared).unwrap();
        assert_eq!(unsigned.as_object().unwrap().len(), 9);
        assert_eq!(
            unsigned["intervention_source"].as_object().unwrap().len(),
            13
        );
        unsigned
            .as_object_mut()
            .unwrap()
            .remove("preparation_sha256");
        assert_eq!(unsigned.as_object().unwrap().len(), 8);
        let expected =
            generation_digest(ROVENT_UTILITY_INTERVENTION_PREPARATION_DOMAIN, &unsigned).unwrap();
        assert_eq!(prepared.canonical_sha256().unwrap(), expected);
        assert!(!prepared.execution_authorized);
        assert_eq!(prepared.performed_optimizer_updates, 0);
        assert_ne!(crate::digest(&canonical), prepared.preparation_sha256);
    }

    #[test]
    fn forbidden_execution_unknown_fields_and_old_domains_are_rejected() {
        for mutation in 0..8 {
            let mut value = serde_json::to_value(prepared()).unwrap();
            match mutation {
                0 => value["execution_authorized"] = json!(true),
                1 => value["performed_optimizer_updates"] = json!(1),
                2 => value["performed_optimizer_updates"] = json!(false),
                3 => value["source_authority_verified"] = json!(true),
                4 => value["schema"] = json!("rz-rovent-utility-training-preparation/1"),
                5 => value["schema"] = json!("rz-rovent-utility-training-preparation/2"),
                6 => value["recipe"]["unexpected"] = json!(0),
                7 => value["intervention_source"]["durable_ack"]["path"] = json!("invented.json"),
                _ => unreachable!(),
            }
            assert!(
                Preparation::from_json(&value.to_string()).is_err(),
                "mutation {mutation}"
            );
        }
    }

    #[test]
    fn duplicate_objects_and_positional_replacements_cannot_hide_fields() {
        let original = prepared().canonical_bytes().unwrap();
        let text = std::str::from_utf8(&original).unwrap();
        let duplicate = text.replacen(
            "\"execution_authorized\":false",
            "\"execution_authorized\":true,\"execution_authorized\":false",
            1,
        );
        assert!(Preparation::from_json(&duplicate).is_err());
        for pointer in [
            "",
            "/recipe",
            "/registration",
            "/parameter_spec",
            "/intervention_source",
            "/intervention_source/publication",
            "/intervention_source/role_queries/0",
            "/intervention_source/frozen_pc_sources/0/artifact",
            "/intervention_source/ancestor_datasets/0/metadata",
        ] {
            let mut value: Value = serde_json::from_slice(&original).unwrap();
            let target = value.pointer_mut(pointer).unwrap();
            let positional: Vec<_> = target.as_object().unwrap().values().cloned().collect();
            *target = Value::Array(positional);
            assert!(
                Preparation::from_json(&value.to_string()).is_err(),
                "pointer {pointer}"
            );
        }
    }

    #[test]
    fn changed_source_seals_order_and_original_parts_are_not_repaired() {
        let original = prepared();
        for mutation in 0..8 {
            let mut changed = original.clone();
            match mutation {
                0 => changed.intervention_source.neutral_target_sha256 = "0".repeat(64),
                1 => changed.intervention_source.durable_ack.sha256 = "0".repeat(64),
                2 => changed.intervention_source.role_queries.swap(0, 1),
                3 => {
                    changed.intervention_source.role_queries[1].recipient =
                        RoventInterventionRecipient::Proposer
                }
                4 => changed.intervention_source.frozen_pc_sources.swap(0, 1),
                5 => changed
                    .intervention_source
                    .ancestor_datasets
                    .push(changed.intervention_source.ancestor_datasets[0].clone()),
                6 => changed.registration.original = 4,
                7 => changed.parameter_spec.original = 5,
                _ => unreachable!(),
            }
            assert!(changed.validate().is_err(), "mutation {mutation}");
        }
        let raw = "x".repeat(ROVENT_UTILITY_INTERVENTION_PREPARATION_MAX_BYTES + 1);
        assert!(Preparation::from_json(&raw).is_err());
    }

    #[test]
    fn semantic_bridge_failure_is_not_replaced_by_a_default_success() {
        assert!(
            Preparation::prepare(
                Part { original: 1 },
                Part { original: 99 },
                Part { original: 3 },
                source()
            )
            .is_err()
        );
        let mut source = source();
        source.publication.bytes = 0;
        assert!(
            Preparation::prepare(
                Part { original: 1 },
                Part { original: 2 },
                Part { original: 3 },
                source
            )
            .is_err()
        );
    }

    #[test]
    fn pair_rosters_are_exact_unique_lists_and_known_is_an_original_subset() {
        let original = source();
        original.validate().unwrap();
        for mutation in 0..4 {
            let mut changed = original.clone();
            match mutation {
                0 => changed
                    .known_pair_sha256
                    .push(changed.known_pair_sha256[0].clone()),
                1 => changed
                    .all_pair_sha256
                    .push(changed.all_pair_sha256[0].clone()),
                2 => changed.known_pair_sha256[0] = "0".repeat(64),
                3 => changed.all_pair_sha256.push("0".repeat(64)),
                _ => unreachable!(),
            }
            assert!(
                changed.validate().is_err(),
                "pair roster mutation {mutation}"
            );
        }
        let mut raw = serde_json::to_value(prepared()).unwrap();
        raw["intervention_source"]["known_pair_sha256"] = json!("f".repeat(64));
        assert!(Preparation::from_json(&raw.to_string()).is_err());
    }

    #[test]
    fn current_intervention_publication_cannot_be_its_own_ancestor() {
        let mut changed = source();
        let publication = changed.publication.sha256.clone();
        let RoventInterventionAncestorSource::RAuxiliary {
            publication_sha256, ..
        } = &mut changed.ancestor_datasets[0]
        else {
            unreachable!()
        };
        *publication_sha256 = publication;
        assert!(changed.validate().is_err());
    }
}
