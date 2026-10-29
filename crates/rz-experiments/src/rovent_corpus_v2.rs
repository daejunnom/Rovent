//! 새 submitted-input/1과 raw/3만 소비하는 명시 Rovent 자료 형식이다.
//! 기존 manifest/index/record /1을 자동 변환하거나 재라벨하지 않는다.
use crate::ManifestError;
use crate::pals_data_v3::*;
use crate::rovent_corpus::{
    HEAD_DOMAIN, MAX_CORPUS_ROWS, ORIGIN_DOMAIN, RoventCorpusChunk, RoventCorpusManifest,
    RoventCorpusOrigin, RoventHeadSource, RoventLineage, RoventNativeLifecycle, RoventTeacherPin,
};
use crate::rovent_submitted_input::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

fn exact_fields(value: &serde_json::Value, names: &[&str]) -> bool {
    value.as_object().is_some_and(|object| {
        object.len() == names.len() && names.iter().all(|name| object.contains_key(*name))
    })
}

/// Raw JSON spelling is sealed as a string outside the submitted-input f32
/// domain. Decode once without silently overwriting duplicate nested keys.
struct UniqueVisibleInputJson(serde_json::Value);
impl<'de> Deserialize<'de> for UniqueVisibleInputJson {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = UniqueVisibleInputJson;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("bounded JSON without duplicate object keys")
            }
            fn visit_bool<E: serde::de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(UniqueVisibleInputJson(serde_json::Value::Bool(value)))
            }
            fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(UniqueVisibleInputJson(serde_json::Value::Number(
                    value.into(),
                )))
            }
            fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(UniqueVisibleInputJson(serde_json::Value::Number(
                    value.into(),
                )))
            }
            fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Self::Value, E> {
                let number = serde_json::Number::from_f64(value)
                    .ok_or_else(|| E::custom("non-finite original visible JSON number"))?;
                Ok(UniqueVisibleInputJson(serde_json::Value::Number(number)))
            }
            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(UniqueVisibleInputJson(serde_json::Value::String(
                    value.into(),
                )))
            }
            fn visit_string<E: serde::de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(UniqueVisibleInputJson(serde_json::Value::String(value)))
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(UniqueVisibleInputJson(serde_json::Value::Null))
            }
            fn visit_none<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                self.visit_unit()
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut out = Vec::new();
                while let Some(value) = seq.next_element::<UniqueVisibleInputJson>()? {
                    out.push(value.0);
                }
                Ok(UniqueVisibleInputJson(serde_json::Value::Array(out)))
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                let mut out = serde_json::Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    if out.contains_key(&key) {
                        return Err(serde::de::Error::custom(
                            "duplicate original visible input JSON key",
                        ));
                    }
                    let value = map.next_value::<UniqueVisibleInputJson>()?;
                    out.insert(key, value.0);
                }
                Ok(UniqueVisibleInputJson(serde_json::Value::Object(out)))
            }
        }
        deserializer.deserialize_any(Visitor)
    }
}

/// The returned input is a pure decoded value. The caller keeps the unchanged
/// original UTF-8 string in its outer receipt; this function issues no authority.
pub fn parse_rovent_visible_input_json(
    value: &str,
) -> Result<RoventSubmittedInputV1, ManifestError> {
    ensure(
        !value.is_empty() && value.len() <= GENERATION_MAX_ROW_BYTES,
        "bounded original visible input JSON string",
    )?;
    let decoded: UniqueVisibleInputJson = serde_json::from_str(value)
        .map_err(|e| ManifestError::Integrity(format!("original visible input JSON: {e}")))?;
    let input: RoventSubmittedInputV1 =
        serde_json::from_value(decoded.0).map_err(|e| ManifestError::Integrity(e.to_string()))?;
    input.validate()?;
    Ok(input)
}

#[cfg(test)]
mod visible_input_json_tests {
    use super::*;

    #[test]
    fn nested_and_escaped_duplicate_keys_are_not_overwritten() {
        for raw in [
            r#"{"encoded":{"metadata":[{"x":1,"x":2}]}}"#,
            r#"{"encoded":{"\u0078":1,"x":2}}"#,
        ] {
            let error = serde_json::from_str::<UniqueVisibleInputJson>(raw)
                .err()
                .unwrap()
                .to_string();
            assert!(error.contains("duplicate original visible input JSON key"));
        }
    }
    #[test]
    fn original_json_decoder_rejects_trailing_documents_and_nonfinite_numbers() {
        for raw in [r#"{"x":1}{"x":2}"#, r#"{"x":NaN}"#, r#"{"x":1e9999}"#] {
            assert!(serde_json::from_str::<UniqueVisibleInputJson>(raw).is_err());
        }
    }
    #[test]
    fn original_json_decoder_keeps_utf8_integer_and_negative_zero_values() {
        let decoded: UniqueVisibleInputJson =
            serde_json::from_str(r#"{"text":"검토","integer":18446744073709551615,"value":-0.0}"#)
                .unwrap();
        assert_eq!(decoded.0["text"], "검토");
        assert_eq!(decoded.0["integer"].as_u64(), Some(u64::MAX));
        assert_eq!(
            decoded.0["value"].as_f64().unwrap().to_bits(),
            (-0.0f64).to_bits()
        );
    }
    #[test]
    fn visible_input_json_is_bounded_before_decode() {
        for raw in [String::new(), " ".repeat(GENERATION_MAX_ROW_BYTES + 1)] {
            assert!(
                parse_rovent_visible_input_json(&raw)
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("bounded original visible input JSON string")
            );
        }
    }
}

pub const ROVENT_EXTERNAL_COLLECTION_ROOT: &str = "rz-rovent-collection-root-admission/2";
pub const ROVENT_PROFILED_EXTERNAL_SOURCE: &str = "rz-rovent-collection-profiled-external-source/1";

/// raw3 전용 명시 판별이다. 원 root1을 넓히거나 메타데이터에서 실행 권한을
/// 복원하지 않는다. 공식 엔진 profile과 전체 lock의 대조는 역사 owner가 맡는다.
pub fn validate_submitted_root_admission(
    root: &serde_json::Value,
    expected_sha256: &serde_json::Value,
) -> Result<(), ManifestError> {
    let common = [
        "schema",
        "native_origin",
        "native_origin_sha256",
        "process_epoch",
        "game_generation",
        "root_generation",
        "implementation_sha256",
        "deadline_clock",
        "deadline_tick",
        "root_budget_sha256",
        "selective_policy_sha256",
        "root_state",
        "store_root_state",
        "root_moves",
        "depth_plies",
        "mate_moves",
        "requested_nodes",
        "root_descriptor",
        "root_descriptor_sha256",
    ];
    let domain = root["schema"].as_str().unwrap_or("");
    let mut roster = common.to_vec();
    match domain {
        "rz-rovent-collection-root-admission/1" => {}
        ROVENT_EXTERNAL_COLLECTION_ROOT => {
            roster.extend(["external_source", "external_source_sha256"]);
            let source = &root["external_source"];
            ensure(
                exact_fields(
                    source,
                    &[
                        "schema",
                        "root_source_sequence",
                        "worker_owner",
                        "clock_epoch",
                        "authority",
                        "engine_profile",
                        "purpose",
                        "process",
                        "profile",
                        "profile_file_sha256",
                        "profile_file_bytes",
                        "profile_canonical_sha256",
                        "profile_capability_sha256",
                        "binary",
                        "startup_input_lock_sha256",
                        "observed_uci",
                        "options_application_observed",
                        "model_loading_observed",
                    ],
                ),
                "external collection source exact roster",
            )?;
            let authority = &source["authority"];
            let process = &source["process"];
            let binary: crate::ArtifactRef = serde_json::from_value(source["binary"].clone())
                .map_err(|error| ManifestError::Integrity(error.to_string()))?;
            binary.validate()?;
            ensure(
                source["schema"] == ROVENT_PROFILED_EXTERNAL_SOURCE
                    && root["external_source_sha256"]
                        == generation_digest(ROVENT_PROFILED_EXTERNAL_SOURCE, source)?
                    && source["root_source_sequence"].as_u64().is_some()
                    && source["worker_owner"]
                        .as_u64()
                        .is_some_and(|n| n <= u16::MAX as u64)
                    && source["clock_epoch"] == root["deadline_clock"]
                    && exact_fields(
                        authority,
                        &[
                            "process_epoch",
                            "game_generation",
                            "root_generation",
                            "implementation_sha256",
                        ],
                    )
                    && [
                        "process_epoch",
                        "game_generation",
                        "root_generation",
                        "implementation_sha256",
                    ]
                    .into_iter()
                    .all(|field| authority[field] == root[field])
                    && source["purpose"] == "research_checker"
                    && source["profile"]["purpose"] == source["purpose"]
                    && source["profile"]["engine"] == source["engine_profile"]
                    && matches!(
                        source["engine_profile"].as_str(),
                        Some("stockfish19_embedded_nnue" | "reckless090_embedded_nnue")
                    )
                    && source["profile"]["identity"]["binary_sha256"] == binary.sha256
                    && exact_fields(process, &["pid", "process_group", "proc_start_ticks"])
                    && ["pid", "process_group"].into_iter().all(|field| {
                        process[field]
                            .as_u64()
                            .is_some_and(|n| n > 0 && n <= u32::MAX as u64)
                    })
                    && process["proc_start_ticks"].as_u64().is_some_and(|n| n > 0)
                    && source["profile_file_bytes"]
                        .as_u64()
                        .is_some_and(|n| n > 0 && n <= 64 << 10)
                    && [
                        "profile_file_sha256",
                        "profile_canonical_sha256",
                        "profile_capability_sha256",
                        "startup_input_lock_sha256",
                    ]
                    .into_iter()
                    .all(|field| source[field].as_str().is_some_and(generation_sha))
                    && source["startup_input_lock_sha256"]
                        == root["native_origin"]["input_lock_sha256"]
                    && exact_fields(&source["observed_uci"], &["name", "author"])
                    && source["observed_uci"]["name"] == source["profile"]["expected_uci_name"]
                    && source["observed_uci"]["name"]
                        .as_str()
                        .is_some_and(|s| !s.is_empty() && s.len() <= 256)
                    && (source["observed_uci"]["author"].is_null()
                        || source["observed_uci"]["author"]
                            .as_str()
                            .is_some_and(|s| s.len() <= 256))
                    && source["options_application_observed"].is_null()
                    && source["model_loading_observed"].is_null()
                    && matches!(
                        root["native_origin"]["mode"].as_str(),
                        Some("HybridPcr" | "CpuPcr")
                    ),
                "external collection source/root/process/profile evidence join",
            )?;
        }
        _ => {
            return Err(ManifestError::Integrity(
                "unsupported raw3 collection root domain".into(),
            ));
        }
    }
    ensure(
        exact_fields(root, &roster) && *expected_sha256 == generation_digest(domain, root)?,
        "raw3 exact collection root roster/domain seal",
    )
}

impl RoventNativeLifecycle {
    /// raw/3는 raw/1 또는 raw/2 decoder를 넓히지 않고 독립적으로 선택한다.
    /// 공통 실제 physical owner chain 계산만 기존 구현을 재사용한다.
    pub fn validate_submitted_input_v3(
        &self,
        manifest: &RoventCorpusManifest,
    ) -> Result<(), ManifestError> {
        self.validate_submitted_input_for_domain(
            manifest,
            ROVENT_NATIVE_RAW_V3_DOMAIN,
            validate_submitted_root_admission,
            |root| root["native_origin"]["purpose"] == "collection",
        )
    }

    /// Shared receipt checks only. The caller must supply the exact raw
    /// domain and root validator for its own separately registered family.
    /// It does not recreate a live native, root, or physical authority.
    pub(crate) fn validate_submitted_input_for_domain<F, P>(
        &self,
        manifest: &RoventCorpusManifest,
        domain: &str,
        validate_root: F,
        root_is_collection: P,
    ) -> Result<(), ManifestError>
    where
        F: FnOnce(&serde_json::Value, &serde_json::Value) -> Result<(), ManifestError>,
        P: FnOnce(&serde_json::Value) -> bool,
    {
        self.validate_domain(manifest, domain)?;
        for (raw, names) in [
            (
                &self.raw_submission,
                &[
                    "schema",
                    "kind",
                    "sequence",
                    "process_epoch",
                    "game_generation",
                    "root_generation",
                    "request_sequence",
                    "head",
                    "source_sha256",
                    "model_sha256",
                    "encoding_sha256",
                    "input_digest",
                    "view_digest",
                    "input_sha256",
                    "ticket",
                    "rules_replay",
                    "submitted_review",
                ][..],
            ),
            (
                &self.raw_completion,
                &[
                    "schema",
                    "kind",
                    "sequence",
                    "process_epoch",
                    "game_generation",
                    "root_generation",
                    "request_sequence",
                    "head",
                    "source_sha256",
                    "input_sha256",
                    "submitted_sha256",
                    "execution_epoch",
                    "execution_sequence",
                    "completed_nn_inputs",
                    "physical_confirmed",
                    "owner_run_invoked",
                    "measurement_complete",
                    "physical_fresh",
                    "output_digest",
                    "owner_service_elapsed_ns",
                    "ticket",
                    "output",
                ][..],
            ),
            (
                &self.raw_retirement,
                &[
                    "schema",
                    "kind",
                    "sequence",
                    "process_epoch",
                    "game_generation",
                    "root_generation",
                    "request_sequence",
                    "head",
                    "source_sha256",
                    "input_sha256",
                    "submitted_sha256",
                    "completed_sha256",
                    "output_digest",
                    "accepted_retired",
                    "ticket",
                    "acceptance_kind",
                    "root_admission",
                    "root_admission_sha256",
                ][..],
            ),
        ] {
            ensure(
                exact_fields(raw, names),
                "explicit raw3 original field roster",
            )?;
        }
        let root = &self.raw_retirement["root_admission"];
        validate_root(root, &self.raw_retirement["root_admission_sha256"])?;
        ensure(
            matches!(self.head.as_str(), "proposer" | "critic")
                && self.raw_submission["rules_replay"]["schema"]
                    == "rz-rovent-submitted-rules-replay/1"
                && root_is_collection(root)
                && matches!(
                    self.raw_retirement["acceptance_kind"].as_str(),
                    Some("accepted_role_step" | "root_ranking_published")
                )
                && (self.raw_retirement["acceptance_kind"] != "root_ranking_published"
                    || self.head == "proposer"),
            "raw3 original collection/Rules/accepted owner chain",
        )?;
        let review = &self.raw_submission["submitted_review"];
        ensure(
            exact_fields(
                review,
                &[
                    "schema",
                    "sequence",
                    "generation",
                    "root_state",
                    "policy",
                    "input_sha256",
                    "view_seal_sha256",
                    "record_scope_sha256",
                    "anchor_states",
                    "source_anchors",
                    "record_metadata",
                ],
            ) && review["schema"] == "rz-rovent-submitted-review-evidence/1"
                && review["sequence"] == self.raw_submission["ticket"]["view_sequence"]
                && review["generation"].as_u64() == Some(self.root_generation)
                && review["root_state"] == root["store_root_state"]
                && review["view_seal_sha256"] == self.view_digest
                && ["input_sha256", "view_seal_sha256", "record_scope_sha256"]
                    .iter()
                    .all(|field| review[*field].as_str().is_some_and(generation_sha)),
            "raw3 exact submitted ReviewView evidence join",
        )?;
        let metadata = review["record_metadata"]
            .as_array()
            .ok_or_else(|| ManifestError::Integrity("submitted record metadata array".into()))?;
        let sources = review["source_anchors"].as_array().ok_or_else(|| {
            ManifestError::Integrity("original record source snapshots array".into())
        })?;
        ensure(
            metadata.len() <= 64
                && sources.len() == metadata.len()
                && metadata.len()
                    == self.raw_submission["ticket"]["selected_records"]
                        .as_u64()
                        .unwrap_or(u64::MAX) as usize,
            "raw3 source metadata/snapshot exact coverage",
        )?;
        for (index, meta) in metadata.iter().enumerate() {
            ensure(
                exact_fields(
                    meta,
                    &[
                        "local_index",
                        "record_revision",
                        "parent_revision",
                        "supersedes_revision",
                        "origin_state",
                        "source_state",
                        "source_revision",
                        "source_line_offset",
                        "original_first_move",
                        "value",
                        "completed_depth",
                        "score_scope",
                        "critical",
                    ],
                ) && meta["local_index"].as_u64() == Some(index as u64)
                    && meta["record_revision"].as_u64().is_some_and(|v| v > 0)
                    && meta["source_revision"].as_u64().is_some_and(|v| v > 0)
                    && meta["origin_state"].is_u64()
                    && meta["source_state"].is_u64()
                    && meta["source_line_offset"]
                        .as_u64()
                        .is_some_and(|v| v <= 256)
                    && (meta["value"].is_null()
                        || meta["value"]
                            .as_i64()
                            .is_some_and(|value| i32::try_from(value).is_ok()))
                    && meta["completed_depth"]
                        .as_u64()
                        .is_some_and(|value| u16::try_from(value).is_ok())
                    && (meta["score_scope"].is_null()
                        || matches!(
                            meta["score_scope"].as_str(),
                            Some("FrontierOnly" | "CompletedIteration" | "RulesTerminal")
                        ))
                    && meta["critical"].is_boolean()
                    && sources[index]["schema"] == "rz-rovent-rules-position-replay/1",
                "raw3 source metadata occurrence and exact replay domain",
            )?;
        }
        Ok(())
    }
}

pub const ROVENT_DIVERGENCE_TARGET_DOMAIN: &str = "rz-rovent-divergence-target/1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventCorpusManifestV2 {
    pub schema: String,
    pub corpus_id: String,
    pub source_commit: String,
    pub source_sha256: String,
    pub runtime_sha256: String,
    pub model_semantics_sha256: String,
    pub model_epoch: u64,
    pub context_epoch: u64,
    pub recipe_sha256: String,
    pub stage_id: String,
    pub round: u64,
    pub round_id: String,
    pub student: String,
    pub learner: String,
    pub one_learner: bool,
    pub inherited_input_sha256: Vec<String>,
    pub inherited_origin_sha256: Vec<String>,
    pub teachers: BTreeMap<String, RoventTeacherPin>,
    pub opening_splits: BTreeMap<String, GenerationSplitV3>,
    pub ancestor_index_sha256: Vec<String>,
}
impl RoventCorpusManifestV2 {
    /// 등록 필드의 기존 순수 검증만 재사용한다. 이 projection을 legacy manifest
    /// 자산으로 발행하거나 기존 index/record에 연결하지 않는다.
    pub fn common_registration_projection(&self) -> RoventCorpusManifest {
        RoventCorpusManifest {
            schema: crate::rovent_corpus::CORPUS_DOMAIN.into(),
            corpus_id: self.corpus_id.clone(),
            source_commit: self.source_commit.clone(),
            source_sha256: self.source_sha256.clone(),
            runtime_sha256: self.runtime_sha256.clone(),
            model_semantics_sha256: self.model_semantics_sha256.clone(),
            model_epoch: self.model_epoch,
            context_epoch: self.context_epoch,
            recipe_sha256: self.recipe_sha256.clone(),
            stage_id: self.stage_id.clone(),
            round: self.round,
            round_id: self.round_id.clone(),
            student: self.student.clone(),
            learner: self.learner.clone(),
            one_learner: self.one_learner,
            inherited_input_sha256: self.inherited_input_sha256.clone(),
            inherited_origin_sha256: self.inherited_origin_sha256.clone(),
            teachers: self.teachers.clone(),
            opening_splits: self.opening_splits.clone(),
            ancestor_index_sha256: self.ancestor_index_sha256.clone(),
        }
    }
    pub fn validate(&self) -> Result<(), ManifestError> {
        ensure(
            self.schema == ROVENT_CORPUS_V2_DOMAIN,
            "explicit Rovent manifest/2 required",
        )?;
        self.common_registration_projection().validate()
    }
    pub fn sha256(&self) -> Result<String, ManifestError> {
        self.validate()?;
        generation_digest(ROVENT_CORPUS_V2_DOMAIN, self)
    }
}

/// counter local index는 counter_question_id의 실제 view namespace에 속한다.
/// 과거 target 입력에 counter가 없던 경우 미래 source view를 과거 view로
/// 재태그하지 않으며 원 opaque 공격→counter causal witness를 추가 요구한다.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventDivergenceTargetV1 {
    pub schema: String,
    pub observed_sequence: u64,
    pub question_id: String,
    pub view_sha256: String,
    pub raw_evidence_sha256: String,
    pub source: String,
    pub slot: u16,
    pub attack_ply: u16,
    pub counter_question_id: String,
    pub counter_native_view_sha256: String,
    pub counter_local_index: u16,
    pub counter_revision: u64,
    pub repair_question_id: String,
    pub status: GenerationDivergenceStatusV3,
}
impl RoventDivergenceTargetV1 {
    /// 형식·고정 입력 대조만 수행한다. raw owner witness의 실제 인과/수락은
    /// producer와 historical 전체 native 기록 join에서 별도로 검증한다.
    pub fn value(&self, input: &RoventSubmittedInputV1) -> Result<f64, ManifestError> {
        input.validate()?;
        let mapping = input
            .attack_mapping
            .get(usize::from(self.slot))
            .ok_or_else(|| {
                ManifestError::Integrity("target attack slot absent from submitted input".into())
            })?;
        ensure(
            self.schema == ROVENT_DIVERGENCE_TARGET_DOMAIN
                && input.role == GenerationRoleV3::Critic
                && self.source == "rovent_resolved_pc_divergence"
                && self.observed_sequence >= input.capture_sequence
                && self.question_id == input.question_id
                && self.view_sha256 == input.view_sha256
                && generation_sha(&self.raw_evidence_sha256)
                && generation_sha(&self.counter_question_id)
                && generation_sha(&self.counter_native_view_sha256)
                && generation_sha(&self.repair_question_id)
                && mapping.slot == self.slot
                && mapping.attack_ply == self.attack_ply
                && self.counter_local_index < 64
                && self.counter_revision > 0,
            "resolved divergence must join its original slot and distinct counter source namespace",
        )?;
        if let Some(index) = mapping.selected_counter_local_index {
            ensure(
                self.counter_question_id == input.question_id
                    && self.counter_local_index == index
                    && mapping.selected_counter_revision == Some(self.counter_revision),
                "visible source counter differs from submitted mapping",
            )?;
        }
        Ok(
            if self.status == GenerationDivergenceStatusV3::SupportedAfterRepair {
                1.0
            } else {
                0.0
            },
        )
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventTargetsV2 {
    pub policy: Option<GenerationPolicyTargetV3>,
    pub wdl: Option<GenerationWdlTargetV3>,
    pub divergence: Option<RoventDivergenceTargetV1>,
    /// 네 새 제품 profile은 V-free다. 기존 V 자료/codec/optimizer는 별도 보존한다.
    pub task: Option<GenerationTaskTargetV3>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventTrainingRecordV2 {
    pub schema: String,
    pub role: GenerationRoleV3,
    pub split: GenerationSplitV3,
    pub input: RoventSubmittedInputV1,
    pub input_sha256: String,
    pub origin_sha256: String,
    pub record_sha256: String,
    pub lineage: RoventLineage,
    pub native: RoventNativeLifecycle,
    pub head_sources: BTreeMap<String, RoventHeadSource>,
    pub targets: RoventTargetsV2,
    pub divergence_index: Option<u16>,
    // 실제 R/NNUEC 입력은 별도 R 자료로 보존한다. PC 과거 입력에 합치지 않는다.
    pub nnuec_input: Option<serde_json::Value>,
    pub nnuec_input_sha256: Option<String>,
    pub nnuec_model_sha256: Option<String>,
    pub nnuec_semantics: Option<String>,
    pub nnuec_wdl: Option<GenerationWdlTargetV3>,
}
impl RoventTrainingRecordV2 {
    pub fn origin(&self, manifest: &RoventCorpusManifestV2) -> Result<String, ManifestError> {
        generation_digest(
            ORIGIN_DOMAIN,
            &serde_json::json!({
            "source_sha256":manifest.source_sha256,"recipe_sha256":manifest.recipe_sha256,
            "stage_id":manifest.stage_id,"round":manifest.round,"round_id":manifest.round_id,
            "student":manifest.student,"learner":manifest.learner,
            "actor_id":self.lineage.actor_id,"teacher_actor_id":self.lineage.teacher_actor_id,
            "input_sha256":self.input_sha256,"game_id":self.input.game_id,"opening_id":self.input.opening_id,
            "line_genealogy_id":self.input.line_genealogy_id,"transposition_sha256":self.input.transposition_sha256,
            "ancestor_origin_sha256":self.lineage.ancestor_origin_sha256}),
        )
    }
    pub fn sha256(&self) -> Result<String, ManifestError> {
        self.sha256_for_domain(ROVENT_TRAINING_RECORD_V2_DOMAIN)
    }

    pub(crate) fn sha256_for_domain(&self, domain: &str) -> Result<String, ManifestError> {
        let mut raw =
            serde_json::to_value(self).map_err(|e| ManifestError::Integrity(e.to_string()))?;
        raw["record_sha256"] = serde_json::json!("");
        raw["input"] = serde_json::json!({"sealed_input_sha256":self.input_sha256});
        generation_digest(domain, &raw)
    }
    pub fn seal(&mut self, manifest: &RoventCorpusManifestV2) -> Result<(), ManifestError> {
        self.input_sha256 = self.input.sha256()?;
        self.origin_sha256 = self.origin(manifest)?;
        self.record_sha256 = self.sha256()?;
        self.validate(manifest)
    }
    pub fn validate(&self, manifest: &RoventCorpusManifestV2) -> Result<(), ManifestError> {
        manifest.validate()?;
        self.input.validate()?;
        self.native
            .validate_submitted_input_v3(&manifest.common_registration_projection())?;
        self.validate_record_body(manifest, ROVENT_TRAINING_RECORD_V2_DOMAIN)
    }

    /// Pure shared record/target checks after the caller validates its own
    /// explicit native lifecycle domain. No record is relabeled or published.
    pub(crate) fn validate_record_body(
        &self,
        manifest: &RoventCorpusManifestV2,
        record_domain: &str,
    ) -> Result<(), ManifestError> {
        ensure(
            self.schema == record_domain
                && self.role == self.input.role
                && self.input_sha256 == self.input.sha256()?
                && self.origin_sha256 == self.origin(manifest)?
                && self.record_sha256 == self.sha256_for_domain(record_domain)?
                && self.input.model_epoch == manifest.model_epoch
                && self.input.context_epoch == manifest.context_epoch
                && manifest.opening_splits.get(&self.input.opening_id) == Some(&self.split)
                && generation_identifier(&self.lineage.actor_id)
                && self.lineage.ancestor_origin_sha256.len() <= 256
                && self
                    .lineage
                    .ancestor_origin_sha256
                    .iter()
                    .all(|v| generation_sha(v))
                && self
                    .lineage
                    .ancestor_origin_sha256
                    .iter()
                    .collect::<BTreeSet<_>>()
                    .len()
                    == self.lineage.ancestor_origin_sha256.len()
                && self
                    .lineage
                    .teacher_actor_id
                    .as_ref()
                    .is_none_or(|actor| manifest.teachers.contains_key(actor)),
            "new record source/input/stage/split/origin seal",
        )?;
        ensure(
            self.native.head
                == match self.role {
                    GenerationRoleV3::Proposer => "proposer",
                    GenerationRoleV3::Critic => "critic",
                    GenerationRoleV3::Validator => {
                        return Err(ManifestError::Integrity("V-free new record".into()));
                    }
                }
                && self.native.submitted_sequence == self.input.capture_sequence
                && [
                    &self.native.raw_submission,
                    &self.native.raw_completion,
                    &self.native.raw_retirement,
                ]
                .iter()
                .all(|r| r["input_sha256"] == self.input_sha256)
                && self.native.raw_submission["ticket"]["question_digest"]
                    == self.input.question_id,
            "raw3 original native input/role/question join",
        )?;
        if let Some(policy) = &self.targets.policy {
            self.input.policy_probabilities(policy)?;
        }
        if let Some(wdl) = &self.targets.wdl {
            self.input.wdl_probabilities(wdl)?;
        }
        if let Some(divergence) = &self.targets.divergence {
            divergence.value(&self.input)?;
        }
        ensure(
            self.targets.task.is_none()
                && self.nnuec_input.is_none()
                && self.nnuec_input_sha256.is_none()
                && self.nnuec_model_sha256.is_none()
                && self.nnuec_semantics.is_none()
                && self.nnuec_wdl.is_none(),
            "new PC record does not manufacture V/R/NNUEC training inputs",
        )?;
        ensure(
            self.divergence_index == self.targets.divergence.as_ref().map(|target| target.slot),
            "new divergence index is the actual original head slot",
        )?;
        for (name, head) in &self.head_sources {
            ensure(
                ["policy", "wdl", "divergence"].contains(&name.as_str())
                    && head.anchor == self.input.rules_anchor
                    && head.observed_sequence >= self.input.capture_sequence
                    && generation_identifier(&head.actor_id)
                    && generation_identifier(&head.source)
                    && head.raw_receipt["input_sha256"] == self.input_sha256
                    && head.raw_receipt["anchor"]
                        == serde_json::to_value(&self.input.rules_anchor)
                            .map_err(|e| ManifestError::Integrity(e.to_string()))?
                    && generation_digest(HEAD_DOMAIN, &head.raw_receipt)?
                        == head.raw_receipt_sha256,
                "new record original head body/anchor/chronology seal",
            )?;
        }
        for (name, present) in [
            ("policy", self.targets.policy.is_some()),
            ("wdl", self.targets.wdl.is_some()),
            ("divergence", self.targets.divergence.is_some()),
        ] {
            ensure(
                !present || self.head_sources.contains_key(name),
                "new known target lacks original head source",
            )?;
        }
        // 원 head별 teacher/Rules/divergence owner 근거의 전체 검증은 아래
        // 공유 함수로 수행한다. metadata flag만으로 이 검사를 생략하지 않는다.
        validate_head_sources(self, manifest)?;
        ensure(
            serde_json::to_vec(self)
                .map_err(|e| ManifestError::Integrity(e.to_string()))?
                .len()
                <= GENERATION_MAX_ROW_BYTES,
            "new Rovent row byte cap",
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventCorpusIndexV2 {
    pub schema: String,
    pub manifest_sha256: String,
    pub record_count: u64,
    pub chunks: Vec<RoventCorpusChunk>,
    pub origins: Vec<RoventCorpusOrigin>,
}
impl RoventCorpusIndexV2 {
    pub fn validate(&self, manifest: &RoventCorpusManifestV2) -> Result<(), ManifestError> {
        manifest.validate()?;
        ensure(
            self.schema == ROVENT_INDEX_V2_DOMAIN
                && self.manifest_sha256 == manifest.sha256()?
                && self.record_count <= MAX_CORPUS_ROWS as u64
                && self.origins.len() as u64 == self.record_count
                && self
                    .chunks
                    .iter()
                    .try_fold(0u64, |sum, c| sum.checked_add(c.records))
                    .is_some_and(|n| n == self.record_count),
            "new corpus index domain/manifest/count",
        )?;
        crate::rovent_corpus::RoventCorpusIndex {
            schema: crate::rovent_corpus::INDEX_DOMAIN.into(),
            manifest_sha256: self.manifest_sha256.clone(),
            record_count: self.record_count,
            chunks: self.chunks.clone(),
            origins: self.origins.clone(),
        }
        .validate()
    }
    pub fn sha256(&self) -> Result<String, ManifestError> {
        generation_digest(ROVENT_INDEX_V2_DOMAIN, self)
    }
}

fn validate_teacher_input_binding(binding: &serde_json::Value, input_sha256: &str) -> bool {
    if binding["schema"] == "rz-rovent-collection-teacher-binding/1" {
        binding["input_scope"]["input_sha256"] == input_sha256
    } else {
        binding["input_sha256"] == input_sha256
    }
}

#[cfg(test)]
mod teacher_input_binding_tests {
    use super::validate_teacher_input_binding;
    use serde_json::json;

    #[test]
    fn collection_scope_is_bound_to_original_input_not_an_unrelated_field() {
        let input = "a".repeat(64);
        let other = "b".repeat(64);
        let mut binding = json!({
            "schema": "rz-rovent-collection-teacher-binding/1",
            "input_scope": { "input_sha256": input },
        });
        assert!(validate_teacher_input_binding(&binding, &input));
        binding["input_sha256"] = json!(input);
        binding["input_scope"]["input_sha256"] = json!(other);
        assert!(!validate_teacher_input_binding(&binding, &input));
        binding["input_scope"] = json!({});
        assert!(!validate_teacher_input_binding(&binding, &input));

        // Preserve the pre-existing top-level binding validator contract.
        // This fixture does not attest any real profiled teacher execution.
        let top_level = json!({"input_sha256": input});
        assert!(validate_teacher_input_binding(&top_level, &input));
        assert!(!validate_teacher_input_binding(&top_level, &other));
    }
}

fn validate_head_sources(
    row: &RoventTrainingRecordV2,
    manifest: &RoventCorpusManifestV2,
) -> Result<(), ManifestError> {
    for (name, label) in [
        (
            "policy",
            row.targets
                .policy
                .as_ref()
                .map(|t| (&t.raw_evidence_sha256, t.observed_sequence)),
        ),
        (
            "wdl",
            row.targets
                .wdl
                .as_ref()
                .map(|t| (&t.raw_evidence_sha256, t.observed_sequence)),
        ),
        (
            "divergence",
            row.targets
                .divergence
                .as_ref()
                .map(|t| (&t.raw_evidence_sha256, t.observed_sequence)),
        ),
    ] {
        if let Some((sha, sequence)) = label {
            let source = &row.head_sources[name];
            ensure(
                &source.raw_receipt_sha256 == sha && source.observed_sequence == sequence,
                "known target does not join its actual head receipt",
            )?;
        }
    }
    if let Some(policy) = &row.targets.policy {
        let head = &row.head_sources["policy"];
        let pin = manifest
            .teachers
            .get(&head.actor_id)
            .ok_or_else(|| ManifestError::Integrity("unknown frozen teacher actor".into()))?;
        let raw = &head.raw_receipt;
        // The collection teacher uses the actual submitted input scope; the
        // pre-existing validator also accepts a top-level input digest. Keep
        // the two shapes distinct instead of rejecting a genuine Collection
        // receipt because its digest is nested in the scope.
        ensure(
            row.lineage.teacher_actor_id.as_ref() == Some(&head.actor_id)
                && head.source == policy.source
                && head.binary_sha256.as_ref() == Some(&pin.binary_sha256)
                && head.profile_sha256.as_ref() == Some(&pin.profile_sha256)
                && policy.teacher_binary_sha256 == pin.binary_sha256
                && policy.teacher_profile_sha256 == pin.profile_sha256
                && raw["schema"] == HEAD_DOMAIN
                && raw["kind"] == "teacher_policy"
                && raw["actor_id"] == head.actor_id
                && raw["teacher_pin"]
                    == serde_json::to_value(pin)
                        .map_err(|e| ManifestError::Integrity(e.to_string()))?
                && validate_teacher_input_binding(&raw["binding"], &row.input_sha256),
            "new teacher policy frozen actor/profile/binary/original binding",
        )?;
    }
    if let Some(wdl) = &row.targets.wdl {
        let head = &row.head_sources["wdl"];
        let raw = &head.raw_receipt;
        let authority: GenerationTrajectoryAuthorityV3 =
            serde_json::from_value(raw["trajectory"].clone())
                .map_err(|e| ManifestError::Integrity(e.to_string()))?;
        authority.validate()?;
        ensure(
            head.source == wdl.source
                && raw["schema"] == HEAD_DOMAIN
                && raw["kind"] == "rules_wdl"
                && authority.game_id == row.input.game_id
                && authority.anchors.contains(&row.input.rules_anchor)
                && authority.ending == wdl.ending
                && authority.outcome == wdl.outcome
                && authority.sha256()? == wdl.trajectory_membership_sha256
                && authority.result_evidence_sha256 == wdl.rules_result_sha256
                && raw["rules_receipt"]["schema"] == "rz-pals-rules-outcome-evidence/1"
                && raw["rules_receipt"]["rules_terminal"] == true
                && raw["rules_receipt"]["game_id"] == row.input.game_id
                && raw["rules_receipt"]["ending"]
                    == serde_json::to_value(wdl.ending)
                        .map_err(|e| ManifestError::Integrity(e.to_string()))?
                && raw["rules_receipt"]["outcome"]
                    == serde_json::to_value(wdl.outcome)
                        .map_err(|e| ManifestError::Integrity(e.to_string()))?
                && generation_digest("rz-pals-rules-outcome-evidence/1", &raw["rules_receipt"])?
                    == authority.result_evidence_sha256,
            "new WDL exact actual trajectory/Rules result evidence",
        )?;
    }
    if let Some(head) = row.head_sources.get("divergence") {
        validate_divergence_roster(row, head)?;
        if let Some(target) = &row.targets.divergence {
            validate_divergence_raw(row, target, head)?;
        }
    }
    Ok(())
}

fn wdl_expectation(endpoint: &serde_json::Value) -> Result<f32, ManifestError> {
    let bits = endpoint["root_wdl_bits"]
        .as_array()
        .filter(|a| a.len() == 3)
        .ok_or_else(|| ManifestError::Integrity("actual Fresh root WDL bits".into()))?;
    let mut wdl = [0.0f32; 3];
    for (i, word) in bits.iter().enumerate() {
        wdl[i] = f32::from_bits(
            u32::try_from(
                word.as_u64()
                    .ok_or_else(|| ManifestError::Integrity("Fresh bit word".into()))?,
            )
            .map_err(|_| ManifestError::Integrity("Fresh bit word overflow".into()))?,
        );
    }
    ensure(
        wdl.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            && (wdl.iter().sum::<f32>() - 1.0).abs() <= 1e-4,
        "actual Fresh normalized finite WDL",
    )?;
    Ok(wdl[0] - wdl[2])
}

fn divergence_visible_source(source: &serde_json::Value) -> &serde_json::Value {
    if source["schema"] == "rz-rovent-pc-divergence-source-with-attack/1" {
        &source["visible_source"]
    } else {
        source
    }
}

fn validate_divergence_source_roster<'a>(
    row: &RoventTrainingRecordV2,
    raw: &'a serde_json::Value,
    observed_sequence: u64,
) -> Result<&'a serde_json::Value, ManifestError> {
    let source = divergence_visible_source(raw);
    let expected_native = if raw["schema"] == "rz-rovent-pc-divergence-source-with-attack/1" {
        let attack = &raw["attack_ranked_source"];
        let evidence = &raw["visible_record_evidence"];
        ensure(
            exact_fields(
                raw,
                &[
                    "schema",
                    "native_source_sha256",
                    "visible_source",
                    "visible_source_sequence",
                    "attack_ranked_source",
                    "visible_record_evidence",
                ],
            ) && raw["native_source_sha256"]
                .as_str()
                .is_some_and(generation_sha)
                && exact_fields(
                    attack,
                    &[
                        "schema",
                        "native_attack_sha256",
                        "accepted",
                        "root_budget_sha256",
                        "source_sequence",
                        "divergence_index",
                        "divergence_ply",
                        "first_move",
                        "proposal",
                        "perspective",
                    ],
                )
                && attack["schema"] == "rz-rovent-pc-attack-ranked-source/1"
                && attack["native_attack_sha256"]
                    .as_str()
                    .is_some_and(generation_sha)
                && attack["source_sequence"].as_u64().is_some_and(|v| v > 0)
                && raw["visible_source_sequence"]
                    .as_u64()
                    .zip(attack["source_sequence"].as_u64())
                    .is_some_and(|(visible, initial)| initial < visible)
                && attack["accepted"]["ticket"] == row.native.raw_submission["ticket"]
                && attack["accepted"]["output_digest"] == row.native.output_digest
                && exact_fields(evidence, &["input_json", "input_sha256", "native"]),
            "new initial attack explicit original/future source projection roster",
        )?;
        let input =
            parse_rovent_visible_input_json(evidence["input_json"].as_str().ok_or_else(|| {
                ManifestError::Integrity("original visible input JSON string required".into())
            })?)?;
        let native: RoventNativeLifecycle = serde_json::from_value(evidence["native"].clone())
            .map_err(|e| ManifestError::Integrity(e.to_string()))?;
        input.validate()?;
        ensure(
            evidence["input_sha256"] == input.sha256()?
                && input.role == GenerationRoleV3::Critic
                && native.head == "critic"
                && native.submitted_sequence == input.capture_sequence
                && row.input.capture_sequence < input.capture_sequence
                && observed_sequence >= native.retired_sequence
                && native.raw_submission["ticket"]["question_digest"] == input.question_id
                && input.game_id == row.input.game_id
                && input.opening_id == row.input.opening_id
                && input.actual_history == row.input.actual_history
                && input.rules_anchor == row.input.rules_anchor
                && input.model_epoch == row.input.model_epoch
                && input.context_epoch == row.input.context_epoch
                && native.raw_retirement["root_admission"]
                    == row.native.raw_retirement["root_admission"],
            "new future visible original submitted seal/root scope",
        )?;
        Some(&evidence["native"])
    } else {
        // The absent-attack source has no schema discriminator and retains its
        // original exact roster and native source digest unchanged.
        None
    };
    let (ticket, output) = if let Some(expected_native) = expected_native {
        (
            &expected_native["raw_submission"]["ticket"],
            expected_native["output_digest"].clone(),
        )
    } else {
        (
            &row.native.raw_submission["ticket"],
            serde_json::Value::String(row.native.output_digest.clone()),
        )
    };
    ensure(
        exact_fields(
            source,
            &[
                "native_source_sha256",
                "accepted",
                "root_budget_sha256",
                "counter_local_index",
                "counter_revision",
                "divergence_index",
                "divergence_ply",
                "first_move",
                "proposal",
                "counter",
                "perspective",
            ],
        ) && source["native_source_sha256"]
            .as_str()
            .is_some_and(generation_sha)
            && source["root_budget_sha256"]
                == row.native.raw_retirement["root_admission"]["root_budget_sha256"]
            && source["accepted"]["ticket"] == *ticket
            && source["accepted"]["output_digest"] == output,
        "new divergence original visible source exact roster/input/output scope",
    )?;
    Ok(source)
}

fn validate_divergence_roster(
    row: &RoventTrainingRecordV2,
    head: &RoventHeadSource,
) -> Result<(), ManifestError> {
    let raw = &head.raw_receipt;
    let witness = &raw["witness"];
    let source =
        validate_divergence_source_roster(row, &witness["source"], head.observed_sequence)?;
    ensure(
        exact_fields(
            raw,
            &[
                "schema",
                "kind",
                "input_sha256",
                "anchor",
                "observed_sequence",
                "witness",
            ],
        ) && raw["schema"] == HEAD_DOMAIN
            && raw["kind"] == "pc_divergence"
            && raw["input_sha256"] == row.input_sha256
            && raw["observed_sequence"].as_u64() == Some(head.observed_sequence)
            && head.actor_id == row.lineage.actor_id
            && head.source == "rovent_actual_pc_divergence"
            && head.binary_sha256.is_none()
            && head.profile_sha256.is_none()
            && exact_fields(
                witness,
                &[
                    "schema",
                    "native_witness_sha256",
                    "native_hash_kind",
                    "event_sequence",
                    "reason",
                    "resolution",
                    "target_known",
                    "source",
                    "repair",
                    "recheck",
                    "repaired_endpoint",
                    "counter_endpoint",
                ],
            )
            && witness["schema"] == "rz-rovent-pc-divergence-witness/1"
            && witness["native_witness_sha256"]
                .as_str()
                .is_some_and(generation_sha)
            && matches!(
                witness["native_hash_kind"].as_str(),
                Some("frozen_pair" | "unresolved")
            )
            && witness["event_sequence"].is_u64()
            && witness["reason"].as_str().is_some_and(|value| {
                !value.is_empty() && value.len() <= 4096 && !value.chars().any(char::is_control)
            })
            && witness["target_known"].is_boolean(),
        "new divergence exact original raw owner roster/input/output scope",
    )?;
    for (key, endpoint) in [
        ("repair", false),
        ("recheck", false),
        ("repaired_endpoint", true),
        ("counter_endpoint", true),
    ] {
        let fact = &witness[key];
        if !fact.is_null() {
            ensure(
                exact_fields(
                    fact,
                    if endpoint {
                        &[
                            "native_endpoint_sha256",
                            "accepted",
                            "line",
                            "pair",
                            "model",
                            "perspective",
                            "raw_wdl_bits",
                            "root_wdl_bits",
                            "physical",
                        ]
                    } else {
                        &[
                            "native_line_sha256",
                            "accepted",
                            "line_id",
                            "observation_id",
                            "revision",
                            "kind",
                            "moves",
                        ]
                    },
                ),
                "new divergence optional fact exact roster",
            )?;
        }
    }
    for fact in [
        &source["accepted"],
        &witness["source"]["attack_ranked_source"]["accepted"],
        &witness["repair"]["accepted"],
        &witness["recheck"]["accepted"],
        &witness["repaired_endpoint"]["accepted"],
        &witness["counter_endpoint"]["accepted"],
    ] {
        if !fact.is_null() {
            ensure(
                exact_fields(
                    fact,
                    &[
                        "frame",
                        "ticket",
                        "output_digest",
                        "kind",
                        "peer",
                        "seal_sha256",
                    ],
                ),
                "new divergence original accepted origin exact roster",
            )?;
        }
    }
    // Full native binary framing, Rules lines and frozen model joins belong to
    // rz-arena's historical owner. This structural reader creates no authority.
    Ok(())
}

/// 역사적 원 owner body의 의미 대조다. 이 계산은 현재 AcceptedRoleStep이나
/// 실제 모델 실행 권한을 발급하지 않는다. 원 future native rows의 전수 join과
/// Rules 수열 재생은 상위 historical owner가 추가로 수행한다.
pub fn validate_divergence_raw(
    row: &RoventTrainingRecordV2,
    target: &RoventDivergenceTargetV1,
    head: &RoventHeadSource,
) -> Result<(), ManifestError> {
    target.value(&row.input)?;
    validate_divergence_roster(row, head)?;
    let raw = &head.raw_receipt;
    let witness = &raw["witness"];
    let source = divergence_visible_source(&witness["source"]);
    let attack = if witness["source"]["schema"] == "rz-rovent-pc-divergence-source-with-attack/1" {
        Some(&witness["source"]["attack_ranked_source"])
    } else {
        None
    };
    let counter_ticket = &source["accepted"]["ticket"];
    ensure(
        exact_fields(
            raw,
            &[
                "schema",
                "kind",
                "input_sha256",
                "anchor",
                "observed_sequence",
                "witness",
            ],
        ) && raw["schema"] == HEAD_DOMAIN
            && raw["kind"] == "pc_divergence"
            && head.source == "rovent_actual_pc_divergence"
            && head.binary_sha256.is_none()
            && head.profile_sha256.is_none()
            && witness["schema"] == "rz-rovent-pc-divergence-witness/1"
            && witness["event_sequence"].as_u64().is_some_and(|v| v > 0)
            && witness["native_hash_kind"] == "frozen_pair"
            && witness["target_known"] == true
            && witness["native_witness_sha256"]
                .as_str()
                .is_some_and(generation_sha)
            && source["counter_local_index"].as_u64()
                == Some(u64::from(target.counter_local_index))
            && source["counter_revision"].as_u64() == Some(target.counter_revision)
            && source["divergence_ply"].as_u64() == Some(u64::from(target.attack_ply))
            && counter_ticket["question_digest"] == target.counter_question_id
            && counter_ticket["view_digest"] == target.counter_native_view_sha256
            && witness["repair"]["accepted"]["ticket"]["question_digest"]
                == target.repair_question_id,
        "actual divergence source counter namespace and accepted Repair",
    )?;
    let original = &row.native.raw_submission["ticket"];
    if let Some(attack) = attack {
        ensure(
            attack["accepted"]["ticket"] == *original
                && attack["divergence_index"].as_u64() == Some(u64::from(target.slot))
                && attack["divergence_ply"].as_u64() == Some(u64::from(target.attack_ply))
                && row.input.attack_mapping[usize::from(target.slot)]
                    .selected_counter_local_index
                    .is_none()
                && !row.input.selected_view.query.counter_present
                && original["request_sequence"]
                    .as_u64()
                    .zip(counter_ticket["request_sequence"].as_u64())
                    .is_some_and(|(initial, visible)| initial < visible),
            "known initial attack requires its original no-counter slot and future visible owner",
        )?;
    } else {
        ensure(
            counter_ticket == original
                && source["divergence_index"].as_u64() == Some(u64::from(target.slot))
                && row.input.attack_mapping[usize::from(target.slot)]
                    .selected_counter_local_index
                    .is_some(),
            "MissingOriginalAttackSource: no capability may insert a future counter into a past input",
        )?;
    }
    let repair = &witness["repaired_endpoint"];
    let counter = &witness["counter_endpoint"];
    ensure(
        repair.is_object()
            && counter.is_object()
            && repair["pair"] == counter["pair"]
            && repair["pair"].is_object()
            && repair["model"] == counter["model"]
            && repair["perspective"] == counter["perspective"]
            && repair["line"] == witness["repair"]["moves"]
            && counter["line"] == witness["recheck"]["moves"],
        "actual divergence matched frozen endpoint pair",
    )?;
    let pair = &repair["pair"];
    ensure(
        exact_fields(
            pair,
            &[
                "public_context",
                "authority",
                "root_state",
                "context_policy",
                "selected_records",
                "anchors",
                "model",
                "encoding",
                "precision",
                "frozen_epoch",
                "record_revision",
                "role",
                "perspective",
            ],
        ) && pair["public_context"].as_str().is_some_and(generation_sha)
            && pair["role"] == "proposer"
            && pair["perspective"] == source["perspective"],
        "complete frozen pair seal fields",
    )?;
    let mut previous = counter_ticket["request_sequence"].as_u64().ok_or_else(|| {
        ManifestError::Integrity("original visible source request sequence".into())
    })?;
    for (fact, role) in [
        (&witness["repair"], "proposer"),
        (&witness["recheck"], "critic"),
        (repair, "proposer"),
        (counter, "proposer"),
    ] {
        let ticket = &fact["accepted"]["ticket"];
        let sequence = ticket["request_sequence"]
            .as_u64()
            .ok_or_else(|| ManifestError::Integrity("actual accepted ticket sequence".into()))?;
        ensure(
            ticket["head"].is_null()
                && ticket["representation"]["role"] == role
                && ticket["process_epoch"].as_u64() == Some(row.native.process_epoch)
                && ticket["game_generation"].as_u64() == Some(row.native.game_generation)
                && ticket["root_generation"].as_u64() == Some(row.native.root_generation)
                && ticket["implementation_sha256"] == original["implementation_sha256"]
                && fact["accepted"]["output_digest"]
                    .as_str()
                    .is_some_and(generation_sha),
            "divergence future accepted head/root provenance",
        )?;
        if role == "critic" || fact == &witness["repair"] {
            ensure(sequence > previous, "actual Repair/recheck chronology")?;
            previous = sequence;
        } else {
            ensure(
                sequence > previous,
                "actual Fresh follows accepted C recheck",
            )?;
        }
    }
    for endpoint in [repair, counter] {
        let ticket = &endpoint["accepted"]["ticket"];
        ensure(
            endpoint["physical"]["measurement_complete"] == true
                && endpoint["physical"]["physical_fresh"] == true
                && endpoint["physical"]["completed_nn_inputs"]
                    .as_u64()
                    .is_some_and(|v| v > 0)
                && endpoint["physical"]["execution_epoch"].as_u64()
                    == Some(row.native.process_epoch),
            "actual divergence requires both original Fresh physical completions",
        )?;
        ensure(
            pair["authority"]["process_epoch"] == ticket["process_epoch"]
                && pair["authority"]["game_generation"] == ticket["game_generation"]
                && pair["authority"]["root_generation"] == ticket["root_generation"]
                && pair["authority"]["implementation_sha256"] == ticket["implementation_sha256"]
                && pair["root_state"] == ticket["root_state"]
                && pair["context_policy"] == ticket["context_policy"]
                && pair["selected_records"] == ticket["selected_records"]
                && pair["anchors"] == ticket["anchors"]
                && pair["record_revision"] == ticket["representation"]["record_revision"],
            "frozen pair does not join each original endpoint ticket",
        )?;
        for field in ["model", "encoding", "precision", "frozen_epoch"] {
            ensure(
                pair[field] == ticket["representation"][field]
                    && ticket["representation"][field] == original["representation"][field],
                "divergence original frozen model/encoding/precision changed",
            )?;
        }
    }
    let ordering = wdl_expectation(repair)?
        .partial_cmp(&wdl_expectation(counter)?)
        .ok_or_else(|| ManifestError::Integrity("invalid conditional comparison".into()))?;
    let expected = match ordering {
        std::cmp::Ordering::Greater => "refuted",
        std::cmp::Ordering::Less => "supported",
        std::cmp::Ordering::Equal => "unknown",
    };
    ensure(
        witness["resolution"] == expected
            && match target.status {
                GenerationDivergenceStatusV3::SupportedAfterRepair => expected == "supported",
                GenerationDivergenceStatusV3::RefutedByRepair => expected == "refuted",
            },
        "known conditional divergence cannot be a tie/unknown or external agreement",
    )
}
