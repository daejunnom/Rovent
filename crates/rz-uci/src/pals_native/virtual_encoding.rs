//! Pure V3 Rules/selected-review declaration. Never invokes a model or teacher.
//! The compact schema is separate from V1/V2's sixteen record/query features.
#[path = "virtual_selected_input.rs"]
mod selected_input;
pub use selected_input::prepare_selected_virtual_review_input;
#[cfg(feature = "onnx-cpu")]
pub use selected_input::{encode_review_divergences, exact_review_anchor_input};

use rz_eval::pals_model::virtual_review::{QUERY_FEATURE_NAMES, RECORD_FEATURE_NAMES};
#[cfg(feature = "search-work-receipts")]
use rz_eval::pals_model::{PalsModelConfig, PalsModelProfile};
use sha2::{Digest, Sha256};

pub const PALS_VIRTUAL_RULES_ENCODING: &str = "rz-pals-rules-virtual-review-v1";
pub const PALS_VIRTUAL_RULES_BASE_SEMANTICS: &str = concat!(
    "board:a1-h8;empty=0;white_PNBRQK=1..6;black_PNBRQK=7..12;",
    "history:newest_first_exact_fen;root:local_anchor0;",
    "anchors:exact_Rules_board_metadata_and_full_history_seal;",
    "records:immutable_selected_bundle_coherent_virtual_lines;",
    "candidate_order:Rules;divergence_features:8_registered_Rules_fields;",
    "query_features:six_purpose_onehot,perspective,counter_presence;",
    "actual_game_moves:Rules_reproduction_only_not_review_tensor_tokens"
);
pub const PALS_VIRTUAL_REVIEW_FEATURES: [&str; 14] = [
    "review.anchor_board",
    "review.anchor_rules_metadata",
    "review.record_anchor_local",
    "review.record_moves",
    "review.record_lengths",
    "review.record_masks",
    "review.record_kind",
    "review.parent_local",
    "review.supersedes_local",
    "review.query_moves",
    "review.query_lengths",
    "review.query_masks",
    "review.query_anchor_local",
    "review.physical_line_bucket",
];
pub fn pals_virtual_rules_semantic_fields() -> impl Iterator<Item = &'static str> {
    [
        PALS_VIRTUAL_RULES_ENCODING,
        PALS_VIRTUAL_RULES_BASE_SEMANTICS,
    ]
    .into_iter()
    .chain(super::PALS_METADATA_FEATURES)
    .chain(RECORD_FEATURE_NAMES)
    .chain(QUERY_FEATURE_NAMES)
    .chain(super::PALS_DIVERGENCE_FEATURES)
    .chain(PALS_VIRTUAL_REVIEW_FEATURES)
}
pub fn pals_virtual_rules_semantic_digest() -> [u8; 32] {
    let mut hash = Sha256::new();
    for field in pals_virtual_rules_semantic_fields() {
        hash.update((field.len() as u64).to_le_bytes());
        hash.update(field.as_bytes());
    }
    hash.finalize().into()
}
/// Separate exact source observation pin. It is never a weight/cache identity
/// or proof that a trained model understands the declared features.
pub fn pals_virtual_rules_source_digest() -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"rz-pals-virtual-rules-encoder-source/1\0");
    for (name, source) in [
        (
            "pals_native.rs",
            include_bytes!("../pals_native.rs").as_slice(),
        ),
        (
            "pals_native/virtual_encoding.rs",
            include_bytes!("virtual_encoding.rs").as_slice(),
        ),
        (
            "pals_native/virtual_selected_input.rs",
            include_bytes!("virtual_selected_input.rs").as_slice(),
        ),
        (
            "pals_native/ticketed/virtual_bridge.rs",
            include_bytes!("ticketed/virtual_bridge.rs").as_slice(),
        ),
        (
            "rz-eval/pals_model/virtual_review.rs",
            include_bytes!("../../../rz-eval/src/pals_model/virtual_review.rs").as_slice(),
        ),
    ] {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((source.len() as u64).to_le_bytes());
        hash.update(source);
    }
    hash.finalize().into()
}
#[cfg(feature = "search-work-receipts")]
fn hex(digest: [u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}
#[cfg(feature = "search-work-receipts")]
pub fn pals_virtual_rules_descriptor() -> serde_json::Value {
    serde_json::json!({
        "schema":"rovezero.pals-rules-descriptor.v3",
        "rules_input_profile":PALS_VIRTUAL_RULES_ENCODING,
        "config":PalsModelConfig::for_profile(PalsModelProfile::VirtualReviewV1),
        "rules_base_semantics":PALS_VIRTUAL_RULES_BASE_SEMANTICS,
        "metadata_features":super::PALS_METADATA_FEATURES,
        "record_features":RECORD_FEATURE_NAMES,"query_features":QUERY_FEATURE_NAMES,
        "divergence_features":super::PALS_DIVERGENCE_FEATURES,
        "virtual_review_features":PALS_VIRTUAL_REVIEW_FEATURES,
        "bounds":{"max_records":64,"max_anchors":4,"max_line_plies":256,
            "line_buckets":[8,16,32,64,128,256],"max_candidates":256,
            "token_budgets":[128,256,512],"root_anchor":0},
        "semantic_fields":pals_virtual_rules_semantic_fields().collect::<Vec<_>>(),
        "semantic_digest_algorithm":"sha256_u64le_length_prefixed_utf8_fields",
        "rules_input_semantic_sha256":hex(pals_virtual_rules_semantic_digest()),
        "rules_encoder_source_sha256":hex(pals_virtual_rules_source_digest()),
        "learned_input_compatibility":"unverified_declaration_only",
    })
}

/// Pure pre-child evidence join. It validates caller-supplied bytes against
/// the currently compiled encoder declaration and never reads files or loads
/// a model. A descriptor proves source identity, not learned compatibility.
#[cfg(feature = "search-work-receipts")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VirtualRulesDescriptorSummary {
    descriptor_sha256: [u8; 32],
    pub config: PalsModelConfig,
    pub semantic_sha256: [u8; 32],
    pub encoder_source_sha256: [u8; 32],
}
#[cfg(feature = "search-work-receipts")]
impl VirtualRulesDescriptorSummary {
    pub fn descriptor_sha256(&self) -> [u8; 32] {
        self.descriptor_sha256
    }
}
#[cfg(feature = "search-work-receipts")]
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct VirtualRulesDescriptorDocument {
    schema: String,
    rules_input_profile: String,
    config: PalsModelConfig,
    rules_base_semantics: String,
    metadata_features: Vec<String>,
    record_features: Vec<String>,
    query_features: Vec<String>,
    divergence_features: Vec<String>,
    virtual_review_features: Vec<String>,
    bounds: VirtualRulesDescriptorBounds,
    semantic_fields: Vec<String>,
    semantic_digest_algorithm: String,
    rules_input_semantic_sha256: String,
    rules_encoder_source_sha256: String,
    learned_input_compatibility: String,
}
#[cfg(feature = "search-work-receipts")]
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct VirtualRulesDescriptorBounds {
    max_records: u16,
    max_anchors: u8,
    max_line_plies: u16,
    line_buckets: Vec<u16>,
    max_candidates: u16,
    token_budgets: Vec<u16>,
    root_anchor: u8,
}
#[cfg(feature = "search-work-receipts")]
pub fn validate_pals_virtual_rules_descriptor_bytes(
    bytes: &[u8],
) -> Result<VirtualRulesDescriptorSummary, rz_eval::error::BackendError> {
    use rz_eval::error::{BackendError, FailureKind, FailureStage};
    let fail = |detail| {
        BackendError::new(
            FailureKind::IdentityMismatch,
            FailureStage::Admission,
            detail,
        )
    };
    if bytes.is_empty() || bytes.len() > 64 * 1024 {
        return Err(fail("V3 Rules descriptor exceeds bounded bytes"));
    }
    // Typed serde structs reject both duplicate known fields and unknown
    // fields, including nested bounds/config, before joining any authority.
    let d: VirtualRulesDescriptorDocument = serde_json::from_slice(bytes)
        .map_err(|_| fail("cannot decode strict V3 Rules descriptor"))?;
    let semantic_sha256 = pals_virtual_rules_semantic_digest();
    let encoder_source_sha256 = pals_virtual_rules_source_digest();
    let expected_fields = pals_virtual_rules_semantic_fields().collect::<Vec<_>>();
    if d.schema != "rovezero.pals-rules-descriptor.v3"
        || d.rules_input_profile != PALS_VIRTUAL_RULES_ENCODING
        || d.config != PalsModelConfig::for_profile(PalsModelProfile::VirtualReviewV1)
        || d.rules_base_semantics != PALS_VIRTUAL_RULES_BASE_SEMANTICS
        || d.metadata_features != super::PALS_METADATA_FEATURES
        || d.record_features != RECORD_FEATURE_NAMES
        || d.query_features != QUERY_FEATURE_NAMES
        || d.divergence_features != super::PALS_DIVERGENCE_FEATURES
        || d.virtual_review_features != PALS_VIRTUAL_REVIEW_FEATURES
        || d.bounds.max_records != 64
        || d.bounds.max_anchors != 4
        || d.bounds.max_line_plies != 256
        || d.bounds.line_buckets != [8, 16, 32, 64, 128, 256]
        || d.bounds.max_candidates != 256
        || d.bounds.token_budgets != [128, 256, 512]
        || d.bounds.root_anchor != 0
        || d.semantic_fields != expected_fields
        || d.semantic_digest_algorithm != "sha256_u64le_length_prefixed_utf8_fields"
        || d.rules_input_semantic_sha256 != hex(semantic_sha256)
        || d.rules_encoder_source_sha256 != hex(encoder_source_sha256)
        || d.learned_input_compatibility != "unverified_declaration_only"
    {
        return Err(fail(
            "V3 Rules descriptor differs from the compiled source declaration",
        ));
    }
    Ok(VirtualRulesDescriptorSummary {
        descriptor_sha256: Sha256::digest(bytes).into(),
        config: d.config,
        semantic_sha256,
        encoder_source_sha256,
    })
}
#[cfg(all(test, feature = "search-work-receipts"))]
mod tests {
    use super::*;
    #[cfg(feature = "search-work-receipts")]
    #[test]
    fn v3_descriptor_has_eight_axes_and_its_own_source_and_semantic_domains() {
        let descriptor = pals_virtual_rules_descriptor();
        assert_eq!(pals_virtual_rules_semantic_fields().count(), 56);
        assert_eq!(descriptor["record_features"].as_array().unwrap().len(), 8);
        assert_eq!(descriptor["query_features"].as_array().unwrap().len(), 8);
        assert_eq!(descriptor["bounds"]["max_anchors"], 4);
        assert_ne!(
            pals_virtual_rules_semantic_digest(),
            super::super::pals_rules_encoding_semantic_digest()
        );
        assert_ne!(
            pals_virtual_rules_source_digest(),
            pals_virtual_rules_semantic_digest()
        );
        assert!(!PALS_VIRTUAL_RULES_BASE_SEMANTICS.contains("latest_kinds_and_author"));
    }
    #[cfg(feature = "search-work-receipts")]
    #[test]
    fn external_descriptor_join_requires_actual_compiled_source_and_strict_fields() {
        let descriptor = pals_virtual_rules_descriptor();
        let bytes = serde_json::to_vec(&descriptor).unwrap();
        let summary = validate_pals_virtual_rules_descriptor_bytes(&bytes).unwrap();
        assert_eq!(
            summary.descriptor_sha256(),
            <[u8; 32]>::from(Sha256::digest(&bytes))
        );
        assert_eq!(
            summary.semantic_sha256,
            pals_virtual_rules_semantic_digest()
        );
        for path in ["rules_encoder_source_sha256", "rules_input_semantic_sha256"] {
            let mut changed = descriptor.clone();
            changed[path] = serde_json::json!("00".repeat(32));
            assert!(
                validate_pals_virtual_rules_descriptor_bytes(
                    &serde_json::to_vec(&changed).unwrap()
                )
                .is_err()
            );
        }
        let mut changed = descriptor.clone();
        changed["extra"] = serde_json::json!(true);
        assert!(
            validate_pals_virtual_rules_descriptor_bytes(&serde_json::to_vec(&changed).unwrap())
                .is_err()
        );
        let duplicated = format!(
            "{{\"schema\":\"rovezero.pals-rules-descriptor.v3\",{}",
            std::str::from_utf8(&bytes).unwrap().trim_start_matches('{')
        );
        assert!(validate_pals_virtual_rules_descriptor_bytes(duplicated.as_bytes()).is_err());
    }
}
