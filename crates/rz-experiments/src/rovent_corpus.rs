//! Rovent corpus wire formats. Parsing/hash validation never grants live owner
//! authority. The arena producer consumes the original typed native/teacher
//! receipts; these separate domains never reinterpret historical PALS data.
use crate::pals_data_v3::*;
use crate::{ManifestError, decode_json};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const CORPUS_DOMAIN: &str = "rz-rovent-corpus-manifest/1";
pub const RECORD_DOMAIN: &str = "rz-rovent-training-record/1";
pub const INDEX_DOMAIN: &str = "rz-rovent-corpus-index/1";
pub const ORIGIN_DOMAIN: &str = "rz-rovent-training-origin/1";
pub const HEAD_DOMAIN: &str = "rz-rovent-head-source/1";
pub const NATIVE_RAW_DOMAIN: &str = "rz-rovent-native-owner-event/1";
/// Explicit original Rules replay/root-admission evidence. Legacy /1 readers
/// continue to reject this domain and never upgrade a record implicitly.
pub const NATIVE_RAW_RULES_V2_DOMAIN: &str = "rz-rovent-native-owner-event/2";
pub const NNUEC_INPUT_DOMAIN: &str = "rz-rovent-nnuec-reference-input/1";
pub const MAX_CORPUS_ROWS: usize = 262_144;
pub const MAX_CHUNK_BYTES: usize = 32 * 1024 * 1024;
pub const MAX_CORPUS_BYTES: u64 = 18 * 1024 * 1024 * 1024;

fn check(ok: bool, why: &str) -> Result<(), ManifestError> {
    if ok {
        Ok(())
    } else {
        Err(ManifestError::Integrity(why.into()))
    }
}
fn unique_shas(values: &[String], maximum: usize) -> bool {
    values.len() <= maximum
        && values.iter().all(|s| generation_sha(s))
        && values.windows(2).all(|w| w[0] < w[1])
}
fn native_bits(
    raw: &serde_json::Value,
    key: &str,
    min: usize,
    max: usize,
) -> Result<Vec<u32>, ManifestError> {
    let a = raw[key]
        .as_array()
        .ok_or_else(|| ManifestError::Integrity("missing native f32 bits".into()))?;
    check(a.len() >= min && a.len() <= max, "native f32 array shape")?;
    a.iter()
        .map(|v| {
            let b = v
                .as_u64()
                .filter(|v| *v <= u64::from(u32::MAX))
                .ok_or_else(|| ManifestError::Integrity("native f32 bit word".into()))?
                as u32;
            check(f32::from_bits(b).is_finite(), "nonfinite native f32 word")?;
            Ok(b)
        })
        .collect()
}
/// Reconstructs the existing actual registry's accepted-head digest from the
/// lossless raw body. It cannot reconstruct live AcceptedRoleStep authority.
fn native_output_digest(raw: &serde_json::Value, head: &str) -> Result<String, ManifestError> {
    native_bits(raw, "private_latent_bits", 6144, 6144)?;
    let p = &raw["payload"];
    let mut hash = Sha256::new();
    let mut values = Vec::new();
    match head {
        "proposer" => {
            check(p["kind"] == "proposal", "native proposer payload")?;
            values = native_bits(p, "candidate_logits_bits", 1, 256)?;
            hash.update(b"rz-pals-accepted-proposer-head/1\0");
            hash.update((values.len() as u64).to_le_bytes());
        }
        "critic" => {
            check(p["kind"] == "counterexample", "native critic payload")?;
            let a = native_bits(p, "candidate_logits_bits", 0, 256)?;
            let b = native_bits(p, "divergence_logits_bits", 0, 256)?;
            check(!a.is_empty() || !b.is_empty(), "empty native critic heads")?;
            hash.update(b"rz-rovent-accepted-critic-head/1\0");
            hash.update((a.len() as u64).to_le_bytes());
            hash.update((b.len() as u64).to_le_bytes());
            values.extend(a);
            values.extend(b);
        }
        "validator" => {
            check(p["kind"] == "task_ranking", "native validator payload")?;
            hash.update(b"rz-pals-accepted-validator-head/1\0");
            values = native_bits(p, "task_logits_bits", 7, 7)?;
        }
        _ => return Err(ManifestError::Integrity("unknown native head".into())),
    }
    if head != "validator" {
        let w = native_bits(p, "wdl_bits", 3, 3)?;
        let probabilities = w.iter().map(|v| f32::from_bits(*v)).collect::<Vec<_>>();
        check(
            probabilities.iter().all(|p| (0.0..=1.0).contains(p))
                && (probabilities.iter().sum::<f32>() - 1.0).abs() <= 1e-4,
            "native WDL invalid",
        )?;
        values.extend(w);
    }
    for v in values {
        hash.update(v.to_le_bytes());
    }
    Ok(format!("{:x}", hash.finalize()))
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventTeacherPin {
    pub engine: String,
    pub source_commit: String,
    pub binary_sha256: String,
    pub binary_bytes: u64,
    pub profile_sha256: String,
    pub canonical_sha256: String,
    pub capability_sha256: String,
    pub options: BTreeMap<String, String>,
    pub assets: BTreeMap<String, String>,
}
impl RoventTeacherPin {
    pub fn validate(&self) -> Result<(), ManifestError> {
        check(
            ["stockfish", "reckless"].contains(&self.engine.as_str())
                && self.source_commit.len() == 40
                && self
                    .source_commit
                    .bytes()
                    .all(|v| v.is_ascii_hexdigit() && !v.is_ascii_uppercase())
                && self.binary_bytes > 0
                && [
                    &self.binary_sha256,
                    &self.profile_sha256,
                    &self.canonical_sha256,
                    &self.capability_sha256,
                ]
                .iter()
                .all(|v| generation_sha(v))
                && self.options.len() <= 128
                && self.assets.len() <= 128
                && self
                    .options
                    .iter()
                    .all(|(k, v)| generation_identifier(k) && v.len() <= 8192)
                && self
                    .assets
                    .iter()
                    .all(|(k, v)| generation_identifier(k) && generation_sha(v)),
            "teacher source/profile/binary/options/assets declaration",
        )
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventCorpusManifest {
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
impl RoventCorpusManifest {
    pub fn validate(&self) -> Result<(), ManifestError> {
        check(
            self.schema == CORPUS_DOMAIN
                && self.one_learner
                && [
                    &self.corpus_id,
                    &self.stage_id,
                    &self.round_id,
                    &self.student,
                    &self.learner,
                ]
                .iter()
                .all(|v| generation_identifier(v))
                && self.source_commit.len() == 40
                && self
                    .source_commit
                    .bytes()
                    .all(|v| v.is_ascii_hexdigit() && !v.is_ascii_uppercase())
                && [
                    &self.source_sha256,
                    &self.runtime_sha256,
                    &self.model_semantics_sha256,
                    &self.recipe_sha256,
                ]
                .iter()
                .all(|v| generation_sha(v))
                && self.model_epoch > 0
                && self.context_epoch > 0
                && unique_shas(&self.inherited_input_sha256, MAX_CORPUS_ROWS)
                && unique_shas(&self.inherited_origin_sha256, MAX_CORPUS_ROWS)
                && unique_shas(&self.ancestor_index_sha256, 256)
                && !self.opening_splits.is_empty()
                && self.opening_splits.len() <= 8192
                && self.opening_splits.keys().all(|v| generation_identifier(v))
                && self.teachers.len() <= 16,
            "Rovent corpus recipe/stage/student/one-learner/source/split/ancestor declaration",
        )?;
        for (actor, pin) in &self.teachers {
            check(generation_identifier(actor), "teacher actor")?;
            pin.validate()?;
        }
        Ok(())
    }
    pub fn sha256(&self) -> Result<String, ManifestError> {
        self.validate()?;
        generation_digest(CORPUS_DOMAIN, self)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventLineage {
    pub actor_id: String,
    pub teacher_actor_id: Option<String>,
    pub ancestor_origin_sha256: Vec<String>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventNativeLifecycle {
    pub head: String,
    pub source_sha256: String,
    pub model_sha256: String,
    pub encoding_sha256: String,
    pub input_digest: String,
    pub view_digest: String,
    pub output_digest: String,
    pub process_epoch: u64,
    pub game_generation: u64,
    pub root_generation: u64,
    pub request_sequence: u64,
    pub execution_epoch: u64,
    pub execution_sequence: u64,
    pub completed_nn_inputs: u64,
    pub submitted_sequence: u64,
    pub completed_sequence: u64,
    pub retired_sequence: u64,
    pub owner_run_invoked: bool,
    pub physical_confirmed: bool,
    pub accepted_retired: bool,
    pub raw_submission: serde_json::Value,
    pub raw_completion: serde_json::Value,
    pub raw_retirement: serde_json::Value,
    pub raw_submission_sha256: String,
    pub raw_completion_sha256: String,
    pub raw_retirement_sha256: String,
}
impl RoventNativeLifecycle {
    pub fn validate(&self, manifest: &RoventCorpusManifest) -> Result<(), ManifestError> {
        self.validate_domain(manifest, NATIVE_RAW_DOMAIN)
    }
    pub fn validate_rules_replay_v2(
        &self,
        manifest: &RoventCorpusManifest,
    ) -> Result<(), ManifestError> {
        self.validate_domain(manifest, NATIVE_RAW_RULES_V2_DOMAIN)?;
        for (raw, fields) in [
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
            check(
                raw.as_object().is_some_and(|object| {
                    object.len() == fields.len()
                        && fields.iter().all(|field| object.contains_key(*field))
                }),
                "explicit /2 raw field set",
            )?;
        }
        let retirement = &self.raw_retirement;
        check(
            matches!(self.head.as_str(), "proposer" | "critic")
                && self.raw_submission["rules_replay"]["schema"]
                    == "rz-rovent-submitted-rules-replay/1"
                && matches!(
                    retirement["acceptance_kind"].as_str(),
                    Some("accepted_role_step" | "root_ranking_published")
                )
                && (retirement["acceptance_kind"] != "root_ranking_published"
                    || self.head == "proposer")
                && retirement["root_admission"]["schema"]
                    == "rz-rovent-collection-root-admission/1"
                && retirement["root_admission_sha256"]
                    == generation_digest(
                        "rz-rovent-collection-root-admission/1",
                        &retirement["root_admission"],
                    )?
                && retirement["root_admission"]["native_origin"]["purpose"] == "collection",
            "explicit /2 actual admission and Rules replay body/seal",
        )
    }
    pub(crate) fn validate_domain(
        &self,
        manifest: &RoventCorpusManifest,
        domain: &str,
    ) -> Result<(), ManifestError> {
        check(
            self.source_sha256 == manifest.source_sha256
                && self.owner_run_invoked
                && self.physical_confirmed
                && self.accepted_retired
                && self.completed_nn_inputs > 0
                && self.process_epoch > 0
                && self.game_generation > 0
                && self.root_generation > 0
                && self.request_sequence > 0
                && self.execution_epoch == self.process_epoch
                && self.execution_sequence > 0
                && self.submitted_sequence < self.completed_sequence
                && self.completed_sequence < self.retired_sequence
                && [
                    &self.model_sha256,
                    &self.encoding_sha256,
                    &self.input_digest,
                    &self.view_digest,
                    &self.output_digest,
                ]
                .iter()
                .all(|v| generation_sha(v)),
            "native submit/complete/retire copied owner chain",
        )?;
        for (raw, expected, kind) in [
            (
                &self.raw_submission,
                &self.raw_submission_sha256,
                "submitted",
            ),
            (
                &self.raw_completion,
                &self.raw_completion_sha256,
                "completed",
            ),
            (&self.raw_retirement, &self.raw_retirement_sha256, "retired"),
        ] {
            check(
                raw["schema"] == domain
                    && raw["kind"] == kind
                    && raw["process_epoch"].as_u64() == Some(self.process_epoch)
                    && raw["game_generation"].as_u64() == Some(self.game_generation)
                    && raw["root_generation"].as_u64() == Some(self.root_generation)
                    && raw["request_sequence"].as_u64() == Some(self.request_sequence)
                    && raw["source_sha256"] == self.source_sha256
                    && raw["head"] == self.head
                    && generation_digest(domain, raw)? == *expected,
                "native raw owner receipt body/seal/request",
            )?;
        }
        check(
            self.raw_submission["model_sha256"] == self.model_sha256
                && self.raw_submission["input_digest"] == self.input_digest
                && self.raw_submission["view_digest"] == self.view_digest
                && self.raw_completion["output_digest"] == self.output_digest
                && self.raw_retirement["output_digest"] == self.output_digest
                && self.raw_completion["physical_confirmed"] == true
                && self.raw_completion["owner_run_invoked"] == true
                && self.raw_completion["measurement_complete"] == true
                && self.raw_completion["physical_fresh"] == true
                && self.raw_retirement["accepted_retired"] == true
                && self.raw_submission["sequence"].as_u64() == Some(self.submitted_sequence)
                && self.raw_completion["sequence"].as_u64() == Some(self.completed_sequence)
                && self.raw_retirement["sequence"].as_u64() == Some(self.retired_sequence)
                && self.raw_completion["submitted_sha256"] == self.raw_submission_sha256
                && self.raw_retirement["submitted_sha256"] == self.raw_submission_sha256
                && self.raw_retirement["completed_sha256"] == self.raw_completion_sha256
                && self.raw_completion["execution_epoch"].as_u64() == Some(self.execution_epoch)
                && self.raw_completion["execution_sequence"].as_u64()
                    == Some(self.execution_sequence)
                && self.raw_completion["completed_nn_inputs"].as_u64()
                    == Some(self.completed_nn_inputs)
                && self.raw_submission["ticket"] == self.raw_completion["ticket"]
                && self.raw_submission["ticket"] == self.raw_retirement["ticket"]
                && self.raw_completion["output"]["request_epoch"].as_u64()
                    == Some(self.process_epoch)
                && self.raw_completion["output"]["request_sequence"].as_u64()
                    == Some(self.request_sequence)
                && self.raw_completion["output"]["game_generation"].as_u64()
                    == Some(self.game_generation)
                && self.raw_completion["output"]["root_generation"].as_u64()
                    == Some(self.root_generation)
                && self.raw_completion["output"]["representation"]
                    == self.raw_submission["ticket"]["representation"]
                && self.raw_completion["output"]["implementation_sha256"]
                    == self.raw_submission["ticket"]["implementation_sha256"]
                && native_output_digest(&self.raw_completion["output"], &self.head)?
                    == self.output_digest,
            "native raw and summary disagree",
        )
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventHeadSource {
    pub actor_id: String,
    pub source: String,
    pub binary_sha256: Option<String>,
    pub profile_sha256: Option<String>,
    pub raw_receipt: serde_json::Value,
    pub raw_receipt_sha256: String,
    pub observed_sequence: u64,
    pub anchor: GenerationRulesAnchorV3,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventTrainingRecord {
    pub schema: String,
    pub role: GenerationRoleV3,
    pub split: GenerationSplitV3,
    pub input: GenerationInputV3,
    pub input_sha256: String,
    pub origin_sha256: String,
    pub record_sha256: String,
    pub lineage: RoventLineage,
    pub native: RoventNativeLifecycle,
    pub head_sources: BTreeMap<String, RoventHeadSource>,
    pub targets: GenerationTargetsV3,
    pub divergence_index: Option<u16>,
    pub nnuec_input: Option<serde_json::Value>,
    pub nnuec_input_sha256: Option<String>,
    pub nnuec_model_sha256: Option<String>,
    pub nnuec_semantics: Option<String>,
    pub nnuec_wdl: Option<GenerationWdlTargetV3>,
}
impl RoventTrainingRecord {
    pub fn origin(&self, m: &RoventCorpusManifest) -> Result<String, ManifestError> {
        generation_digest(
            ORIGIN_DOMAIN,
            &serde_json::json!({
            "source_sha256":m.source_sha256,"recipe_sha256":m.recipe_sha256,"stage_id":m.stage_id,
            "round":m.round,"round_id":m.round_id,"student":m.student,"learner":m.learner,
            "actor_id":self.lineage.actor_id,"teacher_actor_id":self.lineage.teacher_actor_id,
            "input_sha256":self.input_sha256,"game_id":self.input.game_id,"opening_id":self.input.opening_id,
            "line_genealogy_id":self.input.line_genealogy_id,"transposition_sha256":self.input.transposition_sha256,
            "ancestor_origin_sha256":self.lineage.ancestor_origin_sha256}),
        )
    }
    pub fn sha256(&self) -> Result<String, ManifestError> {
        let mut unsealed =
            serde_json::to_value(self).map_err(|e| ManifestError::Integrity(e.to_string()))?;
        unsealed["record_sha256"] = serde_json::json!("");
        // The original input has its own existing f32-bit semantic seal. Do
        // not hash language-specific JSON floating-point spellings again.
        unsealed["input"] = serde_json::json!({"sealed_input_sha256":self.input_sha256});
        generation_digest(RECORD_DOMAIN, &unsealed)
    }
    pub fn seal(&mut self, m: &RoventCorpusManifest) -> Result<(), ManifestError> {
        self.input_sha256 = self.input.sha256()?;
        self.origin_sha256 = self.origin(m)?;
        self.record_sha256 = self.sha256()?;
        self.validate(m)
    }
    pub fn seal_rules_replay_v2(&mut self, m: &RoventCorpusManifest) -> Result<(), ManifestError> {
        self.input_sha256 = self.input.sha256()?;
        self.origin_sha256 = self.origin(m)?;
        self.record_sha256 = self.sha256()?;
        self.validate_rules_replay_v2(m)
    }
    pub fn validate(&self, m: &RoventCorpusManifest) -> Result<(), ManifestError> {
        self.validate_native_format(m, false)
    }
    pub fn validate_rules_replay_v2(&self, m: &RoventCorpusManifest) -> Result<(), ManifestError> {
        self.validate_native_format(m, true)
    }
    fn validate_native_format(
        &self,
        m: &RoventCorpusManifest,
        rules_v2: bool,
    ) -> Result<(), ManifestError> {
        m.validate()?;
        self.input.validate()?;
        if rules_v2 {
            self.native.validate_rules_replay_v2(m)?;
        } else {
            self.native.validate(m)?;
        }
        check(
            self.schema == RECORD_DOMAIN
                && self.role == self.input.role
                && self.input_sha256 == self.input.sha256()?
                && self.origin_sha256 == self.origin(m)?
                && self.record_sha256 == self.sha256()?
                && self.input.model_epoch == m.model_epoch
                && self.input.context_epoch == m.context_epoch
                && m.opening_splits.get(&self.input.opening_id) == Some(&self.split)
                && generation_identifier(&self.lineage.actor_id)
                && unique_shas(&self.lineage.ancestor_origin_sha256, 256)
                && self
                    .lineage
                    .teacher_actor_id
                    .as_ref()
                    .is_none_or(|a| m.teachers.contains_key(a)),
            "record source/input/recipe/split/origin seal",
        )?;
        let head = match self.role {
            GenerationRoleV3::Proposer => "proposer",
            GenerationRoleV3::Critic => "critic",
            GenerationRoleV3::Validator => "validator",
        };
        check(
            self.native.head == head
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
            "native original input/head/question does not join record",
        )?;
        // All old head semantics except teacher identity remain unchanged.
        let non_policy = GenerationTargetsV3 {
            policy: None,
            wdl: self.targets.wdl.clone(),
            divergence: self.targets.divergence.clone(),
            task: self.targets.task.clone(),
        };
        non_policy.validate(&self.input)?;
        if let Some(p) = &self.targets.policy {
            check(
                p.source == "rovent_completed_teacher_multipv"
                    && p.completed
                    && !p.stale
                    && p.observed_sequence >= self.input.capture_sequence
                    && p.question_id == self.input.question_id
                    && p.view_sha256 == self.input.view_sha256
                    && p.anchor == self.input.rules_anchor
                    && p.go_sequence > 0
                    && p.completed_depth > 0
                    && !p.ranked_moves.is_empty()
                    && p.ranked_moves.len() <= 4
                    && p.ranked_moves.iter().collect::<BTreeSet<_>>().len() == p.ranked_moves.len()
                    && p.ranked_moves
                        .iter()
                        .all(|v| self.input.legal_moves.contains(v)),
                "Rovent conditional teacher policy is not a value target",
            )?;
            let h = self.head_sources.get("policy").ok_or_else(|| {
                ManifestError::Integrity("missing policy raw owner evidence".into())
            })?;
            let t = m
                .teachers
                .get(&h.actor_id)
                .ok_or_else(|| ManifestError::Integrity("unknown teacher actor".into()))?;
            check(
                self.lineage.teacher_actor_id.as_ref() == Some(&h.actor_id)
                    && h.source == p.source
                    && h.binary_sha256.as_ref() == Some(&t.binary_sha256)
                    && h.profile_sha256.as_ref() == Some(&t.profile_sha256)
                    && p.teacher_binary_sha256 == t.binary_sha256
                    && p.teacher_profile_sha256 == t.profile_sha256
                    && p.raw_evidence_sha256 == h.raw_receipt_sha256,
                "teacher actor/profile/binary/receipt attribution",
            )?;
        }
        for (head, h) in &self.head_sources {
            check(
                ["policy", "wdl", "divergence", "task", "nnuec_wdl"].contains(&head.as_str())
                    && h.anchor == self.input.rules_anchor
                    && h.observed_sequence >= self.input.capture_sequence
                    && generation_identifier(&h.actor_id)
                    && generation_identifier(&h.source)
                    && h.raw_receipt["input_sha256"] == self.input_sha256
                    && h.raw_receipt["anchor"]
                        == serde_json::to_value(&self.input.rules_anchor)
                            .map_err(|e| ManifestError::Integrity(e.to_string()))?
                    && generation_digest(HEAD_DOMAIN, &h.raw_receipt)? == h.raw_receipt_sha256,
                "head provenance body/anchor/chronology seal",
            )?;
        }
        for (name, present) in [
            ("wdl", self.targets.wdl.is_some()),
            ("divergence", self.targets.divergence.is_some()),
            ("task", self.targets.task.is_some()),
            ("nnuec_wdl", self.nnuec_wdl.is_some()),
        ] {
            check(
                !present || self.head_sources.contains_key(name),
                "known target lacks head provenance",
            )?;
        }
        match (&self.nnuec_input, &self.nnuec_input_sha256) {
            (None, None) => check(
                self.nnuec_wdl.is_none()
                    && self.nnuec_model_sha256.is_none()
                    && self.nnuec_semantics.is_none(),
                "NNUEC label without Rust reference input",
            )?,
            (Some(input), Some(seal)) => check(
                generation_digest(NNUEC_INPUT_DOMAIN, input)? == *seal
                    && self
                        .nnuec_model_sha256
                        .as_ref()
                        .is_some_and(|v| generation_sha(v))
                    && self
                        .nnuec_semantics
                        .as_ref()
                        .is_some_and(|v| generation_identifier(v)),
                "NNUEC Rust reference input seal",
            )?,
            _ => return Err(ManifestError::Integrity("partial NNUEC input seal".into())),
        }
        for (head, target) in [
            ("wdl", self.targets.wdl.as_ref()),
            ("nnuec_wdl", self.nnuec_wdl.as_ref()),
        ] {
            if let Some(wdl) = target {
                wdl.probabilities(&self.input)?;
                let h = &self.head_sources[head];
                check(
                    wdl.raw_evidence_sha256 == h.raw_receipt_sha256
                        && wdl.source == h.source
                        && wdl.observed_sequence == h.observed_sequence,
                    "WDL original head source/seal/sequence",
                )?;
                let authority: GenerationTrajectoryAuthorityV3 =
                    serde_json::from_value(h.raw_receipt["trajectory"].clone())
                        .map_err(|e| ManifestError::Integrity(e.to_string()))?;
                authority.validate()?;
                let rules = &h.raw_receipt["rules_receipt"];
                check(
                    h.raw_receipt["schema"] == HEAD_DOMAIN
                        && h.raw_receipt["kind"] == "rules_wdl"
                        && authority.game_id == self.input.game_id
                        && authority.anchors.contains(&self.input.rules_anchor)
                        && authority.ending == wdl.ending
                        && authority.outcome == wdl.outcome
                        && authority.sha256()? == wdl.trajectory_membership_sha256
                        && authority.result_evidence_sha256 == wdl.rules_result_sha256
                        && rules["schema"] == "rz-pals-rules-outcome-evidence/1"
                        && rules["rules_terminal"] == true
                        && rules["game_id"] == self.input.game_id
                        && rules["ending"] == serde_json::json!(wdl.ending)
                        && rules["outcome"] == serde_json::json!(wdl.outcome)
                        && generation_digest("rz-pals-rules-outcome-evidence/1", rules)?
                            == authority.result_evidence_sha256,
                    "WDL original Rules outcome/trajectory/anchor chain",
                )?;
            }
        }
        check(
            self.divergence_index.is_none_or(|i| {
                self.role == GenerationRoleV3::Critic
                    && usize::from(i) < self.input.divergence_record_ids.len()
            }),
            "divergence selected row index",
        )?;
        check(
            serde_json::to_vec(self)
                .map_err(|e| ManifestError::Integrity(e.to_string()))?
                .len()
                <= GENERATION_MAX_ROW_BYTES,
            "Rovent row byte cap",
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventCorpusOrigin {
    pub record_sha256: String,
    pub input_sha256: String,
    pub origin_sha256: String,
    pub split: GenerationSplitV3,
    pub game_id: String,
    pub opening_id: String,
    pub line_genealogy_id: String,
    pub transposition_sha256: String,
    pub actor_id: String,
    pub teacher_actor_id: Option<String>,
    pub ancestor_origin_sha256: Vec<String>,
}
impl RoventCorpusOrigin {
    pub fn from_record(r: &RoventTrainingRecord) -> Self {
        Self {
            record_sha256: r.record_sha256.clone(),
            input_sha256: r.input_sha256.clone(),
            origin_sha256: r.origin_sha256.clone(),
            split: r.split,
            game_id: r.input.game_id.clone(),
            opening_id: r.input.opening_id.clone(),
            line_genealogy_id: r.input.line_genealogy_id.clone(),
            transposition_sha256: r.input.transposition_sha256.clone(),
            actor_id: r.lineage.actor_id.clone(),
            teacher_actor_id: r.lineage.teacher_actor_id.clone(),
            ancestor_origin_sha256: r.lineage.ancestor_origin_sha256.clone(),
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventCorpusChunk {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    pub records: u64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventCorpusIndex {
    pub schema: String,
    pub manifest_sha256: String,
    pub record_count: u64,
    pub chunks: Vec<RoventCorpusChunk>,
    pub origins: Vec<RoventCorpusOrigin>,
}
impl RoventCorpusIndex {
    pub fn sha256(&self) -> Result<String, ManifestError> {
        generation_digest(INDEX_DOMAIN, self)
    }
    pub fn validate(&self) -> Result<(), ManifestError> {
        check(
            self.schema == INDEX_DOMAIN
                && generation_sha(&self.manifest_sha256)
                && self.record_count as usize == self.origins.len()
                && self.origins.len() <= MAX_CORPUS_ROWS
                && self.chunks.len() <= 8192
                && self
                    .chunks
                    .iter()
                    .try_fold(0u64, |n, c| n.checked_add(c.records))
                    == Some(self.record_count)
                && self
                    .chunks
                    .iter()
                    .try_fold(0u64, |n, c| n.checked_add(c.bytes))
                    .is_some_and(|n| n <= MAX_CORPUS_BYTES),
            "corpus index domain/count/bounds",
        )?;
        let mut paths = BTreeSet::new();
        let mut records = BTreeSet::new();
        for c in &self.chunks {
            check(
                c.path.len() <= 256
                    && !c.path.is_empty()
                    && !c.path.contains(['\\', ':'])
                    && c.path.split('/').all(|s| !matches!(s, "" | "." | ".."))
                    && c.bytes > 0
                    && c.bytes <= MAX_CHUNK_BYTES as u64
                    && c.records > 0
                    && generation_sha(&c.sha256)
                    && paths.insert(c.path.clone()),
                "corpus chunk path/pin/count",
            )?;
        }
        for o in &self.origins {
            check(
                [
                    &o.record_sha256,
                    &o.input_sha256,
                    &o.origin_sha256,
                    &o.transposition_sha256,
                ]
                .iter()
                .all(|v| generation_sha(v))
                    && [&o.game_id, &o.opening_id, &o.line_genealogy_id, &o.actor_id]
                        .iter()
                        .all(|v| generation_identifier(v))
                    && unique_shas(&o.ancestor_origin_sha256, 256)
                    && records.insert(o.record_sha256.clone()),
                "origin identity/seals",
            )?;
        }
        Ok(())
    }
}
/// Validates data only. It cannot reconstruct AcceptedRoleStep, a teacher
/// owner, a Rules trajectory capability or authorization to run training.
pub struct RoventCorpusReader {
    manifest: RoventCorpusManifest,
    index: RoventCorpusIndex,
    rules_replay_v2: bool,
}
impl RoventCorpusReader {
    pub fn new(
        manifest: RoventCorpusManifest,
        index: RoventCorpusIndex,
        ancestors: &[RoventCorpusIndex],
    ) -> Result<Self, ManifestError> {
        manifest.validate()?;
        index.validate()?;
        check(
            index.manifest_sha256 == manifest.sha256()?,
            "index manifest seal",
        )?;
        let mut actual = ancestors
            .iter()
            .map(|i| i.sha256())
            .collect::<Result<Vec<_>, _>>()?;
        actual.sort();
        check(
            actual == manifest.ancestor_index_sha256,
            "missing/extra global ancestor indices",
        )?;
        let mut namespaces = BTreeMap::<(&'static str, String), GenerationSplitV3>::new();
        let mut known_origins = BTreeSet::new();
        let mut known_inputs = BTreeSet::new();
        for ancestor in ancestors {
            ancestor.validate()?;
        }
        for ancestor in ancestors {
            for o in &ancestor.origins {
                known_origins.insert(o.origin_sha256.clone());
                known_inputs.insert(o.input_sha256.clone());
            }
        }
        for i in ancestors.iter().chain(std::iter::once(&index)) {
            for o in &i.origins {
                for (namespace, value) in [
                    ("game", &o.game_id),
                    ("opening", &o.opening_id),
                    ("genealogy", &o.line_genealogy_id),
                    ("state", &o.transposition_sha256),
                    ("input", &o.input_sha256),
                    ("origin", &o.origin_sha256),
                ] {
                    if let Some(old) = namespaces.insert((namespace, value.clone()), o.split) {
                        check(old == o.split, "global ancestor/holdout leakage")?;
                    }
                }
            }
        }
        check(
            manifest
                .inherited_input_sha256
                .iter()
                .all(|v| known_inputs.contains(v))
                && manifest
                    .inherited_origin_sha256
                    .iter()
                    .all(|v| known_origins.contains(v))
                && index.origins.iter().all(|o| {
                    o.ancestor_origin_sha256
                        .iter()
                        .all(|v| known_origins.contains(v))
                }),
            "unresolved inherited/global ancestor origin",
        )?;
        Ok(Self {
            manifest,
            index,
            rules_replay_v2: false,
        })
    }
    /// Explicit /2 data decoder. This does not create a current native,
    /// physical, Rules-owner or optimizer capability from historical bytes.
    pub fn new_rules_replay_v2(
        manifest: RoventCorpusManifest,
        index: RoventCorpusIndex,
        ancestors: &[RoventCorpusIndex],
    ) -> Result<Self, ManifestError> {
        let mut reader = Self::new(manifest, index, ancestors)?;
        reader.rules_replay_v2 = true;
        Ok(reader)
    }
    pub fn manifest(&self) -> &RoventCorpusManifest {
        &self.manifest
    }
    pub fn index(&self) -> &RoventCorpusIndex {
        &self.index
    }
    pub fn read_chunk(
        &self,
        chunk: usize,
        bytes: &[u8],
    ) -> Result<Vec<RoventTrainingRecord>, ManifestError> {
        let c = self
            .index
            .chunks
            .get(chunk)
            .ok_or_else(|| ManifestError::Integrity("chunk slot".into()))?;
        check(
            bytes.len() as u64 == c.bytes
                && bytes.len() <= MAX_CHUNK_BYTES
                && crate::digest(bytes) == c.sha256,
            "chunk original file pin",
        )?;
        let text = std::str::from_utf8(bytes).map_err(|e| ManifestError::Parse(e.to_string()))?;
        let mut rows = Vec::new();
        let mut seen = BTreeSet::new();
        for line in text.lines() {
            check(
                !line.is_empty()
                    && line.len() <= GENERATION_MAX_ROW_BYTES
                    && rows.len() < c.records as usize,
                "bounded JSONL row",
            )?;
            let row: RoventTrainingRecord = decode_json(line)?;
            if self.rules_replay_v2 {
                row.validate_rules_replay_v2(&self.manifest)?;
            } else {
                row.validate(&self.manifest)?;
            }
            check(
                seen.insert(row.record_sha256.clone())
                    && self
                        .index
                        .origins
                        .iter()
                        .any(|o| *o == RoventCorpusOrigin::from_record(&row)),
                "record missing/changed origin index",
            )?;
            rows.push(row);
        }
        check(rows.len() as u64 == c.records, "chunk row count")?;
        Ok(rows)
    }
    /// Whole-corpus membership is checked only when every pinned chunk is
    /// supplied. Independently valid chunks may not duplicate one index row
    /// while silently omitting another one.
    /// The callback may stage data, but must not start optimization before
    /// this whole pass returns success. A late corrupt chunk invalidates it.
    pub fn visit_all_chunks<'a>(
        &self,
        chunks: impl IntoIterator<Item = &'a [u8]>,
        mut consume: impl FnMut(RoventTrainingRecord) -> Result<(), ManifestError>,
    ) -> Result<(), ManifestError> {
        let mut count = 0usize;
        let mut seen = BTreeSet::new();
        for (slot, bytes) in chunks.into_iter().enumerate() {
            check(slot < self.index.chunks.len(), "extra corpus chunk")?;
            count += 1;
            for row in self.read_chunk(slot, bytes)? {
                check(
                    seen.insert(row.record_sha256.clone()),
                    "duplicate record across original chunks",
                )?;
                consume(row)?;
            }
        }
        check(
            count == self.index.chunks.len()
                && seen.len() == self.index.origins.len()
                && self
                    .index
                    .origins
                    .iter()
                    .all(|o| seen.contains(&o.record_sha256)),
            "corpus origin index not fully covered by original chunks",
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn manifest() -> RoventCorpusManifest {
        RoventCorpusManifest {
            schema: CORPUS_DOMAIN.into(),
            corpus_id: "corpus".into(),
            source_commit: "a".repeat(40),
            source_sha256: "a".repeat(64),
            runtime_sha256: "b".repeat(64),
            model_semantics_sha256: "c".repeat(64),
            model_epoch: 1,
            context_epoch: 1,
            recipe_sha256: "d".repeat(64),
            stage_id: "stage".into(),
            round: 0,
            round_id: "common-round".into(),
            student: "student".into(),
            learner: "student".into(),
            one_learner: true,
            inherited_input_sha256: vec![],
            inherited_origin_sha256: vec![],
            teachers: BTreeMap::new(),
            opening_splits: BTreeMap::from([("opening".into(), GenerationSplitV3::Train)]),
            ancestor_index_sha256: vec![],
        }
    }
    fn origin(split: GenerationSplitV3) -> RoventCorpusOrigin {
        RoventCorpusOrigin {
            record_sha256: "1".repeat(64),
            input_sha256: "2".repeat(64),
            origin_sha256: "3".repeat(64),
            split,
            game_id: "game".into(),
            opening_id: "opening".into(),
            line_genealogy_id: "line".into(),
            transposition_sha256: "4".repeat(64),
            actor_id: "actor".into(),
            teacher_actor_id: None,
            ancestor_origin_sha256: vec![],
        }
    }
    fn index(manifest_sha256: String, origin: RoventCorpusOrigin) -> RoventCorpusIndex {
        RoventCorpusIndex {
            schema: INDEX_DOMAIN.into(),
            manifest_sha256,
            record_count: 1,
            chunks: vec![RoventCorpusChunk {
                path: "chunks/000000.jsonl".into(),
                bytes: 1,
                sha256: "5".repeat(64),
                records: 1,
            }],
            origins: vec![origin],
        }
    }
    #[test]
    fn teacher_pin_is_not_stockfish_only_or_a_runtime_receipt() {
        let pin = RoventTeacherPin {
            engine: "reckless".into(),
            source_commit: "a".repeat(40),
            binary_sha256: "b".repeat(64),
            binary_bytes: 1,
            profile_sha256: "c".repeat(64),
            canonical_sha256: "d".repeat(64),
            capability_sha256: "e".repeat(64),
            options: BTreeMap::new(),
            assets: BTreeMap::new(),
        };
        assert!(pin.validate().is_ok());
        let mut wrong = pin;
        wrong.binary_sha256 = "Stockfish 19".into();
        assert!(wrong.validate().is_err());
    }
    #[test]
    fn index_rejects_traversal_and_unframed_count() {
        let mut i = index("a".repeat(64), origin(GenerationSplitV3::Train));
        assert!(i.validate().is_ok());
        i.chunks[0].path = "../chunk.jsonl".into();
        assert!(i.validate().is_err());
        i.chunks[0].path = "chunks/000000.jsonl".into();
        i.origins.clear();
        assert!(i.validate().is_err());
    }
    #[test]
    fn current_row_cannot_supply_its_own_inherited_authority() {
        let mut m = manifest();
        let o = origin(GenerationSplitV3::Train);
        m.inherited_origin_sha256 = vec![o.origin_sha256.clone()];
        let i = index(m.sha256().unwrap(), o);
        assert!(RoventCorpusReader::new(m, i, &[]).is_err());
    }
    #[test]
    fn transposition_leakage_is_checked_across_global_ancestors() {
        let a = index("6".repeat(64), origin(GenerationSplitV3::Holdout));
        let mut m = manifest();
        m.ancestor_index_sha256 = vec![a.sha256().unwrap()];
        let mut o = origin(GenerationSplitV3::Train);
        o.game_id = "other-game".into();
        o.opening_id = "other-opening".into();
        o.line_genealogy_id = "other-line".into();
        o.record_sha256 = "7".repeat(64);
        o.input_sha256 = "8".repeat(64);
        o.origin_sha256 = "9".repeat(64);
        let i = index(m.sha256().unwrap(), o);
        assert!(RoventCorpusReader::new(m, i, &[a]).is_err());
    }
    #[test]
    fn native_head_digest_preserves_f32_bits_and_rejects_nonfinite_or_missing_latent() {
        let mut raw = serde_json::json!({"payload":{"kind":"proposal","candidate_logits_bits":[0],"wdl_bits":[1f32.to_bits(),0,0]},
            "private_latent_bits":vec![0u32;6144]});
        let positive = native_output_digest(&raw, "proposer").unwrap();
        raw["payload"]["candidate_logits_bits"][0] = serde_json::json!((-0.0f32).to_bits());
        assert_ne!(positive, native_output_digest(&raw, "proposer").unwrap());
        raw["payload"]["candidate_logits_bits"][0] = serde_json::json!(f32::NAN.to_bits());
        assert!(native_output_digest(&raw, "proposer").is_err());
        raw["payload"]["candidate_logits_bits"][0] = serde_json::json!(0);
        raw["private_latent_bits"] = serde_json::json!([]);
        assert!(native_output_digest(&raw, "proposer").is_err());
    }
}
