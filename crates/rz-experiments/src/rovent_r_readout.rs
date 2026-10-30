//! 별도 actual-R readout 자료의 순수 wire codec 초안이다.
//!
//! P/C native receipt를 합성하지 않으며, decode/seal은 live CPU·Rules·학습
//! 권한을 발급하지 않는다. 실제 arena producer만 원 capture/acceptance
//! custody를 소비한다. 기존 PALS 및 Rovent P/C codec은 변경하지 않는다.
use crate::ManifestError;
use crate::pals_data_v3::{
    GenerationRulesAnchorV3, GenerationSplitV3, GenerationTrajectoryAuthorityV3,
    GenerationWdlTargetV3, generation_digest, generation_identifier, generation_sha,
    validate_generation_moves,
};
use crate::rovent_corpus::{
    MAX_CHUNK_BYTES, MAX_CORPUS_BYTES, MAX_CORPUS_ROWS, RoventCorpusChunk, RoventCorpusManifest,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const R_READOUT_RECORD_DOMAIN: &str = "rz-rovent-r-readout-record/1";
pub const R_READOUT_MANIFEST_DOMAIN: &str = "rz-rovent-r-readout-corpus-manifest/1";
pub const R_READOUT_INDEX_DOMAIN: &str = "rz-rovent-r-readout-corpus-index/1";
pub const R_READOUT_ORIGIN_DOMAIN: &str = "rz-rovent-r-readout-origin/1";
pub const R_READOUT_INPUT_DOMAIN: &str = "rz-rovent-r-readout-semantic-input/1";
pub const R_READOUT_FLOAT_BITS_DOMAIN: &str = "rz-rovent-r-readout-f32-framing/1";
pub const R_CAPTURE_INPUT_DOMAIN: &str = "rz-nnuec-actual-r-readout-input/1";
pub const R_READOUT_WDL_DOMAIN: &str = "rz-rovent-r-readout-wdl-evidence/1";
pub const R_READOUT_RETIREMENT_DOMAIN: &str = "rz-rovent-r-readout-retirement/1";
pub const MAX_R_READOUT_BYTES: usize = 16 * 1024 * 1024;
const MAX_R_DEPTH: usize = 64;

fn require(ok: bool, why: &str) -> Result<(), ManifestError> {
    if ok {
        Ok(())
    } else {
        Err(ManifestError::Integrity(why.into()))
    }
}
fn raw_sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn ordered_shas(values: &[String], cap: usize) -> bool {
    values.len() <= cap
        && values.iter().all(|s| generation_sha(s))
        && values.windows(2).all(|v| v[0] < v[1])
}

/// New-domain binary framing, independent of Rust/Python JSON exponent spelling.
/// Tags: null=0, false=1, true=2, unsigned=3, negative magnitude=4,
/// finite FP32 bits=5, UTF8=6, array=7, object=8. Lengths are u64 LE;
/// object keys are sorted by UTF8 bytes. JSON float 1.0 stays distinct from 1.
/// This API only computes a digest; it does not verify an original owner.
pub fn r_readout_f32_digest(domain: &str, value: &Value) -> Result<String, ManifestError> {
    fn length(hash: &mut Sha256, n: usize) -> Result<(), ManifestError> {
        hash.update(
            u64::try_from(n)
                .map_err(|_| ManifestError::Integrity("R frame length".into()))?
                .to_le_bytes(),
        );
        Ok(())
    }
    fn walk(hash: &mut Sha256, value: &Value, depth: usize) -> Result<(), ManifestError> {
        require(depth <= MAX_R_DEPTH, "R semantic input recursion bound")?;
        match value {
            Value::Null => hash.update([0]),
            Value::Bool(false) => hash.update([1]),
            Value::Bool(true) => hash.update([2]),
            Value::Number(number) if number.is_f64() => {
                let number = number
                    .as_f64()
                    .ok_or_else(|| ManifestError::Integrity("R float".into()))?
                    as f32;
                require(number.is_finite(), "R semantic input finite f32")?;
                hash.update([5]);
                hash.update(number.to_bits().to_le_bytes());
            }
            Value::Number(number) => {
                if let Some(n) = number.as_u64() {
                    hash.update([3]);
                    hash.update(n.to_le_bytes());
                } else {
                    let n = number
                        .as_i64()
                        .ok_or_else(|| ManifestError::Integrity("R integer".into()))?;
                    hash.update([4]);
                    hash.update(n.unsigned_abs().to_le_bytes());
                }
            }
            Value::String(text) => {
                hash.update([6]);
                length(hash, text.len())?;
                hash.update(text.as_bytes());
            }
            Value::Array(values) => {
                hash.update([7]);
                length(hash, values.len())?;
                for value in values {
                    walk(hash, value, depth + 1)?;
                }
            }
            Value::Object(values) => {
                hash.update([8]);
                length(hash, values.len())?;
                let values = values.iter().collect::<BTreeMap<_, _>>();
                for (key, value) in values {
                    length(hash, key.len())?;
                    hash.update(key.as_bytes());
                    walk(hash, value, depth + 1)?;
                }
            }
        }
        Ok(())
    }
    require(generation_identifier(domain), "R semantic input domain")?;
    let mut hash = Sha256::new();
    hash.update(domain.as_bytes());
    hash.update([0]);
    walk(&mut hash, value, 0)?;
    Ok(format!("{:x}", hash.finalize()))
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRAuthority {
    pub process_epoch: u64,
    pub game_generation: u64,
    pub root_generation: u64,
    pub implementation_sha256: String,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRJob {
    pub authority: RoventRAuthority,
    pub job_sequence: u64,
    pub state_owner: u64,
    pub state_revision: u64,
    pub state_semantic_sha256: String,
    pub question_sha256: String,
    pub root_restriction_sha256: String,
    pub evaluation_namespace_sha256: String,
    pub search_conditions_sha256: String,
    pub total_native_node_credit: u64,
    pub hard_deadline_clock_epoch: u64,
    pub hard_deadline_ns: u64,
}
impl RoventRJob {
    fn validate(&self) -> Result<(), ManifestError> {
        require(
            self.authority.process_epoch > 0
                && generation_sha(&self.authority.implementation_sha256)
                && self.job_sequence > 0
                && self.total_native_node_credit > 0
                && self.hard_deadline_clock_epoch == self.authority.process_epoch
                && [
                    &self.state_semantic_sha256,
                    &self.question_sha256,
                    &self.root_restriction_sha256,
                    &self.evaluation_namespace_sha256,
                    &self.search_conditions_sha256,
                ]
                .iter()
                .all(|s| generation_sha(s)),
            "R original CPU Job identity/credit/deadline",
        )
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRQuestion {
    pub job: RoventRJob,
    pub worker_owner: u16,
    pub frame_sequence: u64,
    pub capture_sequence: u64,
    /// Original zero-based score-attempt sequence; sampling may leave gaps.
    pub readout_sequence: u64,
    pub observed_tick_ns: u64,
}

/// Original replay body belongs to the existing Rules codec. This wrapper
/// preserves it losslessly; arena historical verification restores it through
/// that single owner rather than implementing Rules a second time here.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRRulesEvidence {
    pub anchor: GenerationRulesAnchorV3,
    pub replay: Value,
    pub replay_sha256: String,
}
impl RoventRRulesEvidence {
    fn validate(&self) -> Result<(), ManifestError> {
        self.anchor.validate()?;
        require(
            self.replay["schema"] == "rz-rovent-rules-position-replay/1"
                && self.replay["anchor"]
                    == serde_json::to_value(&self.anchor)
                        .map_err(|e| ManifestError::Integrity(e.to_string()))?
                && generation_digest("rz-rovent-rules-position-replay/1", &self.replay)?
                    == self.replay_sha256,
            "R original Rules replay body/seal",
        )
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRContextRecord {
    /// NNUEC V2 context line IDs are original selected-view local indices.
    pub local_id: u64,
    pub record_id: String,
    pub record_revision: u64,
    pub parent_revision: Option<u64>,
    pub supersedes_revision: Option<u64>,
    pub origin_state: u64,
    pub kind: u8,
    pub anchor: u8,
    pub moves: Vec<u16>,
    pub original_first_move: Option<u16>,
    pub parent: Option<u16>,
    pub supersedes: Option<u16>,
    pub source_state: u64,
    pub source_revision: u64,
    pub source_line_offset: u16,
    /// Original raw observations, never extra numerical NNUEC features.
    pub record_perspective: String,
    pub encoded_perspective: String,
    pub completed_depth: u16,
    pub score_scope: Option<String>,
    pub value: Option<i32>,
    pub cpu_observation: Option<u64>,
    pub critical: bool,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRContextQuery {
    pub purpose: String,
    pub prefix: Vec<u16>,
    pub proposal: Vec<u16>,
    pub counterexample: Option<Vec<u16>>,
    pub legal: Vec<u16>,
    pub divergences: Vec<u16>,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRSelectionPolicy {
    pub mode: String,
    pub token_budget: u16,
    pub max_records: u16,
    pub max_anchors: u8,
    pub required_bundle: Option<u64>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRBoundEvidence {
    pub model_semantics: String,
    /// Actual parsed parameter identity, not a checkpoint file's raw SHA.
    pub parameter_sha256: String,
    pub training_state: Value,
    /// Exact original loaded checkpoint file SHA, observed by loaded source custody.
    pub weights_file_sha256: String,
    /// Collection declaration epochs, never an invented NNUEC model epoch.
    pub collection_model_epoch: u64,
    pub collection_context_epoch: u64,
    pub root_state: u64,
    pub generation: u64,
    pub sequence: u64,
    pub full_view_sha256: String,
    pub view_seal_sha256: String,
    pub record_scope_sha256: String,
    pub policy_sha256: String,
    pub context_sha256: String,
    pub context_query: RoventRContextQuery,
    pub selection_policy: RoventRSelectionPolicy,
    pub actual_tokens: u16,
    pub token_budget: u16,
    pub length_bucket: u16,
    pub context_records: Vec<RoventRContextRecord>,
    pub context_anchors: Vec<RoventRRulesEvidence>,
    /// Original Request source snapshots in the exact selected-record order.
    /// Numerical/rebased anchors cannot replace these historical prefixes.
    pub context_source_anchors: Vec<RoventRRulesEvidence>,
    pub context_anchor_states: Vec<u64>,
    pub local_anchor: RoventRRulesEvidence,
    pub local_encoded_perspective: String,
    pub current_rules: RoventRRulesEvidence,
}
impl RoventRBoundEvidence {
    fn validate(&self) -> Result<(), ManifestError> {
        require(
            generation_identifier(&self.model_semantics)
                && ["bootstrap", "untrained", "learned"]
                    .iter()
                    .any(|kind| self.training_state["kind"] == *kind)
                && self.collection_model_epoch > 0
                && self.collection_context_epoch > 0
                && self.sequence > 0
                && self.context_records.len() <= 64
                && self.actual_tokens <= self.token_budget
                && self.token_budget > 0
                && self.token_budget <= 512
                && [8, 16, 32, 64, 128, 256].contains(&self.length_bucket)
                && (1..=4).contains(&self.context_anchors.len())
                && self.context_source_anchors.len() == self.context_records.len()
                && self.context_anchor_states.len() == self.context_anchors.len()
                && ["current_root_only", "anchored_compact"]
                    .contains(&self.selection_policy.mode.as_str())
                && self.selection_policy.token_budget == self.token_budget
                && [128, 256, 512].contains(&self.selection_policy.token_budget)
                && (1..=64).contains(&self.selection_policy.max_records)
                && (1..=4).contains(&self.selection_policy.max_anchors)
                && self.context_records.len() <= usize::from(self.selection_policy.max_records)
                && self.context_anchors.len() <= usize::from(self.selection_policy.max_anchors)
                && (self.selection_policy.mode != "current_root_only"
                    || self.selection_policy.max_anchors == 1)
                && [
                    &self.parameter_sha256,
                    &self.weights_file_sha256,
                    &self.full_view_sha256,
                    &self.view_seal_sha256,
                    &self.record_scope_sha256,
                    &self.policy_sha256,
                    &self.context_sha256,
                ]
                .iter()
                .all(|s| generation_sha(s)),
            "R original Bound/model/context/view declaration",
        )?;
        for (i, r) in self.context_records.iter().enumerate() {
            require(
                r.local_id == i as u64
                    && generation_identifier(&r.record_id)
                    && r.kind <= 3
                    && usize::from(r.source_line_offset)
                        .checked_add(r.moves.len())
                        .is_some_and(|length| length <= 256)
                    && ["white", "black"].contains(&r.record_perspective.as_str())
                    && usize::from(r.anchor) < self.context_anchors.len()
                    && r.origin_state == self.context_anchor_states[usize::from(r.anchor)]
                    && r.encoded_perspective
                        == if self.context_anchors[usize::from(r.anchor)]
                            .anchor
                            .white_to_move
                        {
                            "white"
                        } else {
                            "black"
                        }
                    && match (r.parent, r.parent_revision) {
                        (None, None) => true,
                        (Some(p), Some(revision)) => {
                            usize::from(p) < i
                                && self.context_records[usize::from(p)].record_revision == revision
                        }
                        _ => false,
                    }
                    && match (r.supersedes, r.supersedes_revision) {
                        (None, None) => true,
                        (Some(p), Some(revision)) => {
                            usize::from(p) < i
                                && self.context_records[usize::from(p)].record_revision == revision
                        }
                        _ => false,
                    },
                "R exact selected-context local mapping",
            )?;
            validate_generation_moves(&r.moves, 256)?;
            if let Some(movement) = r.original_first_move {
                validate_generation_moves(&[movement], 1)?;
            }
        }
        require(
            generation_identifier(&self.context_query.purpose)
                && self.context_query.divergences.len() <= 256
                && self.context_query.divergences.iter().all(|p| *p <= 256),
            "R original context query",
        )?;
        for line in [
            &self.context_query.prefix,
            &self.context_query.proposal,
            &self.context_query.legal,
        ] {
            validate_generation_moves(line, 256)?;
        }
        if let Some(line) = &self.context_query.counterexample {
            validate_generation_moves(line, 256)?;
        }
        for anchor in &self.context_anchors {
            anchor.validate()?;
        }
        for anchor in &self.context_source_anchors {
            anchor.validate()?;
        }
        self.local_anchor.validate()?;
        require(
            self.local_encoded_perspective
                == if self.local_anchor.anchor.white_to_move {
                    "white"
                } else {
                    "black"
                },
            "R original local anchor perspective",
        )?;
        self.current_rules.validate()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRInputEvidence {
    /// Exact original Rust UTF8 serialization. Never Python-reserialize this to
    /// reconstruct the native seal; JSON spelling and object order are material.
    pub reference_input_json: String,
    pub reference_input_bytes: u64,
    pub reference_input_raw_sha256: String,
    pub native_input_sha256: String,
    pub semantic_input_sha256: String,
}
impl RoventRInputEvidence {
    pub fn from_original_json(
        original: String,
        bound: &RoventRBoundEvidence,
        original_native_sha256: &str,
    ) -> Result<Self, ManifestError> {
        require(
            original.len() <= MAX_R_READOUT_BYTES,
            "R original input byte cap",
        )?;
        let parsed: Value = crate::decode_json(&original)?;
        require(
            parsed["schema"] == "rz-nnuec-fp32-input/2",
            "R actual capture input domain; legacy V1 is not relabeled",
        )?;
        let mut hash = Sha256::new();
        hash.update(R_CAPTURE_INPUT_DOMAIN.as_bytes());
        hash.update([0]);
        hash.update(original.as_bytes());
        let native = format!("{:x}", hash.finalize());
        require(
            native == original_native_sha256,
            "R capture's original native input seal",
        )?;
        let semantic = r_readout_f32_digest(
            R_READOUT_INPUT_DOMAIN,
            &serde_json::json!({
                "model_semantics":bound.model_semantics,"parameter_sha256":bound.parameter_sha256,
                "collection_model_epoch":bound.collection_model_epoch,"collection_context_epoch":bound.collection_context_epoch,"reference_input":parsed
            }),
        )?;
        Ok(Self {
            reference_input_bytes: original.len() as u64,
            reference_input_raw_sha256: raw_sha(original.as_bytes()),
            reference_input_json: original,
            native_input_sha256: native,
            semantic_input_sha256: semantic,
        })
    }
    pub fn validate(&self, bound: &RoventRBoundEvidence) -> Result<(), ManifestError> {
        let expected = Self::from_original_json(
            self.reference_input_json.clone(),
            bound,
            &self.native_input_sha256,
        )?;
        require(
            &expected == self,
            "R original raw/native/semantic input seal separation",
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRNumericBits {
    pub schema: String,
    pub parameter_sha256: String,
    pub board_accumulators_bits: [Vec<u32>; 2],
    pub context_accumulator_bits: Vec<u32>,
    pub local_accumulator_bits: Vec<u32>,
    pub hidden_bits: Vec<u32>,
    pub raw_outputs_bits: [u32; 11],
    pub value_bits: u32,
    pub wdl_logits_bits: [u32; 3],
    pub wdl_bits: [u32; 3],
}
impl RoventRNumericBits {
    fn validate(&self, bound: &RoventRBoundEvidence) -> Result<(), ManifestError> {
        require(
            self.schema == "rz-nnuec-fp32-output/2"
                && self.parameter_sha256 == bound.parameter_sha256
                && self
                    .board_accumulators_bits
                    .iter()
                    .all(|v| !v.is_empty() && v.len() <= 4096)
                && [
                    &self.context_accumulator_bits,
                    &self.local_accumulator_bits,
                    &self.hidden_bits,
                ]
                .iter()
                .all(|v| !v.is_empty() && v.len() <= 4096),
            "R numeric output model/shape",
        )?;
        for bits in self
            .board_accumulators_bits
            .iter()
            .flat_map(|v| v.iter())
            .chain(self.context_accumulator_bits.iter())
            .chain(self.local_accumulator_bits.iter())
            .chain(self.hidden_bits.iter())
            .chain(self.raw_outputs_bits.iter())
            .chain(std::iter::once(&self.value_bits))
            .chain(self.wdl_logits_bits.iter())
            .chain(self.wdl_bits.iter())
        {
            require(
                f32::from_bits(*bits).is_finite(),
                "R finite original f32 output",
            )?;
        }
        let w = self.wdl_bits.map(f32::from_bits);
        require(
            w.iter().all(|v| (0.0..=1.0).contains(v))
                && (w.iter().sum::<f32>() - 1.0).abs() <= 1e-4,
            "R numeric WDL prediction",
        )
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RoventRCaptureEnding {
    CpuTerminal { completion: String },
    NotStarted,
    Failed,
    Unknown,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRCaptureLimits {
    pub maximum_readouts: u16,
    pub every_nth_score: u32,
    pub maximum_input_bytes: u64,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRCaptureRetirement {
    pub job: RoventRJob,
    pub worker_owner: u16,
    pub frame_sequence: u64,
    pub capture_sequence: u64,
    pub ending: RoventRCaptureEnding,
    pub limits: RoventRCaptureLimits,
    pub attempted: u64,
    pub selected: u64,
    pub retained_readouts: u16,
    pub overflow: u64,
    /// Retained serialized input bytes, not heap or temporary peak memory.
    pub retained_input_bytes: u64,
    /// Selected-input preparation only, not job/action wall time.
    pub selected_input_preparation_ns: u64,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRPcGenealogy {
    pub record_sha256: String,
    pub input_sha256: String,
    pub origin_sha256: String,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRLineage {
    pub game_id: String,
    pub opening_id: String,
    pub line_genealogy_id: String,
    pub transposition_sha256: String,
    pub actor_id: String,
    pub teacher_actor_id: Option<String>,
    pub ancestor_origin_sha256: Vec<String>,
    pub pc_genealogy: Option<RoventRPcGenealogy>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRWdlEvidence {
    pub target: GenerationWdlTargetV3,
    /// Original Rules membership/result and full original actual trajectory;
    /// the arena's single Rules owner replays it before historical eligibility.
    pub raw_receipt: Value,
    pub raw_receipt_sha256: String,
}
/// The separate seven-output utility shadow is not part of the base 11-output
/// NNUEC, nor a seven-way task supervisor. An actual action/closure witness
/// must be added as a separate explicit domain before known targets exist.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRUtilityState {
    pub support_mask: [bool; 7],
    pub target_known_mask: [bool; 7],
    pub unknown_reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRReadoutRecord {
    pub schema: String,
    pub manifest_sha256: String,
    pub split: GenerationSplitV3,
    pub lineage: RoventRLineage,
    pub native_admission: Value,
    pub native_admission_sha256: String,
    pub root_descriptor: crate::RoventRootDescriptor,
    pub root_descriptor_sha256: String,
    pub startup_input_lock_sha256: String,
    pub question: RoventRQuestion,
    pub bound: RoventRBoundEvidence,
    pub input: RoventRInputEvidence,
    pub numeric: RoventRNumericBits,
    pub retirement: RoventRCaptureRetirement,
    /// Historical projection seal; it never reconstructs a worker capability.
    pub retirement_sha256: String,
    /// Missing current semantic acceptance stays unknown, never "accepted=true".
    pub semantic_consumption: Option<Value>,
    pub whole_action_cost: Option<Value>,
    pub wdl: Option<RoventRWdlEvidence>,
    pub utility: RoventRUtilityState,
    pub origin_sha256: String,
    pub record_sha256: String,
}
impl RoventRReadoutRecord {
    fn origin_digest(&self) -> Result<String, ManifestError> {
        generation_digest(
            R_READOUT_ORIGIN_DOMAIN,
            &serde_json::json!({
                "manifest_sha256":self.manifest_sha256,"lineage":self.lineage,"question":self.question,
                "native_admission_sha256":self.native_admission_sha256,"root_descriptor_sha256":self.root_descriptor_sha256,
                "native_input_sha256":self.input.native_input_sha256,"semantic_input_sha256":self.input.semantic_input_sha256
            }),
        )
    }
    fn record_digest(&self) -> Result<String, ManifestError> {
        let mut value =
            serde_json::to_value(self).map_err(|e| ManifestError::Integrity(e.to_string()))?;
        value
            .as_object_mut()
            .ok_or_else(|| ManifestError::Integrity("R record object".into()))?
            .remove("record_sha256");
        // All numeric outputs are u32 words. Parsed numeric inputs are sealed
        // separately; the original UTF8 string remains material to this record.
        r_readout_f32_digest(R_READOUT_RECORD_DOMAIN, &value)
    }
    pub fn seal(&mut self, manifest: &RoventRReadoutManifest) -> Result<(), ManifestError> {
        self.retirement_sha256 = generation_digest(R_READOUT_RETIREMENT_DOMAIN, &self.retirement)?;
        self.origin_sha256 = self.origin_digest()?;
        self.record_sha256 = self.record_digest()?;
        self.validate(manifest)
    }
    pub fn validate(&self, manifest: &RoventRReadoutManifest) -> Result<(), ManifestError> {
        manifest.validate()?;
        self.question.job.validate()?;
        self.bound.validate()?;
        self.input.validate(&self.bound)?;
        self.numeric.validate(&self.bound)?;
        self.root_descriptor.validate()?;
        let descriptor = &self.root_descriptor.input;
        let authority = &self.question.job.authority;
        require(
            self.schema == R_READOUT_RECORD_DOMAIN
                && self.manifest_sha256 == manifest.sha256()?
                && self.question.frame_sequence > 0
                && self.question.capture_sequence > 0
                && self.question.job.authority.root_generation == self.bound.generation
                && self.bound.collection_model_epoch == manifest.collection.model_epoch
                && self.bound.collection_context_epoch == manifest.collection.context_epoch
                && self.retirement.job == self.question.job
                && self.retirement.worker_owner == self.question.worker_owner
                && self.retirement.frame_sequence == self.question.frame_sequence
                && self.retirement.capture_sequence == self.question.capture_sequence
                && generation_digest(R_READOUT_RETIREMENT_DOMAIN, &self.retirement)?
                    == self.retirement_sha256
                && self.retirement.selected > 0
                && self.question.readout_sequence < self.retirement.attempted
                && (1..=64).contains(&self.retirement.limits.maximum_readouts)
                && self.retirement.limits.every_nth_score > 0
                && self.question.readout_sequence
                    % u64::from(self.retirement.limits.every_nth_score)
                    == 0
                && self.retirement.selected <= u64::from(self.retirement.limits.maximum_readouts)
                && self.retirement.retained_readouts > 0
                && u64::from(self.retirement.retained_readouts) <= self.retirement.selected
                && self.retirement.selected <= self.retirement.attempted
                && self.retirement.limits.maximum_input_bytes > 0
                && self.retirement.limits.maximum_input_bytes <= MAX_R_READOUT_BYTES as u64
                && self.retirement.retained_input_bytes
                    <= self.retirement.limits.maximum_input_bytes
                && self.input.reference_input_bytes <= self.retirement.retained_input_bytes
                && self
                    .retirement
                    .error
                    .as_ref()
                    .is_none_or(|e| generation_identifier(e))
                && [
                    &self.native_admission_sha256,
                    &self.root_descriptor_sha256,
                    &self.startup_input_lock_sha256,
                ]
                .iter()
                .all(|s| generation_sha(s))
                && self.native_admission["schema"] == "rz-rovent-collection-root-admission/1"
                && generation_digest(
                    "rz-rovent-collection-root-admission/1",
                    &self.native_admission,
                )? == self.native_admission_sha256
                && self.root_descriptor_sha256
                    == self
                        .root_descriptor
                        .digest()?
                        .0
                        .iter()
                        .map(|v| format!("{v:02x}"))
                        .collect::<String>()
                && descriptor.startup_lock_sha256 == self.startup_input_lock_sha256
                && descriptor.authority.epoch == authority.process_epoch
                && descriptor.authority.game == authority.game_generation
                && descriptor.authority.root == authority.root_generation
                && descriptor.authority.implementation_sha256 == authority.implementation_sha256
                && self.question.job.hard_deadline_clock_epoch == descriptor.clock.hard.clock_epoch
                && self.question.job.hard_deadline_ns <= descriptor.clock.hard.at_ns
                && self.question.job.state_semantic_sha256
                    == self.bound.local_anchor.anchor.rules_state_sha256,
            "R original question/readout/retirement/admission framing",
        )?;
        require(
            manifest
                .collection
                .opening_splits
                .get(&self.lineage.opening_id)
                == Some(&self.split)
                && [
                    &self.lineage.game_id,
                    &self.lineage.opening_id,
                    &self.lineage.line_genealogy_id,
                    &self.lineage.actor_id,
                ]
                .iter()
                .all(|s| generation_identifier(s))
                && generation_sha(&self.lineage.transposition_sha256)
                && ordered_shas(&self.lineage.ancestor_origin_sha256, 256)
                && self
                    .lineage
                    .teacher_actor_id
                    .as_ref()
                    .is_none_or(|a| manifest.collection.teachers.contains_key(a)),
            "R role-less lineage/split",
        )?;
        if let Some(link) = &self.lineage.pc_genealogy {
            require(
                [&link.record_sha256, &link.input_sha256, &link.origin_sha256]
                    .iter()
                    .all(|s| generation_sha(s)),
                "R optional PC genealogy seals",
            )?;
        }
        // Until actual typed consumption/cost witnesses are connected there is
        // no public JSON route that turns a worker observation into acceptance.
        require(
            self.semantic_consumption.is_none() && self.whole_action_cost.is_none(),
            "R missing original semantic/cost witness",
        )?;
        require(
            self.utility.support_mask == [false; 7]
                && self.utility.target_known_mask == [false; 7]
                && generation_identifier(&self.utility.unknown_reason),
            "V-free R utility/task targets remain unknown",
        )?;
        if let Some(wdl) = &self.wdl {
            let raw = &wdl.raw_receipt;
            let fields = [
                "schema",
                "kind",
                "native_input_sha256",
                "semantic_input_sha256",
                "anchor",
                "trajectory",
                "rules_receipt",
                "rules_replay",
            ];
            let trajectory: GenerationTrajectoryAuthorityV3 =
                serde_json::from_value(raw["trajectory"].clone())
                    .map_err(|e| ManifestError::Integrity(e.to_string()))?;
            trajectory.validate()?;
            let result_fields = ["schema", "game_id", "ending", "outcome", "rules_terminal"];
            require(
                wdl.target.source == "rules_selfplay_trajectory"
                    && wdl.target.game_id == self.lineage.game_id
                    && wdl.target.anchor == self.bound.current_rules.anchor
                    && raw.as_object().is_some_and(|v| {
                        v.len() == fields.len() && fields.iter().all(|k| v.contains_key(*k))
                    })
                    && raw["schema"] == R_READOUT_WDL_DOMAIN
                    && raw["kind"] == "rules_wdl"
                    && raw["native_input_sha256"] == self.input.native_input_sha256
                    && raw["semantic_input_sha256"] == self.input.semantic_input_sha256
                    && raw["anchor"]
                        == serde_json::to_value(&wdl.target.anchor)
                            .map_err(|e| ManifestError::Integrity(e.to_string()))?
                    && trajectory.game_id == self.lineage.game_id
                    && trajectory.anchors.contains(&wdl.target.anchor)
                    && trajectory.sha256()? == wdl.target.trajectory_membership_sha256
                    && trajectory.ending == wdl.target.ending
                    && trajectory.outcome == wdl.target.outcome
                    && trajectory.result_evidence_sha256 == wdl.target.rules_result_sha256
                    && raw["rules_receipt"].as_object().is_some_and(|v| {
                        v.len() == result_fields.len()
                            && result_fields.iter().all(|k| v.contains_key(*k))
                    })
                    && raw["rules_receipt"]["schema"] == "rz-pals-rules-outcome-evidence/1"
                    && raw["rules_receipt"]["game_id"] == self.lineage.game_id
                    && raw["rules_receipt"]["rules_terminal"] == true
                    && raw["rules_receipt"]["ending"]
                        == serde_json::to_value(trajectory.ending)
                            .map_err(|e| ManifestError::Integrity(e.to_string()))?
                    && raw["rules_receipt"]["outcome"]
                        == serde_json::to_value(trajectory.outcome)
                            .map_err(|e| ManifestError::Integrity(e.to_string()))?
                    && generation_digest(
                        "rz-pals-rules-outcome-evidence/1",
                        &raw["rules_receipt"],
                    )? == trajectory.result_evidence_sha256
                    && raw["rules_replay"]["schema"]
                        == "rz-rovent-actual-rules-trajectory-replay/1"
                    && generation_sha(&wdl.raw_receipt_sha256)
                    && generation_digest(R_READOUT_WDL_DOMAIN, raw)? == wdl.raw_receipt_sha256
                    && wdl.target.raw_evidence_sha256 == wdl.raw_receipt_sha256,
                "R WDL requires exact current Rules trajectory evidence",
            )?;
        }
        require(
            self.origin_sha256 == self.origin_digest()?
                && self.record_sha256 == self.record_digest()?,
            "R origin/record semantic seals",
        )?;
        require(
            serde_json::to_vec(self)
                .map_err(|e| ManifestError::Integrity(e.to_string()))?
                .len()
                <= MAX_R_READOUT_BYTES,
            "R row bytes",
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRReadoutManifest {
    pub schema: String,
    /// Original recipe/source/teacher declarations, not a P/C row conversion.
    pub collection: RoventCorpusManifest,
    pub pc_ancestor_index_sha256: Vec<String>,
    pub r_ancestor_index_sha256: Vec<String>,
}
impl RoventRReadoutManifest {
    pub fn validate(&self) -> Result<(), ManifestError> {
        self.collection.validate()?;
        require(
            self.schema == R_READOUT_MANIFEST_DOMAIN
                && ordered_shas(&self.pc_ancestor_index_sha256, 256)
                && ordered_shas(&self.r_ancestor_index_sha256, 256),
            "R manifest source/stage/ancestor domains",
        )
    }
    pub fn sha256(&self) -> Result<String, ManifestError> {
        self.validate()?;
        generation_digest(R_READOUT_MANIFEST_DOMAIN, self)
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRReadoutOrigin {
    pub source_domain: String,
    pub record_sha256: String,
    pub input_sha256: String,
    pub origin_sha256: String,
    pub split: GenerationSplitV3,
    pub lineage: RoventRLineage,
}
impl RoventRReadoutOrigin {
    pub fn from_record(record: &RoventRReadoutRecord) -> Self {
        Self {
            source_domain: R_READOUT_RECORD_DOMAIN.into(),
            record_sha256: record.record_sha256.clone(),
            input_sha256: record.input.semantic_input_sha256.clone(),
            origin_sha256: record.origin_sha256.clone(),
            split: record.split,
            lineage: record.lineage.clone(),
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRReadoutIndex {
    pub schema: String,
    pub manifest_sha256: String,
    pub record_count: u64,
    pub chunks: Vec<RoventCorpusChunk>,
    pub origins: Vec<RoventRReadoutOrigin>,
}
impl RoventRReadoutIndex {
    pub fn sha256(&self) -> Result<String, ManifestError> {
        self.validate()?;
        generation_digest(R_READOUT_INDEX_DOMAIN, self)
    }
    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            self.schema == R_READOUT_INDEX_DOMAIN
                && generation_sha(&self.manifest_sha256)
                && self.record_count == self.origins.len() as u64
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
            "R index counts/bounds",
        )?;
        let mut paths = BTreeSet::new();
        let mut occurrences = BTreeSet::new();
        for chunk in &self.chunks {
            require(
                !chunk.path.is_empty()
                    && chunk.path.len() <= 256
                    && !chunk.path.contains(['\\', ':'])
                    && chunk.path.split('/').all(|s| !matches!(s, "" | "." | ".."))
                    && chunk.bytes > 0
                    && chunk.bytes <= MAX_CHUNK_BYTES as u64
                    && chunk.records > 0
                    && generation_sha(&chunk.sha256)
                    && paths.insert(chunk.path.clone()),
                "R chunk relative path/pin",
            )?;
        }
        for origin in &self.origins {
            require(
                origin.source_domain == R_READOUT_RECORD_DOMAIN
                    && [
                        &origin.record_sha256,
                        &origin.input_sha256,
                        &origin.origin_sha256,
                        &origin.lineage.transposition_sha256,
                    ]
                    .iter()
                    .all(|s| generation_sha(s))
                    && occurrences.insert((
                        origin.source_domain.clone(),
                        origin.record_sha256.clone(),
                        origin.origin_sha256.clone(),
                    )),
                "R occurrence identity/seals",
            )?;
        }
        // Deliberately no uniqueness restriction on input_sha256: repeat
        // readouts remain distinct occurrences and form one leakage component.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn r_f32_frame_preserves_signed_zero_and_numeric_kind() {
        let hash = |v: &str| {
            r_readout_f32_digest(
                R_READOUT_FLOAT_BITS_DOMAIN,
                &serde_json::from_str(v).unwrap(),
            )
            .unwrap()
        };
        assert_ne!(hash("-0.0"), hash("0.0"));
        assert_ne!(hash("1"), hash("1.0"));
        assert_eq!(hash("1e-8"), hash("0.00000001"));
        assert_eq!(
            hash("{\"b\":1.0,\"a\":1e-8}"),
            hash("{\"a\":0.00000001,\"b\":1.0}")
        );
    }
}
