//! Actual generation V3, separate from the historical diagnostic data codecs.
//! Rules/native/CPU owners issue the evidence consumed here. A serialized
//! `source_kind` declaration alone is never collection authority.
use crate::{ManifestError, digest};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

mod targets;
pub use targets::*;
mod frozen_action;
pub use frozen_action::*;

pub const GENERATION_DATA_DOMAIN: &str = "rz-pals-generation-data/3";
pub const GENERATION_INPUT_DOMAIN: &str = "rz-pals-generation-input/3";
pub const GENERATION_VIEW_DOMAIN: &str = "rz-pals-selected-review-view/1";
pub const GENERATION_MAX_RECORDS: usize = 262_144;
pub const GENERATION_MAX_ROW_BYTES: usize = 1024 * 1024;
pub const GENERATION_TASKS: [&str; 7] = [
    "defend_response",
    "attack_repair",
    "widen_responses",
    "lower_selectivity",
    "resume_task",
    "cross_profile_recheck",
    "defer",
];
pub const GENERATION_BUCKETS: [u16; 6] = [8, 16, 32, 64, 128, 256];

pub(crate) fn ensure(ok: bool, message: &str) -> Result<(), ManifestError> {
    if ok {
        Ok(())
    } else {
        Err(ManifestError::Integrity(message.into()))
    }
}
pub fn generation_identifier(value: &str) -> bool {
    !value.is_empty() && value.len() <= 256 && !value.chars().any(char::is_control)
}
pub fn generation_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
/// Python's compact UTF-8 sorted [domain, value] representation. The model
/// payload uses actual f32 JSON values, never a platform-specific float hash.
pub fn generation_digest<T: Serialize>(domain: &str, value: &T) -> Result<String, ManifestError> {
    let mut json = serde_json::to_value((domain, value))
        .map_err(|e| ManifestError::Integrity(e.to_string()))?;
    json.sort_all_objects();
    Ok(digest(
        &serde_json::to_vec(&json).map_err(|e| ManifestError::Integrity(e.to_string()))?,
    ))
}
pub fn validate_generation_moves(moves: &[u16], maximum: usize) -> Result<(), ManifestError> {
    ensure(
        moves.len() <= maximum
            && moves
                .iter()
                .all(|m| m >> 12 <= 4 && (m & 63) != ((m >> 6) & 63)),
        "invalid bounded Move16 input",
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationRoleV3 {
    Proposer,
    Critic,
    Validator,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationSplitV3 {
    Train,
    Validation,
    Holdout,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationRulesAnchorV3 {
    pub rules_state_sha256: String,
    pub rules_history_sha256: String,
    pub board_fen: String,
    pub white_to_move: bool,
}
impl GenerationRulesAnchorV3 {
    pub fn validate(&self) -> Result<(), ManifestError> {
        let fen: Vec<_> = self.board_fen.split_whitespace().collect();
        ensure(
            generation_sha(&self.rules_state_sha256)
                && generation_sha(&self.rules_history_sha256)
                && self.board_fen.len() <= 512
                && !self.board_fen.chars().any(char::is_control)
                && fen.len() == 6
                && fen[1] == if self.white_to_move { "w" } else { "b" },
            "Rules anchor identity/FEN/viewpoint mismatch",
        )
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationRecordKindV3 {
    Proposal,
    Counter,
    Repair,
    Continuation,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationSelectedRecordV3 {
    pub record_id: String,
    pub kind: GenerationRecordKindV3,
    pub anchor: u8,
    pub moves: Vec<u16>,
    pub parent: Option<u16>,
    pub supersedes: Option<u16>,
    pub observed_sequence: u64,
    pub perspective: i8,
    pub completed_iteration: bool,
    pub rules_terminal: bool,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationSelectedQueryV3 {
    pub purpose: String,
    pub anchors: [u8; 3],
    pub prefix: Vec<u16>,
    pub proposal: Vec<u16>,
    pub counter: Vec<u16>,
    /// An explicit Some(empty) terminal witness differs from absent counter.
    pub counter_present: bool,
    pub white_to_move: bool,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationSelectedViewV3 {
    pub context_policy_sha256: String,
    pub anchors: Vec<GenerationRulesAnchorV3>,
    pub records: Vec<GenerationSelectedRecordV3>,
    pub query: GenerationSelectedQueryV3,
    pub token_budget: u16,
    pub length_bucket: u16,
}
impl GenerationSelectedViewV3 {
    pub fn validate(&self) -> Result<(), ManifestError> {
        ensure(
            generation_sha(&self.context_policy_sha256)
                && [128, 256, 512].contains(&self.token_budget)
                && !self.anchors.is_empty()
                && self.anchors.len() <= 4
                && self.records.len() <= 64,
            "selected review capacity/profile",
        )?;
        for anchor in &self.anchors {
            anchor.validate()?;
        }
        let mut ids = BTreeSet::new();
        for (slot, row) in self.records.iter().enumerate() {
            ensure(
                generation_identifier(&row.record_id)
                    && ids.insert(&row.record_id)
                    && usize::from(row.anchor) < self.anchors.len()
                    && [-1, 1].contains(&row.perspective),
                "selected record identity/anchor/perspective",
            )?;
            validate_generation_moves(&row.moves, 256)?;
            for relation in [row.parent, row.supersedes].into_iter().flatten() {
                ensure(
                    usize::from(relation) < self.records.len() && usize::from(relation) != slot,
                    "selected record absent/self relation",
                )?;
            }
        }
        fn visit(
            slot: usize,
            rows: &[GenerationSelectedRecordV3],
            states: &mut [u8],
        ) -> Result<(), ManifestError> {
            ensure(states[slot] != 1, "selected record relationship cycle")?;
            if states[slot] == 2 {
                return Ok(());
            }
            states[slot] = 1;
            for relation in [rows[slot].parent, rows[slot].supersedes]
                .into_iter()
                .flatten()
            {
                visit(relation as usize, rows, states)?;
            }
            states[slot] = 2;
            Ok(())
        }
        let mut states = vec![0; self.records.len()];
        for slot in 0..self.records.len() {
            visit(slot, &self.records, &mut states)?;
        }
        ensure(
            [
                "propose_policy",
                "reply_policy",
                "repair_policy",
                "divergence_policy",
                "value_fresh",
                "task_policy",
            ]
            .contains(&self.query.purpose.as_str())
                && self
                    .query
                    .anchors
                    .iter()
                    .all(|a| usize::from(*a) < self.anchors.len())
                && self.query.white_to_move
                    == (self.anchors[self.query.anchors[0] as usize].white_to_move
                        ^ (self.query.prefix.len() % 2 == 1))
                && (self.query.counter_present || self.query.counter.is_empty()),
            "selected query purpose/anchor/viewpoint/counter presence",
        )?;
        let mut count = 0usize;
        let mut longest = 0usize;
        for moves in self.records.iter().map(|r| &r.moves).chain([
            &self.query.prefix,
            &self.query.proposal,
            &self.query.counter,
        ]) {
            validate_generation_moves(moves, 256)?;
            count += moves.len();
            longest = longest.max(moves.len());
        }
        ensure(
            count <= usize::from(self.token_budget)
                && GENERATION_BUCKETS
                    .into_iter()
                    .find(|b| usize::from(*b) >= longest)
                    == Some(self.length_bucket),
            "selected view actual token/minimal length bucket mismatch",
        )
    }
    pub fn sha256(&self) -> Result<String, ManifestError> {
        self.validate()?;
        generation_digest(GENERATION_VIEW_DOMAIN, self)
    }
}

/// Only the exact selected immutable view lives in encoder input. Played game
/// moves and future target fields are outside this physical model payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationInputV3 {
    pub game_id: String,
    pub opening_id: String,
    pub line_genealogy_id: String,
    pub transposition_sha256: String,
    pub role: GenerationRoleV3,
    pub model_epoch: u64,
    pub context_epoch: u64,
    pub capture_sequence: u64,
    pub question_id: String,
    pub worker_id: String,
    pub job_id: String,
    pub slice_id: String,
    pub rules_anchor: GenerationRulesAnchorV3,
    pub actual_history: Vec<u16>,
    pub selected_view: GenerationSelectedViewV3,
    pub view_sha256: String,
    pub legal_moves: Vec<u16>,
    pub action_support_mask: [bool; 7],
    /// Strict V3 JSON ABI with array Move16 components, not Rust struct tokens.
    pub encoded_model_input: serde_json::Value,
    pub divergence_record_ids: Vec<String>,
}
impl GenerationInputV3 {
    pub fn validate(&self) -> Result<(), ManifestError> {
        self.validate_common_fields()?;
        let ids: BTreeSet<_> = self.divergence_record_ids.iter().collect();
        ensure(
            ids.len() == self.divergence_record_ids.len()
                && self.divergence_record_ids.len()
                    == self.encoded_model_input["divergences"]
                        .as_array()
                        .map_or(usize::MAX, Vec::len)
                && ids.into_iter().all(|id| {
                    self.selected_view
                        .records
                        .iter()
                        .any(|r| &r.record_id == id && r.kind == GenerationRecordKindV3::Counter)
                }),
            "submitted divergence candidate/order mismatch",
        )
    }
    /// 새 Rovent 입력도 사용하는 공통 계산이다. legacy V3의 record-index
    /// 검증이나 입력 권한을 발급하지 않으며 공개 validate는 이를 계속 강제한다.
    pub(crate) fn validate_common_fields(&self) -> Result<(), ManifestError> {
        for value in [
            &self.game_id,
            &self.opening_id,
            &self.line_genealogy_id,
            &self.question_id,
            &self.worker_id,
            &self.job_id,
            &self.slice_id,
        ] {
            ensure(generation_identifier(value), "generation input identifier")?;
        }
        ensure(
            generation_sha(&self.transposition_sha256),
            "generation transposition identity",
        )?;
        self.rules_anchor.validate()?;
        self.selected_view.validate()?;
        ensure(
            self.selected_view.query.white_to_move == self.rules_anchor.white_to_move
                && self.view_sha256 == self.selected_view.sha256()?,
            "generation input query viewpoint/view seal mismatch",
        )?;
        ensure(
            self.selected_view
                .records
                .iter()
                .all(|r| r.observed_sequence <= self.capture_sequence),
            "future record in submitted input",
        )?;
        validate_generation_moves(&self.actual_history, 16_384)?;
        validate_generation_moves(&self.legal_moves, 256)?;
        ensure(
            self.legal_moves.iter().collect::<BTreeSet<_>>().len() == self.legal_moves.len(),
            "duplicate legal generation candidates",
        )?;
        validate_encoded(self)
    }
    pub fn sha256(&self) -> Result<String, ManifestError> {
        self.validate()?;
        // JSON decimal formatting differs between serde_json and Python. Hash
        // actual model f32 bits while leaving the persisted numeric payload
        // untouched. This projection is specific to the new V3 input domain.
        let mut value =
            serde_json::to_value(self).map_err(|e| ManifestError::Integrity(e.to_string()))?;
        project_generation_input_float_bits(&mut value)?;
        generation_digest(GENERATION_INPUT_DOMAIN, &value)
    }
}

/// 검증된 입력의 기존 FP32 bit projection만 공유한다. JSON 파일 bytes/hash나
/// legacy domain을 변경하지 않으며 새 입력은 자기 domain으로 별도 봉인한다.
pub(crate) fn project_generation_input_float_bits(
    value: &mut serde_json::Value,
) -> Result<(), ManifestError> {
    let payload = &mut value["encoded_model_input"];
    for anchor in payload["anchors"]
        .as_array_mut()
        .ok_or_else(|| ManifestError::Integrity("encoded anchors array".into()))?
    {
        for field in anchor["metadata"].as_array_mut().unwrap() {
            *field = serde_json::Value::String(format!(
                "{:08x}",
                (field.as_f64().unwrap() as f32).to_bits()
            ));
        }
    }
    for row in payload["divergences"].as_array_mut().unwrap() {
        for field in row.as_array_mut().unwrap() {
            *field = serde_json::Value::String(format!(
                "{:08x}",
                (field.as_f64().unwrap() as f32).to_bits()
            ));
        }
    }
    Ok(())
}

fn validate_encoded(input: &GenerationInputV3) -> Result<(), ManifestError> {
    use serde_json::{Value, json};
    fn keys(value: &Value, required: &[&str]) -> bool {
        value.as_object().is_some_and(|o| {
            o.len() == required.len() && required.iter().all(|key| o.contains_key(*key))
        })
    }
    let payload = &input.encoded_model_input;
    ensure(
        keys(
            payload,
            &[
                "schema",
                "role",
                "anchors",
                "records",
                "query",
                "candidates",
                "divergences",
                "token_budget",
            ],
        ) && payload["schema"] == "rovezero.pals-virtual-review-input.v1"
            && payload["role"] == serde_json::to_value(input.role).unwrap()
            && payload["token_budget"] == input.selected_view.token_budget,
        "encoded V3 schema/private role/profile",
    )?;
    let anchors = payload["anchors"]
        .as_array()
        .ok_or_else(|| ManifestError::Integrity("encoded anchors array".into()))?;
    ensure(
        anchors.len() == input.selected_view.anchors.len(),
        "encoded anchor allocation",
    )?;
    for anchor in anchors {
        ensure(
            keys(anchor, &["board", "metadata"])
                && anchor["board"].as_array().is_some_and(|a| {
                    a.len() == 64 && a.iter().all(|v| v.as_u64().is_some_and(|x| x <= 12))
                })
                && anchor["metadata"].as_array().is_some_and(|a| {
                    a.len() == 16
                        && a.iter().all(|v| {
                            v.as_f64()
                                .is_some_and(|x| x.is_finite() && x.abs() <= f32::MAX as f64)
                        })
                }),
            "encoded Rules anchor finite board/metadata",
        )?;
    }
    let components = |m: &u16| json!([m & 63, (m >> 6) & 63, m >> 12]);
    let moves = |line: &[u16]| Value::Array(line.iter().map(components).collect());
    ensure(
        payload["candidates"] == moves(&input.legal_moves),
        "encoded legal candidate order",
    )?;
    let records = payload["records"]
        .as_array()
        .ok_or_else(|| ManifestError::Integrity("encoded records array".into()))?;
    ensure(
        records.len() == input.selected_view.records.len(),
        "encoded selected records allocation",
    )?;
    for (source, encoded) in input.selected_view.records.iter().zip(records) {
        let kind = match source.kind {
            GenerationRecordKindV3::Proposal => "proposal",
            GenerationRecordKindV3::Counter => "counterexample",
            GenerationRecordKindV3::Repair => "repair",
            GenerationRecordKindV3::Continuation => "cpu_verification",
        };
        ensure(
            *encoded
                == json!({"kind":kind,"perspective":source.perspective,"completed_iteration":source.completed_iteration,"rules_terminal":source.rules_terminal,"line":moves(&source.moves),"anchor":source.anchor,"parent":source.parent,"supersedes":source.supersedes}),
            "encoded record changed submitted view",
        )?;
    }
    let q = &input.selected_view.query;
    ensure(
        payload["query"]
            == json!({"purpose":q.purpose,"perspective":if q.white_to_move {1} else {-1},"anchors":q.anchors,"prefix":moves(&q.prefix),"proposal":moves(&q.proposal),"counter":moves(&q.counter),"counter_present":q.counter_present}),
        "encoded query changed submitted view",
    )?;
    ensure(
        payload["divergences"].as_array().is_some_and(|a| {
            a.len() <= 128
                && a.iter().all(|row| {
                    row.as_array().is_some_and(|r| {
                        r.len() == 8
                            && r.iter().all(|v| {
                                v.as_f64()
                                    .is_some_and(|x| x.is_finite() && x.abs() <= f32::MAX as f64)
                            })
                    })
                })
        }) && (input.role == GenerationRoleV3::Critic
            || payload["divergences"].as_array().unwrap().is_empty()),
        "encoded divergence finite/private allocation",
    )
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationRecordV3 {
    pub schema: String,
    pub source_kind: String,
    pub eligibility: String,
    pub input: GenerationInputV3,
    pub input_sha256: String,
    pub future_targets: GenerationTargetsV3,
}
impl GenerationRecordV3 {
    pub fn validate(&self) -> Result<(), ManifestError> {
        ensure(
            self.schema == GENERATION_DATA_DOMAIN
                && self.source_kind == "actual_selfplay"
                && self.eligibility == "actual_generation_candidate",
            "diagnostic data is not an actual generation candidate",
        )?;
        ensure(
            self.input_sha256 == self.input.sha256()?,
            "submitted generation input seal mismatch",
        )?;
        self.future_targets.validate(&self.input)?;
        ensure(
            serde_json::to_vec(self)
                .map_err(|e| ManifestError::Integrity(e.to_string()))?
                .len()
                <= GENERATION_MAX_ROW_BYTES,
            "generation row byte limit",
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationProducerAuthorityV3 {
    pub schema: String,
    pub source_kind: String,
    pub source_sha256: String,
    pub model_sha256: String,
    pub runtime_sha256: String,
    pub teacher_binary_sha256: String,
    pub teacher_profile_sha256: String,
    pub opening_split_sha256: String,
    pub registered_before_outcomes: bool,
    pub input_seals: Vec<String>,
    pub actual_game_ids: Vec<String>,
}
impl GenerationProducerAuthorityV3 {
    pub fn validate(
        &self,
        splits: &BTreeMap<String, GenerationSplitV3>,
    ) -> Result<(), ManifestError> {
        ensure(
            self.schema == "rz-pals-actual-collection-authority/1"
                && self.source_kind == "actual_selfplay"
                && self.registered_before_outcomes
                && [
                    &self.source_sha256,
                    &self.model_sha256,
                    &self.runtime_sha256,
                    &self.teacher_binary_sha256,
                    &self.teacher_profile_sha256,
                ]
                .into_iter()
                .all(|s| generation_sha(s))
                && self.opening_split_sha256
                    == generation_digest("rz-pals-opening-split/1", splits)?
                && self.input_seals.len() <= GENERATION_MAX_RECORDS
                && self.actual_game_ids.len() <= 8192
                && self.input_seals.iter().all(|s| generation_sha(s))
                && self
                    .actual_game_ids
                    .iter()
                    .all(|s| generation_identifier(s))
                && self.input_seals.iter().collect::<BTreeSet<_>>().len() == self.input_seals.len()
                && self.actual_game_ids.iter().collect::<BTreeSet<_>>().len()
                    == self.actual_game_ids.len(),
            "actual publication authority/split/capacity",
        )
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct GenerationLeakageReportV3 {
    pub cross_split_records_excluded: usize,
    pub cross_split_identities: usize,
}
/// All sides of known cross-split game/genealogy/state/transposition leakage are
/// excluded symmetrically. Opening assignments must exist before any outcome.
pub fn generation_split_records(
    records: &[GenerationRecordV3],
    splits: &BTreeMap<String, GenerationSplitV3>,
) -> Result<
    (
        BTreeMap<GenerationSplitV3, Vec<usize>>,
        GenerationLeakageReportV3,
    ),
    ManifestError,
> {
    ensure(
        records.len() <= GENERATION_MAX_RECORDS,
        "generation record capacity",
    )?;
    let mut domains: BTreeMap<(&str, &str), BTreeSet<GenerationSplitV3>> = BTreeMap::new();
    for record in records {
        record.validate()?;
        let input = &record.input;
        let split = *splits
            .get(&input.opening_id)
            .ok_or_else(|| ManifestError::Integrity("unregistered opening".into()))?;
        for key in [
            ("game", input.game_id.as_str()),
            ("genealogy", input.line_genealogy_id.as_str()),
            ("state", input.rules_anchor.rules_state_sha256.as_str()),
            ("transposition", input.transposition_sha256.as_str()),
        ] {
            domains.entry(key).or_default().insert(split);
        }
    }
    let leaked: BTreeSet<_> = domains
        .into_iter()
        .filter_map(|(key, s)| (s.len() > 1).then_some(key))
        .collect();
    let mut report = GenerationLeakageReportV3 {
        cross_split_records_excluded: 0,
        cross_split_identities: leaked.len(),
    };
    let mut result = BTreeMap::from([
        (GenerationSplitV3::Train, vec![]),
        (GenerationSplitV3::Validation, vec![]),
        (GenerationSplitV3::Holdout, vec![]),
    ]);
    for (index, record) in records.iter().enumerate() {
        let input = &record.input;
        if [
            ("game", input.game_id.as_str()),
            ("genealogy", input.line_genealogy_id.as_str()),
            ("state", input.rules_anchor.rules_state_sha256.as_str()),
            ("transposition", input.transposition_sha256.as_str()),
        ]
        .iter()
        .any(|key| leaked.contains(key))
        {
            report.cross_split_records_excluded += 1;
        } else {
            result
                .get_mut(&splits[&input.opening_id])
                .unwrap()
                .push(index);
        }
    }
    Ok((result, report))
}

#[cfg(test)]
mod tests;
