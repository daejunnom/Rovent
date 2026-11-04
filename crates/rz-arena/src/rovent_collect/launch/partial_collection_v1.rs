//! Accounting for an explicitly separate, incomplete collection observation.
//!
//! This type does not serialize a corpus, establish original row authority,
//! transfer ownership, or establish native/physical completion. File hashes,
//! sizes and record IDs are checked; the serializable failure projection cannot
//! reconstruct or replace the original sealed evidence owner.
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::{Dir, OpenOptions};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{self, BufRead, BufReader, Read, Write},
};

pub(super) const PARTIAL_COLLECTION_DOMAIN: &str = "rz-rovent-partial-collection/1";
pub const PARTIAL_COLLECTION_SPEC_DOMAIN: &str = "rz-rovent-partial-collection-spec/1";
const DIRECTORY: &str = "partial_collection";
const FILE: &str = "partial_collection.json";
const PENDING: &str = "partial_collection.pending.json";
const FAILURE_FILE: &str = "failure-evidence.jsonl";
const MAX_RECORDS: usize = 65_536;
const MAX_RECEIPT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ID_BYTES: usize = 256;
const MAX_REFERENCE_BYTES: usize = 4096;
pub(super) const MAX_FAILURE_ROW_BYTES: u64 = 8 * 1024 * 1024;
const FAILURE_DOMAIN: &str = "rz-rovent-partial-failure-evidence/1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PartialAccountingError {
    InvalidLimit,
    InvalidIdentifier,
    InvalidReference,
    DuplicateRow,
    UnknownRow,
    AlreadyResolved,
    RowLimit,
    ByteLimit,
    PendingRows,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum PartialEndReason {
    InputExhausted,
    Interrupted,
    CapacityExceeded,
    SinkFailure,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
enum RowState {
    Pending,
    Valid {
        record_reference: String,
        bytes: u64,
    },
    Failed {
        evidence_reference: String,
        bytes: u64,
        evidence_status: String,
        policy_target_known: bool,
        failure_classification: String,
    },
}

/// These fields are observations, not a publication or training receipt.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartialCollectionReceipt {
    schema: String,
    status: String,
    training_eligible: bool,
    publication_authority: bool,
    physical_completion: String,
    end_reason: PartialEndReason,
    valid_rows: usize,
    failed_rows: usize,
    pending_rows: usize,
    declared_payload_bytes: u64,
    row_limit: usize,
    byte_limit: u64,
    rows: BTreeMap<String, RowState>,
    chunks: Vec<PartialChunkPin>,
    failure_evidence: Option<PartialFailurePin>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartialCollectionSpec {
    pub schema: String,
    pub collection_plan_sha256: String,
    pub row_limit: usize,
    pub byte_limit: u64,
    pub receipt_byte_limit: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PartialChunkPin {
    pub(super) path: String,
    pub(super) bytes: u64,
    pub(super) sha256: String,
    pub(super) rows: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PartialFailurePin {
    pub(super) path: String,
    pub(super) bytes: u64,
    pub(super) sha256: String,
    pub(super) rows: u64,
}

fn digest_string(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}
fn invalid_partial(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

impl PartialCollectionSpec {
    pub fn from_json(input: &str, expected_plan_sha256: &str) -> io::Result<Self> {
        if input.len() > 64 * 1024 {
            return Err(invalid_partial("partial spec too large"));
        }
        let spec: Self =
            rz_experiments::decode_json(input).map_err(|e| io::Error::other(e.to_string()))?;
        spec.validate(expected_plan_sha256)?;
        Ok(spec)
    }
    pub fn validate(&self, expected_plan_sha256: &str) -> io::Result<()> {
        if self.schema != PARTIAL_COLLECTION_SPEC_DOMAIN
            || !digest_string(expected_plan_sha256)
            || self.collection_plan_sha256 != expected_plan_sha256
            || self.row_limit == 0
            || self.row_limit > MAX_RECORDS
            || self.byte_limit == 0
            || self.receipt_byte_limit == 0
            || self.receipt_byte_limit > MAX_RECEIPT_BYTES
        {
            return Err(invalid_partial("invalid partial collection spec"));
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PartialStoredReceipt {
    spec: PartialCollectionSpec,
    receipt: PartialCollectionReceipt,
}

/// A serialization of observable values, not the original authority object.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PartialFailureEvidenceRow {
    schema: String,
    row_id: String,
    original_row: serde_json::Value,
    error: String,
    teacher_evidence: serde_json::Value,
    observation_provenance: String,
    owner_reconstructable: bool,
    physical_completion: String,
}

fn exact_keys(v: &serde_json::Value, keys: &[&str]) -> bool {
    v.as_object()
        .is_some_and(|m| m.len() == keys.len() && keys.iter().all(|k| m.contains_key(*k)))
}
fn optional_u64(v: &serde_json::Value) -> bool {
    v.is_null() || v.as_u64().is_some()
}
fn optional_i64(v: &serde_json::Value) -> bool {
    v.is_null() || v.as_i64().is_some()
}
fn portable_duration(v: &serde_json::Value) -> bool {
    exact_keys(v, &["seconds", "nanoseconds"])
        && v["seconds"].as_u64().is_some()
        && v["nanoseconds"].as_u64().is_some_and(|n| n < 1_000_000_000)
}
fn portable_work(v: &serde_json::Value) -> bool {
    exact_keys(v, &["nodes", "qnodes", "tt_hits"])
        && ["nodes", "qnodes", "tt_hits"]
            .iter()
            .all(|k| optional_u64(&v[*k]))
}
fn portable_move(v: &serde_json::Value) -> bool {
    let square = |s: &serde_json::Value| {
        exact_keys(s, &["file", "rank"])
            && ["file", "rank"]
                .iter()
                .all(|k| s[*k].as_u64().is_some_and(|n| n < 8))
    };
    exact_keys(v, &["from", "to", "promotion"])
        && square(&v["from"])
        && square(&v["to"])
        && (v["promotion"].is_null()
            || matches!(
                v["promotion"].as_str(),
                Some("pawn" | "knight" | "bishop" | "rook" | "queen" | "king")
            ))
}
fn portable_report(v: &serde_json::Value) -> bool {
    let keys = [
        "identity",
        "observed_uci",
        "request_id",
        "best_move",
        "pv",
        "score",
        "bound",
        "wdl_per_mille",
        "perspective",
        "requested_depth",
        "reported_depth",
        "seldepth",
        "root_restricted",
        "completion",
        "work",
        "elapsed",
    ];
    if !exact_keys(v, &keys)
        || serde_json::from_value::<rz_search::cpu_checker::ExternalCheckerIdentity>(
            v["identity"].clone(),
        )
        .is_err()
        || !exact_keys(&v["observed_uci"], &["name", "author"])
        || v["observed_uci"]["name"].as_str().is_none()
        || !(v["observed_uci"]["author"].is_null() || v["observed_uci"]["author"].is_string())
        || v["request_id"].as_u64().is_none()
        || !(v["best_move"].is_null() || portable_move(&v["best_move"]))
        || !v["pv"]
            .as_array()
            .is_some_and(|a| a.iter().all(portable_move))
        || !matches!(
            v["bound"].as_str(),
            Some("exact_reported" | "lower" | "upper" | "unknown")
        )
        || !matches!(v["perspective"].as_str(), Some("white" | "black"))
        || v["requested_depth"].as_u64().is_none()
        || !optional_u64(&v["reported_depth"])
        || !optional_u64(&v["seldepth"])
        || v["root_restricted"].as_bool().is_none()
        || !matches!(
            v["completion"].as_str(),
            Some(
                "pending" | "best_move" | "stopped_deadline" | "stopped_canceled" | "stopped_yield"
            )
        )
        || !portable_work(&v["work"])
        || !portable_duration(&v["elapsed"])
    {
        return false;
    }
    let score = &v["score"];
    let score_ok = match score["kind"].as_str() {
        Some("unknown") => exact_keys(score, &["kind"]),
        Some("centipawns" | "mate_moves") => {
            exact_keys(score, &["kind", "value"]) && score["value"].as_i64().is_some()
        }
        _ => false,
    };
    let wdl_ok = v["wdl_per_mille"].is_null()
        || v["wdl_per_mille"].as_array().is_some_and(|a| {
            a.len() == 3
                && a.iter()
                    .all(|n| n.as_u64().is_some_and(|x| x <= u16::MAX as u64))
        });
    score_ok && wdl_ok
}
fn portable_attempt(v: &serde_json::Value) -> bool {
    if !exact_keys(v, &["work", "elapsed", "external"])
        || !portable_work(&v["work"])
        || !portable_duration(&v["elapsed"])
    {
        return false;
    }
    let external = &v["external"];
    if !exact_keys(external, &["request_id", "partial_report", "process"])
        || external["request_id"].as_u64().is_none()
        || !(external["partial_report"].is_null() || portable_report(&external["partial_report"]))
    {
        return false;
    }
    let p = &external["process"];
    let keys = [
        "process_identity",
        "stop_sent",
        "quit_sent",
        "exit_observed",
        "stdout_drained",
        "stderr_drained",
        "exit_code",
        "exit_signal",
        "cleanup_complete",
        "quarantined",
        "ownership_lost",
        "stdout_bytes",
        "stderr_bytes",
    ];
    let process = &p["process_identity"];
    exact_keys(p, &keys)
        && (process.is_null()
            || (exact_keys(process, &["pid", "process_group", "proc_start_ticks"])
                && process["pid"].as_u64().is_some()
                && process["process_group"].as_i64().is_some()
                && process["proc_start_ticks"].as_u64().is_some()))
        && [
            "stop_sent",
            "quit_sent",
            "exit_observed",
            "stdout_drained",
            "stderr_drained",
            "cleanup_complete",
            "quarantined",
            "ownership_lost",
        ]
        .iter()
        .all(|k| p[*k].as_bool().is_some())
        && optional_i64(&p["exit_code"])
        && optional_i64(&p["exit_signal"])
        && p["stdout_bytes"].as_u64().is_some()
        && p["stderr_bytes"].as_u64().is_some()
}

fn portable_teacher_fields(teacher: &serde_json::Map<String, serde_json::Value>) -> bool {
    let Some(dispatch) = teacher.get("dispatch") else {
        return false;
    };
    let Some(completion) = teacher.get("completion") else {
        return false;
    };
    let Some(scope) = teacher.get("input_scope") else {
        return false;
    };
    let Some(startup) = teacher.get("startup") else {
        return false;
    };
    let Some(profiled) = teacher.get("profiled_report") else {
        return false;
    };
    let process = |v: &serde_json::Value| {
        exact_keys(v, &["pid", "process_group", "proc_start_ticks"])
            && ["pid", "process_group", "proc_start_ticks"]
                .iter()
                .all(|k| v[*k].as_u64().is_some())
    };
    let nonempty = |v: &serde_json::Value| v.as_str().is_some_and(|s| !s.is_empty());
    let seq_keys = [
        "cancel_sequence",
        "yield_sequence",
        "publish_sequence",
        "cancel_ack",
        "yield_ack",
        "publish_ack",
    ];
    let controls = &completion["controls"];
    let control_keys = [
        "cancel_sequence",
        "yield_sequence",
        "publish_sequence",
        "cancel_ack",
        "yield_ack",
        "publish_ack",
        "terminal",
    ];
    let deadline = &dispatch["original_deadline"];
    let authority = &scope["authority"];
    let scope_digests = [
        "input_sha256",
        "view_sha256",
        "source_sha256",
        "rules_semantic",
        "input_lock",
        "root_admission",
        "root_budget",
        "frozen_actor",
    ];
    let scope_keys = [
        "authority",
        "input_sha256",
        "view_sha256",
        "source_sha256",
        "role",
        "rules_semantic",
        "input_lock",
        "root_admission",
        "root_budget",
        "frozen_actor",
    ];
    let opinion_ok = |v: &serde_json::Value| {
        let score = &v["score"];
        let score_ok = match score["kind"].as_str() {
            Some("unknown") => exact_keys(score, &["kind"]),
            Some("centipawns" | "mate_moves") => {
                exact_keys(score, &["kind", "value"]) && score["value"].as_i64().is_some()
            }
            _ => false,
        };
        let wdl = &v["reported_wdl"];
        exact_keys(
            v,
            &[
                "rank",
                "depth",
                "seldepth",
                "score",
                "bound",
                "reported_wdl",
                "pv",
            ],
        ) && v["rank"].as_u64().is_some()
            && v["depth"].as_u64().is_some()
            && optional_u64(&v["seldepth"])
            && score_ok
            && matches!(
                v["bound"].as_str(),
                Some("exact_reported" | "lower" | "upper" | "unknown")
            )
            && (wdl.is_null()
                || wdl
                    .as_array()
                    .is_some_and(|a| a.len() == 3 && a.iter().all(|n| n.as_u64().is_some())))
            && v["pv"].as_array().is_some_and(|a| a.iter().all(nonempty))
    };
    exact_keys(
        dispatch,
        &[
            "collection_owner",
            "sequence",
            "process",
            "constraint",
            "original_deadline_debug",
            "original_debug",
            "original_deadline",
            "soft_quantum_ns",
            "profile",
            "evaluation_namespace_sha256",
            "root_restriction_sha256",
            "original_root_moves",
        ],
    ) && dispatch["collection_owner"].as_u64().is_some()
        && dispatch["sequence"].as_u64().is_some()
        && process(&dispatch["process"])
        && serde_json::from_value::<rz_search::external_cpu::ExternalGoConstraint>(
            dispatch["constraint"].clone(),
        )
        .is_ok()
        && nonempty(&dispatch["original_deadline_debug"])
        && nonempty(&dispatch["original_debug"])
        && exact_keys(deadline, &["clock_epoch", "at_ns"])
        && deadline["clock_epoch"].as_u64().is_some()
        && deadline["at_ns"].as_u64().is_some()
        && dispatch["soft_quantum_ns"].as_u64().is_some()
        && serde_json::from_value::<rz_search::external_cpu::ExternalEngineProfile>(
            dispatch["profile"].clone(),
        )
        .is_ok()
        && ["evaluation_namespace_sha256", "root_restriction_sha256"]
            .iter()
            .all(|k| dispatch[*k].as_str().is_some_and(digest_string))
        && (dispatch["original_root_moves"].is_null()
            || dispatch["original_root_moves"]
                .as_array()
                .is_some_and(|a| a.iter().all(nonempty)))
        && exact_keys(
            completion,
            &[
                "collection_owner",
                "sequence",
                "matches_dispatch_identity",
                "process",
                "go_issue",
                "issued_go",
                "issued_constraint",
                "controls",
                "timing_debug",
                "original_debug",
            ],
        )
        && completion["collection_owner"].as_u64().is_some()
        && completion["sequence"].as_u64().is_some()
        && completion["matches_dispatch_identity"].as_bool().is_some()
        && (completion["process"].is_null() || process(&completion["process"]))
        && matches!(
            completion["go_issue"].as_str(),
            Some("not_issued" | "unknown" | "issued")
        )
        && (completion["issued_go"].is_null() || completion["issued_go"].is_string())
        && (completion["issued_constraint"].is_null()
            || serde_json::from_value::<rz_search::external_cpu::ExternalGoConstraint>(
                completion["issued_constraint"].clone(),
            )
            .is_ok())
        && exact_keys(controls, &control_keys)
        && seq_keys.iter().all(|k| controls[*k].as_u64().is_some())
        && controls["terminal"].as_bool().is_some()
        && nonempty(&completion["timing_debug"])
        && nonempty(&completion["original_debug"])
        && exact_keys(scope, &scope_keys)
        && scope_digests
            .iter()
            .all(|k| scope[*k].as_str().is_some_and(digest_string))
        && nonempty(&scope["role"])
        && exact_keys(
            authority,
            &[
                "process_epoch",
                "game_generation",
                "root_generation",
                "implementation_sha256",
            ],
        )
        && ["process_epoch", "game_generation", "root_generation"]
            .iter()
            .all(|k| authority[*k].as_u64().is_some())
        && authority["implementation_sha256"]
            .as_str()
            .is_some_and(digest_string)
        && exact_keys(
            startup,
            &[
                "actor_id",
                "pin",
                "canonical_sha256",
                "capability_sha256",
                "input_lock_sha256",
                "observed_uci",
                "process_identity",
            ],
        )
        && nonempty(&startup["actor_id"])
        && serde_json::from_value::<rz_experiments::rovent_corpus::RoventTeacherPin>(
            startup["pin"].clone(),
        )
        .is_ok()
        && ["canonical_sha256", "capability_sha256", "input_lock_sha256"]
            .iter()
            .all(|k| startup[*k].as_str().is_some_and(digest_string))
        && exact_keys(&startup["observed_uci"], &["name", "author"])
        && nonempty(&startup["observed_uci"]["name"])
        && (startup["observed_uci"]["author"].is_null()
            || startup["observed_uci"]["author"].is_string())
        && process(&startup["process_identity"])
        && ["reported_node_overshoot", "readiness_barrier_completed"]
            .iter()
            .all(|k| profiled[*k].as_bool().is_some())
        && optional_u64(&profiled["strict_native_nodes"])
        && (profiled["option_application_observed"].is_null()
            || profiled["option_application_observed"].is_boolean())
        && serde_json::from_value::<rz_search::external_cpu::ExternalEngineProfile>(
            profiled["profile"].clone(),
        )
        .is_ok()
        && serde_json::from_value::<rz_search::external_cpu::ExternalEnginePurpose>(
            profiled["purpose"].clone(),
        )
        .is_ok()
        && ["requested", "issued"].iter().all(|k| {
            serde_json::from_value::<rz_search::external_cpu::ExternalGoConstraint>(
                profiled[*k].clone(),
            )
            .is_ok()
        })
        && nonempty(&profiled["issued_go"])
        && teacher
            .get("opinions")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|a| a.iter().all(opinion_ok))
        && teacher.get("stop_control").is_some_and(|v| {
            v.is_null()
                || (exact_keys(v, &["sequence", "action"])
                    && v["sequence"].as_u64().is_some()
                    && matches!(
                        v["action"].as_str(),
                        Some("continue" | "yield" | "cancel" | "revoked")
                    ))
        })
}

fn verify_failure_evidence(
    directory: &Dir,
    spec: &PartialCollectionSpec,
    receipt: &PartialCollectionReceipt,
) -> io::Result<()> {
    let Some(pin) = &receipt.failure_evidence else {
        return if receipt.failed_rows == 0 {
            Ok(())
        } else {
            Err(invalid_partial("partial failure evidence missing"))
        };
    };
    let total_chunks = receipt
        .chunks
        .iter()
        .try_fold(0u64, |n, chunk| n.checked_add(chunk.bytes))
        .ok_or_else(|| invalid_partial("partial evidence budget overflow"))?;
    if total_chunks
        .checked_add(pin.bytes)
        .is_none_or(|n| n > spec.byte_limit)
    {
        return Err(invalid_partial("partial evidence budget"));
    }
    let mut options = OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    let file = directory.open_with(FAILURE_FILE, &options)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() != pin.bytes {
        return Err(invalid_partial("partial failure evidence file/size"));
    }
    let mut reader = BufReader::new(file.take(pin.bytes.saturating_add(1)));
    let mut hash = Sha256::new();
    let mut seen = BTreeSet::new();
    let mut count = 0u64;
    let mut size = 0u64;
    loop {
        let mut line = Vec::new();
        let n = (&mut reader)
            .take(MAX_FAILURE_ROW_BYTES + 1)
            .read_until(b'\n', &mut line)?;
        if n == 0 {
            break;
        }
        if line.last() != Some(&b'\n') || n as u64 > MAX_FAILURE_ROW_BYTES + 1 {
            return Err(invalid_partial(
                "partial failure evidence row bound/newline",
            ));
        }
        count += 1;
        size = size
            .checked_add(n as u64)
            .ok_or_else(|| invalid_partial("partial failure evidence size overflow"))?;
        if count > pin.rows {
            return Err(invalid_partial("partial failure evidence rows"));
        }
        hash.update(&line);
        let text = std::str::from_utf8(&line[..n - 1])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        let row: PartialFailureEvidenceRow = rz_experiments::decode_json(text)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        let teacher = row
            .teacher_evidence
            .as_object()
            .ok_or_else(|| invalid_partial("partial teacher evidence object"))?;
        let exclusions = serde_json::json!([
            "owner_capabilities",
            "absolute_monotonic_instants",
            "unretained_stdout_stderr",
            "opaque_rules_history_snapshot"
        ]);
        let complete_teacher = ["dispatch", "completion", "input_scope", "startup"]
            .iter()
            .all(|key| teacher.get(*key).is_some_and(serde_json::Value::is_object))
            && teacher.get("raw_report").is_some_and(portable_report)
            && teacher.get("attempt").is_some_and(portable_attempt)
            && teacher.get("profiled_report").is_some_and(|v| {
                exact_keys(
                    v,
                    &[
                        "profile",
                        "purpose",
                        "requested",
                        "issued",
                        "issued_go",
                        "reported_node_overshoot",
                        "strict_native_nodes",
                        "option_application_observed",
                        "readiness_barrier_completed",
                    ],
                )
            })
            && teacher
                .get("requested_multipv")
                .and_then(serde_json::Value::as_u64)
                .is_some()
            && teacher.get("stop_control").is_some()
            && teacher
                .get("opinions")
                .is_some_and(serde_json::Value::is_array)
            && teacher.get("schema").and_then(serde_json::Value::as_str)
                == Some("portable_teacher_observation_v1")
            && teacher.get("opaque_exclusions") == Some(&exclusions);
        let original_json = serde_json::to_string(&row.original_row)
            .map_err(|e| io::Error::other(e.to_string()))?;
        let original =
            rz_experiments::rovent_corpus_lock_v2::record::RoventTrainingRecordLockV2::from_json(
                &original_json,
            )
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        let sequence = Some(original.0.native.retired_sequence);
        if row.schema != FAILURE_DOMAIN
            || row.error.is_empty()
            || !row.original_row.is_object()
            || row.observation_provenance != "retained_original_sealed_serializable_projection"
            || row.owner_reconstructable
            || row.physical_completion != "not_attested"
            || !complete_teacher
            || !portable_teacher_fields(teacher)
            || teacher.get("conditional_policy_target_known")
                != Some(&serde_json::Value::Bool(false))
            || teacher.get("portable_observation_complete") != Some(&serde_json::Value::Bool(true))
            || teacher.get("truncated") != Some(&serde_json::Value::Bool(false))
            || sequence.map(|n| format!("failure-sequence-{n}")) != Some(row.row_id.clone())
            || !seen.insert(row.row_id.clone())
        {
            return Err(invalid_partial("partial failure evidence shape or scope"));
        }
        match receipt.rows.get(&row.row_id) {
            Some(RowState::Failed {
                evidence_reference,
                bytes,
                ..
            }) if evidence_reference == &format!("{FAILURE_FILE}#{}", row.row_id)
                && *bytes == n as u64 => {}
            _ => return Err(invalid_partial("partial failure evidence identity")),
        }
    }
    if count != pin.rows || size != pin.bytes || format!("{:x}", hash.finalize()) != pin.sha256 {
        return Err(invalid_partial("partial failure evidence checksum/count"));
    }
    Ok(())
}

/// Create a fresh isolated output child. Existing directories are not reused.
pub(super) fn create_partial_directory(parent: &Dir) -> io::Result<Dir> {
    parent.create_dir(DIRECTORY)?;
    parent.open_dir_nofollow(DIRECTORY)
}

fn scan_chunks(
    directory: &Dir,
    spec: &PartialCollectionSpec,
    chunks: &[PartialChunkPin],
    mut visit: impl FnMut(&str, &str, u64) -> io::Result<()>,
) -> io::Result<u64> {
    let mut seen = BTreeSet::new();
    for pin in chunks {
        if !seen.insert(pin.path.as_str()) {
            return Err(invalid_partial("duplicate partial chunk path"));
        }
    }
    seen.clear();
    let mut seen_rows = BTreeSet::new();
    let mut found = 0u64;
    let (mut total, mut rows) = (0u64, 0u64);
    let chunk_dir = if chunks.is_empty() {
        None
    } else {
        Some(directory.open_dir_nofollow("chunks")?)
    };
    for pin in chunks {
        let Some(name) = pin.path.strip_prefix("chunks/") else {
            return Err(invalid_partial("partial chunk path"));
        };
        if name.len() != 12
            || !name.ends_with(".jsonl")
            || !name.as_bytes()[..6].iter().all(u8::is_ascii_digit)
            || !seen.insert(pin.path.as_str())
            || !digest_string(&pin.sha256)
            || pin.bytes == 0
            || pin.rows == 0
            || pin.bytes > rz_experiments::rovent_corpus::MAX_CHUNK_BYTES as u64
        {
            return Err(invalid_partial("partial chunk pin"));
        }
        total = total
            .checked_add(pin.bytes)
            .ok_or_else(|| invalid_partial("partial chunk size overflow"))?;
        rows = rows
            .checked_add(pin.rows)
            .ok_or_else(|| invalid_partial("partial chunk row overflow"))?;
        if total > spec.byte_limit {
            return Err(invalid_partial("partial chunk byte limit"));
        }
        let mut options = OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No);
        let file = chunk_dir
            .as_ref()
            .expect("nonempty chunks")
            .open_with(name, &options)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() || metadata.len() != pin.bytes {
            return Err(invalid_partial("partial chunk file/size"));
        }
        let mut bounded = BufReader::new(file.take(pin.bytes.saturating_add(1)));
        let mut hasher = Sha256::new();
        let (mut observed, mut lines) = (0u64, 0u64);
        loop {
            let mut buffer = Vec::new();
            let n = (&mut bounded)
                .take((rz_experiments::pals_data_v3::GENERATION_MAX_ROW_BYTES as u64) + 1)
                .read_until(b'\n', &mut buffer)?;
            if n == 0 {
                break;
            }
            if buffer.last() != Some(&b'\n') {
                return Err(invalid_partial("partial chunk row bound or newline"));
            }
            observed += n as u64;
            hasher.update(&buffer);
            lines += 1;
            if found >= spec.row_limit as u64 {
                return Err(invalid_partial("partial chunk row limit"));
            }
            let text = std::str::from_utf8(&buffer[..n - 1])
                .map_err(|e| io::Error::other(e.to_string()))?;
            let row = rz_experiments::rovent_corpus_lock_v2::record::RoventTrainingRecordLockV2::from_json(text)
                .map_err(|e| io::Error::other(e.to_string()))?;
            let id = row.0.record_sha256.clone();
            if row.sha256().map_err(|e| io::Error::other(e.to_string()))? != id {
                return Err(invalid_partial("partial row seal differs"));
            }
            if !seen_rows.insert(id.clone()) {
                return Err(invalid_partial("duplicate partial row"));
            }
            visit(&id, &pin.path, n as u64)?;
            found += 1;
        }
        if observed != pin.bytes
            || lines != pin.rows
            || format!("{:x}", hasher.finalize()) != pin.sha256
        {
            return Err(invalid_partial("partial chunk contents differ"));
        }
    }
    if rows != found {
        return Err(invalid_partial("partial valid row/chunk count differs"));
    }
    Ok(found)
}

fn verify_chunks(
    directory: &Dir,
    spec: &PartialCollectionSpec,
    receipt: &PartialCollectionReceipt,
) -> io::Result<()> {
    let rows = scan_chunks(
        directory,
        spec,
        &receipt.chunks,
        |id, path, size| match receipt.rows.get(id) {
            Some(RowState::Valid {
                record_reference,
                bytes,
            }) if record_reference == path && *bytes == size => Ok(()),
            _ => Err(invalid_partial(
                "partial valid row identity/path/size differs",
            )),
        },
    )?;
    if rows != receipt.valid_rows as u64 {
        return Err(invalid_partial("partial valid row count"));
    }
    Ok(())
}

/// Account rows only after checking actual flushed bytes and seals. The caller
/// remains responsible for original producer authority and physical custody.
pub(super) fn account_flushed_chunks(
    directory: &Dir,
    spec: &PartialCollectionSpec,
    accounting: &mut PartialCollectionAccounting,
    chunks: &[PartialChunkPin],
) -> io::Result<()> {
    spec.validate(&spec.collection_plan_sha256)?;
    if accounting.row_limit != spec.row_limit || accounting.byte_limit != spec.byte_limit {
        return Err(invalid_partial("partial accounting spec differs"));
    }
    let mut next = accounting.clone();
    scan_chunks(directory, spec, chunks, |id, path, size| {
        next.begin_row(id)
            .map_err(|e| io::Error::other(format!("partial row admission: {e:?}")))?;
        next.record_valid(id, path, size)
            .map_err(|e| io::Error::other(format!("partial row accounting: {e:?}")))?;
        Ok(())
    })?;
    *accounting = next;
    Ok(())
}

struct LimitedBuffer {
    bytes: Vec<u8>,
    limit: usize,
}
impl Write for LimitedBuffer {
    fn write(&mut self, input: &[u8]) -> io::Result<usize> {
        if self
            .bytes
            .len()
            .checked_add(input.len())
            .is_none_or(|n| n > self.limit)
        {
            return Err(io::Error::other("partial receipt byte limit"));
        }
        self.bytes
            .try_reserve(input.len())
            .map_err(|_| io::Error::other("partial receipt allocation"))?;
        self.bytes.extend_from_slice(input);
        Ok(input.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Writes only the separate partial receipt; caller retains all native owners.
/// Existing paths are never replaced; a failed attempt leaves its pending file.
pub(super) fn write_partial_receipt(
    directory: &Dir,
    spec: &PartialCollectionSpec,
    receipt: &PartialCollectionReceipt,
) -> io::Result<()> {
    spec.validate(&spec.collection_plan_sha256)?;
    receipt.validate(spec)?;
    verify_chunks(directory, spec, receipt)?;
    verify_failure_evidence(directory, spec, receipt)?;
    let stored = PartialStoredReceipt {
        spec: spec.clone(),
        receipt: receipt.clone(),
    };
    let mut output = LimitedBuffer {
        bytes: Vec::new(),
        limit: spec.receipt_byte_limit as usize,
    };
    serde_json::to_writer(&mut output, &stored).map_err(|e| io::Error::other(e.to_string()))?;
    let mut options = OpenOptions::new();
    options
        .write(true)
        .create_new(true)
        .follow(FollowSymlinks::No);
    let mut file = directory.open_with(PENDING, &options)?;
    file.write_all(&output.bytes)?;
    file.sync_all()?;
    let root = crate::pals_collect_v3::training_process::directory_file(directory)
        .map_err(|e| io::Error::other(e.to_string()))?;
    root.sync_all()?;
    directory.hard_link(PENDING, directory, FILE)?;
    root.sync_all()?;
    directory.remove_file(PENDING)?;
    root.sync_all()?;
    Ok(())
}

pub fn read_partial_receipt(
    parent: &Dir,
    spec: &PartialCollectionSpec,
) -> io::Result<PartialCollectionReceipt> {
    let directory = parent.open_dir_nofollow(DIRECTORY)?;
    read_partial_receipt_in(&directory, spec)
}

fn read_partial_receipt_in(
    directory: &Dir,
    spec: &PartialCollectionSpec,
) -> io::Result<PartialCollectionReceipt> {
    let receipt = read_partial_receipt_metadata_in(directory, spec)?;
    verify_chunks(directory, spec, &receipt)?;
    verify_failure_evidence(directory, spec, &receipt)?;
    Ok(receipt)
}

fn read_partial_receipt_metadata_in(
    directory: &Dir,
    spec: &PartialCollectionSpec,
) -> io::Result<PartialCollectionReceipt> {
    spec.validate(&spec.collection_plan_sha256)?;
    let mut options = OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    let file = directory.open_with(FILE, &options)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > spec.receipt_byte_limit {
        return Err(invalid_partial("partial receipt file/size"));
    }
    let mut raw = Vec::new();
    file.take(spec.receipt_byte_limit + 1)
        .read_to_end(&mut raw)?;
    if raw.len() as u64 > spec.receipt_byte_limit {
        return Err(invalid_partial("partial receipt grew past limit"));
    }
    let text = std::str::from_utf8(&raw).map_err(|e| io::Error::other(e.to_string()))?;
    let stored: PartialStoredReceipt =
        rz_experiments::decode_json(text).map_err(|e| io::Error::other(e.to_string()))?;
    if stored.spec != *spec {
        return Err(invalid_partial("partial receipt spec differs"));
    }
    stored.receipt.validate(spec)?;
    Ok(stored.receipt)
}

/// Visit one pinned chunk after complete receipt and chunk verification.
/// Memory is limited to one 32MiB chunk and one parsed row. A callback error
/// may leave prior callback effects; it does not roll them back. This grants no
/// original authority or training eligibility.
/// A reader pins one validated receipt and its original directory capability.
/// Each visit independently verifies the bytes it will pass to the callback.
pub struct PartialCollectionReader {
    directory: Dir,
    spec: PartialCollectionSpec,
    receipt: PartialCollectionReceipt,
}

impl PartialCollectionReader {
    pub fn open(parent: &Dir, spec: &PartialCollectionSpec) -> io::Result<Self> {
        let directory = parent.open_dir_nofollow(DIRECTORY)?;
        let receipt = read_partial_receipt_in(&directory, spec)?;
        Ok(Self {
            directory,
            spec: spec.clone(),
            receipt,
        })
    }

    pub fn chunk_count(&self) -> usize {
        self.receipt.chunks.len()
    }

    pub fn receipt(&self) -> &PartialCollectionReceipt {
        &self.receipt
    }

    /// Rechecks the current file against the pinned receipt and the original
    /// spec. This observation has a time boundary: a later mutation can still
    /// occur, and neither this nor a clean process exit establishes a fence.
    pub fn verify_failure_evidence_current(&self) -> io::Result<bool> {
        verify_failure_evidence(&self.directory, &self.spec, &self.receipt)?;
        Ok(self.receipt.failure_evidence.is_some())
    }

    pub fn visit_chunk(
        &self,
        chunk_index: usize,
        visit: impl FnMut(
            &rz_experiments::rovent_corpus_lock_v2::record::RoventTrainingRecordLockV2,
        ) -> io::Result<()>,
    ) -> io::Result<usize> {
        visit_partial_chunk_in(&self.directory, &self.receipt, chunk_index, visit)
    }
}

fn visit_partial_chunk_in(
    directory: &Dir,
    receipt: &PartialCollectionReceipt,
    chunk_index: usize,
    mut visit: impl FnMut(
        &rz_experiments::rovent_corpus_lock_v2::record::RoventTrainingRecordLockV2,
    ) -> io::Result<()>,
) -> io::Result<usize> {
    let pin = receipt
        .chunks
        .get(chunk_index)
        .ok_or_else(|| invalid_partial("partial chunk index"))?;
    let name = pin
        .path
        .strip_prefix("chunks/")
        .ok_or_else(|| invalid_partial("partial chunk path"))?;
    if name.len() != 12
        || !name.ends_with(".jsonl")
        || !name.as_bytes()[..6].iter().all(u8::is_ascii_digit)
        || pin.bytes == 0
        || pin.rows == 0
        || !digest_string(&pin.sha256)
    {
        return Err(invalid_partial("partial chunk pin"));
    }
    let chunk_dir = directory.open_dir_nofollow("chunks")?;
    let mut options = OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    let file = chunk_dir.open_with(name, &options)?;
    let metadata = file.metadata()?;
    if !metadata.is_file()
        || metadata.len() != pin.bytes
        || pin.bytes > rz_experiments::rovent_corpus::MAX_CHUNK_BYTES as u64
    {
        return Err(invalid_partial("partial chunk file/size"));
    }
    let mut raw = Vec::new();
    file.take(pin.bytes + 1).read_to_end(&mut raw)?;
    if raw.len() as u64 != pin.bytes
        || format!("{:x}", Sha256::digest(&raw)) != pin.sha256
        || raw.last() != Some(&b'\n')
    {
        return Err(invalid_partial("partial chunk bytes differ"));
    }
    let mut count = 0usize;
    let mut seen_ids = BTreeSet::new();
    // Preflight all rows before callbacks. Both passes use the same buffer.
    for line in raw.split_inclusive(|b| *b == b'\n') {
        if line.len() > rz_experiments::pals_data_v3::GENERATION_MAX_ROW_BYTES + 1
            || line.last() != Some(&b'\n')
        {
            return Err(invalid_partial("partial row bound or newline"));
        }
        let text = std::str::from_utf8(&line[..line.len() - 1])
            .map_err(|e| io::Error::other(e.to_string()))?;
        let row =
            rz_experiments::rovent_corpus_lock_v2::record::RoventTrainingRecordLockV2::from_json(
                text,
            )
            .map_err(|e| io::Error::other(e.to_string()))?;
        let id = &row.0.record_sha256;
        if row.sha256().map_err(|e| io::Error::other(e.to_string()))? != *id {
            return Err(invalid_partial("partial row seal differs"));
        }
        if !seen_ids.insert(id.clone()) {
            return Err(invalid_partial("duplicate partial row"));
        }
        match receipt.rows.get(id) {
            Some(RowState::Valid {
                record_reference,
                bytes,
            }) if record_reference == &pin.path && *bytes == line.len() as u64 => {}
            _ => {
                return Err(invalid_partial(
                    "partial inspection row identity/path/size differs",
                ));
            }
        }
        count += 1;
    }
    let expected = receipt.rows.values().filter(|row| matches!(row, RowState::Valid { record_reference, .. } if record_reference == &pin.path)).count();
    if count as u64 != pin.rows || count != expected {
        return Err(invalid_partial("partial chunk row count"));
    }
    for line in raw.split_inclusive(|b| *b == b'\n') {
        let text = std::str::from_utf8(&line[..line.len() - 1])
            .map_err(|e| io::Error::other(e.to_string()))?;
        let row =
            rz_experiments::rovent_corpus_lock_v2::record::RoventTrainingRecordLockV2::from_json(
                text,
            )
            .map_err(|e| io::Error::other(e.to_string()))?;
        visit(&row)?;
    }
    Ok(count)
}

/// Status of the original failure-evidence file at the instant observed.
/// An unavailable file remains a failure; it is not converted to success.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InspectionFailureEvidenceStatus {
    NotApplicable,
    VerifiedAtObservation,
    Missing { detail: String },
    Invalid { detail: String },
    IoError { detail: String },
}

fn inspection_failure_status(
    directory: &Dir,
    spec: &PartialCollectionSpec,
    receipt: &PartialCollectionReceipt,
) -> InspectionFailureEvidenceStatus {
    if receipt.failure_evidence.is_none() {
        return InspectionFailureEvidenceStatus::NotApplicable;
    }
    match verify_failure_evidence(directory, spec, receipt) {
        Ok(()) => InspectionFailureEvidenceStatus::VerifiedAtObservation,
        Err(error) => {
            let detail: String = error.to_string().chars().take(1024).collect();
            match error.kind() {
                io::ErrorKind::NotFound => InspectionFailureEvidenceStatus::Missing { detail },
                io::ErrorKind::InvalidData | io::ErrorKind::UnexpectedEof => {
                    InspectionFailureEvidenceStatus::Invalid { detail }
                }
                _ => InspectionFailureEvidenceStatus::IoError { detail },
            }
        }
    }
}

/// Metadata and selected chunks are inspection-only, with no training,
/// original authority or physical-completion capability. Unlike the strict
/// reader, an unavailable failure file is reported without blocking access to
/// an independently verified valid chunk.
pub struct PartialInspectionReader {
    directory: Dir,
    spec: PartialCollectionSpec,
    receipt: PartialCollectionReceipt,
    failure_status_at_open: InspectionFailureEvidenceStatus,
}

impl PartialInspectionReader {
    pub fn open(parent: &Dir, spec: &PartialCollectionSpec) -> io::Result<Self> {
        let directory = parent.open_dir_nofollow(DIRECTORY)?;
        let receipt = read_partial_receipt_metadata_in(&directory, spec)?;
        validate_inspection_chunk_metadata(spec, &receipt)?;
        let failure_status_at_open = inspection_failure_status(&directory, spec, &receipt);
        Ok(Self {
            directory,
            spec: spec.clone(),
            receipt,
            failure_status_at_open,
        })
    }
    pub fn chunk_count(&self) -> usize {
        self.receipt.chunks.len()
    }
    pub fn receipt(&self) -> &PartialCollectionReceipt {
        &self.receipt
    }
    pub fn failure_status_at_open(&self) -> &InspectionFailureEvidenceStatus {
        &self.failure_status_at_open
    }
    pub fn failure_status_current(&self) -> InspectionFailureEvidenceStatus {
        inspection_failure_status(&self.directory, &self.spec, &self.receipt)
    }
    /// Verifies the whole selected chunk and its row seals before the first
    /// callback, then invokes the callback from the same immutable buffer.
    /// A callback error does not roll back effects of earlier callbacks.
    pub fn visit_chunk(
        &self,
        chunk_index: usize,
        visit: impl FnMut(
            &rz_experiments::rovent_corpus_lock_v2::record::RoventTrainingRecordLockV2,
        ) -> io::Result<()>,
    ) -> io::Result<usize> {
        visit_partial_chunk_in(&self.directory, &self.receipt, chunk_index, visit)
    }
}

fn validate_inspection_chunk_metadata(
    spec: &PartialCollectionSpec,
    receipt: &PartialCollectionReceipt,
) -> io::Result<()> {
    let mut declared = BTreeMap::<&str, (u64, u64)>::new();
    let mut total = 0u64;
    for pin in &receipt.chunks {
        let Some(name) = pin.path.strip_prefix("chunks/") else {
            return Err(invalid_partial("partial chunk path"));
        };
        if name.len() != 12
            || !name.ends_with(".jsonl")
            || !name.as_bytes()[..6].iter().all(u8::is_ascii_digit)
            || pin.bytes == 0
            || pin.bytes > rz_experiments::rovent_corpus::MAX_CHUNK_BYTES as u64
            || pin.rows == 0
            || !digest_string(&pin.sha256)
            || declared.insert(pin.path.as_str(), (0, 0)).is_some()
        {
            return Err(invalid_partial("partial chunk pin"));
        }
        total = total
            .checked_add(pin.bytes)
            .ok_or_else(|| invalid_partial("partial chunk bytes overflow"))?;
    }
    if receipt.failure_evidence.as_ref().is_some_and(|p| {
        total
            .checked_add(p.bytes)
            .is_none_or(|n| n > spec.byte_limit)
    }) || total > spec.byte_limit
    {
        return Err(invalid_partial("partial chunk budget"));
    }
    for row in receipt.rows.values() {
        if let RowState::Valid {
            record_reference,
            bytes,
        } = row
        {
            let entry = declared
                .get_mut(record_reference.as_str())
                .ok_or_else(|| invalid_partial("partial row chunk missing"))?;
            entry.0 = entry
                .0
                .checked_add(1)
                .ok_or_else(|| invalid_partial("partial row count overflow"))?;
            entry.1 = entry
                .1
                .checked_add(*bytes)
                .ok_or_else(|| invalid_partial("partial row bytes overflow"))?;
        }
    }
    for pin in &receipt.chunks {
        if declared.get(pin.path.as_str()) != Some(&(pin.rows, pin.bytes)) {
            return Err(invalid_partial(
                "partial chunk declaration/accounting differs",
            ));
        }
    }
    Ok(())
}

/// Convenient single-visit wrapper; repeated visits should reuse a reader.
pub fn visit_partial_chunk(
    parent: &Dir,
    spec: &PartialCollectionSpec,
    chunk_index: usize,
    visit: impl FnMut(
        &rz_experiments::rovent_corpus_lock_v2::record::RoventTrainingRecordLockV2,
    ) -> io::Result<()>,
) -> io::Result<usize> {
    PartialCollectionReader::open(parent, spec)?.visit_chunk(chunk_index, visit)
}

impl PartialCollectionReceipt {
    pub fn valid_rows(&self) -> usize {
        self.valid_rows
    }
    pub fn failed_rows(&self) -> usize {
        self.failed_rows
    }
    pub fn pending_rows(&self) -> usize {
        self.pending_rows
    }

    pub(super) fn with_chunks(mut self, chunks: Vec<PartialChunkPin>) -> Self {
        self.chunks = chunks;
        self
    }
    pub(super) fn with_failure_evidence(mut self, pin: PartialFailurePin) -> Self {
        self.failure_evidence = Some(pin);
        self
    }
    fn validate(&self, spec: &PartialCollectionSpec) -> io::Result<()> {
        if self.schema != PARTIAL_COLLECTION_DOMAIN
            || self.status != "incomplete"
            || self.training_eligible
            || self.publication_authority
            || self.physical_completion != "not_attested"
            || self.row_limit != spec.row_limit
            || self.byte_limit != spec.byte_limit
            || self.rows.len() > spec.row_limit
        {
            return Err(invalid_partial("partial receipt identity or limits"));
        }
        let (mut valid, mut failed, mut pending, mut bytes) = (0usize, 0usize, 0usize, 0u64);
        for (id, row) in &self.rows {
            if id.is_empty() || id.len() > MAX_ID_BYTES || id.chars().any(char::is_control) {
                return Err(invalid_partial("partial invalid row identity"));
            }
            let (reference, size) = match row {
                RowState::Pending => {
                    pending += 1;
                    continue;
                }
                RowState::Valid {
                    record_reference,
                    bytes,
                } => {
                    valid += 1;
                    (record_reference, *bytes)
                }
                RowState::Failed {
                    evidence_reference,
                    bytes,
                    evidence_status,
                    policy_target_known,
                    failure_classification,
                } => {
                    if evidence_status != "serialized_evidence_owner_not_reconstructable"
                        || *bytes == 0
                        || *policy_target_known
                        || failure_classification != "conditional_policy_unknown_first_observed"
                    {
                        return Err(invalid_partial("partial failure custody status"));
                    }
                    failed += 1;
                    (evidence_reference, *bytes)
                }
            };
            if reference.is_empty()
                || reference.len() > MAX_REFERENCE_BYTES
                || reference.chars().any(char::is_control)
                || (matches!(row, RowState::Valid { .. }) && size == 0)
            {
                return Err(invalid_partial("partial missing row reference"));
            }
            bytes = bytes
                .checked_add(size)
                .ok_or_else(|| invalid_partial("partial byte overflow"))?;
        }
        if (valid, failed, pending, bytes)
            != (
                self.valid_rows,
                self.failed_rows,
                self.pending_rows,
                self.declared_payload_bytes,
            )
            || bytes > spec.byte_limit
            || (self.end_reason == PartialEndReason::InputExhausted && pending != 0)
        {
            return Err(invalid_partial("partial row accounting differs"));
        }
        match &self.failure_evidence {
            Some(pin)
                if failed > 0
                    && pin.path == FAILURE_FILE
                    && pin.bytes > 0
                    && pin.bytes <= spec.byte_limit
                    && pin.rows == failed as u64
                    && digest_string(&pin.sha256) => {}
            None if failed == 0 => {}
            _ => return Err(invalid_partial("partial failure evidence pin")),
        }
        Ok(())
    }
}

/// Caller retains every original row and failure evidence until independently
/// safe transfer/release. A reference is required for each resolved entry but is
/// never treated as proof that a file exists or a transfer is durable.
#[derive(Clone)]
pub(super) struct PartialCollectionAccounting {
    row_limit: usize,
    byte_limit: u64,
    declared_payload_bytes: u64,
    rows: BTreeMap<String, RowState>,
}

impl PartialCollectionAccounting {
    pub(super) fn new(row_limit: usize, byte_limit: u64) -> Result<Self, PartialAccountingError> {
        if row_limit == 0 || byte_limit == 0 {
            return Err(PartialAccountingError::InvalidLimit);
        }
        Ok(Self {
            row_limit,
            byte_limit,
            declared_payload_bytes: 0,
            rows: BTreeMap::new(),
        })
    }

    /// Registers custody before attempting work; failures leave state intact.
    pub(super) fn begin_row(&mut self, id: &str) -> Result<(), PartialAccountingError> {
        if id.is_empty() || id.len() > MAX_ID_BYTES || id.chars().any(char::is_control) {
            return Err(PartialAccountingError::InvalidIdentifier);
        }
        if self.rows.contains_key(id) {
            return Err(PartialAccountingError::DuplicateRow);
        }
        if self.rows.len() >= self.row_limit {
            return Err(PartialAccountingError::RowLimit);
        }
        self.rows.insert(id.to_owned(), RowState::Pending);
        Ok(())
    }

    pub(super) fn record_valid(
        &mut self,
        id: &str,
        record_reference: &str,
        bytes: u64,
    ) -> Result<(), PartialAccountingError> {
        self.resolve(id, record_reference, bytes, false)
    }

    /// The reference must name the caller's separate retained failure evidence.
    /// It does not prove its validity or that any other target is trustworthy.
    pub(super) fn record_failure(
        &mut self,
        id: &str,
        evidence_reference: &str,
        bytes: u64,
    ) -> Result<(), PartialAccountingError> {
        self.resolve(id, evidence_reference, bytes, true)
    }

    fn resolve(
        &mut self,
        id: &str,
        reference: &str,
        bytes: u64,
        failure: bool,
    ) -> Result<(), PartialAccountingError> {
        if reference.is_empty()
            || reference.len() > MAX_REFERENCE_BYTES
            || reference.chars().any(char::is_control)
        {
            return Err(PartialAccountingError::InvalidReference);
        }
        match self.rows.get(id) {
            None => return Err(PartialAccountingError::UnknownRow),
            Some(RowState::Pending) => {}
            Some(_) => return Err(PartialAccountingError::AlreadyResolved),
        }
        let total = self
            .declared_payload_bytes
            .checked_add(bytes)
            .filter(|total| bytes > 0 && *total <= self.byte_limit)
            .ok_or(PartialAccountingError::ByteLimit)?;
        let state = if failure {
            RowState::Failed {
                evidence_reference: reference.to_owned(),
                bytes,
                evidence_status: "serialized_evidence_owner_not_reconstructable".into(),
                policy_target_known: false,
                failure_classification: "conditional_policy_unknown_first_observed".into(),
            }
        } else {
            RowState::Valid {
                record_reference: reference.to_owned(),
                bytes,
            }
        };
        *self.rows.get_mut(id).expect("row checked above") = state;
        self.declared_payload_bytes = total;
        Ok(())
    }

    /// Exhaustion can only be declared when all admitted rows have outcomes.
    /// Every receipt remains partial even in that case. Other end reasons keep
    /// pending IDs visible; they do not release or prove completion of owners.
    pub(super) fn finish(
        self,
        reason: PartialEndReason,
    ) -> Result<PartialCollectionReceipt, (PartialAccountingError, Self)> {
        let pending_rows = self
            .rows
            .values()
            .filter(|v| matches!(v, RowState::Pending))
            .count();
        if reason == PartialEndReason::InputExhausted && pending_rows != 0 {
            return Err((PartialAccountingError::PendingRows, self));
        }
        let valid_rows = self
            .rows
            .values()
            .filter(|v| matches!(v, RowState::Valid { .. }))
            .count();
        let failed_rows = self
            .rows
            .values()
            .filter(|v| matches!(v, RowState::Failed { .. }))
            .count();
        Ok(PartialCollectionReceipt {
            schema: PARTIAL_COLLECTION_DOMAIN.into(),
            status: "incomplete".into(),
            training_eligible: false,
            publication_authority: false,
            physical_completion: "not_attested".into(),
            end_reason: reason,
            valid_rows,
            failed_rows,
            pending_rows,
            declared_payload_bytes: self.declared_payload_bytes,
            row_limit: self.row_limit,
            byte_limit: self.byte_limit,
            rows: self.rows,
            chunks: Vec::new(),
            failure_evidence: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inspection_reports_missing_failure_and_never_visits_invalid_chunk() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "rz-independent-inspection-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        let parent = Dir::open_ambient_dir(&path, cap_std::ambient_authority()).unwrap();
        let output = create_partial_directory(&parent).unwrap();
        output.create_dir("chunks").unwrap();
        let chunk = b"{}\n"; // Deliberately invalid lock-v2 row, never authority.
        std::fs::write(
            path.join(DIRECTORY).join("chunks").join("000000.jsonl"),
            chunk,
        )
        .unwrap();
        let mut state = PartialCollectionAccounting::new(8, 4096).unwrap();
        state.begin_row("row").unwrap();
        state
            .record_valid("row", "chunks/000000.jsonl", chunk.len() as u64)
            .unwrap();
        state.begin_row("failure-sequence-1").unwrap();
        state
            .record_failure(
                "failure-sequence-1",
                "failure-evidence.jsonl#failure-sequence-1",
                20,
            )
            .unwrap();
        let receipt = state
            .finish(PartialEndReason::InputExhausted)
            .ok()
            .unwrap()
            .with_chunks(vec![PartialChunkPin {
                path: "chunks/000000.jsonl".into(),
                bytes: chunk.len() as u64,
                sha256: format!("{:x}", Sha256::digest(chunk)),
                rows: 1,
            }])
            .with_failure_evidence(PartialFailurePin {
                path: FAILURE_FILE.into(),
                bytes: 20,
                sha256: "0".repeat(64),
                rows: 1,
            });
        // Synthetic, metadata-only corruption fixture; the strict writer
        // cannot and does not publish this fabricated receipt.
        let stored = PartialStoredReceipt {
            spec: spec(),
            receipt,
        };
        std::fs::write(
            path.join(DIRECTORY).join(FILE),
            serde_json::to_vec(&stored).unwrap(),
        )
        .unwrap();
        assert!(PartialCollectionReader::open(&parent, &spec()).is_err());
        let reader = PartialInspectionReader::open(&parent, &spec()).unwrap();
        assert_eq!(reader.chunk_count(), 1);
        assert!(matches!(
            reader.failure_status_at_open(),
            InspectionFailureEvidenceStatus::Missing { .. }
        ));
        let mut calls = 0;
        assert!(
            reader
                .visit_chunk(0, |_| {
                    calls += 1;
                    Ok(())
                })
                .is_err()
        );
        assert_eq!(calls, 0);
        std::fs::write(
            path.join(DIRECTORY).join(FAILURE_FILE),
            b"invalid failure data",
        )
        .unwrap();
        assert!(matches!(
            reader.failure_status_current(),
            InspectionFailureEvidenceStatus::Invalid { .. }
        ));
        assert!(matches!(
            reader.failure_status_at_open(),
            InspectionFailureEvidenceStatus::Missing { .. }
        ));
        drop(reader);
        drop(output);
        drop(parent);
        std::fs::remove_dir_all(path).unwrap();
    }

    fn spec() -> PartialCollectionSpec {
        PartialCollectionSpec {
            schema: PARTIAL_COLLECTION_SPEC_DOMAIN.into(),
            collection_plan_sha256: "a".repeat(64),
            row_limit: 8,
            byte_limit: 4096,
            receipt_byte_limit: 4096,
        }
    }

    #[test]
    fn spec_rejects_wrong_domain_plan_and_duplicate_fields() {
        let source = serde_json::to_string(&spec()).unwrap();
        assert!(PartialCollectionSpec::from_json(&source, &"a".repeat(64)).is_ok());
        assert!(PartialCollectionSpec::from_json(&source, &"b".repeat(64)).is_err());
        let mut wrong = spec();
        wrong.schema = "strict-v2".into();
        assert!(
            PartialCollectionSpec::from_json(
                &serde_json::to_string(&wrong).unwrap(),
                &"a".repeat(64)
            )
            .is_err()
        );
        let duplicate = source.replacen("{", "{\"row_limit\":1,", 1);
        assert!(PartialCollectionSpec::from_json(&duplicate, &"a".repeat(64)).is_err());
    }

    #[test]
    fn receipt_rejects_claimed_success_and_known_failed_policy() {
        let mut state = PartialCollectionAccounting::new(8, 4096).unwrap();
        state.begin_row("failed").unwrap();
        state
            .record_failure("failed", "failure-evidence.jsonl#failed", 1)
            .unwrap();
        let mut receipt = state
            .finish(PartialEndReason::InputExhausted)
            .ok()
            .unwrap()
            .with_failure_evidence(PartialFailurePin {
                path: FAILURE_FILE.into(),
                bytes: 1,
                sha256: "0".repeat(64),
                rows: 1,
            });
        assert!(receipt.validate(&spec()).is_ok());
        receipt.status = "complete".into();
        assert!(receipt.validate(&spec()).is_err());
        receipt.status = "incomplete".into();
        if let Some(RowState::Failed {
            policy_target_known,
            ..
        }) = receipt.rows.get_mut("failed")
        {
            *policy_target_known = true;
        }
        assert!(receipt.validate(&spec()).is_err());
    }

    #[test]
    fn partial_receipt_is_separate_and_tampering_is_rejected() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "rz-partial-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        let parent = Dir::open_ambient_dir(&path, cap_std::ambient_authority()).unwrap();
        let output = create_partial_directory(&parent).unwrap();
        let mut state = PartialCollectionAccounting::new(8, 4096).unwrap();
        state.begin_row("failure-sequence-1").unwrap();
        let work = serde_json::json!({"nodes":null,"qnodes":null,"tt_hits":null});
        let elapsed = serde_json::json!({"seconds":0,"nanoseconds":0});
        let report = serde_json::json!({"identity":{"adapter_semantics":"fixture","binary_sha256":"a","launch_arguments_sha256":"b",
            "declared_name":"fixture","declared_version":"fixture","declared_source":"fixture","declared_license":"fixture",
            "options":{},"assets":[],"model_metadata":{"weights_sha256":null,"training":"unknown","declared_rights":null,"precision":null}},
            "observed_uci":{"name":"fixture","author":null},"request_id":1,"best_move":null,"pv":[],
            "score":{"kind":"unknown"},"bound":"unknown","wdl_per_mille":null,"perspective":"white","requested_depth":1,
            "reported_depth":null,"seldepth":null,"root_restricted":false,"completion":"best_move","work":work,"elapsed":elapsed});
        let attempt = serde_json::json!({"work":work,"elapsed":elapsed,"external":{"request_id":1,"partial_report":null,
            "process":{"process_identity":null,"stop_sent":false,"quit_sent":false,"exit_observed":false,
                "stdout_drained":false,"stderr_drained":false,"exit_code":null,"exit_signal":null,
                "cleanup_complete":false,"quarantined":false,"ownership_lost":false,"stdout_bytes":0,"stderr_bytes":0}}});
        assert!(portable_report(&report));
        assert!(portable_attempt(&attempt));
        let mut malformed_report = report.clone();
        malformed_report
            .as_object_mut()
            .unwrap()
            .remove("completion");
        assert!(!portable_report(&malformed_report));
        let mut malformed_attempt = attempt.clone();
        malformed_attempt["external"]["process"]["stdout_bytes"] = "missing".into();
        assert!(!portable_attempt(&malformed_attempt));
        let evidence = serde_json::json!({"schema":FAILURE_DOMAIN,"row_id":"failure-sequence-1",
            "original_row":{"native":{"retired_sequence":1}},"error":"fixture only",
            "teacher_evidence":{"schema":"portable_teacher_observation_v1","dispatch":{},"completion":{},"input_scope":{},"startup":{},
                "raw_report":report, "attempt":attempt, "opinions":[],
                "profiled_report":{"profile":{},"purpose":"teacher","requested":{},"issued":{},"issued_go":true,
                    "reported_node_overshoot":null,"strict_native_nodes":false,"option_application_observed":false,"readiness_barrier_completed":true},
                "requested_multipv":1,"stop_control":null,
                "opaque_exclusions":["owner_capabilities","absolute_monotonic_instants","unretained_stdout_stderr","opaque_rules_history_snapshot"],
                "conditional_policy_target_known":false,"portable_observation_complete":true,"truncated":false},
            "observation_provenance":"retained_original_sealed_serializable_projection",
            "owner_reconstructable":false,"physical_completion":"not_attested"});
        let mut evidence_bytes = serde_json::to_vec(&evidence).unwrap();
        evidence_bytes.push(b'\n');
        std::fs::write(path.join(DIRECTORY).join(FAILURE_FILE), &evidence_bytes).unwrap();
        state
            .record_failure(
                "failure-sequence-1",
                "failure-evidence.jsonl#failure-sequence-1",
                evidence_bytes.len() as u64,
            )
            .unwrap();
        let receipt = state
            .finish(PartialEndReason::InputExhausted)
            .ok()
            .unwrap()
            .with_failure_evidence(PartialFailurePin {
                path: FAILURE_FILE.into(),
                bytes: evidence_bytes.len() as u64,
                sha256: format!("{:x}", Sha256::digest(&evidence_bytes)),
                rows: 1,
            });
        std::fs::write(path.join(DIRECTORY).join(FAILURE_FILE), b"corrupt\n").unwrap();
        assert!(write_partial_receipt(&output, &spec(), &receipt).is_err());
        assert!(!path.join(DIRECTORY).join(FILE).exists());
        std::fs::write(path.join(DIRECTORY).join(FAILURE_FILE), &evidence_bytes).unwrap();
        // This intentionally incomplete synthetic original row must never pass
        // the real lock-v2 decoder, even if the file's digest is correct.
        assert!(write_partial_receipt(&output, &spec(), &receipt).is_err());
        assert!(!path.join(DIRECTORY).join(FILE).exists());
        let empty = PartialCollectionAccounting::new(8, 4096)
            .unwrap()
            .finish(PartialEndReason::InputExhausted)
            .ok()
            .unwrap();
        write_partial_receipt(&output, &spec(), &empty).unwrap();
        assert!(read_partial_receipt(&parent, &spec()).is_ok());
        {
            use super::super::partial_followup_v1::{
                FollowupDecision, FollowupDependency, FollowupResources,
                PARTIAL_COLLECTION_EXIT_CODE, PriorResourceState, ProcessTreeState, ProducerExit,
                decide_partial_followup,
            };
            let reader = PartialCollectionReader::open(&parent, &spec()).unwrap();
            assert!(!reader.verify_failure_evidence_current().unwrap());
            let inspection = PartialInspectionReader::open(&parent, &spec()).unwrap();
            assert!(
                super::super::partial_followup_v1::visit_independent_partial_chunk(
                    &inspection,
                    0,
                    FollowupResources::Isolated,
                    PriorResourceState::Unknown,
                    ProducerExit::Exited(PARTIAL_COLLECTION_EXIT_CODE),
                    ProcessTreeState::GoneObserved,
                    |_| panic!("empty receipt cannot yield a valid row"),
                )
                .is_err()
            );
            let decide = |dep, resources| {
                decide_partial_followup(
                    dep,
                    resources,
                    PriorResourceState::Unknown,
                    ProducerExit::Exited(PARTIAL_COLLECTION_EXIT_CODE),
                    ProcessTreeState::GoneObserved,
                    Some(&reader),
                )
            };
            assert_eq!(
                decide(
                    FollowupDependency::FailureEvidenceInspection,
                    FollowupResources::Isolated,
                ),
                FollowupDecision::BlockedPartialEvidence
            );
            assert_eq!(
                decide(
                    FollowupDependency::PartialInspection,
                    FollowupResources::Isolated
                ),
                FollowupDecision::BlockedPartialEvidence
            );
            assert_eq!(
                decide(
                    FollowupDependency::StrictCorpus,
                    FollowupResources::Isolated
                ),
                FollowupDecision::BlockedStrictCorpus
            );
            assert_eq!(
                decide(
                    FollowupDependency::Independent,
                    FollowupResources::ReusesPrior
                ),
                FollowupDecision::BlockedSharedResource
            );
            assert_eq!(
                decide(FollowupDependency::Independent, FollowupResources::Isolated),
                FollowupDecision::Run
            );
        }
        assert!(!path.join("publication.json").exists());
        assert!(!path.join(DIRECTORY).join("publication.json").exists());
        assert!(write_partial_receipt(&output, &spec(), &empty).is_err());
        let receipt_path = path.join(DIRECTORY).join(FILE);
        let mut raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&receipt_path).unwrap()).unwrap();
        raw["receipt"]["rows"]["failure-sequence-1"]["policy_target_known"] = true.into();
        std::fs::write(&receipt_path, serde_json::to_vec(&raw).unwrap()).unwrap();
        assert!(read_partial_receipt(&parent, &spec()).is_err());
        drop(output);
        drop(parent);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn duplicate_and_tampered_chunk_pins_are_rejected() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "rz-partial-chunks-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        let directory = Dir::open_ambient_dir(&path, cap_std::ambient_authority()).unwrap();
        directory.create_dir("chunks").unwrap();
        let chunk = b"{}\n";
        std::fs::write(path.join("chunks").join("000000.jsonl"), chunk).unwrap();
        let pin = PartialChunkPin {
            path: "chunks/000000.jsonl".into(),
            bytes: chunk.len() as u64,
            sha256: "0".repeat(64),
            rows: 1,
        };
        assert!(
            scan_chunks(
                &directory,
                &spec(),
                &[pin.clone(), pin.clone()],
                |_, _, _| Ok(())
            )
            .is_err()
        );
        assert!(scan_chunks(&directory, &spec(), &[pin], |_, _, _| Ok(())).is_err());
        drop(directory);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn valid_and_failure_remain_distinct_and_ineligible() {
        let mut state = PartialCollectionAccounting::new(3, 100).unwrap();
        state.begin_row("bad").unwrap();
        state
            .record_failure("bad", "failure-evidence.jsonl#bad", 1)
            .unwrap();
        state.begin_row("good").unwrap();
        state.record_valid("good", "records/good", 30).unwrap();
        let receipt = state.finish(PartialEndReason::InputExhausted).ok().unwrap();
        let value = serde_json::to_value(receipt).unwrap();
        assert_eq!(value["schema"], PARTIAL_COLLECTION_DOMAIN);
        assert_eq!(value["status"], "incomplete");
        assert_eq!(value["training_eligible"], false);
        assert_eq!(value["publication_authority"], false);
        assert_eq!(value["physical_completion"], "not_attested");
        assert_eq!(value["valid_rows"], 1);
        assert_eq!(value["failed_rows"], 1);
        assert_eq!(
            value["rows"]["bad"]["evidence_reference"],
            "failure-evidence.jsonl#bad"
        );
        assert_eq!(
            value["rows"]["bad"]["evidence_status"],
            "serialized_evidence_owner_not_reconstructable"
        );
        assert_eq!(value["rows"]["bad"]["policy_target_known"], false);
        assert_eq!(
            value["rows"]["bad"]["failure_classification"],
            "conditional_policy_unknown_first_observed"
        );
    }

    #[test]
    fn limits_fail_without_losing_pending_or_original_accounting() {
        let mut state = PartialCollectionAccounting::new(1, 5).unwrap();
        state.begin_row("one").unwrap();
        assert_eq!(
            state.begin_row("one"),
            Err(PartialAccountingError::DuplicateRow)
        );
        assert_eq!(
            state.begin_row("two"),
            Err(PartialAccountingError::RowLimit)
        );
        assert_eq!(
            state.record_failure("one", "", 1),
            Err(PartialAccountingError::InvalidReference)
        );
        assert_eq!(
            state.record_failure("one", "evidence", 6),
            Err(PartialAccountingError::ByteLimit)
        );
        let (error, mut state) = state
            .finish(PartialEndReason::InputExhausted)
            .err()
            .unwrap();
        assert_eq!(error, PartialAccountingError::PendingRows);
        state.record_failure("one", "evidence", 1).unwrap();
        assert_eq!(
            state.record_valid("one", "record", 1),
            Err(PartialAccountingError::AlreadyResolved)
        );
        let receipt = state.finish(PartialEndReason::Interrupted).ok().unwrap();
        assert_eq!(
            (
                receipt.failed_rows,
                receipt.pending_rows,
                receipt.declared_payload_bytes
            ),
            (1, 0, 1)
        );
    }

    #[test]
    fn interruption_preserves_pending_id_without_claiming_completion() {
        let mut state = PartialCollectionAccounting::new(1, 1).unwrap();
        state.begin_row("pending").unwrap();
        let receipt = state.finish(PartialEndReason::Interrupted).ok().unwrap();
        assert_eq!(receipt.pending_rows, 1);
        assert!(matches!(receipt.rows["pending"], RowState::Pending));
        assert_eq!(receipt.physical_completion, "not_attested");
    }
}
