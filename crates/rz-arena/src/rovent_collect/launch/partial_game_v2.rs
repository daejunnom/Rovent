//! A separate game-level index of independently checked /1 partial outputs.
//!
//! This index is an observation only: it neither proves the original in-memory
//! owner nor closes a process, establishes a device fence or grants training
//! eligibility. Unknown roots remain unknown. Every read revalidates the pins
//! and root identities of all referenced /1 outputs.
//! The byte limit covers the referenced /1 payloads and receipts and this
//! index, not unrelated files or the total filesystem allocation.
use super::partial_collection_v1::{
    MAX_FAILURE_ROW_BYTES, PartialCollectionReader, PartialCollectionSpec,
};
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::{Dir, OpenOptions};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    io::{self, BufRead, BufReader, Read, Write},
    path::{Component, Path},
};

pub const GAME_PARTIAL_INDEX_V2_DOMAIN: &str = "rz-rovent-game-partial-index/2";
/// Dedicated nonzero result for a written game partial index; never strict success.
pub const GAME_PARTIAL_EXIT_CODE: u8 = 3;
const INDEX_FILE: &str = "partial-game-v2.json";
const INDEX_PENDING: &str = "partial-game-v2.pending.json";
const FAILURE_PATH: &str = "failure-evidence.jsonl";
const MAX_INDEX_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GamePartialPinV2 {
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum GamePartialRootV2 {
    Available {
        slot: u16,
        root_admission_sha256: String,
        relative_path: String,
        spec: PartialCollectionSpec,
        receipt: GamePartialPinV2,
    },
    /// An unverified clue, not proof that a root was admitted or completed.
    Unknown {
        slot: u16,
        root_admission_sha256: Option<String>,
        reason: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GamePartialIndexV2 {
    pub schema: String,
    pub collection_plan_sha256: String,
    pub partial_spec_sha256: String,
    pub status: String,
    pub training_eligible: bool,
    pub publication_authority: bool,
    pub physical_completion: String,
    /// Registered capacity, not the number of roots actually admitted.
    pub root_capacity: u16,
    pub max_total_rows: u64,
    pub max_total_bytes: u64,
    pub index_max_bytes: u64,
    pub observed_valid_rows: u64,
    pub observed_failed_rows: u64,
    pub roots: Vec<GamePartialRootV2>,
}

fn bad(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn digest_ok(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}
fn valid_reason(value: &str) -> bool {
    !value.is_empty() && value.len() <= 4096 && !value.chars().any(char::is_control)
}
fn valid_relative_root(relative: &str) -> bool {
    !relative.is_empty()
        && !relative.contains(['\\', ':', '\0'])
        && !relative
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
        && Path::new(relative)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn open_relative_directory(parent: &Dir, relative: &str) -> io::Result<Dir> {
    if !valid_relative_root(relative) {
        return Err(bad("game partial root path"));
    }
    let path = Path::new(relative);
    let mut current = parent.try_clone()?;
    for component in path.components() {
        match component {
            Component::Normal(name) => current = current.open_dir_nofollow(name)?,
            _ => return Err(bad("game partial root path")),
        }
    }
    Ok(current)
}
fn read_file(directory: &Dir, name: &str, limit: u64) -> io::Result<Vec<u8>> {
    let mut options = OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    let file = directory.open_with(name, &options)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > limit {
        return Err(bad("game partial file limit"));
    }
    let mut bytes = Vec::new();
    file.take(limit.saturating_add(1)).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(bad("game partial file grew"));
    }
    Ok(bytes)
}

/// Capture a file pin only; the game-index writer separately validates /1
/// semantics, source binding and every actual row's root admission.
pub fn pin_root_partial_receipt(
    parent: &Dir,
    relative_path: &str,
    spec: &PartialCollectionSpec,
) -> io::Result<GamePartialPinV2> {
    spec.validate(&spec.collection_plan_sha256)?;
    let root = open_relative_directory(parent, relative_path)?;
    let partial = root.open_dir_nofollow("partial_collection")?;
    let raw = read_file(&partial, "partial_collection.json", spec.receipt_byte_limit)?;
    if raw.is_empty() {
        return Err(bad("empty original partial receipt"));
    }
    Ok(GamePartialPinV2 {
        bytes: raw.len() as u64,
        sha256: format!("{:x}", Sha256::digest(&raw)),
    })
}
fn root_of_record(
    row: &rz_experiments::rovent_corpus_lock_v2::record::RoventTrainingRecordLockV2,
) -> Option<&str> {
    row.0
        .native
        .raw_retirement
        .get("root_admission_sha256")?
        .as_str()
}

/// Verify original failure projection bytes against the *same /1 receipt*;
/// the original in-process sealed owner remains non-reconstructable.
pub(super) fn verify_failure_root(
    directory: &Dir,
    pin: &serde_json::Value,
    expected: &str,
    limit: u64,
) -> io::Result<u64> {
    let bytes = pin
        .get("bytes")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| bad("failure pin bytes"))?;
    let rows = pin
        .get("rows")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| bad("failure pin rows"))?;
    let digest = pin
        .get("sha256")
        .and_then(serde_json::Value::as_str)
        .filter(|s| digest_ok(s))
        .ok_or_else(|| bad("failure pin digest"))?;
    if pin.get("path").and_then(serde_json::Value::as_str) != Some(FAILURE_PATH)
        || bytes > limit
        || rows == 0
    {
        return Err(bad("failure pin scope"));
    }
    let mut options = OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    let file = directory.open_with(FAILURE_PATH, &options)?;
    if !file.metadata()?.is_file() || file.metadata()?.len() != bytes {
        return Err(bad("failure file size"));
    }
    let mut reader = BufReader::new(file.take(bytes.saturating_add(1)));
    let (mut count, mut size) = (0u64, 0u64);
    let mut hash = Sha256::new();
    loop {
        let mut line = Vec::new();
        let n = (&mut reader)
            .take(MAX_FAILURE_ROW_BYTES + 1)
            .read_until(b'\n', &mut line)?;
        if n == 0 {
            break;
        }
        if line.last() != Some(&b'\n') {
            return Err(bad("failure line bound"));
        }
        size = size
            .checked_add(n as u64)
            .ok_or_else(|| bad("failure byte overflow"))?;
        count = count
            .checked_add(1)
            .ok_or_else(|| bad("failure row overflow"))?;
        hash.update(&line);
        let text =
            std::str::from_utf8(&line[..line.len() - 1]).map_err(|_| bad("failure UTF-8"))?;
        let value: serde_json::Value =
            rz_experiments::decode_json(text).map_err(|_| bad("failure JSON"))?;
        let row = value
            .get("original_row")
            .ok_or_else(|| bad("failure row missing"))?;
        if row
            .pointer("/native/raw_retirement/root_admission_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(expected)
        {
            return Err(bad("failure row belongs to another root"));
        }
    }
    if count != rows || size != bytes || format!("{:x}", hash.finalize()) != digest {
        return Err(bad("failure pin changed"));
    }
    Ok(count)
}

fn validate_index_shape(index: &GamePartialIndexV2) -> io::Result<()> {
    if index.schema != GAME_PARTIAL_INDEX_V2_DOMAIN
        || !digest_ok(&index.collection_plan_sha256)
        || !digest_ok(&index.partial_spec_sha256)
        || index.status != "incomplete"
        || index.training_eligible
        || index.publication_authority
        || index.physical_completion != "not_attested"
        || index.root_capacity == 0
        || index.root_capacity > 64
        || index.roots.len() != index.root_capacity as usize
        || index.max_total_rows == 0
        || index.max_total_bytes == 0
        || index.index_max_bytes == 0
        || index.index_max_bytes > MAX_INDEX_BYTES
    {
        return Err(bad("game partial index identity or limits"));
    }
    let mut slots = BTreeSet::new();
    for root in &index.roots {
        let slot = match root {
            GamePartialRootV2::Unknown { slot, .. } | GamePartialRootV2::Available { slot, .. } => {
                *slot
            }
        };
        if slot >= index.root_capacity || !slots.insert(slot) {
            return Err(bad("game partial root slot differs"));
        }
    }
    let size = serde_json::to_vec(index).map_err(io::Error::other)?.len() as u64;
    if size > index.index_max_bytes || size > index.max_total_bytes {
        return Err(bad("game partial index byte limit"));
    }
    Ok(())
}

fn validate_game_partial_metadata(
    index: &GamePartialIndexV2,
    source_plan_bytes: &[u8],
    source_spec_bytes: &[u8],
) -> io::Result<PartialCollectionSpec> {
    if source_plan_bytes.len() as u64 > super::ROVENT_LOCK_V2_PLAN_MAX_BYTES
        || format!("{:x}", Sha256::digest(source_plan_bytes)) != index.collection_plan_sha256
    {
        return Err(bad("game partial source plan bytes or digest"));
    }
    let source_plan: super::RoventCollectionLockV2LaunchPlan = rz_experiments::decode_json(
        std::str::from_utf8(source_plan_bytes).map_err(|_| bad("source plan UTF-8"))?,
    )
    .map_err(|_| bad("source plan JSON"))?;
    if source_plan.schema != super::ROVENT_COLLECTION_LOCK_V2_LAUNCH_PLAN
        || source_plan.maximum_roots != index.root_capacity
    {
        return Err(bad("game partial original plan capacity differs"));
    }
    let source_spec = PartialCollectionSpec::from_json(
        std::str::from_utf8(source_spec_bytes).map_err(|_| bad("source partial spec UTF-8"))?,
        &index.collection_plan_sha256,
    )?;
    if format!("{:x}", Sha256::digest(source_spec_bytes)) != index.partial_spec_sha256 {
        return Err(bad("game partial original spec digest differs"));
    }
    if index.max_total_rows > source_spec.row_limit as u64 {
        return Err(bad("game partial row limit exceeds original spec"));
    }
    validate_index_shape(index)?;
    let mut roots = BTreeSet::new();
    let mut paths = BTreeSet::new();
    let mut declared_receipt_bytes = 0u64;
    for entry in &index.roots {
        match entry {
            GamePartialRootV2::Unknown {
                root_admission_sha256,
                ..
            } => {
                if let Some(id) = root_admission_sha256 {
                    if !roots.insert(id.clone()) {
                        return Err(bad("duplicate root"));
                    }
                }
            }
            GamePartialRootV2::Available {
                root_admission_sha256,
                relative_path,
                receipt,
                ..
            } => {
                if !roots.insert(root_admission_sha256.clone())
                    || !paths.insert(relative_path.clone())
                {
                    return Err(bad("duplicate root or path"));
                }
                declared_receipt_bytes = declared_receipt_bytes
                    .checked_add(receipt.bytes)
                    .ok_or_else(|| bad("game partial declared byte overflow"))?;
            }
        }
    }
    let index_bytes = serde_json::to_vec(index).map_err(io::Error::other)?.len() as u64;
    if index
        .observed_valid_rows
        .checked_add(index.observed_failed_rows)
        .is_none_or(|n| n > index.max_total_rows)
        || declared_receipt_bytes
            .checked_add(index_bytes)
            .is_none_or(|n| n > index.max_total_bytes)
    {
        return Err(bad("game partial declared totals or limits differ"));
    }
    Ok(source_spec)
}

fn validate_root_declaration(
    index: &GamePartialIndexV2,
    source_spec: &PartialCollectionSpec,
    entry: &GamePartialRootV2,
) -> io::Result<()> {
    match entry {
        GamePartialRootV2::Unknown {
            root_admission_sha256,
            reason,
            ..
        } => {
            if !valid_reason(reason)
                || root_admission_sha256
                    .as_ref()
                    .is_some_and(|id| !digest_ok(id))
            {
                return Err(bad("unknown root declaration"));
            }
        }
        GamePartialRootV2::Available {
            root_admission_sha256,
            relative_path,
            spec,
            receipt,
            ..
        } => {
            if !digest_ok(root_admission_sha256)
                || !valid_relative_root(relative_path)
                || !digest_ok(&receipt.sha256)
                || receipt.bytes == 0
                || receipt.bytes > spec.receipt_byte_limit
            {
                return Err(bad("available root declaration"));
            }
            spec.validate(&index.collection_plan_sha256)?;
            if spec.row_limit > source_spec.row_limit
                || spec.byte_limit > source_spec.byte_limit
                || spec.receipt_byte_limit > source_spec.receipt_byte_limit
            {
                return Err(bad("game partial root limit exceeds original spec"));
            }
        }
    }
    Ok(())
}

/// Validate /1 receipts and root-local data; this does not restore authority.
pub fn validate_game_partial_index(
    parent: &Dir,
    index: &GamePartialIndexV2,
    source_plan_bytes: &[u8],
    source_spec_bytes: &[u8],
) -> io::Result<()> {
    let source_spec = validate_game_partial_metadata(index, source_plan_bytes, source_spec_bytes)?;
    let mut roots = BTreeSet::new();
    let mut paths = BTreeSet::new();
    let (mut valid, mut failed, mut total_bytes) = (0u64, 0u64, 0u64);
    for entry in &index.roots {
        validate_root_declaration(index, &source_spec, entry)?;
        match entry {
            GamePartialRootV2::Unknown {
                root_admission_sha256,
                reason,
                ..
            } => {
                if !valid_reason(reason) {
                    return Err(bad("unknown root declaration"));
                }
                if let Some(id) = root_admission_sha256 {
                    if !digest_ok(id) || !roots.insert(id.clone()) {
                        return Err(bad("duplicate or invalid root"));
                    }
                }
            }
            GamePartialRootV2::Available {
                root_admission_sha256,
                relative_path,
                spec,
                receipt,
                ..
            } => {
                if !digest_ok(root_admission_sha256)
                    || !roots.insert(root_admission_sha256.clone())
                    || !paths.insert(relative_path.clone())
                    || !digest_ok(&receipt.sha256)
                    || receipt.bytes == 0
                    || receipt.bytes > spec.receipt_byte_limit
                {
                    return Err(bad("available root declaration"));
                }
                spec.validate(&index.collection_plan_sha256)?;
                if spec.row_limit > source_spec.row_limit
                    || spec.byte_limit > source_spec.byte_limit
                    || spec.receipt_byte_limit > source_spec.receipt_byte_limit
                {
                    return Err(bad("game partial root limit exceeds original spec"));
                }
                let root = open_relative_directory(parent, relative_path)?;
                let partial = root.open_dir_nofollow("partial_collection")?;
                let raw = read_file(&partial, "partial_collection.json", spec.receipt_byte_limit)?;
                if raw.len() as u64 != receipt.bytes
                    || format!("{:x}", Sha256::digest(&raw)) != receipt.sha256
                {
                    return Err(bad("original partial receipt pin differs"));
                }
                let wrapped: serde_json::Value = rz_experiments::decode_json(
                    std::str::from_utf8(&raw).map_err(|_| bad("partial receipt UTF-8"))?,
                )
                .map_err(|_| bad("partial receipt JSON"))?;
                let reader = PartialCollectionReader::open(&root, spec)?;
                if wrapped.get("spec")
                    != Some(&serde_json::to_value(spec).map_err(io::Error::other)?)
                    || wrapped.get("receipt")
                        != Some(&serde_json::to_value(reader.receipt()).map_err(io::Error::other)?)
                {
                    return Err(bad("partial reader and pinned receipt differ"));
                }
                if reader.receipt().pending_rows() != 0 {
                    return Err(bad("available root has pending rows"));
                }
                if reader.receipt().valid_rows() == 0 && reader.receipt().failed_rows() == 0 {
                    return Err(bad("available root has no root-scoped row evidence"));
                }
                for i in 0..reader.chunk_count() {
                    reader.visit_chunk(i, |row| {
                        if root_of_record(row) != Some(root_admission_sha256.as_str()) {
                            return Err(bad("valid row belongs to another root"));
                        }
                        Ok(())
                    })?;
                }
                let report = wrapped
                    .get("receipt")
                    .ok_or_else(|| bad("partial receipt absent"))?;
                let failure_pin = report
                    .get("failure_evidence")
                    .ok_or_else(|| bad("failure pin field absent"))?;
                let actual_failed = if failure_pin.is_null() {
                    0
                } else {
                    verify_failure_root(
                        &partial,
                        failure_pin,
                        root_admission_sha256,
                        spec.byte_limit,
                    )?
                };
                if actual_failed != reader.receipt().failed_rows() as u64 {
                    return Err(bad("failure count differs"));
                }
                valid = valid
                    .checked_add(reader.receipt().valid_rows() as u64)
                    .ok_or_else(|| bad("valid row overflow"))?;
                failed = failed
                    .checked_add(actual_failed)
                    .ok_or_else(|| bad("failure row overflow"))?;
                let payload = report
                    .get("declared_payload_bytes")
                    .and_then(serde_json::Value::as_u64)
                    .ok_or_else(|| bad("payload bytes absent"))?;
                total_bytes = total_bytes
                    .checked_add(payload)
                    .and_then(|v| v.checked_add(receipt.bytes))
                    .ok_or_else(|| bad("game partial byte overflow"))?;
            }
        }
    }
    let index_bytes = serde_json::to_vec(index).map_err(io::Error::other)?.len() as u64;
    if valid != index.observed_valid_rows
        || failed != index.observed_failed_rows
        || valid
            .checked_add(failed)
            .is_none_or(|n| n > index.max_total_rows)
        || index_bytes > index.index_max_bytes
        || total_bytes
            .checked_add(index_bytes)
            .is_none_or(|n| n > index.max_total_bytes)
    {
        return Err(bad("game partial observed totals or limits differ"));
    }
    Ok(())
}

/// Publish a new index only after the referenced /1 outputs have been checked.
pub fn write_game_partial_index(
    parent: &Dir,
    index: &GamePartialIndexV2,
    source_plan_bytes: &[u8],
    source_spec_bytes: &[u8],
) -> io::Result<()> {
    validate_game_partial_index(parent, index, source_plan_bytes, source_spec_bytes)?;
    let data = serde_json::to_vec(index).map_err(io::Error::other)?;
    if data.len() as u64 > index.index_max_bytes {
        return Err(bad("game partial index byte limit"));
    }
    let mut options = OpenOptions::new();
    options
        .write(true)
        .create_new(true)
        .follow(FollowSymlinks::No);
    let mut file = parent.open_with(INDEX_PENDING, &options)?;
    file.write_all(&data)?;
    file.sync_all()?;
    let directory = crate::pals_collect_v3::training_process::directory_file(parent)
        .map_err(|e| io::Error::other(e.to_string()))?;
    directory.sync_all()?;
    parent.hard_link(INDEX_PENDING, parent, INDEX_FILE)?;
    directory.sync_all()?;
    parent.remove_file(INDEX_PENDING)?;
    directory.sync_all()?;
    Ok(())
}

/// Plan/spec and index declarations only. No referenced /1 root file or
/// original native owner has been checked by this value.
pub struct GamePartialIndexMetadataV2 {
    index: GamePartialIndexV2,
    source_spec: PartialCollectionSpec,
}

impl GamePartialIndexMetadataV2 {
    pub fn index(&self) -> &GamePartialIndexV2 {
        &self.index
    }
    pub fn into_index(self) -> GamePartialIndexV2 {
        self.index
    }
    /// Validate only this slot's declaration against the original spec. This
    /// does not read its files or establish its original authority.
    pub fn validate_root(&self, slot: u16) -> io::Result<()> {
        let entry = self
            .index
            .roots
            .iter()
            .find(|entry| match entry {
                GamePartialRootV2::Available { slot: current, .. }
                | GamePartialRootV2::Unknown { slot: current, .. } => *current == slot,
            })
            .ok_or_else(|| bad("game partial root slot absent"))?;
        validate_root_declaration(&self.index, &self.source_spec, entry)
    }
}

/// Read only the game index and its source declarations. Inspectors must
/// separately verify the selected /1 root, and must not promote metadata to
/// a complete game, corpus, physical or training receipt.
pub fn read_game_partial_index_metadata(
    parent: &Dir,
    max_index_bytes: u64,
    source_plan_bytes: &[u8],
    source_spec_bytes: &[u8],
) -> io::Result<GamePartialIndexMetadataV2> {
    if max_index_bytes == 0 || max_index_bytes > MAX_INDEX_BYTES {
        return Err(bad("game partial read limit"));
    }
    let data = read_file(parent, INDEX_FILE, max_index_bytes)?;
    let text = std::str::from_utf8(&data).map_err(|_| bad("game partial index UTF-8"))?;
    let index: GamePartialIndexV2 =
        rz_experiments::decode_json(text).map_err(|_| bad("game partial index JSON"))?;
    if index.index_max_bytes > max_index_bytes || data.len() as u64 > index.index_max_bytes {
        return Err(bad("game partial declared index limit"));
    }
    let source_spec = validate_game_partial_metadata(&index, source_plan_bytes, source_spec_bytes)?;
    Ok(GamePartialIndexMetadataV2 { index, source_spec })
}

/// Recheck current /1 receipt pins and files; later mutations are not excluded.
pub fn read_game_partial_index(
    parent: &Dir,
    max_index_bytes: u64,
    source_plan_bytes: &[u8],
    source_spec_bytes: &[u8],
) -> io::Result<GamePartialIndexV2> {
    let index = read_game_partial_index_metadata(
        parent,
        max_index_bytes,
        source_plan_bytes,
        source_spec_bytes,
    )?
    .into_index();
    validate_game_partial_index(parent, &index, source_plan_bytes, source_spec_bytes)?;
    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unknown_only() -> GamePartialIndexV2 {
        GamePartialIndexV2 {
            schema: GAME_PARTIAL_INDEX_V2_DOMAIN.into(),
            collection_plan_sha256: "a".repeat(64),
            partial_spec_sha256: "b".repeat(64),
            status: "incomplete".into(),
            training_eligible: false,
            publication_authority: false,
            physical_completion: "not_attested".into(),
            root_capacity: 1,
            max_total_rows: 10,
            max_total_bytes: 8192,
            index_max_bytes: 8192,
            observed_valid_rows: 0,
            observed_failed_rows: 0,
            roots: vec![GamePartialRootV2::Unknown {
                slot: 0,
                root_admission_sha256: None,
                reason: "original root status unavailable".into(),
            }],
        }
    }

    #[test]
    fn unknown_roots_do_not_claim_success_and_count_toward_budget() {
        let mut index = unknown_only();
        assert!(validate_index_shape(&index).is_ok());
        index.physical_completion = "complete".into();
        assert!(validate_index_shape(&index).is_err());
        index.physical_completion = "not_attested".into();
        index.training_eligible = true;
        assert!(validate_index_shape(&index).is_err());
        index.training_eligible = false;
        index.max_total_bytes = 1;
        assert!(validate_index_shape(&index).is_err());
    }

    #[test]
    fn duplicate_root_slots_and_parent_paths_are_rejected() {
        let parent =
            Dir::open_ambient_dir(std::env::temp_dir(), cap_std::ambient_authority()).unwrap();
        let mut index = unknown_only();
        index.roots.push(GamePartialRootV2::Unknown {
            slot: 0,
            root_admission_sha256: Some("b".repeat(64)),
            reason: "unknown".into(),
        });
        assert!(validate_index_shape(&index).is_err());
        assert!(open_relative_directory(&parent, "root/../outside").is_err());
        assert!(open_relative_directory(&parent, "root//other").is_err());
    }

    #[test]
    fn root_declaration_failure_is_scoped_to_its_slot() {
        let mut index = unknown_only();
        index.root_capacity = 2;
        index.roots.push(GamePartialRootV2::Unknown {
            slot: 1,
            root_admission_sha256: None,
            reason: String::new(),
        });
        let source_spec = PartialCollectionSpec {
            schema: super::super::partial_collection_v1::PARTIAL_COLLECTION_SPEC_DOMAIN.into(),
            collection_plan_sha256: index.collection_plan_sha256.clone(),
            row_limit: 10,
            byte_limit: 8192,
            receipt_byte_limit: 8192,
        };
        assert!(validate_index_shape(&index).is_ok());
        assert!(validate_root_declaration(&index, &source_spec, &index.roots[0]).is_ok());
        assert!(validate_root_declaration(&index, &source_spec, &index.roots[1]).is_err());
    }
}
