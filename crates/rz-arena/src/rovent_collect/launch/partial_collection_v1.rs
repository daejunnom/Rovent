//! Accounting for an explicitly separate, incomplete collection observation.
//!
//! This type does not serialize a corpus, validate a row, persist evidence,
//! transfer ownership, or establish native/physical completion. In particular,
//! the supplied references and byte counts are declarations by the caller;
//! they cannot substitute for the original sealed evidence or its owner.
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
const MAX_RECORDS: usize = 65_536;
const MAX_RECEIPT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ID_BYTES: usize = 256;
const MAX_REFERENCE_BYTES: usize = 4096;

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
    verify_chunks(directory, spec, &stored.receipt)?;
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
    receipt: PartialCollectionReceipt,
}

impl PartialCollectionReader {
    pub fn open(parent: &Dir, spec: &PartialCollectionSpec) -> io::Result<Self> {
        let directory = parent.open_dir_nofollow(DIRECTORY)?;
        let receipt = read_partial_receipt_in(&directory, spec)?;
        Ok(Self { directory, receipt })
    }

    pub fn chunk_count(&self) -> usize {
        self.receipt.chunks.len()
    }

    pub fn receipt(&self) -> &PartialCollectionReceipt {
        &self.receipt
    }

    pub fn visit_chunk(
        &self,
        chunk_index: usize,
        mut visit: impl FnMut(
            &rz_experiments::rovent_corpus_lock_v2::record::RoventTrainingRecordLockV2,
        ) -> io::Result<()>,
    ) -> io::Result<usize> {
        let receipt = &self.receipt;
        let directory = &self.directory;
        let pin = receipt
            .chunks
            .get(chunk_index)
            .ok_or_else(|| invalid_partial("partial chunk index"))?;
        let name = pin
            .path
            .strip_prefix("chunks/")
            .ok_or_else(|| invalid_partial("partial chunk path"))?;
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
        // Preflight all rows before callbacks. Both passes use the same buffer.
        for line in raw.split_inclusive(|b| *b == b'\n') {
            if line.len() > rz_experiments::pals_data_v3::GENERATION_MAX_ROW_BYTES + 1
                || line.last() != Some(&b'\n')
            {
                return Err(invalid_partial("partial row bound or newline"));
            }
            let text = std::str::from_utf8(&line[..line.len() - 1])
                .map_err(|e| io::Error::other(e.to_string()))?;
            let row = rz_experiments::rovent_corpus_lock_v2::record::RoventTrainingRecordLockV2::from_json(text)
            .map_err(|e| io::Error::other(e.to_string()))?;
            let id = &row.0.record_sha256;
            if row.sha256().map_err(|e| io::Error::other(e.to_string()))? != *id {
                return Err(invalid_partial("partial row seal differs"));
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
        if count as u64 != pin.rows {
            return Err(invalid_partial("partial chunk row count"));
        }
        for line in raw.split_inclusive(|b| *b == b'\n') {
            let text = std::str::from_utf8(&line[..line.len() - 1])
                .map_err(|e| io::Error::other(e.to_string()))?;
            let row = rz_experiments::rovent_corpus_lock_v2::record::RoventTrainingRecordLockV2::from_json(text)
            .map_err(|e| io::Error::other(e.to_string()))?;
            visit(&row)?;
        }
        Ok(count)
    }
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
    pub(super) fn with_chunks(mut self, chunks: Vec<PartialChunkPin>) -> Self {
        self.chunks = chunks;
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
                    if evidence_status != "in_memory_custody_not_durable"
                        || *bytes != 0
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
            .filter(|total| {
                ((failure && bytes == 0) || (!failure && bytes > 0)) && *total <= self.byte_limit
            })
            .ok_or(PartialAccountingError::ByteLimit)?;
        let state = if failure {
            RowState::Failed {
                evidence_reference: reference.to_owned(),
                bytes,
                evidence_status: "in_memory_custody_not_durable".into(),
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
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            .record_failure("failed", "in-memory-custody:failed", 0)
            .unwrap();
        let mut receipt = state.finish(PartialEndReason::InputExhausted).ok().unwrap();
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
        state.begin_row("failed").unwrap();
        state
            .record_failure("failed", "in-memory-custody:failed", 0)
            .unwrap();
        let receipt = state.finish(PartialEndReason::InputExhausted).ok().unwrap();
        write_partial_receipt(&output, &spec(), &receipt).unwrap();
        assert!(read_partial_receipt(&parent, &spec()).is_ok());
        assert!(PartialCollectionReader::open(&parent, &spec()).is_ok());
        assert!(!path.join("publication.json").exists());
        assert!(!path.join(DIRECTORY).join("publication.json").exists());
        assert!(write_partial_receipt(&output, &spec(), &receipt).is_err());
        let receipt_path = path.join(DIRECTORY).join(FILE);
        let mut raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&receipt_path).unwrap()).unwrap();
        raw["receipt"]["rows"]["failed"]["policy_target_known"] = true.into();
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
            .record_failure("bad", "in-memory-custody:bad", 0)
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
            "in-memory-custody:bad"
        );
        assert_eq!(
            value["rows"]["bad"]["evidence_status"],
            "in_memory_custody_not_durable"
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
        state.record_failure("one", "evidence", 0).unwrap();
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
            (1, 0, 0)
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
