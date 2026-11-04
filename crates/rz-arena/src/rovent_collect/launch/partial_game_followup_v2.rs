//! Explicit follow-up boundaries for a game partial index.
//!
//! No runtime scheduler invokes this module automatically. Caller-provided
//! isolation, process-tree and physical observations must come from their
//! original supervisors; this API does not create those observations.
//! Only game-index declarations and original plan/spec are checked at open;
//! each root is separately validated when selected. Every visit rechecks the
//! selected root receipt and relevant data against the
//! immutable snapshot. Row callbacks receive the bounded bytes just verified; later
//! filesystem changes and native/GPU owner completion remain separate.
use super::partial_collection_v1::{
    MAX_FAILURE_ROW_BYTES, PartialCollectionReader, PartialCollectionSpec,
};
use super::partial_game_v2::{
    GAME_PARTIAL_EXIT_CODE, GamePartialIndexMetadataV2, GamePartialIndexV2, GamePartialPinV2,
    GamePartialRootV2, read_game_partial_index, read_game_partial_index_metadata,
    verify_failure_root,
};
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::{Dir, OpenOptions};
use sha2::{Digest, Sha256};
use std::{
    cell::{RefCell, RefMut},
    collections::BTreeMap,
    io::{self, BufRead, BufReader, Read, Seek, SeekFrom},
    path::{Component, Path},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GameProducerExitV2 {
    Exited(i32),
    Unknown,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GameProcessTreeV2 {
    GoneObserved,
    Unknown,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GamePhysicalStateV2 {
    CompletionObserved,
    Unknown,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GameResourceScopeV2 {
    IndependentlyIsolated,
    ReusesPrior,
}

/// Per-root readback only; even `Verified` is not owner or physical authority.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum GameRootInspectionStatusV2 {
    Verified {
        slot: u16,
        valid_rows: usize,
        failed_rows: usize,
    },
    Unknown {
        slot: u16,
        reason: String,
    },
    Failed {
        slot: u16,
        error: String,
    },
}

fn bad(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn selected_counts_within_declared(
    declared_valid: u64,
    declared_failed: u64,
    selected_valid: usize,
    selected_failed: usize,
) -> bool {
    u64::try_from(selected_valid).is_ok_and(|n| n <= declared_valid)
        && u64::try_from(selected_failed).is_ok_and(|n| n <= declared_failed)
}
fn resource_allowed(
    scope: GameResourceScopeV2,
    physical: GamePhysicalStateV2,
    tree: GameProcessTreeV2,
) -> bool {
    scope == GameResourceScopeV2::IndependentlyIsolated
        || (physical == GamePhysicalStateV2::CompletionObserved
            && tree == GameProcessTreeV2::GoneObserved)
}

/// An unrelated task can run despite a prior failure only when its resource
/// admission is independent, or the old shared resource and tree are closed.
pub fn run_independent_game_followup<T>(
    scope: GameResourceScopeV2,
    physical: GamePhysicalStateV2,
    tree: GameProcessTreeV2,
    task: impl FnOnce() -> T,
) -> io::Result<T> {
    if !resource_allowed(scope, physical, tree) {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "prior shared resource is not confirmed closed",
        ));
    }
    Ok(task())
}

/// Inspection only. The API intentionally has no strict-corpus or training
/// admission. The borrowed plan and spec bytes must remain the registered
/// originals; their actual SHA and typed contents are checked on each read.
pub struct GamePartialInspectionV2<'a> {
    parent: &'a Dir,
    max_index_bytes: u64,
    source_plan_bytes: &'a [u8],
    source_spec_bytes: &'a [u8],
    metadata: GamePartialIndexMetadataV2,
    roots: RefCell<BTreeMap<u16, RootSnapshot>>,
}

struct RootSnapshot {
    root_admission_sha256: String,
    root: Dir,
    spec: PartialCollectionSpec,
    receipt: GamePartialPinV2,
    reader: PartialCollectionReader,
}

fn verify_root_current(root: &RootSnapshot, wrapped: &serde_json::Value) -> io::Result<()> {
    for chunk in 0..root.reader.chunk_count() {
        root.reader.visit_chunk(chunk, |row| {
            if row
                .0
                .native
                .raw_retirement
                .get("root_admission_sha256")
                .and_then(serde_json::Value::as_str)
                != Some(root.root_admission_sha256.as_str())
            {
                return Err(bad("valid row belongs to another root"));
            }
            Ok(())
        })?;
    }
    if root.reader.receipt().failed_rows() > 0 {
        let pin = wrapped
            .pointer("/receipt/failure_evidence")
            .ok_or_else(|| bad("failure pin missing"))?;
        let partial = root.root.open_dir_nofollow("partial_collection")?;
        if verify_failure_root(
            &partial,
            pin,
            &root.root_admission_sha256,
            root.spec.byte_limit,
        )? != root.reader.receipt().failed_rows() as u64
        {
            return Err(bad("failure count differs"));
        }
    }
    Ok(())
}

impl<'a> GamePartialInspectionV2<'a> {
    pub fn open(
        parent: &'a Dir,
        max_index_bytes: u64,
        source_plan_bytes: &'a [u8],
        source_spec_bytes: &'a [u8],
        producer_exit: GameProducerExitV2,
        tree: GameProcessTreeV2,
        scope: GameResourceScopeV2,
        physical: GamePhysicalStateV2,
    ) -> io::Result<Self> {
        if producer_exit != GameProducerExitV2::Exited(i32::from(GAME_PARTIAL_EXIT_CODE))
            || tree != GameProcessTreeV2::GoneObserved
            || !resource_allowed(scope, physical, tree)
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "game partial producer or resource admission unavailable",
            ));
        }
        let metadata = read_game_partial_index_metadata(
            parent,
            max_index_bytes,
            source_plan_bytes,
            source_spec_bytes,
        )?;
        Ok(Self {
            parent,
            max_index_bytes,
            source_plan_bytes,
            source_spec_bytes,
            metadata,
            roots: RefCell::new(BTreeMap::new()),
        })
    }

    /// Observation at call time only, not a durable assertion about future
    /// filesystem state or a proof of the original opaque in-memory owner.
    pub fn current_index(&self) -> io::Result<GamePartialIndexV2> {
        read_game_partial_index(
            self.parent,
            self.max_index_bytes,
            self.source_plan_bytes,
            self.source_spec_bytes,
        )
    }

    /// The metadata-only immutable index used by subsequent root visits.
    /// Later changes to the on-disk index do not alter this snapshot.
    pub fn snapshot_index(&self) -> &GamePartialIndexV2 {
        self.metadata.index()
    }

    /// Explicitly inspect each root independently. Failure of one root is
    /// recorded in its own status and does not stop verification of others.
    /// This is a point-in-time observation and may be invalidated later.
    pub fn inspect_root_statuses(&self) -> Vec<GameRootInspectionStatusV2> {
        self.metadata
            .index()
            .roots
            .iter()
            .map(|entry| match entry {
                GamePartialRootV2::Unknown { slot, reason, .. } => {
                    match self.metadata.validate_root(*slot) {
                        Ok(()) => GameRootInspectionStatusV2::Unknown {
                            slot: *slot,
                            reason: reason.clone(),
                        },
                        Err(error) => GameRootInspectionStatusV2::Failed {
                            slot: *slot,
                            error: error.to_string().chars().take(512).collect(),
                        },
                    }
                }
                GamePartialRootV2::Available { slot, .. } => match self.root(*slot) {
                    Ok((root, wrapped)) => match verify_root_current(&root, &wrapped) {
                        Ok(()) => GameRootInspectionStatusV2::Verified {
                            slot: *slot,
                            valid_rows: root.reader.receipt().valid_rows(),
                            failed_rows: root.reader.receipt().failed_rows(),
                        },
                        Err(error) => GameRootInspectionStatusV2::Failed {
                            slot: *slot,
                            error: error.to_string().chars().take(512).collect(),
                        },
                    },
                    Err(error) => GameRootInspectionStatusV2::Failed {
                        slot: *slot,
                        error: error.to_string().chars().take(512).collect(),
                    },
                },
            })
            .collect()
    }

    fn root(&self, slot: u16) -> io::Result<(RefMut<'_, RootSnapshot>, serde_json::Value)> {
        let mut roots = self.roots.try_borrow_mut().map_err(|_| {
            io::Error::new(
                io::ErrorKind::WouldBlock,
                "game partial root inspection already active",
            )
        })?;
        if !roots.contains_key(&slot) {
            self.metadata.validate_root(slot)?;
            let entry = self
                .metadata
                .index()
                .roots
                .iter()
                .find(|entry| match entry {
                    GamePartialRootV2::Available {
                        slot: candidate, ..
                    }
                    | GamePartialRootV2::Unknown {
                        slot: candidate, ..
                    } => *candidate == slot,
                })
                .ok_or_else(|| bad("game partial root slot absent"))?;
            let GamePartialRootV2::Available {
                root_admission_sha256,
                relative_path,
                spec,
                receipt,
                ..
            } = entry
            else {
                return Err(bad("game partial root observation is unknown"));
            };
            let root = open_root(self.parent, relative_path)?;
            let reader = PartialCollectionReader::open(&root, spec)?;
            if !selected_counts_within_declared(
                self.metadata.index().observed_valid_rows,
                self.metadata.index().observed_failed_rows,
                reader.receipt().valid_rows(),
                reader.receipt().failed_rows(),
            ) {
                return Err(bad("selected root exceeds game declared row totals"));
            }
            let wrapped = verify_pinned_receipt(&root, spec, receipt, &reader)?;
            let snapshot = RootSnapshot {
                root_admission_sha256: root_admission_sha256.clone(),
                root,
                spec: spec.clone(),
                receipt: receipt.clone(),
                reader,
            };
            verify_root_current(&snapshot, &wrapped)?;
            roots.insert(slot, snapshot);
        }
        let root = roots.get(&slot).expect("root inserted");
        let wrapped = verify_pinned_receipt(&root.root, &root.spec, &root.receipt, &root.reader)?;
        Ok((
            RefMut::map(roots, |roots| roots.get_mut(&slot).expect("root inserted")),
            wrapped,
        ))
    }

    /// Valid rows are inspection-only. The original owner, model eligibility,
    /// GPU fence and training eligibility are not inferred from this callback.
    pub fn visit_valid_chunk(
        &self,
        slot: u16,
        chunk_index: usize,
        mut visit: impl FnMut(
            &rz_experiments::rovent_corpus_lock_v2::record::RoventTrainingRecordLockV2,
        ) -> io::Result<()>,
    ) -> io::Result<usize> {
        let (root, _) = self.root(slot)?;
        if root.reader.receipt().valid_rows() == 0 {
            return Err(bad("game partial root has no valid rows"));
        }
        root.reader.visit_chunk(chunk_index, |row| {
            if row
                .0
                .native
                .raw_retirement
                .get("root_admission_sha256")
                .and_then(serde_json::Value::as_str)
                != Some(root.root_admission_sha256.as_str())
            {
                return Err(bad("game partial valid row root changed"));
            }
            visit(row)
        })
    }

    /// Verify the whole pinned failure stream, then rewind the same open file
    /// and check each line's original hash immediately before its callback.
    /// Callbacks are not transactional: if a later line or callback fails,
    /// earlier callbacks remain the caller's responsibility.
    pub fn visit_failures(
        &self,
        slot: u16,
        mut visit: impl FnMut(&serde_json::Value) -> io::Result<()>,
    ) -> io::Result<usize> {
        let (root, wrapped) = self.root(slot)?;
        if root.reader.receipt().failed_rows() == 0 {
            return Ok(0);
        }
        let pin = wrapped
            .pointer("/receipt/failure_evidence")
            .filter(|v| v.is_object())
            .ok_or_else(|| bad("failure pin missing"))?;
        let bytes = pin
            .get("bytes")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| bad("failure bytes missing"))?;
        let rows = pin
            .get("rows")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| bad("failure row count missing"))?;
        let digest = pin
            .get("sha256")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| bad("failure digest missing"))?;
        if pin.get("path").and_then(serde_json::Value::as_str) != Some("failure-evidence.jsonl")
            || usize::try_from(rows).ok() != Some(root.reader.receipt().failed_rows())
        {
            return Err(bad("failure pin scope differs"));
        }
        let partial = root.root.open_dir_nofollow("partial_collection")?;
        let mut options = OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No);
        let file = partial.open_with("failure-evidence.jsonl", &options)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() || metadata.len() != bytes {
            return Err(bad("failure file changed"));
        }
        let mut input = BufReader::new(file);
        let mut hashes = Vec::<(String, [u8; 32])>::new();
        hashes
            .try_reserve(root.reader.receipt().failed_rows())
            .map_err(|_| io::Error::other("failure line hash allocation"))?;
        let mut ids = std::collections::BTreeSet::new();
        let (mut observed, mut count) = (0u64, 0u64);
        let mut all_hash = Sha256::new();
        loop {
            let mut line = Vec::new();
            let n = (&mut input)
                .take(MAX_FAILURE_ROW_BYTES + 1)
                .read_until(b'\n', &mut line)?;
            if n == 0 {
                break;
            }
            if line.last() != Some(&b'\n') {
                return Err(bad("failure line bound or newline"));
            }
            observed = observed
                .checked_add(n as u64)
                .ok_or_else(|| bad("failure bytes overflow"))?;
            if observed > bytes {
                return Err(bad("failure file grew"));
            }
            count = count
                .checked_add(1)
                .filter(|n| *n <= rows)
                .ok_or_else(|| bad("failure count overflow"))?;
            all_hash.update(&line);
            let text =
                std::str::from_utf8(&line[..line.len() - 1]).map_err(|_| bad("failure UTF-8"))?;
            let row: serde_json::Value =
                rz_experiments::decode_json(text).map_err(|_| bad("failure JSON"))?;
            if row
                .pointer("/original_row/native/raw_retirement/root_admission_sha256")
                .and_then(serde_json::Value::as_str)
                != Some(root.root_admission_sha256.as_str())
            {
                return Err(bad("failure root changed"));
            }
            let id = row
                .get("row_id")
                .and_then(serde_json::Value::as_str)
                .filter(|id| !id.is_empty() && id.len() <= 256 && !id.chars().any(char::is_control))
                .ok_or_else(|| bad("failure id"))?;
            if !ids.insert(id.to_owned()) {
                return Err(bad("duplicate failure id"));
            }
            hashes.push((id.to_owned(), Sha256::digest(&line).into()));
        }
        if observed != bytes || count != rows || format!("{:x}", all_hash.finalize()) != digest {
            return Err(bad("failure evidence changed"));
        }
        input.seek(SeekFrom::Start(0))?;
        let (mut second_bytes, mut second_count) = (0u64, 0usize);
        loop {
            let mut line = Vec::new();
            let n = (&mut input)
                .take(MAX_FAILURE_ROW_BYTES + 1)
                .read_until(b'\n', &mut line)?;
            if n == 0 {
                break;
            }
            if line.last() != Some(&b'\n') {
                return Err(bad("failure line changed"));
            }
            second_bytes = second_bytes
                .checked_add(n as u64)
                .ok_or_else(|| bad("failure bytes overflow"))?;
            if second_bytes > bytes {
                return Err(bad("failure file grew"));
            }
            let (expected_id, expected_hash) = hashes
                .get(second_count)
                .ok_or_else(|| bad("failure row count changed"))?;
            let actual_hash: [u8; 32] = Sha256::digest(&line).into();
            if &actual_hash != expected_hash {
                return Err(bad("failure line changed"));
            }
            let text =
                std::str::from_utf8(&line[..line.len() - 1]).map_err(|_| bad("failure UTF-8"))?;
            let row: serde_json::Value =
                rz_experiments::decode_json(text).map_err(|_| bad("failure JSON"))?;
            if row.get("row_id").and_then(serde_json::Value::as_str) != Some(expected_id.as_str())
                || row
                    .pointer("/original_row/native/raw_retirement/root_admission_sha256")
                    .and_then(serde_json::Value::as_str)
                    != Some(root.root_admission_sha256.as_str())
            {
                return Err(bad("failure identity changed"));
            }
            visit(&row)?;
            second_count += 1;
        }
        if second_bytes != bytes || second_count != hashes.len() {
            return Err(bad("failure stream changed"));
        }
        Ok(second_count)
    }

    /// Read one original failure projection after rechecking the whole pinned
    /// failure stream. A single line is buffered, so large streams do not need
    /// to be retained in memory. No sealed owner is reconstructed.
    pub fn visit_failure(
        &self,
        slot: u16,
        row_id: &str,
        visit: impl FnOnce(&serde_json::Value) -> io::Result<()>,
    ) -> io::Result<bool> {
        if row_id.is_empty() || row_id.len() > 256 || row_id.chars().any(char::is_control) {
            return Err(bad("game partial failure id"));
        }
        let (root, wrapped) = self.root(slot)?;
        if root.reader.receipt().failed_rows() == 0 {
            return Ok(false);
        }
        let pin = wrapped
            .pointer("/receipt/failure_evidence")
            .filter(|v| v.is_object())
            .ok_or_else(|| bad("failure pin missing"))?;
        let bytes = pin
            .get("bytes")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| bad("failure bytes missing"))?;
        let expected_rows = pin
            .get("rows")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| bad("failure row count missing"))?;
        let digest = pin
            .get("sha256")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| bad("failure digest missing"))?;
        if pin.get("path").and_then(serde_json::Value::as_str) != Some("failure-evidence.jsonl") {
            return Err(bad("failure path differs"));
        }
        let partial = root.root.open_dir_nofollow("partial_collection")?;
        let mut options = OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No);
        let file = partial.open_with("failure-evidence.jsonl", &options)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() || metadata.len() != bytes {
            return Err(bad("failure file changed"));
        }
        let mut input = BufReader::new(file.take(bytes.saturating_add(1)));
        let (mut read_bytes, mut count) = (0u64, 0u64);
        let mut digest_actual = Sha256::new();
        let mut selected: Option<Vec<u8>> = None;
        loop {
            let mut line = Vec::new();
            let n = (&mut input)
                .take(MAX_FAILURE_ROW_BYTES + 1)
                .read_until(b'\n', &mut line)?;
            if n == 0 {
                break;
            }
            if line.last() != Some(&b'\n') {
                return Err(bad("failure line bound or newline"));
            }
            read_bytes = read_bytes
                .checked_add(n as u64)
                .ok_or_else(|| bad("failure bytes overflow"))?;
            count = count
                .checked_add(1)
                .ok_or_else(|| bad("failure count overflow"))?;
            digest_actual.update(&line);
            let text =
                std::str::from_utf8(&line[..line.len() - 1]).map_err(|_| bad("failure UTF-8"))?;
            let row: serde_json::Value =
                rz_experiments::decode_json(text).map_err(|_| bad("failure JSON"))?;
            if row
                .pointer("/original_row/native/raw_retirement/root_admission_sha256")
                .and_then(serde_json::Value::as_str)
                != Some(root.root_admission_sha256.as_str())
            {
                return Err(bad("failure root changed"));
            }
            if row.get("row_id").and_then(serde_json::Value::as_str) == Some(row_id) {
                if selected.is_some() {
                    return Err(bad("duplicate failure id"));
                }
                selected = Some(line);
            }
        }
        if read_bytes != bytes
            || count != expected_rows
            || format!("{:x}", digest_actual.finalize()) != digest
        {
            return Err(bad("failure evidence changed"));
        }
        if let Some(line) = selected {
            let text =
                std::str::from_utf8(&line[..line.len() - 1]).map_err(|_| bad("failure UTF-8"))?;
            let row = rz_experiments::decode_json(text).map_err(|_| bad("failure JSON"))?;
            visit(&row)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

fn open_root(parent: &Dir, relative: &str) -> io::Result<Dir> {
    if relative.is_empty()
        || relative.contains(['\\', ':', '\0'])
        || relative
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(bad("game partial root path"));
    }
    let mut root = parent.try_clone()?;
    for component in Path::new(relative).components() {
        match component {
            Component::Normal(name) => root = root.open_dir_nofollow(name)?,
            _ => return Err(bad("game partial root path")),
        }
    }
    Ok(root)
}

fn verify_pinned_receipt(
    root: &Dir,
    spec: &PartialCollectionSpec,
    pin: &GamePartialPinV2,
    reader: &PartialCollectionReader,
) -> io::Result<serde_json::Value> {
    let directory = root.open_dir_nofollow("partial_collection")?;
    let mut options = OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    let file = directory.open_with("partial_collection.json", &options)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() != pin.bytes || pin.bytes > spec.receipt_byte_limit {
        return Err(bad("partial receipt size"));
    }
    let mut bytes = Vec::new();
    file.take(pin.bytes.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 != pin.bytes || format!("{:x}", Sha256::digest(&bytes)) != pin.sha256 {
        return Err(bad("partial receipt pin differs"));
    }
    let text = std::str::from_utf8(&bytes).map_err(|_| bad("partial receipt UTF-8"))?;
    let wrapped: serde_json::Value =
        rz_experiments::decode_json(text).map_err(|_| bad("partial receipt JSON"))?;
    if wrapped.get("spec") != Some(&serde_json::to_value(spec).map_err(io::Error::other)?)
        || wrapped.get("receipt")
            != Some(&serde_json::to_value(reader.receipt()).map_err(io::Error::other)?)
    {
        return Err(bad("partial reader and pinned receipt differ"));
    }
    Ok(wrapped)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selected_root_cannot_exceed_declared_game_counts() {
        assert!(!selected_counts_within_declared(0, 1, 1, 0));
        assert!(!selected_counts_within_declared(2, 0, 1, 1));
        assert!(selected_counts_within_declared(2, 3, 1, 1));
    }
    #[test]
    fn independent_work_does_not_clear_unknown_shared_owner() {
        assert_eq!(
            run_independent_game_followup(
                GameResourceScopeV2::IndependentlyIsolated,
                GamePhysicalStateV2::Unknown,
                GameProcessTreeV2::Unknown,
                || 7
            )
            .unwrap(),
            7
        );
        assert!(
            run_independent_game_followup(
                GameResourceScopeV2::ReusesPrior,
                GamePhysicalStateV2::Unknown,
                GameProcessTreeV2::GoneObserved,
                || panic!("must not run")
            )
            .is_err()
        );
        assert!(
            run_independent_game_followup(
                GameResourceScopeV2::ReusesPrior,
                GamePhysicalStateV2::CompletionObserved,
                GameProcessTreeV2::Unknown,
                || panic!("must not run")
            )
            .is_err()
        );
    }

    #[test]
    fn inspection_rejects_wrong_exit_unfinished_tree_and_unknown_shared_resource() {
        let parent =
            Dir::open_ambient_dir(std::env::temp_dir(), cap_std::ambient_authority()).unwrap();
        assert!(
            GamePartialInspectionV2::open(
                &parent,
                1024,
                b"",
                b"",
                GameProducerExitV2::Exited(0),
                GameProcessTreeV2::GoneObserved,
                GameResourceScopeV2::IndependentlyIsolated,
                GamePhysicalStateV2::Unknown
            )
            .is_err()
        );
        assert!(
            GamePartialInspectionV2::open(
                &parent,
                1024,
                b"",
                b"",
                GameProducerExitV2::Exited(i32::from(GAME_PARTIAL_EXIT_CODE)),
                GameProcessTreeV2::Unknown,
                GameResourceScopeV2::IndependentlyIsolated,
                GamePhysicalStateV2::Unknown
            )
            .is_err()
        );
        assert!(
            GamePartialInspectionV2::open(
                &parent,
                1024,
                b"",
                b"",
                GameProducerExitV2::Exited(i32::from(GAME_PARTIAL_EXIT_CODE)),
                GameProcessTreeV2::GoneObserved,
                GameResourceScopeV2::ReusesPrior,
                GamePhysicalStateV2::Unknown
            )
            .is_err()
        );
    }
}
