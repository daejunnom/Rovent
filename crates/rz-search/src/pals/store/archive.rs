//! Atomic, bounded archive storage. Cold records have no growing RAM directory.
//! Lookup scans only this game's committed segments under an explicit I/O budget.
use super::*;
use cap_fs_ext::{DirExt, FollowSymlinks, MetadataExt as _, OpenOptionsFollowExt};
use cap_std::fs::{Dir, OpenOptions as CapabilityOpenOptions};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

mod codec;
mod foreign_permit;
use codec::{Bundle, LineWire, ObservationWire, RoleRecordWire, StateWire};
pub use foreign_permit::{
    ForeignAppendPermit, ForeignGoIssuedProof, ProfiledForeignAppendPermit,
    ProfiledForeignGoIssuedProof,
};
const MAGIC: &[u8; 8] = b"RZPALS01";
const HEADER: u64 = 56;
pub const GAME_ARCHIVE_BYTES: u64 = 256 * 1024 * 1024;
pub const GLOBAL_ARCHIVE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
pub const ARCHIVE_RECORD_PAYLOAD_BYTES: u64 = 16 * 1024 * 1024;
pub const ARCHIVE_MEASUREMENT_CONTRACT: &str = "rz-pals-cold-archive-owner-accounting-v4/1";
pub const ARCHIVE_CAPACITY_METHOD: &str = "hot_unique_owned_capacity_subset";

/// Explicit V4 selectors. The index limits are positive upper bounds on a cold
/// directory that this implementation does not allocate. Load pins count owners
/// of dependency closures, independently of the cold directory's zero entries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveRuntimeLimits {
    pub game_bytes_max: u64,
    pub global_bytes_max: u64,
    pub index_entries_max: u32,
    pub index_bytes_max: u64,
    pub load_bytes_max: u64,
    pub load_deadline_max_ms: u64,
    pub record_payload_bytes_max: u64,
    pub max_load_pins: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ArchiveOwnerLifecycle {
    #[default]
    Open,
    Closed,
    Failed,
}

/// Fixed-size, owner-cumulative facts. Capacity bytes are Rust-owned requested
/// capacities; shared history Arcs, hash/BTree allocator internals and RSS are
/// excluded. None is unknown, never a declared bound or an invented zero.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ArchiveOwnerSnapshot {
    pub measurement_contract: &'static str,
    pub owner_method: &'static str,
    pub compiled_source_sha256: [u8; 32],
    pub root_sha256: Option<[u8; 32]>,
    pub repository_sha256: Option<[u8; 32]>,
    pub owner_id: u64,
    pub event_sequence: u64,
    pub generation: u64,
    pub reserved_generations: u64,
    pub last_committed_generation: Option<u64>,
    pub last_verified_generation: Option<u64>,
    pub lifecycle: ArchiveOwnerLifecycle,
    pub runtime_limits: Option<ArchiveRuntimeLimits>,
    pub root_boundary_checked: bool,
    pub committed_chunks: u64,
    pub integrity_verified_chunks: u64,
    pub committed_bytes: u64,
    pub integrity_verified_bytes: u64,
    pub ram_released_bytes: Option<u64>,
    pub ram_release_method: &'static str,
    pub ram_reclaim_events: u64,
    pub integrity_before_reclaim_events: u64,
    pub unverified_reclaim_events: u64,
    /// Retained failed transaction file extent, including a renamed segment
    /// whose later verification or quota scan failed. This is not RAM usage.
    pub pending_commit_bytes: Option<u64>,
    /// Actual serialized RAM buffers retained by this manager after an operation.
    /// The current synchronous writer retains none; hot source facts are separate.
    pub pending_buffers_retained: bool,
    pub pending_files_retained: bool,
    pub global_managed_bytes: Option<u64>,
    pub global_scan_event_sequence: Option<u64>,
    pub global_scan_complete: bool,
    pub global_scan_after_last_commit: bool,
    pub global_scope_kind: &'static str,
    pub index_entries_peak: u32,
    pub index_bytes_peak: u64,
    pub cold_index_kind: &'static str,
    pub pinned_entries_peak: u32,
    pub load_pins_max: u32,
    pub loaded_closure_bytes_peak: Option<u64>,
    pub loaded_closure_bytes_method: Option<&'static str>,
    pub loads_requested: u64,
    pub loads_completed: u64,
    pub load_bytes_total: u64,
    pub load_bytes_peak: u64,
    pub load_elapsed_peak_ms: u64,
    pub owner_generation_checks: u64,
    pub owner_generation_check_failures: u64,
    pub quota_failures: u64,
    pub io_failures: u64,
    pub pin_saturation_failures: u64,
    pub archive_write_bytes_total: Option<u64>,
    /// Actual process-session authority shared across replacement game owners.
    /// Legacy owners without runtime selectors have no session write ledger.
    pub session_write_ledger_id: Option<u64>,
    pub session_write_bytes_max: Option<u64>,
    pub session_write_bytes_consumed: Option<u64>,
    pub write_bytes_reserved: u64,
    pub write_bytes_remaining: Option<u64>,
    pub admission_closed: bool,
    pub cleanup_complete: bool,
    pub complete: bool,
}

#[derive(Clone, Debug)]
pub struct ArchiveConfig {
    pub root: PathBuf,
    pub repository_root: PathBuf,
    pub game_bytes: u64,
    pub global_bytes: u64,
    pub record_bytes: u64,
    pub max_load_pins: usize,
}
impl ArchiveConfig {
    pub fn new(root: PathBuf, repository_root: PathBuf) -> Self {
        Self {
            root,
            repository_root,
            game_bytes: GAME_ARCHIVE_BYTES,
            global_bytes: GLOBAL_ARCHIVE_BYTES,
            record_bytes: ARCHIVE_RECORD_PAYLOAD_BYTES,
            max_load_pins: 16,
        }
    }
    /// Documented small-game selection. This changes the selected payload bound
    /// to min(16MiB, game quota), never the game/global or I/O allowance.
    pub fn with_quotas(
        root: PathBuf,
        repository_root: PathBuf,
        game_bytes: u64,
        global_bytes: u64,
    ) -> Result<Self, StoreError> {
        if game_bytes == 0
            || game_bytes > GAME_ARCHIVE_BYTES
            || global_bytes == 0
            || global_bytes > GLOBAL_ARCHIVE_BYTES
            || game_bytes > global_bytes
        {
            return Err(StoreError::InvalidConditions("archive quotas"));
        }
        let mut config = Self::new(root, repository_root);
        config.game_bytes = game_bytes;
        config.global_bytes = global_bytes;
        config.record_bytes = ARCHIVE_RECORD_PAYLOAD_BYTES.min(game_bytes);
        Ok(config)
    }
}

/// A startup-owned directory, rather than authority recovered from a pathname.
/// The path is retained only for boundary checks and public path accounting;
/// archive I/O stays relative to the held directory. New game owners share this
/// capability without reopening a replaced ambient parent.
#[derive(Clone, Debug)]
pub struct ArchiveRootCapability {
    root: PathBuf,
    directory: Arc<Dir>,
    identity: ArchiveDirectoryIdentity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ArchiveDirectoryIdentity {
    device: u64,
    inode: u64,
}
impl ArchiveDirectoryIdentity {
    fn capture(directory: &Dir) -> Result<Self, StoreError> {
        // dir_metadata obtains the information from the held directory handle.
        // cap-fs-ext exposes the Windows by-handle identity as well as Unix IDs.
        let metadata = directory
            .dir_metadata()
            .map_err(|error| io("held archive directory identity", error))?;
        if !metadata.is_dir() {
            return Err(StoreError::ArchiveIntegrity(
                "held archive root is not a directory",
            ));
        }
        Ok(Self {
            device: metadata.dev(),
            inode: metadata.ino(),
        })
    }
}
impl ArchiveRootCapability {
    pub fn from_directory(root: PathBuf, directory: Dir) -> Result<Self, StoreError> {
        if !root.is_absolute() {
            return Err(StoreError::InvalidConditions(
                "held archive root must be absolute",
            ));
        }
        for ancestor in root.ancestors() {
            no_link(ancestor)?;
        }
        let root =
            fs::canonicalize(root).map_err(|error| io("held archive root canonicalize", error))?;
        let identity = ArchiveDirectoryIdentity::capture(&directory)?;
        let capability = Self {
            root,
            directory: Arc::new(directory),
            identity,
        };
        capability.verify_named_root()?;
        Ok(capability)
    }

    pub(in crate::pals) fn root_path(&self) -> &Path {
        &self.root
    }

    pub(in crate::pals) fn held_directory(&self) -> &Dir {
        &self.directory
    }

    pub(in crate::pals) fn verify_named_root(&self) -> Result<(), StoreError> {
        for ancestor in self.root.ancestors() {
            no_link(ancestor)?;
        }
        let named = Dir::open_ambient_dir(&self.root, cap_std::ambient_authority())
            .map_err(|error| io("named archive root ownership", error))?;
        if ArchiveDirectoryIdentity::capture(&named)? != self.identity
            || ArchiveDirectoryIdentity::capture(&self.directory)? != self.identity
        {
            return Err(StoreError::ArchiveIntegrity(
                "held archive root was renamed or replaced",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ArchiveIoBudget {
    pub deadline: Instant,
    pub max_bytes: u64,
}
impl ArchiveIoBudget {
    fn check(self) -> Result<(), StoreError> {
        if Instant::now() >= self.deadline {
            Err(StoreError::ArchiveDeadline)
        } else {
            Ok(())
        }
    }
    fn debit(&mut self, bytes: u64) -> Result<(), StoreError> {
        self.check()?;
        self.max_bytes = self
            .max_bytes
            .checked_sub(bytes)
            .ok_or(StoreError::ArchiveByteBudget)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StorePins {
    pub states: BTreeSet<StateId>,
    pub lines: BTreeSet<LineId>,
    pub observations: BTreeSet<ObservationId>,
    pub situations: BTreeSet<SituationId>,
    pub executions: BTreeSet<ExecutionId>,
}
impl StorePins {
    fn extend(&mut self, other: &Self) {
        self.states.extend(&other.states);
        self.lines.extend(&other.lines);
        self.observations.extend(&other.observations);
        self.situations.extend(&other.situations);
        self.executions.extend(&other.executions);
    }
    fn count(&self) -> usize {
        self.states.len()
            + self.lines.len()
            + self.observations.len()
            + self.situations.len()
            + self.executions.len()
    }
}

/// The optional retirement bound applies before allocating a new unique pin.
/// Legacy archive expansion uses the same graph traversal without a new bound.
type PinClosureCheck<'a> = Option<&'a mut dyn FnMut() -> Result<(), StoreError>>;

fn check_pin_closure(check: &mut PinClosureCheck<'_>) -> Result<(), StoreError> {
    match check {
        Some(check) => check(),
        None => Ok(()),
    }
}

fn clone_closure_set<T: Ord + Copy>(
    source: &BTreeSet<T>,
    check: &mut PinClosureCheck<'_>,
) -> Result<BTreeSet<T>, StoreError> {
    check_pin_closure(check)?;
    let copied = if check.is_none() {
        source.clone()
    } else {
        let mut copied = BTreeSet::new();
        for id in source {
            check_pin_closure(check)?;
            copied.insert(*id);
            check_pin_closure(check)?;
        }
        copied
    };
    check_pin_closure(check)?;
    Ok(copied)
}

fn insert_closure_pin<T: Ord>(
    set: &mut BTreeSet<T>,
    value: T,
    count: &mut usize,
    maximum: Option<usize>,
    check: &mut PinClosureCheck<'_>,
) -> Result<(), StoreError> {
    check_pin_closure(check)?;
    if let Some(maximum) = maximum {
        if set.contains(&value) {
            return check_pin_closure(check);
        }
        let next = count
            .checked_add(1)
            .filter(|next| *next <= maximum)
            .ok_or(StoreError::Capacity("retirement dependency closure"))?;
        set.insert(value);
        *count = next;
    } else {
        set.insert(value);
    }
    check_pin_closure(check)
}

/// A handle proves the live archive owner and committed segment generation.
/// The serial ID is the original logical record, never a reused hot slot.
pub struct ColdHandle<T> {
    owner: Weak<()>,
    generation: u64,
    checksum: [u8; 32],
    record: T,
    _kind: PhantomData<T>,
}
impl<T: Copy> Clone for ColdHandle<T> {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner.clone(),
            generation: self.generation,
            checksum: self.checksum,
            record: self.record,
            _kind: PhantomData,
        }
    }
}
impl<T: std::fmt::Debug> std::fmt::Debug for ColdHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColdHandle")
            .field("generation", &self.generation)
            .field("record", &self.record)
            .finish_non_exhaustive()
    }
}
impl<T: Copy> ColdHandle<T> {
    pub fn original_id(&self) -> T {
        self.record
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ArchiveStats {
    pub states: usize,
    pub lines: usize,
    pub situations: usize,
    pub observations: usize,
    pub dependencies: usize,
    pub executions: usize,
    pub consumers: usize,
    pub history_bytes: usize,
}
impl ArchiveStats {
    fn hot(store: &PalsStores) -> Self {
        Self {
            states: store.states.len(),
            lines: store.lines.len(),
            situations: store.situations.len(),
            observations: store.observations.len(),
            dependencies: store.dependencies.len(),
            executions: store.tasks.len(),
            consumers: store.tasks.consumer_count(),
            history_bytes: store.states.retained_history_bytes(),
        }
    }
}
#[derive(Debug)]
pub struct ArchiveReceipt {
    owner: Weak<()>,
    generation: u64,
    checksum: [u8; 32],
    pub before: ArchiveStats,
    pub after: ArchiveStats,
    pub committed_bytes: u64,
    pub io_bytes: u64,
}
impl ArchiveReceipt {
    fn handle<T: Copy>(&self, record: T) -> ColdHandle<T> {
        ColdHandle {
            owner: self.owner.clone(),
            generation: self.generation,
            checksum: self.checksum,
            record,
            _kind: PhantomData,
        }
    }
    pub fn state_handle(&self, id: StateId) -> ColdHandle<StateId> {
        self.handle(id)
    }
    pub fn line_handle(&self, id: LineId) -> ColdHandle<LineId> {
        self.handle(id)
    }
    pub fn observation_handle(&self, id: ObservationId) -> ColdHandle<ObservationId> {
        self.handle(id)
    }
    pub fn execution_handle(&self, id: ExecutionId) -> ColdHandle<ExecutionId> {
        self.handle(id)
    }
    pub fn situation_handle(&self, id: SituationId) -> ColdHandle<SituationId> {
        self.handle(id)
    }
}
#[derive(Debug)]
pub struct ArchiveLoadPin {
    owner: Weak<()>,
    serial: u64,
}
#[derive(Debug)]
pub struct LoadedArchive {
    pub pin: ArchiveLoadPin,
    pub situations: Vec<(SituationId, SituationId)>,
    pub engine_records: Vec<crate::pals::engine::RoleRecord>,
    pub engine_nodes: Vec<EngineArchiveNode>,
    pub io_bytes: u64,
}
/// Ordered edges use original exact-state IDs; metadata is caller-defined,
/// versioned bytes for the remaining engine-owned node fields (maximum 4KiB).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineArchiveNode {
    pub state: StateId,
    pub situation: SituationId,
    pub edges: Vec<(Move16, Option<StateId>)>,
    pub metadata: Vec<u8>,
}
#[derive(Debug)]
pub struct EngineArchiveReceipt {
    pub archive: ArchiveReceipt,
    pub records: usize,
    pub nodes: usize,
    actual_root: Option<(SituationId, StateId, u64)>,
}
impl EngineArchiveReceipt {
    pub(crate) fn actual_root(&self) -> Option<(SituationId, StateId, u64)> {
        self.actual_root
    }
}

mod sealed {
    pub trait Sealed {}
}
/// Sealed record kinds prevent callers from inventing another archive namespace.
pub trait ArchiveRecordKind: sealed::Sealed + Copy {
    #[doc(hidden)]
    fn seed(self, pins: &mut StorePins);
}
macro_rules! kind {
    ($t:ty,$field:ident) => {
        impl sealed::Sealed for $t {}
        impl ArchiveRecordKind for $t {
            fn seed(self, p: &mut StorePins) {
                p.$field.insert(self);
            }
        }
    };
}
kind!(StateId, states);
kind!(LineId, lines);
kind!(ObservationId, observations);
kind!(ExecutionId, executions);
kind!(SituationId, situations);

#[derive(Debug)]
pub(super) struct ArchiveManager {
    config: ArchiveConfig,
    root_capability: Option<ArchiveRootCapability>,
    directory_capability: Option<Arc<Dir>>,
    directory_identity: Option<ArchiveDirectoryIdentity>,
    identity: Arc<()>,
    directory: PathBuf,
    next_generation: u64,
    pins: StorePins,
    load_pins: BTreeMap<u64, StorePins>,
    next_pin: u64,
    auto_budget: Option<ArchiveIoBudget>,
    allocation_active: bool,
    state_owner: Weak<()>,
    line_owner: Weak<()>,
    runtime_limits: Option<ArchiveRuntimeLimits>,
    write_ledger: Option<Arc<ArchiveWriteLedger>>,
    write_budget_inherited: bool,
    used: bool,
    telemetry: std::sync::Mutex<ArchiveOwnerSnapshot>,
    pending_cleanup: bool,
    #[cfg(test)]
    test_commit_fault: Option<TestCommitFault>,
}
#[cfg(test)]
#[derive(Clone, Copy, Debug)]
enum TestCommitFault {
    PartialPayload(usize),
    ExpirePostCommitScan,
}

/// Fixed-size session accounting only. Archive handle ownership, generations,
/// disk quotas and read allowances remain on the independent game manager.
#[derive(Debug)]
pub(super) struct ArchiveWriteLedger {
    id: u64,
    maximum: u64,
    root: PathBuf,
    root_capability: Option<ArchiveRootCapability>,
    runtime_limits: ArchiveRuntimeLimits,
    state: std::sync::Mutex<ArchiveWriteLedgerState>,
}
#[derive(Debug, Default)]
struct ArchiveWriteLedgerState {
    consumed: u64,
    reserved: u64,
    invalid: bool,
}
impl ArchiveWriteLedger {
    fn new(
        root: PathBuf,
        runtime_limits: ArchiveRuntimeLimits,
        root_capability: Option<ArchiveRootCapability>,
    ) -> Result<Arc<Self>, StoreError> {
        static LEDGERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = LEDGERS
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |value| value.checked_add(1),
            )
            .map_err(|_| StoreError::RevisionExhausted)?;
        Ok(Arc::new(Self {
            id,
            maximum: runtime_limits.game_bytes_max,
            root,
            root_capability,
            runtime_limits,
            state: std::sync::Mutex::new(ArchiveWriteLedgerState::default()),
        }))
    }
    fn facts(&self) -> Option<(u64, u64)> {
        let state = self.state.lock().ok()?;
        (!state.invalid).then_some((state.consumed, state.reserved))
    }
    fn remaining(&self) -> Result<u64, StoreError> {
        let (consumed, reserved) = self
            .facts()
            .ok_or(StoreError::ArchiveQuota("archive write ledger unknown"))?;
        consumed
            .checked_add(reserved)
            .and_then(|used| self.maximum.checked_sub(used))
            .ok_or(StoreError::ArchiveQuota("archive lifetime write bytes"))
    }
    fn reserve(self: &Arc<Self>, bytes: u64) -> Result<ArchiveWriteReservation, StoreError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| StoreError::ArchiveQuota("archive write ledger unknown"))?;
        if state.invalid {
            return Err(StoreError::ArchiveQuota("archive write ledger unknown"));
        }
        let reserved = state
            .reserved
            .checked_add(bytes)
            .filter(|reserved| {
                state
                    .consumed
                    .checked_add(*reserved)
                    .is_some_and(|total| total <= self.maximum)
            })
            .ok_or(StoreError::ArchiveQuota("archive lifetime write bytes"))?;
        state.reserved = reserved;
        Ok(ArchiveWriteReservation {
            ledger: Arc::clone(self),
            remaining: bytes,
        })
    }
}
struct ArchiveWriteReservation {
    ledger: Arc<ArchiveWriteLedger>,
    remaining: u64,
}
impl ArchiveWriteReservation {
    fn debit_written(&mut self, bytes: u64) -> std::io::Result<()> {
        let mut state = self
            .ledger
            .state
            .lock()
            .map_err(|_| std::io::Error::other("archive write ledger unknown"))?;
        let debit = self
            .remaining
            .checked_sub(bytes)
            .zip(state.reserved.checked_sub(bytes))
            .zip(state.consumed.checked_add(bytes))
            .filter(|(_, consumed)| *consumed <= self.ledger.maximum);
        let Some(((remaining, reserved), consumed)) = debit else {
            state.invalid = true;
            return Err(std::io::Error::other("archive write ledger debit invalid"));
        };
        self.remaining = remaining;
        state.reserved = reserved;
        state.consumed = consumed;
        Ok(())
    }
}
impl Drop for ArchiveWriteReservation {
    fn drop(&mut self) {
        // Successful physical writes are already consumed. Only the unwritten
        // portion of this reservation is releasable after any failure or unwind.
        let mut state = self.ledger.state.lock().unwrap_or_else(|poison| {
            let mut state = poison.into_inner();
            state.invalid = true;
            state
        });
        if let Some(reserved) = state.reserved.checked_sub(self.remaining) {
            state.reserved = reserved;
        } else {
            state.invalid = true;
        }
    }
}
fn add_counter(value: &mut u64, amount: u64, complete: &mut bool) {
    if let Some(next) = value.checked_add(amount) {
        *value = next;
    } else {
        *complete = false;
    }
}
fn capacity_bytes(capacity: usize, item: usize) -> Option<u64> {
    capacity
        .checked_mul(item)
        .and_then(|bytes| u64::try_from(bytes).ok())
}
fn task_key_capacity(key: &TaskKey) -> Option<u64> {
    capacity_bytes(key.root_moves.capacity(), std::mem::size_of::<Move16>())?.checked_add(
        key.cpu_condition
            .as_ref()
            .map_or(0, |condition| condition.capacity() as u64),
    )
}
fn observation_capacity(observation: &Observation) -> Option<u64> {
    let condition = observation
        .cpu_condition
        .as_ref()
        .map_or(0, |condition| condition.capacity() as u64);
    condition.checked_add(if observation.external_report.is_some() {
        std::mem::size_of::<ExternalCheckerReport>() as u64
    } else {
        0
    })
}
fn task_capacity(task: &TaskRecord) -> Option<u64> {
    task_key_capacity(&task.key)?.checked_add(capacity_bytes(
        task.consumers.capacity(),
        std::mem::size_of::<ConsumerRecord>(),
    )?)
}
/// The literal bytes compiled into this producer, with names and lengths framed
/// to distinguish the Store, archive authority and hot capacity implementation.
pub fn archive_accounting_source_sha256() -> [u8; 32] {
    static DIGEST: std::sync::OnceLock<[u8; 32]> = std::sync::OnceLock::new();
    *DIGEST.get_or_init(|| {
        let mut digest = Sha256::new();
        for (name, source) in [
            ("store.rs", include_str!("../store.rs")),
            ("store/archive.rs", include_str!("archive.rs")),
            ("store/hot.rs", include_str!("hot.rs")),
            (
                "store/archive/foreign_permit.rs",
                include_str!("archive/foreign_permit.rs"),
            ),
        ] {
            digest.update((name.len() as u64).to_le_bytes());
            digest.update(name.as_bytes());
            digest.update((source.len() as u64).to_le_bytes());
            digest.update(source.as_bytes());
        }
        digest.finalize().into()
    })
}
fn io(stage: &'static str, error: std::io::Error) -> StoreError {
    StoreError::ArchiveIo {
        stage,
        kind: error.kind(),
    }
}
fn no_link(path: &Path) -> Result<(), StoreError> {
    let metadata = fs::symlink_metadata(path).map_err(|e| io("archive path metadata", e))?;
    if metadata.file_type().is_symlink() {
        return Err(StoreError::ArchiveIntegrity("archive symbolic link"));
    }
    // Windows junctions are reparse points as well as ordinary symlinks.
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        if metadata.file_attributes() & 0x400 != 0 {
            return Err(StoreError::ArchiveIntegrity("archive reparse point"));
        }
    }
    Ok(())
}

// A streaming facade keeps legacy std::fs behaviour while the opted-in owner
// never turns a display path back into I/O authority. It does not build a RAM
// directory proportional to archived games or segments.
enum ArchiveEntries {
    Ambient(fs::ReadDir),
    Held {
        entries: cap_std::fs::ReadDir,
        parent: Arc<Dir>,
        display: PathBuf,
    },
}
enum ArchiveEntry {
    Ambient(fs::DirEntry),
    Held {
        entry: cap_std::fs::DirEntry,
        parent: Arc<Dir>,
        display: PathBuf,
    },
}
struct ArchiveEntryMetadata {
    bytes: u64,
    file: bool,
}
impl ArchiveEntryMetadata {
    fn len(&self) -> u64 {
        self.bytes
    }
    fn is_file(&self) -> bool {
        self.file
    }
}
struct ArchiveEntryType {
    file: bool,
    directory: bool,
}
impl ArchiveEntryType {
    fn is_file(&self) -> bool {
        self.file
    }
    fn is_dir(&self) -> bool {
        self.directory
    }
}
impl ArchiveEntries {
    fn held(parent: Arc<Dir>, display: PathBuf) -> std::io::Result<Self> {
        Ok(Self::Held {
            entries: parent.entries()?,
            parent,
            display,
        })
    }
}
impl Iterator for ArchiveEntries {
    type Item = std::io::Result<ArchiveEntry>;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Ambient(entries) => entries.next().map(|item| item.map(ArchiveEntry::Ambient)),
            Self::Held {
                entries,
                parent,
                display,
            } => entries.next().map(|item| {
                item.map(|entry| {
                    let path = display.join(entry.file_name());
                    ArchiveEntry::Held {
                        entry,
                        parent: Arc::clone(parent),
                        display: path,
                    }
                })
            }),
        }
    }
}
impl ArchiveEntry {
    fn file_name(&self) -> std::ffi::OsString {
        match self {
            Self::Ambient(entry) => entry.file_name(),
            Self::Held { entry, .. } => entry.file_name(),
        }
    }
    fn path(&self) -> PathBuf {
        match self {
            Self::Ambient(entry) => entry.path(),
            Self::Held { display, .. } => display.clone(),
        }
    }
    fn file_type(&self) -> std::io::Result<ArchiveEntryType> {
        match self {
            Self::Ambient(entry) => entry.file_type().map(|kind| ArchiveEntryType {
                file: kind.is_file(),
                directory: kind.is_dir(),
            }),
            Self::Held { entry, .. } => entry.file_type().map(|kind| ArchiveEntryType {
                file: kind.is_file(),
                directory: kind.is_dir(),
            }),
        }
    }
    fn metadata(&self) -> std::io::Result<ArchiveEntryMetadata> {
        match self {
            Self::Ambient(entry) => entry.metadata().map(|metadata| ArchiveEntryMetadata {
                bytes: metadata.len(),
                file: metadata.is_file(),
            }),
            Self::Held { entry, .. } => entry.metadata().map(|metadata| ArchiveEntryMetadata {
                bytes: metadata.len(),
                file: metadata.is_file(),
            }),
        }
    }
    fn no_link(&self) -> Result<(), StoreError> {
        match self {
            Self::Ambient(entry) => no_link(&entry.path()),
            Self::Held { entry, parent, .. } => {
                let metadata = parent
                    .symlink_metadata(entry.file_name())
                    .map_err(|e| io("held archive entry metadata", e))?;
                if metadata.is_symlink() {
                    return Err(StoreError::ArchiveIntegrity("archive symbolic link"));
                }
                #[cfg(windows)]
                {
                    use cap_std::fs::MetadataExt;
                    if metadata.file_attributes() & 0x400 != 0 {
                        return Err(StoreError::ArchiveIntegrity("archive reparse point"));
                    }
                }
                Ok(())
            }
        }
    }
    fn entries(&self) -> std::io::Result<ArchiveEntries> {
        match self {
            Self::Ambient(entry) => fs::read_dir(entry.path()).map(ArchiveEntries::Ambient),
            Self::Held {
                entry,
                parent,
                display,
            } => {
                let directory = parent.open_dir_nofollow(entry.file_name())?;
                ArchiveEntries::held(Arc::new(directory), display.clone())
            }
        }
    }
}
impl ArchiveManager {
    fn open(config: ArchiveConfig) -> Result<Self, StoreError> {
        Self::open_inner(config, None)
    }
    fn open_with_capability(
        config: ArchiveConfig,
        capability: ArchiveRootCapability,
    ) -> Result<Self, StoreError> {
        capability.verify_named_root()?;
        Self::open_inner(config, Some(capability))
    }
    fn open_inner(
        config: ArchiveConfig,
        capability: Option<ArchiveRootCapability>,
    ) -> Result<Self, StoreError> {
        if !config.root.is_absolute()
            || !config.repository_root.is_absolute()
            || config.game_bytes == 0
            || config.global_bytes == 0
            || config.game_bytes > GAME_ARCHIVE_BYTES
            || config.global_bytes > GLOBAL_ARCHIVE_BYTES
            || config.game_bytes > config.global_bytes
            || config.record_bytes == 0
            || config.record_bytes > config.game_bytes
            || config.max_load_pins == 0
            || config.max_load_pins > 16
        {
            return Err(StoreError::InvalidConditions("archive path or quotas"));
        }
        let repository =
            fs::canonicalize(&config.repository_root).map_err(|e| io("repository path", e))?;
        let mut ancestor = config.root.as_path();
        while !ancestor.exists() {
            ancestor = ancestor
                .parent()
                .ok_or(StoreError::InvalidConditions("archive root"))?;
        }
        for path in ancestor.ancestors() {
            no_link(path)?;
        }
        let ancestor = fs::canonicalize(ancestor).map_err(|e| io("archive ancestor", e))?;
        if ancestor.starts_with(&repository) {
            return Err(StoreError::InvalidConditions(
                "archive must be outside repository",
            ));
        }
        if let Some(capability) = &capability {
            capability.verify_named_root()?;
            if fs::canonicalize(&config.root).map_err(|e| io("archive root canonicalize", e))?
                != capability.root
            {
                return Err(StoreError::ArchiveIntegrity(
                    "archive configuration differs from held root",
                ));
            }
        } else {
            fs::create_dir_all(&config.root).map_err(|e| io("archive root create", e))?;
        }
        no_link(&config.root)?;
        let root =
            fs::canonicalize(&config.root).map_err(|e| io("archive root canonicalize", e))?;
        if root.starts_with(&repository) {
            return Err(StoreError::InvalidConditions(
                "archive must be outside repository",
            ));
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| StoreError::InvalidConditions("archive clock"))?
            .as_nanos();
        static OWNERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let serial = OWNERS
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |value| value.checked_add(1),
            )
            .map_err(|_| StoreError::RevisionExhausted)?;
        let game_name = format!("game-{now}-{}-{serial}", std::process::id());
        let directory = root.join(&game_name);
        let (directory_capability, directory_identity) = if let Some(capability) = &capability {
            capability.verify_named_root()?;
            capability
                .directory
                .create_dir(&game_name)
                .map_err(|e| io("held archive game create", e))?;
            let game = capability
                .directory
                .open_dir_nofollow(&game_name)
                .map_err(|e| io("held archive game open", e))?;
            let identity = ArchiveDirectoryIdentity::capture(&game)?;
            (Some(Arc::new(game)), Some(identity))
        } else {
            fs::create_dir(&directory).map_err(|e| io("archive game create", e))?;
            (None, None)
        };
        let manager = Self {
            telemetry: std::sync::Mutex::new(ArchiveOwnerSnapshot {
                measurement_contract: ARCHIVE_MEASUREMENT_CONTRACT,
                owner_method: "archive_actual_owner_events_v4",
                compiled_source_sha256: archive_accounting_source_sha256(),
                root_sha256: root
                    .to_str()
                    .map(|path| Sha256::digest(path.as_bytes()).into()),
                repository_sha256: repository
                    .to_str()
                    .map(|path| Sha256::digest(path.as_bytes()).into()),
                owner_id: serial,
                event_sequence: 1,
                root_boundary_checked: true,
                ram_released_bytes: Some(0),
                ram_release_method: ARCHIVE_CAPACITY_METHOD,
                pending_commit_bytes: Some(0),
                pending_buffers_retained: false,
                global_scope_kind: "canonical_no_link_managed_root",
                cold_index_kind: "directory_scan",
                load_pins_max: config.max_load_pins as u32,
                loaded_closure_bytes_peak: Some(0),
                loaded_closure_bytes_method: Some(ARCHIVE_CAPACITY_METHOD),
                archive_write_bytes_total: Some(0),
                complete: true,
                ..ArchiveOwnerSnapshot::default()
            }),
            config: ArchiveConfig {
                root,
                repository_root: repository,
                ..config
            },
            root_capability: capability,
            directory_capability,
            directory_identity,
            identity: Arc::new(()),
            directory,
            next_generation: 0,
            pins: StorePins::default(),
            load_pins: BTreeMap::new(),
            next_pin: 0,
            auto_budget: None,
            allocation_active: false,
            state_owner: Weak::new(),
            line_owner: Weak::new(),
            runtime_limits: None,
            write_ledger: None,
            write_budget_inherited: false,
            used: false,
            pending_cleanup: false,
            #[cfg(test)]
            test_commit_fault: None,
        };
        // An ambient parent replacement during owner opening cannot silently
        // publish an owner for the replacement. Any created game remains in the
        // held directory as failure evidence; it is never redirected or erased.
        manager.verify_directory_boundary()?;
        Ok(manager)
    }
    fn verify_directory_boundary(&self) -> Result<(), StoreError> {
        let Some(root) = &self.root_capability else {
            return Ok(());
        };
        root.verify_named_root()?;
        let name = self
            .directory
            .file_name()
            .ok_or(StoreError::ArchiveIntegrity("archive game name"))?;
        let named = root
            .directory
            .open_dir_nofollow(name)
            .map_err(|e| io("named archive game ownership", e))?;
        let held = self
            .directory_capability
            .as_ref()
            .ok_or(StoreError::ArchiveIntegrity(
                "held archive game capability missing",
            ))?;
        let expected = self.directory_identity.ok_or(StoreError::ArchiveIntegrity(
            "held archive game identity missing",
        ))?;
        if ArchiveDirectoryIdentity::capture(&named)? != expected
            || ArchiveDirectoryIdentity::capture(held)? != expected
        {
            return Err(StoreError::ArchiveIntegrity(
                "held archive game was renamed or replaced",
            ));
        }
        Ok(())
    }
    fn root_entries(&self) -> std::io::Result<ArchiveEntries> {
        match &self.root_capability {
            Some(root) => {
                ArchiveEntries::held(Arc::clone(&root.directory), self.config.root.clone())
            }
            None => fs::read_dir(&self.config.root).map(ArchiveEntries::Ambient),
        }
    }
    fn game_entries(&self) -> std::io::Result<ArchiveEntries> {
        match &self.directory_capability {
            Some(directory) => ArchiveEntries::held(Arc::clone(directory), self.directory.clone()),
            None => fs::read_dir(&self.directory).map(ArchiveEntries::Ambient),
        }
    }
    fn game_file(&self, name: &std::ffi::OsStr, create_new: bool) -> std::io::Result<File> {
        match &self.directory_capability {
            Some(directory) => {
                let mut options = CapabilityOpenOptions::new();
                options.follow(FollowSymlinks::No);
                #[cfg(target_os = "linux")]
                {
                    use cap_std::fs::OpenOptionsExt;
                    // Check the actual descriptor type after opening without
                    // blocking forever on a substituted FIFO/device pathname.
                    options.custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK);
                }
                if create_new {
                    options.write(true).create_new(true);
                } else {
                    options.read(true);
                }
                directory
                    .open_with(Path::new(name), &options)
                    .map(|file| file.into_std())
            }
            None if create_new => OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(self.directory.join(name)),
            None => File::open(self.directory.join(name)),
        }
    }
    fn game_file_size(&self, name: &std::ffi::OsStr) -> Result<u64, StoreError> {
        match &self.directory_capability {
            Some(directory) => directory
                .symlink_metadata(Path::new(name))
                .map_err(|e| io("held archive extent", e))
                .and_then(|metadata| {
                    if !metadata.is_file() || metadata.is_symlink() {
                        return Err(StoreError::ArchiveIntegrity("archive record type"));
                    }
                    Ok(metadata.len())
                }),
            None => fs::metadata(self.directory.join(name))
                .map(|metadata| metadata.len())
                .map_err(|e| io("engine archive committed size", e)),
        }
    }
    fn rename_game_file(
        &self,
        source: &std::ffi::OsStr,
        target: &std::ffi::OsStr,
    ) -> std::io::Result<()> {
        match &self.directory_capability {
            Some(directory) => directory.rename(Path::new(source), directory, Path::new(target)),
            None => fs::rename(self.directory.join(source), self.directory.join(target)),
        }
    }
    fn remove_game_file(&self, name: &std::ffi::OsStr) -> std::io::Result<()> {
        match &self.directory_capability {
            Some(directory) => directory.remove_file(Path::new(name)),
            None => fs::remove_file(self.directory.join(name)),
        }
    }
    #[cfg(unix)]
    fn sync_game_directory(&self) -> std::io::Result<()> {
        match &self.directory_capability {
            Some(directory) => {
                let mut options = CapabilityOpenOptions::new();
                options.read(true).follow(FollowSymlinks::No);
                #[cfg(target_os = "linux")]
                {
                    use cap_std::fs::OpenOptionsExt;
                    options.custom_flags(
                        nix::libc::O_DIRECTORY | nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK,
                    );
                }
                directory.open_with(".", &options)?.into_std().sync_all()
            }
            None => File::open(&self.directory)?.sync_all(),
        }
    }
    fn note(&self, event: impl FnOnce(&mut ArchiveOwnerSnapshot)) {
        let mut meter = self.telemetry.lock().unwrap_or_else(|poison| {
            let mut meter = poison.into_inner();
            meter.complete = false;
            meter
        });
        let meter = &mut *meter;
        add_counter(&mut meter.event_sequence, 1, &mut meter.complete);
        event(meter);
    }
    fn failure(&self, error: &StoreError) {
        self.note(|meter| {
            meter.lifecycle = ArchiveOwnerLifecycle::Failed;
            match error {
                StoreError::ArchiveQuota(_) => {
                    add_counter(&mut meter.quota_failures, 1, &mut meter.complete)
                }
                StoreError::ArchiveIo { .. } | StoreError::ArchiveIntegrity(_) => {
                    add_counter(&mut meter.io_failures, 1, &mut meter.complete)
                }
                StoreError::PinSaturated(_) => {
                    add_counter(&mut meter.pin_saturation_failures, 1, &mut meter.complete)
                }
                _ => {}
            }
        });
    }
    fn failure_after(&self, before: ArchiveOwnerSnapshot, error: &StoreError) {
        let current = self.snapshot();
        let already_recorded = match error {
            StoreError::ArchiveIo { .. } | StoreError::ArchiveIntegrity(_) => {
                current.io_failures != before.io_failures
            }
            StoreError::ArchiveQuota(_) => current.quota_failures != before.quota_failures,
            StoreError::PinSaturated(_) => {
                current.pin_saturation_failures != before.pin_saturation_failures
            }
            _ => false,
        };
        if !already_recorded {
            self.failure(error);
        }
    }
    fn finish_read<T>(
        &self,
        before: ArchiveOwnerSnapshot,
        started: Instant,
        bytes: u64,
        load: bool,
        result: &Result<T, StoreError>,
    ) {
        self.note(|meter| {
            add_counter(&mut meter.load_bytes_total, bytes, &mut meter.complete);
            meter.load_bytes_peak = meter.load_bytes_peak.max(bytes);
            meter.load_elapsed_peak_ms = meter
                .load_elapsed_peak_ms
                .max(u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX));
            if load && result.is_ok() {
                add_counter(&mut meter.loads_completed, 1, &mut meter.complete);
            }
        });
        if let Err(error) = result {
            self.failure_after(before, error);
        }
    }
    fn snapshot(&self) -> ArchiveOwnerSnapshot {
        let meter = self
            .telemetry
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut snapshot = *meter;
        snapshot.generation = self.next_generation;
        snapshot.reserved_generations = self.next_generation;
        snapshot.runtime_limits = self.runtime_limits;
        if let Some(ledger) = &self.write_ledger {
            let facts = ledger.facts();
            snapshot.session_write_ledger_id = Some(ledger.id);
            snapshot.session_write_bytes_max = Some(ledger.maximum);
            snapshot.session_write_bytes_consumed = facts.map(|facts| facts.0);
            snapshot.write_bytes_remaining = facts.and_then(|(consumed, reserved)| {
                consumed
                    .checked_add(reserved)
                    .and_then(|used| ledger.maximum.checked_sub(used))
            });
            snapshot.complete &= snapshot.session_write_bytes_consumed.is_some()
                && snapshot.write_bytes_remaining.is_some();
        }
        snapshot.complete &= !self.telemetry.is_poisoned()
            && snapshot.global_scan_complete
            && snapshot.global_scan_after_last_commit
            && snapshot.global_managed_bytes.is_some()
            && snapshot.pending_commit_bytes.is_some()
            && snapshot.ram_released_bytes.is_some()
            && snapshot.archive_write_bytes_total.is_some()
            && snapshot.loaded_closure_bytes_peak.is_some()
            && snapshot.loaded_closure_bytes_method.is_some()
            && !snapshot.pending_files_retained
            && !snapshot.pending_buffers_retained
            && snapshot.root_sha256.is_some()
            && snapshot.repository_sha256.is_some();
        snapshot
    }
    fn load_budget(&self, original: ArchiveIoBudget) -> Result<ArchiveIoBudget, StoreError> {
        original.check()?;
        let Some(limits) = self.runtime_limits else {
            return Ok(original);
        };
        let maximum_deadline = Instant::now()
            .checked_add(std::time::Duration::from_millis(
                limits.load_deadline_max_ms,
            ))
            .ok_or(StoreError::InvalidConditions("archive load duration"))?;
        Ok(ArchiveIoBudget {
            deadline: original.deadline.min(maximum_deadline),
            max_bytes: original.max_bytes.min(limits.load_bytes_max),
        })
    }
    fn path(&self, generation: u64) -> PathBuf {
        self.directory.join(format!("seg-{generation:020}.arc"))
    }
    fn verify_owner<T>(&self, h: &ColdHandle<T>) -> Result<(), StoreError> {
        self.note(|meter| add_counter(&mut meter.owner_generation_checks, 1, &mut meter.complete));
        if !h.owner.ptr_eq(&Arc::downgrade(&self.identity)) || h.generation >= self.next_generation
        {
            self.note(|meter| {
                add_counter(
                    &mut meter.owner_generation_check_failures,
                    1,
                    &mut meter.complete,
                )
            });
            return Err(StoreError::InvalidHandle("cold owner or generation"));
        }
        Ok(())
    }
    fn verify_hot_owner(&self, store: &PalsStores) -> Result<(), StoreError> {
        if !self
            .state_owner
            .ptr_eq(&Arc::downgrade(&store.states.identity))
            || !self
                .line_owner
                .ptr_eq(&Arc::downgrade(&store.lines.identity))
        {
            return Err(StoreError::InvalidHandle("archive hot-store owner changed"));
        }
        Ok(())
    }
    fn usage(&self, budget: ArchiveIoBudget) -> Result<(u64, u64), StoreError> {
        let result = self.usage_inner(budget);
        self.note(|meter| match result {
            Ok((_, total)) => {
                meter.global_managed_bytes = Some(total);
                meter.global_scan_event_sequence = Some(meter.event_sequence);
                meter.global_scan_complete = true;
                meter.global_scan_after_last_commit = true;
            }
            Err(_) => {
                meter.global_scan_complete = false;
                meter.global_scan_after_last_commit = false;
            }
        });
        result
    }
    fn usage_inner(&self, budget: ArchiveIoBudget) -> Result<(u64, u64), StoreError> {
        budget.check()?;
        self.verify_directory_boundary()?;
        let mut total = 0u64;
        let mut game = 0u64;
        for entry in self
            .root_entries()
            .map_err(|e| io("global archive directory", e))?
        {
            budget.check()?;
            let entry = entry.map_err(|e| io("global archive entry", e))?;
            let path = entry.path();
            entry.no_link()?;
            if entry.file_name() == ".quota.lock" {
                if !entry
                    .file_type()
                    .map_err(|e| io("archive lock type", e))?
                    .is_file()
                {
                    return Err(StoreError::ArchiveIntegrity("archive quota lock type"));
                }
                continue;
            }
            if !entry
                .file_type()
                .map_err(|e| io("global archive type", e))?
                .is_dir()
                || !entry.file_name().to_string_lossy().starts_with("game-")
            {
                return Err(StoreError::ArchiveIntegrity(
                    "archive root is not exclusively managed",
                ));
            }
            for record in entry
                .entries()
                .map_err(|e| io("game archive directory", e))?
            {
                budget.check()?;
                let record = record.map_err(|e| io("game archive entry", e))?;
                record.no_link()?;
                let metadata = record.metadata().map_err(|e| io("archive size", e))?;
                if !metadata.is_file() {
                    return Err(StoreError::ArchiveIntegrity("archive record type"));
                }
                total = total
                    .checked_add(metadata.len())
                    .ok_or(StoreError::ArchiveQuota("global archive overflow"))?;
                if path == self.directory {
                    game = game
                        .checked_add(metadata.len())
                        .ok_or(StoreError::ArchiveQuota("game archive overflow"))?;
                }
            }
        }
        budget.check()?;
        self.verify_directory_boundary()?;
        Ok((game, total))
    }
    fn commit(
        &mut self,
        bundle: &Bundle,
        budget: &mut ArchiveIoBudget,
        before: ArchiveStats,
    ) -> Result<ArchiveReceipt, StoreError> {
        self.used = true;
        if self.snapshot().admission_closed {
            return Err(StoreError::InvalidConditions(
                "archive owner admission closed",
            ));
        }
        let baseline = self.snapshot();
        let result = self.commit_inner(bundle, budget, before);
        if let Err(error) = &result {
            self.failure_after(baseline, error);
        }
        result
    }
    fn commit_inner(
        &mut self,
        bundle: &Bundle,
        budget: &mut ArchiveIoBudget,
        before: ArchiveStats,
    ) -> Result<ArchiveReceipt, StoreError> {
        budget.check()?;
        self.verify_directory_boundary()?;
        let next = self
            .next_generation
            .checked_add(1)
            .ok_or(StoreError::RevisionExhausted)?;
        let lock = QuotaLock::take(&self.config.root, self.root_capability.as_ref())?;
        let (game, total) = self.usage(*budget)?;
        let remaining = self
            .config
            .game_bytes
            .checked_sub(game)
            .ok_or(StoreError::ArchiveQuota("game archive"))?
            .min(
                self.config
                    .global_bytes
                    .checked_sub(total)
                    .ok_or(StoreError::ArchiveQuota("global archive"))?,
            );
        let maximum = self
            .config
            .record_bytes
            .min(remaining.saturating_sub(HEADER))
            .min(budget.max_bytes.saturating_sub(2 * HEADER) / 2);
        let write_remaining = if self.runtime_limits.is_some() {
            Some(
                self.write_ledger
                    .as_ref()
                    .ok_or(StoreError::ArchiveQuota("archive write ledger missing"))?
                    .remaining()?,
            )
        } else {
            None
        };
        let maximum =
            maximum.min(write_remaining.map_or(u64::MAX, |bytes| bytes.saturating_sub(HEADER)));
        if maximum == 0 {
            return Err(
                if write_remaining.is_some_and(|remaining| remaining <= HEADER) {
                    StoreError::ArchiveQuota("archive lifetime write bytes")
                } else if remaining <= HEADER {
                    StoreError::ArchiveQuota("archive disk bytes")
                } else {
                    StoreError::ArchiveByteBudget
                },
            );
        }
        let mut writer = BoundedBuffer {
            bytes: Vec::new(),
            maximum,
            deadline: budget.deadline,
            error: None,
        };
        if serde_json::to_writer(&mut writer, bundle).is_err() {
            let error = writer
                .error
                .unwrap_or(StoreError::ArchiveIntegrity("archive encode"));
            return Err(
                if error == StoreError::ArchiveByteBudget
                    && write_remaining.is_some_and(|bytes| maximum == bytes.saturating_sub(HEADER))
                {
                    StoreError::ArchiveQuota("archive lifetime write bytes")
                } else if error == StoreError::ArchiveByteBudget
                    && maximum == remaining.saturating_sub(HEADER)
                {
                    StoreError::ArchiveQuota(
                        if self.config.game_bytes.saturating_sub(game)
                            <= self.config.global_bytes.saturating_sub(total)
                        {
                            "game archive"
                        } else {
                            "global archive"
                        },
                    )
                } else if error == StoreError::ArchiveByteBudget
                    && maximum == self.config.record_bytes
                {
                    StoreError::ArchiveQuota("archive segment bytes")
                } else {
                    error
                },
            );
        }
        let bytes = writer.bytes;
        let extent = HEADER + bytes.len() as u64;
        // Serialization may run concurrently with another inherited owner.
        // Reserve the whole extent atomically before creating or writing a file.
        let mut write_reservation = self
            .write_ledger
            .as_ref()
            .map(|ledger| ledger.reserve(extent))
            .transpose()?;
        let checksum: [u8; 32] = Sha256::digest(&bytes).into();
        let generation = self.next_generation;
        // Failed verification can leave a committed file. Reserve its number
        // before writing so a retry cannot overwrite retained evidence.
        self.next_generation = next;
        self.note(|meter| {
            meter.reserved_generations = next;
            meter.generation = next;
            meter.write_bytes_reserved = extent;
            meter.pending_commit_bytes = Some(extent);
        });
        let temporary = self.directory.join(format!("seg-{generation:020}.pending"));
        let final_path = self.path(generation);
        let file = self
            .game_file(temporary.file_name().expect("private segment name"), true)
            .map_err(|e| io("archive temporary create", e));
        let mut file = match file {
            Ok(file) => Some(file),
            Err(error) => {
                self.note(|meter| {
                    meter.write_bytes_reserved = 0;
                    meter.pending_commit_bytes = Some(0);
                });
                return Err(error);
            }
        };
        self.pending_cleanup = self.runtime_limits.is_some();
        self.note(|meter| meter.global_scan_after_last_commit = false);
        let mut written = 0u64;
        let mut renamed = false;
        let result = (|| {
            budget.debit(extent)?;
            let mut output = CountedWriter {
                inner: file.as_mut().expect("owned file"),
                written: &mut written,
                reservation: write_reservation.as_mut(),
            };
            output
                .write_all(MAGIC)
                .and_then(|_| output.write_all(&generation.to_le_bytes()))
                .and_then(|_| output.write_all(&(bytes.len() as u64).to_le_bytes()))
                .and_then(|_| output.write_all(&checksum))
                .map_err(|e| io("archive header write", e))?;
            #[cfg(test)]
            if let Some(TestCommitFault::PartialPayload(count)) = self.test_commit_fault {
                self.test_commit_fault = None;
                output
                    .write_all(&bytes[..count.min(bytes.len())])
                    .map_err(|e| io("archive partial payload write", e))?;
                return Err(io(
                    "injected archive partial write",
                    std::io::Error::other("fixture"),
                ));
            }
            output
                .write_all(&bytes)
                .and_then(|_| output.inner.sync_all())
                .map_err(|e| io("archive write and sync", e))?;
            budget.check()?;
            drop(file.take());
            self.verify_directory_boundary()?;
            self.rename_game_file(
                temporary.file_name().expect("private segment name"),
                final_path.file_name().expect("private segment name"),
            )
            .map_err(|e| io("archive atomic commit", e))?;
            renamed = true;
            self.note(|meter| {
                add_counter(&mut meter.committed_chunks, 1, &mut meter.complete);
                add_counter(&mut meter.committed_bytes, extent, &mut meter.complete);
                meter.last_committed_generation = Some(generation);
                meter.global_scan_after_last_commit = false;
            });
            #[cfg(unix)]
            self.sync_game_directory()
                .map_err(|e| io("archive directory sync", e))?;
            self.read(generation, Some(checksum), budget)?;
            self.note(|meter| {
                add_counter(&mut meter.integrity_verified_chunks, 1, &mut meter.complete);
                add_counter(
                    &mut meter.integrity_verified_bytes,
                    extent,
                    &mut meter.complete,
                );
                meter.last_verified_generation = Some(generation);
            });
            if self.runtime_limits.is_some() {
                let post_commit_budget = *budget;
                #[cfg(test)]
                let post_commit_budget = if matches!(
                    self.test_commit_fault,
                    Some(TestCommitFault::ExpirePostCommitScan)
                ) {
                    self.test_commit_fault = None;
                    ArchiveIoBudget {
                        deadline: Instant::now(),
                        ..post_commit_budget
                    }
                } else {
                    post_commit_budget
                };
                let (game, global) = self.usage(post_commit_budget)?;
                if game > self.config.game_bytes || global > self.config.global_bytes {
                    return Err(StoreError::ArchiveQuota(
                        "post-commit managed archive bytes",
                    ));
                }
            }
            Ok(())
        })();
        let actual = if let Some(file) = &file {
            file.metadata().ok().map(|metadata| metadata.len())
        } else if renamed {
            Some(written)
        } else {
            self.game_file_size(temporary.file_name().expect("private segment name"))
                .ok()
        };
        drop(file);
        // Drop releases only unwritten reserved bytes; CountedWriter already
        // debited every successful write, including a failed publication.
        drop(write_reservation);
        self.note(|meter| {
            let actual_written = if self.runtime_limits.is_some() {
                Some(written)
            } else {
                actual
            };
            meter.archive_write_bytes_total = actual_written.and_then(|amount| {
                meter
                    .archive_write_bytes_total
                    .and_then(|total| total.checked_add(amount))
            });
            meter.write_bytes_reserved = 0;
            meter.write_bytes_remaining = self
                .write_ledger
                .as_ref()
                .and_then(|ledger| ledger.remaining().ok());
            if result.is_ok() {
                meter.pending_commit_bytes = Some(0);
                meter.pending_files_retained = false;
            } else {
                meter.pending_commit_bytes = actual;
                meter.pending_files_retained = true;
            }
            if self.runtime_limits.is_some() && result.is_err() {
                meter.admission_closed = true;
            }
        });
        if result.is_ok() {
            self.pending_cleanup = false;
        }
        if self.runtime_limits.is_none()
            && result.is_err()
            && self
                .game_file_size(temporary.file_name().expect("private segment name"))
                .is_ok()
        {
            self.remove_game_file(temporary.file_name().expect("private segment name"))
                .map_err(|e| io("archive temporary cleanup", e))?;
            self.note(|meter| {
                meter.pending_commit_bytes = Some(0);
                meter.pending_files_retained = false;
            });
        }
        result?;
        drop(lock);
        Ok(ArchiveReceipt {
            owner: Arc::downgrade(&self.identity),
            generation,
            checksum,
            before,
            after: before,
            committed_bytes: HEADER + bytes.len() as u64,
            io_bytes: 2 * (HEADER + bytes.len() as u64),
        })
    }
    fn read(
        &self,
        generation: u64,
        expected: Option<[u8; 32]>,
        budget: &mut ArchiveIoBudget,
    ) -> Result<(Bundle, [u8; 32]), StoreError> {
        let result = self.read_inner(generation, expected, budget);
        if let Err(error) = &result {
            self.failure(error);
        }
        result
    }
    fn read_inner(
        &self,
        generation: u64,
        expected: Option<[u8; 32]>,
        budget: &mut ArchiveIoBudget,
    ) -> Result<(Bundle, [u8; 32]), StoreError> {
        budget.check()?;
        self.verify_directory_boundary()?;
        let path = self.path(generation);
        if self.directory_capability.is_none() {
            no_link(&path)?;
        }
        let mut file = self
            .game_file(path.file_name().expect("private segment name"), false)
            .map_err(|e| io("archive open", e))?;
        let metadata = file.metadata().map_err(|e| io("archive file size", e))?;
        if self.directory_capability.is_some() && !metadata.is_file() {
            return Err(StoreError::ArchiveIntegrity("archive record type"));
        }
        let size = metadata.len();
        if size < HEADER || size > self.config.record_bytes.saturating_add(HEADER) {
            return Err(StoreError::ArchiveIntegrity("archive file bounds"));
        }
        budget.debit(size)?;
        let mut header = [0u8; HEADER as usize];
        file.read_exact(&mut header)
            .map_err(|e| io("archive header read", e))?;
        let stored_generation = u64::from_le_bytes(header[8..16].try_into().expect("fixed header"));
        let length = u64::from_le_bytes(header[16..24].try_into().expect("fixed header"));
        let checksum: [u8; 32] = header[24..56].try_into().expect("fixed header");
        if &header[..8] != MAGIC
            || generation != stored_generation
            || length != size - HEADER
            || expected.is_some_and(|expected| expected != checksum)
        {
            return Err(StoreError::ArchiveIntegrity(
                "archive header or handle checksum",
            ));
        }
        let count = usize::try_from(length).map_err(|_| StoreError::ArchiveByteBudget)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(count)
            .map_err(|_| StoreError::Capacity("archive load buffer"))?;
        let mut digest = Sha256::new();
        let mut scratch = [0u8; 8192];
        while bytes.len() < count {
            budget.check()?;
            let take = (count - bytes.len()).min(scratch.len());
            file.read_exact(&mut scratch[..take])
                .map_err(|e| io("archive payload read", e))?;
            digest.update(&scratch[..take]);
            bytes.extend_from_slice(&scratch[..take]);
        }
        if <[u8; 32]>::from(digest.finalize()) != checksum {
            return Err(StoreError::ArchiveIntegrity("archive checksum"));
        }
        budget.check()?;
        let bundle: Bundle = serde_json::from_slice(&bytes)
            .map_err(|_| StoreError::ArchiveIntegrity("archive decode"))?;
        if bundle.version != 1
            || bundle.generation != generation
            || bundle.rules_version != rz_position::RULES_VERSION
            || bundle.variant != rz_position::RULES_VARIANT
        {
            return Err(StoreError::ArchiveIntegrity("archive schema or generation"));
        }
        self.verify_directory_boundary()?;
        Ok((bundle, checksum))
    }
}
struct CountedWriter<'a, W> {
    inner: &'a mut W,
    written: &'a mut u64,
    reservation: Option<&'a mut ArchiveWriteReservation>,
}
impl<W: Write> Write for CountedWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let count = self.inner.write(bytes)?;
        *self.written = self
            .written
            .checked_add(count as u64)
            .ok_or_else(|| std::io::Error::other("archive write count overflow"))?;
        if let Some(reservation) = &mut self.reservation {
            reservation.debit_written(count as u64)?;
        }
        Ok(count)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}
enum QuotaLock {
    Ambient(PathBuf),
    Held { root: Arc<Dir>, file: File },
}
impl QuotaLock {
    fn take(root: &Path, capability: Option<&ArchiveRootCapability>) -> Result<Self, StoreError> {
        if let Some(capability) = capability {
            let mut options = CapabilityOpenOptions::new();
            options
                .write(true)
                .create_new(true)
                .follow(FollowSymlinks::No);
            let file = capability
                .directory
                .open_with(".quota.lock", &options)
                .map_err(|e| io("held archive quota lock", e))?
                .into_std();
            return Ok(Self::Held {
                root: Arc::clone(&capability.directory),
                file,
            });
        }
        let path = root.join(".quota.lock");
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| io("archive quota lock", e))?;
        Ok(Self::Ambient(path))
    }
}
impl Drop for QuotaLock {
    fn drop(&mut self) {
        match self {
            Self::Ambient(path) => {
                let _ = fs::remove_file(path);
            }
            Self::Held { root, file } => {
                // Never unlink a substituted lock or redirect cleanup through a
                // renamed ambient parent. This is not an atomic compare-unlink
                // guarantee against another writer mutating the private root.
                let owned = cap_std::fs::Metadata::from_file(file);
                let mut options = CapabilityOpenOptions::new();
                options.read(true).follow(FollowSymlinks::No);
                #[cfg(target_os = "linux")]
                {
                    use cap_std::fs::OpenOptionsExt;
                    options.custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK);
                }
                // Capture named metadata from a real open file, not pathname
                // metadata lacking the Windows by-handle file identifier.
                let named_file = root.open_with(".quota.lock", &options);
                let named = named_file
                    .as_ref()
                    .ok()
                    .and_then(|file| file.metadata().ok());
                if owned
                    .as_ref()
                    .ok()
                    .zip(named.as_ref())
                    .is_some_and(|(owned, named)| {
                        named.is_file()
                            && !named.is_symlink()
                            && owned.dev() == named.dev()
                            && owned.ino() == named.ino()
                    })
                {
                    let _ = root.remove_file(".quota.lock");
                }
            }
        }
    }
}
struct BoundedBuffer {
    bytes: Vec<u8>,
    maximum: u64,
    deadline: Instant,
    error: Option<StoreError>,
}
impl Write for BoundedBuffer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let error = if Instant::now() >= self.deadline {
            Some(StoreError::ArchiveDeadline)
        } else if (self.bytes.len() as u64)
            .checked_add(bytes.len() as u64)
            .is_none_or(|n| n > self.maximum)
        {
            Some(StoreError::ArchiveByteBudget)
        } else {
            let needed = self.bytes.len() + bytes.len();
            let target = needed.max(
                self.bytes
                    .capacity()
                    .saturating_mul(2)
                    .max(8192)
                    .min(self.maximum as usize),
            );
            if needed > self.bytes.capacity()
                && self
                    .bytes
                    .try_reserve_exact(target - self.bytes.len())
                    .is_err()
            {
                Some(StoreError::Capacity("archive encode buffer"))
            } else {
                None
            }
        };
        if let Some(error) = error {
            self.error = Some(error);
            return Err(std::io::Error::other("bounded archive encoding"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// The allowance is borrowed by one synchronous operation. Drop also clears it
/// when the caller catches an unwind, without extending its deadline or bytes.
struct ArchiveAllocationScope<'a> {
    stores: &'a mut PalsStores,
    budget: &'a mut ArchiveIoBudget,
    owner: Weak<()>,
    initial: ArchiveIoBudget,
    old_pins: Option<StorePins>,
}
impl ArchiveAllocationScope<'_> {
    fn finish(&mut self) -> Result<(), StoreError> {
        let Some(old_pins) = self.old_pins.take() else {
            return Ok(());
        };
        let manager = self
            .stores
            .archive
            .as_mut()
            .ok_or(StoreError::InvalidHandle("archive allocation owner"))?;
        if !self.owner.ptr_eq(&Arc::downgrade(&manager.identity)) {
            return Err(StoreError::InvalidHandle("archive allocation owner"));
        }
        let was_active = manager.allocation_active;
        let remaining = manager.auto_budget.take();
        manager.allocation_active = false;
        manager.pins = old_pins;
        let remaining = remaining.ok_or(StoreError::ArchiveIntegrity(
            "archive allocation allowance disappeared",
        ))?;
        *self.budget = remaining;
        if !was_active
            || remaining.deadline != self.initial.deadline
            || remaining.max_bytes > self.initial.max_bytes
        {
            return Err(StoreError::ArchiveIntegrity(
                "archive allocation allowance changed",
            ));
        }
        self.stores
            .archive
            .as_ref()
            .expect("same archive owner")
            .verify_hot_owner(self.stores)
    }
}
impl Drop for ArchiveAllocationScope<'_> {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

impl PalsStores {
    fn with_archive_read_budget<T>(
        &self,
        original: &mut ArchiveIoBudget,
        operation: impl FnOnce(&mut ArchiveIoBudget) -> Result<T, StoreError>,
    ) -> Result<T, StoreError> {
        self.reject_archive_allocation_scope()?;
        original.check()?;
        let manager = self.archive.as_ref().ok_or(StoreError::ArchiveDisabled)?;
        let mut budget = manager.load_budget(*original)?;
        let initial = budget.max_bytes;
        let started = Instant::now();
        let baseline = manager.snapshot();
        manager.note(|_| {}); // Accepted explicit read scope; never getter work.
        let result = operation(&mut budget).and_then(|value| {
            budget.check()?;
            manager.verify_directory_boundary()?;
            Ok(value)
        });
        let bytes = initial - budget.max_bytes;
        original.max_bytes -= bytes;
        // Directory/uniqueness scans consume the selected read allowance but
        // do not invent a cold-handle load or an owner/generation check.
        manager.finish_read(baseline, started, bytes, false, &result);
        result
    }
    pub fn enable_archive(&mut self, config: ArchiveConfig) -> Result<(), StoreError> {
        self.enable_archive_inner(config, None)
    }
    /// Opt-in startup authority. Legacy path-based callers retain their format
    /// and behaviour; this owner additionally retains the caller's directory.
    pub fn enable_archive_with_capability(
        &mut self,
        config: ArchiveConfig,
        capability: ArchiveRootCapability,
    ) -> Result<(), StoreError> {
        self.enable_archive_inner(config, Some(capability))
    }
    pub(in crate::pals) fn validate_unused_archive_setup(&self) -> Result<(), StoreError> {
        if self.generation != 0
            || self.root.is_some()
            || self.hot_stats() != ArchiveStats::default()
            || !self.derived.entries.is_empty()
            || !self.checked_cpu_pvs.is_empty()
            || self.review_pin_count()? != 0
            || self.archive.is_some()
            || self.archive_write_ledger.is_some()
        {
            return Err(StoreError::InvalidConditions(
                "archive setup needs unused Store",
            ));
        }
        Ok(())
    }
    fn enable_archive_inner(
        &mut self,
        config: ArchiveConfig,
        capability: Option<ArchiveRootCapability>,
    ) -> Result<(), StoreError> {
        if self.archive.is_some() {
            return Err(StoreError::InvalidConditions("archive already enabled"));
        }
        if self.archive_write_ledger.is_some() {
            return Err(StoreError::InvalidConditions(
                "archive session write authority requires a replacement store",
            ));
        }
        let mut manager = match capability {
            Some(capability) => ArchiveManager::open_with_capability(config, capability)?,
            None => ArchiveManager::open(config)?,
        };
        manager.state_owner = Arc::downgrade(&self.states.identity);
        manager.line_owner = Arc::downgrade(&self.lines.identity);
        self.archive = Some(manager);
        Ok(())
    }
    pub fn archive_enabled(&self) -> bool {
        self.archive.is_some()
    }
    pub fn archive_configuration(&self) -> Option<&ArchiveConfig> {
        self.archive.as_ref().map(|manager| &manager.config)
    }
    pub fn archive_runtime_limits(&self) -> Option<&ArchiveRuntimeLimits> {
        self.archive
            .as_ref()
            .and_then(|manager| manager.runtime_limits.as_ref())
    }
    pub fn set_archive_runtime_limits(
        &mut self,
        limits: ArchiveRuntimeLimits,
    ) -> Result<(), StoreError> {
        let hot_in_use = self.hot_stats() != ArchiveStats::default();
        let manager = self.archive.as_mut().ok_or(StoreError::ArchiveDisabled)?;
        if hot_in_use
            || self.archive_write_ledger.is_some()
            || manager.used
            || manager.runtime_limits.is_some()
            || manager.allocation_active
            || manager.auto_budget.is_some()
            || !manager.load_pins.is_empty()
            || manager.pins.count() != 0
            || manager.snapshot().event_sequence != 1
        {
            return Err(StoreError::InvalidConditions(
                "archive selectors are one-shot before use",
            ));
        }
        if limits.game_bytes_max == 0
            || limits.game_bytes_max > GAME_ARCHIVE_BYTES
            || limits.global_bytes_max == 0
            || limits.global_bytes_max > GLOBAL_ARCHIVE_BYTES
            || limits.game_bytes_max > limits.global_bytes_max
            || limits.record_payload_bytes_max == 0
            || limits.record_payload_bytes_max > ARCHIVE_RECORD_PAYLOAD_BYTES
            || limits.record_payload_bytes_max > limits.game_bytes_max
            || limits.max_load_pins == 0
            || limits.max_load_pins > 16
            || limits.index_entries_max == 0
            || limits.index_entries_max > 1_048_576
            || limits.index_bytes_max == 0
            || limits.index_bytes_max > ARCHIVE_RECORD_PAYLOAD_BYTES
            || limits.load_bytes_max == 0
            || limits.load_bytes_max > limits.game_bytes_max
            || limits.load_bytes_max > ARCHIVE_RECORD_PAYLOAD_BYTES
            || limits.load_deadline_max_ms == 0
            || Instant::now()
                .checked_add(std::time::Duration::from_millis(
                    limits.load_deadline_max_ms,
                ))
                .is_none()
        {
            return Err(StoreError::InvalidConditions("archive runtime selectors"));
        }
        let ledger = ArchiveWriteLedger::new(
            manager.config.root.clone(),
            limits,
            manager.root_capability.clone(),
        )?;
        manager.config.game_bytes = limits.game_bytes_max;
        manager.config.global_bytes = limits.global_bytes_max;
        manager.config.record_bytes = limits.record_payload_bytes_max;
        manager.config.max_load_pins = limits.max_load_pins as usize;
        manager.runtime_limits = Some(limits);
        manager.write_ledger = Some(Arc::clone(&ledger));
        self.archive_write_ledger = Some(ledger);
        manager.note(|meter| {
            meter.runtime_limits = Some(limits);
            meter.load_pins_max = limits.max_load_pins;
            meter.write_bytes_remaining = Some(limits.game_bytes_max);
        });
        Ok(())
    }
    /// Attach an unused replacement owner to the existing session's actual
    /// cumulative write authority. The source may already have closed: only its
    /// ledger survives close, never its file or ColdHandle authority. Rejection
    /// changes neither owner. Legacy None/None selectors keep their old meaning.
    pub fn inherit_archive_write_budget_from(
        &mut self,
        previous: &PalsStores,
    ) -> Result<(), StoreError> {
        let hot_in_use = self.hot_stats() != ArchiveStats::default()
            || self.root.is_some()
            || self.generation != 0
            || !self.checked_cpu_pvs.is_empty()
            || !self.derived.entries.is_empty();
        let manager = self.archive.as_mut().ok_or(StoreError::ArchiveDisabled)?;
        let Some(limits) = manager.runtime_limits else {
            if previous.archive_write_ledger.is_none()
                && previous
                    .archive
                    .as_ref()
                    .is_none_or(|source| source.runtime_limits.is_none())
            {
                return Ok(());
            }
            return Err(StoreError::InvalidConditions(
                "archive write budget selectors mismatch",
            ));
        };
        let source =
            previous
                .archive_write_ledger
                .as_ref()
                .ok_or(StoreError::InvalidConditions(
                    "archive write budget selectors mismatch",
                ))?;
        if let Some(capability) = &source.root_capability {
            capability.verify_named_root()?;
        }
        manager.verify_directory_boundary()?;
        let same_capability = match (&source.root_capability, &manager.root_capability) {
            (None, None) => true,
            (Some(source), Some(target)) => {
                source.root == target.root && source.identity == target.identity
            }
            _ => false,
        };
        if source.root != manager.config.root || source.runtime_limits != limits || !same_capability
        {
            return Err(StoreError::InvalidConditions(
                "archive write budget domain mismatch",
            ));
        }
        let initial = manager.snapshot();
        let destination_unused = manager
            .write_ledger
            .as_ref()
            .is_some_and(|ledger| ledger.facts() == Some((0, 0)));
        if hot_in_use
            || manager.used
            || manager.write_budget_inherited
            || manager.next_generation != 0
            || manager.allocation_active
            || manager.auto_budget.is_some()
            || !manager.load_pins.is_empty()
            || manager.pins.count() != 0
            || manager.pending_cleanup
            || initial.event_sequence != 2
            || initial.lifecycle != ArchiveOwnerLifecycle::Open
            || initial.admission_closed
            || initial.archive_write_bytes_total != Some(0)
            || !destination_unused
        {
            return Err(StoreError::InvalidConditions(
                "archive write budget inheritance is one-shot before use",
            ));
        }
        let (_, reserved) = source
            .facts()
            .ok_or(StoreError::ArchiveQuota("archive write ledger unknown"))?;
        if reserved != 0 {
            return Err(StoreError::InvalidConditions(
                "archive write budget inheritance requires idle ledger",
            ));
        }
        manager.write_ledger = Some(Arc::clone(source));
        manager.write_budget_inherited = true;
        self.archive_write_ledger = Some(Arc::clone(source));
        manager.note(|_| {});
        Ok(())
    }
    /// O(1), with no filesystem access or event increment.
    pub fn archive_owner_snapshot(&self) -> Result<Option<ArchiveOwnerSnapshot>, StoreError> {
        Ok(self.archive.as_ref().map(ArchiveManager::snapshot))
    }
    /// The caller must already hold its physical/owned-release fence. This
    /// method closes only the archive authority, retains files and hot facts,
    /// and never creates another cleanup allowance or performs I/O.
    pub fn close_archive_owner(&mut self) -> Result<Option<ArchiveOwnerSnapshot>, StoreError> {
        let Some(manager) = self.archive.as_ref() else {
            return Ok(None);
        };
        let snapshot = manager.snapshot();
        if manager.allocation_active
            || self.review_pin_count()? != 0
            || manager.auto_budget.is_some()
            || !manager.load_pins.is_empty()
            || manager.pending_cleanup
            || snapshot.write_bytes_reserved != 0
            || snapshot.pending_commit_bytes != Some(0)
            || snapshot.pending_files_retained
            || snapshot.pending_buffers_retained
        {
            let error = StoreError::InvalidConditions("archive close retains pending ownership");
            manager.failure(&error);
            return Err(error);
        }
        let manager = self.archive.take().expect("checked archive owner");
        manager.note(|meter| {
            meter.admission_closed = true;
            meter.cleanup_complete = true;
            meter.lifecycle = ArchiveOwnerLifecycle::Closed;
        });
        let snapshot = manager.snapshot();
        drop(manager); // Expire the actual Arc; copied telemetry is not authority.
        Ok(Some(snapshot))
    }
    /// Explicit actual quota measurement under an existing caller allowance.
    /// This gives an unused selected owner a baseline without getter-side I/O.
    pub fn measure_archive_usage_with_budget(
        &mut self,
        budget: &mut ArchiveIoBudget,
    ) -> Result<(), StoreError> {
        self.reject_archive_allocation_scope()?;
        let manager = self.archive.as_mut().ok_or(StoreError::ArchiveDisabled)?;
        manager.used = true;
        let result = manager.usage(*budget).and_then(|(game, global)| {
            if game > manager.config.game_bytes || global > manager.config.global_bytes {
                Err(StoreError::ArchiveQuota("observed managed archive bytes"))
            } else {
                Ok(())
            }
        });
        if let Err(error) = &result {
            manager.failure(error);
        }
        result
    }
    pub(super) fn reject_archive_allocation_scope(&self) -> Result<(), StoreError> {
        if self
            .archive
            .as_ref()
            .is_some_and(|manager| manager.allocation_active)
        {
            Err(StoreError::InvalidConditions(
                "archive allocation allowance is active",
            ))
        } else {
            Ok(())
        }
    }

    /// Scope one allocating facade call under the caller's unchanged deadline
    /// and remaining byte allowance. Pins include resident engine ownership and
    /// caller-owned temporary IDs. Both success and failure return actual I/O
    /// debits, restore the previous pins, and clear the temporary allowance.
    pub fn with_archive_allocation<T>(
        &mut self,
        pins: StorePins,
        budget: &mut ArchiveIoBudget,
        operation: impl FnOnce(&mut PalsStores) -> Result<T, StoreError>,
    ) -> Result<T, StoreError> {
        budget.check()?;
        let manager = self.archive.as_ref().ok_or(StoreError::ArchiveDisabled)?;
        if manager.allocation_active || manager.auto_budget.is_some() {
            return Err(StoreError::InvalidConditions(
                "archive allocation allowance is active",
            ));
        }
        manager.verify_hot_owner(self)?;
        let baseline = manager.snapshot();
        let owner = Arc::downgrade(&manager.identity);
        // The caller already owns its temporary/resident pins. Validate their
        // union before moving the previous pins into the restoration guard.
        let mut combined = pins;
        combined.extend(&manager.pins);
        if let Err(error) = self.validate_pins(&combined) {
            manager.failure_after(baseline, &error);
            return Err(error);
        }
        let initial = *budget;
        let manager = self.archive.as_mut().expect("archive checked");
        let old_pins = std::mem::replace(&mut manager.pins, combined);
        manager.auto_budget = Some(initial);
        manager.allocation_active = true;
        manager.used = true;
        let mut scope = ArchiveAllocationScope {
            stores: self,
            budget,
            owner,
            initial,
            old_pins: Some(old_pins),
        };
        let result = operation(&mut *scope.stores);
        scope.finish()?;
        if let Err(error) = &result {
            scope
                .stores
                .archive
                .as_ref()
                .expect("same archive owner")
                .failure_after(baseline, error);
        }
        result
    }
    pub(super) fn automatic_archive_enabled(&self) -> bool {
        self.archive
            .as_ref()
            .is_some_and(|manager| manager.auto_budget.is_some())
    }
    pub(super) fn check_archive_allocation_deadline(&self) -> Result<(), StoreError> {
        if let Some(budget) = self
            .archive
            .as_ref()
            .and_then(|manager| manager.auto_budget)
        {
            budget.check()?;
        }
        Ok(())
    }
    /// One search owns this explicit I/O allowance. Automatic pressure handling
    /// debits it, so multiple allocations cannot each spend the full allowance.
    pub fn set_archive_io_budget(&mut self, budget: ArchiveIoBudget) -> Result<(), StoreError> {
        self.reject_archive_allocation_scope()?;
        budget.check()?;
        self.archive
            .as_mut()
            .ok_or(StoreError::ArchiveDisabled)?
            .auto_budget = Some(budget);
        self.archive.as_mut().expect("checked archive owner").used = true;
        Ok(())
    }
    pub fn set_archive_pins(&mut self, pins: StorePins) -> Result<(), StoreError> {
        self.reject_archive_allocation_scope()?;
        if let Err(error) = self.validate_pins(&pins) {
            if let Some(manager) = &self.archive {
                manager.failure(&error);
            }
            return Err(error);
        }
        self.archive
            .as_mut()
            .ok_or(StoreError::ArchiveDisabled)?
            .pins = pins;
        self.archive.as_mut().expect("checked archive owner").used = true;
        Ok(())
    }
    pub fn hot_stats(&self) -> ArchiveStats {
        ArchiveStats::hot(self)
    }
    pub(super) fn automatic_reclaim(
        &mut self,
        extra: StorePins,
        force: bool,
    ) -> Result<bool, StoreError> {
        let Some(manager) = self.archive.as_ref() else {
            return Ok(false);
        };
        let Some(mut budget) = manager.auto_budget else {
            return Ok(false);
        };
        budget.check()?;
        let result = self.reclaim_for_allocation(extra, force, &mut budget);
        self.archive.as_mut().expect("archive restored").auto_budget = Some(budget);
        result
    }
    fn reclaim_for_allocation(
        &mut self,
        extra: StorePins,
        force: bool,
        budget: &mut ArchiveIoBudget,
    ) -> Result<bool, StoreError> {
        if !force && !self.pressure() {
            return Ok(false);
        }
        let manager = self.archive.as_ref().ok_or(StoreError::ArchiveDisabled)?;
        let mut combined = extra;
        combined.extend(&manager.pins);
        self.validate_pins(&combined)?;
        let old_pins = std::mem::replace(
            &mut self.archive.as_mut().expect("archive checked").pins,
            combined,
        );
        let result = self.archive_inactive_inner(budget);
        let manager = self.archive.as_mut().expect("archive restored");
        manager.pins = old_pins;
        result.map(|receipt| receipt.is_some())
    }
    /// A foreign physical request must remain unique across hot and cold facts.
    /// The explicit allowance covers scans and any pressure/single retry archive
    /// writes, including failed I/O. Hot-only legacy append never scans disk.
    pub fn append_observation_checked_with_archive_budget(
        &mut self,
        observation: Observation,
        budget: &mut ArchiveIoBudget,
    ) -> Result<ObservationId, StoreError> {
        self.append_observation_checked_with_archive_budget_controlled(observation, budget, || {
            Ok(())
        })
    }
    pub fn append_observation_checked_with_archive_budget_controlled(
        &mut self,
        observation: Observation,
        budget: &mut ArchiveIoBudget,
        controls: impl FnMut() -> Result<(), StoreError>,
    ) -> Result<ObservationId, StoreError> {
        self.append_observation_checked_inner(observation, budget, controls, true)
    }

    /// Retain a completed foreign raw report during logical cancellation only
    /// while the original I/O allowance still permits the cold uniqueness scan.
    /// Exactly one hot append is attempted; no reclamation, retry, or logical
    /// task consumption occurs, and the deadline is never extended.
    pub fn append_observation_checked_first_with_archive_budget(
        &mut self,
        observation: Observation,
        budget: &mut ArchiveIoBudget,
    ) -> Result<ObservationId, StoreError> {
        self.append_observation_checked_inner(observation, budget, || Ok(()), false)
    }

    fn append_observation_checked_inner(
        &mut self,
        observation: Observation,
        budget: &mut ArchiveIoBudget,
        mut controls: impl FnMut() -> Result<(), StoreError>,
        reclaim: bool,
    ) -> Result<ObservationId, StoreError> {
        self.reject_archive_allocation_scope()?;
        budget.check()?;
        controls()?;
        self.validate_observation_handles(&observation)?;
        if !self.archive_enabled() {
            let result = self.append_observation_once(observation);
            controls()?;
            if result.is_ok() {
                budget.check()?;
            }
            return result;
        }
        if observation.external_report.is_some() {
            self.with_archive_read_budget(budget, |bounded| {
                self.check_foreign_cold_request(&observation, bounded)
            })?;
        }
        if !reclaim {
            budget.check()?;
            let result = self.append_observation_once(observation);
            if result.is_ok() {
                budget.check()?;
            }
            return result;
        }
        let mut pins = StorePins::default();
        pins.states.insert(observation.state);
        pins.lines.extend(observation.line);
        pins.lines.extend(observation.cpu_pv);
        pins.observations.extend(observation.supersedes);
        pins.executions.extend(observation.execution);
        pins.observations
            .extend(observation.score.model_dependencies().into_iter().flatten());
        self.reclaim_for_allocation(pins.clone(), false, budget)?;
        budget.check()?;
        controls()?;
        let result = self.append_observation_once(observation.clone());
        if matches!(result, Err(StoreError::Capacity(_))) {
            controls()?;
            if self.reclaim_for_allocation(pins, true, budget)? {
                budget.check()?;
                controls()?;
                let result = self.append_observation_once(observation);
                controls()?;
                return result;
            }
        }
        controls()?;
        result
    }
    fn check_foreign_cold_request(
        &self,
        observation: &Observation,
        budget: &mut ArchiveIoBudget,
    ) -> Result<(), StoreError> {
        let manager = self.archive.as_ref().ok_or(StoreError::ArchiveDisabled)?;
        manager.verify_hot_owner(self)?;
        manager.verify_directory_boundary()?;
        for entry in manager
            .game_entries()
            .map_err(|e| io("foreign archive uniqueness directory", e))?
        {
            budget.check()?;
            let entry = entry.map_err(|e| io("foreign archive uniqueness entry", e))?;
            let name = entry.file_name();
            let Some(number) = name
                .to_str()
                .and_then(|name| name.strip_prefix("seg-"))
                .and_then(|name| name.strip_suffix(".arc"))
            else {
                continue;
            };
            let generation = number
                .parse::<u64>()
                .map_err(|_| StoreError::ArchiveIntegrity("archive segment name"))?;
            if generation >= manager.next_generation {
                continue;
            }
            let (bundle, _) = manager.read(generation, None, budget)?;
            self.check_bundle_counts(&bundle)?;
            if bundle
                .observations
                .iter()
                .any(|(_, cold)| cold.conflicts_external_request(observation))
            {
                return Err(StoreError::InvalidEvidence(
                    "foreign physical request belongs to another archived task",
                ));
            }
        }
        Ok(())
    }
    pub(crate) fn write_retirement_state<W: Write + ?Sized>(
        &self,
        id: StateId,
        writer: &mut W,
    ) -> Result<(), StoreError> {
        codec::write_retirement_wire(writer, &StateWire::encode(id, self.states.get(id)?)?)
    }

    /// Serialize the retirement owner's retained source, including an original
    /// snapshot that is no longer a current hot lookup. This grants no StateId.
    pub(crate) fn write_retirement_snapshot<W: Write + ?Sized>(
        &self,
        id: StateId,
        snapshot: &PositionSnapshot,
        writer: &mut W,
    ) -> Result<(), StoreError> {
        codec::write_retirement_wire(writer, &StateWire::encode(id, snapshot)?)
    }

    pub(crate) fn write_retirement_line<W: Write + ?Sized>(
        &self,
        id: LineId,
        writer: &mut W,
    ) -> Result<(), StoreError> {
        codec::write_retirement_wire(writer, &LineWire::encode(id, self.lines.get(id)?))
    }

    pub(crate) fn write_retirement_observation<W: Write + ?Sized>(
        &self,
        id: ObservationId,
        writer: &mut W,
    ) -> Result<(), StoreError> {
        codec::write_retirement_wire(
            writer,
            &ObservationWire::encode(self.observations.get(id)?)?,
        )
    }

    /// Unpublished failure evidence remains unpublished; the sink binds this
    /// exact value to its original retirement custody rather than a new lookup.
    pub(crate) fn write_retirement_observation_value<W: Write + ?Sized>(
        &self,
        value: &Observation,
        writer: &mut W,
    ) -> Result<(), StoreError> {
        codec::write_retirement_wire(writer, &ObservationWire::encode(value)?)
    }

    pub(crate) fn write_retirement_role_record<W: Write + ?Sized>(
        &self,
        record: &crate::pals::engine::RoleRecord,
        writer: &mut W,
    ) -> Result<(), StoreError> {
        codec::write_retirement_wire(writer, &RoleRecordWire::encode(record)?)
    }

    pub(crate) fn write_retirement_external_report<W: Write + ?Sized>(
        &self,
        report: &ExternalCheckerReport,
        writer: &mut W,
    ) -> Result<(), StoreError> {
        codec::write_retirement_external_report(report, writer)
    }

    // Authored equivalence fixtures retain the uncontrolled form; actual
    // retirement serialization must use the original controlled window below.
    #[cfg(test)]
    pub(crate) fn retirement_dependency_closure(
        &self,
        pins: &StorePins,
        max_records: usize,
    ) -> Result<StorePins, StoreError> {
        if pins.count() > max_records {
            return Err(StoreError::Capacity("retirement dependency closure"));
        }
        self.expand_pins_inner(pins.clone(), Some(max_records), true, None)
    }

    /// Actual retirement traversal uses the original finite-window/cancel
    /// check during cloning and graph walks, before any stream footer exists.
    pub(crate) fn retirement_dependency_closure_controlled(
        &self,
        pins: &StorePins,
        max_records: usize,
        check: &mut dyn FnMut() -> Result<(), StoreError>,
    ) -> Result<StorePins, StoreError> {
        check()?;
        if pins.count() > max_records {
            return Err(StoreError::Capacity("retirement dependency closure"));
        }
        let mut check = Some(check);
        check_pin_closure(&mut check)?;
        let copied = StorePins {
            states: clone_closure_set(&pins.states, &mut check)?,
            lines: clone_closure_set(&pins.lines, &mut check)?,
            observations: clone_closure_set(&pins.observations, &mut check)?,
            situations: clone_closure_set(&pins.situations, &mut check)?,
            executions: clone_closure_set(&pins.executions, &mut check)?,
        };
        check_pin_closure(&mut check)?;
        self.expand_pins_inner(copied, Some(max_records), true, check)
    }

    pub(crate) fn retirement_dependency_edges(
        &self,
    ) -> impl Iterator<Item = (ObservationId, &BTreeSet<SituationId>)> {
        self.dependencies
            .dependents
            .iter()
            .map(|(observation, situations)| (*observation, situations))
    }

    fn validate_pins(&self, pins: &StorePins) -> Result<(), StoreError> {
        self.validate_pins_inner(pins, &mut None)
    }

    fn validate_pins_inner(
        &self,
        pins: &StorePins,
        check: &mut PinClosureCheck<'_>,
    ) -> Result<(), StoreError> {
        check_pin_closure(check)?;
        if pins.states.len() > self.limits.states
            || pins.lines.len() > self.limits.line_chunks
            || pins.situations.len() > self.limits.situations
            || pins.observations.len() > self.limits.observations
            || pins.executions.len() > self.limits.executions
        {
            return Err(StoreError::PinSaturated("pin index"));
        }
        for id in &pins.states {
            check_pin_closure(check)?;
            self.states.get(*id)?;
            check_pin_closure(check)?;
        }
        for id in &pins.lines {
            check_pin_closure(check)?;
            self.lines.get(*id)?;
            check_pin_closure(check)?;
        }
        for id in &pins.observations {
            check_pin_closure(check)?;
            self.observations.get(*id)?;
            check_pin_closure(check)?;
        }
        for id in &pins.situations {
            check_pin_closure(check)?;
            self.situations.get(*id)?;
            check_pin_closure(check)?;
        }
        for id in &pins.executions {
            check_pin_closure(check)?;
            self.tasks.get(*id)?;
            check_pin_closure(check)?;
        }
        check_pin_closure(check)
    }
    fn expand_pins(&self, pins: StorePins) -> Result<StorePins, StoreError> {
        self.expand_pins_inner(pins, None, false, None)
    }

    fn expand_pins_inner(
        &self,
        mut pins: StorePins,
        maximum: Option<usize>,
        include_task_consumers: bool,
        mut check: PinClosureCheck<'_>,
    ) -> Result<StorePins, StoreError> {
        check_pin_closure(&mut check)?;
        let mut count = pins.count();
        loop {
            check_pin_closure(&mut check)?;
            let before = pins.count();
            for id in clone_closure_set(&pins.situations, &mut check)? {
                check_pin_closure(&mut check)?;
                let s = self.situations.get(id)?;
                insert_closure_pin(&mut pins.states, s.state, &mut count, maximum, &mut check)?;
                insert_closure_pin(&mut pins.lines, s.focus, &mut count, maximum, &mut check)?;
                for (line, conclusion) in s.conclusions.iter() {
                    check_pin_closure(&mut check)?;
                    insert_closure_pin(&mut pins.lines, *line, &mut count, maximum, &mut check)?;
                    if let Some(o) = conclusion.evidence {
                        insert_closure_pin(
                            &mut pins.observations,
                            o,
                            &mut count,
                            maximum,
                            &mut check,
                        )?;
                    }
                    if let ContinuationStatus::RepairedBy(line) = conclusion.status {
                        insert_closure_pin(&mut pins.lines, line, &mut count, maximum, &mut check)?;
                    }
                    check_pin_closure(&mut check)?;
                }
                check_pin_closure(&mut check)?;
            }
            for id in clone_closure_set(&pins.lines, &mut check)? {
                check_pin_closure(&mut check)?;
                let line = self.lines.get(id)?;
                insert_closure_pin(
                    &mut pins.states,
                    line.start_state,
                    &mut count,
                    maximum,
                    &mut check,
                )?;
                if let Some(parent) = line.parent {
                    insert_closure_pin(&mut pins.lines, parent, &mut count, maximum, &mut check)?;
                }
                check_pin_closure(&mut check)?;
            }
            for id in clone_closure_set(&pins.observations, &mut check)? {
                check_pin_closure(&mut check)?;
                let o = self.observations.get(id)?;
                insert_closure_pin(&mut pins.states, o.state, &mut count, maximum, &mut check)?;
                for line in [o.line, o.cpu_pv].into_iter().flatten() {
                    check_pin_closure(&mut check)?;
                    insert_closure_pin(&mut pins.lines, line, &mut count, maximum, &mut check)?;
                }
                for observation in o
                    .supersedes
                    .into_iter()
                    .chain(o.score.model_dependencies().into_iter().flatten())
                {
                    check_pin_closure(&mut check)?;
                    insert_closure_pin(
                        &mut pins.observations,
                        observation,
                        &mut count,
                        maximum,
                        &mut check,
                    )?;
                }
                if let Some(execution) = o.execution {
                    insert_closure_pin(
                        &mut pins.executions,
                        execution,
                        &mut count,
                        maximum,
                        &mut check,
                    )?;
                }
                check_pin_closure(&mut check)?;
            }
            for id in clone_closure_set(&pins.executions, &mut check)? {
                check_pin_closure(&mut check)?;
                let task = self.tasks.get(id)?;
                insert_closure_pin(
                    &mut pins.states,
                    task.key.state,
                    &mut count,
                    maximum,
                    &mut check,
                )?;
                if let Some(line) = task.key.line {
                    insert_closure_pin(&mut pins.lines, line, &mut count, maximum, &mut check)?;
                }
                if let Some(execution) = task.resumed_from {
                    insert_closure_pin(
                        &mut pins.executions,
                        execution,
                        &mut count,
                        maximum,
                        &mut check,
                    )?;
                }
                if include_task_consumers {
                    for consumer in &task.consumers {
                        check_pin_closure(&mut check)?;
                        insert_closure_pin(
                            &mut pins.situations,
                            consumer.consumer.situation,
                            &mut count,
                            maximum,
                            &mut check,
                        )?;
                        check_pin_closure(&mut check)?;
                    }
                }
                match task.status {
                    TaskStatus::Completed(o) => {
                        insert_closure_pin(
                            &mut pins.observations,
                            o,
                            &mut count,
                            maximum,
                            &mut check,
                        )?;
                    }
                    TaskStatus::Paused { evidence, .. }
                    | TaskStatus::RetiredPaused { evidence, .. } => {
                        if let Some(observation) = evidence {
                            insert_closure_pin(
                                &mut pins.observations,
                                observation,
                                &mut count,
                                maximum,
                                &mut check,
                            )?;
                        }
                    }
                    _ => {}
                }
                check_pin_closure(&mut check)?;
            }
            for (observation, situations) in &self.dependencies.dependents {
                check_pin_closure(&mut check)?;
                let mut retained = false;
                for situation in situations {
                    check_pin_closure(&mut check)?;
                    if pins.situations.contains(situation) {
                        retained = true;
                        break;
                    }
                    check_pin_closure(&mut check)?;
                }
                if retained {
                    insert_closure_pin(
                        &mut pins.observations,
                        *observation,
                        &mut count,
                        maximum,
                        &mut check,
                    )?;
                }
                check_pin_closure(&mut check)?;
            }
            check_pin_closure(&mut check)?;
            if pins.count() == before {
                break;
            }
        }
        self.validate_pins_inner(&pins, &mut check)?;
        check_pin_closure(&mut check)?;
        Ok(pins)
    }
    fn protected_pins(&self, manager: &ArchiveManager) -> Result<StorePins, StoreError> {
        let mut pins = manager.pins.clone();
        pins.extend(&self.review_pin_union()?);
        pins.situations.extend(self.root);
        for loaded in manager.load_pins.values() {
            pins.extend(loaded);
        }
        for (id, task) in self.tasks.tasks.entries() {
            // CancellationRequested is still physical work, never quiescent.
            let active = matches!(
                task.status,
                TaskStatus::InFlight
                    | TaskStatus::CancellationRequested
                    | TaskStatus::Paused { .. }
            );
            let live_consumer = task.consumers.iter().any(|c| {
                !c.cancelled
                    && !c.consumed
                    && c.consumer.generation == self.generation
                    && self.situations.get(c.consumer.situation).is_ok()
            });
            if active || live_consumer {
                pins.executions.insert(ExecutionId(id));
                for c in &task.consumers {
                    if self.situations.get(c.consumer.situation).is_ok() {
                        pins.situations.insert(c.consumer.situation);
                    }
                }
            }
        }
        self.expand_pins(pins)
    }
    fn bundle_for(&self, selection: &StorePins, generation: u64) -> Result<Bundle, StoreError> {
        let copied = self.expand_pins(selection.clone())?;
        let mut bundle = Bundle::new(generation);
        for id in &copied.states {
            bundle
                .states
                .push(StateWire::encode(*id, self.states.get(*id)?)?);
        }
        for id in &copied.lines {
            bundle
                .lines
                .push(LineWire::encode(*id, self.lines.get(*id)?));
        }
        for id in &copied.observations {
            bundle
                .observations
                .push((*id, ObservationWire::encode(self.observations.get(*id)?)?));
        }
        for id in &copied.situations {
            bundle
                .situations
                .push((*id, self.situations.get(*id)?.clone()));
        }
        for id in &copied.executions {
            bundle.tasks.push((*id, self.tasks.get(*id)?.clone()));
        }
        for (observation, ids) in &self.dependencies.dependents {
            if copied.observations.contains(observation)
                || ids.iter().any(|id| copied.situations.contains(id))
            {
                bundle
                    .dependencies
                    .push((*observation, ids.iter().copied().collect()));
            }
        }
        bundle.checked_pvs = self
            .checked_cpu_pvs
            .keys()
            .filter(|line| copied.lines.contains(line))
            .copied()
            .collect();
        Ok(bundle)
    }
    fn pressure(&self) -> bool {
        let stats = self.hot_stats();
        let l = self.limits;
        [
            (stats.states, l.states),
            (stats.lines, l.line_chunks),
            (stats.situations, l.situations),
            (stats.observations, l.observations),
            (stats.dependencies, l.dependency_edges),
            (stats.executions, l.executions),
            (stats.consumers, l.consumers),
            (stats.history_bytes, l.retained_history_bytes),
        ]
        .into_iter()
        .any(|(used, maximum)| maximum != 0 && (used as u128) * 5 >= (maximum as u128) * 4)
    }
    pub fn archive_pressure(&self) -> bool {
        self.pressure()
    }
    pub fn reclaim_if_needed(
        &mut self,
        budget: ArchiveIoBudget,
    ) -> Result<Option<ArchiveReceipt>, StoreError> {
        self.reject_archive_allocation_scope()?;
        if !self.pressure() {
            return Ok(None);
        }
        self.archive_inactive(budget)
    }
    pub fn archive_inactive(
        &mut self,
        mut budget: ArchiveIoBudget,
    ) -> Result<Option<ArchiveReceipt>, StoreError> {
        self.archive_inactive_with_budget(&mut budget)
    }
    pub fn archive_inactive_with_budget(
        &mut self,
        budget: &mut ArchiveIoBudget,
    ) -> Result<Option<ArchiveReceipt>, StoreError> {
        self.reject_archive_allocation_scope()?;
        self.archive_inactive_inner(budget)
    }
    fn archive_inactive_inner(
        &mut self,
        budget: &mut ArchiveIoBudget,
    ) -> Result<Option<ArchiveReceipt>, StoreError> {
        budget.check()?;
        let mut manager = self.archive.take().ok_or(StoreError::ArchiveDisabled)?;
        manager.used = true;
        let baseline = manager.snapshot();
        let result = (|| {
            manager.verify_hot_owner(self)?;
            let pinned = self.protected_pins(&manager)?;
            let selection = StorePins {
                states: self
                    .states
                    .snapshots
                    .entries()
                    .map(|(id, _)| StateId(id))
                    .filter(|id| !pinned.states.contains(id))
                    .collect(),
                lines: self
                    .lines
                    .chunks
                    .entries()
                    .map(|(id, _)| LineId(id))
                    .filter(|id| !pinned.lines.contains(id))
                    .collect(),
                observations: self
                    .observations
                    .observations
                    .entries()
                    .map(|(id, _)| ObservationId(id))
                    .filter(|id| !pinned.observations.contains(id))
                    .collect(),
                situations: self
                    .situations
                    .iter()
                    .map(|(id, _)| id)
                    .filter(|id| !pinned.situations.contains(id))
                    .collect(),
                executions: self
                    .tasks
                    .tasks
                    .entries()
                    .map(|(id, _)| ExecutionId(id))
                    .filter(|id| !pinned.executions.contains(id))
                    .collect(),
            };
            if selection.count() == 0 {
                return Err(StoreError::PinSaturated("all hot records pinned"));
            }
            for id in &selection.situations {
                id.generation
                    .checked_add(1)
                    .ok_or(StoreError::RevisionExhausted)?;
            }
            let bundle = self.bundle_for(&selection, manager.next_generation)?;
            let mut receipt = manager.commit(&bundle, budget, self.hot_stats())?;
            let capacity_before = self.owned_archive_capacity_subset();
            // Every operation below removes an already validated hot record;
            // no RAM is released until committed bytes were reread and checked.
            for id in &selection.states {
                if let Some(state) = self.states.snapshots.remove(id.0) {
                    self.states.history_bytes -=
                        state.retained_history_bytes().expect("admitted charge");
                }
            }
            self.states.index.retain(|_, ids| {
                ids.retain(|id| !selection.states.contains(id));
                !ids.is_empty()
            });
            for id in &selection.lines {
                self.lines.chunks.remove(id.0);
                self.checked_cpu_pvs.remove(id);
            }
            self.lines
                .roots
                .retain(|_, id| !selection.lines.contains(id));
            self.lines.index.retain(|(parent, _), id| {
                !selection.lines.contains(parent) && !selection.lines.contains(id)
            });
            for id in &selection.observations {
                self.observations.observations.remove(id.0);
            }
            for id in &selection.situations {
                self.situations.remove(*id)?;
            }
            self.dependencies.dependents.retain(|o, ids| {
                if selection.observations.contains(o) {
                    return false;
                }
                ids.retain(|s| !selection.situations.contains(s));
                !ids.is_empty()
            });
            self.dependencies.edges = self
                .dependencies
                .dependents
                .values()
                .map(BTreeSet::len)
                .sum();
            for id in &selection.executions {
                if let Some(task) = self.tasks.tasks.remove(id.0) {
                    for consumer in task.consumers {
                        self.tasks.consumers.remove(&consumer.consumer.id);
                    }
                }
            }
            self.tasks
                .latest
                .retain(|_, id| !selection.executions.contains(id));
            self.tasks
                .owned_instances
                .retain(|id| !selection.executions.contains(id));
            self.states.snapshots.compact();
            self.states.index.shrink_to_fit();
            self.lines.chunks.compact();
            self.observations.observations.compact();
            self.tasks.tasks.compact();
            let capacity_after = self.owned_archive_capacity_subset();
            manager.note(|meter| {
                add_counter(&mut meter.ram_reclaim_events, 1, &mut meter.complete);
                if meter.last_verified_generation == Some(receipt.generation) {
                    add_counter(
                        &mut meter.integrity_before_reclaim_events,
                        1,
                        &mut meter.complete,
                    );
                } else {
                    add_counter(&mut meter.unverified_reclaim_events, 1, &mut meter.complete);
                    meter.complete = false;
                }
                let released = capacity_before
                    .zip(capacity_after)
                    .and_then(|(before, after)| before.checked_sub(after));
                meter.ram_released_bytes = released.and_then(|released| {
                    meter
                        .ram_released_bytes
                        .and_then(|total| total.checked_add(released))
                });
            });
            receipt.after = self.hot_stats();
            Ok(Some(receipt))
        })();
        if let Err(error) = &result {
            manager.failure_after(baseline, error);
        }
        self.archive = Some(manager);
        result
    }
    /// Exact requested capacities of hot record arrays, state index value Vecs,
    /// line index key Vecs, task root/consumer Vecs, cpu_condition Strings and
    /// ExternalCheckerReport Box bodies. Namespace strings nested in identities,
    /// shared Position history and hash/BTree nodes are outside this subset.
    fn owned_archive_capacity_subset(&self) -> Option<u64> {
        let mut total = self
            .states
            .snapshots
            .owned_capacity_bytes()?
            .checked_add(self.lines.chunks.owned_capacity_bytes()?)?
            .checked_add(self.observations.observations.owned_capacity_bytes()?)?
            .checked_add(self.tasks.tasks.owned_capacity_bytes()?)?
            .checked_add(capacity_bytes(
                self.situations.slots.capacity(),
                std::mem::size_of::<SituationSlot>(),
            )?)?;
        for ids in self.states.index.values() {
            total = total.checked_add(capacity_bytes(
                ids.capacity(),
                std::mem::size_of::<StateId>(),
            )?)?;
        }
        for (_, moves) in self.lines.index.keys() {
            total = total.checked_add(capacity_bytes(
                moves.capacity(),
                std::mem::size_of::<Move16>(),
            )?)?;
        }
        for observation in self.observations.observations.iter() {
            total = total.checked_add(observation_capacity(observation)?)?;
        }
        for task in self.tasks.tasks.iter() {
            total = total.checked_add(task_capacity(task)?)?;
        }
        for key in self.tasks.latest.keys() {
            total = total.checked_add(task_key_capacity(key)?)?;
        }
        Some(total)
    }
    fn pinned_capacity_subset(&self, pins: &StorePins) -> Option<u64> {
        let mut total = 0u64;
        for id in &pins.observations {
            let observation = self.observations.get(*id).ok()?;
            total = total.checked_add(observation_capacity(observation)?)?;
        }
        for id in &pins.executions {
            let task = self.tasks.get(*id).ok()?;
            total = total.checked_add(task_capacity(task)?)?;
        }
        Some(total)
    }
    /// Store-independent engine payloads commit through the same quotas and
    /// checksum barrier. The parent may trim only after receiving this receipt.
    pub fn archive_engine_records(
        &mut self,
        records: &[crate::pals::engine::RoleRecord],
        nodes: &[EngineArchiveNode],
        budget: ArchiveIoBudget,
    ) -> Result<EngineArchiveReceipt, StoreError> {
        self.archive_engine_records_with_pins(records, nodes, StorePins::default(), budget)
    }
    pub fn archive_engine_records_with_pins(
        &mut self,
        records: &[crate::pals::engine::RoleRecord],
        nodes: &[EngineArchiveNode],
        selection: StorePins,
        mut budget: ArchiveIoBudget,
    ) -> Result<EngineArchiveReceipt, StoreError> {
        self.archive_engine_records_with_pins_and_budget(records, nodes, selection, &mut budget)
    }
    pub fn archive_engine_records_with_pins_and_budget(
        &mut self,
        records: &[crate::pals::engine::RoleRecord],
        nodes: &[EngineArchiveNode],
        mut selection: StorePins,
        budget: &mut ArchiveIoBudget,
    ) -> Result<EngineArchiveReceipt, StoreError> {
        self.reject_archive_allocation_scope()?;
        budget.check()?;
        if records.len() > self.limits.observations || nodes.len() > self.limits.situations {
            return Err(StoreError::Capacity("engine archive record count"));
        }
        for record in records {
            if record.line.len() > self.limits.line_plies {
                return Err(StoreError::Capacity("engine archive line"));
            }
            selection.states.insert(record.origin_state);
            selection.observations.extend(record.cpu_observation);
        }
        for node in nodes {
            if node.edges.len() > self.limits.root_moves_per_task || node.metadata.len() > 4096 {
                return Err(StoreError::Capacity("engine archive node"));
            }
            selection.states.insert(node.state);
            selection.situations.insert(node.situation);
            for (movement, state) in &node.edges {
                movement.unpack()?;
                selection.states.extend(state);
            }
        }
        let actual_root = self.root.and_then(|root| {
            self.situations.get(root).ok().and_then(|situation| {
                nodes
                    .iter()
                    .any(|node| node.state == situation.state && node.situation == root)
                    .then_some((root, situation.state, self.generation))
            })
        });
        let mut manager = self.archive.take().ok_or(StoreError::ArchiveDisabled)?;
        manager.used = true;
        let baseline = manager.snapshot();
        let result = (|| {
            manager.verify_hot_owner(self)?;
            let mut bundle = self.bundle_for(&selection, manager.next_generation)?;
            bundle.engine_records = records
                .iter()
                .map(RoleRecordWire::encode)
                .collect::<Result<_, _>>()?;
            bundle.engine_nodes = nodes.to_vec();
            let archive = manager.commit(&bundle, budget, self.hot_stats())?;
            Ok(EngineArchiveReceipt {
                archive,
                records: records.len(),
                nodes: nodes.len(),
                actual_root,
            })
        })();
        if let Err(error) = &result {
            manager.failure_after(baseline, error);
        }
        self.archive = Some(manager);
        result
    }
    /// Final idle-game checkpoint, including raw cancelled/unconsumed facts.
    /// This changes no task status, acceptance, consumption or hot ownership.
    /// The concrete caller owns the physical fence; this guard checks only the
    /// original Store's logical work and private IDs before the existing codec.
    pub(crate) fn archive_all_hot_engine_records_with_budget(
        &mut self,
        records: &[crate::pals::engine::RoleRecord],
        nodes: &[EngineArchiveNode],
        budget: &mut ArchiveIoBudget,
    ) -> Result<EngineArchiveReceipt, StoreError> {
        self.reject_archive_allocation_scope()?;
        budget.check()?;
        if self.review_pin_count()? != 0 || self.tasks.active_states().next().is_some() {
            return Err(StoreError::InvalidConditions(
                "final archive checkpoint still owns logical work",
            ));
        }
        let selection = StorePins {
            states: self
                .states
                .snapshots
                .entries()
                .map(|(id, _)| StateId(id))
                .collect(),
            lines: self
                .lines
                .chunks
                .entries()
                .map(|(id, _)| LineId(id))
                .collect(),
            observations: self
                .observations
                .observations
                .entries()
                .map(|(id, _)| ObservationId(id))
                .collect(),
            situations: self.situations.iter().map(|(id, _)| id).collect(),
            executions: self
                .tasks
                .tasks
                .entries()
                .map(|(id, _)| ExecutionId(id))
                .collect(),
        };
        self.archive_engine_records_with_pins_and_budget(records, nodes, selection, budget)
    }
    pub fn lookup_cold_state(
        &self,
        snapshot: &PositionSnapshot,
        mut budget: ArchiveIoBudget,
    ) -> Result<Option<ColdHandle<StateId>>, StoreError> {
        self.lookup_cold_state_with_budget(snapshot, &mut budget)
    }
    pub fn lookup_cold_state_with_budget(
        &self,
        snapshot: &PositionSnapshot,
        budget: &mut ArchiveIoBudget,
    ) -> Result<Option<ColdHandle<StateId>>, StoreError> {
        self.with_archive_read_budget(budget, |budget| {
            self.lookup_cold_state_inner(snapshot, budget)
        })
    }
    fn lookup_cold_state_inner(
        &self,
        snapshot: &PositionSnapshot,
        budget: &mut ArchiveIoBudget,
    ) -> Result<Option<ColdHandle<StateId>>, StoreError> {
        self.reject_archive_allocation_scope()?;
        budget.check()?;
        let manager = self.archive.as_ref().ok_or(StoreError::ArchiveDisabled)?;
        manager.verify_directory_boundary()?;
        for entry in manager
            .game_entries()
            .map_err(|e| io("archive lookup directory", e))?
        {
            budget.check()?;
            let entry = entry.map_err(|e| io("archive lookup entry", e))?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(number) = name
                .strip_prefix("seg-")
                .and_then(|n| n.strip_suffix(".arc"))
            else {
                continue;
            };
            let generation = number
                .parse::<u64>()
                .map_err(|_| StoreError::ArchiveIntegrity("archive segment name"))?;
            if generation >= manager.next_generation {
                continue;
            }
            let (bundle, checksum) = manager.read(generation, None, budget)?;
            self.check_bundle_counts(&bundle)?;
            for state in bundle.states {
                budget.check()?;
                if state.decode()?.same_state(snapshot) {
                    return Ok(Some(ColdHandle {
                        owner: Arc::downgrade(&manager.identity),
                        generation,
                        checksum,
                        record: state.id,
                        _kind: PhantomData,
                    }));
                }
            }
        }
        Ok(None)
    }
    pub fn lookup_cold_state_id(
        &self,
        state: StateId,
        mut budget: ArchiveIoBudget,
    ) -> Result<Option<ColdHandle<StateId>>, StoreError> {
        self.lookup_cold_state_id_with_budget(state, &mut budget)
    }
    pub fn lookup_cold_state_id_with_budget(
        &self,
        state: StateId,
        budget: &mut ArchiveIoBudget,
    ) -> Result<Option<ColdHandle<StateId>>, StoreError> {
        self.with_archive_read_budget(budget, |budget| {
            self.lookup_cold_state_id_inner(state, budget)
        })
    }
    fn lookup_cold_state_id_inner(
        &self,
        state: StateId,
        budget: &mut ArchiveIoBudget,
    ) -> Result<Option<ColdHandle<StateId>>, StoreError> {
        self.reject_archive_allocation_scope()?;
        budget.check()?;
        let manager = self.archive.as_ref().ok_or(StoreError::ArchiveDisabled)?;
        if state.0 >= self.states.snapshots.next_id() {
            return Err(StoreError::InvalidHandle("cold state serial"));
        }
        manager.verify_directory_boundary()?;
        for entry in manager
            .game_entries()
            .map_err(|e| io("archive state lookup directory", e))?
        {
            budget.check()?;
            let entry = entry.map_err(|e| io("archive state lookup entry", e))?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(number) = name
                .strip_prefix("seg-")
                .and_then(|n| n.strip_suffix(".arc"))
            else {
                continue;
            };
            let generation = number
                .parse::<u64>()
                .map_err(|_| StoreError::ArchiveIntegrity("archive segment name"))?;
            if generation >= manager.next_generation {
                continue;
            }
            let (bundle, checksum) = manager.read(generation, None, budget)?;
            self.check_bundle_counts(&bundle)?;
            if bundle.states.iter().any(|wire| wire.id == state) {
                return Ok(Some(ColdHandle {
                    owner: Arc::downgrade(&manager.identity),
                    generation,
                    checksum,
                    record: state,
                    _kind: PhantomData,
                }));
            }
        }
        Ok(None)
    }
    /// Return the newest committed engine payload for an original state serial.
    /// No partial lookup success is returned if its scan exhausts the budget.
    pub fn lookup_engine_archive(
        &self,
        state: StateId,
        mut budget: ArchiveIoBudget,
    ) -> Result<Option<EngineArchiveReceipt>, StoreError> {
        self.lookup_engine_archive_with_budget(state, &mut budget)
    }
    pub fn lookup_engine_archive_with_budget(
        &self,
        state: StateId,
        budget: &mut ArchiveIoBudget,
    ) -> Result<Option<EngineArchiveReceipt>, StoreError> {
        self.lookup_engine_archive_filtered_with_budget(state, budget, false)
    }
    /// A newer record-only segment cannot hide the newest complete Node DTO.
    pub fn lookup_engine_node_archive(
        &self,
        state: StateId,
        mut budget: ArchiveIoBudget,
    ) -> Result<Option<EngineArchiveReceipt>, StoreError> {
        self.lookup_engine_node_archive_with_budget(state, &mut budget)
    }
    pub fn lookup_engine_node_archive_with_budget(
        &self,
        state: StateId,
        budget: &mut ArchiveIoBudget,
    ) -> Result<Option<EngineArchiveReceipt>, StoreError> {
        self.lookup_engine_archive_filtered_with_budget(state, budget, true)
    }
    fn lookup_engine_archive_filtered_with_budget(
        &self,
        state: StateId,
        budget: &mut ArchiveIoBudget,
        require_node: bool,
    ) -> Result<Option<EngineArchiveReceipt>, StoreError> {
        self.with_archive_read_budget(budget, |budget| {
            self.lookup_engine_archive_filtered_inner(state, budget, require_node)
        })
    }
    fn lookup_engine_archive_filtered_inner(
        &self,
        state: StateId,
        budget: &mut ArchiveIoBudget,
        require_node: bool,
    ) -> Result<Option<EngineArchiveReceipt>, StoreError> {
        self.reject_archive_allocation_scope()?;
        budget.check()?;
        let initial = budget.max_bytes;
        let manager = self.archive.as_ref().ok_or(StoreError::ArchiveDisabled)?;
        let mut newest = None;
        manager.verify_directory_boundary()?;
        for entry in manager
            .game_entries()
            .map_err(|e| io("engine archive lookup directory", e))?
        {
            budget.check()?;
            let entry = entry.map_err(|e| io("engine archive lookup entry", e))?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(number) = name
                .strip_prefix("seg-")
                .and_then(|n| n.strip_suffix(".arc"))
            else {
                continue;
            };
            let generation = number
                .parse::<u64>()
                .map_err(|_| StoreError::ArchiveIntegrity("archive segment name"))?;
            if generation >= manager.next_generation {
                continue;
            }
            let (bundle, checksum) = manager.read(generation, None, budget)?;
            self.check_bundle_counts(&bundle)?;
            if (bundle.engine_nodes.iter().any(|node| node.state == state)
                || !require_node
                    && bundle
                        .engine_records
                        .iter()
                        .any(|r| r.origin_state() == state))
                && newest
                    .as_ref()
                    .is_none_or(|r: &EngineArchiveReceipt| r.archive.generation < generation)
            {
                newest = Some(EngineArchiveReceipt {
                    records: bundle.engine_records.len(),
                    nodes: bundle.engine_nodes.len(),
                    actual_root: None,
                    archive: ArchiveReceipt {
                        owner: Arc::downgrade(&manager.identity),
                        generation,
                        checksum,
                        before: self.hot_stats(),
                        after: self.hot_stats(),
                        committed_bytes: manager.game_file_size(
                            manager
                                .path(generation)
                                .file_name()
                                .expect("private segment name"),
                        )?,
                        io_bytes: 0,
                    },
                });
            }
        }
        if let Some(receipt) = newest.as_mut() {
            receipt.archive.io_bytes = initial - budget.max_bytes;
        }
        Ok(newest)
    }
    fn check_bundle_counts(&self, bundle: &Bundle) -> Result<(), StoreError> {
        let l = self.limits;
        if bundle.states.len() > l.states
            || bundle.lines.len() > l.line_chunks
            || bundle.observations.len() > l.observations
            || bundle.situations.len() > l.situations
            || bundle.tasks.len() > l.executions
            || bundle.engine_records.len() > l.observations
            || bundle.engine_nodes.len() > l.situations
            || bundle.checked_pvs.len() > l.line_chunks
            || bundle
                .dependencies
                .iter()
                .map(|(_, ids)| ids.len())
                .sum::<usize>()
                > l.dependency_edges
        {
            return Err(StoreError::ArchiveIntegrity("archive index counts"));
        }
        if bundle
            .states
            .iter()
            .map(|s| s.id)
            .collect::<BTreeSet<_>>()
            .len()
            != bundle.states.len()
            || bundle
                .lines
                .iter()
                .map(|l| l.id)
                .collect::<BTreeSet<_>>()
                .len()
                != bundle.lines.len()
            || bundle
                .observations
                .iter()
                .map(|(id, _)| *id)
                .collect::<BTreeSet<_>>()
                .len()
                != bundle.observations.len()
            || bundle
                .tasks
                .iter()
                .map(|(id, _)| *id)
                .collect::<BTreeSet<_>>()
                .len()
                != bundle.tasks.len()
            || bundle
                .situations
                .iter()
                .map(|(id, _)| *id)
                .collect::<BTreeSet<_>>()
                .len()
                != bundle.situations.len()
        {
            return Err(StoreError::ArchiveIntegrity("duplicate archive serial"));
        }
        Ok(())
    }
    pub fn release_loaded(&mut self, pin: ArchiveLoadPin) -> Result<(), StoreError> {
        let manager = self.archive.as_mut().ok_or(StoreError::ArchiveDisabled)?;
        if !pin.owner.ptr_eq(&Arc::downgrade(&manager.identity))
            || manager.load_pins.remove(&pin.serial).is_none()
        {
            return Err(StoreError::InvalidHandle("archive load pin"));
        }
        manager.note(|_| {});
        Ok(())
    }
    /// The engine first commits the old root topology. This attested handoff
    /// expires only logical root acceptance, retaining physical task ownership.
    pub fn retire_actual_root_for_archive(
        &mut self,
        receipt: &EngineArchiveReceipt,
    ) -> Result<Option<SituationId>, StoreError> {
        let manager = self.archive.as_ref().ok_or(StoreError::ArchiveDisabled)?;
        manager.verify_owner(&receipt.archive.state_handle(StateId(0)))?;
        let Some(root) = self.root else {
            return Ok(None);
        };
        let state = self.situations.get(root)?.state;
        if receipt.actual_root != Some((root, state, self.generation)) {
            return Err(StoreError::InvalidConditions(
                "root retirement receipt does not attest actual root",
            ));
        }
        let next = self
            .generation
            .checked_add(1)
            .ok_or(StoreError::RevisionExhausted)?;
        self.generation = next;
        self.root = None;
        Ok(Some(root))
    }

    /// Restore only the requested record's exact dependency closure. Hot reads
    /// never perform I/O implicitly. The returned pin is explicitly releasable.
    pub fn pin_load<T: ArchiveRecordKind>(
        &mut self,
        handle: &ColdHandle<T>,
        mut budget: ArchiveIoBudget,
    ) -> Result<LoadedArchive, StoreError> {
        self.pin_load_with_budget(handle, &mut budget)
    }
    pub fn pin_load_with_budget<T: ArchiveRecordKind>(
        &mut self,
        handle: &ColdHandle<T>,
        original: &mut ArchiveIoBudget,
    ) -> Result<LoadedArchive, StoreError> {
        self.reject_archive_allocation_scope()?;
        let mut budget = self
            .archive
            .as_ref()
            .ok_or(StoreError::ArchiveDisabled)?
            .load_budget(*original)?;
        let initial = budget.max_bytes;
        let started = Instant::now();
        let mut manager = self.archive.take().ok_or(StoreError::ArchiveDisabled)?;
        manager.used = true;
        let baseline = manager.snapshot();
        manager.note(|meter| add_counter(&mut meter.loads_requested, 1, &mut meter.complete));
        let result = (|| {
            manager.verify_owner(handle)?;
            let initial = budget.max_bytes;
            let (bundle, _) =
                manager.read(handle.generation, Some(handle.checksum), &mut budget)?;
            let mut pins = StorePins::default();
            handle.record.seed(&mut pins);
            self.restore_bundle(
                &mut manager,
                bundle,
                pins,
                false,
                budget,
                initial - budget.max_bytes,
            )
        })();
        let bytes = initial - budget.max_bytes;
        original.max_bytes -= bytes;
        manager.finish_read(baseline, started, bytes, true, &result);
        self.archive = Some(manager);
        result
    }
    pub fn load_engine_archive(
        &mut self,
        receipt: &EngineArchiveReceipt,
        mut budget: ArchiveIoBudget,
    ) -> Result<LoadedArchive, StoreError> {
        self.load_engine_archive_with_budget(receipt, &mut budget)
    }
    pub fn load_engine_archive_with_budget(
        &mut self,
        receipt: &EngineArchiveReceipt,
        original: &mut ArchiveIoBudget,
    ) -> Result<LoadedArchive, StoreError> {
        self.reject_archive_allocation_scope()?;
        let mut budget = self
            .archive
            .as_ref()
            .ok_or(StoreError::ArchiveDisabled)?
            .load_budget(*original)?;
        let initial = budget.max_bytes;
        let started = Instant::now();
        let mut manager = self.archive.take().ok_or(StoreError::ArchiveDisabled)?;
        manager.used = true;
        let baseline = manager.snapshot();
        manager.note(|meter| add_counter(&mut meter.loads_requested, 1, &mut meter.complete));
        let result = (|| {
            let handle = receipt.archive.state_handle(StateId(0));
            manager.verify_owner(&handle)?;
            let initial = budget.max_bytes;
            let (bundle, _) =
                manager.read(handle.generation, Some(handle.checksum), &mut budget)?;
            let mut pins = StorePins::default();
            for record in &bundle.engine_records {
                record.add_pins(&mut pins);
            }
            for node in &bundle.engine_nodes {
                pins.states.insert(node.state);
                pins.situations.insert(node.situation);
                for (_, state) in &node.edges {
                    pins.states.extend(state);
                }
            }
            // Opaque Node metadata can refer to raw CPU/model evidence even
            // when no equivalent Situation dependency edge was installed.
            for (id, observation) in &bundle.observations {
                if bundle
                    .engine_nodes
                    .iter()
                    .any(|node| node.state == observation.state_id())
                {
                    pins.observations.insert(*id);
                }
            }
            self.restore_bundle(
                &mut manager,
                bundle,
                pins,
                true,
                budget,
                initial - budget.max_bytes,
            )
        })();
        let bytes = initial - budget.max_bytes;
        original.max_bytes -= bytes;
        manager.finish_read(baseline, started, bytes, true, &result);
        self.archive = Some(manager);
        result
    }
    /// Inspect bounded immutable engine payloads without admitting hot records.
    pub fn read_engine_archive(
        &self,
        receipt: &EngineArchiveReceipt,
        mut budget: ArchiveIoBudget,
    ) -> Result<
        (
            Vec<crate::pals::engine::RoleRecord>,
            Vec<EngineArchiveNode>,
            u64,
        ),
        StoreError,
    > {
        self.read_engine_archive_with_budget(receipt, &mut budget)
    }
    pub fn read_engine_archive_with_budget(
        &self,
        receipt: &EngineArchiveReceipt,
        original: &mut ArchiveIoBudget,
    ) -> Result<
        (
            Vec<crate::pals::engine::RoleRecord>,
            Vec<EngineArchiveNode>,
            u64,
        ),
        StoreError,
    > {
        self.reject_archive_allocation_scope()?;
        let manager = self.archive.as_ref().ok_or(StoreError::ArchiveDisabled)?;
        let mut budget = manager.load_budget(*original)?;
        let initial = budget.max_bytes;
        let started = Instant::now();
        let baseline = manager.snapshot();
        manager.note(|meter| add_counter(&mut meter.loads_requested, 1, &mut meter.complete));
        let result = (|| {
            let handle = receipt.archive.state_handle(StateId(0));
            manager.verify_owner(&handle)?;
            let initial = budget.max_bytes;
            let (bundle, _) =
                manager.read(handle.generation, Some(handle.checksum), &mut budget)?;
            self.check_bundle_counts(&bundle)?;
            let records = bundle
                .engine_records
                .into_iter()
                .map(RoleRecordWire::decode)
                .collect::<Result<Vec<_>, _>>()?;
            for record in &records {
                if record.line.len() > self.limits.line_plies {
                    return Err(StoreError::ArchiveIntegrity("engine line bounds"));
                }
            }
            for node in &bundle.engine_nodes {
                if node.metadata.len() > 4096 || node.edges.len() > self.limits.root_moves_per_task
                {
                    return Err(StoreError::ArchiveIntegrity("engine node bounds"));
                }
                for (movement, _) in &node.edges {
                    movement.unpack()?;
                }
            }
            budget.check()?;
            Ok((records, bundle.engine_nodes, initial - budget.max_bytes))
        })();
        let bytes = initial - budget.max_bytes;
        original.max_bytes -= bytes;
        manager.finish_read(baseline, started, bytes, true, &result);
        result
    }
    /// Load the selected archived node's reachable continuation subtree. The
    /// remaining historical node graph stays cold and consumes no hot slots.
    pub fn load_engine_archive_for_state(
        &mut self,
        receipt: &EngineArchiveReceipt,
        state: StateId,
        mut budget: ArchiveIoBudget,
    ) -> Result<LoadedArchive, StoreError> {
        self.load_engine_archive_for_state_with_budget(receipt, state, &mut budget)
    }
    pub fn load_engine_archive_for_state_with_budget(
        &mut self,
        receipt: &EngineArchiveReceipt,
        state: StateId,
        original: &mut ArchiveIoBudget,
    ) -> Result<LoadedArchive, StoreError> {
        self.reject_archive_allocation_scope()?;
        let mut budget = self
            .archive
            .as_ref()
            .ok_or(StoreError::ArchiveDisabled)?
            .load_budget(*original)?;
        let initial = budget.max_bytes;
        let started = Instant::now();
        let mut manager = self.archive.take().ok_or(StoreError::ArchiveDisabled)?;
        manager.used = true;
        let baseline = manager.snapshot();
        manager.note(|meter| add_counter(&mut meter.loads_requested, 1, &mut meter.complete));
        let result = (|| {
            let handle = receipt.archive.state_handle(state);
            manager.verify_owner(&handle)?;
            let initial = budget.max_bytes;
            let (mut bundle, _) =
                manager.read(handle.generation, Some(handle.checksum), &mut budget)?;
            self.check_bundle_counts(&bundle)?;
            if !bundle.engine_nodes.iter().any(|n| n.state == state) {
                return Err(StoreError::InvalidHandle("cold engine root membership"));
            }
            let mut reachable = BTreeSet::from([state]);
            loop {
                budget.check()?;
                let before = reachable.len();
                for node in &bundle.engine_nodes {
                    if reachable.contains(&node.state) {
                        for (_, child) in &node.edges {
                            reachable.extend(child);
                        }
                    }
                }
                if reachable.len() > self.limits.states {
                    return Err(StoreError::PinSaturated("engine subtree states"));
                }
                if before == reachable.len() {
                    break;
                }
            }
            bundle
                .engine_nodes
                .retain(|node| reachable.contains(&node.state));
            bundle
                .engine_records
                .retain(|record| reachable.contains(&record.origin_state()));
            let mut pins = StorePins {
                states: reachable,
                ..StorePins::default()
            };
            for node in &bundle.engine_nodes {
                pins.situations.insert(node.situation);
            }
            for record in &bundle.engine_records {
                record.add_pins(&mut pins);
            }
            for (id, observation) in &bundle.observations {
                if pins.states.contains(&observation.state_id()) {
                    pins.observations.insert(*id);
                }
            }
            self.restore_bundle(
                &mut manager,
                bundle,
                pins,
                true,
                budget,
                initial - budget.max_bytes,
            )
        })();
        let bytes = initial - budget.max_bytes;
        original.max_bytes -= bytes;
        manager.finish_read(baseline, started, bytes, true, &result);
        self.archive = Some(manager);
        result
    }
    fn restore_bundle(
        &mut self,
        manager: &mut ArchiveManager,
        bundle: Bundle,
        mut pins: StorePins,
        include_engine: bool,
        budget: ArchiveIoBudget,
        io_bytes: u64,
    ) -> Result<LoadedArchive, StoreError> {
        budget.check()?;
        self.check_bundle_counts(&bundle)?;
        if !manager
            .state_owner
            .ptr_eq(&Arc::downgrade(&self.states.identity))
            || !manager
                .line_owner
                .ptr_eq(&Arc::downgrade(&self.lines.identity))
        {
            return Err(StoreError::InvalidHandle("archive hot-store owner changed"));
        }
        if manager.load_pins.len() >= manager.config.max_load_pins {
            return Err(StoreError::PinSaturated("archive load pins"));
        }
        let pin_serial = manager.next_pin;
        let next_pin = pin_serial
            .checked_add(1)
            .ok_or(StoreError::RevisionExhausted)?;
        // Membership of the seed is checked even if its numeric serial currently
        // exists hot, so a typed handle cannot attest another record in a segment.
        for id in &pins.states {
            if !bundle.states.iter().any(|s| s.id == *id) {
                return Err(StoreError::InvalidHandle("cold state membership"));
            }
        }
        for id in &pins.lines {
            if !bundle.lines.iter().any(|l| l.id == *id) {
                return Err(StoreError::InvalidHandle("cold line membership"));
            }
        }
        for id in &pins.observations {
            if !bundle.observations.iter().any(|(o, _)| o == id) {
                return Err(StoreError::InvalidHandle("cold observation membership"));
            }
        }
        for id in &pins.executions {
            if !bundle.tasks.iter().any(|(e, _)| e == id) {
                return Err(StoreError::InvalidHandle("cold execution membership"));
            }
        }
        for id in &pins.situations {
            if !bundle.situations.iter().any(|(s, _)| s == id) {
                return Err(StoreError::InvalidHandle("cold situation membership"));
            }
        }
        loop {
            budget.check()?;
            let before = pins.count();
            for id in pins.lines.clone() {
                if let Some(line) = bundle.lines.iter().find(|line| line.id == id) {
                    pins.states.insert(line.state);
                    pins.lines.extend(line.parent);
                } else {
                    let line = self.lines.get(id)?;
                    pins.states.insert(line.start_state);
                    pins.lines.extend(line.parent);
                }
            }
            for id in pins.observations.clone() {
                if let Some((_, observation)) = bundle.observations.iter().find(|(o, _)| *o == id) {
                    observation.add_pins(&mut pins);
                } else {
                    let o = self.observations.get(id)?;
                    pins.states.insert(o.state);
                    pins.lines.extend(o.line);
                    pins.lines.extend(o.cpu_pv);
                    pins.observations.extend(o.supersedes);
                    pins.observations
                        .extend(o.score.model_dependencies().into_iter().flatten());
                    pins.executions.extend(o.execution);
                }
            }
            for id in pins.executions.clone() {
                let task = if let Some((_, task)) = bundle.tasks.iter().find(|(e, _)| *e == id) {
                    task
                } else {
                    self.tasks.get(id)?
                };
                pins.states.insert(task.key.state);
                pins.lines.extend(task.key.line);
                pins.executions.extend(task.resumed_from);
                match task.status {
                    TaskStatus::Completed(o) => {
                        pins.observations.insert(o);
                    }
                    TaskStatus::Paused { evidence, .. }
                    | TaskStatus::RetiredPaused { evidence, .. } => {
                        pins.observations.extend(evidence)
                    }
                    _ => {}
                }
            }
            for id in pins.situations.clone() {
                let situation =
                    if let Some((_, s)) = bundle.situations.iter().find(|(s, _)| *s == id) {
                        s
                    } else {
                        self.situations.get(id)?
                    };
                pins.states.insert(situation.state);
                pins.lines.insert(situation.focus);
                for (line, c) in situation.conclusions.iter() {
                    pins.lines.insert(*line);
                    pins.observations.extend(c.evidence);
                    if let ContinuationStatus::RepairedBy(line) = c.status {
                        pins.lines.insert(line);
                    }
                }
            }
            for (observation, situations) in &bundle.dependencies {
                if situations.iter().any(|id| pins.situations.contains(id)) {
                    pins.observations.insert(*observation);
                }
            }
            if pins.count() == before {
                break;
            }
        }
        for (count, maximum) in [
            (
                manager
                    .load_pins
                    .values()
                    .map(|p| p.states.len())
                    .sum::<usize>()
                    + pins.states.len(),
                self.limits.states,
            ),
            (
                manager
                    .load_pins
                    .values()
                    .map(|p| p.lines.len())
                    .sum::<usize>()
                    + pins.lines.len(),
                self.limits.line_chunks,
            ),
            (
                manager
                    .load_pins
                    .values()
                    .map(|p| p.observations.len())
                    .sum::<usize>()
                    + pins.observations.len(),
                self.limits.observations,
            ),
            (
                manager
                    .load_pins
                    .values()
                    .map(|p| p.situations.len())
                    .sum::<usize>()
                    + pins.situations.len(),
                self.limits.situations,
            ),
            (
                manager
                    .load_pins
                    .values()
                    .map(|p| p.executions.len())
                    .sum::<usize>()
                    + pins.executions.len(),
                self.limits.executions,
            ),
        ] {
            if count > maximum {
                return Err(StoreError::PinSaturated("load pin index"));
            }
        }
        let mut states = BTreeMap::new();
        let mut history = 0usize;
        for wire in &bundle.states {
            if pins.states.contains(&wire.id) {
                budget.check()?;
                if wire.id.0 >= self.states.snapshots.next_id() {
                    return Err(StoreError::InvalidHandle("cold state serial"));
                }
                if let Some(hot) = self.states.snapshots.get(wire.id.0) {
                    if StateWire::encode(wire.id, hot)? != *wire {
                        return Err(StoreError::ArchiveIntegrity("state serial collision"));
                    }
                } else {
                    history = history
                        .checked_add(wire.history_charge()?)
                        .ok_or(StoreError::Capacity("archive history"))?;
                    if self
                        .states
                        .history_bytes
                        .checked_add(history)
                        .is_none_or(|n| n > self.limits.retained_history_bytes)
                    {
                        return Err(StoreError::PinSaturated("loaded history"));
                    }
                    let snapshot = wire.decode()?;
                    if states.insert(wire.id, snapshot).is_some() {
                        return Err(StoreError::ArchiveIntegrity("duplicate state"));
                    }
                }
            }
        }
        for id in &pins.states {
            if !states.contains_key(id) {
                self.states.get(*id)?;
            }
        }
        let mut lines = BTreeMap::new();
        for wire in &bundle.lines {
            if pins.lines.contains(&wire.id) {
                if wire.id.0 >= self.lines.chunks.next_id() {
                    return Err(StoreError::InvalidHandle("cold line serial"));
                }
                let line = wire.decode(self.limits.line_plies)?;
                if let Some(hot) = self.lines.chunks.get(wire.id.0) {
                    if *hot != line {
                        return Err(StoreError::ArchiveIntegrity("line serial collision"));
                    }
                } else if lines.insert(wire.id, line).is_some() {
                    return Err(StoreError::ArchiveIntegrity("duplicate line"));
                }
            }
        }
        for id in &pins.lines {
            let line = lines
                .get(id)
                .or_else(|| self.lines.chunks.get(id.0))
                .ok_or(StoreError::InvalidHandle("cold line dependency"))?;
            if !pins.states.contains(&line.start_state) {
                return Err(StoreError::ArchiveIntegrity("line state closure"));
            }
            if let Some(parent) = line.parent {
                let p = lines
                    .get(&parent)
                    .or_else(|| self.lines.chunks.get(parent.0))
                    .ok_or(StoreError::InvalidHandle("cold line parent"))?;
                if p.start_state != line.start_state
                    || p.plies.checked_add(line.len) != Some(line.plies)
                {
                    return Err(StoreError::ArchiveIntegrity("line accumulated plies"));
                }
            }
        }
        let mut observations = BTreeMap::new();
        for (id, wire) in bundle.observations {
            if pins.observations.contains(&id) {
                budget.check()?;
                if id.0 >= self.observations.observations.next_id() {
                    return Err(StoreError::InvalidHandle("cold observation serial"));
                }
                let observation = wire.decode()?;
                if let Some(hot) = self.observations.observations.get(id.0) {
                    if *hot != observation {
                        return Err(StoreError::ArchiveIntegrity("observation serial collision"));
                    }
                } else if observations.insert(id, observation).is_some() {
                    return Err(StoreError::ArchiveIntegrity("duplicate observation"));
                }
            }
        }
        let mut tasks = BTreeMap::new();
        let mut consumers = 0usize;
        let mut task_validator = TaskTable::new(
            self.limits.executions,
            self.limits.consumers,
            self.limits.root_moves_per_task,
        );
        for (id, mut task) in bundle.tasks {
            if pins.executions.contains(&id) {
                budget.check()?;
                if id.0 >= self.tasks.tasks.next_id() {
                    return Err(StoreError::InvalidHandle("cold execution serial"));
                }
                if self.tasks.tasks.get(id.0).is_none() {
                    if matches!(task.key.question, TaskQuestion::ProfiledExternal { .. })
                        && task.status != TaskStatus::Failed
                    {
                        return Err(StoreError::ArchiveIntegrity(
                            "profiled external task was not raw terminal feedback",
                        ));
                    }
                    // Serde Vec growth is an allocation detail, not part of the
                    // retained exact task key. Admit its bounded ordered data.
                    task.key.root_moves = task.key.root_moves.into_boxed_slice().into_vec();
                    if matches!(
                        task.status,
                        TaskStatus::InFlight
                            | TaskStatus::CancellationRequested
                            | TaskStatus::Paused { .. }
                    ) {
                        return Err(StoreError::ArchiveIntegrity(
                            "active execution was archived",
                        ));
                    }
                    task_validator.request(
                        task.key.clone(),
                        TaskConsumer {
                            id: id.0 as u64,
                            situation: SituationId {
                                slot: 0,
                                generation: 0,
                            },
                            revision: 0,
                            generation: 0,
                            deadline_tick: u64::MAX,
                        },
                        0,
                    )?;
                    consumers = consumers
                        .checked_add(task.consumers.len())
                        .ok_or(StoreError::Capacity("archive consumers"))?;
                    if tasks.insert(id, task).is_some() {
                        return Err(StoreError::ArchiveIntegrity("duplicate execution"));
                    }
                }
            }
        }
        for id in &pins.observations {
            let o = observations
                .get(id)
                .or_else(|| self.observations.observations.get(id.0))
                .ok_or(StoreError::InvalidHandle("cold observation dependency"))?;
            if let RawScore::ConditionalWdl { perspective, .. }
            | RawScore::ConditionalRepairWdl { perspective, .. } = o.score
            {
                let snapshot = states
                    .get(&o.state)
                    .or_else(|| self.states.snapshots.get(o.state.0))
                    .ok_or(StoreError::InvalidHandle("cold conditional state"))?;
                if perspective != snapshot.side_to_move() {
                    return Err(StoreError::ArchiveIntegrity("conditional root perspective"));
                }
                let get = |id: ObservationId| {
                    observations
                        .get(&id)
                        .or_else(|| self.observations.observations.get(id.0))
                        .ok_or(StoreError::InvalidHandle("cold conditional raw evidence"))
                };
                validate_model_derivation(o, get)?;
            }
            for line in [o.line, o.cpu_pv].into_iter().flatten() {
                let l = lines
                    .get(&line)
                    .or_else(|| self.lines.chunks.get(line.0))
                    .ok_or(StoreError::InvalidHandle("cold observation line"))?;
                if l.start_state != o.state {
                    return Err(StoreError::ArchiveIntegrity("observation line state"));
                }
            }
            if let Some(previous) = o.supersedes {
                let p = observations
                    .get(&previous)
                    .or_else(|| self.observations.observations.get(previous.0))
                    .ok_or(StoreError::InvalidHandle("cold supersedes"))?;
                if p.state != o.state {
                    return Err(StoreError::ArchiveIntegrity("supersedes state"));
                }
            }
            if let Some(execution) = o.execution {
                let task = tasks
                    .get(&execution)
                    .or_else(|| self.tasks.tasks.get(execution.0))
                    .ok_or(StoreError::InvalidHandle("cold observation execution"))?;
                match (task.key.question, o.scope) {
                    (
                        TaskQuestion::ProfiledExternal {
                            requested: question,
                        },
                        EvidenceScope::ProfiledExternalUci { requested, .. },
                    ) if question == requested => {}
                    (TaskQuestion::ProfiledExternal { .. }, _)
                    | (_, EvidenceScope::ProfiledExternalUci { .. }) => {
                        return Err(StoreError::ArchiveIntegrity(
                            "profiled observation task constraint",
                        ));
                    }
                    _ => {}
                }
                if task.key.state != o.state
                    || task.key.line != o.line
                    || task.key.value_identity != o.value_identity
                    || task.key.checker_identity != o.checker_identity
                    || task.key.cpu_condition != o.cpu_condition
                {
                    return Err(StoreError::ArchiveIntegrity("observation task namespace"));
                }
            }
        }
        let selected_situations = bundle
            .situations
            .into_iter()
            .filter(|(id, _)| pins.situations.contains(id))
            .collect::<Vec<_>>();
        let new_situations = selected_situations
            .iter()
            .filter(|(id, _)| self.situations.get(*id).is_err())
            .count();
        let edges = bundle
            .dependencies
            .into_iter()
            .filter(|(o, ids)| {
                pins.observations.contains(o) || ids.iter().any(|id| pins.situations.contains(id))
            })
            .collect::<Vec<_>>();
        let additional_edges = edges
            .iter()
            .map(|(o, ids)| {
                ids.iter()
                    .filter(|s| {
                        !self
                            .dependencies
                            .dependents
                            .get(o)
                            .is_some_and(|old| old.contains(s))
                    })
                    .count()
            })
            .sum::<usize>();
        let l = self.limits;
        for (used, added, maximum, name) in [
            (self.states.len(), states.len(), l.states, "loaded states"),
            (self.lines.len(), lines.len(), l.line_chunks, "loaded lines"),
            (
                self.observations.len(),
                observations.len(),
                l.observations,
                "loaded observations",
            ),
            (
                self.tasks.len(),
                tasks.len(),
                l.executions,
                "loaded executions",
            ),
            (
                self.tasks.consumer_count(),
                consumers,
                l.consumers,
                "loaded consumers",
            ),
            (
                self.situations.len(),
                new_situations,
                l.situations,
                "loaded situations",
            ),
            (
                self.dependencies.len(),
                additional_edges,
                l.dependency_edges,
                "loaded dependencies",
            ),
            (
                self.states.history_bytes,
                history,
                l.retained_history_bytes,
                "loaded history",
            ),
        ] {
            if used.checked_add(added).is_none_or(|n| n > maximum) {
                return Err(StoreError::PinSaturated(name));
            }
        }
        // Reserve fallible record/index allocations before publishing any data.
        self.states.snapshots.reserve(states.len())?;
        self.lines.chunks.reserve(lines.len())?;
        self.observations.observations.reserve(observations.len())?;
        self.tasks.tasks.reserve(tasks.len())?;
        self.states
            .index
            .try_reserve(states.len())
            .map_err(|_| StoreError::Capacity("load state index"))?;
        self.situations
            .slots
            .try_reserve_exact(
                new_situations.saturating_sub(self.situations.slots.len() - self.situations.live),
            )
            .map_err(|_| StoreError::Capacity("load situation allocation"))?;
        let mut seen_consumers = BTreeSet::new();
        for task in tasks.values() {
            for consumer in &task.consumers {
                if self.tasks.consumers.contains(&consumer.consumer.id)
                    || !seen_consumers.insert(consumer.consumer.id)
                {
                    return Err(StoreError::ArchiveIntegrity("consumer serial collision"));
                }
            }
        }
        let mut checked = Vec::new();
        for id in bundle.checked_pvs {
            if pins.lines.contains(&id) {
                budget.check()?;
                let line = lines
                    .get(&id)
                    .or_else(|| self.lines.chunks.get(id.0))
                    .ok_or(StoreError::InvalidHandle("cold checked PV"))?;
                let snapshot = states
                    .get(&line.start_state)
                    .or_else(|| self.states.snapshots.get(line.start_state.0))
                    .ok_or(StoreError::InvalidHandle("cold checked PV state"))?;
                let trace = snapshot
                    .uci_replay(rz_position::MAX_UCI_REPLAY_PLIES)
                    .map_err(|_| StoreError::ArchiveIntegrity("PV Rules trace"))?;
                let mut position = match trace.origin {
                    rz_position::HistoryOrigin::StartPosition => Position::startpos(),
                    rz_position::HistoryOrigin::Fen => Position::from_fen(&trace.start_fen)
                        .map_err(|_| StoreError::ArchiveIntegrity("PV root"))?,
                };
                for movement in trace.moves {
                    position
                        .make_move(movement)
                        .map_err(|_| StoreError::ArchiveIntegrity("PV Rules root history"))?;
                }
                let mut at = Some(id);
                let mut chain = Vec::new();
                while let Some(id) = at {
                    let line = lines
                        .get(&id)
                        .or_else(|| self.lines.chunks.get(id.0))
                        .ok_or(StoreError::InvalidHandle("PV line chain"))?;
                    chain.push(line);
                    at = line.parent;
                }
                for line in chain.into_iter().rev() {
                    for movement in line.chunk_moves() {
                        budget.check()?;
                        position
                            .make_move(movement.unpack()?)
                            .map_err(|_| StoreError::ArchiveIntegrity("PV Rules replay"))?;
                    }
                }
                checked.push(id);
            }
        }
        for o in observations.values() {
            if o.cpu_pv
                .is_some_and(|id| !checked.contains(&id) && !self.checked_cpu_pvs.contains_key(&id))
            {
                return Err(StoreError::ArchiveIntegrity("cold CPU PV attestation"));
            }
            if let RawScore::Terminal { winner } = o.score {
                let snapshot = states
                    .get(&o.state)
                    .or_else(|| self.states.snapshots.get(o.state.0))
                    .ok_or(StoreError::InvalidHandle("cold terminal state"))?;
                let position = Position::from_snapshot(
                    snapshot,
                    rz_position::PositionLimits {
                        max_history_positions: rz_position::MAX_UCI_REPLAY_PLIES + 1,
                        ..rz_position::PositionLimits::default()
                    },
                )
                .map_err(|_| StoreError::ArchiveIntegrity("terminal Rules owner"))?;
                let legal = position.ordered_legal_moves();
                if !matches!(position.play_status_from_view(&legal),Ok(PlayStatus::Terminal {winner:actual,..}) if actual==winner)
                {
                    return Err(StoreError::ArchiveIntegrity("terminal Rules fact"));
                }
            }
        }
        let engine_records = if include_engine {
            bundle
                .engine_records
                .into_iter()
                .map(RoleRecordWire::decode)
                .collect::<Result<_, _>>()?
        } else {
            vec![]
        };
        let engine_nodes = if include_engine {
            bundle.engine_nodes
        } else {
            vec![]
        };
        for record in &engine_records {
            if record.line.len() > l.line_plies {
                return Err(StoreError::ArchiveIntegrity("engine line bounds"));
            }
        }
        for node in &engine_nodes {
            if node.metadata.len() > 4096 || node.edges.len() > l.root_moves_per_task {
                return Err(StoreError::ArchiveIntegrity("engine node bounds"));
            }
        }
        budget.check()?;
        // All fallible I/O, Rules decoding, namespace checks and capacity checks
        // precede hot publication. Old situation generations stay invalid.
        for (id, state) in states {
            let key = state.repetition_identity();
            self.states.snapshots.restore(id.0, state)?;
            self.states.index.entry(key).or_default().push(id);
        }
        self.states.history_bytes += history;
        for (id, line) in lines {
            if let Some(parent) = line.parent {
                self.lines
                    .index
                    .insert((parent, line.chunk_moves().to_vec()), id);
            } else {
                self.lines.roots.entry(line.start_state).or_insert(id);
            }
            self.lines.chunks.restore(id.0, line)?;
        }
        for (id, o) in observations {
            self.observations.observations.restore(id.0, o)?;
        }
        let mut remap = Vec::new();
        let mut mapped_pins = pins.clone();
        mapped_pins.situations.clear();
        for (old, situation) in selected_situations {
            let current = if self.situations.get(old).is_ok() {
                old
            } else {
                let new = self.situations.insert(situation.state, situation.focus)?;
                *self.situations.get_mut(new)? = situation;
                new
            };
            remap.push((old, current));
            mapped_pins.situations.insert(current);
        }
        for (id, mut task) in tasks {
            for c in &mut task.consumers {
                if let Some((_, new)) = remap.iter().find(|(old, _)| *old == c.consumer.situation) {
                    c.consumer.situation = *new;
                }
                self.tasks.consumers.insert(c.consumer.id);
            }
            self.tasks
                .latest
                .entry(task.key.clone())
                .and_modify(|old| {
                    if id.0 > old.0 {
                        *old = id
                    }
                })
                .or_insert(id);
            self.tasks.tasks.restore(id.0, task)?;
        }
        for (observation, ids) in edges {
            let entry = self.dependencies.dependents.entry(observation).or_default();
            for old in ids {
                let current = remap
                    .iter()
                    .find(|(old_id, _)| *old_id == old)
                    .map_or(old, |(_, new)| *new);
                entry.insert(current);
            }
        }
        self.dependencies.edges = self
            .dependencies
            .dependents
            .values()
            .map(BTreeSet::len)
            .sum();
        for line in checked {
            self.checked_cpu_pvs.insert(
                line,
                CheckedCpuPv {
                    state_store: Arc::downgrade(&self.states.identity),
                    line_pool: Arc::downgrade(&self.lines.identity),
                },
            );
        }
        manager.load_pins.insert(pin_serial, mapped_pins);
        let mut loaded = StorePins::default();
        for pins in manager.load_pins.values() {
            loaded.extend(pins);
        }
        let capacity = self.pinned_capacity_subset(&loaded);
        manager.note(|meter| {
            meter.pinned_entries_peak = meter
                .pinned_entries_peak
                .max(manager.load_pins.len() as u32);
            meter.loaded_closure_bytes_peak = meter
                .loaded_closure_bytes_peak
                .zip(capacity)
                .map(|(peak, current)| peak.max(current));
            if meter.loaded_closure_bytes_peak.is_none() {
                meter.loaded_closure_bytes_method = None;
            }
        });
        manager.next_pin = next_pin;
        Ok(LoadedArchive {
            pin: ArchiveLoadPin {
                owner: Arc::downgrade(&manager.identity),
                serial: pin_serial,
            },
            situations: remap,
            engine_records,
            engine_nodes,
            io_bytes,
        })
    }
}

#[cfg(test)]
mod tests;
