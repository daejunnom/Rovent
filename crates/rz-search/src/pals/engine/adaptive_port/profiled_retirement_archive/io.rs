//! Capability-relative, bounded retirement I/O. A verified artifact is issued
//! only after physical file and directory synchronization and full readback.
//! File extents below are disk/write facts, never host allocation or RSS.

use crate::pals::store::{ArchiveRootCapability, StoreError};
use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::OpenOptions;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

const MAX_MANAGED_BYTES: u64 = 18 * 1024 * 1024 * 1024;
const MAX_FILES: u32 = 32_768;
const MAX_SCAN_ENTRIES: u32 = 65_536;
const LOCK_NAME: &str = ".retirement-quota.lock";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct IoLimits {
    pub max_file_bytes: u64,
    pub max_total_bytes: u64,
    pub max_files: u32,
    pub max_scan_entries: u32,
}
impl IoLimits {
    fn validate(self) -> Result<(), IoError> {
        if self.max_file_bytes == 0
            || self.max_total_bytes == 0
            || self.max_total_bytes > MAX_MANAGED_BYTES
            || self.max_file_bytes > self.max_total_bytes / 2
            || self.max_files < 2
            || self.max_files > MAX_FILES
            || self.max_scan_entries <= self.max_files
            || self.max_scan_entries > MAX_SCAN_ENTRIES
        {
            return Err(IoError::Boundary("retirement I/O limits"));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(super) enum IoError {
    UnsupportedDurablePlatform,
    Boundary(&'static str),
    Cancelled,
    Deadline,
    ByteBudget,
    Store(StoreError),
    Io {
        stage: &'static str,
        error: io::Error,
    },
}
impl IoError {
    fn at(stage: &'static str, error: io::Error) -> Self {
        Self::Io { stage, error }
    }
}
impl std::fmt::Display for IoError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "retirement archive I/O: {self:?}")
    }
}
impl std::error::Error for IoError {}
impl From<StoreError> for IoError {
    fn from(error: StoreError) -> Self {
        Self::Store(error)
    }
}

#[derive(Debug)]
pub(super) enum IoFailure<E> {
    Io(IoError),
    /// Both the original codec failure and any underlying counted-writer
    /// boundary survive. The codec's I/O wrapper cannot hide a quota failure.
    Encode {
        error: E,
        io: Option<IoError>,
    },
}
impl<E> From<IoError> for IoFailure<E> {
    fn from(error: IoError) -> Self {
        Self::Io(error)
    }
}

/// Fixed-size observations only. A poisoned/unknown write is never refunded
/// into an apparent zero. None means unavailable, not an unused allowance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct IoObservation {
    pub session_written_bytes: Option<u64>,
    pub session_reserved_bytes: u64,
    /// Successful data read/write return counts; filesystem metadata I/O,
    /// storage-media traffic and RSS are outside this observation's scope.
    pub data_io_bytes: Option<u64>,
    pub managed_disk_bytes: Option<u64>,
    /// Data-file pathname count, excluding the empty private quota lock.
    /// During publication both temporary hard-link paths count separately.
    pub managed_files: Option<u32>,
    pub verified_artifacts: u64,
    pub poisoned: bool,
    pub failed_file_retained: bool,
    pub lock_retained: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FileIdentity {
    device: u64,
    inode: u64,
}
struct FileState {
    identity: FileIdentity,
    bytes: u64,
    links: u64,
}
impl FileState {
    #[cfg(unix)]
    fn observe(file: &File) -> Result<Self, IoError> {
        use std::os::unix::fs::MetadataExt;
        let metadata = file
            .metadata()
            .map_err(|error| IoError::at("held file metadata", error))?;
        if !metadata.is_file() {
            return Err(IoError::Boundary("retirement file is not regular"));
        }
        Ok(Self {
            identity: FileIdentity {
                device: metadata.dev(),
                inode: metadata.ino(),
            },
            bytes: metadata.len(),
            links: metadata.nlink(),
        })
    }
    #[cfg(not(unix))]
    fn observe(_file: &File) -> Result<Self, IoError> {
        Err(IoError::UnsupportedDurablePlatform)
    }
}

struct PendingFile {
    file: File,
    identity: Option<FileIdentity>,
    pending_name: String,
    final_name: String,
    final_reader: Option<File>,
    final_linked: bool,
    pending_unlinked: bool,
    extent: Option<u64>,
}
struct HeldLock {
    _file: File,
    identity: Option<FileIdentity>,
    removed: bool,
}

pub(super) struct IoOwner {
    capability: ArchiveRootCapability,
    limits: IoLimits,
    next_sequence: u64,
    session_written: Option<u64>,
    session_reserved: u64,
    data_io_bytes: Option<u64>,
    managed_bytes: Option<u64>,
    managed_files: Option<u32>,
    verified_artifacts: u64,
    poisoned: bool,
    pending: Option<PendingFile>,
    lock: Option<HeldLock>,
}

/// Move-only proof material. Only this module constructs it; JSON metadata
/// cannot synthesize the retained, read-only descriptor or its identity.
pub(super) struct IoVerifiedArtifact<S> {
    _retained_capability: ArchiveRootCapability,
    _retained_file: File,
    _retained_identity: FileIdentity,
    name: String,
    sha256: [u8; 32],
    bytes: u64,
    io_bytes: u64,
    summary: S,
}
impl<S> IoVerifiedArtifact<S> {
    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn file(&self) -> &File {
        &self._retained_file
    }
    pub(super) fn name(&self) -> &str {
        &self.name
    }
    pub(super) fn sha256(&self) -> [u8; 32] {
        self.sha256
    }
    pub(super) fn bytes(&self) -> u64 {
        self.bytes
    }
    /// Successful data read/write counts for this publication only. Directory
    /// enumeration, metadata operations and physical storage traffic excluded.
    pub(super) fn io_bytes(&self) -> u64 {
        self.io_bytes
    }
    pub(super) fn summary(&self) -> &S {
        &self.summary
    }

    /// Identity/type/extent check on the original retained FD, not a pathname
    /// or caller-supplied bool. Content revalidation remains bounded below.
    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn revalidate(&self) -> Result<(), IoError> {
        self._retained_capability.verify_named_root()?;
        let state = FileState::observe(&self._retained_file)?;
        if state.identity != self._retained_identity
            || state.bytes != self.bytes
            || state.links != 1
        {
            return Err(IoError::Boundary("retirement artifact descriptor changed"));
        }
        let named = open_relative(&self._retained_capability, &self.name, false)?;
        let state = FileState::observe(&named)?;
        if state.identity != self._retained_identity
            || state.bytes != self.bytes
            || state.links != 1
        {
            return Err(IoError::Boundary(
                "retirement artifact named descriptor changed",
            ));
        }
        Ok(())
    }
}

struct IoBudget<'a> {
    until: Instant,
    cancel: &'a AtomicBool,
    remaining: u64,
    used: u64,
    cumulative: &'a mut Option<u64>,
}
impl IoBudget<'_> {
    fn check(&self) -> Result<(), IoError> {
        if self.cancel.load(Ordering::Acquire) {
            return Err(IoError::Cancelled);
        }
        if Instant::now() >= self.until {
            return Err(IoError::Deadline);
        }
        Ok(())
    }
    fn debit(&mut self, amount: u64) -> Result<(), IoError> {
        self.used = self.used.checked_add(amount).ok_or_else(|| {
            *self.cumulative = None;
            IoError::Boundary("retirement data I/O count overflow")
        })?;
        *self.cumulative = self.cumulative.and_then(|total| total.checked_add(amount));
        if self.cumulative.is_none() {
            return Err(IoError::Boundary("retirement cumulative data I/O unknown"));
        }
        self.remaining = self
            .remaining
            .checked_sub(amount)
            .ok_or(IoError::ByteBudget)?;
        self.check()
    }
    fn unknown(&mut self) {
        *self.cumulative = None;
    }
}

#[derive(Default)]
struct Usage {
    bytes: u64,
    files: u32,
    last_sequence: u64,
}

impl IoOwner {
    pub(super) fn new(
        capability: ArchiveRootCapability,
        protected_execution_root: &Path,
        limits: IoLimits,
        until: Instant,
        cancel: &AtomicBool,
    ) -> Result<Self, IoError> {
        if !cfg!(target_os = "linux") {
            return Err(IoError::UnsupportedDurablePlatform);
        }
        if cancel.load(Ordering::Acquire) {
            return Err(IoError::Cancelled);
        }
        if Instant::now() >= until {
            return Err(IoError::Deadline);
        }
        limits.validate()?;
        if !protected_execution_root.is_absolute() {
            return Err(IoError::Boundary(
                "absolute protected execution root required",
            ));
        }
        capability.verify_named_root()?;
        // This protects the actual private execution input root. It does not
        // require or assert that a source checkout exists on the deployed host.
        let protected_execution_root = std::fs::canonicalize(protected_execution_root)
            .map_err(|error| IoError::at("canonical protected execution root", error))?;
        if capability
            .root_path()
            .starts_with(&protected_execution_root)
            || protected_execution_root.starts_with(capability.root_path())
        {
            return Err(IoError::Boundary(
                "retirement root must be separate from protected execution root",
            ));
        }
        let mut owner = Self {
            capability,
            limits,
            next_sequence: 1,
            session_written: Some(0),
            session_reserved: 0,
            data_io_bytes: Some(0),
            managed_bytes: None,
            managed_files: None,
            verified_artifacts: 0,
            poisoned: false,
            pending: None,
            lock: None,
        };
        let mut metadata_only_data_count = Some(0);
        let budget = IoBudget {
            until,
            cancel,
            remaining: 0,
            used: 0,
            cumulative: &mut metadata_only_data_count,
        };
        let usage = owner.scan(&budget)?;
        owner.next_sequence = usage
            .last_sequence
            .checked_add(1)
            .ok_or(IoError::Boundary("retirement file sequence exhausted"))?;
        owner.managed_bytes = Some(usage.bytes);
        owner.managed_files = Some(usage.files);
        Ok(owner)
    }

    pub(super) fn observation(&self) -> IoObservation {
        IoObservation {
            session_written_bytes: self.session_written,
            session_reserved_bytes: self.session_reserved,
            data_io_bytes: self.data_io_bytes,
            managed_disk_bytes: self.managed_bytes,
            managed_files: self.managed_files,
            verified_artifacts: self.verified_artifacts,
            poisoned: self.poisoned,
            failed_file_retained: self.pending.is_some(),
            lock_retained: self.lock.is_some(),
        }
    }

    pub(super) fn data_io_bytes(&self) -> Option<u64> {
        self.data_io_bytes
    }

    pub(super) fn write_stream<S, E>(
        &mut self,
        until: Instant,
        cancel: &AtomicBool,
        max_io_bytes: u64,
        encode: impl FnOnce(&mut dyn Write) -> Result<S, E>,
    ) -> Result<IoVerifiedArtifact<S>, IoFailure<E>> {
        if self.poisoned || self.pending.is_some() || self.lock.is_some() {
            return Err(IoError::Boundary("retirement I/O admission closed").into());
        }
        // An unwinding codec cannot reopen admission or hide its partial file.
        self.poisoned = true;
        self.managed_bytes = None;
        self.managed_files = None;
        // Leave the owner unknown during execution/unwind. The exact local
        // counter is returned only when the operation itself returns normally.
        let mut data_io_bytes = self.data_io_bytes.take();
        let mut budget = IoBudget {
            until,
            cancel,
            remaining: max_io_bytes,
            used: 0,
            cumulative: &mut data_io_bytes,
        };
        let result = self.write_inner(&mut budget, encode);
        self.data_io_bytes = data_io_bytes;
        if result.is_ok() {
            self.poisoned = false;
        } else {
            self.managed_bytes = None;
            self.managed_files = None;
            if let Some(pending) = &mut self.pending {
                pending.extent = pending.file.metadata().ok().map(|metadata| metadata.len());
            }
        }
        result
    }

    fn write_inner<S, E>(
        &mut self,
        budget: &mut IoBudget<'_>,
        encode: impl FnOnce(&mut dyn Write) -> Result<S, E>,
    ) -> Result<IoVerifiedArtifact<S>, IoFailure<E>> {
        budget.check()?;
        self.capability.verify_named_root().map_err(IoError::from)?;
        self.take_lock()?;
        let usage = self.scan(budget)?;
        if usage
            .files
            .checked_add(2)
            .is_none_or(|files| files > self.limits.max_files)
        {
            return Err(IoError::Boundary("retirement file-count reservation").into());
        }
        let spent = self
            .session_written
            .ok_or(IoError::Boundary("retirement write count unknown"))?;
        let disk_remaining = self
            .limits
            .max_total_bytes
            .checked_sub(usage.bytes)
            .ok_or(IoError::Boundary("retirement managed disk quota"))?;
        let session_remaining = self
            .limits
            .max_total_bytes
            .checked_sub(spent)
            .ok_or(IoError::Boundary("retirement session write quota"))?;
        // During no-clobber publication both names can exist. The per-entry
        // disk scan conservatively counts both, including retained failures.
        let maximum = self
            .limits
            .max_file_bytes
            .min(disk_remaining / 2)
            .min(session_remaining)
            .min(budget.remaining / 3);
        if maximum == 0 {
            return Err(IoError::ByteBudget.into());
        }
        // Another already-admitted owner may have published while this owner
        // was idle. Choose the sequence from the scan protected by this lock.
        let after_latest = usage
            .last_sequence
            .checked_add(1)
            .ok_or(IoError::Boundary("retirement file sequence exhausted"))?;
        let sequence = self.next_sequence.max(after_latest);
        self.next_sequence = sequence
            .checked_add(1)
            .ok_or(IoError::Boundary("retirement file sequence exhausted"))?;
        self.session_reserved = maximum;
        let pending_name = format!("retirement-{sequence:020}.pending");
        let final_name = format!("retirement-{sequence:020}.json");
        let file = self.open(&pending_name, true)?;
        self.pending = Some(PendingFile {
            file,
            identity: None,
            pending_name,
            final_name,
            final_reader: None,
            final_linked: false,
            pending_unlinked: false,
            extent: None,
        });
        let state = FileState::observe(&self.pending.as_ref().expect("retained pending").file)?;
        self.pending.as_mut().expect("retained pending").identity = Some(state.identity);
        self.pending.as_mut().expect("retained pending").extent = Some(state.bytes);
        if state.bytes != 0 || state.links != 1 {
            return Err(IoError::Boundary("new retirement pending file identity").into());
        }
        let pending = self.pending.as_mut().expect("retained pending");
        let mut writer = StreamWriter {
            file: &mut pending.file,
            session_written: &mut self.session_written,
            session_reserved: &mut self.session_reserved,
            pending_extent: &mut pending.extent,
            budget,
            maximum,
            bytes: 0,
            hash: Sha256::new(),
            failure: None,
        };
        let encoded = encode(&mut writer);
        let written = writer.bytes;
        let expected_sha: [u8; 32] = writer.hash.finalize().into();
        let writer_failure = writer.failure;
        // The file extent reconciles a failed write whose OS error provided
        // no byte count. Unknown extent closes accounting rather than refunding.
        let actual = self
            .pending
            .as_ref()
            .and_then(|pending| pending.file.metadata().ok())
            .map(|metadata| metadata.len());
        self.session_written = actual
            .filter(|actual| *actual >= written && *actual <= maximum)
            .and_then(|actual| spent.checked_add(actual));
        if let Some(extra) = actual.and_then(|actual| actual.checked_sub(written)) {
            self.session_reserved = match self.session_reserved.checked_sub(extra) {
                Some(remaining) => remaining,
                None => {
                    self.session_written = None;
                    self.session_reserved
                }
            };
        }
        self.pending.as_mut().expect("retained pending").extent = actual;
        let summary = match encoded {
            Ok(summary) => {
                if let Some(error) = writer_failure {
                    return Err(IoFailure::Io(error));
                }
                summary
            }
            Err(error) => {
                return Err(IoFailure::Encode {
                    error,
                    io: writer_failure,
                });
            }
        };
        if written == 0 || actual != Some(written) || self.session_written.is_none() {
            return Err(IoError::Boundary("retirement stream extent mismatch").into());
        }
        budget.check()?;
        self.pending
            .as_mut()
            .expect("retained pending")
            .file
            .flush()
            .map_err(|error| IoError::at("flush retirement pending file", error))?;
        self.pending
            .as_ref()
            .expect("retained pending")
            .file
            .sync_all()
            .map_err(|error| IoError::at("sync retirement pending file", error))?;
        {
            let pending = self.pending.as_mut().expect("retained pending");
            readback(
                &mut pending.file,
                state.identity,
                written,
                1,
                expected_sha,
                budget,
            )?;
        }
        self.capability.verify_named_root().map_err(IoError::from)?;
        {
            let pending = self.pending.as_ref().expect("retained pending");
            self.capability
                .held_directory()
                .hard_link(
                    &pending.pending_name,
                    self.capability.held_directory(),
                    &pending.final_name,
                )
                .map_err(|error| IoError::at("publish retirement without overwrite", error))?;
        }
        self.pending
            .as_mut()
            .expect("retained pending")
            .final_linked = true;
        let final_name = &self.pending.as_ref().expect("retained pending").final_name;
        let reader = self.open(final_name, false)?;
        self.pending
            .as_mut()
            .expect("retained pending")
            .final_reader = Some(reader);
        {
            let pending = self.pending.as_ref().expect("retained pending");
            let named =
                FileState::observe(pending.final_reader.as_ref().expect("retained reader"))?;
            let original = FileState::observe(&pending.file)?;
            if Some(named.identity) != pending.identity
                || Some(original.identity) != pending.identity
                || named.bytes != written
                || original.bytes != written
                || named.links != 2
                || original.links != 2
            {
                return Err(IoError::Boundary("retirement published link identity").into());
            }
        }
        // Scan the exact temporary two-link state while its quota lock is held.
        self.scan(budget)?;
        {
            let pending = self.pending.as_ref().expect("retained pending");
            let original_name = self.open(&pending.pending_name, false)?;
            let named_state = FileState::observe(&original_name)?;
            if Some(named_state.identity) != pending.identity
                || named_state.bytes != written
                || named_state.links != 2
            {
                return Err(
                    IoError::Boundary("retirement pending name replaced before unlink").into(),
                );
            }
        }
        let pending_name = &self
            .pending
            .as_ref()
            .expect("retained pending")
            .pending_name;
        self.capability
            .held_directory()
            .remove_file(pending_name)
            .map_err(|error| IoError::at("unlink verified retirement pending name", error))?;
        self.pending
            .as_mut()
            .expect("retained pending")
            .pending_unlinked = true;
        self.sync_directory()?;
        {
            let pending = self.pending.as_mut().expect("retained pending");
            readback(
                pending.final_reader.as_mut().expect("retained reader"),
                state.identity,
                written,
                1,
                expected_sha,
                budget,
            )?;
        }
        let usage = self.scan(budget)?;
        self.release_lock()?;
        self.sync_directory()?;
        budget.check()?;
        self.capability.verify_named_root().map_err(IoError::from)?;
        let pending = self.pending.as_ref().expect("retained pending");
        let named = self.open(&pending.final_name, false)?;
        let named_state = FileState::observe(&named)?;
        if Some(named_state.identity) != pending.identity
            || named_state.bytes != written
            || named_state.links != 1
        {
            return Err(IoError::Boundary("retirement final named descriptor changed").into());
        }
        // No fallible operation follows removal from the retained owner.
        let verified_artifacts = self
            .verified_artifacts
            .checked_add(1)
            .ok_or(IoError::Boundary("retirement artifact count exhausted"))?;
        let pending = self.pending.take().expect("retained pending");
        self.session_reserved = 0;
        self.managed_bytes = Some(usage.bytes);
        self.managed_files = Some(usage.files);
        self.verified_artifacts = verified_artifacts;
        self.lock.take();
        Ok(IoVerifiedArtifact {
            _retained_capability: self.capability.clone(),
            _retained_file: named,
            _retained_identity: state.identity,
            name: pending.final_name,
            sha256: expected_sha,
            bytes: written,
            io_bytes: budget.used,
            summary,
        })
    }

    fn open(&self, name: &str, create_new: bool) -> Result<File, IoError> {
        open_relative(&self.capability, name, create_new)
    }

    fn take_lock(&mut self) -> Result<(), IoError> {
        let file = self.open(LOCK_NAME, true)?;
        self.lock = Some(HeldLock {
            _file: file,
            identity: None,
            removed: false,
        });
        let state = FileState::observe(&self.lock.as_ref().expect("retained lock")._file)?;
        self.lock.as_mut().expect("retained lock").identity = Some(state.identity);
        if state.bytes != 0 || state.links != 1 {
            return Err(IoError::Boundary("retirement quota lock identity"));
        }
        Ok(())
    }

    fn release_lock(&mut self) -> Result<(), IoError> {
        let lock = self
            .lock
            .as_ref()
            .ok_or(IoError::Boundary("retirement quota lock missing"))?;
        let named = self.open(LOCK_NAME, false)?;
        let state = FileState::observe(&named)?;
        if Some(state.identity) != lock.identity
            || state.bytes != 0
            || state.links != 1
            || lock.removed
        {
            return Err(IoError::Boundary("retirement quota lock replaced"));
        }
        self.capability
            .held_directory()
            .remove_file(LOCK_NAME)
            .map_err(|error| IoError::at("release original retirement quota lock", error))?;
        self.lock.as_mut().expect("retained lock").removed = true;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn sync_directory(&self) -> Result<(), IoError> {
        use cap_std::fs::OpenOptionsExt;
        let mut options = OpenOptions::new();
        options
            .read(true)
            .follow(FollowSymlinks::No)
            .custom_flags(nix::libc::O_DIRECTORY | nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK);
        self.capability
            .held_directory()
            .open_with(".", &options)
            .and_then(|directory| directory.into_std().sync_all())
            .map_err(|error| IoError::at("sync held retirement directory", error))
    }
    #[cfg(not(target_os = "linux"))]
    fn sync_directory(&self) -> Result<(), IoError> {
        Err(IoError::UnsupportedDurablePlatform)
    }

    fn scan(&self, budget: &IoBudget<'_>) -> Result<Usage, IoError> {
        budget.check()?;
        self.capability.verify_named_root()?;
        let entries = self
            .capability
            .held_directory()
            .entries()
            .map_err(|error| IoError::at("scan held retirement root", error))?;
        let mut result = Usage::default();
        let mut scanned = 0_u32;
        let mut saw_lock = false;
        let mut saw_pending = false;
        let mut saw_final = false;
        for entry in entries {
            budget.check()?;
            scanned = scanned
                .checked_add(1)
                .filter(|count| *count <= self.limits.max_scan_entries)
                .ok_or(IoError::Boundary("retirement scan entry limit"))?;
            let entry =
                entry.map_err(|error| IoError::at("read retirement directory entry", error))?;
            let name = entry.file_name();
            let name = name
                .to_str()
                .filter(|name| name.len() <= 64)
                .ok_or(IoError::Boundary("retirement filename encoding or length"))?;
            let metadata = self
                .capability
                .held_directory()
                .symlink_metadata(name)
                .map_err(|error| IoError::at("retirement entry no-follow metadata", error))?;
            if !metadata.is_file() || metadata.is_symlink() {
                return Err(IoError::Boundary(
                    "retirement root contains non-regular entry",
                ));
            }
            let file = self.open(name, false)?;
            let state = FileState::observe(&file)?;
            if name == LOCK_NAME {
                let lock = self
                    .lock
                    .as_ref()
                    .ok_or(IoError::Boundary("foreign retirement quota lock"))?;
                if Some(state.identity) != lock.identity
                    || state.links != 1
                    || state.bytes != 0
                    || lock.removed
                {
                    return Err(IoError::Boundary("retirement quota lock scan identity"));
                }
                saw_lock = true;
                continue;
            }
            let sequence = parse_final_name(name);
            let active = self.pending.as_ref().filter(|pending| {
                (!pending.pending_unlinked && name == pending.pending_name)
                    || (pending.final_linked && name == pending.final_name)
            });
            match active {
                Some(pending) => {
                    saw_pending |= name == pending.pending_name;
                    saw_final |= name == pending.final_name;
                    let expected_links = if pending.final_linked && !pending.pending_unlinked {
                        2
                    } else {
                        1
                    };
                    if Some(state.identity) != pending.identity
                        || state.links != expected_links
                        || Some(state.bytes) != pending.extent
                    {
                        return Err(IoError::Boundary(
                            "retirement active extent or alias changed",
                        ));
                    }
                }
                None if sequence.is_some() && state.links == 1 => {}
                None => {
                    return Err(IoError::Boundary(
                        "foreign retirement file or hard-link alias",
                    ));
                }
            }
            if state.bytes == 0 || state.bytes > self.limits.max_file_bytes {
                return Err(IoError::Boundary("retirement file extent quota"));
            }
            result.files = result
                .files
                .checked_add(1)
                .filter(|count| *count <= self.limits.max_files)
                .ok_or(IoError::Boundary("retirement managed file quota"))?;
            result.bytes = result
                .bytes
                .checked_add(state.bytes)
                .filter(|bytes| *bytes <= self.limits.max_total_bytes)
                .ok_or(IoError::Boundary("retirement managed disk quota"))?;
            if let Some(sequence) = sequence {
                result.last_sequence = result.last_sequence.max(sequence);
            }
        }
        if self.lock.as_ref().is_some_and(|lock| !lock.removed) && !saw_lock {
            return Err(IoError::Boundary(
                "original retirement quota lock missing from scan",
            ));
        }
        if let Some(pending) = &self.pending {
            if (!pending.pending_unlinked && !saw_pending) || (pending.final_linked && !saw_final) {
                return Err(IoError::Boundary(
                    "original retirement extent missing from scan",
                ));
            }
        }
        budget.check()?;
        self.capability.verify_named_root()?;
        Ok(result)
    }
}

fn open_relative(
    capability: &ArchiveRootCapability,
    name: &str,
    create_new: bool,
) -> Result<File, IoError> {
    let mut options = OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    if create_new {
        options.write(true).create_new(true);
    }
    #[cfg(target_os = "linux")]
    {
        use cap_std::fs::OpenOptionsExt;
        options.custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK);
        if create_new {
            options.mode(0o600);
        }
    }
    capability
        .held_directory()
        .open_with(name, &options)
        .map(|file| file.into_std())
        .map_err(|error| IoError::at("open retirement file without following links", error))
}

fn parse_final_name(name: &str) -> Option<u64> {
    let number = name.strip_prefix("retirement-")?.strip_suffix(".json")?;
    if number.len() != 20 || !number.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let sequence: u64 = number.parse().ok()?;
    (sequence != 0).then_some(sequence)
}

struct StreamWriter<'a, 'b> {
    file: &'a mut File,
    session_written: &'a mut Option<u64>,
    session_reserved: &'a mut u64,
    pending_extent: &'a mut Option<u64>,
    budget: &'a mut IoBudget<'b>,
    maximum: u64,
    bytes: u64,
    hash: Sha256,
    failure: Option<IoError>,
}
impl StreamWriter<'_, '_> {
    fn reject(&mut self, error: IoError) -> io::Error {
        if self.failure.is_none() {
            self.failure = Some(error);
        }
        io::Error::other("bounded retirement stream failed; original cause retained")
    }
}
impl Write for StreamWriter<'_, '_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.failure.is_some() {
            return Err(io::Error::other("retirement stream already failed"));
        }
        if let Err(error) = self.budget.check() {
            return Err(self.reject(error));
        }
        if bytes.len() as u64 > self.maximum.saturating_sub(self.bytes)
            || bytes.len() as u64 > self.budget.remaining
        {
            return Err(self.reject(IoError::ByteBudget));
        }
        let count = match self.file.write(bytes) {
            Ok(count) => count,
            Err(error) => {
                *self.session_written = None;
                *self.pending_extent = None;
                self.budget.unknown();
                return Err(self.reject(IoError::at("write retirement stream", error)));
            }
        };
        self.bytes += count as u64; // maximum was checked before the write.
        *self.session_written = self
            .session_written
            .and_then(|total| total.checked_add(count as u64));
        *self.session_reserved = match self.session_reserved.checked_sub(count as u64) {
            Some(remaining) => remaining,
            None => {
                *self.session_written = None;
                return Err(self.reject(IoError::Boundary("retirement reservation debit invalid")));
            }
        };
        *self.pending_extent = Some(self.bytes);
        self.hash.update(&bytes[..count]);
        if let Err(error) = self.budget.debit(count as u64) {
            return Err(self.reject(error));
        }
        Ok(count)
    }
    fn flush(&mut self) -> io::Result<()> {
        if let Err(error) = self.budget.check() {
            return Err(self.reject(error));
        }
        self.file
            .flush()
            .map_err(|error| self.reject(IoError::at("flush retirement stream", error)))
    }
}

fn readback(
    file: &mut File,
    identity: FileIdentity,
    bytes: u64,
    links: u64,
    expected_sha: [u8; 32],
    budget: &mut IoBudget<'_>,
) -> Result<(), IoError> {
    budget.check()?;
    let before = FileState::observe(file)?;
    if before.identity != identity || before.bytes != bytes || before.links != links {
        return Err(IoError::Boundary("retirement readback descriptor changed"));
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| IoError::at("seek retirement readback", error))?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut read = 0_u64;
    while read < bytes {
        budget.check()?;
        let wanted = (bytes - read)
            .min(buffer.len() as u64)
            .min(budget.remaining);
        if wanted == 0 {
            return Err(IoError::ByteBudget);
        }
        let count = match file.read(&mut buffer[..wanted as usize]) {
            Ok(count) => count,
            Err(error) => {
                budget.unknown();
                return Err(IoError::at("read retirement verification bytes", error));
            }
        };
        if count == 0 {
            return Err(IoError::Boundary("short retirement readback"));
        }
        budget.debit(count as u64)?;
        read += count as u64;
        hash.update(&buffer[..count]);
    }
    budget.check()?;
    // The held regular FD must have the exact registered extent before and
    // after reading exactly that extent. Do not speculate with an EOF probe:
    // a concurrent append could otherwise read one byte outside the budget.
    if <[u8; 32]>::from(hash.finalize()) != expected_sha {
        return Err(IoError::Boundary("retirement readback checksum mismatch"));
    }
    let after = FileState::observe(file)?;
    if after.identity != identity || after.bytes != bytes || after.links != links {
        return Err(IoError::Boundary(
            "retirement readback descriptor changed after hash",
        ));
    }
    Ok(())
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    //! Actual filesystem tests for this I/O component only. These do not mint
    //! Core retirement capabilities, native packets or training acceptance.
    use super::*;
    use cap_std::{ambient_authority, fs::Dir};
    use std::convert::Infallible;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicU64;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    struct Fixture {
        base: PathBuf,
        archive: PathBuf,
        protected_execution_root: PathBuf,
    }
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(1);
            let serial = NEXT.fetch_add(1, Ordering::Relaxed);
            let time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let parent = std::fs::canonicalize(std::env::temp_dir()).unwrap();
            let base = parent.join(format!(
                "rovezero-retirement-io-{}-{time}-{serial}",
                std::process::id()
            ));
            std::fs::create_dir(&base).unwrap();
            let archive = base.join("archive");
            let protected_execution_root = base.join("execution-input");
            std::fs::create_dir(&archive).unwrap();
            std::fs::create_dir(&protected_execution_root).unwrap();
            Self {
                base,
                archive,
                protected_execution_root,
            }
        }
        fn capability(&self) -> ArchiveRootCapability {
            let held = Dir::open_ambient_dir(&self.archive, ambient_authority()).unwrap();
            ArchiveRootCapability::from_directory(self.archive.clone(), held).unwrap()
        }
        fn owner(&self, limits: IoLimits, cancel: &AtomicBool) -> IoOwner {
            IoOwner::new(
                self.capability(),
                &self.protected_execution_root,
                limits,
                until(),
                cancel,
            )
            .unwrap()
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            // This fixture owns a freshly created, absolute temporary directory.
            // Remove only immediate entries; never recurse or follow a link.
            for directory in [&self.archive, &self.protected_execution_root] {
                let Ok(metadata) = std::fs::symlink_metadata(directory) else {
                    continue;
                };
                if metadata.is_symlink() || !metadata.is_dir() {
                    continue;
                }
                if let Ok(entries) = std::fs::read_dir(directory) {
                    for entry in entries.flatten() {
                        if entry.path().parent() == Some(directory.as_path()) {
                            if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
                                let _ = std::fs::remove_dir(entry.path());
                            } else {
                                let _ = std::fs::remove_file(entry.path());
                            }
                        }
                    }
                }
                let _ = std::fs::remove_dir(directory);
            }
            let _ = std::fs::remove_dir(&self.base);
        }
    }

    fn until() -> Instant {
        Instant::now() + Duration::from_secs(10)
    }
    fn limits() -> IoLimits {
        IoLimits {
            max_file_bytes: 1024,
            max_total_bytes: 8192,
            max_files: 8,
            max_scan_entries: 16,
        }
    }

    #[test]
    fn actual_stream_is_synced_rehashed_committed_without_clobber_and_charged() {
        let fixture = Fixture::new();
        let cancel = AtomicBool::new(false);
        let mut owner = fixture.owner(limits(), &cancel);
        let bytes = b"{\"record\":1}\n";
        let artifact = owner
            .write_stream(until(), &cancel, 4096, |writer| {
                writer.write_all(bytes)?;
                Ok::<_, io::Error>(17_u64)
            })
            .unwrap();
        assert_eq!(artifact.bytes(), bytes.len() as u64);
        assert_eq!(artifact.io_bytes(), 3 * bytes.len() as u64);
        assert_eq!(artifact.sha256(), <[u8; 32]>::from(Sha256::digest(bytes)));
        assert_eq!(*artifact.summary(), 17);
        artifact.revalidate().unwrap();
        let mut held = artifact.file().try_clone().unwrap();
        held.seek(SeekFrom::Start(0)).unwrap();
        let mut actual = [0_u8; 13];
        held.read_exact(&mut actual).unwrap();
        assert_eq!(&actual, bytes);
        let first_name = artifact.name().to_owned();
        let first_identity = FileState::observe(artifact.file()).unwrap().identity;
        let second = owner
            .write_stream(until(), &cancel, 4096, |writer| {
                writer.write_all(b"second")?;
                Ok::<_, io::Error>(())
            })
            .unwrap();
        assert_ne!(second.name(), first_name);
        assert_eq!(
            FileState::observe(artifact.file()).unwrap().identity,
            first_identity
        );
        assert_eq!(
            std::fs::read(fixture.archive.join(first_name)).unwrap(),
            bytes
        );
        let observation = owner.observation();
        assert_eq!(
            observation.session_written_bytes,
            Some(bytes.len() as u64 + 6)
        );
        assert_eq!(
            observation.data_io_bytes,
            Some(3 * (bytes.len() as u64 + 6))
        );
        assert_eq!(observation.session_reserved_bytes, 0);
        assert_eq!(observation.managed_disk_bytes, Some(bytes.len() as u64 + 6));
        assert_eq!(observation.managed_files, Some(2));
        assert_eq!(observation.verified_artifacts, 2);
        assert!(!observation.poisoned);
        assert!(!fixture.archive.join(LOCK_NAME).exists());
    }

    #[test]
    fn partial_encode_failure_retains_original_fd_reservation_and_exact_cost() {
        let fixture = Fixture::new();
        let cancel = AtomicBool::new(false);
        let mut owner = fixture.owner(limits(), &cancel);
        let result = owner.write_stream(until(), &cancel, 4096, |writer| {
            writer.write_all(b"partial").unwrap();
            Err::<(), _>("original codec failure")
        });
        assert!(matches!(
            result,
            Err(IoFailure::Encode {
                error: "original codec failure",
                io: None
            })
        ));
        let pending = owner
            .pending
            .as_ref()
            .expect("exact original pending owner retained");
        assert_eq!(FileState::observe(&pending.file).unwrap().bytes, 7);
        assert_eq!(
            std::fs::read(fixture.archive.join(&pending.pending_name)).unwrap(),
            b"partial"
        );
        assert!(!fixture.archive.join(&pending.final_name).exists());
        let before = owner.observation();
        assert_eq!(before.session_written_bytes, Some(7));
        assert_eq!(before.data_io_bytes, Some(7));
        assert!(before.session_reserved_bytes > 0);
        assert_eq!(before.managed_disk_bytes, None);
        assert!(before.poisoned && before.failed_file_retained && before.lock_retained);
        let rejected = owner.write_stream(until(), &cancel, 4096, |_| Ok::<_, Infallible>(()));
        assert!(matches!(
            rejected,
            Err(IoFailure::Io(IoError::Boundary(
                "retirement I/O admission closed"
            )))
        ));
        assert_eq!(owner.observation(), before);
        assert!(fixture.archive.join(LOCK_NAME).exists());
    }

    #[test]
    fn unwinding_codec_keeps_partial_fd_and_unknown_disk_and_io_observations() {
        let fixture = Fixture::new();
        let cancel = AtomicBool::new(false);
        let mut owner = fixture.owner(limits(), &cancel);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<IoVerifiedArtifact<()>, IoFailure<Infallible>> =
                owner.write_stream(until(), &cancel, 4096, |writer| {
                    writer.write_all(b"before panic").unwrap();
                    panic!("fixture codec panic");
                });
        }));
        assert!(panic.is_err());
        let pending = owner
            .pending
            .as_ref()
            .expect("panic preserves original descriptor");
        assert_eq!(FileState::observe(&pending.file).unwrap().bytes, 12);
        assert!(fixture.archive.join(&pending.pending_name).exists());
        let observation = owner.observation();
        assert_eq!(observation.session_written_bytes, Some(12));
        assert_eq!(observation.data_io_bytes, None);
        assert_eq!(observation.managed_disk_bytes, None);
        assert_eq!(observation.managed_files, None);
        assert!(
            observation.poisoned && observation.failed_file_retained && observation.lock_retained
        );
        assert!(observation.session_reserved_bytes > 0);
        assert!(
            owner
                .write_stream(until(), &cancel, 4096, |_| Ok::<_, Infallible>(()))
                .is_err()
        );
    }

    #[test]
    fn foreign_entries_aliases_and_quota_failure_do_not_delete_evidence() {
        let cancel = AtomicBool::new(false);
        for kind in ["unknown", "alias", "symlink", "directory"] {
            let fixture = Fixture::new();
            let final_path = fixture.archive.join("retirement-00000000000000000001.json");
            std::fs::write(&final_path, b"existing").unwrap();
            match kind {
                "unknown" => {
                    std::fs::write(fixture.archive.join("foreign.txt"), b"foreign").unwrap()
                }
                "alias" => std::fs::hard_link(
                    &final_path,
                    fixture.archive.join("retirement-00000000000000000002.json"),
                )
                .unwrap(),
                "symlink" => std::os::unix::fs::symlink(
                    &final_path,
                    fixture.archive.join("retirement-00000000000000000002.json"),
                )
                .unwrap(),
                "directory" => {
                    std::fs::create_dir(fixture.archive.join("foreign-directory")).unwrap()
                }
                _ => unreachable!(),
            }
            assert!(
                IoOwner::new(
                    fixture.capability(),
                    &fixture.protected_execution_root,
                    limits(),
                    until(),
                    &cancel
                )
                .is_err()
            );
            assert_eq!(std::fs::read(&final_path).unwrap(), b"existing");
        }
        // A final target introduced after the locked scan must survive the
        // no-clobber publication attempt byte-for-byte.
        let collision = Fixture::new();
        let mut collision_owner = collision.owner(limits(), &cancel);
        let original_final = collision
            .archive
            .join("retirement-00000000000000000001.json");
        let collision_result = collision_owner.write_stream(until(), &cancel, 4096, |writer| {
            std::fs::write(&original_final, b"original final").unwrap();
            writer.write_all(b"new pending")?;
            Ok::<_, io::Error>(())
        });
        assert!(matches!(
            collision_result,
            Err(IoFailure::Io(IoError::Io {
                stage: "publish retirement without overwrite",
                ..
            }))
        ));
        assert_eq!(std::fs::read(&original_final).unwrap(), b"original final");
        let retained = collision_owner.pending.as_ref().unwrap();
        assert_eq!(
            std::fs::read(collision.archive.join(&retained.pending_name)).unwrap(),
            b"new pending"
        );
        assert!(collision_owner.observation().poisoned);
        let fixture = Fixture::new();
        let small = IoLimits {
            max_file_bytes: 8,
            max_total_bytes: 16,
            max_files: 4,
            max_scan_entries: 8,
        };
        let mut owner = fixture.owner(small, &cancel);
        let result = owner.write_stream(until(), &cancel, 4096, |writer| {
            writer.write_all(b"1234")?;
            writer.write_all(b"56789")?;
            Ok::<_, io::Error>(())
        });
        assert!(matches!(
            result,
            Err(IoFailure::Encode {
                io: Some(IoError::ByteBudget),
                ..
            })
        ));
        let pending = owner.pending.as_ref().unwrap();
        assert_eq!(
            std::fs::read(fixture.archive.join(&pending.pending_name)).unwrap(),
            b"1234"
        );
        assert_eq!(owner.observation().session_written_bytes, Some(4));
        assert_eq!(owner.observation().session_reserved_bytes, 4);
        assert!(owner.observation().poisoned);
        assert!(
            IoLimits {
                max_total_bytes: MAX_MANAGED_BYTES + 1,
                ..limits()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn already_admitted_owners_select_latest_sequence_under_lock() {
        let fixture = Fixture::new();
        let cancel = AtomicBool::new(false);
        let mut first = fixture.owner(limits(), &cancel);
        let mut second = fixture.owner(limits(), &cancel);
        let first_artifact = first
            .write_stream(until(), &cancel, 4096, |writer| {
                writer.write_all(b"first owner")?;
                Ok::<_, io::Error>(())
            })
            .unwrap();
        let second_artifact = second
            .write_stream(until(), &cancel, 4096, |writer| {
                writer.write_all(b"second owner")?;
                Ok::<_, io::Error>(())
            })
            .unwrap();
        assert_eq!(parse_final_name(first_artifact.name()), Some(1));
        assert_eq!(parse_final_name(second_artifact.name()), Some(2));
        assert_eq!(
            std::fs::read(fixture.archive.join(first_artifact.name())).unwrap(),
            b"first owner"
        );
        assert_eq!(
            std::fs::read(fixture.archive.join(second_artifact.name())).unwrap(),
            b"second owner"
        );
        assert!(!second.observation().poisoned);
        assert_eq!(
            second.data_io_bytes(),
            Some(3 * b"second owner".len() as u64)
        );

        let overflow = Fixture::new();
        let mut owner = overflow.owner(limits(), &cancel);
        let original = overflow
            .archive
            .join(format!("retirement-{:020}.json", u64::MAX));
        std::fs::write(&original, b"last sequence").unwrap();
        let rejected = owner.write_stream(until(), &cancel, 4096, |_| Ok::<_, Infallible>(()));
        assert!(matches!(
            rejected,
            Err(IoFailure::Io(IoError::Boundary(
                "retirement file sequence exhausted"
            )))
        ));
        assert_eq!(owner.observation().session_written_bytes, Some(0));
        assert!(owner.pending.is_none());
        assert!(owner.observation().poisoned);
        assert_eq!(std::fs::read(original).unwrap(), b"last sequence");
    }

    #[test]
    fn exact_readback_budget_needs_no_eof_probe_and_zero_budget_reads_nothing() {
        let fixture = Fixture::new();
        let cancel = AtomicBool::new(false);
        let path = fixture.archive.join("retirement-00000000000000000001.json");
        let bytes = b"exact extent";
        std::fs::write(&path, bytes).unwrap();
        let mut file = File::open(&path).unwrap();
        let identity = FileState::observe(&file).unwrap().identity;
        let expected_sha: [u8; 32] = Sha256::digest(bytes).into();
        let mut cumulative = Some(0);
        let mut exact = IoBudget {
            until: until(),
            cancel: &cancel,
            remaining: bytes.len() as u64,
            used: 0,
            cumulative: &mut cumulative,
        };
        readback(
            &mut file,
            identity,
            bytes.len() as u64,
            1,
            expected_sha,
            &mut exact,
        )
        .unwrap();
        assert_eq!(exact.remaining, 0);
        assert_eq!(exact.used, bytes.len() as u64);
        assert_eq!(*exact.cumulative, Some(bytes.len() as u64));
        assert_eq!(file.stream_position().unwrap(), bytes.len() as u64);

        let mut zero_cumulative = Some(0);
        let mut zero = IoBudget {
            until: until(),
            cancel: &cancel,
            remaining: 0,
            used: 0,
            cumulative: &mut zero_cumulative,
        };
        let rejected = readback(
            &mut file,
            identity,
            bytes.len() as u64,
            1,
            expected_sha,
            &mut zero,
        );
        assert!(matches!(rejected, Err(IoError::ByteBudget)));
        assert_eq!(zero.used, 0);
        assert_eq!(*zero.cumulative, Some(0));
        assert_eq!(file.stream_position().unwrap(), 0);

        let mut append = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        append.write_all(b" appended").unwrap();
        let changed_extent = readback(
            &mut file,
            identity,
            bytes.len() as u64,
            1,
            expected_sha,
            &mut zero,
        );
        assert!(matches!(
            changed_extent,
            Err(IoError::Boundary("retirement readback descriptor changed"))
        ));
        assert_eq!(zero.used, 0);
        assert_eq!(file.stream_position().unwrap(), 0);
        assert_eq!(std::fs::read(path).unwrap(), b"exact extent appended");
    }
}
