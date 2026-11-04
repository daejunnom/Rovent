//! Portable checkpoint publication from a current actual subprocess owner.
//! The last commit publishes immutable data only. It cannot recreate a live
//! native witness, current process custody, or tensor-validation authority.
use super::super::linux;
use super::r_auxiliary_ledger::RoventAuxiliaryLedger;
use super::*;
use crate::pals_collect_v3::training_process::{directory_file, open_directory};
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::{Dir, OpenOptions, OpenOptionsExt};
use nix::sys::stat::{Mode, mkdirat};
use sha2::{Digest as _, Sha256};
use std::{
    io::Write,
    os::unix::fs::{FileExt, MetadataExt},
};
const MAX_ARCHIVE_FILES: usize = 262_144;
const MAX_ARCHIVE_DIRECTORIES: usize = 8_192;

/// The caller supplies an actual held managed parent directory and a fresh
/// basename. A saved path/JSON is insufficient to construct the live source.
pub struct RoventCheckpointArchiveTarget {
    pub parent_directory: File,
    pub parent_path: PathBuf,
    pub leaf_name: String,
    pub maximum_output_bytes: u64,
    pub deadline: Instant,
    pub cancel: Arc<AtomicBool>,
    pub read_budget: Arc<RoventRecoveryReadBudget>,
}
struct ArchiveFile {
    relative: String,
    file: File,
    pin: Option<RoventTrainingFilePin>,
}
struct ArchiveBuilder {
    target: RoventCheckpointArchiveTarget,
    root: PathBuf,
    leaf_created: bool,
    directories: BTreeMap<String, Dir>,
    files: Vec<ArchiveFile>,
    output_reserved: u64,
    output_written: u64,
    output_accounting_unknown: bool,
    publication_linked: bool,
    started: Instant,
    copied_ns: Option<u64>,
    readback_ns: Option<u64>,
    committed_ns: Option<u64>,
    finished_ns: Option<u64>,
}

/// Format-independent destination I/O only. The caller must supply its own
/// original typed live boundary before selecting source files. No method here
/// accepts a training-domain declaration or manufactures a completion ACK.
pub(in crate::rovent_training_process) struct CheckpointArchiveIo(ArchiveBuilder);
impl CheckpointArchiveIo {
    pub(in crate::rovent_training_process) fn new(target: RoventCheckpointArchiveTarget) -> Self {
        Self(ArchiveBuilder {
            root: target.parent_path.join(&target.leaf_name),
            target,
            leaf_created: false,
            directories: BTreeMap::new(),
            files: Vec::new(),
            output_reserved: 0,
            output_written: 0,
            output_accounting_unknown: false,
            publication_linked: false,
            started: Instant::now(),
            copied_ns: None,
            readback_ns: None,
            committed_ns: None,
            finished_ns: None,
        })
    }
    pub(in crate::rovent_training_process) fn prepare(&mut self) -> Result<(), ArenaError> {
        self.0.prepare()
    }
    pub(in crate::rovent_training_process) fn root(&self) -> &Path {
        &self.0.root
    }
    pub(in crate::rovent_training_process) fn read_budget(&self) -> &Arc<RoventRecoveryReadBudget> {
        &self.0.target.read_budget
    }
    pub(in crate::rovent_training_process) fn reserved_output_bytes(&self) -> u64 {
        self.0.output_reserved
    }
    pub(in crate::rovent_training_process) fn copy_file(
        &mut self,
        relative: &str,
        pin: &RoventTrainingFilePin,
        file: &File,
    ) -> Result<RoventPublicationFile, ArenaError> {
        self.0.copy_file(relative, pin, file)
    }
    pub(in crate::rovent_training_process) fn write_bytes(
        &mut self,
        relative: &str,
        raw: &[u8],
    ) -> Result<RoventPublicationFile, ArenaError> {
        self.0.write_bytes(relative, raw)
    }
    pub(in crate::rovent_training_process) fn absolute(
        &self,
        reference: &RoventPublicationFile,
    ) -> Result<RoventTrainingFilePin, ArenaError> {
        self.0.absolute(reference)
    }
    pub(in crate::rovent_training_process) fn commit_publication(
        &mut self,
        pending: &RoventPublicationFile,
    ) -> Result<RoventTrainingFilePin, ArenaError> {
        self.0.commit_publication(pending)
    }
    pub(in crate::rovent_training_process) fn named_root(&self) -> Result<(), ArenaError> {
        self.0.named_root()
    }
    pub(in crate::rovent_training_process) fn observation(&self, committed: bool) -> Value {
        json!({"schema":"rz-rovent-checkpoint-archive-io-observation/1",
            "root":self.0.root,"leaf_created":self.0.leaf_created,
            "output_reserved_bytes":self.0.output_reserved,"output_written_bytes":self.0.output_written,
            "output_accounting_unknown":self.0.output_accounting_unknown,
            "read_accounting":self.0.target.read_budget.observation(),
            "publication_linked":self.0.publication_linked,"publication_committed":committed,
            "maximum_files":MAX_ARCHIVE_FILES,"maximum_directories":MAX_ARCHIVE_DIRECTORIES,
            "held_files":self.0.files.len(),"held_directories":self.0.directories.len(),
            "scope":"bounded_destination_io_only_not_source_or_tensor_authority",
            "read_credit_refunded":false,"output_reservations_refunded":false,"execution_authorized":false})
    }
    pub(in crate::rovent_training_process) fn revalidate(
        &self,
        budget: &RoventRecoveryReadBudget,
    ) -> Result<(), ArenaError> {
        budget.reserve(0)?;
        self.0.named_root()?;
        for held in &self.0.files {
            let pin = held
                .pin
                .as_ref()
                .ok_or_else(|| invalid("archive I/O retains an unfinished file"))?;
            check(
                pin.path == self.0.root.join(&held.relative),
                "archive I/O retained original name differs",
            )?;
            budget.verify_held(&held.file, pin, false)?;
        }
        self.0.named_root()?;
        budget.reserve(0)
    }
}
/// Failure retains the original target FD, created directory/file FDs and
/// partial bytes. No Drop cleanup, retry, deletion or old process action runs.
pub struct RoventCheckpointArchiveFailure {
    pub error: ArenaError,
    builder: ArchiveBuilder,
}
impl std::fmt::Debug for RoventCheckpointArchiveFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoventCheckpointArchiveFailure")
            .field("error", &self.error)
            .field("observation", &self.observation())
            .finish_non_exhaustive()
    }
}
impl RoventCheckpointArchiveFailure {
    pub fn observation(&self) -> Value {
        self.builder.observation(false)
    }
}
/// Exact published data, backed by retained destination descriptors. This is
/// neither an execution permit nor acceptance of the recovered tensor state.
pub struct PublishedRoventCheckpointArchive {
    publication: RoventCheckpointPublication,
    pin: RoventTrainingFilePin,
    builder: ArchiveBuilder,
}
enum ActualPublicationRun<'a> {
    Closed(&'a OriginalProcessOutput),
    Boundary,
}
struct ActualPublicationSource<'a> {
    registration: &'a RoventRecoveredTrainingRegistration,
    journal: &'a TrainerLedger,
    held: &'a linux::LinuxHeld,
    dataset: &'a Arc<VerifiedRoventDataset>,
    checkpoint: Option<&'a Arc<VerifiedRoventCheckpointSource>>,
    child_identity: RoventHistoricalChildIdentity,
    expected_window: Option<&'a RoventTrainWindowV2>,
    run: ActualPublicationRun<'a>,
    r_auxiliary_ledger: Option<RoventAuxiliaryLedger>,
}
impl PublishedRoventCheckpointArchive {
    pub fn publication(&self) -> &RoventCheckpointPublication {
        &self.publication
    }
    pub fn publication_pin(&self) -> &RoventTrainingFilePin {
        &self.pin
    }
    pub fn root(&self) -> &Path {
        &self.builder.root
    }
    pub fn observation(&self) -> Value {
        let mut value = self.builder.observation(true);
        value["publication"] = json!(self.pin);
        value
    }
    /// Recheck the actual destination descriptors and their current names
    /// after child retirement. This is a bounded byte/identity observation,
    /// not Rules replay, tensor validation, or a new native execution lease.
    /// The caller supplies its current shared read budget/deadline; an expired
    /// original publication budget is never silently renewed here.
    pub(super) fn revalidate(
        &self,
        shared_budget: &RoventRecoveryReadBudget,
    ) -> Result<(), ArenaError> {
        shared_budget.reserve(0)?;
        check(
            self.builder.publication_linked
                && self.builder.committed_ns.is_some()
                && self.pin.path == self.builder.root.join("publication.json"),
            "archive revalidation requires the actual committed publication owner",
        )?;
        self.builder.named_root()?;
        let mut observed_publication = false;
        for held in &self.builder.files {
            let pin = held
                .pin
                .as_ref()
                .ok_or_else(|| invalid("committed archive retains an unsealed partial file"))?;
            check(
                pin.path == self.builder.root.join(&held.relative),
                "archive retained file name differs from its original pin",
            )?;
            // verify_held checks the complete byte hash/length, single link,
            // and exact held-versus-current-named inode before and after read.
            shared_budget.verify_held(&held.file, pin, false)?;
            if held.relative == "publication.json" {
                check(
                    !observed_publication && pin == &self.pin,
                    "archive final publication pin/FD is ambiguous",
                )?;
                observed_publication = true;
            }
        }
        check(
            observed_publication,
            "committed archive lost its final publication FD",
        )?;
        self.builder.named_root()?;
        shared_budget.reserve(0)
    }
}

/// Only a successful new actual subprocess owner can enter this factory. The
/// caller retains that source owner across failure; no raw-publication factory
/// can stand in for it. The selected checkpoint may precede later updates.
pub fn publish_rovent_training_checkpoint(
    source: &VerifiedRoventRecoveredTrainingWindow,
    selected_checkpoint: usize,
    target: RoventCheckpointArchiveTarget,
) -> Result<PublishedRoventCheckpointArchive, Box<RoventCheckpointArchiveFailure>> {
    publish_input(
        Ok(ActualPublicationSource {
            registration: &source.registration,
            journal: &source.journal,
            held: &source.held,
            dataset: &source.dataset,
            checkpoint: source.checkpoint.as_ref(),
            child_identity: source.child_identity.clone(),
            expected_window: None,
            run: ActualPublicationRun::Closed(&source.process),
            r_auxiliary_ledger: source.r_auxiliary_ledger.clone(),
        }),
        selected_checkpoint,
        target,
    )
}
/// The only pre-terminal entry point accepts the runtime's private current
/// checkpoint boundary borrow, created after real peer admission and original
/// checkpoint retirement. It never accepts fabricated process/terminal JSON.
pub(super) fn publish_rovent_training_checkpoint_boundary(
    source: &runtime::ActualRoventCheckpointBoundary<'_>,
    target: RoventCheckpointArchiveTarget,
) -> Result<PublishedRoventCheckpointArchive, Box<RoventCheckpointArchiveFailure>> {
    let input = source.child_identity().and_then(|child_identity| {
        Ok(ActualPublicationSource {
            registration: source.registration(),
            journal: source.journal(),
            held: source.held(),
            dataset: source.dataset(),
            checkpoint: source.checkpoint_history(),
            child_identity,
            expected_window: Some(source.window()),
            run: ActualPublicationRun::Boundary,
            r_auxiliary_ledger: source.r_auxiliary_ledger()?,
        })
    });
    publish_input(input, source.checkpoint_index(), target)
}
fn publish_input(
    source: Result<ActualPublicationSource<'_>, ArenaError>,
    selected_checkpoint: usize,
    target: RoventCheckpointArchiveTarget,
) -> Result<PublishedRoventCheckpointArchive, Box<RoventCheckpointArchiveFailure>> {
    let mut builder = ArchiveBuilder {
        root: target.parent_path.join(&target.leaf_name),
        target,
        leaf_created: false,
        directories: BTreeMap::new(),
        files: Vec::new(),
        output_reserved: 0,
        output_written: 0,
        output_accounting_unknown: false,
        publication_linked: false,
        started: Instant::now(),
        copied_ns: None,
        readback_ns: None,
        committed_ns: None,
        finished_ns: None,
    };
    let result = (|| {
        let source = source?;
        builder.prepare()?;
        let publication = builder.copy_actual(&source, selected_checkpoint)?;
        builder.copied_ns = u64::try_from(builder.started.elapsed().as_nanos()).ok();
        publication.validate()?;
        let raw = serde_json::to_vec(&publication).map_err(|e| invalid(e.to_string()))?;
        check(
            raw.len() as u64 <= MAX_METADATA_BYTES,
            "portable checkpoint publication byte cap",
        )?;
        let pending = builder.write_bytes("publication.pending.json", &raw)?;
        // Re-run all historical byte, row, update-chain, boundary and closure
        // checks before the final name becomes visible. This produces only a
        // temporary metadata owner, never live or tensor validation authority.
        let verified = VerifiedRoventCheckpointSource::open_with_r_auxiliary(
            &builder.root,
            builder.absolute(&pending)?,
            Arc::clone(source.dataset),
            source.dataset.ancestors.clone(),
            actual_r_auxiliary_datasets(&source)?,
            &builder.target.read_budget,
        )?;
        check(
            verified.publication() == &publication,
            "portable publication readback changed metadata",
        )?;
        drop(verified);
        builder.readback_ns = u64::try_from(builder.started.elapsed().as_nanos()).ok();
        let pin = builder.commit_publication(&pending)?;
        builder.committed_ns = u64::try_from(builder.started.elapsed().as_nanos()).ok();
        Ok((publication, pin))
    })();
    builder.finished_ns = u64::try_from(builder.started.elapsed().as_nanos()).ok();
    match result {
        Ok((publication, pin)) => Ok(PublishedRoventCheckpointArchive {
            publication,
            pin,
            builder,
        }),
        Err(error) => Err(Box::new(RoventCheckpointArchiveFailure { error, builder })),
    }
}
impl ArchiveBuilder {
    fn observation(&self, committed: bool) -> Value {
        json!({"schema":"rz-rovent-checkpoint-archive-observation/1",
        "root":self.root,"leaf_created":self.leaf_created,"output_reserved_bytes":self.output_reserved,
        "output_written_bytes":self.output_written,"output_accounting_unknown":self.output_accounting_unknown,
        "read_accounting":self.target.read_budget.observation(),"publication_linked":self.publication_linked,
        "maximum_files":MAX_ARCHIVE_FILES,"maximum_directories":MAX_ARCHIVE_DIRECTORIES,
        "held_files":self.files.len(),"held_directories":self.directories.len(),
        "archive_wall_ns":self.finished_ns,"copy_phase_finished_ns":self.copied_ns,"readback_phase_finished_ns":self.readback_ns,
        "commit_phase_finished_ns":self.committed_ns,"clock_scope":"current_parent_monotonic_from_publication_entry",
        "read_credit_refunded":false,"output_reservations_refunded":false,
        "publication_committed":committed,"tensor_state_validated":false,"execution_authorized":false,
        "past_native_authority_reissued":false,"past_physical_completion_reissued":false,
        "archive_copy_cost_is_separate_from_recorded_training_cost":true})
    }
    fn active(&self) -> Result<(), ArenaError> {
        check(
            Instant::now() < self.target.deadline && !self.target.cancel.load(Ordering::Acquire),
            "portable publication cancelled/expired",
        )
    }
    fn prepare(&mut self) -> Result<(), ArenaError> {
        self.active()?;
        check(
            self.target.parent_path.is_absolute()
                && self
                    .target
                    .parent_path
                    .components()
                    .all(|p| matches!(p, Component::RootDir | Component::Normal(_)))
                && generation_identifier(&self.target.leaf_name)
                && self.target.leaf_name.len() <= 128
                && !self.target.leaf_name.contains('/')
                && !self.target.leaf_name.contains('\\')
                && (1..=MAX_TRAINER_OUTPUT_BYTES).contains(&self.target.maximum_output_bytes),
            "managed finite portable archive target",
        )?;
        let meta = self
            .target
            .parent_directory
            .metadata()
            .map_err(|_| invalid("archive original parent metadata"))?;
        check(
            meta.is_dir() && meta.uid() == nix::unistd::geteuid().as_raw(),
            "archive parent must be an original current-owner directory",
        )?;
        let named = directory_file(&open_directory(&self.target.parent_path)?)?
            .metadata()
            .map_err(|_| invalid("archive named parent metadata"))?;
        check(
            meta.dev() == named.dev() && meta.ino() == named.ino(),
            "archive named parent differs from original descriptor",
        )?;
        mkdirat(
            &self.target.parent_directory,
            self.target.leaf_name.as_str(),
            Mode::S_IRWXU,
        )
        .map_err(|_| invalid("archive leaf must be fresh and exclusive"))?;
        self.leaf_created = true;
        let parent = Dir::from_std_file(
            self.target
                .parent_directory
                .try_clone()
                .map_err(|_| invalid("archive parent FD clone"))?,
        );
        let root = parent
            .open_dir_nofollow(&self.target.leaf_name)
            .map_err(|_| invalid("archive new leaf nofollow open"))?;
        // Retain the first opened descriptor before metadata/sync checks.
        self.directories.insert(String::new(), root);
        let meta = directory_file(self.root_dir()?)?
            .metadata()
            .map_err(|_| invalid("archive new leaf metadata"))?;
        check(
            meta.uid() == nix::unistd::geteuid().as_raw() && meta.mode() & 0o777 == 0o700,
            "archive leaf ownership/mode changed",
        )?;
        self.named_root()?;
        self.target
            .parent_directory
            .sync_all()
            .map_err(|_| invalid("archive parent sync"))
    }
    fn root_dir(&self) -> Result<&Dir, ArenaError> {
        self.directories
            .get("")
            .ok_or_else(|| invalid("original archive root descriptor absent"))
    }
    fn named_root(&self) -> Result<(), ArenaError> {
        let held = directory_file(self.root_dir()?)?
            .metadata()
            .map_err(|_| invalid("archive held root metadata"))?;
        let named = directory_file(&open_directory(&self.root)?)?
            .metadata()
            .map_err(|_| invalid("archive named root metadata"))?;
        check(
            held.dev() == named.dev() && held.ino() == named.ino(),
            "archive original root path was replaced",
        )
    }
    fn parent_dir(&mut self, relative: &str) -> Result<Dir, ArenaError> {
        let item = RoventPublicationFile {
            path: relative.into(),
            sha256: "a".repeat(64),
            bytes: 1,
        };
        item.validate()?;
        let mut prefix = String::new();
        let parts = relative.split('/').collect::<Vec<_>>();
        for name in &parts[..parts.len() - 1] {
            let next = if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{prefix}/{name}")
            };
            if !self.directories.contains_key(&next) {
                self.active()?;
                check(
                    self.directories.len() < MAX_ARCHIVE_DIRECTORIES,
                    "portable archive directory capacity exhausted",
                )?;
                let parent = self
                    .directories
                    .get(&prefix)
                    .ok_or_else(|| invalid("archive retained parent directory absent"))?;
                mkdirat(&directory_file(parent)?, *name, Mode::S_IRWXU)
                    .map_err(|_| invalid("archive subdirectory must be newly owned"))?;
                let child = parent
                    .open_dir_nofollow(name)
                    .map_err(|_| invalid("archive subdirectory nofollow open"))?;
                self.directories.insert(next.clone(), child);
                let child = self
                    .directories
                    .get(&next)
                    .ok_or_else(|| invalid("archive new directory owner absent"))?;
                let stat = directory_file(child)?
                    .metadata()
                    .map_err(|_| invalid("archive directory metadata"))?;
                check(
                    stat.uid() == nix::unistd::geteuid().as_raw() && stat.mode() & 0o777 == 0o700,
                    "archive directory owner/mode",
                )?;
            }
            prefix = next;
        }
        let parent = self
            .directories
            .get(&prefix)
            .ok_or_else(|| invalid("archive original parent absent"))?;
        Ok(Dir::from_std_file(directory_file(parent)?))
    }
    fn create_file(&mut self, relative: &str, bytes: u64) -> Result<(usize, Dir), ArenaError> {
        self.active()?;
        self.named_root()?;
        check(
            self.files.len() < MAX_ARCHIVE_FILES,
            "portable archive file/FD capacity exhausted",
        )?;
        check(
            bytes > 0 && bytes <= MAX_TRAINER_OUTPUT_BYTES,
            "portable file byte bounds",
        )?;
        self.output_reserved = self
            .output_reserved
            .checked_add(bytes)
            .filter(|n| *n <= self.target.maximum_output_bytes)
            .ok_or_else(|| {
                invalid("portable archive output quota exhausted; reservations are not refunded")
            })?;
        check(
            !self.files.iter().any(|f| f.relative == relative),
            "portable destination path already owned",
        )?;
        let parent = self.parent_dir(relative)?;
        let name = Path::new(relative)
            .file_name()
            .ok_or_else(|| invalid("portable basename"))?;
        let mut options = OpenOptions::new();
        options
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o400)
            .follow(FollowSymlinks::No)
            .custom_flags(libc::O_CLOEXEC | libc::O_NONBLOCK);
        let file = parent
            .open_with(name, &options)
            .map_err(|_| invalid("portable file exclusive create"))?
            .into_std();
        let slot = self.files.len();
        self.files.push(ArchiveFile {
            relative: relative.into(),
            file,
            pin: None,
        });
        let stat = self.files[slot]
            .file
            .metadata()
            .map_err(|_| invalid("portable created file metadata"))?;
        check(
            stat.is_file()
                && stat.len() == 0
                && stat.nlink() == 1
                && stat.uid() == nix::unistd::geteuid().as_raw(),
            "portable file original ownership/type",
        )?;
        Ok((slot, parent))
    }
    fn write_chunk(&mut self, slot: usize, raw: &[u8]) -> Result<(), ArenaError> {
        let mut offset = 0;
        while offset < raw.len() {
            self.active()?;
            let n = self.files[slot].file.write(&raw[offset..]).map_err(|_| {
                self.output_accounting_unknown = true;
                invalid("portable file write failed; partial bytes retained")
            })?;
            check(n > 0, "portable write made no progress")?;
            offset += n;
            self.output_written = self
                .output_written
                .checked_add(n as u64)
                .ok_or_else(|| invalid("portable observed write count overflow"))?;
        }
        Ok(())
    }
    fn seal_file(
        &mut self,
        slot: usize,
        parent: &Dir,
        sha256: String,
        bytes: u64,
    ) -> Result<RoventPublicationFile, ArenaError> {
        self.active()?;
        self.files[slot]
            .file
            .sync_all()
            .map_err(|_| invalid("portable file fsync"))?;
        directory_file(parent)?
            .sync_all()
            .map_err(|_| invalid("portable containing directory fsync"))?;
        let pin = RoventTrainingFilePin {
            path: self.root.join(&self.files[slot].relative),
            sha256,
            bytes,
        };
        self.target
            .read_budget
            .verify_held(&self.files[slot].file, &pin, false)?;
        let stat = self.files[slot]
            .file
            .metadata()
            .map_err(|_| invalid("portable sealed metadata"))?;
        check(
            stat.is_file() && stat.len() == bytes && stat.nlink() == 1,
            "portable sealed length/link count",
        )?;
        self.files[slot].pin = Some(pin.clone());
        self.named_root()?;
        Ok(RoventPublicationFile {
            path: self.files[slot].relative.clone(),
            sha256: pin.sha256,
            bytes,
        })
    }
    fn write_bytes(
        &mut self,
        relative: &str,
        raw: &[u8],
    ) -> Result<RoventPublicationFile, ArenaError> {
        check(
            !raw.is_empty() && raw.len() as u64 <= MAX_METADATA_BYTES,
            "bounded generated portable metadata",
        )?;
        let (slot, parent) = self.create_file(relative, raw.len() as u64)?;
        self.write_chunk(slot, raw)?;
        self.seal_file(
            slot,
            &parent,
            format!("{:x}", Sha256::digest(raw)),
            raw.len() as u64,
        )
    }
    fn copy_file(
        &mut self,
        relative: &str,
        pin: &RoventTrainingFilePin,
        file: &File,
    ) -> Result<RoventPublicationFile, ArenaError> {
        if let Some(old) = self.files.iter().find(|f| f.relative == relative) {
            let old = old
                .pin
                .as_ref()
                .ok_or_else(|| invalid("portable duplicate refers to an unfinished file"))?;
            check(
                old.sha256 == pin.sha256 && old.bytes == pin.bytes,
                "portable source path collision",
            )?;
            self.target.read_budget.verify_held(file, pin, false)?;
            return Ok(RoventPublicationFile {
                path: relative.into(),
                sha256: pin.sha256.clone(),
                bytes: pin.bytes,
            });
        }
        check(
            generation_sha(&pin.sha256) && pin.bytes > 0 && pin.bytes <= MAX_TRAINER_OUTPUT_BYTES,
            "portable original byte pin",
        )?;
        let original_metadata = file
            .metadata()
            .map_err(|_| invalid("portable original FD metadata"))?;
        check(
            original_metadata.is_file()
                && original_metadata.len() == pin.bytes
                && original_metadata.nlink() == 1,
            "portable original file must match the complete byte length and single-link type",
        )?;
        let before = FileIdentity::from(&original_metadata);
        let (slot, parent) = self.create_file(relative, pin.bytes)?;
        let mut hash = Sha256::new();
        let mut offset = 0_u64;
        let mut block = [0_u8; 65536];
        while offset < pin.bytes {
            self.active()?;
            let request = (pin.bytes - offset).min(block.len() as u64) as usize;
            self.target.read_budget.reserve(request as u64)?;
            let n = match file.read_at(&mut block[..request], offset) {
                Ok(n) if n > 0 => n,
                _ => {
                    self.target
                        .read_budget
                        .uncertain
                        .store(true, Ordering::Release);
                    return Err(invalid(
                        "portable original read failed/ended early; credit retained and unknown",
                    ));
                }
            };
            hash.update(&block[..n]);
            self.write_chunk(slot, &block[..n])?;
            offset += n as u64;
        }
        check(
            format!("{:x}", hash.finalize()) == pin.sha256
                && before
                    == FileIdentity::from(
                        &file
                            .metadata()
                            .map_err(|_| invalid("portable original after-copy metadata"))?,
                    ),
            "portable original bytes/identity changed while copying",
        )?;
        self.seal_file(slot, &parent, pin.sha256.clone(), pin.bytes)
    }
    fn absolute(&self, file: &RoventPublicationFile) -> Result<RoventTrainingFilePin, ArenaError> {
        archive_pin(&self.root, file)
    }
    fn copy_historical(
        &mut self,
        source: &VerifiedRoventCheckpointSource,
        relative_publication: &str,
        depth: usize,
    ) -> Result<RoventPublicationFile, ArenaError> {
        check(depth < 64, "portable historical checkpoint ancestry bound")?;
        self.active()?;
        let (current, raw) =
            self.target
                .read_budget
                .read(source.publication_pin(), MAX_METADATA_BYTES, false)?;
        debug_assert!(raw.is_empty());
        let publication = self.copy_file(relative_publication, &current.pin, &current.file)?;
        for artifact in source.publication().artifacts() {
            if source
                .publication()
                .parent_publication
                .as_ref()
                .is_some_and(|p| p.path == artifact.path)
            {
                continue;
            }
            if artifact.path == source.publication().tensor.path {
                let (pin, file) = source.tensor_file();
                self.copy_file(&artifact.path, pin, file)?;
            } else {
                let (old, _) = self.target.read_budget.read(
                    &archive_pin(source.archive_root(), artifact)?,
                    MAX_TRAINER_OUTPUT_BYTES,
                    false,
                )?;
                self.copy_file(&artifact.path, &old.pin, &old.file)?;
            }
        }
        if let Some(parent) = source.parent() {
            let relative = source
                .publication()
                .parent_publication
                .as_ref()
                .ok_or_else(|| invalid("historical parent owner lacks declared pin"))?;
            let copied = self.copy_historical(parent, &relative.path, depth + 1)?;
            check(
                copied == *relative,
                "portable parent publication changed exact bytes",
            )?;
        }
        Ok(publication)
    }
    fn copy_actual(
        &mut self,
        source: &ActualPublicationSource<'_>,
        selected: usize,
    ) -> Result<RoventCheckpointPublication, ArenaError> {
        self.active()?;
        let journal = source.journal;
        let receipt = match &source.run {
            ActualPublicationRun::Closed(process) => {
                let _actual_timing = process.checked_timing()?;
                let _actual_loaded_image = process.checked_loaded_image()?;
                let receipt = &process.process().receipt;
                check(
                    journal.ended
                        && receipt.group_cleanup == crate::process::CleanupStatus::Gone
                        && receipt.exit_code == Some(0)
                        && receipt.exit_signal.is_none(),
                    "portable final publication requires actual current process closure",
                )?;
                Some(receipt)
            }
            ActualPublicationRun::Boundary => {
                check(
                    !journal.ended
                        && selected.checked_add(1) == Some(journal.checkpoint_states.len()),
                    "boundary publication requires the currently retired checkpoint",
                )?;
                None
            }
        };
        check(
            source.registration.action == RoventRecoveredTrainingAction::Train
                && journal.begun
                && journal.pending.is_empty(),
            "portable publication requires an actual committed accumulation boundary",
        )?;
        let spec: RoventTrainWindowV2 =
            serde_json::from_value(journal.window.clone()).map_err(|e| invalid(e.to_string()))?;
        spec.validate()?;
        check(
            source
                .expected_window
                .is_none_or(|original| original == &spec),
            "boundary current typed window differs from actual journal",
        )?;
        let state = journal
            .checkpoint_states
            .get(selected)
            .ok_or_else(|| invalid("selected actual checkpoint does not exist"))?;
        let tensor_pin: RoventTrainingFilePin = serde_json::from_value(state["checkpoint"].clone())
            .map_err(|e| invalid(e.to_string()))?;
        let retirement_pin: RoventTrainingFilePin = serde_json::from_value(
            journal
                .checkpoint_receipts
                .get(selected)
                .ok_or_else(|| invalid("selected actual checkpoint has no original retirement"))?
                .clone(),
        )
        .map_err(|e| invalid(e.to_string()))?;
        let namespace = format!(
            "windows/{:x}",
            Sha256::digest(
                format!("{}\0{}", source.registration.execution_id, spec.window_id).as_bytes()
            )
        );
        let copy_live = |this: &mut Self,
                         pin: &RoventTrainingFilePin,
                         name: &str|
         -> Result<RoventPublicationFile, ArenaError> {
            let held = find_live_file(source.held, pin)?;
            this.copy_file(&format!("{namespace}/{name}"), pin, &held.file)
        };
        let tensor = copy_live(self, &tensor_pin, "tensor.pt")?;
        let checkpoint_retirement = copy_live(self, &retirement_pin, "checkpoint.json")?;
        let terminal_receipt = if receipt.is_some() {
            let terminal_pin = source
                .held
                .journals
                .iter()
                .find(|f| f.pin.path == source.registration.receipt_path)
                .ok_or_else(|| invalid("actual terminal original FD absent"))?
                .pin
                .clone();
            copy_live(self, &terminal_pin, "terminal.json")?
        } else {
            self.write_bytes(&format!("{namespace}/boundary-report.json"),&serde_json::to_vec(&json!({"schema":ROVENT_CHECKPOINT_BOUNDARY_REPORT,
                "execution_id":source.registration.execution_id,"window_id":spec.window_id,"completed_updates":state["completed_updates"],
                "physical_completion":"unknown","arena_eligible":false,"status":"checkpoint_boundary_only",
                "checkpoint_retirement_sha256":checkpoint_retirement.sha256})).map_err(|e|invalid(e.to_string()))?)?
        };
        let original_recipe = copy_live(self, &source.registration.recipe, "recipe.json")?;
        let original_membership =
            copy_live(self, &source.registration.membership, "membership.json")?;
        let original_window_spec =
            copy_live(self, &source.registration.window_spec, "window.json")?;
        let mut update_receipts = Vec::with_capacity(journal.updates.len());
        for i in 0..journal.updates.len() {
            let path = source
                .registration
                .control_root
                .join(format!("update-{:06}.json", i + 1));
            let held = source
                .held
                .journals
                .iter()
                .find(|f| f.pin.path == path)
                .ok_or_else(|| invalid("original actual update FD absent"))?;
            let raw = self
                .target
                .read_budget
                .verify_held(&held.file, &held.pin, true)?;
            let value: Value = decode(&raw)?;
            check(
                value["update"] == journal.updates[i] && value["sequence"] == (i + 1) as u64,
                "original actual update FD differs from committed journal",
            )?;
            update_receipts.push(copy_live(
                self,
                &held.pin,
                &format!("update-{:06}.json", i + 1),
            )?);
        }
        let parent_publication = source
            .checkpoint
            .as_ref()
            .map(|parent| {
                parent.revalidate(&self.target.read_budget)?;
                self.copy_historical(
                    parent,
                    &format!("parents/{}.json", parent.publication_pin().sha256),
                    0,
                )
            })
            .transpose()?;
        let initialization_source = match &source.registration.initialization_source {
            RoventTrainingInputSource::Initialize {
                pc_checkpoint,
                nnuec_checkpoint,
            } => {
                check(
                    source.checkpoint.is_none(),
                    "actual initialize unexpectedly owns a historical parent",
                )?;
                RoventCheckpointInitializationSource::Initialize {
                    pc_checkpoint_sha256: pc_checkpoint.sha256.clone(),
                    nnuec_checkpoint_sha256: nnuec_checkpoint.sha256.clone(),
                }
            }
            RoventTrainingInputSource::Resume { .. } => {
                RoventCheckpointInitializationSource::Resume {
                    tensor_checkpoint_sha256: source
                        .checkpoint
                        .as_ref()
                        .ok_or_else(|| invalid("actual resume historical owner absent"))?
                        .tensor_file()
                        .0
                        .sha256
                        .clone(),
                }
            }
            RoventTrainingInputSource::Fork {
                parent_stage_id, ..
            } => RoventCheckpointInitializationSource::Fork {
                tensor_checkpoint_sha256: source
                    .checkpoint
                    .as_ref()
                    .ok_or_else(|| invalid("actual fork historical owner absent"))?
                    .tensor_file()
                    .0
                    .sha256
                    .clone(),
                parent_stage_id: parent_stage_id.clone(),
            },
        };
        let boundary = actual_boundary(source, state)?;
        let r_auxiliary = match &spec.nnuec_r_auxiliary {
            RoventRAuxiliaryPresence::Absent => {
                check(
                    source.r_auxiliary_ledger.is_none(),
                    "disabled actual window retained an R ledger",
                )?;
                RoventRAuxiliaryPresence::Absent
            }
            RoventRAuxiliaryPresence::Null => {
                check(
                    source.r_auxiliary_ledger.is_none() && state.get("r_auxiliary").is_none(),
                    "explicit null actual R checkpoint changed presence",
                )?;
                RoventRAuxiliaryPresence::Null
            }
            RoventRAuxiliaryPresence::Enabled(recipe) => {
                let prefix = actual_r_auxiliary_prefix(
                    source,
                    &spec,
                    &boundary.completed_updates,
                    self.target.deadline,
                    &self.target.cancel,
                )?;
                prefix.validate_checkpoint_projection(&state["r_auxiliary"])?;
                let dataset = prefix.dataset();
                dataset.revalidate(&self.target.read_budget)?;
                check(
                    dataset.manifest_file_sha256() == recipe.manifest_file_sha256
                        && dataset.index_file_sha256() == recipe.index_file_sha256,
                    "actual R checkpoint changed original dataset file pins",
                )?;
                let parent_ledger_projection = self.write_bytes(
                    &format!("{namespace}/r-auxiliary-parent-ledger.json"),
                    &serde_json::to_vec(&prefix.observation())
                        .map_err(|e| invalid(e.to_string()))?,
                )?;
                RoventRAuxiliaryPresence::Enabled(RoventCheckpointRAuxiliary {
                    schema: ROVENT_CHECKPOINT_R_AUXILIARY.into(),
                    dataset_publication_sha256: dataset.publication_pin().sha256.clone(),
                    manifest_file_sha256: dataset.manifest_file_sha256().to_owned(),
                    index_file_sha256: dataset.index_file_sha256().to_owned(),
                    parent_ledger_projection,
                })
            }
        };
        let recorded_tail_cost = if receipt.is_some() {
            let end_cost = journal
                .end_cost
                .as_ref()
                .ok_or_else(|| invalid("actual current end cost is unknown"))?;
            let delta = actual_tail_delta(&boundary.snapshot_cost, end_cost)?;
            let evidence=self.write_bytes(&format!("{namespace}/tail-cost.json"),&serde_json::to_vec(&json!({"schema":ROVENT_RECORDED_TAIL_COST,
                "execution_id":source.registration.execution_id,"window_id":spec.window_id,"snapshot_cost":boundary.snapshot_cost,"end_cost":end_cost,"delta":delta})).map_err(|e|invalid(e.to_string()))?)?;
            RoventRecordedTailCost::Known { delta, evidence }
        } else {
            RoventRecordedTailCost::Unknown {
                reason: "original_window_unclosed_after_boundary".into(),
                evidence: None,
            }
        };
        let closure = RoventOriginalTrainingClosure {
            schema: if receipt.is_some() {
                ROVENT_ORIGINAL_TRAINING_CLOSURE
            } else {
                ROVENT_CHECKPOINT_BOUNDARY_CLOSURE
            }
            .into(),
            execution_id: source.registration.execution_id.clone(),
            window_id: spec.window_id.clone(),
            terminal_receipt_sha256: terminal_receipt.sha256.clone(),
            terminal_receipt_bytes: terminal_receipt.bytes,
            process_receipt: receipt.cloned(),
            child_identity: source.child_identity.clone(),
            child_reap: if receipt.is_some() {
                RoventHistoricalChildReap::OwnedChildReaped
            } else {
                RoventHistoricalChildReap::Unresolved
            },
            physical_completion: if receipt.is_some() {
                RoventHistoricalPhysicalCompletion::Confirmed
            } else {
                RoventHistoricalPhysicalCompletion::Unknown
            },
        };
        let closure_file = self.write_bytes(
            &format!("{namespace}/closure.json"),
            &serde_json::to_vec(&closure).map_err(|e| invalid(e.to_string()))?,
        )?;
        let mut ancestor_dataset_publication_sha256 = source
            .dataset
            .ancestors
            .iter()
            .map(|d| d.registration.publication.sha256.clone())
            .collect::<Vec<_>>();
        ancestor_dataset_publication_sha256.sort();
        Ok(RoventCheckpointPublication {
            schema: ROVENT_CHECKPOINT_PUBLICATION.into(),
            execution_id: source.registration.execution_id.clone(),
            window_id: spec.window_id,
            collection_source_sha256: spec.collection_source_sha256,
            collection_runtime_library_sha256: spec.collection_runtime_library_sha256,
            trainer_source_sha256: spec.trainer_source_sha256,
            trainer_python_executable_sha256: spec.trainer_python_executable_sha256,
            trainer_dependency_set_sha256: spec.trainer_dependency_set_sha256,
            training_model_semantics_sha256: spec.training_model_semantics_sha256,
            initialization_source,
            collection_model_epoch: spec.collection_model_epoch,
            collection_context_epoch: spec.collection_context_epoch,
            training_model_epoch: spec.training_model_epoch,
            recipe_sha256: spec.recipe_sha256,
            stage_id: spec.stage_id,
            training_profile_sha256: journal.profile_sha256.clone(),
            membership_sha256: journal.membership_sha256.clone(),
            current_dataset_publication_sha256: source
                .dataset
                .registration
                .publication
                .sha256
                .clone(),
            ancestor_dataset_publication_sha256,
            tensor,
            checkpoint_retirement,
            update_receipts,
            terminal_receipt,
            original_recipe,
            original_membership,
            original_window_spec,
            parent_publication,
            boundary,
            recorded_tail_cost,
            recorded_closure: RoventRecordedClosure {
                process_scope: if receipt.is_some() {
                    "original_owned_child_reaped"
                } else {
                    "original_window_not_closed"
                }
                .into(),
                physical_scope: if receipt.is_some() {
                    "original_end_window_confirmed"
                } else {
                    "checkpoint_boundary_does_not_prove_completion"
                }
                .into(),
                evidence: closure_file,
            },
            r_auxiliary,
        })
    }
    fn commit_publication(
        &mut self,
        pending: &RoventPublicationFile,
    ) -> Result<RoventTrainingFilePin, ArenaError> {
        self.active()?;
        self.named_root()?;
        // Every nested directory entry must be durable before publishing the
        // root name. Syncing only each data file's immediate parent misses an
        // intermediate `windows/<namespace>` entry after a crash.
        let mut directories = self.directories.iter().collect::<Vec<_>>();
        directories.sort_by(|(a, _), (b, _)| {
            b.matches('/')
                .count()
                .cmp(&a.matches('/').count())
                .then_with(|| b.len().cmp(&a.len()))
        });
        for (_, directory) in directories {
            self.active()?;
            directory_file(directory)?
                .sync_all()
                .map_err(|_| invalid("portable archive subtree fsync"))?;
        }
        let slot = self
            .files
            .iter()
            .position(|f| f.relative == pending.path)
            .ok_or_else(|| invalid("pending publication FD absent"))?;
        let root = self.root_dir()?;
        // hard_link is create-new and atomic. It never overwrites a prior
        // publication; all data files and the pending bytes are already synced.
        root.hard_link(&pending.path, root, "publication.json")
            .map_err(|_| invalid("atomic create-new publication link"))?;
        self.publication_linked = true;
        let root = self.root_dir()?;
        let mut options = OpenOptions::new();
        options
            .read(true)
            .follow(FollowSymlinks::No)
            .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC);
        let final_file = root
            .open_with("publication.json", &options)
            .map_err(|_| invalid("publication linked readback"))?
            .into_std();
        let original = self.files[slot]
            .file
            .metadata()
            .map_err(|_| invalid("pending publication metadata"))?;
        let linked = final_file
            .metadata()
            .map_err(|_| invalid("linked publication metadata"))?;
        check(
            original.dev() == linked.dev() && original.ino() == linked.ino() && linked.nlink() == 2,
            "publication link differs from original pending inode",
        )?;
        root.remove_file(&pending.path).map_err(|_| {
            invalid("publication original temp retirement failed; both names retained")
        })?;
        directory_file(root)?
            .sync_all()
            .map_err(|_| invalid("publication directory commit fsync"))?;
        let pin = RoventTrainingFilePin {
            path: self.root.join("publication.json"),
            sha256: pending.sha256.clone(),
            bytes: pending.bytes,
        };
        self.target
            .read_budget
            .verify_held(&self.files[slot].file, &pin, false)?;
        check(
            self.files[slot]
                .file
                .metadata()
                .map_err(|_| invalid("publication final metadata"))?
                .nlink()
                == 1,
            "publication original temp still linked",
        )?;
        self.files[slot].relative = "publication.json".into();
        self.files[slot].pin = Some(pin.clone());
        self.named_root()?;
        Ok(pin)
    }
}
fn find_live_file<'a>(
    held: &'a linux::LinuxHeld,
    pin: &RoventTrainingFilePin,
) -> Result<&'a linux::HeldFile, ArenaError> {
    let mut matches = held
        .inputs
        .iter()
        .chain(&held.checkpoints)
        .chain(&held.journals)
        .chain(&held.artifacts)
        .filter(|f| f.pin.path == pin.path);
    let file = matches
        .next()
        .ok_or_else(|| invalid("actual source original FD absent"))?;
    check(
        matches.next().is_none()
            && file.pin == *pin
            && file.identity
                == FileIdentity::from(
                    &file
                        .file
                        .metadata()
                        .map_err(|_| invalid("actual source FD metadata"))?,
                ),
        "actual source original FD identity/pin is ambiguous or changed",
    )?;
    Ok(file)
}
fn actual_boundary(
    source: &ActualPublicationSource<'_>,
    state: &Value,
) -> Result<RoventCheckpointBoundary, ArenaError> {
    let target: BTreeMap<String, u64> = serde_json::from_value(state["completed_updates"].clone())
        .map_err(|e| invalid(e.to_string()))?;
    let mut inputs = source
        .dataset
        .manifest
        .inherited_input_sha256
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut origins = source
        .dataset
        .manifest
        .inherited_origin_sha256
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if let Some(parent) = &source.checkpoint {
        inputs.extend(parent.retained_inputs().iter().cloned());
        origins.extend(parent.retained_origins().iter().cloned());
    }
    let (inputs, origins) = prefix_sets(
        source.journal.initial.clone(),
        &source.journal.updates,
        inputs,
        origins,
        &target,
    )?;
    let boundary = RoventCheckpointBoundary {
        completed_updates: target,
        ancestor_updates: source.journal.ancestors.clone(),
        model_state_sha256: sha(&state["model_state_sha256"])?.into(),
        optimizer_state_sha256: sha(&state["optimizer_state_sha256"])?.into(),
        scheduler_state_sha256: sha(&state["scheduler_state_sha256"])?.into(),
        rng_state_sha256: sha(&state["rng_state_sha256"])?.into(),
        sampler_state_sha256: sha(&state["sampler_state_sha256"])?.into(),
        retained_input_set_sha256: checkpoint::set_seal("input", &inputs)?,
        retained_input_count: inputs.len() as u64,
        retained_origin_set_sha256: checkpoint::set_seal("origin", &origins)?,
        retained_origin_count: origins.len() as u64,
        snapshot_cost: serde_json::from_value(state["cost"].clone())
            .map_err(|e| invalid(e.to_string()))?,
    };
    Ok(boundary)
}
fn actual_r_auxiliary_datasets(
    source: &ActualPublicationSource<'_>,
) -> Result<Vec<Arc<VerifiedRoventRAuxiliaryDataset>>, ArenaError> {
    fn add(
        owner: &Arc<VerifiedRoventRAuxiliaryDataset>,
        all: &mut BTreeMap<String, Arc<VerifiedRoventRAuxiliaryDataset>>,
        depth: usize,
    ) -> Result<(), ArenaError> {
        check(depth <= 256, "actual R archive dataset ancestry depth")?;
        let pin = &owner.publication_pin().sha256;
        if let Some(previous) = all.get(pin) {
            check(
                previous.manifest_file_sha256() == owner.manifest_file_sha256()
                    && previous.index_file_sha256() == owner.index_file_sha256(),
                "actual R archive dataset publication conflicts with retained owner",
            )?;
            return Ok(());
        }
        check(all.len() < 257, "actual R archive dataset owner capacity")?;
        all.insert(pin.clone(), Arc::clone(owner));
        for ancestor in owner.ancestors() {
            add(ancestor, all, depth + 1)?;
        }
        Ok(())
    }
    let mut all = BTreeMap::new();
    if let Some(auxiliary) = &source.r_auxiliary_ledger {
        add(auxiliary.dataset(), &mut all, 0)?;
    }
    let mut history = source.checkpoint.map(Arc::as_ref);
    let mut depth = 0;
    while let Some(checkpoint) = history {
        depth += 1;
        check(depth <= 64, "actual R archive checkpoint history depth")?;
        if let Some(auxiliary) = checkpoint.r_auxiliary_ledger() {
            add(auxiliary.dataset(), &mut all, 0)?;
        }
        history = checkpoint.parent().map(Arc::as_ref);
    }
    Ok(all.into_values().collect())
}

/// Replays the actual journal against the same opaque dataset rather than
/// truncating a later ledger's counters/sets. The full tail is checked, while
/// only the selected checkpoint prefix is serialized into its sidecar.
fn actual_r_auxiliary_prefix(
    source: &ActualPublicationSource<'_>,
    spec: &RoventTrainWindowV2,
    target: &BTreeMap<String, u64>,
    deadline: Instant,
    cancel: &AtomicBool,
) -> Result<RoventAuxiliaryLedger, ArenaError> {
    let current = source
        .r_auxiliary_ledger
        .as_ref()
        .ok_or_else(|| invalid("actual R checkpoint lacks its current original ledger"))?;
    let recipe = spec
        .nnuec_r_auxiliary
        .enabled()
        .ok_or_else(|| invalid("actual R checkpoint lacks its explicit window recipe"))?;
    check(
        recipe == current.recipe()
            && !current.is_pending()
            && current.dataset().joint_current_pc_publication_sha256()?
                == source.dataset.registration.publication.sha256,
        "actual R checkpoint window/ledger differs or accumulation is pending",
    )?;
    let dataset = Arc::clone(current.dataset());
    let mut auxiliary = match (
        &source.registration.initialization_source,
        source.checkpoint,
    ) {
        (RoventTrainingInputSource::Initialize { .. }, None) => RoventAuxiliaryLedger::new(
            dataset,
            recipe.clone(),
            BTreeSet::new(),
            BTreeSet::new(),
            0,
        )?,
        (RoventTrainingInputSource::Resume { .. }, Some(parent)) => {
            let mut prefix = parent
                .r_auxiliary_ledger()
                .ok_or_else(|| invalid("actual resumed R checkpoint lost its historical prefix"))?
                .clone();
            prefix.bind_current_dataset(dataset, recipe)?;
            prefix
        }
        (RoventTrainingInputSource::Fork { .. }, Some(parent)) => {
            if let Some(parent) = parent.r_auxiliary_ledger() {
                parent.fork(dataset, recipe.clone())?
            } else {
                RoventAuxiliaryLedger::from_non_auxiliary_checkpoint(
                    dataset,
                    recipe.clone(),
                    parent,
                )?
            }
        }
        _ => {
            return Err(invalid(
                "actual R checkpoint initialization/history mismatch",
            ));
        }
    };
    auxiliary.set_execution_phase(
        &spec.phase,
        if matches!(
            source.registration.initialization_source,
            RoventTrainingInputSource::Resume { .. }
        ) {
            source.checkpoint.map(Arc::as_ref)
        } else {
            None
        },
    )?;
    let mut replay = TrainerLedger::new(
        source.journal.window.clone(),
        source.journal.recipe.clone(),
        source.journal.membership_sha256.clone(),
        source.journal.corpus_sha256.clone(),
        source.journal.rows.clone(),
        source.journal.initial.clone(),
        source.journal.ancestors.clone(),
    )?;
    replay.profile_sha256 = source.journal.profile_sha256.clone();
    replay.begun = true;
    replay.consumed_inputs.extend(
        source
            .dataset
            .manifest
            .inherited_input_sha256
            .iter()
            .cloned(),
    );
    replay.consumed_origins.extend(
        source
            .dataset
            .manifest
            .inherited_origin_sha256
            .iter()
            .cloned(),
    );
    if let Some(parent) = source.checkpoint {
        replay
            .consumed_inputs
            .extend(parent.retained_inputs().iter().cloned());
        replay
            .consumed_origins
            .extend(parent.retained_origins().iter().cloned());
    }
    auxiliary.validate_joint_retained(&replay)?;
    let mut prefix = (&replay.completed == target).then(|| auxiliary.clone());
    for update in &source.journal.updates {
        check(
            Instant::now() < deadline && !cancel.load(Ordering::Acquire),
            "actual R checkpoint prefix replay cancelled/expired",
        )?;
        checkpoint::replay_update_body(
            &mut replay,
            update,
            "rz-rovent-trainer-consume/2",
            "rz-rovent-trainer-update/2",
            "rz-rovent-recovered-learner-update/1",
            Some(&mut auxiliary),
        )?;
        if prefix.is_none() && &replay.completed == target {
            auxiliary.validate_joint_retained(&replay)?;
            prefix = Some(auxiliary.clone());
        }
    }
    check(
        auxiliary.observation() == current.observation(),
        "actual full R update-chain replay differs from its original current ledger",
    )?;
    prefix.ok_or_else(|| invalid("selected R checkpoint prefix is absent from actual updates"))
}
fn prefix_sets(
    mut completed: BTreeMap<String, u64>,
    updates: &[Value],
    mut inputs: BTreeSet<String>,
    mut origins: BTreeSet<String>,
    target: &BTreeMap<String, u64>,
) -> Result<(BTreeSet<String>, BTreeSet<String>), ArenaError> {
    check(
        completed.len() == 3
            && target.len() == 3
            && ROLES
                .iter()
                .all(|r| completed.contains_key(*r) && target.contains_key(*r)),
        "exact actual prefix role maps",
    )?;
    let mut reached = &completed == target;
    for update in updates {
        if reached {
            break;
        }
        let role = update["event"]["role"]
            .as_str()
            .filter(|r| ROLES.contains(r))
            .ok_or_else(|| invalid("actual prefix update role"))?;
        let next = completed[role]
            .checked_add(1)
            .ok_or_else(|| invalid("actual prefix count overflow"))?;
        check(
            update["event"]["completed_role_updates"] == next,
            "actual prefix counter does not follow original journal",
        )?;
        completed.insert(role.into(), next);
        for (key, set) in [
            ("sample_inputs", &mut inputs),
            ("sample_origins", &mut origins),
        ] {
            let values = update["event"][key]
                .as_array()
                .filter(|v| v.len() == 8)
                .ok_or_else(|| invalid("actual prefix sample group"))?;
            for value in values {
                set.insert(sha(value)?.into());
            }
        }
        reached = &completed == target;
    }
    check(
        reached,
        "selected checkpoint counts are absent from actual committed journal",
    )?;
    Ok((inputs, origins))
}
fn actual_tail_delta(
    snapshot: &BTreeMap<String, u64>,
    end: &Value,
) -> Result<BTreeMap<String, u64>, ArenaError> {
    check(
        snapshot.len() == 8 && COST_FIELDS.iter().all(|k| snapshot.contains_key(*k)),
        "exact snapshot cost fields",
    )?;
    validate_cost(end)?;
    let mut delta = BTreeMap::new();
    for key in COST_FIELDS {
        delta.insert(
            key.into(),
            count(&end[key])?
                .checked_sub(snapshot[key])
                .ok_or_else(|| invalid("actual post-snapshot cost regressed"))?,
        );
    }
    Ok(delta)
}

#[cfg(test)]
#[path = "checkpoint_publication/tests.rs"]
mod tests;
