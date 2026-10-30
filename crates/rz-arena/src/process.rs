//! Bounded native runner supervision on Linux.
//!
//! This owns a fresh process group, not a security sandbox. The child limit is
//! a `/proc` snapshot of that group: transient forks and children escaping with
//! a new session/group require a future cgroup adapter. A leader is not reaped
//! while group cleanup is in progress, so its reserved PID cannot name another
//! group. The tree API retains the original child after unverified cleanup;
//! a PID reservation is proven only while reaping ownership is intact. The
//! older flat APIs retain their original terminal-child reaping behavior.
//! This requires unchanged default SIGCHLD handling and no foreign child reaper.

use crate::ArenaError;
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::fs::File;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};
type StdoutObserver<'a> = &'a mut dyn FnMut(&[u8]);
type LoadedImageVerifier<'a> = &'a mut dyn FnMut(&File) -> Result<(), ()>;

pub const MAX_ORIGINAL_INPUT_BYTES: usize = 2 * 1024 * 1024;

/// A process-local original clock. No serialized duration reconstructs this
/// ownership, and creating the declaration grants no execution/closure proof.
#[derive(Clone, Copy, Debug)]
pub struct OriginalProcessWindow {
    started: Instant,
    execution: Instant,
    whole: Instant,
}
impl OriginalProcessWindow {
    pub fn new(started: Instant, execution: Instant, whole: Instant) -> Result<Self, ArenaError> {
        if started >= execution || execution >= whole || started > Instant::now() {
            return Err(ArenaError::Budget(
                "original process window must be ordered and process-local".into(),
            ));
        }
        Ok(Self {
            started,
            execution,
            whole,
        })
    }
    fn check_before_spawn(self, limits: ProcessLimits) -> Result<(), ArenaError> {
        limits.validate()?;
        let cleanup = Duration::from_millis(limits.shutdown_grace_ms)
            .checked_mul(2)
            .ok_or_else(|| ArenaError::Budget("original cleanup policy overflow".into()))?;
        if self.execution.duration_since(self.started) > Duration::from_millis(limits.wall_ms)
            || self.whole.duration_since(self.execution) > cleanup
        {
            return Err(ArenaError::Budget(
                "original window exceeds declared process limits".into(),
            ));
        }
        if Instant::now() >= self.execution {
            return Err(ArenaError::Budget(
                "original execution window expired before spawn".into(),
            ));
        }
        Ok(())
    }
    #[cfg(target_os = "linux")]
    fn elapsed_ns(self, at: Instant) -> Option<u64> {
        u64::try_from(at.checked_duration_since(self.started)?.as_nanos()).ok()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OriginalProcessInput<'a> {
    pub bytes: &'a [u8],
    pub window: OriginalProcessWindow,
}

/// 수치 캡처의 고정 FP32 조건만 전달한다. 임의 환경이나 부모의 CUDA/
/// 인증/검색 경로를 상속하는 접점이 아니며 기존 실행의 환경은 보존한다.
#[derive(Clone, Copy, Debug)]
pub(crate) enum OriginalNumericalEnvironment {
    // Keep the closed policy constructible at the platform refusal boundary.
    // Its actual original-process consumer is Linux-only.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    Fp32Tf32Off,
}

/// Observed transport only. This is not model/source admission, physical native
/// completion, a cgroup resource proof or a checked next Query capability.
#[derive(Debug, Serialize)]
pub struct OriginalProcessObservation {
    schema: &'static str,
    input_bytes: usize,
    written_bytes: Option<usize>,
    stdin_closed_after_input: bool,
    stdout_eof: bool,
    stderr_eof: bool,
    original_supervisor_entry_ns: Option<u64>,
    original_spawn_ns: Option<u64>,
    original_exit_observed_ns: Option<u64>,
    exit_observed_before_execution: bool,
    original_finished_ns: Option<u64>,
    finished_before_whole: bool,
    cancelled_at_return: bool,
    ownership_lost: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    numerical_environment: Option<&'static str>,
    // Process-local observations are not added to the legacy transport/1 JSON.
    #[serde(skip)]
    loaded_image: Option<LoadedImageObservation>,
    #[serde(skip)]
    first_input_written_ns: Option<u64>,
    #[serde(skip)]
    loaded_image_progress: LoadedImageProgress,
}
#[derive(Debug, Default)]
struct LoadedImageProgress {
    attempts: u64,
    pending: u64,
    first_pending_code: Option<&'static str>,
    last_pending_code: Option<&'static str>,
}
impl LoadedImageProgress {
    #[cfg(target_os = "linux")]
    fn begin_attempt(&mut self) -> Result<(), &'static str> {
        self.attempts = self
            .attempts
            .checked_add(1)
            .ok_or("process.loaded_image_attempt_overflow")?;
        Ok(())
    }
    #[cfg(target_os = "linux")]
    fn record_pending(&mut self, code: &'static str) {
        // At most one pending observation per counted attempt.
        self.pending += 1;
        self.first_pending_code.get_or_insert(code);
        self.last_pending_code = Some(code);
    }
}
#[derive(Debug)]
struct LoadedImageObservation {
    pid: u32,
    device: u64,
    inode: u64,
    observed_ns: u64,
}
impl OriginalProcessObservation {
    #[cfg(target_os = "linux")]
    fn new(input_bytes: usize) -> Self {
        Self {
            schema: "rz-original-process-transport/1",
            input_bytes,
            written_bytes: None,
            stdin_closed_after_input: false,
            stdout_eof: false,
            stderr_eof: false,
            original_supervisor_entry_ns: None,
            original_spawn_ns: None,
            original_exit_observed_ns: None,
            exit_observed_before_execution: false,
            original_finished_ns: None,
            finished_before_whole: false,
            cancelled_at_return: false,
            ownership_lost: false,
            numerical_environment: None,
            loaded_image: None,
            first_input_written_ns: None,
            loaded_image_progress: LoadedImageProgress::default(),
        }
    }
    pub fn written_bytes(&self) -> Option<usize> {
        self.written_bytes
    }
    pub fn numerical_environment(&self) -> Option<&'static str> {
        self.numerical_environment
    }
    pub fn stdout_eof(&self) -> bool {
        self.stdout_eof
    }
    pub fn stderr_eof(&self) -> bool {
        self.stderr_eof
    }
    pub fn finished_before_whole(&self) -> bool {
        self.finished_before_whole
    }
    pub fn exit_observed_before_execution(&self) -> bool {
        self.exit_observed_before_execution
    }
    pub fn original_exit_observed_ns(&self) -> Option<u64> {
        self.original_exit_observed_ns
    }
    pub fn loaded_image_attempts(&self) -> u64 {
        self.loaded_image_progress.attempts
    }
    pub fn loaded_image_pending_observations(&self) -> u64 {
        self.loaded_image_progress.pending
    }
    pub fn loaded_image_first_pending_code(&self) -> Option<&'static str> {
        self.loaded_image_progress.first_pending_code
    }
    pub fn loaded_image_last_pending_code(&self) -> Option<&'static str> {
        self.loaded_image_progress.last_pending_code
    }
}

#[derive(Debug)]
pub struct OriginalProcessOutput {
    process: ProcessOutput,
    observation: OriginalProcessObservation,
    window: OriginalProcessWindow,
}

/// Actual supervisor timestamps borrowed from its private output owner. These
/// describe transport, not model execution, loaded-image admission or Query order.
/// The launch timestamp precedes Command::spawn; it is not a child-ready event.
#[derive(Debug)]
pub struct OriginalProcessTiming<'a> {
    output: &'a OriginalProcessOutput,
    stamps: [u64; 4],
}
impl OriginalProcessTiming<'_> {
    pub fn original_started(&self) -> Instant {
        self.output.window.started
    }
    pub fn execution_deadline(&self) -> Instant {
        self.output.window.execution
    }
    pub fn whole_deadline(&self) -> Instant {
        self.output.window.whole
    }
    pub fn supervisor_entry_ns(&self) -> u64 {
        self.stamps[0]
    }
    pub fn launch_started_ns(&self) -> u64 {
        self.stamps[1]
    }
    pub fn exit_observed_ns(&self) -> u64 {
        self.stamps[2]
    }
    pub fn supervisor_finished_ns(&self) -> u64 {
        self.stamps[3]
    }
}

/// Parent observation of the actual child's executable before original stdin.
/// This checks file identity, not executable content, dependencies, native NN
/// completion, cgroup enforcement or next-Query admission. No serialized report
/// can construct this borrow from a PID or an alleged loaded-image timestamp.
#[derive(Debug)]
pub struct OriginalLoadedImage<'a> {
    output: &'a OriginalProcessOutput,
    image: &'a LoadedImageObservation,
    first_input_written_ns: u64,
}
impl OriginalLoadedImage<'_> {
    pub fn process_id(&self) -> u32 {
        self.output.process.receipt.pid
    }
    pub fn file_device(&self) -> u64 {
        self.image.device
    }
    pub fn file_inode(&self) -> u64 {
        self.image.inode
    }
    pub fn observed_ns(&self) -> u64 {
        self.image.observed_ns
    }
    pub fn first_input_written_ns(&self) -> u64 {
        self.first_input_written_ns
    }
    pub fn observation(&self) -> &OriginalProcessObservation {
        self.output.observation()
    }
    pub fn assurance_scope(&self) -> &'static str {
        "actual_parent_same_inode_before_original_stdin_pending_content_and_native_admission"
    }
}
impl OriginalProcessOutput {
    pub fn process(&self) -> &ProcessOutput {
        &self.process
    }
    pub fn observation(&self) -> &OriginalProcessObservation {
        &self.observation
    }
    /// Requires real pipe/group/reap closure and ordered timestamps on the exact
    /// original process clock. A JSON observation cannot construct this borrow.
    pub fn checked_timing(&self) -> Result<OriginalProcessTiming<'_>, ArenaError> {
        if !self.transport_complete() {
            return Err(ArenaError::Invalid(
                "original process closure/timing unavailable".into(),
            ));
        }
        let stamps = self
            .ordered_stamps()
            .ok_or_else(|| ArenaError::Invalid("original process timing order".into()))?;
        Ok(OriginalProcessTiming {
            output: self,
            stamps,
        })
    }
    /// Adds a real parent-loaded-image observation to complete transport. It is
    /// deliberately separate from the existing transport/1 closure contract.
    pub fn checked_loaded_image(&self) -> Result<OriginalLoadedImage<'_>, ArenaError> {
        let timing = self.checked_timing()?;
        let image = self
            .observation
            .loaded_image
            .as_ref()
            .ok_or_else(|| ArenaError::Invalid("original loaded image unobserved".into()))?;
        let first_input_written_ns = self
            .observation
            .first_input_written_ns
            .ok_or_else(|| ArenaError::Invalid("original first input write unobserved".into()))?;
        if image.pid != self.process.receipt.pid
            || image.observed_ns < timing.launch_started_ns()
            || image.observed_ns > first_input_written_ns
            || first_input_written_ns > timing.exit_observed_ns()
        {
            return Err(ArenaError::Invalid(
                "original loaded image owner/time order".into(),
            ));
        }
        Ok(OriginalLoadedImage {
            output: self,
            image,
            first_input_written_ns,
        })
    }
    fn ordered_stamps(&self) -> Option<[u64; 4]> {
        let o = &self.observation;
        let stamps = [
            o.original_supervisor_entry_ns?,
            o.original_spawn_ns?,
            o.original_exit_observed_ns?,
            o.original_finished_ns?,
        ];
        let execution = u64::try_from(
            self.window
                .execution
                .checked_duration_since(self.window.started)?
                .as_nanos(),
        )
        .ok()?;
        let whole = u64::try_from(
            self.window
                .whole
                .checked_duration_since(self.window.started)?
                .as_nanos(),
        )
        .ok()?;
        (stamps.windows(2).all(|pair| pair[0] <= pair[1])
            && stamps[2] < execution
            && stamps[3] < whole)
            .then_some(stamps)
    }
    pub fn transport_complete(&self) -> bool {
        let o = &self.observation;
        let r = &self.process.receipt;
        o.written_bytes == Some(o.input_bytes)
            && o.stdin_closed_after_input
            && o.stdout_eof
            && o.stderr_eof
            && o.finished_before_whole
            && o.exit_observed_before_execution
            && o.original_supervisor_entry_ns.is_some()
            && o.original_spawn_ns.is_some()
            && o.original_exit_observed_ns.is_some()
            && o.original_finished_ns.is_some()
            && self.ordered_stamps().is_some()
            && !o.cancelled_at_return
            && !o.ownership_lost
            && r.stop == ProcessStop::Exited
            && r.exit_code == Some(0)
            && r.exit_signal.is_none()
            && r.group_cleanup == CleanupStatus::Gone
            && r.errors.is_empty()
            && self.process.pending_child.is_none()
    }
    /// Consume the observation capability to transfer pending-child custody.
    pub fn into_process(self) -> ProcessOutput {
        self.process
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ProcessLimits {
    pub wall_ms: u64,
    pub shutdown_grace_ms: u64,
    pub max_output_bytes: u64,
    /// Includes the runner leader and its engines, within the owned group.
    pub max_child_processes: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessStop {
    Exited,
    WallLimit,
    OutputLimit,
    Cancelled,
    IoFailure,
    ChildLimit,
    ArtifactLimit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CleanupStatus {
    Gone,
    Unverified,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessReceipt {
    pub supervisor_version: u32,
    pub pid: u32,
    pub elapsed_ns: u64,
    /// Observed group cleanup begins after terminal/stop detection and includes
    /// the final group check and owned leader reap. It excludes engine play.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cleanup_elapsed_ns: Option<u64>,
    /// Missing historical fields deserialize as unknown; a current run records
    /// a stable reason whenever cleanup completion/timing could not be observed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cleanup_timing_unavailable_reason: Option<String>,
    pub stop: ProcessStop,
    pub exit_code: Option<i32>,
    pub exit_signal: Option<i32>,
    pub group_cleanup: CleanupStatus,
    /// Bytes retained in the output artifact, never above the combined limit.
    pub stdout_bytes: u64,
    pub stderr_bytes: u64,
    pub observed_output_bytes: u64,
    pub descendant_cleanup_required: bool,
    /// Deliberately distinguishes observed enforcement from a kernel hard cap.
    pub child_limit_enforcement: String,
    /// Peak total length observed in watched regular output files. A tree
    /// observation that stops at a limit may describe only the visited prefix.
    pub watched_artifact_bytes: Option<u64>,
    pub artifact_limit_enforcement: Option<String>,
    /// Stable bounded step/code evidence; no arguments, paths or child payload.
    pub errors: Vec<String>,
}

#[derive(Debug)]
pub struct ProcessOutput {
    pub receipt: ProcessReceipt,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    /// A leader retained after the bounded cleanup interval. For the tree API,
    /// this includes an observed terminal leader when cleanup is Unverified.
    /// With intact ownership, not reaping reserves its PID/process-group ID.
    /// After ownership_lost, this is only the original handle for quarantine;
    /// it proves no reservation and grants no numeric PID wait/signal authority.
    /// A typed external owner must resolve physical cleanup and authority before
    /// reaping or dropping it. No subsequent kill/reap/drop is automatic.
    /// Flat APIs retain their older pending-only behavior and return no handle
    /// after ownership was lost to an unsupported external reaper.
    pub pending_child: Option<std::process::Child>,
}

#[derive(Debug, Clone)]
pub struct ArtifactWatch {
    /// Supported files are direct children of the pinned working directory.
    pub relative_files: Vec<PathBuf>,
    pub max_total_bytes: u64,
}

/// Periodic observation of an exclusively owned output directory tree.
/// This measures file lengths, not allocated blocks or a kernel disk quota.
/// Runtime-created subdirectories are included; symlinks, regular files with
/// multiple hard links and all non-directory/non-regular kinds are rejected.
#[derive(Debug, Clone, Copy)]
pub struct OwnedArtifactTreeWatch {
    pub max_total_bytes: u64,
    pub max_file_bytes: u64,
    /// Maximum regular files in a generic watch; an admitted delegated domain
    /// also counts each explicitly verified alias as a nondirectory entry.
    /// Directories have a separate equal count limit,
    /// and at most twice this many entries are examined per snapshot, including
    /// entries that disappear before their inode can be opened.
    pub max_files: usize,
    /// Root depth is zero; a direct child directory has depth one.
    pub max_depth: usize,
}

impl OwnedArtifactTreeWatch {
    const MAX_FILES: usize = 4096;
    const MAX_DEPTH: usize = 32;

    /// Use the supervisor's finite traversal bounds with the caller's original
    /// byte budgets. Admission still validates positive byte limits before spawn.
    #[cfg(target_os = "linux")]
    pub(crate) const fn bounded_output(max_total_bytes: u64, max_file_bytes: u64) -> Self {
        Self {
            max_total_bytes,
            max_file_bytes,
            max_files: Self::MAX_FILES,
            max_depth: Self::MAX_DEPTH,
        }
    }

    fn validate(&self) -> Result<(), ArenaError> {
        self.validate_with_file_ceiling(Self::MAX_FILES)
    }
    fn validate_with_file_ceiling(&self, file_ceiling: usize) -> Result<(), ArenaError> {
        if self.max_total_bytes == 0
            || self.max_file_bytes == 0
            || !(1..=32_768).contains(&file_ceiling)
            || !(1..=file_ceiling).contains(&self.max_files)
            || self.max_depth > Self::MAX_DEPTH
        {
            return Err(ArenaError::Budget(format!(
                "artifact tree requires positive bytes, 1..{file_ceiling} files and depth at most 32"
            )));
        }
        Ok(())
    }
}

/// Explicit finite Rovent training profile. The generic/public watch keeps its
/// original 4096-file validation. Only the crate's new trainer entrypoint may
/// consume this private wrapper with a separately registered ceiling.
#[cfg(all(target_os = "linux", any(feature = "pals-collection-onnx", test)))]
pub(crate) struct RoventTrainingArtifactTreeWatch(OwnedArtifactTreeWatch);
#[cfg(all(target_os = "linux", any(feature = "pals-collection-onnx", test)))]
impl RoventTrainingArtifactTreeWatch {
    pub(crate) fn registered(
        max_total_bytes: u64,
        max_file_bytes: u64,
        max_files: usize,
    ) -> Result<Self, ArenaError> {
        let watch = OwnedArtifactTreeWatch {
            max_total_bytes,
            max_file_bytes,
            max_files,
            max_depth: OwnedArtifactTreeWatch::MAX_DEPTH,
        };
        watch.validate_with_file_ceiling(max_files)?;
        Ok(Self(watch))
    }
}

/// A crate-owned, already admitted immutable subtree. Generic watches never
/// construct this capability or admit links. Implementations must bind both the
/// retained directory inode and its original named parent/entry on every match
/// check, including checks against a different directory.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) trait DelegatedArtifactTreeWatch {
    fn matches_directory(&self, directory: &File) -> Result<bool, &'static str>;
    fn observe_directory(&self, directory: &File) -> Result<DelegatedTreeSnapshot, &'static str>;
}

/// Descendants only: the caller already counted the delegated root directory.
/// `files` includes explicitly verified aliases; `bytes` sums their nofollow
/// sizes and regular sizes, never a second copy of an alias target payload.
/// `depth` is the maximum descendant directory depth, with this root at zero.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DelegatedTreeSnapshot {
    pub(crate) bytes: u64,
    pub(crate) files: usize,
    pub(crate) directories: usize,
    pub(crate) entries: usize,
    pub(crate) max_file_bytes: u64,
    pub(crate) depth: usize,
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
impl DelegatedTreeSnapshot {
    /// Internal units and count coherence, before each caller applies its own
    /// original aggregate caps. This does not establish admission or ownership.
    pub(crate) fn validate(&self) -> Result<(), &'static str> {
        let counted = self
            .files
            .checked_add(self.directories)
            .ok_or("process.artifact_tree.delegated_snapshot")?;
        let upper_bytes = u64::try_from(self.files)
            .ok()
            .and_then(|files| files.checked_mul(self.max_file_bytes));
        if counted > self.entries
            || self.depth > self.directories
            || (self.directories == 0 && self.depth != 0)
            || (self.directories != 0 && self.depth == 0)
            || self.bytes > upper_bytes.ok_or("process.artifact_tree.delegated_snapshot")?
            || (self.files == 0 && self.max_file_bytes != 0)
        {
            return Err("process.artifact_tree.delegated_snapshot");
        }
        Ok(())
    }
}

impl ArtifactWatch {
    fn validate(&self) -> Result<(), ArenaError> {
        if self.max_total_bytes == 0
            || self.relative_files.is_empty()
            || self.relative_files.len() > 16
        {
            return Err(ArenaError::Budget(
                "artifact watch requires 1..16 files and positive bytes".into(),
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        for path in &self.relative_files {
            let mut parts = path.components();
            if !matches!(parts.next(), Some(Component::Normal(_))) || parts.next().is_some() {
                return Err(ArenaError::Invalid(
                    "artifact watch requires direct relative filenames".into(),
                ));
            }
            let Some(name) = path.to_str() else {
                return Err(ArenaError::Invalid(
                    "artifact watch filenames require UTF-8".into(),
                ));
            };
            if name.starts_with('.')
                || name.len() > 255
                || !name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
                || !seen.insert(name)
            {
                return Err(ArenaError::Invalid(
                    "artifact watch filename is unsafe or duplicated".into(),
                ));
            }
        }
        Ok(())
    }
}

impl ProcessLimits {
    fn validate(self) -> Result<(), ArenaError> {
        if self.wall_ms == 0
            || self.shutdown_grace_ms == 0
            || self.max_output_bytes == 0
            || self.max_child_processes == 0
        {
            return Err(ArenaError::Budget("process limits must be positive".into()));
        }
        let total_ms = self
            .shutdown_grace_ms
            .checked_mul(2)
            .and_then(|grace| self.wall_ms.checked_add(grace))
            .ok_or_else(|| ArenaError::Budget("process duration overflow".into()))?;
        if total_ms.checked_mul(1_000_000).is_none()
            || usize::try_from(self.max_output_bytes).is_err()
            || self.max_child_processes > i32::MAX as u32
        {
            return Err(ArenaError::Budget(
                "process limits exceed representable bounds".into(),
            ));
        }
        Ok(())
    }
}

/// Execute exactly the already-opened native ELF artifact. Linux resolves the
/// open descriptor at exec; replacing its former path cannot swap the program.
/// Scripts are rejected because their interpreter may reopen a closed fd.
pub fn supervise(
    program: &File,
    args: &[OsString],
    cwd: &Path,
    limits: ProcessLimits,
    cancel: Option<&AtomicBool>,
) -> Result<ProcessOutput, ArenaError> {
    limits.validate()?;
    #[cfg(target_os = "linux")]
    {
        linux::supervise(
            program,
            args,
            linux::pin_directory(cwd)?,
            limits,
            cancel,
            None,
            None,
            linux::InvocationIo::default(),
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (program, args, cwd, cancel);
        Err(ArenaError::Invalid(
            "native process supervision currently requires Linux".into(),
        ))
    }
}

/// Additionally stop when watched regular artifacts exceed their combined
/// byte cap. Observation is periodic, not a kernel disk quota. Missing files
/// are allowed until created; symlinks/FIFOs/devices cause an explicit failure.
pub fn supervise_with_watch(
    program: &File,
    args: &[OsString],
    cwd: &Path,
    limits: ProcessLimits,
    cancel: Option<&AtomicBool>,
    watch: &ArtifactWatch,
) -> Result<ProcessOutput, ArenaError> {
    limits.validate()?;
    watch.validate()?;
    #[cfg(target_os = "linux")]
    {
        linux::supervise(
            program,
            args,
            linux::pin_directory(cwd)?,
            limits,
            cancel,
            Some(linux::ArtifactObservation::Flat(watch)),
            None,
            linux::InvocationIo::default(),
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (program, args, cwd, cancel);
        Err(ArenaError::Invalid(
            "native process supervision currently requires Linux".into(),
        ))
    }
}

/// Use an already-pinned working directory capability without reopening its
/// former path. The cloned handle is shared by launch and artifact observation.
pub fn supervise_in_directory(
    program: &File,
    args: &[OsString],
    cwd: &File,
    limits: ProcessLimits,
    cancel: Option<&AtomicBool>,
    watch: Option<&ArtifactWatch>,
) -> Result<ProcessOutput, ArenaError> {
    limits.validate()?;
    if let Some(watch) = watch {
        watch.validate()?;
    }
    #[cfg(target_os = "linux")]
    {
        if !cwd
            .metadata()
            .map_err(|_| ArenaError::Io("process.cwd_metadata".into()))?
            .is_dir()
        {
            return Err(ArenaError::Invalid(
                "runner cwd handle must be a directory".into(),
            ));
        }
        let directory = cwd
            .try_clone()
            .map_err(|_| ArenaError::Io("process.cwd_clone".into()))?;
        linux::supervise(
            program,
            args,
            directory,
            limits,
            cancel,
            watch.map(linux::ArtifactObservation::Flat),
            None,
            linux::InvocationIo::default(),
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (program, args, cwd, cancel);
        Err(ArenaError::Invalid(
            "native process supervision currently requires Linux".into(),
        ))
    }
}

/// Supervise using a cloned directory capability and bounded recursive tree
/// snapshots. Each inode is opened relative to a pinned directory without
/// following symlinks; regular payloads are never opened for reading. This is
/// observation, not a hard disk quota or an atomic view of a changing tree.
/// When cleanup remains Unverified and leader ownership is intact, even a
/// terminal leader is returned unreaped in pending_child. The caller must keep
/// that reservation until its typed owner confirms physical cleanup; polling
/// Child::try_wait also reaps and therefore releases the reservation.
/// If reaping ownership is lost, the original Child is still returned for
/// quarantine, with ownership_lost evidence: no reservation or further numeric
/// PID wait/signal authority is established by preserving that handle.
pub fn supervise_in_directory_with_tree(
    program: &File,
    args: &[OsString],
    directory: &File,
    limits: ProcessLimits,
    cancel: Option<&AtomicBool>,
    watch: &OwnedArtifactTreeWatch,
) -> Result<ProcessOutput, ArenaError> {
    supervise_tree_observed(program, args, directory, limits, cancel, watch, None)
}

/// Passive finite stream observer. It cannot alter output, cancellation or gates.
pub(crate) fn supervise_tree_observed(
    program: &File,
    args: &[OsString],
    directory: &File,
    limits: ProcessLimits,
    cancel: Option<&AtomicBool>,
    watch: &OwnedArtifactTreeWatch,
    stdout_observer: Option<StdoutObserver<'_>>,
) -> Result<ProcessOutput, ArenaError> {
    limits.validate()?;
    watch.validate()?;
    #[cfg(target_os = "linux")]
    {
        if !directory
            .metadata()
            .map_err(|_| ArenaError::Io("process.cwd_metadata".into()))?
            .is_dir()
        {
            return Err(ArenaError::Invalid(
                "runner cwd handle must be a directory".into(),
            ));
        }
        let directory = directory
            .try_clone()
            .map_err(|_| ArenaError::Io("process.cwd_clone".into()))?;
        linux::supervise(
            program,
            args,
            directory,
            limits,
            cancel,
            Some(linux::ArtifactObservation::Tree(watch)),
            stdout_observer,
            linux::InvocationIo::default(),
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (program, args, directory, cancel, stdout_observer);
        Err(ArenaError::Invalid(
            "native process supervision currently requires Linux".into(),
        ))
    }
}

/// Finite UCI preflight input through the same pinned ELF/process-group owner.
/// Commands are written nonblocking; the wall/output/tree bounds remain active.
pub fn supervise_protocol_in_directory(
    program: &File,
    args: &[OsString],
    directory: &File,
    limits: ProcessLimits,
    cancel: Option<&AtomicBool>,
    watch: &OwnedArtifactTreeWatch,
    commands: &[u8],
) -> Result<ProcessOutput, ArenaError> {
    limits.validate()?;
    watch.validate()?;
    if commands.is_empty() || commands.len() > 64 * 1024 || !commands.ends_with(b"\n") {
        return Err(ArenaError::Budget(
            "finite protocol input requires 1..64KiB and a final newline".into(),
        ));
    }
    #[cfg(target_os = "linux")]
    {
        if !directory
            .metadata()
            .map_err(|_| ArenaError::Io("process.cwd_metadata".into()))?
            .is_dir()
        {
            return Err(ArenaError::Invalid(
                "process working descriptor is not a directory".into(),
            ));
        }
        let directory = directory
            .try_clone()
            .map_err(|_| ArenaError::Io("process.cwd_clone".into()))?;
        linux::supervise(
            program,
            args,
            directory,
            limits,
            cancel,
            Some(linux::ArtifactObservation::Tree(watch)),
            None,
            linux::InvocationIo {
                commands: Some(commands),
                ..Default::default()
            },
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (program, args, directory, cancel, commands);
        Err(ArenaError::Invalid(
            "native UCI preflight requires Linux".into(),
        ))
    }
}

/// Replay-sized bytes through the same nonblocking pipe/group owner, under an
/// original absolute execution and whole deadline. Existing protocol limits are
/// unchanged. Blocking OS preflight/spawn/observation is not a hard-timeout proof;
/// late completion is recorded and cannot pass transport_complete().
pub fn supervise_input_in_original_window(
    program: &File,
    args: &[OsString],
    directory: &File,
    limits: ProcessLimits,
    cancel: Option<&AtomicBool>,
    watch: &OwnedArtifactTreeWatch,
    input: OriginalProcessInput<'_>,
) -> Result<OriginalProcessOutput, ArenaError> {
    supervise_original_input(
        program, args, directory, limits, cancel, watch, input, None, None, None, None,
    )
}

/// Only internal callers may add content checks on the actual proc executable.
/// Failure is captured after spawn and follows the existing child custody path.
/// A callback returning Ok is not exposed as a generic content-admission token.
#[cfg(all(target_os = "linux", any(feature = "pals-collection-onnx", test)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn supervise_verified_input_in_original_window(
    program: &File,
    args: &[OsString],
    directory: &File,
    limits: ProcessLimits,
    cancel: Option<&AtomicBool>,
    watch: &OwnedArtifactTreeWatch,
    input: OriginalProcessInput<'_>,
    verify_loaded: LoadedImageVerifier<'_>,
) -> Result<OriginalProcessOutput, ArenaError> {
    supervise_original_input(
        program,
        args,
        directory,
        limits,
        cancel,
        watch,
        input,
        Some(verify_loaded),
        None,
        None,
        None,
    )
}

#[cfg(all(target_os = "linux", any(feature = "pals-collection-onnx", test)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn supervise_numerical_input_in_original_window(
    program: &File,
    args: &[OsString],
    directory: &File,
    limits: ProcessLimits,
    cancel: Option<&AtomicBool>,
    watch: &OwnedArtifactTreeWatch,
    input: OriginalProcessInput<'_>,
    verify_loaded: LoadedImageVerifier<'_>,
    environment: OriginalNumericalEnvironment,
) -> Result<OriginalProcessOutput, ArenaError> {
    supervise_original_input(
        program,
        args,
        directory,
        limits,
        cancel,
        watch,
        input,
        Some(verify_loaded),
        Some(environment),
        None,
        None,
    )
}

/// New trainer-only file-count profile. All execution, byte, deadline, link
/// and owned-tree guards use the same supervisor as the generic 4096 profile.
#[cfg(all(target_os = "linux", any(feature = "pals-collection-onnx", test)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn supervise_rovent_training_numerical_input_in_original_window(
    program: &File,
    args: &[OsString],
    directory: &File,
    limits: ProcessLimits,
    cancel: Option<&AtomicBool>,
    watch: &RoventTrainingArtifactTreeWatch,
    input: OriginalProcessInput<'_>,
    verify_loaded: LoadedImageVerifier<'_>,
    environment: OriginalNumericalEnvironment,
) -> Result<OriginalProcessOutput, ArenaError> {
    supervise_original_input(
        program,
        args,
        directory,
        limits,
        cancel,
        &watch.0,
        input,
        Some(verify_loaded),
        Some(environment),
        None,
        Some(watch.0.max_files),
    )
}

/// The same original process owner with one crate-admitted immutable subtree.
/// Ordinary callers have no link permission and keep the original entrypoint.
#[cfg(all(target_os = "linux", any(feature = "pals-collection-onnx", test)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn supervise_verified_input_with_delegated_tree_in_original_window(
    program: &File,
    args: &[OsString],
    directory: &File,
    limits: ProcessLimits,
    cancel: Option<&AtomicBool>,
    watch: &OwnedArtifactTreeWatch,
    input: OriginalProcessInput<'_>,
    verify_loaded: LoadedImageVerifier<'_>,
    delegated: &dyn DelegatedArtifactTreeWatch,
) -> Result<OriginalProcessOutput, ArenaError> {
    supervise_original_input(
        program,
        args,
        directory,
        limits,
        cancel,
        watch,
        input,
        Some(verify_loaded),
        None,
        Some(delegated),
        None,
    )
}

#[cfg(all(target_os = "linux", any(feature = "pals-collection-onnx", test)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn supervise_numerical_input_with_delegated_tree_in_original_window(
    program: &File,
    args: &[OsString],
    directory: &File,
    limits: ProcessLimits,
    cancel: Option<&AtomicBool>,
    watch: &OwnedArtifactTreeWatch,
    input: OriginalProcessInput<'_>,
    verify_loaded: LoadedImageVerifier<'_>,
    environment: OriginalNumericalEnvironment,
    delegated: &dyn DelegatedArtifactTreeWatch,
) -> Result<OriginalProcessOutput, ArenaError> {
    supervise_original_input(
        program,
        args,
        directory,
        limits,
        cancel,
        watch,
        input,
        Some(verify_loaded),
        Some(environment),
        Some(delegated),
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn supervise_original_input(
    program: &File,
    args: &[OsString],
    directory: &File,
    limits: ProcessLimits,
    cancel: Option<&AtomicBool>,
    watch: &OwnedArtifactTreeWatch,
    input: OriginalProcessInput<'_>,
    verify_loaded: Option<LoadedImageVerifier<'_>>,
    numerical_environment: Option<OriginalNumericalEnvironment>,
    delegated: Option<&dyn DelegatedArtifactTreeWatch>,
    artifact_file_ceiling: Option<usize>,
) -> Result<OriginalProcessOutput, ArenaError> {
    input.window.check_before_spawn(limits)?;
    match artifact_file_ceiling {
        Some(ceiling) => watch.validate_with_file_ceiling(ceiling)?,
        None => watch.validate()?,
    }
    if input.bytes.is_empty() || input.bytes.len() > MAX_ORIGINAL_INPUT_BYTES {
        return Err(ArenaError::Budget(
            "original input requires 1..2MiB bytes".into(),
        ));
    }
    #[cfg(target_os = "linux")]
    {
        let mut observation = OriginalProcessObservation::new(input.bytes.len());
        observation.numerical_environment = numerical_environment.map(|_| "NVIDIA_TF32_OVERRIDE=0");
        observation.original_supervisor_entry_ns = input.window.elapsed_ns(Instant::now());
        if !directory
            .metadata()
            .map_err(|_| ArenaError::Io("process.cwd_metadata".into()))?
            .is_dir()
        {
            return Err(ArenaError::Invalid(
                "original process cwd handle must be a directory".into(),
            ));
        }
        let directory = directory
            .try_clone()
            .map_err(|_| ArenaError::Io("process.cwd_clone".into()))?;
        let process = linux::supervise(
            program,
            args,
            directory,
            limits,
            cancel,
            Some(match delegated {
                Some(delegated) => linux::ArtifactObservation::TreeDelegated(watch, delegated),
                None => linux::ArtifactObservation::Tree(watch),
            }),
            None,
            linux::InvocationIo {
                commands: Some(input.bytes),
                original: Some(input.window),
                observation: Some(&mut observation),
                verify_loaded,
                numerical_environment,
            },
        )?;
        let returned = Instant::now();
        observation.original_finished_ns = input.window.elapsed_ns(returned);
        observation.finished_before_whole = returned < input.window.whole;
        observation.cancelled_at_return =
            cancel.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire));
        Ok(OriginalProcessOutput {
            process,
            observation,
            window: input.window,
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (
            program,
            args,
            directory,
            cancel,
            verify_loaded,
            numerical_environment,
            delegated,
        );
        Err(ArenaError::Invalid(
            "original input supervision currently requires Linux".into(),
        ))
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use cap_std::fs::Dir;
    use nix::errno::Errno;
    use nix::fcntl::{FcntlArg, OFlag, fcntl, open, openat};
    use nix::sys::signal::{Signal, killpg};
    use nix::sys::stat::Mode;
    use nix::sys::wait::{Id, WaitPidFlag, WaitStatus, waitid};
    use nix::unistd::Pid;
    use std::io::{self, Read, Write};
    use std::os::fd::{AsFd, AsRawFd};
    use std::os::unix::fs::{FileExt, MetadataExt};
    use std::os::unix::process::{CommandExt, ExitStatusExt};
    use std::process::{Command, Stdio};
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};

    const POLL: Duration = Duration::from_millis(5);

    #[derive(Default)]
    pub(super) struct InvocationIo<'a, 'b, 'v> {
        pub commands: Option<&'a [u8]>,
        pub original: Option<OriginalProcessWindow>,
        pub observation: Option<&'b mut OriginalProcessObservation>,
        pub verify_loaded: Option<LoadedImageVerifier<'v>>,
        pub numerical_environment: Option<OriginalNumericalEnvironment>,
    }

    #[derive(Clone, Copy)]
    pub(super) enum ArtifactObservation<'a> {
        Flat(&'a ArtifactWatch),
        Tree(&'a OwnedArtifactTreeWatch),
        TreeDelegated(
            &'a OwnedArtifactTreeWatch,
            &'a dyn DelegatedArtifactTreeWatch,
        ),
    }

    impl<'a> ArtifactObservation<'a> {
        fn enforcement(self) -> String {
            match self {
                Self::Flat(_) => "regular_file_snapshot".into(),
                Self::Tree(_) | Self::TreeDelegated(..) => "owned_tree_snapshot".into(),
            }
        }

        fn tree(
            self,
        ) -> Option<(
            &'a OwnedArtifactTreeWatch,
            Option<&'a dyn DelegatedArtifactTreeWatch>,
        )> {
            match self {
                Self::Flat(_) => None,
                Self::Tree(watch) => Some((watch, None)),
                Self::TreeDelegated(watch, delegated) => Some((watch, Some(delegated))),
            }
        }
    }

    pub(super) fn pin_directory(cwd: &Path) -> Result<File, ArenaError> {
        open(
            cwd,
            OFlag::O_PATH | OFlag::O_DIRECTORY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
            Mode::empty(),
        )
        .map(File::from)
        .map_err(|_| ArenaError::Invalid("runner cwd must be a direct directory".into()))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn supervise(
        program: &File,
        args: &[OsString],
        cwd_file: File,
        limits: ProcessLimits,
        cancel: Option<&AtomicBool>,
        watch: Option<ArtifactObservation<'_>>,
        mut stdout_observer: Option<StdoutObserver<'_>>,
        io: InvocationIo<'_, '_, '_>,
    ) -> Result<ProcessOutput, ArenaError> {
        let InvocationIo {
            commands,
            original,
            mut observation,
            mut verify_loaded,
            numerical_environment,
        } = io;
        if let Some(window) = original {
            window.check_before_spawn(limits)?;
        }
        native_elf(program)?;
        default_child_disposition()?;
        // Verify required observation support before starting an executable.
        group_snapshot(Pid::this().as_raw() as u32, 0)
            .map_err(|_| ArenaError::Io("process.proc_preflight".into()))?;
        if cancel.is_some_and(|flag| flag.load(Ordering::Acquire)) {
            return Err(ArenaError::Invalid("process.cancelled_before_spawn".into()));
        }
        if let Some(window) = original {
            window.check_before_spawn(limits)?;
        }
        let started = Instant::now();
        if let (Some(window), Some(o)) = (original, observation.as_deref_mut()) {
            o.original_spawn_ns = window.elapsed_ns(started);
        }
        let wall = Duration::from_millis(limits.wall_ms);
        let grace = Duration::from_millis(limits.shutdown_grace_ms);
        started
            .checked_add(wall)
            .and_then(|time| time.checked_add(grace))
            .and_then(|time| time.checked_add(grace))
            .ok_or_else(|| ArenaError::Budget("process monotonic deadline overflow".into()))?;
        let executable = format!("/proc/self/fd/{}", program.as_raw_fd());
        let pinned_cwd = format!("/proc/self/fd/{}", cwd_file.as_raw_fd());
        let mut child = Command::new(executable)
            .args(args)
            .current_dir(pinned_cwd)
            .env_clear()
            .env("LANG", "C")
            .env("PATH", "/usr/bin:/bin")
            .envs(source_profile_environment(std::env::var_os(
                "RZ_ARENA_SOURCE_PROFILE",
            )))
            .envs(
                numerical_environment
                    .map(|OriginalNumericalEnvironment::Fp32Tf32Off| ("NVIDIA_TF32_OVERRIDE", "0")),
            )
            .stdin(if commands.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0)
            .spawn()
            .map_err(|error| ArenaError::Io(format!("process.spawn:{:?}", error.kind())))?;
        let pid = child.id();
        crate::emit_native_phase("runner_spawned");
        let group = Pid::from_raw(
            i32::try_from(pid).map_err(|_| ArenaError::Io("process.pid_out_of_range".into()))?,
        );
        let mut receipt = ProcessReceipt {
            supervisor_version: 1,
            pid,
            elapsed_ns: 0,
            cleanup_elapsed_ns: None,
            cleanup_timing_unavailable_reason: None,
            stop: ProcessStop::Exited,
            exit_code: None,
            exit_signal: None,
            group_cleanup: CleanupStatus::Unverified,
            stdout_bytes: 0,
            stderr_bytes: 0,
            observed_output_bytes: 0,
            descendant_cleanup_required: false,
            child_limit_enforcement: "process_group_snapshot".into(),
            watched_artifact_bytes: watch.map(|_| 0),
            artifact_limit_enforcement: watch.map(ArtifactObservation::enforcement),
            errors: Vec::new(),
        };
        let mut stdin = child.stdin.take();
        let mut input_offset = 0;
        let mut stdout = child.stdout.take();
        let mut stderr = child.stderr.take();
        let pipe_setup = stdout
            .as_ref()
            .ok_or(Errno::EBADF)
            .and_then(nonblocking)
            .and_then(|()| stderr.as_ref().ok_or(Errno::EBADF).and_then(nonblocking))
            .and_then(|()| match stdin.as_ref() {
                Some(pipe) => nonblocking(pipe),
                None => Ok(()),
            });
        let mut stop = if pipe_setup.is_err() {
            evidence(&mut receipt, "process.pipe_nonblocking");
            // A blocking descriptor is never read by this loop.
            stdout = None;
            stderr = None;
            stdin = None;
            Some(ProcessStop::IoFailure)
        } else {
            None
        };
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut shutdown = None;
        let mut killed = false;
        let mut leader_done = false;
        let mut ownership_lost = false;
        let mut stdout_eof = false;
        let mut stderr_eof = false;
        let mut cleanup_started = None;
        let preserve_unverified = watch.is_some_and(|watch| watch.tree().is_some());
        let mut loaded_image_checked = false;
        let mut loaded_image_progress = LoadedImageProgress::default();
        loop {
            let now = Instant::now();
            if stop.is_none() {
                if cancel.is_some_and(|flag| flag.load(Ordering::Acquire)) {
                    stop = Some(ProcessStop::Cancelled);
                } else if original.map_or_else(
                    || now.duration_since(started) >= wall,
                    |window| now >= window.execution,
                ) {
                    stop = Some(ProcessStop::WallLimit);
                }
            }
            if let Some(ArtifactObservation::Flat(watch)) = watch {
                match watched_size(&cwd_file, watch) {
                    Ok(bytes) => {
                        receipt.watched_artifact_bytes =
                            Some(receipt.watched_artifact_bytes.unwrap_or(0).max(bytes));
                        if bytes > watch.max_total_bytes {
                            stop.get_or_insert(ProcessStop::ArtifactLimit);
                        }
                    }
                    Err(code) => {
                        evidence(&mut receipt, code);
                        stop.get_or_insert(ProcessStop::IoFailure);
                    }
                }
            }
            if let Some((watch, delegated)) = watch.and_then(ArtifactObservation::tree) {
                observe_tree(&cwd_file, watch, delegated, &mut receipt, &mut stop);
            }
            if stop.is_none()
                && !loaded_image_checked
                && let Some(window) = original
            {
                // Keep the same owned, unreaped child and original E while its
                // executable observation is pending. Never write before the
                // matching image and optional content verification both finish.
                // No content-verification refusal is retried.
                match loaded_image_progress
                    .begin_attempt()
                    .and_then(|()| observe_loaded_image(program, pid, window))
                {
                    Ok((loaded, image)) => {
                        if let Some(o) = observation.as_deref_mut() {
                            o.loaded_image = Some(image);
                        }
                        if let Some(verifier) = verify_loaded.as_mut()
                            && verifier(&loaded).is_err()
                        {
                            evidence(&mut receipt, "process.loaded_image_verification_refused");
                            stop = Some(ProcessStop::IoFailure);
                        } else {
                            loaded_image_checked = true;
                        }
                    }
                    Err(code) if loaded_image_observation_pending(code) => {
                        loaded_image_progress.record_pending(code);
                    }
                    Err(code) => {
                        evidence(&mut receipt, code);
                        stop = Some(ProcessStop::IoFailure);
                    }
                }
            }
            if stop.is_none()
                && let Some(window) = original
            {
                // Every write rechecks control after artifact/image observation.
                // OS calls can finish late; they do not renew the original clock.
                if cancel.is_some_and(|flag| flag.load(Ordering::Acquire)) {
                    stop = Some(ProcessStop::Cancelled);
                } else if Instant::now() >= window.execution {
                    stop = Some(ProcessStop::WallLimit);
                }
            }
            if stop.is_none()
                && (original.is_none() || loaded_image_checked)
                && let (Some(pipe), Some(commands)) = (stdin.as_mut(), commands)
            {
                let end = (input_offset + 4096).min(commands.len());
                match pipe.write(&commands[input_offset..end]) {
                    Ok(0) => {
                        evidence(&mut receipt, "process.pipe_write_zero");
                        stop = Some(ProcessStop::IoFailure);
                    }
                    Ok(n) => {
                        if input_offset == 0
                            && let (Some(window), Some(o)) = (original, observation.as_deref_mut())
                        {
                            o.first_input_written_ns = window.elapsed_ns(Instant::now());
                        }
                        input_offset += n;
                        if input_offset == commands.len() {
                            stdin = None;
                        }
                    }
                    Err(e)
                        if matches!(
                            e.kind(),
                            io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                        ) => {}
                    Err(_) => {
                        evidence(&mut receipt, "process.pipe_write");
                        stop = Some(ProcessStop::IoFailure);
                    }
                }
            }
            // Each pass drains a finite amount from each pipe, preserving time,
            // cancellation and process checks even under continuous output.
            let prior_out = out.len();
            let prior_err = err.len();
            let stdout_open = stdout.is_some();
            let stderr_open = stderr.is_some();
            let drained = [
                drain(&mut stdout, &mut out, limits.max_output_bytes, &mut receipt),
                drain(&mut stderr, &mut err, limits.max_output_bytes, &mut receipt),
            ];
            stdout_eof |= stdout_open && stdout.is_none() && drained[0].is_ok();
            stderr_eof |= stderr_open && stderr.is_none() && drained[1].is_ok();
            for result in drained {
                match result {
                    Ok(true) if stop.is_none() => stop = Some(ProcessStop::OutputLimit),
                    Err(()) => {
                        evidence(&mut receipt, "process.pipe_read");
                        if stop.is_none() {
                            stop = Some(ProcessStop::IoFailure);
                        }
                    }
                    _ => {}
                }
            }
            // Observe only newly retained bytes; the raw pipe budget and later
            // provider/clock acceptance still own the complete unchanged output.
            if out.len() > prior_out {
                crate::native_diagnostics::observe_stream(true, &out[prior_out..]);
                if let Some(observer) = stdout_observer.as_mut() {
                    observer(&out[prior_out..]);
                }
            }
            if err.len() > prior_err {
                crate::native_diagnostics::observe_stream(false, &err[prior_err..]);
            }
            let previously_exited = leader_done;
            if !leader_done {
                match waitid(
                    Id::Pid(group),
                    WaitPidFlag::WEXITED | WaitPidFlag::WNOHANG | WaitPidFlag::WNOWAIT,
                ) {
                    Ok(WaitStatus::Exited(_, code)) => {
                        leader_done = true;
                        crate::emit_native_phase("runner_exit_observed");
                        if preserve_unverified {
                            receipt.exit_code = Some(code);
                        }
                    }
                    Ok(WaitStatus::Signaled(_, signal, _)) => {
                        leader_done = true;
                        crate::emit_native_phase("runner_exit_observed");
                        if preserve_unverified {
                            receipt.exit_signal = Some(signal as i32);
                        }
                    }
                    Ok(WaitStatus::StillAlive) => {}
                    Ok(_) => {
                        evidence(&mut receipt, "process.wait_status");
                        stop.get_or_insert(ProcessStop::IoFailure);
                    }
                    Err(Errno::EINTR) => {}
                    Err(_) => {
                        evidence(&mut receipt, "process.wait_observation");
                        // Ownership is no longer proven. An auto/foreign reaper
                        // may have released this numeric PID: never signal a
                        // potentially recycled group ID after this point.
                        stop = Some(ProcessStop::IoFailure);
                        ownership_lost = true;
                        evidence(&mut receipt, "process.ownership_lost");
                        break;
                    }
                }
            }
            if !previously_exited
                && leader_done
                && let (Some(window), Some(o)) = (original, observation.as_deref_mut())
            {
                let observed_at = Instant::now();
                o.original_exit_observed_ns = window.elapsed_ns(observed_at);
                o.exit_observed_before_execution = observed_at < window.execution;
            }
            if leader_done || stop.is_some() {
                cleanup_started.get_or_insert_with(Instant::now);
            }
            let last_group = match group_snapshot(pid, pid) {
                Ok(snapshot) => {
                    if snapshot.members > limits.max_child_processes {
                        stop.get_or_insert(ProcessStop::ChildLimit);
                    }
                    if leader_done && snapshot.other_members > 0 {
                        receipt.descendant_cleanup_required = true;
                    }
                    Some(snapshot)
                }
                Err(error) => {
                    evidence(&mut receipt, "process.proc_observation");
                    evidence(&mut receipt, &error.code());
                    stop.get_or_insert(ProcessStop::IoFailure);
                    None
                }
            };
            // The group observation itself can trigger a child-limit or I/O
            // stop. Begin timing before its subsequent termination/cleanup.
            if stop.is_some() {
                cleanup_started.get_or_insert_with(Instant::now);
            }
            let no_others = last_group.is_some_and(|snapshot| snapshot.other_members == 0);
            if original.is_some_and(|window| Instant::now() >= window.whole) {
                evidence(&mut receipt, "process.original_whole_window_expired");
                if stop.is_none() || stop == Some(ProcessStop::Exited) {
                    stop = Some(ProcessStop::WallLimit);
                }
                // waitid above preserved/rechecked ownership; no reap is made
                // from a recycled PID. Final status remains unverified here.
                signal(
                    group,
                    Signal::SIGKILL,
                    &mut receipt,
                    "process.kill_original_whole",
                );
                break;
            }
            if leader_done && no_others && stdout.is_none() && stderr.is_none() {
                // The child may have created its final outputs between this
                // pass's snapshot and exit observation. Inspect that final
                // tree before reporting a completed run, without changing the
                // older flat API's observation schedule.
                if let Some((watch, delegated)) = watch.and_then(ArtifactObservation::tree) {
                    observe_tree(&cwd_file, watch, delegated, &mut receipt, &mut stop);
                }
                receipt.group_cleanup = CleanupStatus::Gone;
                break;
            }
            if leader_done {
                stop.get_or_insert(ProcessStop::Exited);
            }
            if stop.is_some() && shutdown.is_none() {
                // The observation/drain work above can take time. Grace begins
                // at the actual termination request, not this pass's old clock.
                shutdown = Some(Instant::now());
                signal(group, Signal::SIGTERM, &mut receipt, "process.term");
            }
            if let Some(shutdown_at) = shutdown {
                let age = shutdown_at.elapsed();
                let original_kill_due = original.is_some_and(|window| {
                    let half = window.whole.duration_since(window.execution) / 2;
                    Instant::now() >= window.execution + half
                });
                if (age >= grace || original_kill_due) && !killed {
                    signal(group, Signal::SIGKILL, &mut receipt, "process.kill");
                    killed = true;
                }
                if age >= grace.saturating_mul(2) {
                    // This never joins a reader blocked by an escaped child.
                    evidence(&mut receipt, "process.cleanup_unverified");
                    break;
                }
            }
            std::thread::sleep(POLL);
        }
        if original.is_some() && !loaded_image_checked {
            evidence(
                &mut receipt,
                "process.loaded_image_not_confirmed_before_input",
            );
            if let Some(code) = loaded_image_progress.last_pending_code {
                evidence(&mut receipt, code);
            }
            if stop.is_none() || stop == Some(ProcessStop::Exited) {
                stop = Some(ProcessStop::IoFailure);
            }
        }
        let pending_child = finish_child(
            child,
            &mut receipt,
            &mut stop,
            leader_done,
            ownership_lost,
            preserve_unverified,
        );
        let cleanup_finished = Instant::now();
        record_cleanup_timing(
            &mut receipt,
            cleanup_started,
            cleanup_finished,
            ownership_lost,
            pending_child.is_some(),
        );
        receipt.stop = stop.unwrap_or(ProcessStop::Exited);
        receipt.elapsed_ns = u64::try_from(started.elapsed().as_nanos())
            .map_err(|_| ArenaError::Budget("process elapsed duration overflow".into()))?;
        receipt.stdout_bytes = out.len() as u64;
        receipt.stderr_bytes = err.len() as u64;
        if let Some(o) = observation {
            o.written_bytes = Some(input_offset);
            o.stdin_closed_after_input =
                commands.is_some_and(|bytes| input_offset == bytes.len()) && stdin.is_none();
            o.stdout_eof = stdout_eof;
            o.stderr_eof = stderr_eof;
            o.ownership_lost = ownership_lost;
            o.loaded_image_progress = loaded_image_progress;
        }
        Ok(ProcessOutput {
            receipt,
            stdout: out,
            stderr: err,
            pending_child,
        })
    }

    fn record_cleanup_timing(
        receipt: &mut ProcessReceipt,
        started: Option<Instant>,
        finished: Instant,
        ownership_lost: bool,
        pending_child: bool,
    ) {
        let unavailable = if ownership_lost
            || receipt
                .errors
                .iter()
                .any(|code| code == "process.ownership_lost")
        {
            Some("process_cleanup_ownership_lost_completion_unobserved")
        } else if pending_child || receipt.group_cleanup != CleanupStatus::Gone {
            Some("process_cleanup_completion_unverified")
        } else if receipt.errors.iter().any(|code| code == "process.reap") {
            Some("process_cleanup_reap_failed")
        } else if started.is_none() {
            Some("process_cleanup_interval_not_started")
        } else {
            None
        };
        if let Some(reason) = unavailable {
            receipt.cleanup_elapsed_ns = None;
            receipt.cleanup_timing_unavailable_reason = Some(reason.into());
        } else {
            match u64::try_from(
                finished
                    .duration_since(started.expect("observed cleanup start"))
                    .as_nanos(),
            ) {
                Ok(elapsed) => {
                    receipt.cleanup_elapsed_ns = Some(elapsed);
                    receipt.cleanup_timing_unavailable_reason = None;
                }
                Err(_) => {
                    receipt.cleanup_elapsed_ns = None;
                    receipt.cleanup_timing_unavailable_reason =
                        Some("process_cleanup_duration_overflow".into());
                }
            }
        }
    }

    fn finish_child(
        mut child: std::process::Child,
        receipt: &mut ProcessReceipt,
        stop: &mut Option<ProcessStop>,
        leader_done: bool,
        ownership_lost: bool,
        preserve_unverified: bool,
    ) -> Option<std::process::Child> {
        if preserve_unverified && ownership_lost {
            // Preserve the original handle and evidence, not numeric PID
            // authority: a foreign/automatic reaper may already have freed it.
            receipt.group_cleanup = CleanupStatus::Unverified;
            *stop = Some(ProcessStop::IoFailure);
            evidence(receipt, "process.ownership_lost");
        }
        let retain_child = if preserve_unverified {
            ownership_lost || !leader_done || receipt.group_cleanup == CleanupStatus::Unverified
        } else {
            !ownership_lost && !leader_done
        };
        if leader_done && !retain_child && !ownership_lost {
            match child.wait() {
                Ok(status) => {
                    receipt.exit_code = status.code();
                    receipt.exit_signal = status.signal();
                }
                Err(_) => {
                    evidence(receipt, "process.reap");
                    if preserve_unverified {
                        receipt.group_cleanup = CleanupStatus::Unverified;
                        *stop = Some(ProcessStop::IoFailure);
                        evidence(receipt, "process.ownership_lost");
                        return Some(child);
                    }
                }
            }
        } else {
            // This supervisor did not reap it. With ownership_lost this is
            // not a claim that a foreign reaper left the child unreaped.
            evidence(receipt, "process.leader_unreaped");
        }
        retain_child.then_some(child)
    }

    fn loaded_image_observation_pending(code: &str) -> bool {
        matches!(
            code,
            "process.loaded_image_open" | "process.loaded_image_differs_from_pinned_file"
        )
    }

    fn observe_loaded_image(
        program: &File,
        pid: u32,
        window: OriginalProcessWindow,
    ) -> Result<(File, LoadedImageObservation), &'static str> {
        let expected = program
            .metadata()
            .map_err(|_| "process.pinned_image_metadata")?;
        // Resolve the actual unreaped child's proc link, not an argv/path report.
        // The File pins this observation while metadata is read, then closes.
        let loaded =
            File::open(format!("/proc/{pid}/exe")).map_err(|_| "process.loaded_image_open")?;
        let actual = loaded
            .metadata()
            .map_err(|_| "process.loaded_image_metadata")?;
        if !expected.is_file() {
            return Err("process.pinned_image_not_regular");
        }
        if !actual.is_file() {
            return Err("process.loaded_image_not_regular");
        }
        if expected.dev() != actual.dev() || expected.ino() != actual.ino() {
            return Err("process.loaded_image_differs_from_pinned_file");
        }
        let observed_ns = window
            .elapsed_ns(Instant::now())
            .ok_or("process.loaded_image_clock_unavailable")?;
        Ok((
            loaded,
            LoadedImageObservation {
                pid,
                device: actual.dev(),
                inode: actual.ino(),
                observed_ns,
            },
        ))
    }

    fn native_elf(program: &File) -> Result<(), ArenaError> {
        let metadata = program
            .metadata()
            .map_err(|_| ArenaError::Io("process.program_metadata".into()))?;
        if !metadata.is_file() {
            return Err(ArenaError::Invalid(
                "runner must be a regular native ELF file".into(),
            ));
        }
        let mut header = [0_u8; 4];
        program
            .read_exact_at(&mut header, 0)
            .map_err(|_| ArenaError::Invalid("runner ELF header unavailable".into()))?;
        if header != *b"\x7fELF" {
            return Err(ArenaError::Invalid(
                "runner scripts and non-ELF files are unsupported".into(),
            ));
        }
        Ok(())
    }

    fn default_child_disposition() -> Result<(), ArenaError> {
        // These masks reject visible ignored/caught handlers. They cannot
        // expose SIG_DFL+SA_NOCLDWAIT: unchanged default flags and exclusive
        // reaping remain a caller precondition, not a claimed /proc proof.
        let file = File::open("/proc/self/status")
            .map_err(|_| ArenaError::Io("process.signal_preflight".into()))?;
        let mut bytes = Vec::new();
        file.take(65_537)
            .read_to_end(&mut bytes)
            .map_err(|_| ArenaError::Io("process.signal_preflight".into()))?;
        if bytes.len() > 65_536 {
            return Err(ArenaError::Io("process.signal_preflight_size".into()));
        }
        let mut ignored = None;
        let mut caught = None;
        for line in bytes.split(|byte| *byte == b'\n') {
            for (prefix, target) in [
                (b"SigIgn:".as_slice(), &mut ignored),
                (b"SigCgt:".as_slice(), &mut caught),
            ] {
                if let Some(value) = line.strip_prefix(prefix) {
                    *target = std::str::from_utf8(value)
                        .ok()
                        .and_then(|value| u64::from_str_radix(value.trim(), 16).ok());
                }
            }
        }
        let mask = 1_u64 << (Signal::SIGCHLD as u32 - 1);
        match (ignored, caught) {
            (Some(ignored), Some(caught)) if (ignored | caught) & mask == 0 => Ok(()),
            (Some(_), Some(_)) => Err(ArenaError::Invalid(
                "process requires default SIGCHLD disposition and exclusive child reaping".into(),
            )),
            _ => Err(ArenaError::Io("process.signal_preflight_fields".into())),
        }
    }

    fn watched_size(cwd: &File, watch: &ArtifactWatch) -> Result<u64, &'static str> {
        let mut bytes = 0_u64;
        for path in &watch.relative_files {
            // O_PATH only obtains an inode reference. No FIFO/device open can
            // block, and NOFOLLOW returns a symlink inode instead of its target.
            let file = match openat(
                cwd,
                path.as_path(),
                OFlag::O_PATH | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
                Mode::empty(),
            ) {
                Ok(fd) => File::from(fd),
                Err(Errno::ENOENT) => continue,
                Err(_) => return Err("process.artifact_open"),
            };
            let metadata = file.metadata().map_err(|_| "process.artifact_metadata")?;
            if !metadata.is_file() {
                return Err("process.artifact_kind");
            }
            bytes = bytes
                .checked_add(metadata.len())
                .ok_or("process.artifact_size_overflow")?;
        }
        Ok(bytes)
    }

    struct TreeWatchFailure {
        bytes: u64,
        stop: ProcessStop,
        code: &'static str,
    }

    #[derive(Default)]
    struct TreeSnapshot {
        bytes: u64,
        files: usize,
        directories: usize,
        entries: usize,
        delegated_matches: usize,
    }

    fn observe_tree(
        cwd: &File,
        watch: &OwnedArtifactTreeWatch,
        delegated: Option<&dyn DelegatedArtifactTreeWatch>,
        receipt: &mut ProcessReceipt,
        stop: &mut Option<ProcessStop>,
    ) {
        let observed = match delegated {
            Some(delegated) => watched_tree_size_with_delegation(cwd, watch, Some(delegated)),
            None => watched_tree_size(cwd, watch),
        };
        let (bytes, failure) = match observed {
            Ok(bytes) => (bytes, None),
            Err(failure) => (failure.bytes, Some(failure)),
        };
        receipt.watched_artifact_bytes =
            Some(receipt.watched_artifact_bytes.unwrap_or(0).max(bytes));
        if let Some(failure) = failure {
            evidence(receipt, failure.code);
            // A terminal leader is not a successful artifact verdict. A later
            // observed tree failure must not be hidden by the provisional
            // Exited stop while pipes/group cleanup were still in progress.
            if stop.is_none() || *stop == Some(ProcessStop::Exited) {
                *stop = Some(failure.stop);
            }
        }
    }

    fn watched_tree_size(
        cwd: &File,
        watch: &OwnedArtifactTreeWatch,
    ) -> Result<u64, TreeWatchFailure> {
        watched_tree_size_with_delegation(cwd, watch, None)
    }

    fn watched_tree_size_with_delegation(
        cwd: &File,
        watch: &OwnedArtifactTreeWatch,
        delegated: Option<&dyn DelegatedArtifactTreeWatch>,
    ) -> Result<u64, TreeWatchFailure> {
        let mut snapshot = TreeSnapshot::default();
        let result = visit_tree(cwd, watch, delegated, 0, &mut snapshot).and_then(|()| {
            if delegated.is_some() && snapshot.delegated_matches != 1 {
                return Err((
                    ProcessStop::IoFailure,
                    "process.artifact_tree.delegated_missing",
                ));
            }
            Ok(())
        });
        match result {
            Ok(()) => Ok(snapshot.bytes),
            Err((stop, code)) => Err(TreeWatchFailure {
                bytes: snapshot.bytes,
                stop,
                code,
            }),
        }
    }

    fn visit_tree(
        directory: &File,
        watch: &OwnedArtifactTreeWatch,
        delegated: Option<&dyn DelegatedArtifactTreeWatch>,
        depth: usize,
        snapshot: &mut TreeSnapshot,
    ) -> Result<(), (ProcessStop, &'static str)> {
        if let Some(delegated) = delegated {
            let matched = delegated
                .matches_directory(directory)
                .map_err(|code| (ProcessStop::IoFailure, code))?;
            if matched {
                // The stage root itself is never an exempt domain. One
                // exact entry must be observed once per complete snapshot.
                if depth == 0 || snapshot.delegated_matches != 0 {
                    return Err((
                        ProcessStop::IoFailure,
                        "process.artifact_tree.delegated_identity",
                    ));
                }
                snapshot.delegated_matches += 1;
                let observed = delegated
                    .observe_directory(directory)
                    .map_err(|code| (ProcessStop::IoFailure, code))?;
                observed
                    .validate()
                    .map_err(|code| (ProcessStop::IoFailure, code))?;
                return merge_delegated(watch, depth, snapshot, &observed);
            }
        }
        // Opening '.' obtains a fresh readable directory description from the
        // already pinned inode, including an O_PATH root. It never reopens the
        // entry's former name, and its offset is not shared with the caller.
        let readable = openat(
            directory,
            ".",
            OFlag::O_RDONLY | OFlag::O_DIRECTORY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
            Mode::empty(),
        )
        .map(File::from)
        .map_err(|_| {
            (
                ProcessStop::IoFailure,
                "process.artifact_tree.directory_open",
            )
        })?;
        let capability = Dir::from_std_file(readable);
        let entries = capability
            .entries()
            .map_err(|_| (ProcessStop::IoFailure, "process.artifact_tree.enumerate"))?;
        for entry in entries {
            let entry =
                entry.map_err(|_| (ProcessStop::IoFailure, "process.artifact_tree.enumerate"))?;
            // Count even a disappearing entry: directory churn cannot turn a
            // nominal file limit into an unbounded enumeration pass.
            snapshot.entries += 1;
            if snapshot.entries > watch.max_files * 2 {
                return Err((
                    ProcessStop::ArtifactLimit,
                    "process.artifact_tree.entry_count",
                ));
            }
            let name = entry.file_name();
            let mut components = Path::new(&name).components();
            if !matches!(components.next(), Some(Component::Normal(_)))
                || components.next().is_some()
            {
                return Err((ProcessStop::IoFailure, "process.artifact_tree.entry_name"));
            }
            // O_PATH references every inode kind without opening a FIFO,
            // socket or device payload. NOFOLLOW preserves a symlink itself.
            let inode = match openat(
                directory,
                Path::new(&name),
                OFlag::O_PATH | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
                Mode::empty(),
            ) {
                Ok(fd) => File::from(fd),
                Err(Errno::ENOENT) => continue,
                Err(_) => {
                    return Err((ProcessStop::IoFailure, "process.artifact_tree.inode_open"));
                }
            };
            let metadata = inode
                .metadata()
                .map_err(|_| (ProcessStop::IoFailure, "process.artifact_tree.metadata"))?;
            if metadata.is_dir() {
                snapshot.directories += 1;
                if snapshot.directories > watch.max_files {
                    return Err((
                        ProcessStop::ArtifactLimit,
                        "process.artifact_tree.directory_count",
                    ));
                }
                if depth >= watch.max_depth {
                    return Err((ProcessStop::ArtifactLimit, "process.artifact_tree.depth"));
                }
                visit_tree(&inode, watch, delegated, depth + 1, snapshot)?;
            } else if metadata.is_file() {
                if metadata.nlink() != 1 {
                    return Err((ProcessStop::IoFailure, "process.artifact_tree.hard_link"));
                }
                snapshot.files += 1;
                if snapshot.files > watch.max_files {
                    return Err((
                        ProcessStop::ArtifactLimit,
                        "process.artifact_tree.file_count",
                    ));
                }
                let Some(bytes) = snapshot.bytes.checked_add(metadata.len()) else {
                    snapshot.bytes = u64::MAX;
                    return Err((
                        ProcessStop::ArtifactLimit,
                        "process.artifact_tree.size_overflow",
                    ));
                };
                snapshot.bytes = bytes;
                if metadata.len() > watch.max_file_bytes {
                    return Err((
                        ProcessStop::ArtifactLimit,
                        "process.artifact_tree.file_bytes",
                    ));
                }
                if snapshot.bytes > watch.max_total_bytes {
                    return Err((
                        ProcessStop::ArtifactLimit,
                        "process.artifact_tree.total_bytes",
                    ));
                }
            } else {
                return Err((ProcessStop::IoFailure, "process.artifact_tree.kind"));
            }
        }
        Ok(())
    }

    fn merge_delegated(
        watch: &OwnedArtifactTreeWatch,
        depth: usize,
        snapshot: &mut TreeSnapshot,
        observed: &DelegatedTreeSnapshot,
    ) -> Result<(), (ProcessStop, &'static str)> {
        if observed.max_file_bytes > watch.max_file_bytes {
            return Err((
                ProcessStop::ArtifactLimit,
                "process.artifact_tree.file_bytes",
            ));
        }
        if depth
            .checked_add(observed.depth)
            .is_none_or(|depth| depth > watch.max_depth)
        {
            return Err((ProcessStop::ArtifactLimit, "process.artifact_tree.depth"));
        }
        snapshot.files = snapshot.files.checked_add(observed.files).ok_or((
            ProcessStop::ArtifactLimit,
            "process.artifact_tree.file_count",
        ))?;
        if snapshot.files > watch.max_files {
            return Err((
                ProcessStop::ArtifactLimit,
                "process.artifact_tree.file_count",
            ));
        }
        snapshot.directories = snapshot
            .directories
            .checked_add(observed.directories)
            .ok_or((
                ProcessStop::ArtifactLimit,
                "process.artifact_tree.directory_count",
            ))?;
        if snapshot.directories > watch.max_files {
            return Err((
                ProcessStop::ArtifactLimit,
                "process.artifact_tree.directory_count",
            ));
        }
        snapshot.entries = snapshot.entries.checked_add(observed.entries).ok_or((
            ProcessStop::ArtifactLimit,
            "process.artifact_tree.entry_count",
        ))?;
        if snapshot.entries > watch.max_files * 2 {
            return Err((
                ProcessStop::ArtifactLimit,
                "process.artifact_tree.entry_count",
            ));
        }
        snapshot.bytes = snapshot.bytes.checked_add(observed.bytes).ok_or((
            ProcessStop::ArtifactLimit,
            "process.artifact_tree.size_overflow",
        ))?;
        if snapshot.bytes > watch.max_total_bytes {
            return Err((
                ProcessStop::ArtifactLimit,
                "process.artifact_tree.total_bytes",
            ));
        }
        Ok(())
    }

    fn nonblocking<T: AsFd>(pipe: &T) -> Result<(), Errno> {
        let flags = OFlag::from_bits_truncate(fcntl(pipe, FcntlArg::F_GETFL)?);
        fcntl(pipe, FcntlArg::F_SETFL(flags | OFlag::O_NONBLOCK))?;
        Ok(())
    }

    fn drain<T: Read>(
        pipe: &mut Option<T>,
        retained: &mut Vec<u8>,
        cap: u64,
        receipt: &mut ProcessReceipt,
    ) -> Result<bool, ()> {
        let Some(reader) = pipe.as_mut() else {
            return Ok(false);
        };
        let mut buffer = [0_u8; 8192];
        for _ in 0..16 {
            let stored = receipt.stdout_bytes + receipt.stderr_bytes;
            let remaining = cap - stored;
            let request = usize::try_from(remaining.saturating_add(1))
                .unwrap_or(usize::MAX)
                .min(buffer.len());
            match reader.read(&mut buffer[..request]) {
                Ok(0) => {
                    *pipe = None;
                    return Ok(false);
                }
                Ok(bytes) => {
                    receipt.observed_output_bytes = receipt
                        .observed_output_bytes
                        .checked_add(bytes as u64)
                        .ok_or(())?;
                    let keep = bytes.min(remaining as usize);
                    retained.extend_from_slice(&buffer[..keep]);
                    // Increment the combined counter without binding a pipe's
                    // identity; final stdout/stderr lengths replace it below.
                    receipt.stdout_bytes += keep as u64;
                    if bytes > keep {
                        *pipe = None;
                        return Ok(true);
                    }
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(false),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => {
                    *pipe = None;
                    return Err(());
                }
            }
        }
        Ok(false)
    }

    #[derive(Clone, Copy)]
    struct GroupSnapshot {
        members: u32,
        other_members: u32,
    }

    #[derive(Debug)]
    struct ProcObservationError {
        stage: &'static str,
        errno: Option<i32>,
    }

    impl ProcObservationError {
        fn invalid(stage: &'static str) -> Self {
            Self { stage, errno: None }
        }

        fn io(stage: &'static str, error: io::Error) -> Self {
            Self {
                stage,
                errno: error.raw_os_error(),
            }
        }

        fn code(&self) -> String {
            format!(
                "process.proc_observation.{}:{}",
                self.stage,
                self.errno.unwrap_or(0)
            )
        }
    }

    fn group_snapshot(group: u32, leader: u32) -> Result<GroupSnapshot, ProcObservationError> {
        let mut members = 0_u32;
        let mut other_members = 0_u32;
        for entry in std::fs::read_dir("/proc")
            .map_err(|error| ProcObservationError::io("enumerate", error))?
        {
            let entry = entry.map_err(|error| ProcObservationError::io("entry", error))?;
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            let Ok(pid) = name.parse::<u32>() else {
                continue;
            };
            let stat = match File::open(entry.path().join("stat")) {
                Ok(file) => file,
                Err(error) if disappeared(&error) => continue,
                Err(error) => return Err(ProcObservationError::io("stat_open", error)),
            };
            let mut bytes = Vec::new();
            match stat.take(4097).read_to_end(&mut bytes) {
                Ok(_) if bytes.len() <= 4096 => {}
                Err(error) if disappeared(&error) => continue,
                Err(error) => return Err(ProcObservationError::io("stat_read", error)),
                _ => return Err(ProcObservationError::invalid("stat_size")),
            }
            let Some(pgid) = parse_proc_group(&bytes)? else {
                continue;
            };
            if pgid == group {
                members = members
                    .checked_add(1)
                    .ok_or_else(|| ProcObservationError::invalid("member_count"))?;
                if pid != leader {
                    other_members = other_members
                        .checked_add(1)
                        .ok_or_else(|| ProcObservationError::invalid("member_count"))?;
                }
            }
        }
        Ok(GroupSnapshot {
            members,
            other_members,
        })
    }

    fn parse_proc_group(bytes: &[u8]) -> Result<Option<u32>, ProcObservationError> {
        // comm may contain spaces, ')' and non-UTF-8 bytes. Only decode the
        // ASCII numeric suffix after its final ')' delimiter.
        let delimiter = bytes
            .iter()
            .rposition(|byte| *byte == b')')
            .ok_or_else(|| {
                ProcObservationError::invalid(if bytes.is_empty() {
                    "stat_empty"
                } else {
                    "stat_delimiter"
                })
            })?;
        let fields = std::str::from_utf8(&bytes[delimiter + 1..])
            .map_err(|_| ProcObservationError::invalid("stat_utf8"))?;
        let pgid = fields
            .split_whitespace()
            .nth(2)
            .ok_or_else(|| ProcObservationError::invalid("stat_group_missing"))?
            .parse::<i32>()
            .map_err(|_| ProcObservationError::invalid("stat_group_invalid"))?;
        // Linux do_task_stat starts pgid at -1. If the task has been retired
        // and lock_task_sighand fails, the kernel prints that sentinel instead
        // of a group ID. It cannot name our positive, unreaped leader's group.
        // Malformed records and other negative IDs still fail observation.
        match pgid {
            -1 => Ok(None),
            0.. => Ok(Some(pgid as u32)),
            _ => Err(ProcObservationError::invalid("stat_group_invalid")),
        }
    }

    fn disappeared(error: &io::Error) -> bool {
        error.kind() == io::ErrorKind::NotFound || error.raw_os_error() == Some(Errno::ESRCH as i32)
    }

    /// One explicit passive diagnostic flag; never forward ambient variables,
    /// library search paths, credentials or provider/precision selections.
    fn source_profile_environment(
        value: Option<std::ffi::OsString>,
    ) -> Option<(&'static str, &'static str)> {
        (value.as_deref() == Some(std::ffi::OsStr::new("1")))
            .then_some(("RZ_NATIVE_SOURCE_PROFILE", "1"))
    }

    fn signal(group: Pid, signal: Signal, receipt: &mut ProcessReceipt, step: &str) {
        match killpg(group, signal) {
            Ok(()) | Err(Errno::ESRCH) => {}
            Err(error) => evidence(receipt, &format!("{step}:{}", error as i32)),
        }
    }

    fn evidence(receipt: &mut ProcessReceipt, code: &str) {
        if receipt.errors.len() < 16 && !receipt.errors.iter().any(|item| item == code) {
            receipt.errors.push(code.chars().take(256).collect());
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn rovent_artifact_count_profile_does_not_relax_generic_watch() {
            let generic = OwnedArtifactTreeWatch::bounded_output(1024, 512);
            assert_eq!(generic.max_files, 4096);
            assert!(
                OwnedArtifactTreeWatch {
                    max_files: 4097,
                    ..generic
                }
                .validate()
                .is_err()
            );
            let registered = RoventTrainingArtifactTreeWatch::registered(1024, 512, 32768).unwrap();
            assert_eq!(registered.0.max_files, 32768);
            assert!(registered.0.validate().is_err());
            assert!(RoventTrainingArtifactTreeWatch::registered(1024, 512, 32769).is_err());
            assert!(RoventTrainingArtifactTreeWatch::registered(0, 512, 32768).is_err());
        }

        fn original_watch() -> OwnedArtifactTreeWatch {
            OwnedArtifactTreeWatch {
                max_total_bytes: 1024,
                max_file_bytes: 1024,
                max_files: 4,
                max_depth: 0,
            }
        }

        fn original_limits() -> ProcessLimits {
            ProcessLimits {
                wall_ms: 5000,
                shutdown_grace_ms: 250,
                max_output_bytes: 256 * 1024,
                max_child_processes: 1,
            }
        }

        struct OriginalTestDirectory(std::path::PathBuf);
        impl OriginalTestDirectory {
            fn new() -> Self {
                static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let path = std::env::temp_dir().join(format!(
                    "rz-original-process-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
                std::fs::create_dir(&path).unwrap();
                Self(path)
            }
            fn file(&self) -> File {
                File::open(&self.0).unwrap()
            }
        }
        impl Drop for OriginalTestDirectory {
            fn drop(&mut self) {
                // These tests create no artifacts or descendants. Never recurse
                // through a replacement entry or remove another test's tree.
                let _ = std::fs::remove_dir(&self.0);
            }
        }

        struct FakeDelegatedTree {
            root: OriginalTestDirectory,
            directory: File,
            name: PathBuf,
            snapshot: DelegatedTreeSnapshot,
        }
        impl FakeDelegatedTree {
            fn new() -> Self {
                let root = OriginalTestDirectory::new();
                let name = root.0.join("registered-cache");
                std::fs::create_dir(&name).unwrap();
                std::fs::write(name.join("payload.bin"), b"1234").unwrap();
                std::os::unix::fs::symlink("payload.bin", name.join("alias")).unwrap();
                Self {
                    directory: File::open(&name).unwrap(),
                    name,
                    root,
                    snapshot: DelegatedTreeSnapshot {
                        bytes: 15,
                        files: 2,
                        directories: 0,
                        entries: 2,
                        max_file_bytes: 11,
                        depth: 0,
                    },
                }
            }
        }
        impl DelegatedArtifactTreeWatch for FakeDelegatedTree {
            fn matches_directory(&self, directory: &File) -> Result<bool, &'static str> {
                let held = self
                    .directory
                    .metadata()
                    .map_err(|_| "test.delegated.metadata")?;
                let named =
                    std::fs::symlink_metadata(&self.name).map_err(|_| "test.delegated.binding")?;
                if !named.is_dir() || (named.dev(), named.ino()) != (held.dev(), held.ino()) {
                    return Err("test.delegated.binding");
                }
                let observed = directory
                    .metadata()
                    .map_err(|_| "test.delegated.metadata")?;
                Ok((observed.dev(), observed.ino()) == (held.dev(), held.ino()))
            }
            fn observe_directory(
                &self,
                directory: &File,
            ) -> Result<DelegatedTreeSnapshot, &'static str> {
                if !self.matches_directory(directory)? {
                    return Err("test.delegated.identity");
                }
                let names: std::collections::BTreeSet<_> = std::fs::read_dir(&self.name)
                    .map_err(|_| "test.delegated.enumerate")?
                    .map(|entry| entry.unwrap().file_name())
                    .collect();
                if names != [OsString::from("alias"), OsString::from("payload.bin")].into() {
                    return Err("test.delegated.unknown");
                }
                Ok(self.snapshot)
            }
        }
        impl Drop for FakeDelegatedTree {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(self.name.join("alias"));
                let _ = std::fs::remove_file(self.name.join("payload.bin"));
                let _ = std::fs::remove_dir(&self.name);
            }
        }

        #[test]
        fn delegated_tree_keeps_original_process_clock_and_generic_link_refusal() {
            let tree = FakeDelegatedTree::new();
            let cwd = tree.root.file();
            let watch = OwnedArtifactTreeWatch::bounded_output(32, 16);
            assert_eq!(
                watched_tree_size(&cwd, &watch).unwrap_err().code,
                "process.artifact_tree.kind"
            );
            assert_eq!(
                watched_tree_size_with_delegation(&cwd, &watch, Some(&tree))
                    .unwrap_or_else(|failure| panic!("{}", failure.code)),
                15
            );
            let program = File::open("/bin/cat").unwrap();
            let started = Instant::now();
            let input = OriginalProcessInput {
                bytes: b"original-delegated-input",
                window: OriginalProcessWindow::new(
                    started,
                    started + Duration::from_secs(5),
                    started + Duration::from_millis(5500),
                )
                .unwrap(),
            };
            let mut verify = |_: &File| Ok(());
            let result = supervise_numerical_input_with_delegated_tree_in_original_window(
                &program,
                &[],
                &cwd,
                original_limits(),
                None,
                &watch,
                input,
                &mut verify,
                OriginalNumericalEnvironment::Fp32Tf32Off,
                &tree,
            )
            .unwrap();
            assert!(result.transport_complete(), "{result:?}");
            assert_eq!(result.process().stdout, input.bytes);
            assert_eq!(result.process().receipt.watched_artifact_bytes, Some(15));
            assert_eq!(result.checked_timing().unwrap().original_started(), started);
            assert_eq!(
                result.observation().numerical_environment(),
                Some("NVIDIA_TF32_OVERRIDE=0")
            );
        }

        #[test]
        fn delegated_tree_refuses_extra_links_unknown_entries_rename_and_missing_match() {
            let tree = FakeDelegatedTree::new();
            let cwd = tree.root.file();
            let watch = OwnedArtifactTreeWatch::bounded_output(32, 16);
            let outside = tree.root.0.join("outside-alias");
            std::os::unix::fs::symlink("registered-cache/payload.bin", &outside).unwrap();
            assert_eq!(
                watched_tree_size_with_delegation(&cwd, &watch, Some(&tree))
                    .unwrap_err()
                    .code,
                "process.artifact_tree.kind"
            );
            std::fs::remove_file(outside).unwrap();
            let inside = tree.name.join("undeclared");
            std::os::unix::fs::symlink("payload.bin", &inside).unwrap();
            assert_eq!(
                watched_tree_size_with_delegation(&cwd, &watch, Some(&tree))
                    .unwrap_err()
                    .code,
                "test.delegated.unknown"
            );
            std::fs::remove_file(inside).unwrap();
            let moved = tree.root.0.join("moved-cache");
            std::fs::rename(&tree.name, &moved).unwrap();
            assert_eq!(
                watched_tree_size_with_delegation(&cwd, &watch, Some(&tree))
                    .unwrap_err()
                    .code,
                "test.delegated.binding"
            );
            std::fs::rename(&moved, &tree.name).unwrap();
            let unrelated = OriginalTestDirectory::new();
            assert_eq!(
                watched_tree_size_with_delegation(&unrelated.file(), &watch, Some(&tree))
                    .unwrap_err()
                    .code,
                "process.artifact_tree.delegated_missing"
            );
            assert_eq!(
                watched_tree_size_with_delegation(&tree.directory, &watch, Some(&tree))
                    .unwrap_err()
                    .code,
                "process.artifact_tree.delegated_identity"
            );
        }

        #[test]
        fn delegated_tree_sums_original_caps_and_refuses_incoherent_or_unknown_snapshot() {
            let mut tree = FakeDelegatedTree::new();
            let cwd = tree.root.file();
            let plain = tree.root.0.join("plain.bin");
            std::fs::write(&plain, b"123").unwrap();
            // Both the 15-byte delegated domain and the 3-byte generic file
            // fit independently; their total must still fail the original cap.
            let total = OwnedArtifactTreeWatch::bounded_output(16, 16);
            assert_eq!(
                watched_tree_size_with_delegation(&cwd, &total, Some(&tree))
                    .unwrap_err()
                    .code,
                "process.artifact_tree.total_bytes"
            );
            std::fs::remove_file(plain).unwrap();
            let file = OwnedArtifactTreeWatch::bounded_output(32, 10);
            assert_eq!(
                watched_tree_size_with_delegation(&cwd, &file, Some(&tree))
                    .unwrap_err()
                    .code,
                "process.artifact_tree.file_bytes"
            );
            let watch = OwnedArtifactTreeWatch::bounded_output(32, 16);
            let limit_cases = [
                (
                    OwnedArtifactTreeWatch {
                        max_files: 1,
                        ..watch
                    },
                    tree.snapshot,
                    "process.artifact_tree.file_count",
                ),
                (
                    OwnedArtifactTreeWatch {
                        max_depth: 0,
                        ..watch
                    },
                    tree.snapshot,
                    "process.artifact_tree.depth",
                ),
                (
                    OwnedArtifactTreeWatch {
                        max_files: 2,
                        ..watch
                    },
                    DelegatedTreeSnapshot {
                        entries: 4,
                        ..tree.snapshot
                    },
                    "process.artifact_tree.entry_count",
                ),
                (
                    OwnedArtifactTreeWatch {
                        max_files: 2,
                        ..watch
                    },
                    DelegatedTreeSnapshot {
                        directories: 2,
                        entries: 4,
                        depth: 1,
                        ..tree.snapshot
                    },
                    "process.artifact_tree.directory_count",
                ),
                (
                    OwnedArtifactTreeWatch {
                        max_depth: 1,
                        ..watch
                    },
                    DelegatedTreeSnapshot {
                        directories: 1,
                        entries: 3,
                        depth: 1,
                        ..tree.snapshot
                    },
                    "process.artifact_tree.depth",
                ),
            ];
            for (limits, observed, code) in limit_cases {
                let mut aggregate = TreeSnapshot {
                    directories: 1,
                    entries: 1,
                    ..Default::default()
                };
                assert_eq!(
                    merge_delegated(&limits, 1, &mut aggregate, &observed)
                        .unwrap_err()
                        .1,
                    code
                );
            }
            tree.snapshot.entries = 1;
            assert_eq!(
                watched_tree_size_with_delegation(&cwd, &watch, Some(&tree))
                    .unwrap_err()
                    .code,
                "process.artifact_tree.delegated_snapshot"
            );
            assert!(
                DelegatedTreeSnapshot {
                    bytes: 1,
                    files: 0,
                    directories: 0,
                    entries: 0,
                    max_file_bytes: 0,
                    depth: 0
                }
                .validate()
                .is_err()
            );
            assert!(
                DelegatedTreeSnapshot {
                    bytes: 0,
                    files: 0,
                    directories: 1,
                    entries: 1,
                    max_file_bytes: 0,
                    depth: 0
                }
                .validate()
                .is_err()
            );
        }

        #[test]
        fn bounded_output_watch_reaches_original_supervisor_and_keeps_owned_tree_guards() {
            struct TreeEntries(std::path::PathBuf);
            impl Drop for TreeEntries {
                fn drop(&mut self) {
                    let _ = std::fs::remove_file(self.0.join("artifact.bin"));
                    let _ = std::fs::remove_dir(&self.0);
                }
            }

            let directory = OriginalTestDirectory::new();
            let entries = TreeEntries(directory.0.join("outputs"));
            std::fs::create_dir(&entries.0).unwrap();
            let artifact = entries.0.join("artifact.bin");
            std::fs::write(&artifact, b"12345678").unwrap();
            let cwd = directory.file();
            // This is the same production constructor used by both numerical
            // owners, with small byte budgets so the real tree limits are tested.
            let watch = OwnedArtifactTreeWatch::bounded_output(16, 8);
            let program = File::open("/bin/cat").unwrap();
            let input = b"original-numerical-input";
            let started = Instant::now();
            let window = OriginalProcessWindow::new(
                started,
                started + Duration::from_secs(5),
                started + Duration::from_millis(5500),
            )
            .unwrap();
            let output = supervise_input_in_original_window(
                &program,
                &[],
                &cwd,
                original_limits(),
                None,
                &watch,
                OriginalProcessInput {
                    bytes: input,
                    window,
                },
            )
            .unwrap();
            assert!(output.transport_complete(), "{output:?}");
            assert_eq!(output.process().stdout, input);
            assert_eq!(output.process().receipt.watched_artifact_bytes, Some(8));
            assert_eq!(output.checked_timing().unwrap().original_started(), started);

            let total_cap = OwnedArtifactTreeWatch::bounded_output(7, 8);
            total_cap.validate().unwrap();
            let failure = watched_tree_size(&cwd, &total_cap).unwrap_err();
            assert_eq!(failure.stop, ProcessStop::ArtifactLimit);
            assert_eq!(failure.code, "process.artifact_tree.total_bytes");

            std::fs::write(&artifact, b"123456789").unwrap();
            let failure = watched_tree_size(&cwd, &watch).unwrap_err();
            assert_eq!(failure.stop, ProcessStop::ArtifactLimit);
            assert_eq!(failure.code, "process.artifact_tree.file_bytes");

            std::fs::remove_file(&artifact).unwrap();
            std::os::unix::fs::symlink("/nonexistent-unowned-artifact", &artifact).unwrap();
            let failure = watched_tree_size(&cwd, &watch).unwrap_err();
            assert_eq!(failure.stop, ProcessStop::IoFailure);
            assert_eq!(failure.code, "process.artifact_tree.kind");
        }

        #[test]
        fn original_window_refuses_reordering_expiry_and_resource_extension() {
            let now = Instant::now();
            assert!(OriginalProcessWindow::new(now, now, now + Duration::from_secs(1)).is_err());
            assert!(OriginalProcessWindow::new(now, now + Duration::from_secs(1), now).is_err());
            let old = now - Duration::from_secs(2);
            let expired =
                OriginalProcessWindow::new(old, old + Duration::from_secs(1), now).unwrap();
            assert!(expired.check_before_spawn(original_limits()).is_err());
            let too_long = OriginalProcessWindow::new(
                now,
                now + Duration::from_secs(6),
                now + Duration::from_millis(6500),
            )
            .unwrap();
            assert!(too_long.check_before_spawn(original_limits()).is_err());
            let extra_cleanup = OriginalProcessWindow::new(
                now,
                now + Duration::from_secs(1),
                now + Duration::from_secs(2),
            )
            .unwrap();
            assert!(extra_cleanup.check_before_spawn(original_limits()).is_err());
        }

        #[test]
        fn original_input_transports_more_than_protocol_limit_and_requires_real_closure() {
            let directory = OriginalTestDirectory::new();
            let program = File::open("/bin/cat").unwrap();
            // No newline: this lane transports frozen JSON bytes, not UCI lines.
            let input = vec![b'x'; 96 * 1024];
            let started = Instant::now();
            let window = OriginalProcessWindow::new(
                started,
                started + Duration::from_secs(5),
                started + Duration::from_millis(5500),
            )
            .unwrap();
            let mut output = supervise_input_in_original_window(
                &program,
                &[],
                &directory.file(),
                original_limits(),
                None,
                &original_watch(),
                OriginalProcessInput {
                    bytes: &input,
                    window,
                },
            )
            .unwrap();
            assert_eq!(output.process().stdout, input);
            assert_eq!(output.observation().written_bytes(), Some(input.len()));
            assert!(output.transport_complete(), "{output:?}");
            assert!(output.observation().exit_observed_before_execution());
            assert!(output.observation().original_exit_observed_ns().is_some());
            let timing = output.checked_timing().unwrap();
            assert_eq!(timing.original_started(), started);
            assert_eq!(timing.execution_deadline(), window.execution);
            assert_eq!(timing.whole_deadline(), window.whole);
            assert!(timing.supervisor_entry_ns() <= timing.launch_started_ns());
            assert!(timing.launch_started_ns() <= timing.exit_observed_ns());
            assert!(timing.exit_observed_ns() <= timing.supervisor_finished_ns());
            let image = output.checked_loaded_image().unwrap();
            let metadata = program.metadata().unwrap();
            assert_eq!(image.process_id(), output.process().receipt.pid);
            assert_eq!(image.file_device(), metadata.dev());
            assert_eq!(image.file_inode(), metadata.ino());
            assert!(timing.launch_started_ns() <= image.observed_ns());
            assert!(image.observed_ns() <= image.first_input_written_ns());
            assert!(image.first_input_written_ns() <= timing.exit_observed_ns());
            let serialized = serde_json::to_value(output.observation()).unwrap();
            assert_eq!(serialized["schema"], "rz-original-process-transport/1");
            assert!(serialized.get("loaded_image").is_none());
            assert!(serialized.get("first_input_written_ns").is_none());
            assert!(serialized.get("loaded_image_progress").is_none());
            assert!(image.observation().loaded_image_attempts() > 0);
            assert!(
                image.observation().loaded_image_pending_observations()
                    < image.observation().loaded_image_attempts()
            );
            let original_write = output.observation.first_input_written_ns.take();
            assert!(output.checked_loaded_image().is_err());
            output.observation.first_input_written_ns = original_write;
            let original_image = output.observation.loaded_image.take();
            // Complete transport alone never constructs a loaded-image borrow.
            assert!(output.transport_complete());
            assert!(output.checked_loaded_image().is_err());
            output.observation.loaded_image = original_image;
            let old_image_time = output
                .observation
                .loaded_image
                .as_ref()
                .unwrap()
                .observed_ns;
            output
                .observation
                .loaded_image
                .as_mut()
                .unwrap()
                .observed_ns = output.observation.first_input_written_ns.unwrap() + 1;
            assert!(output.checked_loaded_image().is_err());
            output
                .observation
                .loaded_image
                .as_mut()
                .unwrap()
                .observed_ns = old_image_time;
            let old_image_pid = output.observation.loaded_image.as_ref().unwrap().pid;
            output.observation.loaded_image.as_mut().unwrap().pid = old_image_pid + 1;
            assert!(output.checked_loaded_image().is_err());
            output.observation.loaded_image.as_mut().unwrap().pid = old_image_pid;
            assert!(output.checked_loaded_image().is_ok());
            let old_spawn = output.observation.original_spawn_ns;
            output.observation.original_spawn_ns =
                Some(output.observation.original_exit_observed_ns.unwrap() + 1);
            assert!(!output.transport_complete());
            assert!(output.checked_timing().is_err());
            output.observation.original_spawn_ns = old_spawn;
            let old_finish = output.observation.original_finished_ns;
            output.observation.original_finished_ns = Some(5_500_000_000);
            assert!(!output.transport_complete());
            assert!(output.checked_timing().is_err());
            output.observation.original_finished_ns = old_finish;
            output.observation.exit_observed_before_execution = false;
            assert!(!output.transport_complete());
            output.observation.exit_observed_before_execution = true;
            output.observation.stdout_eof = false;
            assert!(!output.transport_complete());
            output.observation.stdout_eof = true;
            output.observation.finished_before_whole = false;
            assert!(!output.transport_complete());
            output.observation.finished_before_whole = true;
            output.observation.written_bytes = Some(input.len() - 1);
            assert!(!output.transport_complete());
            // The old protocol API still refuses this input before spawning.
            assert!(
                supervise_protocol_in_directory(
                    &program,
                    &[],
                    &directory.file(),
                    original_limits(),
                    None,
                    &original_watch(),
                    &input,
                )
                .is_err()
            );
        }

        #[test]
        fn loaded_image_observation_rejects_a_different_actual_executable() {
            let started = Instant::now();
            let window = OriginalProcessWindow::new(
                started,
                started + Duration::from_secs(5),
                started + Duration::from_millis(5500),
            )
            .unwrap();
            // A real proc executable is compared, without a caller-supplied
            // path, forged serialized report or a fabricated success owner.
            let actual = File::open("/proc/self/exe").unwrap();
            let (_, image) = observe_loaded_image(&actual, std::process::id(), window).unwrap();
            assert_eq!(image.inode, actual.metadata().unwrap().ino());
            let other = File::open("/bin/cat").unwrap();
            assert_eq!(
                observe_loaded_image(&other, std::process::id(), window).unwrap_err(),
                "process.loaded_image_differs_from_pinned_file"
            );
        }

        #[test]
        fn loaded_image_pending_does_not_retry_content_or_metadata_failures() {
            assert!(loaded_image_observation_pending(
                "process.loaded_image_open"
            ));
            assert!(loaded_image_observation_pending(
                "process.loaded_image_differs_from_pinned_file"
            ));
            for code in [
                "process.loaded_image_verification_refused",
                "process.pinned_image_metadata",
                "process.loaded_image_metadata",
                "process.pinned_image_not_regular",
                "process.loaded_image_not_regular",
                "process.loaded_image_clock_unavailable",
                "process.loaded_image_attempt_overflow",
            ] {
                assert!(!loaded_image_observation_pending(code));
            }
            let mut progress = LoadedImageProgress::default();
            for code in [
                "process.loaded_image_open",
                "process.loaded_image_differs_from_pinned_file",
            ] {
                progress.begin_attempt().unwrap();
                progress.record_pending(code);
            }
            progress.begin_attempt().unwrap();
            assert_eq!((progress.attempts, progress.pending), (3, 2));
            assert_eq!(
                progress.first_pending_code,
                Some("process.loaded_image_open")
            );
            assert_eq!(
                progress.last_pending_code,
                Some("process.loaded_image_differs_from_pinned_file")
            );
            progress.attempts = u64::MAX;
            assert_eq!(
                progress.begin_attempt(),
                Err("process.loaded_image_attempt_overflow")
            );
        }

        #[test]
        fn original_execution_does_not_restart_clock_for_a_child_that_wont_read() {
            let directory = OriginalTestDirectory::new();
            let program = File::open("/bin/sleep").unwrap();
            let input = vec![b'x'; 96 * 1024];
            let started = Instant::now() - Duration::from_millis(200);
            let window = OriginalProcessWindow::new(
                started,
                started + Duration::from_millis(350),
                started + Duration::from_millis(850),
            )
            .unwrap();
            let limits = ProcessLimits {
                wall_ms: 350,
                ..original_limits()
            };
            let output = supervise_input_in_original_window(
                &program,
                &[OsString::from("10")],
                &directory.file(),
                limits,
                None,
                &original_watch(),
                OriginalProcessInput {
                    bytes: &input,
                    window,
                },
            )
            .unwrap();
            assert_eq!(
                output.process().receipt.stop,
                ProcessStop::WallLimit,
                "{output:?}"
            );
            assert!(output.observation.original_spawn_ns.unwrap() >= 200_000_000);
            assert!(output.observation.written_bytes.unwrap() < input.len());
            assert!(!output.transport_complete());
            assert!(output.process().pending_child.is_none(), "{output:?}");
        }

        #[test]
        fn original_loaded_image_verifier_rechecks_cancel_and_execution_before_input() {
            let directory = OriginalTestDirectory::new();
            let program = File::open("/bin/cat").unwrap();
            for cancel_in_verifier in [true, false] {
                let cancel = AtomicBool::new(false);
                let started = Instant::now();
                let execution = started + Duration::from_secs(1);
                let window = OriginalProcessWindow::new(
                    started,
                    execution,
                    started + Duration::from_millis(1500),
                )
                .unwrap();
                let mut calls = 0;
                let output = {
                    let mut verifier = |_: &File| {
                        calls += 1;
                        if cancel_in_verifier {
                            cancel.store(true, Ordering::Release);
                        } else {
                            // A real successful callback can finish after original
                            // E. It cannot renew E or authorize sending any input.
                            std::thread::sleep(
                                execution.saturating_duration_since(Instant::now())
                                    + Duration::from_millis(10),
                            );
                        }
                        Ok(())
                    };
                    supervise_verified_input_in_original_window(
                        &program,
                        &[],
                        &directory.file(),
                        ProcessLimits {
                            wall_ms: 1000,
                            ..original_limits()
                        },
                        Some(&cancel),
                        &original_watch(),
                        OriginalProcessInput {
                            bytes: b"must-not-be-sent",
                            window,
                        },
                        &mut verifier,
                    )
                    .unwrap()
                };
                assert_eq!(calls, 1, "{output:?}");
                assert_eq!(output.observation().written_bytes(), Some(0));
                assert!(output.process().stdout.is_empty());
                assert!(!output.transport_complete());
                assert!(output.checked_loaded_image().is_err());
                assert_eq!(
                    output.process().receipt.stop,
                    if cancel_in_verifier {
                        ProcessStop::Cancelled
                    } else {
                        ProcessStop::WallLimit
                    }
                );
                assert_eq!(output.process().receipt.group_cleanup, CleanupStatus::Gone);
                assert!(output.process().pending_child.is_none(), "{output:?}");
            }
        }

        #[test]
        fn original_output_limit_and_cancel_do_not_certify_transport() {
            let directory = OriginalTestDirectory::new();
            let program = File::open("/bin/cat").unwrap();
            let input = vec![b'x'; 32 * 1024];
            let started = Instant::now();
            let window = OriginalProcessWindow::new(
                started,
                started + Duration::from_secs(5),
                started + Duration::from_millis(5500),
            )
            .unwrap();
            let limits = ProcessLimits {
                max_output_bytes: 16,
                ..original_limits()
            };
            let output = supervise_input_in_original_window(
                &program,
                &[],
                &directory.file(),
                limits,
                None,
                &original_watch(),
                OriginalProcessInput {
                    bytes: &input,
                    window,
                },
            )
            .unwrap();
            assert_eq!(output.process().receipt.stop, ProcessStop::OutputLimit);
            assert!(!output.transport_complete());
            let cancelled = AtomicBool::new(true);
            assert!(
                supervise_input_in_original_window(
                    &program,
                    &[],
                    &directory.file(),
                    original_limits(),
                    Some(&cancelled),
                    &original_watch(),
                    OriginalProcessInput {
                        bytes: &input,
                        window
                    },
                )
                .is_err()
            );
        }

        #[test]
        fn source_profile_forwarding_is_one_exact_opt_in_value() {
            assert_eq!(source_profile_environment(None), None);
            for value in ["", "0", "true", "1 ", "--profile"] {
                assert_eq!(source_profile_environment(Some(value.into())), None);
            }
            assert_eq!(
                source_profile_environment(Some("1".into())),
                Some(("RZ_NATIVE_SOURCE_PROFILE", "1"))
            );
        }

        #[test]
        fn retired_proc_group_sentinel_is_not_a_live_group_or_parse_failure() {
            assert_eq!(parse_proc_group(b"12 (rz) X 1 -1 12\n").unwrap(), None);
            assert_eq!(
                parse_proc_group(b"12 (rz )\xff) S 1 42 12\n").unwrap(),
                Some(42)
            );
            assert_eq!(parse_proc_group(b"12 (rz) S 1 0 12\n").unwrap(), Some(0));
            for invalid in [
                b"12 (rz) S 1 -2 12\n".as_slice(),
                b"12 (rz) S 1 invalid 12\n",
                b"12 (rz) S 1 2147483648 12\n",
                b"12 (rz) S 1\n",
                b"12 (rz) S 1 \xff 12\n",
                b"",
            ] {
                assert!(parse_proc_group(invalid).is_err());
            }
        }

        #[test]
        fn proc_snapshot_during_owned_process_churn() {
            let mut child = Command::new("/bin/sh")
                .args([
                    "-c",
                    "i=0; while [ $i -lt 3000 ]; do /bin/true; i=$((i+1)); done",
                ])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .process_group(0)
                .spawn()
                .unwrap();
            let pid = child.id();
            let start = Instant::now();
            let mut failure = None;
            while start.elapsed() < Duration::from_secs(3) && child.try_wait().unwrap().is_none() {
                if let Err(error) = group_snapshot(pid, pid) {
                    failure = Some(error);
                    break;
                }
            }
            if child.try_wait().unwrap().is_none() {
                assert!(matches!(
                    killpg(Pid::from_raw(i32::try_from(pid).unwrap()), Signal::SIGKILL),
                    Ok(()) | Err(Errno::ESRCH)
                ));
                child.wait().unwrap();
            }
            assert!(failure.is_none(), "{failure:?}");
        }

        fn externally_reaped_child() -> std::process::Child {
            let child = Command::new("/bin/true")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .process_group(0)
                .spawn()
                .unwrap();
            let pid = Pid::from_raw(i32::try_from(child.id()).unwrap());
            // Consume only this exact test-owned child, without a global child
            // reaper. Child has not cached the status, so its later wait hits
            // the real OS ECHILD boundary that a foreign reaper would create.
            assert_eq!(
                waitid(Id::Pid(pid), WaitPidFlag::WEXITED).unwrap(),
                WaitStatus::Exited(pid, 0)
            );
            child
        }

        fn receipt(pid: u32, cleanup: CleanupStatus) -> ProcessReceipt {
            ProcessReceipt {
                supervisor_version: 1,
                pid,
                elapsed_ns: 0,
                cleanup_elapsed_ns: None,
                cleanup_timing_unavailable_reason: None,
                stop: ProcessStop::Exited,
                exit_code: Some(0),
                exit_signal: None,
                group_cleanup: cleanup,
                stdout_bytes: 0,
                stderr_bytes: 0,
                observed_output_bytes: 0,
                descendant_cleanup_required: false,
                child_limit_enforcement: "process_group_snapshot".into(),
                watched_artifact_bytes: None,
                artifact_limit_enforcement: None,
                errors: Vec::new(),
            }
        }

        #[test]
        fn historical_process_receipts_keep_cleanup_timing_unknown() {
            let receipt = receipt(123, CleanupStatus::Gone);
            let historical = serde_json::to_value(receipt).unwrap();
            assert!(historical.get("cleanup_elapsed_ns").is_none());
            assert!(
                historical
                    .get("cleanup_timing_unavailable_reason")
                    .is_none()
            );
            let decoded: ProcessReceipt = serde_json::from_value(historical).unwrap();
            assert!(decoded.cleanup_elapsed_ns.is_none());
            assert!(decoded.cleanup_timing_unavailable_reason.is_none());
        }

        #[test]
        fn tree_observation_ownership_loss_keeps_original_handle_without_reaping() {
            for tree in [false, true] {
                let child = externally_reaped_child();
                let pid = child.id();
                let mut receipt = receipt(pid, CleanupStatus::Unverified);
                receipt.errors = vec![
                    "process.wait_observation".into(),
                    "process.ownership_lost".into(),
                ];
                let mut stop = Some(ProcessStop::IoFailure);
                let pending = finish_child(child, &mut receipt, &mut stop, false, true, tree);
                record_cleanup_timing(&mut receipt, None, Instant::now(), true, pending.is_some());
                assert!(receipt.cleanup_elapsed_ns.is_none());
                assert_eq!(
                    receipt.cleanup_timing_unavailable_reason.as_deref(),
                    Some("process_cleanup_ownership_lost_completion_unobserved")
                );
                assert_eq!(
                    pending.as_ref().map(std::process::Child::id),
                    tree.then_some(pid)
                );
                assert_eq!(receipt.group_cleanup, CleanupStatus::Unverified);
                assert_eq!(stop, Some(ProcessStop::IoFailure));
                assert!(
                    receipt
                        .errors
                        .iter()
                        .any(|code| code == "process.ownership_lost")
                );
                // Ownership loss is evidence for quarantine, never permission
                // to issue a numeric PID wait/signal from this returned handle.
                assert!(!receipt.errors.iter().any(|code| code == "process.reap"));
            }
        }

        #[test]
        fn tree_reap_failure_keeps_original_handle_and_invalidates_cleanup() {
            for tree in [false, true] {
                let child = externally_reaped_child();
                let pid = child.id();
                let mut receipt = receipt(pid, CleanupStatus::Gone);
                let mut stop = Some(ProcessStop::Exited);
                let pending = finish_child(child, &mut receipt, &mut stop, true, false, tree);
                record_cleanup_timing(
                    &mut receipt,
                    Some(Instant::now()),
                    Instant::now(),
                    false,
                    pending.is_some(),
                );
                assert!(receipt.cleanup_elapsed_ns.is_none());
                assert_eq!(
                    receipt.cleanup_timing_unavailable_reason.as_deref(),
                    Some(if tree {
                        "process_cleanup_ownership_lost_completion_unobserved"
                    } else {
                        "process_cleanup_reap_failed"
                    })
                );
                assert_eq!(
                    pending.as_ref().map(std::process::Child::id),
                    tree.then_some(pid)
                );
                assert!(receipt.errors.iter().any(|code| code == "process.reap"));
                assert_eq!(
                    receipt.group_cleanup,
                    if tree {
                        CleanupStatus::Unverified
                    } else {
                        CleanupStatus::Gone
                    }
                );
                assert_eq!(
                    stop,
                    Some(if tree {
                        ProcessStop::IoFailure
                    } else {
                        ProcessStop::Exited
                    })
                );
                assert_eq!(
                    receipt
                        .errors
                        .iter()
                        .any(|code| code == "process.ownership_lost"),
                    tree
                );
            }
        }
    }
}
