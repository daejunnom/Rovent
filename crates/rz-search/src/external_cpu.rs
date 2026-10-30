//! Startup-selected external UCI CPU_R checker, independent of arena opponents.
//!
//! External cp/mate/bounds and reported selective depth remain external reports.
//! A bestmove does not certify minimax completion or a Rules game outcome. One
//! owner issues one active `go`; cancellation is followed by bounded stop/drain.
//! Linux execution uses a pinned native ELF, an owned process group, nonblocking
//! pipes and waitid(WNOWAIT). Windows remains explicitly unsupported here.

use crate::cpu::{CpuLimits, CpuResumeToken};
use crate::cpu_checker::*;
use rz_position::{BoardMove, HistoryOrigin, PlayStatus, Position};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ffi::OsString;
#[cfg(target_os = "linux")]
use std::fs::File;
#[cfg(target_os = "linux")]
use std::io::Read;
use std::path::PathBuf;
#[cfg(target_os = "linux")]
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub mod adaptive;
pub mod collection_teacher;
pub mod process_scope;
mod profiled_researcher;
pub mod profiled_worker;
pub mod profiles;
pub mod worker;

pub use adaptive::*;
pub use profiled_researcher::ProfiledResearcherAdmission;
pub use profiled_worker::*;
pub use profiles::*;
#[cfg(all(test, target_os = "linux"))]
pub(crate) use tests::FIXTURE_LOCK as EXTERNAL_FIXTURE_LOCK;

pub const EXTERNAL_UCI_CHECKER_VERSION: &str = "rz-external-uci-checker/0.1";
pub const EXTERNAL_UCI_SCORE_SEMANTICS: &str = "uci-native-cp-or-signed-mate-moves;root-side-to-move;reported-bounds;selectivity-unknown;no-cp-wdl-calibration;no-rules-proof";

#[cfg(target_os = "linux")]
#[derive(Debug, Default)]
struct SealedExecutionCounters {
    sealed_spawn_returned: std::sync::atomic::AtomicU64,
    original_fallback_attempted: std::sync::atomic::AtomicU64,
}

/// Lightweight observation which does not retain the executable or original FD.
#[cfg(target_os = "linux")]
#[derive(Clone, Debug)]
pub struct SealedExecutionProbe {
    counters: Arc<SealedExecutionCounters>,
}

#[cfg(target_os = "linux")]
impl SealedExecutionProbe {
    pub fn execution_observations(&self) -> (u64, u64) {
        (self.counters.sealed_spawn_returned.load(Ordering::Acquire),
         self.counters.original_fallback_attempted.load(Ordering::Acquire))
    }
}

/// A byte-verified, immutable executable prepared before a search root starts.
/// The private descriptor and hash remain coupled; constructing this token does
/// not grant process, root, or completion authority.
#[derive(Clone, Debug)]
pub struct VerifiedSealedExecutable {
    #[cfg(target_os = "linux")]
    file: Arc<File>,
    #[cfg(target_os = "linux")]
    original: Arc<File>,
    #[cfg(target_os = "linux")]
    sha256: String,
    #[cfg(target_os = "linux")]
    counters: Arc<SealedExecutionCounters>,
}

#[cfg(target_os = "linux")]
impl VerifiedSealedExecutable {
    /// Copy the already admitted original into a bounded anonymous file, verify
    /// the copied bytes and seal writes/size changes before returning its owner.
    /// `check` is the original preparation cancellation/deadline check.
    pub fn prepare(
        original: &File,
        expected_sha256: &str,
        mut check: impl FnMut() -> Result<(), CheckerError>,
    ) -> Result<Self, CheckerError> {
        use nix::fcntl::{FcntlArg, SealFlag, fcntl};
        use nix::sys::memfd::{MFdFlags, memfd_create};
        use std::io::Write;
        use std::os::unix::fs::{FileExt, MetadataExt};

        if expected_sha256.len() != 64
            || !expected_sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(error("sealed_prepare", "invalid_sha256"));
        }
        check()?;
        let before = original
            .metadata()
            .map_err(|_| error("sealed_prepare", "original_metadata"))?;
        if !before.is_file()
            || before.len() < 4
            || before.len() > 256 * 1024 * 1024
            || before.mode() & 0o111 == 0
        {
            return Err(error("sealed_prepare", "original_kind_or_size"));
        }
        let fd = memfd_create(
            "rovent-teacher",
            MFdFlags::MFD_CLOEXEC | MFdFlags::MFD_ALLOW_SEALING,
        )
        .map_err(|_| error("sealed_prepare", "memfd_create"))?;
        let mut file = File::from(fd);
        if !file.metadata().is_ok_and(|metadata| metadata.mode() & 0o111 != 0) {
            return Err(error("sealed_prepare", "memfd_not_executable"));
        }
        let mut total = 0u64;
        let mut buffer = [0u8; 65_536];
        loop {
            check()?;
            let count = original
                .read_at(&mut buffer, total)
                .map_err(|_| error("sealed_prepare", "original_read"))?;
            if count == 0 {
                break;
            }
            if total == 0 && (count < 4 || buffer[..4] != *b"\x7fELF") {
                return Err(error("sealed_prepare", "native_elf_required"));
            }
            total = total
                .checked_add(count as u64)
                .ok_or_else(|| error("sealed_prepare", "length_overflow"))?;
            if total > before.len() {
                return Err(error("sealed_prepare", "original_changed"));
            }
            file.write_all(&buffer[..count])
                .map_err(|_| error("sealed_prepare", "copy"))?;
        }
        let after = original
            .metadata()
            .map_err(|_| error("sealed_prepare", "original_recheck"))?;
        if total != before.len()
            || after.len() != before.len()
            || after.dev() != before.dev()
            || after.ino() != before.ino()
            || after.mtime() != before.mtime()
            || after.mtime_nsec() != before.mtime_nsec()
            || after.ctime() != before.ctime()
            || after.ctime_nsec() != before.ctime_nsec()
        {
            return Err(error("sealed_prepare", "original_changed"));
        }
        check()?;
        let seals = SealFlag::F_SEAL_WRITE
            | SealFlag::F_SEAL_GROW
            | SealFlag::F_SEAL_SHRINK
            | SealFlag::F_SEAL_SEAL;
        fcntl(&file, FcntlArg::F_ADD_SEALS(seals)).map_err(|_| error("sealed_prepare", "seal"))?;
        let present = fcntl(&file, FcntlArg::F_GET_SEALS)
            .map_err(|_| error("sealed_prepare", "seal_readback"))?;
        if (present & seals.bits()) != seals.bits() {
            return Err(error("sealed_prepare", "seal_incomplete"));
        }
        // Hash the sealed file itself, not the copy buffer: only this binds the
        // proof to the immutable bytes that the process will actually execute.
        let mut hash = Sha256::new();
        let mut verified = 0u64;
        while verified < total {
            check()?;
            let count = file
                .read_at(&mut buffer, verified)
                .map_err(|_| error("sealed_prepare", "sealed_read"))?;
            if count == 0 {
                return Err(error("sealed_prepare", "sealed_short_read"));
            }
            verified = verified
                .checked_add(count as u64)
                .ok_or_else(|| error("sealed_prepare", "length_overflow"))?;
            if verified > total {
                return Err(error("sealed_prepare", "sealed_length"));
            }
            hash.update(&buffer[..count]);
        }
        if format!("{:x}", hash.finalize()) != expected_sha256 {
            return Err(error("sealed_prepare", "hash_mismatch"));
        }
        check()?;
        let original = original.try_clone().map_err(|_| error("sealed_prepare", "original_clone"))?;
        Ok(Self {
            file: Arc::new(file),
            original: Arc::new(original),
            sha256: expected_sha256.to_owned(),
            counters: Arc::new(SealedExecutionCounters::default()),
        })
    }

    /// Cumulative source-slot observations, not process completion evidence.
    pub fn execution_observations(&self) -> (u64, u64) {
        (self.counters.sealed_spawn_returned.load(Ordering::Acquire),
         self.counters.original_fallback_attempted.load(Ordering::Acquire))
    }

    /// Retain counters without extending the sealed file's memory lifetime.
    pub fn execution_probe(&self) -> SealedExecutionProbe {
        SealedExecutionProbe { counters: Arc::clone(&self.counters) }
    }
}

/// Retained executable/cwd capabilities for a private V5 snapshot. Path names
/// remain declarations in ExternalCpuConfig; an actual start uses these held
/// objects and repeats byte/ELF verification before spawning. This pin is not a
/// startup, readiness or teacher completion authority.
#[derive(Clone, Debug)]
pub struct ExternalLaunchPin {
    #[cfg(target_os = "linux")]
    executable: Arc<File>,
    #[cfg(target_os = "linux")]
    sealed_sha256: Option<String>,
    #[cfg(target_os = "linux")]
    fallback_executable: Option<Arc<File>>,
    #[cfg(target_os = "linux")]
    sealed_counters: Option<Arc<SealedExecutionCounters>>,
    #[cfg(target_os = "linux")]
    directory: Arc<File>,
    #[cfg(not(target_os = "linux"))]
    _unsupported: (),
    #[cfg(target_os = "linux")]
    collection_process_permit: Option<process_scope::CollectionProcessPermit>,
}

impl ExternalLaunchPin {
    #[cfg(target_os = "linux")]
    pub fn from_owned_descriptors(executable: File, directory: File) -> Result<Self, CheckerError> {
        use std::os::unix::fs::MetadataExt;
        let binary = executable
            .metadata()
            .map_err(|_| error("launch_pin", "executable_metadata"))?;
        let cwd = directory
            .metadata()
            .map_err(|_| error("launch_pin", "directory_metadata"))?;
        if !binary.is_file()
            || binary.len() == 0
            || binary.len() > 256 * 1024 * 1024
            || binary.mode() & 0o111 == 0
            || !cwd.is_dir()
        {
            return Err(error("launch_pin", "descriptor_kind_or_size"));
        }
        Ok(Self {
            executable: Arc::new(executable),
            sealed_sha256: None,
            fallback_executable: None,
            sealed_counters: None,
            directory: Arc::new(directory),
            collection_process_permit: None,
        })
    }
    /// Bind the verified sealed file to the same retained working directory.
    /// Process scope and root admission are still required at actual startup.
    #[cfg(target_os = "linux")]
    pub fn from_verified_sealed(
        executable: VerifiedSealedExecutable,
        directory: File,
    ) -> Result<Self, CheckerError> {
        if !directory.metadata().is_ok_and(|m| m.is_dir()) {
            return Err(error("launch_pin", "directory_metadata"));
        }
        Ok(Self {
            executable: executable.file,
            sealed_sha256: Some(executable.sha256),
            fallback_executable: Some(executable.original),
            sealed_counters: Some(executable.counters),
            directory: Arc::new(directory),
            collection_process_permit: None,
        })
    }
    /// Only the Collection worker can attach its exact registered member.
    /// Public legacy pinned checkers remain on the original singleton gate.
    pub(crate) fn with_collection_permit(
        self,
        permit: process_scope::CollectionProcessPermit,
    ) -> Self {
        #[cfg(target_os = "linux")]
        {
            let mut pin = self;
            pin.collection_process_permit = Some(permit);
            pin
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = permit;
            self
        }
    }
}

/// Files are opened, streamed and verified before spawn. Identity declarations
/// do not prove training, license rights, an applied option or actual precision.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExternalCpuConfig {
    pub program: PathBuf,
    pub arguments: Vec<OsString>,
    pub working_directory: PathBuf,
    pub identity: ExternalCheckerIdentity,
    pub expected_uci_name: Option<String>,
    pub max_depth: u16,
    pub max_prefix_plies: usize,
    pub handshake_timeout: Duration,
    pub max_task_wall_time: Duration,
    /// Reserved inside every task for stop, bestmove drain and the ready barrier.
    pub stop_grace: Duration,
    pub shutdown_grace: Duration,
    /// Combined stdout/stderr bytes over this process's whole lifetime.
    pub max_output_bytes: usize,
    pub max_line_bytes: usize,
}

impl ExternalCpuConfig {
    pub fn validate(&self) -> Result<(), CheckerError> {
        self.identity.validate()?;
        if self.identity.adapter_semantics != EXTERNAL_UCI_SCORE_SEMANTICS {
            return Err(CheckerError::Invalid("external score semantics mismatch"));
        }
        if !self.program.is_absolute() || !self.working_directory.is_absolute() {
            return Err(CheckerError::Invalid("external paths must be absolute"));
        }
        if self.max_depth == 0 || self.max_depth > 64 || self.max_prefix_plies > 4096 {
            return Err(CheckerError::Invalid("external depth/history cap"));
        }
        if self.arguments.len() > 64
            || self
                .arguments
                .iter()
                .try_fold(0usize, |total, arg| {
                    total.checked_add(arg.as_encoded_bytes().len())
                })
                .is_none_or(|bytes| bytes > 16_384)
            || self
                .arguments
                .iter()
                .any(|arg| arg.as_encoded_bytes().contains(&0))
        {
            return Err(CheckerError::Invalid("external argument budget"));
        }
        if arguments_sha256(&self.arguments) != self.identity.launch_arguments_sha256 {
            return Err(CheckerError::Invalid("external argument identity mismatch"));
        }
        if !self.identity.assets.is_empty() {
            return Err(CheckerError::Unsupported(
                "external separately loaded assets require a pinned launch profile",
            ));
        }
        for time in [
            self.handshake_timeout,
            self.max_task_wall_time,
            self.stop_grace,
            self.shutdown_grace,
        ] {
            if time.is_zero() || time > Duration::from_secs(180) {
                return Err(CheckerError::Invalid("external finite wall budget"));
            }
        }
        if self.max_output_bytes == 0
            || self.max_output_bytes > 16 * 1024 * 1024
            || self.max_line_bytes == 0
            || self.max_line_bytes > 65_536
            || self.max_line_bytes > self.max_output_bytes
        {
            return Err(CheckerError::Invalid("external output budget"));
        }
        if self
            .expected_uci_name
            .as_ref()
            .is_some_and(|name| !bounded_text(name, 256))
        {
            return Err(CheckerError::Invalid("external expected name"));
        }
        Ok(())
    }
}

/// Domain-separated byte identity; argument boundaries are preserved.
pub fn arguments_sha256(arguments: &[OsString]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"rz-external-uci-arguments/1\0");
    hash.update((arguments.len() as u64).to_le_bytes());
    for arg in arguments {
        hash.update((arg.as_encoded_bytes().len() as u64).to_le_bytes());
        hash.update(arg.as_encoded_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn bounded_text(text: &String, limit: usize) -> bool {
    !text.trim().is_empty()
        && text.len() <= limit
        && text.capacity() <= limit
        && !text.chars().any(char::is_control)
}

fn error(stage: &'static str, code: &'static str) -> CheckerError {
    CheckerError::External { stage, code }
}

fn bounded_deadline(
    start: Instant,
    duration: Duration,
    supplied: Option<Instant>,
) -> Result<Instant, CheckerError> {
    let local = start
        .checked_add(duration)
        .ok_or(CheckerError::Invalid("external deadline overflow"))?;
    Ok(supplied.map_or(local, |deadline| deadline.min(local)))
}

/// The caller's task wall includes readiness, analysis and physical stop/drain.
/// A grace is a reserved portion of that wall, never a new window after expiry.
#[derive(Clone, Copy, Debug)]
struct ExternalTaskBudget {
    analysis_deadline: Instant,
    task_deadline: Instant,
    stop_reserve: Duration,
}

impl ExternalTaskBudget {
    fn new(
        start: Instant,
        task_deadline: Instant,
        stop_reserve: Duration,
    ) -> Result<Self, CheckerError> {
        let analysis_deadline = task_deadline
            .checked_sub(stop_reserve)
            .ok_or_else(|| error("admission", "insufficient_stop_reserve"))?;
        let budget = Self {
            analysis_deadline,
            task_deadline,
            stop_reserve,
        };
        budget.analysis_millis(start)?;
        Ok(budget)
    }

    fn analysis_millis(self, now: Instant) -> Result<u128, CheckerError> {
        let remaining = self
            .analysis_deadline
            .saturating_duration_since(now)
            .as_millis();
        if remaining == 0 {
            return Err(error("admission", "insufficient_stop_reserve"));
        }
        Ok(remaining)
    }

    fn drain_deadline(self, observed: Instant) -> Result<Instant, CheckerError> {
        bounded_deadline(observed, self.stop_reserve, Some(self.task_deadline))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum OptionKind {
    Spin { min: i64, max: i64 },
    Check,
    Combo(Vec<String>),
    String,
    Button,
}

fn parse_option(line: &str) -> Result<(String, OptionKind), CheckerError> {
    let (name, declaration) = line
        .strip_prefix("option name ")
        .and_then(|body| body.split_once(" type "))
        .ok_or_else(|| error("handshake", "malformed_option"))?;
    if name.is_empty() || name.len() > 256 || name.chars().any(char::is_control) {
        return Err(error("handshake", "invalid_option_name"));
    }
    let tokens: Vec<&str> = declaration.split_ascii_whitespace().collect();
    let kind = match tokens.first().copied() {
        Some("spin") => {
            let field = |key: &str| -> Result<i64, CheckerError> {
                let index = tokens
                    .iter()
                    .position(|&word| word == key)
                    .ok_or_else(|| error("handshake", "spin_range_missing"))?;
                tokens
                    .get(index + 1)
                    .and_then(|value| value.parse().ok())
                    .ok_or_else(|| error("handshake", "spin_range_invalid"))
            };
            let min = field("min")?;
            let max = field("max")?;
            if min > max {
                return Err(error("handshake", "spin_range_invalid"));
            }
            OptionKind::Spin { min, max }
        }
        Some("check") => OptionKind::Check,
        Some("combo") => {
            let mut values = Vec::new();
            let mut at = 0;
            while at < tokens.len() {
                if tokens[at] == "var" {
                    let begin = at + 1;
                    at = begin;
                    while at < tokens.len() && tokens[at] != "var" {
                        at += 1;
                    }
                    if begin == at {
                        return Err(error("handshake", "empty_combo_value"));
                    }
                    values.push(tokens[begin..at].join(" "));
                } else {
                    at += 1;
                }
            }
            if values.is_empty() || values.len() > 64 {
                return Err(error("handshake", "combo_values_invalid"));
            }
            OptionKind::Combo(values)
        }
        Some("string") => OptionKind::String,
        Some("button") => OptionKind::Button,
        _ => return Err(error("handshake", "unsupported_option_type")),
    };
    Ok((name.to_owned(), kind))
}

fn validate_option(kind: &OptionKind, value: &str) -> bool {
    match kind {
        OptionKind::Spin { min, max } => value
            .parse::<i64>()
            .is_ok_and(|number| number >= *min && number <= *max),
        OptionKind::Check => matches!(value, "true" | "false"),
        OptionKind::Combo(values) => values.iter().any(|candidate| candidate == value),
        OptionKind::String => value.len() <= 4096 && !value.chars().any(char::is_control),
        OptionKind::Button => value.is_empty(),
    }
}

#[derive(Clone, Debug, Default)]
struct InfoFrame {
    depth: Option<u16>,
    seldepth: Option<u16>,
    nodes: Option<u64>,
    score: Option<ExternalRawScore>,
    bound: Option<ExternalBound>,
    wdl: Option<[u16; 3]>,
    pv: Option<Vec<BoardMove>>,
    multipv: u16,
}

/// A conditional teacher opinion from one complete, legal info tuple. Rank is
/// not a value label and external CP/mate remains raw. Collection must seal the
/// exact question/anchor and require completed stop/readiness before use.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExternalPolicyOpinion {
    pub rank: u16,
    pub depth: u16,
    pub seldepth: Option<u16>,
    pub pv: Vec<BoardMove>,
    pub score: ExternalRawScore,
    pub bound: ExternalBound,
    pub reported_wdl: Option<[u16; 3]>,
}

#[derive(Clone, Debug)]
pub struct ExternalCooperativeReport {
    pub report: ExternalCheckerReport,
    pub opinions: Vec<ExternalPolicyOpinion>,
    pub stop_control: Option<ExternalControlObservation>,
    /// Unknown by design: reported UCI nodes are not strict native credit.
    pub strict_native_nodes: Option<u64>,
    pub reported_node_request: u64,
    pub requested_multipv: u16,
    /// UCI readyok acknowledges readiness, not option-value readback.
    pub option_application_observed: Option<bool>,
    pub reported_node_overshoot: bool,
    pub readiness_barrier_completed: bool,
    /// Coherent complete rank-set opinion only. This remains a conditional
    /// policy target, never a WDL label or a minimax completion assertion.
    pub policy_target_known: bool,
    pub go_issue: ExternalGoIssue,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExternalGoIssue {
    NotIssued,
    /// A write may have been partially delivered before failure.
    Unknown,
    Issued,
}

impl ExternalCooperativeReport {
    /// Sparse conditional teacher opinion: softmax(-rank). Raw CP/mate/WDL
    /// reports are not consulted and no value/utility target is manufactured.
    pub fn conditional_rank_policy(&self) -> Option<Vec<(BoardMove, f32)>> {
        if !self.policy_target_known || self.opinions.is_empty() || self.opinions.len() > 4 {
            return None;
        }
        let sum = self
            .opinions
            .iter()
            .map(|opinion| (-(f64::from(opinion.rank))).exp())
            .sum::<f64>();
        if !sum.is_finite() || sum <= 0.0 {
            return None;
        }
        self.opinions
            .iter()
            .map(|opinion| {
                Some((
                    *opinion.pv.first()?,
                    ((-f64::from(opinion.rank)).exp() / sum) as f32,
                ))
            })
            .collect()
    }
}

/// Each accepted score/PV/depth tuple comes from one complete info line. Sparse
/// lines update observed work only; they cannot stitch a score onto a later PV.
fn parse_info(line: &str, max_pv: usize) -> Result<InfoFrame, CheckerError> {
    let words: Vec<&str> = line.split_ascii_whitespace().collect();
    if words.first() != Some(&"info") {
        return Err(error("analysis", "not_info"));
    }
    let mut frame = InfoFrame {
        multipv: 1,
        ..InfoFrame::default()
    };
    let mut at = 1;
    while at < words.len() {
        match words[at] {
            "string" => break,
            "depth" | "seldepth" | "multipv" => {
                let number = words
                    .get(at + 1)
                    .and_then(|word| word.parse::<u16>().ok())
                    .ok_or_else(|| error("analysis", "invalid_depth_field"))?;
                match words[at] {
                    "depth" => frame.depth = Some(number),
                    "seldepth" => frame.seldepth = Some(number),
                    _ => frame.multipv = number,
                }
                at += 2;
            }
            "nodes" => {
                frame.nodes = Some(
                    words
                        .get(at + 1)
                        .and_then(|word| word.parse().ok())
                        .ok_or_else(|| error("analysis", "invalid_nodes"))?,
                );
                at += 2;
            }
            "score" => {
                let value = words
                    .get(at + 2)
                    .and_then(|word| word.parse::<i32>().ok())
                    .ok_or_else(|| error("analysis", "invalid_score"))?;
                frame.score = Some(match words.get(at + 1).copied() {
                    Some("cp") => ExternalRawScore::Centipawns(value),
                    Some("mate") => ExternalRawScore::MateMoves(value),
                    _ => return Err(error("analysis", "unsupported_score_unit")),
                });
                frame.bound = Some(ExternalBound::ExactReported);
                at += 3;
                if let Some(&kind) = words.get(at) {
                    if matches!(kind, "lowerbound" | "upperbound") {
                        frame.bound = Some(if kind == "lowerbound" {
                            ExternalBound::Lower
                        } else {
                            ExternalBound::Upper
                        });
                        at += 1;
                    }
                }
            }
            "wdl" => {
                let mut values = [0u16; 3];
                for (offset, value) in values.iter_mut().enumerate() {
                    *value = words
                        .get(at + 1 + offset)
                        .and_then(|word| word.parse().ok())
                        .ok_or_else(|| error("analysis", "invalid_wdl"))?;
                }
                if values.iter().map(|&value| u32::from(value)).sum::<u32>() != 1000 {
                    return Err(error("analysis", "invalid_wdl_sum"));
                }
                frame.wdl = Some(values);
                at += 4;
            }
            "pv" => {
                if words.len() - at - 1 > max_pv {
                    return Err(error("analysis", "pv_budget"));
                }
                let pv = words[at + 1..]
                    .iter()
                    .map(|word| {
                        BoardMove::from_uci(word).map_err(|_| error("analysis", "invalid_pv_move"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if pv.is_empty() {
                    return Err(error("analysis", "empty_pv"));
                }
                frame.pv = Some(pv);
                break;
            }
            // These fields are preserved in the bounded transcript. Unknown
            // extension fields do not supply score/depth/coverage observations.
            _ => at += 1,
        }
    }
    if frame.multipv == 0 {
        return Err(error("analysis", "invalid_multipv"));
    }
    Ok(frame)
}

fn validate_pv(
    position: &Position,
    pv: &[BoardMove],
    roots: Option<&[BoardMove]>,
) -> Result<(), CheckerError> {
    if roots.is_some_and(|mask| pv.first().is_some_and(|mv| !mask.contains(mv))) {
        return Err(error("analysis", "root_mask_escape"));
    }
    let mut current = position.clone();
    for &mv in pv {
        let view = current.ordered_legal_moves();
        if current
            .play_status_from_view(&view)
            .map_err(|_| error("analysis", "rules_classification"))?
            != PlayStatus::Ongoing
        {
            return Err(error("analysis", "pv_after_rules_terminal"));
        }
        current
            .make_from_view(&view, mv)
            .map_err(|_| error("analysis", "illegal_pv"))?;
    }
    Ok(())
}

pub struct ExternalUciCpuChecker {
    config: ExternalCpuConfig,
    identity: CheckerIdentity,
    conditions: String,
    observed_uci: ExternalUciIdentity,
    owner: Option<process::OwnedProcess>,
    last_attempt: Option<CheckerAttempt>,
    request_id: u64,
    active: bool,
    failed: bool,
    failed_output: Option<(Vec<u8>, Vec<u8>)>,
    advertised_options: BTreeMap<String, OptionKind>,
    cooperative_opinions: Vec<ExternalPolicyOpinion>,
    // Keep only actual exact info tuples. A later aspiration bound must not
    // erase the last complete MultiPV iteration.
    cooperative_exact_candidates: Vec<ExternalPolicyOpinion>,
    cooperative_exact_completed: Vec<ExternalPolicyOpinion>,
    stop_control: Option<ExternalControlObservation>,
    go_issue: ExternalGoIssue,
    startup_multipv: Option<u16>,
    last_requested_multipv: Option<u16>,
    launch_pin: Option<ExternalLaunchPin>,
    engine_profile: Option<(ExternalEngineProfile, ExternalEnginePurpose)>,
    last_issued_constraint: Option<ExternalGoConstraint>,
    last_issued_go: Option<String>,
}

impl ExternalUciCpuChecker {
    /// Engine/purpose admission for new profiles. Legacy `create` remains the
    /// unchanged compatibility path; an arena declaration cannot start this
    /// research checker or inherit its root/combined-limit capabilities.
    pub fn create_profiled(
        config: ExternalCpuConfig,
        profile: ExternalEngineProfile,
        purpose: ExternalEnginePurpose,
    ) -> Result<Self, CheckerError> {
        if purpose == ExternalEnginePurpose::ArenaOpponent {
            return Err(CheckerError::Unsupported(
                "arena opponent must use the arena process owner",
            ));
        }
        profile.validate_identity(
            purpose,
            &config.identity,
            config.expected_uci_name.as_deref().unwrap_or_default(),
        )?;
        let mut checker = Self::create(config)?;
        checker.engine_profile = Some((profile, purpose));
        checker.conditions = format!(
            "{};score:{};profile:{profile:?};purpose:{purpose:?};depth-cap:{};prefix-cap:{};resume:unsupported;root-searchmoves:{};combined-go:{};explicit-query-constraints;task-wall:includes-stop-and-ready;single-active;game:ucinewgame+ready;output-cap:{};line-cap:{};task-ms:{};stop-ms:{};shutdown-ms:{}",
            EXTERNAL_UCI_CHECKER_VERSION,
            EXTERNAL_UCI_SCORE_SEMANTICS,
            checker.config.max_depth,
            checker.config.max_prefix_plies,
            profile.capabilities().searchmoves,
            profile.capabilities().combined_depth_nodes_movetime,
            checker.config.max_output_bytes,
            checker.config.max_line_bytes,
            checker.config.max_task_wall_time.as_millis(),
            checker.config.stop_grace.as_millis(),
            checker.config.shutdown_grace.as_millis(),
        );
        Ok(checker)
    }

    /// Create an unstarted owner so failed startup evidence remains available.
    /// The caller must use `start` and inspect `last_attempt` on failure.
    pub fn create(config: ExternalCpuConfig) -> Result<Self, CheckerError> {
        config.validate()?;
        let conditions = format!(
            "{};score:{};depth-cap:{};prefix-cap:{};resume:unsupported;root:searchmoves;go:depth+reported-node-limit+finite-wall;task-wall:includes-stop-and-ready;single-active;game:ucinewgame+ready;output-cap:{};line-cap:{};task-ms:{};stop-ms:{};shutdown-ms:{}",
            EXTERNAL_UCI_CHECKER_VERSION,
            EXTERNAL_UCI_SCORE_SEMANTICS,
            config.max_depth,
            config.max_prefix_plies,
            config.max_output_bytes,
            config.max_line_bytes,
            config.max_task_wall_time.as_millis(),
            config.stop_grace.as_millis(),
            config.shutdown_grace.as_millis()
        );
        Ok(Self {
            identity: CheckerIdentity::ExternalUci(config.identity.clone()),
            config,
            conditions,
            observed_uci: ExternalUciIdentity {
                name: String::new(),
                author: None,
            },
            owner: None,
            last_attempt: None,
            request_id: 0,
            active: false,
            failed: false,
            failed_output: None,
            advertised_options: BTreeMap::new(),
            cooperative_opinions: Vec::new(),
            cooperative_exact_candidates: Vec::new(),
            cooperative_exact_completed: Vec::new(),
            stop_control: None,
            go_issue: ExternalGoIssue::NotIssued,
            startup_multipv: None,
            last_requested_multipv: None,
            launch_pin: None,
            engine_profile: None,
            last_issued_constraint: None,
            last_issued_go: None,
        })
    }

    /// Streamed executable validation, spawn and handshake are on the caller's
    /// finite startup clock. This object survives failure with evidence intact.
    pub fn start(&mut self, deadline: Instant, cancel: &AtomicBool) -> Result<(), CheckerError> {
        self.start_with_cleanup_fence(deadline, cancel, None)
    }

    /// The private worker can shorten physical cleanup while startup is in
    /// flight. The public synchronous start keeps its registered deadline.
    fn start_with_cleanup_fence(
        &mut self,
        deadline: Instant,
        cancel: &AtomicBool,
        cleanup_fence: Option<&dyn Fn() -> Option<Instant>>,
    ) -> Result<(), CheckerError> {
        self.last_attempt = None;
        if self.failed || self.owner.is_some() {
            return Err(error("startup", "already_started_or_failed"));
        }
        let start = Instant::now();
        let deadline = bounded_deadline(start, self.config.handshake_timeout, Some(deadline))?;
        if cancel.load(Ordering::Acquire) || start >= deadline {
            return Err(error("startup", "cancel_or_deadline"));
        }
        match process::OwnedProcess::spawn(&self.config, deadline, cancel, self.launch_pin.as_ref())
        {
            Ok(owner) => self.owner = Some(owner),
            Err(failure) => {
                self.failed = true;
                self.last_attempt = Some(CheckerAttempt {
                    work: CheckerWork::default(),
                    elapsed: start.elapsed(),
                    external: Some(ExternalAttemptEvidence {
                        request_id: 0,
                        partial_report: None,
                        process: failure.process,
                    }),
                });
                self.failed_output = Some((failure.stdout, failure.stderr));
                return Err(failure.error);
            }
        }
        let result = self.handshake(deadline, cancel);
        if result.is_err() {
            self.failed = true;
            let fence = cleanup_fence
                .and_then(|read| read())
                .map_or(deadline, |fence| fence.min(deadline));
            let cleanup =
                bounded_deadline(Instant::now(), self.config.shutdown_grace, Some(fence))?;
            let _ = self.shutdown(cleanup);
        }
        self.last_attempt = Some(CheckerAttempt {
            work: CheckerWork::default(),
            elapsed: start.elapsed(),
            external: Some(ExternalAttemptEvidence {
                request_id: 0,
                partial_report: None,
                process: self
                    .owner
                    .as_ref()
                    .map_or_else(empty_shutdown, |owner| owner.evidence()),
            }),
        });
        result
    }

    pub fn config(&self) -> &ExternalCpuConfig {
        &self.config
    }

    /// Transcript storage is bounded over the entire process lifetime. Absence
    /// of an option application response remains unknown, even after readyok.
    pub fn diagnostic_output(&self) -> (&[u8], &[u8]) {
        self.owner.as_ref().map_or_else(
            || {
                self.failed_output
                    .as_ref()
                    .map_or((&[][..], &[][..]), |(stdout, stderr)| {
                        (stdout.as_slice(), stderr.as_slice())
                    })
            },
            |owner| owner.output(),
        )
    }

    fn owner(&mut self) -> Result<&mut process::OwnedProcess, CheckerError> {
        self.owner
            .as_mut()
            .ok_or_else(|| error("process", "owner_closed"))
    }

    fn handshake(&mut self, deadline: Instant, cancel: &AtomicBool) -> Result<(), CheckerError> {
        self.owner()?.send("uci", deadline)?;
        let mut options = BTreeMap::new();
        loop {
            let line = self
                .owner()?
                .next_line(deadline, cancel)?
                .ok_or_else(|| error("handshake", "timeout_or_cancel"))?;
            if line == "uciok" {
                break;
            }
            if let Some(name) = line.strip_prefix("id name ") {
                if !self.observed_uci.name.is_empty() || name.is_empty() || name.len() > 256 {
                    return Err(error("handshake", "invalid_engine_name"));
                }
                self.observed_uci.name = name.to_owned();
            } else if let Some(author) = line.strip_prefix("id author ") {
                if self.observed_uci.author.is_some() || author.len() > 256 {
                    return Err(error("handshake", "invalid_engine_author"));
                }
                self.observed_uci.author = Some(author.to_owned());
            } else if line.starts_with("option ") {
                let (name, kind) = parse_option(&line)?;
                if name == "MultiPV" {
                    let words = line.split_ascii_whitespace().collect::<Vec<_>>();
                    let value = words
                        .windows(2)
                        .find(|words| words[0] == "default")
                        .and_then(|words| words[1].parse::<u16>().ok())
                        .filter(|value| *value != 0 && validate_option(&kind, &value.to_string()))
                        .ok_or_else(|| error("handshake", "invalid_multipv_default"))?;
                    self.startup_multipv = Some(value);
                    self.last_requested_multipv = Some(value);
                }
                if options.len() >= 256 || options.insert(name, kind).is_some() {
                    return Err(error("handshake", "duplicate_or_excess_options"));
                }
            } else if line.starts_with("bestmove") || line == "readyok" {
                return Err(error("handshake", "unexpected_response"));
            }
        }
        if self.observed_uci.name.is_empty()
            || self
                .config
                .expected_uci_name
                .as_ref()
                .is_some_and(|expected| *expected != self.observed_uci.name)
        {
            return Err(error("handshake", "engine_name_mismatch"));
        }
        for (name, value) in self.config.identity.options.clone() {
            let kind = options
                .get(&name)
                .ok_or_else(|| error("handshake", "unsupported_requested_option"))?;
            if !validate_option(kind, &value) {
                return Err(error("handshake", "requested_option_out_of_range"));
            }
            let command = if matches!(kind, OptionKind::Button) {
                format!("setoption name {name}")
            } else {
                format!("setoption name {name} value {value}")
            };
            self.owner()?.send(&command, deadline)?;
            if name == "MultiPV" {
                self.last_requested_multipv = value.parse().ok();
            }
        }
        self.advertised_options = options;
        self.ready(deadline, cancel)
    }

    fn ready(&mut self, deadline: Instant, cancel: &AtomicBool) -> Result<(), CheckerError> {
        self.ready_with_fence(deadline, cancel, None)
    }

    fn next_line_with_fence(
        &mut self,
        deadline: Instant,
        cancel: &AtomicBool,
        fence: Option<&dyn Fn() -> Option<Instant>>,
    ) -> Result<Option<String>, CheckerError> {
        loop {
            let until = fence
                .and_then(|read| read())
                .map_or(deadline, |f| f.min(deadline));
            if cancel.load(Ordering::Acquire) || Instant::now() >= until {
                return Ok(None);
            }
            let poll_until = if fence.is_some() {
                (Instant::now() + Duration::from_millis(10)).min(until)
            } else {
                until
            };
            let line = self.owner()?.next_line(poll_until, cancel)?;
            if line.is_some() || fence.is_none() {
                return Ok(line);
            }
        }
    }

    fn ready_with_fence(
        &mut self,
        deadline: Instant,
        cancel: &AtomicBool,
        fence: Option<&dyn Fn() -> Option<Instant>>,
    ) -> Result<(), CheckerError> {
        let actual_deadline = || {
            fence
                .and_then(|read| read())
                .map_or(deadline, |until| until.min(deadline))
        };
        if cancel.load(Ordering::Acquire) || Instant::now() >= actual_deadline() {
            return Err(error("readiness", "cancel_or_deadline_before_send"));
        }
        self.owner()?.send("isready", actual_deadline())?;
        loop {
            let line = self
                .next_line_with_fence(deadline, cancel, fence)?
                .ok_or_else(|| error("readiness", "timeout_or_cancel"))?;
            if line == "readyok" {
                if cancel.load(Ordering::Acquire) || Instant::now() >= actual_deadline() {
                    return Err(error("readiness", "late_readyok"));
                }
                return Ok(());
            }
            if line.starts_with("bestmove") || line == "uciok" {
                return Err(error("readiness", "late_or_duplicate_response"));
            }
            // Engine diagnostics cannot confer work or analysis evidence here.
        }
    }

    /// Runs only on the unchanged private owner after every accepted go has
    /// reached its bestmove/ready barrier. This is not process shutdown or TT
    /// reset; readiness proves a live reusable UCI owner without requiring EOF.
    fn quiesce_root(
        &mut self,
        deadline: Instant,
        cancel: &AtomicBool,
        fence: &dyn Fn() -> Option<Instant>,
    ) -> Result<CheckerShutdown, CheckerError> {
        if self.failed || self.active {
            return Err(error("root_quiesce", "physical_completion_unknown"));
        }
        let result = self.ready_with_fence(deadline, cancel, Some(fence));
        if let Err(failure) = result {
            self.failed = true;
            return Err(failure);
        }
        let state = self.owner()?.evidence();
        if state.process_identity.is_none()
            || state.quit_sent
            || state.exit_observed
            || state.cleanup_complete
            || state.quarantined
            || state.ownership_lost
        {
            self.failed = true;
            return Err(error("root_quiesce", "owner_not_reusable"));
        }
        Ok(state)
    }

    fn reset_game_with_fence(
        &mut self,
        deadline: Instant,
        cancel: &AtomicBool,
        fence: Option<&dyn Fn() -> Option<Instant>>,
    ) -> Result<(), CheckerError> {
        if self.failed || self.active {
            return Err(error("new_game", "checker_not_available"));
        }
        let until = fence
            .and_then(|read| read())
            .map_or(deadline, |f| deadline.min(f));
        if cancel.load(Ordering::Acquire) || Instant::now() >= until {
            return Err(error("new_game", "cancel_or_deadline_before_reset"));
        }
        self.owner()?.send("ucinewgame", until)?;
        if let Err(failure) = self.ready_with_fence(deadline, cancel, fence) {
            self.failed = true;
            return Err(failure);
        }
        self.last_attempt = None;
        Ok(())
    }

    fn run(
        &mut self,
        position: &Position,
        roots: Option<&[BoardMove]>,
        limits: CpuLimits,
        cancel: &AtomicBool,
    ) -> Result<CheckerReport, CheckerError> {
        self.last_attempt = None;
        let constraint = ExternalGoConstraint::Combined {
            depth: limits.max_depth,
            reported_nodes: limits.max_nodes,
        };
        if let Some((profile, _)) = self.engine_profile {
            profile.require_query(constraint, roots.is_some())?;
        }
        self.run_with_control(position, roots, limits, cancel, None, constraint, true)
    }

    /// New explicit-constraint path for full-legal-root single-limit engines.
    /// It reuses the same physical owner, stop/bestmove/readiness barrier and
    /// elapsed provenance; absent limits are not filled with fictitious caps.
    pub fn analyze_profiled(
        &mut self,
        position: &Position,
        roots: Option<&[BoardMove]>,
        requested: ExternalGoConstraint,
        hard_deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<ProfiledExternalReport, CheckerError> {
        self.last_attempt = None;
        let (profile, purpose) = self.engine_profile.ok_or(CheckerError::Invalid(
            "explicit external query requires an admitted engine profile",
        ))?;
        profile.require_query(requested, roots.is_some())?;
        if requested
            .requested_depth()
            .is_some_and(|depth| depth > self.config.max_depth)
        {
            return Err(CheckerError::Invalid(
                "external requested depth exceeds profile cap",
            ));
        }
        let limits = CpuLimits {
            max_depth: requested.requested_depth().unwrap_or(0),
            max_nodes: requested.requested_nodes().unwrap_or(0),
            deadline: Some(hard_deadline),
        };
        let CheckerReport::ExternalUci(raw) =
            self.run_with_control(position, roots, limits, cancel, None, requested, false)?
        else {
            return Err(error("analysis", "foreign_report_type"));
        };
        let issued = self.last_issued_constraint.ok_or(CheckerError::Invalid(
            "external query completed without issued constraint evidence",
        ))?;
        let issued_go = self.last_issued_go.take().ok_or(CheckerError::Invalid(
            "external query completed without issued command evidence",
        ))?;
        Ok(ProfiledExternalReport {
            profile,
            purpose,
            requested,
            issued,
            issued_go,
            reported_node_overshoot: requested
                .requested_nodes()
                .is_some_and(|limit| raw.work.nodes.is_some_and(|nodes| nodes > limit)),
            raw,
            strict_native_nodes: None,
            option_application_observed: None,
            readiness_barrier_completed: true,
        })
    }

    /// The synchronous checker is owned by a private external worker. The
    /// manager never calls this blocking pipe/drain path. Yield is a partial
    /// stop observation, not an exact continuation token or completed bound.
    pub fn analyze_cooperative(
        &mut self,
        position: &Position,
        roots: Option<&[BoardMove]>,
        mut limits: CpuLimits,
        cancel: &AtomicBool,
        control: &dyn ExternalControl,
        policy: ExternalSlicePolicy,
    ) -> Result<ExternalCooperativeReport, CheckerError> {
        self.last_attempt = None;
        if let Some((profile, _)) = self.engine_profile {
            profile.require_query(
                ExternalGoConstraint::Combined {
                    depth: limits.max_depth,
                    reported_nodes: policy.requested_reported_nodes,
                },
                roots.is_some(),
            )?;
        }
        let candidates = roots.map_or_else(
            || position.ordered_legal_moves().moves().len(),
            <[BoardMove]>::len,
        );
        policy.validate(candidates)?;
        if self.config.stop_grace != policy.stop_drain_reserve {
            return Err(CheckerError::Invalid(
                "external configured stop reserve differs from slice",
            ));
        }
        if !limits.deadline.is_some_and(|deadline| {
            deadline.saturating_duration_since(Instant::now())
                >= policy.quantum + policy.stop_drain_reserve
        }) {
            return Err(error("admission", "insufficient_quantum_and_stop_reserve"));
        }
        let kind = self
            .advertised_options
            .get("MultiPV")
            .ok_or_else(|| error("admission", "multipv_not_advertised"))?;
        if !validate_option(kind, &policy.multipv.to_string()) {
            return Err(error("admission", "multipv_out_of_range"));
        }
        // Policy limits are requested foreign work, never CpuJob native credits.
        limits.max_nodes = policy.requested_reported_nodes;
        self.clear_cooperative_opinions();
        self.stop_control = None;
        self.go_issue = ExternalGoIssue::NotIssued;
        let CheckerReport::ExternalUci(report) = self.run_with_control(
            position,
            roots,
            limits,
            cancel,
            Some((control, policy)),
            ExternalGoConstraint::Combined {
                depth: limits.max_depth,
                reported_nodes: limits.max_nodes,
            },
            false,
        )?
        else {
            return Err(error("analysis", "foreign_report_type"));
        };
        self.select_completed_exact_policy(&report, policy.multipv);
        self.cooperative_opinions
            .sort_by_key(|opinion| opinion.rank);
        let policy_target_known = report.completion == ExternalCompletion::BestMove
            && self.cooperative_opinions.len() == usize::from(policy.multipv)
            && self
                .cooperative_opinions
                .iter()
                .enumerate()
                .all(|(index, opinion)| {
                    opinion.rank == index as u16 + 1
                        && Some(opinion.depth)
                            == self.cooperative_opinions.first().map(|first| first.depth)
                        && opinion.bound == ExternalBound::ExactReported
                        && !self.cooperative_opinions[..index]
                            .iter()
                            .any(|prior| prior.pv.first() == opinion.pv.first())
                })
            && self
                .cooperative_opinions
                .first()
                .and_then(|opinion| opinion.pv.first())
                .copied()
                == report.best_move;
        Ok(ExternalCooperativeReport {
            reported_node_overshoot: report
                .work
                .nodes
                .is_some_and(|nodes| nodes > policy.requested_reported_nodes),
            report,
            opinions: self.cooperative_opinions.clone(),
            stop_control: self.stop_control,
            strict_native_nodes: None,
            reported_node_request: policy.requested_reported_nodes,
            requested_multipv: policy.multipv,
            option_application_observed: None,
            readiness_barrier_completed: true,
            policy_target_known,
            go_issue: self.go_issue,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_with_control(
        &mut self,
        position: &Position,
        roots: Option<&[BoardMove]>,
        limits: CpuLimits,
        cancel: &AtomicBool,
        cooperative: Option<(&dyn ExternalControl, ExternalSlicePolicy)>,
        constraint: ExternalGoConstraint,
        enforce_reported_node_cap: bool,
    ) -> Result<CheckerReport, CheckerError> {
        self.last_attempt = None;
        self.go_issue = ExternalGoIssue::NotIssued;
        self.last_issued_constraint = None;
        self.last_issued_go = None;
        let start = Instant::now();
        if self.failed || self.active {
            return Err(error("analysis", "checker_not_available"));
        }
        if constraint
            .requested_depth()
            .is_some_and(|depth| depth == 0 || depth > self.config.max_depth)
            || constraint.requested_nodes() == Some(0)
        {
            return Err(CheckerError::Invalid("external requested depth/nodes"));
        }
        let legal_view = position.ordered_legal_moves();
        if position
            .play_status_from_view(&legal_view)
            .map_err(|_| error("analysis", "rules_classification"))?
            != PlayStatus::Ongoing
        {
            return Err(CheckerError::Unsupported(
                "Rules terminal is resolved without external UCI",
            ));
        }
        if let Some(mask) = roots {
            let legal = legal_view.moves();
            if mask.is_empty()
                || mask.len() > 256
                || mask
                    .iter()
                    .enumerate()
                    .any(|(at, mv)| !legal.contains(mv) || mask[..at].contains(mv))
            {
                return Err(CheckerError::Invalid("external root mask"));
            }
        }
        let deadline = bounded_deadline(start, self.config.max_task_wall_time, limits.deadline)?;
        if cancel.load(Ordering::Acquire) || start >= deadline {
            return Err(error("analysis", "cancel_or_deadline_before_go"));
        }
        let budget = ExternalTaskBudget::new(start, deadline, self.config.stop_grace)?;
        let replay = position
            .snapshot()
            .uci_replay(self.config.max_prefix_plies)
            .map_err(|_| error("position", "history_replay_refused"))?;
        let mut command = if replay.origin == HistoryOrigin::StartPosition {
            "position startpos".to_owned()
        } else {
            format!("position fen {}", replay.start_fen)
        };
        if !replay.moves.is_empty() {
            command.push_str(" moves");
            for mv in replay.moves {
                command.push(' ');
                command.push_str(&mv.to_string());
            }
        }
        budget.analysis_millis(Instant::now())?;
        self.request_id = self
            .request_id
            .checked_add(1)
            .ok_or(CheckerError::Invalid("external request id overflow"))?;
        let mut report = ExternalCheckerReport {
            identity: self.config.identity.clone(),
            observed_uci: self.observed_uci.clone(),
            request_id: self.request_id,
            best_move: None,
            pv: Vec::new(),
            score: ExternalRawScore::Unknown,
            bound: ExternalBound::Unknown,
            wdl_per_mille: None,
            perspective: position.side_to_move(),
            requested_depth: limits.max_depth,
            reported_depth: None,
            seldepth: None,
            root_restricted: roots.is_some(),
            completion: ExternalCompletion::Pending,
            work: CheckerWork {
                nodes: None,
                qnodes: None,
                tt_hits: None,
            },
            elapsed: Duration::ZERO,
        };
        self.capture(&mut report, start);
        let result = self.run_active(
            position,
            roots,
            limits,
            cancel,
            start,
            budget,
            &command,
            &mut report,
            cooperative,
            constraint,
            enforce_reported_node_cap,
        );
        self.capture(&mut report, start);
        if let Err(failure) = result {
            if !self.active
                && matches!(
                    failure,
                    CheckerError::External {
                        stage: "admission",
                        code: "insufficient_stop_reserve"
                            | "insufficient_quantum_and_stop_reserve"
                            | "control_before_go"
                    }
                )
            {
                // Readiness/position may have consumed the analysis window, but
                // no go was sent. Preserve the idle owner and do not manufacture
                // an analysis attempt or borrow a cleanup wall for this refusal.
                self.last_attempt = None;
                return Err(failure);
            }
            self.failed = true;
            // Failed analysis may use only the task's remaining wall and the
            // cooperative owner's earlier physical fence. Never open a fresh
            // cleanup grace after a shortened shutdown fence has expired.
            // Retained quarantine can still be drained by explicit shutdown.
            let cleanup_fence = cooperative
                .and_then(|(control, _)| control.physical_fence_deadline())
                .map_or(budget.task_deadline, |fence| {
                    fence.min(budget.task_deadline)
                });
            let cleanup = bounded_deadline(
                Instant::now(),
                self.config.shutdown_grace,
                Some(cleanup_fence),
            )?;
            let _ = self.shutdown(cleanup);
            self.capture(&mut report, start);
            return Err(failure);
        }
        Ok(CheckerReport::ExternalUci(report))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_active(
        &mut self,
        position: &Position,
        roots: Option<&[BoardMove]>,
        limits: CpuLimits,
        cancel: &AtomicBool,
        start: Instant,
        budget: ExternalTaskBudget,
        position_command: &str,
        report: &mut ExternalCheckerReport,
        cooperative: Option<(&dyn ExternalControl, ExternalSlicePolicy)>,
        constraint: ExternalGoConstraint,
        enforce_reported_node_cap: bool,
    ) -> Result<(), CheckerError> {
        budget.analysis_millis(Instant::now())?;
        let requested_width = cooperative.map(|(_, policy)| policy.multipv).or_else(|| {
            self.config
                .identity
                .options
                .get("MultiPV")
                .and_then(|value| value.parse().ok())
                .or(self.startup_multipv)
        });
        if let Some(width) = requested_width {
            if self.last_requested_multipv != Some(width) {
                self.owner()?.send(
                    &format!("setoption name MultiPV value {width}"),
                    budget.analysis_deadline,
                )?;
                self.last_requested_multipv = Some(width);
            }
        }
        self.ready(budget.analysis_deadline, cancel)?;
        self.owner()?
            .send(position_command, budget.analysis_deadline)?;
        if cooperative
            .is_some_and(|(control, _)| control.observe().action != ExternalControlAction::Continue)
        {
            return Err(error("admission", "control_before_go"));
        }
        if cancel.load(Ordering::Acquire) {
            return Err(error("analysis", "cancel_or_deadline_before_go"));
        }
        let ready_at = Instant::now();
        let remaining_ms = budget.analysis_millis(ready_at)?;
        let budget = if let Some((_, policy)) = cooperative {
            if remaining_ms < policy.quantum.as_millis() {
                return Err(error("admission", "insufficient_quantum_and_stop_reserve"));
            }
            ExternalTaskBudget {
                analysis_deadline: ready_at
                    .checked_add(policy.quantum)
                    .ok_or(CheckerError::Invalid("external quantum deadline overflow"))?
                    .min(budget.analysis_deadline),
                ..budget
            }
        } else {
            budget
        };
        let remaining_ms = budget.analysis_millis(ready_at)?;
        let (mut go, issued) = constraint.command(remaining_ms);
        if let Some(mask) = roots {
            go.push_str(" searchmoves");
            for mv in mask {
                go.push(' ');
                go.push_str(&mv.to_string());
            }
        }
        self.go_issue = ExternalGoIssue::Unknown;
        self.last_issued_constraint = Some(issued);
        if self.engine_profile.is_some() {
            self.last_issued_go = Some(go.clone());
        }
        self.owner()?.send(&go, budget.analysis_deadline)?;
        self.go_issue = ExternalGoIssue::Issued;
        self.active = true;
        if let Some((control, _)) = cooperative {
            control.observe_owner_phase(ExternalOwnerPhase::GoIssued {
                at: Instant::now(),
                quantum_deadline: budget.analysis_deadline,
            });
        }
        let no_cancel = AtomicBool::new(false);
        let mut stopped_deadline = None;
        loop {
            let now = Instant::now();
            let observed = cooperative.map(|(control, _)| control.observe());
            let cooperative_stop =
                observed.is_some_and(|observed| observed.action != ExternalControlAction::Continue);
            if stopped_deadline.is_none()
                && (cancel.load(Ordering::Acquire)
                    || cooperative_stop
                    || now >= budget.analysis_deadline)
            {
                report.completion = if cancel.load(Ordering::Acquire)
                    || observed.is_some_and(|value| {
                        matches!(
                            value.action,
                            ExternalControlAction::Cancel | ExternalControlAction::Revoked
                        )
                    }) {
                    ExternalCompletion::StoppedCanceled
                } else if observed.is_some_and(|value| value.action == ExternalControlAction::Yield)
                {
                    ExternalCompletion::StoppedYield
                } else if cooperative.is_some() {
                    // Soft quantum completion is a partial external Yield. It
                    // does not consume the original hard deadline or mint a
                    // same-stack resume token.
                    ExternalCompletion::StoppedYield
                } else {
                    ExternalCompletion::StoppedDeadline
                };
                if cooperative_stop && !cancel.load(Ordering::Acquire) {
                    self.stop_control = observed;
                }
                let mut stop_deadline = budget.drain_deadline(now)?;
                if let Some(limit) =
                    cooperative.and_then(|(control, _)| control.physical_fence_deadline())
                {
                    stop_deadline = stop_deadline.min(limit);
                }
                self.owner()?.send("stop", stop_deadline)?;
                self.owner()?.mark_stop();
                if let Some((control, _)) = cooperative {
                    control
                        .observe_owner_phase(ExternalOwnerPhase::StopSent { at: Instant::now() });
                }
                stopped_deadline = Some(stop_deadline);
            }
            let wait_deadline = stopped_deadline.unwrap_or(budget.analysis_deadline);
            let token = if stopped_deadline.is_some() {
                &no_cancel
            } else {
                cancel
            };
            let active_control = if stopped_deadline.is_none() {
                cooperative.map(|(control, _)| control)
            } else {
                None
            };
            let read_fence =
                || cooperative.and_then(|(control, _)| control.physical_fence_deadline());
            let line = if stopped_deadline.is_some() && cooperative.is_some() {
                self.next_line_with_fence(wait_deadline, token, Some(&read_fence))?
            } else {
                self.owner()?
                    .next_line_controlled(wait_deadline, token, active_control)?
            };
            let Some(line) = line else {
                if stopped_deadline.is_some() {
                    return Err(error("stop", "bestmove_not_drained"));
                }
                continue;
            };
            if line.starts_with("info ") {
                let frame = parse_info(&line, 128)?;
                if let Some(nodes) = frame.nodes {
                    if report.work.nodes.is_some_and(|old| nodes < old) {
                        return Err(error("analysis", "nodes_regressed"));
                    }
                    report.work.nodes = Some(nodes);
                    self.capture(report, start);
                    if enforce_reported_node_cap && nodes > limits.max_nodes {
                        return Err(error("analysis", "reported_node_budget_exceeded"));
                    }
                }
                if cooperative.is_some_and(|(_, policy)| frame.multipv > policy.multipv) {
                    return Err(error("analysis", "multipv_outside_requested_width"));
                }
                if cooperative.is_some() {
                    if let (Some(score), Some(bound), Some(depth), Some(pv)) =
                        (frame.score, frame.bound, frame.depth, frame.pv.as_ref())
                    {
                        validate_pv(position, pv, roots)?;
                        let opinion = ExternalPolicyOpinion {
                            rank: frame.multipv,
                            depth,
                            seldepth: frame.seldepth,
                            pv: pv.clone(),
                            score,
                            bound,
                            reported_wdl: frame.wdl,
                        };
                        if opinion.rank == 1 {
                            if let Some((control, _)) = cooperative {
                                control.publish_progress(&opinion, report.work.nodes);
                            }
                        }
                        if let Some((_, policy)) = cooperative {
                            self.observe_exact_policy_iteration(&opinion, policy.multipv);
                        }
                        if let Some(prior) = self
                            .cooperative_opinions
                            .iter_mut()
                            .find(|prior| prior.rank == opinion.rank)
                        {
                            *prior = opinion;
                        } else {
                            self.cooperative_opinions.push(opinion);
                        }
                    }
                }
                if frame.multipv != 1 {
                    continue;
                }
                if let Some(pv) = &frame.pv {
                    validate_pv(position, pv, roots)?;
                }
                if let (Some(score), Some(bound), Some(depth), Some(pv)) =
                    (frame.score, frame.bound, frame.depth, frame.pv)
                {
                    report.score = score;
                    report.bound = bound;
                    report.reported_depth = Some(depth);
                    report.seldepth = frame.seldepth;
                    report.pv = pv;
                    report.wdl_per_mille = frame.wdl;
                    self.capture(report, start);
                }
            } else if let Some(body) = line.strip_prefix("bestmove ") {
                let parts: Vec<&str> = body.split_ascii_whitespace().collect();
                if parts.len() != 1 && (parts.len() != 3 || parts[1] != "ponder") {
                    return Err(error("analysis", "malformed_bestmove"));
                }
                let best = BoardMove::from_uci(parts[0])
                    .map_err(|_| error("analysis", "invalid_bestmove"))?;
                validate_pv(position, &[best], roots)?;
                if report.pv.first() != Some(&best) {
                    // Preserve the original tuple only in prior evidence; a
                    // bestmove mismatch cannot borrow the last line's score.
                    report.pv = vec![best];
                    report.score = ExternalRawScore::Unknown;
                    report.bound = ExternalBound::Unknown;
                    report.reported_depth = None;
                    report.seldepth = None;
                    report.wdl_per_mille = None;
                }
                report.best_move = Some(best);
                if matches!(report.completion, ExternalCompletion::Pending) {
                    report.completion = ExternalCompletion::BestMove;
                }
                self.active = false;
                if let Some((control, _)) = cooperative {
                    control.observe_owner_phase(ExternalOwnerPhase::BestMoveDrained {
                        at: Instant::now(),
                    });
                }
                let ready_deadline = match stopped_deadline {
                    Some(deadline) => deadline,
                    None => {
                        let deadline = budget.drain_deadline(Instant::now())?;
                        cooperative
                            .and_then(|(control, _)| control.physical_fence_deadline())
                            .map_or(deadline, |limit| deadline.min(limit))
                    }
                };
                let read_fence =
                    || cooperative.and_then(|(control, _)| control.physical_fence_deadline());
                let readiness = self.ready_with_fence(
                    ready_deadline,
                    &no_cancel,
                    cooperative.map(|_| &read_fence as &dyn Fn() -> Option<Instant>),
                );
                // Final cancellation/deadline remains a partial external
                // observation even when bestmove races with the stop boundary.
                if matches!(report.completion, ExternalCompletion::BestMove) {
                    if cancel.load(Ordering::Acquire) {
                        report.completion = ExternalCompletion::StoppedCanceled;
                    } else if cooperative.is_none() && Instant::now() >= budget.analysis_deadline {
                        report.completion = ExternalCompletion::StoppedDeadline;
                    }
                }
                readiness?;
                if let Some((control, _)) = cooperative {
                    control.observe_owner_phase(ExternalOwnerPhase::ReadyBarrierCompleted {
                        at: Instant::now(),
                    });
                    let latest = control.observe();
                    if matches!(
                        latest.action,
                        ExternalControlAction::Cancel | ExternalControlAction::Revoked
                    ) {
                        report.completion = ExternalCompletion::StoppedCanceled;
                        self.stop_control = Some(latest);
                    }
                }
                return Ok(());
            } else if matches!(line.as_str(), "uciok" | "readyok") {
                return Err(error("analysis", "unexpected_barrier_response"));
            }
        }
    }

    /// Begin a new physical query without reusing a previous query's data.
    fn clear_cooperative_opinions(&mut self) {
        self.cooperative_opinions.clear();
        self.cooperative_exact_candidates.clear();
        self.cooperative_exact_completed.clear();
    }

    /// Record exact, legal tuples as reported; a bounded or incomplete next
    /// depth never replaces the most recent complete equal-depth set.
    fn observe_exact_policy_iteration(&mut self, opinion: &ExternalPolicyOpinion, width: u16) {
        if opinion.rank == 0 || opinion.rank > width { return; }
        if opinion.bound != ExternalBound::ExactReported || opinion.pv.is_empty() {
            if self.cooperative_exact_candidates.first().is_some_and(|first| first.depth == opinion.depth) {
                self.cooperative_exact_candidates.clear();
            }
            return;
        }
        // Only an observed consecutive rank-ordered group is a completed
        // iteration; never splice a partial later MultiPV update into it.
        if opinion.rank == 1 {
            self.cooperative_exact_candidates.clear();
            self.cooperative_exact_candidates.push(opinion.clone());
        } else if self.cooperative_exact_candidates.last().is_some_and(|last|
            last.rank + 1 == opinion.rank && last.depth == opinion.depth)
        {
            self.cooperative_exact_candidates.push(opinion.clone());
        } else {
            self.cooperative_exact_candidates.clear();
            return;
        }
        if self.cooperative_exact_candidates.len() != usize::from(width) { return; }
        let depth = self.cooperative_exact_candidates[0].depth;
        if self.cooperative_exact_candidates.iter().enumerate().all(|(index, item)|
            item.rank == index as u16 + 1 && item.depth == depth
                && !self.cooperative_exact_candidates[..index].iter().any(|prior| prior.pv.first() == item.pv.first()))
            && self.cooperative_exact_completed.first().is_none_or(|old| old.depth < depth)
        {
            self.cooperative_exact_completed.clone_from(&self.cooperative_exact_candidates);
        }
    }

    /// Use a completed iteration only after the same query naturally returned
    /// bestmove and readiness, and only when its first move agrees. Never turn
    /// a stopped/cancelled execution into a completed policy observation.
    fn select_completed_exact_policy(&mut self, report: &ExternalCheckerReport, width: u16) {
        if report.completion == ExternalCompletion::BestMove
            && self.cooperative_exact_completed.len() == usize::from(width)
            && self.cooperative_exact_completed.first().and_then(|item| item.pv.first()).copied() == report.best_move
        {
            self.cooperative_opinions.clone_from(&self.cooperative_exact_completed);
        }
    }

    fn capture(&mut self, report: &mut ExternalCheckerReport, start: Instant) {
        // One clock observation belongs to both copies of this physical fact.
        // Reading it again for the attempt makes an otherwise identical final
        // report fail the immutable receipt comparison in the search owner.
        report.elapsed = start.elapsed();
        let process = self
            .owner
            .as_ref()
            .map_or_else(empty_shutdown, |owner| owner.evidence());
        self.last_attempt = Some(CheckerAttempt {
            work: report.work,
            elapsed: report.elapsed,
            external: Some(ExternalAttemptEvidence {
                request_id: report.request_id,
                partial_report: Some(report.clone()),
                process,
            }),
        });
    }
}

fn empty_shutdown() -> CheckerShutdown {
    CheckerShutdown {
        process_identity: None,
        stop_sent: false,
        quit_sent: false,
        exit_observed: false,
        exit_code: None,
        exit_signal: None,
        ownership_lost: false,
        stdout_drained: false,
        stderr_drained: false,
        cleanup_complete: false,
        quarantined: false,
        stdout_bytes: 0,
        stderr_bytes: 0,
    }
}

impl CpuChecker for ExternalUciCpuChecker {
    fn startup_uci(&self) -> Option<ExternalUciIdentity> {
        (self.owner.is_some() && !self.failed && !self.observed_uci.name.is_empty())
            .then(|| self.observed_uci.clone())
    }
    fn start(&mut self, deadline: Instant, cancel: &AtomicBool) -> Result<(), CheckerError> {
        ExternalUciCpuChecker::start(self, deadline, cancel)
    }
    fn identity(&self) -> &CheckerIdentity {
        &self.identity
    }
    fn conditions(&self) -> &str {
        &self.conditions
    }
    fn validate_namespace(&self) -> Result<(), CheckerError> {
        self.identity.validate()?;
        // Preserve CpuChecker's metadata boundary before checking availability.
        if self.conditions.is_empty()
            || self.conditions.len() > 8192
            || self.conditions.chars().any(char::is_control)
        {
            return Err(CheckerError::Invalid(
                "checker conditions exceed finite bound",
            ));
        }
        if self.failed || self.active {
            return Err(error("admission", "checker_not_available"));
        }
        // No owner is expected before explicit startup. A retained owner's
        // terminal evidence never grants a new search or resets its old attempt.
        if let Some(owner) = &self.owner {
            let state = owner.evidence();
            if state.quarantined {
                return Err(error("admission", "owner_quarantined"));
            }
            if state.ownership_lost {
                return Err(error("admission", "owner_ownership_lost"));
            }
            if state.cleanup_complete {
                return Err(error("admission", "owner_closed"));
            }
        }
        Ok(())
    }
    fn capabilities(&self) -> CheckerCapabilities {
        let root_moves = self
            .engine_profile
            .is_none_or(|(profile, _)| profile.capabilities().searchmoves);
        CheckerCapabilities {
            max_depth: self.config.max_depth,
            max_prefix_plies: self.config.max_prefix_plies,
            max_root_moves: if root_moves { 256 } else { 0 },
            root_moves,
            divergence: true,
            resume: false,
            selective_search: None,
        }
    }
    fn preflight_task(&self, limits: CpuLimits) -> Result<(), CheckerError> {
        if let Some((profile, _)) = self.engine_profile {
            profile.require_query(
                ExternalGoConstraint::Combined {
                    depth: limits.max_depth,
                    reported_nodes: limits.max_nodes,
                },
                false,
            )?;
        }
        if self.failed || self.active {
            return Err(error("analysis", "checker_not_available"));
        }
        if limits.max_depth == 0
            || limits.max_depth > self.config.max_depth
            || limits.max_nodes == 0
        {
            return Err(CheckerError::Invalid("external requested depth/nodes"));
        }
        let start = Instant::now();
        let deadline = bounded_deadline(start, self.config.max_task_wall_time, limits.deadline)?;
        if start >= deadline {
            return Err(error("analysis", "cancel_or_deadline_before_go"));
        }
        ExternalTaskBudget::new(start, deadline, self.config.stop_grace).map(|_| ())
    }
    fn last_attempt(&self) -> Option<&CheckerAttempt> {
        self.last_attempt.as_ref()
    }
    fn reset_attempt(&mut self) {
        self.last_attempt = None;
    }
    fn analyze(
        &mut self,
        p: &Position,
        l: CpuLimits,
        c: &AtomicBool,
    ) -> Result<CheckerReport, CheckerError> {
        self.run(p, None, l, c)
    }
    fn analyze_root_moves(
        &mut self,
        p: &Position,
        m: &[BoardMove],
        l: CpuLimits,
        c: &AtomicBool,
    ) -> Result<CheckerReport, CheckerError> {
        self.run(p, Some(m), l, c)
    }
    fn analyze_divergence(
        &mut self,
        p: &Position,
        prefix: &[BoardMove],
        l: CpuLimits,
        c: &AtomicBool,
    ) -> Result<CheckerReport, CheckerError> {
        self.last_attempt = None;
        if prefix.len() > self.config.max_prefix_plies {
            return Err(CheckerError::Invalid("external divergence prefix cap"));
        }
        let mut target = p.clone();
        for &mv in prefix {
            target
                .make_move(mv)
                .map_err(|_| error("divergence", "illegal_prefix"))?;
        }
        self.run(&target, None, l, c)
    }
    fn resume(
        &mut self,
        _p: &Position,
        _t: &CpuResumeToken,
        _l: CpuLimits,
        _c: &AtomicBool,
    ) -> Result<CheckerReport, CheckerError> {
        self.last_attempt = None;
        Err(CheckerError::Unsupported(
            "external UCI internal stack resume",
        ))
    }
    fn new_game(&mut self, deadline: Instant, cancel: &AtomicBool) -> Result<(), CheckerError> {
        self.reset_game_with_fence(deadline, cancel, None)
    }
    fn shutdown(&mut self, deadline: Instant) -> Result<CheckerShutdown, CheckerError> {
        let Some(owner) = self.owner.as_mut() else {
            // A failed spawn can return actual proof that this owner never
            // acquired a process, or completed its setup cleanup. Reuse only
            // that exact request-0 state; do not invent exit or drained pipes.
            let cleanup = self.last_attempt.as_ref().and_then(|attempt| {
                attempt.external.as_ref().filter(|external| {
                    self.failed
                        && external.request_id == 0
                        && external.partial_report.is_none()
                        && external.process.cleanup_complete
                        && !external.process.quarantined
                        && !external.process.ownership_lost
                })
            });
            if let Some(evidence) = cleanup {
                return Ok(evidence.process);
            }
            return Err(error("shutdown", "owner_closed"));
        };
        let result = owner.shutdown(deadline, self.active);
        self.active = false;
        if let Some(attempt) = self
            .last_attempt
            .as_mut()
            .and_then(|attempt| attempt.external.as_mut())
        {
            attempt.process = owner.evidence();
        }
        result
    }
}

#[cfg(target_os = "linux")]
mod process {
    use super::*;
    use nix::errno::Errno;
    use nix::fcntl::{FcntlArg, OFlag, fcntl};
    use nix::sys::signal::{Signal, killpg};
    use nix::sys::wait::{Id, WaitPidFlag, WaitStatus, waitid};
    use nix::unistd::Pid;
    use std::collections::VecDeque;
    use std::io::{ErrorKind, Write};
    use std::os::fd::{AsFd, AsRawFd};
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
    use std::os::unix::process::CommandExt;
    use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};

    pub(super) struct OwnedProcess {
        child: Option<Child>,
        stdin: Option<ChildStdin>,
        stdout: Option<ChildStdout>,
        stderr: Option<ChildStderr>,
        stdin_nonblocking: bool,
        stdout_nonblocking: bool,
        stderr_nonblocking: bool,
        _program: File,
        _directory: File,
        pid: Pid,
        start_time: u64,
        authority: bool,
        max_output: usize,
        max_line: usize,
        stdout_log: Vec<u8>,
        stderr_log: Vec<u8>,
        pending_line: Vec<u8>,
        lines: VecDeque<String>,
        state: CheckerShutdown,
        shutdown_grace: Duration,
        process_slot: process_scope::ProcessLease,
    }

    pub(super) struct SpawnFailure {
        pub(super) error: CheckerError,
        pub(super) process: CheckerShutdown,
        pub(super) stdout: Vec<u8>,
        pub(super) stderr: Vec<u8>,
    }

    fn nonblocking<F: AsFd>(fd: &F) -> Result<(), CheckerError> {
        let flags = fcntl(fd, FcntlArg::F_GETFL).map_err(|_| error("spawn", "pipe_flags"))?;
        fcntl(
            fd,
            FcntlArg::F_SETFL(OFlag::from_bits_truncate(flags) | OFlag::O_NONBLOCK),
        )
        .map_err(|_| error("spawn", "pipe_nonblocking"))?;
        Ok(())
    }

    #[derive(Clone, Copy)]
    struct ProcIdentity {
        group: i32,
        start: u64,
    }

    fn proc_identity(pid: i32) -> Result<Option<ProcIdentity>, CheckerError> {
        read_proc_identity(File::open(format!("/proc/{pid}/stat")))
    }

    fn proc_stat_disappeared(failure: &std::io::Error) -> bool {
        failure.kind() == ErrorKind::NotFound || failure.raw_os_error() == Some(nix::libc::ESRCH)
    }

    fn read_proc_identity<R: Read>(
        source: std::io::Result<R>,
    ) -> Result<Option<ProcIdentity>, CheckerError> {
        let file = match source {
            Ok(file) => file,
            Err(failure) if proc_stat_disappeared(&failure) => return Ok(None),
            Err(_) => return Err(error("ownership", "proc_stat_unavailable")),
        };
        let mut bytes = Vec::new();
        match file.take(4097).read_to_end(&mut bytes) {
            Ok(_) => {}
            // A task can exit after its stat FD was opened during the bounded
            // group scan. Never parse bytes from a failed partial read. Leader
            // callers still reject None as a missing/disappeared owned leader.
            Err(failure) if proc_stat_disappeared(&failure) => return Ok(None),
            Err(_) => return Err(error("ownership", "proc_stat_read")),
        }
        if bytes.len() > 4096 {
            return Err(error("ownership", "proc_stat_budget"));
        }
        let text =
            std::str::from_utf8(&bytes).map_err(|_| error("ownership", "proc_stat_encoding"))?;
        let body = text
            .rsplit_once(") ")
            .map(|(_, body)| body)
            .ok_or_else(|| error("ownership", "proc_stat_format"))?;
        let fields: Vec<&str> = body.split_ascii_whitespace().collect();
        let group = fields
            .get(2)
            .and_then(|value| value.parse().ok())
            .ok_or_else(|| error("ownership", "proc_group"))?;
        let start = fields
            .get(19)
            .and_then(|value| value.parse().ok())
            .ok_or_else(|| error("ownership", "proc_start"))?;
        Ok(Some(ProcIdentity { group, start }))
    }

    fn other_group_members(pid: Pid) -> Result<usize, CheckerError> {
        let mut scanned = 0usize;
        let mut count = 0usize;
        for entry in std::fs::read_dir("/proc").map_err(|_| error("cleanup", "proc_group_scan"))? {
            let entry = entry.map_err(|_| error("cleanup", "proc_entry"))?;
            let Some(id) = entry
                .file_name()
                .to_str()
                .and_then(|name| name.parse::<i32>().ok())
            else {
                continue;
            };
            scanned += 1;
            if scanned > 65_536 {
                return Err(error("cleanup", "proc_scan_budget"));
            }
            if id != pid.as_raw()
                && proc_identity(id)?.is_some_and(|identity| identity.group == pid.as_raw())
            {
                count += 1;
            }
        }
        Ok(count)
    }

    impl OwnedProcess {
        pub(super) fn spawn(
            config: &ExternalCpuConfig,
            deadline: Instant,
            cancel: &AtomicBool,
            launch_pin: Option<&ExternalLaunchPin>,
        ) -> Result<Self, Box<SpawnFailure>> {
            let mut process_slot = Some(
                process_scope::ProcessLease::acquire(
                    launch_pin.and_then(|pin| pin.collection_process_permit.as_ref()),
                )
                .map_err(|error| {
                    Box::new(SpawnFailure {
                        error,
                        process: empty_shutdown(),
                        stdout: Vec::new(),
                        stderr: Vec::new(),
                    })
                })?,
            );
            let mut failure_evidence = None;
            let result = Self::spawn_owned(
                config,
                deadline,
                cancel,
                launch_pin,
                &mut failure_evidence,
                &mut process_slot,
            );
            result.map_err(|failure| {
                Box::new(failure_evidence.unwrap_or_else(|| {
                    let mut process = empty_shutdown();
                    process.cleanup_complete = true;
                    SpawnFailure {
                        error: failure,
                        process,
                        stdout: Vec::new(),
                        stderr: Vec::new(),
                    }
                }))
            })
        }
        fn spawn_owned(
            config: &ExternalCpuConfig,
            deadline: Instant,
            cancel: &AtomicBool,
            launch_pin: Option<&ExternalLaunchPin>,
            failure_evidence: &mut Option<SpawnFailure>,
            process_slot: &mut Option<process_scope::ProcessLease>,
        ) -> Result<Self, CheckerError> {
            use std::os::unix::fs::FileExt;
            let program = match launch_pin {
                Some(pin) => pin
                    .executable
                    .try_clone()
                    .map_err(|_| error("spawn", "pinned_program_clone"))?,
                None => std::fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(nix::libc::O_NOFOLLOW)
                    .open(&config.program)
                    .map_err(|_| error("spawn", "program_open"))?,
            };
            let metadata = program
                .metadata()
                .map_err(|_| error("spawn", "program_metadata"))?;
            if !metadata.is_file()
                || metadata.len() > 256 * 1024 * 1024
                || metadata.mode() & 0o111 == 0
            {
                return Err(error("spawn", "program_kind_or_size"));
            }
            let mut header = [0u8; 4];
            program
                .read_exact_at(&mut header, 0)
                .map_err(|_| error("spawn", "program_header"))?;
            if header != *b"\x7fELF" {
                return Err(error("spawn", "native_elf_required"));
            }
            if let Some(sealed_sha256) = launch_pin.and_then(|pin| pin.sealed_sha256.as_ref()) {
                use nix::fcntl::{FcntlArg, SealFlag, fcntl};
                let required = SealFlag::F_SEAL_WRITE
                    | SealFlag::F_SEAL_GROW
                    | SealFlag::F_SEAL_SHRINK
                    | SealFlag::F_SEAL_SEAL;
                let actual = fcntl(&program, FcntlArg::F_GET_SEALS)
                    .map_err(|_| error("spawn", "sealed_program_readback"))?;
                if (actual & required.bits()) != required.bits()
                    || sealed_sha256 != &config.identity.binary_sha256
                {
                    return Err(error("spawn", "sealed_program_identity"));
                }
            } else {
                let mut hash = Sha256::new();
                hash.update(header);
                let mut buffer = [0u8; 65_536];
                let mut total = 4u64;
                loop {
                    // Report both conditions when both are observed at this check.
                    // They are independent observations, not an ordering claim:
                    // cancellation can race with the deadline and is read first.
                    let cancelled = cancel.load(Ordering::Acquire);
                    let expired = Instant::now() >= deadline;
                    match (cancelled, expired) {
                        (false, false) => {}
                        (true, false) => return Err(error("spawn", "hash_cancelled")),
                        (false, true) => return Err(error("spawn", "hash_deadline_expired")),
                        (true, true) => {
                            return Err(error("spawn", "hash_cancelled_and_deadline_expired"));
                        }
                    }
                    let count = program
                        .read_at(&mut buffer, total)
                        .map_err(|_| error("spawn", "program_hash_read"))?;
                    if count == 0 {
                        break;
                    }
                    total = total
                        .checked_add(count as u64)
                        .ok_or_else(|| error("spawn", "program_hash_overflow"))?;
                    if total > 256 * 1024 * 1024 {
                        return Err(error("spawn", "program_hash_budget"));
                    }
                    hash.update(&buffer[..count]);
                }
                let after = program
                    .metadata()
                    .map_err(|_| error("spawn", "program_recheck"))?;
                if total != metadata.len()
                    || after.len() != metadata.len()
                    || after.dev() != metadata.dev()
                    || after.ino() != metadata.ino()
                    || after.mtime() != metadata.mtime()
                    || after.mtime_nsec() != metadata.mtime_nsec()
                    || after.ctime() != metadata.ctime()
                    || after.ctime_nsec() != metadata.ctime_nsec()
                {
                    return Err(error("spawn", "program_changed_during_hash"));
                }
                if format!("{:x}", hash.finalize()) != config.identity.binary_sha256 {
                    return Err(error("spawn", "program_hash_mismatch"));
                }
            }
            let directory = match launch_pin {
                Some(pin) => pin
                    .directory
                    .try_clone()
                    .map_err(|_| error("spawn", "pinned_directory_clone"))?,
                None => std::fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(nix::libc::O_DIRECTORY | nix::libc::O_NOFOLLOW)
                    .open(&config.working_directory)
                    .map_err(|_| error("spawn", "directory_open"))?,
            };
            if !directory
                .metadata()
                .map_err(|_| error("spawn", "directory_metadata"))?
                .is_dir()
            {
                return Err(error("spawn", "directory_kind"));
            }
            let mut command = Command::new(format!("/proc/self/fd/{}", program.as_raw_fd()));
            if cancel.load(Ordering::Acquire) || Instant::now() >= deadline {
                return Err(error("spawn", "cancel_or_deadline_before_spawn"));
            }
            command
                .args(&config.arguments)
                .current_dir(format!("/proc/self/fd/{}", directory.as_raw_fd()))
                .env_clear()
                .env("PATH", "/usr/bin:/bin")
                .env("LC_ALL", "C")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .process_group(0);
            // Before spawn, dropping this reservation means NotStarted. After
            // spawn it is immediately installed in the original OS owner.
            let mut lease = process_slot
                .take()
                .expect("reserved process slot before spawn");
            lease.ensure_spawn_allowed()?;
            let mut child = match command.spawn() {
                Ok(child) => {
                    if let Some(counters) = launch_pin.and_then(|pin| pin.sealed_counters.as_ref()) {
                        if launch_pin.is_some_and(|pin| pin.sealed_sha256.is_some()) {
                            let _ = counters.sealed_spawn_returned.fetch_update(
                                Ordering::AcqRel, Ordering::Acquire, |n| Some(n.saturating_add(1)));
                        }
                    }
                    child
                }
                Err(failure)
                    if launch_pin.is_some_and(|pin| pin.sealed_sha256.is_some())
                        && matches!(failure.raw_os_error(),
                            Some(nix::libc::EACCES | nix::libc::EPERM | nix::libc::ETXTBSY)) =>
                {
                    let pin = launch_pin.expect("sealed pin checked above");
                    let original = pin.fallback_executable.as_ref()
                        .ok_or_else(|| error("spawn", "sealed_fallback_missing"))?;
                    let mut fallback = pin.clone();
                    fallback.executable = Arc::clone(original);
                    fallback.sealed_sha256 = None;
                    fallback.fallback_executable = None;
                    if let Some(counters) = pin.sealed_counters.as_ref() {
                        let _ = counters.original_fallback_attempted.fetch_update(
                            Ordering::AcqRel, Ordering::Acquire, |n| Some(n.saturating_add(1)));
                    }
                    // No child was returned: transfer the *same* reservation to
                    // the ordinary original-FD hash path, without extending its
                    // deadline or reacquiring a one-shot collection permit.
                    *process_slot = Some(lease);
                    return Self::spawn_owned(config, deadline, cancel, Some(&fallback),
                        failure_evidence, process_slot);
                }
                Err(_) => return Err(error("spawn", "native_spawn")),
            };
            lease.mark_started();
            // Linux kernel PIDs fit signed pid_t. Install the owned child before
            // any fallible setup and retain optional pipes even on setup error.
            let pid = Pid::from_raw(child.id() as i32);
            let stdin = child.stdin.take();
            let stdout = child.stdout.take();
            let stderr = child.stderr.take();
            // Child ownership is installed before any fallible pipe/proc setup.
            // An initialization failure receives the same bounded cleanup.
            let mut owner = Self {
                child: Some(child),
                stdin,
                stdout,
                stderr,
                stdin_nonblocking: false,
                stdout_nonblocking: false,
                stderr_nonblocking: false,
                _program: program,
                _directory: directory,
                pid,
                start_time: 0,
                authority: true,
                max_output: config.max_output_bytes,
                max_line: config.max_line_bytes,
                stdout_log: Vec::new(),
                stderr_log: Vec::new(),
                pending_line: Vec::new(),
                lines: VecDeque::new(),
                state: empty_shutdown(),
                shutdown_grace: config.shutdown_grace,
                process_slot: lease,
            };
            let setup = (|| {
                let identity = proc_identity(pid.as_raw())?
                    .ok_or_else(|| error("spawn", "leader_identity_missing"))?;
                if identity.group != pid.as_raw() {
                    return Err(error("spawn", "leader_group_mismatch"));
                }
                owner.start_time = identity.start;
                owner.state.process_identity = Some(crate::cpu_checker::ExternalProcessIdentity {
                    pid: u32::try_from(pid.as_raw())
                        .map_err(|_| error("spawn", "leader_pid_range"))?,
                    process_group: u32::try_from(identity.group)
                        .map_err(|_| error("spawn", "leader_group_range"))?,
                    proc_start_ticks: identity.start,
                });
                nonblocking(
                    owner
                        .stdin
                        .as_ref()
                        .ok_or_else(|| error("spawn", "stdin_missing"))?,
                )?;
                owner.stdin_nonblocking = true;
                nonblocking(
                    owner
                        .stdout
                        .as_ref()
                        .ok_or_else(|| error("spawn", "stdout_missing"))?,
                )?;
                owner.stdout_nonblocking = true;
                nonblocking(
                    owner
                        .stderr
                        .as_ref()
                        .ok_or_else(|| error("spawn", "stderr_missing"))?,
                )?;
                owner.stderr_nonblocking = true;
                Ok::<(), CheckerError>(())
            })();
            if let Err(failure) = setup {
                let cleanup =
                    bounded_deadline(Instant::now(), owner.shutdown_grace, Some(deadline))?;
                let _ = owner.shutdown(cleanup, false);
                *failure_evidence = Some(SpawnFailure {
                    error: failure.clone(),
                    process: owner.evidence(),
                    stdout: owner.stdout_log.clone(),
                    stderr: owner.stderr_log.clone(),
                });
                return Err(failure);
            }
            Ok(owner)
        }
        fn status(&mut self) -> Result<WaitStatus, CheckerError> {
            if !self.authority {
                return Err(error("ownership", "authority_lost"));
            }
            match waitid(
                Id::Pid(self.pid),
                WaitPidFlag::WEXITED | WaitPidFlag::WNOHANG | WaitPidFlag::WNOWAIT,
            ) {
                Ok(status) => Ok(status),
                Err(Errno::ECHILD) => {
                    self.authority = false;
                    self.state.ownership_lost = true;
                    self.state.quarantined = true;
                    Err(error("ownership", "foreign_reaper_or_no_child"))
                }
                Err(_) => Err(error("ownership", "waitid_failed")),
            }
        }
        fn verify_signal_authority(&mut self) -> Result<(), CheckerError> {
            self.status()?;
            let identity = proc_identity(self.pid.as_raw())?
                .ok_or_else(|| error("ownership", "leader_disappeared"))?;
            if identity.start != self.start_time || identity.group != self.pid.as_raw() {
                self.authority = false;
                self.state.ownership_lost = true;
                self.state.quarantined = true;
                return Err(error("ownership", "leader_identity_changed"));
            }
            Ok(())
        }
        fn signal(&mut self, signal: Signal) -> Result<(), CheckerError> {
            self.verify_signal_authority()?;
            killpg(self.pid, signal).map_err(|_| error("cleanup", "group_signal_failed"))
        }
        pub(super) fn send(&mut self, line: &str, deadline: Instant) -> Result<(), CheckerError> {
            if !self.stdin_nonblocking {
                return Err(error("stdin", "nonblocking_not_ready"));
            }
            if line.len() > 65_536 || line.contains(['\r', '\n', '\0']) {
                return Err(error("stdin", "command_budget_or_injection"));
            }
            let mut bytes = line.as_bytes().to_vec();
            bytes.push(b'\n');
            let mut written = 0;
            while written < bytes.len() {
                if Instant::now() >= deadline {
                    return Err(error("stdin", "write_deadline"));
                }
                match self
                    .stdin
                    .as_mut()
                    .ok_or_else(|| error("stdin", "closed"))?
                    .write(&bytes[written..])
                {
                    Ok(0) => return Err(error("stdin", "zero_write")),
                    Ok(count) => written += count,
                    Err(failure) if failure.kind() == ErrorKind::WouldBlock => {
                        self.pump(false)?;
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    Err(failure) if failure.kind() == ErrorKind::Interrupted => {}
                    Err(_) => return Err(error("stdin", "write_failure")),
                }
            }
            Ok(())
        }
        fn pump(&mut self, discard: bool) -> Result<(), CheckerError> {
            let mut buffer = [0u8; 4096];
            // Fair finite reads; a streaming stdout cannot starve stderr/clock.
            for stdout in [true, false] {
                if stdout && !self.stdout_nonblocking || !stdout && !self.stderr_nonblocking {
                    continue;
                }
                for _ in 0..16 {
                    let result = if stdout {
                        self.stdout
                            .as_mut()
                            .ok_or_else(|| error("stdout", "pipe_missing"))?
                            .read(&mut buffer)
                    } else {
                        self.stderr
                            .as_mut()
                            .ok_or_else(|| error("stderr", "pipe_missing"))?
                            .read(&mut buffer)
                    };
                    match result {
                        Ok(0) => {
                            if stdout {
                                self.state.stdout_drained = true;
                            } else {
                                self.state.stderr_drained = true;
                            }
                            break;
                        }
                        Ok(count) => {
                            if stdout {
                                self.state.stdout_bytes =
                                    self.state.stdout_bytes.saturating_add(count as u64);
                            } else {
                                self.state.stderr_bytes =
                                    self.state.stderr_bytes.saturating_add(count as u64);
                            }
                            if discard {
                                continue;
                            }
                            if self
                                .stdout_log
                                .len()
                                .checked_add(self.stderr_log.len())
                                .and_then(|bytes| bytes.checked_add(count))
                                .is_none_or(|bytes| bytes > self.max_output)
                            {
                                return Err(error("output", "combined_output_budget"));
                            }
                            if stdout {
                                self.stdout_log.extend_from_slice(&buffer[..count]);
                                for &byte in &buffer[..count] {
                                    if byte == b'\n' {
                                        if self.pending_line.last() == Some(&b'\r') {
                                            self.pending_line.pop();
                                        }
                                        let text = std::str::from_utf8(&self.pending_line)
                                            .map_err(|_| error("stdout", "invalid_utf8"))?
                                            .to_owned();
                                        self.pending_line.clear();
                                        if self.lines.len() >= 256 {
                                            return Err(error("stdout", "pending_line_budget"));
                                        }
                                        if text.chars().any(|c| c.is_control() && c != '\t') {
                                            return Err(error("stdout", "control_character"));
                                        }
                                        self.lines.push_back(text);
                                    } else {
                                        if self.pending_line.len() >= self.max_line {
                                            return Err(error("stdout", "line_budget"));
                                        }
                                        self.pending_line.push(byte);
                                    }
                                }
                            } else {
                                self.stderr_log.extend_from_slice(&buffer[..count]);
                            }
                        }
                        Err(failure) if failure.kind() == ErrorKind::WouldBlock => break,
                        Err(failure) if failure.kind() == ErrorKind::Interrupted => continue,
                        Err(_) => return Err(error("output", "pipe_read")),
                    }
                }
            }
            Ok(())
        }
        pub(super) fn next_line(
            &mut self,
            deadline: Instant,
            cancel: &AtomicBool,
        ) -> Result<Option<String>, CheckerError> {
            self.next_line_controlled(deadline, cancel, None)
        }
        pub(super) fn next_line_controlled(
            &mut self,
            deadline: Instant,
            cancel: &AtomicBool,
            control: Option<&dyn ExternalControl>,
        ) -> Result<Option<String>, CheckerError> {
            loop {
                if let Some(line) = self.lines.pop_front() {
                    return Ok(Some(line));
                }
                if cancel.load(Ordering::Acquire)
                    || Instant::now() >= deadline
                    || control.is_some_and(|control| {
                        control.observe().action != ExternalControlAction::Continue
                    })
                {
                    return Ok(None);
                }
                self.pump(false)?;
                if let Some(line) = self.lines.pop_front() {
                    return Ok(Some(line));
                }
                match self.status()? {
                    WaitStatus::StillAlive => {}
                    _ => return Err(error("process", "unexpected_exit")),
                }
                if self.state.stdout_drained {
                    return Err(error("stdout", "eof_before_response"));
                }
                if other_group_members(self.pid)? > 0 {
                    return Err(error("process", "child_process_limit"));
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        pub(super) fn mark_stop(&mut self) {
            self.state.stop_sent = true;
        }
        pub(super) fn output(&self) -> (&[u8], &[u8]) {
            (&self.stdout_log, &self.stderr_log)
        }
        pub(super) fn evidence(&self) -> CheckerShutdown {
            self.state
        }
        pub(super) fn shutdown(
            &mut self,
            deadline: Instant,
            active: bool,
        ) -> Result<CheckerShutdown, CheckerError> {
            let result = self.shutdown_inner(deadline, active);
            if result.is_err() {
                self.state.quarantined = true;
            }
            result
        }
        fn shutdown_inner(
            &mut self,
            deadline: Instant,
            active: bool,
        ) -> Result<CheckerShutdown, CheckerError> {
            if self.state.cleanup_complete {
                return Ok(self.state);
            }
            if !self.authority {
                self.state.quarantined = true;
                return Err(error("shutdown", "ownership_lost"));
            }
            if active && self.send("stop", deadline).is_ok() {
                self.state.stop_sent = true;
            }
            if self.send("quit", deadline).is_ok() {
                self.state.quit_sent = true;
            }
            self.stdin.take();
            let begin = Instant::now();
            let mut killed = false;
            let kill_at = begin
                .checked_add(deadline.saturating_duration_since(begin) / 2)
                .unwrap_or(begin);
            while Instant::now() < deadline {
                // Drain without retaining more output after any failure. Byte
                // observations remain actual reads; discarded content is not a
                // fabricated full transcript or a claim about unread bytes.
                let _ = self.pump(true);
                let status = self.status()?;
                let others = other_group_members(self.pid)?;
                if !matches!(status, WaitStatus::StillAlive) {
                    self.state.exit_observed = true;
                    match status {
                        WaitStatus::Exited(_, code) => self.state.exit_code = Some(code),
                        WaitStatus::Signaled(_, signal, _) => {
                            self.state.exit_signal = Some(signal as i32)
                        }
                        _ => {}
                    }
                    if others == 0 && self.state.stdout_drained && self.state.stderr_drained {
                        // The unreaped leader reserves PID/PGID until this final
                        // check. No numeric group operation occurs after reap.
                        self.child
                            .as_mut()
                            .ok_or_else(|| error("shutdown", "missing_child"))?
                            .wait()
                            .map_err(|_| error("shutdown", "leader_reap"))?;
                        self.child.take();
                        self.authority = false;
                        self.state.cleanup_complete = true;
                        self.state.quarantined = false;
                        self.process_slot.complete_physical();
                        return Ok(self.state);
                    }
                }
                if !killed && (others > 0 || Instant::now() >= kill_at) {
                    self.signal(Signal::SIGKILL)?;
                    killed = true;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            self.state.quarantined = true;
            Err(error("shutdown", "cleanup_unconfirmed"))
        }
    }
    impl Drop for OwnedProcess {
        fn drop(&mut self) {
            if self.state.cleanup_complete {
                return;
            }
            let deadline = Instant::now()
                .checked_add(self.shutdown_grace)
                .unwrap_or_else(Instant::now);
            if !self.state.quarantined {
                let _ = self.shutdown(deadline, true);
            }
            if !self.state.cleanup_complete {
                // Preserve an unreaped handle; a failed owner cannot create a
                // replacement. No kill/reap/drop follows lost PID authority.
                if let Some(child) = self.child.take() {
                    std::mem::forget(child);
                }
                // The ProcessLease drops as unconfirmed and keeps its original
                // singleton/group gate occupied. No reservation is refunded.
            }
        }
    }

    #[cfg(test)]
    mod proc_identity_tests {
        use super::*;
        use std::io::Cursor;

        fn stat_bytes() -> Vec<u8> {
            let mut fields = ["0"; 20];
            fields[0] = "S";
            fields[2] = "41";
            fields[19] = "9001";
            format!("123 (proc fixture) {}\n", fields.join(" ")).into_bytes()
        }

        struct ReadThenError {
            bytes: Cursor<Vec<u8>>,
            errno: i32,
        }
        impl Read for ReadThenError {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                let count = self.bytes.read(buffer)?;
                if count != 0 {
                    Ok(count)
                } else {
                    Err(std::io::Error::from_raw_os_error(self.errno))
                }
            }
        }

        #[test]
        fn vanished_proc_stat_open_or_read_returns_no_identity() {
            for errno in [nix::libc::ENOENT, nix::libc::ESRCH] {
                let opened: std::io::Result<Cursor<Vec<u8>>> =
                    Err(std::io::Error::from_raw_os_error(errno));
                assert!(read_proc_identity(opened).unwrap().is_none());
                let reader = ReadThenError {
                    bytes: Cursor::new(Vec::new()),
                    errno,
                };
                assert!(read_proc_identity(Ok(reader)).unwrap().is_none());
            }
        }

        #[test]
        fn vanished_proc_stat_discards_even_parseable_partial_bytes() {
            for errno in [nix::libc::ENOENT, nix::libc::ESRCH] {
                let reader = ReadThenError {
                    bytes: Cursor::new(stat_bytes()),
                    errno,
                };
                assert!(read_proc_identity(Ok(reader)).unwrap().is_none());
            }
        }

        #[test]
        fn proc_stat_permission_and_io_errors_remain_failures() {
            for errno in [nix::libc::EACCES, nix::libc::EIO] {
                let opened: std::io::Result<Cursor<Vec<u8>>> =
                    Err(std::io::Error::from_raw_os_error(errno));
                assert_eq!(
                    read_proc_identity(opened).err(),
                    Some(error("ownership", "proc_stat_unavailable"))
                );
                for bytes in [Vec::new(), stat_bytes()] {
                    let reader = ReadThenError {
                        bytes: Cursor::new(bytes),
                        errno,
                    };
                    assert_eq!(
                        read_proc_identity(Ok(reader)).err(),
                        Some(error("ownership", "proc_stat_read"))
                    );
                }
            }
        }

        #[test]
        fn proc_stat_encoding_and_format_errors_remain_failures() {
            for (bytes, code) in [
                (vec![0xff], "proc_stat_encoding"),
                (b"missing stat delimiter".to_vec(), "proc_stat_format"),
                (b"123 (fixture) S 1 invalid".to_vec(), "proc_group"),
                (b"123 (fixture) S 1 41".to_vec(), "proc_start"),
            ] {
                assert_eq!(
                    read_proc_identity(Ok(Cursor::new(bytes))).err(),
                    Some(error("ownership", code))
                );
            }
        }

        #[test]
        fn proc_stat_keeps_4096_byte_parse_and_bounded_overflow() {
            let mut bytes = stat_bytes();
            bytes.resize(4096, b' ');
            let identity = read_proc_identity(Ok(Cursor::new(bytes.clone())))
                .unwrap()
                .unwrap();
            assert_eq!(identity.group, 41);
            assert_eq!(identity.start, 9001);
            bytes.resize(8192, b' ');
            let mut reader = Cursor::new(bytes);
            assert_eq!(
                read_proc_identity(Ok(&mut reader)).err(),
                Some(error("ownership", "proc_stat_budget"))
            );
            assert_eq!(reader.position(), 4097);
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod process {
    use super::*;
    pub(super) struct OwnedProcess;
    pub(super) struct SpawnFailure {
        pub(super) error: CheckerError,
        pub(super) process: CheckerShutdown,
        pub(super) stdout: Vec<u8>,
        pub(super) stderr: Vec<u8>,
    }
    impl OwnedProcess {
        pub(super) fn spawn(
            _config: &ExternalCpuConfig,
            _deadline: Instant,
            _cancel: &AtomicBool,
            _launch_pin: Option<&ExternalLaunchPin>,
        ) -> Result<Self, Box<SpawnFailure>> {
            Err(Box::new(SpawnFailure {
                error: CheckerError::Unsupported("external UCI checker requires Linux/WSL"),
                process: empty_shutdown(),
                stdout: Vec::new(),
                stderr: Vec::new(),
            }))
        }
        pub(super) fn send(&mut self, _line: &str, _deadline: Instant) -> Result<(), CheckerError> {
            Err(CheckerError::Unsupported("external UCI Linux process"))
        }
        pub(super) fn next_line(
            &mut self,
            _deadline: Instant,
            _cancel: &AtomicBool,
        ) -> Result<Option<String>, CheckerError> {
            Err(CheckerError::Unsupported("external UCI Linux process"))
        }
        pub(super) fn next_line_controlled(
            &mut self,
            _deadline: Instant,
            _cancel: &AtomicBool,
            _control: Option<&dyn ExternalControl>,
        ) -> Result<Option<String>, CheckerError> {
            Err(CheckerError::Unsupported("external UCI Linux process"))
        }
        pub(super) fn mark_stop(&mut self) {}
        pub(super) fn output(&self) -> (&[u8], &[u8]) {
            (&[], &[])
        }
        pub(super) fn evidence(&self) -> CheckerShutdown {
            empty_shutdown()
        }
        pub(super) fn shutdown(
            &mut self,
            _deadline: Instant,
            _active: bool,
        ) -> Result<CheckerShutdown, CheckerError> {
            Err(CheckerError::Unsupported("external UCI Linux process"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_budget_reserves_full_stop_and_ready_window_inside_parent_wall() {
        let start = Instant::now();
        let until = start + Duration::from_secs(2);
        let reserve = Duration::from_secs(1);
        let budget = ExternalTaskBudget::new(start, until, reserve).unwrap();
        assert_eq!(budget.task_deadline, until);
        assert_eq!(budget.analysis_deadline, start + Duration::from_secs(1));
        assert_eq!(budget.analysis_millis(start).unwrap(), 1000);
        assert_eq!(
            budget.drain_deadline(budget.analysis_deadline).unwrap(),
            until
        );
        assert_eq!(
            budget
                .drain_deadline(budget.analysis_deadline + Duration::from_millis(5))
                .unwrap(),
            until
        );
        assert!(matches!(
            ExternalTaskBudget::new(start, start + reserve, reserve),
            Err(CheckerError::External {
                stage: "admission",
                code: "insufficient_stop_reserve"
            })
        ));
        assert!(matches!(
            ExternalTaskBudget::new(start, start + reserve + Duration::from_micros(999), reserve),
            Err(CheckerError::External {
                code: "insufficient_stop_reserve",
                ..
            })
        ));
    }

    #[test]
    fn external_scores_remain_native_units_bounds_and_unknown_work() {
        let cp = parse_info("info depth 8 seldepth 14 nodes 432 score cp -731 upperbound wdl 10 200 790 pv e2e4 e7e5", 16).unwrap();
        assert_eq!(cp.score, Some(ExternalRawScore::Centipawns(-731)));
        assert_eq!(cp.bound, Some(ExternalBound::Upper));
        assert_eq!(cp.wdl, Some([10, 200, 790]));
        assert_eq!(cp.nodes, Some(432));
        let mate = parse_info("info depth 10 score mate -3 lowerbound pv e2e4", 16).unwrap();
        assert_eq!(mate.score, Some(ExternalRawScore::MateMoves(-3)));
        assert_eq!(mate.nodes, None);
        assert_eq!(mate.bound, Some(ExternalBound::Lower));
        let sparse = parse_info("info nodes 0 string score cp 99 pv e2e4", 16).unwrap();
        assert_eq!(sparse.nodes, Some(0));
        assert_eq!(sparse.score, None);
        assert!(parse_info("info score mate NaN pv e2e4", 16).is_err());
        assert!(parse_info("info wdl 1 2 3 pv e2e4", 16).is_err());
    }
    #[test]
    fn root_mask_and_rules_pv_are_independently_checked() {
        let p = Position::startpos();
        let e4 = BoardMove::from_uci("e2e4").unwrap();
        let d4 = BoardMove::from_uci("d2d4").unwrap();
        validate_pv(&p, &[e4], Some(&[e4])).unwrap();
        assert!(validate_pv(&p, &[d4], Some(&[e4])).is_err());
        assert!(validate_pv(&p, &[e4, d4], None).is_err());
    }
    #[test]
    fn option_ranges_and_argument_boundaries_are_not_inferred() {
        let (_, kind) =
            parse_option("option name Threads type spin default 1 min 1 max 1024").unwrap();
        assert!(validate_option(&kind, "2"));
        assert!(!validate_option(&kind, "0"));
        assert!(!validate_option(&kind, "2\nquit"));
        let (_, combo) =
            parse_option("option name NumaPolicy type combo default auto var auto var none")
                .unwrap();
        assert!(validate_option(&combo, "none"));
        assert!(!validate_option(&combo, "unknown"));
        assert_ne!(
            arguments_sha256(&["ab".into(), "c".into()]),
            arguments_sha256(&["a".into(), "bc".into()])
        );
    }

    #[cfg(target_os = "linux")]
    pub(crate) static FIXTURE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[cfg(target_os = "linux")]
    #[test]
    fn sealed_teacher_pin_preserves_bytes_and_original_process_lifetime() {
        use std::os::unix::fs::FileExt;
        let _guard = FIXTURE_LOCK.lock().unwrap();
        let config = fixture_config("fake_uci_normal");
        let original = File::open(&config.program).unwrap();
        let expected = &config.identity.binary_sha256;
        let mut checks = 0;
        let token = VerifiedSealedExecutable::prepare(&original, expected, || {
            checks += 1;
            Ok(())
        }).unwrap();
        let probe = token.execution_probe();
        let sealed_file = Arc::downgrade(&token.file);
        assert!(checks > 1);
        assert!(token.file.write_at(b"changed", 0).is_err());
        assert!(VerifiedSealedExecutable::prepare(&original, &"0".repeat(64), || Ok(())).is_err());
        assert!(VerifiedSealedExecutable::prepare(&original, expected, || {
            Err(CheckerError::Invalid("cancelled preparation"))
        }).is_err());
        let cwd = File::open(&config.working_directory).unwrap();
        let pin = ExternalLaunchPin::from_verified_sealed(token.clone(), cwd).unwrap();
        let mut wrong = config.clone();
        wrong.identity.binary_sha256 = "0".repeat(64);
        let mut denied = ExternalUciCpuChecker::create(wrong).unwrap();
        denied.launch_pin = Some(pin.clone());
        assert!(denied.start(Instant::now() + Duration::from_secs(10), &AtomicBool::new(false)).is_err());
        let mut checker = ExternalUciCpuChecker::create(config).unwrap();
        checker.launch_pin = Some(pin);
        checker.start(Instant::now() + Duration::from_secs(10), &AtomicBool::new(false)).unwrap();
        let closed = checker.shutdown(Instant::now() + Duration::from_secs(3)).unwrap();
        assert!(closed.exit_observed && closed.stdout_drained && closed.stderr_drained);
        assert!(closed.cleanup_complete);
        assert_eq!(token.execution_observations(), (1, 0));
        drop(denied);
        drop(checker);
        drop(token);
        assert!(sealed_file.upgrade().is_none());
        assert_eq!(probe.execution_observations(), (1, 0));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn sealed_teacher_copy_does_not_follow_later_original_mutation() {
        use std::io::Write;
        use std::os::unix::fs::{FileExt, PermissionsExt};
        struct Cleanup(std::path::PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); }
        }
        let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("rz-sealed-mutation-{}-{nonce}", std::process::id()));
        let _cleanup = Cleanup(path.clone());
        let bytes = std::fs::read("/bin/true").unwrap();
        let mut out = std::fs::OpenOptions::new().write(true).create_new(true).open(&path).unwrap();
        out.write_all(&bytes).unwrap();
        out.set_permissions(std::fs::Permissions::from_mode(0o700)).unwrap();
        drop(out);
        let original = File::open(&path).unwrap();
        let expected = format!("{:x}", Sha256::digest(&bytes));
        let token = VerifiedSealedExecutable::prepare(&original, &expected, || Ok(())).unwrap();
        let writer = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        writer.write_at(b"changed", 0).unwrap();
        let mut sealed_header = [0u8; 4];
        token.file.read_exact_at(&mut sealed_header, 0).unwrap();
        assert_eq!(sealed_header, *b"\x7fELF");
        assert!(VerifiedSealedExecutable::prepare(&original, &expected, || Ok(())).is_err());
    }

    #[cfg(target_os = "linux")]
    fn fixture_mode(name: &str) -> bool {
        let args: Vec<_> = std::env::args().collect();
        args.iter().any(|arg| arg == "--exact") && args.iter().any(|arg| arg == name)
    }

    #[cfg(target_os = "linux")]
    fn fixture_loop(mode: u8) {
        use std::io::{BufRead, Write};
        let mut position = Position::startpos();
        let mut readiness_count = 0usize;
        let mut pending_best = None;
        let mut multipv = 1_u16;
        for line in std::io::stdin().lock().lines() {
            let line = line.expect("fixture stdin");
            if line == "uci" {
                println!("id name RZ external checker fixture");
                println!("id author RoveZero tests");
                println!("option name Threads type spin default 1 min 1 max 2");
                println!("option name MultiPV type spin default 1 min 1 max 4");
                println!("uciok");
            } else if let Some(value) = line.strip_prefix("setoption name MultiPV value ") {
                multipv = value.parse().unwrap();
            } else if line == "isready" {
                readiness_count += 1;
                if mode == 3 && readiness_count > 1 {
                    println!("uciok");
                } else {
                    println!("readyok");
                }
            } else if let Some(body) = line.strip_prefix("position startpos") {
                position = Position::startpos();
                if let Some(moves) = body.trim().strip_prefix("moves ") {
                    for mv in moves.split_ascii_whitespace() {
                        position
                            .make_move(BoardMove::from_uci(mv).expect("fixture move"))
                            .expect("fixture legal replay");
                    }
                }
            } else if line.starts_with("go ") {
                let words: Vec<&str> = line.split_ascii_whitespace().collect();
                let depth = words
                    .iter()
                    .position(|word| *word == "depth")
                    .map(|at| words[at + 1])
                    .unwrap_or("2");
                let requested: u64 = words
                    .iter()
                    .position(|word| *word == "nodes")
                    .map(|at| words[at + 1].parse().unwrap())
                    .unwrap_or(10);
                let best = words
                    .iter()
                    .position(|word| *word == "searchmoves")
                    .map(|at| BoardMove::from_uci(words[at + 1]).unwrap())
                    .unwrap_or_else(|| position.legal_moves()[0]);
                let nodes = if mode == 1 {
                    requested + 1
                } else {
                    requested.min(10)
                };
                if mode == 2 {
                    println!("info depth {depth} nodes {nodes} score cp NaN pv {best}");
                } else {
                    println!("info depth {depth} seldepth 12 nodes {nodes} score cp 21 pv {best}");
                    if matches!(mode, 6 | 7 | 8 | 9) {
                        let legal = position.legal_moves();
                        for rank in 2..=multipv {
                            let other = legal
                                .iter()
                                .filter(|mv| **mv != best)
                                .nth(usize::from(rank - 2))
                                .unwrap();
                            let depth = if mode == 7 { "1" } else { depth };
                            println!(
                                "info depth {depth} multipv {rank} nodes {nodes} score cp 11 pv {other}"
                            );
                        }
                    }
                    if matches!(mode, 8 | 9) {
                        let next_depth = depth.parse::<u16>().unwrap() + 1;
                        println!("info depth {next_depth} multipv 1 nodes {nodes} score cp 32 lowerbound pv {best}");
                    }
                }
                if mode == 4 || mode == 5 {
                    pending_best = Some(best);
                } else if mode == 9 {
                    let other = position.legal_moves().into_iter().find(|mv| *mv != best).unwrap();
                    println!("bestmove {other}");
                } else {
                    println!("bestmove {best}");
                }
            } else if line == "stop" {
                if mode == 4
                    && let Some(best) = pending_best.take()
                {
                    println!("bestmove {best}");
                }
            } else if line == "quit" {
                break;
            }
            std::io::stdout().flush().unwrap();
        }
    }

    // These are native test-binary subprocess modes, not shell/Python engines.
    // Ordinary all-target test execution returns immediately in these tests.
    #[cfg(target_os = "linux")]
    #[test]
    fn fake_uci_normal() {
        if fixture_mode("external_cpu::tests::fake_uci_normal") {
            fixture_loop(0);
        }
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn fake_uci_overshoot() {
        if fixture_mode("external_cpu::tests::fake_uci_overshoot") {
            fixture_loop(1);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn fake_uci_parse_failure() {
        if fixture_mode("external_cpu::tests::fake_uci_parse_failure") {
            fixture_loop(2);
        }
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn fake_uci_pre_go_failure() {
        if fixture_mode("external_cpu::tests::fake_uci_pre_go_failure") {
            fixture_loop(3);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn fake_uci_waits_for_stop() {
        if fixture_mode("external_cpu::tests::fake_uci_waits_for_stop") {
            fixture_loop(4);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn fake_uci_ignores_stop() {
        if fixture_mode("external_cpu::tests::fake_uci_ignores_stop") {
            fixture_loop(5);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn fake_uci_multipv() {
        if fixture_mode("external_cpu::tests::fake_uci_multipv") {
            fixture_loop(6);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn fake_uci_mixed_multipv() {
        if fixture_mode("external_cpu::tests::fake_uci_mixed_multipv") {
            fixture_loop(7);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn fake_uci_final_bound() {
        if fixture_mode("external_cpu::tests::fake_uci_final_bound") { fixture_loop(8); }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn fake_uci_final_bound_changed_best() {
        if fixture_mode("external_cpu::tests::fake_uci_final_bound_changed_best") { fixture_loop(9); }
    }

    #[cfg(target_os = "linux")]
    struct FixtureControl(rz_runtime::adaptive::ControlMailbox<u64>);

    #[cfg(target_os = "linux")]
    impl ExternalControl for FixtureControl {
        fn observe(&self) -> ExternalControlObservation {
            let snapshot = self.0.snapshot();
            if snapshot.is_cancelled() {
                ExternalControlObservation {
                    sequence: snapshot.cancel_sequence,
                    action: ExternalControlAction::Cancel,
                }
            } else if let Some(sequence) = snapshot.pending_yield() {
                ExternalControlObservation {
                    sequence,
                    action: ExternalControlAction::Yield,
                }
            } else {
                ExternalControlObservation {
                    sequence: 0,
                    action: ExternalControlAction::Continue,
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn cooperative_fixture(mode: &str) -> ExternalUciCpuChecker {
        let mut config = fixture_config(mode);
        config.stop_grace = EXTERNAL_STOP_DRAIN_RESERVE;
        let mut checker = ExternalUciCpuChecker::create(config).unwrap();
        checker
            .start(
                Instant::now() + Duration::from_secs(10),
                &AtomicBool::new(false),
            )
            .unwrap();
        checker
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cooperative_quantum_yields_partial_and_ready_before_a_second_go() {
        let _guard = FIXTURE_LOCK.lock().unwrap();
        let mut checker = cooperative_fixture("fake_uci_waits_for_stop");
        let cancel = AtomicBool::new(false);
        for _ in 0..2 {
            let control = FixtureControl(rz_runtime::adaptive::ControlMailbox::new(
                1,
                rz_runtime::CompletionSignal::default(),
            ));
            let hard = Instant::now() + Duration::from_secs(2);
            let result = checker
                .analyze_cooperative(
                    &Position::startpos(),
                    None,
                    CpuLimits {
                        max_depth: 8,
                        max_nodes: 1,
                        deadline: Some(hard),
                    },
                    &cancel,
                    &control,
                    ExternalSlicePolicy::default(),
                )
                .unwrap();
            assert_eq!(result.report.completion, ExternalCompletion::StoppedYield);
            assert_eq!(result.strict_native_nodes, None);
            assert_eq!(result.report.work.nodes, Some(10));
            assert!(result.readiness_barrier_completed);
            assert!(result.stop_control.is_none()); // quantum, not a fabricated control ack
            let attempt = checker.last_attempt().unwrap();
            assert_eq!(attempt.elapsed, result.report.elapsed);
            assert_eq!(
                attempt.external.as_ref().unwrap().partial_report.as_ref(),
                Some(&result.report)
            );
            assert!(attempt.elapsed > Duration::ZERO);
            assert!(Instant::now() < hard);
        }
        assert_eq!(checker.request_id, 2);
        assert!(
            checker
                .shutdown(Instant::now() + Duration::from_secs(2))
                .unwrap()
                .cleanup_complete
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cooperative_reported_overshoot_remains_raw_and_unknown_native_credit() {
        let _guard = FIXTURE_LOCK.lock().unwrap();
        let mut checker = cooperative_fixture("fake_uci_overshoot");
        let control = FixtureControl(rz_runtime::adaptive::ControlMailbox::new(
            1,
            rz_runtime::CompletionSignal::default(),
        ));
        let result = checker
            .analyze_cooperative(
                &Position::startpos(),
                None,
                CpuLimits {
                    max_depth: 8,
                    max_nodes: 5,
                    deadline: Some(Instant::now() + Duration::from_secs(2)),
                },
                &AtomicBool::new(false),
                &control,
                ExternalSlicePolicy {
                    requested_reported_nodes: 5,
                    ..ExternalSlicePolicy::default()
                },
            )
            .unwrap();
        assert_eq!(result.report.work.nodes, Some(6));
        assert!(result.reported_node_overshoot);
        assert_eq!(result.strict_native_nodes, None);
        assert_eq!(result.report.completion, ExternalCompletion::BestMove);
        checker
            .shutdown(Instant::now() + Duration::from_secs(2))
            .unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cooperative_explicit_yield_is_not_cancel_or_an_exact_stack_resume() {
        let _guard = FIXTURE_LOCK.lock().unwrap();
        let mut checker = cooperative_fixture("fake_uci_waits_for_stop");
        let mailbox =
            rz_runtime::adaptive::ControlMailbox::new(1, rz_runtime::CompletionSignal::default());
        let request = mailbox.clone();
        let driver = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(25));
            request.request_yield(7).unwrap();
        });
        let control = FixtureControl(mailbox);
        let result = checker
            .analyze_cooperative(
                &Position::startpos(),
                None,
                CpuLimits {
                    max_depth: 8,
                    max_nodes: 100,
                    deadline: Some(Instant::now() + Duration::from_secs(2)),
                },
                &AtomicBool::new(false),
                &control,
                ExternalSlicePolicy::default(),
            )
            .unwrap();
        driver.join().unwrap();
        assert_eq!(result.report.completion, ExternalCompletion::StoppedYield);
        assert_eq!(
            result.stop_control,
            Some(ExternalControlObservation {
                sequence: 7,
                action: ExternalControlAction::Yield
            })
        );
        assert!(result.readiness_barrier_completed);
        let attempt = checker.last_attempt().unwrap();
        assert_eq!(attempt.elapsed, result.report.elapsed);
        assert_eq!(
            attempt.external.as_ref().unwrap().partial_report.as_ref(),
            Some(&result.report)
        );
        assert!(!checker.capabilities().resume);
        checker
            .shutdown(Instant::now() + Duration::from_secs(2))
            .unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cooperative_failed_stop_cannot_reopen_an_expired_physical_fence() {
        struct ShortFence(Instant);
        impl ExternalControl for ShortFence {
            fn observe(&self) -> ExternalControlObservation {
                ExternalControlObservation {
                    sequence: 0,
                    action: ExternalControlAction::Continue,
                }
            }
            fn physical_fence_deadline(&self) -> Option<Instant> {
                Some(self.0)
            }
        }
        let _guard = FIXTURE_LOCK.lock().unwrap();
        let mut checker = cooperative_fixture("fake_uci_ignores_stop");
        let control = ShortFence(Instant::now() + Duration::from_millis(120));
        let result = checker.analyze_cooperative(
            &Position::startpos(),
            None,
            CpuLimits {
                max_depth: 8,
                max_nodes: 100,
                deadline: Some(Instant::now() + Duration::from_secs(2)),
            },
            &AtomicBool::new(false),
            &control,
            ExternalSlicePolicy::default(),
        );
        assert!(result.is_err());
        let process = checker
            .last_attempt()
            .unwrap()
            .external
            .as_ref()
            .unwrap()
            .process;
        assert!(process.quarantined);
        assert!(!process.cleanup_complete);
        // A renewed cleanup grace would issue quit after the expired fence.
        assert!(!process.quit_sent);
        assert_eq!(checker.go_issue, ExternalGoIssue::Issued);
        // An explicit, separately registered global cleanup can drain the
        // retained owner; the failed task does not fabricate that proof.
        assert!(
            checker
                .shutdown(Instant::now() + Duration::from_secs(2))
                .unwrap()
                .cleanup_complete
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn completed_exact_iteration_survives_final_bound_without_fabricating_completion() {
        let _guard = FIXTURE_LOCK.lock().unwrap();
        for (name, width, expected) in [
            ("fake_uci_final_bound", 1, true),
            ("fake_uci_final_bound", 4, true),
            ("fake_uci_final_bound_changed_best", 1, false),
            ("fake_uci_final_bound_changed_best", 4, false),
            ("fake_uci_mixed_multipv", 4, false),
            ("fake_uci_waits_for_stop", 1, false),
        ] {
            let mut checker = cooperative_fixture(name);
            let control = FixtureControl(rz_runtime::adaptive::ControlMailbox::new(
                1, rz_runtime::CompletionSignal::default()));
            let report = checker.analyze_cooperative(
                &Position::startpos(), None,
                CpuLimits { max_depth: 8, max_nodes: 100, deadline: Some(Instant::now() + Duration::from_secs(2)) },
                &AtomicBool::new(false), &control,
                ExternalSlicePolicy { multipv: width, ..ExternalSlicePolicy::default() },
            ).unwrap();
            assert_eq!(report.policy_target_known, expected, "{name} width={width}");
            if name == "fake_uci_final_bound" {
                assert_eq!(report.report.bound, ExternalBound::Lower);
                assert!(report.opinions.iter().all(|opinion| opinion.bound == ExternalBound::ExactReported && opinion.depth == 8));
            }
            if name == "fake_uci_waits_for_stop" {
                assert_ne!(report.report.completion, ExternalCompletion::BestMove);
            }
            checker.shutdown(Instant::now() + Duration::from_secs(2)).unwrap();
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn coherent_multipv_known_and_mixed_depth_masked_without_value_labels() {
        let _guard = FIXTURE_LOCK.lock().unwrap();
        for (mode, expected) in [
            ("fake_uci_multipv", true),
            ("fake_uci_mixed_multipv", false),
        ] {
            let mut checker = cooperative_fixture(mode);
            let control = FixtureControl(rz_runtime::adaptive::ControlMailbox::new(
                1,
                rz_runtime::CompletionSignal::default(),
            ));
            let report = checker
                .analyze_cooperative(
                    &Position::startpos(),
                    None,
                    CpuLimits {
                        max_depth: 8,
                        max_nodes: 100,
                        deadline: Some(Instant::now() + Duration::from_secs(2)),
                    },
                    &AtomicBool::new(false),
                    &control,
                    ExternalSlicePolicy {
                        multipv: 4,
                        ..ExternalSlicePolicy::default()
                    },
                )
                .unwrap();
            assert_eq!(report.opinions.len(), 4);
            assert_eq!(report.policy_target_known, expected);
            assert_eq!(report.conditional_rank_policy().is_some(), expected);
            assert_eq!(report.report.wdl_per_mille, None);
            if expected {
                assert_eq!(checker.last_requested_multipv, Some(4));
                checker
                    .analyze(
                        &Position::startpos(),
                        CpuLimits {
                            max_depth: 8,
                            max_nodes: 100,
                            deadline: Some(Instant::now() + Duration::from_secs(2)),
                        },
                        &AtomicBool::new(false),
                    )
                    .unwrap();
                assert_eq!(checker.last_requested_multipv, Some(1));
            }
            let commands = std::str::from_utf8(checker.diagnostic_output().0).unwrap();
            assert!(!commands.is_empty());
            checker
                .shutdown(Instant::now() + Duration::from_secs(2))
                .unwrap();
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn initial_game_binding_cannot_cross_workers_or_survive_root_quiesce() {
        use rz_contracts::pals::SearchAuthority;
        use rz_contracts::pals_adaptive::CpuWorkerId;
        use rz_contracts::{
            ClockDomain, Digest as ContractDigest, GameGeneration, ProcessEpoch, RootGeneration,
        };
        use worker::*;
        let _guard = FIXTURE_LOCK.lock().unwrap();
        let mut config = fixture_config("fake_uci_waits_for_stop");
        config.stop_grace = EXTERNAL_STOP_DRAIN_RESERVE;
        let epoch = ProcessEpoch(29);
        let clock = ExternalWorkerClock {
            domain: ClockDomain(epoch),
            origin: Instant::now(),
        };
        let until = Instant::now() + Duration::from_secs(10);
        fn ready(worker: &ExternalCpuWorker, until: Instant) {
            while worker.startup_report().is_none() {
                let version = worker.signal().version();
                assert!(Instant::now() < until);
                worker
                    .signal()
                    .wait_changed(version, Duration::from_millis(20));
            }
            assert!(worker.startup_report().unwrap().result.is_ok());
        }
        fn close(worker: &mut ExternalCpuWorker, until: Instant) {
            worker.request_shutdown(until);
            let closed = loop {
                let version = worker.signal().version();
                if let Some(closed) = worker.try_join() {
                    break closed.unwrap();
                }
                assert!(Instant::now() < until);
                worker
                    .signal()
                    .wait_changed(version, Duration::from_millis(20));
            };
            assert!(closed.cleanup_complete && closed.exit_observed && !closed.quarantined);
        }
        let mut first =
            ExternalCpuWorker::spawn(config.clone(), CpuWorkerId(0), clock, until).unwrap();
        ready(&first, until);
        let authority = SearchAuthority {
            epoch,
            game: GameGeneration(1),
            root: RootGeneration(1),
            implementation: ContractDigest([7; 32]),
        };
        let proof = first
            .bind_initial_game(authority, until, &AtomicBool::new(false))
            .unwrap()
            .unwrap();
        assert!(
            first
                .bind_initial_game(authority, until, &AtomicBool::new(false))
                .unwrap()
                .is_none()
        );
        assert!(first.root_fence().is_none());
        assert!(first.new_game_fence().is_none());
        // The production process slot admits one external child at a time.
        // Keep only the stale token while fully closing the original owner.
        close(&mut first, until);
        let mut second = ExternalCpuWorker::spawn(config, CpuWorkerId(1), clock, until).unwrap();
        ready(&second, until);
        assert!(second.consume_initial_game_binding(proof).is_err());

        let proof = second
            .bind_initial_game(authority, until, &AtomicBool::new(false))
            .unwrap()
            .unwrap();
        second.request_root_quiesce(authority, until).unwrap();
        assert!(second.consume_initial_game_binding(proof).is_err());
        let fence = loop {
            let version = second.signal().version();
            if let Some(fence) = second.root_fence() {
                break fence.unwrap();
            }
            assert!(Instant::now() < until);
            second
                .signal()
                .wait_changed(version, Duration::from_millis(20));
        };
        assert!(fence.permits_reuse());
        assert_eq!(fence.execution_high_water(), 0);
        assert_eq!(fence.game_reset_high_water(), 0);
        assert!(second.new_game_fence().is_none());
        close(&mut second, until);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn external_private_worker_seals_progress_report_and_joins_owned_process() {
        use rz_contracts::pals::SearchAuthority;
        use rz_contracts::pals_adaptive::{
            CpuJobId, CpuWorkerId, ExternalCpuJobEnvelope, ExternalSliceGrant,
        };
        use rz_contracts::{
            ClockDomain, Deadline, Digest as ContractDigest, ExecutionId, GameGeneration,
            MonotonicTick, OwnerId, ProcessEpoch, RootGeneration,
        };
        use worker::*;
        let _guard = FIXTURE_LOCK.lock().unwrap();
        let mut config = fixture_config("fake_uci_waits_for_stop");
        config.stop_grace = EXTERNAL_STOP_DRAIN_RESERVE;
        let epoch = ProcessEpoch(9);
        let owner = CpuWorkerId(0);
        let clock = ExternalWorkerClock {
            domain: ClockDomain(epoch),
            origin: Instant::now(),
        };
        let startup_deadline = Instant::now() + Duration::from_secs(10);
        let mut worker =
            ExternalCpuWorker::spawn(config.clone(), owner, clock, startup_deadline).unwrap();
        while worker.startup_report().is_none() {
            let version = worker.signal().version();
            assert!(Instant::now() < startup_deadline);
            worker
                .signal()
                .wait_changed(version, Duration::from_millis(20));
        }
        assert!(worker.startup_report().unwrap().result.is_ok());
        let position = Position::startpos();
        let state_identity =
            rz_position::contracts::ContractPosition::new(OwnerId(43), position.clone())
                .export()
                .unwrap()
                .snapshot()
                .identity();
        let roots = external_root_restriction_digest(None);
        let evaluation = external_evaluation_namespace(&config.identity).unwrap();
        let policy = ExternalSlicePolicy::default();
        let conditions = external_search_conditions(&config, 8, policy);
        let job = CpuJobId {
            authority: SearchAuthority {
                epoch,
                game: GameGeneration(1),
                root: RootGeneration(1),
                implementation: ContractDigest([3; 32]),
            },
            sequence: 1,
        };
        let execution = ExecutionId::new(epoch, 1);
        let scope = ExternalWorkerScope {
            job,
            execution,
            owner,
        };
        let first_until = Instant::now() + Duration::from_secs(3);
        assert!(
            worker
                .bind_initial_game(job.authority, first_until, &AtomicBool::new(true))
                .is_err()
        );
        assert!(
            worker
                .bind_initial_game(job.authority, Instant::now(), &AtomicBool::new(false))
                .is_err()
        );
        let wrong_epoch = SearchAuthority {
            epoch: ProcessEpoch(epoch.0 + 1),
            ..job.authority
        };
        assert!(
            worker
                .bind_initial_game(wrong_epoch, first_until, &AtomicBool::new(false))
                .is_err()
        );
        let initial = worker
            .bind_initial_game(job.authority, first_until, &AtomicBool::new(false))
            .unwrap()
            .unwrap();
        assert_eq!(initial.authority(), job.authority);
        assert_eq!(initial.owner(), owner);
        assert_eq!(initial.source(), evaluation);
        assert!(initial.process_identity().pid > 0);
        assert!(worker.root_fence().is_none());
        assert!(worker.new_game_fence().is_none());
        assert!(
            worker
                .bind_initial_game(job.authority, first_until, &AtomicBool::new(false))
                .unwrap()
                .is_none()
        );
        worker.consume_initial_game_binding(initial).unwrap();
        // First-game binding is not an old-root fence. The real subsequent
        // game reset must still wait for actual root quiesce/ready evidence.
        assert!(matches!(
            worker.request_new_game(
                SearchAuthority {
                    game: GameGeneration(2),
                    root: RootGeneration(2),
                    ..job.authority
                },
                first_until,
            ),
            Err(CheckerError::External {
                stage: "new_game",
                code: "actual_root_fence_missing",
            })
        ));
        let control = rz_runtime::adaptive::ControlMailbox::new(scope, worker.signal());
        control.request_publish_progress(1).unwrap();
        let at = u64::try_from((Instant::now() - clock.origin).as_nanos()).unwrap() + 3_000_000_000;
        let template = ExternalJob {
            envelope: ExternalCpuJobEnvelope {
                id: job,
                state_identity,
                question: external_question_digest(state_identity, roots, evaluation, conditions),
                root_restriction: roots,
                evaluation_namespace: evaluation,
                search_conditions: conditions,
                requested_reported_node_hint: policy.requested_reported_nodes,
                hard_deadline: Deadline {
                    clock: clock.domain,
                    at: MonotonicTick(at),
                },
            },
            grant: ExternalSliceGrant {
                job,
                execution,
                owner,
                requested_reported_node_hint: policy.requested_reported_nodes,
                soft_quantum: policy.quantum,
                control_sequence: 1,
            },
            execution,
            owner,
            position,
            root_moves: None,
            max_depth: 8,
            policy,
            control: control.clone(),
        };
        let submitted = worker.submit_attested(template.clone()).unwrap();
        assert!(
            worker
                .bind_initial_game(job.authority, first_until, &AtomicBool::new(false))
                .unwrap()
                .is_none()
        );
        assert!(submitted.matches_job(&template));
        assert_eq!(submitted.scope(), scope);
        assert_eq!(submitted.evaluation_namespace(), evaluation);
        let mut different_question = template.clone();
        different_question.envelope.question = ContractDigest([9; 32]);
        assert!(!submitted.matches_job(&different_question));
        let until = Instant::now() + Duration::from_secs(3);
        let completion = loop {
            let version = worker.signal().version();
            if let Some(completion) = worker.take_report_attested() {
                break completion;
            }
            assert!(Instant::now() < until);
            worker
                .signal()
                .wait_changed(version, Duration::from_millis(20));
        };
        assert!(completion.matches_dispatch(&submitted));
        assert_eq!(
            completion.process_identity(),
            Some(submitted.process_identity())
        );
        let report = completion.report().clone();
        assert_eq!(report.scope, scope);
        assert_eq!(report.go_issue, ExternalGoIssue::Issued);
        assert_eq!(
            report.result.unwrap().report.completion,
            ExternalCompletion::StoppedYield
        );
        assert_eq!(report.controls.publish_ack, 1);
        assert!(report.controls.terminal);
        let (_, progress) = worker.take_progress().unwrap();
        assert_eq!(progress.scope, scope);
        assert_eq!(progress.strict_native_nodes, None);
        assert_eq!(progress.completed_iteration_observed, None);
        assert!(!report.exact_resume_supported);

        fn retarget(
            template: &ExternalJob,
            authority: SearchAuthority,
            sequence: u64,
            clock: ExternalWorkerClock,
            signal: rz_runtime::CompletionSignal,
        ) -> ExternalJob {
            let mut next = template.clone();
            next.envelope.id = CpuJobId {
                authority,
                sequence,
            };
            next.envelope.hard_deadline.at = MonotonicTick(
                u64::try_from((Instant::now() - clock.origin).as_nanos()).unwrap() + 3_000_000_000,
            );
            next.execution = ExecutionId::new(authority.epoch, sequence);
            next.grant.job = next.envelope.id;
            next.grant.execution = next.execution;
            next.grant.control_sequence = sequence;
            next.control = rz_runtime::adaptive::ControlMailbox::new(next.scope(), signal);
            next
        }
        fn root_fence(worker: &ExternalCpuWorker) -> ExternalRootFence {
            let until = Instant::now() + Duration::from_secs(3);
            loop {
                let version = worker.signal().version();
                if let Some(fence) = worker.root_fence() {
                    return fence.unwrap();
                }
                assert!(Instant::now() < until);
                worker
                    .signal()
                    .wait_changed(version, Duration::from_millis(20));
            }
        }
        fn terminal(worker: &ExternalCpuWorker) -> ExternalWorkerReport {
            let until = Instant::now() + Duration::from_secs(3);
            loop {
                let version = worker.signal().version();
                if let Some(report) = worker.take_report() {
                    return report;
                }
                assert!(Instant::now() < until);
                worker
                    .signal()
                    .wait_changed(version, Duration::from_millis(20));
            }
        }
        let second_authority = SearchAuthority {
            root: RootGeneration(2),
            ..job.authority
        };
        assert!(worker.configuration_matches(&config, owner));
        let mut wrong_config = config.clone();
        wrong_config.max_depth -= 1;
        assert!(!worker.configuration_matches(&wrong_config, owner));
        assert!(worker.replace_root(second_authority).is_err());
        worker
            .request_root_quiesce(job.authority, Instant::now() + Duration::from_secs(3))
            .unwrap();
        assert!(
            worker
                .submit(retarget(
                    &template,
                    second_authority,
                    2,
                    clock,
                    worker.signal()
                ))
                .is_err()
        );
        let first_fence = root_fence(&worker);
        assert!(first_fence.permits_reuse());
        assert_eq!(first_fence.authority(), job.authority);
        assert_eq!(first_fence.owner(), owner);
        assert_eq!(first_fence.execution_high_water(), 1);
        assert_eq!(first_fence.job_high_water(), 1);
        assert_eq!(first_fence.native_request_high_water(), 1);
        assert_eq!(first_fence.game_reset_high_water(), 0);
        assert!(worker.replace_root(job.authority).is_err());
        assert!(
            worker
                .replace_root(SearchAuthority {
                    implementation: ContractDigest([8; 32]),
                    ..second_authority
                })
                .is_err()
        );
        worker.replace_root(second_authority).unwrap();
        assert!(worker.root_fence().is_none());
        assert!(
            worker
                .submit(retarget(
                    &template,
                    job.authority,
                    2,
                    clock,
                    worker.signal()
                ))
                .is_err()
        );
        let second = retarget(&template, second_authority, 2, clock, worker.signal());
        second.control.request_publish_progress(2).unwrap();
        let second_controls = second.control.clone();
        worker.submit(second).unwrap();
        let progress_until = Instant::now() + Duration::from_secs(3);
        loop {
            let version = worker.signal().version();
            if worker.take_progress().is_some() {
                break;
            }
            assert!(Instant::now() < progress_until);
            worker
                .signal()
                .wait_changed(version, Duration::from_millis(20));
        }
        // Explicit cancellation and root revocation may race with the natural
        // soft quantum; both retain their original job and physical barrier.
        let _ = second_controls.request_cancel(3);
        worker
            .request_root_quiesce(second_authority, Instant::now() + Duration::from_secs(3))
            .unwrap();
        let second_report = terminal(&worker);
        assert_eq!(second_report.scope.job.authority, second_authority);
        assert_eq!(second_report.go_issue, ExternalGoIssue::Issued);
        assert!(
            second_report
                .result
                .as_ref()
                .unwrap()
                .readiness_barrier_completed
        );
        let second_fence = root_fence(&worker);
        assert_eq!(
            second_fence.process_identity(),
            first_fence.process_identity()
        );
        assert_eq!(second_fence.native_request_high_water(), 2);
        assert_eq!(second_fence.execution_high_water(), 2);
        assert_eq!(second_fence.job_high_water(), 2);
        assert_eq!(second_fence.game_reset_high_water(), 0);

        let third_authority = SearchAuthority {
            game: GameGeneration(2),
            root: RootGeneration(3),
            ..second_authority
        };
        assert!(worker.replace_root(third_authority).is_err());
        worker
            .request_new_game(third_authority, Instant::now() + Duration::from_secs(3))
            .unwrap();
        let new_game_until = Instant::now() + Duration::from_secs(3);
        let new_game = loop {
            let version = worker.signal().version();
            if let Some(fence) = worker.new_game_fence() {
                break fence.unwrap();
            }
            assert!(Instant::now() < new_game_until);
            worker
                .signal()
                .wait_changed(version, Duration::from_millis(20));
        };
        assert!(new_game.permits_reuse());
        assert_eq!(new_game.previous_authority(), second_authority);
        assert_eq!(new_game.authority(), third_authority);
        assert_eq!(new_game.process_identity(), first_fence.process_identity());
        assert_eq!(new_game.game_reset_high_water(), 1);
        assert!(
            worker
                .replace_root(SearchAuthority {
                    game: GameGeneration(1),
                    ..third_authority
                })
                .is_err()
        );
        worker.replace_root(third_authority).unwrap();
        assert!(worker.new_game_fence().is_none());
        assert!(
            worker
                .request_new_game(third_authority, Instant::now() + Duration::from_secs(3))
                .is_err()
        );
        worker
            .submit(retarget(
                &template,
                third_authority,
                3,
                clock,
                worker.signal(),
            ))
            .unwrap();
        assert!(terminal(&worker).result.is_ok());
        worker
            .request_root_quiesce(third_authority, Instant::now() + Duration::from_secs(3))
            .unwrap();
        let third_fence = root_fence(&worker);
        assert_eq!(
            third_fence.process_identity(),
            first_fence.process_identity()
        );
        assert_eq!(third_fence.native_request_high_water(), 3);
        assert_eq!(third_fence.execution_high_water(), 3);
        assert_eq!(third_fence.job_high_water(), 3);
        assert_eq!(third_fence.game_reset_high_water(), 1);

        // A manager that has not yet consumed a reliable terminal cannot
        // delay closure forever or obtain a fake reusable root fence.
        let fourth_authority = SearchAuthority {
            root: RootGeneration(4),
            ..third_authority
        };
        worker.replace_root(fourth_authority).unwrap();
        let fourth = retarget(&template, fourth_authority, 4, clock, worker.signal());
        let fourth_scope = fourth.scope();
        let fourth_controls = fourth.control.clone();
        worker.submit(fourth).unwrap();
        let terminal_until = Instant::now() + Duration::from_secs(3);
        while !fourth_controls.snapshot().terminal {
            let version = worker.signal().version();
            assert!(Instant::now() < terminal_until);
            worker
                .signal()
                .wait_changed(version, Duration::from_millis(20));
        }
        worker
            .request_root_quiesce(
                fourth_authority,
                Instant::now() + Duration::from_millis(100),
            )
            .unwrap();
        let failure_until = Instant::now() + Duration::from_secs(3);
        let failure = loop {
            let version = worker.signal().version();
            if let Some(fence) = worker.root_fence() {
                break fence.unwrap_err();
            }
            assert!(Instant::now() < failure_until);
            worker
                .signal()
                .wait_changed(version, Duration::from_millis(20));
        };
        assert_eq!(
            failure,
            error("root_quiesce", "deadline_before_terminal_drain")
        );
        let retained = terminal(&worker);
        assert_eq!(retained.scope, fourth_scope);
        assert_eq!(retained.go_issue, ExternalGoIssue::Issued);
        assert!(retained.result.unwrap().readiness_barrier_completed);
        assert_eq!(worker.root_fence(), Some(Err(failure.clone())));
        worker
            .request_root_quiesce(fourth_authority, Instant::now() + Duration::from_secs(3))
            .unwrap();
        assert_eq!(worker.root_fence(), Some(Err(failure)));
        assert!(
            worker
                .replace_root(SearchAuthority {
                    root: RootGeneration(5),
                    ..fourth_authority
                })
                .is_err()
        );
        assert!(worker.shutdown_snapshot().is_none());
        worker.request_shutdown(Instant::now() + Duration::from_secs(2));
        let until = Instant::now() + Duration::from_secs(3);
        let closed = loop {
            let version = worker.signal().version();
            if let Some(closed) = worker.try_join() {
                break closed.unwrap();
            }
            assert!(Instant::now() < until);
            worker
                .signal()
                .wait_changed(version, Duration::from_millis(20));
        };
        assert!(closed.cleanup_complete && !closed.quarantined);
        assert!(closed.exit_observed && closed.stdout_drained && closed.stderr_drained);
        assert!(closed.quit_sent);
        assert_eq!(closed.process_identity, first_fence.process_identity());
        assert_eq!(worker.shutdown_snapshot(), Some(Ok(closed)));
        assert!(worker.try_join().is_none());
        assert_eq!(worker.shutdown_snapshot(), Some(Ok(closed)));
    }

    #[cfg(target_os = "linux")]
    pub(super) fn fixture_config(mode: &str) -> ExternalCpuConfig {
        let program = std::env::current_exe().unwrap();
        let arguments = vec![
            "--exact".into(),
            format!("external_cpu::tests::{mode}").into(),
            "--nocapture".into(),
            "--quiet".into(),
            "--test-threads=1".into(),
        ];
        let mut file = File::open(&program).unwrap();
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 65_536];
        loop {
            let count = file.read(&mut buffer).unwrap();
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
        }
        let identity = ExternalCheckerIdentity {
            adapter_semantics: EXTERNAL_UCI_SCORE_SEMANTICS.to_owned(),
            binary_sha256: format!("{:x}", hash.finalize()),
            launch_arguments_sha256: arguments_sha256(&arguments),
            declared_name: "RZ external checker fixture".to_owned(),
            declared_version: "native-test/1".to_owned(),
            declared_source: "RoveZero MIT fixture".to_owned(),
            declared_license: "MIT".to_owned(),
            options: BTreeMap::from([("Threads".to_owned(), "2".to_owned())]),
            assets: Vec::new(),
            model_metadata: ExternalModelMetadata {
                weights_sha256: None,
                training: ExternalTrainingKnowledge::Unknown,
                declared_rights: None,
                precision: None,
            },
        };
        ExternalCpuConfig {
            working_directory: program.parent().unwrap().to_owned(),
            program,
            arguments,
            identity,
            expected_uci_name: Some("RZ external checker fixture".to_owned()),
            max_depth: 16,
            max_prefix_plies: 4096,
            handshake_timeout: Duration::from_secs(10),
            max_task_wall_time: Duration::from_secs(2),
            stop_grace: Duration::from_secs(1),
            shutdown_grace: Duration::from_secs(2),
            max_output_bytes: 65_536,
            max_line_bytes: 4096,
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn namespace_admission_preserves_unstarted_checker_and_checks_metadata_first() {
        let mut checker = ExternalUciCpuChecker::create(fixture_config("fake_uci_normal")).unwrap();
        checker.validate_namespace().unwrap();
        assert!(checker.owner.is_none());
        assert!(checker.last_attempt().is_none());
        assert_eq!(checker.request_id, 0);

        checker.active = true;
        assert_eq!(
            checker.validate_namespace(),
            Err(error("admission", "checker_not_available"))
        );
        checker.active = false;
        checker.validate_namespace().unwrap();
        checker.failed = true;
        assert_eq!(
            checker.validate_namespace(),
            Err(error("admission", "checker_not_available"))
        );
        checker.conditions.clear();
        assert_eq!(
            checker.validate_namespace(),
            Err(CheckerError::Invalid(
                "checker conditions exceed finite bound"
            ))
        );
        assert!(checker.failed);
        assert!(!checker.active);
        assert!(checker.owner.is_none());
        assert!(checker.last_attempt().is_none());
        assert_eq!(checker.request_id, 0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn native_fixture_root_divergence_and_reset_are_external_reports() {
        let _guard = FIXTURE_LOCK.lock().unwrap();
        let cancel = AtomicBool::new(false);
        let mut checker = ExternalUciCpuChecker::create(fixture_config("fake_uci_normal")).unwrap();
        checker.validate_namespace().unwrap();
        checker
            .start(Instant::now() + Duration::from_secs(10), &cancel)
            .unwrap();
        checker.validate_namespace().unwrap();
        let startup_identity = checker
            .last_attempt()
            .unwrap()
            .external
            .as_ref()
            .unwrap()
            .process
            .process_identity
            .unwrap();
        assert_eq!(startup_identity.pid, startup_identity.process_group);
        assert!(startup_identity.pid > 0);
        let limits = CpuLimits {
            max_depth: 8,
            max_nodes: 80,
            deadline: Some(Instant::now() + Duration::from_secs(2)),
        };
        let d4 = BoardMove::from_uci("d2d4").unwrap();
        let CheckerReport::ExternalUci(report) = checker
            .analyze_root_moves(&Position::startpos(), &[d4], limits, &cancel)
            .unwrap()
        else {
            panic!("foreign converted to own");
        };
        assert_eq!(report.best_move, Some(d4));
        assert_eq!(report.score, ExternalRawScore::Centipawns(21));
        assert_eq!(report.reported_depth, Some(8));
        assert_eq!(report.work.nodes, Some(10));
        assert_eq!(report.work.qnodes, None);
        assert_eq!(report.work.tt_hits, None);
        let before = checker.diagnostic_output().0.len();
        assert!(
            checker
                .new_game(
                    Instant::now() + Duration::from_secs(1),
                    &AtomicBool::new(true)
                )
                .is_err()
        );
        assert_eq!(checker.diagnostic_output().0.len(), before);
        checker
            .new_game(Instant::now() + Duration::from_secs(1), &cancel)
            .unwrap();
        checker.validate_namespace().unwrap();
        let CheckerReport::ExternalUci(branch) = checker
            .analyze_divergence(
                &Position::startpos(),
                &[BoardMove::from_uci("e2e4").unwrap()],
                CpuLimits {
                    deadline: Some(Instant::now() + Duration::from_secs(2)),
                    ..limits
                },
                &cancel,
            )
            .unwrap()
        else {
            panic!("foreign converted to own");
        };
        assert_eq!(branch.perspective, rz_position::Color::Black);
        let process = checker
            .shutdown(Instant::now() + Duration::from_secs(2))
            .unwrap();
        assert!(
            process.cleanup_complete
                && process.exit_observed
                && process.stdout_drained
                && process.stderr_drained
        );
        assert!(!process.ownership_lost && !process.quarantined);
        assert_eq!(process.process_identity, Some(startup_identity));
        let original_attempt = format!("{:?}", checker.last_attempt());
        assert_eq!(
            checker.validate_namespace(),
            Err(error("admission", "owner_closed"))
        );
        assert_eq!(format!("{:?}", checker.last_attempt()), original_attempt);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn node_overshoot_fails_and_preserves_actual_capture() {
        let _guard = FIXTURE_LOCK.lock().unwrap();
        let cancel = AtomicBool::new(false);
        let mut checker =
            ExternalUciCpuChecker::create(fixture_config("fake_uci_overshoot")).unwrap();
        checker
            .start(Instant::now() + Duration::from_secs(10), &cancel)
            .unwrap();
        let result = checker.analyze(
            &Position::startpos(),
            CpuLimits {
                max_depth: 8,
                max_nodes: 80,
                deadline: Some(Instant::now() + Duration::from_secs(2)),
            },
            &cancel,
        );
        assert!(matches!(
            result,
            Err(CheckerError::External {
                code: "reported_node_budget_exceeded",
                ..
            })
        ));
        let attempt = checker.last_attempt().unwrap();
        assert_eq!(attempt.work.nodes, Some(81));
        assert_eq!(attempt.work.qnodes, None);
        assert!(attempt.external.as_ref().unwrap().process.cleanup_complete);
        let partial = attempt
            .external
            .as_ref()
            .unwrap()
            .partial_report
            .as_ref()
            .unwrap();
        assert_eq!(partial.completion, ExternalCompletion::Pending);
        assert_eq!(partial.best_move, None);
        let original_attempt = format!("{:?}", checker.last_attempt());
        let original_request_id = checker.request_id;
        let (stdout, stderr) = checker.diagnostic_output();
        let original_output = (stdout.to_owned(), stderr.to_owned());
        for _ in 0..2 {
            assert_eq!(
                checker.validate_namespace(),
                Err(error("admission", "checker_not_available"))
            );
        }
        assert_eq!(format!("{:?}", checker.last_attempt()), original_attempt);
        assert_eq!(checker.request_id, original_request_id);
        assert_eq!(
            checker.diagnostic_output(),
            (original_output.0.as_slice(), original_output.1.as_slice())
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn native_deadline_stops_and_drains_inside_original_task_wall() {
        let _guard = FIXTURE_LOCK.lock().unwrap();
        let cancel = AtomicBool::new(false);
        let mut config = fixture_config("fake_uci_waits_for_stop");
        config.stop_grace = Duration::from_millis(500);
        let mut checker = ExternalUciCpuChecker::create(config).unwrap();
        checker
            .start(Instant::now() + Duration::from_secs(10), &cancel)
            .unwrap();
        let until = Instant::now() + Duration::from_millis(1500);
        let CheckerReport::ExternalUci(report) = checker
            .analyze(
                &Position::startpos(),
                CpuLimits {
                    max_depth: 8,
                    max_nodes: 80,
                    deadline: Some(until),
                },
                &cancel,
            )
            .unwrap()
        else {
            panic!("foreign converted to own");
        };
        assert_eq!(report.completion, ExternalCompletion::StoppedDeadline);
        assert!(report.best_move.is_some());
        assert_eq!(report.work.nodes, Some(10));
        assert!(Instant::now() < until);
        let attempt = checker.last_attempt().unwrap().external.as_ref().unwrap();
        assert!(attempt.process.stop_sent);
        assert!(!attempt.process.quarantined);
        // An idle, drained checker remains reusable after a bounded stop.
        checker
            .new_game(Instant::now() + Duration::from_secs(1), &cancel)
            .unwrap();
        let cleanup = checker
            .shutdown(Instant::now() + Duration::from_secs(2))
            .unwrap();
        assert!(cleanup.cleanup_complete && !cleanup.quarantined);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn insufficient_stop_reserve_rejects_before_go_without_poisoning_owner() {
        let _guard = FIXTURE_LOCK.lock().unwrap();
        let cancel = AtomicBool::new(false);
        let mut checker = ExternalUciCpuChecker::create(fixture_config("fake_uci_normal")).unwrap();
        checker
            .start(Instant::now() + Duration::from_secs(10), &cancel)
            .unwrap();
        let startup_attempt = checker.last_attempt().unwrap().clone();
        assert_eq!(startup_attempt.external.as_ref().unwrap().request_id, 0);
        let limits = CpuLimits {
            max_depth: 8,
            max_nodes: 80,
            deadline: Some(Instant::now() + Duration::from_millis(500)),
        };
        assert!(matches!(
            checker.preflight_task(limits),
            Err(CheckerError::External {
                stage: "admission",
                code: "insufficient_stop_reserve"
            })
        ));
        assert!(matches!(
            checker.preflight_task(CpuLimits {
                max_nodes: 0,
                ..limits
            }),
            Err(CheckerError::Invalid("external requested depth/nodes"))
        ));
        assert!(matches!(
            checker.preflight_task(CpuLimits {
                deadline: Some(Instant::now()),
                ..limits
            }),
            Err(CheckerError::External {
                stage: "analysis",
                code: "cancel_or_deadline_before_go"
            })
        ));
        // Immutable admission preserves historical startup evidence. Only an
        // actual analyze call resets the ledger for its new analysis attempt.
        let after_preflight = checker.last_attempt().unwrap();
        assert_eq!(after_preflight.work, startup_attempt.work);
        assert_eq!(after_preflight.elapsed, startup_attempt.elapsed);
        let after_preflight = after_preflight.external.as_ref().unwrap();
        let startup_external = startup_attempt.external.as_ref().unwrap();
        assert_eq!(after_preflight.request_id, startup_external.request_id);
        assert_eq!(
            after_preflight.partial_report,
            startup_external.partial_report
        );
        assert_eq!(after_preflight.process, startup_external.process);
        let result = checker.analyze(&Position::startpos(), limits, &cancel);
        assert!(matches!(
            result,
            Err(CheckerError::External {
                stage: "admission",
                code: "insufficient_stop_reserve"
            })
        ));
        assert!(checker.last_attempt().is_none());
        assert!(!checker.active && !checker.failed);
        let transcript = std::str::from_utf8(checker.diagnostic_output().0).unwrap();
        assert!(!transcript.contains("info depth") && !transcript.contains("bestmove "));
        let cleanup = checker
            .shutdown(Instant::now() + Duration::from_secs(2))
            .unwrap();
        assert!(cleanup.cleanup_complete && !cleanup.quarantined);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn expired_task_quarantine_retains_owner_for_explicit_global_shutdown() {
        let _guard = FIXTURE_LOCK.lock().unwrap();
        let cancel = AtomicBool::new(false);
        let mut config = fixture_config("fake_uci_ignores_stop");
        config.stop_grace = Duration::from_millis(150);
        let mut checker = ExternalUciCpuChecker::create(config).unwrap();
        checker
            .start(Instant::now() + Duration::from_secs(10), &cancel)
            .unwrap();
        let result = checker.analyze(
            &Position::startpos(),
            CpuLimits {
                max_depth: 8,
                max_nodes: 80,
                deadline: Some(Instant::now() + Duration::from_millis(500)),
            },
            &cancel,
        );
        assert!(matches!(
            result,
            Err(CheckerError::External {
                stage: "stop",
                code: "bestmove_not_drained"
            })
        ));
        let attempt = checker.last_attempt().unwrap().external.as_ref().unwrap();
        let partial = attempt.partial_report.as_ref().unwrap();
        assert_eq!(partial.completion, ExternalCompletion::StoppedDeadline);
        assert_eq!(partial.best_move, None);
        assert_eq!(partial.work.nodes, Some(10));
        assert!(attempt.process.quarantined);
        assert!(!attempt.process.cleanup_complete);
        assert!(!attempt.process.ownership_lost);
        assert!(checker.failed && checker.owner.is_some());
        // This is a distinct, explicitly supplied global cleanup wall. The task
        // failure did not borrow it or discard the still-owned process handle.
        let cleanup = checker
            .shutdown(Instant::now() + Duration::from_secs(2))
            .unwrap();
        assert!(cleanup.cleanup_complete && !cleanup.quarantined);
        assert!(cleanup.exit_observed && cleanup.stdout_drained && cleanup.stderr_drained);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_and_pre_go_failures_do_not_invent_bestmove_completion() {
        let _guard = FIXTURE_LOCK.lock().unwrap();
        for (mode, expected) in [
            ("fake_uci_parse_failure", "invalid_score"),
            ("fake_uci_pre_go_failure", "late_or_duplicate_response"),
        ] {
            let cancel = AtomicBool::new(false);
            let mut checker = ExternalUciCpuChecker::create(fixture_config(mode)).unwrap();
            checker
                .start(Instant::now() + Duration::from_secs(10), &cancel)
                .unwrap();
            let result = checker.analyze(
                &Position::startpos(),
                CpuLimits {
                    max_depth: 8,
                    max_nodes: 80,
                    deadline: Some(Instant::now() + Duration::from_secs(2)),
                },
                &cancel,
            );
            assert!(matches!(result, Err(CheckerError::External { code, .. }) if code == expected));
            let attempt = checker.last_attempt().unwrap().external.as_ref().unwrap();
            let partial = attempt.partial_report.as_ref().unwrap();
            assert_eq!(partial.completion, ExternalCompletion::Pending);
            assert_eq!(partial.best_move, None);
            assert!(attempt.process.cleanup_complete);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn hash_stop_reports_cancel_and_deadline_independently_without_spawning() {
        let _guard = FIXTURE_LOCK.lock().unwrap();
        let config = fixture_config("fake_uci_normal");
        for (cancelled, expired, expected) in [
            (true, false, "hash_cancelled"),
            (false, true, "hash_deadline_expired"),
            (true, true, "hash_cancelled_and_deadline_expired"),
        ] {
            let deadline = if expired {
                Instant::now()
            } else {
                Instant::now() + Duration::from_secs(10)
            };
            let failure = match process::OwnedProcess::spawn(
                &config,
                deadline,
                &AtomicBool::new(cancelled),
                None,
            ) {
                Ok(_) => panic!("a stopped hash must not spawn the teacher process"),
                Err(failure) => failure,
            };
            assert_eq!(failure.error, error("spawn", expected));
            assert!(failure.process.cleanup_complete);
            assert!(failure.process.process_identity.is_none());
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn failed_spawn_shutdown_reuses_only_actual_request_zero_cleanup() {
        let _guard = FIXTURE_LOCK.lock().unwrap();
        let mut config = fixture_config("fake_uci_normal");
        config.identity.binary_sha256 = "0".repeat(64);
        let mut checker = ExternalUciCpuChecker::create(config).unwrap();
        assert!(matches!(
            checker.shutdown(Instant::now() + Duration::from_secs(1)),
            Err(CheckerError::External {
                code: "owner_closed",
                ..
            })
        ));
        assert!(checker.last_attempt().is_none());
        let failure = checker.start(
            Instant::now() + Duration::from_secs(10),
            &AtomicBool::new(false),
        );
        assert!(matches!(
            failure,
            Err(CheckerError::External {
                code: "program_hash_mismatch",
                ..
            })
        ));
        let actual = checker
            .last_attempt()
            .unwrap()
            .external
            .as_ref()
            .unwrap()
            .process;
        assert!(actual.cleanup_complete);
        assert!(actual.process_identity.is_none());
        assert!(!actual.exit_observed && !actual.stdout_drained && !actual.stderr_drained);
        assert_eq!(
            checker
                .shutdown(Instant::now() + Duration::from_secs(1))
                .unwrap(),
            actual
        );
        assert_eq!(
            checker
                .shutdown(Instant::now() + Duration::from_secs(1))
                .unwrap(),
            actual
        );
        let evidence = checker
            .last_attempt
            .as_mut()
            .unwrap()
            .external
            .as_mut()
            .unwrap();
        evidence.process.quarantined = true;
        assert!(
            checker
                .shutdown(Instant::now() + Duration::from_secs(1))
                .is_err()
        );
        let evidence = checker
            .last_attempt
            .as_mut()
            .unwrap()
            .external
            .as_mut()
            .unwrap();
        evidence.process.quarantined = false;
        evidence.request_id = 1;
        assert!(
            checker
                .shutdown(Instant::now() + Duration::from_secs(1))
                .is_err()
        );
    }
}
