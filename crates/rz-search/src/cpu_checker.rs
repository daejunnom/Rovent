//! Checker 교체 경계. 자체 CPU의 완료 iteration과 외부 UCI의 raw 관측은
//! 별도 타입으로 유지한다. 외부 cp/mate/depth/readyok는 Rules 증명이나 자체
//! 평가 단위, 실제 옵션 적용의 증거로 승격하지 않는다.

use crate::cpu::{
    CpuCapabilities, CpuConfig, CpuError, CpuLimits, CpuPausedStackSnapshot, CpuReport,
    CpuResumePolicy, CpuResumeToken, CpuSearcher, CpuSideScopeSnapshot,
};
use crate::cpu_value::{CpuTrainingState, CpuValueIdentity};
use rz_position::{BoardMove, Color, Position};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const MAX_LABEL: usize = 256;
const MAX_DESCRIPTION: usize = 1024;
const MAX_OPTIONS: usize = 128;
const MAX_ASSETS: usize = 32;
const MAX_IDENTITY_BYTES: usize = 32 * 1024;
const MAX_CONDITIONS_BYTES: usize = 8192;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "identity",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum CheckerIdentity {
    Owned(CpuValueIdentity),
    ExternalUci(ExternalCheckerIdentity),
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalCheckerIdentity {
    pub adapter_semantics: String,
    pub binary_sha256: String,
    pub launch_arguments_sha256: String,
    pub declared_name: String,
    pub declared_version: String,
    pub declared_source: String,
    pub declared_license: String,
    pub options: BTreeMap<String, String>,
    pub assets: Vec<ExternalAssetIdentity>,
    pub model_metadata: ExternalModelMetadata,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalAssetIdentity {
    pub purpose: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalModelMetadata {
    /// A declared digest is not independent proof that an engine loaded it.
    pub weights_sha256: Option<String>,
    pub training: ExternalTrainingKnowledge,
    pub declared_rights: Option<String>,
    pub precision: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalTrainingKnowledge {
    /// No claim about training, architecture, calibration or provenance.
    Unknown,
}

fn bounded_text(value: &String, maximum: usize, allow_empty: bool) -> Result<usize, CheckerError> {
    if (!allow_empty && value.trim().is_empty())
        || value.len() > maximum
        || value.capacity() > maximum
        || value.chars().any(char::is_control)
    {
        return Err(CheckerError::Invalid(
            "identity string length/capacity or text exceeds bound",
        ));
    }
    Ok(value.capacity())
}

fn bounded_sha(value: &String) -> Result<usize, CheckerError> {
    bounded_text(value, 64, false)?;
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(CheckerError::Invalid("identity requires lowercase SHA256"));
    }
    Ok(value.capacity())
}

fn charge(total: &mut usize, bytes: usize) -> Result<(), CheckerError> {
    *total = total
        .checked_add(bytes)
        .ok_or(CheckerError::Invalid("identity byte sum overflow"))?;
    if *total > MAX_IDENTITY_BYTES {
        return Err(CheckerError::Invalid(
            "identity retained capacity exceeds finite bound",
        ));
    }
    Ok(())
}

impl ExternalCheckerIdentity {
    /// Bounds declarations and retained capacities only. The process adapter
    /// separately verifies executable/assets/argv bytes and UCI observations.
    pub fn validate(&self) -> Result<(), CheckerError> {
        let mut total = 0;
        for (text, limit) in [
            (&self.adapter_semantics, MAX_LABEL),
            (&self.declared_name, MAX_LABEL),
            (&self.declared_version, MAX_LABEL),
            (&self.declared_source, MAX_DESCRIPTION),
            (&self.declared_license, MAX_DESCRIPTION),
        ] {
            charge(&mut total, bounded_text(text, limit, false)?)?;
        }
        charge(&mut total, bounded_sha(&self.binary_sha256)?)?;
        charge(&mut total, bounded_sha(&self.launch_arguments_sha256)?)?;
        if self.options.len() > MAX_OPTIONS
            || self.assets.len() > MAX_ASSETS
            || self.assets.capacity() > MAX_ASSETS
        {
            return Err(CheckerError::Invalid(
                "external identity option/asset collection exceeds bound",
            ));
        }
        // BTreeMap has no caller-reservable capacity; its allocated entries are
        // bounded by len. Strings and Vec retain independently bounded capacity.
        for (key, value) in &self.options {
            charge(&mut total, bounded_text(key, 128, false)?)?;
            charge(&mut total, bounded_text(value, MAX_DESCRIPTION, true)?)?;
        }
        for (at, asset) in self.assets.iter().enumerate() {
            charge(&mut total, bounded_text(&asset.purpose, 128, false)?)?;
            charge(&mut total, bounded_sha(&asset.sha256)?)?;
            if self.assets[..at]
                .iter()
                .any(|prior| prior.purpose == asset.purpose)
            {
                return Err(CheckerError::Invalid("duplicate external asset purpose"));
            }
        }
        if let Some(value) = &self.model_metadata.weights_sha256 {
            charge(&mut total, bounded_sha(value)?)?;
        }
        if let Some(value) = &self.model_metadata.declared_rights {
            charge(&mut total, bounded_text(value, MAX_DESCRIPTION, false)?)?;
        }
        if let Some(value) = &self.model_metadata.precision {
            charge(&mut total, bounded_text(value, MAX_LABEL, false)?)?;
        }
        Ok(())
    }
}

impl CheckerIdentity {
    pub fn validate(&self) -> Result<(), CheckerError> {
        match self {
            Self::ExternalUci(value) => value.validate(),
            Self::Owned(value) => validate_owned_identity(value),
        }
    }
}

fn validate_owned_identity(value: &CpuValueIdentity) -> Result<(), CheckerError> {
    value
        .validate()
        .map_err(|error| CheckerError::Owned(CpuError::Value(error)))?;
    bounded_text(&value.semantics, MAX_LABEL, false)?;
    if let Some(weights) = &value.weights_sha256 {
        bounded_sha(weights)?;
    }
    if let CpuTrainingState::Learned {
        run_id,
        dataset_sha256,
        ..
    } = &value.training
    {
        bounded_text(run_id, MAX_LABEL, false)?;
        bounded_sha(dataset_sha256)?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CheckerCapabilities {
    pub max_depth: u16,
    pub max_prefix_plies: usize,
    pub max_root_moves: usize,
    pub root_moves: bool,
    pub divergence: bool,
    pub resume: bool,
    /// Advertised selectivity only. Unknown foreign pruning remains None.
    pub selective_search: Option<bool>,
}

#[derive(Clone, Debug)]
pub enum CheckerReport {
    Owned(CpuReport),
    ExternalUci(ExternalCheckerReport),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExternalUciIdentity {
    pub name: String,
    pub author: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExternalRawScore {
    /// Foreign engine centipawn units, never converted to own raw CPU units.
    Centipawns(i32),
    /// UCI mate-in-moves report, not a Rules-certified mate or mate distance.
    MateMoves(i32),
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExternalBound {
    ExactReported,
    Lower,
    Upper,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExternalCompletion {
    /// No valid bestmove has been observed for this request. Errors and partial
    /// info retain this state instead of inventing a completion event.
    Pending,
    BestMove,
    StoppedDeadline,
    StoppedCanceled,
    /// External stop/drain returned partial observations; this does not own an
    /// exact native stack continuation or certify a completed minimax bound.
    StoppedYield,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExternalCheckerReport {
    pub identity: ExternalCheckerIdentity,
    pub observed_uci: ExternalUciIdentity,
    pub request_id: u64,
    pub best_move: Option<BoardMove>,
    pub pv: Vec<BoardMove>,
    pub score: ExternalRawScore,
    pub bound: ExternalBound,
    /// Optional engine-reported values, not an independently calibrated WDL.
    pub wdl_per_mille: Option<[u16; 3]>,
    pub perspective: Color,
    pub requested_depth: u16,
    pub reported_depth: Option<u16>,
    pub seldepth: Option<u16>,
    pub root_restricted: bool,
    pub completion: ExternalCompletion,
    pub work: CheckerWork,
    pub elapsed: Duration,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CheckerWork {
    /// None is unknown/not observed. A real observed zero stays Some(0).
    pub nodes: Option<u64>,
    pub qnodes: Option<u64>,
    pub tt_hits: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct CheckerAttempt {
    pub work: CheckerWork,
    pub elapsed: Duration,
    pub external: Option<ExternalAttemptEvidence>,
}

#[derive(Clone, Debug)]
pub struct ExternalAttemptEvidence {
    pub request_id: u64,
    pub partial_report: Option<ExternalCheckerReport>,
    pub process: CheckerShutdown,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExternalProcessIdentity {
    pub pid: u32,
    pub process_group: u32,
    /// Observed Linux /proc start-time ticks; not wall-clock or a PID alone.
    pub proc_start_ticks: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CheckerShutdown {
    /// Historical spawn identity after actual PID/group/start-time checks.
    /// None for own checkers, unsupported hosts and pre-spawn failures.
    pub process_identity: Option<ExternalProcessIdentity>,
    pub stop_sent: bool,
    pub quit_sent: bool,
    pub exit_observed: bool,
    pub stdout_drained: bool,
    pub stderr_drained: bool,
    pub exit_code: Option<i32>,
    pub exit_signal: Option<i32>,
    pub cleanup_complete: bool,
    pub quarantined: bool,
    pub ownership_lost: bool,
    pub stdout_bytes: u64,
    pub stderr_bytes: u64,
}

impl CheckerShutdown {
    /// Synchronous own Rust checker has no subprocess/pipes or physical async
    /// work. Logical shutdown is complete; this claims no foreign process exit
    /// or pipe drain, nor release of the adapter's retained allocation yet.
    pub fn owned_no_process() -> Self {
        Self {
            cleanup_complete: true,
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CheckerError {
    Owned(CpuError),
    /// A physical own report was returned and then rejected by immutable
    /// namespace validation. Its independent attempt work remains available;
    /// this is distinct from an admission failure or an execution CpuError.
    OwnedReportRejected {
        reason: &'static str,
    },
    Invalid(&'static str),
    Unsupported(&'static str),
    External {
        stage: &'static str,
        code: &'static str,
    },
}
impl std::fmt::Display for CheckerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Owned(error) => write!(f, "owned checker: {error}"),
            Self::OwnedReportRejected { reason } => {
                write!(f, "own returned report rejected: {reason}")
            }
            Self::Invalid(detail) => write!(f, "checker admission: {detail}"),
            Self::Unsupported(detail) => write!(f, "checker unsupported: {detail}"),
            Self::External { stage, code } => write!(f, "external checker {stage}: {code}"),
        }
    }
}
impl std::error::Error for CheckerError {}

/// Object-safe lifecycle. External implementations own process/pipe deadlines
/// and preserve partial attempt evidence. Foreign reports are not own reports.
pub trait CpuChecker: Send {
    fn identity(&self) -> &CheckerIdentity;
    fn conditions(&self) -> &str;
    fn capabilities(&self) -> CheckerCapabilities;
    /// Read-only identity captured from an actual private worker's engine.
    /// This number alone grants no Ready, Bound, execution or Store authority.
    fn registered_worker_stack_owner_id(&self) -> Option<u64> {
        None
    }
    /// Actual successful UCI handshake identity, not the declared profile name.
    /// Own checkers and failed/partial startup observations remain None.
    fn startup_uci(&self) -> Option<ExternalUciIdentity> {
        None
    }
    /// Explicit finite startup. Synchronous own checkers have no process to
    /// spawn; external adapters must override this with their real handshake.
    fn start(&mut self, deadline: Instant, cancel: &AtomicBool) -> Result<(), CheckerError> {
        if cancel.load(Ordering::Acquire) || Instant::now() >= deadline {
            return Err(CheckerError::Invalid(
                "startup canceled or deadline expired",
            ));
        }
        self.validate_namespace()
    }
    /// Admission-only immutable namespace check. This must not dispatch work,
    /// reset an attempt ledger, transfer a resume token or count a cache visit.
    fn validate_namespace(&self) -> Result<(), CheckerError> {
        self.identity().validate()?;
        if self.conditions().is_empty()
            || self.conditions().len() > MAX_CONDITIONS_BYTES
            || self.conditions().chars().any(char::is_control)
        {
            return Err(CheckerError::Invalid(
                "checker conditions exceed finite bound",
            ));
        }
        Ok(())
    }
    /// Immutable task admission check before any caller reservation or dispatch.
    /// This never clears an attempt ledger or produces a work/report observation.
    /// It does not replace analyze's validation; synchronous own checkers retain
    /// their existing behavior. A foreign insufficient_stop_reserve admission
    /// refusal guarantees that no go or physical analysis attempt was started.
    fn preflight_task(&self, _limits: CpuLimits) -> Result<(), CheckerError> {
        Ok(())
    }
    /// Only the own implementation can disclose its native configuration. A
    /// foreign UCI declaration is never translated into an own search profile.
    fn owned_descriptor(&self) -> Option<OwnedCheckerDescriptor> {
        None
    }
    /// Actual immutable owned resume selection; external declarations expose
    /// no native stack policy or native token ownership.
    fn owned_resume_policy(&self) -> Option<crate::cpu::CpuResumePolicy> {
        None
    }
    fn preserves_paused_on_fresh_tasks(&self) -> bool {
        false
    }
    fn paused_side_scope_snapshot(&self) -> Option<CpuSideScopeSnapshot> {
        None
    }
    /// Actual native PausedStack owner facts. External/unsupported owners
    /// remain unknown; Node/Task retirement is not an owner observation.
    fn paused_stack_snapshot(&self) -> Option<CpuPausedStackSnapshot> {
        None
    }
    /// Read-only liveness check. This cannot resume, consume, recharge or pin a
    /// foreign/native token merely because its metadata has a similar shape.
    fn token_is_current(&self, _token: &CpuResumeToken) -> bool {
        false
    }
    /// Explicit native-owner capability. External implementations keep false:
    /// disposing an own CPU checkpoint never drains or cancels foreign work.
    fn supports_paused_stack_discard(&self) -> bool {
        false
    }
    /// Dispose only the native owner's paused frames, without resetting its
    /// attempt ledger, TT, ordering history, or any physical backend lease.
    fn discard_paused_stack(&mut self) -> bool {
        false
    }
    fn last_attempt(&self) -> Option<&CheckerAttempt>;
    /// Start a new ledger scope even for unsupported/admission failures.
    /// Previous execution work is not this new attempt's actual work.
    fn reset_attempt(&mut self);
    fn analyze(
        &mut self,
        position: &Position,
        limits: CpuLimits,
        cancel: &AtomicBool,
    ) -> Result<CheckerReport, CheckerError>;
    fn analyze_root_moves(
        &mut self,
        _position: &Position,
        _moves: &[BoardMove],
        _limits: CpuLimits,
        _cancel: &AtomicBool,
    ) -> Result<CheckerReport, CheckerError> {
        self.reset_attempt();
        Err(CheckerError::Unsupported("root move restriction"))
    }
    fn analyze_divergence(
        &mut self,
        _position: &Position,
        _prefix: &[BoardMove],
        _limits: CpuLimits,
        _cancel: &AtomicBool,
    ) -> Result<CheckerReport, CheckerError> {
        self.reset_attempt();
        Err(CheckerError::Unsupported("checked divergence prefix"))
    }
    fn resume(
        &mut self,
        _position: &Position,
        _token: &CpuResumeToken,
        _limits: CpuLimits,
        _cancel: &AtomicBool,
    ) -> Result<CheckerReport, CheckerError> {
        self.reset_attempt();
        Err(CheckerError::Unsupported(
            "owned completed-iteration resume",
        ))
    }
    fn new_game(&mut self, deadline: Instant, cancel: &AtomicBool) -> Result<(), CheckerError>;
    fn shutdown(&mut self, deadline: Instant) -> Result<CheckerShutdown, CheckerError>;
}

#[derive(Clone, Debug)]
pub struct OwnedCheckerDescriptor {
    pub config: CpuConfig,
    pub search_identity: &'static str,
    pub search_conditions: String,
    pub capabilities: CpuCapabilities,
}

impl OwnedCheckerDescriptor {
    /// Full owned replay condition namespace. This is metadata, not result authority.
    /// Keep byte-for-byte parity with the existing PALS producer condition format.
    pub fn registered_replay_condition(&self) -> String {
        format!(
            "{};conditions={};profile={:?};q={};tt={};maxdepth={};capabilities={:?}",
            self.search_identity,
            self.search_conditions,
            self.config.profile,
            self.config.quiescence_ply,
            self.config.tt_entries,
            self.config.max_depth,
            self.capabilities
        )
    }
}

pub struct OwnedCpuChecker<C: CpuSearcher> {
    cpu: C,
    identity: CheckerIdentity,
    conditions: String,
    search_identity: &'static str,
    config: CpuConfig,
    capabilities: CheckerCapabilities,
    last_attempt: Option<CheckerAttempt>,
    closed: bool,
    paused_stack_shutdown_complete: bool,
    preserves_paused_on_fresh_tasks: bool,
}

impl<C: CpuSearcher> OwnedCpuChecker<C> {
    pub fn new(cpu: C) -> Result<Self, CheckerError> {
        validate_owned_identity(cpu.value_identity())?;
        let conditions = cpu.search_conditions();
        bounded_text(&conditions, MAX_CONDITIONS_BYTES, false)?;
        let search_identity = cpu.search_identity();
        if search_identity.trim().is_empty()
            || search_identity.len() > MAX_LABEL
            || search_identity.chars().any(char::is_control)
        {
            return Err(CheckerError::Invalid("owned search identity exceeds bound"));
        }
        let config = cpu.config().clone();
        let cap = cpu.capabilities();
        if config.max_depth == 0
            || config.max_depth > 64
            || config.quiescence_ply > 32
            || config.tt_entries > 1_048_576
            || cap.max_depth != config.max_depth
            || cap.max_prefix_plies > 64
            || cap.max_root_moves > 256
        {
            return Err(CheckerError::Invalid(
                "owned configuration/capability exceeds finite admission",
            ));
        }
        let identity = CheckerIdentity::Owned(cpu.value_identity().clone());
        let preserves_paused_on_fresh_tasks = cpu.preserves_paused_on_fresh_tasks();
        let capabilities = CheckerCapabilities {
            max_depth: cap.max_depth,
            max_prefix_plies: cap.max_prefix_plies,
            max_root_moves: cap.max_root_moves,
            root_moves: cap.root_moves,
            divergence: cap.divergence,
            resume: cap.completed_iteration_resume,
            selective_search: Some(cap.selective_reductions),
        };
        Ok(Self {
            cpu,
            identity,
            conditions,
            search_identity,
            config,
            capabilities,
            last_attempt: None,
            closed: false,
            paused_stack_shutdown_complete: false,
            preserves_paused_on_fresh_tasks,
        })
    }

    fn admit(&mut self) -> Result<(), CheckerError> {
        self.last_attempt = None;
        self.validate_frozen()
    }

    fn validate_frozen(&self) -> Result<(), CheckerError> {
        if self.closed {
            return Err(CheckerError::Invalid("owned checker has shut down"));
        }
        let CheckerIdentity::Owned(identity) = &self.identity else {
            return Err(CheckerError::Invalid("owned identity changed type"));
        };
        let config = self.cpu.config();
        let capabilities = self.cpu.capabilities();
        if self.cpu.value_identity() != identity
            || self.cpu.preserves_paused_on_fresh_tasks() != self.preserves_paused_on_fresh_tasks
            || self.cpu.search_identity() != self.search_identity
            || self.cpu.search_conditions() != self.conditions
            || config.profile != self.config.profile
            || config.max_depth != self.config.max_depth
            || config.quiescence_ply != self.config.quiescence_ply
            || config.tt_entries != self.config.tt_entries
            || capabilities.max_depth != self.capabilities.max_depth
            || capabilities.max_prefix_plies != self.capabilities.max_prefix_plies
            || capabilities.max_root_moves != self.capabilities.max_root_moves
            || capabilities.root_moves != self.capabilities.root_moves
            || capabilities.divergence != self.capabilities.divergence
            || capabilities.completed_iteration_resume != self.capabilities.resume
            || Some(capabilities.selective_reductions) != self.capabilities.selective_search
        {
            return Err(CheckerError::Invalid(
                "frozen own value/search conditions changed",
            ));
        }
        Ok(())
    }

    fn capture(
        &mut self,
        started: Instant,
        result: Result<CpuReport, CpuError>,
    ) -> Result<CheckerReport, CheckerError> {
        let work = self
            .cpu
            .last_attempt_work()
            .map_or_else(CheckerWork::default, |work| CheckerWork {
                nodes: Some(work.nodes),
                qnodes: Some(work.quiescence_nodes),
                tt_hits: Some(work.tt_hits),
            });
        self.last_attempt = Some(CheckerAttempt {
            work,
            elapsed: started.elapsed(),
            external: None,
        });
        if let Err(error) = self.validate_frozen() {
            if result.is_ok() {
                return Err(CheckerError::OwnedReportRejected {
                    reason: "frozen own value/search conditions changed after returned report",
                });
            }
            return Err(error);
        }
        let report = result.map_err(CheckerError::Owned)?;
        let CheckerIdentity::Owned(identity) = &self.identity else {
            return Err(CheckerError::Invalid("owned identity changed type"));
        };
        if report.value_identity != *identity
            || report.search_version != self.search_identity
            || report.profile != self.config.profile
        {
            return Err(CheckerError::OwnedReportRejected {
                reason: "own report namespace differs from frozen checker",
            });
        }
        Ok(CheckerReport::Owned(report))
    }
}

impl<C: CpuSearcher> CpuChecker for OwnedCpuChecker<C> {
    fn identity(&self) -> &CheckerIdentity {
        &self.identity
    }
    fn conditions(&self) -> &str {
        &self.conditions
    }
    fn capabilities(&self) -> CheckerCapabilities {
        self.capabilities
    }
    fn validate_namespace(&self) -> Result<(), CheckerError> {
        self.validate_frozen()
    }
    fn owned_descriptor(&self) -> Option<OwnedCheckerDescriptor> {
        Some(OwnedCheckerDescriptor {
            config: self.cpu.config().clone(),
            search_identity: self.cpu.search_identity(),
            search_conditions: self.cpu.search_conditions(),
            capabilities: self.cpu.capabilities(),
        })
    }
    fn owned_resume_policy(&self) -> Option<crate::cpu::CpuResumePolicy> {
        Some(self.cpu.resume_policy())
    }
    fn preserves_paused_on_fresh_tasks(&self) -> bool {
        self.preserves_paused_on_fresh_tasks
    }
    fn paused_side_scope_snapshot(&self) -> Option<CpuSideScopeSnapshot> {
        self.cpu.paused_side_scope_snapshot()
    }
    fn paused_stack_snapshot(&self) -> Option<CpuPausedStackSnapshot> {
        if self.cpu.resume_policy() != CpuResumePolicy::PausedStack {
            return None;
        }
        let mut snapshot = self.cpu.paused_stack_snapshot()?;
        snapshot.admission_closed = self.closed;
        snapshot.owner_released = self.closed
            && self.paused_stack_shutdown_complete
            && snapshot.complete
            && snapshot.tokens_retained == 0
            && snapshot.bytes_current == 0;
        Some(snapshot)
    }
    fn token_is_current(&self, token: &CpuResumeToken) -> bool {
        self.cpu.token_is_current(token)
    }
    fn supports_paused_stack_discard(&self) -> bool {
        self.cpu.supports_paused_stack_discard()
    }
    fn discard_paused_stack(&mut self) -> bool {
        self.cpu.discard_paused_stack()
    }
    fn last_attempt(&self) -> Option<&CheckerAttempt> {
        self.last_attempt.as_ref()
    }
    fn reset_attempt(&mut self) {
        self.last_attempt = None;
    }
    fn analyze(
        &mut self,
        position: &Position,
        limits: CpuLimits,
        cancel: &AtomicBool,
    ) -> Result<CheckerReport, CheckerError> {
        self.admit()?;
        let started = Instant::now();
        let result = self.cpu.analyze(position, limits, cancel);
        self.capture(started, result)
    }
    fn analyze_root_moves(
        &mut self,
        position: &Position,
        moves: &[BoardMove],
        limits: CpuLimits,
        cancel: &AtomicBool,
    ) -> Result<CheckerReport, CheckerError> {
        self.admit()?;
        if !self.capabilities.root_moves || moves.len() > self.capabilities.max_root_moves {
            return Err(CheckerError::Unsupported("root move restriction"));
        }
        let started = Instant::now();
        let result = self.cpu.analyze_root_moves(position, moves, limits, cancel);
        self.capture(started, result)
    }
    fn analyze_divergence(
        &mut self,
        position: &Position,
        prefix: &[BoardMove],
        limits: CpuLimits,
        cancel: &AtomicBool,
    ) -> Result<CheckerReport, CheckerError> {
        self.admit()?;
        if !self.capabilities.divergence || prefix.len() > self.capabilities.max_prefix_plies {
            return Err(CheckerError::Unsupported("checked divergence prefix"));
        }
        let started = Instant::now();
        let result = self
            .cpu
            .analyze_divergence(position, prefix, limits, cancel);
        self.capture(started, result)
    }
    fn resume(
        &mut self,
        position: &Position,
        token: &CpuResumeToken,
        limits: CpuLimits,
        cancel: &AtomicBool,
    ) -> Result<CheckerReport, CheckerError> {
        self.admit()?;
        if !self.capabilities.resume {
            return Err(CheckerError::Unsupported(
                "owned completed-iteration resume",
            ));
        }
        let started = Instant::now();
        let result = self.cpu.resume(position, token, limits, cancel);
        self.capture(started, result)
    }
    fn new_game(&mut self, deadline: Instant, cancel: &AtomicBool) -> Result<(), CheckerError> {
        self.admit()?;
        if cancel.load(Ordering::Acquire) {
            return Err(CheckerError::Invalid("new game canceled before clear"));
        }
        if Instant::now() >= deadline {
            return Err(CheckerError::Invalid(
                "new game deadline expired before clear",
            ));
        }
        self.cpu.clear();
        Ok(())
    }
    fn shutdown(&mut self, _deadline: Instant) -> Result<CheckerShutdown, CheckerError> {
        // No asynchronous CPU work can survive the preceding synchronous call.
        // Closing is idempotent and requires no subprocess wait or extra budget.
        self.closed = true;
        if self.cpu.resume_policy() == CpuResumePolicy::PausedStack {
            self.paused_stack_shutdown_complete = false;
            if !self.cpu.supports_paused_stack_discard() {
                return Err(CheckerError::Unsupported("paused stack shutdown discard"));
            }
            // Close admission before dropping only the actual retained frames;
            // TT/history and the last measured attempt are intentionally kept.
            self.cpu.discard_paused_stack();
            match self.cpu.paused_stack_snapshot() {
                Some(snapshot)
                    if snapshot.complete
                        && snapshot.tokens_retained == 0
                        && snapshot.bytes_current == 0 =>
                {
                    self.paused_stack_shutdown_complete = true;
                }
                Some(_) => {
                    return Err(CheckerError::Invalid(
                        "paused stack shutdown ownership is retained or incomplete",
                    ));
                }
                None => {
                    return Err(CheckerError::Unsupported(
                        "paused stack shutdown owner observation",
                    ));
                }
            }
        }
        Ok(CheckerShutdown::owned_no_process())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{CpuCapabilities, CpuEngine};
    use std::sync::{Arc, atomic::AtomicUsize};

    #[test]
    fn own_preflight_keeps_analysis_deadline_and_invalid_limit_behavior() {
        let cpu = CpuEngine::new(CpuConfig {
            tt_entries: 0,
            ..CpuConfig::default()
        })
        .unwrap();
        let mut checker = OwnedCpuChecker::new(cpu).unwrap();
        let limits = CpuLimits {
            max_depth: 1,
            max_nodes: 1024,
            deadline: Some(Instant::now()),
        };
        checker.preflight_task(limits).unwrap();
        assert!(checker.last_attempt().is_none());
        let CheckerReport::Owned(report) = checker
            .analyze(&Position::startpos(), limits, &AtomicBool::new(false))
            .unwrap()
        else {
            panic!("own checker changed namespace");
        };
        assert_eq!(report.completion, crate::cpu::CpuCompletion::Deadline);
        assert_eq!(report.nodes, 0);
        let invalid = CpuLimits {
            max_nodes: 0,
            ..limits
        };
        checker.preflight_task(invalid).unwrap();
        assert!(matches!(
            checker.analyze(&Position::startpos(), invalid, &AtomicBool::new(false)),
            Err(CheckerError::Owned(CpuError::InvalidLimits(_)))
        ));
    }

    struct UnknownWork {
        cpu: CpuEngine,
        clears: Arc<AtomicUsize>,
    }
    impl CpuSearcher for UnknownWork {
        fn config(&self) -> &CpuConfig {
            self.cpu.config()
        }
        fn value_identity(&self) -> &CpuValueIdentity {
            self.cpu.value_identity()
        }
        fn search_identity(&self) -> &'static str {
            self.cpu.search_identity()
        }
        fn search_conditions(&self) -> String {
            self.cpu.search_conditions()
        }
        fn capabilities(&self) -> CpuCapabilities {
            self.cpu.capabilities()
        }
        fn resume_policy(&self) -> CpuResumePolicy {
            self.cpu.resume_policy()
        }
        fn clear(&mut self) {
            self.clears.fetch_add(1, Ordering::Relaxed);
            self.cpu.clear();
        }
        fn analyze(
            &mut self,
            _: &Position,
            _: CpuLimits,
            _: &AtomicBool,
        ) -> Result<CpuReport, CpuError> {
            Err(CpuError::Unsupported("test checker has no observed work"))
        }
    }

    #[test]
    fn unknown_attempt_never_invents_zero_work_and_trait_is_object_safe() {
        let cpu = CpuEngine::new(CpuConfig {
            tt_entries: 0,
            ..CpuConfig::default()
        })
        .unwrap();
        let mut checker: Box<dyn CpuChecker> = Box::new(
            OwnedCpuChecker::new(UnknownWork {
                cpu,
                clears: Arc::new(AtomicUsize::new(0)),
            })
            .unwrap(),
        );
        let descriptor = checker.owned_descriptor().unwrap();
        assert_eq!(checker.paused_stack_snapshot(), None);
        assert_eq!(descriptor.config.tt_entries, 0);
        assert_eq!(descriptor.search_conditions, checker.conditions());
        assert_eq!(
            descriptor.capabilities.max_depth,
            checker.capabilities().max_depth
        );
        assert_eq!(descriptor.search_identity, crate::cpu::CPU_SEARCH_VERSION);
        assert!(
            checker
                .analyze(
                    &Position::startpos(),
                    CpuLimits::default(),
                    &AtomicBool::new(false)
                )
                .is_err()
        );
        assert_eq!(checker.last_attempt().unwrap().work, CheckerWork::default());
        let shutdown = checker.shutdown(Instant::now()).unwrap();
        assert!(shutdown.cleanup_complete);
        assert_eq!(checker.paused_stack_snapshot(), None);
        assert!(!shutdown.exit_observed);
        assert!(!shutdown.stdout_drained);
        assert_eq!(shutdown.exit_code, None);
        assert_eq!(shutdown.exit_signal, None);
        assert!(
            checker
                .analyze(
                    &Position::startpos(),
                    CpuLimits::default(),
                    &AtomicBool::new(false)
                )
                .is_err()
        );
        assert!(checker.last_attempt().is_none());
    }

    #[test]
    fn canceled_or_expired_new_game_does_not_clear_cpu_state() {
        let clears = Arc::new(AtomicUsize::new(0));
        let cpu = CpuEngine::new(CpuConfig {
            tt_entries: 0,
            ..CpuConfig::default()
        })
        .unwrap();
        let mut checker = OwnedCpuChecker::new(UnknownWork {
            cpu,
            clears: Arc::clone(&clears),
        })
        .unwrap();
        assert!(
            checker
                .new_game(
                    Instant::now() + Duration::from_secs(1),
                    &AtomicBool::new(true)
                )
                .is_err()
        );
        assert!(
            checker
                .new_game(Instant::now(), &AtomicBool::new(false))
                .is_err()
        );
        assert_eq!(clears.load(Ordering::Relaxed), 0);
        checker
            .new_game(
                Instant::now() + Duration::from_secs(1),
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(clears.load(Ordering::Relaxed), 1);
    }

    fn paused_cpu() -> CpuEngine {
        CpuEngine::with_resume_policy(
            CpuConfig {
                max_depth: 2,
                tt_entries: 0,
                ..CpuConfig::default()
            },
            CpuResumePolicy::PausedStack,
        )
        .unwrap()
    }

    #[test]
    fn owned_paused_shutdown_proves_closed_release_without_resetting_work_or_lifetime() {
        let cpu: Box<dyn CpuSearcher> = Box::new(paused_cpu());
        let mut checker = OwnedCpuChecker::new(cpu).unwrap();
        let position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let limits = CpuLimits {
            max_depth: 2,
            max_nodes: 1,
            deadline: None,
        };
        let initial = checker.paused_stack_snapshot().unwrap();
        assert!(initial.complete);
        assert!(!initial.admission_closed && !initial.owner_released);
        assert_eq!(initial, checker.paused_stack_snapshot().unwrap());
        checker.analyze(&position, limits, &cancel).unwrap();
        let before_new_game = checker.paused_stack_snapshot().unwrap();
        assert_eq!(before_new_game.tokens_retained, 1);
        assert!(!before_new_game.owner_released);
        checker
            .new_game(Instant::now() + Duration::from_secs(1), &cancel)
            .unwrap();
        let after_new_game = checker.paused_stack_snapshot().unwrap();
        assert_eq!(after_new_game.owner_id, initial.owner_id);
        assert_eq!(after_new_game.tokens_created, 1);
        assert_eq!(after_new_game.tokens_invalidated, 1);
        assert_eq!(after_new_game.bytes_peak, before_new_game.bytes_peak);
        assert!(after_new_game.complete);
        assert_eq!(after_new_game.tokens_retained, 0);
        assert!(checker.cpu.last_attempt_work().is_none());
        assert!(checker.last_attempt().is_none());

        let CheckerReport::Owned(partial) = checker.analyze(&position, limits, &cancel).unwrap()
        else {
            panic!("own paused checker returned foreign report");
        };
        let token = partial.resume.unwrap();
        let work = checker.cpu.last_attempt_work();
        let attempt = checker.last_attempt().unwrap().clone();
        let before_shutdown = checker.paused_stack_snapshot().unwrap();
        assert_eq!(before_shutdown.tokens_created, 2);
        assert_eq!(before_shutdown.tokens_retained, 1);
        assert!(token.paused_stack_bytes().unwrap() > 0);
        let shutdown = checker.shutdown(Instant::now()).unwrap();
        assert!(shutdown.cleanup_complete);
        assert!(!shutdown.exit_observed && !shutdown.stdout_drained);
        let final_owner = checker.paused_stack_snapshot().unwrap();
        assert!(final_owner.complete && final_owner.admission_closed && final_owner.owner_released);
        assert_eq!(final_owner.owner_id, initial.owner_id);
        assert_eq!(final_owner.tokens_invalidated, 2);
        assert_eq!(final_owner.tokens_retained, 0);
        assert_eq!(final_owner.bytes_current, 0);
        assert_eq!(
            final_owner.event_sequence,
            before_shutdown.event_sequence + 1
        );
        assert_eq!(checker.cpu.last_attempt_work(), work);
        assert_eq!(checker.last_attempt().unwrap().work, attempt.work);
        assert_eq!(checker.last_attempt().unwrap().elapsed, attempt.elapsed);
        assert!(!checker.token_is_current(&token));
        // Cloned/historical metadata remains diagnostic, never current bytes.
        assert!(token.paused_stack_bytes().unwrap() > 0);
        assert!(!checker.cpu.paused_stack_snapshot().unwrap().owner_released);
        assert!(checker.shutdown(Instant::now()).unwrap().cleanup_complete);
        assert_eq!(final_owner, checker.paused_stack_snapshot().unwrap());
        assert!(matches!(
            checker.analyze(&position, limits, &cancel),
            Err(CheckerError::Invalid("owned checker has shut down"))
        ));
        assert_eq!(final_owner, checker.paused_stack_snapshot().unwrap());
    }

    struct UndiscardedOwner(CpuEngine, bool);
    impl CpuSearcher for UndiscardedOwner {
        fn config(&self) -> &CpuConfig {
            self.0.config()
        }
        fn value_identity(&self) -> &CpuValueIdentity {
            self.0.value_identity()
        }
        fn search_identity(&self) -> &'static str {
            self.0.search_identity()
        }
        fn search_conditions(&self) -> String {
            self.0.search_conditions()
        }
        fn capabilities(&self) -> CpuCapabilities {
            self.0.capabilities()
        }
        fn resume_policy(&self) -> CpuResumePolicy {
            self.0.resume_policy()
        }
        fn paused_stack_snapshot(&self) -> Option<CpuPausedStackSnapshot> {
            let mut snapshot = self.0.paused_stack_snapshot()?;
            if self.1 {
                snapshot.complete = false;
            }
            Some(snapshot)
        }
        fn supports_paused_stack_discard(&self) -> bool {
            true
        }
        // Deliberately dishonest capability: the default discard drops nothing.
        fn clear(&mut self) {
            self.0.clear();
        }
        fn analyze(
            &mut self,
            p: &Position,
            l: CpuLimits,
            c: &AtomicBool,
        ) -> Result<CpuReport, CpuError> {
            self.0.analyze(p, l, c)
        }
    }

    #[test]
    fn unknown_or_undiscarded_custom_paused_owner_cannot_claim_shutdown_release() {
        let mut unknown = OwnedCpuChecker::new(UnknownWork {
            cpu: paused_cpu(),
            clears: Arc::new(AtomicUsize::new(0)),
        })
        .unwrap();
        assert_eq!(unknown.paused_stack_snapshot(), None);
        assert!(matches!(
            unknown.shutdown(Instant::now()),
            Err(CheckerError::Unsupported("paused stack shutdown discard"))
        ));
        assert_eq!(unknown.paused_stack_snapshot(), None);

        let mut retained = OwnedCpuChecker::new(UndiscardedOwner(paused_cpu(), false)).unwrap();
        retained
            .analyze(
                &Position::startpos(),
                CpuLimits {
                    max_depth: 2,
                    max_nodes: 1,
                    deadline: None,
                },
                &AtomicBool::new(false),
            )
            .unwrap();
        assert!(matches!(
            retained.shutdown(Instant::now()),
            Err(CheckerError::Invalid(
                "paused stack shutdown ownership is retained or incomplete"
            ))
        ));
        let snapshot = retained.paused_stack_snapshot().unwrap();
        assert!(snapshot.complete && snapshot.admission_closed);
        assert!(!snapshot.owner_released);
        assert_eq!(snapshot.tokens_retained, 1);
        assert!(snapshot.bytes_current > 0);
        assert_eq!(snapshot.tokens_invalidated, 0);

        // Empty retention alone is insufficient when actual observation is
        // incomplete. A later successful verified fence can disclose release.
        assert!(retained.cpu.0.discard_paused_stack());
        retained.cpu.1 = true;
        assert!(matches!(
            retained.shutdown(Instant::now()),
            Err(CheckerError::Invalid(
                "paused stack shutdown ownership is retained or incomplete"
            ))
        ));
        let incomplete = retained.paused_stack_snapshot().unwrap();
        assert_eq!(incomplete.tokens_retained, 0);
        assert_eq!(incomplete.bytes_current, 0);
        assert!(!incomplete.complete && !incomplete.owner_released);
        retained.cpu.1 = false;
        assert!(retained.shutdown(Instant::now()).unwrap().cleanup_complete);
        assert!(retained.paused_stack_snapshot().unwrap().owner_released);
    }

    fn external() -> ExternalCheckerIdentity {
        ExternalCheckerIdentity {
            adapter_semantics: "external-uci-raw-v1".into(),
            binary_sha256: "a".repeat(64),
            launch_arguments_sha256: "b".repeat(64),
            declared_name: "fixture".into(),
            declared_version: "1".into(),
            declared_source: "own-test-fixture".into(),
            declared_license: "MIT".into(),
            options: BTreeMap::new(),
            assets: Vec::new(),
            model_metadata: ExternalModelMetadata {
                weights_sha256: None,
                training: ExternalTrainingKnowledge::Unknown,
                declared_rights: None,
                precision: None,
            },
        }
    }

    #[test]
    fn identity_bounds_retained_capacity_and_preserves_unknown_metadata() {
        let identity = external();
        identity.validate().unwrap();
        let wire = serde_json::to_vec(&CheckerIdentity::ExternalUci(identity.clone())).unwrap();
        let decoded: CheckerIdentity = serde_json::from_slice(&wire).unwrap();
        decoded.validate().unwrap();
        assert_eq!(decoded, CheckerIdentity::ExternalUci(identity.clone()));
        let mut oversized = identity.clone();
        oversized.declared_name = String::with_capacity(MAX_LABEL + 1);
        oversized.declared_name.push_str("fixture");
        assert!(oversized.validate().is_err());
        let mut oversized = identity.clone();
        oversized.assets = Vec::with_capacity(MAX_ASSETS + 1);
        assert!(oversized.validate().is_err());
        let mut malformed = identity;
        malformed.launch_arguments_sha256 = "B".repeat(64);
        assert!(malformed.validate().is_err());
    }

    #[test]
    fn owned_adapter_preserves_report_and_actual_work_namespace() {
        let cpu = CpuEngine::new(CpuConfig {
            tt_entries: 0,
            ..CpuConfig::default()
        })
        .unwrap();
        let mut checker = OwnedCpuChecker::new(cpu).unwrap();
        let result = checker
            .analyze(
                &Position::startpos(),
                CpuLimits {
                    max_depth: 1,
                    max_nodes: 1024,
                    deadline: None,
                },
                &AtomicBool::new(false),
            )
            .unwrap();
        let CheckerReport::Owned(report) = result else {
            panic!("owned checker returned foreign report")
        };
        assert_eq!(
            checker.last_attempt().unwrap().work.nodes,
            Some(report.nodes)
        );
        assert_eq!(
            checker.last_attempt().unwrap().work.qnodes,
            Some(report.quiescence_nodes)
        );
        assert_eq!(
            checker.identity(),
            &CheckerIdentity::Owned(report.value_identity)
        );
    }

    struct DefaultUnsupported(OwnedCpuChecker<CpuEngine>);
    impl CpuChecker for DefaultUnsupported {
        fn identity(&self) -> &CheckerIdentity {
            self.0.identity()
        }
        fn conditions(&self) -> &str {
            self.0.conditions()
        }
        fn capabilities(&self) -> CheckerCapabilities {
            let mut cap = self.0.capabilities();
            cap.root_moves = false;
            cap.divergence = false;
            cap.resume = false;
            cap
        }
        fn last_attempt(&self) -> Option<&CheckerAttempt> {
            self.0.last_attempt()
        }
        fn reset_attempt(&mut self) {
            self.0.reset_attempt();
        }
        fn analyze(
            &mut self,
            position: &Position,
            limits: CpuLimits,
            cancel: &AtomicBool,
        ) -> Result<CheckerReport, CheckerError> {
            self.0.analyze(position, limits, cancel)
        }
        fn new_game(&mut self, deadline: Instant, cancel: &AtomicBool) -> Result<(), CheckerError> {
            self.0.new_game(deadline, cancel)
        }
        fn shutdown(&mut self, deadline: Instant) -> Result<CheckerShutdown, CheckerError> {
            self.0.shutdown(deadline)
        }
    }

    #[test]
    fn default_unsupported_operation_clears_previous_actual_attempt() {
        let cpu = CpuEngine::new(CpuConfig {
            tt_entries: 0,
            ..CpuConfig::default()
        })
        .unwrap();
        let mut checker = DefaultUnsupported(OwnedCpuChecker::new(cpu).unwrap());
        let position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let limits = CpuLimits {
            max_depth: 1,
            max_nodes: 1024,
            deadline: None,
        };
        let report = checker.analyze(&position, limits, &cancel).unwrap();
        assert!(checker.last_attempt().unwrap().work.nodes.unwrap() > 0);
        assert!(
            checker
                .analyze_root_moves(&position, &position.legal_moves()[..1], limits, &cancel)
                .is_err()
        );
        assert!(checker.last_attempt().is_none());
        checker.analyze(&position, limits, &cancel).unwrap();
        assert!(
            checker
                .analyze_divergence(&position, &[], limits, &cancel)
                .is_err()
        );
        assert!(checker.last_attempt().is_none());
        checker.analyze(&position, limits, &cancel).unwrap();
        let CheckerReport::Owned(report) = report else {
            panic!("fixture must produce own CPU report")
        };
        assert!(
            checker
                .resume(&position, report.resume.as_ref().unwrap(), limits, &cancel)
                .is_err()
        );
        assert!(checker.last_attempt().is_none());
    }
}
