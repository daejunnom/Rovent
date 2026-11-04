//! One fresh, training-private Own CPU candidate check, separate from V tasks.
//!
//! Rules/UCI reconstruct the exact known state/history and complete legal order.
//! A single Independent bootstrap search restricts only the root. Raw scores are
//! uncalibrated side-to-move units, never CP/WDL, outcome proofs or preferences.
//! The captured/checker/pre-result hashes are caller declarations: their bytes,
//! registration and production order require an independent caller admission.
//! The dispatcher only compares its verified-image argument; the CLI separately
//! states how it hashed its running executable. No source/build pin is inferred.

use super::{
    CpuTaskAdmission, CpuTaskError, FailedCheckWork, MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES,
    MAX_WALL_TIME_MS, canonical, decode_moves, history_sha, json_digest, milliseconds, pack_moves,
    replay_checked, state_sha, valid_sha,
};
use crate::engine::{OwnerRegistry, RulesUciPort};
use crate::{Command, ParserLimits, PositionPort, parse};
use rz_position::{Color, HistoryCompleteness, PlayStatus, Position, PositionLimits};
use rz_search::cpu::{
    BOOTSTRAP_SCORE_VERSION, CPU_SEARCH_VERSION, CpuCompletion, CpuConfig, CpuEngine, CpuLimits,
    CpuProfile, CpuReport, CpuScoreScope,
};
use rz_search::cpu_value::{CpuTrainingState, CpuValueIdentity};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

pub const CANDIDATE_SCHEMA: &str = "rz-pals-owned-cpu-candidate/1";
pub const CONDITIONS_SCHEMA: &str = "rz-pals-owned-cpu-candidate-conditions/1";
pub const LEGAL_ORDER_DOMAIN: &str = "rz-pals-owned-cpu-candidate-legal-order/1";
pub const DECLARED_REGISTRATION_SCOPE: &str = "declared_not_independently_verified";
pub const DISPATCHER_BINARY_SCOPE: &str = "dispatcher_compared_verified_argument";

/// A single root candidate; no V baseline/after, prefix, resume or recheck fields.
/// Declared hashes are sealed into context, not independently enrolled here.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateRequest {
    pub schema: String,
    pub task_id: String,
    pub captured_input_sha256: String,
    pub checker_namespace_sha256: String,
    pub before_result_anchor_sha256: String,
    pub frozen_epoch: u64,
    pub input_revision: u64,
    pub position_command: String,
    pub expected_board_fen: String,
    pub rules_state_sha256: String,
    pub rules_history_sha256: String,
    pub expected_legal_moves: Vec<u16>,
    pub white_to_move: bool,
    pub candidate: u16,
    pub cpu_binary_sha256: String,
    pub cpu_profile_sha256: String,
    pub horizon: u16,
    pub node_budget: u64,
    pub tt_entries: usize,
    pub quiescence_ply: u16,
    pub max_wall_time_ms: u64,
    pub max_output_bytes: usize,
    pub context_sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateStatus {
    Completed,
    Partial,
    Canceled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchCompletion {
    DepthLimit,
    NodeLimit,
    Deadline,
    Canceled,
    QuiescenceLimit,
    RulesTerminal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreScope {
    FrontierOnly,
    CompletedIteration,
    RulesTerminal,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourcePolicy {
    pub max_wall_time_ms: u64,
    pub max_checks: u8,
    /// Taken from the original wall allowance, never an additional deadline.
    pub search_deadline_reserve_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateConditions {
    pub schema: String,
    pub rules_state_sha256: String,
    pub rules_history_sha256: String,
    pub board_fen: String,
    pub legal_moves: Vec<u16>,
    pub legal_order_sha256: String,
    pub history_completeness: String,
    pub white_to_move: bool,
    pub profile_sha256: String,
    pub profile: String,
    pub value_identity: CpuValueIdentity,
    pub search_version: String,
    /// Common CPU conditions exclude the candidate root mask, recorded below.
    pub search_conditions: String,
    pub horizon: u16,
    pub node_budget: u64,
    pub tt_entries: usize,
    pub quiescence_ply: u16,
    pub root_moves: [u16; 1],
    pub restriction: String,
    pub perspective: String,
    pub resource_policy: ResourcePolicy,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateRawReport {
    pub conditions_sha256: String,
    pub conditions: CandidateConditions,
    pub raw_score: i32,
    pub best_move: Option<u16>,
    pub pv: Vec<u16>,
    pub score_scope: ScoreScope,
    pub completion: SearchCompletion,
    /// Retained if an unexpected native terminal report must be diagnosed.
    pub terminal_reason: Option<String>,
    pub completed_depth: u16,
    pub requested_depth: u16,
    pub nodes: u64,
    pub quiescence_nodes: u64,
    pub tt_hits: u64,
    pub elapsed_ms: u64,
    pub reused_completed_depth: u16,
    pub root_restricted: bool,
    pub score_provenance: String,
    pub pv_rules_validated: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateReceipt {
    pub schema: String,
    pub task_id: String,
    pub captured_input_sha256: String,
    pub checker_namespace_sha256: String,
    pub before_result_anchor_sha256: String,
    pub frozen_epoch: u64,
    pub input_revision: u64,
    pub context_sha256: String,
    pub cpu_binary_sha256: String,
    pub binary_pin_scope: String,
    pub caller_registration_scope: String,
    pub status: CandidateStatus,
    pub cpu_calls: u8,
    pub fresh_engine: bool,
    pub elapsed_ms: u64,
    pub deadline_exceeded: bool,
    pub product_verifier_enabled: bool,
    pub report: CandidateRawReport,
}

/// Nonzero CLI failure preserves observed work and a report even when its
/// post-search validation/serialization/delivery failed. Such a report is not
/// an admitted candidate result. None means unknown work, not zero work.
#[derive(Clone, Debug, Serialize)]
pub struct CandidateTaskError {
    pub schema: &'static str,
    pub code: &'static str,
    pub stage: &'static str,
    // A thin pointer keeps the unboxed error below Clippy's large-Err bound;
    // serde still emits exactly the same bounded JSON string.
    pub message: Box<String>,
    pub known_nodes: Option<u64>,
    pub failed_check_work: Option<Box<FailedCheckWork>>,
    pub report: Option<Box<CandidateRawReport>>,
    pub cpu_calls: u8,
    pub elapsed_ms: Option<u64>,
    pub deadline_exceeded: bool,
    #[serde(skip)]
    pub output_limit: usize,
}

impl CandidateTaskError {
    pub fn new(stage: &'static str, message: impl fmt::Display) -> Self {
        Self {
            schema: CANDIDATE_SCHEMA,
            code: "cpu_candidate_task_failed",
            stage,
            message: Box::new(message.to_string().chars().take(512).collect::<String>()),
            known_nodes: None,
            failed_check_work: None,
            report: None,
            cpu_calls: 0,
            elapsed_ms: None,
            deadline_exceeded: false,
            output_limit: 1024,
        }
    }

    pub fn with_report(mut self, report: &CandidateRawReport) -> Self {
        self.known_nodes = Some(report.nodes);
        self.failed_check_work = Some(Box::new(FailedCheckWork {
            nodes: report.nodes,
            quiescence_nodes: report.quiescence_nodes,
            tt_hits: report.tt_hits,
        }));
        self.report = Some(Box::new(report.clone()));
        self.cpu_calls = 1;
        self
    }
}

impl fmt::Display for CandidateTaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.stage, self.message)
    }
}
impl std::error::Error for CandidateTaskError {}

// Only request-independent parent helpers reach this conversion. No V task is
// dispatched or repackaged, and V baseline/after reports never enter this path.
fn shared_error(error: CpuTaskError) -> CandidateTaskError {
    CandidateTaskError::new(error.stage, error.message)
}

/// Actual fixed Independent bootstrap configuration, distinct from V's domain.
pub fn profile_description(request: &CandidateRequest) -> Value {
    json!({"domain":CANDIDATE_SCHEMA,"search":CPU_SEARCH_VERSION,
        "evaluator":BOOTSTRAP_SCORE_VERSION,"profile":CpuProfile::Independent.identity(),
        "tt_entries":request.tt_entries,"max_depth":request.horizon,
        "quiescence_ply":request.quiescence_ply,"selective_reductions":false})
}

pub fn profile_sha256(request: &CandidateRequest) -> Result<String, CandidateTaskError> {
    json_digest(&profile_description(request)).map_err(shared_error)
}

/// SHA256 of canonical [schema, complete request without context_sha256].
/// Caller anchors are bound as declarations; this does not verify their bytes.
pub fn request_context_sha256(request: &CandidateRequest) -> Result<String, CandidateTaskError> {
    let mut value = serde_json::to_value(request)
        .map_err(|e| CandidateTaskError::new("context_identity", e))?;
    value
        .as_object_mut()
        .ok_or_else(|| CandidateTaskError::new("context_identity", "request is not an object"))?
        .remove("context_sha256");
    json_digest(&json!([CANDIDATE_SCHEMA, value])).map_err(shared_error)
}

fn decode_request(bytes: &[u8]) -> Result<CandidateRequest, CandidateTaskError> {
    if bytes.is_empty() || bytes.len() > MAX_REQUEST_BYTES {
        return Err(CandidateTaskError::new(
            "admission",
            "stdin must contain 1..=512KiB",
        ));
    }
    let request: CandidateRequest =
        serde_json::from_slice(bytes).map_err(|e| CandidateTaskError::new("json_admission", e))?;
    if request.schema != CANDIDATE_SCHEMA
        || request.task_id.is_empty()
        || request.task_id.len() > 128
        || request.task_id.bytes().any(|b| !b.is_ascii_graphic())
        || !(1..=64).contains(&request.horizon)
        || !(1..=u64::from(u32::MAX)).contains(&request.node_budget)
        || !(1..=MAX_WALL_TIME_MS).contains(&request.max_wall_time_ms)
        || !(1024..=MAX_RESPONSE_BYTES).contains(&request.max_output_bytes)
        || request.tt_entries > 1_048_576
        || request.quiescence_ply > 32
        || request.expected_legal_moves.is_empty()
        || request.expected_legal_moves.len() > 256
        || request.expected_board_fen.is_empty()
        || request.expected_board_fen.len() > PositionLimits::default().max_fen_bytes
    {
        return Err(CandidateTaskError::new(
            "admission",
            "unsupported schema or finite request bounds",
        ));
    }
    for hash in [
        &request.captured_input_sha256,
        &request.checker_namespace_sha256,
        &request.before_result_anchor_sha256,
        &request.rules_state_sha256,
        &request.rules_history_sha256,
        &request.cpu_binary_sha256,
        &request.cpu_profile_sha256,
        &request.context_sha256,
    ] {
        if !valid_sha(hash) {
            return Err(CandidateTaskError::new(
                "admission",
                "identity must be lowercase SHA256",
            ));
        }
    }
    decode_moves(&[request.candidate]).map_err(shared_error)?;
    let legal = decode_moves(&request.expected_legal_moves).map_err(shared_error)?;
    if legal
        .iter()
        .enumerate()
        .any(|(at, mv)| legal[..at].contains(mv))
    {
        return Err(CandidateTaskError::new(
            "admission",
            "duplicate captured legal move",
        ));
    }
    Ok(request)
}

pub fn request_admission(
    bytes: &[u8],
    started: Instant,
) -> Result<CpuTaskAdmission, CandidateTaskError> {
    let request = decode_request(bytes)?;
    Ok(CpuTaskAdmission {
        deadline: started
            .checked_add(Duration::from_millis(request.max_wall_time_ms))
            .ok_or_else(|| CandidateTaskError::new("admission", "absolute deadline overflow"))?,
        output_limit: request.max_output_bytes,
    })
}

fn validate_pins(request: &CandidateRequest, binary: &str) -> Result<(), CandidateTaskError> {
    if !valid_sha(binary) || request.cpu_binary_sha256 != binary {
        return Err(CandidateTaskError::new(
            "binary_identity",
            "verified image argument differs from declared CPU binary",
        ));
    }
    if profile_sha256(request)? != request.cpu_profile_sha256 {
        return Err(CandidateTaskError::new(
            "profile_identity",
            "actual fixed CPU configuration digest mismatch",
        ));
    }
    if request_context_sha256(request)? != request.context_sha256 {
        return Err(CandidateTaskError::new(
            "context_identity",
            "complete request context digest mismatch",
        ));
    }
    Ok(())
}

fn reserve_ms(request: &CandidateRequest) -> u64 {
    (request.max_wall_time_ms / 10).min(1000)
}

fn check_deadline(deadline: Instant, stage: &'static str) -> Result<(), CandidateTaskError> {
    if Instant::now() >= deadline {
        Err(CandidateTaskError::new(
            stage,
            "original absolute wall allowance expired",
        ))
    } else {
        Ok(())
    }
}

fn capture_report(
    report: &CpuReport,
    conditions: CandidateConditions,
) -> Result<CandidateRawReport, CandidateTaskError> {
    let completion = match report.completion {
        CpuCompletion::DepthLimit => SearchCompletion::DepthLimit,
        CpuCompletion::NodeLimit => SearchCompletion::NodeLimit,
        CpuCompletion::Deadline => SearchCompletion::Deadline,
        CpuCompletion::Canceled => SearchCompletion::Canceled,
        CpuCompletion::Yielded => {
            return Err(CandidateTaskError::new(
                "unsupported_completion",
                "cooperative Yield requires the adaptive Job/Slice receipt",
            ));
        }
        CpuCompletion::AwaitReciprocal => {
            return Err(CandidateTaskError::new(
                "unsupported_completion",
                "reciprocal CPU wait requires the Rovent question receipt",
            ));
        }
        CpuCompletion::QuiescenceLimit => SearchCompletion::QuiescenceLimit,
        CpuCompletion::Terminal(_) => SearchCompletion::RulesTerminal,
    };
    let score_scope = match report.score_scope {
        CpuScoreScope::FrontierOnly => ScoreScope::FrontierOnly,
        CpuScoreScope::CompletedIteration => ScoreScope::CompletedIteration,
        CpuScoreScope::RulesTerminal => ScoreScope::RulesTerminal,
    };
    Ok(CandidateRawReport {
        conditions_sha256: json_digest(
            &serde_json::to_value(&conditions)
                .map_err(|e| CandidateTaskError::new("cpu_report", e))?,
        )
        .map_err(shared_error)?,
        requested_depth: conditions.horizon,
        conditions,
        raw_score: report.score,
        best_move: pack_moves(&report.best_move.into_iter().collect::<Vec<_>>())
            .map_err(shared_error)?
            .first()
            .copied(),
        pv: pack_moves(&report.pv).map_err(shared_error)?,
        score_scope,
        completion,
        terminal_reason: match report.completion {
            CpuCompletion::Terminal(reason) => Some(format!("{reason:?}")),
            _ => None,
        },
        completed_depth: report.completed_depth,
        nodes: report.nodes,
        quiescence_nodes: report.quiescence_nodes,
        tt_hits: report.tt_hits,
        elapsed_ms: milliseconds(report.elapsed),
        reused_completed_depth: report.reused_completed_depth,
        root_restricted: report.root_restricted,
        score_provenance: report.score_provenance.into(),
        pv_rules_validated: false,
    })
}

fn validate_report(
    report: &CpuReport,
    request: &CandidateRequest,
    raw: &CandidateRawReport,
    position: &Position,
    owners: &OwnerRegistry,
    deadline: Instant,
) -> Result<(), CandidateTaskError> {
    if report.search_version != CPU_SEARCH_VERSION
        || report.profile != CpuProfile::Independent
        || report.value_identity != raw.conditions.value_identity
        || report.value_identity.semantics != BOOTSTRAP_SCORE_VERSION
        || report.value_identity.weights_sha256.is_some()
        || report.value_identity.training != CpuTrainingState::Bootstrap
        || report.score_provenance != BOOTSTRAP_SCORE_VERSION
        || report.completed_depth > request.horizon
        || report.nodes > request.node_budget
        || report.quiescence_nodes > report.nodes
        || report.tt_hits > report.nodes
        || report.reused_completed_depth != 0
        || !report.root_restricted
        || report.pv.len() > usize::from(request.horizon + request.quiescence_ply)
        || (report.score_scope == CpuScoreScope::CompletedIteration) != (report.completed_depth > 0)
        || report.score_scope == CpuScoreScope::RulesTerminal
        || matches!(report.completion, CpuCompletion::Terminal(_))
        || (report.completion == CpuCompletion::DepthLimit
            && report.completed_depth != request.horizon)
        || raw.best_move != Some(request.candidate)
        || raw.pv.first() != Some(&request.candidate)
    {
        return Err(CandidateTaskError::new(
            "cpu_report",
            "single fresh CPU report violated actual scope/profile/root/resource bounds",
        ));
    }
    replay_checked(position, &report.pv, owners, deadline).map_err(shared_error)?;
    check_deadline(deadline, "cpu_report")
}

fn native_failure(error: impl fmt::Display, cpu: &CpuEngine) -> CandidateTaskError {
    let mut error = CandidateTaskError::new("candidate_cpu", error);
    error.cpu_calls = 1;
    if let Some(work) = cpu.last_attempt_work() {
        error.known_nodes = Some(work.nodes);
        error.failed_check_work = Some(Box::new(FailedCheckWork {
            nodes: work.nodes,
            quiescence_nodes: work.quiescence_nodes,
            tt_hits: work.tt_hits,
        }));
    }
    error
}

fn execute(
    request: &CandidateRequest,
    verified_binary: &str,
    started: Instant,
    cancellation: &AtomicBool,
) -> Result<CandidateReceipt, CandidateTaskError> {
    let deadline = started
        .checked_add(Duration::from_millis(request.max_wall_time_ms))
        .ok_or_else(|| CandidateTaskError::new("admission", "absolute deadline overflow"))?;
    check_deadline(deadline, "admission_deadline")?;
    validate_pins(request, verified_binary)?;
    let search_deadline = deadline
        .checked_sub(Duration::from_millis(reserve_ms(request)))
        .ok_or_else(|| CandidateTaskError::new("admission", "search deadline overflow"))?;
    let command = parse(&request.position_command, ParserLimits::default())
        .map_err(|e| CandidateTaskError::new("position_admission", e))?;
    let Command::Position(spec) = command else {
        return Err(CandidateTaskError::new(
            "position_admission",
            "only a position command is accepted",
        ));
    };
    let owners = Arc::new(OwnerRegistry::default());
    let prepared = RulesUciPort::new(Arc::clone(&owners), PositionLimits::default())
        .prepare(&spec)
        .map_err(|e| CandidateTaskError::new("rules_prepare", e))?;
    check_deadline(deadline, "rules_prepare")?;
    let root = prepared.snapshot.rules_position();
    let state = state_sha(root, &owners).map_err(shared_error)?;
    let history = history_sha(root).map_err(shared_error)?;
    let legal = root.ordered_legal_moves();
    let legal_bits = pack_moves(legal.moves()).map_err(shared_error)?;
    if state != request.rules_state_sha256
        || history != request.rules_history_sha256
        || root.to_fen() != request.expected_board_fen
        || legal_bits != request.expected_legal_moves
        || (root.side_to_move() == Color::White) != request.white_to_move
    {
        return Err(CandidateTaskError::new(
            "rules_identity",
            "actual Rules state/full known history/side/legal order differs from declared captured input",
        ));
    }
    if !matches!(
        root.play_status_from_view(&legal)
            .map_err(|e| CandidateTaskError::new("rules_terminal", e))?,
        PlayStatus::Ongoing
    ) {
        return Err(CandidateTaskError::new(
            "terminal_input",
            "Rules terminal input is not a candidate search result",
        ));
    }
    let candidate = decode_moves(&[request.candidate]).map_err(shared_error)?[0];
    if !legal.moves().contains(&candidate) {
        return Err(CandidateTaskError::new(
            "root_admission",
            "candidate is not legal in the captured Rules state",
        ));
    }
    check_deadline(search_deadline, "search_admission_deadline")?;
    let mut cpu = CpuEngine::new(CpuConfig {
        profile: CpuProfile::Independent,
        tt_entries: request.tt_entries,
        max_depth: request.horizon,
        quiescence_ply: request.quiescence_ply,
    })
    .map_err(|e| CandidateTaskError::new("cpu_prepare", e))?;
    let conditions = CandidateConditions {
        schema: CONDITIONS_SCHEMA.into(),
        rules_state_sha256: state,
        rules_history_sha256: history,
        board_fen: root.to_fen(),
        legal_order_sha256: json_digest(&json!([LEGAL_ORDER_DOMAIN, legal_bits]))
            .map_err(shared_error)?,
        legal_moves: legal_bits,
        history_completeness: match root.history_completeness() {
            HistoryCompleteness::Complete => "complete",
            HistoryCompleteness::UnknownPrefix => "unknown_prefix",
        }
        .into(),
        white_to_move: root.side_to_move() == Color::White,
        profile_sha256: profile_sha256(request)?,
        profile: cpu.config().profile.identity().into(),
        value_identity: cpu.value_identity().clone(),
        search_version: CPU_SEARCH_VERSION.into(),
        search_conditions: cpu.search_conditions(),
        horizon: request.horizon,
        node_budget: request.node_budget,
        tt_entries: request.tt_entries,
        quiescence_ply: request.quiescence_ply,
        root_moves: [request.candidate],
        restriction: "candidate_only".into(),
        perspective: "captured_side_to_move".into(),
        resource_policy: ResourcePolicy {
            max_wall_time_ms: request.max_wall_time_ms,
            max_checks: 1,
            search_deadline_reserve_ms: reserve_ms(request),
        },
    };
    check_deadline(search_deadline, "search_admission_deadline")?;
    // The only search invocation: fresh TT, no baseline, resume or second check.
    let native = cpu
        .analyze_root_moves(
            root,
            &[candidate],
            CpuLimits {
                max_depth: request.horizon,
                max_nodes: request.node_budget,
                deadline: Some(search_deadline),
            },
            cancellation,
        )
        .map_err(|e| native_failure(e, &cpu))?;
    let mut report = capture_report(&native, conditions).map_err(|mut error| {
        error.cpu_calls = 1;
        error.known_nodes = Some(native.nodes);
        error.failed_check_work = Some(Box::new(FailedCheckWork {
            nodes: native.nodes,
            quiescence_nodes: native.quiescence_nodes,
            tt_hits: native.tt_hits,
        }));
        error
    })?;
    validate_report(&native, request, &report, root, &owners, deadline)
        .map_err(|error| error.with_report(&report))?;
    report.pv_rules_validated = true;
    let status = match native.completion {
        CpuCompletion::DepthLimit if native.completed_depth == request.horizon => {
            CandidateStatus::Completed
        }
        CpuCompletion::Canceled => CandidateStatus::Canceled,
        _ => CandidateStatus::Partial,
    };
    Ok(CandidateReceipt {
        schema: CANDIDATE_SCHEMA.into(),
        task_id: request.task_id.clone(),
        captured_input_sha256: request.captured_input_sha256.clone(),
        checker_namespace_sha256: request.checker_namespace_sha256.clone(),
        before_result_anchor_sha256: request.before_result_anchor_sha256.clone(),
        frozen_epoch: request.frozen_epoch,
        input_revision: request.input_revision,
        context_sha256: request.context_sha256.clone(),
        cpu_binary_sha256: verified_binary.into(),
        binary_pin_scope: DISPATCHER_BINARY_SCOPE.into(),
        caller_registration_scope: DECLARED_REGISTRATION_SCOPE.into(),
        status,
        cpu_calls: 1,
        fresh_engine: true,
        elapsed_ms: milliseconds(started.elapsed()),
        deadline_exceeded: false,
        product_verifier_enabled: false,
        report,
    })
}

fn finish(
    mut receipt: CandidateReceipt,
    request: &CandidateRequest,
    started: Instant,
) -> Result<Vec<u8>, CandidateTaskError> {
    let deadline = started
        .checked_add(Duration::from_millis(request.max_wall_time_ms))
        .ok_or_else(|| {
            CandidateTaskError::new("admission", "absolute deadline overflow")
                .with_report(&receipt.report)
        })?;
    check_deadline(deadline, "receipt_deadline").map_err(|e| e.with_report(&receipt.report))?;
    receipt.elapsed_ms = milliseconds(started.elapsed());
    let value = serde_json::to_value(&receipt)
        .map_err(|e| CandidateTaskError::new("serialization", e).with_report(&receipt.report))?;
    let mut bytes = canonical(&value).map_err(|e| shared_error(e).with_report(&receipt.report))?;
    bytes.push(b'\n');
    if bytes.len() > request.max_output_bytes {
        return Err(CandidateTaskError::new(
            "output_bound",
            "candidate receipt exceeds admitted output bound",
        )
        .with_report(&receipt.report));
    }
    check_deadline(deadline, "receipt_deadline").map_err(|e| e.with_report(&receipt.report))?;
    Ok(bytes)
}

/// `verified_own_binary_sha256` is supplied by the caller; this function cannot
/// prove how those bytes were obtained, or independently register source/build.
pub fn dispatch_started(
    bytes: &[u8],
    verified_own_binary_sha256: &str,
    started: Instant,
) -> Result<Vec<u8>, CandidateTaskError> {
    dispatch_started_with_cancel(
        bytes,
        verified_own_binary_sha256,
        started,
        &AtomicBool::new(false),
    )
}

/// A one-shot CLI has no cancellation IPC. Its supervisor kill must remain a
/// missing/unknown result; this API preserves actual in-process CPU cancellation.
pub fn dispatch_started_with_cancel(
    bytes: &[u8],
    verified_own_binary_sha256: &str,
    started: Instant,
    cancellation: &AtomicBool,
) -> Result<Vec<u8>, CandidateTaskError> {
    let request = decode_request(bytes)?;
    execute(&request, verified_own_binary_sha256, started, cancellation)
        .and_then(|receipt| finish(receipt, &request, started))
        .map_err(|mut error| {
            error.output_limit = request.max_output_bytes;
            error.elapsed_ms = Some(milliseconds(started.elapsed()));
            error.deadline_exceeded =
                started.elapsed() >= Duration::from_millis(request.max_wall_time_ms);
            error
        })
}

pub fn capabilities() -> Result<Vec<u8>, CandidateTaskError> {
    let value = json!({"schema":CANDIDATE_SCHEMA,"product_verifier_enabled":false,
        "actual_training_executed":false,"teacher_gpu":false,"max_checks":1,
        "max_request_bytes":MAX_REQUEST_BYTES,"max_response_bytes":MAX_RESPONSE_BYTES,
        "max_wall_time_ms":MAX_WALL_TIME_MS,"max_horizon":64,"max_node_budget":u32::MAX,
        "max_tt_entries":1_048_576,"max_quiescence_ply":32,"fresh_engine":true,
        "profile":CpuProfile::Independent.identity(),"search_version":CPU_SEARCH_VERSION,
        "score_semantics":BOOTSTRAP_SCORE_VERSION,"score_units":"uncalibrated_root_side_to_move_raw",
        "root_restriction":"exactly_one_candidate","resume":false,
        "caller_registration_scope":DECLARED_REGISTRATION_SCOPE,
        "binary_pin_scope":DISPATCHER_BINARY_SCOPE});
    let mut bytes = canonical(&value).map_err(shared_error)?;
    bytes.push(b'\n');
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_position::BoardMove;
    use std::sync::atomic::Ordering;

    fn request(command: &str) -> CandidateRequest {
        let Command::Position(spec) = parse(command, ParserLimits::default()).unwrap() else {
            panic!("position fixture")
        };
        let owners = Arc::new(OwnerRegistry::default());
        let prepared = RulesUciPort::new(Arc::clone(&owners), PositionLimits::default())
            .prepare(&spec)
            .unwrap();
        let root = prepared.snapshot.rules_position();
        let legal = pack_moves(root.ordered_legal_moves().moves()).unwrap();
        let mut request = CandidateRequest {
            schema: CANDIDATE_SCHEMA.into(),
            task_id: "candidate-fixture".into(),
            captured_input_sha256: "1".repeat(64),
            checker_namespace_sha256: "2".repeat(64),
            before_result_anchor_sha256: "3".repeat(64),
            frozen_epoch: 4,
            input_revision: 5,
            position_command: command.into(),
            expected_board_fen: root.to_fen(),
            rules_state_sha256: state_sha(root, &owners).unwrap(),
            rules_history_sha256: history_sha(root).unwrap(),
            candidate: legal[0],
            expected_legal_moves: legal,
            white_to_move: root.side_to_move() == Color::White,
            cpu_binary_sha256: "a".repeat(64),
            cpu_profile_sha256: String::new(),
            horizon: 1,
            node_budget: 100_000,
            tt_entries: 64,
            quiescence_ply: 8,
            max_wall_time_ms: 10_000,
            max_output_bytes: MAX_RESPONSE_BYTES,
            context_sha256: String::new(),
        };
        reseal(&mut request);
        request
    }

    fn reseal(request: &mut CandidateRequest) {
        request.cpu_profile_sha256 = profile_sha256(request).unwrap();
        request.context_sha256 = request_context_sha256(request).unwrap();
    }

    fn run(request: &CandidateRequest) -> CandidateReceipt {
        serde_json::from_slice(
            &dispatch_started(
                &serde_json::to_vec(request).unwrap(),
                &request.cpu_binary_sha256,
                Instant::now(),
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn reject(request: &CandidateRequest) -> CandidateTaskError {
        dispatch_started(
            &serde_json::to_vec(request).unwrap(),
            &request.cpu_binary_sha256,
            Instant::now(),
        )
        .unwrap_err()
    }

    #[test]
    fn two_candidates_each_have_one_fresh_restricted_actual_report() {
        let a = request("position startpos");
        let mut b = a.clone();
        b.candidate = a.expected_legal_moves[1];
        b.task_id = "other-candidate".into();
        reseal(&mut b);
        for request in [&a, &b] {
            let receipt = run(request);
            assert_eq!(receipt.status, CandidateStatus::Completed);
            assert_eq!(receipt.cpu_calls, 1);
            assert!(receipt.fresh_engine);
            assert_eq!(receipt.report.completion, SearchCompletion::DepthLimit);
            assert_eq!(receipt.report.best_move, Some(request.candidate));
            assert_eq!(receipt.report.pv.first(), Some(&request.candidate));
            assert_eq!(receipt.report.conditions.root_moves, [request.candidate]);
            assert_eq!(
                receipt.report.conditions.legal_moves,
                request.expected_legal_moves
            );
            assert_eq!(receipt.report.reused_completed_depth, 0);
            assert!(
                receipt.report.nodes > 0 && receipt.report.quiescence_nodes <= receipt.report.nodes
            );
            assert!(receipt.report.tt_hits <= receipt.report.nodes);
            assert!(receipt.report.pv_rules_validated);
            assert_eq!(
                receipt.caller_registration_scope,
                DECLARED_REGISTRATION_SCOPE
            );
            assert_eq!(receipt.binary_pin_scope, DISPATCHER_BINARY_SCOPE);
        }
    }

    #[test]
    fn exact_history_order_and_illegal_candidate_fail_before_cpu() {
        let original = request("position startpos moves g1f3 g8f6 f3g1 f6g8");
        let mut same_board = request(&format!("position fen {}", original.expected_board_fen));
        assert_eq!(same_board.expected_board_fen, original.expected_board_fen);
        assert_ne!(
            same_board.rules_history_sha256,
            original.rules_history_sha256
        );
        same_board.rules_history_sha256 = original.rules_history_sha256.clone();
        reseal(&mut same_board);
        let error = reject(&same_board);
        assert_eq!(error.stage, "rules_identity");
        assert_eq!(error.cpu_calls, 0);
        assert_eq!(error.known_nodes, None);
        let mut bad_order = original.clone();
        bad_order.expected_legal_moves.reverse();
        reseal(&mut bad_order);
        assert_eq!(reject(&bad_order).stage, "rules_identity");
        let mut bad_candidate = request("position startpos");
        bad_candidate.candidate = pack_moves(&[BoardMove::from_uci("a1a2").unwrap()]).unwrap()[0];
        reseal(&mut bad_candidate);
        assert_eq!(reject(&bad_candidate).stage, "root_admission");
    }

    #[test]
    fn pins_schema_and_exact_numeric_bounds_do_not_fallback_or_clamp() {
        let original = request("position startpos");
        let mut wrong = original.clone();
        wrong.cpu_binary_sha256 = "b".repeat(64);
        reseal(&mut wrong);
        assert_eq!(
            dispatch_started(
                &serde_json::to_vec(&wrong).unwrap(),
                &original.cpu_binary_sha256,
                Instant::now()
            )
            .unwrap_err()
            .stage,
            "binary_identity"
        );
        let mut wrong = original.clone();
        wrong.cpu_profile_sha256 = "b".repeat(64);
        wrong.context_sha256 = request_context_sha256(&wrong).unwrap();
        assert_eq!(reject(&wrong).stage, "profile_identity");
        let mut wrong = original.clone();
        wrong.before_result_anchor_sha256 = "b".repeat(64);
        assert_eq!(reject(&wrong).stage, "context_identity");
        for (horizon, nodes) in [(65, 1), (1, u64::from(u32::MAX) + 1)] {
            let mut wrong = original.clone();
            wrong.horizon = horizon;
            wrong.node_budget = nodes;
            reseal(&mut wrong);
            assert_eq!(reject(&wrong).stage, "admission");
        }
        let mut raw = serde_json::to_value(&original).unwrap();
        raw["baseline_depth"] = json!(1);
        assert_eq!(
            dispatch_started(
                &serde_json::to_vec(&raw).unwrap(),
                &original.cpu_binary_sha256,
                Instant::now()
            )
            .unwrap_err()
            .stage,
            "json_admission"
        );
        let mut wrong = original.clone();
        wrong.schema = super::super::CPU_TASK_SCHEMA.into();
        reseal(&mut wrong);
        assert_eq!(reject(&wrong).stage, "admission");
        assert!(
            super::super::request_admission(
                &serde_json::to_vec(&original).unwrap(),
                Instant::now()
            )
            .is_err()
        );
        let raw = serde_json::to_string(&original).unwrap();
        let duplicate = raw.replacen("\"horizon\":1", "\"horizon\":1,\"horizon\":1", 1);
        assert_eq!(
            dispatch_started(
                duplicate.as_bytes(),
                &original.cpu_binary_sha256,
                Instant::now()
            )
            .unwrap_err()
            .stage,
            "json_admission"
        );
    }

    #[test]
    fn tiny_budget_and_actual_cancel_keep_frontier_work_without_completion() {
        let mut request = request("position startpos");
        request.horizon = 2;
        request.node_budget = 1;
        reseal(&mut request);
        let receipt = run(&request);
        assert_eq!(receipt.status, CandidateStatus::Partial);
        assert_eq!(receipt.report.completion, SearchCompletion::NodeLimit);
        assert_eq!(receipt.report.completed_depth, 0);
        assert_eq!(receipt.report.score_scope, ScoreScope::FrontierOnly);
        assert_eq!(receipt.report.nodes, 1);
        assert_eq!(receipt.report.pv.first(), Some(&request.candidate));
        let canceled = AtomicBool::new(false);
        canceled.store(true, Ordering::Relaxed);
        let receipt: CandidateReceipt = serde_json::from_slice(
            &dispatch_started_with_cancel(
                &serde_json::to_vec(&request).unwrap(),
                &request.cpu_binary_sha256,
                Instant::now(),
                &canceled,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(receipt.status, CandidateStatus::Canceled);
        assert_eq!(receipt.cpu_calls, 1);
        assert_eq!(receipt.report.completion, SearchCompletion::Canceled);
        assert_eq!(receipt.report.nodes, 0);
        assert_eq!(receipt.report.completed_depth, 0);
    }

    #[test]
    fn original_start_and_output_failures_preserve_known_and_unknown_work() {
        let mut request = request("position startpos");
        request.max_wall_time_ms = 1;
        reseal(&mut request);
        let error = dispatch_started(
            &serde_json::to_vec(&request).unwrap(),
            &request.cpu_binary_sha256,
            Instant::now() - Duration::from_millis(50),
        )
        .unwrap_err();
        assert!(error.deadline_exceeded);
        assert!(error.elapsed_ms.unwrap() >= 50);
        assert_eq!(error.cpu_calls, 0);
        assert_eq!(error.known_nodes, None);
        request.max_wall_time_ms = 10_000;
        request.max_output_bytes = 1024;
        reseal(&mut request);
        let error = reject(&request);
        assert_eq!(error.stage, "output_bound");
        assert_eq!(error.cpu_calls, 1);
        let report = error.report.unwrap();
        assert!(report.pv_rules_validated);
        assert_eq!(error.known_nodes, Some(report.nodes));
        let work = error.failed_check_work.unwrap();
        assert_eq!(work.nodes, report.nodes);
        assert_eq!(work.quiescence_nodes, report.quiescence_nodes);
        assert_eq!(work.tt_hits, report.tt_hits);
    }

    #[test]
    fn terminal_rules_with_geometric_legal_moves_is_not_candidate_completion() {
        let request = request("position fen 8/8/8/8/8/8/5k2/7K w - - 150 80");
        assert!(!request.expected_legal_moves.is_empty());
        let error = reject(&request);
        assert_eq!(error.stage, "terminal_input");
        assert_eq!(error.cpu_calls, 0);
    }

    #[test]
    fn black_root_and_four_promotions_preserve_rules_perspective() {
        let black = request("position startpos moves e2e4");
        let receipt = run(&black);
        assert!(!receipt.report.conditions.white_to_move);
        assert_eq!(
            receipt.report.conditions.perspective,
            "captured_side_to_move"
        );
        let promotion = request("position fen 7k/P7/8/8/8/8/8/7K w - - 0 1");
        let promotions = decode_moves(&promotion.expected_legal_moves)
            .unwrap()
            .into_iter()
            .filter(|mv| mv.to_string().starts_with("a7a8"))
            .collect::<Vec<_>>();
        assert_eq!(promotions.len(), 4);
        for mv in promotions {
            let mut request = promotion.clone();
            request.candidate = pack_moves(&[mv]).unwrap()[0];
            reseal(&mut request);
            let receipt = run(&request);
            assert_eq!(receipt.report.pv[0], request.candidate);
            assert!(receipt.report.pv_rules_validated);
        }
    }
}
