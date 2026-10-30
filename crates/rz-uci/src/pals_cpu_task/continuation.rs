//! One bounded, non-learning whole-line factual check, separate from V tasks.
//!
//! Rules replays every ordered ply and owns terminal/state/history facts. A
//! nonterminal endpoint receives exactly one fresh unrestricted Independent
//! bootstrap CPU search. Its score is in the endpoint side-to-move namespace,
//! not a calibrated CP value, whole-line utility, preference or training target.
//! Terminal endpoints have no CPU report or fabricated score. Caller anchors,
//! source registration and before-result ordering require independent admission;
//! a dispatcher-supplied binary digest is not itself a loaded-image observation.

use super::candidate::{ScoreScope, SearchCompletion};
use super::semantic::{self, MoveTokenKind, RulesDescriptor, SideToMove};
use super::{
    CpuTaskAdmission, CpuTaskError, FailedCheckWork, MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES,
    MAX_WALL_TIME_MS, canonical, decode_moves, json_digest, milliseconds, pack_moves,
    replay_checked, valid_sha,
};
use crate::engine::{OwnerRegistry, RulesUciPort};
use crate::{Command, ParserLimits, PositionPort, parse};
use rz_position::contracts::ContractPosition;
use rz_position::{BoardMove, PlayStatus, Position, PositionLimits};
use rz_search::cpu::{
    BOOTSTRAP_SCORE_VERSION, CPU_SEARCH_CONDITIONS, CPU_SEARCH_VERSION, CpuCompletion, CpuConfig,
    CpuEngine, CpuLimits, CpuProfile, CpuReport, CpuScoreScope,
};
use rz_search::cpu_value::{
    BootstrapCpuValue, CpuTrainingState, CpuValueEvaluator, CpuValueIdentity,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub const CONTINUATION_SCHEMA: &str = "rz-pals-owned-cpu-line-continuation/1";
pub const CONDITIONS_SCHEMA: &str = "rz-pals-owned-cpu-line-continuation-conditions/1";
pub const LINE_DOMAIN: &str = "rz-pals-owned-cpu-ordered-line/1";
pub const DECLARED_REGISTRATION_SCOPE: &str = "declared_not_independently_verified";
pub const DISPATCHER_BINARY_SCOPE: &str = "dispatcher_compared_verified_argument";
pub const MAX_LINE_PLIES: usize = 64;

/// Independent root and endpoint expectations are declarations bound into the
/// request. Exact Rules reconstruction checks them; it does not enroll a caller
/// or prove the production time of its captured/before-result bytes.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LineContinuationRequest {
    pub schema: String,
    pub task_id: String,
    pub captured_input_sha256: String,
    pub checker_namespace_sha256: String,
    pub before_result_anchor_sha256: String,
    pub frozen_epoch: u64,
    pub input_revision: u64,
    pub position_command: String,
    pub expected_root_board_fen: String,
    pub root_rules_state_sha256: String,
    pub root_rules_history_sha256: String,
    pub expected_root_legal_moves: Vec<u16>,
    pub root_side_to_move: SideToMove,
    /// Order and repetitions are preserved; this is not a root restriction set.
    pub line: Vec<u16>,
    pub line_sha256: String,
    pub expected_endpoint_board_fen: String,
    pub endpoint_rules_state_sha256: String,
    pub endpoint_rules_history_sha256: String,
    pub expected_endpoint_legal_moves: Vec<u16>,
    pub endpoint_side_to_move: SideToMove,
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
pub enum ContinuationStatus {
    Completed,
    Partial,
    Canceled,
    RulesTerminal,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuationResourcePolicy {
    pub max_wall_time_ms: u64,
    pub max_output_bytes: usize,
    pub max_checks: u8,
    /// This reserve is subtracted from the original absolute wall allowance.
    pub search_deadline_reserve_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuationConditions {
    pub schema: String,
    pub root_rules_state_sha256: String,
    pub root_rules_history_sha256: String,
    pub root_legal_order_sha256: String,
    pub endpoint_rules_state_sha256: String,
    pub endpoint_rules_history_sha256: String,
    pub endpoint_legal_order_sha256: String,
    pub line_sha256: String,
    pub line_plies: u16,
    pub endpoint_side_to_move: SideToMove,
    pub profile_sha256: String,
    pub profile: String,
    pub value_identity: CpuValueIdentity,
    pub search_version: String,
    pub search_conditions: String,
    pub horizon: u16,
    pub node_budget: u64,
    pub tt_entries: usize,
    pub quiescence_ply: u16,
    pub restriction: String,
    pub perspective: String,
    pub resource_policy: ContinuationResourcePolicy,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuationRawReport {
    pub raw_score: i32,
    pub best_move: Option<u16>,
    pub pv: Vec<u16>,
    pub score_scope: ScoreScope,
    pub completion: SearchCompletion,
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
pub struct ContinuationReceipt {
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
    pub status: ContinuationStatus,
    pub root: RulesDescriptor,
    pub endpoint: RulesDescriptor,
    pub line: Vec<u16>,
    pub line_sha256: String,
    pub line_rules_validated: bool,
    pub conditions: ContinuationConditions,
    pub conditions_sha256: String,
    pub cpu_calls: u8,
    pub fresh_engine: bool,
    /// None at a Rules terminal. No terminal bootstrap score is invented.
    pub report: Option<ContinuationRawReport>,
    pub elapsed_ms: u64,
    pub deadline_exceeded: bool,
    pub training_target_created: bool,
    pub product_verifier_enabled: bool,
}

/// A failed check/delivery may retain an actual report or completed receipt as
/// diagnostic evidence. Its presence does not turn the failed call into a result.
/// None counters mean unobserved work, not an invented zero.
#[derive(Clone, Debug, Serialize)]
pub struct ContinuationError {
    pub schema: &'static str,
    pub code: &'static str,
    pub stage: &'static str,
    pub message: Box<String>,
    // The boxed scalar keeps this rich, unboxed error below large-Err limits;
    // JSON retains the same integer/null counter representation.
    pub known_nodes: Option<Box<u64>>,
    pub failed_check_work: Option<Box<FailedCheckWork>>,
    pub report: Option<Box<ContinuationRawReport>>,
    pub receipt: Option<Box<ContinuationReceipt>>,
    pub cpu_calls: u8,
    pub elapsed_ms: Option<u64>,
    pub deadline_exceeded: bool,
    #[serde(skip)]
    pub output_limit: usize,
}

impl ContinuationError {
    pub fn new(stage: &'static str, message: impl fmt::Display) -> Self {
        Self {
            schema: CONTINUATION_SCHEMA,
            code: "cpu_line_continuation_failed",
            stage,
            message: Box::new(message.to_string().chars().take(512).collect()),
            known_nodes: None,
            failed_check_work: None,
            report: None,
            receipt: None,
            cpu_calls: 0,
            elapsed_ms: None,
            deadline_exceeded: false,
            output_limit: 1024,
        }
    }

    pub fn with_report(mut self, report: &ContinuationRawReport) -> Self {
        self.known_nodes = Some(Box::new(report.nodes));
        self.failed_check_work = Some(Box::new(FailedCheckWork {
            nodes: report.nodes,
            quiescence_nodes: report.quiescence_nodes,
            tt_hits: report.tt_hits,
        }));
        self.report = Some(Box::new(report.clone()));
        self.cpu_calls = 1;
        self
    }

    pub fn with_receipt(mut self, receipt: &ContinuationReceipt) -> Self {
        if let Some(report) = &receipt.report {
            self = self.with_report(report);
        } else {
            self.cpu_calls = receipt.cpu_calls;
            // Only this actual completed zero-call Rules receipt proves zero.
            self.known_nodes = Some(Box::new(0));
            self.failed_check_work = Some(Box::new(FailedCheckWork {
                nodes: 0,
                quiescence_nodes: 0,
                tt_hits: 0,
            }));
        }
        self.receipt = Some(Box::new(receipt.clone()));
        self
    }
}

impl fmt::Display for ContinuationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.stage, self.message)
    }
}
impl std::error::Error for ContinuationError {}

fn shared_error(error: CpuTaskError) -> ContinuationError {
    ContinuationError::new(error.stage, error.message)
}
fn descriptor_error(error: semantic::SemanticError) -> ContinuationError {
    let mut converted = ContinuationError::new(error.stage, error.message);
    converted.deadline_exceeded = error.deadline_exceeded;
    converted
}

pub fn profile_description(request: &LineContinuationRequest) -> Value {
    json!({"domain":CONTINUATION_SCHEMA,"search":CPU_SEARCH_VERSION,
        "evaluator":BOOTSTRAP_SCORE_VERSION,"profile":CpuProfile::Independent.identity(),
        "tt_entries":request.tt_entries,"max_depth":request.horizon,
        "quiescence_ply":request.quiescence_ply,"selective_reductions":false})
}

pub fn profile_sha256(request: &LineContinuationRequest) -> Result<String, ContinuationError> {
    json_digest(&profile_description(request)).map_err(shared_error)
}

/// Full known root history belongs to the ordered-line namespace. Neither
/// sorting/deduplication nor a digest of endpoint board alone identifies a line.
pub fn line_sha256(request: &LineContinuationRequest) -> Result<String, ContinuationError> {
    json_digest(&json!([
        LINE_DOMAIN,
        request.root_rules_state_sha256,
        request.root_rules_history_sha256,
        request.line
    ]))
    .map_err(shared_error)
}

/// Canonical sorted-key JSON [schema, complete request excluding context_sha256].
pub fn request_context_sha256(
    request: &LineContinuationRequest,
) -> Result<String, ContinuationError> {
    let mut value = serde_json::to_value(request)
        .map_err(|error| ContinuationError::new("context_identity", error))?;
    value
        .as_object_mut()
        .ok_or_else(|| ContinuationError::new("context_identity", "request is not an object"))?
        .remove("context_sha256");
    json_digest(&json!([CONTINUATION_SCHEMA, value])).map_err(shared_error)
}

fn decode_request(bytes: &[u8]) -> Result<LineContinuationRequest, ContinuationError> {
    if bytes.is_empty() || bytes.len() > MAX_REQUEST_BYTES {
        return Err(ContinuationError::new(
            "admission",
            "request exceeds 1..=512KiB",
        ));
    }
    // Typed serde rejects unknown and duplicate fields at this boundary.
    let request: LineContinuationRequest = serde_json::from_slice(bytes)
        .map_err(|error| ContinuationError::new("json_admission", error))?;
    if request.schema != CONTINUATION_SCHEMA
        || request.task_id.is_empty()
        || request.task_id.len() > 128
        || !request.task_id.bytes().all(|byte| byte.is_ascii_graphic())
        || request.line.is_empty()
        || request.line.len() > MAX_LINE_PLIES
        || request.expected_root_legal_moves.len() > 256
        || request.expected_endpoint_legal_moves.len() > 256
        || request.position_command.len() > ParserLimits::default().max_line_bytes
        || request.expected_root_board_fen.len() > PositionLimits::default().max_fen_bytes
        || request.expected_endpoint_board_fen.len() > PositionLimits::default().max_fen_bytes
        || !(1..=64).contains(&request.horizon)
        || !(1..=u64::from(u32::MAX)).contains(&request.node_budget)
        || request.tt_entries > 1_048_576
        || request.quiescence_ply > 32
        || !(1..=MAX_WALL_TIME_MS).contains(&request.max_wall_time_ms)
        || !(1024..=MAX_RESPONSE_BYTES).contains(&request.max_output_bytes)
    {
        return Err(ContinuationError::new(
            "admission",
            "unsupported schema or finite bounds",
        ));
    }
    for pin in [
        &request.captured_input_sha256,
        &request.checker_namespace_sha256,
        &request.before_result_anchor_sha256,
        &request.root_rules_state_sha256,
        &request.root_rules_history_sha256,
        &request.endpoint_rules_state_sha256,
        &request.endpoint_rules_history_sha256,
        &request.cpu_binary_sha256,
        &request.cpu_profile_sha256,
        &request.line_sha256,
        &request.context_sha256,
    ] {
        if !valid_sha(pin) {
            return Err(ContinuationError::new(
                "admission",
                "identity must be lowercase SHA256",
            ));
        }
    }
    decode_moves(&request.line).map_err(shared_error)?;
    // Repetitions are meaningful in a line. Only complete legal arrays are sets.
    for legal in [
        &request.expected_root_legal_moves,
        &request.expected_endpoint_legal_moves,
    ] {
        decode_moves(legal).map_err(shared_error)?;
        if legal
            .iter()
            .enumerate()
            .any(|(at, movement)| legal[..at].contains(movement))
        {
            return Err(ContinuationError::new(
                "admission",
                "duplicate complete legal move",
            ));
        }
    }
    Ok(request)
}

fn absolute_deadline(
    request: &LineContinuationRequest,
    started: Instant,
) -> Result<Instant, ContinuationError> {
    started
        .checked_add(Duration::from_millis(request.max_wall_time_ms))
        .ok_or_else(|| ContinuationError::new("admission", "absolute deadline overflow"))
}

fn check_deadline(deadline: Instant, stage: &'static str) -> Result<(), ContinuationError> {
    if Instant::now() >= deadline {
        let mut error = ContinuationError::new(stage, "admitted stage deadline expired");
        error.deadline_exceeded = true;
        Err(error)
    } else {
        Ok(())
    }
}

pub fn request_admission(
    bytes: &[u8],
    started: Instant,
) -> Result<CpuTaskAdmission, ContinuationError> {
    let request = decode_request(bytes)?;
    let deadline = absolute_deadline(&request, started)?;
    check_deadline(deadline, "admission_deadline")?;
    Ok(CpuTaskAdmission {
        deadline,
        output_limit: request.max_output_bytes,
    })
}

fn validate_pins(request: &LineContinuationRequest, binary: &str) -> Result<(), ContinuationError> {
    if !valid_sha(binary) || request.cpu_binary_sha256 != binary {
        return Err(ContinuationError::new(
            "binary_identity",
            "verified image argument differs",
        ));
    }
    if request.cpu_profile_sha256 != profile_sha256(request)? {
        return Err(ContinuationError::new(
            "profile_identity",
            "fixed CPU profile digest differs",
        ));
    }
    if request.line_sha256 != line_sha256(request)? {
        return Err(ContinuationError::new(
            "line_identity",
            "ordered line/root history digest differs",
        ));
    }
    if request.context_sha256 != request_context_sha256(request)? {
        return Err(ContinuationError::new(
            "context_identity",
            "complete request context differs",
        ));
    }
    Ok(())
}

fn replay_line(
    root: &Position,
    line: &[BoardMove],
    owners: &OwnerRegistry,
    deadline: Instant,
    cancellation: &AtomicBool,
) -> Result<Position, ContinuationError> {
    let owner = owners
        .allocate()
        .map_err(|error| ContinuationError::new("rules_replay", error))?;
    let mut current = ContractPosition::new(owner, root.clone());
    for &movement in line {
        check_deadline(deadline, "rules_replay_deadline")?;
        if cancellation.load(Ordering::Relaxed) {
            return Err(ContinuationError::new(
                "rules_replay_canceled",
                "canceled before a line ply",
            ));
        }
        let view = current
            .export()
            .map_err(|error| ContinuationError::new("rules_replay", error))?;
        // Terminal Rules states may still have geometric legal moves. Playing
        // those would falsely extend the supplied factual continuation.
        if !matches!(
            view.rules().classification().play_status,
            PlayStatus::Ongoing
        ) {
            return Err(ContinuationError::new(
                "line_after_terminal",
                "line continues after Rules termination",
            ));
        }
        let movement = rz_contracts::Move::try_from(movement)
            .map_err(|error| ContinuationError::new("rules_replay", error))?;
        current
            .make_from_view(&view, movement)
            .map_err(|error| ContinuationError::new("rules_replay", error))?;
    }
    check_deadline(deadline, "rules_replay_deadline")?;
    Ok(current.into_position())
}

fn descriptor_matches(
    descriptor: &RulesDescriptor,
    fen: &str,
    state: &str,
    history: &str,
    legal: &[u16],
    turn: SideToMove,
) -> bool {
    descriptor.board_fen == fen
        && descriptor.rules_state_sha256 == state
        && descriptor.rules_history_sha256 == history
        && descriptor.legal_moves == legal
        && descriptor.side_to_move == turn
}

fn conditions(
    request: &LineContinuationRequest,
    root: &RulesDescriptor,
    endpoint: &RulesDescriptor,
) -> Result<ContinuationConditions, ContinuationError> {
    let value_identity = BootstrapCpuValue::default().identity().clone();
    value_identity
        .validate()
        .map_err(|error| ContinuationError::new("value_identity", error))?;
    Ok(ContinuationConditions {
        schema: CONDITIONS_SCHEMA.into(),
        root_rules_state_sha256: root.rules_state_sha256.clone(),
        root_rules_history_sha256: root.rules_history_sha256.clone(),
        root_legal_order_sha256: root.legal_order_sha256.clone(),
        endpoint_rules_state_sha256: endpoint.rules_state_sha256.clone(),
        endpoint_rules_history_sha256: endpoint.rules_history_sha256.clone(),
        endpoint_legal_order_sha256: endpoint.legal_order_sha256.clone(),
        line_sha256: request.line_sha256.clone(),
        line_plies: request.line.len() as u16,
        endpoint_side_to_move: endpoint.side_to_move,
        profile_sha256: profile_sha256(request)?,
        profile: CpuProfile::Independent.identity().into(),
        value_identity,
        search_version: CPU_SEARCH_VERSION.into(),
        search_conditions: format!(
            "{CPU_SEARCH_CONDITIONS};profile={};max_depth={};q_plies={};tt_entries={}",
            CpuProfile::Independent.identity(),
            request.horizon,
            request.quiescence_ply,
            request.tt_entries
        ),
        horizon: request.horizon,
        node_budget: request.node_budget,
        tt_entries: request.tt_entries,
        quiescence_ply: request.quiescence_ply,
        restriction: "unrestricted_endpoint".into(),
        perspective: "endpoint_side_to_move".into(),
        resource_policy: ContinuationResourcePolicy {
            max_wall_time_ms: request.max_wall_time_ms,
            max_checks: 1,
            max_output_bytes: request.max_output_bytes,
            search_deadline_reserve_ms: (request.max_wall_time_ms / 10).min(1000),
        },
    })
}

fn capture_report(
    native: &CpuReport,
    horizon: u16,
) -> Result<ContinuationRawReport, ContinuationError> {
    Ok(ContinuationRawReport {
        raw_score: native.score,
        best_move: pack_moves(&native.best_move.into_iter().collect::<Vec<_>>())
            .map_err(shared_error)?
            .first()
            .copied(),
        pv: pack_moves(&native.pv).map_err(shared_error)?,
        score_scope: match native.score_scope {
            CpuScoreScope::FrontierOnly => ScoreScope::FrontierOnly,
            CpuScoreScope::CompletedIteration => ScoreScope::CompletedIteration,
            CpuScoreScope::RulesTerminal => ScoreScope::RulesTerminal,
        },
        completion: match native.completion {
            CpuCompletion::DepthLimit => SearchCompletion::DepthLimit,
            CpuCompletion::NodeLimit => SearchCompletion::NodeLimit,
            CpuCompletion::Deadline => SearchCompletion::Deadline,
            CpuCompletion::Canceled => SearchCompletion::Canceled,
            CpuCompletion::Yielded => {
                return Err(ContinuationError::new(
                    "unsupported_completion",
                    "cooperative Yield requires the adaptive Job/Slice receipt",
                ));
            }
            CpuCompletion::QuiescenceLimit => SearchCompletion::QuiescenceLimit,
            CpuCompletion::AwaitReciprocal => {
                return Err(ContinuationError::new(
                    "unsupported_completion",
                    "reciprocal CPU wait requires the Rovent question receipt",
                ));
            }
            CpuCompletion::Terminal(_) => SearchCompletion::RulesTerminal,
        },
        completed_depth: native.completed_depth,
        requested_depth: horizon,
        nodes: native.nodes,
        quiescence_nodes: native.quiescence_nodes,
        tt_hits: native.tt_hits,
        elapsed_ms: milliseconds(native.elapsed),
        reused_completed_depth: native.reused_completed_depth,
        root_restricted: native.root_restricted,
        score_provenance: native.score_provenance.into(),
        pv_rules_validated: false,
    })
}

fn observed_work(error: &mut ContinuationError, work: rz_search::cpu::CpuWork) {
    error.cpu_calls = 1;
    error.known_nodes = Some(Box::new(work.nodes));
    error.failed_check_work = Some(Box::new(FailedCheckWork {
        nodes: work.nodes,
        quiescence_nodes: work.quiescence_nodes,
        tt_hits: work.tt_hits,
    }));
}

fn check_endpoint(
    request: &LineContinuationRequest,
    endpoint: &Position,
    conditions: &ContinuationConditions,
    owners: &OwnerRegistry,
    deadline: Instant,
    cancellation: &AtomicBool,
) -> Result<ContinuationRawReport, ContinuationError> {
    let search_deadline = deadline
        .checked_sub(Duration::from_millis(
            conditions.resource_policy.search_deadline_reserve_ms,
        ))
        .ok_or_else(|| ContinuationError::new("admission", "search deadline overflow"))?;
    check_deadline(search_deadline, "search_admission_deadline")?;
    let mut cpu = CpuEngine::new(CpuConfig {
        profile: CpuProfile::Independent,
        tt_entries: request.tt_entries,
        max_depth: request.horizon,
        quiescence_ply: request.quiescence_ply,
    })
    .map_err(|error| ContinuationError::new("cpu_prepare", error))?;
    if cpu.value_identity() != &conditions.value_identity
        || cpu.search_conditions() != conditions.search_conditions
    {
        return Err(ContinuationError::new(
            "cpu_identity",
            "actual fresh CPU value/conditions differ",
        ));
    }
    check_deadline(search_deadline, "search_admission_deadline")?;
    // The only CPU search invocation. The whole supplied line is already
    // replayed; endpoint moves are unrestricted, with a new TT/history owner.
    let native = cpu
        .analyze(
            endpoint,
            CpuLimits {
                max_depth: request.horizon,
                max_nodes: request.node_budget,
                deadline: Some(search_deadline),
            },
            cancellation,
        )
        .map_err(|cause| {
            let mut error = ContinuationError::new("endpoint_cpu", cause);
            error.cpu_calls = 1;
            if let Some(work) = cpu.last_attempt_work() {
                observed_work(&mut error, work);
            }
            error
        })?;
    let mut raw = capture_report(&native, request.horizon).map_err(|mut error| {
        observed_work(
            &mut error,
            rz_search::cpu::CpuWork {
                nodes: native.nodes,
                quiescence_nodes: native.quiescence_nodes,
                tt_hits: native.tt_hits,
            },
        );
        error
    })?;
    let validate = || -> Result<(), ContinuationError> {
        if native.search_version != CPU_SEARCH_VERSION
            || native.profile != CpuProfile::Independent
            || native.value_identity != conditions.value_identity
            || native.value_identity.semantics != BOOTSTRAP_SCORE_VERSION
            || native.value_identity.weights_sha256.is_some()
            || native.value_identity.training != CpuTrainingState::Bootstrap
            || native.score_provenance != BOOTSTRAP_SCORE_VERSION
            || native.root_restricted
            || native.reused_completed_depth != 0
            || native.completed_depth > request.horizon
            || native.nodes > request.node_budget
            || native.quiescence_nodes > native.nodes
            || native.tt_hits > native.nodes
            || native.pv.len() > usize::from(request.horizon + request.quiescence_ply)
            || (native.score_scope == CpuScoreScope::CompletedIteration)
                != (native.completed_depth > 0)
            || native.score_scope == CpuScoreScope::RulesTerminal
            || matches!(native.completion, CpuCompletion::Terminal(_))
            || (native.completion == CpuCompletion::DepthLimit
                && native.completed_depth != request.horizon)
            || native.best_move.is_none()
            || native.pv.first().copied() != native.best_move
        {
            return Err(ContinuationError::new(
                "cpu_report",
                "fresh endpoint report violates profile/resource/scope bounds",
            ));
        }
        replay_checked(endpoint, &native.pv, owners, deadline).map_err(shared_error)?;
        check_deadline(deadline, "cpu_report")
    };
    validate().map_err(|error| error.with_report(&raw))?;
    raw.pv_rules_validated = true;
    Ok(raw)
}

fn execute(
    request: &LineContinuationRequest,
    binary: &str,
    started: Instant,
    cancellation: &AtomicBool,
) -> Result<ContinuationReceipt, ContinuationError> {
    let deadline = absolute_deadline(request, started)?;
    check_deadline(deadline, "admission_deadline")?;
    validate_pins(request, binary)?;
    let Command::Position(spec) = parse(&request.position_command, ParserLimits::default())
        .map_err(|error| ContinuationError::new("position_admission", error))?
    else {
        return Err(ContinuationError::new(
            "position_admission",
            "only a position command is accepted",
        ));
    };
    let owners = Arc::new(OwnerRegistry::default());
    let prepared = RulesUciPort::new(Arc::clone(&owners), PositionLimits::default())
        .prepare(&spec)
        .map_err(|error| ContinuationError::new("rules_prepare", error))?;
    check_deadline(deadline, "rules_prepare")?;
    let root_position = prepared.snapshot.rules_position();
    let root = semantic::describe(root_position, &owners, MoveTokenKind::RootLegal, deadline)
        .map_err(descriptor_error)?;
    if !descriptor_matches(
        &root,
        &request.expected_root_board_fen,
        &request.root_rules_state_sha256,
        &request.root_rules_history_sha256,
        &request.expected_root_legal_moves,
        request.root_side_to_move,
    ) {
        return Err(ContinuationError::new(
            "root_identity",
            "actual root Rules state/full history/legal order/turn differs",
        ));
    }
    let endpoint_position = replay_line(
        root_position,
        &decode_moves(&request.line).map_err(shared_error)?,
        &owners,
        deadline,
        cancellation,
    )?;
    let endpoint = semantic::describe(
        &endpoint_position,
        &owners,
        MoveTokenKind::ClaimEndLegal,
        deadline,
    )
    .map_err(descriptor_error)?;
    if !descriptor_matches(
        &endpoint,
        &request.expected_endpoint_board_fen,
        &request.endpoint_rules_state_sha256,
        &request.endpoint_rules_history_sha256,
        &request.expected_endpoint_legal_moves,
        request.endpoint_side_to_move,
    ) {
        return Err(ContinuationError::new(
            "endpoint_identity",
            "actual endpoint Rules state/full history/legal order/turn differs",
        ));
    }
    let conditions = conditions(request, &root, &endpoint)?;
    let conditions_sha256 = json_digest(
        &serde_json::to_value(&conditions)
            .map_err(|error| ContinuationError::new("conditions_identity", error))?,
    )
    .map_err(shared_error)?;
    let terminal = matches!(
        endpoint_position
            .play_status_from_view(&endpoint_position.ordered_legal_moves())
            .map_err(|error| ContinuationError::new("rules_terminal", error))?,
        PlayStatus::Terminal { .. }
    );
    let report = if terminal {
        None
    } else {
        Some(check_endpoint(
            request,
            &endpoint_position,
            &conditions,
            &owners,
            deadline,
            cancellation,
        )?)
    };
    let status = match report.as_ref().map(|report| report.completion) {
        None => ContinuationStatus::RulesTerminal,
        Some(SearchCompletion::DepthLimit) => ContinuationStatus::Completed,
        Some(SearchCompletion::Canceled) => ContinuationStatus::Canceled,
        Some(_) => ContinuationStatus::Partial,
    };
    Ok(ContinuationReceipt {
        schema: CONTINUATION_SCHEMA.into(),
        task_id: request.task_id.clone(),
        captured_input_sha256: request.captured_input_sha256.clone(),
        checker_namespace_sha256: request.checker_namespace_sha256.clone(),
        before_result_anchor_sha256: request.before_result_anchor_sha256.clone(),
        frozen_epoch: request.frozen_epoch,
        input_revision: request.input_revision,
        context_sha256: request.context_sha256.clone(),
        cpu_binary_sha256: binary.into(),
        binary_pin_scope: DISPATCHER_BINARY_SCOPE.into(),
        caller_registration_scope: DECLARED_REGISTRATION_SCOPE.into(),
        status,
        root,
        endpoint,
        line: request.line.clone(),
        line_sha256: request.line_sha256.clone(),
        line_rules_validated: true,
        conditions,
        conditions_sha256,
        cpu_calls: u8::from(!terminal),
        fresh_engine: !terminal,
        report,
        elapsed_ms: milliseconds(started.elapsed()),
        deadline_exceeded: false,
        training_target_created: false,
        product_verifier_enabled: false,
    })
}

fn finish(
    mut receipt: ContinuationReceipt,
    request: &LineContinuationRequest,
    started: Instant,
) -> Result<Vec<u8>, ContinuationError> {
    let deadline =
        absolute_deadline(request, started).map_err(|error| error.with_receipt(&receipt))?;
    check_deadline(deadline, "receipt_deadline").map_err(|error| error.with_receipt(&receipt))?;
    receipt.elapsed_ms = milliseconds(started.elapsed());
    let value = serde_json::to_value(&receipt)
        .map_err(|error| ContinuationError::new("serialization", error).with_receipt(&receipt))?;
    let mut bytes =
        canonical(&value).map_err(|error| shared_error(error).with_receipt(&receipt))?;
    bytes.push(b'\n');
    if bytes.len() > request.max_output_bytes {
        return Err(ContinuationError::new(
            "output_bound",
            "continuation receipt exceeds admitted output bound",
        )
        .with_receipt(&receipt));
    }
    check_deadline(deadline, "receipt_deadline").map_err(|error| error.with_receipt(&receipt))?;
    Ok(bytes)
}

pub fn dispatch_started(
    bytes: &[u8],
    verified_own_binary_sha256: &str,
    started: Instant,
) -> Result<Vec<u8>, ContinuationError> {
    dispatch_started_with_cancel(
        bytes,
        verified_own_binary_sha256,
        started,
        &AtomicBool::new(false),
    )
}

/// Supervisor termination is a missing result, not an in-process cancellation
/// receipt. This API retains actual cancellation/partial work when observed.
pub fn dispatch_started_with_cancel(
    bytes: &[u8],
    verified_own_binary_sha256: &str,
    started: Instant,
    cancellation: &AtomicBool,
) -> Result<Vec<u8>, ContinuationError> {
    let request = decode_request(bytes)?;
    execute(&request, verified_own_binary_sha256, started, cancellation)
        .and_then(|receipt| finish(receipt, &request, started))
        .map_err(|mut error| {
            error.output_limit = request.max_output_bytes;
            error.elapsed_ms = Some(milliseconds(started.elapsed()));
            // Search reserve expiry is a distinct stage failure; this flag is
            // exclusively the original request wall deadline, never the reserve.
            error.deadline_exceeded =
                started.elapsed() >= Duration::from_millis(request.max_wall_time_ms);
            error
        })
}

pub fn capabilities() -> Result<Vec<u8>, ContinuationError> {
    let value = json!({"schema":CONTINUATION_SCHEMA,"max_checks":1,"terminal_checks":0,
        "max_line_plies":MAX_LINE_PLIES,"max_horizon":64,"max_node_budget":u32::MAX,
        "max_tt_entries":1_048_576,"max_quiescence_ply":32,"max_request_bytes":MAX_REQUEST_BYTES,
        "max_response_bytes":MAX_RESPONSE_BYTES,"max_wall_time_ms":MAX_WALL_TIME_MS,
        "fresh_engine":true,"restriction":"unrestricted_endpoint","resume":false,
        "profile":CpuProfile::Independent.identity(),"search_version":CPU_SEARCH_VERSION,
        "score_semantics":BOOTSTRAP_SCORE_VERSION,"score_units":"uncalibrated_endpoint_side_to_move_raw",
        "rules_descriptor_schema":semantic::DESCRIPTOR_SCHEMA,"line_domain":LINE_DOMAIN,
        "conditions_schema":CONDITIONS_SCHEMA,"caller_registration_scope":DECLARED_REGISTRATION_SCOPE,
        "binary_pin_scope":DISPATCHER_BINARY_SCOPE,"training_target_created":false,
        "product_verifier_enabled":false,"teacher_gpu":false,"ranking_created":false});
    let mut bytes = canonical(&value).map_err(shared_error)?;
    bytes.push(b'\n');
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_position::Color;

    fn fixture(command: &str, line: &[&str]) -> LineContinuationRequest {
        let Command::Position(spec) = parse(command, ParserLimits::default()).unwrap() else {
            panic!("position fixture");
        };
        let owners = Arc::new(OwnerRegistry::default());
        let prepared = RulesUciPort::new(Arc::clone(&owners), PositionLimits::default())
            .prepare(&spec)
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        let root_position = prepared.snapshot.rules_position();
        let moves = line
            .iter()
            .map(|movement| BoardMove::from_uci(movement).unwrap())
            .collect::<Vec<_>>();
        let endpoint_position = replay_line(
            root_position,
            &moves,
            &owners,
            deadline,
            &AtomicBool::new(false),
        )
        .unwrap();
        let root =
            semantic::describe(root_position, &owners, MoveTokenKind::RootLegal, deadline).unwrap();
        let endpoint = semantic::describe(
            &endpoint_position,
            &owners,
            MoveTokenKind::ClaimEndLegal,
            deadline,
        )
        .unwrap();
        let mut request = LineContinuationRequest {
            schema: CONTINUATION_SCHEMA.into(),
            task_id: "line-fixture".into(),
            captured_input_sha256: "1".repeat(64),
            checker_namespace_sha256: "2".repeat(64),
            before_result_anchor_sha256: "3".repeat(64),
            frozen_epoch: 4,
            input_revision: 5,
            position_command: command.into(),
            expected_root_board_fen: root.board_fen,
            root_rules_state_sha256: root.rules_state_sha256,
            root_rules_history_sha256: root.rules_history_sha256,
            expected_root_legal_moves: root.legal_moves,
            root_side_to_move: root.side_to_move,
            line: pack_moves(&moves).unwrap(),
            line_sha256: String::new(),
            expected_endpoint_board_fen: endpoint.board_fen,
            endpoint_rules_state_sha256: endpoint.rules_state_sha256,
            endpoint_rules_history_sha256: endpoint.rules_history_sha256,
            expected_endpoint_legal_moves: endpoint.legal_moves,
            endpoint_side_to_move: endpoint.side_to_move,
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

    fn reseal(request: &mut LineContinuationRequest) {
        request.cpu_profile_sha256 = profile_sha256(request).unwrap();
        request.line_sha256 = line_sha256(request).unwrap();
        request.context_sha256 = request_context_sha256(request).unwrap();
    }

    fn run(request: &LineContinuationRequest) -> ContinuationReceipt {
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

    fn reject(request: &LineContinuationRequest) -> ContinuationError {
        dispatch_started(
            &serde_json::to_vec(request).unwrap(),
            &request.cpu_binary_sha256,
            Instant::now(),
        )
        .unwrap_err()
    }

    #[test]
    fn ordered_legal_repeated_move_replays_whole_line_then_one_fresh_endpoint() {
        let request = fixture(
            "position startpos",
            &["g1f3", "g8f6", "f3g1", "f6g8", "g1f3"],
        );
        assert_eq!(request.line[0], request.line[4]);
        let receipt = run(&request);
        assert_eq!(receipt.line, request.line);
        assert_eq!(receipt.line_sha256, request.line_sha256);
        assert_eq!(receipt.status, ContinuationStatus::Completed);
        assert_eq!(receipt.cpu_calls, 1);
        assert!(receipt.fresh_engine && receipt.line_rules_validated);
        assert_eq!(
            receipt.endpoint.rules_history_sha256,
            request.endpoint_rules_history_sha256
        );
        assert_eq!(
            receipt.endpoint.legal_moves,
            request.expected_endpoint_legal_moves
        );
        let report = receipt.report.unwrap();
        assert_eq!(report.completion, SearchCompletion::DepthLimit);
        assert_eq!(report.completed_depth, 1);
        assert_eq!(report.reused_completed_depth, 0);
        assert!(!report.root_restricted);
        assert!(report.pv_rules_validated && report.nodes > 0);
        assert!(report.quiescence_nodes <= report.nodes && report.tt_hits <= report.nodes);
        assert!(!receipt.training_target_created && !receipt.product_verifier_enabled);
    }

    #[test]
    fn terminal_endpoint_has_exact_rules_facts_and_no_cpu_score() {
        for request in [
            fixture("position startpos", &["f2f3", "e7e5", "g2g4", "d8h4"]),
            fixture("position fen 7k/8/8/8/8/8/R7/K7 w - - 149 80", &["a2a3"]),
        ] {
            let receipt = run(&request);
            assert_eq!(receipt.status, ContinuationStatus::RulesTerminal);
            assert_eq!(receipt.endpoint.play_status, "rules_terminal");
            assert!(receipt.endpoint.terminal_reason.is_some());
            assert_eq!(
                receipt.endpoint.terminal_source.as_deref(),
                Some("rz-position-rules")
            );
            assert_eq!(receipt.cpu_calls, 0);
            assert!(!receipt.fresh_engine);
            assert!(receipt.report.is_none());
            let raw = serde_json::to_value(receipt).unwrap();
            assert!(raw["report"].is_null());
        }
    }

    #[test]
    fn geometric_legal_move_after_automatic_terminal_is_rejected_before_cpu() {
        let mut request = fixture("position fen 7k/8/8/8/8/8/R7/K7 w - - 149 80", &["a2a3"]);
        assert!(!request.expected_endpoint_legal_moves.is_empty());
        request.line.push(request.expected_endpoint_legal_moves[0]);
        reseal(&mut request);
        let error = reject(&request);
        assert_eq!(error.stage, "line_after_terminal");
        assert_eq!(error.cpu_calls, 0);
        assert!(error.known_nodes.is_none() && error.report.is_none());
    }

    #[test]
    fn root_and_endpoint_full_history_order_and_turn_are_independent() {
        let request = fixture("position startpos moves g1f3 g8f6 f3g1 f6g8", &["e2e4"]);
        let mut wrong = request.clone();
        wrong.position_command = format!("position fen {}", request.expected_root_board_fen);
        reseal(&mut wrong);
        assert_eq!(reject(&wrong).stage, "root_identity");
        let mut wrong = request.clone();
        wrong.endpoint_rules_history_sha256 = request.root_rules_history_sha256.clone();
        reseal(&mut wrong);
        assert_eq!(reject(&wrong).stage, "endpoint_identity");
        let mut wrong = request.clone();
        wrong.expected_endpoint_legal_moves.reverse();
        reseal(&mut wrong);
        assert_eq!(reject(&wrong).stage, "endpoint_identity");
        let mut wrong = request.clone();
        wrong.endpoint_side_to_move = SideToMove::White;
        reseal(&mut wrong);
        assert_eq!(reject(&wrong).stage, "endpoint_identity");
        let mut wrong = request.clone();
        wrong.line = pack_moves(&[BoardMove::from_uci("a1a2").unwrap()]).unwrap();
        reseal(&mut wrong);
        assert_eq!(reject(&wrong).stage, "rules_replay");
    }

    #[test]
    fn black_endpoint_raw_score_is_not_flipped_to_the_captured_root() {
        let request = fixture("position startpos", &["e2e4"]);
        assert_eq!(request.root_side_to_move, SideToMove::White);
        assert_eq!(request.endpoint_side_to_move, SideToMove::Black);
        let receipt = run(&request);
        assert_eq!(receipt.conditions.perspective, "endpoint_side_to_move");
        let Command::Position(spec) =
            parse("position startpos moves e2e4", ParserLimits::default()).unwrap()
        else {
            panic!("position fixture");
        };
        let prepared = RulesUciPort::new(
            Arc::new(OwnerRegistry::default()),
            PositionLimits::default(),
        )
        .prepare(&spec)
        .unwrap();
        let endpoint = prepared.snapshot.rules_position();
        assert_eq!(endpoint.side_to_move(), Color::Black);
        let mut reference = CpuEngine::new(CpuConfig {
            profile: CpuProfile::Independent,
            tt_entries: request.tt_entries,
            max_depth: request.horizon,
            quiescence_ply: request.quiescence_ply,
        })
        .unwrap();
        let actual = reference
            .analyze(
                endpoint,
                CpuLimits {
                    max_depth: request.horizon,
                    max_nodes: request.node_budget,
                    deadline: Some(Instant::now() + Duration::from_secs(5)),
                },
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(actual.completion, CpuCompletion::DepthLimit);
        assert_eq!(receipt.report.unwrap().raw_score, actual.score);
    }

    #[test]
    fn node_partial_preserves_actual_frontier_and_no_completion_claim() {
        let mut request = fixture("position startpos", &["e2e4"]);
        request.horizon = 2;
        request.node_budget = 1;
        reseal(&mut request);
        let receipt = run(&request);
        assert_eq!(receipt.status, ContinuationStatus::Partial);
        assert_eq!(receipt.cpu_calls, 1);
        let report = receipt.report.unwrap();
        assert_eq!(report.completion, SearchCompletion::NodeLimit);
        assert_eq!(report.score_scope, ScoreScope::FrontierOnly);
        assert_eq!(report.completed_depth, 0);
        assert_eq!(report.nodes, 1);
        assert_eq!(report.reused_completed_depth, 0);
    }

    #[test]
    fn cancellation_before_replay_and_actual_cpu_cancellation_are_distinct() {
        let request = fixture("position startpos", &["e2e4"]);
        let canceled = AtomicBool::new(true);
        let error = dispatch_started_with_cancel(
            &serde_json::to_vec(&request).unwrap(),
            &request.cpu_binary_sha256,
            Instant::now(),
            &canceled,
        )
        .unwrap_err();
        assert_eq!(error.stage, "rules_replay_canceled");
        assert_eq!(error.cpu_calls, 0);
        assert!(error.known_nodes.is_none());
        // Directly enter the same production endpoint function after an actual
        // checked line; this deterministic fixture cancels at CPU admission.
        let owners = Arc::new(OwnerRegistry::default());
        let Command::Position(spec) =
            parse("position startpos moves e2e4", ParserLimits::default()).unwrap()
        else {
            panic!("position fixture");
        };
        let prepared = RulesUciPort::new(Arc::clone(&owners), PositionLimits::default())
            .prepare(&spec)
            .unwrap();
        let endpoint = prepared.snapshot.rules_position();
        let receipt = run(&request);
        let raw = check_endpoint(
            &request,
            endpoint,
            &receipt.conditions,
            &owners,
            Instant::now() + Duration::from_millis(request.max_wall_time_ms),
            &canceled,
        )
        .unwrap();
        assert_eq!(raw.completion, SearchCompletion::Canceled);
        assert_eq!(raw.nodes, 0);
        assert_eq!(raw.completed_depth, 0);
        assert_eq!(raw.score_scope, ScoreScope::FrontierOnly);
    }

    #[test]
    fn original_deadline_and_bounded_output_keep_actual_or_unknown_work() {
        let mut request = fixture("position startpos", &["e2e4"]);
        request.max_wall_time_ms = 1;
        reseal(&mut request);
        let bytes = serde_json::to_vec(&request).unwrap();
        let started = Instant::now() - Duration::from_millis(50);
        let error = request_admission(&bytes, started).unwrap_err();
        assert!(error.deadline_exceeded);
        let error = dispatch_started(&bytes, &request.cpu_binary_sha256, started).unwrap_err();
        assert!(error.deadline_exceeded && error.elapsed_ms.unwrap() >= 50);
        assert_eq!(error.cpu_calls, 0);
        assert!(error.known_nodes.is_none());
        request.max_wall_time_ms = 10_000;
        request.max_output_bytes = 1024;
        reseal(&mut request);
        let error = reject(&request);
        assert_eq!(error.stage, "output_bound");
        assert_eq!(error.cpu_calls, 1);
        let raw = error.report.as_ref().unwrap();
        assert!(raw.pv_rules_validated);
        assert_eq!(error.known_nodes.as_deref(), Some(&raw.nodes));
        assert_eq!(error.failed_check_work.as_ref().unwrap().nodes, raw.nodes);
        assert!(error.receipt.is_some());
    }

    #[test]
    fn line_promotion_order_and_all_seals_are_preserved_without_legacy_coercion() {
        let mut seals = Vec::new();
        for movement in ["a7a8q", "a7a8r", "a7a8b", "a7a8n"] {
            let request = fixture("position fen 7k/P7/8/8/8/8/8/7K w - - 0 1", &[movement]);
            let receipt = run(&request);
            assert_eq!(receipt.line, request.line);
            assert_eq!(
                receipt.endpoint.board_fen,
                request.expected_endpoint_board_fen
            );
            seals.push(receipt.line_sha256);
        }
        assert!(
            seals
                .iter()
                .enumerate()
                .all(|(at, pin)| !seals[..at].contains(pin))
        );
        let original = fixture("position startpos", &["e2e4"]);
        let mut wrong = original.clone();
        wrong.before_result_anchor_sha256 = "b".repeat(64);
        assert_eq!(reject(&wrong).stage, "context_identity");
        let mut wrong = original.clone();
        wrong.line_sha256 = "b".repeat(64);
        wrong.context_sha256 = request_context_sha256(&wrong).unwrap();
        assert_eq!(reject(&wrong).stage, "line_identity");
        let mut wrong = original.clone();
        wrong.cpu_profile_sha256 = "b".repeat(64);
        wrong.context_sha256 = request_context_sha256(&wrong).unwrap();
        assert_eq!(reject(&wrong).stage, "profile_identity");
        for line in [Vec::new(), vec![original.line[0]; 65]] {
            let mut wrong = original.clone();
            wrong.line = line;
            reseal(&mut wrong);
            assert_eq!(reject(&wrong).stage, "admission");
        }
        let mut raw = serde_json::to_value(&original).unwrap();
        raw["candidate"] = json!(original.line[0]);
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
        let text = serde_json::to_string(&original).unwrap();
        let duplicate = text.replacen("\"horizon\":1", "\"horizon\":1,\"horizon\":1", 1);
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
        assert!(
            super::super::request_admission(
                &serde_json::to_vec(&original).unwrap(),
                Instant::now()
            )
            .is_err()
        );
    }
}
