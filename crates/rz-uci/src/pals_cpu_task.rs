//! Training-private, finite CPU_T task dispatcher. This is not a product V path.
//!
//! One request admits at most two own bootstrap CPU searches. Rules and the UCI
//! parser own all state reconstruction, legality, history and terminal facts.
//! A finite-depth scalar is uncalibrated evidence, never a policy/value target,
//! comparative preference rank, or proof of the game's outcome. Resume tokens
//! stay opaque and inside this invocation; interrupted stacks are not restored.

pub mod candidate;
pub mod continuation;
pub mod semantic;
pub mod strategic_action;

use crate::engine::{OwnerRegistry, RulesUciPort};
use crate::pals_native::pals_history_digest;
use crate::{Command, ParserLimits, PositionPort, parse};
use rz_contracts::pals::Move16;
use rz_position::contracts::ContractPosition;
use rz_position::{BoardMove, Color, PlayStatus, Position, PositionLimits};
use rz_search::cpu::{
    BOOTSTRAP_SCORE_VERSION, CPU_MATE_SCORE, CPU_SEARCH_VERSION, CPU_SEE_SEARCH_VERSION,
    CpuCompletion, CpuConfig, CpuEngine, CpuLimits, CpuOrderingPolicy, CpuProfile, CpuReport,
    CpuScoreScope, CpuSearcher,
};
use rz_search::cpu_value::{CpuTrainingState, CpuValueIdentity};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

pub const CPU_TASK_SCHEMA: &str = "rz-pals-private-cpu-task/1";
/// Explicit opt-in bridge. The frozen legacy task/conditions domains stay intact.
pub const CPU_SEE_TASK_SCHEMA: &str = "rz-pals-private-cpu-task-legal-see/1";
pub const MAX_REQUEST_BYTES: usize = 512 * 1024;
pub const MAX_RESPONSE_BYTES: usize = 1024 * 1024;
pub const MAX_WALL_TIME_MS: u64 = 300_000;
/// Only the executable itself is streamed by the example; no model is loaded.
pub const MAX_SELF_BINARY_BYTES: u64 = 1024 * 1024 * 1024;
const CONDITIONS_SCHEMA: &str = "rz-pals-private-cpu-conditions/1";
const SEE_CONDITIONS_SCHEMA: &str = "rz-pals-private-cpu-conditions-legal-see/1";
const BRANCH_DOMAIN: &str = "rz-pals-private-cpu-branch/1";
const ROOT_ORDER_DOMAIN: &str = "rz-pals-private-cpu-root-order/1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TaskKind {
    DefendResponse,
    AttackRepair,
    WidenResponses,
    LowerSelectivity,
    ResumeTask,
    CrossProfileRecheck,
    Defer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
enum ExplicitOrdering {
    #[serde(rename = "legal_see_v1")]
    LegalSeeV1,
}

fn deserialize_ordering<'de, D>(deserializer: D) -> Result<Option<ExplicitOrdering>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    // Missing means legacy; explicit null/unknown policy is not an omission.
    ExplicitOrdering::deserialize(deserializer).map(Some)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_ordering"
    )]
    ordering_policy: Option<ExplicitOrdering>,
    task: TaskKind,
    parent_input_sha256: String,
    position_command: String,
    expected_board_fen: String,
    rules_state_sha256: String,
    rules_history_sha256: String,
    cpu_binary_sha256: String,
    branch_sha256: String,
    prefix: Vec<u16>,
    root_moves: Vec<u16>,
    baseline_depth: u16,
    requested_depth: u16,
    max_nodes_per_check: u64,
    max_wall_time_ms: u64,
    max_output_bytes: usize,
    tt_entries: usize,
    quiescence_ply: u16,
    cpu_profile_sha256: String,
    recheck_profile_sha256: String,
    context_sha256: String,
}

impl Request {
    fn ordering(&self) -> CpuOrderingPolicy {
        if self.ordering_policy.is_some() {
            CpuOrderingPolicy::LegalSeeV1
        } else {
            CpuOrderingPolicy::LegacyMvvLvaV1
        }
    }
    fn domain(&self) -> &'static str {
        task_domain(self.ordering())
    }
}

fn task_domain(ordering: CpuOrderingPolicy) -> &'static str {
    match ordering {
        CpuOrderingPolicy::LegacyMvvLvaV1 => CPU_TASK_SCHEMA,
        CpuOrderingPolicy::LegalSeeV1 => CPU_SEE_TASK_SCHEMA,
    }
}

fn search_version(ordering: CpuOrderingPolicy) -> &'static str {
    match ordering {
        CpuOrderingPolicy::LegacyMvvLvaV1 => CPU_SEARCH_VERSION,
        CpuOrderingPolicy::LegalSeeV1 => CPU_SEE_SEARCH_VERSION,
    }
}

fn require_ordering(request: &Request, selected: CpuOrderingPolicy) -> Result<(), CpuTaskError> {
    if request.ordering() != selected || request.schema != task_domain(selected) {
        return Err(CpuTaskError::new(
            "ordering_admission",
            "CLI selection, sealed ordering marker and task domain differ",
        ));
    }
    Ok(())
}

/// Explicit failure; a CPU call which failed before a report has unknown work.
/// The CLI exits nonzero for this value. No error is replaced by zero-score or
/// zero-node success. A completed first report remains available to the caller.
#[derive(Clone, Debug, Serialize)]
pub struct CpuTaskError {
    pub code: &'static str,
    pub stage: &'static str,
    pub message: Box<str>,
    pub known_nodes: Option<u64>,
    pub failed_check_work: Option<Box<FailedCheckWork>>,
    pub baseline: Option<Box<RawReport>>,
    pub after: Option<Box<RawReport>>,
    pub elapsed_ms: Option<u64>,
    pub deadline_exceeded: bool,
    #[serde(skip)]
    pub output_limit: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct FailedCheckWork {
    pub nodes: u64,
    pub quiescence_nodes: u64,
    pub tt_hits: u64,
}

impl CpuTaskError {
    fn new(stage: &'static str, message: impl fmt::Display) -> Self {
        Self {
            code: "cpu_task_failed",
            stage,
            message: message
                .to_string()
                .chars()
                .take(512)
                .collect::<String>()
                .into_boxed_str(),
            known_nodes: None,
            failed_check_work: None,
            baseline: None,
            after: None,
            elapsed_ms: None,
            deadline_exceeded: false,
            output_limit: 1024,
        }
    }
    fn with_baseline(mut self, baseline: &RawReport) -> Self {
        self.known_nodes = Some(baseline.nodes);
        self.baseline = Some(Box::new(baseline.clone()));
        self
    }
}
impl fmt::Display for CpuTaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.stage, self.message)
    }
}
impl std::error::Error for CpuTaskError {}

#[derive(Clone, Debug, Serialize)]
pub struct Conditions {
    schema: &'static str,
    rules_state_sha256: String,
    rules_history_sha256: String,
    board_fen: String,
    profile_sha256: String,
    value_identity: CpuValueIdentity,
    search_version: &'static str,
    search_conditions: String,
    quiescence_ply: u16,
    resource_policy: ResourcePolicy,
    /// Full current Rules order, including geometric moves in terminal states.
    legal_moves: Vec<u16>,
    root_moves: Option<Vec<u16>>,
    root_order: Vec<u16>,
    root_order_sha256: String,
    root_selection: &'static str,
    white_to_move: bool,
}

#[derive(Clone, Debug, Serialize)]
struct ResourcePolicy {
    max_wall_time_ms: u64,
    max_nodes_per_check: u64,
    max_checks: u8,
    /// This is part of the original wall allowance, never an extra deadline.
    search_deadline_reserve_ms: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct RawReport {
    pub profile_sha256: String,
    pub conditions_sha256: String,
    pub score_scope: &'static str,
    pub completed_depth: u16,
    pub requested_depth: u16,
    pub nodes: u64,
    pub quiescence_nodes: u64,
    pub completion: &'static str,
    pub root_restricted: bool,
    pub raw_score: i32,
    pub pv: Vec<u16>,
    pub white_to_move: bool,
    pub elapsed_ms: u64,
    pub reused_completed_depth: u16,
    pub score_provenance: &'static str,
    pub conditions: Conditions,
    pub pv_rules_validated: bool,
}

#[derive(Clone, Debug, Serialize)]
struct Response {
    schema: &'static str,
    task: TaskKind,
    context_sha256: String,
    cpu_binary_sha256: String,
    rules_state_sha256: String,
    rules_history_sha256: String,
    board_fen: String,
    branch_sha256: String,
    status: &'static str,
    reason: &'static str,
    baseline: Option<RawReport>,
    after: Option<RawReport>,
    /// Sum of actual reports. Resume's after.nodes is only newly executed work.
    nodes: u64,
    elapsed_ms: u64,
    deadline_exceeded: bool,
    resume_kind: Option<&'static str>,
    product_verifier_enabled: bool,
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn canonical(value: &Value) -> Result<Vec<u8>, CpuTaskError> {
    // Explicit recursion makes sorting independent of serde_json preserve_order.
    fn sorted(value: &Value) -> Value {
        match value {
            Value::Object(fields) => {
                let fields: BTreeMap<_, _> =
                    fields.iter().map(|(k, v)| (k.clone(), sorted(v))).collect();
                Value::Object(fields.into_iter().collect())
            }
            Value::Array(values) => Value::Array(values.iter().map(sorted).collect()),
            _ => value.clone(),
        }
    }
    serde_json::to_vec(&sorted(value)).map_err(|e| CpuTaskError::new("canonical_json", e))
}

fn json_digest(value: &Value) -> Result<String, CpuTaskError> {
    Ok(digest(&canonical(value)?))
}

fn valid_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn decode_request(bytes: &[u8]) -> Result<Request, CpuTaskError> {
    if bytes.is_empty() || bytes.len() > MAX_REQUEST_BYTES {
        return Err(CpuTaskError::new(
            "admission",
            "stdin must contain 1..=512KiB",
        ));
    }
    // Struct deserialization rejects duplicate fields as well as unknown fields;
    // numbers are unsigned integers, never bool/fractional/nonfinite substitutes.
    let request: Request =
        serde_json::from_slice(bytes).map_err(|e| CpuTaskError::new("json_admission", e))?;
    if request.schema != request.domain()
        || !(1..=63).contains(&request.baseline_depth)
        || request.requested_depth <= request.baseline_depth
        || request.requested_depth > 64
        || !(1..=u64::from(u32::MAX)).contains(&request.max_nodes_per_check)
        || !(1..=MAX_WALL_TIME_MS).contains(&request.max_wall_time_ms)
        || !(1024..=MAX_RESPONSE_BYTES).contains(&request.max_output_bytes)
        || request.tt_entries > 1_048_576
        || request.quiescence_ply > 32
        || request.prefix.len() > 64
        || request.root_moves.len() > 256
    {
        return Err(CpuTaskError::new(
            "admission",
            "unsupported schema or finite resource bounds",
        ));
    }
    for hash in [
        &request.parent_input_sha256,
        &request.rules_state_sha256,
        &request.rules_history_sha256,
        &request.cpu_binary_sha256,
        &request.branch_sha256,
        &request.cpu_profile_sha256,
        &request.recheck_profile_sha256,
        &request.context_sha256,
    ] {
        if !valid_sha(hash) {
            return Err(CpuTaskError::new(
                "admission",
                "identity must be lowercase SHA256",
            ));
        }
    }
    if request.expected_board_fen.len() > PositionLimits::default().max_fen_bytes {
        return Err(CpuTaskError::new(
            "admission",
            "expected FEN exceeds Rules bound",
        ));
    }
    // Each kind consumes exactly its declared controls. Extraneous controls do
    // not get silently ignored, even for no-action and unsupported tasks.
    let payload_ok = match request.task {
        TaskKind::DefendResponse => !request.prefix.is_empty() && !request.root_moves.is_empty(),
        TaskKind::AttackRepair => !request.prefix.is_empty() && request.root_moves.is_empty(),
        TaskKind::WidenResponses => request.prefix.is_empty(),
        TaskKind::LowerSelectivity
        | TaskKind::ResumeTask
        | TaskKind::CrossProfileRecheck
        | TaskKind::Defer => request.prefix.is_empty() && request.root_moves.is_empty(),
    };
    if !payload_ok {
        return Err(CpuTaskError::new(
            "admission",
            "task payload contains missing or unused controls",
        ));
    }
    decode_moves(&request.prefix)?;
    let roots = decode_moves(&request.root_moves)?;
    if roots
        .iter()
        .enumerate()
        .any(|(at, mv)| roots[..at].contains(mv))
    {
        return Err(CpuTaskError::new("admission", "duplicate root move"));
    }
    Ok(request)
}

/// Derive the absolute request deadline from the CLI's original start instant.
/// Reading stdin and hashing the executable cannot receive an extra budget.
pub fn request_deadline(bytes: &[u8], started: Instant) -> Result<Instant, CpuTaskError> {
    Ok(request_admission(bytes, started)?.deadline)
}

#[derive(Clone, Copy, Debug)]
pub struct CpuTaskAdmission {
    pub deadline: Instant,
    pub output_limit: usize,
}

pub fn request_admission(bytes: &[u8], started: Instant) -> Result<CpuTaskAdmission, CpuTaskError> {
    request_admission_with_ordering(bytes, started, CpuOrderingPolicy::LegacyMvvLvaV1)
}

/// Selected private CLI admission. The argument is a declaration; independent
/// caller/source/loaded-image observations remain separate execution evidence.
pub fn request_admission_with_ordering(
    bytes: &[u8],
    started: Instant,
    selected: CpuOrderingPolicy,
) -> Result<CpuTaskAdmission, CpuTaskError> {
    let request = decode_request(bytes)?;
    require_ordering(&request, selected)?;
    let deadline = started
        .checked_add(Duration::from_millis(request.max_wall_time_ms))
        .ok_or_else(|| CpuTaskError::new("admission", "absolute deadline overflow"))?;
    Ok(CpuTaskAdmission {
        deadline,
        output_limit: request.max_output_bytes,
    })
}

fn profile(request: &Request, profile: CpuProfile) -> Value {
    let mut value = json!({"domain":request.domain(),"search":search_version(request.ordering()),
        "evaluator":BOOTSTRAP_SCORE_VERSION,"profile":profile.identity(),
        "tt_entries":request.tt_entries,"max_depth":request.requested_depth,
        "quiescence_ply":request.quiescence_ply,"selective_reductions":false});
    if request.ordering() == CpuOrderingPolicy::LegalSeeV1 {
        value["ordering_policy"] = json!(CpuOrderingPolicy::LegalSeeV1.identity());
    }
    value
}

fn validate_pins(request: &Request, binary: &str) -> Result<(), CpuTaskError> {
    if !valid_sha(binary) || request.cpu_binary_sha256 != binary {
        return Err(CpuTaskError::new(
            "binary_identity",
            "own executable digest differs from admitted binary",
        ));
    }
    let branch = json!([BRANCH_DOMAIN, {"parent_input_sha256":request.parent_input_sha256,
        "prefix":request.prefix,"root_moves":request.root_moves}]);
    if json_digest(&branch)? != request.branch_sha256 {
        return Err(CpuTaskError::new(
            "branch_identity",
            "branch controls digest mismatch",
        ));
    }
    let mut raw =
        serde_json::to_value(request).map_err(|e| CpuTaskError::new("context_identity", e))?;
    raw.as_object_mut()
        .expect("Request is object")
        .remove("context_sha256");
    if json_digest(&json!([request.domain(), raw]))? != request.context_sha256 {
        return Err(CpuTaskError::new(
            "context_identity",
            "complete request context digest mismatch",
        ));
    }
    if json_digest(&profile(request, CpuProfile::PlanAssisted))? != request.cpu_profile_sha256
        || json_digest(&profile(request, CpuProfile::Independent))?
            != request.recheck_profile_sha256
    {
        return Err(CpuTaskError::new(
            "profile_identity",
            "actual own CPU profile digest mismatch",
        ));
    }
    Ok(())
}

fn decode_moves(bits: &[u16]) -> Result<Vec<BoardMove>, CpuTaskError> {
    bits.iter()
        .map(|&bits| {
            let movement = Move16::try_from_bits(bits)
                .and_then(Move16::decode)
                .map_err(|e| CpuTaskError::new("move16_admission", e))?;
            BoardMove::try_from(movement).map_err(|e| CpuTaskError::new("move16_admission", e))
        })
        .collect()
}

fn pack_moves(moves: &[BoardMove]) -> Result<Vec<u16>, CpuTaskError> {
    moves
        .iter()
        .map(|&mv| {
            rz_contracts::Move::try_from(mv)
                .map(|mv| Move16::encode(mv).bits())
                .map_err(|e| CpuTaskError::new("move16_export", e))
        })
        .collect()
}

fn state_sha(position: &Position, owners: &OwnerRegistry) -> Result<String, CpuTaskError> {
    let owner = owners
        .allocate()
        .map_err(|e| CpuTaskError::new("rules_identity", e))?;
    let state = ContractPosition::new(owner, position.clone())
        .export()
        .map_err(|e| CpuTaskError::new("rules_identity", e))?;
    Ok(state
        .snapshot()
        .identity()
        .semantic
        .0
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

fn history_sha(position: &Position) -> Result<String, CpuTaskError> {
    Ok(pals_history_digest(position)
        .map_err(|e| CpuTaskError::new("rules_history", e))?
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

fn replay_checked(
    position: &Position,
    moves: &[BoardMove],
    owners: &OwnerRegistry,
    deadline: Instant,
) -> Result<Position, CpuTaskError> {
    let owner = owners
        .allocate()
        .map_err(|e| CpuTaskError::new("rules_replay", e))?;
    let mut position = ContractPosition::new(owner, position.clone());
    for &mv in moves {
        if Instant::now() >= deadline {
            return Err(CpuTaskError::new(
                "rules_replay",
                "absolute deadline expired during checked replay",
            ));
        }
        let view = position
            .export()
            .map_err(|e| CpuTaskError::new("rules_replay", e))?;
        let mv =
            rz_contracts::Move::try_from(mv).map_err(|e| CpuTaskError::new("rules_replay", e))?;
        position
            .make_from_view(&view, mv)
            .map_err(|e| CpuTaskError::new("rules_replay", e))?;
    }
    Ok(position.into_position())
}

fn configured(request: &Request, profile: CpuProfile) -> Result<CpuEngine, CpuTaskError> {
    CpuEngine::with_ordering(
        CpuConfig {
            profile,
            tt_entries: request.tt_entries,
            max_depth: request.requested_depth,
            quiescence_ply: request.quiescence_ply,
        },
        request.ordering(),
    )
    .map_err(|e| CpuTaskError::new("cpu_prepare", e))
}

fn reserve_ms(request: &Request) -> u64 {
    (request.max_wall_time_ms / 10).min(1000)
}

fn cpu_failure(
    stage: &'static str,
    error: impl fmt::Display,
    cpu: &CpuEngine,
    before: Option<&RawReport>,
) -> CpuTaskError {
    let mut error = CpuTaskError::new(stage, error);
    if let Some(before) = before {
        error = error.with_baseline(before);
    }
    if let Some(work) = cpu.last_attempt_work() {
        error.failed_check_work = Some(Box::new(FailedCheckWork {
            nodes: work.nodes,
            quiescence_nodes: work.quiescence_nodes,
            tt_hits: work.tt_hits,
        }));
        error.known_nodes = before.map_or(Some(work.nodes), |before| {
            before.nodes.checked_add(work.nodes)
        });
    }
    error
}

fn conditions(
    request: &Request,
    position: &Position,
    roots: Option<&[BoardMove]>,
    cpu: &CpuEngine,
    owners: &OwnerRegistry,
) -> Result<Conditions, CpuTaskError> {
    let legal = position.ordered_legal_moves();
    if legal.moves().len() > 256 {
        return Err(CpuTaskError::new(
            "rules_admission",
            "current legal array exceeds CPU_T capability",
        ));
    }
    let mut root_order = legal.moves().to_vec();
    if let Some(roots) = roots {
        if roots.is_empty()
            || roots
                .iter()
                .enumerate()
                .any(|(at, mv)| roots[..at].contains(mv) || !legal.moves().contains(mv))
        {
            return Err(CpuTaskError::new(
                "root_admission",
                "restriction is empty, duplicate or illegal",
            ));
        }
        root_order.retain(|mv| roots.contains(mv));
    }
    let root_order = pack_moves(&root_order)?;
    Ok(Conditions {
        schema: if request.ordering() == CpuOrderingPolicy::LegalSeeV1 {
            SEE_CONDITIONS_SCHEMA
        } else {
            CONDITIONS_SCHEMA
        },
        rules_state_sha256: state_sha(position, owners)?,
        rules_history_sha256: history_sha(position)?,
        board_fen: position.to_fen(),
        profile_sha256: json_digest(&profile_value(request, cpu.config().profile))?,
        value_identity: cpu.value_identity().clone(),
        search_version: cpu.search_identity(),
        search_conditions: cpu.search_conditions(),
        quiescence_ply: request.quiescence_ply,
        resource_policy: ResourcePolicy {
            max_wall_time_ms: request.max_wall_time_ms,
            max_nodes_per_check: request.max_nodes_per_check,
            max_checks: 2,
            search_deadline_reserve_ms: reserve_ms(request),
        },
        legal_moves: pack_moves(legal.moves())?,
        root_moves: roots.map(|_| root_order.clone()),
        root_order_sha256: json_digest(&json!([ROOT_ORDER_DOMAIN, root_order]))?,
        root_selection: if roots.is_none() {
            "unrestricted"
        } else if request.task == TaskKind::WidenResponses && request.root_moves.is_empty() {
            "first_actual_rules_legal_move"
        } else {
            "explicit_registered_controls"
        },
        root_order,
        white_to_move: position.side_to_move() == Color::White,
    })
}

// Avoid the local profile argument shadowing the declaration helper.
fn profile_value(request: &Request, selected: CpuProfile) -> Value {
    profile(request, selected)
}

fn raw_report(
    report: &CpuReport,
    request: &Request,
    position: &Position,
    conditions: Conditions,
    requested_depth: u16,
    owners: &OwnerRegistry,
    deadline: Instant,
) -> Result<RawReport, CpuTaskError> {
    let expected_profile = if conditions.profile_sha256 == request.cpu_profile_sha256 {
        CpuProfile::PlanAssisted
    } else if conditions.profile_sha256 == request.recheck_profile_sha256 {
        CpuProfile::Independent
    } else {
        return Err(CpuTaskError::new("cpu_report", "unknown actual profile"));
    };
    if report.search_version != search_version(request.ordering())
        || conditions.search_version != report.search_version
        || report.profile != expected_profile
        || report.value_identity != conditions.value_identity
        || report.value_identity.semantics != BOOTSTRAP_SCORE_VERSION
        || report.value_identity.weights_sha256.is_some()
        || report.value_identity.training != CpuTrainingState::Bootstrap
        || report.completed_depth > requested_depth
        || report.nodes > request.max_nodes_per_check
        || report.quiescence_nodes > report.nodes
        || report.reused_completed_depth > report.completed_depth
        || report.root_restricted != conditions.root_moves.is_some()
        || report.pv.len() > usize::from(requested_depth + request.quiescence_ply)
        || (report.score_scope == CpuScoreScope::CompletedIteration) != (report.completed_depth > 0)
        || (report.score_scope == CpuScoreScope::RulesTerminal)
            != matches!(report.completion, CpuCompletion::Terminal(_))
        || (report.completion == CpuCompletion::DepthLimit
            && report.completed_depth != requested_depth)
    {
        return Err(CpuTaskError::new(
            "cpu_report",
            "own CPU report violated actual namespace/scope/resource bounds",
        ));
    }
    let view = position.ordered_legal_moves();
    let status = position
        .play_status_from_view(&view)
        .map_err(|e| CpuTaskError::new("cpu_report", e))?;
    match status {
        PlayStatus::Terminal { reason, winner } => {
            let score = winner.map_or(0, |winner| {
                if winner == position.side_to_move() {
                    CPU_MATE_SCORE
                } else {
                    -CPU_MATE_SCORE
                }
            });
            if report.completion != CpuCompletion::Terminal(reason)
                || report.score_scope != CpuScoreScope::RulesTerminal
                || report.score != score
                || report.best_move.is_some()
                || !report.pv.is_empty()
            {
                return Err(CpuTaskError::new(
                    "cpu_report",
                    "report disagrees with actual Rules terminal",
                ));
            }
        }
        PlayStatus::Ongoing => {
            if report.score_scope == CpuScoreScope::RulesTerminal
                || report.best_move.is_none()
                || report.pv.first() != report.best_move.as_ref()
            {
                return Err(CpuTaskError::new(
                    "cpu_report",
                    "ongoing Rules position needs an actual legal fallback/PV",
                ));
            }
        }
    }
    // Audit stays inside the original absolute wall allowance. CPU search uses
    // its declared earlier deadline to leave a finite replay/receipt reserve.
    replay_checked(position, &report.pv, owners, deadline)?;
    if let Some(best) = report.best_move {
        if !position.legal_moves().contains(&best)
            || report.pv.first().is_some_and(|mv| *mv != best)
            || conditions.root_order.iter().all(|&bits| {
                Move16::try_from_bits(bits)
                    .and_then(Move16::decode)
                    .ok()
                    .and_then(|mv| BoardMove::try_from(mv).ok())
                    != Some(best)
            })
        {
            return Err(CpuTaskError::new(
                "cpu_report",
                "best move is outside actual Rules root order",
            ));
        }
    }
    let score_scope = match report.score_scope {
        CpuScoreScope::FrontierOnly => "frontier_only",
        CpuScoreScope::CompletedIteration => "completed_iteration",
        CpuScoreScope::RulesTerminal => "rules_terminal",
    };
    let completion = match report.completion {
        CpuCompletion::DepthLimit => "depth_limit",
        CpuCompletion::NodeLimit => "node_limit",
        CpuCompletion::Deadline => "deadline",
        CpuCompletion::Canceled => "canceled",
        CpuCompletion::Yielded => {
            return Err(CpuTaskError::new(
                "unsupported_completion",
                "cooperative Yield requires the adaptive Job/Slice receipt",
            ));
        }
        CpuCompletion::QuiescenceLimit => "quiescence_limit",
        CpuCompletion::AwaitReciprocal => {
            return Err(CpuTaskError::new(
                "unsupported_completion",
                "reciprocal CPU wait requires the Rovent question receipt",
            ));
        }
        CpuCompletion::Terminal(_) => "rules_terminal",
    };
    if report.score_provenance
        != if score_scope == "rules_terminal" {
            "rz-position/rules-terminal"
        } else {
            BOOTSTRAP_SCORE_VERSION
        }
    {
        return Err(CpuTaskError::new(
            "cpu_report",
            "unexpected bootstrap score provenance",
        ));
    }
    Ok(RawReport {
        profile_sha256: conditions.profile_sha256.clone(),
        conditions_sha256: json_digest(
            &serde_json::to_value(&conditions).map_err(|e| CpuTaskError::new("cpu_report", e))?,
        )?,
        score_scope,
        completed_depth: report.completed_depth,
        requested_depth,
        nodes: report.nodes,
        quiescence_nodes: report.quiescence_nodes,
        completion,
        root_restricted: report.root_restricted,
        raw_score: report.score,
        pv: pack_moves(&report.pv)?,
        white_to_move: conditions.white_to_move,
        elapsed_ms: milliseconds(report.elapsed),
        reused_completed_depth: report.reused_completed_depth,
        score_provenance: report.score_provenance,
        conditions,
        pv_rules_validated: true,
    })
}

fn milliseconds(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

/// Caller verifies its own binary digest before admission. This convenience
/// starts the budget here; CLIs must use dispatch_started with the earlier start.
pub fn dispatch(bytes: &[u8], verified_own_binary_sha256: &str) -> Result<Vec<u8>, CpuTaskError> {
    dispatch_started(bytes, verified_own_binary_sha256, Instant::now())
}

pub fn dispatch_started(
    bytes: &[u8],
    verified_own_binary_sha256: &str,
    started: Instant,
) -> Result<Vec<u8>, CpuTaskError> {
    dispatch_started_with_ordering(
        bytes,
        verified_own_binary_sha256,
        started,
        CpuOrderingPolicy::LegacyMvvLvaV1,
    )
}

/// Both the explicit caller choice and sealed domain/marker must agree. The
/// original start instant covers stdin, binary hashing, CPU work and output.
pub fn dispatch_started_with_ordering(
    bytes: &[u8],
    verified_own_binary_sha256: &str,
    started: Instant,
    selected: CpuOrderingPolicy,
) -> Result<Vec<u8>, CpuTaskError> {
    let request = decode_request(bytes)?;
    require_ordering(&request, selected)?;
    let cap = request.max_output_bytes;
    let result = execute(&request, verified_own_binary_sha256, started);
    result.map_err(|mut error| {
        error.output_limit = cap;
        error.elapsed_ms = Some(milliseconds(started.elapsed()));
        error.deadline_exceeded =
            started.elapsed() >= Duration::from_millis(request.max_wall_time_ms);
        error
    })
}

fn execute(
    request: &Request,
    verified_binary: &str,
    started: Instant,
) -> Result<Vec<u8>, CpuTaskError> {
    validate_pins(request, verified_binary)?;
    let deadline = started
        .checked_add(Duration::from_millis(request.max_wall_time_ms))
        .ok_or_else(|| CpuTaskError::new("admission", "absolute deadline overflow"))?;
    let search_deadline = deadline
        .checked_sub(Duration::from_millis(reserve_ms(request)))
        .ok_or_else(|| CpuTaskError::new("admission", "search deadline overflow"))?;
    let command = parse(&request.position_command, ParserLimits::default())
        .map_err(|e| CpuTaskError::new("position_admission", e))?;
    let Command::Position(spec) = command else {
        return Err(CpuTaskError::new(
            "position_admission",
            "only a position command is accepted",
        ));
    };
    let owners = Arc::new(OwnerRegistry::default());
    let prepared = RulesUciPort::new(Arc::clone(&owners), PositionLimits::default())
        .prepare(&spec)
        .map_err(|e| CpuTaskError::new("rules_prepare", e))?;
    let root = prepared.snapshot.rules_position();
    let state = prepared
        .snapshot
        .state()
        .snapshot()
        .identity()
        .semantic
        .0
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    if state != request.rules_state_sha256
        || history_sha(root)? != request.rules_history_sha256
        || root.to_fen() != request.expected_board_fen
    {
        return Err(CpuTaskError::new(
            "rules_identity",
            "actual Rules board/full known history differs from sealed input",
        ));
    }
    let mut response = Response {
        schema: request.domain(),
        task: request.task,
        context_sha256: request.context_sha256.clone(),
        cpu_binary_sha256: verified_binary.to_owned(),
        rules_state_sha256: state,
        rules_history_sha256: request.rules_history_sha256.clone(),
        board_fen: root.to_fen(),
        branch_sha256: request.branch_sha256.clone(),
        status: "unavailable",
        reason: "absolute_deadline_before_baseline",
        baseline: None,
        after: None,
        nodes: 0,
        elapsed_ms: 0,
        deadline_exceeded: false,
        resume_kind: None,
        product_verifier_enabled: false,
    };
    if Instant::now() >= search_deadline {
        return finish(response, request, started, deadline);
    }
    match request.task {
        TaskKind::Defer => {
            response.status = "deferred";
            response.reason = "explicit_defer_no_cpu_check";
            return finish(response, request, started, deadline);
        }
        TaskKind::LowerSelectivity => {
            response.reason = "selective_reductions_already_disabled";
            return finish(response, request, started, deadline);
        }
        _ => {}
    }
    let prefix = decode_moves(&request.prefix)?;
    let target = replay_checked(root, &prefix, &owners, deadline)?;
    let mut restriction = decode_moves(&request.root_moves)?;
    let legal = target.ordered_legal_moves();
    if request.task == TaskKind::WidenResponses {
        if restriction.is_empty() {
            let Some(&first) = legal.moves().first() else {
                response.reason = "no_legal_root_response_to_widen";
                return finish(response, request, started, deadline);
            };
            restriction.push(first);
        }
        if restriction.len() >= legal.moves().len() {
            return Err(CpuTaskError::new(
                "root_admission",
                "widen needs a strict legal response subset",
            ));
        }
    }
    let restricted = matches!(
        request.task,
        TaskKind::DefendResponse | TaskKind::WidenResponses
    );
    let roots = restricted.then_some(restriction.as_slice());
    let mut cpu = configured(request, CpuProfile::PlanAssisted)?;
    let before_conditions = conditions(request, &target, roots, &cpu, &owners)?;
    if Instant::now() >= search_deadline {
        return finish(response, request, started, deadline);
    }
    let cancel = AtomicBool::new(false);
    let baseline_limits = CpuLimits {
        max_depth: request.baseline_depth,
        max_nodes: request.max_nodes_per_check,
        deadline: Some(search_deadline),
    };
    let baseline = if let Some(roots) = roots {
        cpu.analyze_root_moves(&target, roots, baseline_limits, &cancel)
    } else {
        cpu.analyze(&target, baseline_limits, &cancel)
    }
    .map_err(|e| cpu_failure("baseline_cpu", e, &cpu, None))?;
    let before = raw_report(
        &baseline,
        request,
        &target,
        before_conditions,
        request.baseline_depth,
        &owners,
        deadline,
    )
    .map_err(|mut e| {
        e.known_nodes = Some(baseline.nodes);
        e
    })?;
    response.nodes = before.nodes;
    response.baseline = Some(before.clone());
    response.reason = "absolute_deadline_after_baseline";
    if Instant::now() >= search_deadline {
        return finish(response, request, started, deadline);
    }
    let after_limits = CpuLimits {
        max_depth: request.requested_depth,
        max_nodes: request.max_nodes_per_check,
        deadline: Some(search_deadline),
    };
    let (after, after_conditions) = if request.task == TaskKind::ResumeTask {
        let Some(token) = baseline
            .resume
            .as_ref()
            .filter(|_| baseline.completed_depth > 0)
        else {
            response.reason = "baseline_has_no_owned_completed_iteration_token";
            return finish(response, request, started, deadline);
        };
        let conditions = conditions(request, &target, None, &cpu, &owners)
            .map_err(|e| e.with_baseline(&before))?;
        if Instant::now() >= search_deadline {
            return finish(response, request, started, deadline);
        }
        response.resume_kind = Some("completed_iteration");
        let report = cpu
            .resume(&target, token, after_limits, &cancel)
            .map_err(|e| cpu_failure("after_cpu", e, &cpu, Some(&before)))?;
        (report, conditions)
    } else {
        // Cross-profile cannot share TT/history; other checks also use a fresh
        // engine to avoid silently carrying baseline state into an observation.
        drop(cpu);
        let next_profile = if request.task == TaskKind::CrossProfileRecheck {
            CpuProfile::Independent
        } else {
            CpuProfile::PlanAssisted
        };
        let mut after_cpu =
            configured(request, next_profile).map_err(|e| e.with_baseline(&before))?;
        let after_roots = if request.task == TaskKind::DefendResponse {
            roots
        } else {
            None
        };
        let conditions = conditions(request, &target, after_roots, &after_cpu, &owners)
            .map_err(|e| e.with_baseline(&before))?;
        if Instant::now() >= search_deadline {
            return finish(response, request, started, deadline);
        }
        let report = if let Some(roots) = after_roots {
            after_cpu.analyze_root_moves(&target, roots, after_limits, &cancel)
        } else {
            after_cpu.analyze(&target, after_limits, &cancel)
        }
        .map_err(|e| cpu_failure("after_cpu", e, &after_cpu, Some(&before)))?;
        (report, conditions)
    };
    let after = raw_report(
        &after,
        request,
        &target,
        after_conditions,
        request.requested_depth,
        &owners,
        deadline,
    )
    .map_err(|mut e| {
        e.known_nodes = before.nodes.checked_add(after.nodes);
        e.failed_check_work = Some(Box::new(FailedCheckWork {
            nodes: after.nodes,
            quiescence_nodes: after.quiescence_nodes,
            tt_hits: after.tt_hits,
        }));
        e.baseline = Some(Box::new(before.clone()));
        e
    })?;
    response.nodes = before.nodes.checked_add(after.nodes).ok_or_else(|| {
        CpuTaskError::new("accounting", "actual node sum overflow").with_baseline(&before)
    })?;
    response.after = Some(after);
    response.status = "observed";
    response.reason = "two_actual_conditional_cpu_checks";
    finish(response, request, started, deadline)
}

fn finish(
    mut response: Response,
    request: &Request,
    started: Instant,
    deadline: Instant,
) -> Result<Vec<u8>, CpuTaskError> {
    response.elapsed_ms = milliseconds(started.elapsed());
    response.deadline_exceeded = Instant::now() >= deadline;
    let mut bytes =
        canonical(&serde_json::to_value(&response).map_err(|e| CpuTaskError::new("receipt", e))?)?;
    if !response.deadline_exceeded && Instant::now() >= deadline {
        response.deadline_exceeded = true;
        response.elapsed_ms = milliseconds(started.elapsed());
        bytes = canonical(
            &serde_json::to_value(&response).map_err(|e| CpuTaskError::new("receipt", e))?,
        )?;
    }
    bytes.push(b'\n');
    if response.deadline_exceeded || Instant::now() >= deadline {
        let mut error = CpuTaskError::new(
            "receipt_deadline",
            "original absolute wall allowance exceeded; actual observations preserved only as failed-run evidence",
        );
        error.known_nodes = Some(response.nodes);
        error.baseline = response.baseline.map(Box::new);
        error.after = response.after.map(Box::new);
        error.elapsed_ms = Some(milliseconds(started.elapsed()));
        error.deadline_exceeded = true;
        return Err(error);
    }
    if bytes.len() > request.max_output_bytes {
        let mut error =
            CpuTaskError::new("receipt", "actual receipt exceeds admitted output bound");
        error.known_nodes = Some(response.nodes);
        error.baseline = response.baseline.map(Box::new);
        error.after = response.after.map(Box::new);
        return Err(error);
    }
    Ok(bytes)
}

/// Capabilities describe actual implementation limits, with conditional tasks
/// stated separately; they are not claims that a request completed a CPU check.
pub fn capabilities() -> Result<Vec<u8>, CpuTaskError> {
    capabilities_with_ordering(CpuOrderingPolicy::LegacyMvvLvaV1)
}

/// Selected CPU_T implementation metadata, not a completed check or support
/// declaration for native PALS/CPU_R launch, candidate or continuation modes.
pub fn capabilities_with_ordering(ordering: CpuOrderingPolicy) -> Result<Vec<u8>, CpuTaskError> {
    let cpu = CpuEngine::with_ordering(
        CpuConfig {
            profile: CpuProfile::PlanAssisted,
            tt_entries: 0,
            max_depth: 64,
            quiescence_ply: 32,
        },
        ordering,
    )
    .map_err(|e| CpuTaskError::new("capabilities", e))?;
    let cap = cpu.capabilities();
    let mut value = json!({"schema":task_domain(ordering),"training_private_only":true,"product_verifier_enabled":false,
        "actual_training_executed":false,"backward_executed":false,"optimizer_created":false,
        "external_teacher_used":false,"gpu_used":false,"cpu_search":cpu.search_identity(),
        "value_identity":cpu.value_identity(),"profile":cpu.config().profile.identity(),
        "max_request_bytes":MAX_REQUEST_BYTES,"max_response_bytes":MAX_RESPONSE_BYTES,
        "max_wall_time_ms":MAX_WALL_TIME_MS,"max_checks":2,"max_nodes_per_check":u32::MAX,
        "max_depth":cap.max_depth,"max_prefix_plies":cap.max_prefix_plies,"max_root_moves":cap.max_root_moves,
        "selective_reductions":cap.selective_reductions,"resume_kind":"completed_iteration_same_invocation_only",
        "tasks":{"defend_response":"explicit_legal_prefix_and_restricted_response",
            "attack_repair":"explicit_legal_prefix","widen_responses":"strict_legal_root_subset_to_unrestricted",
            "lower_selectivity":"unavailable_reductions_already_disabled",
            "resume_task":"conditional_owned_completed_iteration_token",
            "cross_profile_recheck":"fresh_own_independent_profile","defer":"no_cpu_check"}});
    if ordering == CpuOrderingPolicy::LegalSeeV1 {
        value["ordering_policy"] = json!("legal_see_v1");
    }
    let mut bytes = canonical(&value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BINARY: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    fn request(task: TaskKind) -> Request {
        let root = Position::startpos();
        let owners = OwnerRegistry::default();
        let mut r = Request {
            schema: CPU_TASK_SCHEMA.into(),
            ordering_policy: None,
            task,
            parent_input_sha256: "b".repeat(64),
            position_command: "position startpos".into(),
            expected_board_fen: root.to_fen(),
            rules_state_sha256: state_sha(&root, &owners).unwrap(),
            rules_history_sha256: history_sha(&root).unwrap(),
            cpu_binary_sha256: BINARY.into(),
            branch_sha256: String::new(),
            prefix: vec![],
            root_moves: vec![],
            baseline_depth: 1,
            requested_depth: 2,
            max_nodes_per_check: 4096,
            max_wall_time_ms: 5000,
            max_output_bytes: MAX_RESPONSE_BYTES,
            tt_entries: 64,
            quiescence_ply: 8,
            cpu_profile_sha256: String::new(),
            recheck_profile_sha256: String::new(),
            context_sha256: String::new(),
        };
        seal(&mut r);
        r
    }
    fn seal(r: &mut Request) {
        r.cpu_profile_sha256 = json_digest(&profile(r, CpuProfile::PlanAssisted)).unwrap();
        r.recheck_profile_sha256 = json_digest(&profile(r, CpuProfile::Independent)).unwrap();
        r.branch_sha256 = json_digest(
            &json!([BRANCH_DOMAIN,{"parent_input_sha256":r.parent_input_sha256,
            "prefix":r.prefix,"root_moves":r.root_moves}]),
        )
        .unwrap();
        let mut v = serde_json::to_value(&*r).unwrap();
        v.as_object_mut().unwrap().remove("context_sha256");
        r.context_sha256 = json_digest(&json!([r.domain(), v])).unwrap();
    }
    fn run(r: &Request) -> Value {
        serde_json::from_slice(&dispatch(&serde_json::to_vec(r).unwrap(), BINARY).unwrap()).unwrap()
    }

    fn see_request(task: TaskKind) -> Request {
        let mut request = request(task);
        request.schema = CPU_SEE_TASK_SCHEMA.into();
        request.ordering_policy = Some(ExplicitOrdering::LegalSeeV1);
        seal(&mut request);
        request
    }

    #[test]
    fn ordering_marker_omission_preserves_legacy_wire_and_profile() {
        let request = request(TaskKind::Defer);
        let wire = serde_json::to_value(&request).unwrap();
        assert!(wire.get("ordering_policy").is_none());
        assert_eq!(
            profile(&request, CpuProfile::PlanAssisted),
            json!({
                "domain":CPU_TASK_SCHEMA,"search":CPU_SEARCH_VERSION,
                "evaluator":BOOTSTRAP_SCORE_VERSION,"profile":"cpu-plan-assisted-conservative-v1",
                "tt_entries":64,"max_depth":2,"quiescence_ply":8,"selective_reductions":false
            })
        );
        let cap: Value = serde_json::from_slice(&capabilities().unwrap()).unwrap();
        assert_eq!(cap["schema"], CPU_TASK_SCHEMA);
        assert_eq!(cap["cpu_search"], CPU_SEARCH_VERSION);
        assert!(cap.get("ordering_policy").is_none());
        let answer = run(&request);
        assert_eq!(answer["schema"], CPU_TASK_SCHEMA);
        assert_eq!(answer["nodes"], 0);
    }

    #[test]
    fn explicit_see_selection_requires_matching_domain_marker_and_cli() {
        let request = see_request(TaskKind::Defer);
        let wire = serde_json::to_vec(&request).unwrap();
        assert!(request_admission(&wire, Instant::now()).is_err());
        assert!(dispatch(&wire, BINARY).is_err());
        let answer: Value = serde_json::from_slice(
            &dispatch_started_with_ordering(
                &wire,
                BINARY,
                Instant::now(),
                CpuOrderingPolicy::LegalSeeV1,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(answer["schema"], CPU_SEE_TASK_SCHEMA);
        assert_eq!(answer["status"], "deferred");
        assert_eq!(answer["nodes"], 0);
        assert!(answer["baseline"].is_null());
        assert!(answer["after"].is_null());
        let legacy = serde_json::to_vec(&self::request(TaskKind::Defer)).unwrap();
        assert!(
            request_admission_with_ordering(&legacy, Instant::now(), CpuOrderingPolicy::LegalSeeV1)
                .is_err()
        );
    }

    #[test]
    fn ordering_null_unknown_and_cross_domain_markers_fail_before_search() {
        let legacy = serde_json::to_value(request(TaskKind::Defer)).unwrap();
        for marker in [
            Value::Null,
            json!("unknown"),
            json!(false),
            json!("legal_see_v1"),
        ] {
            let mut changed = legacy.clone();
            changed["ordering_policy"] = marker;
            assert!(decode_request(&serde_json::to_vec(&changed).unwrap()).is_err());
        }
        let mut missing = serde_json::to_value(see_request(TaskKind::Defer)).unwrap();
        missing.as_object_mut().unwrap().remove("ordering_policy");
        assert!(decode_request(&serde_json::to_vec(&missing).unwrap()).is_err());
    }

    #[test]
    fn see_profile_conditions_and_reports_use_actual_selected_engine() {
        let request = see_request(TaskKind::ResumeTask);
        let description = profile(&request, CpuProfile::PlanAssisted);
        assert_eq!(description["domain"], CPU_SEE_TASK_SCHEMA);
        assert_eq!(description["search"], CPU_SEE_SEARCH_VERSION);
        assert_eq!(
            description["ordering_policy"],
            CpuOrderingPolicy::LegalSeeV1.identity()
        );
        assert_eq!(description["evaluator"], BOOTSTRAP_SCORE_VERSION);
        let selected = configured(&request, CpuProfile::PlanAssisted).unwrap();
        assert_eq!(selected.ordering_policy(), CpuOrderingPolicy::LegalSeeV1);
        assert_eq!(
            selected.value_identity().training,
            CpuTrainingState::Bootstrap
        );
        let answer: Value = serde_json::from_slice(
            &dispatch_started_with_ordering(
                &serde_json::to_vec(&request).unwrap(),
                BINARY,
                Instant::now(),
                CpuOrderingPolicy::LegalSeeV1,
            )
            .unwrap(),
        )
        .unwrap();
        let baseline = &answer["baseline"];
        assert_eq!(baseline["conditions"]["schema"], SEE_CONDITIONS_SCHEMA);
        assert_eq!(
            baseline["conditions"]["search_version"],
            CPU_SEE_SEARCH_VERSION
        );
        assert_eq!(
            baseline["conditions"]["search_conditions"],
            selected.search_conditions()
        );
        assert_eq!(baseline["score_provenance"], BOOTSTRAP_SCORE_VERSION);
        if !answer["after"].is_null() {
            assert_eq!(
                answer["after"]["conditions_sha256"],
                baseline["conditions_sha256"]
            );
            assert_eq!(answer["resume_kind"], "completed_iteration");
        }
        assert_eq!(answer["product_verifier_enabled"], false);
    }

    #[test]
    fn see_pins_cannot_use_legacy_context_or_profile_and_capabilities_are_separate() {
        let mut request = see_request(TaskKind::Defer);
        let original = request.context_sha256.clone();
        let mut raw = serde_json::to_value(&request).unwrap();
        raw.as_object_mut().unwrap().remove("context_sha256");
        request.context_sha256 = json_digest(&json!([CPU_TASK_SCHEMA, raw])).unwrap();
        assert_eq!(
            validate_pins(&request, BINARY).unwrap_err().stage,
            "context_identity"
        );
        request.context_sha256 = original;
        request.cpu_profile_sha256 = self::request(TaskKind::Defer).cpu_profile_sha256;
        let mut raw = serde_json::to_value(&request).unwrap();
        raw.as_object_mut().unwrap().remove("context_sha256");
        request.context_sha256 = json_digest(&json!([CPU_SEE_TASK_SCHEMA, raw])).unwrap();
        assert_eq!(
            validate_pins(&request, BINARY).unwrap_err().stage,
            "profile_identity"
        );
        let cap: Value = serde_json::from_slice(
            &capabilities_with_ordering(CpuOrderingPolicy::LegalSeeV1).unwrap(),
        )
        .unwrap();
        assert_eq!(cap["schema"], CPU_SEE_TASK_SCHEMA);
        assert_eq!(cap["cpu_search"], CPU_SEE_SEARCH_VERSION);
        assert_eq!(cap["ordering_policy"], "legal_see_v1");
        assert_eq!(cap["value_identity"]["semantics"], BOOTSTRAP_SCORE_VERSION);
    }

    #[test]
    fn selected_see_admission_keeps_original_absolute_deadline() {
        let mut request = see_request(TaskKind::Defer);
        request.max_wall_time_ms = 1;
        seal(&mut request);
        let bytes = serde_json::to_vec(&request).unwrap();
        let started = Instant::now() - Duration::from_secs(1);
        let admission =
            request_admission_with_ordering(&bytes, started, CpuOrderingPolicy::LegalSeeV1)
                .unwrap();
        assert_eq!(admission.deadline, started + Duration::from_millis(1));
        let error =
            dispatch_started_with_ordering(&bytes, BINARY, started, CpuOrderingPolicy::LegalSeeV1)
                .unwrap_err();
        assert_eq!(error.stage, "receipt_deadline");
        assert!(error.deadline_exceeded);
        assert_eq!(error.known_nodes, Some(0));
    }

    #[test]
    fn see_cross_profile_keeps_ordering_but_uses_fresh_independent_namespace() {
        let request = see_request(TaskKind::CrossProfileRecheck);
        let answer: Value = serde_json::from_slice(
            &dispatch_started_with_ordering(
                &serde_json::to_vec(&request).unwrap(),
                BINARY,
                Instant::now(),
                CpuOrderingPolicy::LegalSeeV1,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(answer["status"], "observed");
        assert!(answer["resume_kind"].is_null());
        for key in ["baseline", "after"] {
            assert_eq!(
                answer[key]["conditions"]["search_version"],
                CPU_SEE_SEARCH_VERSION
            );
            assert_eq!(answer[key]["reused_completed_depth"], 0);
            assert_eq!(answer[key]["score_provenance"], BOOTSTRAP_SCORE_VERSION);
        }
        assert_eq!(
            answer["baseline"]["profile_sha256"],
            request.cpu_profile_sha256
        );
        assert_eq!(
            answer["after"]["profile_sha256"],
            request.recheck_profile_sha256
        );
        assert_ne!(
            answer["baseline"]["conditions_sha256"],
            answer["after"]["conditions_sha256"]
        );
    }

    #[test]
    fn defer_validates_all_pins_and_has_no_fabricated_cpu_work() {
        let r = request(TaskKind::Defer);
        let answer = run(&r);
        assert_eq!(answer["status"], "deferred");
        assert_eq!(answer["nodes"], 0);
        assert!(answer["baseline"].is_null());
        assert!(answer["after"].is_null());
        assert_eq!(answer["product_verifier_enabled"], false);
        for key in [
            "context_sha256",
            "branch_sha256",
            "rules_state_sha256",
            "rules_history_sha256",
            "cpu_profile_sha256",
        ] {
            let mut broken = serde_json::to_value(&r).unwrap();
            broken[key] = json!("c".repeat(64));
            assert!(
                dispatch(&serde_json::to_vec(&broken).unwrap(), BINARY).is_err(),
                "{key}"
            );
        }
        assert!(dispatch(&serde_json::to_vec(&r).unwrap(), &"d".repeat(64)).is_err());
    }

    #[test]
    fn finite_admission_rejects_duplicate_unknown_fractional_and_unused_fields() {
        assert!(dispatch(&vec![b' '; MAX_REQUEST_BYTES + 1], BINARY).is_err());
        let r = request(TaskKind::Defer);
        let bytes = serde_json::to_string(&r).unwrap();
        let duplicate = bytes.replacen('{', "{\"schema\":\"rz-pals-private-cpu-task/1\",", 1);
        assert!(dispatch(duplicate.as_bytes(), BINARY).is_err());
        for (field, value) in [
            ("unknown", json!(true)),
            ("max_nodes_per_check", json!(1.5)),
            ("max_nodes_per_check", json!(u64::from(u32::MAX) + 1)),
            ("max_wall_time_ms", json!(300001)),
            ("root_moves", json!([0])),
            ("prefix", json!([0])),
            ("max_output_bytes", json!(1023)),
        ] {
            let mut v = serde_json::to_value(&r).unwrap();
            v[field] = value;
            assert!(
                dispatch(&serde_json::to_vec(&v).unwrap(), BINARY).is_err(),
                "{field}"
            );
        }
    }

    #[test]
    fn actual_widen_preserves_restricted_baseline_and_unrestricted_after() {
        let mut r = request(TaskKind::WidenResponses);
        r.root_moves = pack_moves(&Position::startpos().legal_moves()[..1]).unwrap();
        seal(&mut r);
        let answer = run(&r);
        assert_eq!(answer["status"], "observed");
        assert_eq!(answer["baseline"]["root_restricted"], true);
        assert_eq!(answer["after"]["root_restricted"], false);
        assert_eq!(
            answer["baseline"]["conditions"]["root_order"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            answer["after"]["conditions"]["root_order"]
                .as_array()
                .unwrap()
                .len(),
            20
        );
        assert_ne!(
            answer["baseline"]["conditions_sha256"],
            answer["after"]["conditions_sha256"]
        );
        assert_eq!(
            answer["nodes"].as_u64().unwrap(),
            answer["baseline"]["nodes"].as_u64().unwrap()
                + answer["after"]["nodes"].as_u64().unwrap()
        );
        assert_eq!(answer["after"]["pv_rules_validated"], true);
        r.root_moves = pack_moves(&Position::startpos().legal_moves()).unwrap();
        seal(&mut r);
        assert!(dispatch(&serde_json::to_vec(&r).unwrap(), BINARY).is_err());
        let answer = run(&request(TaskKind::WidenResponses));
        assert_eq!(
            answer["baseline"]["conditions"]["root_selection"],
            "first_actual_rules_legal_move"
        );
        assert_eq!(
            answer["baseline"]["conditions"]["root_order"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn resume_uses_owned_completed_iteration_and_accounts_new_work_separately() {
        let r = request(TaskKind::ResumeTask);
        let answer = run(&r);
        assert_eq!(answer["status"], "observed");
        assert_eq!(answer["resume_kind"], "completed_iteration");
        assert_eq!(answer["baseline"]["completed_depth"], 1);
        assert_eq!(answer["after"]["reused_completed_depth"], 1);
        assert_eq!(answer["after"]["requested_depth"], 2);
        assert_eq!(
            answer["baseline"]["conditions_sha256"],
            answer["after"]["conditions_sha256"]
        );
        let mut no_token = r;
        no_token.max_nodes_per_check = 1;
        seal(&mut no_token);
        let answer = run(&no_token);
        assert_eq!(answer["status"], "unavailable");
        assert!(answer["after"].is_null());
        assert_eq!(answer["nodes"], answer["baseline"]["nodes"]);
        assert_eq!(answer["baseline"]["score_scope"], "frontier_only");
    }

    #[test]
    fn prefix_replay_records_actual_branch_side_and_history() {
        let mut r = request(TaskKind::AttackRepair);
        r.prefix = pack_moves(&[BoardMove::from_uci("e2e4").unwrap()]).unwrap();
        seal(&mut r);
        let answer = run(&r);
        assert_eq!(answer["baseline"]["white_to_move"], false);
        assert_ne!(
            answer["baseline"]["conditions"]["rules_history_sha256"],
            r.rules_history_sha256
        );
        assert_ne!(
            answer["baseline"]["conditions"]["rules_state_sha256"],
            r.rules_state_sha256
        );
        r.prefix = pack_moves(&[BoardMove::from_uci("e2e5").unwrap()]).unwrap();
        seal(&mut r);
        assert!(dispatch(&serde_json::to_vec(&r).unwrap(), BINARY).is_err());
    }

    #[test]
    fn capabilities_and_unsupported_selectivity_are_explicit() {
        let value: Value = serde_json::from_slice(&capabilities().unwrap()).unwrap();
        assert_eq!(value["selective_reductions"], false);
        assert_eq!(value["max_checks"], 2);
        assert_eq!(value["actual_training_executed"], false);
        assert_eq!(value["gpu_used"], false);
        let answer = run(&request(TaskKind::LowerSelectivity));
        assert_eq!(answer["status"], "unavailable");
        assert_eq!(answer["nodes"], 0);
        assert!(answer["baseline"].is_null());
        assert!(answer["after"].is_null());
        let answer = run(&request(TaskKind::CrossProfileRecheck));
        assert_ne!(
            answer["baseline"]["profile_sha256"],
            answer["after"]["profile_sha256"]
        );
        assert_eq!(answer["after"]["reused_completed_depth"], 0);
    }

    #[test]
    fn deadline_includes_input_preparation_and_never_clamps_or_returns_success() {
        let r = request(TaskKind::Defer);
        let started = Instant::now().checked_sub(Duration::from_secs(6)).unwrap();
        let error =
            dispatch_started(&serde_json::to_vec(&r).unwrap(), BINARY, started).unwrap_err();
        assert_eq!(error.stage, "receipt_deadline");
        assert_eq!(error.known_nodes, Some(0));
        assert!(error.deadline_exceeded);
        assert!(error.elapsed_ms.unwrap() >= 6000);
        assert!(error.baseline.is_none());
        assert!(error.after.is_none());
    }

    #[test]
    fn restricted_defense_uses_actual_post_prefix_legal_responses() {
        let mut r = request(TaskKind::DefendResponse);
        r.prefix = pack_moves(&[BoardMove::from_uci("e2e4").unwrap()]).unwrap();
        r.root_moves = pack_moves(&[BoardMove::from_uci("e7e5").unwrap()]).unwrap();
        seal(&mut r);
        let answer = run(&r);
        assert_eq!(answer["status"], "observed");
        assert_eq!(answer["baseline"]["white_to_move"], false);
        assert_eq!(answer["baseline"]["root_restricted"], true);
        assert_eq!(answer["after"]["root_restricted"], true);
        assert_eq!(
            answer["baseline"]["conditions_sha256"],
            answer["after"]["conditions_sha256"]
        );
        r.root_moves = r.prefix.clone();
        seal(&mut r);
        assert!(dispatch(&serde_json::to_vec(&r).unwrap(), BINARY).is_err());
    }
}
