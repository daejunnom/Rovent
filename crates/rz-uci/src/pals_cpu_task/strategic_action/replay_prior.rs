//! Structured, bounded observations for a subsequent private V query.
//!
//! This consumer compares independent historical bindings and separates complete
//! CPU scope from native closure. It does not validate a dataset/current selector,
//! caller process chronology, loaded executable, whole-action cost or utility.
//! No Debug text, CPU score, native numeric ID or identity hash is a V feature.

use super::super::replay_inputs::{ReplayAuthorities, ReplayExpectedPins, ReplayInputMode};
use super::{
    AdmissionFault, CpuFreshAssetProfile, CpuStageObservation, NativeReplayObservation,
    SnapshotText,
};
use rz_contracts::pals::Move16;
use rz_position::{BoardMove, Color, PositionSnapshot};
use rz_search::cpu::{CPU_MATE_THRESHOLD, CpuCompletion, CpuScoreScope};
use rz_search::cpu_value::CpuValueIdentity;
use rz_search::pals::engine::replay::{
    FreshReplayOwner, ReplayCpuPhase, ReplayOpponentOutcome, ReplayStageReport,
};
use rz_search::pals::engine::{RecheckEndpointEvidence, RoleModel};
use rz_search::pals::store::{BoundKind, RawScore};
use serde::{Deserialize, Serialize, Serializer};
use sha2::{Digest, Sha256};

#[path = "replay_captured_cpu_witness.rs"]
pub mod captured_cpu_witness;

#[path = "replay_reported.rs"]
mod reported;
pub use reported::{
    ReportedPriorConsistency, ReportedRepairConsistency, ReportedRepairRulesConsistency,
    ReportedRulesConsistency, ReportedRulesError, check_reported_prior_consistency,
    check_reported_repair_consistency,
};

#[path = "replay_repair_evidence.rs"]
mod repair_evidence;
pub use repair_evidence::{
    REPAIR_EVIDENCE_SCHEMA, REPAIR_EVIDENCE_SCOPE, RepairAnchorEvidence,
    consume_native_repair_replay_prior, repair_evidence_source_bytes,
    repair_evidence_source_digest,
};

pub const SCHEMA: &str = "rz-pals-frozen-replay-query-prior/1";
pub const SCOPE: &str = "registered_native_replay_projection_pending_caller_chronology";
pub const MAX_PV_MOVES: usize = 96; // registered depth <=64 plus qsearch <=32
pub const MAX_LINE_MOVES: usize = 16;
pub const MAX_STAGES: usize = 4;
/// Fixed scalar/move projection fits inside the existing /2 header reservation.
pub const MAX_PROJECTED_JSON_BYTES: usize = 16 * 1024;

/// Compiled literal metadata only. An independent caller must still register
/// this consumer source/build/binary; this is never a fallback expected pin.
pub fn projection_source_digest() -> [u8; 32] {
    Sha256::digest(include_bytes!("replay_prior.rs")).into()
}

#[derive(Debug)]
pub struct PackedMoves {
    moves: [u16; MAX_PV_MOVES],
    len: usize,
}
impl Default for PackedMoves {
    fn default() -> Self {
        Self {
            moves: [0; MAX_PV_MOVES],
            len: 0,
        }
    }
}
impl PackedMoves {
    fn capture(&mut self, moves: &[BoardMove], maximum: usize) -> Result<(), AdmissionFault> {
        self.len = 0;
        if moves.len() > maximum || maximum > MAX_PV_MOVES {
            return Err(fault("exact move extent exceeds registered projection"));
        }
        for movement in moves {
            let movement = rz_contracts::Move::try_from(*movement)
                .map_err(|_| fault("actual move16 conversion failed"))?;
            self.moves[self.len] = Move16::encode(movement).bits();
            self.len += 1;
        }
        Ok(())
    }
    pub fn as_slice(&self) -> &[u16] {
        &self.moves[..self.len]
    }
}
impl Serialize for PackedMoves {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.as_slice().serialize(s)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriorPhase {
    Baseline,
    After,
    RepairEndpoint,
    RepairOpponentEndpoint,
}
impl From<ReplayCpuPhase> for PriorPhase {
    fn from(value: ReplayCpuPhase) -> Self {
        match value {
            ReplayCpuPhase::Baseline => Self::Baseline,
            ReplayCpuPhase::After => Self::After,
            ReplayCpuPhase::RepairEndpoint => Self::RepairEndpoint,
            ReplayCpuPhase::RepairOpponentEndpoint => Self::RepairOpponentEndpoint,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriorCompletion {
    DepthLimit,
    NodeLimit,
    Deadline,
    Canceled,
    QuiescenceLimit,
    RulesTerminal,
}
impl TryFrom<CpuCompletion> for PriorCompletion {
    type Error = AdmissionFault;
    fn try_from(value: CpuCompletion) -> Result<Self, Self::Error> {
        Ok(match value {
            CpuCompletion::DepthLimit => Self::DepthLimit,
            CpuCompletion::NodeLimit => Self::NodeLimit,
            CpuCompletion::Deadline => Self::Deadline,
            CpuCompletion::Canceled => Self::Canceled,
            CpuCompletion::Yielded => {
                return Err(fault(
                    "cooperative Yield is unsupported by legacy replay prior",
                ));
            }
            CpuCompletion::QuiescenceLimit => Self::QuiescenceLimit,
            CpuCompletion::AwaitReciprocal => {
                return Err(fault(
                    "reciprocal CPU wait is unsupported by legacy replay prior",
                ));
            }
            CpuCompletion::Terminal(_) => Self::RulesTerminal,
        })
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriorOutcome {
    Unobserved,
    RepairNotReady,
    NoEligibleOpponentAnchor,
    NoAlternativeResponse,
    PartialOpponentEndpoint,
    CompletedOpponentEndpoint,
    RulesTerminalOpponentEndpoint,
}

/// Live, source-owned endpoint fact. There is no deserializer, public
/// constructor or wire projection. CPU raw units are not calibrated cp, WDL,
/// Rules proof, a strategic refutation, or a training target.
#[derive(Debug)]
pub struct ConditionalEndpointFact {
    snapshot: PositionSnapshot,
    raw_value: i32,
    perspective: Color,
    root_perspective: Color,
    completed_depth: u16,
    registered_condition_sha256: [u8; 32],
    value_identity: CpuValueIdentity,
}
impl ConditionalEndpointFact {
    pub fn snapshot(&self) -> &PositionSnapshot {
        &self.snapshot
    }
    pub fn raw_value(&self) -> i32 {
        self.raw_value
    }
    pub fn perspective(&self) -> Color {
        self.perspective
    }
    pub fn root_perspective(&self) -> Color {
        self.root_perspective
    }
    pub fn completed_depth(&self) -> u16 {
        self.completed_depth
    }
    pub fn registered_condition_sha256(&self) -> &[u8; 32] {
        &self.registered_condition_sha256
    }
    pub fn value_identity(&self) -> &CpuValueIdentity {
        &self.value_identity
    }
    pub fn raw_value_in_root_perspective(&self) -> i32 {
        if self.perspective == self.root_perspective {
            self.raw_value
        } else {
            -self.raw_value
        }
    }
    pub fn same_conditional_fact(&self, other: &Self) -> bool {
        self.snapshot.same_state(&other.snapshot)
            && self.raw_value == other.raw_value
            && self.perspective == other.perspective
            && self.root_perspective == other.root_perspective
            && self.completed_depth == other.completed_depth
            && self.registered_condition_sha256 == other.registered_condition_sha256
            && self.value_identity == other.value_identity
    }
}

fn live_endpoint_fact<M: RoleModel>(
    owner: &FreshReplayOwner<M>,
    stage: &ReplayStageReport,
) -> Option<ConditionalEndpointFact> {
    let report = stage.report()?;
    if !stage.exact_completed()
        || report.score_scope != CpuScoreScope::CompletedIteration
        || report.completion != CpuCompletion::DepthLimit
        || report.completed_depth != stage.requested_depth()
        || report.root_restricted
        || report.reused_completed_depth != 0
        || report.score.unsigned_abs() >= CPU_MATE_THRESHOLD as u32
        || report.value_identity.validate().is_err()
    {
        return None;
    }
    let id = stage.observation()?;
    let observed = owner.observation(id).ok()?;
    let (snapshot, endpoint_state) = match stage.phase() {
        ReplayCpuPhase::RepairEndpoint => {
            let endpoint = owner.repair_endpoint()?;
            match endpoint.evidence {
                RecheckEndpointEvidence::OwnCpu {
                    observation_id,
                    execution_id,
                    admitted_scope,
                    ..
                } if observation_id == id
                    && execution_id == stage.execution()
                    && admitted_scope == CpuScoreScope::CompletedIteration => {}
                _ => return None,
            }
            (endpoint.snapshot.clone(), endpoint.state)
        }
        ReplayCpuPhase::RepairOpponentEndpoint => {
            let endpoint = owner.opponent_endpoint()?;
            if endpoint.terminal.is_some() || endpoint.stage?.execution() != stage.execution() {
                return None;
            }
            (endpoint.snapshot.clone(), endpoint.state)
        }
        _ => return None,
    };
    let RawScore::Cpu {
        value,
        perspective,
        bound: BoundKind::ExactWithinSearch,
    } = observed.score
    else {
        return None;
    };
    if observed.state != endpoint_state
        || value != report.score
        || perspective != snapshot.side_to_move()
        || observed.value_identity.as_ref() != Some(&report.value_identity)
        || observed.cpu_condition.as_deref() != Some(stage.task_condition())
    {
        return None;
    }
    Some(ConditionalEndpointFact {
        snapshot,
        raw_value: value,
        perspective,
        root_perspective: owner.root_snapshot().side_to_move(),
        completed_depth: report.completed_depth,
        registered_condition_sha256: Sha256::digest(stage.registered_condition().as_bytes()).into(),
        value_identity: report.value_identity.clone(),
    })
}

#[derive(Debug, Default, Serialize)]
pub struct PriorStage {
    phase: Option<PriorPhase>,
    requested_depth: u16,
    node_budget: u64,
    exact_completed: bool,
    report_present: bool,
    completed_depth: Option<u16>,
    completion: Option<PriorCompletion>,
    completed_iteration: bool,
    root_restricted: Option<bool>,
    reused_completed_depth: Option<u16>,
    report_nodes: Option<u64>,
    report_qnodes: Option<u64>,
    report_tt_hits: Option<u64>,
    report_elapsed_ms: Option<u64>,
    attempt_present: bool,
    attempt_nodes: Option<u64>,
    observation_present: bool,
    /// Metadata only; separate from the numeric/move feature view.
    registered_condition_sha256: [u8; 32],
    task_condition_sha256: [u8; 32],
    pv: PackedMoves,
    #[serde(skip)]
    conditional_endpoint: Option<ConditionalEndpointFact>,
}
impl PriorStage {
    fn complete(&self) -> bool {
        self.exact_completed
            && self.report_present
            && self.observation_present
            && self.attempt_present
            && self.completion == Some(PriorCompletion::DepthLimit)
            && self.completed_iteration
            && self.completed_depth == Some(self.requested_depth)
            && self.reused_completed_depth == Some(0)
            && self.report_nodes.is_some_and(|n| n <= self.node_budget)
    }
    pub fn pv(&self) -> &[u16] {
        self.pv.as_slice()
    }
    pub fn phase(&self) -> Option<PriorPhase> {
        self.phase
    }
    pub fn completed_depth(&self) -> Option<u16> {
        self.completed_depth
    }
    pub fn conditional_endpoint(&self) -> Option<&ConditionalEndpointFact> {
        self.conditional_endpoint.as_ref()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriorReadiness {
    Unobserved,
    IncompleteCpuScope,
    NativeClosureUnobserved,
    OriginalWindowFailed,
    CapturedResultFailed,
    CompleteCpuScopePendingCallerChronology,
    RulesTerminalPendingCallerChronology,
}

#[derive(Debug, Serialize)]
pub struct ReplayQueryPrior {
    schema: &'static str,
    assurance_scope: &'static str,
    projection_source_sha256: [u8; 32],
    outcome: PriorOutcome,
    readiness: PriorReadiness,
    work_returned_elapsed_ms: Option<u64>,
    work_returned_within_execution_window: Option<bool>,
    #[serde(skip)]
    native_work_started_at: Option<std::time::Instant>,
    #[serde(skip)]
    native_work_returned_at: Option<std::time::Instant>,
    /// Retained scalar facts do not fabricate a missing report or cost as zero.
    stage_count: usize,
    #[serde(serialize_with = "serialize_stages")]
    stages: [PriorStage; MAX_STAGES],
    opponent_anchor_ply: Option<usize>,
    repaired_line: PackedMoves,
    opponent_counterline: PackedMoves,
    opponent_role_steps: usize,
    accepted_opponent_steps: usize,
    endpoint_observed: bool,
    endpoint_rules_terminal: bool,
    actual_utility_groups: u8,
    authorities: ReplayAuthorities,
}
fn serialize_stages<S: Serializer>(
    stages: &[PriorStage; MAX_STAGES],
    s: S,
) -> Result<S::Ok, S::Error> {
    let count = stages
        .iter()
        .take_while(|stage| stage.phase.is_some())
        .count();
    stages[..count].serialize(s)
}
impl Default for ReplayQueryPrior {
    fn default() -> Self {
        Self {
            schema: SCHEMA,
            assurance_scope: SCOPE,
            projection_source_sha256: projection_source_digest(),
            outcome: PriorOutcome::Unobserved,
            readiness: PriorReadiness::Unobserved,
            stage_count: 0,
            work_returned_elapsed_ms: None,
            work_returned_within_execution_window: None,
            native_work_started_at: None,
            native_work_returned_at: None,
            stages: std::array::from_fn(|_| PriorStage::default()),
            opponent_anchor_ply: None,
            repaired_line: PackedMoves::default(),
            opponent_counterline: PackedMoves::default(),
            opponent_role_steps: 0,
            accepted_opponent_steps: 0,
            endpoint_observed: false,
            endpoint_rules_terminal: false,
            actual_utility_groups: 0,
            authorities: ReplayAuthorities::default(),
        }
    }
}
impl ReplayQueryPrior {
    pub(super) fn observe_work_return(
        &mut self,
        started: std::time::Instant,
        execution: std::time::Instant,
    ) {
        let now = std::time::Instant::now();
        self.work_returned_elapsed_ms =
            u64::try_from(now.saturating_duration_since(started).as_millis()).ok();
        self.work_returned_within_execution_window = Some(now < execution);
        self.native_work_returned_at = Some(now);
    }
    pub(super) fn observe_native_entry(&mut self, actual_entry: std::time::Instant) {
        if self.native_work_started_at.is_none() {
            self.native_work_started_at = Some(actual_entry);
        }
    }
    pub(super) fn capture<M: RoleModel>(
        &mut self,
        owner: &FreshReplayOwner<M>,
    ) -> Result<(), AdmissionFault> {
        if owner.stages().len() > MAX_STAGES {
            return Err(fault("CPU stage extent"));
        }
        self.stage_count = owner.stages().len();
        for (out, stage) in self.stages.iter_mut().zip(owner.stages()) {
            out.phase = Some(stage.phase().into());
            out.requested_depth = stage.requested_depth();
            out.node_budget = stage.node_budget();
            out.exact_completed = stage.exact_completed();
            out.observation_present = stage.observation().is_some();
            out.registered_condition_sha256 =
                Sha256::digest(stage.registered_condition().as_bytes()).into();
            out.task_condition_sha256 = Sha256::digest(stage.task_condition().as_bytes()).into();
            out.attempt_present = stage.attempt().is_some();
            out.attempt_nodes = stage.attempt().and_then(|attempt| attempt.work.nodes);
            if let Some(report) = stage.report() {
                out.report_present = true;
                out.completed_depth = Some(report.completed_depth);
                out.completion = Some(PriorCompletion::try_from(report.completion)?);
                out.completed_iteration = report.score_scope == CpuScoreScope::CompletedIteration;
                out.root_restricted = Some(report.root_restricted);
                out.reused_completed_depth = Some(report.reused_completed_depth);
                out.report_nodes = Some(report.nodes);
                out.report_qnodes = Some(report.quiescence_nodes);
                out.report_tt_hits = Some(report.tt_hits);
                out.report_elapsed_ms = Some(
                    u64::try_from(report.elapsed.as_millis())
                        .map_err(|_| fault("elapsed extent"))?,
                );
                out.pv.capture(&report.pv, MAX_PV_MOVES)?;
            }
            out.conditional_endpoint = live_endpoint_fact(owner, stage);
        }
        self.opponent_anchor_ply = owner.opponent_anchor_ply();
        self.repaired_line
            .capture(owner.repaired_line(), MAX_LINE_MOVES)?;
        self.opponent_counterline
            .capture(owner.opponent_counterline(), MAX_LINE_MOVES)?;
        self.opponent_role_steps = owner.opponent_role_steps().len();
        self.accepted_opponent_steps = owner
            .opponent_role_steps()
            .iter()
            .filter(|step| step.accepted() && step.selected().is_some())
            .count();
        self.endpoint_observed = owner.opponent_endpoint().is_some();
        self.endpoint_rules_terminal = owner
            .opponent_endpoint()
            .is_some_and(|endpoint| endpoint.terminal.is_some());
        Ok(())
    }
    pub(super) fn capture_outcome(&mut self, outcome: &ReplayOpponentOutcome) {
        self.outcome = match outcome {
            ReplayOpponentOutcome::RepairNotReady(_) => PriorOutcome::RepairNotReady,
            ReplayOpponentOutcome::NoEligibleOpponentAnchor => {
                PriorOutcome::NoEligibleOpponentAnchor
            }
            ReplayOpponentOutcome::NoAlternativeResponse => PriorOutcome::NoAlternativeResponse,
            ReplayOpponentOutcome::PartialOpponentEndpoint { .. } => {
                PriorOutcome::PartialOpponentEndpoint
            }
            ReplayOpponentOutcome::CompletedOpponentEndpoint { .. } => {
                PriorOutcome::CompletedOpponentEndpoint
            }
            ReplayOpponentOutcome::RulesTerminalOpponentEndpoint { .. } => {
                PriorOutcome::RulesTerminalOpponentEndpoint
            }
        };
    }
    pub fn readiness(&self) -> PriorReadiness {
        self.readiness
    }
    pub fn native_work_interval(&self) -> Option<(std::time::Instant, std::time::Instant)> {
        Some((self.native_work_started_at?, self.native_work_returned_at?))
    }
    pub fn outcome(&self) -> PriorOutcome {
        self.outcome
    }
    pub fn stages(&self) -> &[PriorStage] {
        &self.stages[..self.stage_count]
    }
    pub fn opponent_counterline(&self) -> &[u16] {
        self.opponent_counterline.as_slice()
    }
    pub fn repaired_line(&self) -> &[u16] {
        self.repaired_line.as_slice()
    }
    pub(super) fn set_readiness(&mut self, value: PriorReadiness) {
        self.readiness = value;
    }
}

fn fault(reason: &'static str) -> AdmissionFault {
    AdmissionFault {
        stage: "query_prior",
        reason,
    }
}

/// Borrowed checked projection. No deserializer or mutable feature accessor is
/// supplied. Actual native replay calls this consumer before returning /2 bytes.
pub struct CheckedNativeReplayPrior<'a> {
    observation: &'a NativeReplayObservation,
    readiness: PriorReadiness,
}
impl<'a> CheckedNativeReplayPrior<'a> {
    /// The original native observation borrowed by this checked capability.
    /// This does not deserialize, copy or admit a reported child observation.
    pub fn observation(&self) -> &'a NativeReplayObservation {
        self.observation
    }
    pub fn projection(&self) -> &'a ReplayQueryPrior {
        self.observation
            .query_prior
            .as_ref()
            .expect("checked /2 projection")
    }
    pub fn readiness(&self) -> PriorReadiness {
        self.readiness
    }
}

pub fn consume_native_replay_prior<'a>(
    observation: &'a NativeReplayObservation,
    expected: &ReplayExpectedPins,
    expected_asset_profile_artifact: &super::super::ArtifactPin,
    expected_asset_profile: &CpuFreshAssetProfile,
) -> Result<CheckedNativeReplayPrior<'a>, AdmissionFault> {
    if observation.repair_anchor_evidence.is_some() {
        return Err(fault("legacy /3 prior rejects Repair evidence lane"));
    }
    consume_native_replay_prior_schema(
        observation,
        expected,
        expected_asset_profile_artifact,
        expected_asset_profile,
        super::QUERY_PRIOR_OBSERVATION_SCHEMA,
    )
}

fn consume_native_replay_prior_schema<'a>(
    observation: &'a NativeReplayObservation,
    expected: &ReplayExpectedPins,
    expected_asset_profile_artifact: &super::super::ArtifactPin,
    expected_asset_profile: &CpuFreshAssetProfile,
    schema: &str,
) -> Result<CheckedNativeReplayPrior<'a>, AdmissionFault> {
    let audit = &observation.input_admission;
    let projection = observation
        .query_prior
        .as_ref()
        .ok_or_else(|| fault("explicit /2 projection required"))?;
    if observation.mode != ReplayInputMode::RepairOpponent4n
        || observation.schema != schema
        || observation.scope != super::SCOPE
        || audit.mode != observation.mode
        || audit.schema != observation.mode.input_schema()
        || audit.scope != super::super::replay_inputs::SCOPE
        || expected
            .registered_artifacts
            .opponent_recheck_source
            .is_none()
        || observation.binary_pin_scope != super::super::replay_inputs::BINARY_DECLARATION_SCOPE
        || audit.binary_pin_scope != observation.binary_pin_scope
        || audit.parent != expected.parent
        || audit.binding != expected.binding
        || audit.registration_artifact != expected.registration_artifact
        || audit.prepared_action_artifact != expected.prepared_action_artifact
        || audit.semantic_receipt_artifact != expected.semantic_receipt_artifact
        || audit.registered_artifacts != expected.registered_artifacts
        || audit.provider_factory_id != expected.provider_factory_id
        || audit.legacy_cpu_profile_sha256 != expected.legacy_cpu_profile_sha256
        || observation.authorities != ReplayAuthorities::default()
        || audit.authorities != ReplayAuthorities::default()
        || projection.authorities != ReplayAuthorities::default()
        || audit.actual_utility_groups != 0
        || projection.actual_utility_groups != 0
        || projection.schema != SCHEMA
        || projection.assurance_scope != SCOPE
        || projection.projection_source_sha256 != projection_source_digest()
        || observation.asset_profile_artifact != *expected_asset_profile_artifact
        || expected_asset_profile.schema != super::ASSET_PROFILE_SCHEMA
        || expected_asset_profile.domain != "cpu_fresh"
        || expected_asset_profile.provider != "cpu"
    {
        return Err(fault(
            "independent registration/parent/current/frozen/query binding or authority differs",
        ));
    }
    if let Some(native) = &observation.native_receipt
        && (super::digest_string(&expected_asset_profile.model_epoch) != Some(native.model_epoch)
            || super::digest_string(&expected_asset_profile.export_manifest.sha256)
                != Some(native.export_manifest_sha256)
            || super::digest_string(&expected_asset_profile.encoding_semantic_sha256)
                != Some(native.encoding_semantic_sha256)
            || super::digest_string(&expected.registered_artifacts.provider_factory_source.sha256)
                != Some(native.adapter_source_sha256))
    {
        return Err(fault(
            "independent model/export/encoding/adapter binding differs",
        ));
    }
    let readiness = classify(observation, projection);
    Ok(CheckedNativeReplayPrior {
        observation,
        readiness,
    })
}

fn classify(
    observation: &NativeReplayObservation,
    projection: &ReplayQueryPrior,
) -> PriorReadiness {
    classify_readiness(&ReadinessObservation::from(observation), projection)
}

/// Common fact view for live native observations and separately labelled caller
/// reports. Constructing this view does not manufacture a native witness.
struct ReadinessObservation<'a> {
    canceled: bool,
    deadline_exceeded: bool,
    original_whole_wall_ms: u64,
    cleanup_reserve_ms: u64,
    elapsed_ms: u64,
    input_admission: ReadinessInput,
    status: &'a str,
    replay_returned_ok: bool,
    owner_created: bool,
    model_returned: bool,
    replay_attempted: bool,
    assets_source_admitted: bool,
    replay_state: &'a SnapshotText,
    opponent_state: Option<&'a SnapshotText>,
    outcome: &'a SnapshotText,
    trace_complete: bool,
    stages: &'a [CpuStageObservation],
    cpu_tasks_requested: Option<u64>,
    cpu_tasks_accounted: Option<u64>,
    cpu_reports_returned: Option<u64>,
    cpu_work_observation_incomplete: Option<bool>,
    cpu_nodes_lower_bound: Option<u64>,
    cleanup_attempted: bool,
    cleanup_returned_ok: bool,
    native_cleanup_missing: bool,
    native_bindings_observed: usize,
    native_ready_observed: usize,
    native_terminal_missing: usize,
    native_completion_unknown_observed: usize,
    native_receipt: Option<ClosureReceipt>,
}
struct ReadinessInput {
    whole_wall_ms: u64,
    cleanup_reserve_ms: u64,
    cpu_allowance: u64,
}
#[derive(Deserialize)]
struct ClosureReceipt {
    physical_shutdown_confirmed: bool,
    native_buffers_released: bool,
    quarantined: bool,
    physical_runs_in_flight: u64,
    observer_failures: u64,
    #[serde(deserialize_with = "required_option")]
    last_observer_failure: Option<serde_json::Value>,
    #[serde(deserialize_with = "required_option")]
    last_failure: Option<serde_json::Value>,
    failed_physical_role_calls: u64,
    invalid_role_outputs: u64,
    canceled_requests: u64,
    expired_requests: u64,
    completed_role_inputs: u64,
    search_consumed_role_inputs: u64,
    delivered_role_inputs: u64,
    physically_completed_role_calls: u64,
}
fn required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}
impl From<&crate::pals_native::NativeRoleReceipt> for ClosureReceipt {
    fn from(n: &crate::pals_native::NativeRoleReceipt) -> Self {
        Self {
            physical_shutdown_confirmed: n.physical_shutdown_confirmed,
            native_buffers_released: n.native_buffers_released,
            quarantined: n.quarantined,
            physical_runs_in_flight: n.physical_runs_in_flight,
            observer_failures: n.observer_failures,
            last_observer_failure: n
                .last_observer_failure
                .as_ref()
                .map(|_| serde_json::Value::Null),
            last_failure: n.last_failure.as_ref().map(|_| serde_json::Value::Null),
            failed_physical_role_calls: n.failed_physical_role_calls,
            invalid_role_outputs: n.invalid_role_outputs,
            canceled_requests: n.canceled_requests,
            expired_requests: n.expired_requests,
            completed_role_inputs: n.completed_role_inputs,
            search_consumed_role_inputs: n.search_consumed_role_inputs,
            delivered_role_inputs: n.delivered_role_inputs,
            physically_completed_role_calls: n.physically_completed_role_calls,
        }
    }
}
impl<'a> From<&'a NativeReplayObservation> for ReadinessObservation<'a> {
    fn from(o: &'a NativeReplayObservation) -> Self {
        Self {
            canceled: o.canceled,
            deadline_exceeded: o.deadline_exceeded,
            original_whole_wall_ms: o.original_whole_wall_ms,
            cleanup_reserve_ms: o.cleanup_reserve_ms,
            elapsed_ms: o.elapsed_ms,
            input_admission: ReadinessInput {
                whole_wall_ms: o.input_admission.whole_wall_ms,
                cleanup_reserve_ms: o.input_admission.cleanup_reserve_ms,
                cpu_allowance: o.input_admission.cpu_allowance,
            },
            status: o.status,
            replay_returned_ok: o.replay_returned_ok,
            owner_created: o.owner_created,
            model_returned: o.model_returned,
            replay_attempted: o.replay_attempted,
            assets_source_admitted: o.assets_source_admitted,
            replay_state: &o.replay_state,
            opponent_state: o.opponent_state.as_ref(),
            outcome: &o.outcome,
            trace_complete: o.trace_complete,
            stages: &o.stages,
            cpu_tasks_requested: o.cpu_tasks_requested,
            cpu_tasks_accounted: o.cpu_tasks_accounted,
            cpu_reports_returned: o.cpu_reports_returned,
            cpu_work_observation_incomplete: o.cpu_work_observation_incomplete,
            cpu_nodes_lower_bound: o.cpu_nodes_lower_bound,
            cleanup_attempted: o.cleanup_attempted,
            cleanup_returned_ok: o.cleanup_returned_ok,
            native_cleanup_missing: o.native_cleanup_missing,
            native_bindings_observed: o.native_bindings_observed,
            native_ready_observed: o.native_ready_observed,
            native_terminal_missing: o.native_terminal_missing,
            native_completion_unknown_observed: o.native_completion_unknown_observed,
            native_receipt: o.native_receipt.as_ref().map(ClosureReceipt::from),
        }
    }
}
fn classify_readiness(
    observation: &ReadinessObservation<'_>,
    projection: &ReplayQueryPrior,
) -> PriorReadiness {
    // The old execution_deadline_exceeded tick includes cleanup. Cleanup may
    // legitimately cross E while remaining within W; use the actual work-return
    // tick for E and preserve the old diagnostic unchanged.
    if observation.canceled
        || observation.deadline_exceeded
        || observation.original_whole_wall_ms != observation.input_admission.whole_wall_ms
        || observation.cleanup_reserve_ms != observation.input_admission.cleanup_reserve_ms
        || observation.cleanup_reserve_ms >= observation.original_whole_wall_ms
        || observation.elapsed_ms > observation.original_whole_wall_ms
    {
        return PriorReadiness::OriginalWindowFailed;
    }
    if observation.status != "observations_returned"
        || !observation.replay_returned_ok
        || !observation.owner_created
        || !observation.model_returned
        || !observation.replay_attempted
        || !observation.assets_source_admitted
        || !observation.replay_state.complete
        || !observation
            .opponent_state
            .as_ref()
            .is_some_and(|state| state.complete)
        || !observation.outcome.complete
        || !observation.trace_complete
    {
        return PriorReadiness::CapturedResultFailed;
    }
    if projection.work_returned_within_execution_window != Some(true)
        || !projection.work_returned_elapsed_ms.is_some_and(|elapsed| {
            elapsed < observation.original_whole_wall_ms - observation.cleanup_reserve_ms
        })
    {
        return PriorReadiness::OriginalWindowFailed;
    }
    let phases = [
        PriorPhase::Baseline,
        PriorPhase::After,
        PriorPhase::RepairEndpoint,
        PriorPhase::RepairOpponentEndpoint,
    ];
    let terminal = projection.outcome == PriorOutcome::RulesTerminalOpponentEndpoint;
    let required = if terminal { 3 } else { 4 };
    if (!terminal && projection.outcome != PriorOutcome::CompletedOpponentEndpoint)
        || projection.stage_count != required
        || observation.stages.len() != required
        || projection.stages[required..]
            .iter()
            .any(|stage| stage.phase.is_some())
        || observation.cpu_tasks_requested != Some(required as u64)
        || observation.cpu_tasks_accounted != Some(required as u64)
        || observation.cpu_reports_returned != Some(required as u64)
        || observation.cpu_work_observation_incomplete != Some(false)
        || !projection.endpoint_observed
        || projection.endpoint_rules_terminal != terminal
        || projection.opponent_anchor_ply.is_none()
        || projection.opponent_role_steps == 0
        || projection.opponent_role_steps > MAX_LINE_MOVES
        || projection.accepted_opponent_steps != projection.opponent_role_steps
        || projection.repaired_line.as_slice().is_empty()
        || projection.opponent_counterline.as_slice().is_empty()
        || projection.repaired_line.as_slice() == projection.opponent_counterline.as_slice()
    {
        return PriorReadiness::IncompleteCpuScope;
    }
    let mut nodes = 0u64;
    for (at, (stage, raw)) in projection
        .stages()
        .iter()
        .zip(observation.stages)
        .enumerate()
    {
        if stage.phase != Some(phases[at])
            || !stage.complete()
            || !raw.exact_completed
            || raw.phase
                != match phases[at] {
                    PriorPhase::Baseline => "baseline",
                    PriorPhase::After => "after",
                    PriorPhase::RepairEndpoint => "repair_endpoint",
                    PriorPhase::RepairOpponentEndpoint => "repair_opponent_endpoint",
                }
            || !raw.report_present
            || raw.observation_id.is_none()
            || !raw.attempt_present
            || raw.attempt_nodes != stage.attempt_nodes
            || !raw.raw_stage.complete
            || !raw.raw_task.complete
            || !raw.raw_observation.as_ref().is_some_and(|s| s.complete)
            || raw.report_nodes != stage.report_nodes
            || raw.requested_depth != stage.requested_depth
            || raw.report_qnodes != stage.report_qnodes
            || raw.report_tt_hits != stage.report_tt_hits
            || raw.node_budget != stage.node_budget
        {
            return PriorReadiness::IncompleteCpuScope;
        }
        let Some(total) = nodes.checked_add(stage.report_nodes.unwrap_or(0)) else {
            return PriorReadiness::IncompleteCpuScope;
        };
        nodes = total;
    }
    if observation.cpu_nodes_lower_bound != Some(nodes)
        || nodes > observation.input_admission.cpu_allowance
    {
        return PriorReadiness::IncompleteCpuScope;
    }
    if !observation.cleanup_attempted
        || !observation.cleanup_returned_ok
        || observation.native_cleanup_missing
        || observation.native_bindings_observed == 0
        || observation.native_ready_observed != observation.native_bindings_observed
        || observation.native_terminal_missing != 0
        || observation.native_completion_unknown_observed != 0
    {
        return PriorReadiness::NativeClosureUnobserved;
    }
    let Some(native) = &observation.native_receipt else {
        return PriorReadiness::NativeClosureUnobserved;
    };
    if !native.physical_shutdown_confirmed
        || !native.native_buffers_released
        || native.quarantined
        || native.physical_runs_in_flight != 0
        || native.observer_failures != 0
        || native.last_observer_failure.is_some()
        || native.last_failure.is_some()
        || native.failed_physical_role_calls != 0
        || native.invalid_role_outputs != 0
        || native.canceled_requests != 0
        || native.expired_requests != 0
        || native.completed_role_inputs != native.search_consumed_role_inputs
        || native.delivered_role_inputs != native.completed_role_inputs
        || native.physically_completed_role_calls != native.completed_role_inputs
        || native.completed_role_inputs != observation.native_ready_observed as u64
    {
        return PriorReadiness::NativeClosureUnobserved;
    }
    if terminal {
        PriorReadiness::RulesTerminalPendingCallerChronology
    } else {
        PriorReadiness::CompleteCpuScopePendingCallerChronology
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::CpuStageObservation;
    use super::*;
    #[test]
    fn live_fact_and_native_clock_are_not_imported_or_added_to_old_wire() {
        let mut prior = ReplayQueryPrior::default();
        let before = serde_json::to_value(&prior).unwrap();
        assert!(prior.native_work_interval().is_none());
        let now = std::time::Instant::now();
        prior.observe_work_return(now, now + std::time::Duration::from_secs(1));
        // A caller-supplied historical S is not actual native entry evidence.
        assert!(prior.native_work_interval().is_none());
        prior.observe_native_entry(now);
        assert!(prior.native_work_interval().is_some());
        let wire = serde_json::to_value(&prior).unwrap();
        assert_eq!(
            before.as_object().unwrap().keys().collect::<Vec<_>>(),
            wire.as_object().unwrap().keys().collect::<Vec<_>>()
        );
        assert!(
            !wire
                .as_object()
                .unwrap()
                .keys()
                .any(|key| key.contains("native_work"))
        );
        for stage in &prior.stages {
            assert!(stage.conditional_endpoint().is_none());
            assert!(
                serde_json::to_value(stage)
                    .unwrap()
                    .get("conditional_endpoint")
                    .is_none()
            );
        }
    }

    /// Controlled declarations only. This helper is never a production self-pin
    /// fallback or evidence of running ONNX/processes/strict parent admission.
    pub(in super::super) fn controlled_expected(o: &NativeReplayObservation) -> ReplayExpectedPins {
        let a = &o.input_admission;
        ReplayExpectedPins {
            registration_artifact: a.registration_artifact.clone(),
            prepared_action_artifact: a.prepared_action_artifact.clone(),
            semantic_receipt_artifact: a.semantic_receipt_artifact.clone(),
            parent: a.parent.clone(),
            binding: a.binding.clone(),
            registered_artifacts: a.registered_artifacts.clone(),
            legacy_cpu_profile_sha256: a.legacy_cpu_profile_sha256.clone(),
            provider_factory_id: a.provider_factory_id.clone(),
            semantic_binary_sha256: "a".repeat(64),
        }
    }
    pub(super) fn fixture() -> NativeReplayObservation {
        let mut o = super::super::tests::receipt(super::super::tests::clock(4 * 1024 * 1024));
        o.mode = ReplayInputMode::RepairOpponent4n;
        o.schema = super::super::QUERY_PRIOR_OBSERVATION_SCHEMA;
        o.input_admission.mode = o.mode;
        o.input_admission.schema = o.mode.input_schema();
        o.input_admission
            .conservative_repair_role_store_overreservation = false;
        o.input_admission
            .registered_artifacts
            .opponent_recheck_source = Some(super::super::super::pin(b"controlled_source"));
        o.input_admission.cpu_allowance = 100;
        o.status = "observations_returned";
        o.model_returned = true;
        o.owner_created = true;
        o.replay_attempted = true;
        o.replay_returned_ok = true;
        o.assets_source_admitted = true;
        o.replay_state.complete = true;
        o.outcome.complete = true;
        o.opponent_state = Some(super::super::SnapshotText {
            text: String::new(),
            complete: true,
        });
        o.cpu_tasks_requested = Some(4);
        o.cpu_tasks_accounted = Some(4);
        o.cpu_reports_returned = Some(4);
        o.cpu_nodes_lower_bound = Some(40);
        o.cpu_work_observation_incomplete = Some(false);
        let mut p = ReplayQueryPrior {
            outcome: PriorOutcome::CompletedOpponentEndpoint,
            work_returned_elapsed_ms: Some(1),
            work_returned_within_execution_window: Some(true),
            stage_count: 4,
            endpoint_observed: true,
            opponent_anchor_ply: Some(3),
            opponent_role_steps: 2,
            accepted_opponent_steps: 2,
            ..ReplayQueryPrior::default()
        };
        p.repaired_line
            .capture(&[BoardMove::from_uci("e2e4").unwrap()], MAX_LINE_MOVES)
            .unwrap();
        p.opponent_counterline
            .capture(&[BoardMove::from_uci("d2d4").unwrap()], MAX_LINE_MOVES)
            .unwrap();
        let phases = [
            PriorPhase::Baseline,
            PriorPhase::After,
            PriorPhase::RepairEndpoint,
            PriorPhase::RepairOpponentEndpoint,
        ];
        let raw_phases = [
            "baseline",
            "after",
            "repair_endpoint",
            "repair_opponent_endpoint",
        ];
        for (at, stage) in p.stages.iter_mut().enumerate() {
            stage.phase = Some(phases[at]);
            stage.requested_depth = 2;
            stage.node_budget = 25;
            stage.exact_completed = true;
            stage.report_present = true;
            stage.completed_depth = Some(2);
            stage.completion = Some(PriorCompletion::DepthLimit);
            stage.completed_iteration = true;
            stage.reused_completed_depth = Some(0);
            stage.report_nodes = Some(10);
            stage.report_qnodes = Some(0);
            stage.report_tt_hits = Some(0);
            stage.observation_present = true;
            stage.attempt_present = true;
            stage.attempt_nodes = Some(10);
            let mut raw = CpuStageObservation::reserved().unwrap();
            raw.phase = raw_phases[at];
            raw.cpu_execution = at;
            raw.requested_depth = 2;
            raw.node_budget = 25;
            raw.exact_completed = true;
            raw.report_present = true;
            raw.report_nodes = Some(10);
            raw.report_qnodes = Some(0);
            raw.report_tt_hits = Some(0);
            raw.observation_id = Some(at);
            raw.attempt_present = true;
            raw.attempt_nodes = Some(10);
            raw.raw_stage.complete = true;
            raw.raw_task.complete = true;
            raw.raw_observation.as_mut().unwrap().complete = true;
            o.stages.push(raw);
        }
        o.query_prior = Some(p);
        o
    }
    fn ready(o: &NativeReplayObservation) -> PriorReadiness {
        consume_native_replay_prior(
            o,
            &controlled_expected(o),
            &o.asset_profile_artifact,
            &super::super::tests::profile(false),
        )
        .unwrap()
        .readiness()
    }
    fn controlled_projection_source() -> super::super::super::ArtifactPin {
        // Independent test fixture choice only, never a production source fallback.
        super::super::super::pin(include_bytes!("replay_prior.rs"))
    }
    fn reported(
        body: serde_json::Value,
        o: &NativeReplayObservation,
    ) -> Result<ReportedPriorConsistency, AdmissionFault> {
        check_reported_prior_consistency(
            body,
            &controlled_expected(o),
            &o.asset_profile_artifact,
            &super::super::tests::profile(false),
            &controlled_projection_source(),
        )
    }
    #[test]
    fn reported_readiness_recomputes_the_same_cpu_window_and_partial_scope_cases() {
        for case in 0..7 {
            let mut o = fixture();
            match case {
                0 => {}
                1 => o.canceled = true,
                2 => o.query_prior.as_mut().unwrap().stages[3].reused_completed_depth = Some(1),
                3 => o.stages[2].report_nodes = Some(20),
                4 => o.cpu_nodes_lower_bound = None,
                5 => {
                    let p = o.query_prior.as_mut().unwrap();
                    p.outcome = PriorOutcome::RulesTerminalOpponentEndpoint;
                    p.endpoint_rules_terminal = true;
                    p.stage_count = 3;
                    p.stages[3] = PriorStage::default();
                    o.stages.truncate(3);
                    o.cpu_tasks_requested = Some(3);
                    o.cpu_tasks_accounted = Some(3);
                    o.cpu_reports_returned = Some(3);
                    o.cpu_nodes_lower_bound = Some(30);
                }
                _ => {
                    o.execution_deadline_exceeded = true;
                    o.elapsed_ms = 40_000;
                }
            }
            let expected = ready(&o);
            o.query_prior.as_mut().unwrap().readiness = expected;
            let checked = reported(serde_json::to_value(&o).unwrap(), &o).unwrap();
            assert_eq!(
                checked.recomputed_reported_readiness(),
                expected,
                "case {case}"
            );
            assert_eq!(
                checked.reported_projection().authorities,
                ReplayAuthorities::default()
            );
            assert_eq!(checked.reported_projection().actual_utility_groups, 0);
        }
    }
    #[test]
    fn reported_readiness_overclaims_wrong_types_and_unknown_projection_are_refused() {
        let mut o = fixture();
        let computed = ready(&o);
        o.query_prior.as_mut().unwrap().readiness = computed;
        let body = serde_json::to_value(&o).unwrap();
        for case in 0..6 {
            let mut changed = body.clone();
            match case {
                0 => {
                    changed["query_prior"]["readiness"] =
                        serde_json::json!("complete_cpu_scope_pending_caller_chronology")
                }
                1 => changed["cpu_nodes_lower_bound"] = serde_json::json!(true),
                2 => changed["query_prior"]["unknown_authority"] = serde_json::json!(true),
                3 => {
                    changed["stages"][3]
                        .as_object_mut()
                        .unwrap()
                        .remove("report_present");
                }
                4 => {
                    changed["query_prior"]["projection_source_sha256"] =
                        serde_json::json!(vec![0u8; 32])
                }
                _ => {
                    changed["input_admission"]["authorities"]["utility_authority"] =
                        serde_json::json!(true)
                }
            }
            assert!(reported(changed, &o).is_err(), "case {case}");
        }
    }
    #[test]
    fn reported_complete_closure_is_only_a_report_and_retains_no_native_witness_authority() {
        let mut o = fixture();
        o.query_prior.as_mut().unwrap().readiness =
            PriorReadiness::CompleteCpuScopePendingCallerChronology;
        let mut body = serde_json::to_value(&o).unwrap();
        body["cleanup_attempted"] = serde_json::json!(true);
        body["cleanup_returned_ok"] = serde_json::json!(true);
        body["native_cleanup_missing"] = serde_json::json!(false);
        body["native_bindings_observed"] = serde_json::json!(1);
        body["native_ready_observed"] = serde_json::json!(1);
        body["native_terminal_missing"] = serde_json::json!(0);
        body["native_completion_unknown_observed"] = serde_json::json!(0);
        let profile = super::super::tests::profile(false);
        let expected = controlled_expected(&o);
        // Synthetic reported closure, deliberately NOT a NativeRoleReceipt.
        body["native_receipt"] = serde_json::json!({
            "model_epoch":super::super::digest_string(&profile.model_epoch).unwrap(),
            "export_manifest_sha256":super::super::digest_string(&profile.export_manifest.sha256).unwrap(),
            "encoding_semantic_sha256":super::super::digest_string(&profile.encoding_semantic_sha256).unwrap(),
            "adapter_source_sha256":super::super::digest_string(&expected.registered_artifacts.provider_factory_source.sha256).unwrap(),
            "physical_shutdown_confirmed":true,"native_buffers_released":true,"quarantined":false,
            "physical_runs_in_flight":0,"observer_failures":0,"last_observer_failure":null,"last_failure":null,
            "failed_physical_role_calls":0,"invalid_role_outputs":0,"canceled_requests":0,"expired_requests":0,
            "completed_role_inputs":1,"search_consumed_role_inputs":1,"delivered_role_inputs":1,"physically_completed_role_calls":1,
        });
        let checked = reported(body.clone(), &o).unwrap();
        assert_eq!(
            checked.recomputed_reported_readiness(),
            PriorReadiness::CompleteCpuScopePendingCallerChronology
        );
        assert_eq!(
            checked.assurance_scope(),
            "reported_consistency_pending_native_witness_and_caller_chronology"
        );
        assert_eq!(
            checked.reported_projection().authorities,
            ReplayAuthorities::default()
        );
        assert_eq!(ready(&o), PriorReadiness::NativeClosureUnobserved);
        for case in 0..5 {
            let mut changed = body.clone();
            match case {
                0 => changed["native_receipt"]["delivered_role_inputs"] = serde_json::json!(2),
                1 => changed["native_receipt"]["physical_runs_in_flight"] = serde_json::json!(1),
                2 => changed["native_receipt"]["model_epoch"] = serde_json::json!(vec![0u8; 32]),
                3 => {
                    changed["native_receipt"]
                        .as_object_mut()
                        .unwrap()
                        .remove("last_failure");
                }
                _ => changed["native_receipt"]["quarantined"] = serde_json::json!(true),
            }
            assert!(reported(changed, &o).is_err(), "case {case}");
        }
    }
    #[test]
    fn four_exact_cpu_stages_without_native_evidence_remain_unadmitted() {
        let o = fixture();
        assert_eq!(ready(&o), PriorReadiness::NativeClosureUnobserved);
        let p = o.query_prior.as_ref().unwrap();
        assert_eq!(p.authorities, ReplayAuthorities::default());
        assert_eq!(p.actual_utility_groups, 0);
        let value = serde_json::to_value(p).unwrap();
        assert!(!value.to_string().contains("cpu_execution"));
        assert!(!value.to_string().contains("raw_stage"));
        assert!(!value.to_string().contains("score\":"));
    }
    #[test]
    fn independent_parent_current_frozen_query_and_source_bindings_do_not_cross() {
        let o = fixture();
        for at in 0..9 {
            let mut e = controlled_expected(&o);
            match at {
                0 => e.parent.parent_input_sha256 = "b".repeat(64),
                1 => e.parent.current_view_sha256 = "b".repeat(64),
                2 => e.parent.frozen_admission_sha256 = "b".repeat(64),
                3 => e.parent.encoding_sha256 = "b".repeat(64),
                4 => e.binding.query_sha256 = "b".repeat(64),
                5 => e.binding.prior_ledger_sha256 = "b".repeat(64),
                6 => e.registration_artifact.sha256 = "b".repeat(64),
                7 => e.registered_artifacts.opponent_recheck_source = None,
                _ => e.provider_factory_id = "other".into(),
            }
            assert!(
                consume_native_replay_prior(
                    &o,
                    &e,
                    &o.asset_profile_artifact,
                    &super::super::tests::profile(false)
                )
                .is_err()
            );
        }
        let mut o = fixture();
        o.authorities.utility_authority = true;
        assert!(
            consume_native_replay_prior(
                &o,
                &controlled_expected(&o),
                &o.asset_profile_artifact,
                &super::super::tests::profile(false)
            )
            .is_err()
        );
        let mut o = fixture();
        o.mode = ReplayInputMode::RepairEndpoint3n;
        assert!(
            consume_native_replay_prior(
                &o,
                &controlled_expected(&o),
                &o.asset_profile_artifact,
                &super::super::tests::profile(false)
            )
            .is_err()
        );
        let o = fixture();
        assert!(
            consume_native_replay_prior(
                &o,
                &controlled_expected(&o),
                &super::super::super::pin(b"other_asset_profile"),
                &super::super::tests::profile(false)
            )
            .is_err()
        );
    }
    #[test]
    fn partial_reordered_frontier_reused_and_missing_work_never_become_complete_scope() {
        for at in 0..8 {
            let mut o = fixture();
            let p = o.query_prior.as_mut().unwrap();
            match at {
                0 => p.outcome = PriorOutcome::PartialOpponentEndpoint,
                1 => p.stages[3].exact_completed = false,
                2 => p.stages.swap(2, 3),
                3 => p.stages[3].completed_iteration = false,
                4 => p.stages[3].reused_completed_depth = Some(1),
                5 => o.cpu_nodes_lower_bound = None,
                6 => p.stages[3].report_nodes = Some(26),
                _ => p.accepted_opponent_steps = 1,
            }
            assert_eq!(ready(&o), PriorReadiness::IncompleteCpuScope);
        }
    }
    #[test]
    fn cleanup_crossing_execution_partition_does_not_relabel_timely_work_as_late() {
        let mut o = fixture();
        o.execution_deadline_exceeded = true;
        o.elapsed_ms = 40_000; // work at 1ms, E=30s, cleanup still within W=60s
        assert_eq!(ready(&o), PriorReadiness::NativeClosureUnobserved);
        o.query_prior
            .as_mut()
            .unwrap()
            .work_returned_within_execution_window = Some(false);
        assert_eq!(ready(&o), PriorReadiness::OriginalWindowFailed);
        o.query_prior
            .as_mut()
            .unwrap()
            .work_returned_within_execution_window = Some(true);
        o.canceled = true;
        assert_eq!(ready(&o), PriorReadiness::OriginalWindowFailed);
    }
    #[test]
    fn terminal_after_three_checks_has_no_fabricated_fourth_cpu_and_still_needs_native_closure() {
        let mut o = fixture();
        let p = o.query_prior.as_mut().unwrap();
        p.outcome = PriorOutcome::RulesTerminalOpponentEndpoint;
        p.endpoint_rules_terminal = true;
        p.stage_count = 3;
        p.stages[3] = PriorStage::default();
        o.stages.truncate(3);
        o.cpu_tasks_requested = Some(3);
        o.cpu_tasks_accounted = Some(3);
        o.cpu_reports_returned = Some(3);
        o.cpu_nodes_lower_bound = Some(30);
        assert_eq!(ready(&o), PriorReadiness::NativeClosureUnobserved);
        assert_eq!(
            serde_json::to_value(o.query_prior.as_ref().unwrap()).unwrap()["stages"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        o.query_prior.as_mut().unwrap().stage_count = 4;
        assert_eq!(ready(&o), PriorReadiness::IncompleteCpuScope);
    }
    #[test]
    fn fixed_projection_extent_and_move16_match_existing_export_without_truncation() {
        let mut p = ReplayQueryPrior::default();
        let movement = BoardMove::from_uci("a7a8n").unwrap();
        let moves = [movement; MAX_PV_MOVES];
        for s in &mut p.stages {
            s.phase = Some(PriorPhase::Baseline);
            s.pv.capture(&moves, MAX_PV_MOVES).unwrap();
            s.report_nodes = Some(u64::MAX);
            s.report_qnodes = Some(u64::MAX);
            s.report_tt_hits = Some(u64::MAX);
            s.report_elapsed_ms = Some(u64::MAX);
            s.attempt_nodes = Some(u64::MAX);
            s.node_budget = u64::MAX;
            s.registered_condition_sha256 = [255; 32];
            s.task_condition_sha256 = [255; 32];
        }
        p.repaired_line
            .capture(&moves[..MAX_LINE_MOVES], MAX_LINE_MOVES)
            .unwrap();
        p.opponent_counterline
            .capture(&moves[..MAX_LINE_MOVES], MAX_LINE_MOVES)
            .unwrap();
        let raw = serde_json::to_vec(&p).unwrap();
        assert!(raw.len() <= MAX_PROJECTED_JSON_BYTES);
        assert_eq!(
            p.stages[0].pv.as_slice(),
            crate::pals_cpu_task::pack_moves(&moves).unwrap()
        );
        assert!(
            p.repaired_line
                .capture(&moves[..MAX_LINE_MOVES + 1], MAX_LINE_MOVES)
                .is_err()
        );
        assert!(p.repaired_line.as_slice().is_empty());
        const {
            assert!(MAX_PROJECTED_JSON_BYTES < super::super::HEADER_RESERVE);
        }
    }
}
