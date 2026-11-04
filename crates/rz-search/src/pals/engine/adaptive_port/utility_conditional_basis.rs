//! Cohort-only historical conditional comparison basis.
//!
//! These bounded bytes preserve an actual publisher's original sources. Pure
//! validation reconstructs no live Store, accepted head, known target or job.
//! Existing raw witness preimages and their hashing contract stay unchanged.
use super::conditional_evidence::{ConditionalEndpoint, EndpointBasis};
use super::utility_route::{
    HistoricalWitnessObservation, HistoricalWitnessRawScore, UtilityWitnessPreimageError,
    UtilityWitnessPreimageExpectation, parse_validated_utility_witness_preimage,
};
use super::utility_route::{UtilityRouteSemanticOutcome, UtilityRouteWitness};
use super::validator_port::ActionWitnessKind;
use super::*;
use crate::cpu_worker::CpuWorkerProfile;
use crate::pals::store::TaskConsumer;
use rz_contracts::pals_adaptive::{CpuJobEnvelope, CpuWorkerId};
use rz_position::{
    HistoryCompleteness, HistoryOrigin, PlayStatus, PositionLimits, PositionSnapshot, UciReplay,
};
use serde::{Deserialize, Serialize};
use std::io::{self, Write};

const DOMAIN: &str = "rz-rovent-utility-cohort-conditional-basis/1";
const MAX_BYTES: usize = 16 * 1024 * 1024;
const MAX_LINE: usize = 256;
mod raw_join;
#[cfg(test)]
mod tests;

/// Original accepted CPU source, projected only from the private consume seal.
/// This owns no View, Bound, worker stack or live Store pin.
#[derive(Clone)]
pub(super) struct UtilityCpuContextMetadata {
    pub(super) generation: u64,
    pub(super) root: StateId,
    pub(super) authority: SearchAuthority,
    pub(super) frame: FrameId,
    pub(super) question: ContractDigest,
    pub(super) task: TaskKey,
    pub(super) execution: ExecutionId,
    pub(super) observation: ObservationId,
    pub(super) job: CpuJobEnvelope,
    pub(super) owner: CpuWorkerId,
    pub(super) profile: CpuWorkerProfile,
    pub(super) original_consumer: TaskConsumer,
}

#[derive(Clone)]
pub(super) struct UtilityCpuAcceptedMetadata {
    pub(super) original_consumer: TaskConsumer,
    pub(super) contextual: Option<UtilityCpuContextMetadata>,
    pub(super) legacy_condition: Option<String>,
}

pub struct UtilityConditionalWitnessBasis {
    original: Vec<u8>,
    sha256: ContractDigest,
}
impl UtilityConditionalWitnessBasis {
    pub fn domain(&self) -> &'static str {
        DOMAIN
    }
    pub fn original_bytes(&self) -> &[u8] {
        &self.original
    }
    pub fn sha256(&self) -> ContractDigest {
        self.sha256
    }
    pub fn owned_host_bytes(&self) -> Option<u64> {
        u64::try_from(std::mem::size_of::<Self>().checked_add(self.original.capacity())?).ok()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UtilityConditionalWitnessBasisUnavailable {
    MissingOriginalComparison,
    MissingResidentEvidence,
    IncompleteContext,
    NamespaceMismatch,
    Capacity,
    InvalidOriginalBasis,
}
impl UtilityConditionalWitnessBasisUnavailable {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MissingOriginalComparison => "missing_original_comparison",
            Self::MissingResidentEvidence => "missing_resident_evidence",
            Self::IncompleteContext => "incomplete_context",
            Self::NamespaceMismatch => "namespace_mismatch",
            Self::Capacity => "capacity",
            Self::InvalidOriginalBasis => "invalid_original_basis",
        }
    }
}

#[derive(Clone, Copy)]
pub struct UtilityWitnessBasisExpectation<'a> {
    pub witness: UtilityWitnessPreimageExpectation<'a>,
    pub observation: ObservationId,
    pub basis_sha256: ContractDigest,
    pub preimage: &'a [u8],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UtilityWitnessBasisError {
    Limit(&'static str),
    Invalid(&'static str),
    Mismatch(&'static str),
    Raw(UtilityWitnessPreimageError),
}
impl std::fmt::Display for UtilityWitnessBasisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "historical conditional basis: {self:?}")
    }
}
impl std::error::Error for UtilityWitnessBasisError {}
type BasisError = UtilityWitnessBasisError;
type Unavailable = UtilityConditionalWitnessBasisUnavailable;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    origin: u8,
    start_fen: String,
    endpoint_fen: String,
    moves: Vec<Move16>,
    revision: u64,
}
impl Snapshot {
    fn capture(snapshot: &PositionSnapshot) -> Result<Self, Unavailable> {
        let trace = snapshot
            .uci_replay(rz_position::MAX_UCI_REPLAY_PLIES)
            .map_err(|_| Unavailable::InvalidOriginalBasis)?;
        Ok(Self {
            origin: match trace.origin {
                HistoryOrigin::StartPosition => 0,
                HistoryOrigin::Fen => 1,
            },
            start_fen: trace.start_fen,
            endpoint_fen: snapshot.to_fen(),
            moves: pack(&trace.moves).map_err(|_| Unavailable::InvalidOriginalBasis)?,
            revision: snapshot.revision(),
        })
    }
    fn decode(&self) -> Result<PositionSnapshot, BasisError> {
        if self.start_fen.len() > 4096
            || self.endpoint_fen.len() > 4096
            || self.moves.len() > rz_position::MAX_UCI_REPLAY_PLIES
        {
            return Err(BasisError::Limit("Rules snapshot"));
        }
        let (origin, completeness) = match self.origin {
            0 => (HistoryOrigin::StartPosition, HistoryCompleteness::Complete),
            1 => (HistoryOrigin::Fen, HistoryCompleteness::UnknownPrefix),
            _ => return Err(BasisError::Invalid("Rules origin")),
        };
        let trace = UciReplay {
            origin,
            completeness,
            start_fen: self.start_fen.clone(),
            moves: unpack(&self.moves)?,
        };
        let snapshot = PositionSnapshot::from_uci_replay(
            &trace,
            self.revision,
            PositionLimits {
                max_history_positions: rz_position::MAX_UCI_REPLAY_PLIES + 1,
                ..Default::default()
            },
        )
        .map_err(|_| BasisError::Invalid("Rules replay"))?;
        if snapshot.to_fen() != self.endpoint_fen {
            return Err(BasisError::Mismatch("Rules endpoint"));
        }
        Ok(snapshot)
    }
}

#[derive(Clone, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct ModelIdentity {
    semantics: String,
    model: String,
    encoding: String,
    precision: String,
    epoch: [u8; 32],
}
impl ModelIdentity {
    fn capture(value: &ModelValueIdentity) -> Self {
        Self {
            semantics: value.semantics.clone(),
            model: value.model.clone(),
            encoding: value.encoding.clone(),
            precision: value.precision.clone(),
            epoch: value.model_epoch,
        }
    }
    fn validate(&self) -> Result<(), BasisError> {
        ModelValueIdentity {
            semantics: self.semantics.clone(),
            model: self.model.clone(),
            encoding: self.encoding.clone(),
            precision: self.precision.clone(),
            model_epoch: self.epoch,
        }
        .validate()
        .map_err(|_| BasisError::Invalid("model namespace"))?;
        if self.semantics != crate::pals::value::MODEL_WDL_VALUE_SEMANTICS {
            return Err(BasisError::Invalid("model WDL semantics"));
        }
        Ok(())
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredLine {
    id: LineId,
    moves: Vec<Move16>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
enum Scope {
    Rules,
    Depth {
        depth: u16,
        profile: u64,
        condition: u64,
    },
    Model {
        model: u64,
        encoding: u64,
        input: u64,
    },
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
enum Score {
    Unknown,
    Estimate {
        bits: u32,
        side: u8,
    },
    Cpu {
        value: i32,
        side: u8,
        bound: BoundKind,
    },
    Wdl {
        bits: [u32; 3],
        side: u8,
    },
    ContextWdl {
        bits: [u32; 3],
        side: u8,
        revision: u64,
    },
    ConditionalWdl {
        bits: u32,
        side: u8,
        before: ObservationId,
        after: ObservationId,
        revision: u64,
    },
    ConditionalRepairWdl {
        bits: u32,
        side: u8,
        before: ObservationId,
        after: ObservationId,
        revision: u64,
    },
    Terminal {
        winner: Option<u8>,
    },
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationSource {
    id: ObservationId,
    state_id: StateId,
    state: Snapshot,
    line: Option<StoredLine>,
    source: u64,
    epoch: u64,
    value: Option<CpuValueIdentity>,
    checker: Option<()>,
    work: Option<[Option<u64>; 3]>,
    external: Option<()>,
    model: Option<ModelIdentity>,
    input: Option<[u8; 32]>,
    condition: Option<String>,
    cpu_pv: Option<StoredLine>,
    scope: Scope,
    score: Score,
    budget: u64,
    kind: ObservationKind,
    supersedes: Option<ObservationId>,
    execution: Option<ExecutionId>,
}
impl ObservationSource {
    fn capture(stores: &PalsStores, id: ObservationId) -> Result<Self, Unavailable> {
        let value = stores
            .observations
            .get(id)
            .map_err(|_| Unavailable::MissingResidentEvidence)?;
        if value.checker_identity.is_some() || value.external_report.is_some() {
            return Err(Unavailable::NamespaceMismatch);
        }
        let line = |id: LineId| -> Result<StoredLine, Unavailable> {
            Ok(StoredLine {
                id,
                moves: pack(
                    &stores
                        .lines
                        .moves(id)
                        .map_err(|_| Unavailable::MissingResidentEvidence)?,
                )
                .map_err(|_| Unavailable::InvalidOriginalBasis)?,
            })
        };
        let scope = match value.scope {
            EvidenceScope::RulesTerminal => Scope::Rules,
            EvidenceScope::DepthLimited {
                depth,
                profile,
                condition,
            } => Scope::Depth {
                depth,
                profile,
                condition,
            },
            EvidenceScope::Model {
                model,
                encoding,
                input,
            } => Scope::Model {
                model,
                encoding,
                input,
            },
            _ => return Err(Unavailable::NamespaceMismatch),
        };
        let score = match value.score {
            RawScore::Unknown => Score::Unknown,
            RawScore::Estimate { value, perspective } => Score::Estimate {
                bits: value.to_bits(),
                side: side(perspective),
            },
            RawScore::Cpu {
                value,
                perspective,
                bound,
            } => Score::Cpu {
                value,
                side: side(perspective),
                bound,
            },
            RawScore::Wdl {
                win,
                draw,
                loss,
                perspective,
            } => Score::Wdl {
                bits: [win.to_bits(), draw.to_bits(), loss.to_bits()],
                side: side(perspective),
            },
            RawScore::ContextWdl {
                win,
                draw,
                loss,
                perspective,
                context_revision,
            } => Score::ContextWdl {
                bits: [win.to_bits(), draw.to_bits(), loss.to_bits()],
                side: side(perspective),
                revision: context_revision,
            },
            RawScore::ConditionalWdl {
                expectation,
                perspective,
                repaired,
                counter,
                context_revision,
            } => Score::ConditionalWdl {
                bits: expectation.to_bits(),
                side: side(perspective),
                before: repaired,
                after: counter,
                revision: context_revision,
            },
            RawScore::ConditionalRepairWdl {
                expectation,
                perspective,
                before,
                after,
                context_revision,
            } => Score::ConditionalRepairWdl {
                bits: expectation.to_bits(),
                side: side(perspective),
                before,
                after,
                revision: context_revision,
            },
            RawScore::Terminal { winner } => Score::Terminal {
                winner: winner.map(side),
            },
            _ => return Err(Unavailable::NamespaceMismatch),
        };
        Ok(Self {
            id,
            state_id: value.state,
            state: Snapshot::capture(
                stores
                    .states
                    .get(value.state)
                    .map_err(|_| Unavailable::MissingResidentEvidence)?,
            )?,
            line: value.line.map(line).transpose()?,
            source: value.source,
            epoch: value.epoch,
            value: value.value_identity.clone(),
            checker: None,
            work: value.checker_work.map(|w| [w.nodes, w.qnodes, w.tt_hits]),
            external: None,
            model: value
                .model_value_identity
                .as_ref()
                .map(ModelIdentity::capture),
            input: value.model_value_input,
            condition: value.cpu_condition.clone(),
            cpu_pv: value.cpu_pv.map(line).transpose()?,
            scope,
            score,
            budget: value.budget,
            kind: value.kind,
            supersedes: value.supersedes,
            execution: value.execution,
        })
    }
}

#[derive(Clone, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct Authority {
    epoch: u64,
    game: u64,
    root: u64,
    implementation: [u8; 32],
}
impl From<SearchAuthority> for Authority {
    fn from(a: SearchAuthority) -> Self {
        Self {
            epoch: a.epoch.0,
            game: a.game.0,
            root: a.root.0,
            implementation: a.implementation.0,
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Consumer {
    consumer: TaskConsumer,
    cancelled: bool,
    consumed: bool,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Task {
    key: TaskKey,
    status: TaskStatus,
    resumed_from: Option<ExecutionId>,
    consumers: Vec<Consumer>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CpuContext {
    generation: u64,
    root: StateId,
    authority: Authority,
    frame_authority: Authority,
    frame_sequence: u64,
    question: [u8; 32],
    task: TaskKey,
    execution: ExecutionId,
    observation: ObservationId,
    consumer: TaskConsumer,
    job_authority: Authority,
    job_sequence: u64,
    job_state_owner: u64,
    job_state_revision: u64,
    job_state_semantic: [u8; 32],
    job_question: [u8; 32],
    root_restriction: [u8; 32],
    evaluation_namespace: [u8; 32],
    search_conditions: [u8; 32],
    total_credit: u64,
    deadline_clock: u64,
    deadline_at: u64,
    owner: u16,
    profile_owner: u16,
    profile_depth: u16,
    profile_evaluation_namespace: [u8; 32],
    profile_search_conditions: [u8; 32],
}
impl From<UtilityCpuContextMetadata> for CpuContext {
    fn from(c: UtilityCpuContextMetadata) -> Self {
        Self {
            generation: c.generation,
            root: c.root,
            authority: c.authority.into(),
            frame_authority: c.frame.authority.into(),
            frame_sequence: c.frame.sequence,
            question: c.question.0,
            task: c.task,
            execution: c.execution,
            observation: c.observation,
            consumer: c.original_consumer,
            job_authority: c.job.id.authority.into(),
            job_sequence: c.job.id.sequence,
            job_state_owner: c.job.state_identity.owner.0,
            job_state_revision: c.job.state_identity.revision.0,
            job_state_semantic: c.job.state_identity.semantic.0,
            job_question: c.job.question.0,
            root_restriction: c.job.root_restriction.0,
            evaluation_namespace: c.job.evaluation_namespace.0,
            search_conditions: c.job.search_conditions.0,
            total_credit: c.job.total_native_node_credit,
            deadline_clock: c.job.hard_deadline.clock.0.0,
            deadline_at: c.job.hard_deadline.at.0,
            owner: c.owner.0,
            profile_owner: c.profile.owner.0,
            profile_depth: c.profile.max_depth,
            profile_evaluation_namespace: c.profile.evaluation_namespace.0,
            profile_search_conditions: c.profile.search_conditions.0,
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
// Historical JSON under an aggregate byte cap; never a worker queue item.
#[allow(clippy::large_enum_variant)]
enum CpuNamespace {
    Legacy { condition: String },
    Contextual { context: CpuContext },
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
#[allow(clippy::large_enum_variant)]
enum Endpoint {
    Rules {
        line: Vec<Move16>,
        snapshot: Snapshot,
        root_value: i32,
    },
    Cpu {
        line: Vec<Move16>,
        snapshot: Snapshot,
        root_value: i32,
        depth: u16,
        plies: usize,
        identity: CpuValueIdentity,
        base_condition: String,
        profile: u64,
        condition: u64,
        observation: ObservationSource,
        task: Task,
        original_consumer: TaskConsumer,
        situation: SituationId,
        namespace: CpuNamespace,
    },
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ComparisonRecipe {
    root: Snapshot,
    before: Endpoint,
    after: Endpoint,
    superseded: Option<ObservationSource>,
    previous: Option<Box<ComparisonRecipe>>,
}
pub(super) struct ConditionalComparisonSource {
    original: Vec<u8>,
    unavailable: Unavailable,
}
impl ConditionalComparisonSource {
    pub(super) fn owned_host_bytes(&self) -> Option<u64> {
        u64::try_from(std::mem::size_of::<Self>().checked_add(self.original.capacity())?).ok()
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PcOrigin {
    authority: Authority,
    frame_sequence: u64,
    request_epoch: u64,
    request_sequence: u64,
    question: [u8; 32],
    view: [u8; 32],
    view_sequence: u64,
    full_view: [u8; 32],
    input: [u8; 32],
    history: [u8; 32],
    candidates: [u8; 32],
    mask_positions: [u8; 32],
    model: [u8; 32],
    encoding: [u8; 32],
    precision: String,
    frozen_epoch: u64,
    revision: u64,
    role: u8,
    deadline_clock: u64,
    deadline_at: u64,
    peer: Option<(u64, [u8; 32])>,
    output: [u8; 32],
    origin_seal: [u8; 32],
    namespace: [u8; 32],
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SupersededModel {
    summary: ObservationSource,
    before: ObservationSource,
    after: ObservationSource,
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
#[allow(clippy::large_enum_variant)]
enum Comparison {
    Model {
        before: ObservationSource,
        after: ObservationSource,
        origins: Box<[PcOrigin; 2]>,
        superseded: Option<SupersededModel>,
    },
    Finite {
        recipe: ComparisonRecipe,
    },
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Basis {
    schema: String,
    domain: String,
    root: Snapshot,
    authority: Authority,
    root_state: StateId,
    kind: u8,
    line: Vec<Move16>,
    source_question: [u8; 32],
    source_frame_sequence: u64,
    identity: [u8; 32],
    raw_evidence: [u8; 32],
    root_budget: [u8; 32],
    original_store_byte_limit: u64,
    observation: ObservationSource,
    comparison: Comparison,
}

struct BoundedWriter {
    bytes: Vec<u8>,
    maximum: usize,
}
impl Write for BoundedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self
            .bytes
            .len()
            .checked_add(bytes.len())
            .is_none_or(|n| n > self.maximum)
            || self.bytes.try_reserve_exact(bytes.len()).is_err()
        {
            return Err(io::Error::other("conditional basis byte limit"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn encoded<T: Serialize>(value: &T, maximum: usize) -> Result<Vec<u8>, Unavailable> {
    let mut writer = BoundedWriter {
        bytes: Vec::new(),
        maximum,
    };
    serde_json::to_writer(&mut writer, value).map_err(|_| Unavailable::Capacity)?;
    Ok(writer.bytes)
}
fn pack(line: &[BoardMove]) -> Result<Vec<Move16>, BasisError> {
    if line.len() > rz_position::MAX_UCI_REPLAY_PLIES {
        return Err(BasisError::Limit("line"));
    }
    line.iter()
        .copied()
        .map(|m| Move16::pack(m).map_err(|_| BasisError::Invalid("move")))
        .collect()
}
fn unpack(line: &[Move16]) -> Result<Vec<BoardMove>, BasisError> {
    line.iter()
        .copied()
        .map(|m| m.unpack().map_err(|_| BasisError::Invalid("move")))
        .collect()
}
fn side(color: Color) -> u8 {
    u8::from(color == Color::Black)
}
fn color(value: u8) -> Result<Color, BasisError> {
    match value {
        0 => Ok(Color::White),
        1 => Ok(Color::Black),
        _ => Err(BasisError::Invalid("perspective")),
    }
}
fn replay(root: &PositionSnapshot, line: &[Move16]) -> Result<PositionSnapshot, BasisError> {
    if line.len() > MAX_LINE {
        return Err(BasisError::Limit("comparison plies"));
    }
    let mut position = Position::from_snapshot(
        root,
        PositionLimits {
            max_history_positions: root
                .known_history_len()
                .checked_add(MAX_LINE)
                .ok_or(BasisError::Limit("history"))?,
            ..Default::default()
        },
    )
    .map_err(|_| BasisError::Invalid("Rules anchor"))?;
    for m in unpack(line)? {
        let legal = position.ordered_legal_moves();
        if position
            .play_status_from_view(&legal)
            .map_err(|_| BasisError::Invalid("Rules status"))?
            != PlayStatus::Ongoing
        {
            return Err(BasisError::Invalid("line past terminal"));
        }
        position
            .make_from_view(&legal, m)
            .map_err(|_| BasisError::Invalid("Rules move"))?;
    }
    Ok(position.snapshot())
}

impl Authority {
    fn value(&self) -> SearchAuthority {
        SearchAuthority {
            epoch: rz_contracts::ProcessEpoch(self.epoch),
            game: rz_contracts::GameGeneration(self.game),
            root: rz_contracts::RootGeneration(self.root),
            implementation: ContractDigest(self.implementation),
        }
    }
}

/// A historical projection only. None of these fields can issue an accepted
/// head, consume a Task, resume a worker, or mint a training target.
pub struct UtilityWitnessBasisMetadata {
    authority: SearchAuthority,
    root: PositionSnapshot,
    root_state: StateId,
    root_budget: ContractDigest,
    original_store_byte_limit: u64,
    source_question: ContractDigest,
    source_frame: FrameId,
    observation: ObservationId,
    kind: ActionWitnessKind,
    line: Vec<BoardMove>,
    before: UtilityWitnessEndpointMetadata,
    after: UtilityWitnessEndpointMetadata,
}
impl UtilityWitnessBasisMetadata {
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn root(&self) -> &PositionSnapshot {
        &self.root
    }
    pub fn root_state(&self) -> StateId {
        self.root_state
    }
    pub fn root_budget(&self) -> ContractDigest {
        self.root_budget
    }
    pub fn original_store_byte_limit(&self) -> u64 {
        self.original_store_byte_limit
    }
    pub fn source_question(&self) -> ContractDigest {
        self.source_question
    }
    pub fn source_frame(&self) -> FrameId {
        self.source_frame
    }
    pub fn observation(&self) -> ObservationId {
        self.observation
    }
    pub fn kind(&self) -> ActionWitnessKind {
        self.kind
    }
    pub fn line(&self) -> &[BoardMove] {
        &self.line
    }
    pub fn before(&self) -> &UtilityWitnessEndpointMetadata {
        &self.before
    }
    pub fn after(&self) -> &UtilityWitnessEndpointMetadata {
        &self.after
    }
}
pub struct UtilityWitnessEndpointMetadata {
    snapshot: PositionSnapshot,
    line: Vec<BoardMove>,
    model: Option<UtilityWitnessPcMetadata>,
    cpu: Option<UtilityWitnessCpuMetadata>,
}
impl UtilityWitnessEndpointMetadata {
    pub fn snapshot(&self) -> &PositionSnapshot {
        &self.snapshot
    }
    pub fn line(&self) -> &[BoardMove] {
        &self.line
    }
    pub fn accepted_pc(&self) -> Option<&UtilityWitnessPcMetadata> {
        self.model.as_ref()
    }
    pub fn consumed_cpu(&self) -> Option<&UtilityWitnessCpuMetadata> {
        self.cpu.as_ref()
    }
    pub fn is_rules_terminal(&self) -> bool {
        self.model.is_none() && self.cpu.is_none()
    }
}
pub struct UtilityWitnessPcMetadata {
    original: PcOrigin,
}
impl UtilityWitnessPcMetadata {
    pub fn authority(&self) -> SearchAuthority {
        self.original.authority.value()
    }
    pub fn frame(&self) -> FrameId {
        FrameId {
            authority: self.authority(),
            sequence: self.original.frame_sequence,
        }
    }
    pub fn request_id(&self) -> RequestId {
        RequestId {
            epoch: rz_contracts::ProcessEpoch(self.original.request_epoch),
            sequence: self.original.request_sequence,
        }
    }
    pub fn input(&self) -> ContractDigest {
        ContractDigest(self.original.input)
    }
    pub fn context_revision(&self) -> u64 {
        self.original.revision
    }
    pub fn question(&self) -> ContractDigest {
        ContractDigest(self.original.question)
    }
    pub fn view_digest(&self) -> ContractDigest {
        ContractDigest(self.original.view)
    }
    pub fn view_sequence(&self) -> u64 {
        self.original.view_sequence
    }
    pub fn full_view(&self) -> ContractDigest {
        ContractDigest(self.original.full_view)
    }
    pub fn output_digest(&self) -> ContractDigest {
        ContractDigest(self.original.output)
    }
    pub fn origin_seal(&self) -> ContractDigest {
        ContractDigest(self.original.origin_seal)
    }
    pub fn namespace(&self) -> ContractDigest {
        ContractDigest(self.original.namespace)
    }
    pub fn role(&self) -> Role {
        if self.original.role == Role::Proposer as u8 {
            Role::Proposer
        } else {
            Role::Critic
        }
    }
}
pub struct UtilityWitnessCpuMetadata {
    observation: ObservationId,
    execution: ExecutionId,
    consumer: TaskConsumer,
    task: TaskKey,
    context: Option<CpuContext>,
}
impl UtilityWitnessCpuMetadata {
    pub fn observation(&self) -> ObservationId {
        self.observation
    }
    pub fn execution(&self) -> ExecutionId {
        self.execution
    }
    pub fn original_consumer(&self) -> TaskConsumer {
        self.consumer
    }
    pub fn task(&self) -> &TaskKey {
        &self.task
    }
    pub fn context_authority(&self) -> Option<SearchAuthority> {
        self.context.as_ref().map(|c| c.authority.value())
    }
    pub fn context_frame(&self) -> Option<FrameId> {
        self.context.as_ref().map(|c| FrameId {
            authority: c.frame_authority.value(),
            sequence: c.frame_sequence,
        })
    }
    pub fn context_question(&self) -> Option<ContractDigest> {
        self.context.as_ref().map(|c| ContractDigest(c.question))
    }
    pub fn context_evaluation_namespace(&self) -> Option<ContractDigest> {
        self.context
            .as_ref()
            .map(|c| ContractDigest(c.evaluation_namespace))
    }
    pub fn context_search_conditions(&self) -> Option<ContractDigest> {
        self.context
            .as_ref()
            .map(|c| ContractDigest(c.search_conditions))
    }
    pub fn context_owner(&self) -> Option<CpuWorkerId> {
        self.context.as_ref().map(|c| CpuWorkerId(c.owner))
    }
    pub fn context_job_sequence(&self) -> Option<u64> {
        self.context.as_ref().map(|c| c.job_sequence)
    }
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// Conservative known backing retained for conditional history. Shared
    /// context occurrences may be charged more than once. Store consumer Vec
    /// capacities, cloned CPU context payload, source Vec capacities, and the
    /// original-consumer slots are included; map/tree/Arc internals, temporary
    /// serialization DTOs and allocator overhead remain unknown. This is not
    /// total heap, RSS, physical sharing or a memory peak.
    pub fn utility_conditional_known_retained_host_bytes(&self) -> Option<u64> {
        let mut bytes = self
            .engine
            .semantic
            .stores
            .tasks
            .historical_consumer_backing_bytes()?;
        for (_, line, source) in self.conditional_witnesses.values() {
            bytes = bytes.checked_add(
                u64::try_from(
                    line.capacity()
                        .checked_mul(std::mem::size_of::<BoardMove>())?,
                )
                .ok()?,
            )?;
            if let Some(source) = source {
                bytes = bytes.checked_add(source.owned_host_bytes()?)?;
            }
        }
        for node in &self.engine.semantic.nodes {
            if let Some(context) = node
                .evidence
                .as_ref()
                .and_then(|value| value.context.as_ref())
            {
                bytes = bytes.checked_add(context.known_owned_backing_bytes()?)?;
            }
        }
        bytes.checked_add(
            u64::try_from(
                self.cpu_admissions
                    .len()
                    .checked_mul(std::mem::size_of::<Option<TaskConsumer>>())?,
            )
            .ok()?,
        )
    }

    /// A narrow admission against the already issued history payload limit.
    /// It does not redefine the runtime's global resource policy or attest
    /// unknown process memory. Overflow/missing known backing is fail-closed.
    pub(super) fn check_utility_conditional_known_backing(
        &self,
        additional: u64,
    ) -> Result<(), PalsError> {
        let limit = u64::try_from(self.utility_continuations.witness_payload_limit())
            .map_err(|_| PalsError::Capacity)?;
        self.utility_conditional_known_retained_host_bytes()
            .and_then(|used| used.checked_add(additional))
            .filter(|total| *total <= limit)
            .map(|_| ())
            .ok_or(PalsError::Capacity)
    }

    pub(super) fn capture_utility_conditional_comparison(
        &self,
        before_line: &[BoardMove],
        before: &ConditionalEndpoint,
        after_line: &[BoardMove],
        after: &ConditionalEndpoint,
        supersedes: Option<ObservationId>,
    ) -> Result<ConditionalComparisonSource, PalsError> {
        // Source insufficiency is a historical mask; it never changes the
        // already checked publisher's semantic decision.
        let captured = (|| -> Result<Vec<u8>, Unavailable> {
            let used = usize::try_from(
                self.utility_conditional_known_retained_host_bytes()
                    .ok_or(Unavailable::Capacity)?,
            )
            .map_err(|_| Unavailable::Capacity)?;
            // The upcoming witness retains this line and a source object in
            // addition to the byte Vec. Their known backing uses the same
            // original local allowance instead of a second hidden budget.
            let additional_fixed = std::mem::size_of::<ConditionalComparisonSource>()
                .checked_add(
                    after_line
                        .len()
                        .checked_mul(std::mem::size_of::<BoardMove>())
                        .ok_or(Unavailable::Capacity)?,
                )
                .ok_or(Unavailable::Capacity)?;
            let maximum = self
                .utility_continuations
                .witness_payload_limit()
                .checked_sub(used)
                .and_then(|remaining| remaining.checked_sub(additional_fixed))
                .ok_or(Unavailable::Capacity)?;
            if maximum == 0 {
                return Err(Unavailable::Capacity);
            }
            let root =
                Snapshot::capture(&self.engine.semantic.nodes[self.root].position.snapshot())?;
            let before = self.capture_conditional_endpoint(before_line, before, maximum)?;
            let after = self.capture_conditional_endpoint(after_line, after, maximum)?;
            let (superseded, previous) = if let Some(id) = supersedes {
                let old = self
                    .conditional_witnesses
                    .get(&id)
                    .and_then(|v| v.2.as_ref())
                    .ok_or(Unavailable::MissingOriginalComparison)?;
                if old.original.is_empty() {
                    return Err(old.unavailable);
                }
                let previous = serde_json::from_slice::<ComparisonRecipe>(&old.original)
                    .map_err(|_| Unavailable::InvalidOriginalBasis)?;
                (
                    Some(ObservationSource::capture(
                        &self.engine.semantic.stores,
                        id,
                    )?),
                    Some(Box::new(previous)),
                )
            } else {
                (None, None)
            };
            let original = encoded(
                &ComparisonRecipe {
                    root,
                    before,
                    after,
                    superseded,
                    previous,
                },
                maximum.min(MAX_BYTES),
            )?;
            let additional = additional_fixed
                .checked_add(original.capacity())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or(Unavailable::Capacity)?;
            self.check_utility_conditional_known_backing(additional)
                .map_err(|_| Unavailable::Capacity)?;
            Ok(original)
        })();
        Ok(match captured {
            Ok(original) => ConditionalComparisonSource {
                original,
                unavailable: Unavailable::InvalidOriginalBasis,
            },
            Err(unavailable) => ConditionalComparisonSource {
                original: Vec::new(),
                unavailable,
            },
        })
    }

    fn capture_conditional_endpoint(
        &self,
        line: &[BoardMove],
        endpoint: &ConditionalEndpoint,
        maximum: usize,
    ) -> Result<Endpoint, Unavailable> {
        if line.len() > MAX_LINE {
            return Err(Unavailable::Capacity);
        }
        // Follow the original materialized graph; this does not issue another
        // Rules traversal, native search, or provisional comparison source.
        let mut node = self.root;
        for movement in line {
            node = self.engine.semantic.nodes[node]
                .edges
                .iter()
                .find(|edge| edge.movement == *movement)
                .map(|edge| edge.child)
                .ok_or(Unavailable::MissingResidentEvidence)?;
        }
        let actual = &self.engine.semantic.nodes[node];
        let snapshot = Snapshot::capture(&actual.position.snapshot())?;
        let packed = pack(line).map_err(|_| Unavailable::InvalidOriginalBasis)?;
        match &endpoint.basis {
            EndpointBasis::Rules => Ok(Endpoint::Rules {
                line: packed,
                snapshot,
                root_value: endpoint.root_value,
            }),
            EndpointBasis::OwnCpu {
                depth,
                plies,
                identity,
                base_condition,
                profile,
                condition,
                observation,
                execution,
            } => {
                let original = self
                    .utility_conditional_cpu_metadata(node, *observation, *execution)
                    .map_err(|_| Unavailable::InvalidOriginalBasis)?
                    .ok_or(Unavailable::IncompleteContext)?;
                let namespace = match (original.contextual, original.legacy_condition) {
                    (Some(context), None) => CpuNamespace::Contextual {
                        context: context.into(),
                    },
                    (None, Some(condition)) => CpuNamespace::Legacy { condition },
                    _ => return Err(Unavailable::IncompleteContext),
                };
                let task = self
                    .engine
                    .semantic
                    .stores
                    .tasks
                    .get(*execution)
                    .map_err(|_| Unavailable::MissingResidentEvidence)?;
                let task = serde_json::from_slice(&encoded(task, maximum)?)
                    .map_err(|_| Unavailable::InvalidOriginalBasis)?;
                Ok(Endpoint::Cpu {
                    line: packed,
                    snapshot,
                    root_value: endpoint.root_value,
                    depth: *depth,
                    plies: *plies,
                    identity: identity.clone(),
                    base_condition: base_condition.clone(),
                    profile: *profile,
                    condition: *condition,
                    observation: ObservationSource::capture(
                        &self.engine.semantic.stores,
                        *observation,
                    )?,
                    task,
                    original_consumer: original.original_consumer,
                    situation: actual.situation,
                    namespace,
                })
            }
        }
    }

    pub(super) fn freeze_utility_conditional_basis(
        &self,
        outcome: &UtilityRouteSemanticOutcome,
        witness: &UtilityRouteWitness,
        remaining: usize,
        preimage: &[u8],
    ) -> Result<UtilityConditionalWitnessBasis, Unavailable> {
        if outcome.id().authority() != self.authority
            || witness.source_frame().authority != self.authority
            || !outcome
                .witnesses()
                .iter()
                .any(|actual| std::ptr::eq(actual, witness))
        {
            return Err(Unavailable::InvalidOriginalBasis);
        }
        let maximum = remaining.min(MAX_BYTES).min(
            usize::try_from(outcome.original_budget().original_store_byte_limit)
                .unwrap_or(usize::MAX),
        );
        if maximum == 0 {
            return Err(Unavailable::Capacity);
        }
        if preimage.len() > MAX_BYTES
            || ContractDigest(Sha256::digest(preimage).into()) != witness.raw_evidence()
        {
            return Err(Unavailable::InvalidOriginalBasis);
        }
        let root = self.engine.semantic.nodes[self.root].position.snapshot();
        let observation =
            ObservationSource::capture(&self.engine.semantic.stores, witness.observation())?;
        let comparison = match &observation.score {
            Score::ConditionalWdl { before, after, .. }
            | Score::ConditionalRepairWdl { before, after, .. } => {
                let before = ObservationSource::capture(&self.engine.semantic.stores, *before)?;
                let after = ObservationSource::capture(&self.engine.semantic.stores, *after)?;
                let origins = Box::new([
                    self.capture_conditional_pc_origin(outcome, &before)?,
                    self.capture_conditional_pc_origin(outcome, &after)?,
                ]);
                let superseded = if let Some(id) = observation.supersedes {
                    let summary = ObservationSource::capture(&self.engine.semantic.stores, id)?;
                    let Score::ConditionalWdl {
                        before: old_before,
                        after: old_after,
                        ..
                    } = summary.score
                    else {
                        return Err(Unavailable::MissingOriginalComparison);
                    };
                    Some(SupersededModel {
                        before: ObservationSource::capture(
                            &self.engine.semantic.stores,
                            old_before,
                        )?,
                        after: ObservationSource::capture(&self.engine.semantic.stores, old_after)?,
                        summary,
                    })
                } else {
                    None
                };
                Comparison::Model {
                    before,
                    after,
                    origins,
                    superseded,
                }
            }
            Score::Estimate { .. } => {
                let original = self
                    .conditional_witnesses
                    .get(&witness.observation())
                    .and_then(|v| v.2.as_ref())
                    .ok_or(Unavailable::MissingOriginalComparison)?;
                if original.original.is_empty() {
                    return Err(original.unavailable);
                }
                Comparison::Finite {
                    recipe: serde_json::from_slice(&original.original)
                        .map_err(|_| Unavailable::InvalidOriginalBasis)?,
                }
            }
            _ => return Err(Unavailable::NamespaceMismatch),
        };
        let basis = Basis {
            schema: DOMAIN.into(),
            domain: DOMAIN.into(),
            root: Snapshot::capture(&root)?,
            authority: self.authority.into(),
            root_state: self.root_state_id(),
            kind: witness.kind() as u8,
            line: pack(witness.line()).map_err(|_| Unavailable::InvalidOriginalBasis)?,
            source_question: witness.source_question().0,
            source_frame_sequence: witness.source_frame().sequence,
            identity: witness.identity().0,
            raw_evidence: witness.raw_evidence().0,
            root_budget: outcome.root_budget().0,
            original_store_byte_limit: outcome.original_budget().original_store_byte_limit,
            observation,
            comparison,
        };
        let original = encoded(&basis, maximum)?;
        let retained = std::mem::size_of::<UtilityConditionalWitnessBasis>()
            .checked_add(original.capacity())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(Unavailable::Capacity)?;
        self.check_utility_conditional_known_backing(retained)
            .map_err(|_| Unavailable::Capacity)?;
        let sha256 = ContractDigest(Sha256::digest(&original).into());
        validate_utility_witness_basis(
            &original,
            UtilityWitnessBasisExpectation {
                witness: UtilityWitnessPreimageExpectation {
                    root: &root,
                    kind: witness.kind(),
                    line: witness.line(),
                    source_question: witness.source_question(),
                    source_frame_sequence: witness.source_frame().sequence,
                    identity: witness.identity(),
                    raw_evidence: witness.raw_evidence(),
                },
                observation: witness.observation(),
                basis_sha256: sha256,
                preimage,
            },
        )
        .map_err(|_| Unavailable::InvalidOriginalBasis)?;
        Ok(UtilityConditionalWitnessBasis { original, sha256 })
    }

    fn capture_conditional_pc_origin(
        &self,
        outcome: &UtilityRouteSemanticOutcome,
        observation: &ObservationSource,
    ) -> Result<PcOrigin, Unavailable> {
        let (Some(input), Some(model), Score::ContextWdl { revision, .. }) = (
            observation.input,
            observation.model.as_ref(),
            &observation.score,
        ) else {
            return Err(Unavailable::IncompleteContext);
        };
        let namespace = self
            .engine
            .semantic
            .model
            .registered_representation_namespace()
            .ok_or(Unavailable::NamespaceMismatch)?;
        let model_value = ModelValueIdentity {
            semantics: model.semantics.clone(),
            model: model.model.clone(),
            encoding: model.encoding.clone(),
            precision: model.precision.clone(),
            model_epoch: model.epoch,
        };
        let mut matches = outcome.pc_consumption().iter().filter(|pc| {
            pc.origin().ticket().representation.input.0 == input
                && pc.origin().ticket().representation.record_revision == *revision
        });
        let source = matches.next().ok_or(Unavailable::IncompleteContext)?;
        if matches.next().is_some() {
            return Err(Unavailable::IncompleteContext);
        }
        let origin = source.origin();
        let ticket = origin.ticket();
        let repr = &ticket.representation;
        namespace
            .verify_ticket(ticket, Some(&model_value))
            .map_err(|_| Unavailable::NamespaceMismatch)?;
        if ticket.authority != self.authority
            || origin.frame().authority != self.authority
            || source.question() != ticket.question
            || source.view() != ticket.view
            || repr.role != ticket.role
            || !matches!(ticket.role, Role::Proposer | Role::Critic)
        {
            return Err(Unavailable::IncompleteContext);
        }
        let precision = match repr.precision {
            rz_contracts::PrecisionProfile::Fp32 => "fp32",
            _ => return Err(Unavailable::NamespaceMismatch),
        };
        let peer = match origin.kind() {
            crate::pals::coordinator::AcceptedRoleOriginKind::LocalAccepted => None,
            crate::pals::coordinator::AcceptedRoleOriginKind::PeerQuestion(peer) => {
                Some((peer.id.sequence, peer.question_digest.0))
            }
        };
        Ok(PcOrigin {
            authority: ticket.authority.into(),
            frame_sequence: origin.frame().sequence,
            request_epoch: ticket.id.epoch.0,
            request_sequence: ticket.id.sequence,
            question: ticket.question.0,
            view: ticket.view.view_digest.0,
            view_sequence: ticket.view.view_sequence,
            full_view: source.full_view().0,
            input: repr.input.0,
            history: repr.history.0,
            candidates: repr.candidates.0,
            mask_positions: repr.mask_positions.0,
            model: repr.model.0,
            encoding: repr.encoding.0,
            precision: precision.into(),
            frozen_epoch: repr.frozen_epoch,
            revision: repr.record_revision,
            role: ticket.role as u8,
            deadline_clock: ticket.original_deadline.clock.0.0,
            deadline_at: ticket.original_deadline.at.0,
            peer,
            output: origin.output_digest().0,
            origin_seal: origin.seal_digest().0,
            namespace: namespace.utility_cohort_digest().0,
        })
    }
}

/// Validate original historical data. The returned projection owns neither a
/// Store lease nor an accepted-head/known-target capability.
pub fn validate_utility_witness_basis(
    bytes: &[u8],
    expected: UtilityWitnessBasisExpectation<'_>,
) -> Result<UtilityWitnessBasisMetadata, BasisError> {
    if bytes.len() > MAX_BYTES || expected.preimage.len() > MAX_BYTES {
        return Err(BasisError::Limit("payload bytes"));
    }
    if ContractDigest(Sha256::digest(bytes).into()) != expected.basis_sha256 {
        return Err(BasisError::Mismatch("basis SHA-256"));
    }
    let basis: Basis =
        serde_json::from_slice(bytes).map_err(|_| BasisError::Invalid("closed JSON schema"))?;
    // One byte representation also rejects unknown nested legacy fields that
    // their older serde types would otherwise ignore.
    if encoded(&basis, MAX_BYTES).map_err(|_| BasisError::Limit("canonical bytes"))? != bytes {
        return Err(BasisError::Invalid("noncanonical basis"));
    }
    let total = bytes
        .len()
        .checked_add(expected.preimage.len())
        .ok_or(BasisError::Limit("shared payload"))?;
    if total > MAX_BYTES
        || u64::try_from(total).map_err(|_| BasisError::Limit("shared payload"))?
            > basis.original_store_byte_limit
    {
        return Err(BasisError::Limit("original shared payload allowance"));
    }
    let root = basis.root.decode()?;
    let line = unpack(&basis.line)?;
    if basis.schema != DOMAIN
        || basis.domain != DOMAIN
        || !root.same_state(expected.witness.root)
        || root.revision() != expected.witness.root.revision()
        || basis.kind != expected.witness.kind as u8
        || line != expected.witness.line
        || basis.source_question != expected.witness.source_question.0
        || basis.source_frame_sequence != expected.witness.source_frame_sequence
        || basis.identity != expected.witness.identity.0
        || basis.raw_evidence != expected.witness.raw_evidence.0
        || basis.observation.id != expected.observation
        || basis.observation.state_id != basis.root_state
        || !basis.observation.state.decode()?.same_state(&root)
        || basis.authority.root == 0
    {
        return Err(BasisError::Mismatch("original witness header/root"));
    }
    let kind = expected.witness.kind;
    validate_summary(&basis.observation, kind, &root)?;
    let raw = parse_validated_utility_witness_preimage(expected.preimage, expected.witness)
        .map_err(BasisError::Raw)?;
    let (before, after) = match basis.comparison {
        Comparison::Model {
            before,
            after,
            origins,
            superseded,
        } => {
            raw_join::validate_raw_join(&basis.observation, &raw, Some(&before), Some(&after))?;
            validate_model_comparison(
                &basis.observation,
                &before,
                &after,
                &origins,
                superseded.as_ref(),
                &root,
                &basis.authority,
                kind,
                &basis.line,
            )?
        }
        Comparison::Finite { recipe } => {
            raw_join::validate_raw_join(&basis.observation, &raw, None, None)?;
            validate_recipe(
                &recipe,
                &basis.observation,
                &root,
                &basis.authority,
                basis.root_state,
                kind,
                &basis.line,
                0,
            )?
        }
    };
    Ok(UtilityWitnessBasisMetadata {
        authority: basis.authority.value(),
        root,
        root_state: basis.root_state,
        root_budget: ContractDigest(basis.root_budget),
        original_store_byte_limit: basis.original_store_byte_limit,
        source_question: ContractDigest(basis.source_question),
        source_frame: FrameId {
            authority: basis.authority.value(),
            sequence: basis.source_frame_sequence,
        },
        observation: basis.observation.id,
        kind,
        line,
        before,
        after,
    })
}

fn validate_summary(
    observation: &ObservationSource,
    kind: ActionWitnessKind,
    root: &PositionSnapshot,
) -> Result<(), BasisError> {
    let expected_kind = match kind {
        ActionWitnessKind::Counter => ObservationKind::Refutation,
        ActionWitnessKind::Repair => ObservationKind::Repair,
    };
    if observation.kind != expected_kind
        || observation
            .line
            .as_ref()
            .is_none_or(|v| v.moves.is_empty() || v.moves.len() > MAX_LINE)
        || !observation.state.decode()?.same_state(root)
        || observation.epoch != 0
        || observation.budget != 0
        || observation.value.is_some()
        || observation.checker.is_some()
        || observation.work.is_some()
        || observation.external.is_some()
        || observation.model.is_some()
        || observation.input.is_some()
        || observation.condition.is_some()
        || observation.cpu_pv.is_some()
        || observation.execution.is_some()
        || (kind == ActionWitnessKind::Counter && observation.supersedes.is_some())
        || (kind == ActionWitnessKind::Repair && observation.supersedes.is_none())
    {
        return Err(BasisError::Invalid("conditional summary metadata"));
    }
    let Scope::Model {
        encoding, input, ..
    } = observation.scope
    else {
        return Err(BasisError::Invalid("conditional summary scope"));
    };
    let (perspective, finite) = match observation.score {
        Score::Estimate { bits, side } => {
            if !f32::from_bits(bits).is_finite() {
                return Err(BasisError::Invalid("finite summary"));
            }
            (side, true)
        }
        Score::ConditionalWdl { side, revision, .. } if kind == ActionWitnessKind::Counter => {
            if input != revision {
                return Err(BasisError::Mismatch("summary context scope"));
            }
            (side, false)
        }
        Score::ConditionalRepairWdl { side, revision, .. } if kind == ActionWitnessKind::Repair => {
            if input != revision {
                return Err(BasisError::Mismatch("summary context scope"));
            }
            (side, false)
        }
        _ => return Err(BasisError::Invalid("summary score kind")),
    };
    if color(perspective)? != root.side_to_move()
        || encoding
            != stable_id(if finite {
                "pals-restricted-summary-v1"
            } else {
                "pals-frozen-pair-v3"
            })
    {
        return Err(BasisError::Mismatch("summary perspective/encoding"));
    }
    replay(
        root,
        &observation
            .line
            .as_ref()
            .ok_or(BasisError::Invalid("summary line"))?
            .moves,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_recipe(
    recipe: &ComparisonRecipe,
    summary: &ObservationSource,
    root: &PositionSnapshot,
    authority: &Authority,
    root_state: StateId,
    kind: ActionWitnessKind,
    actual_line: &[Move16],
    depth: u8,
) -> Result<
    (
        UtilityWitnessEndpointMetadata,
        UtilityWitnessEndpointMetadata,
    ),
    BasisError,
> {
    if depth >= 8 || !recipe.root.decode()?.same_state(root) {
        return Err(BasisError::Limit("original comparison ancestry"));
    }
    let (before, before_value) = validate_endpoint(&recipe.before, root, authority, root_state)?;
    let (after, after_value) = validate_endpoint(&recipe.after, root, authority, root_state)?;
    validate_line_relation(root, kind, &before.line, &after.line)?;
    let comparable = match (&recipe.before, &recipe.after) {
        (Endpoint::Rules { .. }, Endpoint::Rules { .. }) => true,
        (
            Endpoint::Cpu {
                depth: a_depth,
                plies: a_plies,
                identity: a_identity,
                base_condition: a_base,
                profile: a_profile,
                condition: a_condition,
                ..
            },
            Endpoint::Cpu {
                depth: b_depth,
                plies: b_plies,
                identity: b_identity,
                base_condition: b_base,
                profile: b_profile,
                condition: b_condition,
                ..
            },
        ) => {
            a_depth == b_depth
                && a_plies == b_plies
                && a_identity == b_identity
                && a_base == b_base
                && a_profile == b_profile
                && a_condition == b_condition
        }
        _ => false,
    };
    let Score::Estimate { bits, side } = summary.score else {
        return Err(BasisError::Invalid("finite comparison summary"));
    };
    if !comparable
        || !strict(kind, before_value, after_value)
        || bits != (after_value as f32).to_bits()
        || color(side)? != root.side_to_move()
        || after.line != unpack(actual_line)?
    {
        return Err(BasisError::Mismatch("strict finite comparison"));
    }
    match kind {
        ActionWitnessKind::Counter => {
            if recipe.superseded.is_some()
                || recipe.previous.is_some()
                || summary.supersedes.is_some()
                || summary.line.as_ref().is_none_or(|v| {
                    unpack(&v.moves).ok().as_deref() != Some(before.line.as_slice())
                })
            {
                return Err(BasisError::Mismatch("counter proposal source"));
            }
        }
        ActionWitnessKind::Repair => {
            let old = recipe
                .superseded
                .as_ref()
                .ok_or(BasisError::Invalid("original counter summary"))?;
            let previous = recipe
                .previous
                .as_ref()
                .ok_or(BasisError::Invalid("original counter comparison"))?;
            if summary.supersedes != Some(old.id) || old.state_id != root_state {
                return Err(BasisError::Mismatch("supersedes original counter"));
            }
            validate_summary(old, ActionWitnessKind::Counter, root)?;
            let previous_actual = endpoint_line(&previous.after);
            let (_, old_after) = validate_recipe(
                previous,
                old,
                root,
                authority,
                root_state,
                ActionWitnessKind::Counter,
                previous_actual,
                depth + 1,
            )?;
            if !before.snapshot.same_state(&old_after.snapshot)
                || before.line != old_after.line
                || endpoint_value(&previous.after) != before_value
                || summary.line.as_ref().is_none_or(|v| v.moves != actual_line)
            {
                return Err(BasisError::Mismatch("repair original counter endpoint"));
            }
        }
    }
    Ok((before, after))
}
fn strict<T: PartialOrd>(kind: ActionWitnessKind, before: T, after: T) -> bool {
    match kind {
        ActionWitnessKind::Counter => after < before,
        ActionWitnessKind::Repair => after > before,
    }
}
fn endpoint_line(endpoint: &Endpoint) -> &[Move16] {
    match endpoint {
        Endpoint::Rules { line, .. } | Endpoint::Cpu { line, .. } => line,
    }
}
fn endpoint_value(endpoint: &Endpoint) -> i32 {
    match endpoint {
        Endpoint::Rules { root_value, .. } | Endpoint::Cpu { root_value, .. } => *root_value,
    }
}
fn root_value(value: i32, plies: usize) -> Result<i32, BasisError> {
    if plies % 2 == 0 {
        Ok(value)
    } else {
        value
            .checked_neg()
            .ok_or(BasisError::Invalid("CPU value sign overflow"))
    }
}

fn validate_endpoint(
    endpoint: &Endpoint,
    root: &PositionSnapshot,
    authority: &Authority,
    root_state: StateId,
) -> Result<(UtilityWitnessEndpointMetadata, i32), BasisError> {
    let (line, snapshot, value) = match endpoint {
        Endpoint::Rules {
            line,
            snapshot,
            root_value,
        }
        | Endpoint::Cpu {
            line,
            snapshot,
            root_value,
            ..
        } => (line, snapshot, *root_value),
    };
    let snapshot = snapshot.decode()?;
    if !replay(root, line)?.same_state(&snapshot) {
        return Err(BasisError::Mismatch("endpoint exact Rules history"));
    }
    let moves = unpack(line)?;
    let mut metadata = UtilityWitnessEndpointMetadata {
        snapshot: snapshot.clone(),
        line: moves,
        model: None,
        cpu: None,
    };
    match endpoint {
        Endpoint::Rules { .. } => {
            let position = Position::from_snapshot(
                &snapshot,
                PositionLimits {
                    max_history_positions: snapshot.known_history_len(),
                    ..Default::default()
                },
            )
            .map_err(|_| BasisError::Invalid("Rules endpoint"))?;
            let legal = position.ordered_legal_moves();
            let PlayStatus::Terminal { winner, .. } = position
                .play_status_from_view(&legal)
                .map_err(|_| BasisError::Invalid("Rules terminal classification"))?
            else {
                return Err(BasisError::Invalid("Rules endpoint is ongoing"));
            };
            let actual = match winner {
                Some(side) if side == snapshot.side_to_move() => CPU_MATE_SCORE,
                Some(_) => -CPU_MATE_SCORE,
                None => 0,
            };
            if root_value(actual, line.len())? != value {
                return Err(BasisError::Mismatch("Rules terminal root value"));
            }
        }
        Endpoint::Cpu {
            depth,
            plies,
            identity,
            base_condition,
            profile,
            condition,
            observation,
            task,
            original_consumer,
            situation,
            namespace,
            ..
        } => {
            let position = Position::from_snapshot(
                &snapshot,
                PositionLimits {
                    max_history_positions: snapshot.known_history_len(),
                    ..Default::default()
                },
            )
            .map_err(|_| BasisError::Invalid("CPU endpoint Rules snapshot"))?;
            // Only exact automatic termination belongs to this guard. Full
            // classification also previews intended-move draw claims, which
            // would require a spare history slot unrelated to this endpoint.
            let legal = position.ordered_legal_moves();
            if position
                .play_status_from_view(&legal)
                .map_err(|_| BasisError::Invalid("CPU endpoint Rules classification"))?
                != PlayStatus::Ongoing
            {
                return Err(BasisError::Invalid("CPU endpoint is Rules terminal"));
            }
            identity
                .validate()
                .map_err(|_| BasisError::Invalid("CPU value identity"))?;
            let key = &task.key;
            if *plies != line.len()
                || *depth == 0
                || key.requested_depth == 0
                || key.node_budget == 0
                || *depth < key.requested_depth
                || base_condition.is_empty()
                || base_condition.len() > 64 * 1024
                || observation.kind != ObservationKind::CpuAnalysis
                || observation.line.is_some()
                || !observation.state.decode()?.same_state(&snapshot)
                || observation.state_id != key.state
                || observation.execution.is_none()
                || observation.value.as_ref() != Some(identity)
                || observation.checker.is_some()
                || observation.work.is_some()
                || observation.external.is_some()
                || observation.model.is_some()
                || observation.input.is_some()
                || observation.condition.as_ref() != Some(base_condition)
                || observation.supersedes.is_some()
                || observation.budget > key.node_budget
                || observation.epoch != key.epoch
                || key.question != TaskQuestion::AnalyzePosition
                || key.line.is_some()
                || !key.root_moves.is_empty()
                || key.input_revision != 0
                || key.value_identity.as_ref() != Some(identity)
                || key.checker_identity.is_some()
                || key.cpu_condition.as_ref() != Some(base_condition)
                || key.profile != *profile
                || key.condition != *condition
                || !matches!(task.status,TaskStatus::Completed(id) if id==observation.id)
                || original_consumer.situation != *situation
                || original_consumer.generation != authority.root
                || task
                    .consumers
                    .iter()
                    .filter(|c| c.consumer.id == original_consumer.id)
                    .count()
                    != 1
                || !task
                    .consumers
                    .iter()
                    .any(|c| c.consumer == *original_consumer && c.consumed && !c.cancelled)
            {
                return Err(BasisError::Mismatch("completed consumed CPU source"));
            }
            if !matches!(observation.scope,Scope::Depth{depth:d,profile:p,condition:c} if d==*depth&&p==*profile&&c==*condition)
            {
                return Err(BasisError::Mismatch("CPU depth/profile/condition"));
            }
            let Score::Cpu {
                value: raw,
                side,
                bound: BoundKind::ExactWithinSearch,
            } = observation.score
            else {
                return Err(BasisError::Invalid("completed CPU score bound"));
            };
            if color(side)? != snapshot.side_to_move()
                || root_value(raw, *plies)? != value
                || raw.unsigned_abs() > CPU_MATE_SCORE as u32
            {
                return Err(BasisError::Mismatch("CPU score/perspective"));
            }
            if let Some(pv) = &observation.cpu_pv {
                replay(&snapshot, &pv.moves)?;
            }
            let context = match namespace {
                CpuNamespace::Legacy { condition } => {
                    if condition != base_condition
                        || identity.semantics == crate::cpu_value::nnuec::v2::SEMANTICS
                    {
                        return Err(BasisError::Mismatch("original legacy CPU condition"));
                    }
                    None
                }
                CpuNamespace::Contextual { context } => {
                    let c = context;
                    if c.generation != authority.root
                        || c.root != root_state
                        || c.authority != *authority
                        || c.frame_authority != *authority
                        || c.job_authority != *authority
                        || c.frame_sequence == 0
                        || c.job_sequence == 0
                        || c.task != *key
                        || Some(c.execution) != observation.execution
                        || c.observation != observation.id
                        || c.consumer != *original_consumer
                        || c.question != c.job_question
                        || c.owner != c.profile_owner
                        || c.profile_depth < key.requested_depth
                        || c.evaluation_namespace != c.profile_evaluation_namespace
                        || c.search_conditions != c.profile_search_conditions
                        || c.total_credit == 0
                        || key.node_budget > c.total_credit
                        || c.deadline_clock != authority.epoch
                        || c.deadline_at == 0
                        || c.job_state_owner == 0
                        || c.job_state_semantic
                            != rz_position::contracts::rules_snapshot_semantic_digest(&snapshot).0
                    {
                        return Err(BasisError::Mismatch("original contextual CPU source"));
                    }
                    let suffix = format!(";nnuec-context-binding={}", hex(&c.evaluation_namespace));
                    if !base_condition.ends_with(&suffix)
                        || identity.semantics != crate::cpu_value::nnuec::v2::SEMANTICS
                    {
                        return Err(BasisError::Mismatch("contextual evaluation namespace"));
                    }
                    Some(c.clone())
                }
            };
            metadata.cpu = Some(UtilityWitnessCpuMetadata {
                observation: observation.id,
                execution: observation
                    .execution
                    .ok_or(BasisError::Invalid("CPU execution"))?,
                consumer: *original_consumer,
                task: key.clone(),
                context,
            });
        }
    }
    Ok((metadata, value))
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[allow(clippy::too_many_arguments)]
fn validate_model_comparison(
    summary: &ObservationSource,
    before: &ObservationSource,
    after: &ObservationSource,
    origins: &[PcOrigin; 2],
    superseded: Option<&SupersededModel>,
    root: &PositionSnapshot,
    authority: &Authority,
    kind: ActionWitnessKind,
    actual_line: &[Move16],
) -> Result<
    (
        UtilityWitnessEndpointMetadata,
        UtilityWitnessEndpointMetadata,
    ),
    BasisError,
> {
    let (before_snapshot, before_value, before_revision, before_model) =
        validate_model_endpoint(before, root)?;
    let (after_snapshot, after_value, after_revision, after_model) =
        validate_model_endpoint(after, root)?;
    let before_line = suffix(root, &before_snapshot)?;
    let after_line = suffix(root, &after_snapshot)?;
    validate_line_relation(root, kind, &before_line, &after_line)?;
    let (summary_before, summary_after, summary_revision, summary_bits, summary_side) =
        match summary.score {
            Score::ConditionalWdl {
                before,
                after,
                revision,
                bits,
                side,
            } if kind == ActionWitnessKind::Counter => (before, after, revision, bits, side),
            Score::ConditionalRepairWdl {
                before,
                after,
                revision,
                bits,
                side,
            } if kind == ActionWitnessKind::Repair => (before, after, revision, bits, side),
            _ => return Err(BasisError::Invalid("original ContextWDL summary")),
        };
    if before.id == after.id
        || summary_before != before.id
        || summary_after != after.id
        || before_model != after_model
        || before_revision != after_revision
        || summary_revision != before_revision
        || summary_bits != after_value.to_bits()
        || color(summary_side)? != root.side_to_move()
        || !strict(kind, before_value, after_value)
        || after_line != unpack(actual_line)?
        || !matches!(summary.scope,Scope::Model{model,encoding,input} if model==stable_id(&before_model.model)&&encoding==stable_id("pals-frozen-pair-v3")&&input==before_revision)
    {
        return Err(BasisError::Mismatch("strict frozen ContextWDL comparison"));
    }
    for (origin, observation) in origins.iter().zip([before, after]) {
        validate_pc_origin(origin, observation, authority)?;
    }
    if (origins[0].request_epoch, origins[0].request_sequence)
        == (origins[1].request_epoch, origins[1].request_sequence)
        || origins[0].namespace != origins[1].namespace
        || origins[0].model != origins[1].model
        || origins[0].encoding != origins[1].encoding
        || origins[0].frozen_epoch != origins[1].frozen_epoch
        || origins[0].input == origins[1].input
    {
        return Err(BasisError::Mismatch(
            "distinct accepted frozen endpoint inputs",
        ));
    }
    match kind {
        ActionWitnessKind::Counter => {
            if superseded.is_some()
                || summary.supersedes.is_some()
                || summary.line.as_ref().is_none_or(|line| {
                    unpack(&line.moves).ok().as_deref() != Some(before_line.as_slice())
                })
            {
                return Err(BasisError::Mismatch("original model proposal"));
            }
        }
        ActionWitnessKind::Repair => {
            let old = superseded.ok_or(BasisError::Invalid("original model counter"))?;
            validate_summary(&old.summary, ActionWitnessKind::Counter, root)?;
            let (old_before_snapshot, old_before, old_before_revision, old_before_model) =
                validate_model_endpoint(&old.before, root)?;
            let (old_after_snapshot, old_after, old_after_revision, old_after_model) =
                validate_model_endpoint(&old.after, root)?;
            validate_line_relation(
                root,
                ActionWitnessKind::Counter,
                &suffix(root, &old_before_snapshot)?,
                &suffix(root, &old_after_snapshot)?,
            )?;
            let Score::ConditionalWdl {
                before: old_before_id,
                after: old_after_id,
                revision,
                bits,
                side,
            } = old.summary.score
            else {
                return Err(BasisError::Invalid("superseded ContextWDL counter"));
            };
            if summary.supersedes != Some(old.summary.id)
                || old.summary.state_id != summary.state_id
                || old_before_id != old.before.id
                || old_after_id != old.after.id
                || old.before.id == old.after.id
                || old_before_model != old_after_model
                || old_before_model != before_model
                || old_before_revision != old_after_revision
                || revision != old_before_revision
                || bits != old_after.to_bits()
                || color(side)? != root.side_to_move()
                || old_after >= old_before
                || !old_after_snapshot.same_state(&before_snapshot)
                || summary
                    .line
                    .as_ref()
                    .is_none_or(|line| line.moves != actual_line)
                || old.summary.line.as_ref().is_none_or(|line| {
                    suffix(root, &old_before_snapshot).ok().as_deref()
                        != unpack(&line.moves).ok().as_deref()
                })
            {
                return Err(BasisError::Mismatch(
                    "repair exact superseded model comparison",
                ));
            }
        }
    }
    Ok((
        UtilityWitnessEndpointMetadata {
            snapshot: before_snapshot,
            line: before_line,
            model: Some(UtilityWitnessPcMetadata {
                original: origins[0].clone(),
            }),
            cpu: None,
        },
        UtilityWitnessEndpointMetadata {
            snapshot: after_snapshot,
            line: after_line,
            model: Some(UtilityWitnessPcMetadata {
                original: origins[1].clone(),
            }),
            cpu: None,
        },
    ))
}

fn validate_model_endpoint<'a>(
    observation: &'a ObservationSource,
    root: &PositionSnapshot,
) -> Result<(PositionSnapshot, f32, u64, &'a ModelIdentity), BasisError> {
    let snapshot = observation.state.decode()?;
    let model = observation
        .model
        .as_ref()
        .ok_or(BasisError::Invalid("actual model identity"))?;
    model.validate()?;
    let input = observation
        .input
        .ok_or(BasisError::Invalid("actual model input binding"))?;
    let Score::ContextWdl {
        bits,
        side,
        revision,
    } = observation.score
    else {
        return Err(BasisError::Invalid("actual ContextWDL endpoint"));
    };
    let side = color(side)?;
    let wdl = bits.map(f32::from_bits);
    if wdl
        .iter()
        .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        || (wdl.iter().sum::<f32>() - 1.0).abs() > 1e-4
        || side != snapshot.side_to_move()
        || observation.line.is_some()
        || observation.kind != ObservationKind::Proposal
        || observation.epoch != 0
        || observation.budget != 1
        || observation.value.is_some()
        || observation.checker.is_some()
        || observation.work.is_some()
        || observation.external.is_some()
        || observation.condition.is_some()
        || observation.cpu_pv.is_some()
        || observation.supersedes.is_some()
        || observation.execution.is_some()
        || observation.source != stable_id(&model.semantics)
        || !matches!(observation.scope,Scope::Model{model:m,encoding,input:i} if m==stable_id(&model.model)&&encoding==stable_id(&model.encoding)&&i==u64::from_le_bytes(input[..8].try_into().map_err(|_|BasisError::Invalid("input prefix"))?))
    {
        return Err(BasisError::Mismatch("original model endpoint metadata"));
    }
    suffix(root, &snapshot)?;
    let expectation = if side == root.side_to_move() {
        wdl[0] - wdl[2]
    } else {
        wdl[2] - wdl[0]
    };
    Ok((snapshot, expectation, revision, model))
}

fn suffix(
    root: &PositionSnapshot,
    endpoint: &PositionSnapshot,
) -> Result<Vec<BoardMove>, BasisError> {
    let start = root
        .uci_replay(rz_position::MAX_UCI_REPLAY_PLIES)
        .map_err(|_| BasisError::Invalid("root trace"))?;
    let end = endpoint
        .uci_replay(rz_position::MAX_UCI_REPLAY_PLIES)
        .map_err(|_| BasisError::Invalid("endpoint trace"))?;
    if start.origin != end.origin
        || start.completeness != end.completeness
        || start.start_fen != end.start_fen
        || !end.moves.starts_with(&start.moves)
    {
        return Err(BasisError::Mismatch("endpoint root history prefix"));
    }
    let suffix = end.moves[start.moves.len()..].to_vec();
    if !replay(root, &pack(&suffix)?)?.same_state(endpoint) {
        return Err(BasisError::Mismatch("endpoint root Rules replay"));
    }
    Ok(suffix)
}

fn validate_line_relation(
    root: &PositionSnapshot,
    kind: ActionWitnessKind,
    before: &[BoardMove],
    after: &[BoardMove],
) -> Result<(), BasisError> {
    if before.is_empty() || after.is_empty() || before.first() != after.first() || before == after {
        return Err(BasisError::Mismatch("original continuation first move"));
    }
    let differing = before.iter().zip(after).position(|(a, b)| a != b);
    let divergence = match (kind, differing) {
        (ActionWitnessKind::Counter, None) => {
            return Err(BasisError::Invalid("counter has no opponent divergence"));
        }
        (_, Some(index)) => index,
        (ActionWitnessKind::Repair, None) => before.len().min(after.len()),
    };
    let anchor = replay(root, &pack(&before[..divergence])?)?;
    let own = anchor.side_to_move() == root.side_to_move();
    if own != (kind == ActionWitnessKind::Repair) {
        return Err(BasisError::Mismatch("conditional divergence side"));
    }
    Ok(())
}

fn validate_pc_origin(
    origin: &PcOrigin,
    observation: &ObservationSource,
    authority: &Authority,
) -> Result<(), BasisError> {
    let model = observation
        .model
        .as_ref()
        .ok_or(BasisError::Invalid("origin model namespace"))?;
    let Score::ContextWdl { revision, .. } = observation.score else {
        return Err(BasisError::Invalid("origin ContextWDL revision"));
    };
    if origin.authority != *authority
        || origin.frame_sequence == 0
        || origin.request_epoch != authority.epoch
        || origin.request_sequence == 0
        || origin.view_sequence == 0
        || origin.deadline_clock != authority.epoch
        || origin.deadline_at == 0
        || Some(origin.input) != observation.input
        || origin.revision != revision
        || origin.precision != "fp32"
        || !matches!(model.precision.as_str(), "fp32" | "fp32_tf32_off")
        || (origin.role != Role::Proposer as u8 && origin.role != Role::Critic as u8)
    {
        return Err(BasisError::Mismatch("accepted endpoint origin identity"));
    }
    let mut namespace = Sha256::new();
    namespace.update(b"rz-rovent-utility-cohort-model-namespace/1\0");
    for text in [
        &model.semantics,
        &model.model,
        &model.encoding,
        &model.precision,
    ] {
        namespace.update((text.len() as u64).to_le_bytes());
        namespace.update(text.as_bytes());
    }
    namespace.update(model.epoch);
    namespace.update(origin.model);
    namespace.update(origin.encoding);
    namespace.update([0]);
    namespace.update(origin.frozen_epoch.to_le_bytes());
    if <[u8; 32]>::from(namespace.finalize()) != origin.namespace {
        return Err(BasisError::Mismatch(
            "registered model representation namespace",
        ));
    }
    // Historical-only recheck of the canonical owner's framing at
    // pals/coordinator/native_acceptance.rs::AcceptedRoleOrigin::seal_digest.
    // This hash comparison constructs no native acceptance capability.
    let mut seal = Sha256::new();
    seal.update(b"rz-rovent-accepted-local-origin-seal/1\0");
    seal.update(origin.frame_sequence.to_le_bytes());
    seal.update(origin.request_sequence.to_le_bytes());
    seal.update(origin.authority.epoch.to_le_bytes());
    seal.update(origin.authority.game.to_le_bytes());
    seal.update(origin.authority.root.to_le_bytes());
    seal.update(origin.authority.implementation);
    seal.update(origin.question);
    seal.update(origin.view);
    seal.update(origin.view_sequence.to_le_bytes());
    seal.update(origin.output);
    for digest in [
        origin.input,
        origin.history,
        origin.candidates,
        origin.mask_positions,
        origin.model,
        origin.encoding,
    ] {
        seal.update(digest);
    }
    seal.update(origin.frozen_epoch.to_le_bytes());
    seal.update(origin.revision.to_le_bytes());
    seal.update([origin.role]);
    seal.update([0]);
    seal.update(origin.deadline_at.to_le_bytes());
    match origin.peer {
        None => seal.update([0]),
        Some((sequence, question)) => {
            if sequence == 0 {
                return Err(BasisError::Invalid("peer origin sequence"));
            }
            seal.update([1]);
            seal.update(sequence.to_le_bytes());
            seal.update(question);
        }
    }
    if <[u8; 32]>::from(seal.finalize()) != origin.origin_seal {
        return Err(BasisError::Mismatch("actual native origin seal"));
    }
    Ok(())
}
