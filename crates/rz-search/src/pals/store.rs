//! Bounded, single-writer PALS situation and evidence storage.
//!
//! Exact Rules history and immutable completed observations outlive derived model
//! caches. A refutation concerns one continuation under its recorded conditions;
//! it never permanently marks the continuation's first move as losing. These
//! stores own no CPU/GPU execution lease: cancelling a consumer does not assert
//! that its physical work has finished.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex, Weak};

use crate::cpu_checker::{
    CheckerIdentity, CheckerWork, ExternalBound, ExternalCheckerReport, ExternalCompletion,
    ExternalRawScore,
};
use crate::cpu_value::CpuValueIdentity;
use rz_position::{
    BoardMove, Color, PieceKind, PlayStatus, Position, PositionSnapshot, RepetitionIdentity, Square,
};

mod archive;
mod hot;
pub mod hot_commit;
pub use archive::{
    ArchiveConfig, ArchiveIoBudget, ArchiveLoadPin, ArchiveOwnerLifecycle, ArchiveOwnerSnapshot,
    ArchiveReceipt, ArchiveRecordKind, ArchiveRootCapability, ArchiveRuntimeLimits, ArchiveStats,
    ColdHandle, EngineArchiveNode, EngineArchiveReceipt, ForeignAppendPermit, ForeignGoIssuedProof,
    LoadedArchive, ProfiledForeignAppendPermit, ProfiledForeignGoIssuedProof, StorePins,
    archive_accounting_source_sha256,
};
use hot::HotRecords;

const REVIEW_PIN_CAPACITY: usize = 16;

macro_rules! index_id {
    ($name:ident) => {
        #[derive(
            Clone,
            Copy,
            Debug,
            Eq,
            PartialEq,
            Ord,
            PartialOrd,
            Hash,
            serde::Serialize,
            serde::Deserialize,
        )]
        pub struct $name(pub usize);
    };
}
index_id!(StateId);
index_id!(LineId);
index_id!(ObservationId);
index_id!(ExecutionId);

/// Reusing a slot issues another generation, so stale IDs cannot name new work.
#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct SituationId {
    pub slot: usize,
    pub generation: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StoreError {
    Capacity(&'static str),
    InvalidHandle(&'static str),
    InvalidMove,
    InvalidEvidence(&'static str),
    InvalidConditions(&'static str),
    ExpiredConsumer,
    StaleConsumer,
    CancelledConsumer,
    AlreadyConsumed,
    NotCompleted,
    RevisionExhausted,
    ArchiveDisabled,
    ArchiveBudgetRequired,
    ArchiveQuota(&'static str),
    ArchiveIo {
        stage: &'static str,
        kind: std::io::ErrorKind,
    },
    ArchiveIntegrity(&'static str),
    ArchiveCanceled,
    ArchiveDeadline,
    ArchiveByteBudget,
    /// The cold uniqueness proof belongs to an older store/archive fact set.
    /// The caller must report unknown and request a new bounded worker scan.
    ForeignPermitStale,
    PinSaturated(&'static str),
    /// Admission is waiting for existing selected-view owners to retire. This
    /// is neither archive evidence deletion nor permission to refund work.
    ReviewLeaseBackpressure {
        current: usize,
        capacity: usize,
        required_new: usize,
        replacement_reserve: usize,
    },
    ColdRecord(&'static str),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PALS store: {self:?}")
    }
}
impl std::error::Error for StoreError {}

/// Counts include retained completed records. Capacity exhaustion is explicit;
/// completed evidence is never silently evicted to admit another branch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StoreLimits {
    pub states: usize,
    pub retained_history_bytes: usize,
    pub line_chunks: usize,
    pub line_plies: usize,
    pub situations: usize,
    pub conclusions_per_situation: usize,
    pub observations: usize,
    pub dependency_edges: usize,
    pub executions: usize,
    pub consumers: usize,
    pub root_moves_per_task: usize,
    pub derived_entries: usize,
    pub derived_bytes: usize,
}

impl Default for StoreLimits {
    fn default() -> Self {
        Self {
            states: 8_192,
            retained_history_bytes: 64 * 1024 * 1024,
            line_chunks: 32_768,
            line_plies: 256,
            situations: 8_192,
            conclusions_per_situation: 256,
            observations: 65_536,
            dependency_edges: 131_072,
            executions: 32_768,
            consumers: 65_536,
            root_moves_per_task: 256,
            derived_entries: 4_096,
            derived_bytes: 8 * 1024 * 1024,
        }
    }
}

/// Coordinate move packing only; Rules still owns legality, castling and EP.
#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct Move16(u16);

impl Move16 {
    pub fn pack(mv: BoardMove) -> Result<Self, StoreError> {
        if mv.from == mv.to {
            return Err(StoreError::InvalidMove);
        }
        let promotion = match mv.promotion {
            None => 0,
            Some(PieceKind::Queen) => 1,
            Some(PieceKind::Rook) => 2,
            Some(PieceKind::Bishop) => 3,
            Some(PieceKind::Knight) => 4,
            _ => return Err(StoreError::InvalidMove),
        };
        Ok(Self(
            mv.from.index() as u16 | ((mv.to.index() as u16) << 6) | (promotion << 12),
        ))
    }

    pub fn unpack(self) -> Result<BoardMove, StoreError> {
        if self.0 & 0x8000 != 0 {
            return Err(StoreError::InvalidMove);
        }
        let promotion = match (self.0 >> 12) & 7 {
            0 => None,
            1 => Some(PieceKind::Queen),
            2 => Some(PieceKind::Rook),
            3 => Some(PieceKind::Bishop),
            4 => Some(PieceKind::Knight),
            _ => return Err(StoreError::InvalidMove),
        };
        BoardMove::new(
            Square::new((self.0 & 63) as u8).map_err(|_| StoreError::InvalidMove)?,
            Square::new(((self.0 >> 6) & 63) as u8).map_err(|_| StoreError::InvalidMove)?,
            promotion,
        )
        .map_err(|_| StoreError::InvalidMove)
    }

    pub fn bits(self) -> u16 {
        self.0
    }
}

#[derive(Debug)]
pub struct StateStore {
    identity: Arc<()>,
    snapshots: HotRecords<PositionSnapshot>,
    index: HashMap<RepetitionIdentity, Vec<StateId>>,
    limit: usize,
    history_limit: usize,
    history_bytes: usize,
}

impl StateStore {
    pub fn new(limit: usize, retained_history_bytes: usize) -> Self {
        Self {
            identity: Arc::new(()),
            snapshots: HotRecords::new(),
            index: HashMap::new(),
            limit,
            history_limit: retained_history_bytes,
            history_bytes: 0,
        }
    }

    /// The repetition key only selects a bucket. Exact history equality decides
    /// reuse; same FEN with missing/different history receives another StateId.
    pub fn find(&self, snapshot: &PositionSnapshot) -> Option<StateId> {
        self.index
            .get(&snapshot.repetition_identity())?
            .iter()
            .copied()
            .find(|id| self.snapshots[id.0].same_state(snapshot))
    }

    pub fn insert(&mut self, snapshot: PositionSnapshot) -> Result<StateId, StoreError> {
        if let Some(id) = self.find(&snapshot) {
            return Ok(id);
        }
        if self.snapshots.len() >= self.limit {
            return Err(StoreError::Capacity("states"));
        }
        let charge = snapshot
            .retained_history_bytes()
            .ok_or(StoreError::Capacity("state history bytes"))?;
        let new_bytes = self
            .history_bytes
            .checked_add(charge)
            .filter(|bytes| *bytes <= self.history_limit)
            .ok_or(StoreError::Capacity("state history bytes"))?;
        let key = snapshot.repetition_identity();
        self.index
            .try_reserve(1)
            .map_err(|_| StoreError::Capacity("state index allocation"))?;
        self.snapshots.reserve(1)?;
        self.index
            .entry(key.clone())
            .or_default()
            .try_reserve(1)
            .map_err(|_| StoreError::Capacity("state bucket allocation"))?;
        let id = StateId(self.snapshots.next_id());
        self.snapshots.push(snapshot)?;
        self.index.entry(key).or_default().push(id);
        self.history_bytes = new_bytes;
        Ok(id)
    }

    pub fn get(&self, id: StateId) -> Result<&PositionSnapshot, StoreError> {
        self.snapshots.get(id.0).ok_or_else(|| {
            if id.0 < self.snapshots.next_id() {
                StoreError::ColdRecord("state")
            } else {
                StoreError::InvalidHandle("state")
            }
        })
    }

    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    pub fn retained_history_bytes(&self) -> usize {
        self.history_bytes
    }
}

pub const MOVES_PER_CHUNK: usize = 16;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LineChunk {
    pub parent: Option<LineId>,
    pub start_state: StateId,
    moves: [Move16; MOVES_PER_CHUNK],
    len: usize,
    plies: usize,
}

impl LineChunk {
    pub fn chunk_moves(&self) -> &[Move16] {
        &self.moves[..self.len]
    }

    pub fn plies(&self) -> usize {
        self.plies
    }
}

#[derive(Debug)]
pub struct LinePool {
    /// A weak attestation retains this allocation, so replacing a public pool
    /// cannot reissue its checked numeric handles under a new owner.
    identity: Arc<()>,
    chunks: HotRecords<LineChunk>,
    roots: BTreeMap<StateId, LineId>,
    index: BTreeMap<(LineId, Vec<Move16>), LineId>,
    limit: usize,
    ply_limit: usize,
}

impl LinePool {
    pub fn new(limit: usize, ply_limit: usize) -> Self {
        Self {
            identity: Arc::new(()),
            chunks: HotRecords::new(),
            roots: BTreeMap::new(),
            index: BTreeMap::new(),
            limit,
            ply_limit,
        }
    }

    pub fn root(&mut self, start_state: StateId) -> Result<LineId, StoreError> {
        if let Some(id) = self.roots.get(&start_state) {
            return Ok(*id);
        }
        if self.chunks.len() >= self.limit {
            return Err(StoreError::Capacity("line chunks"));
        }
        let id = LineId(self.chunks.next_id());
        self.chunks.push(LineChunk {
            parent: None,
            start_state,
            moves: [Move16(0); MOVES_PER_CHUNK],
            len: 0,
            plies: 0,
        })?;
        self.roots.insert(start_state, id);
        Ok(id)
    }

    /// Each append retains its immutable parent and adds chunks of <=16 moves.
    /// Repeating the same suffix from the same prefix reuses existing chunks.
    /// Preflight prevents a capacity error from inserting a partial suffix.
    pub fn append(&mut self, prefix: LineId, moves: &[BoardMove]) -> Result<LineId, StoreError> {
        let head = self.get(prefix)?;
        let plies = head
            .plies
            .checked_add(moves.len())
            .filter(|plies| *plies <= self.ply_limit)
            .ok_or(StoreError::Capacity("line plies"))?;
        if moves.is_empty() {
            return Ok(prefix);
        }
        let packed = moves
            .iter()
            .copied()
            .map(Move16::pack)
            .collect::<Result<Vec<_>, _>>()?;
        let mut existing = prefix;
        let mut missing = 0;
        let mut diverged = false;
        for chunk in packed.chunks(MOVES_PER_CHUNK) {
            let hit = if diverged {
                None
            } else {
                self.index.get(&(existing, chunk.to_vec()))
            };
            if let Some(id) = hit {
                existing = *id;
            } else {
                diverged = true;
                missing += 1;
            }
        }
        if self.chunks.len().saturating_add(missing) > self.limit {
            return Err(StoreError::Capacity("line chunks"));
        }
        self.chunks.reserve(missing)?;
        let mut parent = prefix;
        let start_state = self.get(prefix)?.start_state;
        let mut accumulated = self.get(prefix)?.plies;
        for chunk in packed.chunks(MOVES_PER_CHUNK) {
            let key = (parent, chunk.to_vec());
            if let Some(id) = self.index.get(&key) {
                parent = *id;
                accumulated = self.chunks[parent.0].plies;
                continue;
            }
            let mut payload = [Move16(0); MOVES_PER_CHUNK];
            payload[..chunk.len()].copy_from_slice(chunk);
            accumulated += chunk.len();
            let id = LineId(self.chunks.next_id());
            self.chunks.push(LineChunk {
                parent: Some(parent),
                start_state,
                moves: payload,
                len: chunk.len(),
                plies: accumulated,
            })?;
            self.index.insert(key, id);
            parent = id;
        }
        debug_assert_eq!(self.get(parent)?.plies, plies);
        Ok(parent)
    }

    pub fn get(&self, id: LineId) -> Result<&LineChunk, StoreError> {
        self.chunks.get(id.0).ok_or_else(|| {
            if id.0 < self.chunks.next_id() {
                StoreError::ColdRecord("line")
            } else {
                StoreError::InvalidHandle("line")
            }
        })
    }

    pub fn moves(&self, id: LineId) -> Result<Vec<BoardMove>, StoreError> {
        let mut chain = Vec::new();
        let mut at = Some(id);
        while let Some(id) = at {
            let node = self.get(id)?;
            chain.push(id);
            at = node.parent;
        }
        let mut result = Vec::with_capacity(self.get(id)?.plies);
        for id in chain.into_iter().rev() {
            for mv in self.get(id)?.chunk_moves() {
                result.push(mv.unpack()?);
            }
        }
        Ok(result)
    }

    pub fn first_move(&self, id: LineId) -> Result<Option<BoardMove>, StoreError> {
        let mut at = id;
        let mut first = None;
        loop {
            let node = self.get(at)?;
            if let Some(mv) = node.chunk_moves().first() {
                first = Some(mv.unpack()?);
            }
            match node.parent {
                Some(parent) => at = parent,
                None => return Ok(first),
            }
        }
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum BoundKind {
    ExactWithinSearch,
    LowerWithinSearch,
    UpperWithinSearch,
}

/// Store-owned external constraint namespace. Missing limits stay absent; these
/// reported UCI limits never reserve or certify native search credit.
#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, serde::Serialize, serde::Deserialize,
)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProfiledExternalConstraint {
    Combined { depth: u16, reported_nodes: u64 },
    Nodes { reported_nodes: u64 },
    Depth { depth: u16 },
    MoveTime { milliseconds: u64 },
}

impl From<crate::external_cpu::ExternalGoConstraint> for ProfiledExternalConstraint {
    fn from(value: crate::external_cpu::ExternalGoConstraint) -> Self {
        use crate::external_cpu::ExternalGoConstraint as Source;
        match value {
            Source::Combined {
                depth,
                reported_nodes,
            } => Self::Combined {
                depth,
                reported_nodes,
            },
            Source::Nodes { reported_nodes } => Self::Nodes { reported_nodes },
            Source::Depth { depth } => Self::Depth { depth },
            Source::MoveTime { milliseconds } => Self::MoveTime { milliseconds },
        }
    }
}

impl ProfiledExternalConstraint {
    pub fn requested_depth(self) -> Option<u16> {
        match self {
            Self::Combined { depth, .. } | Self::Depth { depth } => Some(depth),
            Self::Nodes { .. } | Self::MoveTime { .. } => None,
        }
    }

    pub fn requested_nodes(self) -> Option<u64> {
        match self {
            Self::Combined { reported_nodes, .. } | Self::Nodes { reported_nodes } => {
                Some(reported_nodes)
            }
            Self::Depth { .. } | Self::MoveTime { .. } => None,
        }
    }

    pub fn validate(self) -> Result<(), StoreError> {
        if self
            .requested_depth()
            .is_some_and(|depth| depth == 0 || depth > 64)
            || self.requested_nodes() == Some(0)
            || matches!(
                self,
                Self::MoveTime {
                    milliseconds: 0 | 180_001..
                }
            )
        {
            return Err(StoreError::InvalidConditions(
                "profiled external constraint bounds",
            ));
        }
        Ok(())
    }

    /// Only movetime may be shortened by the original physical wall allowance.
    pub fn validate_issued(self, issued: Self) -> Result<(), StoreError> {
        self.validate()?;
        issued.validate()?;
        if self == issued
            || matches!((self, issued),
            (Self::MoveTime { milliseconds: requested }, Self::MoveTime { milliseconds: actual })
                if actual <= requested)
        {
            Ok(())
        } else {
            Err(StoreError::InvalidEvidence(
                "profiled issued constraint differs",
            ))
        }
    }
}

/// A finite CPU search result is scoped by depth/profile/conditions. It does not
/// constitute a proof of the complete chess game, even when its TT says exact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceScope {
    RulesTerminal,
    DepthLimited {
        depth: u16,
        profile: u64,
        condition: u64,
    },
    /// Foreign UCI output preserves its own depth and bound vocabulary. A
    /// bestmove response is not an own completed iteration or a chess proof.
    ExternalUci {
        requested_depth: u16,
        reported_depth: Option<u16>,
        seldepth: Option<u16>,
        bound: ExternalBound,
    },
    Model {
        model: u64,
        encoding: u64,
        input: u64,
    },
    /// A distinct raw UCI namespace, including queries with no depth limit.
    /// It is never an own completed depth, native credit or a Rules proof.
    ProfiledExternalUci {
        requested: ProfiledExternalConstraint,
        issued: ProfiledExternalConstraint,
        reported_depth: Option<u16>,
        seldepth: Option<u16>,
        bound: ExternalBound,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RawScore {
    Unknown,
    Estimate {
        value: f32,
        perspective: Color,
    },
    Cpu {
        value: i32,
        perspective: Color,
        bound: BoundKind,
    },
    ExternalUci {
        value: ExternalRawScore,
        bound: ExternalBound,
        perspective: Color,
        wdl_per_mille: Option<[u16; 3]>,
    },
    Wdl {
        win: f32,
        draw: f32,
        loss: f32,
        perspective: Color,
    },
    /// Actual prepared model output under one immutable public PALS context.
    /// Legacy Wdl retains its original unversioned raw-output contract.
    ContextWdl {
        win: f32,
        draw: f32,
        loss: f32,
        perspective: Color,
        context_revision: u64,
    },
    /// Restricted continuation expectation derived from two immutable raw model
    /// observations. This does not claim a new prepared model input or WDL.
    ConditionalWdl {
        expectation: f32,
        perspective: Color,
        repaired: ObservationId,
        counter: ObservationId,
        context_revision: u64,
    },
    /// Strict improvement over the re-evaluated counter under the same public
    /// context. Supersedes links the separate active refutation observation.
    ConditionalRepairWdl {
        expectation: f32,
        perspective: Color,
        before: ObservationId,
        after: ObservationId,
        context_revision: u64,
    },
    Terminal {
        winner: Option<Color>,
    },
}

impl RawScore {
    fn model_dependencies(self) -> Option<[ObservationId; 2]> {
        match self {
            Self::ConditionalWdl {
                repaired: before,
                counter: after,
                ..
            }
            | Self::ConditionalRepairWdl { before, after, .. } => Some([before, after]),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ObservationKind {
    Proposal,
    Refutation,
    Repair,
    CpuAnalysis,
    ExternalCpuAnalysis,
    ExactTerminal,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Observation {
    pub state: StateId,
    pub line: Option<LineId>,
    pub source: u64,
    pub epoch: u64,
    /// Exact immutable CPU evaluator namespace. It is independent of neural
    /// model/epoch handles and preserves real weight absence or checkpoint ID.
    pub value_identity: Option<CpuValueIdentity>,
    pub checker_identity: Option<CheckerIdentity>,
    /// Optional actual counters; unknown work stays None, never an invented 0.
    pub checker_work: Option<CheckerWork>,
    /// Full unprojected foreign report. `cpu_pv` retains only its bounded,
    /// Rules-checked line projection, while this preserves raw units/metadata.
    pub external_report: Option<Box<ExternalCheckerReport>>,
    pub model_value_identity: Option<super::value::ModelValueIdentity>,
    pub model_value_input: Option<[u8; 32]>,
    /// Exact canonical search implementation/configuration/capability and task
    /// conditions, including ordered restrictions. Numeric IDs are metadata.
    pub cpu_condition: Option<String>,
    /// An immutable, sequentially Rules-checked CPU PV prefix from `state`.
    /// The prefix and finite CPU estimate never certify a terminal outcome.
    pub cpu_pv: Option<LineId>,
    pub scope: EvidenceScope,
    pub score: RawScore,
    pub budget: u64,
    pub kind: ObservationKind,
    pub supersedes: Option<ObservationId>,
    pub execution: Option<ExecutionId>,
}

impl Observation {
    fn validate(&self) -> Result<(), StoreError> {
        let owned_scope = matches!(self.scope, EvidenceScope::DepthLimited { .. });
        let external_scope = matches!(
            self.scope,
            EvidenceScope::ExternalUci { .. } | EvidenceScope::ProfiledExternalUci { .. }
        );
        let cpu_scope = owned_scope || external_scope;
        if owned_scope != self.value_identity.is_some()
            || cpu_scope != self.cpu_condition.is_some()
            || (self.cpu_pv.is_some() && !cpu_scope)
            || external_scope != self.external_report.is_some()
            || (!cpu_scope && self.checker_work.is_some())
        {
            return Err(StoreError::InvalidEvidence(
                "CPU scope requires its exact value namespace and PV scope",
            ));
        }
        match (&self.checker_identity, &self.value_identity) {
            (Some(CheckerIdentity::Owned(actual)), Some(value))
                if owned_scope && actual == value => {}
            (None, Some(_)) if owned_scope => {} // Legacy own CPU namespace.
            (Some(CheckerIdentity::ExternalUci(_)), None) if external_scope => {}
            (None, None) if !cpu_scope => {}
            _ => {
                return Err(StoreError::InvalidEvidence(
                    "checker and score namespaces differ",
                ));
            }
        }
        if let Some(identity) = &self.checker_identity {
            identity.validate().map_err(|_| {
                StoreError::InvalidEvidence("invalid or unbounded checker namespace")
            })?;
        }
        if self.checker_work.is_some_and(|work| {
            matches!((work.nodes, work.qnodes), (Some(nodes), Some(qnodes)) if qnodes > nodes)
        }) {
            return Err(StoreError::InvalidEvidence("checker work counters contradict"));
        }
        if external_scope {
            self.validate_external_report()?;
        } else if self.kind == ObservationKind::ExternalCpuAnalysis
            || matches!(self.score, RawScore::ExternalUci { .. })
        {
            return Err(StoreError::InvalidEvidence(
                "foreign result needs external scope",
            ));
        }
        let model_wdl = matches!(
            self.score,
            RawScore::Wdl { .. } | RawScore::ContextWdl { .. }
        );
        if model_wdl && !matches!(self.scope, EvidenceScope::Model { .. }) {
            return Err(StoreError::InvalidEvidence("model WDL needs model scope"));
        }
        if model_wdl != self.model_value_identity.is_some()
            || model_wdl != self.model_value_input.is_some()
        {
            return Err(StoreError::InvalidEvidence(
                "model WDL needs full value and input identity",
            ));
        }
        if let RawScore::ConditionalWdl { expectation, .. } = self.score {
            if !expectation.is_finite()
                || !(-1.0..=1.0).contains(&expectation)
                || !matches!(self.scope, EvidenceScope::Model { .. })
                || self.kind != ObservationKind::Refutation
                || self.line.is_none()
            {
                return Err(StoreError::InvalidEvidence(
                    "conditional model expectation needs a finite refutation line",
                ));
            }
        }
        if let RawScore::ConditionalRepairWdl { expectation, .. } = self.score {
            if !expectation.is_finite()
                || !(-1.0..=1.0).contains(&expectation)
                || !matches!(self.scope, EvidenceScope::Model { .. })
                || self.kind != ObservationKind::Repair
                || self.line.is_none()
                || self.supersedes.is_none()
            {
                return Err(StoreError::InvalidEvidence(
                    "conditional model repair needs a finite line and refutation link",
                ));
            }
        }
        if let Some(identity) = &self.model_value_identity {
            identity.validate().map_err(|_| {
                StoreError::InvalidEvidence("invalid or unbounded model value namespace")
            })?;
        }
        if let Some(identity) = &self.value_identity {
            identity.validate().map_err(|_| {
                StoreError::InvalidEvidence("invalid or unbounded CPU value namespace")
            })?;
        }
        if self
            .cpu_condition
            .as_ref()
            .is_some_and(|value| !valid_cpu_condition(value, external_scope))
        {
            return Err(StoreError::InvalidEvidence(
                "invalid or unbounded exact CPU conditions",
            ));
        }
        match self.score {
            RawScore::Estimate { value, .. } if !value.is_finite() => {
                return Err(StoreError::InvalidEvidence("non-finite estimate"));
            }
            RawScore::Wdl {
                win, draw, loss, ..
            }
            | RawScore::ContextWdl {
                win, draw, loss, ..
            } if ![win, draw, loss]
                .into_iter()
                .all(|x| x.is_finite() && (0.0..=1.0).contains(&x))
                || (win + draw + loss - 1.0).abs() > 1e-4 =>
            {
                return Err(StoreError::InvalidEvidence("invalid WDL"));
            }
            RawScore::Terminal { .. } if self.scope != EvidenceScope::RulesTerminal => {
                return Err(StoreError::InvalidEvidence("unverified terminal claim"));
            }
            _ => {}
        }
        if self.scope == EvidenceScope::RulesTerminal
            && (!matches!(self.score, RawScore::Terminal { .. })
                || self.kind != ObservationKind::ExactTerminal)
        {
            return Err(StoreError::InvalidEvidence(
                "terminal scope needs Rules result",
            ));
        }
        if matches!(self.score, RawScore::Cpu { .. })
            && !matches!(self.scope, EvidenceScope::DepthLimited { .. })
        {
            return Err(StoreError::InvalidEvidence("CPU result needs search scope"));
        }
        Ok(())
    }

    fn validate_external_report(&self) -> Result<(), StoreError> {
        let Some(CheckerIdentity::ExternalUci(identity)) = &self.checker_identity else {
            return Err(StoreError::InvalidEvidence(
                "foreign report needs checker identity",
            ));
        };
        let report = self
            .external_report
            .as_deref()
            .ok_or(StoreError::InvalidEvidence("foreign report is absent"))?;
        report.identity.validate().map_err(|_| {
            StoreError::InvalidEvidence("invalid or unbounded retained foreign namespace")
        })?;
        let matching_scope = match self.scope {
            EvidenceScope::ExternalUci {
                requested_depth,
                reported_depth,
                seldepth,
                bound,
            } => {
                requested_depth > 0
                    && requested_depth == report.requested_depth
                    && reported_depth == report.reported_depth
                    && seldepth == report.seldepth
                    && bound == report.bound
            }
            EvidenceScope::ProfiledExternalUci {
                requested,
                issued,
                reported_depth,
                seldepth,
                bound,
            } => {
                requested.validate_issued(issued)?;
                report.requested_depth == requested.requested_depth().unwrap_or(0)
                    && reported_depth == report.reported_depth
                    && seldepth == report.seldepth
                    && bound == report.bound
            }
            _ => false,
        };
        if report.identity != *identity
            || self.source != external_source_id(&identity.adapter_semantics)
            || self.checker_work != Some(report.work)
            || self.kind != ObservationKind::ExternalCpuAnalysis
            || !matching_scope
            || self.score
                != (RawScore::ExternalUci {
                    value: report.score,
                    bound: report.bound,
                    perspective: report.perspective,
                    wdl_per_mille: report.wdl_per_mille,
                })
            || report.pv.len() > MAX_EXTERNAL_PV_PLIES
            || report.pv.capacity() > MAX_EXTERNAL_PV_PLIES
            || report.observed_uci.name.trim().is_empty()
            || report.observed_uci.name.len() > 256
            || report.observed_uci.name.capacity() > 256
            || report.observed_uci.name.chars().any(char::is_control)
            || report.observed_uci.author.as_ref().is_some_and(|author| {
                author.len() > 256
                    || author.capacity() > 256
                    || author.chars().any(char::is_control)
            })
            || report.wdl_per_mille.is_some_and(|wdl| {
                wdl.into_iter().any(|value| value > 1000)
                    || wdl.into_iter().map(u32::from).sum::<u32>() != 1000
            })
            || report.work.nodes.is_some_and(|nodes| nodes != self.budget)
            || (report.work.nodes.is_none() && self.budget != 0)
        {
            return Err(StoreError::InvalidEvidence(
                "foreign report scope, counters or retained payload differ",
            ));
        }
        // Zero `budget` with unknown nodes is an observation bookkeeping slot,
        // not a zero-work claim; consumption still requires known bounded work.
        Ok(())
    }
}

/// Validate the immutable raw evidence used by a derived model refutation. The
/// engine owns the live root/context revision check; this verifies only the
/// retained evidence namespace, perspective conversion and strict decrease.
fn validate_conditional_wdl(
    observation: &Observation,
    repaired: &Observation,
    counter: &Observation,
) -> Result<(), StoreError> {
    let RawScore::ConditionalWdl {
        expectation,
        perspective,
        context_revision,
        ..
    } = observation.score
    else {
        return Ok(());
    };
    observation.validate()?;
    let (repaired_expectation, counter_expectation) =
        validate_context_model_pair(repaired, counter, perspective, context_revision)?;
    if counter_expectation >= repaired_expectation || expectation != counter_expectation {
        return Err(StoreError::InvalidEvidence(
            "conditional model expectation is not its strict counter decrease",
        ));
    }
    Ok(())
}

fn validate_context_model_pair(
    before: &Observation,
    after: &Observation,
    perspective: Color,
    context_revision: u64,
) -> Result<(f32, f32), StoreError> {
    before.validate()?;
    after.validate()?;
    if before.model_value_identity.is_none()
        || before.model_value_identity != after.model_value_identity
        || before.model_value_input.is_none()
        || after.model_value_input.is_none()
    {
        return Err(StoreError::InvalidEvidence(
            "conditional model evidence namespaces differ",
        ));
    }
    let root_expectation = |raw: &Observation| {
        if let RawScore::ContextWdl {
            win,
            loss,
            perspective: raw_perspective,
            context_revision: raw_revision,
            ..
        } = raw.score
            && raw_revision == context_revision
        {
            Ok(if raw_perspective == perspective {
                win - loss
            } else {
                loss - win
            })
        } else {
            Err(StoreError::InvalidEvidence(
                "conditional evidence needs actual raw WDL under its context revision",
            ))
        }
    };
    Ok((root_expectation(before)?, root_expectation(after)?))
}

fn validate_conditional_repair_wdl(
    observation: &Observation,
    before: &Observation,
    after: &Observation,
    refutation: &Observation,
) -> Result<(), StoreError> {
    let RawScore::ConditionalRepairWdl {
        expectation,
        perspective,
        context_revision,
        ..
    } = observation.score
    else {
        return Ok(());
    };
    observation.validate()?;
    refutation.validate()?;
    if refutation.kind != ObservationKind::Refutation || refutation.state != observation.state {
        return Err(StoreError::InvalidEvidence(
            "conditional model repair supersedes another refutation scope",
        ));
    }
    let (before_expectation, after_expectation) =
        validate_context_model_pair(before, after, perspective, context_revision)?;
    if after_expectation <= before_expectation || expectation != after_expectation {
        return Err(StoreError::InvalidEvidence(
            "conditional model repair is not its strict improvement",
        ));
    }
    Ok(())
}

fn validate_model_derivation<'a>(
    observation: &Observation,
    get: impl Fn(ObservationId) -> Result<&'a Observation, StoreError>,
) -> Result<(), StoreError> {
    match observation.score {
        RawScore::ConditionalWdl {
            repaired, counter, ..
        } => validate_conditional_wdl(observation, get(repaired)?, get(counter)?),
        RawScore::ConditionalRepairWdl { before, after, .. } => {
            let refutation = observation.supersedes.ok_or(StoreError::InvalidEvidence(
                "conditional model repair lacks superseded refutation",
            ))?;
            validate_conditional_repair_wdl(
                observation,
                get(before)?,
                get(after)?,
                get(refutation)?,
            )
        }
        _ => Ok(()),
    }
}

const MAX_EXTERNAL_PV_PLIES: usize = 256;

fn external_source_id(semantics: &str) -> u64 {
    semantics.bytes().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

#[derive(Debug)]
pub struct ObservationStore {
    observations: HotRecords<Observation>,
    limit: usize,
    /// A foreign append proof cannot survive replacement of this fact owner.
    identity: Arc<()>,
}

impl ObservationStore {
    /// Monotonic issued ID high-water; hot resident length decreases on
    /// archival and is not a witness-baseline sequence.
    pub fn next_observation_sequence(&self) -> usize {
        self.observations.next_id()
    }
    pub fn new(limit: usize) -> Self {
        Self {
            observations: HotRecords::new(),
            limit,
            identity: Arc::new(()),
        }
    }

    pub fn append(&mut self, observation: Observation) -> Result<ObservationId, StoreError> {
        observation.validate()?;
        validate_model_derivation(&observation, |id| self.get(id))?;
        if let Some(previous) = observation.supersedes {
            self.get(previous)?;
        }
        if self.observations.len() >= self.limit {
            return Err(StoreError::Capacity("observations"));
        }
        let id = ObservationId(self.observations.next_id());
        self.observations.push(observation)?;
        Ok(id)
    }

    pub fn get(&self, id: ObservationId) -> Result<&Observation, StoreError> {
        self.observations.get(id.0).ok_or_else(|| {
            if id.0 < self.observations.next_id() {
                StoreError::ColdRecord("observation")
            } else {
                StoreError::InvalidHandle("observation")
            }
        })
    }

    pub fn len(&self) -> usize {
        self.observations.len()
    }
    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ContinuationStatus {
    Unresolved,
    Supported,
    Refuted,
    RepairedBy(LineId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ContinuationConclusion {
    pub status: ContinuationStatus,
    pub evidence: Option<ObservationId>,
    pub revision: u64,
}

/// Mutable interpretation; immutable raw observations remain independently readable.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ConclusionView {
    continuations: BTreeMap<LineId, ContinuationConclusion>,
}

impl ConclusionView {
    pub fn get(&self, line: LineId) -> Option<&ContinuationConclusion> {
        self.continuations.get(&line)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&LineId, &ContinuationConclusion)> {
        self.continuations.iter()
    }

    fn ensure_capacity(&self, lines: &[LineId], limit: usize) -> Result<(), StoreError> {
        let additional = lines
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .filter(|line| !self.continuations.contains_key(line))
            .count();
        if self.continuations.len().saturating_add(additional) > limit {
            return Err(StoreError::Capacity("situation conclusions"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Situation {
    pub state: StateId,
    pub focus: LineId,
    pub revision: u64,
    pub dirty: bool,
    pub conclusions: ConclusionView,
}

#[derive(Debug)]
struct SituationSlot {
    generation: u64,
    value: Option<Situation>,
}

#[derive(Debug)]
pub struct SituationArena {
    slots: Vec<SituationSlot>,
    limit: usize,
    live: usize,
}

impl SituationArena {
    pub fn new(limit: usize) -> Self {
        Self {
            slots: Vec::new(),
            limit,
            live: 0,
        }
    }

    pub fn insert(&mut self, state: StateId, focus: LineId) -> Result<SituationId, StoreError> {
        let value = Situation {
            state,
            focus,
            revision: 0,
            dirty: false,
            conclusions: ConclusionView::default(),
        };
        if let Some((slot, free)) = self
            .slots
            .iter_mut()
            .enumerate()
            .find(|(_, slot)| slot.value.is_none())
        {
            let id = SituationId {
                slot,
                generation: free.generation,
            };
            free.value = Some(value);
            self.live += 1;
            return Ok(id);
        }
        if self.slots.len() >= self.limit {
            return Err(StoreError::Capacity("situations"));
        }
        let id = SituationId {
            slot: self.slots.len(),
            generation: 0,
        };
        self.slots.push(SituationSlot {
            generation: 0,
            value: Some(value),
        });
        self.live += 1;
        Ok(id)
    }

    pub fn get(&self, id: SituationId) -> Result<&Situation, StoreError> {
        self.slots
            .get(id.slot)
            .filter(|slot| slot.generation == id.generation)
            .and_then(|slot| slot.value.as_ref())
            .ok_or(StoreError::InvalidHandle("situation generation"))
    }

    pub fn get_mut(&mut self, id: SituationId) -> Result<&mut Situation, StoreError> {
        self.slots
            .get_mut(id.slot)
            .filter(|slot| slot.generation == id.generation)
            .and_then(|slot| slot.value.as_mut())
            .ok_or(StoreError::InvalidHandle("situation generation"))
    }

    pub fn remove(&mut self, id: SituationId) -> Result<Situation, StoreError> {
        self.get(id)?;
        let slot = &mut self.slots[id.slot];
        let next = slot
            .generation
            .checked_add(1)
            .ok_or(StoreError::RevisionExhausted)?;
        let value = slot
            .value
            .take()
            .ok_or(StoreError::InvalidHandle("situation"))?;
        slot.generation = next;
        self.live -= 1;
        Ok(value)
    }

    pub fn iter(&self) -> impl Iterator<Item = (SituationId, &Situation)> {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            slot.value.as_ref().map(|value| {
                (
                    SituationId {
                        slot: index,
                        generation: slot.generation,
                    },
                    value,
                )
            })
        })
    }

    pub fn len(&self) -> usize {
        self.live
    }
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }
}

#[derive(Debug)]
pub struct DependencyIndex {
    dependents: BTreeMap<ObservationId, BTreeSet<SituationId>>,
    edges: usize,
    limit: usize,
    #[cfg(test)]
    fail_allocations: usize,
}

impl DependencyIndex {
    pub fn new(limit: usize) -> Self {
        Self {
            dependents: BTreeMap::new(),
            edges: 0,
            limit,
            #[cfg(test)]
            fail_allocations: 0,
        }
    }

    pub fn add(
        &mut self,
        observation: ObservationId,
        situation: SituationId,
    ) -> Result<(), StoreError> {
        if self
            .dependents
            .get(&observation)
            .is_some_and(|ids| ids.contains(&situation))
        {
            return Ok(());
        }
        #[cfg(test)]
        if self.fail_allocations != 0 {
            self.fail_allocations -= 1;
            return Err(StoreError::Capacity("injected dependency allocation"));
        }
        if self.edges >= self.limit {
            return Err(StoreError::Capacity("dependency edges"));
        }
        self.dependents
            .entry(observation)
            .or_default()
            .insert(situation);
        self.edges += 1;
        Ok(())
    }

    pub fn affected(&self, observation: ObservationId) -> impl Iterator<Item = SituationId> + '_ {
        self.dependents
            .get(&observation)
            .into_iter()
            .flat_map(|ids| ids.iter().copied())
    }

    /// Only registered live dependents advance revision. Removed generations are
    /// ignored rather than dirtying the next occupant of the same arena slot.
    pub fn invalidate(
        &self,
        observation: ObservationId,
        arena: &mut SituationArena,
    ) -> Result<usize, StoreError> {
        let affected = self
            .affected(observation)
            .filter(|id| arena.get(*id).is_ok())
            .collect::<Vec<_>>();
        for id in &affected {
            arena
                .get(*id)?
                .revision
                .checked_add(1)
                .ok_or(StoreError::RevisionExhausted)?;
        }
        for id in &affected {
            let situation = arena.get_mut(*id)?;
            situation.revision += 1;
            situation.dirty = true;
        }
        Ok(affected.len())
    }

    pub fn len(&self) -> usize {
        self.edges
    }
    pub fn is_empty(&self) -> bool {
        self.edges == 0
    }
}

#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, serde::Serialize, serde::Deserialize,
)]
pub enum TaskQuestion {
    AnalyzePosition,
    AnalyzeRootMoves,
    FindAlternative {
        divergence_ply: u16,
    },
    ModelProposal,
    ModelRefutation,
    ModelRepair,
    ProfiledExternal {
        requested: ProfiledExternalConstraint,
    },
}

/// Exact task reuse namespace. A different requested depth, input revision,
/// profile, history StateId, candidate order, or model epoch is another question.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, serde::Serialize, serde::Deserialize)]
pub struct TaskKey {
    pub state: StateId,
    pub line: Option<LineId>,
    pub question: TaskQuestion,
    pub root_moves: Vec<Move16>,
    pub model: u64,
    pub epoch: u64,
    /// CPU tasks use the complete immutable value namespace, never a truncated
    /// model hash or a fictitious neural epoch. Model tasks leave this absent.
    pub value_identity: Option<CpuValueIdentity>,
    pub checker_identity: Option<CheckerIdentity>,
    /// Full canonical CPU implementation/configuration/capability, task kind,
    /// ordered root restrictions, profile and window. An immutable exact key;
    /// `condition` is only an auxiliary numeric identifier.
    pub cpu_condition: Option<String>,
    pub profile: u64,
    /// Issuer-owned identity for exact search conditions, not a truncated hash.
    /// Model/profile IDs likewise name immutable registered configurations.
    pub condition: u64,
    pub input_revision: u64,
    pub requested_depth: u16,
    pub node_budget: u64,
}

impl TaskKey {
    /// Known separately allocated payload of this owned key. Fixed fields are
    /// charged by its containing owner. Tree/hash nodes and allocator overhead
    /// are not included, so this is never a total heap or resident peak.
    pub(crate) fn known_owned_payload_bytes(&self) -> Option<u64> {
        fn identity_bytes(value: &CpuValueIdentity) -> Option<usize> {
            let mut bytes = value
                .semantics
                .capacity()
                .checked_add(value.weights_sha256.as_ref().map_or(0, String::capacity))?;
            if let crate::cpu_value::CpuTrainingState::Learned {
                run_id,
                dataset_sha256,
                ..
            } = &value.training
            {
                bytes = bytes
                    .checked_add(run_id.capacity())?
                    .checked_add(dataset_sha256.capacity())?;
            }
            Some(bytes)
        }
        let mut bytes = self
            .root_moves
            .capacity()
            .checked_mul(std::mem::size_of::<Move16>())?
            .checked_add(self.cpu_condition.as_ref().map_or(0, String::capacity))?;
        if let Some(identity) = &self.value_identity {
            bytes = bytes.checked_add(identity_bytes(identity)?)?;
        }
        // Conditional own-CPU sources never carry a foreign checker. Its
        // unsupported owned payload remains unknown rather than zero bytes.
        if self.checker_identity.is_some() {
            return None;
        }
        u64::try_from(bytes).ok()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TaskConsumer {
    pub id: u64,
    pub situation: SituationId,
    pub revision: u64,
    pub generation: u64,
    /// Monotonic caller clock units, not wall-clock timestamps.
    pub deadline_tick: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum TaskStatus {
    InFlight,
    CancellationRequested,
    Paused {
        checkpoint: u64,
        evidence: Option<ObservationId>,
    },
    /// A concrete CPU owner rejected reuse; partial evidence stays historical.
    RetiredPaused {
        checkpoint: u64,
        evidence: Option<ObservationId>,
    },
    Completed(ObservationId),
    Failed,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ConsumerRecord {
    consumer: TaskConsumer,
    cancelled: bool,
    consumed: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TaskRecord {
    pub key: TaskKey,
    pub status: TaskStatus,
    pub resumed_from: Option<ExecutionId>,
    consumers: Vec<ConsumerRecord>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaskAdmission {
    Start(ExecutionId),
    Join(ExecutionId),
    Reuse {
        execution: ExecutionId,
        observation: ObservationId,
    },
    Resume {
        execution: ExecutionId,
        previous: ExecutionId,
        checkpoint: u64,
    },
}

#[derive(Debug)]
pub struct TaskTable {
    tasks: HotRecords<TaskRecord>,
    latest: BTreeMap<TaskKey, ExecutionId>,
    /// Only active private worker instances live here. Existing archive pins
    /// retain them, so this index is bounded by the hot execution capacity.
    owned_instances: BTreeSet<ExecutionId>,
    consumers: BTreeSet<u64>,
    execution_limit: usize,
    consumer_limit: usize,
    root_move_limit: usize,
}

/// An owned physical CPU job may share its exact computation key with another
/// job without joining that job's execution or borrowing its paused stack.
/// This admission mode is process-local; archived task records keep their
/// original semantic key and execution provenance without a new codec field.
#[derive(Clone, Copy)]
enum TaskDispatch {
    Shared,
    OwnedCpu { previous: Option<ExecutionId> },
}

fn valid_cpu_condition(value: &String, external: bool) -> bool {
    let maximum = if external { 8192 } else { 2048 };
    !value.trim().is_empty()
        && value.len() <= maximum
        && value.capacity() <= maximum
        && !value.chars().any(char::is_control)
}

impl TaskTable {
    pub fn new(executions: usize, consumers: usize, root_moves_per_task: usize) -> Self {
        Self {
            tasks: HotRecords::new(),
            latest: BTreeMap::new(),
            owned_instances: BTreeSet::new(),
            consumers: BTreeSet::new(),
            execution_limit: executions,
            consumer_limit: consumers,
            root_move_limit: root_moves_per_task,
        }
    }

    pub fn request(
        &mut self,
        key: TaskKey,
        consumer: TaskConsumer,
        now_tick: u64,
    ) -> Result<TaskAdmission, StoreError> {
        self.request_dispatch(key, consumer, now_tick, TaskDispatch::Shared)
    }

    fn request_cpu_instance(
        &mut self,
        key: TaskKey,
        consumer: TaskConsumer,
        now_tick: u64,
        previous: Option<ExecutionId>,
    ) -> Result<TaskAdmission, StoreError> {
        self.request_dispatch(key, consumer, now_tick, TaskDispatch::OwnedCpu { previous })
    }

    fn request_dispatch(
        &mut self,
        key: TaskKey,
        consumer: TaskConsumer,
        now_tick: u64,
        dispatch: TaskDispatch,
    ) -> Result<TaskAdmission, StoreError> {
        if consumer.deadline_tick <= now_tick {
            return Err(StoreError::ExpiredConsumer);
        }
        if key.root_moves.len() > self.root_move_limit
            || key.root_moves.capacity() > self.root_move_limit
        {
            return Err(StoreError::Capacity("task root moves"));
        }
        let cpu_question = matches!(
            key.question,
            TaskQuestion::AnalyzePosition
                | TaskQuestion::AnalyzeRootMoves
                | TaskQuestion::FindAlternative { .. }
                | TaskQuestion::ProfiledExternal { .. }
        );
        let external = matches!(key.checker_identity, Some(CheckerIdentity::ExternalUci(_)));
        if let TaskQuestion::ProfiledExternal { requested } = key.question {
            requested.validate()?;
            if !external
                || key.requested_depth != requested.requested_depth().unwrap_or(0)
                || key.node_budget != requested.requested_nodes().unwrap_or(0)
            {
                return Err(StoreError::InvalidConditions(
                    "profiled external task constraint differs",
                ));
            }
        }
        if (cpu_question && !external) != key.value_identity.is_some()
            || cpu_question != key.cpu_condition.is_some()
            || (!cpu_question && key.checker_identity.is_some())
        {
            return Err(StoreError::InvalidConditions(
                "CPU task requires an exact evaluator namespace",
            ));
        }
        if let Some(identity) = &key.checker_identity {
            identity.validate().map_err(|_| {
                StoreError::InvalidConditions("invalid or unbounded checker namespace")
            })?;
            if let CheckerIdentity::Owned(actual) = identity {
                if key.value_identity.as_ref() != Some(actual) {
                    return Err(StoreError::InvalidConditions(
                        "own checker and value namespace differ",
                    ));
                }
            }
        }
        if let Some(identity) = &key.value_identity {
            identity.validate().map_err(|_| {
                StoreError::InvalidConditions("invalid or unbounded CPU value namespace")
            })?;
        }
        if key
            .cpu_condition
            .as_ref()
            .is_some_and(|value| !valid_cpu_condition(value, external))
        {
            return Err(StoreError::InvalidConditions(
                "invalid or unbounded exact CPU conditions",
            ));
        }
        if cpu_question
            && !matches!(key.question, TaskQuestion::ProfiledExternal { .. })
            && (key.requested_depth == 0 || key.node_budget == 0)
        {
            return Err(StoreError::InvalidConditions(
                "CPU task requires a finite positive budget",
            ));
        }
        if key.question == TaskQuestion::AnalyzeRootMoves && key.root_moves.is_empty() {
            return Err(StoreError::InvalidConditions(
                "restricted CPU task requires ordered root moves",
            ));
        }
        if external && key.question == TaskQuestion::AnalyzePosition && !key.root_moves.is_empty() {
            return Err(StoreError::InvalidConditions(
                "unrestricted external task has root moves",
            ));
        }
        let mut distinct_roots = BTreeSet::new();
        for movement in &key.root_moves {
            movement.unpack()?;
            if !distinct_roots.insert(*movement) {
                return Err(StoreError::InvalidConditions("duplicate task root move"));
            }
        }
        if self.consumers.contains(&consumer.id) {
            return Err(StoreError::InvalidConditions("duplicate consumer ID"));
        }
        if self.consumers.len() >= self.consumer_limit {
            return Err(StoreError::Capacity("task consumers"));
        }
        if matches!(dispatch, TaskDispatch::OwnedCpu { .. })
            && (!cpu_question || external || key.value_identity.is_none())
        {
            return Err(StoreError::InvalidConditions(
                "independent dispatch requires an owned CPU task",
            ));
        }
        let reusable = match dispatch {
            TaskDispatch::Shared => self.latest.get(&key).copied().filter(|id| {
                !self.owned_instances.contains(id)
                    && matches!(
                        self.tasks[id.0].status,
                        TaskStatus::InFlight | TaskStatus::Completed(_)
                    )
            }),
            TaskDispatch::OwnedCpu { .. } => None,
        };
        let (execution, admission) = if let Some(id) = reusable {
            let admission = match self.tasks[id.0].status {
                TaskStatus::Completed(observation) => TaskAdmission::Reuse {
                    execution: id,
                    observation,
                },
                _ => TaskAdmission::Join(id),
            };
            (id, admission)
        } else {
            if self.tasks.len() >= self.execution_limit {
                return Err(StoreError::Capacity("task executions"));
            }
            let id = ExecutionId(self.tasks.next_id());
            let paused = match dispatch {
                TaskDispatch::Shared => self.latest.get(&key).and_then(|previous| {
                    if self.owned_instances.contains(previous) {
                        return None;
                    }
                    match self.tasks[previous.0].status {
                        TaskStatus::Paused { checkpoint, .. } => Some((*previous, checkpoint)),
                        _ => None,
                    }
                }),
                TaskDispatch::OwnedCpu { previous: None } => None,
                TaskDispatch::OwnedCpu {
                    previous: Some(previous),
                } => {
                    let task = self.get(previous)?;
                    if task.key != key {
                        return Err(StoreError::InvalidConditions(
                            "independent CPU resume changed its semantic key",
                        ));
                    }
                    let TaskStatus::Paused { checkpoint, .. } = task.status else {
                        return Err(StoreError::InvalidConditions(
                            "independent CPU resume requires its live paused execution",
                        ));
                    };
                    Some((previous, checkpoint))
                }
            };
            self.tasks.push(TaskRecord {
                key: key.clone(),
                status: TaskStatus::InFlight,
                resumed_from: paused.map(|(previous, _)| previous),
                consumers: Vec::new(),
            })?;
            self.latest.insert(key, id);
            if matches!(dispatch, TaskDispatch::OwnedCpu { .. }) {
                self.owned_instances.insert(id);
            }
            let admission = match paused {
                Some((previous, checkpoint)) => TaskAdmission::Resume {
                    execution: id,
                    previous,
                    checkpoint,
                },
                None => TaskAdmission::Start(id),
            };
            (id, admission)
        };
        self.tasks[execution.0].consumers.push(ConsumerRecord {
            consumer,
            cancelled: false,
            consumed: false,
        });
        self.consumers.insert(consumer.id);
        Ok(admission)
    }

    pub fn get(&self, id: ExecutionId) -> Result<&TaskRecord, StoreError> {
        self.tasks.get(id.0).ok_or_else(|| {
            if id.0 < self.tasks.next_id() {
                StoreError::ColdRecord("execution")
            } else {
                StoreError::InvalidHandle("execution")
            }
        })
    }

    /// Publication retains the completed observation even if every consumer has
    /// expired. It grants no old or new root permission to use that observation.
    pub fn complete(
        &mut self,
        execution: ExecutionId,
        observation: ObservationId,
    ) -> Result<(), StoreError> {
        let task = self
            .tasks
            .get_mut(execution.0)
            .ok_or(StoreError::InvalidHandle("execution"))?;
        if !matches!(
            task.status,
            TaskStatus::InFlight | TaskStatus::CancellationRequested
        ) {
            return Err(StoreError::InvalidConditions("execution already terminal"));
        }
        task.status = TaskStatus::Completed(observation);
        self.owned_instances.remove(&execution);
        Ok(())
    }

    pub fn fail(&mut self, execution: ExecutionId) -> Result<(), StoreError> {
        let task = self
            .tasks
            .get_mut(execution.0)
            .ok_or(StoreError::InvalidHandle("execution"))?;
        if !matches!(
            task.status,
            TaskStatus::InFlight | TaskStatus::CancellationRequested
        ) {
            return Err(StoreError::InvalidConditions("execution already terminal"));
        }
        task.status = TaskStatus::Failed;
        self.owned_instances.remove(&execution);
        Ok(())
    }

    /// The backend must first establish a supported, physically quiescent resume
    /// boundary. The opaque checkpoint is owned by that same backend/profile;
    /// this table does not copy a stack or transfer it to another CPU engine.
    pub fn pause(
        &mut self,
        execution: ExecutionId,
        checkpoint: u64,
        evidence: Option<ObservationId>,
    ) -> Result<(), StoreError> {
        let task = self
            .tasks
            .get_mut(execution.0)
            .ok_or(StoreError::InvalidHandle("execution"))?;
        if matches!(
            task.key.checker_identity,
            Some(CheckerIdentity::ExternalUci(_))
        ) {
            return Err(StoreError::InvalidConditions(
                "external UCI has no owned resume checkpoint",
            ));
        }
        if !matches!(
            task.status,
            TaskStatus::InFlight | TaskStatus::CancellationRequested
        ) {
            return Err(StoreError::InvalidConditions("execution already terminal"));
        }
        task.status = TaskStatus::Paused {
            checkpoint,
            evidence,
        };
        Ok(())
    }

    pub fn cancel_consumer(&mut self, execution: ExecutionId, id: u64) -> Result<(), StoreError> {
        let task = self
            .tasks
            .get_mut(execution.0)
            .ok_or(StoreError::InvalidHandle("execution"))?;
        let record = task
            .consumers
            .iter_mut()
            .find(|record| record.consumer.id == id)
            .ok_or(StoreError::InvalidHandle("consumer"))?;
        record.cancelled = true;
        if task.status == TaskStatus::InFlight
            && task.consumers.iter().all(|record| record.cancelled)
        {
            task.status = TaskStatus::CancellationRequested;
        }
        Ok(())
    }

    /// Exactly one consumption per live consumer. Completion reuse may add a new
    /// consumer, but cannot resurrect the deadline or generation of an old one.
    pub fn consume(
        &mut self,
        execution: ExecutionId,
        consumer_id: u64,
        now_tick: u64,
        current_generation: u64,
        arena: &SituationArena,
    ) -> Result<ObservationId, StoreError> {
        let task = self
            .tasks
            .get_mut(execution.0)
            .ok_or(StoreError::InvalidHandle("execution"))?;
        let observation = match task.status {
            TaskStatus::Completed(id) => id,
            _ => return Err(StoreError::NotCompleted),
        };
        let record = task
            .consumers
            .iter_mut()
            .find(|record| record.consumer.id == consumer_id)
            .ok_or(StoreError::InvalidHandle("consumer"))?;
        if record.consumed {
            return Err(StoreError::AlreadyConsumed);
        }
        if record.cancelled {
            return Err(StoreError::CancelledConsumer);
        }
        if record.consumer.deadline_tick <= now_tick {
            return Err(StoreError::ExpiredConsumer);
        }
        let situation = arena
            .get(record.consumer.situation)
            .map_err(|_| StoreError::StaleConsumer)?;
        if record.consumer.generation != current_generation
            || record.consumer.revision != situation.revision
            || task.key.state != situation.state
        {
            return Err(StoreError::StaleConsumer);
        }
        record.consumed = true;
        Ok(observation)
    }

    pub fn len(&self) -> usize {
        self.tasks.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }
    pub fn consumer_count(&self) -> usize {
        self.consumers.len()
    }
    /// Actual retained consumer-vector capacities, including completed and
    /// cancelled rows. This issues no consumer and does not change flags.
    /// Task arrays, tree nodes, and allocator overhead remain outside this
    /// known subset; callers must not report it as total Store memory.
    pub(crate) fn historical_consumer_backing_bytes(&self) -> Option<u64> {
        self.tasks.iter().try_fold(0_u64, |sum, task| {
            let bytes = task
                .consumers
                .capacity()
                .checked_mul(std::mem::size_of::<ConsumerRecord>())?;
            sum.checked_add(u64::try_from(bytes).ok()?)
        })
    }
    /// Read an already recorded use of this exact original consumer. This
    /// historical projection never admits work, reopens a deadline or makes
    /// a cancelled consumer live again.
    pub(crate) fn historical_consumer_was_consumed(
        &self,
        execution: ExecutionId,
        original: TaskConsumer,
    ) -> Result<bool, StoreError> {
        let task = self.get(execution)?;
        let record = task
            .consumers
            .iter()
            .find(|record| record.consumer.id == original.id)
            .ok_or(StoreError::InvalidHandle("consumer"))?;
        if record.consumer != original {
            return Err(StoreError::StaleConsumer);
        }
        Ok(record.consumed)
    }
    pub fn paused_executions_for_state(
        &self,
        state: StateId,
    ) -> Result<Vec<ExecutionId>, StoreError> {
        let count = self
            .tasks
            .entries()
            .filter(|(_, t)| {
                t.key.state == state
                    && matches!(t.status, TaskStatus::Paused { .. })
                    && t.key.value_identity.is_some()
                    && !matches!(
                        t.key.checker_identity,
                        Some(CheckerIdentity::ExternalUci(_))
                    )
            })
            .count();
        let mut result = Vec::new();
        result
            .try_reserve_exact(count)
            .map_err(|_| StoreError::Capacity("paused execution index"))?;
        result.extend(
            self.tasks
                .entries()
                .filter(|(_, t)| {
                    t.key.state == state
                        && matches!(t.status, TaskStatus::Paused { .. })
                        && t.key.value_identity.is_some()
                        && !matches!(
                            t.key.checker_identity,
                            Some(CheckerIdentity::ExternalUci(_))
                        )
                })
                .map(|(id, _)| ExecutionId(id)),
        );
        Ok(result)
    }
    /// The concrete CPU owner must first establish which tokens remain current.
    /// This table certifies no backend stack ownership or physical completion.
    pub fn retire_paused_except(
        &mut self,
        current: &BTreeSet<ExecutionId>,
    ) -> Result<usize, StoreError> {
        for id in current {
            self.get(*id)?;
        }
        let count = self
            .tasks
            .entries()
            .filter(|(id, t)| {
                !current.contains(&ExecutionId(*id))
                    && matches!(t.status, TaskStatus::Paused { .. })
                    && t.key.value_identity.is_some()
                    && !matches!(
                        t.key.checker_identity,
                        Some(CheckerIdentity::ExternalUci(_))
                    )
            })
            .count();
        let mut ids = Vec::new();
        ids.try_reserve_exact(count)
            .map_err(|_| StoreError::Capacity("retired pause index"))?;
        ids.extend(
            self.tasks
                .entries()
                .filter(|(id, t)| {
                    !current.contains(&ExecutionId(*id))
                        && matches!(t.status, TaskStatus::Paused { .. })
                        && t.key.value_identity.is_some()
                        && !matches!(
                            t.key.checker_identity,
                            Some(CheckerIdentity::ExternalUci(_))
                        )
                })
                .map(|(id, _)| id),
        );
        for id in &ids {
            let task = self.tasks.get_mut(*id).expect("checked pause execution");
            let TaskStatus::Paused {
                checkpoint,
                evidence,
            } = task.status
            else {
                unreachable!("filtered pause")
            };
            task.status = TaskStatus::RetiredPaused {
                checkpoint,
                evidence,
            };
            for c in &mut task.consumers {
                c.cancelled = true;
            }
            self.owned_instances.remove(&ExecutionId(*id));
        }
        Ok(ids.len())
    }
    pub fn active_states(&self) -> impl Iterator<Item = StateId> + '_ {
        self.tasks
            .iter()
            .filter(|t| {
                matches!(
                    t.status,
                    TaskStatus::InFlight
                        | TaskStatus::CancellationRequested
                        | TaskStatus::Paused { .. }
                )
            })
            .map(|t| t.key.state)
    }
    /// Engine compaction supplies current node indices only after committing
    /// the old topology. Retired checkpoint metadata is never rewritten.
    pub fn remap_paused_checkpoints(
        &mut self,
        nodes: &BTreeMap<StateId, u64>,
    ) -> Result<usize, StoreError> {
        let count = self
            .tasks
            .entries()
            .filter(|(_, task)| {
                matches!(task.status, TaskStatus::Paused { .. })
                    && task.key.value_identity.is_some()
                    && !matches!(
                        task.key.checker_identity,
                        Some(CheckerIdentity::ExternalUci(_))
                    )
            })
            .count();
        let mut remaps = Vec::new();
        remaps
            .try_reserve_exact(count)
            .map_err(|_| StoreError::Capacity("pause checkpoint remap"))?;
        for (id, task) in self.tasks.entries() {
            if matches!(task.status, TaskStatus::Paused { .. })
                && task.key.value_identity.is_some()
                && !matches!(
                    task.key.checker_identity,
                    Some(CheckerIdentity::ExternalUci(_))
                )
            {
                let checkpoint =
                    nodes
                        .get(&task.key.state)
                        .copied()
                        .ok_or(StoreError::InvalidHandle(
                            "active pause state missing from remap",
                        ))?;
                remaps.push((id, checkpoint));
            }
        }
        for (id, checkpoint) in &remaps {
            let task = self.tasks.get_mut(*id).expect("checked remap execution");
            let TaskStatus::Paused { evidence, .. } = task.status else {
                unreachable!("checked pause remap")
            };
            task.status = TaskStatus::Paused {
                checkpoint: *checkpoint,
                evidence,
            };
        }
        Ok(remaps.len())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct DerivedCacheKey {
    pub state: StateId,
    pub line: Option<LineId>,
    pub model: u64,
    pub epoch: u64,
    pub encoding: u64,
    pub mask: u64,
    pub precision: u64,
}

/// Optional CPU representation bytes; physical GPU pages live in the runtime's
/// event/lease owner. Evicting these bytes never removes facts or completed tasks.
#[derive(Debug)]
pub struct DerivedCache {
    entries: BTreeMap<DerivedCacheKey, Vec<u8>>,
    entry_limit: usize,
    byte_limit: usize,
    bytes: usize,
}

impl DerivedCache {
    pub fn new(entries: usize, bytes: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            entry_limit: entries,
            byte_limit: bytes,
            bytes: 0,
        }
    }

    pub fn insert(&mut self, key: DerivedCacheKey, bytes: Vec<u8>) -> Result<(), StoreError> {
        let previous = self.entries.get(&key).map_or(0, Vec::capacity);
        let new_total = self
            .bytes
            .checked_sub(previous)
            .and_then(|value| value.checked_add(bytes.capacity()))
            .filter(|total| *total <= self.byte_limit)
            .ok_or(StoreError::Capacity("derived cache bytes"))?;
        if !self.entries.contains_key(&key) && self.entries.len() >= self.entry_limit {
            return Err(StoreError::Capacity("derived cache entries"));
        }
        self.entries.insert(key, bytes);
        self.bytes = new_total;
        Ok(())
    }

    pub fn get(&self, key: &DerivedCacheKey) -> Option<&[u8]> {
        self.entries.get(key).map(Vec::as_slice)
    }

    pub fn evict_all(&mut self) -> usize {
        let freed = self.bytes;
        self.entries.clear();
        self.bytes = 0;
        freed
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

#[derive(Debug)]
struct CheckedCpuPv {
    line_pool: Weak<()>,
    state_store: Weak<()>,
}

/// Single-writer facade validates cross-store handles before publishing a fact.
#[derive(Debug)]
pub struct PalsStores {
    pub states: StateStore,
    pub lines: LinePool,
    pub situations: SituationArena,
    pub observations: ObservationStore,
    pub dependencies: DependencyIndex,
    pub tasks: TaskTable,
    pub derived: DerivedCache,
    limits: StoreLimits,
    root: Option<SituationId>,
    generation: u64,
    checked_cpu_pvs: BTreeMap<LineId, CheckedCpuPv>,
    archive: Option<archive::ArchiveManager>,
    /// Session write authority survives archive close without retaining files,
    /// hot-store owners, cold handles or permission to admit another write.
    archive_write_ledger: Option<Arc<archive::ArchiveWriteLedger>>,
    /// Immutable submitted review views own independent, bounded hot leases.
    /// Archive authority and other consumers remain separate owners.
    review_pins: Arc<Mutex<ReviewPinRegistry>>,
}

#[derive(Debug, Default)]
struct ReviewPinRegistry {
    next: u64,
    leases: BTreeMap<u64, StorePins>,
}

/// Concrete store ownership; numeric handles never confer this authority.
#[derive(Clone, Debug)]
pub struct StoreReadOwner {
    states: Weak<()>,
    lines: Weak<()>,
}
impl StoreReadOwner {
    pub fn matches(&self, stores: &PalsStores) -> bool {
        self.states.ptr_eq(&Arc::downgrade(&stores.states.identity))
            && self.lines.ptr_eq(&Arc::downgrade(&stores.lines.identity))
    }
}

/// Process-local custody of one actual CPU dispatch. Numeric execution IDs and
/// equal computation keys alone cannot authorize an independent stack resume.
/// Active/paused records use the existing archive protected-pins closure; this
/// handle cannot reconstruct a worker or resume authority from a cold record.
#[derive(Clone, Debug)]
pub(crate) struct CpuTaskDispatch {
    owner: StoreReadOwner,
    generation: u64,
    execution: ExecutionId,
    consumer: TaskConsumer,
}
impl CpuTaskDispatch {
    pub(crate) fn identifies(&self, execution: ExecutionId, consumer: TaskConsumer) -> bool {
        self.execution == execution && self.consumer == consumer
    }
}

/// Dropping the last immutable view releases only its own hot lease. This is
/// neither archive deletion nor a CPU/GPU physical completion acknowledgement.
#[derive(Debug)]
pub struct StoreReadPin {
    owner: StoreReadOwner,
    registry: Arc<Mutex<ReviewPinRegistry>>,
    serial: u64,
    generation: u64,
}
impl StoreReadPin {
    pub fn validates(&self, stores: &PalsStores) -> bool {
        self.owner.matches(stores)
            && self.generation == stores.generation()
            && Arc::ptr_eq(&self.registry, &stores.review_pins)
            && self
                .registry
                .lock()
                .is_ok_and(|pins| pins.leases.contains_key(&self.serial))
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    /// Read only this original lease's validated seed. This neither joins the
    /// view union nor creates another lease or retirement/admission authority.
    pub(crate) fn retirement_seed(&self, stores: &PalsStores) -> Result<StorePins, StoreError> {
        if !self.owner.matches(stores)
            || self.generation != stores.generation()
            || !Arc::ptr_eq(&self.registry, &stores.review_pins)
        {
            return Err(StoreError::StaleConsumer);
        }
        self.registry
            .lock()
            .map_err(|_| StoreError::InvalidConditions("review pin owner poisoned"))?
            .leases
            .get(&self.serial)
            .cloned()
            .ok_or(StoreError::InvalidHandle("review pin lease"))
    }
    pub fn reborrow(&self, stores: &PalsStores) -> Result<StoreReadPin, StoreError> {
        if !self.validates(stores) {
            return Err(StoreError::StaleConsumer);
        }
        let pins = self
            .registry
            .lock()
            .map_err(|_| StoreError::InvalidConditions("review pin owner poisoned"))?
            .leases
            .get(&self.serial)
            .cloned()
            .ok_or(StoreError::InvalidHandle("review pin lease"))?;
        stores.pin_review(pins)
    }
}
impl Drop for StoreReadPin {
    fn drop(&mut self) {
        // Recover poison for release only. Admission/collection remains closed
        // on poison; discarding a poisoned ownership lock would leak a lease.
        let mut pins = self
            .registry
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        pins.leases.remove(&self.serial);
    }
}

impl PalsStores {
    pub fn new(limits: StoreLimits) -> Self {
        Self {
            states: StateStore::new(limits.states, limits.retained_history_bytes),
            lines: LinePool::new(limits.line_chunks, limits.line_plies),
            situations: SituationArena::new(limits.situations),
            observations: ObservationStore::new(limits.observations),
            dependencies: DependencyIndex::new(limits.dependency_edges),
            tasks: TaskTable::new(
                limits.executions,
                limits.consumers,
                limits.root_moves_per_task,
            ),
            derived: DerivedCache::new(limits.derived_entries, limits.derived_bytes),
            limits,
            root: None,
            generation: 0,
            checked_cpu_pvs: BTreeMap::new(),
            archive: None,
            archive_write_ledger: None,
            review_pins: Arc::new(Mutex::new(ReviewPinRegistry::default())),
        }
    }

    pub fn root(&self) -> Option<SituationId> {
        self.root
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn read_owner(&self) -> StoreReadOwner {
        StoreReadOwner {
            states: Arc::downgrade(&self.states.identity),
            lines: Arc::downgrade(&self.lines.identity),
        }
    }
    /// Validate seed handles now, then retain their dependency closure through
    /// the archive's existing protected_pins/expand_pins path on reclamation.
    pub fn pin_review(&self, pins: StorePins) -> Result<StoreReadPin, StoreError> {
        if pins.states.len() > self.limits.states
            || pins.lines.len() > self.limits.line_chunks
            || pins.observations.len() > self.limits.observations
            || pins.situations.len() > self.limits.situations
            || pins.executions.len() > self.limits.executions
        {
            return Err(StoreError::PinSaturated("review seed index"));
        }
        for id in &pins.states {
            self.states.get(*id)?;
        }
        for id in &pins.lines {
            self.lines.get(*id)?;
        }
        for id in &pins.observations {
            self.observations.get(*id)?;
        }
        for id in &pins.situations {
            self.situations.get(*id)?;
        }
        for id in &pins.executions {
            self.tasks.get(*id)?;
        }
        let mut registry = self
            .review_pins
            .lock()
            .map_err(|_| StoreError::InvalidConditions("review pin owner poisoned"))?;
        if registry.leases.len() >= REVIEW_PIN_CAPACITY {
            return Err(StoreError::PinSaturated("review view leases"));
        }
        let serial = registry.next;
        registry.next = serial.checked_add(1).ok_or(StoreError::RevisionExhausted)?;
        registry.leases.insert(serial, pins);
        Ok(StoreReadPin {
            owner: self.read_owner(),
            registry: Arc::clone(&self.review_pins),
            serial,
            generation: self.generation,
        })
    }
    pub fn review_pin_count(&self) -> Result<usize, StoreError> {
        self.review_pins
            .lock()
            .map(|pins| pins.leases.len())
            .map_err(|_| StoreError::InvalidConditions("review pin owner poisoned"))
    }
    /// The exact same hard ceiling enforced by pin_review. Reading it grants
    /// no lease and does not change the Store's capacity policy.
    pub const fn review_pin_capacity(&self) -> usize {
        REVIEW_PIN_CAPACITY
    }
    pub(super) fn review_pin_union(&self) -> Result<StorePins, StoreError> {
        let registry = self
            .review_pins
            .lock()
            .map_err(|_| StoreError::InvalidConditions("review pin owner poisoned"))?;
        let mut combined = StorePins::default();
        for pins in registry.leases.values() {
            combined.states.extend(&pins.states);
            combined.lines.extend(&pins.lines);
            combined.observations.extend(&pins.observations);
            combined.situations.extend(&pins.situations);
            combined.executions.extend(&pins.executions);
        }
        Ok(combined)
    }

    /// Admission controls are rechecked after immutable I/O commits and before
    /// each allocation attempt. The caller can borrow cancellation authority;
    /// no callback or borrowed pointer escapes this synchronous operation.
    fn allocate_hot_with_controls<T>(
        &mut self,
        pins: StorePins,
        mut controls: impl FnMut() -> Result<(), StoreError>,
        mut operation: impl FnMut(&mut Self) -> Result<T, StoreError>,
    ) -> Result<T, StoreError> {
        controls()?;
        self.automatic_reclaim(pins.clone(), false)?;
        self.check_archive_allocation_deadline()?;
        controls()?;
        let result = operation(self);
        if matches!(result, Err(StoreError::Capacity(_))) {
            controls()?;
            if self.automatic_reclaim(pins, true)? {
                self.check_archive_allocation_deadline()?;
                controls()?;
                let result = operation(self);
                controls()?;
                return result;
            }
        }
        controls()?;
        result
    }

    fn allocate_once_with_controls<T>(
        &mut self,
        mut controls: impl FnMut() -> Result<(), StoreError>,
        operation: impl FnOnce(&mut Self) -> Result<T, StoreError>,
    ) -> Result<T, StoreError> {
        controls()?;
        let result = operation(self);
        controls()?;
        result
    }

    pub fn insert_situation(
        &mut self,
        snapshot: PositionSnapshot,
    ) -> Result<SituationId, StoreError> {
        if !self.automatic_archive_enabled() {
            return self.insert_situation_once(snapshot);
        }
        self.insert_situation_controlled(snapshot, || Ok(()))
    }

    pub fn insert_situation_controlled(
        &mut self,
        snapshot: PositionSnapshot,
        controls: impl FnMut() -> Result<(), StoreError>,
    ) -> Result<SituationId, StoreError> {
        if !self.automatic_archive_enabled() {
            return self.allocate_once_with_controls(controls, |stores| {
                stores.insert_situation_once(snapshot)
            });
        }
        let mut pins = StorePins::default();
        if let Some(id) = self.states.find(&snapshot) {
            pins.states.insert(id);
        }
        self.allocate_hot_with_controls(pins, controls, |stores| {
            stores.insert_situation_once(snapshot.clone())
        })
    }
    fn insert_situation_once(
        &mut self,
        snapshot: PositionSnapshot,
    ) -> Result<SituationId, StoreError> {
        if self.situations.len() >= self.limits.situations {
            return Err(StoreError::Capacity("situations"));
        }
        let existing = self.states.find(&snapshot);
        if (existing.is_none()
            || existing.is_some_and(|state| !self.lines.roots.contains_key(&state)))
            && self.lines.len() >= self.limits.line_chunks
        {
            return Err(StoreError::Capacity("line chunks"));
        }
        let state = self.states.insert(snapshot)?;
        let line = self.lines.root(state)?;
        self.situations.insert(state, line)
    }

    pub fn append_line(
        &mut self,
        prefix: LineId,
        checked_moves: &[BoardMove],
    ) -> Result<LineId, StoreError> {
        self.append_line_controlled(prefix, checked_moves, || Ok(()))
    }

    pub fn append_line_controlled(
        &mut self,
        prefix: LineId,
        checked_moves: &[BoardMove],
        controls: impl FnMut() -> Result<(), StoreError>,
    ) -> Result<LineId, StoreError> {
        self.states.get(self.lines.get(prefix)?.start_state)?;
        let mut pins = StorePins::default();
        pins.lines.insert(prefix);
        self.allocate_hot_with_controls(pins, controls, |stores| {
            stores.lines.append(prefix, checked_moves)
        })
    }

    /// Mint a CPU PV handle only after replay through the Rules owner from the
    /// complete registered state. The checked-handle set is bounded by the
    /// existing immutable line-chunk limit and does not store duplicate moves.
    pub fn append_cpu_pv(
        &mut self,
        position: &Position,
        moves: &[BoardMove],
    ) -> Result<LineId, StoreError> {
        self.append_cpu_pv_controlled(position, moves, || Ok(()))
    }

    pub fn append_cpu_pv_controlled(
        &mut self,
        position: &Position,
        moves: &[BoardMove],
        controls: impl FnMut() -> Result<(), StoreError>,
    ) -> Result<LineId, StoreError> {
        let mut pins = StorePins::default();
        if let Some(state) = self.states.find(&position.snapshot()) {
            pins.states.insert(state);
        }
        self.allocate_hot_with_controls(pins, controls, |stores| {
            stores.append_cpu_pv_once(position, moves)
        })
    }
    fn append_cpu_pv_once(
        &mut self,
        position: &Position,
        moves: &[BoardMove],
    ) -> Result<LineId, StoreError> {
        if moves.len() > self.limits.line_plies {
            return Err(StoreError::Capacity("CPU PV plies"));
        }
        let state = self
            .states
            .find(&position.snapshot())
            .ok_or(StoreError::InvalidEvidence(
                "CPU PV starts at an unregistered exact state",
            ))?;
        let mut replay = position.clone();
        for movement in moves {
            replay
                .make_move(*movement)
                .map_err(|_| StoreError::InvalidEvidence("CPU PV failed Rules replay"))?;
        }
        self.append_validated_cpu_pv_once(state, moves)
    }

    /// Private allocation boundary shared by a completed Rules replay and an
    /// opaque, original-controller checked path. It issues no replay authority.
    fn append_validated_cpu_pv_once(
        &mut self,
        state: StateId,
        moves: &[BoardMove],
    ) -> Result<LineId, StoreError> {
        let root = self.lines.root(state)?;
        let line = self.lines.append(root, moves)?;
        if self.checked_cpu_pvs.get(&line).is_some_and(|checked| {
            !checked
                .line_pool
                .ptr_eq(&Arc::downgrade(&self.lines.identity))
                || !checked
                    .state_store
                    .ptr_eq(&Arc::downgrade(&self.states.identity))
        }) {
            return Err(StoreError::InvalidEvidence(
                "checked CPU PV handle cannot be reissued by another store",
            ));
        }
        if !self.checked_cpu_pvs.contains_key(&line)
            && self.checked_cpu_pvs.len() >= self.limits.line_chunks
        {
            return Err(StoreError::Capacity("checked CPU PV handles"));
        }
        self.checked_cpu_pvs.insert(
            line,
            CheckedCpuPv {
                line_pool: Arc::downgrade(&self.lines.identity),
                state_store: Arc::downgrade(&self.states.identity),
            },
        );
        Ok(line)
    }

    pub fn append_observation(
        &mut self,
        observation: Observation,
    ) -> Result<ObservationId, StoreError> {
        // The legacy/no-automatic-archive path never clones large raw payloads.
        if self.archive_enabled() && observation.external_report.is_some() {
            return Err(StoreError::ArchiveBudgetRequired);
        }
        if !self.automatic_archive_enabled() {
            return self.append_observation_once(observation);
        }
        self.append_observation_controlled(observation, || Ok(()))
    }

    pub fn append_observation_controlled(
        &mut self,
        observation: Observation,
        controls: impl FnMut() -> Result<(), StoreError>,
    ) -> Result<ObservationId, StoreError> {
        if self.archive_enabled() && observation.external_report.is_some() {
            return Err(StoreError::ArchiveBudgetRequired);
        }
        if !self.automatic_archive_enabled() {
            return self.allocate_once_with_controls(controls, |stores| {
                stores.append_observation_once(observation)
            });
        }
        let mut pins = StorePins::default();
        pins.states.insert(observation.state);
        pins.lines.extend(observation.line);
        pins.lines.extend(observation.cpu_pv);
        pins.observations.extend(observation.supersedes);
        pins.observations
            .extend(observation.score.model_dependencies().into_iter().flatten());
        pins.executions.extend(observation.execution);
        self.allocate_hot_with_controls(pins, controls, |stores| {
            stores.append_observation_once(observation.clone())
        })
    }
    fn append_observation_once(
        &mut self,
        observation: Observation,
    ) -> Result<ObservationId, StoreError> {
        if matches!(observation.score, RawScore::Terminal { .. }) {
            return Err(StoreError::InvalidEvidence(
                "Rules terminal must be minted from a checked Position",
            ));
        }
        self.validate_observation_handles(&observation)?;
        self.observations.append(observation)
    }

    fn validate_observation_handles(&self, observation: &Observation) -> Result<(), StoreError> {
        let snapshot = self.states.get(observation.state)?;
        observation.validate()?;
        if let RawScore::ConditionalWdl { perspective, .. }
        | RawScore::ConditionalRepairWdl { perspective, .. } = observation.score
        {
            if perspective != snapshot.side_to_move() {
                return Err(StoreError::InvalidEvidence(
                    "conditional model expectation has another root perspective",
                ));
            }
            validate_model_derivation(observation, |id| self.observations.get(id))?;
        }
        if let Some(report) = observation.external_report.as_deref() {
            if self.observations.observations.iter().any(|previous| {
                previous.external_report.as_deref().is_some_and(|earlier| {
                    earlier.identity == report.identity
                        && earlier.request_id == report.request_id
                        && (previous.execution != observation.execution
                            || previous.state != observation.state)
                })
            }) {
                return Err(StoreError::InvalidEvidence(
                    "foreign physical request belongs to another task",
                ));
            }
            self.validate_external_pv(observation, report)?;
        }
        if let Some(previous) = observation.supersedes {
            if self.observations.get(previous)?.state != observation.state {
                return Err(StoreError::InvalidEvidence(
                    "superseded observation belongs to another exact state",
                ));
            }
        }
        if let Some(line) = observation.line {
            if self.lines.get(line)?.start_state != observation.state {
                return Err(StoreError::InvalidEvidence("line starts at another state"));
            }
        }
        if let Some(pv) = observation.cpu_pv {
            let checked = self
                .checked_cpu_pvs
                .get(&pv)
                .ok_or(StoreError::InvalidEvidence(
                    "CPU PV lacks matching exact-state Rules replay",
                ))?;
            if self.lines.get(pv)?.start_state != observation.state
                || !checked
                    .line_pool
                    .ptr_eq(&Arc::downgrade(&self.lines.identity))
                || !checked
                    .state_store
                    .ptr_eq(&Arc::downgrade(&self.states.identity))
            {
                return Err(StoreError::InvalidEvidence(
                    "CPU PV lacks matching exact-state Rules replay",
                ));
            }
        }
        if let Some(execution) = observation.execution {
            let task = self.tasks.get(execution)?;
            match (task.key.question, observation.scope) {
                (
                    TaskQuestion::ProfiledExternal {
                        requested: question,
                    },
                    EvidenceScope::ProfiledExternalUci { requested, .. },
                ) if question == requested => {}
                (TaskQuestion::ProfiledExternal { .. }, _)
                | (_, EvidenceScope::ProfiledExternalUci { .. }) => {
                    return Err(StoreError::InvalidEvidence(
                        "profiled task and observation constraint differ",
                    ));
                }
                _ => {}
            }
            if task.key.state != observation.state
                || task.key.line != observation.line
                || task.key.value_identity != observation.value_identity
                || task.key.checker_identity != observation.checker_identity
                || task.key.cpu_condition != observation.cpu_condition
            {
                return Err(StoreError::InvalidEvidence(
                    "task and observation conditions differ",
                ));
            }
            if let Some(report) = observation.external_report.as_deref() {
                // A foreign node request is a hint, not strict native credit.
                // Preserve an overshot raw report for the original one-use
                // feedback path. validate_completed_task separately enforces
                // the registered budget before bounded task/cache reuse.
                if task.key.epoch != observation.epoch
                    || report.requested_depth != task.key.requested_depth
                    || report.root_restricted == task.key.root_moves.is_empty()
                    || report.best_move.is_some_and(|movement| {
                        !task.key.root_moves.is_empty()
                            && Move16::pack(movement)
                                .map_or(true, |packed| !task.key.root_moves.contains(&packed))
                    })
                {
                    return Err(StoreError::InvalidEvidence(
                        "foreign task budget or restriction differs",
                    ));
                }
            }
            if let Some(pv) = observation.cpu_pv {
                if !task.key.root_moves.is_empty() {
                    let first = self
                        .lines
                        .first_move(pv)?
                        .ok_or(StoreError::InvalidEvidence(
                            "restricted CPU PV has no first move",
                        ))?;
                    if !task.key.root_moves.contains(&Move16::pack(first)?) {
                        return Err(StoreError::InvalidEvidence(
                            "CPU PV first move is outside the admitted root restriction",
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn position_for_state(&self, state: StateId) -> Result<Position, StoreError> {
        let snapshot = self.states.get(state)?;
        let trace = snapshot
            .uci_replay(rz_position::MAX_UCI_REPLAY_PLIES)
            .map_err(|_| {
                StoreError::InvalidEvidence("exact external state history cannot be replayed")
            })?;
        let mut position = match trace.origin {
            rz_position::HistoryOrigin::StartPosition => Position::startpos(),
            rz_position::HistoryOrigin::Fen => Position::from_fen(&trace.start_fen)
                .map_err(|_| StoreError::InvalidEvidence("external state origin is invalid"))?,
        };
        for movement in trace.moves {
            position.make_move(movement).map_err(|_| {
                StoreError::InvalidEvidence("external state history failed Rules replay")
            })?;
        }
        if !position.snapshot().same_state(snapshot) {
            return Err(StoreError::InvalidEvidence(
                "external state history differs",
            ));
        }
        Ok(position)
    }

    fn validate_external_pv(
        &self,
        observation: &Observation,
        report: &ExternalCheckerReport,
    ) -> Result<(), StoreError> {
        let mut position = self.position_for_state(observation.state)?;
        if report.perspective != position.side_to_move()
            || (report.best_move.is_some() && report.best_move != report.pv.first().copied())
            || (report.completion == ExternalCompletion::BestMove
                && (report.best_move.is_none() || report.pv.is_empty()))
        {
            return Err(StoreError::InvalidEvidence(
                "foreign bestmove or perspective differs",
            ));
        }
        // Recheck the full retained raw PV, including any tail omitted from the
        // line projection. Only Rules owns state transitions and terminal truth.
        for movement in &report.pv {
            let legal = position.ordered_legal_moves();
            if !matches!(
                position.play_status_from_view(&legal),
                Ok(PlayStatus::Ongoing)
            ) {
                return Err(StoreError::InvalidEvidence(
                    "foreign PV continues after Rules terminal",
                ));
            }
            position
                .make_from_view(&legal, *movement)
                .map_err(|_| StoreError::InvalidEvidence("foreign PV failed Rules replay"))?;
        }
        let projected = &report.pv[..report.pv.len().min(self.limits.line_plies)];
        match observation.cpu_pv {
            Some(pv) if self.lines.moves(pv)? == projected => {}
            None if projected.is_empty() => {}
            _ => return Err(StoreError::InvalidEvidence("foreign PV projection differs")),
        }
        Ok(())
    }

    /// Only the Rules-owned automatic termination classifier can create exact
    /// terminal evidence through the product facade. Model/CPU reported mate and
    /// claimable draws cannot select this path by setting an enum field.
    pub fn append_rules_terminal(
        &mut self,
        position: &Position,
        source: u64,
        epoch: u64,
    ) -> Result<ObservationId, StoreError> {
        self.append_rules_terminal_controlled(position, source, epoch, || Ok(()))
    }

    pub fn append_rules_terminal_controlled(
        &mut self,
        position: &Position,
        source: u64,
        epoch: u64,
        controls: impl FnMut() -> Result<(), StoreError>,
    ) -> Result<ObservationId, StoreError> {
        let mut pins = StorePins::default();
        if let Some(state) = self.states.find(&position.snapshot()) {
            pins.states.insert(state);
        }
        self.allocate_hot_with_controls(pins, controls, |stores| {
            stores.append_rules_terminal_once(position, source, epoch)
        })
    }
    fn append_rules_terminal_once(
        &mut self,
        position: &Position,
        source: u64,
        epoch: u64,
    ) -> Result<ObservationId, StoreError> {
        let legal = position.ordered_legal_moves();
        let winner = match position
            .play_status_from_view(&legal)
            .map_err(|_| StoreError::InvalidEvidence("Rules terminal validation failed"))?
        {
            PlayStatus::Terminal { winner, .. } => winner,
            PlayStatus::Ongoing => {
                return Err(StoreError::InvalidEvidence(
                    "Rules position is not terminal",
                ));
            }
        };
        if self.observations.len() >= self.limits.observations {
            return Err(StoreError::Capacity("observations"));
        }
        let state = self.states.insert(position.snapshot())?;
        let observation = Observation {
            state,
            line: None,
            source,
            epoch,
            value_identity: None,
            checker_identity: None,
            checker_work: None,
            external_report: None,
            model_value_identity: None,
            model_value_input: None,
            cpu_condition: None,
            cpu_pv: None,
            scope: EvidenceScope::RulesTerminal,
            score: RawScore::Terminal { winner },
            budget: 0,
            kind: ObservationKind::ExactTerminal,
            supersedes: None,
            execution: None,
        };
        self.validate_observation_handles(&observation)?;
        self.observations.append(observation)
    }

    fn check_conclusion(
        &self,
        situation: SituationId,
        line: LineId,
        evidence: ObservationId,
    ) -> Result<(), StoreError> {
        let current = self.situations.get(situation)?;
        let observation = self.observations.get(evidence)?;
        if self.lines.get(line)?.start_state != current.state
            || observation.state != current.state
            || observation.line != Some(line)
        {
            return Err(StoreError::InvalidEvidence("conclusion conditions differ"));
        }
        Ok(())
    }

    /// Register a hot evidence edge with the same allocation policy as records.
    /// Evidence may originate at a child state while the root depends on it.
    pub fn add_dependency(
        &mut self,
        observation: ObservationId,
        situation: SituationId,
    ) -> Result<(), StoreError> {
        self.add_dependency_controlled(observation, situation, || Ok(()))
    }

    pub fn add_dependency_controlled(
        &mut self,
        observation: ObservationId,
        situation: SituationId,
        controls: impl FnMut() -> Result<(), StoreError>,
    ) -> Result<(), StoreError> {
        self.observations.get(observation)?;
        self.situations.get(situation)?;
        let mut pins = StorePins::default();
        pins.observations.insert(observation);
        pins.situations.insert(situation);
        self.allocate_hot_with_controls(pins, controls, |stores| {
            stores.dependencies.add(observation, situation)
        })
    }

    pub fn refute_continuation(
        &mut self,
        situation: SituationId,
        line: LineId,
        evidence: ObservationId,
    ) -> Result<(), StoreError> {
        self.refute_continuation_controlled(situation, line, evidence, || Ok(()))
    }

    pub fn refute_continuation_controlled(
        &mut self,
        situation: SituationId,
        line: LineId,
        evidence: ObservationId,
        controls: impl FnMut() -> Result<(), StoreError>,
    ) -> Result<(), StoreError> {
        self.check_conclusion(situation, line, evidence)?;
        let mut pins = StorePins::default();
        pins.situations.insert(situation);
        pins.lines.insert(line);
        pins.observations.insert(evidence);
        let revision = self.allocate_hot_with_controls(pins, controls, |stores| {
            stores.refute_continuation_once(situation, line, evidence)
        })?;
        let current = self.situations.get_mut(situation)?;
        current.revision = revision;
        current.dirty = true;
        current.conclusions.continuations.insert(
            line,
            ContinuationConclusion {
                status: ContinuationStatus::Refuted,
                evidence: Some(evidence),
                revision,
            },
        );
        Ok(())
    }

    fn refute_continuation_once(
        &mut self,
        situation: SituationId,
        line: LineId,
        evidence: ObservationId,
    ) -> Result<u64, StoreError> {
        self.check_conclusion(situation, line, evidence)?;
        if self.observations.get(evidence)?.kind != ObservationKind::Refutation {
            return Err(StoreError::InvalidEvidence(
                "refutation requires refutation record",
            ));
        }
        let current = self.situations.get(situation)?;
        current
            .conclusions
            .ensure_capacity(&[line], self.limits.conclusions_per_situation)?;
        let revision = current
            .revision
            .checked_add(1)
            .ok_or(StoreError::RevisionExhausted)?;
        self.dependencies.add(evidence, situation)?;
        Ok(revision)
    }

    pub fn repair(
        &mut self,
        situation: SituationId,
        refuted_line: LineId,
        repaired_line: LineId,
        evidence: ObservationId,
    ) -> Result<(), StoreError> {
        self.repair_controlled(situation, refuted_line, repaired_line, evidence, || Ok(()))
    }

    pub fn repair_controlled(
        &mut self,
        situation: SituationId,
        refuted_line: LineId,
        repaired_line: LineId,
        evidence: ObservationId,
        controls: impl FnMut() -> Result<(), StoreError>,
    ) -> Result<(), StoreError> {
        self.check_conclusion(situation, repaired_line, evidence)?;
        self.lines.get(refuted_line)?;
        let mut pins = StorePins::default();
        pins.situations.insert(situation);
        pins.lines.extend([refuted_line, repaired_line]);
        pins.observations.insert(evidence);
        let revision = self.allocate_hot_with_controls(pins, controls, |stores| {
            stores.repair_once(situation, refuted_line, repaired_line, evidence)
        })?;
        let current = self.situations.get_mut(situation)?;
        current.revision = revision;
        current.dirty = true;
        current.conclusions.continuations.insert(
            refuted_line,
            ContinuationConclusion {
                status: ContinuationStatus::RepairedBy(repaired_line),
                evidence: Some(evidence),
                revision,
            },
        );
        current.conclusions.continuations.insert(
            repaired_line,
            ContinuationConclusion {
                status: ContinuationStatus::Supported,
                evidence: Some(evidence),
                revision,
            },
        );
        Ok(())
    }

    fn repair_once(
        &mut self,
        situation: SituationId,
        refuted_line: LineId,
        repaired_line: LineId,
        evidence: ObservationId,
    ) -> Result<u64, StoreError> {
        self.check_conclusion(situation, repaired_line, evidence)?;
        let observation = self.observations.get(evidence)?;
        if observation.kind != ObservationKind::Repair {
            return Err(StoreError::InvalidEvidence("repair requires repair record"));
        }
        let current = self.situations.get(situation)?;
        if self.lines.get(refuted_line)?.start_state != current.state {
            return Err(StoreError::InvalidEvidence(
                "repair starts at another state",
            ));
        }
        let previous = current
            .conclusions
            .get(refuted_line)
            .filter(|conclusion| conclusion.status == ContinuationStatus::Refuted)
            .ok_or(StoreError::InvalidConditions(
                "repair needs active refutation",
            ))?;
        if observation.supersedes != previous.evidence {
            return Err(StoreError::InvalidEvidence(
                "repair does not supersede active refutation",
            ));
        }
        if refuted_line == repaired_line {
            return Err(StoreError::InvalidConditions(
                "repair needs revised continuation",
            ));
        }
        current.conclusions.ensure_capacity(
            &[refuted_line, repaired_line],
            self.limits.conclusions_per_situation,
        )?;
        let revision = current
            .revision
            .checked_add(1)
            .ok_or(StoreError::RevisionExhausted)?;
        self.dependencies.add(evidence, situation)?;
        Ok(revision)
    }

    /// The caller supplies a state already produced by checked Rules moves. This
    /// method does not replay moves or create legality rules. An existing exact
    /// state situation is retained; every focus issues new acceptance authority.
    pub fn focus_actual_moves(
        &mut self,
        checked_state: PositionSnapshot,
    ) -> Result<SituationId, StoreError> {
        let next_generation = self
            .generation
            .checked_add(1)
            .ok_or(StoreError::RevisionExhausted)?;
        let matching = self.states.find(&checked_state).and_then(|state| {
            self.situations
                .iter()
                .find(|(_, situation)| situation.state == state)
                .map(|(id, _)| id)
        });
        let root = match matching {
            Some(id) => id,
            None => self.insert_situation(checked_state)?,
        };
        self.root = Some(root);
        self.generation = next_generation;
        Ok(root)
    }

    pub fn request_task(
        &mut self,
        key: TaskKey,
        consumer: TaskConsumer,
        now_tick: u64,
    ) -> Result<TaskAdmission, StoreError> {
        if !self.automatic_archive_enabled() {
            return self.request_task_once(key, consumer, now_tick);
        }
        self.request_task_controlled(key, consumer, now_tick, || Ok(()))
    }

    pub fn request_task_controlled(
        &mut self,
        key: TaskKey,
        consumer: TaskConsumer,
        now_tick: u64,
        controls: impl FnMut() -> Result<(), StoreError>,
    ) -> Result<TaskAdmission, StoreError> {
        let mut admitted = None;
        let result = if !self.automatic_archive_enabled() {
            self.allocate_once_with_controls(controls, |stores| {
                let admission = stores.request_task_once(key, consumer, now_tick)?;
                admitted = Some(admission);
                Ok(admission)
            })
        } else {
            let mut pins = StorePins::default();
            pins.states.insert(key.state);
            pins.lines.extend(key.line);
            pins.situations.insert(consumer.situation);
            self.allocate_hot_with_controls(pins, controls, |stores| {
                let admission = stores.request_task_once(key.clone(), consumer, now_tick)?;
                admitted = Some(admission);
                Ok(admission)
            })
        };
        if result.is_err()
            && let Some(admission) = admitted
        {
            self.retire_unreturned_task_admission(admission, consumer.id)?;
        }
        result
    }

    fn retire_unreturned_task_admission(
        &mut self,
        admission: TaskAdmission,
        consumer_id: u64,
    ) -> Result<(), StoreError> {
        let execution = match admission {
            TaskAdmission::Start(execution)
            | TaskAdmission::Join(execution)
            | TaskAdmission::Resume { execution, .. }
            | TaskAdmission::Reuse { execution, .. } => execution,
        };
        self.tasks.cancel_consumer(execution, consumer_id)?;
        // A new/resumed reservation was never returned to the backend for
        // dispatch. Joined physical work keeps its cancellation/drain lifetime.
        if matches!(
            admission,
            TaskAdmission::Start(_) | TaskAdmission::Resume { .. }
        ) {
            self.tasks.fail(execution)?;
        }
        Ok(())
    }
    fn request_task_once(
        &mut self,
        key: TaskKey,
        consumer: TaskConsumer,
        now_tick: u64,
    ) -> Result<TaskAdmission, StoreError> {
        self.validate_task_request(&key, consumer)?;
        // Low-level tables are exposed for inspection. A caller must not turn
        // retained partial evidence into reusable completion via raw mutation.
        if let Some(execution) = self.tasks.latest.get(&key).copied()
            && let TaskStatus::Completed(observation) = self.tasks.get(execution)?.status
        {
            self.validate_completed_task(execution, observation)?;
        }
        self.tasks.request(key, consumer, now_tick)
    }

    fn validate_task_request(
        &self,
        key: &TaskKey,
        consumer: TaskConsumer,
    ) -> Result<(), StoreError> {
        self.states.get(key.state)?;
        if matches!(key.checker_identity, Some(CheckerIdentity::ExternalUci(_))) {
            let position = self.position_for_state(key.state)?;
            let legal = position.ordered_legal_moves();
            for movement in &key.root_moves {
                if !legal.moves().contains(&movement.unpack()?) {
                    return Err(StoreError::InvalidConditions(
                        "external task root is not legal",
                    ));
                }
            }
        }
        if let Some(line) = key.line {
            if self.lines.get(line)?.start_state != key.state {
                return Err(StoreError::InvalidConditions("task line state"));
            }
        }
        let situation = self.situations.get(consumer.situation)?;
        if situation.state != key.state
            || situation.revision != consumer.revision
            || consumer.generation != self.generation
        {
            return Err(StoreError::StaleConsumer);
        }
        Ok(())
    }

    pub fn complete_task(
        &mut self,
        execution: ExecutionId,
        observation: ObservationId,
    ) -> Result<(), StoreError> {
        self.validate_completed_task(execution, observation)?;
        self.tasks.complete(execution, observation)
    }

    fn validate_completed_task(
        &self,
        execution: ExecutionId,
        observation: ObservationId,
    ) -> Result<(), StoreError> {
        let task = self.tasks.get(execution)?;
        let evidence = self.observations.get(observation)?;
        if matches!(task.key.question, TaskQuestion::ProfiledExternal { .. })
            || matches!(evidence.scope, EvidenceScope::ProfiledExternalUci { .. })
        {
            return Err(StoreError::InvalidEvidence(
                "profiled external feedback is not completed native work",
            ));
        }
        if evidence.execution != Some(execution)
            || evidence.state != task.key.state
            || evidence.line != task.key.line
            || evidence.value_identity != task.key.value_identity
            || evidence.checker_identity != task.key.checker_identity
            || evidence.cpu_condition != task.key.cpu_condition
        {
            return Err(StoreError::InvalidEvidence(
                "completed task record mismatch",
            ));
        }
        self.validate_observation_handles(evidence)?;
        if evidence.epoch != task.key.epoch || evidence.budget > task.key.node_budget {
            return Err(StoreError::InvalidEvidence(
                "completed task epoch or budget mismatch",
            ));
        }
        let cpu_question = matches!(
            task.key.question,
            TaskQuestion::AnalyzePosition
                | TaskQuestion::AnalyzeRootMoves
                | TaskQuestion::FindAlternative { .. }
        );
        if (cpu_question && matches!(evidence.scope, EvidenceScope::Model { .. }))
            || (!cpu_question
                && matches!(
                    evidence.scope,
                    EvidenceScope::DepthLimited { .. } | EvidenceScope::ExternalUci { .. }
                ))
        {
            return Err(StoreError::InvalidEvidence(
                "completed task backend scope mismatch",
            ));
        }
        match evidence.scope {
            EvidenceScope::ExternalUci {
                requested_depth, ..
            } => {
                let report = evidence
                    .external_report
                    .as_deref()
                    .ok_or(StoreError::InvalidEvidence("foreign task lacks report"))?;
                if requested_depth != task.key.requested_depth
                    || report.completion != ExternalCompletion::BestMove
                    || report.work.nodes.is_none()
                    || report.best_move.is_none()
                {
                    return Err(StoreError::InvalidEvidence(
                        "foreign task lacks a valid bounded bestmove completion",
                    ));
                }
            }
            EvidenceScope::DepthLimited {
                depth,
                profile,
                condition,
            } if depth < task.key.requested_depth
                || profile != task.key.profile
                || condition != task.key.condition =>
            {
                return Err(StoreError::InvalidEvidence(
                    "completed CPU scope does not satisfy task",
                ));
            }
            EvidenceScope::Model { model, input, .. }
                if model != task.key.model || input != task.key.input_revision =>
            {
                return Err(StoreError::InvalidEvidence(
                    "completed model scope does not satisfy task",
                ));
            }
            _ => {}
        }
        Ok(())
    }

    pub fn pause_task(
        &mut self,
        execution: ExecutionId,
        checkpoint: u64,
        partial: Option<ObservationId>,
    ) -> Result<(), StoreError> {
        let task = self.tasks.get(execution)?;
        if matches!(
            task.key.checker_identity,
            Some(CheckerIdentity::ExternalUci(_))
        ) {
            return Err(StoreError::InvalidConditions(
                "external UCI has no owned resume checkpoint",
            ));
        }
        if let Some(id) = partial {
            let evidence = self.observations.get(id)?;
            if evidence.execution != Some(execution)
                || evidence.state != task.key.state
                || evidence.line != task.key.line
                || evidence.value_identity != task.key.value_identity
                || evidence.checker_identity != task.key.checker_identity
                || evidence.cpu_condition != task.key.cpu_condition
            {
                return Err(StoreError::InvalidEvidence("paused task record mismatch"));
            }
            self.validate_observation_handles(evidence)?;
            if evidence.epoch != task.key.epoch || evidence.budget > task.key.node_budget {
                return Err(StoreError::InvalidEvidence(
                    "paused task epoch or budget mismatch",
                ));
            }
            match evidence.scope {
                EvidenceScope::DepthLimited {
                    profile, condition, ..
                } if task.key.value_identity.is_none()
                    || profile != task.key.profile
                    || condition != task.key.condition =>
                {
                    return Err(StoreError::InvalidEvidence(
                        "paused CPU scope does not satisfy task",
                    ));
                }
                EvidenceScope::Model { model, input, .. }
                    if task.key.value_identity.is_some()
                        || model != task.key.model
                        || input != task.key.input_revision =>
                {
                    return Err(StoreError::InvalidEvidence(
                        "paused model scope does not satisfy task",
                    ));
                }
                _ => {}
            }
        }
        self.tasks.pause(execution, checkpoint, partial)
    }

    pub fn consume_task(
        &mut self,
        execution: ExecutionId,
        consumer_id: u64,
        now_tick: u64,
    ) -> Result<ObservationId, StoreError> {
        if let TaskStatus::Completed(observation) = self.tasks.get(execution)?.status {
            self.validate_completed_task(execution, observation)?;
        }
        self.tasks.consume(
            execution,
            consumer_id,
            now_tick,
            self.generation,
            &self.situations,
        )
    }

    pub fn evict_derived_cache(&mut self) -> usize {
        self.derived.evict_all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_position::Position;

    fn cpu_value_identity() -> CpuValueIdentity {
        CpuValueIdentity {
            semantics: crate::cpu::BOOTSTRAP_SCORE_VERSION.into(),
            weights_sha256: None,
            training: crate::cpu_value::CpuTrainingState::Bootstrap,
        }
    }

    fn moves(text: &str) -> Vec<BoardMove> {
        text.split_whitespace()
            .map(|mv| BoardMove::from_uci(mv).unwrap())
            .collect()
    }

    fn observation(state: StateId, line: Option<LineId>, kind: ObservationKind) -> Observation {
        Observation {
            state,
            line,
            source: 1,
            epoch: 1,
            value_identity: Some(cpu_value_identity()),
            checker_identity: None,
            checker_work: None,
            external_report: None,
            model_value_identity: None,
            model_value_input: None,
            cpu_condition: Some("fixture-cpu-search-conditions-v1".into()),
            cpu_pv: None,
            scope: EvidenceScope::DepthLimited {
                depth: 4,
                profile: 1,
                condition: 1,
            },
            score: RawScore::Unknown,
            budget: 128,
            kind,
            supersedes: None,
            execution: None,
        }
    }

    fn key(state: StateId) -> TaskKey {
        TaskKey {
            state,
            line: None,
            question: TaskQuestion::AnalyzePosition,
            root_moves: Vec::new(),
            model: 1,
            epoch: 1,
            value_identity: Some(cpu_value_identity()),
            checker_identity: None,
            cpu_condition: Some("fixture-cpu-search-conditions-v1".into()),
            profile: 1,
            condition: 1,
            input_revision: 0,
            requested_depth: 4,
            node_budget: 128,
        }
    }

    fn float_value_identity(weight_digit: char) -> CpuValueIdentity {
        CpuValueIdentity {
            semantics: crate::cpu_value::CPU_VALUE_ARCHITECTURE.into(),
            weights_sha256: Some(weight_digit.to_string().repeat(64)),
            training: crate::cpu_value::CpuTrainingState::Untrained,
        }
    }

    #[test]
    fn historical_consumer_backing_counts_spare_capacity_without_changing_flags() {
        let mut table = TaskTable::new(4, 16, 4);
        let mut consumers = Vec::with_capacity(8);
        consumers.push(ConsumerRecord {
            consumer: TaskConsumer {
                id: 1,
                situation: SituationId {
                    slot: 0,
                    generation: 1,
                },
                revision: 3,
                generation: 4,
                deadline_tick: 5,
            },
            consumed: true,
            cancelled: false,
        });
        let expected = (consumers.capacity() * std::mem::size_of::<ConsumerRecord>()) as u64;
        table
            .tasks
            .push(TaskRecord {
                key: key(StateId(0)),
                status: TaskStatus::Completed(ObservationId(0)),
                resumed_from: None,
                consumers,
            })
            .unwrap();
        let original = table.tasks.get(0).unwrap().consumers[0].consumer;
        let before = serde_json::to_vec(table.tasks.get(0).unwrap()).unwrap();
        assert_eq!(table.historical_consumer_backing_bytes(), Some(expected));
        assert!(
            table
                .historical_consumer_was_consumed(ExecutionId(0), original)
                .unwrap()
        );
        assert_eq!(
            serde_json::to_vec(table.tasks.get(0).unwrap()).unwrap(),
            before
        );
        assert!(expected > std::mem::size_of::<ConsumerRecord>() as u64);
    }

    #[test]
    fn historical_task_key_backing_counts_owned_namespace_capacity() {
        let mut task = key(StateId(0));
        let before = task.known_owned_payload_bytes().unwrap();
        let old_capacity = task.cpu_condition.as_ref().unwrap().capacity();
        task.cpu_condition.as_mut().unwrap().reserve(4096);
        let new_capacity = task.cpu_condition.as_ref().unwrap().capacity();
        assert_eq!(
            task.known_owned_payload_bytes(),
            Some(before + (new_capacity - old_capacity) as u64)
        );
        task.checker_identity = Some(CheckerIdentity::Owned(cpu_value_identity()));
        assert!(task.known_owned_payload_bytes().is_none());
    }

    fn external_identity() -> crate::cpu_checker::ExternalCheckerIdentity {
        use crate::cpu_checker::{ExternalModelMetadata, ExternalTrainingKnowledge};
        crate::cpu_checker::ExternalCheckerIdentity {
            adapter_semantics: "fixture-external-uci-raw/1".into(),
            binary_sha256: "a".repeat(64),
            launch_arguments_sha256: "b".repeat(64),
            declared_name: "fixture".into(),
            declared_version: "1".into(),
            declared_source: "test-source".into(),
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

    fn external_key(state: StateId) -> TaskKey {
        let mut query = key(state);
        query.value_identity = None;
        query.checker_identity = Some(CheckerIdentity::ExternalUci(external_identity()));
        query.cpu_condition = Some("fixture-external-full-conditions".into());
        query
    }

    fn external_observation(state: StateId, execution: ExecutionId, pv: LineId) -> Observation {
        use crate::cpu_checker::ExternalUciIdentity;
        let report = ExternalCheckerReport {
            identity: external_identity(),
            observed_uci: ExternalUciIdentity {
                name: "fixture".into(),
                author: None,
            },
            request_id: 1,
            best_move: Some(BoardMove::from_uci("e2e4").unwrap()),
            pv: moves("e2e4 e7e5"),
            score: ExternalRawScore::MateMoves(19),
            bound: ExternalBound::Lower,
            wdl_per_mille: Some([900, 50, 50]),
            perspective: Color::White,
            requested_depth: 4,
            reported_depth: Some(1),
            seldepth: Some(3),
            root_restricted: false,
            completion: ExternalCompletion::BestMove,
            work: CheckerWork {
                nodes: Some(20),
                qnodes: None,
                tt_hits: None,
            },
            elapsed: std::time::Duration::from_millis(1),
        };
        let mut record = observation(state, None, ObservationKind::ExternalCpuAnalysis);
        record.source = external_source_id(&report.identity.adapter_semantics);
        record.value_identity = None;
        record.checker_identity = Some(CheckerIdentity::ExternalUci(report.identity.clone()));
        record.checker_work = Some(report.work);
        record.cpu_condition = Some("fixture-external-full-conditions".into());
        record.cpu_pv = Some(pv);
        record.scope = EvidenceScope::ExternalUci {
            requested_depth: report.requested_depth,
            reported_depth: report.reported_depth,
            seldepth: report.seldepth,
            bound: report.bound,
        };
        record.score = RawScore::ExternalUci {
            value: report.score,
            bound: report.bound,
            perspective: report.perspective,
            wdl_per_mille: report.wdl_per_mille,
        };
        record.budget = 20;
        record.external_report = Some(Box::new(report));
        record.execution = Some(execution);
        record
    }

    fn profiled_key(state: StateId, requested: ProfiledExternalConstraint) -> TaskKey {
        let mut key = external_key(state);
        key.question = TaskQuestion::ProfiledExternal { requested };
        key.requested_depth = requested.requested_depth().unwrap_or(0);
        key.node_budget = requested.requested_nodes().unwrap_or(0);
        key
    }

    fn profiled_observation(
        state: StateId,
        execution: ExecutionId,
        pv: LineId,
        requested: ProfiledExternalConstraint,
    ) -> Observation {
        let mut raw = external_observation(state, execution, pv);
        let report = raw.external_report.as_mut().unwrap();
        report.requested_depth = requested.requested_depth().unwrap_or(0);
        raw.scope = EvidenceScope::ProfiledExternalUci {
            requested,
            issued: requested,
            reported_depth: report.reported_depth,
            seldepth: report.seldepth,
            bound: report.bound,
        };
        raw
    }

    #[test]
    fn profiled_limits_remain_exact_raw_feedback_and_never_completed_depth() {
        for requested in [
            ProfiledExternalConstraint::Combined {
                depth: 4,
                reported_nodes: 16,
            },
            ProfiledExternalConstraint::Nodes { reported_nodes: 16 },
            ProfiledExternalConstraint::Depth { depth: 4 },
            ProfiledExternalConstraint::MoveTime { milliseconds: 100 },
        ] {
            let mut stores = PalsStores::new(StoreLimits::default());
            let position = Position::startpos();
            let root = stores.focus_actual_moves(position.snapshot()).unwrap();
            let state = stores.situations.get(root).unwrap().state;
            let original = consumer(&stores, root, 1, 100);
            let key = profiled_key(state, requested);
            let TaskAdmission::Start(execution) =
                stores.request_task(key.clone(), original, 0).unwrap()
            else {
                panic!("profiled start")
            };
            let pv = stores
                .append_cpu_pv(&position, &moves("e2e4 e7e5"))
                .unwrap();
            let raw = profiled_observation(state, execution, pv, requested);
            // Reported 20 may overshoot the requested 16, or nodes may not have
            // been requested at all. Neither is a native reservation or proof.
            let observation = stores.append_observation(raw.clone()).unwrap();
            assert!(stores.complete_task(execution, observation).is_err());
            assert!(
                stores
                    .hot_consume_sealed_profiled_foreign_feedback(
                        execution,
                        original,
                        &key,
                        &position.snapshot(),
                        observation,
                        1
                    )
                    .is_err()
            );
            stores.tasks.fail(execution).unwrap();
            assert_eq!(
                stores
                    .hot_consume_sealed_profiled_foreign_feedback(
                        execution,
                        original,
                        &key,
                        &position.snapshot(),
                        observation,
                        1
                    )
                    .unwrap(),
                observation
            );
            assert_eq!(
                stores.hot_consume_sealed_profiled_foreign_feedback(
                    execution,
                    original,
                    &key,
                    &position.snapshot(),
                    observation,
                    1
                ),
                Err(StoreError::AlreadyConsumed)
            );
            assert_eq!(
                stores
                    .hot_validate_consumed_profiled_foreign_feedback(
                        execution,
                        original,
                        &key,
                        &position.snapshot(),
                        observation,
                        2
                    )
                    .unwrap(),
                observation
            );
            assert!(
                stores
                    .hot_consume_sealed_foreign_feedback(
                        execution,
                        original,
                        &key,
                        &position.snapshot(),
                        observation,
                        2
                    )
                    .is_err()
            );
            assert_eq!(
                stores.tasks.get(execution).unwrap().status,
                TaskStatus::Failed
            );
            assert!(matches!(
                stores
                    .request_task(key, consumer(&stores, root, 2, 100), 2)
                    .unwrap(),
                TaskAdmission::Start(_)
            ));
            assert_eq!(stores.observations.get(observation).unwrap(), &raw);
            stores
                .tasks
                .cancel_consumer(execution, original.id)
                .unwrap();
            assert_eq!(
                stores.hot_validate_consumed_profiled_foreign_feedback(
                    execution,
                    original,
                    &profiled_key(state, requested),
                    &position.snapshot(),
                    observation,
                    3
                ),
                Err(StoreError::CancelledConsumer)
            );
        }
    }

    #[test]
    fn profiled_absent_limits_reject_fake_positive_fields_and_legacy_zero_depth() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let position = Position::startpos();
        let root = stores.focus_actual_moves(position.snapshot()).unwrap();
        let state = stores.situations.get(root).unwrap().state;
        let requested = ProfiledExternalConstraint::MoveTime { milliseconds: 100 };
        let key = profiled_key(state, requested);
        for (depth, nodes) in [(1, 0), (0, 1), (1, 1)] {
            let mut fabricated = key.clone();
            fabricated.requested_depth = depth;
            fabricated.node_budget = nodes;
            assert!(
                stores
                    .request_task(fabricated, consumer(&stores, root, 1, 100), 0)
                    .is_err()
            );
        }
        let original = consumer(&stores, root, 1, 100);
        let TaskAdmission::Start(execution) =
            stores.request_task(key.clone(), original, 0).unwrap()
        else {
            panic!("profiled start")
        };
        let pv = stores
            .append_cpu_pv(&position, &moves("e2e4 e7e5"))
            .unwrap();
        let mut raw = profiled_observation(state, execution, pv, requested);
        raw.checker_work = Some(CheckerWork::default());
        raw.budget = 0;
        raw.external_report.as_mut().unwrap().work = CheckerWork::default();
        let unknown = stores.append_observation(raw.clone()).unwrap();
        let mut legacy = raw.clone();
        legacy.scope = EvidenceScope::ExternalUci {
            requested_depth: 0,
            reported_depth: Some(1),
            seldepth: Some(3),
            bound: ExternalBound::Lower,
        };
        assert!(legacy.validate().is_err());
        let mut legacy_key = key.clone();
        legacy_key.question = TaskQuestion::AnalyzePosition;
        assert!(
            stores
                .request_task(legacy_key, consumer(&stores, root, 2, 100), 0)
                .is_err()
        );
        let mut wrong_question = raw.clone();
        wrong_question.scope = EvidenceScope::ProfiledExternalUci {
            requested: ProfiledExternalConstraint::Nodes {
                reported_nodes: 100,
            },
            issued: ProfiledExternalConstraint::Nodes {
                reported_nodes: 100,
            },
            reported_depth: Some(1),
            seldepth: Some(3),
            bound: ExternalBound::Lower,
        };
        assert!(stores.append_observation(wrong_question).is_err());
        stores.tasks.fail(execution).unwrap();
        assert_eq!(
            stores
                .hot_consume_sealed_profiled_foreign_feedback(
                    execution,
                    original,
                    &key,
                    &position.snapshot(),
                    unknown,
                    1
                )
                .unwrap(),
            unknown
        );
        stores.hot_focus_actual_moves(position.snapshot()).unwrap();
        assert_eq!(
            stores.hot_validate_consumed_profiled_foreign_feedback(
                execution,
                original,
                &key,
                &position.snapshot(),
                unknown,
                2
            ),
            Err(StoreError::StaleConsumer)
        );
    }

    #[test]
    fn external_bestmove_preserves_foreign_scope_and_reuses_only_the_exact_question() {
        let mut stores = PalsStores::new(StoreLimits {
            line_plies: 1,
            ..StoreLimits::default()
        });
        let position = Position::startpos();
        let root = stores.focus_actual_moves(position.snapshot()).unwrap();
        let state = stores.situations.get(root).unwrap().state;
        let execution = match stores
            .request_task(external_key(state), consumer(&stores, root, 1, 100), 0)
            .unwrap()
        {
            TaskAdmission::Start(id) => id,
            _ => panic!("start"),
        };
        let pv = stores.append_cpu_pv(&position, &moves("e2e4")).unwrap();
        let record = external_observation(state, execution, pv);
        let id = stores.append_observation(record.clone()).unwrap();
        // reported depth 1 < requested 4 is deliberately not an own depth gate.
        stores.complete_task(execution, id).unwrap();
        assert!(
            matches!(stores.request_task(external_key(state), consumer(&stores, root, 2, 100), 1).unwrap(), TaskAdmission::Reuse { observation, .. } if observation == id)
        );
        assert_eq!(stores.consume_task(execution, 2, 2).unwrap(), id);
        assert_eq!(
            stores
                .observations
                .get(id)
                .unwrap()
                .external_report
                .as_ref()
                .unwrap()
                .pv
                .len(),
            2
        );
        let mut changed = external_key(state);
        changed.node_budget += 1;
        let other = match stores
            .request_task(changed, consumer(&stores, root, 3, 100), 1)
            .unwrap()
        {
            TaskAdmission::Start(id) => id,
            _ => panic!("distinct budget must start"),
        };
        let mut duplicate = record.clone();
        duplicate.execution = Some(other);
        assert!(stores.append_observation(duplicate.clone()).is_err());
        let mut wrong_identity = record.clone();
        wrong_identity
            .external_report
            .as_mut()
            .unwrap()
            .identity
            .binary_sha256 = "c".repeat(64);
        assert!(stores.append_observation(wrong_identity).is_err());
        let mut illegal_tail = record;
        illegal_tail.external_report.as_mut().unwrap().pv[1] = BoardMove::from_uci("e7e4").unwrap();
        assert!(stores.append_observation(illegal_tail).is_err());
        let unchecked_duplicate = stores.observations.append(duplicate).unwrap();
        assert!(stores.complete_task(other, unchecked_duplicate).is_err());
    }

    #[test]
    fn unknown_or_stopped_foreign_work_is_preserved_without_consumption_or_resume() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let position = Position::startpos();
        let root = stores.focus_actual_moves(position.snapshot()).unwrap();
        let state = stores.situations.get(root).unwrap().state;
        let execution = match stores
            .request_task(external_key(state), consumer(&stores, root, 1, 100), 0)
            .unwrap()
        {
            TaskAdmission::Start(id) => id,
            _ => panic!("start"),
        };
        let pv = stores
            .append_cpu_pv(&position, &moves("e2e4 e7e5"))
            .unwrap();
        let mut record = external_observation(state, execution, pv);
        record.checker_work = Some(CheckerWork::default());
        record.budget = 0;
        record.external_report.as_mut().unwrap().work = CheckerWork::default();
        let unknown = stores.append_observation(record.clone()).unwrap();
        assert!(stores.complete_task(execution, unknown).is_err());
        assert!(stores.pause_task(execution, 7, Some(unknown)).is_err());
        assert!(stores.tasks.pause(execution, 7, Some(unknown)).is_err());
        record.checker_work = Some(CheckerWork {
            nodes: Some(20),
            ..CheckerWork::default()
        });
        record.budget = 20;
        let report = record.external_report.as_mut().unwrap();
        report.work = record.checker_work.unwrap();
        report.completion = ExternalCompletion::StoppedDeadline;
        let stopped = stores.append_observation(record).unwrap();
        assert!(stores.complete_task(execution, stopped).is_err());
        stores.tasks.fail(execution).unwrap();
        assert_eq!(
            stores.consume_task(execution, 1, 2),
            Err(StoreError::NotCompleted)
        );
        assert_eq!(
            stores
                .observations
                .get(unknown)
                .unwrap()
                .checker_work
                .unwrap()
                .nodes,
            None
        );
        assert!(matches!(
            stores
                .request_task(external_key(state), consumer(&stores, root, 2, 100), 2)
                .unwrap(),
            TaskAdmission::Start(_)
        ));
    }

    #[test]
    fn raw_table_completion_cannot_authorize_foreign_partial_consumption_or_reuse() {
        for (nodes, completion) in [
            (None, ExternalCompletion::BestMove),
            (Some(20), ExternalCompletion::StoppedDeadline),
        ] {
            let mut stores = PalsStores::new(StoreLimits::default());
            let position = Position::startpos();
            let root = stores.focus_actual_moves(position.snapshot()).unwrap();
            let state = stores.situations.get(root).unwrap().state;
            let execution = match stores
                .request_task(external_key(state), consumer(&stores, root, 1, 100), 0)
                .unwrap()
            {
                TaskAdmission::Start(id) => id,
                _ => panic!("start"),
            };
            let pv = stores
                .append_cpu_pv(&position, &moves("e2e4 e7e5"))
                .unwrap();
            let mut record = external_observation(state, execution, pv);
            record.checker_work.as_mut().unwrap().nodes = nodes;
            record.budget = nodes.unwrap_or(0);
            let report = record.external_report.as_mut().unwrap();
            report.work.nodes = nodes;
            report.completion = completion;
            let partial = stores.append_observation(record).unwrap();
            assert!(stores.complete_task(execution, partial).is_err());
            stores.tasks.complete(execution, partial).unwrap();
            assert!(stores.consume_task(execution, 1, 2).is_err());
            assert!(
                stores
                    .request_task(external_key(state), consumer(&stores, root, 2, 100), 2)
                    .is_err()
            );
            assert_eq!(stores.tasks.consumer_count(), 1);
        }
    }

    #[test]
    fn model_wdl_requires_full_identity_and_actual_input_key() {
        let mut record = observation(StateId(0), None, ObservationKind::Proposal);
        record.score = RawScore::Wdl {
            win: 0.5,
            draw: 0.25,
            loss: 0.25,
            perspective: Color::White,
        };
        let mut store = ObservationStore::new(4);
        assert!(store.append(record.clone()).is_err());
        record.value_identity = None;
        record.cpu_condition = None;
        record.scope = EvidenceScope::Model {
            model: 1,
            encoding: 1,
            input: 1,
        };
        record.score = RawScore::Wdl {
            win: 0.5,
            draw: 0.25,
            loss: 0.25,
            perspective: Color::White,
        };
        assert!(store.append(record.clone()).is_err());
        record.model_value_identity = Some(super::super::value::ModelValueIdentity {
            semantics: super::super::value::MODEL_WDL_VALUE_SEMANTICS.into(),
            model: "fixture-model".into(),
            encoding: "entity-fixture/1".into(),
            precision: "f32".into(),
            model_epoch: [1; 32],
        });
        assert!(store.append(record.clone()).is_err());
        record.model_value_input = Some([2; 32]);
        assert!(store.append(record.clone()).is_ok());
        record.model_value_identity.as_mut().unwrap().encoding = String::with_capacity(257);
        record
            .model_value_identity
            .as_mut()
            .unwrap()
            .encoding
            .push_str("short");
        assert!(store.append(record).is_err());
    }

    #[test]
    fn cpu_checkpoint_reuse_requires_the_complete_typed_value_namespace() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let root = stores
            .focus_actual_moves(Position::startpos().snapshot())
            .unwrap();
        let state = stores.situations.get(root).unwrap().state;
        let original_key = key(state);
        let requester = consumer(&stores, root, 1, 100);
        let original = match stores
            .request_task(original_key.clone(), requester, 0)
            .unwrap()
        {
            TaskAdmission::Start(id) => id,
            _ => panic!("start"),
        };
        stores.pause_task(original, 7, None).unwrap();
        let mut learned = float_value_identity('1');
        learned.training = crate::cpu_value::CpuTrainingState::Learned {
            run_id: "declared-run".into(),
            steps: 1,
            dataset_sha256: "3".repeat(64),
        };
        for (index, identity) in [
            float_value_identity('1'),
            float_value_identity('2'),
            learned,
        ]
        .into_iter()
        .enumerate()
        {
            let mut changed = original_key.clone();
            changed.value_identity = Some(identity);
            let requester = consumer(&stores, root, index as u64 + 2, 100);
            assert!(matches!(
                stores.request_task(changed, requester, 1).unwrap(),
                TaskAdmission::Start(_)
            ));
        }
        let requester = consumer(&stores, root, 5, 100);
        assert!(matches!(
            stores.request_task(original_key, requester, 1).unwrap(),
            TaskAdmission::Resume { previous, checkpoint: 7, .. } if previous == original
        ));
        let mut changed_conditions = key(state);
        changed_conditions.cpu_condition = Some("fixture-cpu-search-conditions-v2".into());
        let requester = consumer(&stores, root, 6, 100);
        assert!(matches!(
            stores
                .request_task(changed_conditions, requester, 1)
                .unwrap(),
            TaskAdmission::Start(_)
        ));
    }

    #[test]
    fn wrong_cpu_namespace_cannot_publish_complete_or_pause_a_task() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let root = stores
            .focus_actual_moves(Position::startpos().snapshot())
            .unwrap();
        let state = stores.situations.get(root).unwrap().state;
        let requester = consumer(&stores, root, 1, 100);
        let execution = match stores.request_task(key(state), requester, 0).unwrap() {
            TaskAdmission::Start(id) => id,
            _ => panic!("start"),
        };
        let mut wrong = observation(state, None, ObservationKind::CpuAnalysis);
        wrong.execution = Some(execution);
        wrong.value_identity = Some(float_value_identity('1'));
        assert!(stores.append_observation(wrong.clone()).is_err());
        let unchecked = stores.observations.append(wrong).unwrap();
        assert!(stores.complete_task(execution, unchecked).is_err());
        assert!(stores.pause_task(execution, 7, Some(unchecked)).is_err());
        assert_eq!(
            stores.tasks.get(execution).unwrap().status,
            TaskStatus::InFlight
        );
        let mut missing = key(state);
        missing.value_identity = None;
        let requester = consumer(&stores, root, 2, 100);
        assert!(stores.request_task(missing, requester, 1).is_err());
        let mut wrong_conditions = observation(state, None, ObservationKind::CpuAnalysis);
        wrong_conditions.execution = Some(execution);
        wrong_conditions.cpu_condition = Some("different-cpu-search-conditions".into());
        assert!(stores.append_observation(wrong_conditions.clone()).is_err());
        let unchecked = stores.observations.append(wrong_conditions).unwrap();
        assert!(stores.complete_task(execution, unchecked).is_err());
        assert!(stores.pause_task(execution, 7, Some(unchecked)).is_err());
        for invalid_conditions in [String::new(), "\n".into(), "a".repeat(2049)] {
            let mut invalid = key(state);
            invalid.cpu_condition = Some(invalid_conditions);
            let requester = consumer(&stores, root, 2, 100);
            assert!(stores.request_task(invalid, requester, 1).is_err());
        }
        let mut oversized_allocation = String::with_capacity(4096);
        oversized_allocation.push_str("short");
        let mut invalid = key(state);
        let mut oversized_key = String::with_capacity(4096);
        oversized_key.push_str("short");
        invalid.cpu_condition = Some(oversized_key);
        let requester = consumer(&stores, root, 2, 100);
        assert!(stores.request_task(invalid, requester, 1).is_err());
        let mut invalid = observation(state, None, ObservationKind::CpuAnalysis);
        invalid.cpu_condition = Some(oversized_allocation);
        assert!(stores.append_observation(invalid).is_err());
    }

    #[test]
    fn paused_cpu_evidence_obeys_epoch_budget_profile_and_conditions() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let root = stores
            .focus_actual_moves(Position::startpos().snapshot())
            .unwrap();
        let state = stores.situations.get(root).unwrap().state;
        let requester = consumer(&stores, root, 1, 100);
        let execution = match stores.request_task(key(state), requester, 0).unwrap() {
            TaskAdmission::Start(id) => id,
            _ => panic!("start"),
        };
        let mut partial = observation(state, None, ObservationKind::CpuAnalysis);
        partial.execution = Some(execution);
        partial.scope = EvidenceScope::DepthLimited {
            depth: 1,
            profile: 1,
            condition: 1,
        };
        partial.budget = 20;
        for (epoch, budget, profile, condition) in
            [(2, 20, 1, 1), (1, 129, 1, 1), (1, 20, 2, 1), (1, 20, 1, 2)]
        {
            let mut invalid = partial.clone();
            invalid.epoch = epoch;
            invalid.budget = budget;
            invalid.scope = EvidenceScope::DepthLimited {
                depth: 1,
                profile,
                condition,
            };
            let stored = stores.append_observation(invalid).unwrap();
            assert!(stores.pause_task(execution, 7, Some(stored)).is_err());
            assert_eq!(
                stores.tasks.get(execution).unwrap().status,
                TaskStatus::InFlight
            );
        }
        let stored = stores.append_observation(partial).unwrap();
        stores.pause_task(execution, 7, Some(stored)).unwrap();
        assert!(matches!(
            stores.tasks.get(execution).unwrap().status,
            TaskStatus::Paused { checkpoint: 7, evidence: Some(id) } if id == stored
        ));
    }

    #[test]
    fn cpu_pv_publication_preserves_the_admitted_root_restriction() {
        let position = Position::startpos();
        let mut stores = PalsStores::new(StoreLimits::default());
        let root = stores.focus_actual_moves(position.snapshot()).unwrap();
        let state = stores.situations.get(root).unwrap().state;
        let mut query = key(state);
        query.question = TaskQuestion::AnalyzeRootMoves;
        let requester = consumer(&stores, root, 1, 100);
        assert!(stores.request_task(query.clone(), requester, 0).is_err());
        query.root_moves = vec![Move16::pack(BoardMove::from_uci("e2e4").unwrap()).unwrap()];
        let requester = consumer(&stores, root, 1, 100);
        let execution = match stores.request_task(query, requester, 0).unwrap() {
            TaskAdmission::Start(id) => id,
            _ => panic!("start"),
        };
        let outside = stores.append_cpu_pv(&position, &moves("d2d4")).unwrap();
        let mut record = observation(state, None, ObservationKind::CpuAnalysis);
        record.execution = Some(execution);
        record.cpu_pv = Some(outside);
        assert!(stores.append_observation(record.clone()).is_err());
        let stored = stores.observations.append(record.clone()).unwrap();
        assert!(stores.complete_task(execution, stored).is_err());
        assert!(stores.pause_task(execution, 7, Some(stored)).is_err());
        record.cpu_pv = Some(stores.append_cpu_pv(&position, &moves("e2e4")).unwrap());
        let stored = stores.append_observation(record).unwrap();
        stores.complete_task(execution, stored).unwrap();
    }

    #[test]
    fn cpu_pv_handles_require_rules_replay_from_the_exact_registered_state() {
        let position = Position::startpos();
        let mut stores = PalsStores::new(StoreLimits::default());
        let root = stores.focus_actual_moves(position.snapshot()).unwrap();
        let state = stores.situations.get(root).unwrap().state;
        let legal = moves("e2e4 e7e5 g1f3");
        let checked = stores.append_cpu_pv(&position, &legal).unwrap();
        assert_eq!(stores.lines.moves(checked).unwrap(), legal);
        assert_eq!(stores.append_cpu_pv(&position, &legal).unwrap(), checked);
        let mut record = observation(state, None, ObservationKind::CpuAnalysis);
        record.cpu_pv = Some(checked);
        stores.append_observation(record.clone()).unwrap();
        let lines_before = stores.lines.len();
        assert!(stores.append_cpu_pv(&position, &moves("e2e5")).is_err());
        assert!(
            stores
                .append_cpu_pv(&position, &moves("e2e4 e7e4"))
                .is_err()
        );
        assert_eq!(stores.lines.len(), lines_before);
        let mut played = position;
        played
            .apply_uci_moves(&["g1f3", "g8f6", "f3g1", "f6g8"])
            .unwrap();
        stores.insert_situation(played.snapshot()).unwrap();
        let missing_history = Position::from_fen(&played.to_fen()).unwrap();
        assert!(
            stores
                .append_cpu_pv(&missing_history, &moves("e2e4"))
                .is_err()
        );
        let foreign_root = stores.lines.root(state).unwrap();
        let unverified = stores.lines.append(foreign_root, &moves("d2d5")).unwrap();
        record.cpu_pv = Some(unverified);
        assert!(stores.append_observation(record).is_err());
    }

    #[test]
    fn cpu_pv_attestation_is_rechecked_at_completion_and_survives_no_pool_reissue() {
        let position = Position::startpos();
        let mut stores = PalsStores::new(StoreLimits::default());
        let root = stores.focus_actual_moves(position.snapshot()).unwrap();
        let state = stores.situations.get(root).unwrap().state;
        let requester = consumer(&stores, root, 1, 100);
        let execution = match stores.request_task(key(state), requester, 0).unwrap() {
            TaskAdmission::Start(id) => id,
            _ => panic!("start"),
        };
        let checked = stores.append_cpu_pv(&position, &moves("e2e4")).unwrap();
        let mut record = observation(state, None, ObservationKind::CpuAnalysis);
        record.execution = Some(execution);
        record.cpu_pv = Some(checked);
        let verified = stores.append_observation(record.clone()).unwrap();
        stores.complete_task(execution, verified).unwrap();
        let requester = consumer(&stores, root, 2, 100);
        assert!(matches!(
            stores.request_task(key(state), requester, 1).unwrap(),
            TaskAdmission::Reuse { observation, .. } if observation == verified
        ));
        assert_eq!(
            stores.observations.get(verified).unwrap().cpu_pv,
            Some(checked)
        );
        stores.lines = LinePool::new(32, 256);
        let reissued_root = stores.lines.root(state).unwrap();
        let reissued = stores.lines.append(reissued_root, &moves("d2d4")).unwrap();
        assert_eq!(reissued, checked);
        assert_eq!(
            stores.append_cpu_pv(&position, &moves("d2d4")),
            Err(StoreError::InvalidEvidence(
                "checked CPU PV handle cannot be reissued by another store",
            )),
        );
        record.cpu_pv = Some(reissued);
        assert!(stores.append_observation(record.clone()).is_err());
        assert_eq!(
            stores.consume_task(execution, 2, 2),
            Err(StoreError::InvalidEvidence(
                "CPU PV lacks matching exact-state Rules replay",
            )),
        );
        let requester = consumer(&stores, root, 3, 100);
        let mut changed = key(state);
        changed.requested_depth = 5;
        let other = match stores.request_task(changed, requester, 1).unwrap() {
            TaskAdmission::Start(id) => id,
            _ => panic!("start"),
        };
        record.execution = Some(other);
        record.scope = EvidenceScope::DepthLimited {
            depth: 5,
            profile: 1,
            condition: 1,
        };
        let unchecked = stores.observations.append(record).unwrap();
        assert_eq!(
            stores.complete_task(other, unchecked),
            Err(StoreError::InvalidEvidence(
                "CPU PV lacks matching exact-state Rules replay",
            )),
        );
        assert_eq!(
            stores.pause_task(other, 7, Some(unchecked)),
            Err(StoreError::InvalidEvidence(
                "CPU PV lacks matching exact-state Rules replay",
            )),
        );
        assert_eq!(
            stores.tasks.get(other).unwrap().status,
            TaskStatus::InFlight
        );
    }

    #[test]
    fn checked_cpu_pv_cannot_rebind_to_a_reissued_state_store() {
        let position = Position::startpos();
        let mut stores = PalsStores::new(StoreLimits::default());
        let root = stores.focus_actual_moves(position.snapshot()).unwrap();
        let state = stores.situations.get(root).unwrap().state;
        let checked = stores.append_cpu_pv(&position, &moves("e2e4")).unwrap();
        let mut record = observation(state, None, ObservationKind::CpuAnalysis);
        record.cpu_pv = Some(checked);
        stores.append_observation(record.clone()).unwrap();
        stores.states = StateStore::new(32, 1024 * 1024);
        let reissued = stores.states.insert(position.snapshot()).unwrap();
        assert_eq!(reissued, state);
        assert_eq!(
            stores.append_cpu_pv(&position, &moves("e2e4")),
            Err(StoreError::InvalidEvidence(
                "checked CPU PV handle cannot be reissued by another store",
            )),
        );
        assert!(stores.append_observation(record).is_err());
    }

    fn consumer(
        stores: &PalsStores,
        situation: SituationId,
        id: u64,
        deadline_tick: u64,
    ) -> TaskConsumer {
        TaskConsumer {
            id,
            situation,
            revision: stores.situations.get(situation).unwrap().revision,
            generation: stores.generation(),
            deadline_tick,
        }
    }

    #[test]
    fn move16_roundtrips_castling_ep_and_all_promotions_without_claiming_legality() {
        for text in ["e1g1", "e5d6", "a7a8q", "a7a8r", "a7a8b", "a7a8n"] {
            let mv = BoardMove::from_uci(text).unwrap();
            assert_eq!(Move16::pack(mv).unwrap().unpack().unwrap(), mv);
        }
        assert!(Move16(0xffff).unpack().is_err());
        assert!(Move16(0).unpack().is_err());
    }

    #[test]
    fn same_board_with_different_rule_history_is_not_the_same_state() {
        let mut played = Position::startpos();
        played
            .apply_uci_moves(&["g1f3", "g8f6", "f3g1", "f6g8"])
            .unwrap();
        let missing_history = Position::from_fen(&played.to_fen()).unwrap();
        assert_eq!(played.to_fen(), missing_history.to_fen());
        let mut states = StateStore::new(4, 1024 * 1024);
        let a = states.insert(played.snapshot()).unwrap();
        let b = states.insert(missing_history.snapshot()).unwrap();
        assert_ne!(a, b);
        assert_eq!(states.insert(played.snapshot()).unwrap(), a);
        assert_eq!(states.len(), 2);
    }

    #[test]
    fn immutable_chunks_share_prefix_and_reject_partial_admission() {
        let mut pool = LinePool::new(4, 64);
        let root = pool.root(StateId(0)).unwrap();
        let prefix = pool.append(root, &moves("e2e4 e7e5")).unwrap();
        let old = pool.append(prefix, &moves("g1f3 b8c6")).unwrap();
        let repaired = pool.append(prefix, &moves("f1c4 g8f6")).unwrap();
        assert_eq!(pool.get(old).unwrap().parent, Some(prefix));
        assert_eq!(pool.get(repaired).unwrap().parent, Some(prefix));
        assert_eq!(pool.append(prefix, &moves("g1f3 b8c6")).unwrap(), old);
        let count = pool.len();
        assert!(pool.append(repaired, &moves("d2d3")).is_err());
        assert_eq!(pool.len(), count);
        assert_eq!(pool.moves(old).unwrap(), moves("e2e4 e7e5 g1f3 b8c6"));
    }

    #[test]
    fn refutation_then_repair_preserves_the_same_first_move() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let situation = stores
            .focus_actual_moves(Position::startpos().snapshot())
            .unwrap();
        let current = stores.situations.get(situation).unwrap().clone();
        let old = stores
            .append_line(current.focus, &moves("e2e4 e7e5 g1f3"))
            .unwrap();
        let new = stores
            .append_line(current.focus, &moves("e2e4 c7c5 g1f3"))
            .unwrap();
        let refutation = stores
            .append_observation(observation(
                current.state,
                Some(old),
                ObservationKind::Refutation,
            ))
            .unwrap();
        let original_record = stores.observations.get(refutation).unwrap().clone();
        stores
            .refute_continuation(situation, old, refutation)
            .unwrap();
        let mut repair = observation(current.state, Some(new), ObservationKind::Repair);
        repair.supersedes = Some(refutation);
        let repair = stores.append_observation(repair).unwrap();
        stores.repair(situation, old, new, repair).unwrap();
        assert_eq!(
            stores.lines.first_move(old).unwrap(),
            stores.lines.first_move(new).unwrap()
        );
        let conclusions = &stores.situations.get(situation).unwrap().conclusions;
        assert_eq!(
            conclusions.get(old).unwrap().status,
            ContinuationStatus::RepairedBy(new)
        );
        assert_eq!(
            conclusions.get(new).unwrap().status,
            ContinuationStatus::Supported
        );
        assert_eq!(
            stores.observations.get(refutation).unwrap(),
            &original_record
        );
    }

    #[test]
    fn situation_generation_prevents_aba_and_stale_dependency_invalidation() {
        let mut arena = SituationArena::new(1);
        let a = arena.insert(StateId(0), LineId(0)).unwrap();
        let mut dependencies = DependencyIndex::new(4);
        dependencies.add(ObservationId(0), a).unwrap();
        arena.remove(a).unwrap();
        let b = arena.insert(StateId(1), LineId(1)).unwrap();
        assert_eq!(a.slot, b.slot);
        assert_ne!(a.generation, b.generation);
        assert!(arena.get(a).is_err());
        assert_eq!(
            dependencies
                .invalidate(ObservationId(0), &mut arena)
                .unwrap(),
            0
        );
        assert_eq!(arena.get(b).unwrap().revision, 0);
    }

    #[test]
    fn only_registered_dependents_advance_revision() {
        let mut arena = SituationArena::new(3);
        let affected = arena.insert(StateId(0), LineId(0)).unwrap();
        let unrelated = arena.insert(StateId(1), LineId(1)).unwrap();
        let mut dependencies = DependencyIndex::new(4);
        dependencies.add(ObservationId(0), affected).unwrap();
        dependencies.add(ObservationId(0), affected).unwrap();
        assert_eq!(dependencies.len(), 1);
        assert_eq!(
            dependencies
                .invalidate(ObservationId(0), &mut arena)
                .unwrap(),
            1
        );
        assert!(arena.get(affected).unwrap().dirty);
        assert_eq!(arena.get(affected).unwrap().revision, 1);
        assert!(!arena.get(unrelated).unwrap().dirty);
        assert_eq!(arena.get(unrelated).unwrap().revision, 0);
    }

    #[test]
    fn expired_consumer_cannot_pollute_new_root_but_new_consumer_can_reuse_completed_work() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let position = Position::startpos();
        let root = stores.focus_actual_moves(position.snapshot()).unwrap();
        let state = stores.situations.get(root).unwrap().state;
        let first = consumer(&stores, root, 1, 10);
        let execution = match stores.request_task(key(state), first, 0).unwrap() {
            TaskAdmission::Start(id) => id,
            _ => panic!("must start"),
        };
        let mut record = observation(state, None, ObservationKind::CpuAnalysis);
        record.execution = Some(execution);
        let evidence = stores.append_observation(record).unwrap();
        stores.complete_task(execution, evidence).unwrap();
        assert_eq!(
            stores.consume_task(execution, 1, 10),
            Err(StoreError::ExpiredConsumer)
        );
        stores.focus_actual_moves(position.snapshot()).unwrap();
        assert_eq!(
            stores.consume_task(execution, 1, 5),
            Err(StoreError::StaleConsumer)
        );
        assert_eq!(stores.situations.get(root).unwrap().revision, 0);
        let next = consumer(&stores, root, 2, 30);
        assert_eq!(
            stores.request_task(key(state), next, 12).unwrap(),
            TaskAdmission::Reuse {
                execution,
                observation: evidence
            }
        );
        assert_eq!(stores.consume_task(execution, 2, 13).unwrap(), evidence);
        assert_eq!(
            stores.consume_task(execution, 2, 14),
            Err(StoreError::AlreadyConsumed)
        );
    }

    #[test]
    fn conditions_depth_history_candidate_order_and_consumers_are_independent() {
        let mut table = TaskTable::new(4, 5, 4);
        let first = TaskConsumer {
            id: 1,
            situation: SituationId {
                slot: 0,
                generation: 0,
            },
            revision: 0,
            generation: 1,
            deadline_tick: 20,
        };
        let a = match table.request(key(StateId(0)), first, 0).unwrap() {
            TaskAdmission::Start(id) => id,
            _ => panic!("start"),
        };
        let mut second = first;
        second.id = 2;
        second.deadline_tick = 30;
        assert_eq!(
            table.request(key(StateId(0)), second, 1).unwrap(),
            TaskAdmission::Join(a)
        );
        let mut changed = key(StateId(0));
        changed.requested_depth = 10;
        let mut third = second;
        third.id = 3;
        assert!(matches!(
            table.request(changed, third, 1).unwrap(),
            TaskAdmission::Start(_)
        ));
        let mut reordered = key(StateId(0));
        reordered.root_moves = moves("e2e4 d2d4")
            .into_iter()
            .map(|mv| Move16::pack(mv).unwrap())
            .collect();
        let mut fourth = third;
        fourth.id = 4;
        let original = match table.request(reordered.clone(), fourth, 1).unwrap() {
            TaskAdmission::Start(id) => id,
            _ => panic!("start"),
        };
        reordered.root_moves.reverse();
        let mut fifth = fourth;
        fifth.id = 5;
        assert_ne!(
            table.request(reordered, fifth, 1).unwrap(),
            TaskAdmission::Join(original)
        );
        table.cancel_consumer(a, 1).unwrap();
        assert_eq!(table.get(a).unwrap().status, TaskStatus::InFlight);
        table.cancel_consumer(a, 2).unwrap();
        assert_eq!(
            table.get(a).unwrap().status,
            TaskStatus::CancellationRequested
        );
    }

    #[test]
    fn completed_tasks_and_unknown_evidence_survive_derived_cache_eviction() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let root = stores
            .focus_actual_moves(Position::startpos().snapshot())
            .unwrap();
        let state = stores.situations.get(root).unwrap().state;
        let requester = consumer(&stores, root, 1, 100);
        let execution = match stores.request_task(key(state), requester, 0).unwrap() {
            TaskAdmission::Start(id) => id,
            _ => panic!("start"),
        };
        let mut record = observation(state, None, ObservationKind::CpuAnalysis);
        record.execution = Some(execution);
        let evidence = stores.append_observation(record).unwrap();
        stores.complete_task(execution, evidence).unwrap();
        let cache_key = DerivedCacheKey {
            state,
            line: None,
            model: 1,
            epoch: 1,
            encoding: 1,
            mask: 1,
            precision: 32,
        };
        stores.derived.insert(cache_key, vec![1, 2, 3, 4]).unwrap();
        assert_eq!(stores.evict_derived_cache(), 4);
        assert!(stores.derived.get(&cache_key).is_none());
        assert_eq!(
            stores.tasks.get(execution).unwrap().status,
            TaskStatus::Completed(evidence)
        );
        assert_eq!(
            stores.observations.get(evidence).unwrap().score,
            RawScore::Unknown
        );
        assert_eq!(stores.consume_task(execution, 1, 50).unwrap(), evidence);
    }

    #[test]
    fn bounded_pools_keep_records_on_admission_failure() {
        let limits = StoreLimits {
            states: 1,
            observations: 1,
            derived_entries: 1,
            derived_bytes: 2,
            ..StoreLimits::default()
        };
        let mut stores = PalsStores::new(limits);
        let root = stores
            .focus_actual_moves(Position::startpos().snapshot())
            .unwrap();
        let state = stores.situations.get(root).unwrap().state;
        let id = stores
            .append_observation(observation(state, None, ObservationKind::CpuAnalysis))
            .unwrap();
        assert!(
            stores
                .append_observation(observation(state, None, ObservationKind::Proposal))
                .is_err()
        );
        assert_eq!(
            stores.observations.get(id).unwrap().score,
            RawScore::Unknown
        );
        let mut played = Position::startpos();
        played.make_uci("e2e4").unwrap();
        assert!(stores.focus_actual_moves(played.snapshot()).is_err());
        assert_eq!(stores.root(), Some(root));
        assert_eq!(stores.generation(), 1);
        let cache_key = DerivedCacheKey {
            state,
            line: None,
            model: 1,
            epoch: 1,
            encoding: 1,
            mask: 1,
            precision: 32,
        };
        stores.derived.insert(cache_key, vec![1, 2]).unwrap();
        assert!(stores.derived.insert(cache_key, vec![3, 4, 5]).is_err());
        assert_eq!(stores.derived.get(&cache_key), Some([1, 2].as_slice()));
    }

    #[test]
    fn finite_search_exact_bound_cannot_be_published_as_a_rules_terminal() {
        let mut observations = ObservationStore::new(2);
        let mut record = observation(StateId(0), None, ObservationKind::CpuAnalysis);
        record.score = RawScore::Cpu {
            value: 9_999,
            perspective: Color::White,
            bound: BoundKind::ExactWithinSearch,
        };
        assert!(observations.append(record.clone()).is_ok());
        record.score = RawScore::Terminal {
            winner: Some(Color::White),
        };
        assert_eq!(
            observations.append(record),
            Err(StoreError::InvalidEvidence("unverified terminal claim"))
        );
    }

    #[test]
    fn partial_depth_is_not_complete_and_resume_issues_a_new_execution() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let root = stores
            .focus_actual_moves(Position::startpos().snapshot())
            .unwrap();
        let state = stores.situations.get(root).unwrap().state;
        let mut query = key(state);
        query.requested_depth = 10;
        let requester = consumer(&stores, root, 1, 100);
        let original = match stores.request_task(query.clone(), requester, 0).unwrap() {
            TaskAdmission::Start(id) => id,
            _ => panic!("start"),
        };
        let mut record = observation(state, None, ObservationKind::CpuAnalysis);
        record.execution = Some(original);
        let partial = stores.append_observation(record).unwrap();
        assert_eq!(
            stores.complete_task(original, partial),
            Err(StoreError::InvalidEvidence(
                "completed CPU scope does not satisfy task"
            ))
        );
        stores.pause_task(original, 7, Some(partial)).unwrap();
        stores.evict_derived_cache();
        let requester = consumer(&stores, root, 2, 100);
        let resumed = match stores.request_task(query.clone(), requester, 1).unwrap() {
            TaskAdmission::Resume {
                execution,
                previous,
                checkpoint,
            } => {
                assert_eq!(previous, original);
                assert_eq!(checkpoint, 7);
                execution
            }
            _ => panic!("must resume same exact task conditions"),
        };
        assert_ne!(original, resumed);
        assert_eq!(
            stores.tasks.get(resumed).unwrap().resumed_from,
            Some(original)
        );
        assert!(matches!(
            stores.tasks.get(original).unwrap().status,
            TaskStatus::Paused { .. }
        ));
        query.profile = 2;
        let requester = consumer(&stores, root, 3, 100);
        assert!(matches!(
            stores.request_task(query, requester, 1).unwrap(),
            TaskAdmission::Start(_)
        ));
        assert_eq!(
            stores.consume_task(original, 1, 2),
            Err(StoreError::NotCompleted)
        );
    }

    #[test]
    fn mismatched_profile_or_model_epoch_does_not_complete_matching_board_task() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let root = stores
            .focus_actual_moves(Position::startpos().snapshot())
            .unwrap();
        let state = stores.situations.get(root).unwrap().state;
        let requester = consumer(&stores, root, 1, 100);
        let execution = match stores.request_task(key(state), requester, 0).unwrap() {
            TaskAdmission::Start(id) => id,
            _ => panic!("start"),
        };
        let mut record = observation(state, None, ObservationKind::CpuAnalysis);
        record.execution = Some(execution);
        record.scope = EvidenceScope::DepthLimited {
            depth: 4,
            profile: 2,
            condition: 1,
        };
        let mismatch = stores.append_observation(record.clone()).unwrap();
        assert_eq!(
            stores.complete_task(execution, mismatch),
            Err(StoreError::InvalidEvidence(
                "completed CPU scope does not satisfy task"
            ))
        );
        record.scope = EvidenceScope::DepthLimited {
            depth: 4,
            profile: 1,
            condition: 1,
        };
        record.epoch = 2;
        let mismatch = stores.append_observation(record).unwrap();
        assert_eq!(
            stores.complete_task(execution, mismatch),
            Err(StoreError::InvalidEvidence(
                "completed task epoch or budget mismatch"
            ))
        );
        assert_eq!(
            stores.tasks.get(execution).unwrap().status,
            TaskStatus::InFlight
        );
    }

    #[test]
    fn chunks_never_exceed_sixteen_and_reconstruction_has_exact_order() {
        let mut pool = LinePool::new(5, 64);
        let root = pool.root(StateId(0)).unwrap();
        let suffix = moves("e2e4 e7e5")
            .into_iter()
            .cycle()
            .take(33)
            .collect::<Vec<_>>();
        let line = pool.append(root, &suffix).unwrap();
        assert_eq!(pool.moves(line).unwrap(), suffix);
        assert_eq!(pool.get(line).unwrap().plies(), 33);
        let mut current = Some(line);
        while let Some(id) = current {
            let chunk = pool.get(id).unwrap();
            assert!(chunk.chunk_moves().len() <= MOVES_PER_CHUNK);
            current = chunk.parent;
        }
    }

    #[test]
    fn focused_exact_analysis_situation_is_retained_after_actual_move() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let parent = Position::startpos();
        let old_root = stores.focus_actual_moves(parent.snapshot()).unwrap();
        let child = parent
            .preview_move(BoardMove::from_uci("e2e4").unwrap())
            .unwrap();
        let analyzed = stores.insert_situation(child.snapshot()).unwrap();
        let state = stores.situations.get(analyzed).unwrap().state;
        let evidence = stores
            .append_observation(observation(state, None, ObservationKind::CpuAnalysis))
            .unwrap();
        assert_eq!(
            stores.focus_actual_moves(child.snapshot()).unwrap(),
            analyzed
        );
        assert_ne!(stores.root(), Some(old_root));
        assert_eq!(stores.observations.get(evidence).unwrap().state, state);
        assert!(stores.situations.get(old_root).is_ok());
        assert_eq!(stores.generation(), 2);
    }

    #[test]
    fn facade_mints_exact_terminal_only_from_rules_and_claimable_draw_stays_ongoing() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let ongoing = Position::startpos();
        let root = stores.focus_actual_moves(ongoing.snapshot()).unwrap();
        let state = stores.situations.get(root).unwrap().state;
        let mut fabricated = observation(state, None, ObservationKind::ExactTerminal);
        fabricated.scope = EvidenceScope::RulesTerminal;
        fabricated.score = RawScore::Terminal {
            winner: Some(Color::White),
        };
        assert_eq!(
            stores.append_observation(fabricated),
            Err(StoreError::InvalidEvidence(
                "Rules terminal must be minted from a checked Position"
            ))
        );
        assert_eq!(
            stores.append_rules_terminal(&ongoing, 1, 1),
            Err(StoreError::InvalidEvidence(
                "Rules position is not terminal"
            ))
        );
        let claimable = Position::from_fen("7k/8/8/8/8/8/Q7/K7 w - - 100 1").unwrap();
        assert_eq!(
            stores.append_rules_terminal(&claimable, 1, 1),
            Err(StoreError::InvalidEvidence(
                "Rules position is not terminal"
            ))
        );
        let mate = Position::from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        let id = stores.append_rules_terminal(&mate, 1, 1).unwrap();
        assert_eq!(
            stores.observations.get(id).unwrap().score,
            RawScore::Terminal {
                winner: Some(Color::White)
            }
        );
        let stale = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        let id = stores.append_rules_terminal(&stale, 1, 1).unwrap();
        assert_eq!(
            stores.observations.get(id).unwrap().score,
            RawScore::Terminal { winner: None }
        );
    }

    #[test]
    fn large_spare_owned_capacity_cannot_bypass_payload_limits() {
        let mut cache = DerivedCache::new(1, 2);
        let mut spare = Vec::with_capacity(512);
        spare.push(1);
        let key = DerivedCacheKey {
            state: StateId(0),
            line: None,
            model: 1,
            epoch: 1,
            encoding: 1,
            mask: 1,
            precision: 32,
        };
        assert_eq!(
            cache.insert(key, spare),
            Err(StoreError::Capacity("derived cache bytes"))
        );
        assert_eq!(cache.bytes(), 0);
        let mut table = TaskTable::new(1, 1, 4);
        let mut query = super::tests::key(StateId(0));
        query.root_moves = Vec::with_capacity(512);
        let consumer = TaskConsumer {
            id: 1,
            situation: SituationId {
                slot: 0,
                generation: 0,
            },
            revision: 0,
            generation: 1,
            deadline_tick: 20,
        };
        assert_eq!(
            table.request(query, consumer, 0),
            Err(StoreError::Capacity("task root moves"))
        );
        assert_eq!(table.len(), 0);
    }
}
