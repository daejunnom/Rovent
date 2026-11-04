//! Versioned archive DTOs preserve foreign score units and exact Rules traces.
use super::*;
use serde::{Deserialize, Serialize};

/// Stream the unchanged archive DTO into the caller's already-bounded writer.
/// This function opens no files and does not grant a cold read or write scope.
pub(super) fn write_retirement_wire<W: Write + ?Sized, T: Serialize + ?Sized>(
    writer: &mut W,
    wire: &T,
) -> Result<(), StoreError> {
    serde_json::to_writer(writer, wire).map_err(|error| match error.io_error_kind() {
        Some(kind) => StoreError::ArchiveIo {
            stage: "retirement record serialization",
            kind,
        },
        None => StoreError::ArchiveIntegrity("retirement record serialization"),
    })
}

pub(super) fn write_retirement_external_report<W: Write + ?Sized>(
    report: &ExternalCheckerReport,
    writer: &mut W,
) -> Result<(), StoreError> {
    write_retirement_wire(writer, &ReportWire::encode(report)?)
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Bundle {
    pub version: u32,
    pub rules_version: String,
    pub variant: String,
    pub generation: u64,
    pub states: Vec<StateWire>,
    pub lines: Vec<LineWire>,
    pub observations: Vec<(ObservationId, ObservationWire)>,
    pub situations: Vec<(SituationId, Situation)>,
    pub dependencies: Vec<(ObservationId, Vec<SituationId>)>,
    pub tasks: Vec<(ExecutionId, TaskRecord)>,
    pub checked_pvs: BTreeSet<LineId>,
    pub engine_records: Vec<RoleRecordWire>,
    pub engine_nodes: Vec<EngineArchiveNode>,
}
impl Bundle {
    pub fn new(generation: u64) -> Self {
        Self {
            version: 1,
            rules_version: rz_position::RULES_VERSION.into(),
            variant: rz_position::RULES_VARIANT.into(),
            generation,
            states: vec![],
            lines: vec![],
            observations: vec![],
            situations: vec![],
            dependencies: vec![],
            tasks: vec![],
            checked_pvs: BTreeSet::new(),
            engine_records: vec![],
            engine_nodes: vec![],
        }
    }
}

#[derive(Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StateWire {
    pub id: StateId,
    origin: u8,
    start_fen: String,
    endpoint_fen: String,
    moves: Vec<Move16>,
    revision: u64,
}
impl StateWire {
    pub fn history_charge(&self) -> Result<usize, StoreError> {
        if self.moves.len() > rz_position::MAX_UCI_REPLAY_PLIES {
            return Err(StoreError::ArchiveIntegrity("Rules trace bounds"));
        }
        PositionSnapshot::retained_history_bytes_for_len(self.moves.len() + 1)
            .ok_or(StoreError::Capacity("archive history charge"))
    }
    pub fn encode(id: StateId, state: &PositionSnapshot) -> Result<Self, StoreError> {
        let replay = state
            .uci_replay(rz_position::MAX_UCI_REPLAY_PLIES)
            .map_err(|_| StoreError::InvalidEvidence("archive exact history replay"))?;
        Ok(Self {
            id,
            origin: match replay.origin {
                rz_position::HistoryOrigin::StartPosition => 0,
                rz_position::HistoryOrigin::Fen => 1,
            },
            start_fen: replay.start_fen,
            endpoint_fen: state.to_fen(),
            moves: replay
                .moves
                .into_iter()
                .map(Move16::pack)
                .collect::<Result<_, _>>()?,
            revision: state.revision(),
        })
    }
    pub fn decode(&self) -> Result<PositionSnapshot, StoreError> {
        if self.start_fen.len() > 4096
            || self.endpoint_fen.len() > 4096
            || self.moves.len() > rz_position::MAX_UCI_REPLAY_PLIES
        {
            return Err(StoreError::ArchiveIntegrity("Rules trace bounds"));
        }
        let (origin, completeness) = match self.origin {
            0 => (
                rz_position::HistoryOrigin::StartPosition,
                rz_position::HistoryCompleteness::Complete,
            ),
            1 => (
                rz_position::HistoryOrigin::Fen,
                rz_position::HistoryCompleteness::UnknownPrefix,
            ),
            _ => return Err(StoreError::ArchiveIntegrity("Rules trace origin")),
        };
        let trace = rz_position::UciReplay {
            origin,
            completeness,
            start_fen: self.start_fen.clone(),
            moves: self
                .moves
                .iter()
                .map(|m| m.unpack())
                .collect::<Result<_, _>>()?,
        };
        let snapshot = PositionSnapshot::from_uci_replay(
            &trace,
            self.revision,
            rz_position::PositionLimits {
                max_history_positions: rz_position::MAX_UCI_REPLAY_PLIES + 1,
                ..rz_position::PositionLimits::default()
            },
        )
        .map_err(|_| StoreError::ArchiveIntegrity("Rules trace replay"))?;
        if snapshot.to_fen() != self.endpoint_fen {
            return Err(StoreError::ArchiveIntegrity("Rules endpoint"));
        }
        Ok(snapshot)
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct LineWire {
    pub id: LineId,
    pub parent: Option<LineId>,
    pub state: StateId,
    moves: Vec<Move16>,
    plies: usize,
}
impl LineWire {
    pub fn encode(id: LineId, line: &LineChunk) -> Self {
        Self {
            id,
            parent: line.parent,
            state: line.start_state,
            moves: line.chunk_moves().to_vec(),
            plies: line.plies,
        }
    }
    pub fn decode(&self, ply_limit: usize) -> Result<LineChunk, StoreError> {
        if self.moves.len() > MOVES_PER_CHUNK
            || self.plies > ply_limit
            || self.parent.is_none() && (!self.moves.is_empty() || self.plies != 0)
            || self.parent.is_some_and(|parent| parent.0 >= self.id.0)
        {
            return Err(StoreError::ArchiveIntegrity("line shape or parent"));
        }
        let mut moves = [Move16(0); MOVES_PER_CHUNK];
        for (at, movement) in self.moves.iter().enumerate() {
            movement.unpack()?;
            moves[at] = *movement;
        }
        Ok(LineChunk {
            parent: self.parent,
            start_state: self.state,
            moves,
            len: self.moves.len(),
            plies: self.plies,
        })
    }
}

fn color(value: u8) -> Result<Color, StoreError> {
    match value {
        0 => Ok(Color::White),
        1 => Ok(Color::Black),
        _ => Err(StoreError::ArchiveIntegrity("color")),
    }
}
fn c(value: Color) -> u8 {
    if value == Color::White { 0 } else { 1 }
}
fn bound(value: u8) -> Result<ExternalBound, StoreError> {
    match value {
        0 => Ok(ExternalBound::ExactReported),
        1 => Ok(ExternalBound::Lower),
        2 => Ok(ExternalBound::Upper),
        3 => Ok(ExternalBound::Unknown),
        _ => Err(StoreError::ArchiveIntegrity("external bound")),
    }
}
fn b(value: ExternalBound) -> u8 {
    match value {
        ExternalBound::ExactReported => 0,
        ExternalBound::Lower => 1,
        ExternalBound::Upper => 2,
        ExternalBound::Unknown => 3,
    }
}
fn raw(value: (u8, i32)) -> Result<ExternalRawScore, StoreError> {
    match value.0 {
        0 => Ok(ExternalRawScore::Unknown),
        1 => Ok(ExternalRawScore::Centipawns(value.1)),
        2 => Ok(ExternalRawScore::MateMoves(value.1)),
        _ => Err(StoreError::ArchiveIntegrity("foreign raw units")),
    }
}
fn r(value: ExternalRawScore) -> (u8, i32) {
    match value {
        ExternalRawScore::Unknown => (0, 0),
        ExternalRawScore::Centipawns(n) => (1, n),
        ExternalRawScore::MateMoves(n) => (2, n),
    }
}
fn work(value: CheckerWork) -> [Option<u64>; 3] {
    [value.nodes, value.qnodes, value.tt_hits]
}
fn w(value: [Option<u64>; 3]) -> CheckerWork {
    CheckerWork {
        nodes: value[0],
        qnodes: value[1],
        tt_hits: value[2],
    }
}

#[derive(Debug, Serialize, Deserialize)]
enum ScopeWire {
    Rules,
    Depth(u16, u64, u64),
    External(u16, Option<u16>, Option<u16>, u8),
    Model(u64, u64, u64),
    ProfiledExternal(
        ProfiledExternalConstraint,
        ProfiledExternalConstraint,
        Option<u16>,
        Option<u16>,
        u8,
    ),
}
impl ScopeWire {
    fn encode(value: EvidenceScope) -> Self {
        match value {
            EvidenceScope::RulesTerminal => Self::Rules,
            EvidenceScope::DepthLimited {
                depth,
                profile,
                condition,
            } => Self::Depth(depth, profile, condition),
            EvidenceScope::ExternalUci {
                requested_depth,
                reported_depth,
                seldepth,
                bound,
            } => Self::External(requested_depth, reported_depth, seldepth, b(bound)),
            EvidenceScope::Model {
                model,
                encoding,
                input,
            } => Self::Model(model, encoding, input),
            EvidenceScope::ProfiledExternalUci {
                requested,
                issued,
                reported_depth,
                seldepth,
                bound,
            } => Self::ProfiledExternal(requested, issued, reported_depth, seldepth, b(bound)),
        }
    }
    fn decode(&self) -> Result<EvidenceScope, StoreError> {
        Ok(match *self {
            Self::Rules => EvidenceScope::RulesTerminal,
            Self::Depth(depth, profile, condition) => EvidenceScope::DepthLimited {
                depth,
                profile,
                condition,
            },
            Self::External(requested_depth, reported_depth, seldepth, b) => {
                EvidenceScope::ExternalUci {
                    requested_depth,
                    reported_depth,
                    seldepth,
                    bound: bound(b)?,
                }
            }
            Self::Model(model, encoding, input) => EvidenceScope::Model {
                model,
                encoding,
                input,
            },
            Self::ProfiledExternal(requested, issued, reported_depth, seldepth, b) => {
                requested.validate_issued(issued)?;
                EvidenceScope::ProfiledExternalUci {
                    requested,
                    issued,
                    reported_depth,
                    seldepth,
                    bound: bound(b)?,
                }
            }
        })
    }
}

#[cfg(test)]
mod scope_tests {
    use super::*;

    #[test]
    fn legacy_scope_wire_tags_and_bytes_stay_unchanged() {
        for (scope, bytes) in [
            (EvidenceScope::RulesTerminal, "\"Rules\""),
            (
                EvidenceScope::DepthLimited {
                    depth: 4,
                    profile: 7,
                    condition: 9,
                },
                "{\"Depth\":[4,7,9]}",
            ),
            (
                EvidenceScope::ExternalUci {
                    requested_depth: 4,
                    reported_depth: Some(1),
                    seldepth: Some(3),
                    bound: ExternalBound::Lower,
                },
                "{\"External\":[4,1,3,1]}",
            ),
            (
                EvidenceScope::Model {
                    model: 7,
                    encoding: 8,
                    input: 9,
                },
                "{\"Model\":[7,8,9]}",
            ),
        ] {
            assert_eq!(
                serde_json::to_string(&ScopeWire::encode(scope)).unwrap(),
                bytes
            );
            assert_eq!(
                serde_json::from_str::<ScopeWire>(bytes)
                    .unwrap()
                    .decode()
                    .unwrap(),
                scope
            );
        }
        assert_eq!(
            serde_json::to_string(&TaskQuestion::AnalyzePosition).unwrap(),
            "\"AnalyzePosition\""
        );
        assert_eq!(
            serde_json::to_string(&TaskQuestion::FindAlternative { divergence_ply: 3 }).unwrap(),
            "{\"FindAlternative\":{\"divergence_ply\":3}}"
        );
    }

    #[test]
    fn profiled_scope_has_an_explicit_tag_and_rejects_invalid_issued_limits() {
        for requested in [
            ProfiledExternalConstraint::Combined {
                depth: 4,
                reported_nodes: 16,
            },
            ProfiledExternalConstraint::Nodes { reported_nodes: 16 },
            ProfiledExternalConstraint::Depth { depth: 4 },
            ProfiledExternalConstraint::MoveTime { milliseconds: 100 },
        ] {
            let issued = match requested {
                ProfiledExternalConstraint::MoveTime { .. } => {
                    ProfiledExternalConstraint::MoveTime { milliseconds: 50 }
                }
                other => other,
            };
            let scope = EvidenceScope::ProfiledExternalUci {
                requested,
                issued,
                reported_depth: Some(2),
                seldepth: None,
                bound: ExternalBound::Unknown,
            };
            let json = serde_json::to_string(&ScopeWire::encode(scope)).unwrap();
            assert!(json.starts_with("{\"ProfiledExternal\":"));
            assert_eq!(
                serde_json::from_str::<ScopeWire>(&json)
                    .unwrap()
                    .decode()
                    .unwrap(),
                scope
            );
        }
        for wire in [
            ScopeWire::ProfiledExternal(
                ProfiledExternalConstraint::Nodes { reported_nodes: 0 },
                ProfiledExternalConstraint::Nodes { reported_nodes: 0 },
                None,
                None,
                0,
            ),
            ScopeWire::ProfiledExternal(
                ProfiledExternalConstraint::MoveTime { milliseconds: 50 },
                ProfiledExternalConstraint::MoveTime { milliseconds: 51 },
                None,
                None,
                0,
            ),
            ScopeWire::ProfiledExternal(
                ProfiledExternalConstraint::Depth { depth: 4 },
                ProfiledExternalConstraint::Nodes { reported_nodes: 16 },
                None,
                None,
                0,
            ),
            ScopeWire::ProfiledExternal(
                ProfiledExternalConstraint::Depth { depth: 4 },
                ProfiledExternalConstraint::Depth { depth: 4 },
                None,
                None,
                9,
            ),
        ] {
            assert!(wire.decode().is_err());
        }
        assert!(serde_json::from_str::<ScopeWire>("{\"UnknownProfiledTag\":[]}").is_err());
    }
}
#[derive(Debug, Serialize, Deserialize)]
enum ScoreWire {
    Unknown,
    Estimate(u32, u8),
    Cpu(i32, u8, BoundKind),
    External((u8, i32), u8, u8, Option<[u16; 3]>),
    Wdl([u32; 3], u8),
    ContextWdl([u32; 3], u8, u64),
    ConditionalWdl(u32, u8, ObservationId, ObservationId, u64),
    ConditionalRepairWdl(u32, u8, ObservationId, ObservationId, u64),
    Terminal(Option<u8>),
}
impl ScoreWire {
    fn encode(value: RawScore) -> Self {
        match value {
            RawScore::Unknown => Self::Unknown,
            RawScore::Estimate { value, perspective } => {
                Self::Estimate(value.to_bits(), c(perspective))
            }
            RawScore::Cpu {
                value,
                perspective,
                bound,
            } => Self::Cpu(value, c(perspective), bound),
            RawScore::ExternalUci {
                value,
                bound,
                perspective,
                wdl_per_mille,
            } => Self::External(r(value), b(bound), c(perspective), wdl_per_mille),
            RawScore::Wdl {
                win,
                draw,
                loss,
                perspective,
            } => Self::Wdl(
                [win.to_bits(), draw.to_bits(), loss.to_bits()],
                c(perspective),
            ),
            RawScore::ContextWdl {
                win,
                draw,
                loss,
                perspective,
                context_revision,
            } => Self::ContextWdl(
                [win.to_bits(), draw.to_bits(), loss.to_bits()],
                c(perspective),
                context_revision,
            ),
            RawScore::ConditionalWdl {
                expectation,
                perspective,
                repaired,
                counter,
                context_revision,
            } => Self::ConditionalWdl(
                expectation.to_bits(),
                c(perspective),
                repaired,
                counter,
                context_revision,
            ),
            RawScore::ConditionalRepairWdl {
                expectation,
                perspective,
                before,
                after,
                context_revision,
            } => Self::ConditionalRepairWdl(
                expectation.to_bits(),
                c(perspective),
                before,
                after,
                context_revision,
            ),
            RawScore::Terminal { winner } => Self::Terminal(winner.map(c)),
        }
    }
    fn decode(&self) -> Result<RawScore, StoreError> {
        Ok(match *self {
            Self::Unknown => RawScore::Unknown,
            Self::Estimate(value, perspective) => RawScore::Estimate {
                value: f32::from_bits(value),
                perspective: color(perspective)?,
            },
            Self::Cpu(value, perspective, bound) => RawScore::Cpu {
                value,
                perspective: color(perspective)?,
                bound,
            },
            Self::External(value, b, perspective, wdl_per_mille) => RawScore::ExternalUci {
                value: raw(value)?,
                bound: bound(b)?,
                perspective: color(perspective)?,
                wdl_per_mille,
            },
            Self::Wdl(value, perspective) => RawScore::Wdl {
                win: f32::from_bits(value[0]),
                draw: f32::from_bits(value[1]),
                loss: f32::from_bits(value[2]),
                perspective: color(perspective)?,
            },
            Self::ContextWdl(value, perspective, context_revision) => RawScore::ContextWdl {
                win: f32::from_bits(value[0]),
                draw: f32::from_bits(value[1]),
                loss: f32::from_bits(value[2]),
                perspective: color(perspective)?,
                context_revision,
            },
            Self::ConditionalWdl(expectation, perspective, repaired, counter, context_revision) => {
                RawScore::ConditionalWdl {
                    expectation: f32::from_bits(expectation),
                    perspective: color(perspective)?,
                    repaired,
                    counter,
                    context_revision,
                }
            }
            Self::ConditionalRepairWdl(
                expectation,
                perspective,
                before,
                after,
                context_revision,
            ) => RawScore::ConditionalRepairWdl {
                expectation: f32::from_bits(expectation),
                perspective: color(perspective)?,
                before,
                after,
                context_revision,
            },
            Self::Terminal(winner) => RawScore::Terminal {
                winner: winner.map(color).transpose()?,
            },
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReportWire {
    identity: crate::cpu_checker::ExternalCheckerIdentity,
    name: String,
    author: Option<String>,
    request_id: u64,
    best_move: Option<Move16>,
    pv: Vec<Move16>,
    score: (u8, i32),
    bound: u8,
    wdl: Option<[u16; 3]>,
    perspective: u8,
    requested_depth: u16,
    reported_depth: Option<u16>,
    seldepth: Option<u16>,
    restricted: bool,
    completion: u8,
    work: [Option<u64>; 3],
    elapsed: (u64, u32),
}
impl ReportWire {
    fn encode(report: &ExternalCheckerReport) -> Result<Self, StoreError> {
        Ok(Self {
            identity: report.identity.clone(),
            name: report.observed_uci.name.clone(),
            author: report.observed_uci.author.clone(),
            request_id: report.request_id,
            best_move: report.best_move.map(Move16::pack).transpose()?,
            pv: report
                .pv
                .iter()
                .copied()
                .map(Move16::pack)
                .collect::<Result<_, _>>()?,
            score: r(report.score),
            bound: b(report.bound),
            wdl: report.wdl_per_mille,
            perspective: c(report.perspective),
            requested_depth: report.requested_depth,
            reported_depth: report.reported_depth,
            seldepth: report.seldepth,
            restricted: report.root_restricted,
            completion: match report.completion {
                ExternalCompletion::Pending => 0,
                ExternalCompletion::BestMove => 1,
                ExternalCompletion::StoppedDeadline => 2,
                ExternalCompletion::StoppedCanceled => 3,
                ExternalCompletion::StoppedYield => 4,
            },
            work: work(report.work),
            elapsed: (report.elapsed.as_secs(), report.elapsed.subsec_nanos()),
        })
    }
    fn decode(self) -> Result<ExternalCheckerReport, StoreError> {
        if self.elapsed.1 >= 1_000_000_000 || self.pv.len() > rz_position::MAX_UCI_REPLAY_PLIES {
            return Err(StoreError::ArchiveIntegrity("external report bounds"));
        }
        Ok(ExternalCheckerReport {
            identity: self.identity,
            observed_uci: crate::cpu_checker::ExternalUciIdentity {
                name: self.name,
                author: self.author,
            },
            request_id: self.request_id,
            best_move: self.best_move.map(Move16::unpack).transpose()?,
            pv: self
                .pv
                .into_iter()
                .map(Move16::unpack)
                .collect::<Result<_, _>>()?,
            score: raw(self.score)?,
            bound: bound(self.bound)?,
            wdl_per_mille: self.wdl,
            perspective: color(self.perspective)?,
            requested_depth: self.requested_depth,
            reported_depth: self.reported_depth,
            seldepth: self.seldepth,
            root_restricted: self.restricted,
            completion: match self.completion {
                0 => ExternalCompletion::Pending,
                1 => ExternalCompletion::BestMove,
                2 => ExternalCompletion::StoppedDeadline,
                3 => ExternalCompletion::StoppedCanceled,
                4 => ExternalCompletion::StoppedYield,
                _ => return Err(StoreError::ArchiveIntegrity("external completion")),
            },
            work: w(self.work),
            elapsed: std::time::Duration::new(self.elapsed.0, self.elapsed.1),
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ObservationWire {
    state: StateId,
    line: Option<LineId>,
    source: u64,
    epoch: u64,
    value: Option<CpuValueIdentity>,
    checker: Option<CheckerIdentity>,
    work: Option<[Option<u64>; 3]>,
    external: Option<ReportWire>,
    model: Option<(String, String, String, String, [u8; 32])>,
    input: Option<[u8; 32]>,
    condition: Option<String>,
    pv: Option<LineId>,
    scope: ScopeWire,
    score: ScoreWire,
    budget: u64,
    kind: ObservationKind,
    supersedes: Option<ObservationId>,
    execution: Option<ExecutionId>,
}
impl ObservationWire {
    pub fn state_id(&self) -> StateId {
        self.state
    }
    pub fn conflicts_external_request(&self, observation: &Observation) -> bool {
        matches!((&self.external, observation.external_report.as_deref()),
            (Some(cold), Some(actual)) if cold.identity == actual.identity
                && cold.request_id == actual.request_id
                && (self.state != observation.state || self.execution != observation.execution))
    }
    pub fn add_pins(&self, pins: &mut StorePins) {
        pins.states.insert(self.state);
        pins.lines.extend(self.line);
        pins.lines.extend(self.pv);
        pins.observations.extend(self.supersedes);
        if let ScoreWire::ConditionalWdl(_, _, before, after, _)
        | ScoreWire::ConditionalRepairWdl(_, _, before, after, _) = self.score
        {
            pins.observations.extend([before, after]);
        }
        pins.executions.extend(self.execution);
    }
    pub fn encode(o: &Observation) -> Result<Self, StoreError> {
        Ok(Self {
            state: o.state,
            line: o.line,
            source: o.source,
            epoch: o.epoch,
            value: o.value_identity.clone(),
            checker: o.checker_identity.clone(),
            work: o.checker_work.map(work),
            external: o
                .external_report
                .as_deref()
                .map(ReportWire::encode)
                .transpose()?,
            model: o.model_value_identity.as_ref().map(|m| {
                (
                    m.semantics.clone(),
                    m.model.clone(),
                    m.encoding.clone(),
                    m.precision.clone(),
                    m.model_epoch,
                )
            }),
            input: o.model_value_input,
            condition: o.cpu_condition.clone(),
            pv: o.cpu_pv,
            scope: ScopeWire::encode(o.scope),
            score: ScoreWire::encode(o.score),
            budget: o.budget,
            kind: o.kind,
            supersedes: o.supersedes,
            execution: o.execution,
        })
    }
    pub fn decode(self) -> Result<Observation, StoreError> {
        let result = Observation {
            state: self.state,
            line: self.line,
            source: self.source,
            epoch: self.epoch,
            value_identity: self.value,
            checker_identity: self.checker,
            checker_work: self.work.map(w),
            external_report: self
                .external
                .map(ReportWire::decode)
                .transpose()?
                .map(Box::new),
            model_value_identity: self.model.map(
                |(semantics, model, encoding, precision, model_epoch)| {
                    crate::pals::value::ModelValueIdentity {
                        semantics,
                        model,
                        encoding,
                        precision,
                        model_epoch,
                    }
                },
            ),
            model_value_input: self.input,
            cpu_condition: self.condition,
            cpu_pv: self.pv,
            scope: self.scope.decode()?,
            score: self.score.decode()?,
            budget: self.budget,
            kind: self.kind,
            supersedes: self.supersedes,
            execution: self.execution,
        };
        result.validate()?;
        Ok(result)
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RoleRecordWire {
    revision: u64,
    state: StateId,
    kind: u8,
    line: Vec<Move16>,
    value: Option<i32>,
    depth: u16,
    scope: Option<u8>,
    observation: Option<ObservationId>,
    perspective: u8,
    critical: bool,
    parent_revision: Option<u64>,
    supersedes_revision: Option<u64>,
}
impl RoleRecordWire {
    pub fn origin_state(&self) -> StateId {
        self.state
    }
    pub fn add_pins(&self, pins: &mut StorePins) {
        pins.states.insert(self.state);
        pins.observations.extend(self.observation);
    }
    pub fn encode(o: &crate::pals::engine::RoleRecord) -> Result<Self, StoreError> {
        use crate::cpu::CpuScoreScope;
        use crate::pals::engine::RecordKind;
        Ok(Self {
            revision: o.revision,
            state: o.origin_state,
            kind: match o.kind {
                RecordKind::Proposal => 0,
                RecordKind::Counterexample => 1,
                RecordKind::Repair => 2,
                RecordKind::CpuVerification => 3,
            },
            line: o
                .line
                .iter()
                .copied()
                .map(Move16::pack)
                .collect::<Result<_, _>>()?,
            value: o.value,
            depth: o.completed_depth,
            scope: o.score_scope.map(|s| match s {
                CpuScoreScope::FrontierOnly => 0,
                CpuScoreScope::CompletedIteration => 1,
                CpuScoreScope::RulesTerminal => 2,
            }),
            observation: o.cpu_observation,
            perspective: c(o.perspective),
            critical: o.critical,
            parent_revision: o.parent_revision,
            supersedes_revision: o.supersedes_revision,
        })
    }
    pub fn decode(self) -> Result<crate::pals::engine::RoleRecord, StoreError> {
        use crate::cpu::CpuScoreScope;
        use crate::pals::engine::RecordKind;
        Ok(crate::pals::engine::RoleRecord {
            revision: self.revision,
            origin_state: self.state,
            kind: match self.kind {
                0 => RecordKind::Proposal,
                1 => RecordKind::Counterexample,
                2 => RecordKind::Repair,
                3 => RecordKind::CpuVerification,
                _ => return Err(StoreError::ArchiveIntegrity("role kind")),
            },
            line: self
                .line
                .into_iter()
                .map(Move16::unpack)
                .collect::<Result<_, _>>()?,
            value: self.value,
            completed_depth: self.depth,
            score_scope: self
                .scope
                .map(|s| match s {
                    0 => Ok(CpuScoreScope::FrontierOnly),
                    1 => Ok(CpuScoreScope::CompletedIteration),
                    2 => Ok(CpuScoreScope::RulesTerminal),
                    _ => Err(StoreError::ArchiveIntegrity("CPU role scope")),
                })
                .transpose()?,
            cpu_observation: self.observation,
            perspective: color(self.perspective)?,
            critical: self.critical,
            parent_revision: self.parent_revision,
            supersedes_revision: self.supersedes_revision,
        })
    }
}
