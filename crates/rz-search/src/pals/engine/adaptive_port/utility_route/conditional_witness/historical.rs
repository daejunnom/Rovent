//! Strict historical parsing of the original conditional-witness preimage.
//!
//! This module issues no Store, live source, completed-comparison, known-target
//! or training capability. In particular, an Estimate has no serialized CPU
//! comparison basis, and tag 4 merges Wdl with ContextWdl while omitting actual
//! prepared-input/context-revision provenance. Valid bytes cannot recover it.

use crate::cpu_checker::{ExternalBound, ExternalRawScore};
use crate::pals::engine::adaptive_port::validator_port::ActionWitnessKind;
use crate::pals::store::{BoundKind, Move16, ObservationKind, ProfiledExternalConstraint};
use rz_contracts::Digest;
use rz_position::{BoardMove, Color, PlayStatus, Position, PositionLimits, PositionSnapshot};
use sha2::{Digest as _, Sha256};

const MAX_BYTES: usize = 16 * 1024 * 1024;
const MAX_LINE: usize = 256;
const MAX_DEPTH: u8 = 8;
const MAX_NODES: usize = 255;
const BODY_DOMAIN: &[u8] = b"rz-rovent-utility-actual-conditional-witness/1\0";
const IDENTITY_DOMAIN: &[u8] = b"rz-pals-action-unique-witness/1\0";
const PROFILED_DOMAIN: &[u8] = b"rz-validator-profiled-external-observation/1\0";

#[derive(Clone, Copy)]
pub struct UtilityWitnessPreimageExpectation<'a> {
    pub root: &'a PositionSnapshot,
    pub kind: ActionWitnessKind,
    pub line: &'a [BoardMove],
    pub source_question: Digest,
    pub source_frame_sequence: u64,
    pub identity: Digest,
    pub raw_evidence: Digest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UtilityWitnessPreimageError {
    Limit(&'static str),
    Truncated,
    Invalid(&'static str),
    Mismatch(&'static str),
    TrailingBytes,
}
impl std::fmt::Display for UtilityWitnessPreimageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "historical utility witness: {self:?}")
    }
}
impl std::error::Error for UtilityWitnessPreimageError {}
type Error = UtilityWitnessPreimageError;

/// Validate bounded historical structure and the supplied exact Rules scope.
/// Success is deliberately `()`; it cannot become a live or known witness.
pub fn validate_utility_witness_preimage(
    bytes: &[u8],
    expected: UtilityWitnessPreimageExpectation<'_>,
) -> Result<(), UtilityWitnessPreimageError> {
    parse_validated_utility_witness_preimage(bytes, expected).map(|_| ())
}

/// Read-only raw projection of the same validated traversal. Fields absent
/// from the old framing (scope, actual input/revision, completed CPU basis)
/// remain absent; this is historical data rather than a producer capability.
pub(in crate::pals::engine::adaptive_port) fn parse_validated_utility_witness_preimage<'a>(
    bytes: &'a [u8],
    expected: UtilityWitnessPreimageExpectation<'_>,
) -> Result<HistoricalWitnessObservation<'a>, UtilityWitnessPreimageError> {
    if bytes.len() > MAX_BYTES {
        return Err(Error::Limit("preimage bytes"));
    }
    if expected.line.is_empty() || expected.line.len() > MAX_LINE {
        return Err(Error::Invalid("actual witness line length"));
    }
    if expected.source_frame_sequence == 0 {
        return Err(Error::Invalid("source frame sequence"));
    }
    let root = rz_position::contracts::rules_snapshot_semantic_digest(expected.root);
    let mut identity = Sha256::new();
    identity.update(IDENTITY_DOMAIN);
    identity.update(root.0);
    identity.update([expected.kind as u8]);
    for &movement in expected.line {
        identity.update(
            Move16::pack(movement)
                .map_err(|_| Error::Invalid("actual witness Move16"))?
                .bits()
                .to_le_bytes(),
        );
    }
    if Digest(identity.finalize().into()) != expected.identity {
        return Err(Error::Mismatch("unique witness identity"));
    }
    let mut cursor = Cursor {
        bytes,
        offset: 0,
        nodes: 0,
    };
    if cursor.take(BODY_DOMAIN.len())? != BODY_DOMAIN {
        return Err(Error::Invalid("preimage domain"));
    }
    if Digest(cursor.array()?) != expected.source_question {
        return Err(Error::Mismatch("source question"));
    }
    if cursor.u64()? != expected.source_frame_sequence {
        return Err(Error::Mismatch("source frame sequence"));
    }
    if Digest(cursor.array()?) != expected.identity {
        return Err(Error::Mismatch("header witness identity"));
    }
    let observation = cursor.observation(0)?;
    if cursor.offset != bytes.len() {
        return Err(Error::TrailingBytes);
    }
    if Digest(Sha256::digest(bytes).into()) != expected.raw_evidence {
        return Err(Error::Mismatch("raw evidence SHA-256"));
    }
    let kind = match expected.kind {
        ActionWitnessKind::Counter => ObservationKind::Refutation,
        ActionWitnessKind::Repair => ObservationKind::Repair,
    };
    if observation.kind != kind || observation.state_digest != root {
        return Err(Error::Mismatch("root observation kind/state"));
    }
    let proposal = observation
        .line
        .as_deref()
        .filter(|line| !line.is_empty())
        .ok_or(Error::Invalid("root observation line"))?;
    let actual_endpoint = replay(expected.root, expected.line)?;
    let proposal_endpoint = replay(expected.root, proposal)?;
    match expected.kind {
        ActionWitnessKind::Repair if proposal != expected.line => {
            return Err(Error::Mismatch("Repair observation line"));
        }
        ActionWitnessKind::Counter => {
            if proposal.first() != expected.line.first() {
                return Err(Error::Mismatch("counter original first move"));
            }
            let divergence = proposal
                .iter()
                .zip(expected.line)
                .position(|(left, right)| left != right)
                .ok_or(Error::Invalid("counter has no actual divergence"))?;
            let anchor = replay(expected.root, &proposal[..divergence])?;
            if anchor.side_to_move() == expected.root.side_to_move() {
                return Err(Error::Invalid("counter divergence is not opponent's turn"));
            }
        }
        ActionWitnessKind::Repair => {}
    }
    match observation.summary {
        ParsedScore::Estimate { perspective } => {
            if perspective != expected.root.side_to_move() {
                return Err(Error::Mismatch("root Estimate perspective"));
            }
        }
        ParsedScore::Conditional {
            tag,
            perspective,
            before,
            after,
        } => {
            if perspective != expected.root.side_to_move()
                || tag
                    != match expected.kind {
                        ActionWitnessKind::Counter => 5,
                        ActionWitnessKind::Repair => 6,
                    }
            {
                return Err(Error::Mismatch("root conditional kind/perspective"));
            }
            endpoint_matches(after, &actual_endpoint)?;
            if expected.kind == ActionWitnessKind::Counter {
                endpoint_matches(before, &proposal_endpoint)?;
            }
        }
        _ => return Err(Error::Invalid("root witness score kind")),
    }
    Ok(observation)
}

fn replay(root: &PositionSnapshot, line: &[BoardMove]) -> Result<PositionSnapshot, Error> {
    if line.len() > MAX_LINE {
        return Err(Error::Limit("Rules line plies"));
    }
    let maximum = root
        .known_history_len()
        .checked_add(MAX_LINE)
        .ok_or(Error::Limit("Rules history"))?;
    let mut position = Position::from_snapshot(
        root,
        PositionLimits {
            max_history_positions: maximum,
            ..PositionLimits::default()
        },
    )
    .map_err(|_| Error::Invalid("Rules snapshot"))?;
    for &movement in line {
        let legal = position.ordered_legal_moves();
        if position
            .play_status_from_view(&legal)
            .map_err(|_| Error::Invalid("Rules classification"))?
            != PlayStatus::Ongoing
        {
            return Err(Error::Invalid("line continues after Rules terminal"));
        }
        position
            .make_from_view(&legal, movement)
            .map_err(|_| Error::Invalid("illegal Rules move"))?;
    }
    Ok(position.snapshot())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::pals::engine::adaptive_port) struct HistoricalModelValueIdentity<'a> {
    pub semantics: &'a str,
    pub model: &'a str,
    pub encoding: &'a str,
    pub precision: &'a str,
    pub epoch: [u8; 32],
}
type ModelIdentity<'a> = HistoricalModelValueIdentity<'a>;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::pals::engine::adaptive_port) struct HistoricalCpuValueIdentity<'a> {
    pub semantics: &'a str,
    /// Empty is the original wire spelling of absent weights. No training
    /// state was serialized and none is reconstructed from this field.
    pub weights: &'a str,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::pals::engine::adaptive_port) struct HistoricalProfiledScope {
    pub requested: ProfiledExternalConstraint,
    pub issued: ProfiledExternalConstraint,
    pub reported_depth: Option<u16>,
    pub seldepth: Option<u16>,
    pub bound: ExternalBound,
}
#[derive(Debug)]
pub(in crate::pals::engine::adaptive_port) enum HistoricalWitnessRawScore<'a> {
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
    // Preserve the complete parsed external report. Cohort comparison admits
    // only Rules, internal completed CPU, or original Fresh-WDL evidence;
    // external CP/mate fields must not become a conditional value target.
    #[allow(dead_code)]
    ExternalUci {
        value: ExternalRawScore,
        bound: ExternalBound,
        perspective: Color,
        wdl_per_mille: Option<[u16; 3]>,
    },
    /// The old tag 4 cannot distinguish Wdl from ContextWdl.
    Wdl {
        win: f32,
        draw: f32,
        loss: f32,
        perspective: Color,
    },
    ConditionalWdl {
        expectation: f32,
        perspective: Color,
        repaired: Box<HistoricalWitnessObservation<'a>>,
        counter: Box<HistoricalWitnessObservation<'a>>,
    },
    ConditionalRepairWdl {
        expectation: f32,
        perspective: Color,
        before: Box<HistoricalWitnessObservation<'a>>,
        after: Box<HistoricalWitnessObservation<'a>>,
    },
    Terminal {
        winner: Option<Color>,
    },
}
#[derive(Clone, Copy, Debug)]
struct WdlEndpoint {
    state: Digest,
    perspective: Color,
    expectation: f32,
}
#[derive(Clone, Copy, Debug)]
enum ParsedScore {
    Other,
    Estimate {
        perspective: Color,
    },
    Wdl {
        values: [f32; 3],
        perspective: Color,
    },
    Conditional {
        tag: u8,
        perspective: Color,
        before: WdlEndpoint,
        after: WdlEndpoint,
    },
}
#[derive(Debug)]
pub(in crate::pals::engine::adaptive_port) struct HistoricalWitnessObservation<'a> {
    pub kind: ObservationKind,
    pub state_digest: Digest,
    pub line: Option<Vec<BoardMove>>,
    pub cpu_pv: Option<Vec<BoardMove>>,
    pub budget: u64,
    pub cpu_condition: &'a str,
    pub cpu_identity: Option<HistoricalCpuValueIdentity<'a>>,
    pub model_identity: Option<HistoricalModelValueIdentity<'a>>,
    pub profiled_scope: Option<HistoricalProfiledScope>,
    pub score: HistoricalWitnessRawScore<'a>,
    summary: ParsedScore,
}
impl<'a> HistoricalWitnessObservation<'a> {
    fn wdl_endpoint(&self, perspective: Color) -> Result<(ModelIdentity<'a>, WdlEndpoint), Error> {
        let ParsedScore::Wdl {
            values,
            perspective: side,
        } = self.summary
        else {
            return Err(Error::Invalid("conditional child is not raw WDL"));
        };
        let identity = self
            .model_identity
            .ok_or(Error::Invalid("raw WDL namespace"))?;
        Ok((
            identity,
            WdlEndpoint {
                state: self.state_digest,
                perspective: side,
                expectation: if perspective == side {
                    values[0] - values[2]
                } else {
                    values[2] - values[0]
                },
            },
        ))
    }
}
fn endpoint_matches(endpoint: WdlEndpoint, snapshot: &PositionSnapshot) -> Result<(), Error> {
    if endpoint.state != rz_position::contracts::rules_snapshot_semantic_digest(snapshot)
        || endpoint.perspective != snapshot.side_to_move()
    {
        return Err(Error::Mismatch(
            "conditional raw endpoint Rules/perspective",
        ));
    }
    Ok(())
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
    nodes: usize,
}
impl<'a> Cursor<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], Error> {
        let end = self.offset.checked_add(count).ok_or(Error::Truncated)?;
        let bytes = self.bytes.get(self.offset..end).ok_or(Error::Truncated)?;
        self.offset = end;
        Ok(bytes)
    }
    fn array<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        self.take(N)?.try_into().map_err(|_| Error::Truncated)
    }
    fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.array::<1>()?[0])
    }
    fn u16(&mut self) -> Result<u16, Error> {
        Ok(u16::from_le_bytes(self.array()?))
    }
    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.array()?))
    }
    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.array()?))
    }
    fn present(&mut self) -> Result<bool, Error> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(Error::Invalid("presence tag")),
        }
    }
    fn color(&mut self) -> Result<Color, Error> {
        Ok(if self.present()? {
            Color::Black
        } else {
            Color::White
        })
    }
    fn float(&mut self) -> Result<f32, Error> {
        let value = f32::from_bits(self.u32()?);
        if !value.is_finite() {
            return Err(Error::Invalid("non-finite float"));
        }
        Ok(value)
    }
    fn string(&mut self, maximum: usize, empty: bool) -> Result<&'a str, Error> {
        let count = usize::try_from(self.u64()?).map_err(|_| Error::Limit("string bytes"))?;
        if count > maximum {
            return Err(Error::Limit("string bytes"));
        }
        let text =
            std::str::from_utf8(self.take(count)?).map_err(|_| Error::Invalid("UTF-8 string"))?;
        if text.chars().any(char::is_control)
            || (!empty && text.trim().is_empty())
            || (!text.is_empty() && text.trim().is_empty())
        {
            return Err(Error::Invalid("bounded namespace string"));
        }
        Ok(text)
    }
    fn line(&mut self) -> Result<Option<Vec<BoardMove>>, Error> {
        if !self.present()? {
            return Ok(None);
        }
        let count = usize::try_from(self.u64()?).map_err(|_| Error::Limit("line plies"))?;
        if count > MAX_LINE {
            return Err(Error::Limit("line plies"));
        }
        let mut line = Vec::new();
        line.try_reserve_exact(count)
            .map_err(|_| Error::Limit("line allocation"))?;
        for _ in 0..count {
            let packed = rz_contracts::pals::Move16::try_from_bits(self.u16()?)
                .map_err(|_| Error::Invalid("Move16 bits"))?;
            let movement = packed.decode().map_err(|_| Error::Invalid("Move16 move"))?;
            line.push(BoardMove::try_from(movement).map_err(|_| Error::Invalid("Move16 move"))?);
        }
        Ok(Some(line))
    }
    fn constraint(&mut self) -> Result<ProfiledExternalConstraint, Error> {
        use ProfiledExternalConstraint as Limit;
        let value = match self.u8()? {
            0 => Limit::Combined {
                depth: self.u16()?,
                reported_nodes: self.u64()?,
            },
            1 => Limit::Nodes {
                reported_nodes: self.u64()?,
            },
            2 => Limit::Depth { depth: self.u16()? },
            3 => Limit::MoveTime {
                milliseconds: self.u64()?,
            },
            _ => return Err(Error::Invalid("profiled constraint tag")),
        };
        value
            .validate()
            .map_err(|_| Error::Invalid("profiled constraint bounds"))?;
        Ok(value)
    }
    fn bound(&mut self, maximum: u8) -> Result<u8, Error> {
        let value = self.u8()?;
        if value > maximum {
            return Err(Error::Invalid("score bound tag"));
        }
        Ok(value)
    }
    fn observation(&mut self, depth: u8) -> Result<HistoricalWitnessObservation<'a>, Error> {
        if depth >= MAX_DEPTH {
            return Err(Error::Limit("observation depth"));
        }
        self.nodes = self
            .nodes
            .checked_add(1)
            .ok_or(Error::Limit("observation nodes"))?;
        if self.nodes > MAX_NODES {
            return Err(Error::Limit("observation nodes"));
        }
        let profiled = if self.bytes.get(self.offset) == PROFILED_DOMAIN.first() {
            if self.take(PROFILED_DOMAIN.len())? != PROFILED_DOMAIN {
                return Err(Error::Invalid("profiled observation domain"));
            }
            let requested = self.constraint()?;
            let issued = self.constraint()?;
            requested
                .validate_issued(issued)
                .map_err(|_| Error::Invalid("issued constraint"))?;
            let reported_depth = if self.present()? {
                Some(self.u16()?)
            } else {
                None
            };
            let seldepth = if self.present()? {
                Some(self.u16()?)
            } else {
                None
            };
            let bound = external_bound(self.bound(3)?);
            Some(HistoricalProfiledScope {
                requested,
                issued,
                reported_depth,
                seldepth,
                bound,
            })
        } else {
            None
        };
        let kind = match self.u8()? {
            0 => ObservationKind::Proposal,
            1 => ObservationKind::Refutation,
            2 => ObservationKind::Repair,
            3 => ObservationKind::CpuAnalysis,
            4 => ObservationKind::ExternalCpuAnalysis,
            5 => ObservationKind::ExactTerminal,
            _ => return Err(Error::Invalid("observation kind tag")),
        };
        let state = Digest(self.array()?);
        let line = self.line()?;
        let pv = self.line()?;
        let budget = self.u64()?;
        let condition = self.string(8192, true)?;
        let cpu = self.present()?;
        let cpu_identity = if cpu {
            let semantics = self.string(256, false)?;
            let weights = self.string(64, true)?;
            if !weights.is_empty()
                && (weights.len() != 64
                    || !weights
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
            {
                return Err(Error::Invalid("CPU weights digest"));
            }
            Some(HistoricalCpuValueIdentity { semantics, weights })
        } else {
            None
        };
        let model = if self.present()? {
            Some(ModelIdentity {
                semantics: self.string(256, false)?,
                model: self.string(1024, false)?,
                encoding: self.string(256, false)?,
                precision: self.string(64, false)?,
                epoch: self.array()?,
            })
        } else {
            None
        };
        let tag = self.u8()?;
        if (tag == 4) != model.is_some()
            || (cpu && (condition.is_empty() || condition.len() > 2048 || profiled.is_some()))
            || (!cpu && !condition.is_empty() && tag != 3)
            || (profiled.is_some() && tag != 3)
        {
            return Err(Error::Invalid("raw score namespace framing"));
        }
        let (summary, score) = match tag {
            0 => (ParsedScore::Other, HistoricalWitnessRawScore::Unknown),
            1 => {
                let perspective = self.color()?;
                let value = self.float()?;
                (
                    ParsedScore::Estimate { perspective },
                    HistoricalWitnessRawScore::Estimate { value, perspective },
                )
            }
            2 => {
                let perspective = self.color()?;
                let bound = match self.bound(2)? {
                    0 => BoundKind::ExactWithinSearch,
                    1 => BoundKind::LowerWithinSearch,
                    _ => BoundKind::UpperWithinSearch,
                };
                let value = self.u32()? as i32;
                if !cpu {
                    return Err(Error::Invalid("CPU score namespace"));
                }
                (
                    ParsedScore::Other,
                    HistoricalWitnessRawScore::Cpu {
                        value,
                        perspective,
                        bound,
                    },
                )
            }
            3 => {
                let bound = external_bound(self.bound(3)?);
                let perspective = self.color()?;
                let value = match self.u8()? {
                    0 => ExternalRawScore::Centipawns(self.u32()? as i32),
                    1 => ExternalRawScore::MateMoves(self.u32()? as i32),
                    2 => ExternalRawScore::Unknown,
                    _ => return Err(Error::Invalid("external score tag")),
                };
                let wdl_per_mille = if self.present()? {
                    let wdl = [self.u16()?, self.u16()?, self.u16()?];
                    if wdl.iter().any(|value| *value > 1000)
                        || wdl.into_iter().map(u32::from).sum::<u32>() != 1000
                    {
                        return Err(Error::Invalid("external WDL"));
                    }
                    Some(wdl)
                } else {
                    None
                };
                if cpu
                    || condition.is_empty()
                    || kind != ObservationKind::ExternalCpuAnalysis
                    || profiled.is_some_and(|scope| scope.bound != bound)
                {
                    return Err(Error::Invalid("external score namespace"));
                }
                (
                    ParsedScore::Other,
                    HistoricalWitnessRawScore::ExternalUci {
                        value,
                        bound,
                        perspective,
                        wdl_per_mille,
                    },
                )
            }
            4 => {
                let perspective = self.color()?;
                let values = [self.float()?, self.float()?, self.float()?];
                if cpu
                    || !condition.is_empty()
                    || pv.is_some()
                    || values.iter().any(|value| !(0.0..=1.0).contains(value))
                    || (values.iter().sum::<f32>() - 1.0).abs() > 1e-4
                {
                    return Err(Error::Invalid("raw WDL"));
                }
                (
                    ParsedScore::Wdl {
                        values,
                        perspective,
                    },
                    HistoricalWitnessRawScore::Wdl {
                        win: values[0],
                        draw: values[1],
                        loss: values[2],
                        perspective,
                    },
                )
            }
            5 | 6 => {
                let perspective = self.color()?;
                let expectation = self.float()?;
                if cpu
                    || !condition.is_empty()
                    || pv.is_some()
                    || line.is_none()
                    || !(-1.0..=1.0).contains(&expectation)
                    || kind
                        != if tag == 5 {
                            ObservationKind::Refutation
                        } else {
                            ObservationKind::Repair
                        }
                {
                    return Err(Error::Invalid("conditional score framing"));
                }
                let before_observation = self.observation(depth + 1)?;
                let after_observation = self.observation(depth + 1)?;
                let (before_model, before) = before_observation.wdl_endpoint(perspective)?;
                let (after_model, after) = after_observation.wdl_endpoint(perspective)?;
                if before_model != after_model
                    || expectation != after.expectation
                    || (tag == 5 && after.expectation >= before.expectation)
                    || (tag == 6 && after.expectation <= before.expectation)
                {
                    return Err(Error::Invalid("conditional raw WDL comparison"));
                }
                let summary = ParsedScore::Conditional {
                    tag,
                    perspective,
                    before,
                    after,
                };
                let score = if tag == 5 {
                    HistoricalWitnessRawScore::ConditionalWdl {
                        expectation,
                        perspective,
                        repaired: Box::new(before_observation),
                        counter: Box::new(after_observation),
                    }
                } else {
                    HistoricalWitnessRawScore::ConditionalRepairWdl {
                        expectation,
                        perspective,
                        before: Box::new(before_observation),
                        after: Box::new(after_observation),
                    }
                };
                (summary, score)
            }
            7 => {
                let winner = match self.u8()? {
                    0 => None,
                    1 => Some(Color::White),
                    2 => Some(Color::Black),
                    _ => return Err(Error::Invalid("terminal winner tag")),
                };
                if cpu
                    || !condition.is_empty()
                    || pv.is_some()
                    || kind != ObservationKind::ExactTerminal
                {
                    return Err(Error::Invalid("terminal raw framing"));
                }
                (
                    ParsedScore::Other,
                    HistoricalWitnessRawScore::Terminal { winner },
                )
            }
            _ => return Err(Error::Invalid("raw score tag")),
        };
        if (kind == ObservationKind::ExternalCpuAnalysis && tag != 3)
            || (pv.is_some() && !cpu && tag != 3)
            || (cpu && tag != 0 && tag != 1 && tag != 2)
        {
            return Err(Error::Invalid("observation namespace/score"));
        }
        // Root Estimate witnesses are pure derived observations. No hidden
        // CPU namespace or PV may supply a comparison basis missing on wire.
        if depth == 0 && (cpu || pv.is_some() || !condition.is_empty()) {
            return Err(Error::Invalid("root derived observation framing"));
        }
        Ok(HistoricalWitnessObservation {
            kind,
            state_digest: state,
            line,
            cpu_pv: pv,
            budget,
            cpu_condition: condition,
            cpu_identity,
            model_identity: model,
            profiled_scope: profiled,
            score,
            summary,
        })
    }
}

fn external_bound(tag: u8) -> ExternalBound {
    match tag {
        0 => ExternalBound::ExactReported,
        1 => ExternalBound::Lower,
        2 => ExternalBound::Upper,
        _ => ExternalBound::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent authored wire fixtures test the historical grammar only.
    // They are not Store publications or actual conditional witness receipts.
    fn line(text: &str) -> Vec<BoardMove> {
        text.split_whitespace()
            .map(|text| text.parse().unwrap())
            .collect()
    }
    fn bytes_text(bytes: &mut Vec<u8>, text: &str) {
        bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
        bytes.extend_from_slice(text.as_bytes());
    }
    fn bytes_line(bytes: &mut Vec<u8>, line: Option<&[BoardMove]>) {
        bytes.push(u8::from(line.is_some()));
        if let Some(line) = line {
            bytes.extend_from_slice(&(line.len() as u64).to_le_bytes());
            for &movement in line {
                bytes.extend_from_slice(&Move16::pack(movement).unwrap().bits().to_le_bytes());
            }
        }
    }
    fn unique(root: &PositionSnapshot, kind: ActionWitnessKind, line: &[BoardMove]) -> Digest {
        let mut hash = Sha256::new();
        hash.update(IDENTITY_DOMAIN);
        hash.update(rz_position::contracts::rules_snapshot_semantic_digest(root).0);
        hash.update([kind as u8]);
        for &movement in line {
            hash.update(Move16::pack(movement).unwrap().bits().to_le_bytes());
        }
        Digest(hash.finalize().into())
    }
    fn body(
        root: &PositionSnapshot,
        kind: ActionWitnessKind,
        actual: &[BoardMove],
        raw: &[u8],
    ) -> Vec<u8> {
        let mut bytes = BODY_DOMAIN.to_vec();
        bytes.extend_from_slice(&[13; 32]);
        bytes.extend_from_slice(&7_u64.to_le_bytes());
        bytes.extend_from_slice(&unique(root, kind, actual).0);
        bytes.extend_from_slice(raw);
        bytes
    }
    fn expected<'a>(
        root: &'a PositionSnapshot,
        kind: ActionWitnessKind,
        line: &'a [BoardMove],
        bytes: &[u8],
    ) -> UtilityWitnessPreimageExpectation<'a> {
        UtilityWitnessPreimageExpectation {
            root,
            kind,
            line,
            source_question: Digest([13; 32]),
            source_frame_sequence: 7,
            identity: unique(root, kind, line),
            raw_evidence: Digest(Sha256::digest(bytes).into()),
        }
    }
    fn common(
        root: &PositionSnapshot,
        kind: ObservationKind,
        moves: Option<&[BoardMove]>,
    ) -> Vec<u8> {
        let mut raw = vec![kind as u8];
        raw.extend_from_slice(&rz_position::contracts::rules_snapshot_semantic_digest(root).0);
        bytes_line(&mut raw, moves);
        bytes_line(&mut raw, None);
        raw.extend_from_slice(&0_u64.to_le_bytes());
        bytes_text(&mut raw, "");
        raw.push(0); // No CPU value namespace.
        raw
    }
    fn estimate(root: &PositionSnapshot, kind: ObservationKind, moves: &[BoardMove]) -> Vec<u8> {
        let mut raw = common(root, kind, Some(moves));
        raw.extend_from_slice(&[0, 1, u8::from(root.side_to_move() == Color::Black)]);
        raw.extend_from_slice(&(-100.0_f32).to_bits().to_le_bytes());
        raw
    }
    fn wdl(snapshot: &PositionSnapshot, values: [f32; 3]) -> Vec<u8> {
        let mut raw = common(snapshot, ObservationKind::Proposal, None);
        raw.push(1); // Model identity follows.
        for text in [
            "context-wdl",
            "authored-model",
            "authored-encoding",
            "fp32_tf32_off",
        ] {
            bytes_text(&mut raw, text);
        }
        raw.extend_from_slice(&[17; 32]);
        raw.extend_from_slice(&[4, u8::from(snapshot.side_to_move() == Color::Black)]);
        for value in values {
            raw.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        raw
    }

    #[test]
    fn authored_counter_and_repair_estimates_validate_as_history_only() {
        let root = Position::startpos().snapshot();
        let proposal = line("e2e4 e7e5");
        let counter = line("e2e4 c7c5");
        let raw = estimate(&root, ObservationKind::Refutation, &proposal);
        let bytes = body(&root, ActionWitnessKind::Counter, &counter, &raw);
        validate_utility_witness_preimage(
            &bytes,
            expected(&root, ActionWitnessKind::Counter, &counter, &bytes),
        )
        .unwrap();
        // Counter's root observation is the refuted proposal, not the actual
        // counter line used by the unique witness identity.
        assert_ne!(proposal, counter);
        let repaired = line("e2e4 c7c5 g1f3");
        let bytes = body(
            &root,
            ActionWitnessKind::Repair,
            &repaired,
            &estimate(&root, ObservationKind::Repair, &repaired),
        );
        validate_utility_witness_preimage(
            &bytes,
            expected(&root, ActionWitnessKind::Repair, &repaired, &bytes),
        )
        .unwrap();
    }

    #[test]
    fn every_truncation_and_trailing_byte_is_rejected_without_unbounded_lengths() {
        let root = Position::startpos().snapshot();
        let actual = line("e2e4 c7c5");
        let proposal = line("e2e4 e7e5");
        let bytes = body(
            &root,
            ActionWitnessKind::Counter,
            &actual,
            &estimate(&root, ObservationKind::Refutation, &proposal),
        );
        for length in 0..bytes.len() {
            let short = &bytes[..length];
            assert!(
                validate_utility_witness_preimage(
                    short,
                    expected(&root, ActionWitnessKind::Counter, &actual, short)
                )
                .is_err(),
                "accepted truncated length {length}"
            );
        }
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert_eq!(
            validate_utility_witness_preimage(
                &trailing,
                expected(&root, ActionWitnessKind::Counter, &actual, &trailing)
            ),
            Err(Error::TrailingBytes)
        );
        let mut oversized_line = bytes;
        let count = BODY_DOMAIN.len() + 32 + 8 + 32 + 1 + 32 + 1;
        oversized_line[count..count + 8].copy_from_slice(&257_u64.to_le_bytes());
        assert_eq!(
            validate_utility_witness_preimage(
                &oversized_line,
                expected(&root, ActionWitnessKind::Counter, &actual, &oversized_line)
            ),
            Err(Error::Limit("line plies"))
        );
    }

    #[test]
    fn source_root_identity_tags_move_bits_and_floats_are_independently_checked() {
        let root = Position::startpos().snapshot();
        let actual = line("e2e4 c7c5");
        let proposal = line("e2e4 e7e5");
        let bytes = body(
            &root,
            ActionWitnessKind::Counter,
            &actual,
            &estimate(&root, ObservationKind::Refutation, &proposal),
        );
        let observation = BODY_DOMAIN.len() + 32 + 8 + 32;
        for offset in [
            0,
            BODY_DOMAIN.len(),
            BODY_DOMAIN.len() + 32,
            BODY_DOMAIN.len() + 40,
            observation + 1,
        ] {
            let mut changed = bytes.clone();
            changed[offset] ^= 1;
            assert!(
                validate_utility_witness_preimage(
                    &changed,
                    expected(&root, ActionWitnessKind::Counter, &actual, &changed)
                )
                .is_err()
            );
        }
        let mut presence = bytes.clone();
        presence[observation + 1 + 32] = 2;
        assert_eq!(
            validate_utility_witness_preimage(
                &presence,
                expected(&root, ActionWitnessKind::Counter, &actual, &presence)
            ),
            Err(Error::Invalid("presence tag"))
        );
        let mut packed = bytes.clone();
        let move_offset = observation + 1 + 32 + 1 + 8;
        packed[move_offset..move_offset + 2].copy_from_slice(&u16::MAX.to_le_bytes());
        assert_eq!(
            validate_utility_witness_preimage(
                &packed,
                expected(&root, ActionWitnessKind::Counter, &actual, &packed)
            ),
            Err(Error::Invalid("Move16 bits"))
        );
        let mut nan = bytes.clone();
        let float_offset = nan.len() - 4;
        nan[float_offset..].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        assert_eq!(
            validate_utility_witness_preimage(
                &nan,
                expected(&root, ActionWitnessKind::Counter, &actual, &nan)
            ),
            Err(Error::Invalid("non-finite float"))
        );
        let mut perspective = bytes.clone();
        let color_offset = perspective.len() - 5;
        perspective[color_offset] = 1;
        assert_eq!(
            validate_utility_witness_preimage(
                &perspective,
                expected(&root, ActionWitnessKind::Counter, &actual, &perspective)
            ),
            Err(Error::Mismatch("root Estimate perspective"))
        );
        let mut wrong_hash = expected(&root, ActionWitnessKind::Counter, &actual, &bytes);
        wrong_hash.raw_evidence.0[0] ^= 1;
        assert_eq!(
            validate_utility_witness_preimage(&bytes, wrong_hash),
            Err(Error::Mismatch("raw evidence SHA-256"))
        );
    }

    #[test]
    fn legal_replay_distinguishes_opponent_counter_from_own_turn_and_repair_mismatch() {
        let root = Position::startpos().snapshot();
        let cases = [
            (line("e2e4 e7e5 g1f3"), line("e2e4 e7e5 b1c3")), // Diverges on White's turn.
            (line("e2e4 e7e5"), line("d2d4 d7d5")),           // Different first move.
            (line("e2e4 e7e5"), line("e2e4 e7e5")),           // No divergence.
            (line("e2e4 e7e5"), line("e2e4 e7e6 e1e3")),      // Illegal continuation.
        ];
        for (proposal, actual) in cases {
            let bytes = body(
                &root,
                ActionWitnessKind::Counter,
                &actual,
                &estimate(&root, ObservationKind::Refutation, &proposal),
            );
            assert!(
                validate_utility_witness_preimage(
                    &bytes,
                    expected(&root, ActionWitnessKind::Counter, &actual, &bytes)
                )
                .is_err()
            );
        }
        let actual = line("e2e4 c7c5");
        let other = line("e2e4 e7e5");
        let bytes = body(
            &root,
            ActionWitnessKind::Repair,
            &actual,
            &estimate(&root, ObservationKind::Repair, &other),
        );
        assert_eq!(
            validate_utility_witness_preimage(
                &bytes,
                expected(&root, ActionWitnessKind::Repair, &actual, &bytes)
            ),
            Err(Error::Mismatch("Repair observation line"))
        );
    }

    #[test]
    fn recursive_raw_wdl_comparison_checks_framing_values_and_exact_rules_endpoints() {
        let root = Position::startpos().snapshot();
        let proposal = line("e2e4 e7e5");
        let actual = line("e2e4 c7c5");
        let before = replay(&root, &proposal).unwrap();
        let after = replay(&root, &actual).unwrap();
        let mut raw = common(&root, ObservationKind::Refutation, Some(&proposal));
        raw.extend_from_slice(&[0, 5, 0]);
        raw.extend_from_slice(&(-0.5_f32).to_bits().to_le_bytes());
        raw.extend_from_slice(&wdl(&before, [0.75, 0.0, 0.25]));
        let after_offset = raw.len();
        raw.extend_from_slice(&wdl(&after, [0.25, 0.0, 0.75]));
        let bytes = body(&root, ActionWitnessKind::Counter, &actual, &raw);
        validate_utility_witness_preimage(
            &bytes,
            expected(&root, ActionWitnessKind::Counter, &actual, &bytes),
        )
        .unwrap();
        let mut wrong_endpoint = raw.clone();
        wrong_endpoint[after_offset + 1] ^= 1;
        let bytes = body(&root, ActionWitnessKind::Counter, &actual, &wrong_endpoint);
        assert_eq!(
            validate_utility_witness_preimage(
                &bytes,
                expected(&root, ActionWitnessKind::Counter, &actual, &bytes)
            ),
            Err(Error::Mismatch(
                "conditional raw endpoint Rules/perspective"
            ))
        );
        let mut bad_wdl = raw;
        let final_loss = bad_wdl.len() - 4;
        bad_wdl[final_loss..].copy_from_slice(&0.5_f32.to_bits().to_le_bytes());
        let bytes = body(&root, ActionWitnessKind::Counter, &actual, &bad_wdl);
        assert_eq!(
            validate_utility_witness_preimage(
                &bytes,
                expected(&root, ActionWitnessKind::Counter, &actual, &bytes)
            ),
            Err(Error::Invalid("raw WDL"))
        );
    }

    #[test]
    fn validated_projection_keeps_exact_raw_values_and_borrowed_namespaces() {
        let root = Position::startpos().snapshot();
        let proposal = line("e2e4 e7e5");
        let actual = line("e2e4 c7c5");
        let raw = estimate(&root, ObservationKind::Refutation, &proposal);
        let bytes = body(&root, ActionWitnessKind::Counter, &actual, &raw);
        let projected = parse_validated_utility_witness_preimage(
            &bytes,
            expected(&root, ActionWitnessKind::Counter, &actual, &bytes),
        )
        .unwrap();
        assert_eq!(projected.kind, ObservationKind::Refutation);
        assert_eq!(projected.line.as_deref(), Some(proposal.as_slice()));
        assert!(projected.cpu_pv.is_none());
        assert_eq!(projected.budget, 0);
        assert_eq!(projected.cpu_condition, "");
        assert!(projected.cpu_identity.is_none() && projected.model_identity.is_none());
        assert!(projected.profiled_scope.is_none());
        let HistoricalWitnessRawScore::Estimate { value, perspective } = projected.score else {
            panic!("authored Estimate raw projection")
        };
        assert_eq!(value.to_bits(), (-100.0_f32).to_bits());
        assert_eq!(perspective, Color::White);

        let before = replay(&root, &proposal).unwrap();
        let after = replay(&root, &actual).unwrap();
        let mut raw = common(&root, ObservationKind::Refutation, Some(&proposal));
        raw.extend_from_slice(&[0, 5, 0]);
        raw.extend_from_slice(&(-0.5_f32).to_bits().to_le_bytes());
        raw.extend_from_slice(&wdl(&before, [0.75, 0.0, 0.25]));
        raw.extend_from_slice(&wdl(&after, [0.25, 0.0, 0.75]));
        let bytes = body(&root, ActionWitnessKind::Counter, &actual, &raw);
        let projected = parse_validated_utility_witness_preimage(
            &bytes,
            expected(&root, ActionWitnessKind::Counter, &actual, &bytes),
        )
        .unwrap();
        let HistoricalWitnessRawScore::ConditionalWdl {
            expectation,
            perspective,
            repaired,
            counter,
        } = projected.score
        else {
            panic!("authored conditional raw projection")
        };
        assert_eq!(expectation.to_bits(), (-0.5_f32).to_bits());
        assert_eq!(perspective, Color::White);
        assert_eq!(
            repaired.state_digest,
            rz_position::contracts::rules_snapshot_semantic_digest(&before)
        );
        assert_eq!(
            counter.state_digest,
            rz_position::contracts::rules_snapshot_semantic_digest(&after)
        );
        let identity = counter.model_identity.unwrap();
        assert_eq!(identity.semantics, "context-wdl");
        assert_eq!(identity.model, "authored-model");
        assert_eq!(identity.encoding, "authored-encoding");
        assert_eq!(identity.precision, "fp32_tf32_off");
        assert_eq!(identity.epoch, [17; 32]);
        let HistoricalWitnessRawScore::Wdl {
            win,
            draw,
            loss,
            perspective,
        } = counter.score
        else {
            panic!("tag4 remains an unversioned Wdl/ContextWdl projection")
        };
        assert_eq!(
            [win.to_bits(), draw.to_bits(), loss.to_bits()],
            [0.25_f32.to_bits(), 0.0_f32.to_bits(), 0.75_f32.to_bits()]
        );
        assert_eq!(perspective, Color::White);
    }

    #[test]
    fn raw_reader_rejects_invalid_utf8_constraint_clamps_and_recursion_limits() {
        let root = Position::startpos().snapshot();
        let mut invalid_utf8 = vec![ObservationKind::CpuAnalysis as u8];
        invalid_utf8
            .extend_from_slice(&rz_position::contracts::rules_snapshot_semantic_digest(&root).0);
        invalid_utf8.extend_from_slice(&[0, 0]);
        invalid_utf8.extend_from_slice(&0_u64.to_le_bytes());
        invalid_utf8.extend_from_slice(&1_u64.to_le_bytes());
        invalid_utf8.push(255);
        fn cursor(bytes: &[u8]) -> Cursor<'_> {
            Cursor {
                bytes,
                offset: 0,
                nodes: 0,
            }
        }
        assert!(matches!(
            cursor(&invalid_utf8).observation(1),
            Err(Error::Invalid("UTF-8 string"))
        ));
        assert!(matches!(
            cursor(&[]).observation(MAX_DEPTH),
            Err(Error::Limit("observation depth"))
        ));
        assert!(matches!(
            Cursor {
                bytes: &[],
                offset: 0,
                nodes: MAX_NODES
            }
            .observation(1),
            Err(Error::Limit("observation nodes"))
        ));
        let mut too_long = 8193_u64.to_le_bytes().to_vec();
        assert!(matches!(
            cursor(&too_long).string(8192, true),
            Err(Error::Limit("string bytes"))
        ));
        too_long = [vec![3], 180_001_u64.to_le_bytes().to_vec()].concat();
        assert!(matches!(
            cursor(&too_long).constraint(),
            Err(Error::Invalid("profiled constraint bounds"))
        ));
        let mut wrong_clamp = PROFILED_DOMAIN.to_vec();
        wrong_clamp.push(3);
        wrong_clamp.extend_from_slice(&10_u64.to_le_bytes());
        wrong_clamp.push(3);
        wrong_clamp.extend_from_slice(&11_u64.to_le_bytes());
        assert!(matches!(
            cursor(&wrong_clamp).observation(1),
            Err(Error::Invalid("issued constraint"))
        ));
    }
}
