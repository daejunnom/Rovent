//! Explicit NNUEC V2: typed immutable lines and directed review relations.
//! V1 checkpoints and numerical evidence are not converted or re-labelled.
//! A line depends only on its own exact anchor, moves, kind and frozen model.
//! A pair depends only on its two immutable endpoints and relation kind; an
//! unrelated publication therefore cannot change a cached contribution.

use super::{
    Exact256, NNUEC_MAX_BYTES, NnuecBoard, NnuecCheckpoint, NnuecError, NnuecLocal, NnuecModel,
    NnuecOutput, PARAMETER_MAGNITUDE, ReferenceLine, ReferenceMove, add, board_features,
    color_name, mul, project,
};
use crate::cpu_value::CpuValueIdentity;
use rz_position::{BoardMove, Color, Position, PositionIdentity};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::sync::Arc;

#[path = "nnuec_v2_capture.rs"]
pub mod capture;
#[path = "nnuec_v2_cpu.rs"]
pub mod cpu;
#[path = "nnuec_v2_registration.rs"]
pub mod registration;
#[path = "nnuec_v2_retirement.rs"]
pub mod retirement;
#[path = "nnuec_v2_review.rs"]
pub mod review;
#[cfg(feature = "nnuec-scalar-diagnostics")]
#[path = "nnuec_v2_scalar_diagnostic.rs"]
pub mod scalar_diagnostic;
#[path = "nnuec_v2_utility.rs"]
pub mod utility;
#[path = "nnuec_v2_utility_capture.rs"]
pub mod utility_capture;

pub const SCHEMA: &str = "rz-nnuec-fp32-v2";
pub const SEMANTICS: &str =
    "board800-exactb192-causal-line-kind4-directedpair-originfirst-exactc256-fp32-v2";
pub const INPUT_SCHEMA: &str = "rz-nnuec-fp32-input/2";
pub const OUTPUT_SCHEMA: &str = "rz-nnuec-fp32-output/2";
// V1 bounds give raw E < 1.032e12. Adding a bounded kind row does not
// reach 1.033e12. Two diagonal products + bias yield pair E < 2.067e15,
// and a 256-column projection is < 5.3e20, including FP32 rounding slack.
// We enforce a looser operand bound 2^70, not arbitrary finite FP32 support.
// <=128 lines + <=256 directed edges + one replacement transient sum to
// less than 2^79; scaling by 2^149 needs fewer than 228 magnitude bits.
const MAX_OPERAND: f32 = 1_180_591_620_717_411_303_424.0; // 2^70
const MAX_SUM: f32 = 604_462_909_807_314_587_353_088.0; // 2^79

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum LineKind {
    Proposal,
    Counterexample,
    Repair,
    CpuVerification,
}
impl TryFrom<u8> for LineKind {
    type Error = NnuecError;
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Proposal),
            1 => Ok(Self::Counterexample),
            2 => Ok(Self::Repair),
            3 => Ok(Self::CpuVerification),
            _ => Err(invalid("unknown record kind")),
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum EdgeKind {
    Parent,
    Supersedes,
}
impl TryFrom<u8> for EdgeKind {
    type Error = NnuecError;
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Parent),
            1 => Ok(Self::Supersedes),
            _ => Err(invalid("unknown relation kind")),
        }
    }
}
/// Source is the parent/superseded line; target is the child/new line. IDs
/// perform lookup only. Their magnitude/order never enters a neural feature.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Edge {
    pub source: u64,
    pub target: u64,
    pub kind: EdgeKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelationParameters {
    pub kind_embeddings: Vec<f32>,
    pub pair_parent: Vec<f32>,
    pub pair_child: Vec<f32>,
    pub pair_bias: Vec<f32>,
    pub pair_projection: Vec<f32>,
}
impl RelationParameters {
    fn arrays(&self) -> [&Vec<f32>; 5] {
        [
            &self.kind_embeddings,
            &self.pair_parent,
            &self.pair_child,
            &self.pair_bias,
            &self.pair_projection,
        ]
    }
    fn validate(&self, base: &NnuecCheckpoint) -> Result<(), NnuecError> {
        let e = base.config.line_width;
        for (values, length) in self.arrays().into_iter().zip([
            4 * e,
            2 * e,
            2 * e,
            2 * e,
            base.config.context_width * e,
        ]) {
            if values.len() != length || values.capacity() > (length * 2).max(4) {
                return Err(invalid("V2 relation parameter shape/allocation mismatch"));
            }
            if values
                .iter()
                .any(|value| !value.is_finite() || value.abs() > PARAMETER_MAGNITUDE)
            {
                return Err(invalid("V2 non-finite/excessive relation parameter"));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    pub schema: String,
    pub semantics: String,
    pub base: NnuecCheckpoint,
    pub parameters: RelationParameters,
}
impl Checkpoint {
    pub fn validate(&self) -> Result<(), NnuecError> {
        if self.schema != SCHEMA
            || self.semantics != SEMANTICS
            || self.schema.capacity() > 256
            || self.semantics.capacity() > 256
        {
            return Err(invalid(
                "V2 schema/semantics mismatch; no automatic conversion",
            ));
        }
        self.base.validate()?;
        self.parameters.validate(&self.base)
    }
    pub fn weights_sha256(&self) -> Result<String, NnuecError> {
        self.validate()?;
        let mut hash = Sha256::new();
        hash.update(SCHEMA.as_bytes());
        hash.update([0]);
        hash.update(SEMANTICS.as_bytes());
        hash.update([0]);
        hash.update(self.base.weights_sha256()?.as_bytes());
        hash.update([0]);
        for values in self.parameters.arrays() {
            for value in values {
                hash.update(value.to_bits().to_le_bytes());
            }
        }
        Ok(format!("{:x}", hash.finalize()))
    }
}

pub struct Model {
    base: NnuecModel,
    parameters: Arc<RelationParameters>,
    identity: Arc<CpuValueIdentity>,
}
impl Model {
    pub fn new(checkpoint: Checkpoint) -> Result<Self, NnuecError> {
        let weights_sha256 = checkpoint.weights_sha256()?;
        let identity = Arc::new(CpuValueIdentity {
            semantics: SEMANTICS.into(),
            weights_sha256: Some(weights_sha256),
            training: checkpoint.base.training.clone(),
        });
        // Move base parameters into the existing owner, not a second full copy.
        Ok(Self {
            base: NnuecModel::new(checkpoint.base)?,
            parameters: Arc::new(checkpoint.parameters),
            identity,
        })
    }
    pub fn from_reader(reader: impl Read) -> Result<Self, NnuecError> {
        let mut bytes = Vec::new();
        reader
            .take((NNUEC_MAX_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|e| NnuecError::Io(e.to_string()))?;
        if bytes.len() > NNUEC_MAX_BYTES {
            return Err(NnuecError::Capacity("V2 checkpoint bytes"));
        }
        let checkpoint =
            serde_json::from_slice(&bytes).map_err(|e| NnuecError::Json(e.to_string()))?;
        Self::new(checkpoint)
    }
    pub fn identity(&self) -> &CpuValueIdentity {
        &self.identity
    }
    pub fn base_checkpoint(&self) -> &NnuecCheckpoint {
        self.base.checkpoint()
    }
    pub fn relation_parameters(&self) -> &RelationParameters {
        &self.parameters
    }
    pub fn board(&self, position: &Position) -> Result<NnuecBoard, NnuecError> {
        self.base.board(position)
    }
    pub fn local(&self, anchor: &Position) -> Result<NnuecLocal, NnuecError> {
        self.base.local(anchor)
    }
    pub fn append_local(
        &self,
        local: &mut NnuecLocal,
        movement: BoardMove,
    ) -> Result<super::NnuecLocalUndo, NnuecError> {
        self.base.append_local(local, movement)
    }
    pub fn undo_local(
        &self,
        local: &mut NnuecLocal,
        undo: super::NnuecLocalUndo,
    ) -> Result<(), NnuecError> {
        self.base.undo_local(local, undo)
    }
    pub fn encode_line(
        &self,
        anchor: &Position,
        moves: &[BoardMove],
        kind: LineKind,
    ) -> Result<Arc<Line>, NnuecError> {
        self.encode_review_line(anchor, moves, kind, moves.first().copied())
    }
    /// Only the checked ReviewView factory may preserve an original bundle
    /// move after Rules rebase removes it from the actual encoded suffix.
    fn encode_review_line(
        &self,
        anchor: &Position,
        moves: &[BoardMove],
        kind: LineKind,
        original_first_move: Option<BoardMove>,
    ) -> Result<Arc<Line>, NnuecError> {
        let raw = self.base.encode_line(anchor, moves)?;
        let e = self.base.checkpoint.config.line_width;
        let typed = raw
            .embedding
            .iter()
            .enumerate()
            .map(|(j, value)| {
                add(
                    *value,
                    self.parameters.kind_embeddings[kind as usize * e + j],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let contribution = project(
            &typed,
            &self.base.checkpoint.parameters.context_projection,
            &vec![0.0; self.base.checkpoint.config.context_width],
        )?;
        admit_contribution(&contribution)?;
        Ok(Arc::new(Line {
            raw,
            kind,
            typed,
            contribution,
            model: Arc::clone(&self.identity),
            original_first_move,
        }))
    }
    pub fn empty_context(&self) -> Arc<Context> {
        let exact: Vec<_> = self
            .base
            .checkpoint
            .parameters
            .context_bias
            .iter()
            .map(|value| exact_operand(*value).expect("admitted bias"))
            .collect();
        let values = rounded(&exact).expect("bounded admitted bias");
        Arc::new(Context {
            model: Arc::clone(&self.identity),
            authority: Arc::new(()),
            lines: BTreeMap::new(),
            edges: BTreeMap::new(),
            line_order: Vec::new(),
            edge_order: Vec::new(),
            exact,
            values,
        })
    }
    pub fn with_line(
        &self,
        context: &Arc<Context>,
        id: u64,
        line: Arc<Line>,
    ) -> Result<Arc<Context>, NnuecError> {
        self.check_context(context)?;
        if line.model != self.identity {
            return Err(NnuecError::StateMismatch);
        }
        let old = context.lines.get(&id);
        if old.is_none() && context.lines.len() >= self.base.checkpoint.config.max_lines {
            return Err(NnuecError::Capacity("V2 context lines"));
        }
        let mut next = context.copy_for_change();
        if let Some(old) = old {
            change(&mut next.exact, &old.contribution, true)?;
        }
        change(&mut next.exact, &line.contribution, false)?;
        next.lines.insert(id, line);
        if old.is_none() {
            next.line_order.push(id);
        }
        // Recompute only incoming/outgoing pairs of this immutable line version.
        for (edge, old_pair) in &context.edges {
            if edge.source == id || edge.target == id {
                change(&mut next.exact, &old_pair.contribution, true)?;
                let pair = self.pair(*edge, &next.lines)?;
                change(&mut next.exact, &pair.contribution, false)?;
                next.edges.insert(*edge, pair);
            }
        }
        next.values = rounded(&next.exact)?;
        Ok(Arc::new(next))
    }
    pub fn with_edge(
        &self,
        context: &Arc<Context>,
        edge: Edge,
    ) -> Result<Arc<Context>, NnuecError> {
        self.check_context(context)?;
        if context.edges.contains_key(&edge) {
            return Err(invalid("duplicate V2 relation"));
        }
        if context
            .edges
            .keys()
            .any(|old| old.target == edge.target && old.kind == edge.kind)
        {
            return Err(invalid("more than one parent/supersedes for a record"));
        }
        if context.edges.len() >= 2 * self.base.checkpoint.config.max_lines {
            return Err(NnuecError::Capacity("V2 context relations"));
        }
        let pair = self.pair(edge, &context.lines)?;
        let mut next = context.copy_for_change();
        next.edges.insert(edge, pair);
        if cyclic(&next.lines, next.edges.keys().copied()) {
            return Err(invalid("cyclic V2 relations"));
        }
        change(&mut next.exact, &next.edges[&edge].contribution, false)?;
        next.edge_order.push(edge);
        next.values = rounded(&next.exact)?;
        Ok(Arc::new(next))
    }
    pub fn without_edge(
        &self,
        context: &Arc<Context>,
        edge: Edge,
    ) -> Result<Arc<Context>, NnuecError> {
        self.check_context(context)?;
        let old = context
            .edges
            .get(&edge)
            .ok_or(invalid("unknown V2 relation"))?;
        let mut next = context.copy_for_change();
        change(&mut next.exact, &old.contribution, true)?;
        next.edges.remove(&edge);
        next.edge_order.retain(|value| *value != edge);
        next.values = rounded(&next.exact)?;
        Ok(Arc::new(next))
    }
    /// An isolated line may be removed directly; connected records require a
    /// coherent bundle closure, so no dangling relation is silently forgotten.
    pub fn without_line(
        &self,
        context: &Arc<Context>,
        id: u64,
    ) -> Result<Arc<Context>, NnuecError> {
        self.check_context(context)?;
        if context
            .edges
            .keys()
            .any(|edge| edge.source == id || edge.target == id)
        {
            return Err(invalid("V2 line still has incident relations"));
        }
        let old = context.lines.get(&id).ok_or(invalid("unknown V2 line"))?;
        let mut next = context.copy_for_change();
        change(&mut next.exact, &old.contribution, true)?;
        next.lines.remove(&id);
        next.line_order.retain(|value| *value != id);
        next.values = rounded(&next.exact)?;
        Ok(Arc::new(next))
    }
    pub fn without_bundle(
        &self,
        context: &Arc<Context>,
        ids: &[u64],
    ) -> Result<Arc<Context>, NnuecError> {
        self.check_context(context)?;
        if ids.is_empty() || ids.len() > context.lines.len() {
            return Err(invalid("V2 bundle removal has duplicate/missing records"));
        }
        let removed: BTreeSet<_> = ids.iter().copied().collect();
        if removed.len() != ids.len()
            || removed.is_empty()
            || ids.iter().any(|id| !context.lines.contains_key(id))
        {
            return Err(invalid("V2 bundle removal has duplicate/missing records"));
        }
        if context
            .edges
            .keys()
            .any(|edge| removed.contains(&edge.source) != removed.contains(&edge.target))
        {
            return Err(invalid("V2 bundle removal leaves a dangling relation"));
        }
        let mut next = context.copy_for_change();
        for id in ids {
            change(&mut next.exact, &context.lines[id].contribution, true)?;
            next.lines.remove(id);
        }
        for (edge, pair) in &context.edges {
            if removed.contains(&edge.source) {
                change(&mut next.exact, &pair.contribution, true)?;
                next.edges.remove(edge);
            }
        }
        next.line_order.retain(|id| !removed.contains(id));
        next.edge_order
            .retain(|edge| !removed.contains(&edge.source));
        next.values = rounded(&next.exact)?;
        Ok(Arc::new(next))
    }
    pub fn rebuild_context(&self, context: &Context) -> Result<Arc<Context>, NnuecError> {
        self.check_context(context)?;
        let mut next = context.copy_for_change();
        next.exact = self.empty_context().exact.clone();
        for line in next.lines.values() {
            change(&mut next.exact, &line.contribution, false)?;
        }
        for pair in next.edges.values() {
            change(&mut next.exact, &pair.contribution, false)?;
        }
        next.values = rounded(&next.exact)?;
        Ok(Arc::new(next))
    }
    fn check_context(&self, context: &Context) -> Result<(), NnuecError> {
        if context.model != self.identity {
            return Err(NnuecError::StateMismatch);
        }
        Ok(())
    }
    fn pair(&self, edge: Edge, lines: &BTreeMap<u64, Arc<Line>>) -> Result<Arc<Pair>, NnuecError> {
        if edge.source == edge.target {
            return Err(invalid("self V2 relation"));
        }
        let parent = lines
            .get(&edge.source)
            .ok_or(invalid("dangling V2 relation source"))?;
        let child = lines
            .get(&edge.target)
            .ok_or(invalid("dangling V2 relation target"))?;
        if parent.raw.anchor != child.raw.anchor
            || parent.original_first_move.is_none()
            || parent.original_first_move != child.original_first_move
        {
            return Err(invalid(
                "V2 relation exact anchor/original first move mismatch",
            ));
        }
        let e = self.base.checkpoint.config.line_width;
        let start = edge.kind as usize * e;
        let mut embedding = Vec::with_capacity(e);
        for j in 0..e {
            embedding.push(
                add(
                    add(
                        self.parameters.pair_bias[start + j],
                        mul(parent.typed[j], self.parameters.pair_parent[start + j])?,
                    )?,
                    mul(child.typed[j], self.parameters.pair_child[start + j])?,
                )?
                .max(0.0),
            );
        }
        let contribution = project(
            &embedding,
            &self.parameters.pair_projection,
            &vec![0.0; self.base.checkpoint.config.context_width],
        )?;
        admit_contribution(&contribution)?;
        Ok(Arc::new(Pair { contribution }))
    }
    pub fn forward(
        &self,
        board: &NnuecBoard,
        context: &Context,
        local: &NnuecLocal,
        position: &Position,
    ) -> Result<Output, NnuecError> {
        self.check_context(context)?;
        let mut numeric =
            self.base
                .forward_with_context_values(board, &context.values, local, position)?;
        numeric.schema = OUTPUT_SCHEMA.into();
        numeric.weights_sha256 = self
            .identity
            .weights_sha256
            .clone()
            .expect("admitted V2 identity");
        Ok(Output {
            numeric,
            context_sha256: context.sha256()?,
        })
    }
    pub fn cache_context(
        &self,
        context: &Context,
        local: &NnuecLocal,
        position: &Position,
    ) -> Result<crate::cpu_value::CpuValueCacheContext, NnuecError> {
        self.check_context(context)?;
        if local.model != self.base.identity
            || local.work.position_identity() != position.position_identity()
        {
            return Err(NnuecError::StateMismatch);
        }
        Ok(crate::cpu_value::CpuValueCacheContext::contextual(
            &context.authority,
            &local.anchor,
        ))
    }
    pub fn reference_input(
        &self,
        context: &Context,
        local: &NnuecLocal,
        position: &Position,
    ) -> Result<ReferenceInput, NnuecError> {
        self.check_context(context)?;
        if local.model != self.base.identity
            || local.work.position_identity() != position.position_identity()
        {
            return Err(NnuecError::StateMismatch);
        }
        Ok(ReferenceInput {
            schema: INPUT_SCHEMA.into(),
            board_features: [
                board_features(position, Color::White),
                board_features(position, Color::Black),
            ],
            side_to_move: color_name(position.side_to_move()).into(),
            context_lines: context.reference_lines(),
            context_edges: context
                .edge_order
                .iter()
                .map(|edge| ReferenceEdge {
                    source: edge.source,
                    target: edge.target,
                    kind: edge.kind as u8,
                })
                .collect(),
            local_line: ReferenceLine {
                anchor_features: local.anchor_features.clone(),
                perspective: color_name(local.perspective).into(),
                moves: local
                    .moves
                    .iter()
                    .copied()
                    .map(ReferenceMove::from_board)
                    .collect(),
            },
        })
    }
}

pub struct Line {
    raw: Arc<super::NnuecLine>,
    kind: LineKind,
    typed: Vec<f32>,
    contribution: Vec<f32>,
    model: Arc<CpuValueIdentity>,
    original_first_move: Option<BoardMove>,
}
impl Line {
    pub fn kind(&self) -> LineKind {
        self.kind
    }
    pub fn anchor(&self) -> &PositionIdentity {
        &self.raw.anchor
    }
    pub fn typed_representation(&self) -> &[f32] {
        &self.typed
    }
    pub fn contribution(&self) -> &[f32] {
        &self.contribution
    }
}
struct Pair {
    contribution: Vec<f32>,
}
pub struct Context {
    model: Arc<CpuValueIdentity>,
    authority: Arc<()>,
    lines: BTreeMap<u64, Arc<Line>>,
    edges: BTreeMap<Edge, Arc<Pair>>,
    line_order: Vec<u64>,
    edge_order: Vec<Edge>,
    exact: Vec<Exact256>,
    values: Vec<f32>,
}
impl Context {
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
    pub fn line(&self, id: u64) -> Option<&Arc<Line>> {
        self.lines.get(&id)
    }
    fn copy_for_change(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
            authority: Arc::new(()),
            lines: self.lines.clone(),
            edges: self.edges.clone(),
            line_order: self.line_order.clone(),
            edge_order: self.edge_order.clone(),
            exact: self.exact.clone(),
            values: self.values.clone(),
        }
    }
    fn reference_lines(&self) -> Vec<ReferenceLineV2> {
        self.line_order
            .iter()
            .map(|id| {
                let line = &self.lines[id];
                ReferenceLineV2 {
                    id: *id,
                    kind: line.kind as u8,
                    original_first_move: line.original_first_move.map(ReferenceMove::from_board),
                    anchor_features: line.raw.input.anchor_features.clone(),
                    perspective: line.raw.input.perspective.clone(),
                    moves: line.raw.input.moves.clone(),
                }
            })
            .collect()
    }
    pub fn sha256(&self) -> Result<String, NnuecError> {
        context_sha256(
            &self.reference_lines(),
            &self
                .edge_order
                .iter()
                .map(|edge| ReferenceEdge {
                    source: edge.source,
                    target: edge.target,
                    kind: edge.kind as u8,
                })
                .collect::<Vec<_>>(),
        )
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Output {
    #[serde(flatten)]
    pub numeric: NnuecOutput,
    pub context_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceInput {
    pub schema: String,
    pub board_features: [Vec<usize>; 2],
    pub side_to_move: String,
    pub context_lines: Vec<ReferenceLineV2>,
    pub context_edges: Vec<ReferenceEdge>,
    pub local_line: ReferenceLine,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceLineV2 {
    pub id: u64,
    pub kind: u8,
    /// Bundle admission provenance, not an additional numerical feature. The
    /// actual rebased suffix may begin elsewhere or be empty.
    pub original_first_move: Option<ReferenceMove>,
    pub anchor_features: Vec<usize>,
    pub perspective: String,
    pub moves: Vec<ReferenceMove>,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceEdge {
    pub source: u64,
    pub target: u64,
    pub kind: u8,
}

pub fn context_sha256(
    lines: &[ReferenceLineV2],
    edges: &[ReferenceEdge],
) -> Result<String, NnuecError> {
    if lines.len() > 128 || edges.len() > 256 {
        return Err(NnuecError::Capacity("V2 context seal"));
    }
    let mut hash = Sha256::new();
    hash.update(INPUT_SCHEMA.as_bytes());
    hash.update(b"\0context\0");
    hash.update((lines.len() as u32).to_le_bytes());
    for line in lines {
        LineKind::try_from(line.kind)?;
        let side = match line.perspective.as_str() {
            "white" => 0u8,
            "black" => 1u8,
            _ => return Err(invalid("V2 context seal perspective")),
        };
        if line.anchor_features.len() > 71 || line.moves.len() > 256 {
            return Err(NnuecError::Capacity("V2 context seal line"));
        }
        hash.update(line.id.to_le_bytes());
        hash.update([line.kind, side]);
        match &line.original_first_move {
            Some(movement) => {
                validate_reference_move(movement)?;
                hash.update([
                    1,
                    movement.from_square,
                    movement.to_square,
                    movement.promotion,
                ]);
            }
            None => hash.update([0]),
        }
        hash.update((line.anchor_features.len() as u32).to_le_bytes());
        for feature in &line.anchor_features {
            if *feature >= 800 {
                return Err(invalid("V2 context seal feature"));
            }
            hash.update((*feature as u16).to_le_bytes());
        }
        hash.update((line.moves.len() as u32).to_le_bytes());
        for movement in &line.moves {
            validate_reference_move(movement)?;
            hash.update([movement.from_square, movement.to_square, movement.promotion]);
        }
    }
    hash.update((edges.len() as u32).to_le_bytes());
    for edge in edges {
        EdgeKind::try_from(edge.kind)?;
        hash.update(edge.source.to_le_bytes());
        hash.update(edge.target.to_le_bytes());
        hash.update([edge.kind]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn validate_reference_move(movement: &ReferenceMove) -> Result<(), NnuecError> {
    if movement.from_square >= 64
        || movement.to_square >= 64
        || movement.from_square == movement.to_square
        || movement.promotion > 4
    {
        return Err(invalid("V2 context seal move"));
    }
    Ok(())
}
fn invalid(message: &'static str) -> NnuecError {
    NnuecError::InvalidCheckpoint(message)
}
fn exact_operand(value: f32) -> Result<Exact256, NnuecError> {
    Exact256::from_f32_bounded(value, MAX_OPERAND)
}
fn admit_contribution(values: &[f32]) -> Result<(), NnuecError> {
    for value in values {
        exact_operand(*value)?;
    }
    Ok(())
}
fn change(sum: &mut [Exact256], values: &[f32], remove: bool) -> Result<(), NnuecError> {
    if sum.len() != values.len() {
        return Err(NnuecError::StateMismatch);
    }
    for (sum, value) in sum.iter_mut().zip(values) {
        let operand = exact_operand(*value)?;
        *sum = sum.checked_add(if remove { operand.negated() } else { operand })?;
    }
    Ok(())
}
fn rounded(sum: &[Exact256]) -> Result<Vec<f32>, NnuecError> {
    sum.iter()
        .map(|value| {
            let value = value.to_f32();
            if !value.is_finite() || value.abs() >= MAX_SUM {
                Err(NnuecError::AccumulatorOverflow)
            } else {
                Ok(value)
            }
        })
        .collect()
}
fn cyclic(lines: &BTreeMap<u64, Arc<Line>>, edges: impl Iterator<Item = Edge>) -> bool {
    let mut incoming: BTreeMap<_, usize> = lines.keys().map(|id| (*id, 0)).collect();
    let mut outgoing: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for edge in edges {
        *incoming.get_mut(&edge.target).expect("validated target") += 1;
        outgoing.entry(edge.source).or_default().push(edge.target);
    }
    let mut ready: Vec<_> = incoming
        .iter()
        .filter_map(|(id, count)| (*count == 0).then_some(*id))
        .collect();
    let mut visited = 0;
    while let Some(id) = ready.pop() {
        visited += 1;
        for target in outgoing.get(&id).into_iter().flatten() {
            let count = incoming.get_mut(target).expect("validated target");
            *count -= 1;
            if *count == 0 {
                ready.push(*target);
            }
        }
    }
    visited != lines.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_value::CpuTrainingState;
    use crate::cpu_value::nnuec::{NnuecConfig, NnuecParameters};

    pub(super) fn checkpoint() -> Checkpoint {
        let config = NnuecConfig {
            board_width: 1,
            line_width: 1,
            context_width: 1,
            local_width: 1,
            hidden_width: 1,
            causal_window: 3,
            max_lines: 8,
            max_plies: 16,
        };
        Checkpoint {
            schema: SCHEMA.into(),
            semantics: SEMANTICS.into(),
            base: NnuecCheckpoint {
                schema: super::super::NNUEC_SCHEMA.into(),
                semantics: super::super::NNUEC_SEMANTICS.into(),
                training: CpuTrainingState::Untrained,
                output_scale: 1.0,
                parameters: NnuecParameters {
                    board_weights: vec![0.0; 800],
                    board_bias: vec![0.0],
                    anchor_weights: vec![0.0; 800],
                    anchor_bias: vec![0.0],
                    from_weights: (0..64).map(|i| i as f32 / 64.0).collect(),
                    to_weights: vec![0.0; 64],
                    promotion_weights: vec![0.0; 5],
                    ply_weights: vec![0.0; 16],
                    token_bias: vec![1.0],
                    causal_weights: vec![1.0, 0.5, 0.25],
                    causal_bias: vec![0.0],
                    context_projection: vec![1.0],
                    context_bias: vec![0.125],
                    local_projection: vec![0.0],
                    local_bias: vec![0.0],
                    hidden_weights: vec![0.0, 0.0, 1.0, 0.0],
                    hidden_bias: vec![0.0],
                    output_weights: vec![1.0; 11],
                    output_bias: vec![0.0; 11],
                },
                config,
            },
            parameters: RelationParameters {
                kind_embeddings: vec![0.0, 1.0, 2.0, 3.0],
                pair_parent: vec![2.0, 3.0],
                pair_child: vec![5.0, 7.0],
                pair_bias: vec![0.25, 0.5],
                pair_projection: vec![1.0],
            },
        }
    }
    fn moves(text: &str) -> Vec<BoardMove> {
        text.split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect()
    }
    fn edge(source: u64, target: u64, kind: EdgeKind) -> Edge {
        Edge {
            source,
            target,
            kind,
        }
    }
    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }
    fn populated(model: &Model, ids: [u64; 4]) -> Arc<Context> {
        let root = Position::startpos();
        let mut context = model.empty_context();
        for (id, kind, text) in [
            (ids[0], LineKind::Proposal, "e2e4"),
            (ids[1], LineKind::Counterexample, "e2e4 e7e5"),
            (ids[2], LineKind::Proposal, "d2d4"),
            (ids[3], LineKind::Repair, "d2d4 d7d5"),
        ] {
            context = model
                .with_line(
                    &context,
                    id,
                    model.encode_line(&root, &moves(text), kind).unwrap(),
                )
                .unwrap();
        }
        for relation in [
            edge(ids[0], ids[1], EdgeKind::Parent),
            edge(ids[2], ids[3], EdgeKind::Parent),
        ] {
            context = model.with_edge(&context, relation).unwrap();
        }
        context
    }

    #[test]
    fn v2_has_explicit_separate_checkpoint_and_parameter_identity() {
        let checkpoint = checkpoint();
        let original = checkpoint.weights_sha256().unwrap();
        assert_ne!(original, checkpoint.base.weights_sha256().unwrap());
        let mut changed = checkpoint.clone();
        changed.parameters.kind_embeddings[1] += 0.125;
        assert_ne!(original, changed.weights_sha256().unwrap());
        changed = checkpoint.clone();
        changed.schema = super::super::NNUEC_SCHEMA.into();
        assert!(changed.validate().is_err());
        changed = checkpoint.clone();
        changed.parameters.pair_child.pop();
        assert!(changed.validate().is_err());
        changed = checkpoint;
        changed.parameters.pair_parent[0] = f32::NAN;
        assert!(changed.validate().is_err());
    }

    #[test]
    fn identical_moves_distinguish_record_kind_and_directed_relation() {
        let model = Model::new(checkpoint()).unwrap();
        let root = Position::startpos();
        let proposal = model
            .encode_line(&root, &moves("e2e4"), LineKind::Proposal)
            .unwrap();
        let counter = model
            .encode_line(&root, &moves("e2e4"), LineKind::Counterexample)
            .unwrap();
        let repair = model
            .encode_line(&root, &moves("e2e4"), LineKind::Repair)
            .unwrap();
        assert_ne!(bits(proposal.contribution()), bits(counter.contribution()));
        assert_ne!(bits(counter.contribution()), bits(repair.contribution()));
        let base = model
            .with_line(&model.empty_context(), 0, proposal)
            .unwrap();
        let base = model.with_line(&base, 1, counter).unwrap();
        let parent = model
            .with_edge(&base, edge(0, 1, EdgeKind::Parent))
            .unwrap();
        let reverse = model
            .with_edge(&base, edge(1, 0, EdgeKind::Parent))
            .unwrap();
        let supersedes = model
            .with_edge(&base, edge(0, 1, EdgeKind::Supersedes))
            .unwrap();
        assert_ne!(bits(parent.values()), bits(reverse.values()));
        assert_ne!(bits(parent.values()), bits(supersedes.values()));
    }

    #[test]
    fn local_ids_and_selection_order_do_not_change_numeric_context() {
        let model = Model::new(checkpoint()).unwrap();
        let first = populated(&model, [0, 1, 2, 3]);
        let renamed = populated(&model, [77, 44, 12, 9]);
        assert_eq!(bits(first.values()), bits(renamed.values()));
        assert_ne!(first.sha256().unwrap(), renamed.sha256().unwrap());
        let mut reordered = model.empty_context();
        for id in [3, 1, 2, 0] {
            reordered = model
                .with_line(&reordered, id, Arc::clone(first.line(id).unwrap()))
                .unwrap();
        }
        for relation in [edge(2, 3, EdgeKind::Parent), edge(0, 1, EdgeKind::Parent)] {
            reordered = model.with_edge(&reordered, relation).unwrap();
        }
        assert_eq!(bits(first.values()), bits(reordered.values()));
        assert_ne!(first.sha256().unwrap(), reordered.sha256().unwrap());
        assert_eq!(
            bits(first.values()),
            bits(model.rebuild_context(&first).unwrap().values())
        );
    }

    #[test]
    fn replacing_a_line_recomputes_only_incident_pairs_and_old_context_is_frozen() {
        let model = Model::new(checkpoint()).unwrap();
        let old = populated(&model, [0, 1, 2, 3]);
        let old_values = bits(old.values());
        let old_seal = old.sha256().unwrap();
        let replacement = model
            .encode_line(&Position::startpos(), &moves("e2e4 c7c5"), LineKind::Repair)
            .unwrap();
        let next = model.with_line(&old, 1, replacement).unwrap();
        assert!(!Arc::ptr_eq(
            &old.edges[&edge(0, 1, EdgeKind::Parent)],
            &next.edges[&edge(0, 1, EdgeKind::Parent)]
        ));
        assert!(Arc::ptr_eq(
            &old.edges[&edge(2, 3, EdgeKind::Parent)],
            &next.edges[&edge(2, 3, EdgeKind::Parent)]
        ));
        assert_eq!(old_values, bits(old.values()));
        assert_eq!(old_seal, old.sha256().unwrap());
        assert_ne!(old_values, bits(next.values()));
        assert_eq!(
            bits(next.values()),
            bits(model.rebuild_context(&next).unwrap().values())
        );
        assert!(model.without_line(&next, 0).is_err());
        assert!(model.without_bundle(&next, &[0]).is_err());
        assert!(model.without_bundle(&next, &[0, 1, 1]).is_err());
        let remaining = model.without_bundle(&next, &[0, 1]).unwrap();
        assert_eq!(remaining.line_count(), 2);
        assert_eq!(remaining.edge_count(), 1);
        let empty = model.without_bundle(&remaining, &[2, 3]).unwrap();
        assert_eq!(bits(empty.values()), bits(model.empty_context().values()));
    }

    #[test]
    fn invalid_graph_and_replacement_do_not_mutate_existing_context() {
        let model = Model::new(checkpoint()).unwrap();
        let old = populated(&model, [0, 1, 2, 3]);
        let seal = old.sha256().unwrap();
        for relation in [
            edge(0, 0, EdgeKind::Parent),
            edge(0, 99, EdgeKind::Parent),
            edge(0, 1, EdgeKind::Parent),
            edge(2, 1, EdgeKind::Parent),
            edge(1, 0, EdgeKind::Parent),
            edge(0, 3, EdgeKind::Supersedes),
        ] {
            assert!(model.with_edge(&old, relation).is_err(), "{relation:?}");
            assert_eq!(old.sha256().unwrap(), seal);
        }
        let wrong_first = model
            .encode_line(&Position::startpos(), &moves("g1f3"), LineKind::Repair)
            .unwrap();
        assert!(model.with_line(&old, 1, wrong_first).is_err());
        let mut other_anchor = Position::startpos();
        other_anchor.make_move("e2e4".parse().unwrap()).unwrap();
        let wrong_anchor = model
            .encode_line(&other_anchor, &moves("e7e5"), LineKind::Repair)
            .unwrap();
        assert!(model.with_line(&old, 1, wrong_anchor).is_err());
        assert_eq!(old.sha256().unwrap(), seal);
    }

    #[test]
    fn original_bundle_move_is_required_and_distinct_from_rebased_suffix() {
        let model = Model::new(checkpoint()).unwrap();
        let root = Position::startpos();
        let empty = model.encode_line(&root, &[], LineKind::Proposal).unwrap();
        let context = model
            .with_line(&model.empty_context(), 0, Arc::clone(&empty))
            .unwrap();
        let context = model.with_line(&context, 1, empty).unwrap();
        assert!(
            model
                .with_edge(&context, edge(0, 1, EdgeKind::Parent))
                .is_err()
        );
        let original_move = "e2e4".parse().unwrap();
        let original = Some(original_move);
        let mut child = Position::startpos();
        child.make_move(original_move).unwrap();
        let proposal = model
            .encode_review_line(&child, &[], LineKind::Proposal, original)
            .unwrap();
        let counter = model
            .encode_review_line(&child, &moves("e7e5"), LineKind::Counterexample, original)
            .unwrap();
        let context = model
            .with_line(&model.empty_context(), 0, proposal)
            .unwrap();
        let context = model.with_line(&context, 1, counter).unwrap();
        let context = model
            .with_edge(&context, edge(0, 1, EdgeKind::Parent))
            .unwrap();
        assert_eq!(context.edge_count(), 1);
        let lines = context.reference_lines();
        assert!(lines[0].moves.is_empty());
        assert!(lines[0].original_first_move.is_some());
    }

    #[test]
    fn v2_exact_operand_bounds_are_checked_before_numeric_publication() {
        let maximum = exact_operand(MAX_OPERAND).unwrap();
        assert_eq!(maximum.to_f32().to_bits(), MAX_OPERAND.to_bits());
        assert!(exact_operand(MAX_OPERAND * 2.0).is_err());
        assert!(exact_operand(f32::MAX).is_err());
        assert!(exact_operand(f32::INFINITY).is_err());
        assert_eq!(
            maximum
                .checked_add(maximum.negated())
                .unwrap()
                .to_f32()
                .to_bits(),
            0
        );
        let model = Model::new(checkpoint()).unwrap();
        let context = populated(&model, [0, 1, 2, 3]);
        let oversized = vec![0; 129];
        assert!(model.without_bundle(&context, &oversized).is_err());
        assert_eq!(context.line_count(), 4);
    }
}
