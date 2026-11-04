//! CPU-only context-aware, efficiently updatable FP32 reference.
//!
//! B is an owned two-perspective board accumulator, C is an immutable shared
//! line set, and L is an owned, Rules-checked local suffix. No search, runtime,
//! Python, GPU, or global mutable context is required to use this module.
//! The dimensions are required experimental configuration, not adopted defaults.
//! Encoders, projections and heads use separate FP32 multiply/add operations.
//! B sums FP32 feature operands in signed192; C sums the already-FP32 line
//! projections in checked signed256. Both round once, ties to even. This avoids
//! irreversible FP32 subtraction, and is not integer quantization of weights.
//! Undo restores the original bits; no clipping precedes accumulator sums.

use super::{
    CPU_ACCUMULATOR_WIDTH, CPU_FEATURE_COUNT, CpuAccumulator, CpuAccumulatorUndo, CpuTrainingState,
    CpuValueCacheContext, CpuValueError, CpuValueEvaluator, CpuValueIdentity, Signed192,
    metadata_features, piece_feature, validate_training_declaration,
};
use rz_position::{
    BoardMove, Color, PieceKind, Position, PositionIdentity, PositionLimits, RuleMoveDelta, Square,
    UndoToken,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::sync::Arc;

#[path = "nnuec_rules.rs"]
pub mod rules;
#[path = "nnuec_v2.rs"]
pub mod v2;

pub const NNUEC_SCHEMA: &str = "rz-nnuec-fp32-v1";
pub const NNUEC_SEMANTICS: &str = "board800-exactb192-causal-line-exactc256-fp32-v1";
pub const NNUEC_INPUT_SCHEMA: &str = "rz-nnuec-fp32-input/1";
pub const NNUEC_OUTPUT_SCHEMA: &str = "rz-nnuec-fp32-output/1";
pub const NNUEC_MAX_BYTES: usize = 32 * 1024 * 1024;
pub const NNUEC_OUTPUTS: usize = 11;
pub const NNUEC_ABSOLUTE_TOLERANCE: f32 = 1e-4;
pub const NNUEC_RELATIVE_TOLERANCE: f32 = 1e-3;
const PARAMETER_MAGNITUDE: f32 = 1000.0;
// Even the conservative bound of 800 anchor terms gives |anchor| < 2^20,
// |token| < 806001, |causal token| < 4.031e9, |line sum| < 1.032e12 and a
// 256-column FP32 projection < 2.65e17 (rounding slack included). We separately
// enforce the looser exact operand limit 2^60, rather than admitting arbitrary
// finite FP32 values. Bias + 128 contributions and one replacement transient
// have |sum| < 2^68; scaled by 2^149 they need fewer than 217 magnitude bits.
// B's bias + 800 features + at most 22 transition terms have |sum| < 2^20,
// or fewer than 169 scaled magnitude bits, within the existing signed192.
const MAX_CONTEXT_OPERAND: f32 = 1_152_921_504_606_846_976.0; // 2^60

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NnuecError {
    InvalidConfig(&'static str),
    InvalidCheckpoint(&'static str),
    Capacity(&'static str),
    StateMismatch,
    NonFinite,
    AccumulatorOverflow,
    Rules(String),
    Io(String),
    Json(String),
}

impl std::fmt::Display for NnuecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(s) => write!(f, "NNUEC config: {s}"),
            Self::InvalidCheckpoint(s) => write!(f, "NNUEC checkpoint: {s}"),
            Self::Capacity(s) => write!(f, "NNUEC capacity: {s}"),
            Self::StateMismatch => f.write_str("NNUEC owner/model/Rules state mismatch"),
            Self::NonFinite => f.write_str("NNUEC FP32 operation produced a non-finite result"),
            Self::AccumulatorOverflow => f.write_str("NNUEC exact accumulator range exceeded"),
            Self::Rules(s) => write!(f, "NNUEC Rules transition: {s}"),
            Self::Io(s) => write!(f, "NNUEC checkpoint read: {s}"),
            Self::Json(s) => write!(f, "NNUEC JSON: {s}"),
        }
    }
}
impl std::error::Error for NnuecError {}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NnuecConfig {
    pub board_width: usize,
    pub line_width: usize,
    pub context_width: usize,
    pub local_width: usize,
    pub hidden_width: usize,
    pub causal_window: usize,
    pub max_lines: usize,
    pub max_plies: usize,
}

impl NnuecConfig {
    pub fn validate(&self) -> Result<(), NnuecError> {
        if [
            self.board_width,
            self.line_width,
            self.context_width,
            self.local_width,
        ]
        .into_iter()
        .any(|n| !(1..=256).contains(&n))
            || !(1..=64).contains(&self.hidden_width)
            || ![3, 5].contains(&self.causal_window)
            || !(1..=128).contains(&self.max_lines)
            || !(1..=256).contains(&self.max_plies)
        {
            return Err(NnuecError::InvalidConfig(
                "unsupported dimension or capacity",
            ));
        }
        Ok(())
    }

    pub fn input_width(&self) -> usize {
        2 * self.board_width + self.context_width + self.local_width
    }

    fn dimensions(&self) -> [usize; 8] {
        [
            self.board_width,
            self.line_width,
            self.context_width,
            self.local_width,
            self.hidden_width,
            self.causal_window,
            self.max_lines,
            self.max_plies,
        ]
    }
}

/// Flat row-major FP32 parameters. Embedding tables have one row per feature;
/// dense projections have one row per output. No implicit initializer exists.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NnuecParameters {
    pub board_weights: Vec<f32>,
    pub board_bias: Vec<f32>,
    pub anchor_weights: Vec<f32>,
    pub anchor_bias: Vec<f32>,
    pub from_weights: Vec<f32>,
    pub to_weights: Vec<f32>,
    pub promotion_weights: Vec<f32>,
    pub ply_weights: Vec<f32>,
    pub token_bias: Vec<f32>,
    /// Diagonal causal convolution, lag 0 is the current token.
    pub causal_weights: Vec<f32>,
    pub causal_bias: Vec<f32>,
    pub context_projection: Vec<f32>,
    pub context_bias: Vec<f32>,
    pub local_projection: Vec<f32>,
    pub local_bias: Vec<f32>,
    pub hidden_weights: Vec<f32>,
    pub hidden_bias: Vec<f32>,
    /// Rows: value, W/D/L, state candidate hint, retain/drop/revisit, P/C/R.
    pub output_weights: Vec<f32>,
    pub output_bias: Vec<f32>,
}

impl NnuecParameters {
    fn arrays(&self) -> [&Vec<f32>; 19] {
        [
            &self.board_weights,
            &self.board_bias,
            &self.anchor_weights,
            &self.anchor_bias,
            &self.from_weights,
            &self.to_weights,
            &self.promotion_weights,
            &self.ply_weights,
            &self.token_bias,
            &self.causal_weights,
            &self.causal_bias,
            &self.context_projection,
            &self.context_bias,
            &self.local_projection,
            &self.local_bias,
            &self.hidden_weights,
            &self.hidden_bias,
            &self.output_weights,
            &self.output_bias,
        ]
    }

    fn validate(&self, c: &NnuecConfig) -> Result<(), NnuecError> {
        let e = c.line_width;
        let expected = [
            CPU_FEATURE_COUNT * c.board_width,
            c.board_width,
            CPU_FEATURE_COUNT * e,
            e,
            64 * e,
            64 * e,
            5 * e,
            c.max_plies * e,
            e,
            c.causal_window * e,
            e,
            c.context_width * e,
            c.context_width,
            c.local_width * e,
            c.local_width,
            c.hidden_width * c.input_width(),
            c.hidden_width,
            NNUEC_OUTPUTS * c.hidden_width,
            NNUEC_OUTPUTS,
        ];
        let mut allocated = 0usize;
        for (values, expected) in self.arrays().into_iter().zip(expected) {
            if values.len() != expected || values.capacity() > expected.saturating_mul(2).max(4) {
                return Err(NnuecError::InvalidCheckpoint(
                    "parameter shape/allocation mismatch",
                ));
            }
            allocated = allocated
                .checked_add(values.capacity() * 4)
                .ok_or(NnuecError::Capacity("parameter bytes"))?;
            if values
                .iter()
                .any(|v| !v.is_finite() || v.abs() > PARAMETER_MAGNITUDE)
            {
                return Err(NnuecError::InvalidCheckpoint(
                    "non-finite/excessive parameter",
                ));
            }
        }
        if allocated > NNUEC_MAX_BYTES {
            return Err(NnuecError::Capacity("parameter bytes"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NnuecCheckpoint {
    pub schema: String,
    pub semantics: String,
    pub config: NnuecConfig,
    pub training: CpuTrainingState,
    pub parameters: NnuecParameters,
    pub output_scale: f32,
}

impl NnuecCheckpoint {
    pub fn validate(&self) -> Result<(), NnuecError> {
        self.config.validate()?;
        if self.schema != NNUEC_SCHEMA
            || self.semantics != NNUEC_SEMANTICS
            || self.schema.capacity() > 256
            || self.semantics.capacity() > 256
            || matches!(self.training, CpuTrainingState::Bootstrap)
            || validate_training_declaration(&self.training).is_err()
            || !self.output_scale.is_finite()
            || self.output_scale <= 0.0
            || self.output_scale > 10_000.0
        {
            return Err(NnuecError::InvalidCheckpoint(
                "schema/semantics/training/scale",
            ));
        }
        self.parameters.validate(&self.config)
    }

    /// Binary domain, required dimensions (u32 LE), ordered FP32 parameter bits,
    /// then scale bits. JSON formatting and training declarations are excluded.
    pub fn weights_sha256(&self) -> Result<String, NnuecError> {
        self.validate()?;
        let mut hash = Sha256::new();
        hash.update(NNUEC_SCHEMA.as_bytes());
        hash.update([0]);
        hash.update(NNUEC_SEMANTICS.as_bytes());
        hash.update([0]);
        for n in self.config.dimensions() {
            hash.update((n as u32).to_le_bytes());
        }
        for values in self.parameters.arrays() {
            for v in values {
                hash.update(v.to_bits().to_le_bytes());
            }
        }
        hash.update(self.output_scale.to_bits().to_le_bytes());
        Ok(format!("{:x}", hash.finalize()))
    }
}

pub struct NnuecModel {
    checkpoint: Arc<NnuecCheckpoint>,
    identity: Arc<CpuValueIdentity>,
}

impl NnuecModel {
    pub fn new(checkpoint: NnuecCheckpoint) -> Result<Self, NnuecError> {
        let weights = checkpoint.weights_sha256()?;
        let identity = Arc::new(CpuValueIdentity {
            semantics: NNUEC_SEMANTICS.to_owned(),
            weights_sha256: Some(weights),
            training: checkpoint.training.clone(),
        });
        Ok(Self {
            checkpoint: Arc::new(checkpoint),
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
            return Err(NnuecError::Capacity("checkpoint bytes"));
        }
        let checkpoint =
            serde_json::from_slice(&bytes).map_err(|e| NnuecError::Json(e.to_string()))?;
        Self::new(checkpoint)
    }

    pub fn identity(&self) -> &CpuValueIdentity {
        &self.identity
    }
    pub fn checkpoint(&self) -> &NnuecCheckpoint {
        &self.checkpoint
    }

    pub fn board(&self, position: &Position) -> Result<NnuecBoard, NnuecError> {
        let p = &self.checkpoint.parameters;
        let bias = p
            .board_bias
            .iter()
            .map(|v| exact_board_operand(*v))
            .collect::<Result<Vec<_>, _>>()?;
        let mut exact = [bias.clone(), bias];
        for side in [Color::White, Color::Black] {
            for feature in board_features(position, side) {
                add_exact_board_row(&mut exact[side as usize], &p.board_weights, feature, false)?;
            }
        }
        let values = round_board(&exact);
        Ok(NnuecBoard {
            values,
            exact,
            position: position.position_identity(),
            model: Arc::clone(&self.identity),
            stamp: Arc::new(()),
        })
    }

    pub fn apply_board_delta(
        &self,
        board: &mut NnuecBoard,
        delta: &RuleMoveDelta,
    ) -> Result<NnuecBoardUndo, NnuecError> {
        if board.model != self.identity || board.position != delta.before.position_identity() {
            return Err(NnuecError::StateMismatch);
        }
        if delta.removals.len() > 4 || delta.additions.len() > 4 {
            return Err(NnuecError::Capacity("Rules piece delta"));
        }
        let mut next = board.exact.clone();
        for side in [Color::White, Color::Black] {
            let out = &mut next[side as usize];
            for r in &delta.removals {
                add_exact_board_row(
                    out,
                    &self.checkpoint.parameters.board_weights,
                    piece_feature(r.piece, r.square, side),
                    true,
                )?;
            }
            for a in &delta.additions {
                add_exact_board_row(
                    out,
                    &self.checkpoint.parameters.board_weights,
                    piece_feature(a.piece, a.square, side),
                    false,
                )?;
            }
            let before = metadata_features(&delta.before, side);
            let after = metadata_features(&delta.after, side);
            for f in before.as_slice() {
                if !after.as_slice().contains(f) {
                    add_exact_board_row(out, &self.checkpoint.parameters.board_weights, *f, true)?;
                }
            }
            for f in after.as_slice() {
                if !before.as_slice().contains(f) {
                    add_exact_board_row(out, &self.checkpoint.parameters.board_weights, *f, false)?;
                }
            }
        }
        let after = delta.after.position_identity();
        let stamp = Arc::new(());
        let rounded = round_board(&next);
        let undo = NnuecBoardUndo {
            values: std::mem::replace(&mut board.values, rounded),
            exact: std::mem::replace(&mut board.exact, next),
            before: board.position.clone(),
            after: after.clone(),
            model: Arc::clone(&self.identity),
            before_stamp: Arc::clone(&board.stamp),
            after_stamp: Arc::clone(&stamp),
        };
        board.position = after;
        board.stamp = stamp;
        Ok(undo)
    }

    pub fn restore_board(
        &self,
        board: &mut NnuecBoard,
        undo: NnuecBoardUndo,
        position: &Position,
    ) -> Result<(), NnuecError> {
        if board.model != self.identity
            || undo.model != self.identity
            || board.position != undo.after
            || position.position_identity() != undo.before
            || !Arc::ptr_eq(&board.stamp, &undo.after_stamp)
        {
            return Err(NnuecError::StateMismatch);
        }
        board.values = undo.values;
        board.exact = undo.exact;
        board.position = undo.before;
        board.stamp = undo.before_stamp;
        Ok(())
    }

    pub fn local(&self, anchor: &Position) -> Result<NnuecLocal, NnuecError> {
        let side = anchor.side_to_move();
        let features = board_features(anchor, side);
        let mut anchor_values = self.checkpoint.parameters.anchor_bias.clone();
        for feature in &features {
            add_row(
                &mut anchor_values,
                &self.checkpoint.parameters.anchor_weights,
                *feature,
                1.0,
            )?;
        }
        let embedding = vec![0.0; self.checkpoint.config.line_width];
        let projected = project(
            &embedding,
            &self.checkpoint.parameters.local_projection,
            &self.checkpoint.parameters.local_bias,
        )?;
        Ok(NnuecLocal {
            work: anchor.clone(),
            anchor: anchor.position_identity(),
            perspective: side,
            anchor_features: features,
            anchor_values,
            tokens: Vec::new(),
            moves: Vec::new(),
            embedding,
            projected,
            model: Arc::clone(&self.identity),
            stamp: Arc::new(()),
        })
    }

    /// Appending validates a genuine Rules transition. Only a bounded causal
    /// window and local accumulator change; C and the global store are untouched.
    pub fn append_local(
        &self,
        local: &mut NnuecLocal,
        mv: BoardMove,
    ) -> Result<NnuecLocalUndo, NnuecError> {
        let prepared = self.prepare_local_append(local, mv)?;
        reserve_local_append(local)?;
        let rules = local
            .work
            .make_move(mv)
            .map_err(|e| NnuecError::Rules(e.to_string()))?;
        let stamp = Arc::new(());
        let undo = NnuecLocalUndo {
            rules,
            previous_embedding: std::mem::replace(&mut local.embedding, prepared.embedding),
            previous_projected: std::mem::replace(&mut local.projected, prepared.projected),
            model: Arc::clone(&self.identity),
            before_stamp: Arc::clone(&local.stamp),
            after_stamp: Arc::clone(&stamp),
            expected_length: local.moves.len() + 1,
        };
        local.tokens.push(prepared.token);
        local.moves.push(mv);
        local.stamp = stamp;
        Ok(undo)
    }

    fn prepare_local_append(
        &self,
        local: &NnuecLocal,
        mv: BoardMove,
    ) -> Result<PreparedLocalAppend, NnuecError> {
        if local.model != self.identity {
            return Err(NnuecError::StateMismatch);
        }
        let c = &self.checkpoint.config;
        if local.moves.len() >= c.max_plies {
            return Err(NnuecError::Capacity("local plies"));
        }
        let token = self.token(
            &local.anchor_values,
            local.perspective,
            mv,
            local.moves.len(),
        )?;
        let mut embedding = local.embedding.clone();
        for (j, out) in embedding.iter_mut().enumerate() {
            let mut value = self.checkpoint.parameters.causal_bias[j];
            for lag in 0..c.causal_window {
                let input = if lag == 0 {
                    token[j]
                } else if lag <= local.tokens.len() {
                    local.tokens[local.tokens.len() - lag][j]
                } else {
                    continue;
                };
                value = add(
                    value,
                    mul(
                        input,
                        self.checkpoint.parameters.causal_weights[lag * c.line_width + j],
                    )?,
                )?;
            }
            *out = add(*out, value.max(0.0))?;
        }
        let projected = project(
            &embedding,
            &self.checkpoint.parameters.local_projection,
            &self.checkpoint.parameters.local_bias,
        )?;
        Ok(PreparedLocalAppend {
            token,
            embedding,
            projected,
        })
    }

    pub fn undo_local(
        &self,
        local: &mut NnuecLocal,
        undo: NnuecLocalUndo,
    ) -> Result<(), NnuecError> {
        if local.model != self.identity
            || undo.model != self.identity
            || local.moves.len() != undo.expected_length
            || !Arc::ptr_eq(&local.stamp, &undo.after_stamp)
        {
            return Err(NnuecError::StateMismatch);
        }
        local
            .work
            .unmake(undo.rules)
            .map_err(|e| NnuecError::Rules(e.to_string()))?;
        local.embedding = undo.previous_embedding;
        local.projected = undo.previous_projected;
        local.tokens.pop();
        local.moves.pop();
        local.stamp = undo.before_stamp;
        Ok(())
    }

    /// Re-anchoring or deleting a middle move requires a genuine rebuild.
    pub fn encode_line(
        &self,
        anchor: &Position,
        moves: &[BoardMove],
    ) -> Result<Arc<NnuecLine>, NnuecError> {
        if moves.len() > self.checkpoint.config.max_plies {
            return Err(NnuecError::Capacity("line plies"));
        }
        let mut local = self.local(anchor)?;
        for mv in moves {
            self.append_local(&mut local, *mv)?;
        }
        let contribution = project(
            &local.embedding,
            &self.checkpoint.parameters.context_projection,
            &vec![0.0; self.checkpoint.config.context_width],
        )?;
        for value in &contribution {
            Exact256::from_f32(*value)?;
        }
        Ok(Arc::new(NnuecLine {
            model: Arc::clone(&self.identity),
            anchor: local.anchor,
            input: ReferenceLine {
                anchor_features: local.anchor_features,
                perspective: color_name(local.perspective).into(),
                moves: local
                    .moves
                    .iter()
                    .copied()
                    .map(ReferenceMove::from_board)
                    .collect(),
            },
            embedding: local.embedding,
            contribution,
        }))
    }

    fn token(
        &self,
        anchor: &[f32],
        side: Color,
        mv: BoardMove,
        ply: usize,
    ) -> Result<Vec<f32>, NnuecError> {
        let promotion = promotion_index(mv)?;
        let p = &self.checkpoint.parameters;
        let width = self.checkpoint.config.line_width;
        let from = oriented_square(mv.from, side);
        let to = oriented_square(mv.to, side);
        let mut token = p.token_bias.clone();
        for (j, out) in token.iter_mut().enumerate() {
            *out = add(*out, anchor[j])?;
            *out = add(*out, p.from_weights[from * width + j])?;
            *out = add(*out, p.to_weights[to * width + j])?;
            *out = add(*out, p.promotion_weights[promotion * width + j])?;
            *out = add(*out, p.ply_weights[ply * width + j])?;
        }
        Ok(token)
    }

    pub fn empty_context(&self) -> Arc<NnuecContext> {
        let exact: Vec<Exact256> = self
            .checkpoint
            .parameters
            .context_bias
            .iter()
            .map(|v| Exact256::from_f32(*v).expect("admitted bias within exact context range"))
            .collect();
        let values = exact.iter().map(|value| value.to_f32()).collect();
        Arc::new(NnuecContext {
            model: Arc::clone(&self.identity),
            lines: BTreeMap::new(),
            values,
            exact,
            authority: Arc::new(()),
        })
    }

    /// IDs are ownership keys only. They never enter an embedding or a head.
    /// Returning a new Arc keeps all existing workers' frozen C unchanged.
    pub fn with_line(
        &self,
        context: &Arc<NnuecContext>,
        id: u64,
        line: Arc<NnuecLine>,
    ) -> Result<Arc<NnuecContext>, NnuecError> {
        if context.model != self.identity || line.model != self.identity {
            return Err(NnuecError::StateMismatch);
        }
        if !context.lines.contains_key(&id)
            && context.lines.len() >= self.checkpoint.config.max_lines
        {
            return Err(NnuecError::Capacity("context lines"));
        }
        let mut exact = context.exact.clone();
        if let Some(old) = context.lines.get(&id) {
            Self::add_line(&mut exact, old, true)?;
        }
        Self::add_line(&mut exact, &line, false)?;
        let values = exact.iter().map(|v| v.to_f32()).collect();
        let mut lines = context.lines.clone();
        lines.insert(id, line);
        Ok(Arc::new(NnuecContext {
            model: Arc::clone(&self.identity),
            lines,
            values,
            exact,
            authority: Arc::new(()),
        }))
    }

    pub fn without_line(
        &self,
        context: &Arc<NnuecContext>,
        id: u64,
    ) -> Result<Arc<NnuecContext>, NnuecError> {
        if context.model != self.identity {
            return Err(NnuecError::StateMismatch);
        }
        let old = context.lines.get(&id).ok_or(NnuecError::StateMismatch)?;
        let mut exact = context.exact.clone();
        Self::add_line(&mut exact, old, true)?;
        let values = exact.iter().map(|v| v.to_f32()).collect();
        let mut lines = context.lines.clone();
        lines.remove(&id);
        Ok(Arc::new(NnuecContext {
            model: Arc::clone(&self.identity),
            lines,
            values,
            exact,
            authority: Arc::new(()),
        }))
    }

    pub fn rebuild_context(&self, context: &NnuecContext) -> Result<Arc<NnuecContext>, NnuecError> {
        if context.model != self.identity {
            return Err(NnuecError::StateMismatch);
        }
        let mut exact = self.empty_context().exact.clone();
        for line in context.lines.values() {
            Self::add_line(&mut exact, line, false)?;
        }
        let values = exact.iter().map(|v| v.to_f32()).collect();
        Ok(Arc::new(NnuecContext {
            model: Arc::clone(&self.identity),
            lines: context.lines.clone(),
            values,
            exact,
            authority: Arc::new(()),
        }))
    }

    fn add_line(values: &mut [Exact256], line: &NnuecLine, remove: bool) -> Result<(), NnuecError> {
        for (value, contribution) in values.iter_mut().zip(&line.contribution) {
            let operand = Exact256::from_f32(*contribution)?;
            *value = value.checked_add(if remove { operand.negated() } else { operand })?;
        }
        Ok(())
    }

    pub fn forward(
        &self,
        board: &NnuecBoard,
        context: &NnuecContext,
        local: &NnuecLocal,
        position: &Position,
    ) -> Result<NnuecOutput, NnuecError> {
        if context.model != self.identity {
            return Err(NnuecError::StateMismatch);
        }
        self.forward_with_context_values(board, &context.values, local, position)
    }

    // V2 reuses this exact B/L/head operation order with its separately
    // admitted context and output identity. No checkpoint conversion occurs.
    fn forward_with_context_values(
        &self,
        board: &NnuecBoard,
        values: &[f32],
        local: &NnuecLocal,
        position: &Position,
    ) -> Result<NnuecOutput, NnuecError> {
        if board.model != self.identity
            || local.model != self.identity
            || board.position != position.position_identity()
            || local.work.position_identity() != board.position
            || values.len() != self.checkpoint.config.context_width
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(NnuecError::StateMismatch);
        }
        let side = position.side_to_move() as usize;
        let mut input = Vec::with_capacity(self.checkpoint.config.input_width());
        input.extend(
            board.values[side]
                .iter()
                .chain(&board.values[1 - side])
                .chain(values)
                .chain(&local.projected)
                .map(|v| v.clamp(0.0, 1.0)),
        );
        let mut hidden = project(
            &input,
            &self.checkpoint.parameters.hidden_weights,
            &self.checkpoint.parameters.hidden_bias,
        )?;
        for v in &mut hidden {
            *v = v.max(0.0);
        }
        let raw = project(
            &hidden,
            &self.checkpoint.parameters.output_weights,
            &self.checkpoint.parameters.output_bias,
        )?;
        let raw_outputs: [f32; NNUEC_OUTPUTS] =
            raw.try_into().map_err(|_| NnuecError::StateMismatch)?;
        let value = mul(raw_outputs[0], self.checkpoint.output_scale)?;
        let wdl_logits = [raw_outputs[1], raw_outputs[2], raw_outputs[3]];
        let max = wdl_logits.into_iter().fold(f32::NEG_INFINITY, f32::max);
        let mut wdl = wdl_logits.map(|v| (v - max).exp());
        let sum = add(add(wdl[0], wdl[1])?, wdl[2])?;
        for v in &mut wdl {
            *v /= sum;
            if !v.is_finite() {
                return Err(NnuecError::NonFinite);
            }
        }
        Ok(NnuecOutput {
            schema: NNUEC_OUTPUT_SCHEMA.into(),
            weights_sha256: self
                .identity
                .weights_sha256
                .clone()
                .expect("admitted weights"),
            board_accumulators: board.values.clone(),
            context_accumulator: values.to_vec(),
            local_accumulator: local.projected.clone(),
            hidden,
            raw_outputs,
            value,
            wdl_logits,
            wdl,
        })
    }

    pub fn reference_input(
        &self,
        context: &NnuecContext,
        local: &NnuecLocal,
        position: &Position,
    ) -> Result<NnuecReferenceInput, NnuecError> {
        if context.model != self.identity
            || local.model != self.identity
            || local.work.position_identity() != position.position_identity()
        {
            return Err(NnuecError::StateMismatch);
        }
        Ok(NnuecReferenceInput {
            schema: NNUEC_INPUT_SCHEMA.into(),
            board_features: [
                board_features(position, Color::White),
                board_features(position, Color::Black),
            ],
            side_to_move: color_name(position.side_to_move()).into(),
            context_lines: context.lines.values().map(|l| l.input.clone()).collect(),
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

/// Owned B. It is deliberately not Clone: undo authority belongs to this owner.
pub struct NnuecBoard {
    values: [Vec<f32>; 2],
    exact: [Vec<Signed192>; 2],
    position: PositionIdentity,
    model: Arc<CpuValueIdentity>,
    stamp: Arc<()>,
}
impl NnuecBoard {
    pub fn values(&self) -> &[Vec<f32>; 2] {
        &self.values
    }
}
pub struct NnuecBoardUndo {
    values: [Vec<f32>; 2],
    exact: [Vec<Signed192>; 2],
    before: PositionIdentity,
    after: PositionIdentity,
    model: Arc<CpuValueIdentity>,
    before_stamp: Arc<()>,
    after_stamp: Arc<()>,
}

pub struct NnuecLine {
    model: Arc<CpuValueIdentity>,
    anchor: PositionIdentity,
    input: ReferenceLine,
    embedding: Vec<f32>,
    contribution: Vec<f32>,
}
impl NnuecLine {
    pub fn embedding(&self) -> &[f32] {
        &self.embedding
    }
    pub fn anchor(&self) -> &PositionIdentity {
        &self.anchor
    }
}
pub struct NnuecContext {
    model: Arc<CpuValueIdentity>,
    lines: BTreeMap<u64, Arc<NnuecLine>>,
    values: Vec<f32>,
    exact: Vec<Exact256>,
    authority: Arc<()>,
}
impl NnuecContext {
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn len(&self) -> usize {
        self.lines.len()
    }
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
}
pub struct NnuecLocal {
    work: Position,
    anchor: PositionIdentity,
    perspective: Color,
    anchor_features: Vec<usize>,
    anchor_values: Vec<f32>,
    tokens: Vec<Vec<f32>>,
    moves: Vec<BoardMove>,
    embedding: Vec<f32>,
    projected: Vec<f32>,
    model: Arc<CpuValueIdentity>,
    stamp: Arc<()>,
}
impl NnuecLocal {
    pub fn values(&self) -> &[f32] {
        &self.projected
    }
    pub fn moves(&self) -> &[BoardMove] {
        &self.moves
    }
    pub fn position(&self) -> &Position {
        &self.work
    }
}
pub struct NnuecLocalUndo {
    rules: UndoToken,
    previous_embedding: Vec<f32>,
    previous_projected: Vec<f32>,
    model: Arc<CpuValueIdentity>,
    before_stamp: Arc<()>,
    after_stamp: Arc<()>,
    expected_length: usize,
}

struct PreparedLocalAppend {
    token: Vec<f32>,
    embedding: Vec<f32>,
    projected: Vec<f32>,
}

fn reserve_local_append(local: &mut NnuecLocal) -> Result<(), NnuecError> {
    local
        .tokens
        .try_reserve(1)
        .map_err(|_| NnuecError::Capacity("local token allocation"))?;
    local
        .moves
        .try_reserve(1)
        .map_err(|_| NnuecError::Capacity("local move allocation"))?;
    Ok(())
}

/// Explicit experimental CPU value profile. Existing engine constructors keep
/// their original evaluator. The declared training state remains in identity;
/// constructing this profile does not claim training, strength or Rules proof.
pub struct NnuecCpuValue {
    model: Arc<NnuecModel>,
    context: Arc<NnuecContext>,
}

impl NnuecCpuValue {
    pub fn new(model: Arc<NnuecModel>, context: Arc<NnuecContext>) -> Result<Self, NnuecError> {
        if context.model != model.identity {
            return Err(NnuecError::StateMismatch);
        }
        Ok(Self { model, context })
    }

    /// Read-only conditional value/utility observation. Utility rows have no
    /// authority to retain/drop/route work, regardless of checkpoint labels.
    pub fn observe(
        &self,
        accumulator: &CpuAccumulator,
        position: &Position,
    ) -> Result<NnuecOutput, CpuValueError> {
        self.cache_context(accumulator, position)?;
        let state = accumulator
            .nnuec
            .as_ref()
            .ok_or(CpuValueError::StateMismatch)?;
        self.model
            .forward(&state.board, &self.context, &state.local, position)
            .map_err(cpu_error)
    }
}

pub(crate) struct NnuecNativeAccumulator {
    board: NnuecBoard,
    local: NnuecLocal,
    context_authority: Arc<()>,
}
pub(crate) struct NnuecNativeUndo {
    board: NnuecBoard,
    previous_work: Position,
    previous_embedding: Vec<f32>,
    previous_projected: Vec<f32>,
    before_stamp: Arc<()>,
    after_stamp: Arc<()>,
    board_after_stamp: Arc<()>,
    expected_length: usize,
}

impl std::fmt::Debug for NnuecNativeAccumulator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NnuecNativeAccumulator")
            .field("model", &self.board.model)
            .field("local_plies", &self.local.moves.len())
            .finish_non_exhaustive()
    }
}

impl Clone for NnuecNativeAccumulator {
    fn clone(&self) -> Self {
        let mut board = copy_board(&self.board);
        board.stamp = Arc::new(());
        let local = &self.local;
        Self {
            board,
            context_authority: Arc::clone(&self.context_authority),
            local: NnuecLocal {
                work: local.work.clone(),
                anchor: local.anchor.clone(),
                perspective: local.perspective,
                anchor_features: local.anchor_features.clone(),
                anchor_values: local.anchor_values.clone(),
                tokens: local.tokens.clone(),
                moves: local.moves.clone(),
                embedding: local.embedding.clone(),
                projected: local.projected.clone(),
                model: Arc::clone(&local.model),
                stamp: Arc::new(()),
            },
        }
    }
}

impl NnuecNativeAccumulator {
    pub(crate) fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(board_backing_bytes(&self.board))
            .saturating_add(vector_backing_bytes(&self.local.anchor_features))
            .saturating_add(vector_backing_bytes(&self.local.anchor_values))
            .saturating_add(vector_backing_bytes(&self.local.tokens))
            .saturating_add(self.local.tokens.iter().fold(0usize, |sum, row| {
                sum.saturating_add(vector_backing_bytes(row))
            }))
            .saturating_add(vector_backing_bytes(&self.local.moves))
            .saturating_add(vector_backing_bytes(&self.local.embedding))
            .saturating_add(vector_backing_bytes(&self.local.projected))
            .saturating_add(96)
    }
}
impl NnuecNativeUndo {
    pub(crate) fn retained_bytes(&self) -> usize {
        // The saved Position shares the already charged Rules history chain.
        std::mem::size_of::<Self>()
            .saturating_add(board_backing_bytes(&self.board))
            .saturating_add(vector_backing_bytes(&self.previous_embedding))
            .saturating_add(vector_backing_bytes(&self.previous_projected))
            .saturating_add(96)
    }
}

impl CpuValueEvaluator for NnuecCpuValue {
    fn identity(&self) -> &CpuValueIdentity {
        self.model.identity()
    }
    fn provenance(&self) -> &'static str {
        NNUEC_SEMANTICS
    }
    fn initialize(&self, position: &Position) -> Result<CpuAccumulator, CpuValueError> {
        let board = self.model.board(position).map_err(cpu_error)?;
        let local = self.model.local(position).map_err(cpu_error)?;
        let values = padded_board_values(&board);
        Ok(CpuAccumulator {
            values,
            identity: position.position_identity(),
            namespace: Arc::clone(&self.model.identity),
            exact: None,
            nnuec: Some(Box::new(NnuecNativeAccumulator {
                board,
                local,
                context_authority: Arc::clone(&self.context.authority),
            })),
        })
    }
    fn cache_context(
        &self,
        accumulator: &CpuAccumulator,
        position: &Position,
    ) -> Result<CpuValueCacheContext, CpuValueError> {
        let state = accumulator
            .nnuec
            .as_ref()
            .ok_or(CpuValueError::StateMismatch)?;
        if accumulator.namespace != self.model.identity
            || accumulator.identity != position.position_identity()
            || state.board.position != accumulator.identity
            || state.local.work.position_identity() != accumulator.identity
            || state.board.model != self.model.identity
            || state.local.model != self.model.identity
            || !Arc::ptr_eq(&state.context_authority, &self.context.authority)
        {
            return Err(CpuValueError::StateMismatch);
        }
        Ok(CpuValueCacheContext::contextual(
            &self.context.authority,
            &state.local.anchor,
        ))
    }
    fn score(
        &self,
        accumulator: &CpuAccumulator,
        position: &Position,
    ) -> Result<i32, CpuValueError> {
        let output = self.observe(accumulator, position)?;
        Ok(output.frontier_score())
    }
    fn apply_delta(
        &self,
        accumulator: &mut CpuAccumulator,
        delta: &RuleMoveDelta,
    ) -> Result<CpuAccumulatorUndo, CpuValueError> {
        if accumulator.namespace != self.model.identity
            || accumulator.identity != delta.before.position_identity()
        {
            return Err(CpuValueError::StateMismatch);
        }
        let state = accumulator
            .nnuec
            .as_mut()
            .ok_or(CpuValueError::StateMismatch)?;
        if state.local.work.position_identity() != accumulator.identity
            || !Arc::ptr_eq(&state.context_authority, &self.context.authority)
        {
            return Err(CpuValueError::StateMismatch);
        }
        let prepared = self
            .model
            .prepare_local_append(&state.local, delta.mv)
            .map_err(cpu_error)?;
        // Consume the already checked Rules delta. Restore an immutable view;
        // do not replay legality, make_move or an entire prefix in the evaluator.
        let next_work = Position::from_snapshot(
            &delta.after,
            PositionLimits {
                max_history_positions: delta.after.known_history_len().max(1),
                ..PositionLimits::default()
            },
        )
        .map_err(|_| CpuValueError::StateMismatch)?;
        let mut next_board = copy_board(&state.board);
        self.model
            .apply_board_delta(&mut next_board, delta)
            .map_err(cpu_error)?;
        reserve_local_append(&mut state.local).map_err(cpu_error)?;
        let stamp = Arc::new(());
        let native = NnuecNativeUndo {
            board_after_stamp: Arc::clone(&next_board.stamp),
            board: std::mem::replace(&mut state.board, next_board),
            previous_work: std::mem::replace(&mut state.local.work, next_work),
            previous_embedding: std::mem::replace(&mut state.local.embedding, prepared.embedding),
            previous_projected: std::mem::replace(&mut state.local.projected, prepared.projected),
            before_stamp: Arc::clone(&state.local.stamp),
            after_stamp: Arc::clone(&stamp),
            expected_length: state.local.moves.len() + 1,
        };
        state.local.moves.push(delta.mv);
        state.local.tokens.push(prepared.token);
        state.local.stamp = stamp;
        let undo = CpuAccumulatorUndo {
            changes: None,
            before: accumulator.identity.clone(),
            after: delta.after.position_identity(),
            namespace: Arc::clone(&self.model.identity),
            nnuec: Some(Box::new(native)),
        };
        accumulator.identity = undo.after.clone();
        accumulator.values = padded_board_values(&state.board);
        Ok(undo)
    }
    fn restore(
        &self,
        accumulator: &mut CpuAccumulator,
        undo: CpuAccumulatorUndo,
        position: &Position,
    ) -> Result<(), CpuValueError> {
        if accumulator.namespace != self.model.identity
            || undo.namespace != self.model.identity
            || accumulator.identity != undo.after
            || position.position_identity() != undo.before
            || undo.changes.is_some()
        {
            return Err(CpuValueError::StateMismatch);
        }
        let native = undo.nnuec.ok_or(CpuValueError::StateMismatch)?;
        let state = accumulator
            .nnuec
            .as_mut()
            .ok_or(CpuValueError::StateMismatch)?;
        if !Arc::ptr_eq(&state.local.stamp, &native.after_stamp)
            || !Arc::ptr_eq(&state.board.stamp, &native.board_after_stamp)
            || state.local.moves.len() != native.expected_length
            || native.previous_work.position_identity() != undo.before
            || !Arc::ptr_eq(&state.context_authority, &self.context.authority)
        {
            return Err(CpuValueError::StateMismatch);
        }
        state.board = native.board;
        state.local.work = native.previous_work;
        state.local.embedding = native.previous_embedding;
        state.local.projected = native.previous_projected;
        state.local.stamp = native.before_stamp;
        state.local.moves.pop();
        state.local.tokens.pop();
        accumulator.values = padded_board_values(&state.board);
        accumulator.identity = undo.before;
        Ok(())
    }
}

fn cpu_error(error: NnuecError) -> CpuValueError {
    match error {
        NnuecError::StateMismatch | NnuecError::Rules(_) => CpuValueError::StateMismatch,
        NnuecError::NonFinite => CpuValueError::NonFiniteForward,
        NnuecError::AccumulatorOverflow => CpuValueError::AccumulatorOverflow,
        NnuecError::Capacity(_) => {
            CpuValueError::InvalidCheckpoint("NNUEC configured capacity exceeded")
        }
        NnuecError::InvalidConfig(_) | NnuecError::InvalidCheckpoint(_) => {
            CpuValueError::InvalidCheckpoint("NNUEC invalid model")
        }
        NnuecError::Io(s) => CpuValueError::Io(s),
        NnuecError::Json(s) => CpuValueError::Json(s),
    }
}
fn copy_board(board: &NnuecBoard) -> NnuecBoard {
    NnuecBoard {
        values: board.values.clone(),
        exact: board.exact.clone(),
        position: board.position.clone(),
        model: Arc::clone(&board.model),
        stamp: Arc::clone(&board.stamp),
    }
}
fn padded_board_values(board: &NnuecBoard) -> [[f32; CPU_ACCUMULATOR_WIDTH]; 2] {
    let mut values = [[0.0; CPU_ACCUMULATOR_WIDTH]; 2];
    for (out, source) in values.iter_mut().zip(&board.values) {
        out[..source.len()].copy_from_slice(source);
    }
    values
}
fn vector_backing_bytes<T>(values: &Vec<T>) -> usize {
    values
        .capacity()
        .saturating_mul(std::mem::size_of::<T>())
        .saturating_add(32)
}
fn board_backing_bytes(board: &NnuecBoard) -> usize {
    board
        .values
        .iter()
        .fold(0usize, |sum, v| sum.saturating_add(vector_backing_bytes(v)))
        .saturating_add(
            board
                .exact
                .iter()
                .fold(0usize, |sum, v| sum.saturating_add(vector_backing_bytes(v))),
        )
}

/// The value is a nonterminal learned/frontier number, never a Rules mate proof.
/// Candidate/context/route rows are independent uncalibrated hints. Row 4 is a
/// state-level candidate hint; per-move action scoring is a separate consumer.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NnuecOutput {
    pub schema: String,
    pub weights_sha256: String,
    pub board_accumulators: [Vec<f32>; 2],
    pub context_accumulator: Vec<f32>,
    pub local_accumulator: Vec<f32>,
    pub hidden: Vec<f32>,
    pub raw_outputs: [f32; NNUEC_OUTPUTS],
    pub value: f32,
    pub wdl_logits: [f32; 3],
    pub wdl: [f32; 3],
}
impl NnuecOutput {
    pub fn frontier_score(&self) -> i32 {
        self.value.clamp(-20_000.0, 20_000.0).round() as i32
    }
    pub fn context_utility(&self) -> [f32; 3] {
        [
            self.raw_outputs[5],
            self.raw_outputs[6],
            self.raw_outputs[7],
        ]
    }
    pub fn receiver_utility(&self) -> [f32; 3] {
        [
            self.raw_outputs[8],
            self.raw_outputs[9],
            self.raw_outputs[10],
        ]
    }
}

/// Independent reference wire contains only model-consumed features. Exact
/// Rules/history identities remain in the Rust owner and are not numeric IDs.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NnuecReferenceInput {
    pub schema: String,
    pub board_features: [Vec<usize>; 2],
    pub side_to_move: String,
    pub context_lines: Vec<ReferenceLine>,
    pub local_line: ReferenceLine,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceLine {
    pub anchor_features: Vec<usize>,
    pub perspective: String,
    pub moves: Vec<ReferenceMove>,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceMove {
    pub from_square: u8,
    pub to_square: u8,
    pub promotion: u8,
}
impl ReferenceMove {
    fn from_board(mv: BoardMove) -> Self {
        Self {
            from_square: mv.from.index(),
            to_square: mv.to.index(),
            promotion: promotion_index(mv).expect("Rules move") as u8,
        }
    }
}

fn board_features(position: &Position, side: Color) -> Vec<usize> {
    let snapshot = position.snapshot();
    let mut out = Vec::with_capacity(40);
    for index in 0..64 {
        let square = Square::new(index).expect("bounded square");
        if let Some(piece) = snapshot.piece_at(square) {
            out.push(piece_feature(piece, square, side));
        }
    }
    out.extend(metadata_features(&snapshot, side));
    out
}
fn color_name(side: Color) -> &'static str {
    if side == Color::White {
        "white"
    } else {
        "black"
    }
}
fn oriented_square(square: Square, side: Color) -> usize {
    usize::from(square.index() ^ if side == Color::Black { 56 } else { 0 })
}
fn promotion_index(mv: BoardMove) -> Result<usize, NnuecError> {
    match mv.promotion {
        None => Ok(0),
        Some(PieceKind::Queen) => Ok(1),
        Some(PieceKind::Rook) => Ok(2),
        Some(PieceKind::Bishop) => Ok(3),
        Some(PieceKind::Knight) => Ok(4),
        _ => Err(NnuecError::InvalidConfig("invalid promotion")),
    }
}
fn add(a: f32, b: f32) -> Result<f32, NnuecError> {
    let v = a + b;
    if v.is_finite() {
        Ok(v)
    } else {
        Err(NnuecError::NonFinite)
    }
}
fn mul(a: f32, b: f32) -> Result<f32, NnuecError> {
    let v = a * b;
    if v.is_finite() {
        Ok(v)
    } else {
        Err(NnuecError::NonFinite)
    }
}
fn add_row(out: &mut [f32], weights: &[f32], row: usize, sign: f32) -> Result<(), NnuecError> {
    let width = out.len();
    let weights = weights
        .get(row * width..(row + 1) * width)
        .ok_or(NnuecError::StateMismatch)?;
    for (out, weight) in out.iter_mut().zip(weights) {
        *out = add(*out, mul(sign, *weight)?)?;
    }
    Ok(())
}
fn project(input: &[f32], weights: &[f32], bias: &[f32]) -> Result<Vec<f32>, NnuecError> {
    if weights.len() != input.len() * bias.len() {
        return Err(NnuecError::StateMismatch);
    }
    let mut out = bias.to_vec();
    for (row, out) in out.iter_mut().enumerate() {
        for (input, weight) in input
            .iter()
            .zip(&weights[row * input.len()..(row + 1) * input.len()])
        {
            *out = add(*out, mul(*input, *weight)?)?;
        }
    }
    Ok(out)
}

fn exact_board_operand(value: f32) -> Result<Signed192, NnuecError> {
    Signed192::from_f32(value).map_err(|_| NnuecError::AccumulatorOverflow)
}

fn add_exact_board_row(
    out: &mut [Signed192],
    weights: &[f32],
    row: usize,
    remove: bool,
) -> Result<(), NnuecError> {
    let width = out.len();
    let weights = weights
        .get(row * width..(row + 1) * width)
        .ok_or(NnuecError::StateMismatch)?;
    for (out, weight) in out.iter_mut().zip(weights) {
        let operand = exact_board_operand(*weight)?;
        *out = out
            .checked_add(if remove { operand.negated() } else { operand })
            .map_err(|_| NnuecError::AccumulatorOverflow)?;
    }
    Ok(())
}

fn round_board(exact: &[Vec<Signed192>; 2]) -> [Vec<f32>; 2] {
    std::array::from_fn(|side| exact[side].iter().map(|value| value.to_f32()).collect())
}

/// Exact sum of bounded FP32 context operands expressed in 2^-149 units.
/// This is deliberately local to C, not a quantized network or an accumulator
/// for arbitrary finite FP32 inputs. Its operand and signed-overflow checks are
/// retained even though admitted model bounds leave more than 38 spare bits.
#[derive(Clone, Copy, Default)]
struct Exact256([u64; 4]);

impl Exact256 {
    fn from_f32(value: f32) -> Result<Self, NnuecError> {
        Self::from_f32_bounded(value, MAX_CONTEXT_OPERAND)
    }

    fn from_f32_bounded(value: f32, maximum: f32) -> Result<Self, NnuecError> {
        if !maximum.is_finite() || maximum <= 0.0 || !value.is_finite() || value.abs() > maximum {
            return Err(NnuecError::AccumulatorOverflow);
        }
        let bits = value.to_bits();
        let exponent = ((bits >> 23) & 255) as usize;
        let fraction = bits & 0x7f_ffff;
        let mantissa = if exponent == 0 {
            fraction
        } else {
            fraction | 0x80_0000
        };
        let shift = exponent.saturating_sub(1);
        let word = shift / 64;
        let offset = shift % 64;
        if word >= 4 || shift + 24 >= 256 {
            return Err(NnuecError::AccumulatorOverflow);
        }
        let mut words = [0u64; 4];
        words[word] = u64::from(mantissa) << offset;
        if offset > 40 {
            words[word + 1] = u64::from(mantissa) >> (64 - offset);
        }
        let positive = Self(words);
        Ok(if bits >> 31 != 0 {
            positive.negated()
        } else {
            positive
        })
    }

    fn negative(self) -> bool {
        self.0[3] >> 63 != 0
    }

    fn negated(self) -> Self {
        let mut out = self.0.map(|word| !word);
        let mut carry = true;
        for word in &mut out {
            let (value, overflow) = word.overflowing_add(u64::from(carry));
            *word = value;
            carry = overflow;
        }
        Self(out)
    }

    fn checked_add(self, other: Self) -> Result<Self, NnuecError> {
        let mut words = [0u64; 4];
        let mut carry = false;
        for (index, word) in words.iter_mut().enumerate() {
            let (value, first) = self.0[index].overflowing_add(other.0[index]);
            let (value, second) = value.overflowing_add(u64::from(carry));
            *word = value;
            carry = first || second;
        }
        let out = Self(words);
        if self.negative() == other.negative() && self.negative() != out.negative() {
            return Err(NnuecError::AccumulatorOverflow);
        }
        Ok(out)
    }

    fn to_f32(self) -> f32 {
        let negative = self.negative();
        let words = if negative { self.negated().0 } else { self.0 };
        let Some(mut highest) = words
            .iter()
            .enumerate()
            .rev()
            .find(|(_, word)| **word != 0)
            .map(|(index, word)| index * 64 + 63 - word.leading_zeros() as usize)
        else {
            return 0.0;
        };
        let sign = if negative { 1u32 << 31 } else { 0 };
        if highest <= 22 {
            return f32::from_bits(sign | words[0] as u32);
        }
        let shift = highest - 23;
        let index = shift / 64;
        let offset = shift % 64;
        let mut top = words[index] >> offset;
        if offset != 0 && index + 1 < 4 {
            top |= words[index + 1] << (64 - offset);
        }
        let mut mantissa = (top & 0xff_ffff) as u32;
        if shift != 0 {
            let guard = shift - 1;
            let guard_bit = words[guard / 64] & (1u64 << (guard % 64)) != 0;
            let word = guard / 64;
            let offset = guard % 64;
            let sticky = words[..word].iter().any(|w| *w != 0)
                || (offset != 0 && words[word] & ((1u64 << offset) - 1) != 0);
            if guard_bit && (sticky || mantissa & 1 != 0) {
                mantissa += 1;
                if mantissa == 0x100_0000 {
                    mantissa >>= 1;
                    highest += 1;
                }
            }
        }
        f32::from_bits(sign | (((highest - 22) as u32) << 23) | (mantissa & 0x7f_ffff))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint() -> NnuecCheckpoint {
        let config = NnuecConfig {
            board_width: 3,
            line_width: 2,
            context_width: 3,
            local_width: 2,
            hidden_width: 4,
            causal_window: 3,
            max_lines: 2,
            max_plies: 8,
        };
        let weights = |n: usize| (0..n).map(|i| ((i % 17) as f32 - 8.0) / 128.0).collect();
        let e = config.line_width;
        NnuecCheckpoint {
            schema: NNUEC_SCHEMA.into(),
            semantics: NNUEC_SEMANTICS.into(),
            training: CpuTrainingState::Untrained,
            output_scale: 1000.0,
            parameters: NnuecParameters {
                board_weights: weights(CPU_FEATURE_COUNT * config.board_width),
                board_bias: vec![0.125; config.board_width],
                anchor_weights: weights(CPU_FEATURE_COUNT * e),
                anchor_bias: vec![0.25; e],
                from_weights: (0..64 * e).map(|i| i as f32 / 128.0).collect(),
                to_weights: weights(64 * e),
                promotion_weights: weights(5 * e),
                ply_weights: weights(config.max_plies * e),
                token_bias: vec![0.5; e],
                causal_weights: (0..config.causal_window)
                    .flat_map(|lag| vec![(lag + 1) as f32; e])
                    .collect(),
                causal_bias: vec![0.125; e],
                context_projection: weights(config.context_width * e),
                context_bias: vec![0.0; config.context_width],
                local_projection: weights(config.local_width * e),
                local_bias: vec![0.0; config.local_width],
                hidden_weights: weights(config.hidden_width * config.input_width()),
                hidden_bias: vec![0.125; config.hidden_width],
                output_weights: weights(NNUEC_OUTPUTS * config.hidden_width),
                output_bias: weights(NNUEC_OUTPUTS),
            },
            config,
        }
    }

    fn moves(text: &str) -> Vec<BoardMove> {
        text.split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect()
    }
    fn position(text: &str) -> Position {
        if let Some(fen) = text.strip_prefix("position fen ") {
            Position::from_fen(fen).unwrap()
        } else {
            let mut position = Position::startpos();
            if let Some(rest) = text.strip_prefix("position startpos moves ") {
                for mv in moves(rest) {
                    position.make_move(mv).unwrap();
                }
            } else {
                assert_eq!(text, "position startpos");
            }
            position
        }
    }
    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert!(
                (*a - *b).abs() <= NNUEC_ABSOLUTE_TOLERANCE + NNUEC_RELATIVE_TOLERANCE * b.abs(),
                "{a} != {b}"
            );
        }
    }
    fn bits(a: &[f32]) -> Vec<u32> {
        a.iter().map(|v| v.to_bits()).collect()
    }

    fn zero_parameters(checkpoint: &mut NnuecCheckpoint) {
        let p = &mut checkpoint.parameters;
        for values in [
            &mut p.board_weights,
            &mut p.board_bias,
            &mut p.anchor_weights,
            &mut p.anchor_bias,
            &mut p.from_weights,
            &mut p.to_weights,
            &mut p.promotion_weights,
            &mut p.ply_weights,
            &mut p.token_bias,
            &mut p.causal_weights,
            &mut p.causal_bias,
            &mut p.context_projection,
            &mut p.context_bias,
            &mut p.local_projection,
            &mut p.local_bias,
            &mut p.hidden_weights,
            &mut p.hidden_bias,
            &mut p.output_weights,
            &mut p.output_bias,
        ] {
            values.fill(0.0);
        }
    }

    #[test]
    fn native_value_profile_seals_frozen_context_and_observes_conditional_value() {
        let mut checkpoint = checkpoint();
        zero_parameters(&mut checkpoint);
        checkpoint.output_scale = 1.0;
        checkpoint.parameters.token_bias[0] = 1.0;
        checkpoint.parameters.causal_weights[0] = 1.0;
        checkpoint.parameters.context_projection[0] = 1.0;
        checkpoint.parameters.hidden_weights[2 * checkpoint.config.board_width] = 1.0;
        checkpoint.parameters.output_weights[0] = 100.0;
        checkpoint.parameters.output_bias[5] = -3.0;
        checkpoint.parameters.output_bias[6] = 9.0;
        let model = Arc::new(NnuecModel::new(checkpoint).unwrap());
        let root = position("position startpos");
        let empty = model.empty_context();
        let filled = model
            .with_line(&empty, 1, model.encode_line(&root, &moves("e2e4")).unwrap())
            .unwrap();
        let no_context = NnuecCpuValue::new(Arc::clone(&model), empty).unwrap();
        let contextual = NnuecCpuValue::new(Arc::clone(&model), filled).unwrap();
        let a = no_context.initialize(&root).unwrap();
        let b = contextual.initialize(&root).unwrap();
        assert_eq!(no_context.score(&a, &root).unwrap(), 0);
        assert_eq!(contextual.score(&b, &root).unwrap(), 100);
        assert_eq!(contextual.identity().training, CpuTrainingState::Untrained);
        assert_ne!(
            no_context.cache_context(&a, &root).unwrap(),
            contextual.cache_context(&b, &root).unwrap()
        );
        assert_eq!(
            contextual.cache_context(&a, &root),
            Err(CpuValueError::StateMismatch)
        );
        assert_eq!(
            contextual.observe(&b, &root).unwrap().context_utility(),
            [-3.0, 9.0, 0.0]
        );
        assert!(b.retained_bytes() > std::mem::size_of::<CpuAccumulator>());
    }

    #[test]
    fn native_checked_delta_and_undo_preserve_local_anchor_and_exact_board() {
        let model = Arc::new(NnuecModel::new(checkpoint()).unwrap());
        let evaluator = NnuecCpuValue::new(Arc::clone(&model), model.empty_context()).unwrap();
        let mut root = position("position startpos");
        let mut accumulator = evaluator.initialize(&root).unwrap();
        let original_values = accumulator.values;
        let rules = root.make_move("e2e4".parse().unwrap()).unwrap();
        let undo = evaluator
            .apply_delta(&mut accumulator, rules.delta())
            .unwrap();
        assert!(undo.retained_bytes() > std::mem::size_of::<CpuAccumulatorUndo>());
        let fresh_at_child = evaluator.initialize(&root).unwrap();
        assert_eq!(accumulator.values, fresh_at_child.values);
        assert_ne!(
            evaluator.cache_context(&accumulator, &root).unwrap(),
            evaluator.cache_context(&fresh_at_child, &root).unwrap()
        );
        let native = accumulator.nnuec.as_ref().unwrap();
        assert_eq!(native.local.moves(), &moves("e2e4"));
        assert_eq!(
            native.local.position().position_identity(),
            root.position_identity()
        );
        root.unmake(rules).unwrap();
        evaluator.restore(&mut accumulator, undo, &root).unwrap();
        assert_eq!(accumulator.values, original_values);
        assert!(accumulator.nnuec.as_ref().unwrap().local.moves().is_empty());
        let fresh_root = evaluator.initialize(&root).unwrap();
        assert_eq!(
            evaluator.cache_context(&accumulator, &root).unwrap(),
            evaluator.cache_context(&fresh_root, &root).unwrap()
        );
    }

    #[test]
    fn native_capacity_failure_and_foreign_clone_undo_leave_owned_state_unchanged() {
        let mut checkpoint = checkpoint();
        checkpoint.config.max_plies = 1;
        checkpoint
            .parameters
            .ply_weights
            .truncate(checkpoint.config.line_width);
        checkpoint.parameters.ply_weights.shrink_to_fit();
        let model = Arc::new(NnuecModel::new(checkpoint).unwrap());
        let evaluator = NnuecCpuValue::new(Arc::clone(&model), model.empty_context()).unwrap();
        let mut root = position("position startpos");
        let mut accumulator = evaluator.initialize(&root).unwrap();
        let first = root.make_move("e2e4".parse().unwrap()).unwrap();
        let undo = evaluator
            .apply_delta(&mut accumulator, first.delta())
            .unwrap();
        let values = accumulator.values;
        let identity = accumulator.identity.clone();
        let second = root.make_move("e7e5".parse().unwrap()).unwrap();
        assert_eq!(
            evaluator
                .apply_delta(&mut accumulator, second.delta())
                .err(),
            Some(CpuValueError::InvalidCheckpoint(
                "NNUEC configured capacity exceeded"
            ))
        );
        assert_eq!(accumulator.values, values);
        assert_eq!(accumulator.identity, identity);
        root.unmake(second).unwrap();
        let mut cloned = accumulator.clone();
        root.unmake(first).unwrap();
        assert_eq!(
            evaluator.restore(&mut cloned, undo, &root),
            Err(CpuValueError::StateMismatch)
        );
        assert_eq!(cloned.values, values);
    }

    #[test]
    fn cancellation_does_not_turn_context_deletion_into_a_different_empty_view() {
        let mut checkpoint = checkpoint();
        zero_parameters(&mut checkpoint);
        checkpoint.parameters.context_bias[0] = 1.0;
        checkpoint.parameters.token_bias[0] = 1000.0;
        checkpoint.parameters.causal_weights[0] = 1000.0;
        checkpoint.parameters.context_projection[0] = 1000.0;
        let model = NnuecModel::new(checkpoint).unwrap();
        let root = position("position startpos");
        let empty = model.empty_context();
        let line = model.encode_line(&root, &moves("e2e4")).unwrap();
        let occupied = model.with_line(&empty, 1, line).unwrap();
        let removed = model.without_line(&occupied, 1).unwrap();
        assert_eq!(bits(removed.values()), bits(empty.values()));
        close(
            removed.values(),
            model.rebuild_context(&removed).unwrap().values(),
        );
    }

    #[test]
    fn board_delta_rebuild_handles_half_ulp_hidden_by_large_canceling_features() {
        let mut checkpoint = checkpoint();
        zero_parameters(&mut checkpoint);
        let mut root = position("position startpos");
        let snapshot = root.snapshot();
        for square in (0..8).chain(56..64) {
            let square = Square::new(square).unwrap();
            let piece = snapshot.piece_at(square).unwrap();
            let feature = piece_feature(piece, square, Color::White);
            checkpoint.parameters.board_weights[feature * checkpoint.config.board_width] =
                if square.rank() == 0 { 1000.0 } else { -1000.0 };
        }
        let feature = piece_feature(
            snapshot.piece_at("e2".parse().unwrap()).unwrap(),
            "e2".parse().unwrap(),
            Color::White,
        );
        checkpoint.parameters.board_weights[feature * checkpoint.config.board_width] =
            2.0f32.powi(-12);
        let model = NnuecModel::new(checkpoint).unwrap();
        let mut board = model.board(&root).unwrap();
        let rules = root.make_move("e2e4".parse().unwrap()).unwrap();
        model.apply_board_delta(&mut board, rules.delta()).unwrap();
        let rebuilt = model.board(&root).unwrap();
        close(&board.values[0], &rebuilt.values[0]);
        assert_eq!(bits(&board.values[0]), bits(&rebuilt.values[0]));
    }

    #[test]
    fn exact_context_sum_rounds_ties_sticky_subnormals_and_rejects_oversized_operands() {
        let one = Exact256::from_f32(1.0).unwrap();
        let half = Exact256::from_f32(2.0f32.powi(-24)).unwrap();
        let tiny = Exact256::from_f32(f32::from_bits(1)).unwrap();
        let tie = one.checked_add(half).unwrap();
        assert_eq!(tie.to_f32().to_bits(), 1.0f32.to_bits());
        assert_eq!(
            tie.checked_add(tiny).unwrap().to_f32().to_bits(),
            1.0f32.to_bits() + 1
        );
        assert_eq!(tiny.to_f32().to_bits(), 1);
        assert_eq!(tiny.negated().to_f32().to_bits(), (1 << 31) | 1);
        let maximum = Exact256::from_f32(MAX_CONTEXT_OPERAND).unwrap();
        assert_eq!(
            maximum
                .checked_add(maximum.negated())
                .unwrap()
                .to_f32()
                .to_bits(),
            0
        );
        assert_eq!(
            Exact256::from_f32(f32::MAX).err(),
            Some(NnuecError::AccumulatorOverflow)
        );
        assert_eq!(
            Exact256::from_f32(MAX_CONTEXT_OPERAND * 2.0).err(),
            Some(NnuecError::AccumulatorOverflow)
        );
        let signed_max = Exact256([u64::MAX, u64::MAX, u64::MAX, u64::MAX >> 1]);
        assert_eq!(
            signed_max.checked_add(tiny).err(),
            Some(NnuecError::AccumulatorOverflow)
        );
    }

    #[test]
    fn required_config_parameters_and_weight_domain_are_validated() {
        let good = checkpoint();
        good.validate().unwrap();
        let mut changed = good.clone();
        changed.config.causal_window = 4;
        assert_eq!(
            changed.validate(),
            Err(NnuecError::InvalidConfig(
                "unsupported dimension or capacity"
            ))
        );
        changed = good.clone();
        changed.parameters.board_weights.pop();
        assert!(matches!(
            changed.validate(),
            Err(NnuecError::InvalidCheckpoint(
                "parameter shape/allocation mismatch"
            ))
        ));
        changed = good.clone();
        changed.parameters.output_bias[0] = f32::NAN;
        assert!(matches!(
            changed.validate(),
            Err(NnuecError::InvalidCheckpoint(
                "non-finite/excessive parameter"
            ))
        ));
        changed = good.clone();
        changed.training = CpuTrainingState::Bootstrap;
        assert!(matches!(
            changed.validate(),
            Err(NnuecError::InvalidCheckpoint(_))
        ));
        changed = good.clone();
        changed.parameters.output_bias[0] += 0.125;
        assert_ne!(
            changed.weights_sha256().unwrap(),
            good.weights_sha256().unwrap()
        );
        changed = good.clone();
        changed.training = CpuTrainingState::Learned {
            run_id: "declaration-only".into(),
            steps: 1,
            dataset_sha256: "a".repeat(64),
        };
        assert_eq!(
            changed.weights_sha256().unwrap(),
            good.weights_sha256().unwrap()
        );
        let bytes = serde_json::to_vec(&good).unwrap();
        let restored = NnuecModel::from_reader(bytes.as_slice()).unwrap();
        assert_eq!(restored.identity().training, CpuTrainingState::Untrained);
    }

    #[test]
    fn board_rules_deltas_match_rebuild_and_undo_restores_original_bits() {
        let model = NnuecModel::new(checkpoint()).unwrap();
        for (fen, text) in [
            ("position startpos", "e2e4"),
            ("position fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
            ("position fen 4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", "e5d6"),
            ("position fen 4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),
            ("position fen 4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8r"),
            ("position fen 4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8b"),
            ("position fen 4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n"),
        ] {
            let mut position = position(fen);
            let mut board = model.board(&position).unwrap();
            let original = board.values.clone();
            let rules = position.make_move(text.parse().unwrap()).unwrap();
            let undo = model.apply_board_delta(&mut board, rules.delta()).unwrap();
            let rebuilt = model.board(&position).unwrap();
            close(&board.values[0], &rebuilt.values[0]);
            close(&board.values[1], &rebuilt.values[1]);
            assert_eq!(bits(&board.values[0]), bits(&rebuilt.values[0]));
            assert_eq!(bits(&board.values[1]), bits(&rebuilt.values[1]));
            let child = board.values.clone();
            assert_eq!(
                model.apply_board_delta(&mut board, rules.delta()).err(),
                Some(NnuecError::StateMismatch)
            );
            assert_eq!(bits(&board.values[0]), bits(&child[0]));
            position.unmake(rules).unwrap();
            model.restore_board(&mut board, undo, &position).unwrap();
            assert_eq!(bits(&board.values[0]), bits(&original[0]));
            assert_eq!(bits(&board.values[1]), bits(&original[1]));
        }
    }

    #[test]
    fn causal_lines_preserve_order_and_local_lifo_is_transactional() {
        let model = NnuecModel::new(checkpoint()).unwrap();
        let root = position("position startpos");
        let a = model
            .encode_line(&root, &moves("g1f3 g8f6 b1c3 b8c6"))
            .unwrap();
        let b = model
            .encode_line(&root, &moves("b1c3 b8c6 g1f3 g8f6"))
            .unwrap();
        assert_ne!(bits(a.embedding()), bits(b.embedding()));
        let mut local = model.local(&root).unwrap();
        let empty = bits(local.values());
        let empty_position = local.position().position_identity();
        let first = model
            .append_local(&mut local, "e2e4".parse().unwrap())
            .unwrap();
        let after_first = bits(local.values());
        assert!(matches!(
            model.append_local(&mut local, "e2e4".parse().unwrap()),
            Err(NnuecError::Rules(_))
        ));
        assert_eq!(bits(local.values()), after_first);
        assert_eq!(local.moves().len(), 1);
        let second = model
            .append_local(&mut local, "e7e5".parse().unwrap())
            .unwrap();
        model.undo_local(&mut local, second).unwrap();
        assert_eq!(bits(local.values()), after_first);
        model.undo_local(&mut local, first).unwrap();
        assert_eq!(bits(local.values()), empty);
        assert_eq!(local.position().position_identity(), empty_position);
        let rebuilt = model
            .encode_line(&root, &moves("g1f3 g8f6 b1c3 b8c6"))
            .unwrap();
        assert_eq!(bits(a.embedding()), bits(rebuilt.embedding()));
    }

    #[test]
    fn immutable_context_add_replace_delete_matches_rebuild_without_worker_copy() {
        let model = NnuecModel::new(checkpoint()).unwrap();
        let root = position("position startpos");
        let a = model.encode_line(&root, &moves("e2e4 e7e5")).unwrap();
        let b = model.encode_line(&root, &moves("d2d4 d7d5")).unwrap();
        let c0 = model.empty_context();
        let old = bits(c0.values());
        let c1 = model.with_line(&c0, 17, Arc::clone(&a)).unwrap();
        let c2 = model.with_line(&c1, 99, Arc::clone(&b)).unwrap();
        assert_eq!(bits(c0.values()), old);
        assert!(c0.is_empty());
        assert!(Arc::ptr_eq(
            c1.lines.get(&17).unwrap(),
            c2.lines.get(&17).unwrap()
        ));
        close(c2.values(), model.rebuild_context(&c2).unwrap().values());
        assert_eq!(
            bits(c2.values()),
            bits(model.rebuild_context(&c2).unwrap().values())
        );
        let c3 = model.with_line(&c2, 17, Arc::clone(&b)).unwrap();
        close(c3.values(), model.rebuild_context(&c3).unwrap().values());
        assert_eq!(
            bits(c3.values()),
            bits(model.rebuild_context(&c3).unwrap().values())
        );
        let c4 = model.without_line(&c3, 99).unwrap();
        close(c4.values(), model.rebuild_context(&c4).unwrap().values());
        assert_eq!(
            bits(c4.values()),
            bits(model.rebuild_context(&c4).unwrap().values())
        );
        let before = bits(c2.values());
        assert!(matches!(
            model.with_line(&c2, 100, a),
            Err(NnuecError::Capacity("context lines"))
        ));
        assert_eq!(bits(c2.values()), before);
        let mut foreign_checkpoint = checkpoint();
        foreign_checkpoint.parameters.output_bias[0] += 1.0;
        let foreign = NnuecModel::new(foreign_checkpoint).unwrap();
        assert!(matches!(
            foreign.with_line(&c1, 99, b),
            Err(NnuecError::StateMismatch)
        ));
    }

    #[test]
    fn empty_perspectives_and_four_promotions_remain_distinct_without_rules_authority() {
        let model = NnuecModel::new(checkpoint()).unwrap();
        let root = position("position fen 4k3/P7/8/8/8/8/8/4K3 w - - 0 1");
        let embeddings: Vec<Vec<u32>> = ["a7a8q", "a7a8r", "a7a8b", "a7a8n"]
            .into_iter()
            .map(|s| {
                bits(
                    model
                        .encode_line(&root, &[s.parse().unwrap()])
                        .unwrap()
                        .embedding(),
                )
            })
            .collect();
        for (i, a) in embeddings.iter().enumerate() {
            for b in &embeddings[i + 1..] {
                assert_ne!(a, b);
            }
        }
        for root in [
            position("position startpos"),
            position("position startpos moves e2e4"),
        ] {
            let board = model.board(&root).unwrap();
            let c = model.empty_context();
            let l = model.local(&root).unwrap();
            let output = model.forward(&board, &c, &l, &root).unwrap();
            assert_eq!(output.schema, NNUEC_OUTPUT_SCHEMA);
            assert!((output.wdl.iter().sum::<f32>() - 1.0).abs() < 1e-6);
            assert!(output.wdl.iter().all(|v| *v >= 0.0));
            assert!((-20_000..=20_000).contains(&output.frontier_score()));
            let wire = model.reference_input(&c, &l, &root).unwrap();
            assert_eq!(wire.side_to_move, color_name(root.side_to_move()));
            assert!(wire.local_line.moves.is_empty());
        }
    }

    #[test]
    fn capacity_and_nonfinite_failures_preserve_local_and_context_semantics() {
        let mut tiny = checkpoint();
        tiny.config.max_plies = 1;
        tiny.parameters.ply_weights.truncate(tiny.config.line_width);
        tiny.parameters.ply_weights.shrink_to_fit();
        let model = NnuecModel::new(tiny).unwrap();
        let root = position("position startpos");
        let mut local = model.local(&root).unwrap();
        model
            .append_local(&mut local, "e2e4".parse().unwrap())
            .unwrap();
        let before = bits(local.values());
        let position_before = local.position().position_identity();
        assert!(matches!(
            model.append_local(&mut local, "e7e5".parse().unwrap()),
            Err(NnuecError::Capacity("local plies"))
        ));
        assert_eq!(bits(local.values()), before);
        assert_eq!(local.position().position_identity(), position_before);
        assert_eq!(add(f32::MAX, f32::MAX), Err(NnuecError::NonFinite));
        assert_eq!(mul(f32::MAX, 2.0), Err(NnuecError::NonFinite));
        // A rejected LIFO token cannot rewind a different local owner.
        let mut other = model.local(&root).unwrap();
        let token = model
            .append_local(&mut other, "e2e4".parse().unwrap())
            .unwrap();
        assert_eq!(
            model.undo_local(&mut local, token),
            Err(NnuecError::StateMismatch)
        );
        assert_eq!(bits(local.values()), before);
    }
}
