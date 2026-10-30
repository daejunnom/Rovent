//! CPU 전용 평가 경계와 자체 FP32 accumulator256 → hidden32 → value 모델.
//!
//! 학습 실행을 포함하지 않는다. `Untrained`와 `Learned`는 체크포인트의 선언이며,
//! 후자의 메타데이터 형식 검증은 실제 학습 실행의 독립 증거를 확인하지 않는다.
//! bootstrap material/PST와 학습 가능한 float 모델은 서로 다른 의미 식별을 쓴다.
//! Rules가 만든 `RuleMoveDelta`만 증분 갱신에 사용하고 모델은 legality를 판단하지 않는다.
//! forward는 Rust로 실행하며 Python/ORT/GPU를 호출하지 않는다. feature 합산은
//! 정확한 signed192 정수에서 수행하고 FP32로 한 번 nearest-even 반올림한다.

use crate::cpu::{BOOTSTRAP_SCORE_VERSION, evaluate_bootstrap};
use rz_position::{
    Color, Piece, Position, PositionIdentity, PositionSnapshot, RuleMoveDelta, Square,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::sync::Arc;

pub mod nnuec;

pub const CPU_VALUE_SCHEMA: &str = "rz-cpu-float-value-v2";
pub const CPU_VALUE_ARCHITECTURE: &str = "piece-square-metadata-exact192-acc256-hidden32-fp32-v1";
pub const CPU_ACCUMULATOR_WIDTH: usize = 256;
pub const CPU_HIDDEN_WIDTH: usize = 32;
pub const CPU_FEATURE_COUNT: usize = 800;
pub const CPU_VALUE_MAX_BYTES: usize = 8 * 1024 * 1024;
const PIECE_FEATURES: usize = 12 * 64;
const INPUT_WIDTH: usize = 2 * CPU_ACCUMULATOR_WIDTH;
const MAX_PARAMETER_MAGNITUDE: f32 = 1_000_000.0;
const MAX_FEATURE_CHANGES: usize = 24;

// A finite FP32 parameter is a signed integer times 2^-149. The admitted
// parameter bound and all 800 feature rows plus bias imply |sum| < 2^30,
// hence the scaled integer has magnitude < 2^179. Even a transition's at most
// 22 additional terms stays below 2^179, well below signed192's 2^191 bound.
// We still check arithmetic overflow instead of silently wrapping.
type ExactValues = [[Signed192; CPU_ACCUMULATOR_WIDTH]; 2];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Signed192([u64; 3]);

impl Signed192 {
    fn from_f32(value: f32) -> Result<Self, CpuValueError> {
        if !value.is_finite() || value.abs() > MAX_PARAMETER_MAGNITUDE {
            return Err(CpuValueError::InvalidCheckpoint(
                "exact-sum operand outside parameter bound",
            ));
        }
        let bits = value.to_bits();
        let exponent = ((bits >> 23) & 0xff) as usize;
        let fraction = bits & 0x7f_ffff;
        let mantissa = if exponent == 0 {
            fraction
        } else {
            fraction | 0x80_0000
        };
        let shift = exponent.saturating_sub(1);
        let word = shift / 64;
        let offset = shift % 64;
        let mut magnitude = [0u64; 3];
        if word >= 3 || shift + 24 >= 192 {
            return Err(CpuValueError::AccumulatorOverflow);
        }
        magnitude[word] = u64::from(mantissa) << offset;
        if offset > 40 {
            magnitude[word + 1] = u64::from(mantissa) >> (64 - offset);
        }
        let positive = Self(magnitude);
        Ok(if bits >> 31 != 0 {
            positive.negated()
        } else {
            positive
        })
    }

    fn negative(self) -> bool {
        self.0[2] >> 63 != 0
    }

    fn negated(self) -> Self {
        let mut result = self.0.map(|word| !word);
        let mut carry = true;
        for word in &mut result {
            let (next, overflow) = word.overflowing_add(u64::from(carry));
            *word = next;
            carry = overflow;
        }
        Self(result)
    }

    fn checked_add(self, other: Self) -> Result<Self, CpuValueError> {
        let mut result = [0u64; 3];
        let mut carry = false;
        for (index, word) in result.iter_mut().enumerate() {
            let (sum, first_carry) = self.0[index].overflowing_add(other.0[index]);
            let (sum, second_carry) = sum.overflowing_add(u64::from(carry));
            *word = sum;
            carry = first_carry || second_carry;
        }
        let result = Self(result);
        if self.negative() == other.negative() && self.negative() != result.negative() {
            return Err(CpuValueError::AccumulatorOverflow);
        }
        Ok(result)
    }

    /// One canonical round-to-nearest, ties-to-even from the exact sum.
    /// Exact zero uses +0. Every representable signed192 value scaled by
    /// 2^-149 is finite FP32, including subnormal inputs and cancellation.
    fn to_f32(self) -> f32 {
        let negative = self.negative();
        let magnitude = if negative { self.negated().0 } else { self.0 };
        let Some(mut highest) = magnitude
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
            return f32::from_bits(sign | magnitude[0] as u32);
        }
        let shift = highest - 23;
        let word = shift / 64;
        let offset = shift % 64;
        let mut top = magnitude[word] >> offset;
        if offset != 0 && word + 1 < 3 {
            top |= magnitude[word + 1] << (64 - offset);
        }
        let mut mantissa = (top & 0xff_ffff) as u32;
        if shift != 0 {
            let guard_position = shift - 1;
            let guard = magnitude[guard_position / 64] & (1u64 << (guard_position % 64)) != 0;
            let sticky_word = guard_position / 64;
            let sticky_offset = guard_position % 64;
            let sticky = magnitude[..sticky_word].iter().any(|word| *word != 0)
                || (sticky_offset != 0
                    && magnitude[sticky_word] & ((1u64 << sticky_offset) - 1) != 0);
            if guard && (sticky || mantissa & 1 != 0) {
                mantissa += 1;
                if mantissa == 0x100_0000 {
                    mantissa >>= 1;
                    highest += 1;
                }
            }
        }
        let exponent = (highest - 22) as u32;
        f32::from_bits(sign | (exponent << 23) | (mantissa & 0x7f_ffff))
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct FeatureChange {
    feature: u16,
    sign: i8,
}

#[derive(Clone, Debug)]
struct FeatureChanges {
    values: [[FeatureChange; MAX_FEATURE_CHANGES]; 2],
    lengths: [usize; 2],
}

impl Default for FeatureChanges {
    fn default() -> Self {
        Self {
            values: [[FeatureChange::default(); MAX_FEATURE_CHANGES]; 2],
            lengths: [0; 2],
        }
    }
}

impl FeatureChanges {
    fn push(&mut self, perspective: Color, feature: usize, sign: i8) -> Result<(), CpuValueError> {
        let side = perspective as usize;
        if feature >= CPU_FEATURE_COUNT || self.lengths[side] >= MAX_FEATURE_CHANGES {
            return Err(CpuValueError::InvalidCheckpoint(
                "feature delta bound exceeded",
            ));
        }
        self.values[side][self.lengths[side]] = FeatureChange {
            feature: feature as u16,
            sign,
        };
        self.lengths[side] += 1;
        Ok(())
    }

    fn for_side(&self, perspective: Color) -> &[FeatureChange] {
        &self.values[perspective as usize][..self.lengths[perspective as usize]]
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CpuTrainingState {
    Bootstrap,
    Untrained,
    Learned {
        /// Checkpoint-declared metadata, not independent proof of execution.
        run_id: String,
        steps: u64,
        dataset_sha256: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpuValueIdentity {
    pub semantics: String,
    /// Digest of actual float parameter bytes, not of a path or a label.
    pub weights_sha256: Option<String>,
    pub training: CpuTrainingState,
}

impl CpuValueIdentity {
    /// Validate bounded namespace declarations. Learned metadata is not proof
    /// that training occurred, and a supplied digest is not proof of its bytes.
    /// The checkpoint owner separately validates and hashes actual parameters.
    pub fn validate(&self) -> Result<(), CpuValueError> {
        if self.semantics.trim().is_empty()
            || self.semantics.len() > 256
            || self.semantics.capacity() > 256
            || self.semantics.chars().any(char::is_control)
        {
            return Err(CpuValueError::InvalidCheckpoint(
                "invalid value semantics identity",
            ));
        }
        if self
            .weights_sha256
            .as_ref()
            .is_some_and(|hash| hash.capacity() > 64 || !valid_digest(hash))
        {
            return Err(CpuValueError::InvalidCheckpoint(
                "invalid value weights digest",
            ));
        }
        match &self.training {
            CpuTrainingState::Bootstrap if self.weights_sha256.is_some() => {
                return Err(CpuValueError::InvalidCheckpoint(
                    "bootstrap identity cannot claim float weights",
                ));
            }
            CpuTrainingState::Untrained | CpuTrainingState::Learned { .. }
                if self.weights_sha256.is_none() =>
            {
                return Err(CpuValueError::InvalidCheckpoint(
                    "float value identity requires weights digest",
                ));
            }
            _ => {}
        }
        validate_training_declaration(&self.training)
    }
}

fn validate_training_declaration(training: &CpuTrainingState) -> Result<(), CpuValueError> {
    if let CpuTrainingState::Learned {
        run_id,
        steps,
        dataset_sha256,
    } = training
    {
        if run_id.trim().is_empty()
            || run_id.len() > 128
            || run_id.capacity() > 128
            || run_id.chars().any(char::is_control)
            || *steps == 0
            || dataset_sha256.capacity() > 64
            || !valid_digest(dataset_sha256)
        {
            return Err(CpuValueError::InvalidCheckpoint(
                "missing or invalid declared training metadata",
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum CpuValueError {
    CheckpointSize,
    InvalidCheckpoint(&'static str),
    Io(String),
    Json(String),
    StateMismatch,
    /// The original immutable NNUEC Bound revoked this invocation. This is
    /// distinct from an accumulator, namespace, numerical or owner failure.
    AuthorityRevoked(CpuValueAuthorityRevocation),
    NonFiniteForward,
    ScoreOutsideFrontierNamespace,
    AccumulatorOverflow,
}

impl std::fmt::Display for CpuValueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CheckpointSize => f.write_str("CPU checkpoint exceeds 8 MiB"),
            Self::InvalidCheckpoint(message) => write!(f, "CPU checkpoint: {message}"),
            Self::Io(message) => write!(f, "CPU checkpoint read: {message}"),
            Self::Json(message) => write!(f, "CPU checkpoint JSON: {message}"),
            Self::StateMismatch => {
                f.write_str("CPU accumulator does not match exact Rules state or value namespace")
            }
            Self::AuthorityRevoked(reason) => {
                write!(f, "original CPU NNUEC authority revoked: {reason:?}")
            }
            Self::NonFiniteForward => f.write_str("CPU FP32 forward produced a non-finite value"),
            Self::ScoreOutsideFrontierNamespace => {
                f.write_str("CPU nonterminal evaluator score entered the Rules mate namespace")
            }
            Self::AccumulatorOverflow => f.write_str("CPU signed192 accumulator overflow"),
        }
    }
}

impl std::error::Error for CpuValueError {}

/// Original evaluator control observation, not a cleanup or resume capability.
/// Hard owner/context revocations cannot be relabeled as cancellation/expiry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuValueAuthorityRevocation {
    Cancelled,
    Expired,
    ActorStopped,
    Invalidated,
}

/// Immutable weights for an entire admitted CPU task. Changing weights or
/// semantics requires a new evaluator/engine; implementations must not update
/// them underneath an active task or retain opponent/private PALS state.
pub trait CpuValueEvaluator: Send + Sync {
    fn identity(&self) -> &CpuValueIdentity;
    fn provenance(&self) -> &'static str;
    /// Immutable task registration for a context-bound evaluator. Legacy
    /// values retain their original namespace bytes. A revoked binding is an
    /// error, never None or permission to fall back to a model-only namespace.
    fn job_binding_digest(&self) -> Result<Option<rz_contracts::Digest>, CpuValueError> {
        Ok(None)
    }
    /// Cleanup custody only. It cannot revive a revoked score/resume scope.
    fn nnuec_retirement_owner(&self) -> Option<nnuec::v2::retirement::NnuecValueRetirementOwner> {
        None
    }
    fn initialize(&self, position: &Position) -> Result<CpuAccumulator, CpuValueError>;
    /// Exact context of a position-dependent value. NNUEC additionally seals
    /// the frozen C owner and L anchor; a model/weights identity alone is not a
    /// sufficient TT key for a path-conditioned evaluator.
    fn cache_context(
        &self,
        accumulator: &CpuAccumulator,
        position: &Position,
    ) -> Result<CpuValueCacheContext, CpuValueError> {
        if accumulator.identity != position.position_identity()
            || accumulator.namespace.as_ref() != self.identity()
            || accumulator.nnuec.is_some()
        {
            return Err(CpuValueError::StateMismatch);
        }
        Ok(CpuValueCacheContext::default())
    }
    fn score(
        &self,
        accumulator: &CpuAccumulator,
        position: &Position,
    ) -> Result<i32, CpuValueError>;
    /// Explicit collection only. Legacy evaluators retain their same score
    /// call and report unsupported capture rather than fabricating an input.
    fn score_with_capture(
        &self,
        accumulator: &CpuAccumulator,
        position: &Position,
        capture: &nnuec::v2::capture::NnuecActiveCapture,
    ) -> Result<i32, CpuValueError> {
        let score = self.score(accumulator, position)?;
        capture.unsupported();
        Ok(score)
    }
    /// Transactional update: failure leaves accumulator unchanged.
    fn apply_delta(
        &self,
        accumulator: &mut CpuAccumulator,
        delta: &RuleMoveDelta,
    ) -> Result<CpuAccumulatorUndo, CpuValueError>;
    /// Rules must already have unmade its own undo token. Float evaluators undo
    /// their exact integer feature changes, then perform the same canonical round.
    fn restore(
        &self,
        accumulator: &mut CpuAccumulator,
        undo: CpuAccumulatorUndo,
        position: &Position,
    ) -> Result<(), CpuValueError> {
        if accumulator.identity != undo.after
            || position.position_identity() != undo.before
            || accumulator.namespace.as_ref() != self.identity()
            || accumulator.namespace != undo.namespace
        {
            return Err(CpuValueError::StateMismatch);
        }
        if undo.changes.is_some() || undo.nnuec.is_some() || accumulator.nnuec.is_some() {
            return Err(CpuValueError::StateMismatch);
        }
        accumulator.identity = undo.before;
        Ok(())
    }
}

/// Opaque, cheap exact cache context. The authority pins only a small seal,
/// never an old C view or all of its archived lines. Rules identity supplies
/// exact current history; the local anchor then determines the full suffix.
#[derive(Clone, Debug, Default)]
pub struct CpuValueCacheContext {
    contextual: Option<ContextualCacheContext>,
}

#[derive(Clone, Debug)]
struct ContextualCacheContext {
    authority: Arc<()>,
    local_anchor: PositionIdentity,
}

impl PartialEq for CpuValueCacheContext {
    fn eq(&self, other: &Self) -> bool {
        match (&self.contextual, &other.contextual) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                Arc::ptr_eq(&a.authority, &b.authority) && a.local_anchor == b.local_anchor
            }
            _ => false,
        }
    }
}
impl Eq for CpuValueCacheContext {}

impl CpuValueCacheContext {
    fn contextual(authority: &Arc<()>, local_anchor: &PositionIdentity) -> Self {
        Self {
            contextual: Some(ContextualCacheContext {
                authority: Arc::clone(authority),
                local_anchor: local_anchor.clone(),
            }),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CpuAccumulator {
    values: [[f32; CPU_ACCUMULATOR_WIDTH]; 2],
    identity: PositionIdentity,
    namespace: Arc<CpuValueIdentity>,
    exact: Option<Box<ExactValues>>,
    nnuec: Option<Box<nnuec::NnuecNativeAccumulator>>,
}

impl CpuAccumulator {
    pub fn values(&self) -> &[[f32; CPU_ACCUMULATOR_WIDTH]; 2] {
        &self.values
    }
    /// Conservative inline + owned backing charge. Frozen model/C and shared
    /// Rules history are retained by their existing owner, not copied here.
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(
                self.exact
                    .as_ref()
                    .map_or(0, |_| std::mem::size_of::<ExactValues>()),
            )
            .saturating_add(self.nnuec.as_ref().map_or(0, |v| v.retained_bytes()))
    }
}

pub struct CpuAccumulatorUndo {
    changes: Option<FeatureChanges>,
    before: PositionIdentity,
    after: PositionIdentity,
    namespace: Arc<CpuValueIdentity>,
    nnuec: Option<Box<nnuec::NnuecNativeUndo>>,
}

impl CpuAccumulatorUndo {
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(self.nnuec.as_ref().map_or(0, |v| v.retained_bytes()))
    }
}

pub struct BootstrapCpuValue {
    identity: Arc<CpuValueIdentity>,
}

impl Default for BootstrapCpuValue {
    fn default() -> Self {
        Self {
            identity: Arc::new(CpuValueIdentity {
                semantics: BOOTSTRAP_SCORE_VERSION.to_owned(),
                weights_sha256: None,
                training: CpuTrainingState::Bootstrap,
            }),
        }
    }
}

impl CpuValueEvaluator for BootstrapCpuValue {
    fn identity(&self) -> &CpuValueIdentity {
        &self.identity
    }

    fn provenance(&self) -> &'static str {
        BOOTSTRAP_SCORE_VERSION
    }

    fn initialize(&self, position: &Position) -> Result<CpuAccumulator, CpuValueError> {
        Ok(CpuAccumulator {
            values: [[0.0; CPU_ACCUMULATOR_WIDTH]; 2],
            identity: position.position_identity(),
            namespace: Arc::clone(&self.identity),
            exact: None,
            nnuec: None,
        })
    }

    fn score(
        &self,
        accumulator: &CpuAccumulator,
        position: &Position,
    ) -> Result<i32, CpuValueError> {
        if accumulator.identity != position.position_identity()
            || accumulator.namespace != self.identity
        {
            return Err(CpuValueError::StateMismatch);
        }
        Ok(evaluate_bootstrap(position))
    }

    fn apply_delta(
        &self,
        accumulator: &mut CpuAccumulator,
        delta: &RuleMoveDelta,
    ) -> Result<CpuAccumulatorUndo, CpuValueError> {
        if accumulator.identity != delta.before.position_identity()
            || accumulator.namespace != self.identity
        {
            return Err(CpuValueError::StateMismatch);
        }
        let undo = CpuAccumulatorUndo {
            changes: None,
            before: accumulator.identity.clone(),
            after: delta.after.position_identity(),
            namespace: Arc::clone(&self.identity),
            nnuec: None,
        };
        accumulator.identity = undo.after.clone();
        Ok(undo)
    }
}

/// Row-major feature embeddings and dense matrices. Accumulator activation is
/// clip(0,1), hidden activation is ReLU, and the scalar uses explicit output_scale.
/// All dimensions, dtype/semantics and allocation bounds are fixed by the schema.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpuFloatCheckpoint {
    pub schema: String,
    pub architecture: String,
    pub training: CpuTrainingState,
    pub feature_weights: Vec<f32>,
    pub accumulator_bias: Vec<f32>,
    pub hidden_weights: Vec<f32>,
    pub hidden_bias: Vec<f32>,
    pub output_weights: Vec<f32>,
    pub output_bias: f32,
    pub output_scale: f32,
}

impl CpuFloatCheckpoint {
    pub fn zeros_untrained() -> Self {
        Self {
            schema: CPU_VALUE_SCHEMA.to_owned(),
            architecture: CPU_VALUE_ARCHITECTURE.to_owned(),
            training: CpuTrainingState::Untrained,
            feature_weights: vec![0.0; CPU_FEATURE_COUNT * CPU_ACCUMULATOR_WIDTH],
            accumulator_bias: vec![0.0; CPU_ACCUMULATOR_WIDTH],
            hidden_weights: vec![0.0; CPU_HIDDEN_WIDTH * INPUT_WIDTH],
            hidden_bias: vec![0.0; CPU_HIDDEN_WIDTH],
            output_weights: vec![0.0; CPU_HIDDEN_WIDTH],
            output_bias: 0.0,
            output_scale: 1000.0,
        }
    }

    pub fn validate(&self) -> Result<(), CpuValueError> {
        if self.schema != CPU_VALUE_SCHEMA || self.architecture != CPU_VALUE_ARCHITECTURE {
            return Err(CpuValueError::InvalidCheckpoint(
                "unsupported schema or architecture",
            ));
        }
        if self.feature_weights.len() != CPU_FEATURE_COUNT * CPU_ACCUMULATOR_WIDTH
            || self.accumulator_bias.len() != CPU_ACCUMULATOR_WIDTH
            || self.hidden_weights.len() != CPU_HIDDEN_WIDTH * INPUT_WIDTH
            || self.hidden_bias.len() != CPU_HIDDEN_WIDTH
            || self.output_weights.len() != CPU_HIDDEN_WIDTH
        {
            return Err(CpuValueError::InvalidCheckpoint("matrix shape mismatch"));
        }
        if self
            .feature_weights
            .iter()
            .chain(&self.accumulator_bias)
            .chain(&self.hidden_weights)
            .chain(&self.hidden_bias)
            .chain(&self.output_weights)
            .any(|value| !value.is_finite() || value.abs() > MAX_PARAMETER_MAGNITUDE)
            || !self.output_bias.is_finite()
            || self.output_bias.abs() > MAX_PARAMETER_MAGNITUDE
            || !self.output_scale.is_finite()
            || !(0.0..=10_000.0).contains(&self.output_scale)
            || self.output_scale == 0.0
        {
            return Err(CpuValueError::InvalidCheckpoint(
                "non-finite or excessive parameter",
            ));
        }
        if self.training == CpuTrainingState::Bootstrap {
            return Err(CpuValueError::InvalidCheckpoint(
                "float weights cannot claim bootstrap provenance",
            ));
        }
        validate_training_declaration(&self.training)
    }

    pub fn weights_sha256(&self) -> String {
        let mut digest = Sha256::new();
        digest.update(CPU_VALUE_ARCHITECTURE.as_bytes());
        for value in self
            .feature_weights
            .iter()
            .chain(&self.accumulator_bias)
            .chain(&self.hidden_weights)
            .chain(&self.hidden_bias)
            .chain(&self.output_weights)
            .chain([&self.output_bias, &self.output_scale])
        {
            digest.update(value.to_bits().to_le_bytes());
        }
        format!("{:x}", digest.finalize())
    }
}

pub struct CpuFloatValue {
    checkpoint: Arc<CpuFloatCheckpoint>,
    identity: Arc<CpuValueIdentity>,
}

impl CpuFloatValue {
    pub fn new(checkpoint: CpuFloatCheckpoint) -> Result<Self, CpuValueError> {
        checkpoint.validate()?;
        let identity = CpuValueIdentity {
            semantics: CPU_VALUE_ARCHITECTURE.to_owned(),
            weights_sha256: Some(checkpoint.weights_sha256()),
            training: checkpoint.training.clone(),
        };
        identity.validate()?;
        Ok(Self {
            checkpoint: Arc::new(checkpoint),
            identity: Arc::new(identity),
        })
    }

    /// Bounded reader supplied by the artifact owner; no path probing, model
    /// discovery, directory traversal or implicit checkpoint fallback.
    pub fn from_reader(reader: impl Read) -> Result<Self, CpuValueError> {
        let mut bytes = Vec::new();
        reader
            .take((CPU_VALUE_MAX_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|error| CpuValueError::Io(error.to_string()))?;
        if bytes.len() > CPU_VALUE_MAX_BYTES {
            return Err(CpuValueError::CheckpointSize);
        }
        let checkpoint: CpuFloatCheckpoint = serde_json::from_slice(&bytes)
            .map_err(|error| CpuValueError::Json(error.to_string()))?;
        Self::new(checkpoint)
    }

    pub fn checkpoint(&self) -> &CpuFloatCheckpoint {
        &self.checkpoint
    }

    pub fn forward(&self, accumulator: &CpuAccumulator, side: Color) -> Result<f32, CpuValueError> {
        if accumulator.namespace != self.identity {
            return Err(CpuValueError::StateMismatch);
        }
        let own = side as usize;
        let other = side.opposite() as usize;
        let mut hidden = [0.0f32; CPU_HIDDEN_WIDTH];
        for (row, value) in hidden.iter_mut().enumerate() {
            let weights =
                &self.checkpoint.hidden_weights[row * INPUT_WIDTH..(row + 1) * INPUT_WIDTH];
            let mut sum = self.checkpoint.hidden_bias[row];
            for (column, weight) in weights.iter().enumerate() {
                let perspective = if column < CPU_ACCUMULATOR_WIDTH {
                    own
                } else {
                    other
                };
                let feature =
                    accumulator.values[perspective][column % CPU_ACCUMULATOR_WIDTH].clamp(0.0, 1.0);
                sum += weight * feature;
            }
            *value = sum.max(0.0);
        }
        let mut output = self.checkpoint.output_bias;
        for (value, weight) in hidden.iter().zip(&self.checkpoint.output_weights) {
            output += value * weight;
        }
        output *= self.checkpoint.output_scale;
        if output.is_finite() {
            Ok(output)
        } else {
            Err(CpuValueError::NonFiniteForward)
        }
    }

    fn add_exact_feature(
        &self,
        values: &mut [Signed192; CPU_ACCUMULATOR_WIDTH],
        feature: usize,
    ) -> Result<(), CpuValueError> {
        let row = &self.checkpoint.feature_weights
            [feature * CPU_ACCUMULATOR_WIDTH..(feature + 1) * CPU_ACCUMULATOR_WIDTH];
        for (value, weight) in values.iter_mut().zip(row) {
            *value = value.checked_add(Signed192::from_f32(*weight)?)?;
        }
        Ok(())
    }

    fn changed_sum(
        &self,
        value: Signed192,
        column: usize,
        changes: &[FeatureChange],
        direction: i8,
    ) -> Result<Signed192, CpuValueError> {
        let mut next = value;
        for change in changes {
            let weight = self.checkpoint.feature_weights
                [usize::from(change.feature) * CPU_ACCUMULATOR_WIDTH + column];
            let exact = Signed192::from_f32(weight)?;
            next = next.checked_add(if change.sign * direction < 0 {
                exact.negated()
            } else {
                exact
            })?;
        }
        Ok(next)
    }

    fn apply_changes(
        &self,
        exact: &mut ExactValues,
        changes: &FeatureChanges,
        direction: i8,
    ) -> Result<(), CpuValueError> {
        // Preflight every scalar before publishing any new state. The second
        // pass uses immutable weights and independent columns, so a failure
        // cannot leave a partially applied delta; no per-node matrix clone.
        for perspective in [Color::White, Color::Black] {
            for (column, value) in exact[perspective as usize].iter().enumerate() {
                self.changed_sum(*value, column, changes.for_side(perspective), direction)?;
            }
        }
        for perspective in [Color::White, Color::Black] {
            for (column, value) in exact[perspective as usize].iter_mut().enumerate() {
                *value =
                    self.changed_sum(*value, column, changes.for_side(perspective), direction)?;
            }
        }
        Ok(())
    }
}

impl CpuValueEvaluator for CpuFloatValue {
    fn identity(&self) -> &CpuValueIdentity {
        &self.identity
    }

    fn provenance(&self) -> &'static str {
        CPU_VALUE_ARCHITECTURE
    }

    fn initialize(&self, position: &Position) -> Result<CpuAccumulator, CpuValueError> {
        let snapshot = position.snapshot();
        let mut exact = Box::new([[Signed192::default(); CPU_ACCUMULATOR_WIDTH]; 2]);
        for perspective in [Color::White, Color::Black] {
            for (value, bias) in exact[perspective as usize]
                .iter_mut()
                .zip(&self.checkpoint.accumulator_bias)
            {
                *value = Signed192::from_f32(*bias)?;
            }
            for index in 0..64 {
                let square = Square::new(index).expect("bounded board index");
                if let Some(piece) = snapshot.piece_at(square) {
                    self.add_exact_feature(
                        &mut exact[perspective as usize],
                        piece_feature(piece, square, perspective),
                    )?;
                }
            }
            for feature in metadata_features(&snapshot, perspective) {
                self.add_exact_feature(&mut exact[perspective as usize], feature)?;
            }
        }
        let values = rounded_values(&exact);
        Ok(CpuAccumulator {
            values,
            identity: position.position_identity(),
            namespace: Arc::clone(&self.identity),
            exact: Some(exact),
            nnuec: None,
        })
    }

    fn score(
        &self,
        accumulator: &CpuAccumulator,
        position: &Position,
    ) -> Result<i32, CpuValueError> {
        if accumulator.identity != position.position_identity()
            || accumulator.namespace != self.identity
        {
            return Err(CpuValueError::StateMismatch);
        }
        let raw = self.forward(accumulator, position.side_to_move())?;
        // Finite learned/frontier values cannot enter the Rules mate namespace.
        Ok(raw.clamp(-20_000.0, 20_000.0).round() as i32)
    }

    fn apply_delta(
        &self,
        accumulator: &mut CpuAccumulator,
        delta: &RuleMoveDelta,
    ) -> Result<CpuAccumulatorUndo, CpuValueError> {
        if accumulator.identity != delta.before.position_identity()
            || accumulator.namespace != self.identity
        {
            return Err(CpuValueError::StateMismatch);
        }
        if delta.removals.len() > 4 || delta.additions.len() > 4 {
            return Err(CpuValueError::InvalidCheckpoint(
                "Rules piece delta bound exceeded",
            ));
        }
        let mut changes = FeatureChanges::default();
        for perspective in [Color::White, Color::Black] {
            for removed in &delta.removals {
                changes.push(
                    perspective,
                    piece_feature(removed.piece, removed.square, perspective),
                    -1,
                )?;
            }
            for added in &delta.additions {
                changes.push(
                    perspective,
                    piece_feature(added.piece, added.square, perspective),
                    1,
                )?;
            }
            let before = metadata_features(&delta.before, perspective);
            let after = metadata_features(&delta.after, perspective);
            for &feature in before.as_slice() {
                if !after.as_slice().contains(&feature) {
                    changes.push(perspective, feature, -1)?;
                }
            }
            for &feature in after.as_slice() {
                if !before.as_slice().contains(&feature) {
                    changes.push(perspective, feature, 1)?;
                }
            }
        }
        let exact = accumulator
            .exact
            .as_mut()
            .ok_or(CpuValueError::StateMismatch)?;
        self.apply_changes(exact, &changes, 1)?;
        let undo = CpuAccumulatorUndo {
            changes: Some(changes),
            before: accumulator.identity.clone(),
            after: delta.after.position_identity(),
            namespace: Arc::clone(&self.identity),
            nnuec: None,
        };
        accumulator.values = rounded_values(exact);
        accumulator.identity = undo.after.clone();
        Ok(undo)
    }

    fn restore(
        &self,
        accumulator: &mut CpuAccumulator,
        undo: CpuAccumulatorUndo,
        position: &Position,
    ) -> Result<(), CpuValueError> {
        if accumulator.identity != undo.after
            || position.position_identity() != undo.before
            || accumulator.namespace != self.identity
            || undo.namespace != self.identity
        {
            return Err(CpuValueError::StateMismatch);
        }
        let changes = undo.changes.ok_or(CpuValueError::StateMismatch)?;
        let exact = accumulator
            .exact
            .as_mut()
            .ok_or(CpuValueError::StateMismatch)?;
        self.apply_changes(exact, &changes, -1)?;
        accumulator.values = rounded_values(exact);
        accumulator.identity = undo.before;
        Ok(())
    }
}

fn rounded_values(exact: &ExactValues) -> [[f32; CPU_ACCUMULATOR_WIDTH]; 2] {
    let mut values = [[0.0; CPU_ACCUMULATOR_WIDTH]; 2];
    for (values, sums) in values.iter_mut().zip(exact) {
        for (value, sum) in values.iter_mut().zip(sums) {
            *value = sum.to_f32();
        }
    }
    values
}

fn piece_feature(piece: Piece, square: Square, perspective: Color) -> usize {
    let color = usize::from(piece.color != perspective);
    let index = if perspective == Color::White {
        square.index()
    } else {
        square.index() ^ 56
    };
    (color * 6 + piece.kind as usize) * 64 + usize::from(index)
}

/// 32 metadata rows: side-to-move2, own/opponent castling4, EP-none/files9,
/// halfmove buckets17 (0..9,10..19,...,>=160). Rules uses exact counters/history;
/// this bucketing is declared model encoding and never changes draw authority.
struct MetadataFeatures {
    values: [usize; 7],
    len: usize,
}

impl MetadataFeatures {
    fn as_slice(&self) -> &[usize] {
        &self.values[..self.len]
    }
    fn push(&mut self, feature: usize) {
        self.values[self.len] = feature;
        self.len += 1;
    }
}

impl IntoIterator for MetadataFeatures {
    type Item = usize;
    type IntoIter = std::iter::Take<std::array::IntoIter<usize, 7>>;
    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter().take(self.len)
    }
}

fn metadata_features(snapshot: &PositionSnapshot, perspective: Color) -> MetadataFeatures {
    let mut features = MetadataFeatures {
        values: [0; 7],
        len: 0,
    };
    features.push(PIECE_FEATURES + usize::from(snapshot.side_to_move() != perspective));
    let rights = snapshot.castling_rights();
    let bits = if perspective == Color::White {
        [0u8, 1, 2, 3]
    } else {
        [2u8, 3, 0, 1]
    };
    for (index, bit) in bits.into_iter().enumerate() {
        if rights & (1 << bit) != 0 {
            features.push(PIECE_FEATURES + 2 + index);
        }
    }
    features.push(
        PIECE_FEATURES
            + 6
            + snapshot
                .en_passant_target()
                .map_or(0, |square| usize::from(square.file()) + 1),
    );
    features.push(PIECE_FEATURES + 15 + (snapshot.halfmove_clock() / 10).min(16) as usize);
    features
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{CpuConfig, CpuEngine, CpuLimits};
    use rz_position::BoardMove;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn value_namespace_admission_is_bounded_and_keeps_complete_training_identity() {
        let bootstrap = BootstrapCpuValue::default().identity().clone();
        bootstrap.validate().unwrap();
        let mut value = bootstrap.clone();
        value.semantics = " ".into();
        assert!(value.validate().is_err());
        value.semantics = "x".repeat(257);
        assert!(value.validate().is_err());
        value.semantics = String::with_capacity(4096);
        value.semantics.push_str("small-but-overallocated");
        assert!(value.validate().is_err());
        value = bootstrap.clone();
        value.weights_sha256 = Some("0".repeat(64));
        assert!(value.validate().is_err());
        value.training = CpuTrainingState::Untrained;
        value.validate().unwrap();
        let first = value.clone();
        value.weights_sha256 = Some("1".repeat(64));
        assert_ne!(first.cmp(&value), std::cmp::Ordering::Equal);
        value.training = CpuTrainingState::Learned {
            run_id: "declared-only".into(),
            steps: 1,
            dataset_sha256: "2".repeat(64),
        };
        value.validate().unwrap();
        assert_ne!(first.cmp(&value), std::cmp::Ordering::Equal);
        value.weights_sha256 = None;
        assert!(value.validate().is_err());
        value.weights_sha256 = Some("A".repeat(64));
        assert!(value.validate().is_err());
        let mut overallocated_hash = String::with_capacity(4096);
        overallocated_hash.push_str(&"0".repeat(64));
        value.weights_sha256 = Some(overallocated_hash);
        assert!(value.validate().is_err());
        value.weights_sha256 = Some("0".repeat(64));
        value.training = CpuTrainingState::Learned {
            run_id: " ".into(),
            steps: 0,
            dataset_sha256: "2".repeat(64),
        };
        assert!(value.validate().is_err());
    }

    fn model() -> CpuFloatValue {
        let mut checkpoint = CpuFloatCheckpoint::zeros_untrained();
        // Deterministic test weights, explicitly untrained; no learning run.
        for (index, value) in checkpoint.feature_weights.iter_mut().enumerate() {
            *value = ((index * 13 % 97) as f32 - 48.0) * 0.0001;
        }
        checkpoint.accumulator_bias.fill(0.4);
        for (index, value) in checkpoint.hidden_weights.iter_mut().enumerate() {
            *value = ((index * 7 % 31) as f32 - 15.0) * 0.001;
        }
        checkpoint.hidden_bias.fill(0.03);
        checkpoint.output_weights.fill(0.02);
        CpuFloatValue::new(checkpoint).unwrap()
    }

    // Independent binary-digit oracle: positive/negative per-bit counts are
    // normalized separately, then subtracted. It never uses Signed192 limbs,
    // production carry/sign code, delta updates or f64 approximate summation.
    fn reference_sum(values: impl IntoIterator<Item = f32>) -> f32 {
        let mut positive = [0u32; 192];
        let mut negative = [0u32; 192];
        for value in values {
            let bits = value.to_bits();
            let exponent = ((bits >> 23) & 255) as usize;
            assert!(exponent != 255 && value.abs() <= MAX_PARAMETER_MAGNITUDE);
            let mantissa = (bits & 0x7f_ffff) | if exponent == 0 { 0 } else { 1 << 23 };
            let offset = exponent.saturating_sub(1);
            let counts = if bits >> 31 == 0 {
                &mut positive
            } else {
                &mut negative
            };
            for bit in 0..24 {
                counts[offset + bit] += (mantissa >> bit) & 1;
            }
        }
        for counts in [&mut positive, &mut negative] {
            for bit in 0..191 {
                counts[bit + 1] += counts[bit] / 2;
                counts[bit] %= 2;
            }
            assert_eq!(counts[191], 0);
        }
        let ordering = positive.iter().rev().cmp(negative.iter().rev());
        let (large, small, sign) = match ordering {
            std::cmp::Ordering::Equal => return 0.0,
            std::cmp::Ordering::Greater => (positive, negative, 0),
            std::cmp::Ordering::Less => (negative, positive, 1 << 31),
        };
        let mut digits = [0u32; 192];
        let mut borrow = 0i32;
        for bit in 0..192 {
            let value = large[bit] as i32 - small[bit] as i32 - borrow;
            if value < 0 {
                digits[bit] = (value + 2) as u32;
                borrow = 1;
            } else {
                digits[bit] = value as u32;
                borrow = 0;
            }
        }
        assert_eq!(borrow, 0);
        let mut highest = digits.iter().rposition(|digit| *digit != 0).unwrap();
        if highest <= 22 {
            let fraction = (0..=highest).fold(0, |sum, bit| sum | (digits[bit] << bit));
            return f32::from_bits(sign | fraction);
        }
        let mut mantissa = (0..24).fold(0, |sum, bit| sum | (digits[highest - bit] << (23 - bit)));
        if highest > 23 {
            let guard = highest - 24;
            if digits[guard] != 0
                && (mantissa & 1 != 0 || digits[..guard].iter().any(|digit| *digit != 0))
            {
                mantissa += 1;
                if mantissa == 1 << 24 {
                    mantissa >>= 1;
                    highest += 1;
                }
            }
        }
        f32::from_bits(sign | (((highest - 22) as u32) << 23) | (mantissa & 0x7f_ffff))
    }

    fn reference(model: &CpuFloatValue, position: &Position) -> [[f32; CPU_ACCUMULATOR_WIDTH]; 2] {
        let snapshot = position.snapshot();
        let mut output = [[0.0; CPU_ACCUMULATOR_WIDTH]; 2];
        for perspective in [Color::White, Color::Black] {
            let mut features = Vec::new();
            for square_index in 0..64 {
                let square = Square::new(square_index).unwrap();
                if let Some(piece) = snapshot.piece_at(square) {
                    features.push(piece_feature(piece, square, perspective));
                }
            }
            for feature in metadata_features(&snapshot, perspective) {
                features.push(feature);
            }
            for (column, value) in output[perspective as usize].iter_mut().enumerate() {
                *value = reference_sum(
                    std::iter::once(model.checkpoint.accumulator_bias[column]).chain(
                        features.iter().map(|row| {
                            model.checkpoint.feature_weights[row * CPU_ACCUMULATOR_WIDTH + column]
                        }),
                    ),
                );
            }
        }
        output
    }

    fn assert_close(
        actual: &[[f32; CPU_ACCUMULATOR_WIDTH]; 2],
        expected: &[[f32; CPU_ACCUMULATOR_WIDTH]; 2],
    ) {
        for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{actual} != {expected}"
            );
        }
    }

    #[test]
    fn exact_integer_roundtrip_subnormals_ties_and_overflow_guards() {
        let smallest = f32::from_bits(1);
        for values in [
            vec![1.0, 2.0f32.powi(-24)],
            vec![f32::from_bits(1.0f32.to_bits() + 1), 2.0f32.powi(-24)],
            vec![1.0, 2.0f32.powi(-24), smallest],
            vec![-1.0, -2.0f32.powi(-24), -smallest],
            vec![f32::from_bits(0x7f_ffff), smallest],
            vec![f32::MIN_POSITIVE, -f32::from_bits(0x7f_ffff)],
            vec![1_000_000.0, 0.03, -1_000_000.0, smallest],
            vec![1_000_000.0, -1_000_000.0, smallest],
        ] {
            let expected = reference_sum(values.iter().copied()).to_bits();
            for offset in 0..values.len() {
                let mut exact = Signed192::default();
                for index in 0..values.len() {
                    exact = exact
                        .checked_add(
                            Signed192::from_f32(values[(index + offset) % values.len()]).unwrap(),
                        )
                        .unwrap();
                }
                assert_eq!(exact.to_f32().to_bits(), expected);
            }
        }
        let mut state = 0x49a1_c347u32;
        for _ in 0..10_000 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let value = f32::from_bits(state);
            if value.is_finite() && value.abs() <= MAX_PARAMETER_MAGNITUDE {
                assert_eq!(
                    Signed192::from_f32(value).unwrap().to_f32().to_bits(),
                    if value == 0.0 { 0 } else { value.to_bits() }
                );
            }
        }
        let mut bounded = Signed192::default();
        for _ in 0..(CPU_FEATURE_COUNT + 1) {
            bounded = bounded
                .checked_add(Signed192::from_f32(MAX_PARAMETER_MAGNITUDE).unwrap())
                .unwrap();
        }
        assert_eq!(
            bounded.to_f32(),
            ((CPU_FEATURE_COUNT + 1) as f64 * f64::from(MAX_PARAMETER_MAGNITUDE)) as f32
        );
        let maximum = Signed192([u64::MAX, u64::MAX, i64::MAX as u64]);
        assert!(matches!(
            maximum.checked_add(Signed192([1, 0, 0])),
            Err(CpuValueError::AccumulatorOverflow)
        ));
        let minimum = Signed192([0, 0, 1 << 63]);
        assert!(matches!(
            minimum.checked_add(Signed192([u64::MAX; 3])),
            Err(CpuValueError::AccumulatorOverflow)
        ));
    }

    #[test]
    fn large_cancellation_and_subnormal_delta_are_bit_identical_to_fresh() {
        let mut checkpoint = CpuFloatCheckpoint::zeros_untrained();
        for (square, kind, weight) in [
            (0, rz_position::PieceKind::Rook, 1_000_000.0),
            (1, rz_position::PieceKind::Knight, 0.03),
            (6, rz_position::PieceKind::Knight, -1_000_000.0),
            (18, rz_position::PieceKind::Knight, 0.04),
        ] {
            checkpoint.feature_weights[(kind as usize * 64 + square) * CPU_ACCUMULATOR_WIDTH] =
                weight;
        }
        checkpoint.feature_weights
            [(rz_position::PieceKind::Knight as usize * 64 + 1) * CPU_ACCUMULATOR_WIDTH + 1] =
            f32::from_bits(1);
        checkpoint.feature_weights
            [(rz_position::PieceKind::Knight as usize * 64 + 18) * CPU_ACCUMULATOR_WIDTH + 1] =
            f32::from_bits(2);
        checkpoint.hidden_weights[0] = 1.0;
        checkpoint.hidden_weights[CPU_ACCUMULATOR_WIDTH] = 1.0;
        checkpoint.output_weights[0] = 1.0;
        let model = CpuFloatValue::new(checkpoint).unwrap();
        let mut position = Position::startpos();
        let mut incremental = model.initialize(&position).unwrap();
        let before = incremental.values;
        assert_eq!(
            before[Color::White as usize][0].to_bits(),
            0.03f32.to_bits()
        );
        let rules = position.make_uci("b1c3").unwrap();
        let undo = model.apply_delta(&mut incremental, rules.delta()).unwrap();
        let fresh = model.initialize(&position).unwrap();
        assert_close(incremental.values(), fresh.values());
        assert_close(incremental.values(), &reference(&model, &position));
        assert_eq!(
            incremental.values[Color::White as usize][0].to_bits(),
            0.04f32.to_bits()
        );
        assert_eq!(incremental.values[Color::White as usize][1].to_bits(), 2);
        assert_eq!(
            model
                .forward(&incremental, position.side_to_move())
                .unwrap()
                .to_bits(),
            model
                .forward(&fresh, position.side_to_move())
                .unwrap()
                .to_bits()
        );
        position.unmake(rules).unwrap();
        model.restore(&mut incremental, undo, &position).unwrap();
        assert_close(incremental.values(), &before);
    }

    #[test]
    fn fp32_forward_matches_independent_dense_f64_reference() {
        let model = model();
        let position = Position::startpos();
        let dense = reference(&model, &position);
        let accumulator = model.initialize(&position).unwrap();
        for side in [Color::White, Color::Black] {
            let input: Vec<f64> = dense[side as usize]
                .iter()
                .chain(&dense[side.opposite() as usize])
                .map(|value| f64::from(value.clamp(0.0, 1.0)))
                .collect();
            let hidden: Vec<f64> = model
                .checkpoint
                .hidden_weights
                .chunks_exact(INPUT_WIDTH)
                .zip(&model.checkpoint.hidden_bias)
                .map(|(row, bias)| {
                    (row.iter()
                        .zip(&input)
                        .map(|(weight, value)| f64::from(*weight) * value)
                        .sum::<f64>()
                        + f64::from(*bias))
                    .max(0.0)
                })
                .collect();
            let expected = (hidden
                .iter()
                .zip(&model.checkpoint.output_weights)
                .map(|(value, weight)| value * f64::from(*weight))
                .sum::<f64>()
                + f64::from(model.checkpoint.output_bias))
                * f64::from(model.checkpoint.output_scale);
            let actual = f64::from(model.forward(&accumulator, side).unwrap());
            assert!((actual - expected).abs() <= 1e-4 + 1e-3 * expected.abs());
        }
    }

    #[test]
    fn incremental_special_moves_match_dense_reference_and_undo_is_bit_exact() {
        let model = model();
        let fixtures = [
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
            ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", "e5d6"),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8r"),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8b"),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n"),
            ("4k3/8/8/8/8/3p4/4P3/4K3 w - - 9 8", "e2d3"),
            ("4k3/8/8/8/8/8/P7/4K3 w - - 9 8", "e1d1"),
        ];
        for (fen, mv) in fixtures {
            let mut position = Position::from_fen(fen).unwrap();
            let mut accumulator = model.initialize(&position).unwrap();
            let before = accumulator.values;
            let rules_undo = position
                .make_move(BoardMove::from_uci(mv).unwrap())
                .unwrap();
            let value_undo = model
                .apply_delta(&mut accumulator, rules_undo.delta())
                .unwrap();
            assert_close(accumulator.values(), &reference(&model, &position));
            assert_close(
                accumulator.values(),
                model.initialize(&position).unwrap().values(),
            );
            position.unmake(rules_undo).unwrap();
            model
                .restore(&mut accumulator, value_undo, &position)
                .unwrap();
            assert_eq!(
                accumulator.values.map(|row| row.map(f32::to_bits)),
                before.map(|row| row.map(f32::to_bits))
            );
        }
    }

    #[test]
    fn long_checked_line_restores_every_accumulator_frame() {
        let model = model();
        let mut position = Position::startpos();
        let mut accumulator = model.initialize(&position).unwrap();
        let before = accumulator.values;
        let mut frames = Vec::new();
        for mv in [
            "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7", "f1e1",
            "b7b5", "a4b3", "d7d6", "c2c3", "e8g8",
        ] {
            let rules = position.make_uci(mv).unwrap();
            let value = model.apply_delta(&mut accumulator, rules.delta()).unwrap();
            assert_close(accumulator.values(), &reference(&model, &position));
            frames.push((rules, value));
        }
        while let Some((rules, value)) = frames.pop() {
            position.unmake(rules).unwrap();
            model.restore(&mut accumulator, value, &position).unwrap();
            assert_close(accumulator.values(), &reference(&model, &position));
        }
        assert_eq!(
            accumulator.values.map(|row| row.map(f32::to_bits)),
            before.map(|row| row.map(f32::to_bits))
        );
    }

    #[test]
    fn checkpoint_roundtrip_and_provenance_are_bound_to_parameter_bytes() {
        let model = model();
        let json = serde_json::to_vec(model.checkpoint()).unwrap();
        let loaded = CpuFloatValue::from_reader(json.as_slice()).unwrap();
        assert_eq!(loaded.identity(), model.identity());
        assert_eq!(loaded.identity().training, CpuTrainingState::Untrained);
        let mut changed = loaded.checkpoint().clone();
        changed.feature_weights[0] += 0.001;
        let changed = CpuFloatValue::new(changed).unwrap();
        assert_ne!(
            changed.identity().weights_sha256,
            model.identity().weights_sha256
        );
        assert_ne!(BootstrapCpuValue::default().identity(), model.identity());
    }

    #[test]
    fn invalid_shapes_finite_values_and_learning_claims_are_rejected() {
        let mut checkpoint = CpuFloatCheckpoint::zeros_untrained();
        checkpoint.output_weights.pop();
        assert!(CpuFloatValue::new(checkpoint).is_err());
        let mut checkpoint = CpuFloatCheckpoint::zeros_untrained();
        checkpoint.hidden_weights[0] = f32::NAN;
        assert!(CpuFloatValue::new(checkpoint).is_err());
        let mut checkpoint = CpuFloatCheckpoint::zeros_untrained();
        checkpoint.training = CpuTrainingState::Learned {
            run_id: "invalid-evidence".into(),
            steps: 0,
            dataset_sha256: "0".repeat(64),
        };
        assert!(CpuFloatValue::new(checkpoint).is_err());
        assert!(matches!(
            CpuFloatValue::from_reader(
                std::io::repeat(b' ').take((CPU_VALUE_MAX_BYTES + 1) as u64)
            ),
            Err(CpuValueError::CheckpointSize)
        ));
    }

    #[test]
    fn foreign_delta_and_wrong_restore_are_rejected_without_mutation() {
        let model = model();
        let position = Position::startpos();
        let mut accumulator = model.initialize(&position).unwrap();
        let before = accumulator.values;
        let mut different = Position::from_fen("4k3/8/8/8/8/8/P7/4K3 w - - 0 1").unwrap();
        let foreign = different.make_uci("a2a3").unwrap();
        assert!(matches!(
            model.apply_delta(&mut accumulator, foreign.delta()),
            Err(CpuValueError::StateMismatch)
        ));
        assert_eq!(accumulator.values, before);
        let mut changed = model.checkpoint().clone();
        changed.output_bias += 1.0;
        let other_model = CpuFloatValue::new(changed).unwrap();
        assert!(matches!(
            other_model.score(&accumulator, &position),
            Err(CpuValueError::StateMismatch)
        ));
        assert!(matches!(
            other_model.forward(&accumulator, position.side_to_move()),
            Err(CpuValueError::StateMismatch)
        ));
    }

    #[test]
    fn untrained_float_model_runs_inside_the_same_cpu_search() {
        let model = Arc::new(model());
        let identity = model.identity().clone();
        let mut engine = CpuEngine::with_evaluator(CpuConfig::default(), model).unwrap();
        let position = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap();
        let report = engine
            .analyze(
                &position,
                CpuLimits {
                    max_depth: 1,
                    max_nodes: 10_000,
                    deadline: None,
                },
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(report.value_identity, identity);
        assert_eq!(report.score, crate::cpu::CPU_MATE_SCORE - 1);
    }
}
