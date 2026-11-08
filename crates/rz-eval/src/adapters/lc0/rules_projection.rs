//! Concrete A → C projection. Rules retains legality/history/terminal authority.
//! This consumes owned snapshots and reads their exact raw EP and known history;
//! missing-prefix history remains unknown, while model padding stays an encoder
//! option. The original C model-input codec and ordered action map are reused.

use crate::contracts::{input_key, ordered_policy_indices, MaiaBinding, PreparedRequest};
use crate::model_adapter::{ModelAdapter, ModelCapabilities, PhysicalFailure};
use rz_contracts::{
    ContractError, Digest, ErrorCode, EvalInputKey, EvalRequest, ModelDescriptor, Move, Stage,
};
use rz_contracts::{EvalOutput, ExecutionId, PrecisionProfile};
use rz_encoding::classical::{self, EncodedInput, Frame, HistoryFill, Input, HISTORY_FRAMES};
use rz_position::{contracts::RulesState, Color, Piece, PieceKind, PositionSnapshot, Square};
use std::sync::Arc;

impl ModelAdapter for Lc0ModelAdapter {
    type PreparedInput = PreparedRequest<RulesState>;
    type PreparedBatch = crate::contracts::PreparedBatch<RulesState>;
    type RawOutput = super::RawOutput;
    fn model(&self) -> &Arc<ModelDescriptor> {
        self.model()
    }
    fn backend(&self) -> Digest {
        self.backend()
    }
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            input: "lc0-classical112",
            policy_head: "lc0-1858",
            host_bytes_per_item: crate::contracts::HOST_BYTES_PER_ITEM,
            mixed_legal_batching: true,
        }
    }
    fn input_key(&self, state: &RulesState, legal: &[Move]) -> Result<EvalInputKey, ContractError> {
        self.input_key(state, legal)
    }
    fn prepare(
        &self,
        request: Arc<EvalRequest<RulesState>>,
    ) -> Result<Self::PreparedInput, ContractError> {
        self.prepare(request)
    }
    fn batch(
        &self,
        execution: ExecutionId,
        items: Vec<Self::PreparedInput>,
    ) -> Result<Self::PreparedBatch, ContractError> {
        crate::contracts::PreparedBatch::new(execution, items)
    }
    fn decode(
        &self,
        input: &Self::PreparedInput,
        raw: &Self::RawOutput,
        execution: ExecutionId,
    ) -> Result<EvalOutput, PhysicalFailure> {
        input.physical_output(raw, execution)
    }
    fn validate_runtime(&self, max_batch: usize) -> Result<(), ContractError> {
        if self.model().full_steps() != 1
            || self.model().max_batch_items() != max_batch
            || !self.model().supports(PrecisionProfile::Fp32)
        {
            return Err(failed(
                ErrorCode::UnsupportedContract,
                "LC0 baseline requires fresh FP32 inference",
            ));
        }
        Ok(())
    }
}

/// Legacy name retained without an allocation or wrapper.
pub type ClassicalProjection = Lc0ModelAdapter;

#[derive(Clone, Debug)]
pub struct Lc0ModelAdapter {
    binding: MaiaBinding,
    #[cfg(feature = "experimental-prepared-input")]
    prepared: Arc<std::sync::Mutex<Option<Arc<PreparedRulesInput>>>>,
    #[cfg(feature = "experimental-raw-cache")]
    raw_cache: crate::raw_cache::RawCache,
}

impl Lc0ModelAdapter {
    pub fn new(binding: MaiaBinding) -> Self {
        Self {
            binding,
            #[cfg(feature = "experimental-prepared-input")]
            prepared: Arc::default(),
            #[cfg(feature = "experimental-raw-cache")]
            raw_cache: crate::raw_cache::RawCache::disabled(),
        }
    }
    pub fn model(&self) -> &Arc<ModelDescriptor> {
        self.binding.model()
    }
    pub fn backend(&self) -> Digest {
        self.binding.backend()
    }
    pub fn binding(&self) -> &MaiaBinding {
        &self.binding
    }

    #[cfg(feature = "experimental-raw-cache")]
    pub fn configure_raw_cache(
        &self,
        limits: crate::raw_cache::RawCacheLimits,
    ) -> Result<(), ContractError> {
        self.raw_cache.configure(limits)
    }
    #[cfg(feature = "experimental-raw-cache")]
    pub fn clear_raw_cache(&self) -> Result<(), ContractError> {
        self.raw_cache.clear()
    }
    #[cfg(feature = "experimental-raw-cache")]
    pub fn raw_cache_stats(&self) -> Result<crate::raw_cache::RawCacheStats, ContractError> {
        self.raw_cache.stats()
    }
    #[cfg(feature = "experimental-raw-cache")]
    pub fn raw_cache_provider(&self) -> crate::raw_cache::RawCacheProvider {
        crate::raw_cache::RawCacheProvider {
            projection: self.clone(),
            cache: self.raw_cache.clone(),
        }
    }

    pub fn project(&self, state: &RulesState) -> Result<RulesProjection, ContractError> {
        let snapshot = state.snapshot();
        let mut frames = ProjectionFrames::try_new()?;
        #[cfg(feature = "experimental-history-frames")]
        for history in snapshot
            .recent_history_frames::<HISTORY_FRAMES>()
            .iter()
            .flatten()
        {
            frames.push(Frame {
                pieces: read_pieces(history.piece_bitboards(), |square| history.piece_at(square))?,
                repeated: history.repeated(),
                en_passant_target: history.en_passant_target().map(Square::index),
            })?;
        }
        #[cfg(not(feature = "experimental-history-frames"))]
        for history in snapshot.known_history().take(HISTORY_FRAMES) {
            let pieces = read_pieces(history.piece_bitboards(), |square| history.piece_at(square))?;
            let identity = history.repetition_identity();
            let repeated = history
                .known_history()
                .skip(1)
                .any(|prior| prior.repetition_identity() == identity);
            frames.push(Frame {
                pieces,
                repeated,
                en_passant_target: history.en_passant_target().map(Square::index),
            })?;
        }
        let rights = snapshot.castling_rights();
        Ok(RulesProjection {
            frames,
            black_to_move: snapshot.side_to_move() == Color::Black,
            castling: [
                rights & 2 != 0,
                rights & 1 != 0,
                rights & 8 != 0,
                rights & 4 != 0,
            ],
            halfmove_clock: snapshot.halfmove_clock(),
            history_fill: self.binding.history_fill(),
        })
    }

    pub fn encode(&self, state: &RulesState) -> Result<EncodedInput, ContractError> {
        let projection = self.project(state)?;
        classical::encode(projection.input()).map_err(|_| {
            failed(
                ErrorCode::InvalidInput,
                "Rules projection cannot be encoded by this profile",
            )
        })
    }

    /// Experimental LC0 history semantics. This does not alter the existing
    /// ModelAdapter, v1 binding, cache keys, or UCI selection. A consumer must
    /// explicitly select and register the v2 semantics before using it.
    pub fn project_lc0_v2(&self, state: &RulesState) -> Result<RulesProjection, ContractError> {
        let snapshot = state.snapshot();
        let mut frames = ProjectionFrames::try_new()?;
        for history in snapshot.known_history().take(HISTORY_FRAMES) {
            let identity = history.repetition_identity();
            let ep = lc0_history_ep(&history);
            let repeated = history.known_history().skip(1).any(|prior| {
                prior.repetition_identity() == identity && lc0_history_ep(&prior) == ep
            });
            frames.push(Frame {
                pieces: read_pieces(history.piece_bitboards(), |square| history.piece_at(square))?,
                repeated,
                en_passant_target: ep.map(Square::index),
            })?;
        }
        let rights = snapshot.castling_rights();
        Ok(RulesProjection {
            frames,
            black_to_move: snapshot.side_to_move() == Color::Black,
            castling: [
                rights & 2 != 0,
                rights & 1 != 0,
                rights & 8 != 0,
                rights & 4 != 0,
            ],
            halfmove_clock: snapshot.halfmove_clock(),
            history_fill: self.binding.history_fill(),
        })
    }

    /// Encode v2 without issuing a v1 EvalInputKey or selecting it for play.
    pub fn encode_lc0_v2(&self, state: &RulesState) -> Result<EncodedInput, ContractError> {
        let projection = self.project_lc0_v2(state)?;
        classical::encode(projection.input()).map_err(|_| {
            failed(
                ErrorCode::InvalidInput,
                "LC0 v2 Rules projection cannot be encoded by this profile",
            )
        })
    }

    pub fn input_key(
        &self,
        state: &RulesState,
        ordered_legal: &[Move],
    ) -> Result<EvalInputKey, ContractError> {
        #[cfg(feature = "experimental-prepared-input")]
        {
            Ok(self.prepared_input(state, ordered_legal, None)?.key)
        }
        #[cfg(not(feature = "experimental-prepared-input"))]
        {
            let projection = self.project(state)?;
            // Keep the original dense-input identity codec. Legal order is a
            // separate attestation and is nevertheless checked for representability.
            ordered_policy_indices(projection.input(), ordered_legal)?;
            let encoded = classical::encode(projection.input()).map_err(|_| {
                failed(
                    ErrorCode::InvalidInput,
                    "Rules projection cannot be encoded by this profile",
                )
            })?;
            Ok(input_key(self.model().encoding().handle, &encoded))
        }
    }

    pub fn legal_indices(
        &self,
        state: &RulesState,
        ordered_legal: &[Move],
    ) -> Result<Vec<usize>, ContractError> {
        ordered_policy_indices(self.project(state)?.input(), ordered_legal)
    }

    pub fn prepare(
        &self,
        request: Arc<EvalRequest<RulesState>>,
    ) -> Result<PreparedRequest<RulesState>, ContractError> {
        #[cfg(feature = "experimental-prepared-input")]
        {
            let prepared = self.prepared_input(
                request.position().state(),
                request.legal().moves(),
                Some(&request),
            )?;
            self.attach_raw_cache(self.binding.prepare_rules(request, &prepared)?)
        }
        #[cfg(not(feature = "experimental-prepared-input"))]
        {
            let projection = self.project(request.position().state())?;
            self.attach_raw_cache(self.binding.prepare(request, projection.input())?)
        }
    }

    fn attach_raw_cache(
        &self,
        prepared: PreparedRequest<RulesState>,
    ) -> Result<PreparedRequest<RulesState>, ContractError> {
        #[cfg(feature = "experimental-raw-cache")]
        let prepared = {
            let mut prepared = prepared;
            if self.raw_cache.enabled() {
                prepared.raw_cache = Some(self.raw_cache.clone());
            }
            prepared
        };
        Ok(prepared)
    }

    #[cfg(feature = "experimental-prepared-input")]
    fn prepared_input(
        &self,
        state: &RulesState,
        legal: &[Move],
        request: Option<&EvalRequest<RulesState>>,
    ) -> Result<Arc<PreparedRulesInput>, ContractError> {
        let mut memo = self
            .prepared
            .lock()
            .map_err(|_| failed(ErrorCode::BackendFailure, "prepared input owner poisoned"))?;
        if let Some(prepared) = memo.as_ref() {
            if prepared.identity.matches(state.snapshot()) && prepared.legal.as_ref() == legal {
                if let Some(request) = request {
                    self.binding
                        .validate_preparation(request, prepared.projection.input())?;
                    check_prepared_key(prepared.key, request)?;
                }
                return Ok(Arc::clone(prepared));
            }
        }
        let projection = self.project(state)?;
        if let Some(request) = request {
            self.binding
                .validate_preparation(request, projection.input())?;
        }
        // Match the original error order: input_key maps moves first, while
        // prepare validates the actual input key before mapping legal indices.
        let indices = if request.is_none() {
            Some(ordered_policy_indices(projection.input(), legal)?)
        } else {
            None
        };
        let encoded = classical::encode(projection.input())
            .map_err(|_| failed(ErrorCode::InvalidInput, "invalid Rules input projection"))?;
        let key = input_key(self.model().encoding().handle, &encoded);
        if let Some(request) = request {
            check_prepared_key(key, request)?;
        }
        let indices = match indices {
            Some(indices) => indices,
            None => ordered_policy_indices(projection.input(), legal)?,
        };
        let prepared = Arc::new(PreparedRulesInput {
            identity: state.snapshot().weak_position_identity(),
            legal: legal.into(),
            projection,
            key,
            encoded: Arc::new(encoded),
            indices: indices.into(),
        });
        // One slot, at most 112*64 f32 values + HISTORY_FRAMES + POLICY_SIZE
        // bounded move/index arrays. The weak identity does not retain Rules.
        *memo = Some(Arc::clone(&prepared));
        Ok(prepared)
    }
}

/// Semantic identifier for the explicit experimental projection, not a v1
/// EncodingHandle or a claim that the complete v2 model path is registered.
pub fn lc0_v2_projection_manifest(fill: HistoryFill) -> Digest {
    Digest(crate::asset::sha256(
        format!(
            "rz-lc0-classical-projection-v2;lc0=fd71a2d921b689c5f479d3227c3806c8e272d9c5;ep=fen-root-or-pseudo-attacked-move;history={}",
            fill.reference_option()
        )
        .as_bytes(),
    ))
}

/// LC0 accepts the raw FEN-root EP square but records a played double push
/// only when a side-to-move pawn attacks the target geometrically. Unlike
/// the FIDE repetition identity, this deliberately does not test king safety.
fn lc0_history_ep(history: &PositionSnapshot) -> Option<Square> {
    let target = history.en_passant_target()?;
    if history.known_history_len() == 1 {
        return Some(target);
    }
    let (color, rank) = match history.side_to_move() {
        Color::White => (0, target.rank().checked_sub(1)?),
        Color::Black => (1, target.rank().checked_add(1)?),
    };
    if rank >= 8 {
        return None;
    }
    let pawns = history.piece_bitboards()[color * 6];
    let file = target.file();
    for candidate in [file.checked_sub(1), file.checked_add(1)]
        .into_iter()
        .flatten()
        .filter(|candidate| *candidate < 8)
    {
        let square = rank * 8 + candidate;
        if pawns & (1u64 << square) != 0 {
            return Some(target);
        }
    }
    None
}

#[cfg(feature = "experimental-prepared-input")]
fn check_prepared_key(
    key: EvalInputKey,
    request: &EvalRequest<RulesState>,
) -> Result<(), ContractError> {
    if key != request.context().input {
        return Err(failed(
            ErrorCode::IdentityMismatch,
            "actual tensor identity differs from request",
        ));
    }
    Ok(())
}

#[cfg(feature = "experimental-prepared-input")]
pub(crate) struct PreparedRulesInput {
    identity: rz_position::WeakPositionIdentity,
    legal: Arc<[Move]>,
    pub(crate) projection: RulesProjection,
    pub(crate) key: EvalInputKey,
    pub(crate) encoded: Arc<EncodedInput>,
    pub(crate) indices: Arc<[usize]>,
}

#[cfg(feature = "experimental-prepared-input")]
impl std::fmt::Debug for PreparedRulesInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedRulesInput")
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}

fn read_pieces(
    bitboards: &[u64; 12],
    piece_at: impl Fn(Square) -> Option<Piece>,
) -> Result<[[u64; 6]; 2], ContractError> {
    if cfg!(feature = "experimental-bitboards") {
        return Ok(std::array::from_fn(|color| {
            std::array::from_fn(|kind| bitboards[color * 6 + kind])
        }));
    }
    let mut pieces = [[0; 6]; 2];
    for index in 0..64 {
        if let Some(piece) = piece_at(Square::new(index)?) {
            let color = usize::from(piece.color == Color::Black);
            let kind = match piece.kind {
                PieceKind::Pawn => 0,
                PieceKind::Knight => 1,
                PieceKind::Bishop => 2,
                PieceKind::Rook => 3,
                PieceKind::Queen => 4,
                PieceKind::King => 5,
            };
            pieces[color][kind] |= 1u64 << index;
        }
    }
    Ok(pieces)
}

/// Model-only storage: the eight-frame bound never truncates Rules history.
/// The inline trial removes one allocation but increases the owner's inline
/// size; its actual stack/move/retention cost must be measured independently.
struct ProjectionFrames {
    #[cfg(not(feature = "experimental-projection-frames"))]
    frames: Vec<Frame>,
    #[cfg(feature = "experimental-projection-frames")]
    frames: [Frame; HISTORY_FRAMES],
    #[cfg(feature = "experimental-projection-frames")]
    len: usize,
}

impl ProjectionFrames {
    fn try_new() -> Result<Self, ContractError> {
        #[cfg(not(feature = "experimental-projection-frames"))]
        {
            let mut frames = Vec::new();
            frames.try_reserve_exact(HISTORY_FRAMES).map_err(|_| {
                failed(
                    ErrorCode::ResourceExhausted,
                    "Rules projection allocation failed",
                )
            })?;
            Ok(Self { frames })
        }
        #[cfg(feature = "experimental-projection-frames")]
        {
            Ok(Self {
                frames: [Frame {
                    pieces: [[0; 6]; 2],
                    repeated: false,
                    en_passant_target: None,
                }; HISTORY_FRAMES],
                len: 0,
            })
        }
    }

    fn push(&mut self, frame: Frame) -> Result<(), ContractError> {
        if self.as_slice().len() == HISTORY_FRAMES {
            return Err(failed(
                ErrorCode::ResourceExhausted,
                "Rules projection frame bound exceeded",
            ));
        }
        #[cfg(not(feature = "experimental-projection-frames"))]
        self.frames.push(frame);
        #[cfg(feature = "experimental-projection-frames")]
        {
            self.frames[self.len] = frame;
            self.len += 1;
        }
        Ok(())
    }

    fn as_slice(&self) -> &[Frame] {
        #[cfg(not(feature = "experimental-projection-frames"))]
        {
            &self.frames
        }
        #[cfg(feature = "experimental-projection-frames")]
        {
            &self.frames[..self.len]
        }
    }
}

pub struct RulesProjection {
    frames: ProjectionFrames,
    black_to_move: bool,
    castling: [bool; 4],
    halfmove_clock: u32,
    history_fill: HistoryFill,
}

impl RulesProjection {
    pub fn input(&self) -> Input<'_> {
        Input {
            history: self.frames.as_slice(),
            black_to_move: self.black_to_move,
            castling: self.castling,
            halfmove_clock: self.halfmove_clock,
            history_fill: self.history_fill,
        }
    }
}

fn failed(code: ErrorCode, detail: &'static str) -> ContractError {
    ContractError::new(code, Stage::Admission, detail)
}

#[cfg(test)]
mod storage_tests {
    use super::*;
    use crate::contracts::encoding_manifest;
    use rz_contracts::{EncodingHandle, ModelHandle, OwnerId, SlotGeneration};
    use rz_position::{contracts::ContractPosition, Position};

    fn projection(fill: HistoryFill) -> ClassicalProjection {
        ClassicalProjection::new(
            MaiaBinding::new(
                ModelHandle {
                    owner: OwnerId(71),
                    slot: 1,
                    generation: SlotGeneration(1),
                    manifest: Digest([72; 32]),
                },
                EncodingHandle {
                    owner: OwnerId(71),
                    slot: 2,
                    generation: SlotGeneration(1),
                    manifest: encoding_manifest(fill),
                },
                fill,
                Digest([73; 32]),
                1,
            )
            .unwrap(),
        )
    }

    #[test]
    fn bounded_storage_rejects_overflow_without_changing_frames() {
        let mut storage = ProjectionFrames::try_new().unwrap();
        assert!(storage.as_slice().is_empty());
        let frame = Frame {
            pieces: [[0; 6]; 2],
            repeated: true,
            en_passant_target: Some(20),
        };
        for _ in 0..HISTORY_FRAMES {
            storage.push(frame).unwrap();
        }
        assert_eq!(storage.as_slice(), &[frame; HISTORY_FRAMES]);
        assert_eq!(
            storage.push(frame).unwrap_err().code,
            ErrorCode::ResourceExhausted
        );
        assert_eq!(storage.as_slice(), &[frame; HISTORY_FRAMES]);
    }

    // Baseline diagnostic for LC0 v0.32.1 fd71a2d, compiled independently
    // with NO_PEXT: both special cases report zero repetitions after the
    // indicated legal moves, while the start-position control reports one.
    // The current Rules repetition identity intentionally remains FIDE-legal.
    #[test]
    fn lc0_ep_repetition_baseline_diagnostic() {
        let cases: &[(&str, &[&str], bool)] = &[
            (
                "k3r1n1/3p4/8/4P3/8/8/8/4K1N1 b - - 0 1",
                &["d7d5", "g1f3", "g8f6", "f3g1", "f6g8"],
                false,
            ),
            (
                "k3r1n1/8/8/3pB3/8/8/8/4K1N1 w - d6 0 2",
                &["g1f3", "g8f6", "f3g1", "f6g8"],
                false,
            ),
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                &["e2e4", "g8f6", "g1f3", "f6g8", "f3g1"],
                true,
            ),
        ];
        for fill in [HistoryFill::No, HistoryFill::RepeatOldest] {
            for (fen, moves, lc0_repeated) in cases {
                let mut position = Position::from_fen(fen).unwrap();
                position.apply_uci_moves(moves).unwrap();
                let frozen = ContractPosition::new(OwnerId(83), position)
                    .export()
                    .unwrap();
                let snapshot = frozen.rules().snapshot();
                let rules_repeated = snapshot
                    .known_history()
                    .skip(1)
                    .any(|prior| prior.repetition_identity() == snapshot.repetition_identity());
                let actual = projection(fill).project(frozen.rules()).unwrap();
                let projected_repeated = actual.input().history[0].repeated;
                let legacy_tensor = projection(fill).encode(frozen.rules()).unwrap();
                let v2 = projection(fill).project_lc0_v2(frozen.rules()).unwrap();
                let v2_tensor = projection(fill).encode_lc0_v2(frozen.rules()).unwrap();
                eprintln!(
                    "fill={fill:?} fen={fen} rules={rules_repeated} projection={projected_repeated} lc0={lc0_repeated}"
                );
                assert!(rules_repeated);
                assert!(projected_repeated);
                assert!(legacy_tensor.values()[12 * 64..13 * 64]
                    .iter()
                    .all(|value| *value == 1.0));
                assert_eq!(v2.input().history[0].repeated, *lc0_repeated);
                assert!(v2_tensor.values()[12 * 64..13 * 64]
                    .iter()
                    .all(|value| *value == f32::from(u8::from(*lc0_repeated))));
            }
        }
    }

    #[test]
    fn lc0_v2_distinguishes_fen_root_and_played_ep_without_changing_rules() {
        let root_fen = "k3r1n1/8/8/3pB3/8/8/8/4K1N1 w - d6 0 2";
        let root = ContractPosition::new(OwnerId(84), Position::from_fen(root_fen).unwrap())
            .export()
            .unwrap();
        assert_eq!(
            lc0_history_ep(root.rules().snapshot()).map(Square::index),
            Some(43)
        );
        for fill in [HistoryFill::No, HistoryFill::RepeatOldest] {
            let adapter = projection(fill);
            let input = adapter.encode_lc0_v2(root.rules()).unwrap();
            assert!(input.inferred_ep_predecessor);
            assert_eq!(input.known_frames, 1);
            assert_eq!(
                input.padded_frames,
                match fill {
                    HistoryFill::No => 1,
                    HistoryFill::RepeatOldest => HISTORY_FRAMES - 1,
                }
            );
            assert_ne!(lc0_v2_projection_manifest(fill), encoding_manifest(fill));
        }
        assert_ne!(
            lc0_v2_projection_manifest(HistoryFill::No),
            lc0_v2_projection_manifest(HistoryFill::RepeatOldest)
        );

        let mut no_attacker = Position::startpos();
        no_attacker.apply_uci_moves(&["e2e4"]).unwrap();
        let no_attacker = ContractPosition::new(OwnerId(85), no_attacker)
            .export()
            .unwrap();
        assert_eq!(
            no_attacker
                .rules()
                .snapshot()
                .en_passant_target()
                .map(Square::index),
            Some(20)
        );
        assert_eq!(lc0_history_ep(no_attacker.rules().snapshot()), None);

        let mut white_attacker =
            Position::from_fen("k3r1n1/3p4/8/4P3/8/8/8/4K1N1 b - - 0 1").unwrap();
        white_attacker.apply_uci_moves(&["d7d5"]).unwrap();
        let white_attacker = ContractPosition::new(OwnerId(86), white_attacker)
            .export()
            .unwrap();
        assert_eq!(
            lc0_history_ep(white_attacker.rules().snapshot()).map(Square::index),
            Some(43)
        );

        let mut black_attacker =
            Position::from_fen("k5n1/8/8/8/4p3/8/3P4/4K1N1 w - - 0 1").unwrap();
        black_attacker.apply_uci_moves(&["d2d4"]).unwrap();
        let black_attacker = ContractPosition::new(OwnerId(87), black_attacker)
            .export()
            .unwrap();
        assert_eq!(
            lc0_history_ep(black_attacker.rules().snapshot()).map(Square::index),
            Some(19)
        );
    }

    #[test]
    fn projection_storage_matches_full_history_reference_and_input_identity() {
        let mut traced = Position::startpos();
        traced.apply_uci_moves(&["e2e4", "e7e5", "g1f3"]).unwrap();
        let mut repeated = Position::startpos();
        repeated
            .apply_uci_moves(&[
                "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
            ])
            .unwrap();
        let positions = [
            Position::startpos(),
            traced,
            repeated,
            Position::from_fen("rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq e6 0 2")
                .unwrap(),
            Position::from_fen("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap(),
        ];
        for fill in [HistoryFill::No, HistoryFill::RepeatOldest] {
            let projection = projection(fill);
            for position in positions.iter().cloned() {
                let frozen = ContractPosition::new(OwnerId(80), position)
                    .export()
                    .unwrap();
                let snapshot = frozen.rules().snapshot();
                // Independent complete-prefix reference, including repetition
                // before the model's oldest retained frame.
                let reference: Vec<_> = snapshot
                    .known_history()
                    .take(HISTORY_FRAMES)
                    .map(|history| {
                        let identity = history.repetition_identity();
                        Frame {
                            pieces: std::array::from_fn(|color| {
                                std::array::from_fn(|kind| {
                                    history.piece_bitboards()[color * 6 + kind]
                                })
                            }),
                            repeated: history
                                .known_history()
                                .skip(1)
                                .any(|prior| prior.repetition_identity() == identity),
                            en_passant_target: history.en_passant_target().map(Square::index),
                        }
                    })
                    .collect();
                let actual = projection.project(frozen.rules()).unwrap();
                assert_eq!(actual.input().history, reference.as_slice());
                let encoded = classical::encode(actual.input()).unwrap();
                let expected = classical::encode(Input {
                    history: &reference,
                    ..actual.input()
                })
                .unwrap();
                assert_eq!(encoded, expected);
                assert_eq!(
                    input_key(projection.model().encoding().handle, &encoded),
                    input_key(projection.model().encoding().handle, &expected)
                );
            }
        }
    }

    #[cfg(feature = "experimental-prepared-input")]
    #[test]
    fn one_slot_replacement_keeps_external_consumer_until_its_last_drop() {
        let projection = projection(HistoryFill::No);
        let first_state = ContractPosition::new(OwnerId(81), Position::startpos())
            .export()
            .unwrap();
        let first = projection
            .prepared_input(first_state.rules(), first_state.legal_moves().moves(), None)
            .unwrap();
        let old_payload = Arc::downgrade(&first.encoded);
        let mut moved = Position::startpos();
        moved.apply_uci_moves(&["e2e4"]).unwrap();
        let second_state = ContractPosition::new(OwnerId(82), moved).export().unwrap();
        let second = projection
            .prepared_input(
                second_state.rules(),
                second_state.legal_moves().moves(),
                None,
            )
            .unwrap();
        assert!(old_payload.upgrade().is_some());
        drop(first);
        assert!(old_payload.upgrade().is_none());
        let current_payload = Arc::downgrade(&second.encoded);
        drop(second);
        assert!(current_payload.upgrade().is_some()); // The bounded memo owns it.
        drop(projection);
        assert!(current_payload.upgrade().is_none());
    }
}
