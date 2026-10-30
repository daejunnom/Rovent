//! Explicit native CPU value installation for an accepted NNUEC V2 view.
//!
//! The external evaluator/TT namespace is V2. The unchanged B/L arithmetic
//! remains owned by the V1 base model inside V2; these identities must not be
//! conflated. Context utility rows remain read-only observations. Selecting
//! this experimental evaluator does not change the engine's default value,
//! establish training/strength, or turn a nonterminal value into Rules proof.

use super::super::{
    NnuecError, NnuecNativeAccumulator, NnuecNativeUndo, copy_board, cpu_error,
    padded_board_values, reserve_local_append,
};
use super::review::{BoundNnuecReviewContext, Provenance};
use super::{Model, Output, SEMANTICS};
use crate::cpu_value::{
    CpuAccumulator, CpuAccumulatorUndo, CpuValueCacheContext, CpuValueError, CpuValueEvaluator,
    CpuValueIdentity,
};
use crate::pals::store::PalsStores;
use rz_contracts::Digest;
use rz_position::{Position, PositionLimits, RuleMoveDelta};
use sha2::{Digest as _, Sha256};
use std::sync::Arc;

/// Installed only after the original ReviewView submission is accepted.
/// Keeping the bound owner retains its actual view pin and cancellation
/// authority. CPU accumulator/TT entries retain only the small C authority.
#[derive(Clone)]
pub struct V2CpuValue {
    model: Arc<Model>,
    bound: Arc<BoundNnuecReviewContext>,
    binding: Digest,
}

impl V2CpuValue {
    pub fn from_bound(
        model: Arc<Model>,
        bound: Arc<BoundNnuecReviewContext>,
        stores: &PalsStores,
        original_view_seal: [u8; 32],
    ) -> Result<Self, NnuecError> {
        let provenance = bound.provenance();
        if provenance.model != *model.identity()
            || !bound.authority_is_live()
            || !bound.validates(
                stores,
                provenance.root,
                provenance.generation,
                original_view_seal,
            )
        {
            return Err(NnuecError::StateMismatch);
        }
        model.check_context(bound.context())?;
        let binding = binding_digest(model.identity(), provenance)?;
        if !bound.authority_is_live() {
            return Err(NnuecError::StateMismatch);
        }
        Ok(Self {
            model,
            bound,
            binding,
        })
    }

    fn check_live(&self) -> Result<(), CpuValueError> {
        if let Some(reason) = self.bound.authority_revocation() {
            return Err(CpuValueError::AuthorityRevoked(reason.into()));
        }
        Ok(())
    }

    fn score_native(
        &self,
        accumulator: &CpuAccumulator,
        position: &Position,
        capture: Option<&super::capture::NnuecActiveCapture>,
    ) -> Result<i32, CpuValueError> {
        self.cache_context(accumulator, position)?;
        let state = accumulator
            .nnuec
            .as_ref()
            .ok_or(CpuValueError::StateMismatch)?;
        // The ordinary and observed paths compute precisely one same forward.
        let numeric = self
            .model
            .base
            .forward_with_context_values(
                &state.board,
                self.bound.context().values(),
                &state.local,
                position,
            )
            .map_err(cpu_error)?;
        self.check_live()?;
        if let Some(capture) = capture {
            capture.observe(&self.model, &self.bound, &state.local, position, &numeric);
        }
        let score = numeric.frontier_score();
        #[cfg(feature = "nnuec-scalar-diagnostics")]
        super::scalar_diagnostic::record(
            self.binding,
            self.bound.provenance(),
            numeric.raw_outputs[0].to_bits(),
            numeric.value.to_bits(),
            score,
        );
        Ok(score)
    }

    /// Observe the existing value/WDL/utility head without granting its utility
    /// rows any authority to retain, drop, revisit, or route work.
    pub fn observe(
        &self,
        accumulator: &CpuAccumulator,
        position: &Position,
    ) -> Result<Output, CpuValueError> {
        self.cache_context(accumulator, position)?;
        let state = accumulator
            .nnuec
            .as_ref()
            .ok_or(CpuValueError::StateMismatch)?;
        let output = self
            .model
            .forward(&state.board, self.bound.context(), &state.local, position)
            .map_err(cpu_error)?;
        // A result computed after cancellation is not accepted just because
        // the immutable numerical inputs remain alive.
        self.check_live()?;
        Ok(output)
    }
}

/// Immutable job namespace belongs to the actual evaluator, so recovering its
/// engine cannot silently replace view custody with a weights-only identity.
pub(super) fn binding_digest(
    identity: &CpuValueIdentity,
    provenance: &Provenance,
) -> Result<Digest, NnuecError> {
    let bytes = serde_json::to_vec(identity).map_err(|e| NnuecError::Json(e.to_string()))?;
    let root = u64::try_from(provenance.root.0)
        .map_err(|_| NnuecError::Capacity("prepared value root identity"))?;
    let mut hash = Sha256::new();
    hash.update(b"rz-cpu-prepared-value-binding/1");
    hash.update((bytes.len() as u64).to_be_bytes());
    hash.update(bytes);
    hash.update(provenance.full_view_digest);
    hash.update(provenance.view_seal);
    hash.update(provenance.record_scope_digest);
    hash.update(root.to_be_bytes());
    hash.update(provenance.generation.to_be_bytes());
    Ok(Digest(hash.finalize().into()))
}

impl CpuValueEvaluator for V2CpuValue {
    fn identity(&self) -> &CpuValueIdentity {
        self.model.identity()
    }

    fn provenance(&self) -> &'static str {
        SEMANTICS
    }

    fn job_binding_digest(&self) -> Result<Option<Digest>, CpuValueError> {
        self.check_live()?;
        Ok(Some(self.binding))
    }
    fn nnuec_retirement_owner(&self) -> Option<super::retirement::NnuecValueRetirementOwner> {
        Some(super::retirement::NnuecValueRetirementOwner::new(
            Arc::clone(&self.model),
            Arc::clone(&self.bound),
            self.binding,
        ))
    }

    fn initialize(&self, position: &Position) -> Result<CpuAccumulator, CpuValueError> {
        self.check_live()?;
        // A CPU job may start at a frozen question endpoint rather than the
        // review root. Its exact position and this new L anchor seal the TT.
        let board = self.model.board(position).map_err(cpu_error)?;
        let local = self.model.local(position).map_err(cpu_error)?;
        let values = padded_board_values(&board);
        self.check_live()?;
        Ok(CpuAccumulator {
            values,
            identity: position.position_identity(),
            namespace: Arc::clone(&self.model.identity),
            exact: None,
            nnuec: Some(Box::new(NnuecNativeAccumulator {
                board,
                local,
                context_authority: Arc::clone(&self.bound.context().authority),
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
            || state.board.model != self.model.base.identity
            || state.local.model != self.model.base.identity
            || !Arc::ptr_eq(&state.context_authority, &self.bound.context().authority)
        {
            return Err(CpuValueError::StateMismatch);
        }
        // A known foreign accumulator remains a hard failure even when the
        // original Bound has also been cancelled. No new arithmetic follows
        // a revoked authority; only the read-only identity guards precede it.
        self.check_live()?;
        let context = self
            .model
            .cache_context(self.bound.context(), &state.local, position)
            .map_err(cpu_error)?;
        self.check_live()?;
        Ok(context)
    }

    fn score(
        &self,
        accumulator: &CpuAccumulator,
        position: &Position,
    ) -> Result<i32, CpuValueError> {
        // Same admitted V2 C and same head arithmetic. A per-node score does
        // not need to serialize/hash all frozen review lines for an Output
        // receipt; only the explicit observation path emits that metadata.
        self.score_native(accumulator, position, None)
    }

    fn score_with_capture(
        &self,
        accumulator: &CpuAccumulator,
        position: &Position,
        capture: &super::capture::NnuecActiveCapture,
    ) -> Result<i32, CpuValueError> {
        self.score_native(accumulator, position, Some(capture))
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
        if state.board.position != accumulator.identity
            || state.local.work.position_identity() != accumulator.identity
            || state.board.model != self.model.base.identity
            || state.local.model != self.model.base.identity
            || !Arc::ptr_eq(&state.context_authority, &self.bound.context().authority)
        {
            return Err(CpuValueError::StateMismatch);
        }
        self.check_live()?;
        // This small ownership bridge follows the existing V1 native delta
        // sequence. Every numerical operation remains in the same B/L helper;
        // do not replay legality or introduce a different FP32 update order.
        let prepared = self
            .model
            .base
            .prepare_local_append(&state.local, delta.mv)
            .map_err(cpu_error)?;
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
            .base
            .apply_board_delta(&mut next_board, delta)
            .map_err(cpu_error)?;
        reserve_local_append(&mut state.local).map_err(cpu_error)?;
        self.check_live()?;
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
        // Revocation forbids new work/readout, not disposal of an owned delta.
        // Cleanup may restore only this exact stamped state and undo pair;
        // none of the admission or value authority is revived by restoration.
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
        if state.board.position != accumulator.identity
            || state.local.work.position_identity() != accumulator.identity
            || state.board.model != self.model.base.identity
            || state.local.model != self.model.base.identity
            || !Arc::ptr_eq(&state.local.stamp, &native.after_stamp)
            || !Arc::ptr_eq(&state.board.stamp, &native.board_after_stamp)
            || state.local.moves.len() != native.expected_length
            || native.previous_work.position_identity() != undo.before
            || native.board.position != undo.before
            || native.board.model != self.model.base.identity
            || !Arc::ptr_eq(&state.context_authority, &self.bound.context().authority)
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

#[cfg(test)]
mod tests {
    use super::super::Checkpoint;
    use super::super::review::{PreparationActor, Request};
    use super::*;
    use crate::cpu_value::CpuTrainingState;
    use crate::pals::engine::{RecordKind, RoleQueryPurpose, RoleRecord};
    use crate::pals::review_context::{
        ReviewContextIndex, ReviewQuerySequences, ReviewSelectionPolicy, ReviewView,
    };
    use crate::pals::store::{StateId, StoreLimits};
    use rz_position::{BoardMove, Color};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    struct Fixture {
        model: Arc<Model>,
        bound: Arc<BoundNnuecReviewContext>,
        stores: PalsStores,
        view: Arc<ReviewView>,
        actor: PreparationActor,
        cancel: Arc<AtomicBool>,
    }

    fn checkpoint() -> Checkpoint {
        let mut checkpoint = super::super::tests::checkpoint();
        checkpoint.base.output_scale = 8.0;
        // Nonzero B and L make a delta/rebuild assertion observe both paths.
        checkpoint.base.parameters.board_weights =
            (0..800).map(|i| (i % 17) as f32 / 32.0).collect();
        checkpoint.base.parameters.local_projection[0] = 0.75;
        checkpoint.base.parameters.hidden_weights = vec![0.25, 0.5, 1.0, 0.125];
        checkpoint
    }

    fn record(
        state: StateId,
        revision: u64,
        kind: RecordKind,
        moves: &str,
        parent: Option<u64>,
    ) -> RoleRecord {
        RoleRecord {
            revision,
            parent_revision: parent,
            supersedes_revision: None,
            origin_state: state,
            kind,
            line: moves
                .split_whitespace()
                .map(|movement| movement.parse().unwrap())
                .collect(),
            value: None,
            completed_depth: 0,
            score_scope: None,
            cpu_observation: None,
            perspective: Color::White,
            critical: true,
        }
    }

    fn prepare_fixture(checkpoint: Checkpoint) -> Fixture {
        let model = Arc::new(Model::new(checkpoint).unwrap());
        let mut stores = PalsStores::new(StoreLimits::default());
        let situation = stores
            .focus_actual_moves(Position::startpos().snapshot())
            .unwrap();
        let state = stores.situations.get(situation).unwrap().state;
        let mut index = ReviewContextIndex::default();
        for record in [
            record(state, 1, RecordKind::Proposal, "e2e4", None),
            record(state, 2, RecordKind::Counterexample, "e2e4 e7e5", Some(1)),
        ] {
            index.publish(&stores, &record, None).unwrap();
        }
        let view = Arc::new(
            index
                .select(
                    &stores,
                    state,
                    ReviewQuerySequences {
                        purpose: RoleQueryPurpose::ProposePolicy,
                        prefix: &[],
                        proposal: &[],
                        counterexample: None,
                        legal: &[],
                        divergences: &[],
                    },
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        let cancel = Arc::new(AtomicBool::new(false));
        let request = Request::admit(
            &model,
            Arc::clone(&view),
            &stores,
            state,
            stores.generation(),
            Instant::now() + Duration::from_secs(30),
            Arc::clone(&cancel),
        )
        .unwrap();
        let actor = PreparationActor::spawn(Arc::clone(&model)).unwrap();
        actor
            .try_submit(request)
            .unwrap_or_else(|_| panic!("initial preparation admission"));
        let until = Instant::now() + Duration::from_secs(10);
        let bound = loop {
            if let Some(completion) = actor.try_complete().unwrap() {
                break Arc::new(completion.result.unwrap());
            }
            assert!(Instant::now() < until, "preparation did not complete");
            std::thread::yield_now();
        };
        Fixture {
            model,
            bound,
            stores,
            view,
            actor,
            cancel,
        }
    }

    impl Fixture {
        fn evaluator(&self) -> V2CpuValue {
            V2CpuValue::from_bound(
                Arc::clone(&self.model),
                Arc::clone(&self.bound),
                &self.stores,
                self.view.seal_digest(),
            )
            .unwrap()
        }
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    fn assert_output_bits(actual: &Output, expected: &Output) {
        assert_eq!(actual.numeric.schema, expected.numeric.schema);
        assert_eq!(
            actual.numeric.weights_sha256,
            expected.numeric.weights_sha256
        );
        assert_eq!(actual.context_sha256, expected.context_sha256);
        for side in 0..2 {
            assert_eq!(
                bits(&actual.numeric.board_accumulators[side]),
                bits(&expected.numeric.board_accumulators[side]),
            );
        }
        for (actual, expected) in [
            (
                actual.numeric.context_accumulator.as_slice(),
                expected.numeric.context_accumulator.as_slice(),
            ),
            (
                actual.numeric.local_accumulator.as_slice(),
                expected.numeric.local_accumulator.as_slice(),
            ),
            (
                actual.numeric.hidden.as_slice(),
                expected.numeric.hidden.as_slice(),
            ),
            (&actual.numeric.raw_outputs, &expected.numeric.raw_outputs),
            (&actual.numeric.wdl_logits, &expected.numeric.wdl_logits),
            (&actual.numeric.wdl, &expected.numeric.wdl),
        ] {
            assert_eq!(bits(actual), bits(expected));
        }
        assert_eq!(
            actual.numeric.value.to_bits(),
            expected.numeric.value.to_bits()
        );
    }

    #[test]
    fn installation_requires_original_seal_owner_and_v2_model() {
        let fixture = prepare_fixture(checkpoint());
        let evaluator = fixture.evaluator();
        let root = Position::startpos();
        let accumulator = evaluator.initialize(&root).unwrap();
        assert_eq!(evaluator.identity().semantics, SEMANTICS);
        assert_eq!(evaluator.identity().training, CpuTrainingState::Untrained);
        assert_ne!(evaluator.identity(), fixture.model.base.identity());
        let state = accumulator.nnuec.as_ref().unwrap();
        assert_eq!(accumulator.namespace.as_ref(), fixture.model.identity());
        assert_eq!(state.board.model.as_ref(), fixture.model.base.identity());
        assert_eq!(state.local.model.as_ref(), fixture.model.base.identity());
        let expected = fixture
            .model
            .forward(
                &fixture.model.board(&root).unwrap(),
                fixture.bound.context(),
                &fixture.model.local(&root).unwrap(),
                &root,
            )
            .unwrap();
        assert_output_bits(&evaluator.observe(&accumulator, &root).unwrap(), &expected);
        assert_eq!(
            evaluator.score(&accumulator, &root).unwrap(),
            expected.numeric.frontier_score(),
        );

        let mut wrong_seal = fixture.view.seal_digest();
        wrong_seal[0] ^= 1;
        assert_eq!(
            V2CpuValue::from_bound(
                Arc::clone(&fixture.model),
                Arc::clone(&fixture.bound),
                &fixture.stores,
                wrong_seal,
            )
            .err(),
            Some(NnuecError::StateMismatch),
        );
        let foreign = PalsStores::new(StoreLimits::default());
        assert_eq!(
            V2CpuValue::from_bound(
                Arc::clone(&fixture.model),
                Arc::clone(&fixture.bound),
                &foreign,
                fixture.view.seal_digest(),
            )
            .err(),
            Some(NnuecError::StateMismatch),
        );
        let mut different = checkpoint();
        different.parameters.kind_embeddings[0] += 0.25;
        assert_eq!(
            V2CpuValue::from_bound(
                Arc::new(Model::new(different).unwrap()),
                Arc::clone(&fixture.bound),
                &fixture.stores,
                fixture.view.seal_digest(),
            )
            .err(),
            Some(NnuecError::StateMismatch),
        );
    }

    #[test]
    fn checked_native_delta_rebuild_and_owned_undo_keep_identical_bits() {
        let fixture = prepare_fixture(checkpoint());
        let evaluator = fixture.evaluator();
        let cases = [
            ("startpos", "e2e4"),
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
            ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", "e5d6"),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8r"),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8b"),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n"),
            ("4k3/8/8/8/8/8/p7/4K3 b - - 0 1", "a2a1q"),
            ("4k3/8/8/8/8/8/p7/4K3 b - - 0 1", "a2a1r"),
            ("4k3/8/8/8/8/8/p7/4K3 b - - 0 1", "a2a1b"),
            ("4k3/8/8/8/8/8/p7/4K3 b - - 0 1", "a2a1n"),
        ];
        for (fen, movement) in cases {
            let mut position = if fen == "startpos" {
                Position::startpos()
            } else {
                Position::from_fen(fen).unwrap()
            };
            let mut rebuilt_local = fixture.model.local(&position).unwrap();
            let mut accumulator = evaluator.initialize(&position).unwrap();
            let original = evaluator.observe(&accumulator, &position).unwrap();
            let movement: BoardMove = movement.parse().unwrap();
            let rules = position.make_move(movement).unwrap();
            let undo = evaluator
                .apply_delta(&mut accumulator, rules.delta())
                .unwrap();
            fixture
                .model
                .append_local(&mut rebuilt_local, movement)
                .unwrap();
            let rebuilt_board = fixture.model.board(&position).unwrap();
            let expected = fixture
                .model
                .forward(
                    &rebuilt_board,
                    fixture.bound.context(),
                    &rebuilt_local,
                    &position,
                )
                .unwrap();
            assert_output_bits(
                &evaluator.observe(&accumulator, &position).unwrap(),
                &expected,
            );
            assert!(accumulator.retained_bytes() > std::mem::size_of::<CpuAccumulator>());
            assert!(undo.retained_bytes() > std::mem::size_of::<CpuAccumulatorUndo>());
            let at_child = evaluator.initialize(&position).unwrap();
            assert_eq!(accumulator.values(), at_child.values());
            assert_ne!(
                evaluator.cache_context(&accumulator, &position).unwrap(),
                evaluator.cache_context(&at_child, &position).unwrap(),
            );
            position.unmake(rules).unwrap();
            evaluator
                .restore(&mut accumulator, undo, &position)
                .unwrap();
            assert_output_bits(
                &evaluator.observe(&accumulator, &position).unwrap(),
                &original,
            );
        }
    }

    #[test]
    fn paused_stack_v2_preserves_native_lifo_bits_and_uninterrupted_work() {
        use crate::cpu::{
            CpuCompletion, CpuConfig, CpuEngine, CpuLimits, CpuOrderingPolicy, CpuResumePolicy,
            CpuWork,
        };
        use std::sync::Mutex;

        #[derive(Default)]
        struct NativeTrace {
            initializations: u64,
            deltas: u64,
            restores: u64,
            root_restores: u64,
            peak: usize,
            local_changed: bool,
            before: Vec<Output>,
        }
        struct TracedV2 {
            inner: V2CpuValue,
            root: Position,
            trace: Arc<Mutex<NativeTrace>>,
        }
        impl CpuValueEvaluator for TracedV2 {
            fn identity(&self) -> &CpuValueIdentity {
                self.inner.identity()
            }
            fn provenance(&self) -> &'static str {
                self.inner.provenance()
            }
            fn job_binding_digest(&self) -> Result<Option<Digest>, CpuValueError> {
                self.inner.job_binding_digest()
            }
            fn nnuec_retirement_owner(
                &self,
            ) -> Option<crate::cpu_value::nnuec::v2::retirement::NnuecValueRetirementOwner>
            {
                self.inner.nnuec_retirement_owner()
            }
            fn initialize(&self, position: &Position) -> Result<CpuAccumulator, CpuValueError> {
                let value = self.inner.initialize(position)?;
                assert!(self.root.snapshot().same_state(&position.snapshot()));
                self.trace.lock().unwrap().initializations += 1;
                Ok(value)
            }
            fn cache_context(
                &self,
                value: &CpuAccumulator,
                position: &Position,
            ) -> Result<CpuValueCacheContext, CpuValueError> {
                self.inner.cache_context(value, position)
            }
            fn score(
                &self,
                value: &CpuAccumulator,
                position: &Position,
            ) -> Result<i32, CpuValueError> {
                self.inner.score(value, position)
            }
            fn apply_delta(
                &self,
                value: &mut CpuAccumulator,
                delta: &RuleMoveDelta,
            ) -> Result<CpuAccumulatorUndo, CpuValueError> {
                let position = Position::from_snapshot(&delta.before, Default::default()).unwrap();
                let before = self.inner.observe(value, &position)?;
                // Only the original V2 evaluator creates/mutates the native
                // accumulator and its uncloneable owned undo.
                let undo = self.inner.apply_delta(value, delta)?;
                let state = value.nnuec.as_ref().unwrap();
                assert!(Arc::ptr_eq(
                    &state.context_authority,
                    &self.inner.bound.context().authority,
                ));
                let after = self.inner.observe(value, &state.local.work)?;
                let mut trace = self.trace.lock().unwrap();
                assert!(trace.before.len() < 16, "bounded fixture native undo depth");
                trace.local_changed |= bits(&before.numeric.local_accumulator)
                    != bits(&after.numeric.local_accumulator);
                trace.before.push(before);
                trace.deltas += 1;
                trace.peak = trace.peak.max(trace.before.len());
                Ok(undo)
            }
            fn restore(
                &self,
                value: &mut CpuAccumulator,
                undo: CpuAccumulatorUndo,
                position: &Position,
            ) -> Result<(), CpuValueError> {
                self.inner.restore(value, undo, position)?;
                let actual = self.inner.observe(value, position)?;
                let state = value.nnuec.as_ref().unwrap();
                assert!(Arc::ptr_eq(
                    &state.context_authority,
                    &self.inner.bound.context().authority,
                ));
                let mut trace = self.trace.lock().unwrap();
                let before = trace
                    .before
                    .pop()
                    .expect("actual native delta owns one LIFO undo");
                assert_output_bits(&actual, &before);
                trace.restores += 1;
                if trace.before.is_empty() {
                    // This is the actual search work Position after unmake,
                    // not merely the unchanged caller's borrowed root.
                    assert!(self.root.snapshot().same_state(&position.snapshot()));
                    assert!(state.local.moves().is_empty());
                    trace.root_restores += 1;
                }
                Ok(())
            }
        }

        let fixture = prepare_fixture(checkpoint());
        let position = Position::startpos();
        let before = position.snapshot();
        let frozen_context = bits(fixture.bound.context().values());
        assert!(
            fixture
                .bound
                .context()
                .values()
                .iter()
                .any(|value| *value != 0.0)
        );
        let root_moves = ["e2e4".parse::<BoardMove>().unwrap()];
        let config = CpuConfig {
            tt_entries: 16,
            max_depth: 3,
            quiescence_ply: 4,
            ..CpuConfig::default()
        };
        let limits = |max_nodes| CpuLimits {
            max_depth: 3,
            max_nodes,
            deadline: None,
        };
        let cancellation = AtomicBool::new(false);
        let traced = |trace: &Arc<Mutex<NativeTrace>>| -> Arc<dyn CpuValueEvaluator> {
            Arc::new(TracedV2 {
                inner: fixture.evaluator(),
                root: position.clone(),
                trace: Arc::clone(trace),
            })
        };
        let full_trace = Arc::new(Mutex::new(NativeTrace::default()));
        let mut full = CpuEngine::with_evaluator_and_ordering(
            config.clone(),
            traced(&full_trace),
            CpuOrderingPolicy::LegacyMvvLvaV1,
        )
        .unwrap();
        let expected = full
            .analyze_root_moves(&position, &root_moves, limits(20_000), &cancellation)
            .unwrap();
        assert_eq!(expected.completion, CpuCompletion::DepthLimit);
        assert_eq!(expected.completed_depth, 3);
        let expected_work = full.last_attempt_work().unwrap();
        let expected_trace = {
            let trace = full_trace.lock().unwrap();
            assert!(trace.before.is_empty());
            assert_eq!(trace.initializations, 1);
            assert_eq!(trace.deltas, trace.restores);
            assert!(trace.peak >= 3 && trace.local_changed && trace.root_restores > 0);
            (
                trace.deltas,
                trace.restores,
                trace.root_restores,
                trace.peak,
            )
        };

        for interval in [1_u64, 7] {
            let trace = Arc::new(Mutex::new(NativeTrace::default()));
            let mut sliced = CpuEngine::with_evaluator_ordering_and_resume_policy(
                config.clone(),
                traced(&trace),
                CpuOrderingPolicy::LegacyMvvLvaV1,
                CpuResumePolicy::PausedStack,
            )
            .unwrap();
            let owner = sliced.paused_stack_snapshot().unwrap().owner_id;
            let mut report = sliced
                .analyze_root_moves(&position, &root_moves, limits(interval), &cancellation)
                .unwrap();
            let mut total = CpuWork::default();
            let mut saw_nested_pause = false;
            for _ in 0..20_000 {
                let work = sliced.last_attempt_work().unwrap();
                assert_eq!(
                    work,
                    CpuWork {
                        nodes: report.nodes,
                        quiescence_nodes: report.quiescence_nodes,
                        tt_hits: report.tt_hits,
                    }
                );
                assert!(work.nodes <= interval);
                total.nodes = total.nodes.checked_add(work.nodes).unwrap();
                total.quiescence_nodes = total
                    .quiescence_nodes
                    .checked_add(work.quiescence_nodes)
                    .unwrap();
                total.tt_hits = total.tt_hits.checked_add(work.tt_hits).unwrap();
                assert!(before.same_state(&position.snapshot()));
                if report.completion != CpuCompletion::NodeLimit {
                    break;
                }
                saw_nested_pause |= trace.lock().unwrap().before.len() >= 2;
                let token = report
                    .resume
                    .take()
                    .expect("actual V2 stack pause owns a token");
                assert!(token.is_paused_stack() && sliced.token_is_current(&token));
                assert_eq!(token.cumulative_work(), Some(total));
                assert_eq!(token.paused_stack_bytes(), sliced.paused_stack_bytes());
                let paused = sliced.paused_stack_snapshot().unwrap();
                assert!(paused.complete && paused.owner_id == owner);
                assert_eq!(paused.tokens_retained, 1);
                assert!(paused.bytes_current > 0);
                report = sliced
                    .resume(&position, &token, limits(interval), &cancellation)
                    .unwrap();
                assert!(!sliced.token_is_current(&token));
            }
            assert_eq!(report.completion, CpuCompletion::DepthLimit);
            assert_eq!(
                (report.score, &report.pv, report.completed_depth),
                (expected.score, &expected.pv, expected.completed_depth),
            );
            assert_eq!(total, expected_work);
            assert!(
                saw_nested_pause,
                "the original V2 undo chain must survive a pause"
            );
            let observed = trace.lock().unwrap();
            assert!(observed.before.is_empty() && observed.local_changed);
            assert_eq!(
                observed.initializations, 1,
                "resume must retain the original V2 L"
            );
            assert_eq!(
                (
                    observed.deltas,
                    observed.restores,
                    observed.root_restores,
                    observed.peak,
                ),
                expected_trace,
            );
            let complete = sliced.paused_stack_snapshot().unwrap();
            assert!(complete.complete && complete.owner_id == owner);
            assert_eq!(complete.tokens_retained, 0);
            assert_eq!(complete.bytes_current, 0);
            assert_eq!(complete.replayed_consumed_work, 0);
            assert!(complete.tokens_created > 0 && complete.tokens_resumed > 0);
            assert!(!complete.admission_closed && !complete.owner_released);
            assert!(sliced.paused_stack_bytes().is_none());
            assert!(before.same_state(&position.snapshot()));
            assert_eq!(bits(fixture.bound.context().values()), frozen_context);
        }
    }

    #[test]
    fn equal_numeric_contexts_with_distinct_authority_cannot_share_native_state() {
        let first = prepare_fixture(checkpoint());
        let second = prepare_fixture(checkpoint());
        assert_eq!(first.model.identity(), second.model.identity());
        assert_eq!(
            bits(first.bound.context().values()),
            bits(second.bound.context().values()),
        );
        let a = first.evaluator();
        let b = second.evaluator();
        let position = Position::startpos();
        let accumulator = a.initialize(&position).unwrap();
        let other = b.initialize(&position).unwrap();
        assert_ne!(
            a.cache_context(&accumulator, &position).unwrap(),
            b.cache_context(&other, &position).unwrap(),
        );
        assert_eq!(
            b.cache_context(&accumulator, &position),
            Err(CpuValueError::StateMismatch)
        );
        let mut copied = accumulator.clone();
        let mut work = Position::startpos();
        let rules = work.make_move("e2e4".parse().unwrap()).unwrap();
        assert_eq!(
            b.apply_delta(&mut copied, rules.delta()).err(),
            Some(CpuValueError::StateMismatch),
        );
        assert_eq!(copied.values(), accumulator.values());
        assert_eq!(copied.identity, accumulator.identity);
    }

    #[test]
    fn clone_undo_and_capacity_failures_preserve_the_owned_state() {
        let fixture = prepare_fixture(checkpoint());
        let evaluator = fixture.evaluator();
        let mut position = Position::startpos();
        let mut accumulator = evaluator.initialize(&position).unwrap();
        let rules = position.make_move("e2e4".parse().unwrap()).unwrap();
        let undo = evaluator
            .apply_delta(&mut accumulator, rules.delta())
            .unwrap();
        let mut cloned = accumulator.clone();
        let values = cloned.values;
        let identity = cloned.identity.clone();
        position.unmake(rules).unwrap();
        fixture.cancel.store(true, Ordering::Release);
        assert_eq!(
            evaluator.restore(&mut cloned, undo, &position),
            Err(CpuValueError::StateMismatch),
        );
        assert_eq!(cloned.values, values);
        assert_eq!(cloned.identity, identity);

        let mut limited = checkpoint();
        limited.base.config.max_plies = 2;
        limited.base.parameters.ply_weights.truncate(2);
        limited.base.parameters.ply_weights.shrink_to_fit();
        let limited = prepare_fixture(limited);
        let evaluator = limited.evaluator();
        let mut position = Position::startpos();
        let mut accumulator = evaluator.initialize(&position).unwrap();
        for movement in ["e2e4", "e7e5"] {
            let rules = position.make_move(movement.parse().unwrap()).unwrap();
            let _ = evaluator
                .apply_delta(&mut accumulator, rules.delta())
                .unwrap();
        }
        let expected = evaluator.observe(&accumulator, &position).unwrap();
        let identity = accumulator.identity.clone();
        let third = position.make_move("g1f3".parse().unwrap()).unwrap();
        assert_eq!(
            evaluator.apply_delta(&mut accumulator, third.delta()).err(),
            Some(CpuValueError::InvalidCheckpoint(
                "NNUEC configured capacity exceeded"
            )),
        );
        assert_eq!(accumulator.identity, identity);
        position.unmake(third).unwrap();
        assert_output_bits(
            &evaluator.observe(&accumulator, &position).unwrap(),
            &expected,
        );
    }

    #[test]
    fn late_cancel_rejects_new_work_but_allows_only_owned_undo_cleanup() {
        let fixture = prepare_fixture(checkpoint());
        let evaluator = fixture.evaluator();
        let mut position = Position::startpos();
        let mut accumulator = evaluator.initialize(&position).unwrap();
        let original_values = accumulator.values;
        let rules = position.make_move("e2e4".parse().unwrap()).unwrap();
        let undo = evaluator
            .apply_delta(&mut accumulator, rules.delta())
            .unwrap();
        let child = Position::from_snapshot(&position.snapshot(), Default::default()).unwrap();
        let values = accumulator.values;
        let identity = accumulator.identity.clone();
        let second = position.make_move("e7e5".parse().unwrap()).unwrap();
        fixture.cancel.store(true, Ordering::Release);
        let cancelled = CpuValueError::AuthorityRevoked(
            crate::cpu_value::CpuValueAuthorityRevocation::Cancelled,
        );
        assert!(matches!(
            evaluator.initialize(&child),
            Err(CpuValueError::AuthorityRevoked(
                crate::cpu_value::CpuValueAuthorityRevocation::Cancelled
            ))
        ));
        assert_eq!(
            evaluator.cache_context(&accumulator, &child),
            Err(cancelled.clone())
        );
        assert_eq!(
            evaluator.score(&accumulator, &child),
            Err(cancelled.clone())
        );
        assert!(matches!(
            evaluator.observe(&accumulator, &child),
            Err(CpuValueError::AuthorityRevoked(
                crate::cpu_value::CpuValueAuthorityRevocation::Cancelled
            ))
        ));
        assert_eq!(
            evaluator
                .apply_delta(&mut accumulator, second.delta())
                .err(),
            Some(cancelled.clone()),
        );
        assert_eq!(accumulator.values, values);
        assert_eq!(accumulator.identity, identity);
        position.unmake(second).unwrap();
        position.unmake(rules).unwrap();
        assert_eq!(evaluator.restore(&mut accumulator, undo, &position), Ok(()),);
        assert_eq!(accumulator.values, original_values);
        assert_eq!(accumulator.identity, position.position_identity());
        assert!(accumulator.nnuec.as_ref().unwrap().local.moves().is_empty());
        assert_eq!(
            evaluator.score(&accumulator, &position),
            Err(cancelled.clone()),
        );
        assert_eq!(
            evaluator.cache_context(&accumulator, &position),
            Err(cancelled),
        );
        assert_eq!(
            V2CpuValue::from_bound(
                Arc::clone(&fixture.model),
                Arc::clone(&fixture.bound),
                &fixture.stores,
                fixture.view.seal_digest(),
            )
            .err(),
            Some(NnuecError::StateMismatch),
        );
    }

    #[test]
    fn invalidation_and_actor_shutdown_revoke_an_already_installed_evaluator() {
        let fixture = prepare_fixture(checkpoint());
        let evaluator = fixture.evaluator();
        let mut position = Position::startpos();
        let mut accumulator = evaluator.initialize(&position).unwrap();
        let original_values = accumulator.values;
        let rules = position.make_move("e2e4".parse().unwrap()).unwrap();
        let undo = evaluator
            .apply_delta(&mut accumulator, rules.delta())
            .unwrap();
        fixture.actor.invalidate_cache().unwrap();
        fixture.cancel.store(true, Ordering::Release);
        let invalidated = CpuValueError::AuthorityRevoked(
            crate::cpu_value::CpuValueAuthorityRevocation::Invalidated,
        );
        assert_eq!(
            evaluator.score(&accumulator, &position),
            Err(invalidated.clone())
        );
        position.unmake(rules).unwrap();
        evaluator
            .restore(&mut accumulator, undo, &position)
            .unwrap();
        assert_eq!(accumulator.values, original_values);
        assert_eq!(evaluator.score(&accumulator, &position), Err(invalidated),);
        fixture.actor.shutdown().unwrap();
        assert_eq!(
            evaluator.cache_context(&accumulator, &position),
            Err(CpuValueError::AuthorityRevoked(
                crate::cpu_value::CpuValueAuthorityRevocation::ActorStopped
            ))
        );
    }

    #[test]
    fn revoked_authority_does_not_hide_a_foreign_accumulator() {
        let first = prepare_fixture(checkpoint());
        let second = prepare_fixture(checkpoint());
        let a = first.evaluator();
        let b = second.evaluator();
        let position = Position::startpos();
        let mut foreign = a.initialize(&position).unwrap();
        second.cancel.store(true, Ordering::Release);
        assert_eq!(
            b.cache_context(&foreign, &position),
            Err(CpuValueError::StateMismatch),
        );
        assert_eq!(
            b.score(&foreign, &position),
            Err(CpuValueError::StateMismatch)
        );
        let mut work = Position::startpos();
        let rules = work.make_move("e2e4".parse().unwrap()).unwrap();
        assert_eq!(
            b.apply_delta(&mut foreign, rules.delta()).err(),
            Some(CpuValueError::StateMismatch),
        );
        assert_eq!(foreign.identity, position.position_identity());
    }

    #[test]
    fn original_cancel_inside_value_readout_is_a_known_nonresumable_stop() {
        use crate::cpu::{CpuCompletion, CpuConfig, CpuEngine, CpuOrderingPolicy, CpuResumePolicy};
        use std::sync::atomic::AtomicU64;

        struct CancelOnSecondScore {
            inner: V2CpuValue,
            cancel: Arc<AtomicBool>,
            scores: AtomicU64,
        }
        impl CpuValueEvaluator for CancelOnSecondScore {
            fn identity(&self) -> &CpuValueIdentity {
                self.inner.identity()
            }
            fn provenance(&self) -> &'static str {
                self.inner.provenance()
            }
            fn job_binding_digest(&self) -> Result<Option<Digest>, CpuValueError> {
                self.inner.job_binding_digest()
            }
            fn initialize(&self, p: &Position) -> Result<CpuAccumulator, CpuValueError> {
                self.inner.initialize(p)
            }
            fn cache_context(
                &self,
                a: &CpuAccumulator,
                p: &Position,
            ) -> Result<CpuValueCacheContext, CpuValueError> {
                self.inner.cache_context(a, p)
            }
            fn score(&self, a: &CpuAccumulator, p: &Position) -> Result<i32, CpuValueError> {
                if self.scores.fetch_add(1, Ordering::AcqRel) == 1 {
                    // The actual CPU control check has already admitted this
                    // readout. Revoke the exact original Bound, without a
                    // sleep, new deadline, fake score or synthetic response.
                    self.cancel.store(true, Ordering::Release);
                }
                self.inner.score(a, p)
            }
            fn apply_delta(
                &self,
                a: &mut CpuAccumulator,
                d: &RuleMoveDelta,
            ) -> Result<CpuAccumulatorUndo, CpuValueError> {
                self.inner.apply_delta(a, d)
            }
            fn restore(
                &self,
                a: &mut CpuAccumulator,
                u: CpuAccumulatorUndo,
                p: &Position,
            ) -> Result<(), CpuValueError> {
                self.inner.restore(a, u, p)
            }
        }
        let fixture = prepare_fixture(checkpoint());
        let position = Position::startpos();
        let before = position.snapshot();
        let original = fixture.evaluator();
        let original_score = original
            .score(&original.initialize(&position).unwrap(), &position)
            .unwrap();
        let mut engine = CpuEngine::with_evaluator_ordering_and_resume_policy(
            CpuConfig {
                tt_entries: 0,
                max_depth: 2,
                ..CpuConfig::default()
            },
            Arc::new(CancelOnSecondScore {
                inner: original,
                cancel: Arc::clone(&fixture.cancel),
                scores: AtomicU64::new(0),
            }),
            CpuOrderingPolicy::LegacyMvvLvaV1,
            CpuResumePolicy::PausedStack,
        )
        .unwrap();
        let report = engine
            .analyze(
                &position,
                crate::cpu::CpuLimits {
                    max_depth: 2,
                    max_nodes: 100_000,
                    deadline: None,
                },
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(report.completion, CpuCompletion::Canceled);
        assert_eq!(report.completed_depth, 0);
        assert_eq!(report.score, original_score);
        assert!(report.nodes > 0);
        assert_eq!(engine.last_attempt_work().unwrap().nodes, report.nodes);
        assert!(report.resume.is_none() && engine.paused_stack_bytes().is_none());
        let owner = engine.paused_stack_snapshot().unwrap();
        assert!(owner.complete && owner.owner_id != 0);
        assert_eq!(owner.tokens_retained, 0);
        assert_eq!(owner.bytes_current, 0);
        assert!(before.same_state(&position.snapshot()));
    }

    #[test]
    fn fresh_and_retained_value_preflight_keep_the_original_no_score_revocation() {
        use crate::cpu::{
            CpuConfig, CpuEngine, CpuError, CpuLimits, CpuOrderingPolicy, CpuResumePolicy, CpuWork,
        };
        use crate::cpu_value::CpuValueAuthorityRevocation;

        for retained in [false, true] {
            let fixture = prepare_fixture(checkpoint());
            let position = Position::startpos();
            let before = position.snapshot();
            let mut engine = CpuEngine::with_evaluator_ordering_and_resume_policy(
                CpuConfig {
                    tt_entries: 0,
                    max_depth: 2,
                    ..CpuConfig::default()
                },
                Arc::new(fixture.evaluator()),
                CpuOrderingPolicy::LegacyMvvLvaV1,
                CpuResumePolicy::PausedStack,
            )
            .unwrap();
            let limits = CpuLimits {
                max_depth: 2,
                max_nodes: 1,
                deadline: None,
            };
            let token = retained.then(|| {
                engine
                    .analyze(&position, limits, &AtomicBool::new(false))
                    .unwrap()
                    .resume
                    .unwrap()
            });
            fixture.cancel.store(true, Ordering::Release);
            let result = if let Some(token) = &token {
                engine.resume(&position, token, limits, &AtomicBool::new(false))
            } else {
                engine.analyze(&position, limits, &AtomicBool::new(false))
            };
            assert!(matches!(
                result,
                Err(CpuError::Value(CpuValueError::AuthorityRevoked(
                    CpuValueAuthorityRevocation::Cancelled,
                )))
            ));
            assert_eq!(engine.last_attempt_work(), Some(CpuWork::default()));
            assert!(engine.paused_stack_bytes().is_none());
            let owner = engine.paused_stack_snapshot().unwrap();
            assert!(owner.complete && owner.owner_id != 0);
            assert_eq!(owner.tokens_retained, 0);
            assert_eq!(owner.bytes_current, 0);
            assert!(before.same_state(&position.snapshot()));
        }
    }
}
