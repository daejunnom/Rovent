//! Thin adapter to coordinator contract 0.1 (PR #7).
//!
//! No scheduler, Rules implementation, new shared IDs, or logical Evaluator.
//! A/embedding code supplies a projection of the exact immutable Rules snapshot.
//! D owns current scope/clock, finalization and physical execution ID issuance.

use crate::asset;
use crate::error::{BackendError, FailureKind, FailureStage};
pub use crate::model_adapter::{backend_error, PhysicalFailure};
use crate::{output, RawOutput};
use rz_contracts::*;
use rz_encoding::classical::{self, EncodedInput, HistoryFill, Input, INPUT_VALUES};
use rz_encoding::{policy, POLICY_SIZE};
use sha2::{Digest as _, Sha256};
use std::sync::Arc;

pub use crate::native_runtime_bridge::{
    NativeDiagnosticBatch, NativeDiagnosticReceipt, NativeRuntimeBackend, NativeWorkerOwner,
};
pub use crate::rules_projection::{ClassicalProjection, RulesProjection};
pub use crate::runtime_bridge::{BridgeDiagnostics, MockTicket, ScriptedRuntimeBackend};

/// Conservative C buffer reservation: encoded input + staging, native/owned raw
/// heads, legal f32/f64 conversion and Arc allocation, plus small metadata.
/// Rules snapshots and native session/activation workspace require other budgets.
pub const HOST_BYTES_PER_ITEM: u64 =
    (2 * INPUT_VALUES * 4 + 2 * (POLICY_SIZE + 3) * 4 + POLICY_SIZE * 20 + 4096) as u64;

/// Versioned local encoding manifest. Coordinates are a1=0, h8=63; shared Move
/// uses the actual king destination for standard castling. No Chess960 profile.
pub fn encoding_manifest(fill: HistoryFill) -> Digest {
    Digest(asset::sha256(format!("rz-maia-classical-v1;a1=0;stm-rank-flip;112x8x8;history={};clock=raw;castle=king-destination;policy=qrb-tail,n-base", fill.reference_option()).as_bytes()))
}

pub fn action_map_digest() -> Digest {
    let mut hash = Sha256::new();
    hash.update(b"rz-maia-action-map-v1\0");
    for slot in policy::slots() {
        hash.update([slot.from, slot.to, slot.promotion.map_or(0, |p| p as u8)]);
    }
    Digest(hash.finalize().into())
}

/// Bootstrap/A computes this before constructing EvalContext; prepare recomputes
/// it from the actual tensor. It is a C model-input codec, not a Rules state hash.
pub fn input_key(encoding: EncodingHandle, input: &EncodedInput) -> EvalInputKey {
    let mut hash = Sha256::new();
    hash.update(b"rz-maia-input-v1\0");
    hash.update(encoding.owner.0.to_le_bytes());
    hash.update(encoding.slot.to_le_bytes());
    hash.update(encoding.generation.0.to_le_bytes());
    hash.update(encoding.manifest.0);
    hash.update((input.known_frames as u64).to_le_bytes());
    hash.update((input.padded_frames as u64).to_le_bytes());
    hash.update([u8::from(input.inferred_ep_predecessor)]);
    hash.update(input.history_fill.reference_option().as_bytes());
    #[cfg(not(feature = "experimental-input-hash"))]
    for value in input.values() {
        hash.update(value.to_le_bytes());
    }
    #[cfg(feature = "experimental-input-hash")]
    for chunk in input.values().chunks(256) {
        let mut bytes = [0u8; 1024];
        for (value, dest) in chunk.iter().zip(bytes.chunks_exact_mut(4)) {
            dest.copy_from_slice(&value.to_le_bytes());
        }
        hash.update(&bytes[..chunk.len() * 4]);
    }
    EvalInputKey(Digest(hash.finalize().into()))
}

#[derive(Clone, Debug)]
pub struct MaiaBinding {
    model: Arc<ModelDescriptor>,
    backend: Digest,
    history_fill: HistoryFill,
}

impl MaiaBinding {
    /// Handles are issued by the registry, not by C. For physical ORT inference
    /// use `for_backend`, which binds the verified asset/backend identities.
    /// This constructor also permits CPU/mock contract tests without model files.
    pub fn new(
        model: ModelHandle,
        encoding: EncodingHandle,
        history_fill: HistoryFill,
        backend: Digest,
        max_batch: usize,
    ) -> Result<Self, ContractError> {
        if encoding.manifest != encoding_manifest(history_fill) {
            return Err(error(
                ErrorCode::IdentityMismatch,
                Stage::Contract,
                "encoding manifest differs from selected C profile",
            ));
        }
        if max_batch == 0 || max_batch > 16 {
            return Err(error(
                ErrorCode::ResourceExhausted,
                Stage::Contract,
                "Maia batch limit must be 1..=16",
            ));
        }
        let descriptor = EncodingDescriptor {
            handle: encoding,
            history_length: 8,
            action_map: action_map_digest(),
            history_policy: Digest(asset::sha256(history_fill.reference_option().as_bytes())),
        };
        Ok(Self {
            model: Arc::new(ModelDescriptor::try_new(
                model,
                descriptor,
                vec![PrecisionProfile::Fp32],
                1,
                max_batch,
            )?),
            backend,
            history_fill,
        })
    }

    #[cfg(feature = "onnx")]
    pub fn for_backend(
        backend: &crate::onnx::OnnxBackend,
        model: ModelHandle,
        encoding: EncodingHandle,
        history_fill: HistoryFill,
    ) -> Result<Self, ContractError> {
        if model.manifest.0 != backend.asset_identity() {
            return Err(error(
                ErrorCode::IdentityMismatch,
                Stage::Contract,
                "model registry manifest differs from loaded asset",
            ));
        }
        Self::new(
            model,
            encoding,
            history_fill,
            Digest(backend.identity()),
            backend.config().max_batch,
        )
    }

    pub fn model(&self) -> &Arc<ModelDescriptor> {
        &self.model
    }
    pub fn backend(&self) -> Digest {
        self.backend
    }
    pub fn history_fill(&self) -> HistoryFill {
        self.history_fill
    }

    /// Projection is supplied by the Rules adapter and must come from
    /// request.position().state(). C checks side, profile, input identity and
    /// ordered action mapping; it cannot prove arbitrary generic P's legality.
    pub fn prepare<P>(
        &self,
        request: Arc<EvalRequest<P>>,
        projection: Input<'_>,
    ) -> Result<PreparedRequest<P>, ContractError> {
        self.validate_preparation(&request, projection)?;
        let encoded = classical::encode(projection).map_err(|_| {
            error(
                ErrorCode::InvalidInput,
                Stage::Admission,
                "invalid Rules input projection",
            )
        })?;
        if input_key(request.context().encoding, &encoded) != request.context().input {
            return Err(error(
                ErrorCode::IdentityMismatch,
                Stage::Admission,
                "actual tensor identity differs from request",
            ));
        }
        let indices = ordered_indices(projection, request.legal())?;
        #[cfg(feature = "experimental-prepared-input")]
        let encoded = Arc::new(encoded);
        #[cfg(feature = "experimental-prepared-input")]
        let indices: Arc<[usize]> = indices.into();
        Ok(PreparedRequest {
            request,
            encoded,
            indices,
            backend: self.backend,
            #[cfg(feature = "experimental-raw-cache")]
            raw_cache: None,
        })
    }

    pub(crate) fn validate_preparation<P>(
        &self,
        request: &EvalRequest<P>,
        projection: Input<'_>,
    ) -> Result<(), ContractError> {
        let context = request.context();
        context.revision.validate()?;
        if context.model != self.model.handle()
            || context.encoding != self.model.encoding().handle
            || context.backend != self.backend
            || request.model().encoding() != self.model.encoding()
            || request.model().full_steps() != 1
            || request.model().max_batch_items() != self.model.max_batch_items()
        {
            return Err(error(
                ErrorCode::IdentityMismatch,
                Stage::Admission,
                "request does not match loaded model/encoding/backend",
            ));
        }
        if context.precision != PrecisionProfile::Fp32
            || context.compute.min_steps != 1
            || context.compute.max_steps != 1
        {
            return Err(error(
                ErrorCode::UnsupportedContract,
                Stage::Admission,
                "Maia supports only fresh full FP32 inference",
            ));
        }
        if projection.history_fill != self.history_fill
            || projection.black_to_move != (request.position().side_to_move() == Color::Black)
        {
            return Err(error(
                ErrorCode::IdentityMismatch,
                Stage::Admission,
                "Rules projection side/history profile differs",
            ));
        }
        if request.byte_budget().host < HOST_BYTES_PER_ITEM {
            return Err(error(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "request host staging reservation is too small",
            ));
        }
        Ok(())
    }

    #[cfg(feature = "experimental-prepared-input")]
    pub(crate) fn prepare_rules(
        &self,
        request: Arc<EvalRequest<rz_position::contracts::RulesState>>,
        prepared: &crate::rules_projection::PreparedRulesInput,
    ) -> Result<PreparedRequest<rz_position::contracts::RulesState>, ContractError> {
        self.validate_preparation(&request, prepared.projection.input())?;
        if prepared.key != request.context().input {
            return Err(error(
                ErrorCode::IdentityMismatch,
                Stage::Admission,
                "actual tensor identity differs from request",
            ));
        }
        Ok(PreparedRequest {
            request,
            encoded: Arc::clone(&prepared.encoded),
            indices: Arc::clone(&prepared.indices),
            backend: self.backend,
            #[cfg(feature = "experimental-raw-cache")]
            raw_cache: None,
        })
    }
}

pub struct PreparedRequest<P> {
    request: Arc<EvalRequest<P>>,
    encoded: EncodedStorage,
    indices: IndexStorage,
    backend: Digest,
    #[cfg(feature = "experimental-raw-cache")]
    pub(crate) raw_cache: Option<crate::raw_cache::RawCache>,
}

#[cfg(not(feature = "experimental-prepared-input"))]
type EncodedStorage = EncodedInput;
#[cfg(feature = "experimental-prepared-input")]
type EncodedStorage = Arc<EncodedInput>;
#[cfg(not(feature = "experimental-prepared-input"))]
type IndexStorage = Vec<usize>;
#[cfg(feature = "experimental-prepared-input")]
type IndexStorage = Arc<[usize]>;

impl<P> PreparedRequest<P> {
    pub fn request(&self) -> &Arc<EvalRequest<P>> {
        &self.request
    }
    pub fn encoded(&self) -> &EncodedInput {
        &self.encoded
    }
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Physical output conversion only. This does NOT publish Completed or
    /// consume the request. D and B must recheck current authority/deadline via
    /// EvalOutput::validate_for immediately before finalization/backup.
    pub fn output(
        &self,
        raw: &RawOutput,
        execution: ExecutionId,
    ) -> Result<EvalOutput, ContractError> {
        self.physical_output(raw, execution)
            .map_err(|failure| failure.contract)
    }

    /// Physical conversion with the bounded model-validator cause retained for
    /// the runtime owner. Common-only callers can keep using `output` above.
    pub fn physical_output(
        &self,
        raw: &RawOutput,
        execution: ExecutionId,
    ) -> Result<EvalOutput, PhysicalFailure> {
        let output = self.convert_output(raw, execution)?;
        #[cfg(feature = "experimental-raw-cache")]
        if let Some(cache) = &self.raw_cache {
            cache.stage(self.request.context(), self.encoded(), raw, execution);
        }
        Ok(output)
    }

    #[cfg(feature = "experimental-raw-cache")]
    pub(crate) fn reused_output(
        &self,
        raw: &RawOutput,
        source: ExecutionId,
    ) -> Result<EvalOutput, ContractError> {
        self.convert_output(raw, source)
            .map_err(|failure| failure.contract)
    }

    fn convert_output(
        &self,
        raw: &RawOutput,
        execution: ExecutionId,
    ) -> Result<EvalOutput, PhysicalFailure> {
        let context = self.request.context();
        if execution.epoch != context.request.epoch {
            return Err(error(
                ErrorCode::IdentityMismatch,
                Stage::Output,
                "physical execution belongs to a different epoch",
            )
            .into());
        }
        #[cfg(feature = "experimental-policy-buffer")]
        let heads = output::validate_maia_contract(raw, &self.indices);
        #[cfg(not(feature = "experimental-policy-buffer"))]
        let heads = output::validate_maia(raw, &self.indices);
        let heads = heads.map_err(|failure| {
            let kind = if matches!(failure, output::OutputError::AllocationFailed) {
                FailureKind::ResourceExhausted
            } else {
                FailureKind::NumericalFailure
            };
            PhysicalFailure::from(
                BackendError::new(kind, FailureStage::Output, "invalid Maia policy/WDL heads")
                    .with_output_cause(&failure),
            )
        })?;
        #[cfg(feature = "experimental-policy-buffer")]
        let (policy, [w, d, l]) = heads;
        #[cfg(not(feature = "experimental-policy-buffer"))]
        let [w, d, l] = heads.wdl();
        #[cfg(not(feature = "experimental-policy-buffer"))]
        let policy = heads.policy().iter().map(|&p| f64::from(p)).collect();
        Ok(EvalOutput {
            context,
            legal: self.request.legal().clone(),
            policy: LegalPolicy::try_new(policy, output::PROBABILITY_SUM_TOLERANCE)?,
            wdl: Wdl::try_new(w, d, l, output::PROBABILITY_SUM_TOLERANCE as f32)?,
            viewpoint: Viewpoint::SideToMove,
            actual: ActualCompute {
                precision: PrecisionProfile::Fp32,
                steps: 1,
                full: true,
                backend: self.backend,
                execution: Some(execution),
                provenance: CacheProvenance::Computed,
            },
        })
    }
}

/// One runtime-issued physical execution, retaining all immutable logical inputs.
pub struct PreparedBatch<P> {
    execution: ExecutionId,
    requests: Vec<PreparedRequest<P>>,
}

impl<P> PreparedBatch<P> {
    pub fn new(
        execution: ExecutionId,
        requests: Vec<PreparedRequest<P>>,
    ) -> Result<Self, ContractError> {
        let first = requests.first().ok_or_else(|| {
            error(
                ErrorCode::InvalidInput,
                Stage::Admission,
                "empty physical batch",
            )
        })?;
        if requests.len() > first.request.model().max_batch_items() || requests.len() > 16 {
            return Err(error(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "physical batch exceeds model limit",
            ));
        }
        let first_context = first.request.context();
        let mut ids = std::collections::HashSet::new();
        for item in &requests {
            let context = item.request.context();
            if context.request.epoch != execution.epoch
                || !ids.insert(context.request)
                || context.model != first_context.model
                || context.encoding != first_context.encoding
                || context.backend != first_context.backend
                || context.precision != first_context.precision
                || context.compute != first_context.compute
            {
                return Err(error(
                    ErrorCode::IdentityMismatch,
                    Stage::Admission,
                    "mixed or duplicate physical batch context",
                ));
            }
        }
        Ok(Self {
            execution,
            requests,
        })
    }
    pub fn execution(&self) -> ExecutionId {
        self.execution
    }
    pub fn requests(&self) -> &[PreparedRequest<P>] {
        &self.requests
    }
}

#[cfg(feature = "onnx")]
pub type OnnxWorker<P> =
    crate::worker::SingleWorker<PreparedBatch<P>, Result<Vec<EvalOutput>, PhysicalFailure>>;

/// Owns one ORT session on one physical thread. Runtime maps the returned lease
/// into its Backend interface; no reverse dependency from C to rz-runtime.
#[cfg(feature = "onnx")]
pub fn spawn_onnx_worker<P: Send + Sync + 'static>(
    backend: crate::onnx::OnnxBackend,
) -> Result<OnnxWorker<P>, ContractError> {
    spawn_onnx_worker_profiled(backend, None)
}

#[cfg(feature = "onnx")]
pub fn spawn_onnx_worker_profiled<P: Send + Sync + 'static>(
    mut backend: crate::onnx::OnnxBackend,
    trace: Option<rz_telemetry::source::SourceJournal<CompletionContext>>,
) -> Result<OnnxWorker<P>, ContractError> {
    validate_onnx_worker_trace(&backend, trace.is_some())?;
    crate::worker::SingleWorker::spawn_with_outcome(move |batch: &PreparedBatch<P>| {
        run_onnx_worker_batch(&mut backend, batch, trace.as_ref())
    })
    .map_err(|failure| backend_error(&failure))
}

/// Opt-in owner-preserving path. A successful session-release proof belongs
/// only to this backend and is neither a device-wide fence nor recovery from
/// an unconfirmed native failure. The caller keeps `original` on admission
/// failure and owns the corresponding recovery reservation. If native entry
/// already occurred, the caller must keep both and its original startup scope
/// alive on failure; dropping an unpublished reservation is not a fence.
#[cfg(feature = "onnx")]
pub fn spawn_onnx_worker_retained_profiled<P: Send + Sync + 'static>(
    original: &mut Option<crate::onnx::OnnxBackend>,
    reservation: &mut crate::worker::PhysicalRecoveryReservation,
    trace: Option<rz_telemetry::source::SourceJournal<CompletionContext>>,
) -> Result<OnnxWorker<P>, PhysicalFailure> {
    let backend = original.as_ref().ok_or_else(|| {
        error(
            ErrorCode::InvalidInput,
            Stage::Admission,
            "missing original ONNX backend",
        )
    })?;
    validate_onnx_worker_trace(backend, trace.is_some())?;
    crate::worker::SingleWorker::spawn_with_retained_synchronous_owner(
        original,
        reservation,
        move |backend, batch: &PreparedBatch<P>| {
            run_onnx_worker_batch(backend, batch, trace.as_ref())
        },
        |backend, _until| backend.synchronous_shutdown_proof(),
    )
    .map_err(PhysicalFailure::from)
}

#[cfg(feature = "onnx")]
fn validate_onnx_worker_trace(
    backend: &crate::onnx::OnnxBackend,
    requested: bool,
) -> Result<(), ContractError> {
    if requested && !backend.config().source_profile_supported() {
        return Err(error(
            ErrorCode::UnsupportedContract,
            Stage::Contract,
            "source profiling requires default B1 native execution",
        ));
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn run_onnx_worker_batch<P: Send + Sync + 'static>(
    backend: &mut crate::onnx::OnnxBackend,
    batch: &PreparedBatch<P>,
    trace: Option<&rz_telemetry::source::SourceJournal<CompletionContext>>,
) -> crate::worker::PhysicalRun<Result<Vec<EvalOutput>, PhysicalFailure>> {
    use rz_telemetry::source::SourceStage;
    use std::time::Instant;
    let started = trace.map(|_| Instant::now());
    let completed = (|| {
        for item in batch.requests() {
            if item.backend.0 != backend.identity()
                || item.request.context().model.manifest.0 != backend.asset_identity()
            {
                return Err(error(
                    ErrorCode::IdentityMismatch,
                    Stage::Admission,
                    "batch does not belong to this physical backend",
                )
                .into());
            }
        }
        let inputs = batch
            .requests()
            .iter()
            .map(PreparedRequest::encoded)
            .collect::<Vec<_>>();
        let raw = if let Some(trace) = trace {
            let (result, timing) = backend.run_profiled(&inputs);
            let key = batch.requests().first().map(|item| CompletionContext {
                request: item.request.context(),
                execution: Some(batch.execution()),
            });
            for (stage, span, succeeded) in [
                (SourceStage::NativePreparation, timing.preparation, true),
                (
                    SourceStage::NativeInvocation,
                    timing.invocation,
                    timing.completion_attested,
                ),
                (SourceStage::NativeOutput, timing.output, result.is_ok()),
            ] {
                if let Some((start, end)) = span {
                    trace.record(key, stage, start, end, succeeded);
                }
            }
            result
        } else {
            backend.run(&inputs)
        }
        .map_err(PhysicalFailure::from)?;
        let completed = batch
            .requests()
            .iter()
            .zip(&raw)
            .map(|(request, raw)| request.physical_output(raw, batch.execution()))
            .collect();
        backend.recycle_outputs(raw);
        completed
    })();
    if let (Some(trace), Some(start)) = (trace, started) {
        let key = batch.requests().first().map(|item| CompletionContext {
            request: item.request.context(),
            execution: Some(batch.execution()),
        });
        trace.record(
            key,
            SourceStage::PhysicalWorker,
            start,
            Instant::now(),
            completed.is_ok() && backend.physical_quarantine_cause().is_none(),
        );
    }
    if let Some(cause) = backend.physical_quarantine_cause() {
        // CUDA Run returned an error without a completion fence. No
        // physical_output/ActualCompute or completed error may escape as
        // Ready; the worker retains this batch and backend until exit.
        crate::worker::PhysicalRun::Quarantined(cause.clone())
    } else {
        crate::worker::PhysicalRun::Complete(completed)
    }
}

fn ordered_indices(
    projection: Input<'_>,
    legal: &LegalMoveView,
) -> Result<Vec<usize>, ContractError> {
    ordered_policy_indices(projection, legal.moves())
}

/// Maps an already Rules-attested ordered array. Geometry is not a legality
/// oracle; the caller retains the immutable Rules owner and legal view.
pub fn ordered_policy_indices(
    projection: Input<'_>,
    moves: &[Move],
) -> Result<Vec<usize>, ContractError> {
    let bad = || {
        error(
            ErrorCode::InvalidInput,
            Stage::Admission,
            "legal view cannot be mapped to selected Maia action space",
        )
    };
    if moves.is_empty() || moves.len() > POLICY_SIZE {
        return Err(bad());
    }
    let kings = projection.history.first().ok_or_else(bad)?.pieces
        [usize::from(projection.black_to_move)][5];
    let mut result = Vec::with_capacity(moves.len());
    let mut seen = [false; POLICY_SIZE];
    for movement in moves {
        let from = policy::canonical_square(movement.from.index(), projection.black_to_move)
            .map_err(|_| bad())?;
        let to = policy::canonical_square(movement.to.index(), projection.black_to_move)
            .map_err(|_| bad())?;
        let promotion = movement.promotion.map(|p| match p {
            Promotion::Queen => 'q',
            Promotion::Rook => 'r',
            Promotion::Bishop => 'b',
            Promotion::Knight => 'n',
        });
        let index =
            if kings & (1u64 << movement.from.index()) != 0 && from == 4 && (to == 2 || to == 6) {
                if promotion.is_some() {
                    return Err(bad());
                }
                policy::castling_index(from, if to == 2 { 0 } else { 7 })
            } else {
                policy::index(from, to, promotion)
            }
            .map_err(|_| bad())?;
        if seen[index] {
            return Err(bad());
        }
        seen[index] = true;
        result.push(index);
    }
    Ok(result)
}

fn error(code: ErrorCode, stage: Stage, detail: &'static str) -> ContractError {
    ContractError::new(code, stage, detail)
}
