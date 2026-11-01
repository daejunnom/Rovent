//! One owned native physical worker at D's cooperative Backend boundary.
//!
//! Bootstrap loads the explicit CPU or CUDA session once. Per-root runtimes share this owner and
//! never initialize ORT inside `dispatch`. Logical cancellation belongs to D/B;
//! only a physically completed worker result permits Ready and pin release.

#[cfg(any(feature = "onnx", test))]
use crate::rules_projection::ClassicalProjection;
use crate::{
    error::{BackendError, CauseCode, FailureKind, FailureStage},
    model_adapter::{backend_error, AdapterAdmission, ModelAdapter, PhysicalFailure},
    rules_projection::Lc0ModelAdapter,
    worker::{PhysicalLease, PhysicalPoll, SingleWorker},
};
use rz_contracts::*;
use rz_position::contracts::RulesState;
use rz_runtime::contracts::{ContractClock, ContractsAdapter, RuntimeRequest};
use rz_runtime::{Backend, BackendResult, Resources};
use std::{
    sync::{Arc, Mutex, MutexGuard, TryLockError},
    task::Poll,
    time::Instant,
};

pub type NativePhysicalWorker<A = Lc0ModelAdapter> =
    SingleWorker<<A as ModelAdapter>::PreparedBatch, Result<Vec<EvalOutput>, PhysicalFailure>>;
type NativePhysicalLease<A> =
    PhysicalLease<<A as ModelAdapter>::PreparedBatch, Result<Vec<EvalOutput>, PhysicalFailure>>;
type NativeDelivery<C> = Vec<BackendResult<ContractsAdapter<RulesState, C>>>;
type PreparedDispatch<C, A> = Result<(NativePhysicalLease<A>, NativeDelivery<C>), PhysicalFailure>;

pub const NATIVE_RUNTIME_OVERHEAD_BYTES: u64 = 4096;
/// Fixed first CUDA baseline declaration. This is neither measured VRAM nor a
/// total allocation hard cap; the ORT arena and external monitoring are separate.
pub const NATIVE_CUDA_ADMISSION_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_DIAGNOSTICS: usize = 4096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeWorkerOrigin {
    Injected,
    CpuOnnx,
    CudaOnnx,
}

pub use crate::adapters::lc0::NativeAdmissionPolicy;

/// Metadata captured from the admitted, already loaded CUDA session. Paths and
/// raw diagnostics stay outside this receipt; the runtime bundle digest binds
/// the complete pinned nineteen-library profile. Warm placement proves the
/// bootstrap probe, not that any search request has consumed a GPU result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeCudaMetadata {
    pub device_id: i32,
    pub arena_bytes: u64,
    pub runtime_bundle_digest: Digest,
    pub placement_profile_digest: Digest,
    pub executed_cuda_nodes: usize,
}

/// Evidence from the system CUDA runtime domain. The identity is deliberately
/// separate from `runtime_bundle_digest`: it does not identify a copied bundle.
/// The completed warm probe does not attest completion of any later request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeSystemCudaMetadata {
    pub device_id: i32,
    pub arena_bytes: u64,
    pub system_runtime_identity: Digest,
    pub placement_profile_digest: Digest,
    pub executed_cuda_nodes: usize,
    pub warm_probe_synchronous_run_returned: bool,
}

#[cfg(feature = "onnx")]
#[derive(Clone, Copy)]
enum CudaRuntimeSource {
    LegacyBundle,
    System,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeDiagnosticKind {
    DispatchRefused,
    PhysicalFailure,
    Quarantined,
}

/// Native text stays bounded and hidden by BackendError's ordinary Debug/Display.
/// `ticket` includes a burned dispatch attempt; `context.execution` is Some only
/// after the physical worker accepted that input.
#[derive(Clone, Debug)]
pub struct NativeDiagnosticReceipt {
    pub ticket: (ExecutionId, RequestId),
    pub context: CompletionContext,
    pub kind: NativeDiagnosticKind,
    pub failure: PhysicalFailure,
}

#[derive(Debug)]
pub struct NativeDiagnosticBatch {
    pub entries: Vec<NativeDiagnosticReceipt>,
    pub boundary_error: Option<ContractError>,
    pub poison_error: Option<ContractError>,
}

#[derive(Clone, Copy, Debug)]
pub struct NativeOwnerStatus {
    pub active: bool,
    pub admission_error: Option<ContractError>,
    pub occupied_diagnostics: usize,
    pub reserved_diagnostics: usize,
}

struct OwnerState<A: ModelAdapter> {
    #[cfg(feature = "experimental-batch")]
    batch_journal: Option<crate::batch_journal::BatchJournal>,
    worker: NativePhysicalWorker<A>,
    epoch: Option<ProcessEpoch>,
    last_execution: Option<u64>,
    last_request: Option<u64>,
    batch_scope: Option<(GameGeneration, RootGeneration)>,
    batch_floor: u64,
    batch_seen: Vec<u64>,
    diagnostics: Vec<DiagnosticSlot>,
    active: bool,
    shutdown_started: bool,
    closed: Option<ContractError>,
    boundary_error: Option<ContractError>,
    poison_error: Option<ContractError>,
    poison_observed: bool,
}

// The owner's 1..=4096 inline slots are reserved and initialized before launch;
// allocation failure rejects construction. Boxing a receipt would introduce an
// additional allocation while recording a physically completed failure.
#[allow(
    clippy::large_enum_variant,
    reason = "bounded inline diagnostic slots are allocated before native launch"
)]
enum DiagnosticSlot {
    Free,
    Reserved,
    Occupied(NativeDiagnosticReceipt),
}

struct OwnerInner<A: ModelAdapter> {
    max_batch: usize,
    #[cfg(feature = "experimental-notify")]
    signal: rz_runtime::CompletionSignal,
    projection: A,
    origin: NativeWorkerOrigin,
    admission_policy: AdapterAdmission,
    cuda_metadata: Option<NativeCudaMetadata>,
    system_cuda_metadata: Option<NativeSystemCudaMetadata>,
    source_trace: Option<rz_telemetry::source::SourceJournal<CompletionContext>>,
    state: Mutex<OwnerState<A>>,
}

/// Process-owned session/worker and original diagnostic receipts. Keep this
/// capability in bootstrap through serve success, failure and physical drain.
/// Identity high-water marks use constant space. Every physical attempt reserves
/// diagnostic capacity before launch; only actual failure/audit receipts occupy
/// it afterward. A successful physical result releases its reserved slot.
#[derive(Clone)]
pub struct NativeWorkerOwner<A: ModelAdapter = Lc0ModelAdapter>(Arc<OwnerInner<A>>);

impl<A: ModelAdapter> NativeWorkerOwner<A> {
    #[cfg(feature = "experimental-batch")]
    pub fn enable_batch_journal(&self) -> Result<(), ContractError> {
        let mut state = self.state();
        if state.active || state.last_request.is_some() || state.batch_journal.is_some() {
            return Err(failure(
                ErrorCode::InvalidInput,
                Stage::Admission,
                "batch journal must be enabled before any request",
            ));
        }
        state.batch_journal = Some(crate::batch_journal::BatchJournal::new(self.max_batch())?);
        Ok(())
    }
    #[cfg(feature = "experimental-batch")]
    pub fn batch_register(&self, context: EvalContext) -> Result<(), ContractError> {
        if let Some(journal) = &mut self.state().batch_journal {
            journal.register(context)?;
        }
        Ok(())
    }
    #[cfg(feature = "experimental-batch")]
    pub fn batch_logical(
        &self,
        request: RequestId,
        outcome: u8,
        cancel: bool,
    ) -> Result<(), ContractError> {
        if let Some(journal) = &mut self.state().batch_journal {
            let row = journal.row(request)?;
            if outcome > 6 || (outcome != 0 && row.7 != 0 && row.7 != outcome) {
                return Err(failure(
                    ErrorCode::InvalidInput,
                    Stage::Output,
                    "batch logical outcome cannot be rewritten",
                ));
            }
            row.6 |= cancel;
            if outcome != 0 {
                row.7 = outcome;
            }
        }
        Ok(())
    }
    #[cfg(feature = "experimental-batch")]
    pub fn batch_consume(&self, context: CompletionContext) -> Result<(), ContractError> {
        if let Some(journal) = &mut self.state().batch_journal {
            journal.consume(context)?;
        }
        Ok(())
    }
    #[cfg(feature = "experimental-batch")]
    pub fn batch_journal_receipt(
        &self,
        transfer: bool,
    ) -> Option<crate::batch_journal::BatchJournalReceipt> {
        let mut state = self.state();
        if transfer && !state.active {
            state.batch_journal.take().map(|mut j| j.take())
        } else {
            state.batch_journal.as_ref().map(|j| j.snapshot())
        }
    }
    /// Injection exercises the same physical Lease bridge without claiming NN
    /// execution. The supplied closure must satisfy SingleWorker's Run contract.
    pub fn from_worker(
        worker: NativePhysicalWorker<A>,
        projection: A,
        diagnostic_capacity: usize,
    ) -> Result<Self, PhysicalFailure> {
        Self::from_worker_with_admission(
            worker,
            projection,
            diagnostic_capacity,
            NativeAdmissionPolicy::Cpu,
        )
    }

    /// Exercises admission and quarantine with an injected physical worker.
    /// The origin remains Injected for every policy; this API cannot attest NN
    /// execution, warm placement or a loaded CUDA provider.
    pub fn from_worker_with_admission(
        worker: NativePhysicalWorker<A>,
        projection: A,
        diagnostic_capacity: usize,
        admission_policy: impl Into<AdapterAdmission>,
    ) -> Result<Self, PhysicalFailure> {
        Self::from_worker_limit(
            worker,
            projection,
            diagnostic_capacity,
            admission_policy.into(),
            1,
        )
    }
    #[cfg(feature = "experimental-batch")]
    pub fn from_worker_batched(
        worker: NativePhysicalWorker<A>,
        projection: A,
        diagnostic_capacity: usize,
        admission_policy: impl Into<AdapterAdmission>,
        max_batch: usize,
    ) -> Result<Self, PhysicalFailure> {
        Self::from_worker_limit(
            worker,
            projection,
            diagnostic_capacity,
            admission_policy.into(),
            max_batch,
        )
    }
    fn from_worker_limit(
        worker: NativePhysicalWorker<A>,
        projection: A,
        diagnostic_capacity: usize,
        admission_policy: AdapterAdmission,
        max_batch: usize,
    ) -> Result<Self, PhysicalFailure> {
        if diagnostic_capacity == 0 || diagnostic_capacity > MAX_DIAGNOSTICS {
            return Err(failure(
                ErrorCode::ResourceExhausted,
                Stage::Contract,
                "native owner requires a finite 1..=4096 diagnostic capacity",
            )
            .into());
        }
        projection.validate_runtime(max_batch)?;
        if !(1..=16).contains(&max_batch)
            || (max_batch > 1 && !cfg!(feature = "experimental-batch"))
        {
            return Err(failure(
                ErrorCode::UnsupportedContract,
                Stage::Contract,
                "runtime batch capacity is unsupported",
            )
            .into());
        }
        let mut diagnostics = Vec::new();
        diagnostics
            .try_reserve_exact(diagnostic_capacity)
            .map_err(allocation_failure)?;
        diagnostics.resize_with(diagnostic_capacity, || DiagnosticSlot::Free);
        Ok(Self(Arc::new(OwnerInner {
            max_batch,
            #[cfg(feature = "experimental-notify")]
            signal: worker.completion_signal(),
            projection,
            origin: NativeWorkerOrigin::Injected,
            admission_policy,
            cuda_metadata: None,
            system_cuda_metadata: None,
            source_trace: None,
            state: Mutex::new(OwnerState {
                #[cfg(feature = "experimental-batch")]
                batch_journal: None,
                worker,
                epoch: None,
                last_execution: None,
                last_request: None,
                batch_scope: None,
                batch_floor: 0,
                batch_seen: Vec::new(),
                diagnostics,
                active: false,
                shutdown_started: false,
                closed: None,
                boundary_error: None,
                poison_error: None,
                poison_observed: false,
            }),
        })))
    }

    pub fn origin(&self) -> NativeWorkerOrigin {
        self.0.origin
    }

    pub fn admission_policy(&self) -> AdapterAdmission {
        self.0.admission_policy
    }

    pub fn cuda_metadata(&self) -> Option<&NativeCudaMetadata> {
        self.0.cuda_metadata.as_ref()
    }

    /// Actual system-domain observation after warm placement and mapping
    /// checks, not a legacy CUDA bundle receipt or a later-run fence.
    pub fn system_cuda_metadata(&self) -> Option<&NativeSystemCudaMetadata> {
        self.0.system_cuda_metadata.as_ref()
    }

    #[cfg(feature = "experimental-notify")]
    pub fn completion_signal(&self) -> rz_runtime::CompletionSignal {
        self.0.signal.clone()
    }

    pub fn max_batch(&self) -> usize {
        self.0.max_batch
    }
    pub fn projection(&self) -> &A {
        &self.0.projection
    }

    pub fn source_trace(&self) -> Option<&rz_telemetry::source::SourceJournal<CompletionContext>> {
        self.0.source_trace.as_ref()
    }

    pub fn admission_error(&self) -> Option<ContractError> {
        self.state().closed
    }

    /// Closing admission never closes a live physical completion/pin path.
    pub fn close_admission(&self, error: ContractError) {
        let mut state = self.state();
        state.closed.get_or_insert(error);
        state.boundary_error.get_or_insert(error);
    }

    pub fn try_close_admission(&self, error: ContractError) -> Result<bool, ContractError> {
        let mut state = match self.0.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Ok(false),
            Err(TryLockError::Poisoned(poisoned)) => {
                let mut state = poisoned.into_inner();
                Self::mark_poison(&mut state);
                state
            }
        };
        state.closed.get_or_insert(error);
        state.boundary_error.get_or_insert(error);
        Ok(true)
    }

    /// Transfers original receipts only after output allocation succeeds. Poison
    /// recovery closes admission and accompanies recovered originals with an
    /// explicit boundary error; shutdown must not silently take/drop this batch.
    pub fn take_diagnostics(&self) -> Result<NativeDiagnosticBatch, ContractError> {
        let mut state = self.state();
        Self::take_from(&mut state, usize::MAX)
    }

    /// No wait for an owner lock. Transfer at most the consumer's remaining
    /// receipt budget; every unselected original and Reserved slot stays owned.
    pub fn try_take_diagnostics(
        &self,
        max_entries: usize,
    ) -> Result<Option<NativeDiagnosticBatch>, ContractError> {
        let mut state = match self.0.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(TryLockError::Poisoned(poisoned)) => {
                let mut state = poisoned.into_inner();
                Self::mark_poison(&mut state);
                state
            }
        };
        Self::take_from(&mut state, max_entries).map(Some)
    }

    pub fn try_status(&self) -> Result<Option<NativeOwnerStatus>, ContractError> {
        let state = match self.0.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(TryLockError::Poisoned(poisoned)) => {
                let mut state = poisoned.into_inner();
                Self::mark_poison(&mut state);
                state
            }
        };
        Ok(Some(NativeOwnerStatus {
            active: state.active,
            admission_error: state.closed,
            occupied_diagnostics: state
                .diagnostics
                .iter()
                .filter(|entry| matches!(entry, DiagnosticSlot::Occupied(_)))
                .count(),
            reserved_diagnostics: state
                .diagnostics
                .iter()
                .filter(|entry| matches!(entry, DiagnosticSlot::Reserved))
                .count(),
        }))
    }

    /// Terminal process shutdown, distinct from per-root logical/request drain.
    /// No owner-lock wait and no caller-side join. Once the owner lock is acquired,
    /// close future construction and dispatch, then start its one reaper only
    /// after every physical lease and reserved diagnostic has been finalized.
    /// Occupied original diagnostics remain available for the final collector.
    /// Ready(Ok) includes session/closure destruction and native-thread TLS exit.
    /// A quarantined lease remains active and pinned, so it cannot yield success.
    pub fn try_shutdown(&self) -> Poll<Result<(), PhysicalFailure>> {
        let mut state = match self.0.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Poll::Pending,
            Err(TryLockError::Poisoned(poisoned)) => {
                let mut state = poisoned.into_inner();
                Self::mark_poison(&mut state);
                state
            }
        };
        // Healthy closure is neither a native boundary error nor a discarded
        // receipt. Only this separate terminal flag closes normal admission.
        // WouldBlock remains an unlinearized Pending request: the caller must
        // retry within its deadline and cannot claim confirmed shutdown yet.
        state.shutdown_started = true;
        if state.poison_observed {
            // Delivery of the diagnostic batch does not recover a poisoned
            // owner or make a later shutdown acknowledgement successful.
            return Poll::Ready(Err(owner_poison_error().into()));
        }
        if state.active
            || state
                .diagnostics
                .iter()
                .any(|entry| matches!(entry, DiagnosticSlot::Reserved))
        {
            return Poll::Pending;
        }
        state
            .worker
            .try_shutdown()
            .map(|result| result.map_err(PhysicalFailure::from))
    }

    fn take_from(
        state: &mut OwnerState<A>,
        max_entries: usize,
    ) -> Result<NativeDiagnosticBatch, ContractError> {
        let count = state
            .diagnostics
            .iter()
            .filter(|entry| matches!(entry, DiagnosticSlot::Occupied(_)))
            .count()
            .min(max_entries);
        let mut entries = Vec::new();
        entries.try_reserve_exact(count).map_err(|_| {
            failure(
                ErrorCode::ResourceExhausted,
                Stage::Output,
                "cannot allocate native diagnostic delivery; originals retained",
            )
        })?;
        for entry in &mut state.diagnostics {
            if entries.len() == count {
                break;
            }
            if matches!(entry, DiagnosticSlot::Occupied(_)) {
                let DiagnosticSlot::Occupied(receipt) =
                    std::mem::replace(entry, DiagnosticSlot::Free)
                else {
                    unreachable!("matched occupied receipt")
                };
                entries.push(receipt);
            }
        }
        Ok(NativeDiagnosticBatch {
            entries,
            boundary_error: state.boundary_error.take(),
            poison_error: state.poison_error.take(),
        })
    }

    fn state(&self) -> MutexGuard<'_, OwnerState<A>> {
        match self.0.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                Self::mark_poison(&mut state);
                state
            }
        }
    }

    fn mark_poison(state: &mut OwnerState<A>) {
        let error = owner_poison_error();
        state.closed.get_or_insert(error);
        if !state.poison_observed {
            state.boundary_error.get_or_insert(error);
            state.poison_error = Some(error);
            state.poison_observed = true;
        }
    }

    fn record(&self, slot: usize, receipt: NativeDiagnosticReceipt) {
        let mut state = self.state();
        // Each burned attempt reserves one exclusive slot before launch. A
        // dispatch refusal has no Lease; each accepted Lease records at most once.
        state.diagnostics[slot] = DiagnosticSlot::Occupied(receipt);
    }
}

pub struct NativeRuntimeBackend<C: ContractClock + Send, A: ModelAdapter = Lc0ModelAdapter> {
    owner: NativeWorkerOwner<A>,
    clock: C,
}

pub struct NativeRuntimeLease<C: ContractClock, A: ModelAdapter = Lc0ModelAdapter> {
    physical: NativePhysicalLease<A>,
    owner: NativeWorkerOwner<A>,
    slot: usize,
    ticket: (ExecutionId, RequestId),
    context: CompletionContext,
    contexts: Vec<CompletionContext>,
    results: NativeDelivery<C>,
    consumed: bool,
    quarantined: bool,
}

impl<C: ContractClock + Send, A: ModelAdapter> NativeRuntimeBackend<C, A> {
    pub fn new(owner: NativeWorkerOwner<A>, clock: C) -> Result<Self, ContractError> {
        {
            let state = owner.state();
            if let Some(error) = state.closed {
                return Err(error);
            }
            if state.shutdown_started {
                return Err(process_shutdown_error());
            }
            if state.epoch.is_some_and(|epoch| epoch != clock.domain().0) {
                return Err(failure(
                    ErrorCode::IdentityMismatch,
                    Stage::Contract,
                    "native owner clock domain changed",
                ));
            }
        }
        Ok(Self { owner, clock })
    }

    fn quarantine(&self, lease: &mut NativeRuntimeLease<C, A>, error: BackendError) {
        let contract = backend_error(&error);
        self.owner.record(
            lease.slot,
            NativeDiagnosticReceipt {
                ticket: lease.ticket,
                context: lease.context,
                kind: NativeDiagnosticKind::Quarantined,
                failure: error.into(),
            },
        );
        self.owner.state().closed.get_or_insert(contract);
        lease.quarantined = true;
    }
}

impl<C: ContractClock + Send, A: ModelAdapter> Backend<ContractsAdapter<RulesState, C>>
    for NativeRuntimeBackend<C, A>
{
    type Lease = NativeRuntimeLease<C, A>;

    fn additional_resources(&self, requests: &[Arc<RuntimeRequest<RulesState>>]) -> Resources {
        let resources = self.owner.admission_policy().execution_resources();
        let count = requests.len() as u64;
        Resources {
            host_bytes: count.saturating_mul(resources.host_bytes),
            device_bytes: count.saturating_mul(resources.device_bytes),
            pinned_bytes: count.saturating_mul(resources.pinned_bytes),
        }
    }

    fn dispatch(
        &mut self,
        execution: &ExecutionId,
        requests: &[Arc<RuntimeRequest<RulesState>>],
    ) -> Result<Self::Lease, ContractError> {
        if requests.is_empty() || requests.len() > self.owner.max_batch() {
            return Err(failure(
                ErrorCode::UnsupportedContract,
                Stage::Admission,
                "native CPU baseline requires single-item dispatch",
            ));
        }
        let request = requests[0].eval();
        let context = request.context();
        if requests.iter().any(|item| {
            let c = item.eval().context();
            c.request.epoch != execution.epoch
                || c.selection.epoch != execution.epoch
                || c.game != context.game
                || c.root != context.root
                || item.execution().is_some_and(|bound| bound != *execution)
        }) {
            return Err(failure(
                ErrorCode::IdentityMismatch,
                Stage::Admission,
                "batch epochs/scopes/execution bindings differ",
            ));
        }
        if execution.epoch != self.clock.domain().0
            || context.request.epoch != execution.epoch
            || context.selection.epoch != execution.epoch
        {
            return Err(failure(
                ErrorCode::IdentityMismatch,
                Stage::Admission,
                "native physical/request/clock epoch differs",
            ));
        }
        let ticket = (*execution, context.request);
        let mut contexts = Vec::new();
        contexts
            .try_reserve_exact(requests.len() - 1)
            .map_err(|_| {
                failure(
                    ErrorCode::ResourceExhausted,
                    Stage::Admission,
                    "batch completion context allocation failed",
                )
            })?;
        contexts.extend(requests.iter().skip(1).map(|item| CompletionContext {
            request: item.eval().context(),
            execution: Some(*execution),
        }));
        let slot = {
            let mut state = self.owner.state();
            if let Some(error) = state.closed {
                return Err(error);
            }
            if state.shutdown_started {
                return Err(process_shutdown_error());
            }
            if state.epoch.is_some_and(|epoch| epoch != execution.epoch)
                || state
                    .last_execution
                    .is_some_and(|last| execution.sequence <= last)
                || (self.owner.max_batch() == 1
                    && state
                        .last_request
                        .is_some_and(|last| context.request.sequence <= last))
            {
                return Err(failure(
                    ErrorCode::IdentityMismatch,
                    Stage::Admission,
                    "native physical identity reused or moved backwards",
                ));
            }
            if self.owner.max_batch() > 1 {
                let scope = (context.game, context.root);
                if state.batch_scope != Some(scope) {
                    state.batch_floor = state.last_request.unwrap_or(0);
                    state.batch_seen.clear();
                    state.batch_scope = Some(scope);
                }
                if state.batch_seen.len() + requests.len() > 1024 {
                    return Err(failure(
                        ErrorCode::ResourceExhausted,
                        Stage::Admission,
                        "experimental root physical request ledger exhausted",
                    ));
                }
                state.batch_seen.try_reserve(requests.len()).map_err(|_| {
                    failure(
                        ErrorCode::ResourceExhausted,
                        Stage::Admission,
                        "batch identity allocation failed",
                    )
                })?;
                let mut ids = Vec::with_capacity(requests.len());
                for item in requests {
                    let sequence = item.eval().context().request.sequence;
                    if sequence <= state.batch_floor
                        || state.batch_seen.contains(&sequence)
                        || ids.contains(&sequence)
                    {
                        return Err(failure(
                            ErrorCode::IdentityMismatch,
                            Stage::Admission,
                            "batch physical request identity reused",
                        ));
                    }
                    ids.push(sequence);
                }
                state.batch_seen.extend(ids);
            }
            state.epoch = Some(execution.epoch);
            state.last_execution = Some(execution.sequence);
            state.last_request = requests
                .iter()
                .map(|r| r.eval().context().request.sequence)
                .chain(state.last_request)
                .max();
            let Some(slot) = state
                .diagnostics
                .iter()
                .position(|entry| matches!(entry, DiagnosticSlot::Free))
            else {
                let error = failure(
                    ErrorCode::ResourceExhausted,
                    Stage::Admission,
                    "native diagnostic store is full; admission closed",
                );
                state.closed = Some(error);
                state.boundary_error = Some(error);
                return Err(error);
            };
            state.diagnostics[slot] = DiagnosticSlot::Reserved;
            slot
        };
        let prepared: PreparedDispatch<C, A> = (|| {
            let mut items = Vec::new();
            items
                .try_reserve_exact(requests.len())
                .map_err(allocation_failure)?;
            for item in requests {
                let request = item.eval();
                request
                    .deadline()
                    .accepts(self.clock.domain(), self.clock.now())?;
                if request.cancel_token().is_canceled() {
                    return Err(failure(
                        ErrorCode::Canceled,
                        Stage::Admission,
                        "native batch canceled before preparation",
                    )
                    .into());
                }
                let encoding_started = self.owner.source_trace().map(|_| Instant::now());
                let prepared = self.owner.projection().prepare(Arc::clone(request));
                if let (Some(trace), Some(start)) = (self.owner.source_trace(), encoding_started) {
                    trace.record(
                        Some(CompletionContext {
                            request: request.context(),
                            execution: Some(*execution),
                        }),
                        rz_telemetry::source::SourceStage::EncodingPreparation,
                        start,
                        Instant::now(),
                        prepared.is_ok(),
                    );
                }
                items.push(prepared?);
            }
            let batch = self.owner.projection().batch(*execution, items)?;
            let mut results = Vec::new();
            results
                .try_reserve_exact(requests.len())
                .map_err(allocation_failure)?;
            for item in requests {
                item.eval()
                    .deadline()
                    .accepts(self.clock.domain(), self.clock.now())?;
                if item.eval().cancel_token().is_canceled() {
                    return Err(failure(
                        ErrorCode::Canceled,
                        Stage::Admission,
                        "native batch canceled during preparation",
                    )
                    .into());
                }
            }
            let mut state = self.owner.state();
            if let Some(error) = state.closed {
                return Err(error.into());
            }
            if state.shutdown_started {
                return Err(process_shutdown_error().into());
            }
            if state.active {
                return Err(failure(
                    ErrorCode::ResourceExhausted,
                    Stage::Admission,
                    "native owner still holds a physical lease",
                )
                .into());
            }
            if requests[0]
                .execution()
                .is_some_and(|bound| bound != *execution)
            {
                return Err(failure(
                    ErrorCode::IdentityMismatch,
                    Stage::Admission,
                    "native request is already bound to a different execution",
                )
                .into());
            }
            #[cfg(feature = "experimental-batch")]
            let journal_contexts = state.batch_journal.as_ref().map(|_| {
                let mut contexts = [context; 16];
                for (slot, request) in contexts.iter_mut().zip(requests) {
                    *slot = request.eval().context();
                }
                contexts
            });
            #[cfg(feature = "experimental-batch")]
            if let (Some(journal), Some(contexts)) = (&mut state.batch_journal, &journal_contexts) {
                journal.dispatch(&contexts[..requests.len()], *execution)?;
            }
            let physical = match state.worker.submit(batch) {
                Ok(lease) => lease,
                Err(error) => {
                    #[cfg(feature = "experimental-batch")]
                    if let (Some(journal), Some(contexts)) =
                        (&mut state.batch_journal, &journal_contexts)
                    {
                        journal.undo_dispatch(&contexts[..requests.len()]);
                    }
                    return Err(PhysicalFailure::from(error));
                }
            };
            // Every fallible operation precedes successful native handoff. The
            // returned lease is registered even if the worker already failed.
            state.active = true;
            Ok((physical, results))
        })();
        let (physical, results) = match prepared {
            Ok(value) => value,
            Err(error) => {
                let contract = error.contract;
                self.owner.record(
                    slot,
                    NativeDiagnosticReceipt {
                        ticket,
                        context: CompletionContext {
                            request: context,
                            execution: None,
                        },
                        kind: NativeDiagnosticKind::DispatchRefused,
                        failure: error,
                    },
                );
                return Err(contract);
            }
        };
        Ok(NativeRuntimeLease {
            physical,
            owner: self.owner.clone(),
            slot,
            ticket,
            context: CompletionContext {
                request: context,
                execution: Some(*execution),
            },
            contexts,
            results,
            consumed: false,
            quarantined: false,
        })
    }

    fn poll(&mut self, lease: &mut Self::Lease) -> Poll<NativeDelivery<C>> {
        if lease.consumed || lease.quarantined {
            return Poll::Pending;
        }
        if !Arc::ptr_eq(&self.owner.0, &lease.owner.0) {
            // Preserve the actual lease owner; a foreign caller cannot complete it.
            let error = failure(
                ErrorCode::IdentityMismatch,
                Stage::Backend,
                "native lease belongs to a different owner",
            );
            lease.owner.record(
                lease.slot,
                NativeDiagnosticReceipt {
                    ticket: lease.ticket,
                    context: lease.context,
                    kind: NativeDiagnosticKind::Quarantined,
                    failure: error.into(),
                },
            );
            let mut state = lease.owner.state();
            state.closed.get_or_insert(error);
            state.boundary_error.get_or_insert(error);
            lease.quarantined = true;
            return Poll::Pending;
        }
        let result = match lease.physical.poll() {
            PhysicalPoll::Pending | PhysicalPoll::Consumed => return Poll::Pending,
            PhysicalPoll::Quarantined => {
                let cause = match lease.physical.quarantine_cause() {
                    Ok(Some(cause)) | Err(cause) => cause,
                    Ok(None) => BackendError::new(
                        FailureKind::BackendFailure,
                        FailureStage::Backend,
                        "native completion disconnected without physical completion proof",
                    ),
                };
                self.quarantine(lease, cause);
                return Poll::Pending;
            }
            PhysicalPoll::Ready(result) => result,
        };
        lease.consumed = true;
        let outputs = match result {
            Ok(outputs) if outputs.len() == 1 + lease.contexts.len() => Ok(outputs),
            Ok(outputs) => Err(BackendError::new(
                FailureKind::BackendFailure,
                FailureStage::Output,
                "native output count differs from physical inputs",
            )
            .with_diagnostic(
                "NativeOutputCount",
                &format!("expected={} actual={}", lease.contexts.len(), outputs.len()),
            )
            .into()),
            Err(error) => Err(error),
        };
        #[cfg(feature = "experimental-batch")]
        {
            let mut state = self.owner.state();
            if let Some(journal) = &mut state.batch_journal {
                if let Err(error) = journal.physical(
                    std::iter::once(lease.context).chain(lease.contexts.iter().copied()),
                    outputs.is_ok(),
                ) {
                    state.closed.get_or_insert(error);
                    state.boundary_error.get_or_insert(error);
                }
            }
        }
        match outputs {
            Ok(outputs) => {
                self.owner.state().diagnostics[lease.slot] = DiagnosticSlot::Free;
                for (context, output) in std::iter::once(&lease.context)
                    .chain(&lease.contexts)
                    .zip(outputs)
                {
                    lease.results.push(BackendResult {
                        request_id: context.request.request,
                        output: Ok(output),
                    });
                }
            }
            Err(error) => {
                let contract = error.contract;
                self.owner.record(
                    lease.slot,
                    NativeDiagnosticReceipt {
                        ticket: lease.ticket,
                        context: lease.context,
                        kind: NativeDiagnosticKind::PhysicalFailure,
                        failure: error,
                    },
                );
                for context in std::iter::once(&lease.context).chain(&lease.contexts) {
                    lease.results.push(BackendResult {
                        request_id: context.request.request,
                        output: Err(contract),
                    });
                }
            }
        }
        self.owner.state().active = false;
        Poll::Ready(std::mem::take(&mut lease.results))
    }
}

fn allocation_failure(error: std::collections::TryReserveError) -> PhysicalFailure {
    BackendError::new(
        FailureKind::ResourceExhausted,
        FailureStage::Admission,
        "native bridge allocation failed before launch",
    )
    .with_external_cause(CauseCode::InputAllocation, &error)
    .into()
}

fn failure(code: ErrorCode, stage: Stage, detail: &'static str) -> ContractError {
    ContractError::new(code, stage, detail)
}

fn process_shutdown_error() -> ContractError {
    failure(
        ErrorCode::Canceled,
        Stage::Admission,
        "native owner process shutdown has closed admission",
    )
}

impl NativeWorkerOwner<Lc0ModelAdapter> {
    /// Consumes an already loaded, verified CPU session. Native load/provider
    /// initialization is a bootstrap operation, outside every search deadline.
    #[cfg(feature = "onnx")]
    pub fn from_onnx(
        backend: crate::onnx::OnnxBackend,
        projection: ClassicalProjection,
        diagnostic_capacity: usize,
    ) -> Result<Self, PhysicalFailure> {
        Self::from_onnx_profiled(backend, projection, diagnostic_capacity, None)
    }

    #[cfg(feature = "onnx")]
    pub fn from_onnx_profiled(
        backend: crate::onnx::OnnxBackend,
        projection: ClassicalProjection,
        diagnostic_capacity: usize,
        trace: Option<rz_telemetry::source::SourceJournal<CompletionContext>>,
    ) -> Result<Self, PhysicalFailure> {
        Self::from_onnx_limit(backend, projection, diagnostic_capacity, 1, trace)
    }
    #[cfg(all(feature = "onnx", feature = "experimental-batch"))]
    pub fn from_onnx_batched(
        backend: crate::onnx::OnnxBackend,
        projection: ClassicalProjection,
        diagnostic_capacity: usize,
    ) -> Result<Self, PhysicalFailure> {
        let max_batch = backend.config().max_batch;
        Self::from_onnx_limit(backend, projection, diagnostic_capacity, max_batch, None)
    }
    #[cfg(feature = "onnx")]
    fn from_onnx_limit(
        backend: crate::onnx::OnnxBackend,
        projection: ClassicalProjection,
        diagnostic_capacity: usize,
        max_batch: usize,
        trace: Option<rz_telemetry::source::SourceJournal<CompletionContext>>,
    ) -> Result<Self, PhysicalFailure> {
        if !matches!(backend.config().provider, crate::onnx::Provider::Cpu)
            || backend.config().max_batch != max_batch
            || projection.backend().0 != backend.identity()
            || projection.model().handle().manifest.0 != backend.asset_identity()
        {
            return Err(failure(
                ErrorCode::IdentityMismatch,
                Stage::Contract,
                "native CPU owner differs from loaded provider/model/batch identity",
            )
            .into());
        }
        let worker = crate::contracts::spawn_onnx_worker_profiled(backend, trace.clone())?;
        let mut owner = Self::from_worker_limit(
            worker,
            projection,
            diagnostic_capacity,
            NativeAdmissionPolicy::Cpu.into(),
            max_batch,
        )?;
        // No clone has escaped this constructor; the actual origin is immutable.
        let inner = Arc::get_mut(&mut owner.0).expect("new native owner is exclusively owned");
        inner.origin = NativeWorkerOrigin::CpuOnnx;
        inner.source_trace = trace;
        Ok(owner)
    }

    /// Consumes a verified CUDA B1/FP32 session after the actual warm placement
    /// probe. Bootstrap, not a search deadline, owns initialization. The first
    /// profile uses one intra-op thread and a one-GiB ORT arena declaration.
    /// No unchecked caller value can issue CudaOnnx origin or placement metadata.
    #[cfg(feature = "onnx")]
    pub fn from_cuda_onnx(
        backend: crate::onnx::OnnxBackend,
        projection: ClassicalProjection,
        diagnostic_capacity: usize,
    ) -> Result<Self, PhysicalFailure> {
        Self::from_cuda_onnx_profiled(backend, projection, diagnostic_capacity, None)
    }

    #[cfg(feature = "onnx")]
    pub fn from_cuda_onnx_profiled(
        backend: crate::onnx::OnnxBackend,
        projection: ClassicalProjection,
        diagnostic_capacity: usize,
        trace: Option<rz_telemetry::source::SourceJournal<CompletionContext>>,
    ) -> Result<Self, PhysicalFailure> {
        Self::from_cuda_onnx_limit(
            backend,
            projection,
            diagnostic_capacity,
            1,
            trace,
            CudaRuntimeSource::LegacyBundle,
        )
    }
    #[cfg(all(feature = "onnx", feature = "experimental-batch"))]
    pub fn from_cuda_onnx_batched(
        backend: crate::onnx::OnnxBackend,
        projection: ClassicalProjection,
        diagnostic_capacity: usize,
    ) -> Result<Self, PhysicalFailure> {
        let max_batch = backend.config().max_batch;
        Self::from_cuda_onnx_limit(
            backend,
            projection,
            diagnostic_capacity,
            max_batch,
            None,
            CudaRuntimeSource::LegacyBundle,
        )
    }

    /// The system-runtime owner is explicit and cannot issue a historical
    /// bundle witness. The same physical worker and failure handling apply.
    #[cfg(feature = "onnx")]
    pub fn from_system_cuda_onnx(
        backend: crate::onnx::OnnxBackend,
        projection: ClassicalProjection,
        diagnostic_capacity: usize,
    ) -> Result<Self, PhysicalFailure> {
        Self::from_system_cuda_onnx_profiled(backend, projection, diagnostic_capacity, None)
    }

    #[cfg(feature = "onnx")]
    pub fn from_system_cuda_onnx_profiled(
        backend: crate::onnx::OnnxBackend,
        projection: ClassicalProjection,
        diagnostic_capacity: usize,
        trace: Option<rz_telemetry::source::SourceJournal<CompletionContext>>,
    ) -> Result<Self, PhysicalFailure> {
        Self::from_cuda_onnx_limit(
            backend,
            projection,
            diagnostic_capacity,
            1,
            trace,
            CudaRuntimeSource::System,
        )
    }

    #[cfg(all(feature = "onnx", feature = "experimental-batch"))]
    pub fn from_system_cuda_onnx_batched(
        backend: crate::onnx::OnnxBackend,
        projection: ClassicalProjection,
        diagnostic_capacity: usize,
    ) -> Result<Self, PhysicalFailure> {
        let max_batch = backend.config().max_batch;
        Self::from_cuda_onnx_limit(
            backend,
            projection,
            diagnostic_capacity,
            max_batch,
            None,
            CudaRuntimeSource::System,
        )
    }

    #[cfg(feature = "onnx")]
    fn from_cuda_onnx_limit(
        backend: crate::onnx::OnnxBackend,
        projection: ClassicalProjection,
        diagnostic_capacity: usize,
        max_batch: usize,
        trace: Option<rz_telemetry::source::SourceJournal<CompletionContext>>,
        runtime_source: CudaRuntimeSource,
    ) -> Result<Self, PhysicalFailure> {
        // Preserve the actual native cause of an earlier uncertain Run before
        // reporting a generic profile mismatch. Drop retains its session/input.
        if let Some(cause) = backend.physical_quarantine_cause() {
            return Err(cause.clone().into());
        }
        let crate::onnx::Provider::Cuda {
            device_id,
            arena_bytes,
        } = backend.config().provider
        else {
            return Err(failure(
                ErrorCode::IdentityMismatch,
                Stage::Contract,
                "native CUDA owner requires an explicitly loaded CUDA provider",
            )
            .into());
        };
        if device_id != 0
            || arena_bytes != backend.asset_profile().cuda_arena_bytes()
            || backend.config().max_batch != max_batch
            || backend.config().intra_threads != 1
            || projection.backend().0 != backend.identity()
            || projection.model().handle().manifest.0 != backend.asset_identity()
            || backend.has_unconfirmed_physical_completion()
        {
            return Err(failure(
                ErrorCode::IdentityMismatch,
                Stage::Contract,
                "native CUDA owner differs from verified device0/B1/thread1/model/backend/arena profile",
            )
            .into());
        }
        let evidence = backend
            .cuda_evidence()
            .filter(|e| e.executed_cuda_nodes > 0)
            .ok_or_else(|| {
                failure(
                    ErrorCode::IdentityMismatch,
                    Stage::Contract,
                    "native CUDA owner requires actual warm CUDA kernel placement",
                )
            })?;
        let (metadata, system_metadata) = match runtime_source {
            CudaRuntimeSource::LegacyBundle => {
                let digest = backend.runtime_bundle_digest().ok_or_else(|| {
                    failure(
                        ErrorCode::IdentityMismatch,
                        Stage::Contract,
                        "native CUDA owner requires the complete pinned runtime bundle",
                    )
                })?;
                (
                    Some(NativeCudaMetadata {
                        device_id,
                        arena_bytes: arena_bytes as u64,
                        runtime_bundle_digest: Digest(digest),
                        placement_profile_digest: Digest(evidence.profile_sha256),
                        executed_cuda_nodes: evidence.executed_cuda_nodes,
                    }),
                    None,
                )
            }
            CudaRuntimeSource::System => {
                let digest = backend.system_cuda_runtime_identity().ok_or_else(|| {
                    failure(
                        ErrorCode::IdentityMismatch,
                        Stage::Contract,
                        "native CUDA owner requires an observed system runtime",
                    )
                })?;
                (
                    None,
                    Some(NativeSystemCudaMetadata {
                        device_id,
                        arena_bytes: arena_bytes as u64,
                        system_runtime_identity: Digest(digest),
                        placement_profile_digest: Digest(evidence.profile_sha256),
                        executed_cuda_nodes: evidence.executed_cuda_nodes,
                        // `cuda_evidence` is assigned only after a successful
                        // synchronous warm Run and placement-profile validation.
                        warm_probe_synchronous_run_returned: true,
                    }),
                )
            }
        };
        // A successful historical probe alone does not authorize current maps.
        // Loader failure stays typed and latched, with all native pins retained.
        backend.verify_cuda_runtime_mappings()?;
        let admission = backend.asset_profile().admission_policy();
        let worker = crate::contracts::spawn_onnx_worker_profiled(backend, trace.clone())?;
        let mut owner = Self::from_worker_limit(
            worker,
            projection,
            diagnostic_capacity,
            admission.into(),
            max_batch,
        )?;
        let inner = Arc::get_mut(&mut owner.0).expect("new native owner is exclusively owned");
        inner.origin = NativeWorkerOrigin::CudaOnnx;
        inner.cuda_metadata = metadata;
        inner.system_cuda_metadata = system_metadata;
        inner.source_trace = trace;
        Ok(owner)
    }
}

fn owner_poison_error() -> ContractError {
    failure(
        ErrorCode::BackendFailure,
        Stage::Backend,
        "native physical/diagnostic owner poisoned; admission closed",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::PreparedBatch;
    use crate::contracts::{encoding_manifest, MaiaBinding};
    use rz_encoding::classical::HistoryFill;

    fn owner() -> NativeWorkerOwner {
        let binding = MaiaBinding::new(
            ModelHandle {
                owner: OwnerId(911),
                slot: 1,
                generation: SlotGeneration(1),
                manifest: Digest([91; 32]),
            },
            EncodingHandle {
                owner: OwnerId(911),
                slot: 2,
                generation: SlotGeneration(1),
                manifest: encoding_manifest(HistoryFill::No),
            },
            HistoryFill::No,
            Digest([92; 32]),
            1,
        )
        .unwrap();
        let worker = SingleWorker::spawn(|_: &PreparedBatch<RulesState>| Ok(Vec::new())).unwrap();
        NativeWorkerOwner::from_worker(worker, ClassicalProjection::new(binding), 2).unwrap()
    }

    fn occupied(owner: &NativeWorkerOwner) -> NativeDiagnosticReceipt {
        let live = rz_position::contracts::ContractPosition::new(
            OwnerId(912),
            rz_position::Position::startpos(),
        );
        let state = live.export().unwrap();
        let request = EvalContext {
            revision: CONTRACT_REVISION,
            request: RequestId::new(ProcessEpoch(913), 1),
            selection: SelectionId::new(ProcessEpoch(913), 1),
            game: GameGeneration(1),
            root: RootGeneration(1),
            state: state.snapshot().identity(),
            legal_order: state.legal_moves().order(),
            input: owner
                .projection()
                .input_key(state.rules(), state.legal_moves().moves())
                .unwrap(),
            model: owner.projection().model().handle(),
            encoding: owner.projection().model().encoding().handle,
            precision: PrecisionProfile::Fp32,
            compute: ComputeBudget {
                min_steps: 1,
                max_steps: 1,
                require_full: true,
            },
            backend: owner.projection().backend(),
        };
        let execution = ExecutionId::new(ProcessEpoch(913), 1);
        NativeDiagnosticReceipt {
            ticket: (execution, request.request),
            context: CompletionContext {
                request,
                execution: Some(execution),
            },
            kind: NativeDiagnosticKind::PhysicalFailure,
            failure: BackendError::new(
                FailureKind::BackendFailure,
                FailureStage::Backend,
                "original native failure before owner poison",
            )
            .into(),
        }
    }

    #[test]
    fn busy_owner_lock_is_unlinearized_pending_then_normal_shutdown_is_clean() {
        let owner = owner();
        {
            let state = owner.state();
            assert!(matches!(owner.try_shutdown(), Poll::Pending));
            assert!(!state.shutdown_started);
        }
        let until = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            match owner.try_shutdown() {
                Poll::Ready(result) => {
                    result.unwrap();
                    break;
                }
                Poll::Pending => {
                    assert!(
                        std::time::Instant::now() < until,
                        "shutdown reaper test budget"
                    );
                    std::thread::yield_now();
                }
            }
        }
        assert!(owner.state().shutdown_started);
        let batch = owner.take_diagnostics().unwrap();
        assert!(batch.entries.is_empty());
        assert!(batch.boundary_error.is_none());
        assert!(batch.poison_error.is_none());
    }

    #[test]
    fn poison_preserves_first_close_or_store_full_and_existing_slots() {
        for original in [
            failure(
                ErrorCode::BackendFailure,
                Stage::Backend,
                "original explicit owner close",
            ),
            failure(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "original native diagnostic store full",
            ),
        ] {
            let owner = owner();
            let receipt = occupied(&owner);
            let expected = receipt.context;
            {
                let mut state = owner.state();
                state.diagnostics[0] = DiagnosticSlot::Reserved;
                state.diagnostics[1] = DiagnosticSlot::Occupied(receipt);
            }
            owner.close_admission(original);
            let poisoned = owner.clone();
            assert!(std::thread::spawn(move || {
                let _state = poisoned.0.state.lock().unwrap();
                panic!("injected native owner mutex poison");
            })
            .join()
            .is_err());
            let mut batch = owner.try_take_diagnostics(1).unwrap().unwrap();
            assert_eq!(batch.boundary_error, Some(original));
            assert_eq!(batch.poison_error.unwrap().code, ErrorCode::BackendFailure);
            let receipt = batch.entries.pop().unwrap();
            assert_eq!(receipt.context, expected);
            assert_eq!(
                receipt.failure.backend.unwrap().detail,
                "original native failure before owner poison"
            );
            let snapshot = owner.try_status().unwrap().unwrap();
            assert_eq!(snapshot.admission_error, Some(original));
            assert_eq!(snapshot.reserved_diagnostics, 1);
            assert_eq!(snapshot.occupied_diagnostics, 0);
            let acknowledged = owner.take_diagnostics().unwrap();
            assert!(acknowledged.entries.is_empty());
            assert!(acknowledged.boundary_error.is_none());
            assert!(
                acknowledged.poison_error.is_none(),
                "a retained poison fact is delivered once"
            );
            assert!(matches!(owner.try_shutdown(), Poll::Ready(Err(error))
                if error.contract == owner_poison_error() && error.backend.is_none()));
            assert_eq!(
                owner.try_status().unwrap().unwrap().admission_error,
                Some(original)
            );
        }
    }
}
