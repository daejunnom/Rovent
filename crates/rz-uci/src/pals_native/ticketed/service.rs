use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use rz_contracts::pals::{RepresentationKey, Role, SearchAuthority};
use rz_contracts::pals_adaptive::{ReviewViewSeal, RoleTicketEnvelope};
use rz_contracts::{Deadline, Digest, ExecutionId};
use rz_runtime::contracts::{ContractClock, ContractSystemClock};
use rz_runtime::{Clock, CompletionSignal};
use rz_search::pals::coordinator::{
    NativeAcceptanceIssuer, NativeAcceptancePermit, NativeAcceptanceReceipt,
    NativeAcceptanceVerifier, PhysicalRoleCompletion, RoleCompletion, RoleFailure,
};

#[path = "service_recovery.rs"]
mod recoverable;

const CAPACITY: usize = 16;
const ADAPTIVE_COMMAND_CAPACITY: usize = 8;
// Stop ordinary admission at 64 queued observations. At most sixteen work
// commands, sixteen acknowledgements, sixteen unsubmitted retirements and the
// active completion/finish can then publish. Their terminal reserve is separate.
const REPORT_HIGH_WATER: usize = CAPACITY * 4;
const EVENTS: usize = CAPACITY * 12;

/// Monotonic observation on the original factory thread. Entering native code
/// does not by itself prove either device execution or physical completion.
pub struct FactoryNativeEntry {
    entered: bool,
}

pub enum TicketNativeRecoveryProbe {
    Pending,
    Unknown,
    Confirmed(rz_eval::worker::PhysicalRecoveryConfirmation),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TicketNativeRelease {
    Pending,
    Released,
    Unknown,
}
impl FactoryNativeEntry {
    pub fn mark_native_entered(&mut self) {
        self.entered = true;
    }
    pub fn with_native<T>(&mut self, call: impl FnOnce() -> T) -> T {
        self.mark_native_entered();
        call()
    }
    pub fn entered(&self) -> bool {
        self.entered
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeFactoryStartupStatus {
    Pending,
    Ready,
    BeforeNative,
    PhysicalUnknown,
    Unobserved,
}

pub struct FactoryStartupCustody<C> {
    owner: C,
    entry: FactoryNativeEntry,
    recovery_key: super::recovery::TicketNativeRecoveryKey,
}
impl<C> FactoryStartupCustody<C> {
    pub fn bind_inner(
        &mut self,
        key: &rz_eval::worker::PhysicalRecoveryKey,
        epoch: u64,
    ) -> Result<(), super::recovery::TicketRecoveryError> {
        if self.entry.entered {
            return Err(super::recovery::TicketRecoveryError::WrongState);
        }
        self.recovery_key.bind_inner_before_native(key, epoch)
    }
    pub fn parts(&mut self) -> (&mut C, &mut FactoryNativeEntry) {
        (&mut self.owner, &mut self.entry)
    }
}

/// Operations are invoked only on the original factory worker while it is
/// parked. Absent same-owner evidence, the default remains Unknown.
pub trait FactoryStartupRecovery: Send + 'static {
    fn reobserve(&mut self, _: Instant) -> Result<TicketNativeRecoveryProbe, RoleFailure> {
        Ok(TicketNativeRecoveryProbe::Unknown)
    }
    fn release(
        &mut self,
        _: &rz_eval::worker::PhysicalRecoveryConfirmation,
        _: Instant,
    ) -> Result<TicketNativeRelease, RoleFailure> {
        Ok(TicketNativeRelease::Unknown)
    }
}

pub struct FactoryStartupSpawnRejection<C> {
    pub error: TicketServiceError,
    pub owner: C,
    /// Present only when a never-started original thread still needs join.
    pub recovery_key: Option<super::recovery::TicketNativeRecoveryKey>,
    /// An original thread that never received a start token; caller may poll
    /// and join it without delaying unrelated work.
    pub unstarted_thread: Option<JoinHandle<()>>,
}
impl<C> std::fmt::Debug for FactoryStartupSpawnRejection<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FactoryStartupSpawnRejection")
            .field("error", &self.error)
            .field("original_owner_returned", &true)
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct PreparationId(pub u64);

/// Not a submitted neural request. Preparation happens on the native owner;
/// its exact RepresentationKey returns before the manager issues RequestId.
pub struct TicketPreparation<V, Q> {
    pub question: Digest,
    pub seal: ReviewViewSeal,
    pub role: Role,
    pub deadline: Deadline,
    pub view: Arc<V>,
    pub query: Q,
}

/// A read-only handle to the same native ticket cancellation word. It grants
/// no right to cancel, accept, consume, or establish physical completion.
#[derive(Clone)]
pub struct NativeTicketCancelObservation {
    original: Arc<AtomicBool>,
}
impl NativeTicketCancelObservation {
    pub fn is_canceled(&self) -> bool {
        self.original.load(Ordering::Acquire)
    }
    /// Establish the ordering of an optional submission against cancellation.
    /// A successful claim does not dispatch, accept or complete any work.
    pub fn try_claim_submission(&self) -> bool {
        self.original
            .compare_exchange(false, false, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
}

/// Borrowed context produced only on the native worker after it validates
/// the original ticket against its held preparation. It conveys no acceptance
/// or physical-completion authority and cannot be constructed or cloned by a
/// backend caller. The exact question and prepared input remain borrowed.
pub struct TicketOwnerDispatchContext<'a, V, Q, P> {
    verifier: &'a NativeAcceptanceVerifier,
    ticket: &'a RoleTicketEnvelope,
    question: &'a TicketPreparation<V, Q>,
    prepared: &'a P,
    canceled: &'a Arc<AtomicBool>,
}

impl<'a, V, Q, P> TicketOwnerDispatchContext<'a, V, Q, P> {
    pub fn ticket(&self) -> &'a RoleTicketEnvelope {
        self.ticket
    }
    pub fn question(&self) -> &'a TicketPreparation<V, Q> {
        self.question
    }
    pub fn prepared(&self) -> &'a P {
        self.prepared
    }
    /// Retain the same original native cancellation observation, not a new
    /// cancel authority or evidence that the NN request was accepted.
    pub fn original_cancellation(&self) -> NativeTicketCancelObservation {
        NativeTicketCancelObservation {
            original: Arc::clone(self.canceled),
        }
    }
    pub fn same_owner(&self, verifier: &NativeAcceptanceVerifier) -> bool {
        self.verifier.same_owner(verifier)
    }

    /// Internal projection retains the same ticket and exact original view.
    /// The projection owner must first validate that `prepared` is the
    /// projection retained with this original service preparation.
    pub(crate) fn project<Q2, P2>(
        self,
        question: &'a TicketPreparation<V, Q2>,
        prepared: &'a P2,
    ) -> Result<TicketOwnerDispatchContext<'a, V, Q2, P2>, RoleFailure> {
        if question.question != self.question.question
            || question.seal != self.question.seal
            || question.role != self.question.role
            || question.deadline != self.question.deadline
            || !Arc::ptr_eq(&question.view, &self.question.view)
        {
            return Err(RoleFailure::InvalidOutput);
        }
        Ok(TicketOwnerDispatchContext {
            verifier: self.verifier,
            ticket: self.ticket,
            question,
            prepared,
            canceled: self.canceled,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedTicketSeal {
    pub preparation: PreparationId,
    pub question: Digest,
    pub view: ReviewViewSeal,
    pub representation: RepresentationKey,
    pub role: Role,
    pub original_deadline: Deadline,
}
impl PreparedTicketSeal {
    fn matches(self, ticket: &RoleTicketEnvelope) -> bool {
        self.question == ticket.question
            && self.view == ticket.view
            && self.representation == ticket.representation
            && self.role == ticket.role
            && self.original_deadline == ticket.original_deadline
    }
}

pub struct TicketPhysicalResult<O> {
    pub physical: PhysicalRoleCompletion,
    pub output: Result<O, RoleFailure>,
}

/// A collection/semantic report, separate from the owner's physical result.
/// At most two reports can accompany a backend callback; neither is proof of
/// acceptance, NN completion, or a physical fence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TicketNativeObservationStage {
    Submitted,
    Completed,
    Retired,
    Retrieval,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TicketNativeObservationDisposition {
    Observation,
    RequiredAfterPhysical,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TicketNativeObservationFailure {
    pub ticket: RoleTicketEnvelope,
    pub stage: TicketNativeObservationStage,
    pub disposition: TicketNativeObservationDisposition,
    pub cause: RoleFailure,
}

/// Bounded observations carried atomically with their original physical
/// terminal. Empty is the compatibility default for existing backends.
#[derive(Default, Debug)]
pub struct TicketNativeObservationFailures {
    entries: [Option<TicketNativeObservationFailure>; 2],
}
impl TicketNativeObservationFailures {
    pub fn iter(&self) -> impl Iterator<Item = &TicketNativeObservationFailure> {
        self.entries.iter().flatten()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.iter().all(Option::is_none)
    }
}
/// Unknown measurement is neither completed-input zero nor logical consumption.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TicketExecutionEvidence {
    pub execution: Option<ExecutionId>,
    pub completed_nn_inputs: Option<u64>,
    pub measurement_complete: bool,
    pub physical_fresh: Option<bool>,
    /// Only actual owner counter deltas are known here. Missing evidence is
    /// unknown, never an inferred cache hit or proof of no validator call.
    pub accounting: Option<TicketNativeAccountingEvidence>,
}

/// Ticket-scoped native owner observations. These do not acknowledge the
/// output, consume a search evaluation, or prove process-wide V absence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TicketNativeAccountingEvidence {
    pub(crate) public_cache_hits: u64,
    pub(crate) public_cache_misses: u64,
    pub(crate) public_nn_runs_attempted: u64,
    pub(crate) public_nn_runs_completed: u64,
    pub(crate) public_nn_runs_failed_known: u64,
    pub(crate) role_nn_runs_attempted: u64,
    pub(crate) role_nn_runs_completed: u64,
    pub(crate) role_nn_runs_failed_known: u64,
    pub(crate) validator_nn_runs_attempted: u64,
    pub(crate) validator_nn_runs_completed: u64,
    pub(crate) validator_nn_runs_failed_known: u64,
}
impl TicketNativeAccountingEvidence {
    pub fn public_cache_hits(&self) -> u64 {
        self.public_cache_hits
    }
    pub fn public_cache_misses(&self) -> u64 {
        self.public_cache_misses
    }
    pub fn public_nn_runs_attempted(&self) -> u64 {
        self.public_nn_runs_attempted
    }
    pub fn public_nn_runs_completed(&self) -> u64 {
        self.public_nn_runs_completed
    }
    pub fn public_nn_runs_failed_known(&self) -> u64 {
        self.public_nn_runs_failed_known
    }
    pub fn role_nn_runs_attempted(&self) -> u64 {
        self.role_nn_runs_attempted
    }
    pub fn role_nn_runs_completed(&self) -> u64 {
        self.role_nn_runs_completed
    }
    pub fn role_nn_runs_failed_known(&self) -> u64 {
        self.role_nn_runs_failed_known
    }
    pub fn validator_nn_runs_attempted(&self) -> u64 {
        self.validator_nn_runs_attempted
    }
    pub fn validator_nn_runs_completed(&self) -> u64 {
        self.validator_nn_runs_completed
    }
    pub fn validator_nn_runs_failed_known(&self) -> u64 {
        self.validator_nn_runs_failed_known
    }
}

/// Session, binding and physical leases are backend-owned. A returned unknown
/// result is not permission to release them. Fresh outputs may await logical
/// acknowledgement while the worker immediately starts the next physical call.
pub trait TicketNativeBackend<V, Q>: Send + 'static {
    type Prepared: Send + 'static;
    type Output: Send + Sync + 'static;
    fn clock(&self) -> ContractSystemClock;
    fn prepare(
        &mut self,
        question: &TicketPreparation<V, Q>,
        canceled: &AtomicBool,
    ) -> Result<(Self::Prepared, RepresentationKey), RoleFailure>;
    fn prepared_host_bytes(&self, prepared: &Self::Prepared) -> u64;
    fn question_host_bytes(&self, _: &TicketPreparation<V, Q>) -> u64 {
        0
    }
    fn run(
        &mut self,
        ticket: &RoleTicketEnvelope,
        question: &TicketPreparation<V, Q>,
        prepared: &Self::Prepared,
        canceled: &AtomicBool,
    ) -> TicketPhysicalResult<Self::Output>;
    /// The service uses this entrypoint for actual dispatch. Existing backends
    /// retain their original execution behavior through this default method.
    fn run_owned(
        &mut self,
        context: TicketOwnerDispatchContext<'_, V, Q, Self::Prepared>,
        canceled: &AtomicBool,
    ) -> TicketPhysicalResult<Self::Output> {
        self.run(context.ticket, context.question, context.prepared, canceled)
    }
    fn output_host_bytes(&self, output: &Self::Output) -> u64;
    /// Exact original P/V head digest for an owner acknowledgement. Other
    /// payloads/backends retain the historical acceptance path explicitly.
    fn acceptance_output_digest(&self, _: &Self::Output) -> Option<Digest> {
        None
    }
    fn execution_evidence(&mut self, _: &RoleTicketEnvelope) -> Option<TicketExecutionEvidence> {
        None
    }
    /// At most two separately retained failures from the immediately preceding
    /// run, accept or discard callback. Backends must retain its original cause
    /// and must not use this to downgrade physical or pre-execution failures.
    fn take_observation_failure(&mut self) -> Option<TicketNativeObservationFailure> {
        None
    }
    /// Only a manager acknowledgement reaches this callback. A backend may
    /// register a seed only in the namespace of this original ticket/query.
    fn accept(
        &mut self,
        ticket: &RoleTicketEnvelope,
        question: &TicketPreparation<V, Q>,
        prepared: &Self::Prepared,
        output: &Self::Output,
    ) -> Result<(), RoleFailure>;
    fn discard(&mut self, ticket: &RoleTicketEnvelope) -> Result<(), RoleFailure>;
    /// Fence this root and revoke its retained authority without closing the
    /// model/session or joining the process-owned physical worker. Default is
    /// fail-closed: a backend must implement its actual idle/lifetime proof.
    fn quiesce(&mut self, _: Instant) -> bool {
        false
    }
    /// Observe the same quarantined owner. Unsupported or timed-out fence
    /// providers must return false; this cannot manufacture a completion.
    fn reobserve_quarantined(
        &mut self,
        _: Instant,
    ) -> Result<TicketNativeRecoveryProbe, RoleFailure> {
        Ok(TicketNativeRecoveryProbe::Unknown)
    }
    fn recovery_owner(&self) -> Option<rz_eval::worker::PhysicalRecoveryKey> {
        None
    }
    /// Actual outstanding lease scope, not an inferred logical request or a
    /// completion claim. Unscoped control/startup work returns None.
    fn recovery_work_scope(&self) -> Option<rz_eval::worker::PhysicalWorkScope> {
        None
    }
    /// Release is a separate operation using the same original proof. A
    /// backend lacking this path must remain Unknown instead of dropping state.
    fn release_quarantined(
        &mut self,
        _: &rz_eval::worker::PhysicalRecoveryConfirmation,
        _: Instant,
    ) -> Result<TicketNativeRelease, RoleFailure> {
        Ok(TicketNativeRelease::Unknown)
    }
    /// True requires the actual physical fence and worker join, not a timeout.
    fn finish(&mut self, until: Instant) -> bool;
    /// Preserve semantic shutdown failures independently of physical custody.
    /// Existing backends keep their historical `finish` behavior by default.
    fn finish_report(&mut self, until: Instant) -> TicketNativeFinishReport {
        TicketNativeFinishReport {
            physical_shutdown_confirmed: self.finish(until),
            physical_failure: None,
            semantic_failure: None,
            session_release: None,
            runtime_mapping_status: TicketRuntimeMappingStatus::NotObserved,
            system_placement_status: TicketSystemPlacementStatus::NotObserved,
        }
    }
}

/// Runtime mapping observations are separate from strict placement acceptance,
/// neural execution and physical completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TicketRuntimeMappingStatus {
    NotObserved,
    NonCuda,
    Observed,
    Unknown,
}

/// A placement witness covers only this system-runtime product scope. An
/// incomplete observation is never an accepted witness or a GPU fence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TicketSystemPlacementStatus {
    NotObserved,
    Incomplete,
    /// Mappings include an identity outside the strict plan or predate its capture.
    /// This does not establish a mapping conflict or constitute an accepted witness.
    UnregisteredOrPreexistingMappings,
    Confirmed,
}

/// Physical completion permits resource release; it does not imply that all
/// requested shutdown operations succeeded semantically.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TicketNativeFinishReport {
    pub physical_shutdown_confirmed: bool,
    pub physical_failure: Option<RoleFailure>,
    pub semantic_failure: Option<RoleFailure>,
    /// This owner's normal session release, not a GPU-wide fence or Run receipt.
    pub session_release: Option<rz_eval::worker::SessionReleased>,
    pub runtime_mapping_status: TicketRuntimeMappingStatus,
    pub system_placement_status: TicketSystemPlacementStatus,
}

fn take_observation_failures<V, Q, B: TicketNativeBackend<V, Q>>(
    backend: &mut B,
    ticket: RoleTicketEnvelope,
) -> TicketNativeObservationFailures {
    let mut failures = TicketNativeObservationFailures::default();
    for entry in &mut failures.entries {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            backend.take_observation_failure()
        })) {
            Ok(Some(failure)) => *entry = Some(failure),
            Ok(None) => break,
            Err(_) => {
                *entry = Some(TicketNativeObservationFailure {
                    ticket,
                    stage: TicketNativeObservationStage::Retrieval,
                    disposition: TicketNativeObservationDisposition::RequiredAfterPhysical,
                    cause: RoleFailure::Backend(
                        "observation failure retrieval panicked; original report unavailable"
                            .into(),
                    ),
                });
                break;
            }
        }
    }
    failures
}

/// Produced only by the owner after root cancellation, physical completion and
/// ticket cleanup. It proves root reuse, never process shutdown or session drop.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeRootFence {
    authority: SearchAuthority,
    physical_completion_confirmed: bool,
    ticket_cleanup_confirmed: bool,
}
impl NativeRootFence {
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn permits_reuse(&self) -> bool {
        self.physical_completion_confirmed && self.ticket_cleanup_confirmed
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TicketServiceLimits {
    pub review_token_budget: u16,
    pub prepared_host_bytes_per_ticket: u64,
    pub output_host_bytes_per_ticket: u64,
}
impl Default for TicketServiceLimits {
    fn default() -> Self {
        Self {
            review_token_budget: 512,
            prepared_host_bytes_per_ticket: 1024 * 1024,
            output_host_bytes_per_ticket: 1024 * 1024,
        }
    }
}

/// Owner-issued observation of whether this original ticket entered the
/// backend run callback. It is not proof that a graph started or completed.
/// A false witness is solely the service's pre-run cancellation branch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TicketOwnerRunWitness {
    ticket: RoleTicketEnvelope,
    invoked: bool,
}
impl TicketOwnerRunWitness {
    pub fn ticket(&self) -> &RoleTicketEnvelope {
        &self.ticket
    }
    pub fn backend_run_invoked(&self) -> bool {
        self.invoked
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TicketServiceError {
    Capacity,
    Closed,
    UnknownPreparation,
    Duplicate,
    IdentityMismatch,
    WorkerGone,
    Backend(RoleFailure),
}

// The bounded event queue retains the original ticket, completion evidence and
// non-clone acceptance permit/receipt together, without a per-event allocation.
#[allow(clippy::large_enum_variant)]
pub enum TicketNativeEvent<O> {
    Ready {
        clock: ContractSystemClock,
    },
    StartupFailed(RoleFailure),
    Prepared(PreparedTicketSeal),
    PreparationFailed {
        preparation: PreparationId,
        failure: RoleFailure,
    },
    Started(RoleTicketEnvelope),
    /// Same-process monotonic start of the owner run section. This does not
    /// claim that a native NN lease was already admitted at this instant.
    OwnerRunStarted {
        ticket: RoleTicketEnvelope,
        started_at: Instant,
    },
    Completed {
        completion: RoleCompletion<Arc<O>>,
        evidence: Option<TicketExecutionEvidence>,
        owner_run: TicketOwnerRunWitness,
        /// Actual owner run section, including admission/transfer/fence/decode;
        /// excludes preparation/queue wait and is not pure NN kernel time.
        owner_service_elapsed: Option<Duration>,
        /// Recorded independently from, and inseparably with, this terminal.
        observation_failures: TicketNativeObservationFailures,
    },
    Rejected {
        preparation: PreparationId,
        error: TicketServiceError,
    },
    Retired {
        ticket: RoleTicketEnvelope,
        accepted: bool,
        result: Result<(), RoleFailure>,
        native_receipt: Option<NativeAcceptanceReceipt>,
        rejected_permit: Option<NativeAcceptancePermit>,
        observation_failures: TicketNativeObservationFailures,
    },
    AcceptanceRejected {
        preparation: PreparationId,
        ticket: RoleTicketEnvelope,
        error: TicketServiceError,
        permit: NativeAcceptancePermit,
    },
    RootQuiesced(NativeRootFence),
    Finished {
        physical_shutdown_confirmed: bool,
    },
}

// Keep the original question/ticket inline in the bounded command channel;
// dispatch must not allocate an extra box or separate its authority payload.
#[allow(clippy::large_enum_variant)]
enum Command<V, Q> {
    Prepare(PreparationId, TicketPreparation<V, Q>, Arc<AtomicBool>),
    Execute(PreparationId, RoleTicketEnvelope, bool),
}
/// Original queued request recovered after the owning physical work is fenced.
pub enum TicketPendingCommand<V, Q> {
    Prepare(PreparationId, TicketPreparation<V, Q>, Arc<AtomicBool>),
    Execute(PreparationId, RoleTicketEnvelope, bool),
}
impl<V, Q> From<Command<V, Q>> for TicketPendingCommand<V, Q> {
    fn from(command: Command<V, Q>) -> Self {
        match command {
            Command::Prepare(id, question, canceled) => Self::Prepare(id, question, canceled),
            Command::Execute(id, ticket, flag) => Self::Execute(id, ticket, flag),
        }
    }
}
// One bounded acknowledgement owns its original ticket and non-clone permit
// until the physical owner accepts or returns that same permit on failure.
#[allow(clippy::large_enum_variant)]
enum Ack {
    Accept(
        PreparationId,
        RoleTicketEnvelope,
        Option<NativeAcceptancePermit>,
    ),
    Discard(PreparationId, RoleTicketEnvelope),
}
/// The actual pending acknowledgement transferred from its original worker.
/// It is custody only; obtaining it is not an acceptance or completion.
pub struct TicketPendingAcknowledgement {
    original: Ack,
}
/// Exact startup custody returned after the inner owner has been released.
pub struct TicketRetainedFactoryState<C> {
    pub owner: C,
}
impl TicketPendingAcknowledgement {
    pub fn ticket(&self) -> &RoleTicketEnvelope {
        match &self.original {
            Ack::Accept(_, ticket, _) | Ack::Discard(_, ticket) => ticket,
        }
    }
    pub fn into_permit(self) -> Option<NativeAcceptancePermit> {
        match self.original {
            Ack::Accept(_, _, permit) => permit,
            Ack::Discard(_, _) => None,
        }
    }
}
struct PreparationControl {
    canceled: Arc<AtomicBool>,
    /// Successfully admitted Execute commands not yet taken by the owner.
    /// This is protected by the same lock as cancellation retirement: a
    /// queued execution is submitted work even before Held::ticket is set.
    queued_executions: usize,
}
#[cfg(test)]
struct CancellationScanTestGate {
    preparation: PreparationId,
    entered: mpsc::SyncSender<()>,
    release: mpsc::Receiver<()>,
    until: Instant,
}
#[cfg(test)]
#[derive(Default)]
struct CancellationScanTestState {
    installed: bool,
    pending: Option<CancellationScanTestGate>,
}
#[cfg(test)]
struct AdmissionTestGate {
    entered: mpsc::SyncSender<()>,
    release: mpsc::Receiver<()>,
    until: Instant,
}
pub(super) struct Shared {
    #[cfg(feature = "native-validation-diagnostics")]
    validation_trace: std::sync::OnceLock<
        rz_eval::pals_onnx::virtual_review::native_validation_diagnostics::GpuLifecycleTrace,
    >,
    controls: Mutex<BTreeMap<PreparationId, PreparationControl>>,
    // Recoverable admission and shutdown share a short, nonblocking gate.
    admission: Mutex<()>,
    pending_capacity: usize,
    command_capacity: usize,
    closed: AtomicBool,
    root_closed: AtomicBool,
    quarantine: AtomicBool,
    factory_status: AtomicU8,
    shutdown_until: Mutex<Option<Instant>>,
    root_authority: Mutex<Option<SearchAuthority>>,
    root_quiesce: Mutex<Option<(SearchAuthority, Instant)>>,
    root_fence: Mutex<Option<NativeRootFence>>,
    ready_clock: Mutex<Option<ContractSystemClock>>,
    acceptance_verifier: Mutex<Option<NativeAcceptanceVerifier>>,
    pending_reports: AtomicUsize,
    finish_report: Mutex<Option<TicketNativeFinishReport>>,
    finish_report_snapshot: Mutex<Option<TicketNativeFinishReport>>,
    requires_finish_report: bool,
    signal: CompletionSignal,
    ready_cpu: Mutex<
        Option<
            Result<rz_runtime::adaptive::CpuThreadReadback, rz_runtime::adaptive::CpuResourceError>,
        >,
    >,
    #[cfg(test)]
    cancellation_scan_test: Mutex<CancellationScanTestState>,
    #[cfg(test)]
    admission_test: Mutex<Option<AdmissionTestGate>>,
}
impl Shared {
    pub(super) fn take_finish_report(&self) -> Option<TicketNativeFinishReport> {
        self.finish_report
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
    }
    pub(super) fn has_finish_report(&self) -> bool {
        self.finish_report
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }
    fn seal_admission(&self) {
        let _guard = self.admission.lock().unwrap_or_else(|e| e.into_inner());
        self.closed.store(true, Ordering::Release);
    }
    #[cfg(test)]
    fn wait_admission_test(&self) {
        let gate = self
            .admission_test
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(gate) = gate {
            let _ = gate.entered.send(());
            let _ = gate
                .release
                .recv_timeout(gate.until.saturating_duration_since(Instant::now()));
        }
    }
    pub(super) fn recovery_dequeued(&self) {
        self.pending_reports.fetch_sub(1, Ordering::AcqRel);
    }
    fn reports_available(&self) -> bool {
        self.pending_reports.load(Ordering::Acquire) < REPORT_HIGH_WATER
    }
    fn retire(&self, id: PreparationId) {
        self.controls
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&id);
    }
}
struct EventPublisher<O> {
    sender: mpsc::SyncSender<TicketNativeEvent<O>>,
    shared: Arc<Shared>,
    undelivered: RefCell<Vec<super::recovery::TicketUndeliveredEvent>>,
    recoverable: bool,
}
impl<O: Send + Sync + 'static> EventPublisher<O> {
    // SendError returns the original terminal/ack event, including any owned
    // permit, instead of replacing failed delivery with a smaller error token.
    fn send(&self, event: TicketNativeEvent<O>) -> Result<(), ()> {
        self.shared.pending_reports.fetch_add(1, Ordering::AcqRel);
        let sent = if self.recoverable {
            self.sender.try_send(event)
        } else {
            self.sender
                .send(event)
                .map_err(|mpsc::SendError(event)| mpsc::TrySendError::Disconnected(event))
        };
        let result = match sent {
            Ok(()) => Ok(()),
            Err(mpsc::TrySendError::Full(event) | mpsc::TrySendError::Disconnected(event)) => {
                self.shared.pending_reports.fetch_sub(1, Ordering::AcqRel);
                self.shared.closed.store(true, Ordering::Release);
                // The original event stays in the same worker state. This
                // allocates only if the bounded preallocated reserve is used.
                self.undelivered
                    .borrow_mut()
                    .push(super::recovery::TicketUndeliveredEvent::new(event));
                Err(())
            }
        };
        self.shared.signal.notify();
        result
    }
}

/// A nonblocking manager handle. Ordinary queue saturation cannot prevent
/// cancellation or shutdown: these use a separate atomic control mailbox.
enum OwnerThread {
    Legacy(Option<JoinHandle<()>>),
    Recoverable(Arc<super::recovery::TicketOwnerJoin>),
}

/// Recoverable sessions register the same receiver before starting their
/// worker. The historical path retains its original lock-free receiver.
enum OwnerEvents<O> {
    Legacy(mpsc::Receiver<TicketNativeEvent<O>>),
    Recoverable(Arc<Mutex<mpsc::Receiver<TicketNativeEvent<O>>>>),
}
impl<O> OwnerEvents<O> {
    fn try_recv(&self) -> Result<TicketNativeEvent<O>, mpsc::TryRecvError> {
        match self {
            Self::Legacy(receiver) => receiver.try_recv(),
            Self::Recoverable(receiver) => receiver
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .try_recv(),
        }
    }
}

pub struct TicketedNativeService<V, Q, B: TicketNativeBackend<V, Q>> {
    commands: mpsc::SyncSender<Command<V, Q>>,
    acknowledgements: mpsc::SyncSender<Ack>,
    events: Option<OwnerEvents<B::Output>>,
    shared: Arc<Shared>,
    thread: OwnerThread,
    recovery_key: Option<super::recovery::TicketNativeRecoveryKey>,
    next_preparation: u64,
}
impl<V: Send + Sync + 'static, Q: Send + 'static, B: TicketNativeBackend<V, Q>>
    TicketedNativeService<V, Q, B>
{
    /// Factory execution (including model loading/startup) is off the manager.
    pub fn spawn(
        factory: impl FnOnce() -> Result<B, RoleFailure> + Send + 'static,
        limits: TicketServiceLimits,
    ) -> Result<Self, TicketServiceError> {
        // Historical callers retain both sixteen-slot queues. V5 declares its
        // smaller ordinary command admission independently below.
        Self::spawn_bounded(factory, limits, CAPACITY, CAPACITY)
    }

    /// Apply the registered V5 limits to actual queued preparations, retained
    /// inputs/outputs and the ordinary worker command channel. Cancel/stop use
    /// their atomic mailbox; terminal publication retains the full reserve.
    pub fn spawn_with_capacity(
        factory: impl FnOnce() -> Result<B, RoleFailure> + Send + 'static,
        limits: TicketServiceLimits,
        pending_capacity: usize,
        command_capacity: usize,
    ) -> Result<Self, TicketServiceError> {
        if !(1..=CAPACITY).contains(&pending_capacity)
            || !(1..=ADAPTIVE_COMMAND_CAPACITY).contains(&command_capacity)
        {
            return Err(TicketServiceError::Capacity);
        }
        Self::spawn_bounded(factory, limits, pending_capacity, command_capacity)
    }

    /// Reserve a recoverable original owner before allowing factory work. A
    /// spawn failure returns the exact C; no reconstructed owner is issued.
    pub fn spawn_with_startup_custody<C: FactoryStartupRecovery>(
        owner: C,
        factory: impl FnOnce(&mut FactoryStartupCustody<C>) -> Result<B, RoleFailure> + Send + 'static,
        limits: TicketServiceLimits,
        pending_capacity: usize,
        command_capacity: usize,
    ) -> Result<Self, FactoryStartupSpawnRejection<C>> {
        if !(1..=CAPACITY).contains(&pending_capacity)
            || !(1..=ADAPTIVE_COMMAND_CAPACITY).contains(&command_capacity)
            || ![128, 256, 512].contains(&limits.review_token_budget)
            || limits.prepared_host_bytes_per_ticket == 0
            || limits.output_host_bytes_per_ticket == 0
        {
            return Err(FactoryStartupSpawnRejection {
                error: TicketServiceError::Capacity,
                owner,
                recovery_key: None,
                unstarted_thread: None,
            });
        }
        let (reservation, key) = match super::recovery::TicketNativeRecoveryReservation::reserve() {
            Ok(value) => value,
            Err(_) => {
                return Err(FactoryStartupSpawnRejection {
                    error: TicketServiceError::Capacity,
                    owner,
                    recovery_key: None,
                    unstarted_thread: None,
                });
            }
        };
        let join = Arc::new(super::recovery::TicketOwnerJoin::new());
        if reservation.attach(Arc::clone(&join)).is_err() {
            let _ = reservation.cancel_unstarted();
            return Err(FactoryStartupSpawnRejection {
                error: TicketServiceError::WorkerGone,
                owner,
                recovery_key: None,
                unstarted_thread: None,
            });
        }
        let (commands, command_receiver) = mpsc::sync_channel(command_capacity);
        let (acknowledgements, ack_receiver) = mpsc::sync_channel(pending_capacity);
        let (event_sender, events) = mpsc::sync_channel(EVENTS);
        let shared = Arc::new(Shared {
            #[cfg(feature = "native-validation-diagnostics")]
            validation_trace: std::sync::OnceLock::new(),
            controls: Mutex::new(BTreeMap::new()),
            admission: Mutex::new(()),
            pending_capacity,
            command_capacity,
            closed: AtomicBool::new(false),
            root_closed: AtomicBool::new(false),
            quarantine: AtomicBool::new(false),
            factory_status: AtomicU8::new(0),
            shutdown_until: Mutex::new(None),
            root_authority: Mutex::new(None),
            root_quiesce: Mutex::new(None),
            root_fence: Mutex::new(None),
            ready_clock: Mutex::new(None),
            acceptance_verifier: Mutex::new(None),
            pending_reports: AtomicUsize::new(0),
            finish_report: Mutex::new(None),
            finish_report_snapshot: Mutex::new(None),
            requires_finish_report: true,
            signal: CompletionSignal::default(),
            ready_cpu: Mutex::new(None),
            #[cfg(test)]
            cancellation_scan_test: Mutex::new(CancellationScanTestState::default()),
            #[cfg(test)]
            admission_test: Mutex::new(None),
        });
        let events = Arc::new(Mutex::new(events));
        if key
            .install_shared_receiver(&events, Arc::clone(&shared))
            .is_err()
        {
            let _ = reservation.cancel_spawn_failure();
            return Err(FactoryStartupSpawnRejection {
                error: TicketServiceError::WorkerGone,
                owner,
                recovery_key: None,
                unstarted_thread: None,
            });
        }
        let publisher = EventPublisher {
            sender: event_sender,
            shared: Arc::clone(&shared),
            undelivered: RefCell::new(Vec::with_capacity(EVENTS)),
            recoverable: true,
        };
        let worker_shared = Arc::clone(&shared);
        let worker_key = key.clone();
        let transfer = Arc::new(Mutex::new(Some((owner, reservation))));
        let worker_transfer = Arc::clone(&transfer);
        let (start, started) = mpsc::sync_channel::<()>(1);
        let spawned = thread::Builder::new()
            .name("pals-native-ticket-owner".into())
            .spawn(move || {
                if started.recv().is_err() {
                    return;
                }
                let Some((owner, mut reservation)) = worker_transfer
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .take()
                else {
                    return;
                };
                let mut custody = FactoryStartupCustody {
                    owner,
                    entry: FactoryNativeEntry { entered: false },
                    recovery_key: worker_key,
                };
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    factory(&mut custody)
                }));
                match result {
                    Ok(Ok(backend)) => {
                        let mut state = WorkerState {
                            backend,
                            held: BTreeMap::new(),
                            commands: command_receiver,
                            acks: ack_receiver,
                            events: publisher,
                            active: None,
                            active_ack: None,
                            active_prepared: None,
                            active_output: None,
                            active_panics: Vec::with_capacity(4),
                            active_observations: TicketNativeObservationFailures::default(),
                            active_ack_result: None,
                            recovery_acks: Vec::with_capacity(pending_capacity),
                            recovery_commands: Vec::with_capacity(command_capacity),
                            recovery_root_fence: None,
                        };
                        // Publish readiness only after the actual worker
                        // state owns the backend and its queues.
                        worker_shared.factory_status.store(1, Ordering::Release);
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            recoverable::run_worker_loop(&mut state, &worker_shared, limits)
                        }));
                        let (finished, panicked) = match result {
                            Ok(finished) => (finished, false),
                            Err(payload) => {
                                state.active_panics.push(payload);
                                (false, true)
                            }
                        };
                        worker_shared.seal_admission();
                        if finished
                            && state.events.undelivered.borrow().is_empty()
                            && state.held.is_empty()
                            && state.active.is_none()
                            && state.active_ack.is_none()
                            && state
                                .recovery_commands
                                .iter()
                                .all(|item| !item.is_pending())
                            && state.recovery_acks.iter().all(|item| !item.is_pending())
                            && state.active_prepared.is_none()
                            && state.active_output.is_none()
                            && state.active_panics.is_empty()
                            && state.active_observations.is_empty()
                            && state.active_ack_result.is_none()
                            && state.recovery_root_fence.is_none()
                        {
                            let _ = reservation.finish_clean();
                            return;
                        }
                        worker_shared.quarantine.store(!finished, Ordering::Release);
                        for cancel in worker_shared
                            .controls
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .values()
                        {
                            cancel.canceled.store(true, Ordering::Release);
                        }
                        if panicked {
                            let _ = state.events.send(TicketNativeEvent::Finished {
                                physical_shutdown_confirmed: false,
                            });
                        }
                        worker_shared.signal.notify();
                        // Bind only the actual outstanding physical lease, the
                        // currently retained Execute command and its original
                        // held ticket. Different/missing scopes remain Unknown.
                        if let (
                            Some(Command::Execute(id, active_ticket, _)),
                            Some(inner),
                            Some(work),
                        ) = (
                            state.active.as_ref(),
                            state.backend.recovery_owner(),
                            state.backend.recovery_work_scope(),
                        ) {
                            if let rz_eval::worker::PhysicalWorkScope::Ticket { ticket, .. } = work
                            {
                                if ticket == *active_ticket
                                    && state
                                        .held
                                        .get(id)
                                        .is_some_and(|held| held.ticket == Some(ticket))
                                {
                                    let _ = reservation.bind_active_work(&inner, work);
                                }
                            }
                        }
                        let WorkerState {
                            backend,
                            events,
                            held,
                            commands,
                            active,
                            active_ack,
                            acks,
                            recovery_acks,
                            recovery_commands,
                            active_prepared,
                            active_output,
                            active_panics,
                            active_observations,
                            active_ack_result,
                            recovery_root_fence,
                        } = &mut state;
                        let mut retained_state: Option<super::recovery::TicketUndeliveredEvent> =
                            None;
                        reservation.park(
                            finished,
                            |op| match op {
                                super::recovery::TicketRecoveryOperation::Observe(until) => {
                                    super::recovery::TicketRecoveryOperationResult::Observe(
                                        backend
                                            .reobserve_quarantined(until)
                                            .unwrap_or(TicketNativeRecoveryProbe::Unknown),
                                    )
                                }
                                super::recovery::TicketRecoveryOperation::Release(proof, until) => {
                                    super::recovery::TicketRecoveryOperationResult::Release(
                                        backend
                                            .release_quarantined(proof, until)
                                            .unwrap_or(TicketNativeRelease::Unknown),
                                    )
                                }
                                super::recovery::TicketRecoveryOperation::TakeAck => {
                                    if let Some(original) =
                                        recovery_acks.iter().find(|item| item.is_pending()).cloned()
                                    {
                                        super::recovery::TicketRecoveryOperationResult::Ack(Some(
                                            original,
                                        ))
                                    } else {
                                        let ack =
                                            active_ack.take().or_else(|| acks.try_recv().ok());
                                        let original = ack.map(|original| {
                                            super::recovery::TicketUndeliveredEvent::new(
                                                TicketPendingAcknowledgement { original },
                                            )
                                        });
                                        if let Some(original) = &original {
                                            recovery_acks.push(original.clone());
                                        }
                                        super::recovery::TicketRecoveryOperationResult::Ack(
                                            original,
                                        )
                                    }
                                }
                                super::recovery::TicketRecoveryOperation::TakeState { .. } => {
                                    if retained_state.is_none() {
                                        let (_sender, empty) = mpsc::channel();
                                        let retained = TicketRetainedOwnerState {
                                            held: std::mem::take(held),
                                            active_command: active.take().map(Into::into),
                                            queued_commands: std::mem::replace(commands, empty),
                                            original_command_handles: std::mem::take(
                                                recovery_commands,
                                            ),
                                            active_prepared: active_prepared.take(),
                                            active_output: active_output.take(),
                                            panic_payloads: std::mem::take(active_panics),
                                            observations: std::mem::take(active_observations),
                                            active_ack_result: active_ack_result.take(),
                                            original_root_fence: recovery_root_fence.take(),
                                        };
                                        retained_state = Some(
                                            super::recovery::TicketUndeliveredEvent::new(retained),
                                        );
                                    }
                                    super::recovery::TicketRecoveryOperationResult::State(
                                        retained_state.clone(),
                                    )
                                }
                                super::recovery::TicketRecoveryOperation::CanRelease => {
                                    if active_ack.is_none()
                                        && recovery_acks.iter().all(|item| !item.is_pending())
                                    {
                                        if let Ok(original) = acks.try_recv() {
                                            recovery_acks.push(
                                                super::recovery::TicketUndeliveredEvent::new(
                                                    TicketPendingAcknowledgement { original },
                                                ),
                                            );
                                        }
                                    }
                                    super::recovery::TicketRecoveryOperationResult::CanRelease(
                                        active_ack.is_none()
                                            && recovery_acks.iter().all(|item| !item.is_pending())
                                            && retained_state
                                                .as_ref()
                                                .is_some_and(|state| !state.is_pending()),
                                    )
                                }
                            },
                            &events.undelivered,
                        );
                    }
                    result => {
                        let cause = match result {
                            Ok(Err(cause)) => cause,
                            Err(payload) => {
                                publisher
                                    .undelivered
                                    .borrow_mut()
                                    .push(super::recovery::TicketUndeliveredEvent::new(payload));
                                RoleFailure::Backend("native startup factory panicked".into())
                            }
                            Ok(Ok(_)) => unreachable!(),
                        };
                        let entered = custody.entry.entered();
                        worker_shared.seal_admission();
                        worker_shared
                            .factory_status
                            .store(if entered { 3 } else { 2 }, Ordering::Release);
                        let _ = publisher.send(TicketNativeEvent::StartupFailed(cause));
                        if !entered {
                            let events = publisher.undelivered.into_inner();
                            if let Err((_owner, _events)) =
                                reservation.before_native(custody.owner, events)
                            {
                                // Registry identity corruption cannot authorize dropping
                                // the original owner. Keep it on this live thread.
                                loop {
                                    thread::park();
                                }
                            }
                            return;
                        }
                        worker_shared.quarantine.store(true, Ordering::Release);
                        let _ = publisher.send(TicketNativeEvent::Finished {
                            physical_shutdown_confirmed: false,
                        });
                        worker_shared.signal.notify();
                        let mut owner = Some(custody.owner);
                        let mut retained_factory: Option<super::recovery::TicketUndeliveredEvent> =
                            None;
                        reservation.park(
                            false,
                            |op| match op {
                                super::recovery::TicketRecoveryOperation::Observe(until) => {
                                    super::recovery::TicketRecoveryOperationResult::Observe(
                                        owner
                                            .as_mut()
                                            .and_then(|owner| owner.reobserve(until).ok())
                                            .unwrap_or(TicketNativeRecoveryProbe::Unknown),
                                    )
                                }
                                super::recovery::TicketRecoveryOperation::Release(proof, until) => {
                                    super::recovery::TicketRecoveryOperationResult::Release(
                                        owner
                                            .as_mut()
                                            .and_then(|owner| owner.release(proof, until).ok())
                                            .unwrap_or(TicketNativeRelease::Unknown),
                                    )
                                }
                                super::recovery::TicketRecoveryOperation::TakeAck => {
                                    super::recovery::TicketRecoveryOperationResult::Ack(None)
                                }
                                super::recovery::TicketRecoveryOperation::TakeState {
                                    inner_released,
                                } => {
                                    if inner_released && retained_factory.is_none() {
                                        if let Some(owner) = owner.take() {
                                            retained_factory =
                                                Some(super::recovery::TicketUndeliveredEvent::new(
                                                    TicketRetainedFactoryState { owner },
                                                ));
                                        }
                                    }
                                    super::recovery::TicketRecoveryOperationResult::State(
                                        retained_factory.clone(),
                                    )
                                }
                                super::recovery::TicketRecoveryOperation::CanRelease => {
                                    super::recovery::TicketRecoveryOperationResult::CanRelease(
                                        retained_factory
                                            .as_ref()
                                            .is_some_and(|state| !state.is_pending()),
                                    )
                                }
                            },
                            &publisher.undelivered,
                        );
                    }
                }
            });
        let handle = match spawned {
            Ok(handle) => handle,
            Err(_) => {
                let (owner, reservation) = transfer
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .take()
                    .expect("unstarted transfer retains original owner");
                let _ = reservation.cancel_spawn_failure();
                return Err(FactoryStartupSpawnRejection {
                    error: TicketServiceError::WorkerGone,
                    owner,
                    recovery_key: None,
                    unstarted_thread: None,
                });
            }
        };
        if let Err(handle) = join.install(handle) {
            drop(start);
            let (owner, reservation) = transfer
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take()
                .expect("unstarted transfer retains original owner");
            reservation.mark_start_not_delivered();
            key.release_shared_receiver();
            return Err(FactoryStartupSpawnRejection {
                error: TicketServiceError::WorkerGone,
                owner,
                recovery_key: Some(key),
                unstarted_thread: Some(handle),
            });
        }
        // The worker cannot access the native factory until the original join
        // handle is installed in both service and recovery registry.
        if start.send(()).is_err() {
            // The token was not delivered, so the factory could not run. Keep
            // its actual join handle in the registry without blocking here.
            let (owner, reservation) = transfer
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take()
                .expect("failed start retains original owner");
            reservation.mark_start_not_delivered();
            key.release_shared_receiver();
            return Err(FactoryStartupSpawnRejection {
                error: TicketServiceError::WorkerGone,
                owner,
                recovery_key: Some(key),
                unstarted_thread: None,
            });
        }
        Ok(Self {
            commands,
            acknowledgements,
            events: Some(OwnerEvents::Recoverable(events)),
            shared,
            thread: OwnerThread::Recoverable(join),
            recovery_key: Some(key),
            next_preparation: 1,
        })
    }

    fn spawn_bounded(
        factory: impl FnOnce() -> Result<B, RoleFailure> + Send + 'static,
        limits: TicketServiceLimits,
        pending_capacity: usize,
        command_capacity: usize,
    ) -> Result<Self, TicketServiceError> {
        if ![128, 256, 512].contains(&limits.review_token_budget)
            || limits.prepared_host_bytes_per_ticket == 0
            || limits.output_host_bytes_per_ticket == 0
        {
            return Err(TicketServiceError::Capacity);
        }
        let (commands, receiver) = mpsc::sync_channel(command_capacity);
        let (acknowledgements, ack_receiver) = mpsc::sync_channel(pending_capacity);
        let (event_sender, events) = mpsc::sync_channel(EVENTS);
        let shared = Arc::new(Shared {
            #[cfg(feature = "native-validation-diagnostics")]
            validation_trace: std::sync::OnceLock::new(),
            controls: Mutex::new(BTreeMap::new()),
            admission: Mutex::new(()),
            pending_capacity,
            command_capacity,
            closed: AtomicBool::new(false),
            root_closed: AtomicBool::new(false),
            quarantine: AtomicBool::new(false),
            factory_status: AtomicU8::new(0),
            shutdown_until: Mutex::new(None),
            root_authority: Mutex::new(None),
            root_quiesce: Mutex::new(None),
            root_fence: Mutex::new(None),
            ready_clock: Mutex::new(None),
            acceptance_verifier: Mutex::new(None),
            pending_reports: AtomicUsize::new(0),
            finish_report: Mutex::new(None),
            finish_report_snapshot: Mutex::new(None),
            requires_finish_report: false,
            signal: CompletionSignal::default(),
            ready_cpu: Mutex::new(None),
            #[cfg(test)]
            cancellation_scan_test: Mutex::new(CancellationScanTestState::default()),
            #[cfg(test)]
            admission_test: Mutex::new(None),
        });
        let event_sender = EventPublisher {
            sender: event_sender,
            shared: Arc::clone(&shared),
            undelivered: RefCell::new(Vec::with_capacity(EVENTS)),
            recoverable: false,
        };
        let thread_shared = Arc::clone(&shared);
        let thread = thread::Builder::new()
            .name("pals-native-ticket-owner".into())
            .spawn(move || match factory() {
                Ok(backend) => {
                    thread_shared.factory_status.store(1, Ordering::Release);
                    worker(
                        backend,
                        receiver,
                        ack_receiver,
                        event_sender,
                        thread_shared,
                        limits,
                    )
                }
                Err(failure) => {
                    thread_shared.factory_status.store(4, Ordering::Release);
                    thread_shared.closed.store(true, Ordering::Release);
                    thread_shared
                        .controls
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .clear();
                    let _ = event_sender.send(TicketNativeEvent::StartupFailed(failure));
                }
            })
            .map_err(|_| TicketServiceError::WorkerGone)?;
        Ok(Self {
            commands,
            acknowledgements,
            events: Some(OwnerEvents::Legacy(events)),
            shared,
            thread: OwnerThread::Legacy(Some(thread)),
            recovery_key: None,
            next_preparation: 1,
        })
    }

    pub fn pending_capacity(&self) -> usize {
        self.shared.pending_capacity
    }
    pub fn command_capacity(&self) -> usize {
        self.shared.command_capacity
    }

    /// Explicit validation only, before the first actual preparation.
    #[cfg(feature = "native-validation-diagnostics")]
    pub fn enable_gpu_lifecycle_validation(
        &mut self,
        trace: rz_eval::pals_onnx::virtual_review::native_validation_diagnostics::GpuLifecycleTrace,
    ) -> Result<(), TicketServiceError> {
        if self.next_preparation != 1
            || !self
                .shared
                .controls
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_empty()
        {
            return Err(TicketServiceError::Duplicate);
        }
        self.shared
            .validation_trace
            .set(trace)
            .map_err(|_| TicketServiceError::Duplicate)
    }
    // Failed admission returns the exact immutable question/view ownership;
    // no extra allocation or early drop is introduced at this boundary.
    #[allow(clippy::result_large_err)]
    pub fn prepare(
        &mut self,
        question: TicketPreparation<V, Q>,
    ) -> Result<PreparationId, (TicketServiceError, TicketPreparation<V, Q>)> {
        let _admission = self.recovery_key.as_ref().map(|_| {
            self.shared
                .admission
                .lock()
                .unwrap_or_else(|e| e.into_inner())
        });
        if self.recovery_key.is_some() && self.startup_status() != NativeFactoryStartupStatus::Ready
        {
            return Err((TicketServiceError::Closed, question));
        }
        if self.shared.closed.load(Ordering::Acquire)
            || self.shared.root_closed.load(Ordering::Acquire)
        {
            return Err((TicketServiceError::Closed, question));
        }
        if !self.shared.reports_available() {
            return Err((TicketServiceError::Capacity, question));
        }
        let Some(next) = self.next_preparation.checked_add(1) else {
            return Err((TicketServiceError::Closed, question));
        };
        let id = PreparationId(self.next_preparation);
        let canceled = Arc::new(AtomicBool::new(false));
        {
            let mut controls = self
                .shared
                .controls
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if controls.len() >= self.shared.pending_capacity {
                return Err((TicketServiceError::Capacity, question));
            }
            controls.insert(
                id,
                PreparationControl {
                    canceled: Arc::clone(&canceled),
                    queued_executions: 0,
                },
            );
        }
        match self
            .commands
            .try_send(Command::Prepare(id, question, canceled))
        {
            Ok(()) => {
                self.next_preparation = next;
                self.shared.signal.notify();
                Ok(id)
            }
            Err(error) => {
                self.shared.retire(id);
                let (kind, command) = match error {
                    mpsc::TrySendError::Full(command) => (TicketServiceError::Capacity, command),
                    mpsc::TrySendError::Disconnected(command) => {
                        (TicketServiceError::WorkerGone, command)
                    }
                };
                let Command::Prepare(_, question, _) = command else {
                    unreachable!("submitted prepare")
                };
                Err((kind, question))
            }
        }
    }
    pub fn execute(
        &self,
        preparation: PreparationId,
        ticket: RoleTicketEnvelope,
    ) -> Result<(), TicketServiceError> {
        self.execute_with_admission_clock(preparation, ticket)
            .map(|_| ())
    }
    /// Timestamp the original successful command-queue admission. A failed
    /// admission returns no timestamp; a later receiver clock is not queue wait.
    pub(crate) fn execute_with_admission_clock(
        &self,
        preparation: PreparationId,
        ticket: RoleTicketEnvelope,
    ) -> Result<Instant, TicketServiceError> {
        let _admission = self.recovery_key.as_ref().map(|_| {
            self.shared
                .admission
                .lock()
                .unwrap_or_else(|e| e.into_inner())
        });
        if self.shared.closed.load(Ordering::Acquire)
            || self.shared.root_closed.load(Ordering::Acquire)
        {
            return Err(TicketServiceError::Closed);
        }
        if !self.shared.reports_available() {
            return Err(TicketServiceError::Capacity);
        }
        let mut controls = self
            .shared
            .controls
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let next_queued = controls
            .get(&preparation)
            .map(|control| {
                control
                    .queued_executions
                    .checked_add(1)
                    .ok_or(TicketServiceError::Capacity)
            })
            .transpose()?;
        // Holding this lock through queue admission and registration prevents
        // the owner from retiring a canceled preparation as unsubmitted while
        // its already accepted Execute command is waiting behind a run.
        let admitted_at = Instant::now();
        self.commands
            .try_send(Command::Execute(preparation, ticket, next_queued.is_some()))
            .map_err(send_error)?;
        if let (Some(control), Some(next_queued)) = (controls.get_mut(&preparation), next_queued) {
            control.queued_executions = next_queued;
        }
        drop(controls);
        self.shared.signal.notify();
        #[cfg(feature = "native-validation-diagnostics")]
        if let Some(trace) = self.shared.validation_trace.get() {
            trace.admitted(preparation.0, ticket, admitted_at);
        }
        Ok(admitted_at)
    }
    pub fn cancel(&self, preparation: PreparationId) -> Result<(), TicketServiceError> {
        let controls = self
            .shared
            .controls
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let canceled = controls
            .get(&preparation)
            .ok_or(TicketServiceError::UnknownPreparation)?;
        #[cfg(feature = "native-validation-diagnostics")]
        let validation_before = Instant::now();
        #[cfg(feature = "native-validation-diagnostics")]
        let validation_first = !canceled.canceled.load(Ordering::Acquire);
        canceled.canceled.store(true, Ordering::Release);
        #[cfg(feature = "native-validation-diagnostics")]
        if let Some(trace) = self.shared.validation_trace.get() {
            trace.cancel_flag_store(
                preparation.0,
                validation_before,
                Instant::now(),
                validation_first,
            );
        }
        self.shared.signal.notify();
        Ok(())
    }
    /// Revoke old-root authority before selecting another fixed root. RequestId
    /// issuance is not reset. No token or running frame moves between owners.
    pub fn replace_root(&self, authority: SearchAuthority) -> Result<(), TicketServiceError> {
        let _admission = self.recovery_key.as_ref().map(|_| {
            self.shared
                .admission
                .lock()
                .unwrap_or_else(|e| e.into_inner())
        });
        if self.shared.closed.load(Ordering::Acquire) {
            return Err(TicketServiceError::Closed);
        }
        // Root fencing also publishes terminal observations. Repeated roots
        // cannot bypass the same finite report reserve as ordinary work.
        if !self.shared.reports_available() {
            return Err(TicketServiceError::Capacity);
        }
        let mut current = self
            .shared
            .root_authority
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if current.is_some_and(|old| old == authority) {
            return if self.shared.root_closed.load(Ordering::Acquire) {
                Err(TicketServiceError::Closed)
            } else {
                Ok(())
            };
        }
        if current.is_some_and(|old| {
            old.epoch != authority.epoch
                || authority.game.0 < old.game.0
                || (authority.game == old.game && authority.root.0 <= old.root.0)
        }) {
            return Err(TicketServiceError::IdentityMismatch);
        }
        if self.shared.root_closed.load(Ordering::Acquire)
            && self
                .shared
                .root_fence
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_none_or(|fence| Some(fence.authority()) != *current || !fence.permits_reuse())
        {
            return Err(TicketServiceError::Closed);
        }
        for cancel in self
            .shared
            .controls
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
        {
            cancel.canceled.store(true, Ordering::Release);
        }
        *current = Some(authority);
        *self
            .shared
            .root_fence
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
        self.shared.root_closed.store(false, Ordering::Release);
        self.shared.signal.notify();
        Ok(())
    }
    pub fn accept(
        &self,
        preparation: PreparationId,
        ticket: RoleTicketEnvelope,
    ) -> Result<(), TicketServiceError> {
        let _admission = self.recovery_key.as_ref().map(|_| {
            self.shared
                .admission
                .lock()
                .unwrap_or_else(|e| e.into_inner())
        });
        if self.shared.closed.load(Ordering::Acquire)
            || self.shared.root_closed.load(Ordering::Acquire)
        {
            return Err(TicketServiceError::Closed);
        }
        if !self.shared.reports_available() {
            return Err(TicketServiceError::Capacity);
        }
        self.acknowledgements
            .try_send(Ack::Accept(preparation, ticket, None))
            .map_err(send_error)?;
        self.shared.signal.notify();
        Ok(())
    }
    // A failed enqueue returns the original non-clone acceptance permit so its
    // authority and rejection custody remain with the caller.
    #[allow(clippy::result_large_err)]
    pub fn accept_with_permit(
        &self,
        preparation: PreparationId,
        ticket: RoleTicketEnvelope,
        permit: NativeAcceptancePermit,
    ) -> Result<(), (TicketServiceError, NativeAcceptancePermit)> {
        if permit.ticket() != &ticket {
            return Err((TicketServiceError::IdentityMismatch, permit));
        }
        let _admission = self.recovery_key.as_ref().map(|_| {
            self.shared
                .admission
                .lock()
                .unwrap_or_else(|e| e.into_inner())
        });
        if self.shared.closed.load(Ordering::Acquire)
            || self.shared.root_closed.load(Ordering::Acquire)
        {
            return Err((TicketServiceError::Closed, permit));
        }
        if !self.shared.reports_available() {
            return Err((TicketServiceError::Capacity, permit));
        }
        #[cfg(test)]
        if self.recovery_key.is_some() {
            self.shared.wait_admission_test();
        }
        match self
            .acknowledgements
            .try_send(Ack::Accept(preparation, ticket, Some(permit)))
        {
            Ok(()) => {
                self.shared.signal.notify();
                Ok(())
            }
            Err(error) => {
                let (error, command) = match error {
                    mpsc::TrySendError::Full(command) => (TicketServiceError::Capacity, command),
                    mpsc::TrySendError::Disconnected(command) => {
                        (TicketServiceError::WorkerGone, command)
                    }
                };
                let Ack::Accept(_, _, Some(permit)) = command else {
                    unreachable!("owned acceptance permit");
                };
                Err((error, permit))
            }
        }
    }
    pub fn discard(
        &self,
        preparation: PreparationId,
        ticket: RoleTicketEnvelope,
    ) -> Result<(), TicketServiceError> {
        let _admission = self.recovery_key.as_ref().map(|_| {
            self.shared
                .admission
                .lock()
                .unwrap_or_else(|e| e.into_inner())
        });
        if self.recovery_key.is_some()
            && (self.shared.closed.load(Ordering::Acquire)
                || self.shared.root_closed.load(Ordering::Acquire))
        {
            return Err(TicketServiceError::Closed);
        }
        if !self.shared.reports_available() {
            return Err(TicketServiceError::Capacity);
        }
        self.acknowledgements
            .try_send(Ack::Discard(preparation, ticket))
            .map_err(send_error)?;
        self.shared.signal.notify();
        Ok(())
    }
    /// Transfer the recoverable owner's exact finish report at most once.
    /// The legacy worker does not publish a report and returns None.
    pub fn take_finish_report(&self) -> Option<TicketNativeFinishReport> {
        self.shared.take_finish_report()
    }

    /// Stable diagnostic copy; remains available after the original transfer.
    pub fn finish_report_snapshot(&self) -> Option<TicketNativeFinishReport> {
        self.shared
            .finish_report_snapshot
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// An observed finish without a snapshot is an error only for this path.
    pub fn requires_finish_report(&self) -> bool {
        self.shared.requires_finish_report
    }

    /// Capacity of this service's bounded native event channel. A consumer
    /// buffering original events must stop polling when its own bound is full.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(crate) fn event_queue_capacity(&self) -> usize {
        EVENTS
    }

    pub fn poll(&self) -> Option<TicketNativeEvent<B::Output>> {
        let event = self.events.as_ref()?.try_recv().ok()?;
        self.shared.pending_reports.fetch_sub(1, Ordering::AcqRel);
        #[cfg(feature = "native-validation-diagnostics")]
        if let Some(trace) = self.shared.validation_trace.get() {
            match &event {
                TicketNativeEvent::Completed { completion, .. } => trace.logical_completed(
                    completion.ticket,
                    completion.physical == PhysicalRoleCompletion::Confirmed,
                    matches!(&completion.result, Err(RoleFailure::Canceled)),
                ),
                TicketNativeEvent::Retired {
                    ticket,
                    accepted,
                    result,
                    ..
                } => trace.retired(*ticket, *accepted, result.is_ok()),
                TicketNativeEvent::RootQuiesced(fence) => {
                    trace.root_fence(fence.authority(), fence.permits_reuse())
                }
                TicketNativeEvent::Finished {
                    physical_shutdown_confirmed,
                    ..
                } => trace.finished(*physical_shutdown_confirmed),
                _ => {}
            }
        }
        Some(event)
    }
    pub fn completion_signal(&self) -> CompletionSignal {
        self.shared.signal.clone()
    }
    pub fn signal(&self) -> CompletionSignal {
        self.completion_signal()
    }
    pub fn ready_cpu_thread_observation(
        &self,
    ) -> Option<
        Result<rz_runtime::adaptive::CpuThreadReadback, rz_runtime::adaptive::CpuResourceError>,
    > {
        self.shared
            .ready_cpu
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
    /// Actual process owner startup is retained across roots; a new root does
    /// not require a new Ready event or a second model/session initialization.
    pub fn ready_clock(&self) -> Option<ContractSystemClock> {
        self.shared
            .ready_clock
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
    /// Only the verifier crosses out of the actual process owner. The private
    /// issuer stays on its thread and cannot be replaced by a caller's pair.
    pub fn native_acceptance_verifier(&self) -> Option<NativeAcceptanceVerifier> {
        self.shared
            .acceptance_verifier
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
    pub fn root_fence(&self) -> Option<NativeRootFence> {
        *self
            .shared
            .root_fence
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }
    pub fn occupancy(&self) -> usize {
        self.shared
            .controls
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }
    /// Pause this test owner's cancellation scan once, only after the original
    /// preparation is actually held. Dropping the release sender unblocks it;
    /// no gate can wait past the caller's original deadline.
    #[cfg(test)]
    pub(crate) fn pause_after_cancellation_scan_for_test(
        &self,
        preparation: PreparationId,
        until: Instant,
    ) -> Result<(mpsc::Receiver<()>, mpsc::SyncSender<()>), TicketServiceError> {
        if Instant::now() >= until
            || self.shared.closed.load(Ordering::Acquire)
            || self.shared.root_closed.load(Ordering::Acquire)
        {
            return Err(TicketServiceError::Closed);
        }
        if !self
            .shared
            .controls
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(&preparation)
        {
            return Err(TicketServiceError::UnknownPreparation);
        }
        let mut state = self
            .shared
            .cancellation_scan_test
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if state.installed {
            return Err(TicketServiceError::Duplicate);
        }
        let (entered, observed) = mpsc::sync_channel(1);
        let (release, released) = mpsc::sync_channel(1);
        state.installed = true;
        state.pending = Some(CancellationScanTestGate {
            preparation,
            entered,
            release: released,
            until,
        });
        self.shared.signal.notify();
        Ok((observed, release))
    }
    pub fn is_quarantined(&self) -> bool {
        self.shared.quarantine.load(Ordering::Acquire)
    }
    pub fn recovery_key(&self) -> Option<super::recovery::TicketNativeRecoveryKey> {
        self.recovery_key.clone()
    }
    pub fn startup_status(&self) -> NativeFactoryStartupStatus {
        match self.shared.factory_status.load(Ordering::Acquire) {
            1 => NativeFactoryStartupStatus::Ready,
            2 => NativeFactoryStartupStatus::BeforeNative,
            3 => NativeFactoryStartupStatus::PhysicalUnknown,
            4 => NativeFactoryStartupStatus::Unobserved,
            _ => NativeFactoryStartupStatus::Pending,
        }
    }
    /// Root stop is independent of process shutdown. It closes admission and
    /// revokes all questions through the control mailbox, even at queue pressure.
    pub fn begin_root_quiesce(&self, until: Instant) -> Result<(), TicketServiceError> {
        let _admission = self.recovery_key.as_ref().map(|_| {
            self.shared
                .admission
                .lock()
                .unwrap_or_else(|e| e.into_inner())
        });
        if self.shared.closed.load(Ordering::Acquire) {
            return Err(TicketServiceError::Closed);
        }
        let current = self
            .shared
            .root_authority
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let authority = (*current).ok_or(TicketServiceError::IdentityMismatch)?;
        self.shared.root_closed.store(true, Ordering::Release);
        for flag in self
            .shared
            .controls
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
        {
            flag.canceled.store(true, Ordering::Release);
        }
        let mut request = self
            .shared
            .root_quiesce
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if self
            .root_fence()
            .is_some_and(|fence| fence.authority() == authority && fence.permits_reuse())
        {
            return Ok(());
        }
        *request = Some((authority, request.map_or(until, |(_, old)| old.min(until))));
        self.shared.signal.notify();
        Ok(())
    }
    /// Revoke every queued/running/delivered question even if the work queue is full.
    pub fn begin_shutdown(&self, until: Instant) {
        self.shared.seal_admission();
        for flag in self
            .shared
            .controls
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
        {
            flag.canceled.store(true, Ordering::Release);
        }
        let mut selected = self
            .shared
            .shutdown_until
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *selected = Some(selected.map_or(until, |old| old.min(until)));
        self.shared.signal.notify();
    }
    /// No blocking join on the manager. Completion is the actual owner thread join.
    pub fn try_join(&mut self) -> Result<bool, TicketServiceError> {
        match &mut self.thread {
            OwnerThread::Legacy(thread) => {
                if thread.as_ref().is_some_and(|thread| !thread.is_finished()) {
                    return Ok(false);
                }
                if let Some(thread) = thread.take() {
                    thread.join().map_err(|_| TicketServiceError::WorkerGone)?;
                    #[cfg(feature = "native-validation-diagnostics")]
                    if let Some(trace) = self.shared.validation_trace.get() {
                        trace.joined();
                    }
                }
                Ok(true)
            }
            OwnerThread::Recoverable(thread) => thread
                .try_join()
                .map_err(|_| TicketServiceError::WorkerGone),
        }
    }
}
fn send_error<T>(error: mpsc::TrySendError<T>) -> TicketServiceError {
    match error {
        mpsc::TrySendError::Full(_) => TicketServiceError::Capacity,
        mpsc::TrySendError::Disconnected(_) => TicketServiceError::WorkerGone,
    }
}
impl<V, Q, B: TicketNativeBackend<V, Q>> Drop for TicketedNativeService<V, Q, B> {
    fn drop(&mut self) {
        self.shared.seal_admission();
        if let Some(key) = &self.recovery_key {
            // The registry already owns this exact receiver. Enabling its
            // recovery reader is an infallible atomic transition, not a move
            // from a field that is about to be destroyed.
            key.release_shared_receiver();
            key.try_retire_clean();
        }
        for cancel in self
            .shared
            .controls
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
        {
            cancel.canceled.store(true, Ordering::Release);
        }
        *self
            .shared
            .shutdown_until
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
        self.shared.signal.notify();
        // Detaching is not a physical shutdown claim. The owner still retains
        // the backend and its active inputs until the real fence or quarantine.
    }
}

struct Held<V, Q, P, O> {
    question: TicketPreparation<V, Q>,
    prepared: P,
    seal: PreparedTicketSeal,
    canceled: Arc<AtomicBool>,
    ticket: Option<RoleTicketEnvelope>,
    output: Option<Arc<O>>,
    completed: bool,
    ack_attempted: bool,
    evidence: Option<TicketExecutionEvidence>,
}
type HeldTickets<V, Q, P, O> = BTreeMap<PreparationId, Held<V, Q, P, O>>;

/// Original preparation and its output, transferred only after confirmation.
pub struct TicketRetainedPreparation<V, Q, P, O> {
    pub question: TicketPreparation<V, Q>,
    pub prepared: P,
    pub seal: PreparedTicketSeal,
    pub canceled: Arc<AtomicBool>,
    pub ticket: Option<RoleTicketEnvelope>,
    pub output: Option<Arc<O>>,
    pub completed: bool,
    pub ack_attempted: bool,
    pub evidence: Option<TicketExecutionEvidence>,
}
impl<V, Q, P, O> From<Held<V, Q, P, O>> for TicketRetainedPreparation<V, Q, P, O> {
    fn from(held: Held<V, Q, P, O>) -> Self {
        Self {
            question: held.question,
            prepared: held.prepared,
            seal: held.seal,
            canceled: held.canceled,
            ticket: held.ticket,
            output: held.output,
            completed: held.completed,
            ack_attempted: held.ack_attempted,
            evidence: held.evidence,
        }
    }
}

/// Exact worker-held state. Taking it does not itself confirm GPU completion;
/// the recovery key makes it available only after a same-owner confirmation.
pub struct TicketRetainedOwnerState<V, Q, P, O> {
    held: HeldTickets<V, Q, P, O>,
    pub active_command: Option<TicketPendingCommand<V, Q>>,
    queued_commands: mpsc::Receiver<Command<V, Q>>,
    pub original_command_handles: Vec<super::recovery::TicketUndeliveredEvent>,
    pub active_prepared: Option<Result<(P, RepresentationKey), RoleFailure>>,
    pub active_output: Option<TicketPhysicalResult<O>>,
    pub panic_payloads: Vec<Box<dyn std::any::Any + Send>>,
    pub observations: TicketNativeObservationFailures,
    pub active_ack_result: Option<Result<(), RoleFailure>>,
    pub original_root_fence: Option<NativeRootFence>,
}
impl<V, Q, P, O> TicketRetainedOwnerState<V, Q, P, O> {
    pub fn pop_preparation(
        &mut self,
    ) -> Option<(PreparationId, TicketRetainedPreparation<V, Q, P, O>)> {
        self.held.pop_first().map(|(id, held)| (id, held.into()))
    }
    pub fn pop_queued_command(&mut self) -> Option<TicketPendingCommand<V, Q>> {
        self.queued_commands.try_recv().ok().map(Into::into)
    }
}

/// The complete original worker state remains outside the unwind boundary.
/// No snapshot or replacement object may stand in for any of these fields.
struct WorkerState<V, Q, B: TicketNativeBackend<V, Q>> {
    backend: B,
    held: HeldTickets<V, Q, B::Prepared, B::Output>,
    commands: mpsc::Receiver<Command<V, Q>>,
    acks: mpsc::Receiver<Ack>,
    events: EventPublisher<B::Output>,
    active: Option<Command<V, Q>>,
    active_ack: Option<Ack>,
    active_prepared: Option<Result<(B::Prepared, RepresentationKey), RoleFailure>>,
    active_output: Option<TicketPhysicalResult<B::Output>>,
    active_panics: Vec<Box<dyn std::any::Any + Send>>,
    active_observations: TicketNativeObservationFailures,
    active_ack_result: Option<Result<(), RoleFailure>>,
    recovery_acks: Vec<super::recovery::TicketUndeliveredEvent>,
    recovery_commands: Vec<super::recovery::TicketUndeliveredEvent>,
    recovery_root_fence: Option<NativeRootFence>,
}

fn retire_canceled_unsubmitted<V, Q, P, O: Send + Sync + 'static>(
    held: &mut HeldTickets<V, Q, P, O>,
    events: &EventPublisher<O>,
    shared: &Shared,
) {
    let (unsubmitted, retired_preparations) = {
        let mut controls = shared.controls.lock().unwrap_or_else(|e| e.into_inner());
        let unsubmitted: Vec<_> = held
            .iter()
            .filter_map(|(id, value)| {
                (value.ticket.is_none()
                    && value.canceled.load(Ordering::Acquire)
                    && controls
                        .get(id)
                        .is_none_or(|control| control.queued_executions == 0))
                .then_some(*id)
            })
            .collect();
        let mut retired = Vec::with_capacity(unsubmitted.len());
        for id in &unsubmitted {
            if let Some(value) = held.remove(id) {
                retired.push(value);
            }
            controls.remove(id);
        }
        (unsubmitted, retired)
    };
    // Backend-prepared data and immutable view pins may have nontrivial
    // destructors; never run those while holding the control mailbox lock.
    drop(retired_preparations);
    for id in unsubmitted {
        let _ = events.send(TicketNativeEvent::PreparationFailed {
            preparation: id,
            failure: RoleFailure::Canceled,
        });
    }
}

fn worker<V: Send + Sync + 'static, Q: Send + 'static, B: TicketNativeBackend<V, Q>>(
    mut backend: B,
    commands: mpsc::Receiver<Command<V, Q>>,
    acks: mpsc::Receiver<Ack>,
    events: EventPublisher<B::Output>,
    shared: Arc<Shared>,
    limits: TicketServiceLimits,
) {
    let clock = backend.clock();
    let (acceptance_issuer, acceptance_verifier) = NativeAcceptanceIssuer::new_pair();
    *shared
        .acceptance_verifier
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = Some(acceptance_verifier.clone());
    *shared.ready_clock.lock().unwrap_or_else(|e| e.into_inner()) = Some(clock.clone());
    *shared.ready_cpu.lock().unwrap_or_else(|e| e.into_inner()) =
        Some(rz_runtime::adaptive::observe_current_cpu_thread(
            rz_runtime::adaptive::CpuActor::NativeCaller,
        ));
    let mut held: HeldTickets<V, Q, B::Prepared, B::Output> = BTreeMap::new();
    let mut request_high_water = 0;
    if events
        .send(TicketNativeEvent::Ready {
            clock: clock.clone(),
        })
        .is_err()
    {
        shared.closed.store(true, Ordering::Release);
    }
    loop {
        let observed_signal = shared.signal.version();
        for _ in 0..shared.pending_capacity {
            let Ok(ack) = acks.try_recv() else {
                break;
            };
            acknowledge(
                &mut backend,
                &mut held,
                ack,
                &events,
                &shared,
                &clock,
                limits.review_token_budget,
                &acceptance_issuer,
            );
        }
        retire_canceled_unsubmitted(&mut held, &events, &shared);
        #[cfg(test)]
        {
            let gate = {
                let mut state = shared
                    .cancellation_scan_test
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                if state.pending.as_ref().is_some_and(|gate| {
                    held.get(&gate.preparation)
                        .is_some_and(|value| value.ticket.is_none())
                }) {
                    state.pending.take()
                } else {
                    None
                }
            };
            if let Some(gate) = gate {
                let _ = gate.entered.send(());
                let _ = gate
                    .release
                    .recv_timeout(gate.until.saturating_duration_since(Instant::now()));
            }
        }
        let until = *shared
            .shutdown_until
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(until) = until {
            let known =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| backend.finish(until)))
                    .unwrap_or(false);
            if !known {
                quarantine(backend, held, commands, &shared);
            } else {
                shared
                    .controls
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clear();
            }
            let _ = events.send(TicketNativeEvent::Finished {
                physical_shutdown_confirmed: known,
            });
            return;
        }
        let root_quiesce = *shared
            .root_quiesce
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some((authority, until)) = root_quiesce {
            // This owner is between physical calls. The returned run result
            // already established completion; canceled queued work never starts.
            for command in commands.try_iter() {
                match command {
                    Command::Prepare(id, _, _) => {
                        shared.retire(id);
                        let _ = events.send(TicketNativeEvent::PreparationFailed {
                            preparation: id,
                            failure: RoleFailure::Canceled,
                        });
                    }
                    Command::Execute(id, _, _) => {
                        let _ = events.send(TicketNativeEvent::Rejected {
                            preparation: id,
                            error: TicketServiceError::Closed,
                        });
                    }
                }
            }
            // Cancellation may have arrived after this iteration's first scan.
            // Retire the original never-submitted preparation exactly once,
            // before physical quiesce clears its remaining root custody.
            retire_canceled_unsubmitted(&mut held, &events, &shared);
            let cleanup_known = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                held.values().all(|value| {
                    value
                        .ticket
                        .is_none_or(|ticket| value.completed && backend.discard(&ticket).is_ok())
                })
            }))
            .unwrap_or(false);
            let physical_known = cleanup_known
                && std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    backend.quiesce(until)
                }))
                .unwrap_or(false);
            let fence = NativeRootFence {
                authority,
                physical_completion_confirmed: physical_known,
                ticket_cleanup_confirmed: cleanup_known,
            };
            if !fence.permits_reuse() {
                quarantine(backend, held, commands, &shared);
                let _ = events.send(TicketNativeEvent::RootQuiesced(fence));
                let _ = events.send(TicketNativeEvent::Finished {
                    physical_shutdown_confirmed: false,
                });
                return;
            }
            // Successful root fence can release this root's pins/host outputs,
            // while the backend, model, session and owner thread remain alive.
            held.clear();
            shared
                .controls
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clear();
            // This guard makes old-root fence publication/request removal one
            // transition against replace_root and begin_root_quiesce. A new
            // root's request cannot be overwritten by old-root cleanup.
            let _authority_transition = shared
                .root_authority
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            *shared.root_fence.lock().unwrap_or_else(|e| e.into_inner()) = Some(fence);
            *shared
                .root_quiesce
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = None;
            drop(_authority_transition);
            let _ = events.send(TicketNativeEvent::RootQuiesced(fence));
            continue;
        }
        let command = match commands.try_recv() {
            Ok(command) => command,
            Err(mpsc::TryRecvError::Empty) => {
                shared
                    .signal
                    .wait_changed(observed_signal, Duration::from_millis(50));
                continue;
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                *shared
                    .shutdown_until
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
                continue;
            }
        };
        if let Command::Execute(id, _, true) = &command {
            if let Some(control) = shared
                .controls
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get_mut(id)
            {
                control.queued_executions = control
                    .queued_executions
                    .checked_sub(1)
                    .expect("accepted Execute commands have one queued registration");
            }
        }
        match command {
            Command::Prepare(id, question, canceled) => {
                let valid = question
                    .seal
                    .validate(question.seal.authority, limits.review_token_budget)
                    .and_then(|_| question.deadline.accepts(clock.domain(), clock.now()));
                let prepared = if valid.is_err() {
                    Err(RoleFailure::InvalidOutput)
                } else if canceled.load(Ordering::Acquire) {
                    Err(RoleFailure::Canceled)
                } else {
                    backend.prepare(&question, &canceled)
                };
                match prepared {
                    Ok((prepared, representation))
                        if backend
                            .prepared_host_bytes(&prepared)
                            .checked_add(backend.question_host_bytes(&question))
                            .is_some_and(|bytes| {
                                bytes <= limits.prepared_host_bytes_per_ticket
                            })
                            && representation.role == question.role =>
                    {
                        let seal = PreparedTicketSeal {
                            preparation: id,
                            question: question.question,
                            view: question.seal,
                            representation,
                            role: question.role,
                            original_deadline: question.deadline,
                        };
                        held.insert(
                            id,
                            Held {
                                question,
                                prepared,
                                seal,
                                canceled,
                                ticket: None,
                                output: None,
                                completed: false,
                                ack_attempted: false,
                                evidence: None,
                            },
                        );
                        let _ = events.send(TicketNativeEvent::Prepared(seal));
                    }
                    result => {
                        let failure = result.err().unwrap_or(RoleFailure::InvalidOutput);
                        shared.retire(id);
                        let _ = events.send(TicketNativeEvent::PreparationFailed {
                            preparation: id,
                            failure,
                        });
                    }
                }
            }
            Command::Execute(id, ticket, _) => {
                let mut authority = shared
                    .root_authority
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                let error = match held.get(&id) {
                    None => Some(TicketServiceError::UnknownPreparation),
                    Some(value) if value.ticket.is_some() => Some(TicketServiceError::Duplicate),
                    Some(value)
                        if !value.seal.matches(&ticket)
                            || ticket.id.sequence <= request_high_water
                            || authority.is_some_and(|old| old != ticket.authority)
                            || ticket
                                .validate(clock.domain(), clock.now(), limits.review_token_budget)
                                .is_err() =>
                    {
                        Some(TicketServiceError::IdentityMismatch)
                    }
                    _ => None,
                };
                if let Some(error) = error {
                    let _ = events.send(TicketNativeEvent::Rejected {
                        preparation: id,
                        error,
                    });
                    continue;
                }
                authority.get_or_insert(ticket.authority);
                drop(authority);
                request_high_water = ticket.id.sequence;
                let value = held.get_mut(&id).expect("checked preparation");
                value.ticket = Some(ticket);
                let mut service_started = None;
                let result = if value.canceled.load(Ordering::Acquire) {
                    TicketPhysicalResult {
                        physical: PhysicalRoleCompletion::Confirmed,
                        output: Err(RoleFailure::Canceled),
                    }
                } else {
                    let _ = events.send(TicketNativeEvent::Started(ticket));
                    let started_at = Instant::now();
                    service_started = Some(started_at);
                    let _ = events.send(TicketNativeEvent::OwnerRunStarted { ticket, started_at });
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        backend.run_owned(
                            TicketOwnerDispatchContext {
                                verifier: &acceptance_verifier,
                                ticket: &ticket,
                                question: &value.question,
                                prepared: &value.prepared,
                                canceled: &value.canceled,
                            },
                            &value.canceled,
                        )
                    }))
                    .unwrap_or(TicketPhysicalResult {
                        physical: PhysicalRoleCompletion::Unknown,
                        output: Err(RoleFailure::PhysicalUnknown),
                    })
                };
                let physical = result.physical;
                // A pre-run cancellation has no current owner invocation.
                // An unwound/unknown physical call cannot publish a complete
                // accounting observation even if its backend retained one.
                value.evidence =
                    if service_started.is_some() && physical == PhysicalRoleCompletion::Confirmed {
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            backend.execution_evidence(&ticket)
                        }))
                        .ok()
                        .flatten()
                    } else {
                        None
                    };
                let mut output = match result.output {
                    Ok(output)
                        if backend.output_host_bytes(&output)
                            <= limits.output_host_bytes_per_ticket =>
                    {
                        let output = Arc::new(output);
                        value.output = Some(Arc::clone(&output));
                        Ok(output)
                    }
                    Ok(_) => Err(RoleFailure::InvalidOutput),
                    Err(failure) => Err(failure),
                };
                if physical == PhysicalRoleCompletion::Unknown {
                    output = Err(RoleFailure::PhysicalUnknown);
                } else if value.canceled.load(Ordering::Acquire)
                    || shared.closed.load(Ordering::Acquire)
                    || shared.root_closed.load(Ordering::Acquire)
                    || shared
                        .root_authority
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .is_some_and(|authority| authority != ticket.authority)
                {
                    output = Err(RoleFailure::Canceled);
                } else if ticket
                    .validate(clock.domain(), clock.now(), limits.review_token_budget)
                    .is_err()
                {
                    output = Err(RoleFailure::Expired);
                }
                value.completed = physical == PhysicalRoleCompletion::Confirmed;
                let observation_failures = if service_started.is_some() {
                    take_observation_failures::<V, Q, B>(&mut backend, ticket)
                } else {
                    TicketNativeObservationFailures::default()
                };
                let _ = events.send(TicketNativeEvent::Completed {
                    completion: RoleCompletion {
                        ticket,
                        physical,
                        result: output,
                    },
                    evidence: value.evidence,
                    owner_run: TicketOwnerRunWitness {
                        ticket,
                        invoked: service_started.is_some(),
                    },
                    owner_service_elapsed: service_started.map(|started| started.elapsed()),
                    observation_failures,
                });
                if physical == PhysicalRoleCompletion::Unknown {
                    quarantine(backend, held, commands, &shared);
                    let _ = events.send(TicketNativeEvent::Finished {
                        physical_shutdown_confirmed: false,
                    });
                    return;
                }
                // No logical acknowledgement barrier here. A second Fresh
                // execution begins while this immutable output awaits accept.
            }
        }
    }
}

// This owner-only boundary checks independent ticket/view, clock, publication
// and issuer authority without folding them into a caller-constructible input.
#[allow(clippy::too_many_arguments)]
fn acknowledge<V, Q, B: TicketNativeBackend<V, Q>>(
    backend: &mut B,
    held: &mut HeldTickets<V, Q, B::Prepared, B::Output>,
    ack: Ack,
    events: &EventPublisher<B::Output>,
    shared: &Shared,
    clock: &ContractSystemClock,
    token_budget: u16,
    acceptance_issuer: &NativeAcceptanceIssuer,
) {
    let (id, ticket, accepted, mut permit) = match ack {
        Ack::Accept(id, ticket, permit) => (id, ticket, true, permit),
        Ack::Discard(id, ticket) => (id, ticket, false, None),
    };
    let Some(value) = held.get_mut(&id) else {
        reject_acceptance(
            events,
            id,
            ticket,
            TicketServiceError::UnknownPreparation,
            permit,
        );
        return;
    };
    if !value.completed || value.ticket != Some(ticket) {
        reject_acceptance(
            events,
            id,
            ticket,
            TicketServiceError::IdentityMismatch,
            permit,
        );
        return;
    }
    if accepted && value.ack_attempted {
        reject_acceptance(events, id, ticket, TicketServiceError::Duplicate, permit);
        return;
    }
    if accepted {
        value.ack_attempted = true;
    }
    let mut backend_called = false;
    let result = if accepted {
        if value.canceled.load(Ordering::Acquire)
            || shared.closed.load(Ordering::Acquire)
            || shared.root_closed.load(Ordering::Acquire)
        {
            Err(RoleFailure::Canceled)
        } else if ticket
            .validate(clock.domain(), clock.now(), token_budget)
            .is_err()
        {
            Err(RoleFailure::Expired)
        } else if let Some(output) = &value.output {
            if permit.as_ref().is_some_and(|permit| {
                permit.ticket() != &ticket
                    || backend.acceptance_output_digest(output) != Some(permit.output_digest())
            }) {
                Err(RoleFailure::InvalidOutput)
            } else {
                backend_called = true;
                backend.accept(&ticket, &value.question, &value.prepared, output)
            }
        } else {
            Err(RoleFailure::InvalidOutput)
        }
    } else {
        backend_called = true;
        backend.discard(&ticket)
    };
    let observation_failures = if backend_called {
        take_observation_failures::<V, Q, B>(backend, ticket)
    } else {
        TicketNativeObservationFailures::default()
    };
    if !accepted || result.is_ok() {
        held.remove(&id);
        shared.retire(id);
    }
    let event = TicketNativeEvent::Retired {
        ticket,
        accepted: accepted && result.is_ok(),
        native_receipt: if accepted && result.is_ok() {
            permit
                .take()
                .map(|permit| acceptance_issuer.issue_success(permit))
        } else {
            None
        },
        rejected_permit: permit,
        result,
        observation_failures,
    };
    // An acknowledgement can partially record telemetry before failing. It is
    // attempted once, preserving input/output until explicit discard or the
    // physical shutdown fence. It never authorizes a callback replay.
    // Every acknowledgement, including failed permit/partial callback state,
    // uses reliable terminal publication. Ordinary admission closes at 64;
    // remaining bounded work/acks/retirements fit the 192-event reserve. There
    // is no try_send(Full) drop or coalescing of owner receipts.
    let _ = events.send(event);
}

fn reject_acceptance<O: Send + Sync + 'static>(
    events: &EventPublisher<O>,
    preparation: PreparationId,
    ticket: RoleTicketEnvelope,
    error: TicketServiceError,
    permit: Option<NativeAcceptancePermit>,
) {
    let event = match permit {
        Some(permit) => TicketNativeEvent::AcceptanceRejected {
            preparation,
            ticket,
            error,
            permit,
        },
        None => TicketNativeEvent::Rejected { preparation, error },
    };
    let _ = events.send(event);
}

fn quarantine<V, Q, B: TicketNativeBackend<V, Q>>(
    backend: B,
    held: HeldTickets<V, Q, B::Prepared, B::Output>,
    commands: mpsc::Receiver<Command<V, Q>>,
    shared: &Shared,
) {
    shared.closed.store(true, Ordering::Release);
    shared.quarantine.store(true, Ordering::Release);
    for cancel in shared
        .controls
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .values()
    {
        cancel.canceled.store(true, Ordering::Release);
    }
    // Unknown physical completion cannot be repaired by handle drop. Keeping
    // the complete owner is bounded (16 tickets) and intentionally sticky;
    // reclaim requires a future capability proving this exact physical fence.
    let queued: Vec<_> = commands.try_iter().collect();
    std::mem::forget(Box::new((backend, held, queued, commands)));
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
