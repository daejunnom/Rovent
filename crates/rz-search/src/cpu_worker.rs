//! Private own-CPU worker and manager-owned exact native credit.
//!
//! A job fixes Rules/history, restriction, evaluator, conditions and deadline.
//! A grant changes occupancy only. Tokens never leave this worker and actual
//! invocation deltas, rather than progress snapshots, settle reservations.

use crate::contract_time::{ContractClock, InstantClock};
use crate::cpu::{
    CpuCompletion, CpuControlMailbox, CpuControlledReport, CpuEngine, CpuError,
    CpuIterationProgress, CpuLimits, CpuMailboxScope, CpuReciprocalReply, CpuReciprocalRequest,
    CpuReciprocalScope, CpuResumePolicy, CpuResumeToken, CpuSliceControl, CpuWork,
};
use crate::cpu_value::nnuec::v2::capture::{
    CaptureError, NnuecActiveCapture, NnuecCaptureEnding, NnuecCaptureReservation,
};
use rz_contracts::pals::SearchAuthority;
use rz_contracts::pals_adaptive::{CpuJobEnvelope, CpuJobId, CpuWorkerId, SliceGrant};
use rz_contracts::{Digest, ExecutionId};
use rz_position::contracts::ContractPosition;
use rz_position::{BoardMove, Position};
use rz_runtime::CompletionSignal;
use rz_runtime::adaptive::{
    ControlSnapshot, CpuActor, CpuResourceError, CpuThreadAffinityApplied,
    CpuThreadAffinityFinishError, CpuThreadAffinityObservation, CpuThreadAffinityStartError,
    CpuThreadReadback, CurrentCpuThreadAffinityOwner, LogicalCpu, MailboxError, ReliablePermit,
    WorkerMailbox, observe_current_cpu_thread,
};
use sha2::{Digest as _, Sha256};
use std::collections::HashMap;
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[path = "cpu_worker/prepared_value.rs"]
mod prepared_value;
pub use prepared_value::{
    InstallId, InstallRejected, PreparedValueInstallation, ValueInstallReadback,
    ValueInstallReceipt,
};
#[path = "cpu_worker/deferred_value.rs"]
pub(crate) mod deferred_value;
pub use deferred_value::{
    DeferredNnuecCpuPlan, DeferredNnuecRSource, MaterializationCompletion, MaterializationHandoff,
    MaterializationRequest, MaterializedNnuecEngine, NnuecMaterializationReleaseAck,
    NnuecMaterializationReleaseCompletion, NnuecMaterializationReleaseId, NnuecReadyRegistration,
    StartingNnuecWorker,
};
#[path = "cpu_worker/ledger_registration.rs"]
mod ledger_registration;
pub use ledger_registration::CpuLedgerRegistrationObservation;
#[path = "cpu_worker/retired_value.rs"]
mod retired_value;
pub use retired_value::{
    NnuecEngineReleaseAck, NnuecEngineRetirementCompletion, NnuecEngineRetirementId,
    RecoveredNnuecEngine,
};

pub const CPU_WORKER_COMMAND_CAPACITY: usize = 8;
pub const CPU_JOB_CAPACITY: usize = 64;
pub const CPU_FIXED_SLICE_NODES: u64 = 4096;
pub const CPU_FIXED_SOFT_LEASE: Duration = Duration::from_millis(8);

/// A read-only observation of the original NNUEC Bound control. This is not
/// proof of cleanup, nor permission to accept a failed root as a normal stop.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NnuecAuthorityRevocation {
    Cancelled,
    Expired,
    ActorStopped,
    Invalidated,
}
impl From<NnuecAuthorityRevocation> for crate::cpu_value::CpuValueAuthorityRevocation {
    fn from(reason: NnuecAuthorityRevocation) -> Self {
        match reason {
            NnuecAuthorityRevocation::Cancelled => Self::Cancelled,
            NnuecAuthorityRevocation::Expired => Self::Expired,
            NnuecAuthorityRevocation::ActorStopped => Self::ActorStopped,
            NnuecAuthorityRevocation::Invalidated => Self::Invalidated,
        }
    }
}

/// A typed soft error alone is not enough to recover a worker. Require the
/// actual immutable V2 owner, the original installed binding, and the same
/// original Bound's present control. Expiry may monotonically be followed
/// by its cancellation, but the original Expired observation is not rewritten.
/// Hard revocations and unknown work remain errors. This accepts no value.
fn owned_soft_value_terminal(
    engine: &CpuEngine,
    binding: Option<Digest>,
    error: &CpuError,
    actual_work: Option<CpuWork>,
) -> Option<CpuCompletion> {
    actual_work?;
    let completion = error.value_authority_stop()?;
    let CpuError::Value(crate::cpu_value::CpuValueError::AuthorityRevoked(reason)) = error else {
        return None;
    };
    let owner = engine.nnuec_retirement_owner().ok().flatten()?;
    if binding != Some(owner.binding_digest())
        || !original_soft_revocation_matches(
            *reason,
            owner.authority_revocation(),
            owner.original_hard_deadline(),
            Instant::now(),
        )
    {
        return None;
    }
    Some(completion)
}

/// Pure control observation only. It cannot produce an owner or a terminal
/// receipt. In particular a cancelled-but-not-expired Bound cannot attest a
/// fabricated earlier Expired result merely because both are soft reasons.
fn original_soft_revocation_matches(
    observed: crate::cpu_value::CpuValueAuthorityRevocation,
    current: Option<NnuecAuthorityRevocation>,
    original_deadline: Instant,
    observed_at: Instant,
) -> bool {
    use crate::cpu_value::CpuValueAuthorityRevocation;
    match (observed, current) {
        (CpuValueAuthorityRevocation::Cancelled, Some(NnuecAuthorityRevocation::Cancelled)) => true,
        (
            CpuValueAuthorityRevocation::Expired,
            Some(NnuecAuthorityRevocation::Expired | NnuecAuthorityRevocation::Cancelled),
        ) => observed_at >= original_deadline,
        _ => false,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CpuWorkerError {
    Invalid(&'static str),
    NnuecAuthorityRevoked(NnuecAuthorityRevocation),
    Capacity,
    UnknownJob,
    AlreadyReserved,
    Duplicate,
    UnknownUsage,
    Quarantined,
    Disconnected,
    Cpu(CpuError),
}
impl From<CpuError> for CpuWorkerError {
    fn from(error: CpuError) -> Self {
        Self::Cpu(error)
    }
}

/// Only a worker's definite pre-enqueue rejection can create this receipt.
/// A missing/failed response from an admitted execution cannot release credit.
#[derive(Clone, Debug)]
pub struct CpuAdmissionRejection {
    grant: SliceGrant,
    verified_unqueued: bool,
}
impl CpuAdmissionRejection {
    pub fn grant(&self) -> SliceGrant {
        self.grant
    }
    /// A rejected duplicate, foreign owner or poisoned submission boundary is
    /// not proof that the original execution never entered its owner queue.
    pub fn is_verified_unqueued(&self) -> bool {
        self.verified_unqueued
    }
}

pub struct CpuSubmissionRejected {
    reason: CpuWorkerError,
    // Definite rejection is exceptional. Keep its large immutable authority
    // and prepared Job off the success-path Result stack without changing the
    // owned payload returned to callers.
    proof: Box<CpuAdmissionRejection>,
    job: Option<Box<CpuJob>>,
}
impl std::fmt::Debug for CpuSubmissionRejected {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CpuSubmissionRejected")
            .field("reason", &self.reason)
            .field("proof", &self.proof)
            .field("job_retained", &self.job.is_some())
            .finish()
    }
}
impl CpuSubmissionRejected {
    pub fn reason(&self) -> &CpuWorkerError {
        &self.reason
    }
    pub fn proof(&self) -> &CpuAdmissionRejection {
        &self.proof
    }
    pub fn into_parts(self) -> (CpuWorkerError, CpuAdmissionRejection, Option<CpuJob>) {
        (self.reason, *self.proof, self.job.map(|job| *job))
    }
    fn from_command(reason: CpuWorkerError, command: CpuCommand, verified_unqueued: bool) -> Self {
        let (grant, job) = match command {
            CpuCommand::Begin(job, grant) => (grant, Some(job)),
            CpuCommand::Resume(grant) | CpuCommand::ResumeWithReply(grant, _) => (grant, None),
        };
        // Only this owner's definite pre-enqueue rejection proves that the
        // prepared original never executed. Duplicates/foreign/unknown
        // submissions must leave any primary capture custody untouched.
        if verified_unqueued && let Some(job) = &job {
            let _ = job.retire_nnuec_capture(grant.owner, NnuecCaptureEnding::NotStarted);
        }
        Self {
            reason,
            proof: Box::new(CpuAdmissionRejection {
                grant,
                verified_unqueued,
            }),
            job,
        }
    }
}

pub fn cpu_root_restriction_digest(moves: Option<&[BoardMove]>) -> Digest {
    let mut hash = Sha256::new();
    hash.update(b"rz-cpu-root-restriction/1");
    hash.update([u8::from(moves.is_some())]);
    if let Some(moves) = moves {
        hash.update((moves.len() as u64).to_be_bytes());
        for movement in moves {
            let encoded = rz_contracts::pals::Move16::encode(
                rz_contracts::Move::try_from(*movement).expect("Rules board move is representable"),
            );
            hash.update(encoded.bits().to_be_bytes());
        }
    }
    Digest(hash.finalize().into())
}

pub fn cpu_evaluation_namespace_digest(engine: &CpuEngine) -> Result<Digest, CpuWorkerError> {
    let bytes = serde_json::to_vec(engine.value_identity())
        .map_err(|_| CpuWorkerError::Invalid("CPU value identity serialization"))?;
    let mut hash = Sha256::new();
    hash.update(b"rz-cpu-evaluation-namespace/1");
    hash.update(bytes);
    Ok(Digest(hash.finalize().into()))
}

pub fn cpu_search_conditions_digest(engine: &CpuEngine) -> Digest {
    let mut hash = Sha256::new();
    hash.update(b"rz-cpu-search-conditions/1");
    hash.update(engine.search_conditions().as_bytes());
    Digest(hash.finalize().into())
}

pub fn compiled_cooperative_cpu_owner_sha256() -> [u8; 32] {
    let mut hash = Sha256::new();
    for bytes in [
        b"rz-cooperative-cpu-owner/1".as_slice(),
        include_bytes!("cpu.rs").as_slice(),
        include_bytes!("cpu/stack.rs").as_slice(),
        include_bytes!("cpu_worker.rs").as_slice(),
        include_bytes!("cpu_worker/prepared_value.rs").as_slice(),
        include_bytes!("cpu_worker/deferred_value.rs").as_slice(),
        include_bytes!("cpu_worker/ledger_registration.rs").as_slice(),
    ] {
        hash.update((bytes.len() as u64).to_be_bytes());
        hash.update(bytes);
    }
    hash.finalize().into()
}

pub struct CpuJob {
    envelope: CpuJobEnvelope,
    position: Position,
    root_moves: Option<Vec<BoardMove>>,
    target_depth: u16,
    hard_deadline: Instant,
    clock: InstantClock,
    pub mailbox: Arc<CpuControlMailbox>,
    reciprocal_scope: Option<CpuReciprocalScope>,
    nnuec_capture: Option<NnuecCaptureReservation>,
}

impl CpuJob {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        envelope: CpuJobEnvelope,
        position: ContractPosition,
        root_moves: Option<Vec<BoardMove>>,
        target_depth: u16,
        clock: &InstantClock,
        engine: &CpuEngine,
        owner: CpuWorkerId,
        signal: CompletionSignal,
    ) -> Result<Self, CpuWorkerError> {
        if engine.resume_policy() != CpuResumePolicy::PausedStack {
            return Err(CpuWorkerError::Invalid("CPU cooperative policy"));
        }
        Self::from_profile(
            envelope,
            position,
            root_moves,
            target_depth,
            clock,
            engine.config().max_depth,
            prepared_value::registered_namespace(engine, None)?,
            cpu_search_conditions_digest(engine),
            owner,
            signal,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn from_profile(
        envelope: CpuJobEnvelope,
        position: ContractPosition,
        root_moves: Option<Vec<BoardMove>>,
        target_depth: u16,
        clock: &InstantClock,
        max_depth: u16,
        evaluation_namespace: Digest,
        search_conditions: Digest,
        owner: CpuWorkerId,
        signal: CompletionSignal,
    ) -> Result<Self, CpuWorkerError> {
        let state = position
            .export()
            .map_err(|_| CpuWorkerError::Invalid("Rules export failed"))?;
        if state.snapshot().identity() != envelope.state_identity {
            return Err(CpuWorkerError::Invalid("job Rules identity mismatch"));
        }
        if envelope.total_native_node_credit == 0
            || envelope.id.sequence == 0
            || envelope.id.authority.epoch != envelope.hard_deadline.clock.0
        {
            return Err(CpuWorkerError::Invalid(
                "job credit or authority is invalid",
            ));
        }
        envelope
            .hard_deadline
            .accepts(
                clock.domain(),
                clock
                    .now()
                    .map_err(|_| CpuWorkerError::Invalid("CPU clock observation failed"))?,
            )
            .map_err(|_| CpuWorkerError::Invalid("CPU job deadline is not live"))?;
        let hard_deadline = clock
            .instant_at(envelope.hard_deadline.at)
            .map_err(|_| CpuWorkerError::Invalid("CPU deadline is not representable"))?;
        if target_depth == 0 || target_depth > max_depth {
            return Err(CpuWorkerError::Invalid("CPU cooperative depth or policy"));
        }
        if envelope.root_restriction != cpu_root_restriction_digest(root_moves.as_deref())
            || envelope.evaluation_namespace != evaluation_namespace
            || envelope.search_conditions != search_conditions
        {
            return Err(CpuWorkerError::Invalid("CPU immutable namespace mismatch"));
        }
        if let Some(moves) = &root_moves {
            let legal = state.legal_moves().moves();
            if moves.is_empty()
                || moves.len() > 256
                || moves.iter().enumerate().any(|(i, movement)| {
                    moves[..i].contains(movement)
                        || !legal
                            .contains(&rz_contracts::Move::try_from(*movement).expect("Rules move"))
                })
            {
                return Err(CpuWorkerError::Invalid("CPU root restriction is illegal"));
            }
        }
        let mailbox = Arc::new(CpuControlMailbox::new(
            CpuMailboxScope {
                job: envelope.id,
                owner,
            },
            signal,
        ));
        Ok(Self {
            envelope,
            position: position.into_position(),
            root_moves,
            target_depth,
            hard_deadline,
            clock: *clock,
            mailbox,
            reciprocal_scope: None,
            nnuec_capture: None,
        })
    }
    pub fn envelope(&self) -> CpuJobEnvelope {
        self.envelope
    }
    /// Optional collection custody is attached before Begin. This does not
    /// activate a worker, evaluate a value or reserve additional node credit.
    #[allow(clippy::result_large_err)]
    pub fn enable_nnuec_capture(
        &mut self,
        capture: NnuecCaptureReservation,
    ) -> Result<(), (CpuWorkerError, NnuecCaptureReservation)> {
        if self.nnuec_capture.is_some() {
            return Err((CpuWorkerError::Duplicate, capture));
        }
        if capture
            .prepare_job(
                self.envelope,
                self.mailbox.scope().owner,
                &self.position,
                self.clock,
            )
            .is_err()
        {
            return Err((
                CpuWorkerError::Invalid("NNUEC capture does not match prepared Job"),
                capture,
            ));
        }
        self.nnuec_capture = Some(capture);
        Ok(())
    }
    pub fn nnuec_capture(&self) -> Option<&NnuecCaptureReservation> {
        self.nnuec_capture.as_ref()
    }
    fn retire_nnuec_capture(
        &self,
        owner: CpuWorkerId,
        ending: NnuecCaptureEnding,
    ) -> Result<(), CaptureError> {
        match &self.nnuec_capture {
            Some(capture) => capture
                .retire(self.envelope, owner, ending)
                .inspect_err(|error| {
                    capture.record_error(error.clone());
                }),
            None => Ok(()),
        }
    }

    /// Configure before Begin. The original Job, owner and hard deadline stay
    /// immutable; a question never creates a second engine or paused stack.
    pub fn enable_reciprocal_questions(
        &mut self,
        scope: CpuReciprocalScope,
    ) -> Result<(), CpuWorkerError> {
        use rz_contracts::rovent::SearchRecipient;
        if self.reciprocal_scope.is_some() {
            return Err(CpuWorkerError::Duplicate);
        }
        if scope.job != self.envelope
            || scope.owner != self.mailbox.scope().owner
            || scope.parent.id.authority != self.envelope.id.authority
            || scope.parent.anchor != self.envelope.state_identity
            || scope.parent.question_digest != self.envelope.question
            || scope.parent.original_deadline != self.envelope.hard_deadline
            || scope.parent.recipient != SearchRecipient::Researcher
            || !matches!(
                scope.recipient,
                SearchRecipient::Proposer | SearchRecipient::Critic
            )
            || !(1..=64).contains(&scope.max_questions)
            || scope.node_interval == 0
        {
            return Err(CpuWorkerError::Invalid(
                "CPU reciprocal scope differs from its original Job",
            ));
        }
        self.reciprocal_scope = Some(scope);
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct CpuSliceReport {
    pub grant: SliceGrant,
    /// A real owner-sealed waiting question, distinct from occupancy Yield.
    /// The private continuation token remains exclusively on that CPU worker.
    pub reciprocal_request: Option<CpuReciprocalRequest>,
    /// None is unknown and must quarantine the outstanding reservation.
    pub actual_work: Option<CpuWork>,
    pub result: Result<CpuControlledReport, CpuError>,
    /// Total Job work including earlier slices; progress nodes are invocation-local.
    /// An admission rejection did not inspect or execute a continuation, so
    /// its cumulative snapshot is not settlement evidence; the known delta is zero.
    pub cumulative_nodes: u64,
    pub progress_sequence: u64,
    pub continuation_retained: bool,
    pub terminal: bool,
    pub admission_rejected: bool,
    pub controls: Option<ControlSnapshot>,
}

#[derive(Clone, Debug)]
pub struct CpuJobTermination {
    pub job: CpuJobId,
    pub owner: CpuWorkerId,
    pub last_execution: ExecutionId,
    pub completion: CpuCompletion,
    pub controls: ControlSnapshot,
    pub retained_bytes: usize,
    /// Owner shutdown is distinct from a sequenced job cancel request.
    pub shutdown: bool,
}

#[derive(Clone, Debug)]
// Reliable storage already has a fixed admission/report-seat bound. Boxing
// the slice here would add a heap allocation to every native slice; the idle
// termination variant is less frequent and is kept in the same bounded queue.
#[allow(clippy::large_enum_variant)]
pub enum CpuWorkerEvent {
    Slice(CpuSliceReport),
    JobTerminated(CpuJobTermination),
}

/// Reliable receipts are returned before disposing the closed owner. A joined
/// thread alone is not proof that every admitted execution has settled work.
#[derive(Debug)]
pub struct CpuWorkerShutdown {
    pub owner: CpuWorkerId,
    pub events: Vec<CpuWorkerEvent>,
    pub last_progress: Option<CpuWorkerProgress>,
    pub thread_joined: bool,
    pub thread_panicked: bool,
    pub mailbox: rz_runtime::adaptive::MailboxOccupancy,
    /// Ready-boundary observation, distinct from the declared engine profile.
    /// None is not observed; Err retains a startup probe/role failure and never
    /// attests a complete inventory or an applied resource setting.
    pub ready_readback: Option<Result<CpuThreadReadback, CpuResourceError>>,
    /// Opt-in current-thread ownership evidence. A joined or recovered CPU
    /// engine does not substitute for successful explicit restoration.
    pub thread_affinity: CpuWorkerThreadAffinityShutdown,
    /// The actual engine may be taken only after this known joined exit. This
    /// does not replace settlement of the returned reliable event receipts.
    pub engine_recoverable: bool,
    /// The independently bounded installation terminal is never discarded by
    /// shutdown. A failed/absent channel acknowledgement remains an error.
    pub value_installation_receipt: Option<ValueInstallReceipt>,
    pub value_installation_error: Option<CpuWorkerError>,
}

struct CreditAccount {
    envelope: CpuJobEnvelope,
    spent: u64,
    reservation: Option<SliceGrant>,
    last_settled_sequence: u64,
    last_started_sequence: u64,
    last_control_sequence: u64,
    terminal: bool,
    quarantine: bool,
    owner: Option<CpuWorkerId>,
}

/// Root-scoped single manager writer. Unknown failure retains reservation;
/// snapshots do not settle credit. Completed receipts are externally retained.
pub struct CpuCreditLedger {
    identity: CpuLedgerIdentity,
    jobs: HashMap<CpuJobId, CreditAccount>,
    authority: SearchAuthority,
    clock: InstantClock,
    global_limit: u64,
    global_spent: u64,
    highest_job_sequence: u64,
    highest_execution_sequence: u64,
}

/// In-process identity of the single writer, never a serializable authority.
/// Retaining the Arc prevents a later ledger from reusing this identity even
/// when it has the same root and credit settings.
#[derive(Clone, Debug)]
pub struct CpuLedgerIdentity(Arc<()>);
impl PartialEq for CpuLedgerIdentity {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for CpuLedgerIdentity {}

/// Issued only after this ledger has accepted and settled the supplied report.
/// This attests accounting, not independent worker/physical completion. The
/// actual worker report and its owner channel remain separate evidence.
#[derive(Clone, Debug)]
pub struct SettledCpuSlice {
    ledger: CpuLedgerIdentity,
    grant: SliceGrant,
    actual_work: CpuWork,
    admission_rejected: bool,
    terminal: bool,
}
impl SettledCpuSlice {
    pub fn ledger_identity(&self) -> &CpuLedgerIdentity {
        &self.ledger
    }
    pub fn grant(&self) -> SliceGrant {
        self.grant
    }
    pub fn actual_work(&self) -> CpuWork {
        self.actual_work
    }
    pub fn admission_rejected(&self) -> bool {
        self.admission_rejected
    }
    pub fn terminal(&self) -> bool {
        self.terminal
    }
}

/// Definite pre-enqueue release is not a zero-work worker completion.
#[derive(Clone, Debug)]
pub struct ReleasedCpuReservation {
    ledger: CpuLedgerIdentity,
    grant: SliceGrant,
}
impl ReleasedCpuReservation {
    pub fn ledger_identity(&self) -> &CpuLedgerIdentity {
        &self.ledger
    }
    pub fn grant(&self) -> SliceGrant {
        self.grant
    }
}

/// Read-only dispatch eligibility. A logically waiting parent still owns its
/// private paused stack; it is not an idle physical worker for a child Job.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuOwnerAdmission {
    Unstarted,
    SameOwnerResume,
    BlockedBy(CpuJobId),
}
impl CpuCreditLedger {
    pub fn new(
        authority: SearchAuthority,
        global_limit: u64,
        clock: InstantClock,
    ) -> Result<Self, CpuWorkerError> {
        if global_limit == 0 || clock.domain().0 != authority.epoch {
            return Err(CpuWorkerError::Invalid(
                "CPU ledger root, clock or credit mismatch",
            ));
        }
        Ok(Self {
            identity: CpuLedgerIdentity(Arc::new(())),
            jobs: HashMap::new(),
            authority,
            clock,
            global_limit,
            global_spent: 0,
            highest_job_sequence: 0,
            highest_execution_sequence: 0,
        })
    }
    pub fn identity(&self) -> CpuLedgerIdentity {
        self.identity.clone()
    }
    pub fn register(&mut self, envelope: CpuJobEnvelope) -> Result<(), CpuWorkerError> {
        if envelope.id.authority != self.authority
            || envelope.id.sequence <= self.highest_job_sequence
        {
            return Err(CpuWorkerError::Duplicate);
        }
        envelope
            .validate(
                self.clock.domain(),
                self.clock
                    .now()
                    .map_err(|_| CpuWorkerError::Invalid("CPU ledger clock failed"))?,
            )
            .map_err(|_| CpuWorkerError::Invalid("CPU job deadline or authority invalid"))?;
        if envelope.total_native_node_credit > self.global_limit {
            return Err(CpuWorkerError::Invalid(
                "CPU job exceeds global native credit",
            ));
        }
        self.jobs.retain(|_, account| {
            !account.terminal || account.quarantine || account.reservation.is_some()
        });
        if self.jobs.len() >= CPU_JOB_CAPACITY {
            return Err(CpuWorkerError::Capacity);
        }
        self.highest_job_sequence = envelope.id.sequence;
        self.jobs.insert(
            envelope.id,
            CreditAccount {
                envelope,
                spent: 0,
                reservation: None,
                last_settled_sequence: 0,
                last_started_sequence: 0,
                last_control_sequence: 0,
                terminal: false,
                quarantine: false,
                owner: None,
            },
        );
        Ok(())
    }
    /// Inspect before assigning independent, not-yet-started work. This neither
    /// reserves credit nor moves a token. `reserve` rechecks the same authority
    /// after the manager chooses the owner.
    pub fn owner_admission(
        &self,
        job: CpuJobId,
        owner: CpuWorkerId,
    ) -> Result<CpuOwnerAdmission, CpuWorkerError> {
        let account = self.jobs.get(&job).ok_or(CpuWorkerError::UnknownJob)?;
        if account.quarantine {
            return Err(CpuWorkerError::Quarantined);
        }
        if account.terminal {
            return Err(CpuWorkerError::Invalid("CPU job has terminated"));
        }
        if account.reservation.is_some() {
            return Err(CpuWorkerError::AlreadyReserved);
        }
        if account.owner.is_some_and(|previous| previous != owner) {
            return Err(CpuWorkerError::Invalid(
                "CPU primary job cannot move between owners",
            ));
        }
        if let Some((blocking_job, _)) = self
            .jobs
            .iter()
            .find(|(id, held)| **id != job && !held.terminal && held.owner == Some(owner))
        {
            return Ok(CpuOwnerAdmission::BlockedBy(*blocking_job));
        }
        Ok(if account.owner.is_some() {
            CpuOwnerAdmission::SameOwnerResume
        } else {
            CpuOwnerAdmission::Unstarted
        })
    }

    pub fn reserve(&mut self, grant: SliceGrant) -> Result<(), CpuWorkerError> {
        if grant.job.authority != self.authority
            || grant.execution.sequence <= self.highest_execution_sequence
        {
            return Err(CpuWorkerError::Duplicate);
        }
        if matches!(
            self.owner_admission(grant.job, grant.owner),
            Ok(CpuOwnerAdmission::BlockedBy(_))
        ) {
            return Err(CpuWorkerError::Invalid(
                "CPU owner already holds another primary job",
            ));
        }
        let (_, outstanding) = self.global_usage()?;
        if self
            .global_spent
            .checked_add(outstanding)
            .and_then(|sum| sum.checked_add(grant.node_credit))
            .is_none_or(|sum| sum > self.global_limit)
        {
            return Err(CpuWorkerError::Invalid(
                "CPU grant exceeds global native credit",
            ));
        }
        let account = self
            .jobs
            .get_mut(&grant.job)
            .ok_or(CpuWorkerError::UnknownJob)?;
        if account.quarantine {
            return Err(CpuWorkerError::Quarantined);
        }
        if account.terminal {
            return Err(CpuWorkerError::Invalid("CPU job has terminated"));
        }
        if account.reservation.is_some() {
            return Err(CpuWorkerError::AlreadyReserved);
        }
        if account.owner.is_some_and(|owner| owner != grant.owner) {
            return Err(CpuWorkerError::Invalid(
                "CPU primary job cannot move between owners",
            ));
        }
        grant
            .validate(
                &account.envelope,
                self.clock.domain(),
                self.clock
                    .now()
                    .map_err(|_| CpuWorkerError::Invalid("CPU ledger clock failed"))?,
            )
            .map_err(|_| {
                CpuWorkerError::Invalid("CPU slice lease or immutable deadline invalid")
            })?;
        if grant.control_sequence <= account.last_control_sequence {
            return Err(CpuWorkerError::Invalid("CPU control sequence is stale"));
        }
        if account
            .spent
            .checked_add(grant.node_credit)
            .is_none_or(|sum| sum > account.envelope.total_native_node_credit)
        {
            return Err(CpuWorkerError::Invalid(
                "CPU slice exceeds remaining exact credit",
            ));
        }
        account.owner = Some(grant.owner);
        account.reservation = Some(grant);
        self.highest_execution_sequence = grant.execution.sequence;
        Ok(())
    }
    pub fn settle(&mut self, report: &CpuSliceReport) -> Result<(), CpuWorkerError> {
        let account = self
            .jobs
            .get_mut(&report.grant.job)
            .ok_or(CpuWorkerError::UnknownJob)?;
        if account.quarantine {
            return Err(CpuWorkerError::Quarantined);
        }
        if account.reservation.as_ref() != Some(&report.grant) {
            return Err(CpuWorkerError::Duplicate);
        }
        let Some(work) = report.actual_work else {
            account.quarantine = true;
            return Err(CpuWorkerError::UnknownUsage);
        };
        let invalid_result = match &report.result {
            Ok(value) => {
                let paused = matches!(
                    value.report.completion,
                    CpuCompletion::Yielded | CpuCompletion::AwaitReciprocal
                );
                value.report.nodes != work.nodes
                    || value.report.quiescence_nodes != work.quiescence_nodes
                    || value.report.tt_hits != work.tt_hits
                    || value.report.resume.is_some()
                    || (paused && (report.terminal || !report.continuation_retained))
                    || (!paused && (!report.terminal || report.continuation_retained))
                    || ((value.report.completion == CpuCompletion::AwaitReciprocal)
                        != report.reciprocal_request.is_some())
            }
            Err(_) => {
                (!report.admission_rejected && !report.terminal)
                    || report.reciprocal_request.is_some()
            }
        };
        let spent = account.spent.checked_add(work.nodes);
        let global_spent = self.global_spent.checked_add(work.nodes);
        if work.nodes > report.grant.node_credit
            || spent.is_none_or(|sum| sum > account.envelope.total_native_node_credit)
            || (!report.admission_rejected && spent != Some(report.cumulative_nodes))
            || global_spent.is_none_or(|sum| sum > self.global_limit)
            || invalid_result
            || report.reciprocal_request.as_ref().is_some_and(|request| {
                request.scope().job != account.envelope
                    || request.scope().owner != report.grant.owner
            })
            || (report.terminal && report.continuation_retained)
            || (report.admission_rejected
                && (work != CpuWork::default() || report.result.is_ok() || report.terminal))
        {
            account.quarantine = true;
            return Err(CpuWorkerError::Invalid(
                "CPU actual usage or completion violates reservation",
            ));
        }
        account.spent = spent.expect("checked native credit");
        self.global_spent = global_spent.expect("checked global native credit");
        account.last_settled_sequence = report.grant.execution.sequence;
        if !report.admission_rejected {
            account.last_started_sequence = report.grant.execution.sequence;
        }
        account.last_control_sequence = report.grant.control_sequence;
        account.reservation = None;
        // A definite first-Begin rejection never acquired a private stack.
        // Rejected Resume keeps the existing same-owner continuation authority.
        if report.admission_rejected && account.last_started_sequence == 0 {
            account.owner = None;
        }
        account.terminal |= report.terminal;
        Ok(())
    }
    /// A consumer can bind this successful accounting transition to its own
    /// outstanding token. Cloned receipts still describe one settlement and
    /// must be consumed once by the downstream owner.
    pub fn settle_attested(
        &mut self,
        report: &CpuSliceReport,
    ) -> Result<SettledCpuSlice, CpuWorkerError> {
        self.settle(report)?;
        Ok(SettledCpuSlice {
            ledger: self.identity(),
            grant: report.grant,
            actual_work: report
                .actual_work
                .expect("successful settlement has known work"),
            admission_rejected: report.admission_rejected,
            terminal: report.terminal,
        })
    }
    /// Roll back only a command proved not to have entered its worker queue.
    /// Execution high water remains advanced, so the rejected ID cannot replay.
    pub fn release_unadmitted(
        &mut self,
        rejection: &CpuAdmissionRejection,
    ) -> Result<(), CpuWorkerError> {
        if !rejection.is_verified_unqueued() {
            return Err(CpuWorkerError::UnknownUsage);
        }
        let account = self
            .jobs
            .get_mut(&rejection.grant.job)
            .ok_or(CpuWorkerError::UnknownJob)?;
        if account.quarantine {
            return Err(CpuWorkerError::Quarantined);
        }
        if account.reservation.as_ref() != Some(&rejection.grant) {
            return Err(CpuWorkerError::Duplicate);
        }
        account.last_settled_sequence = rejection.grant.execution.sequence;
        account.last_control_sequence = rejection.grant.control_sequence;
        account.reservation = None;
        if account.last_started_sequence == 0 {
            account.owner = None;
        }
        Ok(())
    }
    pub fn release_unadmitted_attested(
        &mut self,
        rejection: &CpuAdmissionRejection,
    ) -> Result<ReleasedCpuReservation, CpuWorkerError> {
        self.release_unadmitted(rejection)?;
        Ok(ReleasedCpuReservation {
            ledger: self.identity(),
            grant: rejection.grant,
        })
    }
    /// Abandon a queued-but-never-started question. An admitted reservation or
    /// an owned continuation requires its worker's actual terminal receipt.
    pub fn retire_unstarted(&mut self, job: CpuJobId) -> Result<(), CpuWorkerError> {
        let account = self.jobs.get_mut(&job).ok_or(CpuWorkerError::UnknownJob)?;
        if account.quarantine {
            return Err(CpuWorkerError::Quarantined);
        }
        if account.terminal {
            return Err(CpuWorkerError::Duplicate);
        }
        if account.reservation.is_some() {
            return Err(CpuWorkerError::AlreadyReserved);
        }
        if account.last_started_sequence != 0 || account.owner.is_some() || account.spent != 0 {
            return Err(CpuWorkerError::Invalid(
                "CPU started Job needs owner termination",
            ));
        }
        account.terminal = true;
        Ok(())
    }
    pub fn accept_job_termination(
        &mut self,
        termination: &CpuJobTermination,
    ) -> Result<(), CpuWorkerError> {
        let account = self
            .jobs
            .get_mut(&termination.job)
            .ok_or(CpuWorkerError::UnknownJob)?;
        if account.quarantine {
            return Err(CpuWorkerError::Quarantined);
        }
        if account.terminal {
            return Err(CpuWorkerError::Duplicate);
        }
        if account.owner != Some(termination.owner)
            || account.last_started_sequence != termination.last_execution.sequence
            || termination.last_execution.epoch != self.authority.epoch
            || termination.retained_bytes != 0
            || !termination.controls.terminal
            || !matches!(
                termination.completion,
                CpuCompletion::Canceled | CpuCompletion::Deadline
            )
            || (termination.completion == CpuCompletion::Canceled
                && termination.controls.cancel_ack == 0
                && !termination.shutdown)
        {
            return Err(CpuWorkerError::Invalid(
                "CPU paused termination is not attested",
            ));
        }
        account.terminal = true;
        Ok(())
    }
    pub fn global_usage(&self) -> Result<(u64, u64), CpuWorkerError> {
        let outstanding = self
            .jobs
            .values()
            .try_fold(0_u64, |sum, account| {
                sum.checked_add(
                    account
                        .reservation
                        .as_ref()
                        .map_or(0, |grant| grant.node_credit),
                )
            })
            .ok_or(CpuWorkerError::Invalid(
                "CPU outstanding reservations overflow",
            ))?;
        Ok((self.global_spent, outstanding))
    }
    pub fn quarantine(&mut self, job: CpuJobId) -> Result<(), CpuWorkerError> {
        self.jobs
            .get_mut(&job)
            .ok_or(CpuWorkerError::UnknownJob)?
            .quarantine = true;
        Ok(())
    }
    pub fn usage(&self, job: CpuJobId) -> Result<(u64, u64, bool), CpuWorkerError> {
        let account = self.jobs.get(&job).ok_or(CpuWorkerError::UnknownJob)?;
        Ok((
            account.spent,
            account
                .reservation
                .as_ref()
                .map_or(0, |grant| grant.node_credit),
            account.quarantine,
        ))
    }
}

enum CpuCommand {
    Begin(Box<CpuJob>, SliceGrant),
    Resume(SliceGrant),
    ResumeWithReply(SliceGrant, Box<CpuReciprocalReply>),
}
struct TimedCpuCommand {
    original: CpuCommand,
    admitted_at: Instant,
}
impl CpuCommand {
    fn grant(&self) -> SliceGrant {
        match self {
            Self::Begin(_, grant) | Self::Resume(grant) | Self::ResumeWithReply(grant, _) => *grant,
        }
    }
}

/// One worker lifetime belongs to one root authority. Both admitted and
/// definitely declined IDs advance this bounded fence, so a second call cannot
/// mint a refund receipt for an execution that has already entered the queue.
#[derive(Default)]
struct CpuSubmissionFence {
    authority: Option<SearchAuthority>,
    high_water: u64,
}
impl CpuSubmissionFence {
    fn claim(&mut self, owner: CpuWorkerId, grant: SliceGrant) -> Result<(), CpuWorkerError> {
        if grant.owner != owner
            || grant.execution.epoch != grant.job.authority.epoch
            || grant.execution.sequence == 0
            || grant.execution.sequence == u64::MAX
            || grant.node_credit == 0
            || grant.soft_lease.is_zero()
            || grant.control_sequence == 0
        {
            return Err(CpuWorkerError::Invalid(
                "CPU submission owner or grant mismatch",
            ));
        }
        if self
            .authority
            .is_some_and(|authority| authority != grant.job.authority)
        {
            return Err(CpuWorkerError::Invalid("CPU worker root authority changed"));
        }
        if grant.execution.sequence <= self.high_water {
            return Err(CpuWorkerError::Duplicate);
        }
        self.authority = Some(grant.job.authority);
        self.high_water = grant.execution.sequence;
        Ok(())
    }
}

/// Engine owner IDs remain opaque here. In particular, this receipt does
/// not convert the existing 1-based bootstrap owner into runtime's 0-based
/// actor slot or attest manager/native/ORT placement.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CpuWorkerThreadAffinityShutdown {
    NotRequested,
    Requested {
        ready: Option<Result<CpuThreadAffinityApplied, CpuThreadAffinityStartError>>,
        /// None after a started owner means explicit restoration was not
        /// observed (for example, an uncaught unwind), never restored=true.
        finished: Option<Result<CpuThreadAffinityObservation, CpuThreadAffinityFinishError>>,
    },
}

#[cfg(test)]
#[derive(Default)]
struct CpuWorkerTestHooks {
    before_paused_command_poll: Option<Box<dyn FnOnce() + Send + 'static>>,
    affinity_start_failure: Option<CpuThreadAffinityStartError>,
    affinity_finish_failure: Option<CpuThreadAffinityFinishError>,
    panic_after_affinity_ready: bool,
}

struct CpuWorkerExit {
    engine: Option<CpuEngine>,
    affinity_finished: Option<Result<CpuThreadAffinityObservation, CpuThreadAffinityFinishError>>,
}
struct ActiveJob {
    job: CpuJob,
    token: Option<CpuResumeToken>,
    retained_completed_depth: u16,
    spent: u64,
    last_sequence: u64,
    last_control_sequence: u64,
    terminal_permit: Option<
        ReliablePermit<TimedCpuCommand, CpuWorkerEventObservation, CpuWorkerProgressObservation>,
    >,
    nnuec_capture: Option<NnuecActiveCapture>,
}
#[derive(Clone, Debug)]
pub struct CpuWorkerProgress {
    pub job: CpuJobId,
    pub execution: ExecutionId,
    pub owner: CpuWorkerId,
    pub sequence: u64,
    /// Coherent iteration result; nodes count only this invocation, not the Job.
    pub iteration: CpuIterationProgress,
}

/// Original owner clocks travel with their publication, never in a separate
/// mutable "latest timing" slot. This is timing evidence, not a new witness.
#[derive(Clone, Debug)]
pub struct CpuWorkerProgressObservation {
    original: CpuWorkerProgress,
    run_started_at: Instant,
    published_at: Instant,
    interval_started_at: Instant,
    advanced_depth: bool,
    resume_admission_and_restore: Option<Duration>,
}
impl CpuWorkerProgressObservation {
    pub fn progress(&self) -> &CpuWorkerProgress {
        &self.original
    }
    pub fn run_started_at(&self) -> Instant {
        self.run_started_at
    }
    pub fn published_at(&self) -> Instant {
        self.published_at
    }
    pub fn completed_progress_interval(&self) -> Option<Duration> {
        self.advanced_depth
            .then(|| self.published_at.duration_since(self.interval_started_at))
    }
    pub fn resume_admission_and_restore(&self) -> Option<Duration> {
        self.resume_admission_and_restore
    }
}
#[derive(Clone, Copy, Debug)]
pub struct CpuWorkerSliceTiming {
    run_started_at: Instant,
    analyze_returned_at: Instant,
    report_ready_at: Instant,
    resume_admission_and_restore: Option<Duration>,
    #[cfg(feature = "nnuec-scalar-diagnostics")]
    nnuec_scalar: Option<crate::cpu_value::nnuec::v2::scalar_diagnostic::SliceObservation>,
}
impl CpuWorkerSliceTiming {
    #[cfg(feature = "nnuec-scalar-diagnostics")]
    pub fn nnuec_scalar(
        &self,
    ) -> Option<crate::cpu_value::nnuec::v2::scalar_diagnostic::SliceObservation> {
        self.nnuec_scalar
    }
    pub fn run_started_at(&self) -> Instant {
        self.run_started_at
    }
    pub fn analyze_returned_at(&self) -> Instant {
        self.analyze_returned_at
    }
    pub fn report_ready_at(&self) -> Instant {
        self.report_ready_at
    }
    pub fn return_processing(&self) -> Duration {
        self.report_ready_at
            .duration_since(self.analyze_returned_at)
    }
    pub fn resume_admission_and_restore(&self) -> Option<Duration> {
        self.resume_admission_and_restore
    }
}
pub struct CpuWorkerEventObservation {
    original: CpuWorkerEvent,
    timing: Option<CpuWorkerSliceTiming>,
}
impl CpuWorkerEventObservation {
    pub fn into_parts(self) -> (CpuWorkerEvent, Option<CpuWorkerSliceTiming>) {
        (self.original, self.timing)
    }
}

/// Commands and terminal slots share an admission count of eight. Each command
/// reserves its report seat before enqueueing, so terminal send never depends
/// on an unavailable general queue slot. Progress alone is coalesced.
type CpuWorkerMailbox =
    WorkerMailbox<TimedCpuCommand, CpuWorkerEventObservation, CpuWorkerProgressObservation>;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuWorkerProfile {
    pub owner: CpuWorkerId,
    pub max_depth: u16,
    pub evaluation_namespace: Digest,
    pub search_conditions: Digest,
}
pub struct CpuWorker {
    owner: CpuWorkerId,
    max_depth: u16,
    evaluation_namespace: Digest,
    search_conditions: Digest,
    command_capacity: usize,
    paused_bytes_limit: usize,
    mailbox: CpuWorkerMailbox,
    shutdown: Arc<AtomicBool>,
    ready_readback: Arc<OnceLock<Result<CpuThreadReadback, CpuResourceError>>>,
    affinity_ready:
        Option<Arc<OnceLock<Result<CpuThreadAffinityApplied, CpuThreadAffinityStartError>>>>,
    affinity_finished: Option<Result<CpuThreadAffinityObservation, CpuThreadAffinityFinishError>>,
    pub signal: CompletionSignal,
    submission: Mutex<CpuSubmissionFence>,
    /// One independently bounded install seat. A pending owner readback blocks
    /// new job/profile admission; it never transfers an active or paused job.
    installation: Mutex<prepared_value::Endpoint>,
    join: Option<JoinHandle<CpuWorkerExit>>,
    recovered_engine: Option<CpuEngine>,
    joined_known: bool,
    deferred_nnuec_source: Option<DeferredNnuecRSource>,
}

impl CpuWorker {
    pub fn spawn(owner: CpuWorkerId, engine: CpuEngine) -> Result<Self, CpuWorkerError> {
        Self::spawn_notified(owner, engine, CompletionSignal::default())
    }
    /// Pool/native services can share one manager wake without creating a
    /// relay thread. Each Job still has its own immutable control mailbox.
    pub fn spawn_notified(
        owner: CpuWorkerId,
        engine: CpuEngine,
        signal: CompletionSignal,
    ) -> Result<Self, CpuWorkerError> {
        Self::spawn_notified_retaining(owner, engine, signal).map_err(|(_, reason)| reason)
    }
    /// Return the same engine and TT if validation or OS thread creation fails
    /// before this worker exists. Once a thread starts it owns the engine; only
    /// the existing known-exit `take_engine` path can return it afterward.
    // This startup-only error transfers a unique CpuEngine by value. Boxing it
    // would change the ownership-preserving constructor contract merely to
    // reduce an exceptional stack return, rather than the execution hot path.
    #[allow(clippy::result_large_err)]
    pub fn spawn_notified_retaining(
        owner: CpuWorkerId,
        engine: CpuEngine,
        signal: CompletionSignal,
    ) -> Result<Self, (CpuEngine, CpuWorkerError)> {
        Self::spawn_notified_retaining_with_capacity(
            owner,
            engine,
            signal,
            CPU_WORKER_COMMAND_CAPACITY,
        )
    }
    /// Apply the declaration to this actual command mailbox. The immutable
    /// engine already owns its separately configured paused-stack ceiling;
    /// no replacement engine or TT is constructed by this admission path.
    #[allow(clippy::result_large_err)]
    pub fn spawn_notified_retaining_with_capacity(
        owner: CpuWorkerId,
        engine: CpuEngine,
        signal: CompletionSignal,
        command_capacity: usize,
    ) -> Result<Self, (CpuEngine, CpuWorkerError)> {
        Self::spawn_notified_with_capacity_and_launcher(
            owner,
            engine,
            signal,
            command_capacity,
            |builder, task| builder.spawn(task),
            #[cfg(test)]
            None,
        )
    }
    /// Explicit, opt-in ownership for this actual own-CPU worker only. The
    /// default constructors remain observe-only. The guard is constructed
    /// inside the new thread and remains there through token/engine cleanup;
    /// callers receive cloned receipts, never the non-Send owner itself.
    /// Admission remains closed until a successful apply+readback is published.
    /// This is not connected to the production bootstrap and cannot enable
    /// DedicatedControl or change the spawning thread/process affinity.
    #[allow(clippy::result_large_err)]
    pub fn spawn_notified_retaining_with_thread_affinity(
        owner: CpuWorkerId,
        engine: CpuEngine,
        signal: CompletionSignal,
        command_capacity: usize,
        cpus: Vec<LogicalCpu>,
    ) -> Result<Self, (CpuEngine, CpuWorkerError)> {
        Self::spawn_notified_with_capacity_launcher_and_affinity(
            owner,
            engine,
            signal,
            command_capacity,
            Some(cpus),
            |builder, task| builder.spawn(task),
            #[cfg(test)]
            CpuWorkerTestHooks::default(),
        )
    }

    // The private launcher lets a source test exercise a real pre-start
    // ownership failure without exhausting process/OS resources. An Err must
    // drop, rather than invoke, the submitted task, as Thread::Builder does.
    #[cfg(test)]
    #[allow(clippy::result_large_err)]
    fn spawn_notified_with_launcher(
        owner: CpuWorkerId,
        engine: CpuEngine,
        signal: CompletionSignal,
        launcher: impl FnOnce(
            thread::Builder,
            Box<dyn FnOnce() -> CpuWorkerExit + Send + 'static>,
        ) -> std::io::Result<JoinHandle<CpuWorkerExit>>,
    ) -> Result<Self, (CpuEngine, CpuWorkerError)> {
        Self::spawn_notified_with_capacity_and_launcher(
            owner,
            engine,
            signal,
            CPU_WORKER_COMMAND_CAPACITY,
            launcher,
            None,
        )
    }
    #[allow(clippy::result_large_err)]
    fn spawn_notified_with_capacity_and_launcher(
        owner: CpuWorkerId,
        engine: CpuEngine,
        signal: CompletionSignal,
        command_capacity: usize,
        launcher: impl FnOnce(
            thread::Builder,
            Box<dyn FnOnce() -> CpuWorkerExit + Send + 'static>,
        ) -> std::io::Result<JoinHandle<CpuWorkerExit>>,
        #[cfg(test)] before_paused_command_poll: Option<Box<dyn FnOnce() + Send + 'static>>,
    ) -> Result<Self, (CpuEngine, CpuWorkerError)> {
        Self::spawn_notified_with_capacity_launcher_and_affinity(
            owner,
            engine,
            signal,
            command_capacity,
            None,
            launcher,
            #[cfg(test)]
            CpuWorkerTestHooks {
                before_paused_command_poll,
                ..Default::default()
            },
        )
    }

    #[allow(clippy::result_large_err)]
    fn spawn_notified_with_capacity_launcher_and_affinity(
        owner: CpuWorkerId,
        engine: CpuEngine,
        signal: CompletionSignal,
        command_capacity: usize,
        cpus: Option<Vec<LogicalCpu>>,
        launcher: impl FnOnce(
            thread::Builder,
            Box<dyn FnOnce() -> CpuWorkerExit + Send + 'static>,
        ) -> std::io::Result<JoinHandle<CpuWorkerExit>>,
        #[cfg(test)] mut hooks: CpuWorkerTestHooks,
    ) -> Result<Self, (CpuEngine, CpuWorkerError)> {
        if !(1..=CPU_WORKER_COMMAND_CAPACITY).contains(&command_capacity) {
            return Err((engine, CpuWorkerError::Invalid("CPU command capacity")));
        }
        if engine.resume_policy() != CpuResumePolicy::PausedStack {
            return Err((
                engine,
                CpuWorkerError::Invalid("CPU worker requires PausedStack"),
            ));
        }
        let max_depth = engine.config().max_depth;
        let paused_bytes_limit = engine.paused_stack_max_bytes();
        let initial_value_binding = match engine.job_binding_digest() {
            Ok(binding) => binding,
            Err(error) => return Err((engine, CpuWorkerError::Cpu(error))),
        };
        let evaluation_namespace =
            match prepared_value::registered_namespace(&engine, initial_value_binding) {
                Ok(namespace) => namespace,
                Err(reason) => return Err((engine, reason)),
            };
        let search_conditions = cpu_search_conditions_digest(&engine);
        let shutdown = Arc::new(AtomicBool::new(false));
        // One terminal seat belongs to the paused Job. A one-command profile
        // still needs two more seats for its next slice/error receipt, so its
        // owner can Resume after the earlier report is drained.
        let reliable_capacity = (command_capacity * 2).max(3);
        let mailbox =
            match CpuWorkerMailbox::new(command_capacity, reliable_capacity, signal.clone()) {
                Ok(mailbox) => mailbox,
                Err(_) => return Err((engine, CpuWorkerError::Invalid("CPU mailbox capacity"))),
            };
        let worker_shutdown = shutdown.clone();
        let worker_mailbox = mailbox.clone();
        let (installation, mut worker_installation) = prepared_value::channel();
        let ready_readback = Arc::new(OnceLock::new());
        let worker_readback = ready_readback.clone();
        let affinity_ready = cpus.as_ref().map(|_| Arc::new(OnceLock::new()));
        let worker_affinity_ready = affinity_ready.clone();
        // Builder::spawn drops its FnOnce argument when OS creation fails. A
        // shared one-use gate retains the engine on that branch; no engine or
        // TT is cloned. Successful threads take ownership before any work.
        let staged_engine = Arc::new(Mutex::new(Some(engine)));
        let worker_engine = staged_engine.clone();
        let task = Box::new(move || {
            let mut engine = worker_engine
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
                .expect("CPU startup gate contains its unique engine");
            drop(worker_engine);
            // Construct the non-Send/non-Sync guard on this exact worker.
            // No closure capture or JoinHandle result contains that guard.
            let mut affinity_owner = match cpus {
                Some(cpus) => {
                    #[cfg(test)]
                    let started = hooks
                        .affinity_start_failure
                        .take()
                        .map_or_else(|| CurrentCpuThreadAffinityOwner::begin(cpus), Err);
                    #[cfg(not(test))]
                    let started = CurrentCpuThreadAffinityOwner::begin(cpus);
                    match started {
                        Ok(guard) => {
                            let _ = worker_affinity_ready
                                .as_ref()
                                .expect("opt-in affinity owns its ready slot")
                                .set(Ok(guard.applied().clone()));
                            Some(guard)
                        }
                        Err(error) => {
                            let recoverable = error.permits_unstarted_engine_recovery();
                            let _ = worker_affinity_ready
                                .as_ref()
                                .expect("opt-in affinity owns its ready slot")
                                .set(Err(error));
                            let _ = worker_readback.set(Err(CpuResourceError::InvalidReadback));
                            worker_mailbox.close();
                            worker_mailbox.signal().notify();
                            // A rejected search did not consume CPU work, but
                            // failed/unknown mutation rollback cannot authorize
                            // recovery of this worker owner. Preserve its original
                            // typed error; only pre-mutation/verified rollback can
                            // return the same engine after a known join.
                            return CpuWorkerExit {
                                engine: recoverable.then_some(engine),
                                affinity_finished: None,
                            };
                        }
                    }
                }
                None => None,
            };
            // Observe this actual OS thread once, before accepting search
            // work. A failed probe remains explicit and does not install
            // affinity or promote SharedControl to DedicatedControl.
            let observed = u8::try_from(owner.0)
                .map_err(|_| CpuResourceError::InvalidDeclaration)
                .and_then(|id| observe_current_cpu_thread(CpuActor::OwnWorker(id)));
            let _ = worker_readback.set(observed);
            worker_mailbox.signal().notify();
            #[cfg(test)]
            if affinity_owner.is_some() && hooks.panic_after_affinity_ready {
                panic!("injected uncaught worker unwind after affinity readiness");
            }
            let mut active: Option<ActiveJob> = None;
            let mut progress_sequence = 0u64;
            let mut engine_reusable = true;
            let mut value_binding = initial_value_binding;
            loop {
                let observed = worker_mailbox.signal().version();
                // A paused owner keeps a reliable terminal seat. Atomic
                // cancellation is serviced even when normal queues are full.
                if let Some(task) = active.as_mut() {
                    let snapshot = task.job.mailbox.snapshot();
                    let reason = if snapshot.is_cancelled() {
                        Some(CpuCompletion::Canceled)
                    } else if Instant::now() >= task.job.hard_deadline {
                        Some(CpuCompletion::Deadline)
                    } else if worker_shutdown.load(Ordering::Acquire) {
                        Some(CpuCompletion::Canceled)
                    } else {
                        None
                    };
                    if let Some(completion) = reason {
                        engine.discard_paused_stack();
                        if snapshot.cancel_sequence > snapshot.cancel_ack {
                            let _ = task
                                .job
                                .mailbox
                                .acknowledge_cancel(snapshot.cancel_sequence);
                        }
                        let controls = task.job.mailbox.seal_terminal();
                        if task
                            .job
                            .retire_nnuec_capture(
                                owner,
                                NnuecCaptureEnding::CpuTerminal(completion),
                            )
                            .is_err()
                        {
                            engine_reusable = false;
                        }
                        if let Some(mut permit) = task.terminal_permit.take() {
                            let termination = CpuJobTermination {
                                job: task.job.envelope.id,
                                owner,
                                last_execution: ExecutionId::new(
                                    task.job.envelope.id.authority.epoch,
                                    task.last_sequence,
                                ),
                                completion,
                                controls,
                                retained_bytes: engine.paused_stack_bytes().unwrap_or(0),
                                shutdown: worker_shutdown.load(Ordering::Acquire),
                            };
                            let _ = permit.publish(CpuWorkerEventObservation {
                                original: CpuWorkerEvent::JobTerminated(termination),
                                timing: None,
                            });
                        }
                        active = None;
                    }
                }
                // This actual owner decides whether its engine, token and
                // reliable seats are idle. A producer's claim of idleness
                // cannot authorize replacement of a private paused traversal.
                if worker_installation
                    .service(
                        &mut engine,
                        &mut value_binding,
                        active.is_none(),
                        engine_reusable,
                        worker_mailbox.occupancy(),
                        worker_shutdown.load(Ordering::Acquire),
                        &worker_mailbox.signal(),
                    )
                    .is_err()
                {
                    engine_reusable = false;
                    worker_shutdown.store(true, Ordering::Release);
                    worker_mailbox.close();
                }
                // A source fixture can hold this exact boundary after paused
                // termination was checked and before the next shutdown read.
                // The hook does not exist in non-test worker builds.
                #[cfg(test)]
                if active.is_some()
                    && let Some(observe) = hooks.before_paused_command_poll.take()
                {
                    observe();
                }
                let mut command = match worker_mailbox.take_command() {
                    Some(command) => command,
                    None => {
                        if worker_shutdown.load(Ordering::Acquire) {
                            // Shutdown may arrive after the paused-owner
                            // check above. Publish its reserved terminal
                            // receipt before joining; only an idle owner can
                            // exit. Use this same shutdown observation for
                            // both decisions, rather than two racing reads.
                            if active.is_some() {
                                continue;
                            }
                            break;
                        }
                        worker_mailbox.wait_changed(observed, Duration::from_millis(10));
                        continue;
                    }
                };
                let admitted_at = command.command.admitted_at;
                let mut rejected_controls = None;
                #[cfg(feature = "nnuec-scalar-diagnostics")]
                let diagnostic_resume_command = matches!(
                    &command.command.original,
                    CpuCommand::Resume(_) | CpuCommand::ResumeWithReply(_, _)
                );
                let (grant, accepted_command, reciprocal_reply) = match command.command.original {
                    CpuCommand::Begin(job, grant) => {
                        let fresh_eligible = !worker_shutdown.load(Ordering::Acquire)
                            && grant.job == job.envelope.id
                            && grant.owner == owner
                            && job.mailbox.scope()
                                == &CpuMailboxScope {
                                    job: grant.job,
                                    owner,
                                }
                            && prepared_value::registered_namespace(&engine, value_binding)
                                .is_ok_and(|namespace| {
                                    job.envelope.evaluation_namespace == namespace
                                })
                            && job.envelope.search_conditions
                                == cpu_search_conditions_digest(&engine)
                            && job.target_depth <= engine.config().max_depth
                            && job.clock.now().ok().is_some_and(|now| {
                                grant
                                    .validate(&job.envelope, job.clock.domain(), now)
                                    .is_ok()
                            });
                        let capture = if active.is_none() && fresh_eligible {
                            match &job.nnuec_capture {
                                Some(capture) => engine
                                    .job_binding_digest()
                                    .map_err(|_| CaptureError::Stale)
                                    .and_then(|binding| {
                                        binding.ok_or(CaptureError::Invalid(
                                            "capture requires contextual value binding",
                                        ))
                                    })
                                    .and_then(|binding| {
                                        capture.bind_job(
                                            job.envelope,
                                            owner,
                                            &job.position,
                                            job.clock,
                                            binding,
                                        )
                                    })
                                    .map(Some),
                                None => Ok(None),
                            }
                        } else {
                            Ok(None)
                        };
                        let capture_ok = capture.is_ok();
                        if let Err(error) = &capture {
                            if let Some(scope) = &job.nnuec_capture {
                                scope.record_error(error.clone());
                            }
                        }
                        if active.is_none() && fresh_eligible && capture_ok {
                            active = Some(ActiveJob {
                                job: *job,
                                token: None,
                                retained_completed_depth: 0,
                                spent: 0,
                                last_sequence: 0,
                                last_control_sequence: 0,
                                terminal_permit: None,
                                nnuec_capture: capture.expect("checked capture admission"),
                            });
                            (grant, true, None)
                        } else {
                            // This owned Begin never ran. Closing its
                            // question admission is distinct from an
                            // execution terminal or a retained stack ack.
                            // A malformed duplicate must not seal the live
                            // primary when it aliases that exact mailbox.
                            let aliases_primary = active
                                .as_ref()
                                .is_some_and(|task| Arc::ptr_eq(&task.job.mailbox, &job.mailbox));
                            if !aliases_primary {
                                rejected_controls = Some(job.mailbox.seal_terminal());
                                let _ =
                                    job.retire_nnuec_capture(owner, NnuecCaptureEnding::NotStarted);
                            }
                            (grant, false, None)
                        }
                    }
                    CpuCommand::Resume(grant) => (grant, true, None),
                    CpuCommand::ResumeWithReply(grant, reply) => (grant, true, Some(reply)),
                };
                let mut report = CpuSliceReport {
                    grant,
                    reciprocal_request: None,
                    actual_work: Some(CpuWork::default()),
                    result: Err(CpuError::InvalidLimits(
                        "slice does not name the live private job",
                    )),
                    cumulative_nodes: 0,
                    progress_sequence,
                    continuation_retained: false,
                    terminal: false,
                    admission_rejected: true,
                    controls: rejected_controls,
                };
                // Reject malformed/late replies before taking the old terminal
                // seat or invoking the engine. A rejected Resume must preserve
                // the original paused stack and charge no new native work.
                let reciprocal_admission_error =
                    active
                        .as_ref()
                        .and_then(|task| match &task.job.reciprocal_scope {
                            Some(scope) => engine
                                .validate_reciprocal_resume(
                                    task.token.as_ref(),
                                    scope,
                                    reciprocal_reply.as_deref(),
                                )
                                .err(),
                            None if reciprocal_reply.is_some()
                                || engine.pending_reciprocal_request().is_some() =>
                            {
                                Some(CpuError::ResumeMismatch(
                                    "plain CPU Resume cannot bypass a reciprocal question",
                                ))
                            }
                            None => None,
                        });
                let eligible = !worker_shutdown.load(Ordering::Acquire)
                    && accepted_command
                    && reciprocal_admission_error.is_none()
                    && active.as_ref().is_some_and(|task| {
                        grant.job == task.job.envelope.id
                            && grant.owner == owner
                            && grant.execution.epoch == grant.job.authority.epoch
                            && grant.execution.sequence > task.last_sequence
                            && grant.node_credit > 0
                            && !grant.soft_lease.is_zero()
                            && grant.control_sequence > task.last_control_sequence
                            && task.job.clock.now().ok().is_some_and(|now| {
                                grant
                                    .validate(&task.job.envelope, task.job.clock.domain(), now)
                                    .is_ok()
                            })
                            && task
                                .spent
                                .checked_add(grant.node_credit)
                                .is_some_and(|sum| {
                                    sum <= task.job.envelope.total_native_node_credit
                                })
                    });
                if let Some(error) = reciprocal_admission_error {
                    report.result = Err(error);
                }
                let mut owner_timing = None;
                if eligible {
                    let task = active.as_mut().expect("checked private job");
                    // Starting the next slice consumes the previous pause
                    // authority; its unused terminal storage can be released.
                    task.terminal_permit.take();
                    let run_started_at = Instant::now();
                    let lease = run_started_at.checked_add(grant.soft_lease);
                    let resumed_depth = task.token.as_ref().map(|_| task.retained_completed_depth);
                    let mut last_depth = resumed_depth.unwrap_or(0);
                    let mut interval_started_at = run_started_at;
                    let mut first_publication = true;
                    let mut resume_admission_and_restore = None;
                    let mut invoked = false;
                    #[cfg(feature = "nnuec-scalar-diagnostics")]
                    let diagnostic_before = (task.token.is_some(), engine.paused_stack_snapshot());
                    #[cfg(feature = "nnuec-scalar-diagnostics")]
                    let diagnostic =
                        crate::cpu_value::nnuec::v2::scalar_diagnostic::Invocation::begin();
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        CpuSliceControl::new(grant.node_credit, lease, task.job.mailbox.clone())
                            .and_then(|control| match &task.nnuec_capture {
                                Some(capture) => control.with_nnuec_capture(capture.clone()),
                                None => Ok(control),
                            })
                            .and_then(|control| {
                                invoked = true;
                                let limits = CpuLimits {
                                    max_depth: task.job.target_depth,
                                    max_nodes: task.job.envelope.total_native_node_credit
                                        - task.spent,
                                    deadline: Some(task.job.hard_deadline),
                                };
                                let mut observer = |iteration: CpuIterationProgress| {
                                    if iteration.completed_depth == 0 {
                                        return;
                                    }
                                    let published_at = Instant::now();
                                    if first_publication
                                        && resumed_depth == Some(iteration.completed_depth)
                                        && iteration.nodes == 0
                                    {
                                        // The existing stack owner publishes its
                                        // reused iteration after exact restoration.
                                        // This is not a newly completed depth.
                                        resume_admission_and_restore =
                                            published_at.checked_duration_since(admitted_at);
                                    }
                                    first_publication = false;
                                    let advanced_depth = iteration.completed_depth > last_depth;
                                    last_depth = last_depth.max(iteration.completed_depth);
                                    progress_sequence = progress_sequence.saturating_add(1);
                                    let _ = worker_mailbox.publish_progress(
                                        progress_sequence,
                                        CpuWorkerProgressObservation {
                                            original: CpuWorkerProgress {
                                                job: grant.job,
                                                execution: grant.execution,
                                                owner,
                                                sequence: progress_sequence,
                                                iteration,
                                            },
                                            run_started_at,
                                            published_at,
                                            interval_started_at,
                                            advanced_depth,
                                            resume_admission_and_restore,
                                        },
                                    );
                                    interval_started_at = published_at;
                                };
                                match &task.job.reciprocal_scope {
                                    Some(scope) => engine
                                        .analyze_controlled_reciprocal(
                                            &task.job.position,
                                            task.job.root_moves.as_deref(),
                                            limits,
                                            &worker_shutdown,
                                            &control,
                                            task.token.as_ref(),
                                            scope,
                                            reciprocal_reply.as_deref(),
                                            &mut observer,
                                        )
                                        .map(|value| {
                                            report.reciprocal_request = value.request;
                                            value.controlled
                                        }),
                                    None => engine.analyze_controlled(
                                        &task.job.position,
                                        task.job.root_moves.as_deref(),
                                        limits,
                                        &worker_shutdown,
                                        &control,
                                        task.token.as_ref(),
                                        &mut observer,
                                    ),
                                }
                            })
                    }));
                    let analyze_returned_at = Instant::now();
                    owner_timing = Some(CpuWorkerSliceTiming {
                        run_started_at,
                        analyze_returned_at,
                        report_ready_at: analyze_returned_at,
                        resume_admission_and_restore,
                        #[cfg(feature = "nnuec-scalar-diagnostics")]
                        nnuec_scalar: Some(
                            crate::cpu_value::nnuec::v2::scalar_diagnostic::SliceObservation {
                                scalar: diagnostic.finish(),
                                resume_command: diagnostic_resume_command,
                                prior_token_present: diagnostic_before.0,
                                before: diagnostic_before.1,
                                after: engine.paused_stack_snapshot(),
                            },
                        ),
                    });
                    let panicked = result.is_err();
                    let mut result = result.unwrap_or_else(|_| {
                        Err(CpuError::Unsupported("CPU worker panicked; usage unknown"))
                    });
                    // Preserve the existing OwnedCpuChecker post-call
                    // frozen namespace guard in this private execution path.
                    if result.is_ok() {
                        if let Err(error) = engine.validate_frozen_value_namespace() {
                            result = Err(error);
                        }
                    }
                    if result.is_err() {
                        report.reciprocal_request = None;
                    }
                    report.actual_work = if panicked {
                        None
                    } else if invoked {
                        engine.last_attempt_work()
                    } else {
                        Some(CpuWork::default())
                    };
                    if let Some(work) = report.actual_work {
                        if let Some(spent) = task.spent.checked_add(work.nodes) {
                            task.spent = spent;
                        } else {
                            report.actual_work = None;
                            result = Err(CpuError::Unsupported(
                                "CPU cumulative native work overflow; usage unknown",
                            ));
                        }
                    }
                    report.cumulative_nodes = task.spent;
                    task.last_sequence = grant.execution.sequence;
                    task.last_control_sequence = grant.control_sequence;
                    task.token = result
                        .as_ref()
                        .ok()
                        .filter(|value| {
                            matches!(
                                value.report.completion,
                                CpuCompletion::Yielded | CpuCompletion::AwaitReciprocal
                            )
                        })
                        .and_then(|value| value.report.resume.clone());
                    task.retained_completed_depth = result
                        .as_ref()
                        .map_or(0, |value| value.report.completed_depth);
                    // Only this private worker retains the opaque engine
                    // owner. The manager receives liveness, not the token.
                    if let Ok(value) = &mut result {
                        value.report.resume = None;
                    }
                    report.continuation_retained = task
                        .token
                        .as_ref()
                        .is_some_and(|token| engine.token_is_current(token));
                    report.terminal = match &result {
                        Ok(value) => !matches!(
                            value.report.completion,
                            CpuCompletion::Yielded | CpuCompletion::AwaitReciprocal
                        ),
                        Err(_) => true,
                    };
                    report.admission_rejected = false;
                    report.result = result;
                    report.progress_sequence = progress_sequence;
                    let mut nnuec_soft_terminal = report.result.as_ref().err().and_then(|error| {
                        owned_soft_value_terminal(&engine, value_binding, error, report.actual_work)
                    });
                    if report.actual_work.is_none()
                        || (report.result.is_err() && nnuec_soft_terminal.is_none())
                    {
                        engine_reusable = false;
                    }
                    if report.terminal {
                        engine.discard_paused_stack();
                        if nnuec_soft_terminal.is_some()
                            && (!engine.paused_stack_snapshot().is_some_and(|snapshot| {
                                snapshot.complete
                                    && snapshot.owner_id != 0
                                    && snapshot.tokens_retained == 0
                                    && snapshot.bytes_current == 0
                            }) || engine.paused_stack_bytes().is_some()
                                || engine.pending_reciprocal_request().is_some())
                        {
                            // No repair of counters, owner IDs, states or
                            // unknown cleanup. Preserve the original Err.
                            engine_reusable = false;
                            nnuec_soft_terminal = None;
                        }
                        let snapshot = task.job.mailbox.snapshot();
                        if snapshot.cancel_sequence > snapshot.cancel_ack {
                            let _ = task
                                .job
                                .mailbox
                                .acknowledge_cancel(snapshot.cancel_sequence);
                        }
                        task.job.mailbox.seal_terminal();
                        report.controls = Some(task.job.mailbox.snapshot());
                        let ending = if report.actual_work.is_none() {
                            NnuecCaptureEnding::Unknown
                        } else {
                            match &report.result {
                                Ok(value) => {
                                    NnuecCaptureEnding::CpuTerminal(value.report.completion)
                                }
                                Err(_) => nnuec_soft_terminal.map_or(
                                    NnuecCaptureEnding::Failed,
                                    NnuecCaptureEnding::CpuTerminal,
                                ),
                            }
                        };
                        if task.job.retire_nnuec_capture(owner, ending).is_err() {
                            engine_reusable = false;
                            report.result = Err(CpuError::InvalidLimits(
                                "NNUEC capture retirement is not attested",
                            ));
                        }
                        active = None;
                    } else {
                        report.controls = Some(task.job.mailbox.snapshot());
                    }
                    if panicked {
                        worker_shutdown.store(true, Ordering::Release);
                        worker_mailbox.close();
                    }
                } else if let Some(task) = active
                    .as_ref()
                    .filter(|task| task.job.envelope.id == grant.job)
                {
                    report.cumulative_nodes = task.spent;
                    if accepted_command && task.last_sequence == 0 {
                        // Shutdown can arrive between installing a fresh
                        // primary and its first invocation. No stack or
                        // work exists, so return the admission rejection
                        // with its actual closed question scope.
                        report.controls = Some(task.job.mailbox.seal_terminal());
                        let _ = task
                            .job
                            .retire_nnuec_capture(owner, NnuecCaptureEnding::NotStarted);
                        active = None;
                    }
                }
                let retain_permit = !report.admission_rejected && report.continuation_retained;
                if let Some(timing) = &mut owner_timing {
                    timing.report_ready_at = Instant::now();
                }
                if command
                    .reliable
                    .publish(CpuWorkerEventObservation {
                        original: CpuWorkerEvent::Slice(report),
                        timing: owner_timing,
                    })
                    .is_err()
                {
                    engine_reusable = false;
                    break;
                }
                if retain_permit {
                    if let Some(task) = active.as_mut() {
                        task.terminal_permit = Some(command.reliable);
                    }
                }
            }
            if let Some(task) = active {
                task.job.mailbox.seal_terminal();
                let _ = task
                    .job
                    .retire_nnuec_capture(owner, NnuecCaptureEnding::Unknown);
                engine_reusable = false;
            }
            engine.discard_paused_stack();
            engine_reusable &= engine.validate_frozen_value_namespace().is_ok();
            // Restore only after this own-CPU task has stopped and discarded
            // its paused token. An explicit failure stays failed even if the
            // guard's later Drop best-effort cleanup manages to restore it.
            let affinity_finished = affinity_owner.as_mut().map(|guard| {
                #[cfg(test)]
                if let Some(error) = hooks.affinity_finish_failure.take() {
                    return Err(error);
                }
                guard.finish()
            });
            if affinity_finished.as_ref().is_some_and(Result::is_err) {
                engine_reusable = false;
            }
            worker_mailbox.signal().notify();
            CpuWorkerExit {
                engine: engine_reusable.then_some(engine),
                affinity_finished,
            }
        });
        let join = match launcher(
            thread::Builder::new().name(format!("rz-cpu-{}", owner.0)),
            task,
        ) {
            Ok(join) => join,
            Err(_) => {
                let engine = staged_engine
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take()
                    .expect("failed OS spawn did not enter the CPU startup gate");
                return Err((engine, CpuWorkerError::Disconnected));
            }
        };
        Ok(Self {
            owner,
            max_depth,
            evaluation_namespace,
            search_conditions,
            command_capacity,
            paused_bytes_limit,
            shutdown,
            ready_readback,
            affinity_ready,
            affinity_finished: None,
            mailbox,
            signal,
            submission: Mutex::new(CpuSubmissionFence::default()),
            installation: Mutex::new(installation),
            join: Some(join),
            recovered_engine: None,
            joined_known: false,
            deferred_nnuec_source: None,
        })
    }
    /// Build a question from the immutable worker profile without copying its
    /// engine/TT. Controls notify the same owner signal as command admission.
    pub fn prepare_job(
        &self,
        envelope: CpuJobEnvelope,
        position: ContractPosition,
        root_moves: Option<Vec<BoardMove>>,
        target_depth: u16,
        clock: &InstantClock,
    ) -> Result<CpuJob, CpuWorkerError> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(CpuWorkerError::Disconnected);
        }
        if self
            .installation
            .lock()
            .map_err(|_| CpuWorkerError::Quarantined)?
            .pending()
        {
            return Err(CpuWorkerError::AlreadyReserved);
        }
        CpuJob::from_profile(
            envelope,
            position,
            root_moves,
            target_depth,
            clock,
            self.max_depth,
            self.evaluation_namespace,
            self.search_conditions,
            self.owner,
            self.signal.clone(),
        )
    }
    /// Last actual owner profile whose installation readback was collected.
    /// Use acknowledged_profile() to exclude an in-progress replacement.
    pub fn profile(&self) -> CpuWorkerProfile {
        CpuWorkerProfile {
            owner: self.owner,
            max_depth: self.max_depth,
            evaluation_namespace: self.evaluation_namespace,
            search_conditions: self.search_conditions,
        }
    }
    pub fn acknowledged_profile(&self) -> Result<CpuWorkerProfile, CpuWorkerError> {
        if self
            .installation
            .lock()
            .map_err(|_| CpuWorkerError::Quarantined)?
            .pending()
        {
            return Err(CpuWorkerError::AlreadyReserved);
        }
        if self.shutdown.load(Ordering::Acquire) {
            return Err(CpuWorkerError::Disconnected);
        }
        Ok(self.profile())
    }
    pub fn value_install_pending(&self) -> Result<bool, CpuWorkerError> {
        Ok(self
            .installation
            .lock()
            .map_err(|_| CpuWorkerError::Quarantined)?
            .pending())
    }
    /// A historical receipt from another worker with an equal numeric owner
    /// cannot authorize this owner's current value profile.
    pub fn validates_installation_receipt(&self, receipt: &ValueInstallReceipt) -> bool {
        !self.shutdown.load(Ordering::Acquire)
            && receipt.installed_profile() == Some(self.profile())
            && self
                .installation
                .lock()
                .is_ok_and(|endpoint| endpoint.validates_receipt(receipt))
    }
    /// Reserve one installation/readback seat without granting a native node
    /// or claiming that an actual owner is idle. Only its command decides that.
    pub fn try_install_prepared_value(
        &self,
        payload: PreparedValueInstallation,
    ) -> Result<InstallId, InstallRejected> {
        let _fence = match self.submission.lock() {
            Ok(fence) => fence,
            Err(_) => return Err(InstallRejected::new(payload, CpuWorkerError::Quarantined)),
        };
        if self.shutdown.load(Ordering::Acquire) {
            return Err(InstallRejected::new(payload, CpuWorkerError::Disconnected));
        }
        if self.thread_affinity_requested() && !matches!(self.thread_affinity_ready(), Some(Ok(_)))
        {
            return Err(InstallRejected::new(
                payload,
                CpuWorkerError::Invalid("CPU worker affinity apply/readback not ready"),
            ));
        }
        let mut installation = match self.installation.lock() {
            Ok(installation) => installation,
            Err(_) => return Err(InstallRejected::new(payload, CpuWorkerError::Quarantined)),
        };
        let id = installation.try_submit(payload, self.profile())?;
        self.signal.notify();
        Ok(id)
    }
    /// The endpoint verifies this worker's private channel/sequence/binding
    /// before returning a receipt. No caller-supplied profile can update it.
    pub fn try_install_readback(&mut self) -> Result<Option<ValueInstallReceipt>, CpuWorkerError> {
        let _fence = self
            .submission
            .lock()
            .map_err(|_| CpuWorkerError::Quarantined)?;
        let mut installation = self
            .installation
            .lock()
            .map_err(|_| CpuWorkerError::Quarantined)?;
        let receipt = installation.try_ack()?;
        if let Some(profile) = receipt
            .as_ref()
            .and_then(|receipt| receipt.result().ok())
            .map(ValueInstallReadback::profile)
        {
            if profile.owner != self.owner {
                return Err(CpuWorkerError::Quarantined);
            }
            self.max_depth = profile.max_depth;
            self.evaluation_namespace = profile.evaluation_namespace;
            self.search_conditions = profile.search_conditions;
        }
        Ok(receipt)
    }
    /// The actual mailbox and engine ceilings captured before worker start.
    pub fn command_capacity(&self) -> usize {
        self.command_capacity
    }
    pub fn paused_bytes_limit(&self) -> usize {
        self.paused_bytes_limit
    }
    /// Borrow the single real ready observation without blocking or copying
    /// CPU sets on manager ticks. A declaration is not an observed placement.
    pub fn ready_readback(&self) -> Option<&Result<CpuThreadReadback, CpuResourceError>> {
        self.ready_readback.get()
    }
    /// None is either the unchanged observe-only profile, or an opted-in
    /// worker whose install has not returned. Check `thread_affinity_requested`
    /// first; neither case means successful application.
    pub fn thread_affinity_requested(&self) -> bool {
        self.affinity_ready.is_some()
    }
    pub fn thread_affinity_ready(
        &self,
    ) -> Option<&Result<CpuThreadAffinityApplied, CpuThreadAffinityStartError>> {
        self.affinity_ready.as_ref().and_then(|slot| slot.get())
    }
    fn submit(&self, command: CpuCommand) -> Result<(), CpuSubmissionRejected> {
        let mut fence = match self.submission.lock() {
            Ok(fence) => fence,
            Err(_) => {
                return Err(CpuSubmissionRejected::from_command(
                    CpuWorkerError::Quarantined,
                    command,
                    false,
                ));
            }
        };
        if let Err(error) = fence.claim(self.owner, command.grant()) {
            return Err(CpuSubmissionRejected::from_command(error, command, false));
        }
        match self.installation.lock() {
            Ok(installation) if installation.pending() => {
                return Err(CpuSubmissionRejected::from_command(
                    CpuWorkerError::AlreadyReserved,
                    command,
                    true,
                ));
            }
            Err(_) => {
                return Err(CpuSubmissionRejected::from_command(
                    CpuWorkerError::Quarantined,
                    command,
                    true,
                ));
            }
            _ => {}
        }
        if self.shutdown.load(Ordering::Acquire) {
            return Err(CpuSubmissionRejected::from_command(
                CpuWorkerError::Disconnected,
                command,
                true,
            ));
        }
        if self.thread_affinity_requested() && !matches!(self.thread_affinity_ready(), Some(Ok(_)))
        {
            return Err(CpuSubmissionRejected::from_command(
                CpuWorkerError::Invalid("CPU worker affinity apply/readback not ready"),
                command,
                true,
            ));
        }
        self.mailbox
            .admit(
                TimedCpuCommand {
                    original: command,
                    admitted_at: Instant::now(),
                },
                2,
            )
            .map_err(|error| {
                let reason = match error.reason {
                    MailboxError::Closed => CpuWorkerError::Disconnected,
                    MailboxError::QueueFull | MailboxError::ReliableFull => {
                        CpuWorkerError::Capacity
                    }
                    _ => CpuWorkerError::Invalid("CPU mailbox admission failed"),
                };
                CpuSubmissionRejected::from_command(reason, error.value.original, true)
            })
    }
    pub fn begin(&self, job: CpuJob, grant: SliceGrant) -> Result<(), CpuWorkerError> {
        self.begin_retaining(job, grant)
            .map_err(|error| error.reason)
    }
    /// Preserve the immutable Job and an admission receipt on definite failure.
    pub fn begin_retaining(
        &self,
        job: CpuJob,
        grant: SliceGrant,
    ) -> Result<(), CpuSubmissionRejected> {
        if grant.owner != self.owner
            || grant.job != job.envelope.id
            || job.mailbox.scope()
                != &(CpuMailboxScope {
                    job: grant.job,
                    owner: self.owner,
                })
        {
            return Err(CpuSubmissionRejected::from_command(
                CpuWorkerError::Invalid("CPU job owner mismatch"),
                CpuCommand::Begin(Box::new(job), grant),
                false,
            ));
        }
        self.submit(CpuCommand::Begin(Box::new(job), grant))
    }
    /// Consume a prepared question at the actual owner boundary without
    /// enqueueing. This is used when hot-store admission fails after reserve.
    /// A repeated/enqueued ID or a foreign question cannot issue this proof.
    pub fn reject_prepared(
        &self,
        job: CpuJob,
        grant: SliceGrant,
    ) -> Result<CpuSubmissionRejected, CpuWorkerError> {
        // Preserve the legacy caller's error-only disposition. New cleanup
        // consumers retain the original Job through the boundary below.
        self.reject_prepared_retaining(job, grant)
            .map_err(|(error, _job)| error)
    }
    /// Actual worker/fence rejection of a unique prepared Job. Validation or
    /// fence failure returns that same Job without retiring its capture. Only
    /// the successful, definitely unqueued branch may publish NotStarted.
    /// This is not a ledger refund; the caller still settles the original
    /// reservation using the resulting CpuAdmissionRejection proof.
    #[allow(clippy::result_large_err)]
    pub fn reject_prepared_retaining(
        &self,
        job: CpuJob,
        grant: SliceGrant,
    ) -> Result<CpuSubmissionRejected, (CpuWorkerError, CpuJob)> {
        if grant.owner != self.owner
            || grant.job != job.envelope.id
            || job.mailbox.scope()
                != &(CpuMailboxScope {
                    job: grant.job,
                    owner: self.owner,
                })
            || job.envelope.evaluation_namespace != self.evaluation_namespace
            || job.envelope.search_conditions != self.search_conditions
        {
            return Err((
                CpuWorkerError::Invalid("CPU prepared rejection scope mismatch"),
                job,
            ));
        }
        let mut fence = match self.submission.lock() {
            Ok(fence) => fence,
            Err(_) => return Err((CpuWorkerError::Quarantined, job)),
        };
        if let Err(error) = fence.claim(self.owner, grant) {
            return Err((error, job));
        }
        Ok(CpuSubmissionRejected::from_command(
            CpuWorkerError::Invalid("CPU prepared question declined before enqueue"),
            CpuCommand::Begin(Box::new(job), grant),
            true,
        ))
    }
    /// Seal only a new unsubmitted resume execution. The old private stack is
    /// unaffected; no current/previously enqueued execution can be refunded.
    pub fn reject_unsubmitted_resume(
        &self,
        grant: SliceGrant,
    ) -> Result<CpuSubmissionRejected, CpuWorkerError> {
        let mut fence = self
            .submission
            .lock()
            .map_err(|_| CpuWorkerError::Quarantined)?;
        if fence.authority != Some(grant.job.authority) {
            return Err(CpuWorkerError::UnknownJob);
        }
        fence.claim(self.owner, grant)?;
        Ok(CpuSubmissionRejected::from_command(
            CpuWorkerError::Invalid("CPU resume declined before enqueue"),
            CpuCommand::Resume(grant),
            true,
        ))
    }
    pub fn resume(&self, grant: SliceGrant) -> Result<(), CpuWorkerError> {
        self.resume_attested(grant).map_err(|error| error.reason)
    }
    pub fn resume_attested(&self, grant: SliceGrant) -> Result<(), CpuSubmissionRejected> {
        self.submit(CpuCommand::Resume(grant))
    }

    /// Validate and consume the reply on the original private owner as one
    /// command. Rejection before invocation leaves its pending token intact.
    pub fn resume_with_reply_attested(
        &self,
        grant: SliceGrant,
        reply: CpuReciprocalReply,
    ) -> Result<(), CpuSubmissionRejected> {
        self.submit(CpuCommand::ResumeWithReply(grant, Box::new(reply)))
    }
    pub fn try_event(&self) -> Result<Option<CpuWorkerEvent>, CpuWorkerError> {
        self.try_event_observed()
            .map(|event| event.map(|event| event.original))
    }
    pub fn try_event_observed(&self) -> Result<Option<CpuWorkerEventObservation>, CpuWorkerError> {
        if let Some(report) = self.mailbox.take_reliable() {
            return Ok(Some(report));
        }
        if self.join.as_ref().is_some_and(JoinHandle::is_finished) {
            return Err(CpuWorkerError::Disconnected);
        }
        Ok(None)
    }
    pub fn latest_progress(&self) -> Option<CpuWorkerProgress> {
        self.latest_progress_observed()
            .map(|progress| progress.original)
    }
    pub fn latest_progress_observed(&self) -> Option<CpuWorkerProgressObservation> {
        self.mailbox.take_progress().map(|(_, progress)| progress)
    }
    pub fn occupancy(&self) -> rz_runtime::adaptive::MailboxOccupancy {
        self.mailbox.occupancy()
    }
    /// Request closure without waiting for CPU work or joining a running owner.
    pub fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        self.mailbox.close();
        self.signal.notify();
    }
    /// Manager polling joins only an already-finished owner. All reliable
    /// receipts are returned before disposal; None means it is still running.
    /// During closure retain a finite polling tick: the final signal is a wake
    /// hint and can precede JoinHandle's finished flag by a scheduling boundary.
    pub fn try_join(&mut self) -> Result<Option<CpuWorkerShutdown>, CpuWorkerError> {
        let join = self.join.as_ref().ok_or(CpuWorkerError::Duplicate)?;
        if !join.is_finished() {
            return Ok(None);
        }
        let join = self.join.take().expect("finished owner retained");
        Ok(Some(self.collect_joined(join)))
    }
    fn collect_joined(&mut self, join: JoinHandle<CpuWorkerExit>) -> CpuWorkerShutdown {
        let thread_panicked = match join.join() {
            Ok(exit) => {
                self.affinity_finished = exit.affinity_finished;
                self.recovered_engine = exit.engine;
                self.joined_known = self.recovered_engine.is_some();
                false
            }
            Err(_) => {
                self.joined_known = false;
                true
            }
        };
        let mut events = Vec::new();
        while let Some(event) = self.mailbox.take_reliable() {
            events.push(event.original);
        }
        let last_progress = self.latest_progress();
        let (value_installation_receipt, value_installation_error) =
            match self.try_install_readback() {
                Ok(receipt) => (receipt, None),
                Err(error) => (None, Some(error)),
            };
        if value_installation_error.is_some() {
            self.recovered_engine = None;
            self.joined_known = false;
        }
        CpuWorkerShutdown {
            owner: self.owner,
            events,
            last_progress,
            thread_joined: true,
            thread_panicked,
            mailbox: self.mailbox.occupancy(),
            ready_readback: self.ready_readback().cloned(),
            thread_affinity: if self.thread_affinity_requested() {
                CpuWorkerThreadAffinityShutdown::Requested {
                    ready: self.thread_affinity_ready().cloned(),
                    finished: self.affinity_finished.clone(),
                }
            } else {
                CpuWorkerThreadAffinityShutdown::NotRequested
            },
            engine_recoverable: self.joined_known,
            value_installation_receipt,
            value_installation_error,
        }
    }
    /// Move the same engine/TT out only after a known joined owner exit. The
    /// caller settles all returned receipts before spawning the next root.
    /// Caught panics, unknown work and failed owner executions quarantine it.
    pub fn take_engine(&mut self) -> Result<CpuEngine, CpuWorkerError> {
        if self.join.is_some() {
            return Err(CpuWorkerError::AlreadyReserved);
        }
        if !self.joined_known {
            return Err(CpuWorkerError::Quarantined);
        }
        self.recovered_engine
            .take()
            .ok_or(CpuWorkerError::Duplicate)
    }
    /// Explicit blocking shutdown for a caller outside the manager event loop.
    pub fn shutdown(mut self) -> Result<CpuWorkerShutdown, CpuWorkerError> {
        self.request_shutdown();
        let join = self.join.take().ok_or(CpuWorkerError::Duplicate)?;
        Ok(self.collect_joined(join))
    }
}
impl Drop for CpuWorker {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.mailbox.close();
        self.signal.notify();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

#[cfg(test)]
#[path = "cpu_worker_tests.rs"]
mod tests;

#[cfg(test)]
mod owner_clock_tests {
    use super::*;
    use rz_contracts::pals::SearchAuthority;
    use rz_contracts::{GameGeneration, ProcessEpoch, RootGeneration};

    fn original_progress() -> CpuWorkerProgress {
        let epoch = ProcessEpoch(1);
        CpuWorkerProgress {
            job: CpuJobId {
                authority: SearchAuthority {
                    epoch,
                    game: GameGeneration(1),
                    root: RootGeneration(1),
                    implementation: Digest([1; 32]),
                },
                sequence: 1,
            },
            execution: ExecutionId::new(epoch, 1),
            owner: CpuWorkerId(0),
            sequence: 1,
            iteration: CpuIterationProgress {
                best_move: None,
                score: 0,
                completed_depth: 2,
                nodes: 0,
            },
        }
    }

    #[test]
    fn restored_iteration_is_not_a_new_progress_interval() {
        let started = Instant::now();
        let mut publication = CpuWorkerProgressObservation {
            original: original_progress(),
            run_started_at: started,
            published_at: started + Duration::from_millis(3),
            interval_started_at: started,
            advanced_depth: false,
            resume_admission_and_restore: Some(Duration::from_millis(5)),
        };
        assert_eq!(publication.completed_progress_interval(), None);
        assert_eq!(
            publication.resume_admission_and_restore(),
            Some(Duration::from_millis(5))
        );
        publication.interval_started_at = publication.published_at;
        publication.published_at += Duration::from_millis(7);
        publication.advanced_depth = true;
        assert_eq!(
            publication.completed_progress_interval(),
            Some(Duration::from_millis(7))
        );
        // Admission/restore has its own scope and does not become this sample.
        assert_ne!(
            publication.completed_progress_interval(),
            publication.resume_admission_and_restore()
        );
    }

    #[test]
    fn feedback_return_excludes_owner_search_and_paused_residence() {
        let started = Instant::now();
        let timing = CpuWorkerSliceTiming {
            run_started_at: started,
            analyze_returned_at: started + Duration::from_millis(100),
            report_ready_at: started + Duration::from_millis(102),
            resume_admission_and_restore: None,
            #[cfg(feature = "nnuec-scalar-diagnostics")]
            nnuec_scalar: None,
        };
        assert_eq!(timing.return_processing(), Duration::from_millis(2));
        assert_eq!(timing.resume_admission_and_restore(), None);
    }
}
