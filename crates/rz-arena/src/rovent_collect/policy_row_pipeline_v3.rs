//! Shared original custody for one V3 policy row and its game writer.
//!
//! A supervisor must retain the same owner before handing a clone to any
//! worker and through that worker's actual join. This is not a worker-join,
//! GPU fence, durable publication, or training-eligibility proof.

use super::game_authority_v3::{GamePublicationOriginV3, GamePublishGuardV3, GameStepV3};
use super::launch::{
    ManagedOutputOwnerV3, ManagedOutputWriteErrorV3, PreparedPcCollectionInputsV3,
};
use super::producer_v3::{
    PendingPolicyPublicationV3, RoventCorpusProducerLockV3, RoventRetiredLockV3Record,
};
use super::speculative_route_v3::{
    SpeculativeRootV3, TargetBeginRejectionV3, TargetCapabilityTakeV3, TargetPublicationV3,
};
use super::teacher::policy_prepare_v3::{
    PreparedTeacherPolicyLockV3, prepare_teacher_policy_lock_v3_in_slots,
};
use super::teacher::v3_policy::RoventSealedTeacherPolicyV3;
use super::writer_lock_v3::{
    FinishedRoventCorpusLockV3, PolicyAppendResultV3, RoventCorpusWriterLockV3,
};
use super::{ArenaError, MAX_CORPUS_ROWS, RoventChunkPacket};
use rz_contracts::Digest;
use rz_uci::pals_adaptive::RoventCollectionRootAdmissionV3;
use sha2::{Digest as _, Sha256};
use std::path::Component;
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};

/// A failure remains in the same owner; an uncertain append also retains the
/// original writer and pending row and never authorizes retry or flush.
pub(super) enum PolicyPipelineFailureV3 {
    Stage(ArenaError),
    Prepare(ArenaError, Option<Box<dyn std::any::Any + Send>>),
    Bind(ArenaError),
    AppendRejected(ArenaError),
    GateRejected(TargetBeginRejectionV3),
    PolicyUnavailable(ArenaError),
    BeforeAppendPanic(Box<dyn std::any::Any + Send>),
    AppendUncertain,
    Flush(ArenaError),
    FlushPanic(Box<dyn std::any::Any + Send>),
    ChunkPrepare(ArenaError),
    ChunkPreparePanic(Box<dyn std::any::Any + Send>),
    ChunkIo(ArenaError),
    ChunkPanic(Box<dyn std::any::Any + Send>),
    PublicationBeforeWrite(String),
    PublicationUnknown(String),
    PublicationPanic(Box<dyn std::any::Any + Send>),
    ReceiptReadPanic(Box<dyn std::any::Any + Send>),
}

/// Pipeline-local result only; worker, native and parent lease termination
/// remain independently owned and independently evidenced.
pub(super) enum PolicyFinalizationV3 {
    Pending,
    Progress,
    NoData,
    Published,
    Aborted,
    Failed,
    Unknown,
}

pub(super) enum PolicyReceiptReadV3 {
    Busy,
    NotReady,
    Foreign,
    Available(rz_experiments::rovent_corpus_lock_v3::publication::RoventLockV3Publication),
    Unknown,
}

pub(super) enum PolicyPipelineProgressV3 {
    Appended,
    /// A complete prior chunk is in this pipeline's retained output owner.
    ChunkReady,
    Busy,
    NotReady,
    Failed,
    Unknown,
}

pub(super) enum PolicyFlushOutcomeV3 {
    Packet,
    Empty,
    NotReady,
    Failed,
    Unknown,
}

pub(super) enum PolicyChunkPersistenceV3 {
    Persisted,
    NotReady,
    Failed,
    Unknown,
}

pub(super) enum PolicyPrepareOutcomeV3 {
    Prepared,
    NotReady,
    Failed,
    Unknown,
}

pub(super) struct PolicyRowPipelineV3 {
    identity: Arc<()>,
    row: Option<RoventRetiredLockV3Record>,
    prepared: Option<PreparedTeacherPolicyLockV3>,
    pending: Option<PendingPolicyPublicationV3>,
    capability: Option<TargetPublicationV3>,
    writer: Option<RoventCorpusWriterLockV3>,
    output_chunk: Option<RoventChunkPacket>,
    output_completed: Option<PendingPolicyPublicationV3>,
    output_finished: Option<FinishedRoventCorpusLockV3>,
    output_publication:
        Option<rz_experiments::rovent_corpus_lock_v3::publication::RoventLockV3Publication>,
    output_origin: Option<GamePublicationOriginV3>,
    output_no_data: bool,
    output_aborted: bool,
    output_directory: Option<Arc<ManagedOutputOwnerV3>>,
    chunk_attempted: bool,
    chunk_receipts: Vec<rz_experiments::rovent_corpus::RoventCorpusChunk>,
    appended: bool,
    failure: Option<PolicyPipelineFailureV3>,
}

#[derive(Clone)]
pub(crate) struct PolicyRowPipelineOwnerV3(Arc<Mutex<PolicyRowPipelineV3>>);

/// A view of the very same game-scoped pipeline owner. It cannot be created
/// from a digest or from another root's arbitrary output slot.
#[derive(Clone)]
pub(super) struct PolicyOutputOwnerV3(PolicyRowPipelineOwnerV3);

struct PolicyRootRetainedV3 {
    pipeline_identity: Arc<()>,
    admission: RoventCollectionRootAdmissionV3,
    maximum: usize,
    completed: Vec<PendingPolicyPublicationV3>,
    rejected: Vec<RetainedPolicyRejectionV3>,
}

/// Known row-scoped failure and every original value move together. It does
/// not establish that the root, worker, teacher or output is complete.
struct RetainedPolicyRejectionV3 {
    row: Option<RoventRetiredLockV3Record>,
    prepared: Option<PreparedTeacherPolicyLockV3>,
    pending: Option<PendingPolicyPublicationV3>,
    capability: Option<TargetPublicationV3>,
    failure: PolicyPipelineFailureV3,
}

/// Original per-root retention, registered with the supervisor before the
/// worker starts. Retention does not prove worker join or physical release.
#[derive(Clone)]
pub(super) struct PolicyRootRetainedOwnerV3(Arc<Mutex<PolicyRootRetainedV3>>);

pub(super) enum PolicyRetentionOutcomeV3 {
    Transferred,
    Busy,
    NotReady,
    Capacity,
    /// The shared game slot currently holds another legitimate root.
    OtherRoot,
}

/// Retention observation only; none of these flags proves durable output,
/// target validity, worker join, or permission to release an owner.
pub(super) struct PolicyOutputStatusV3 {
    pub(super) chunk_held: bool,
    pub(super) completed_held: bool,
    pub(super) finished_held: bool,
    pub(super) chunk_write_attempted: bool,
    pub(super) synced_chunks: usize,
}

impl PolicyRowPipelineOwnerV3 {
    pub(super) fn new(writer: RoventCorpusWriterLockV3) -> Self {
        Self(Arc::new(Mutex::new(PolicyRowPipelineV3 {
            identity: Arc::new(()),
            row: None,
            prepared: None,
            pending: None,
            capability: None,
            writer: Some(writer),
            output_chunk: None,
            output_completed: None,
            output_finished: None,
            output_publication: None,
            output_origin: None,
            output_no_data: false,
            output_aborted: false,
            output_directory: None,
            chunk_attempted: false,
            chunk_receipts: Vec::new(),
            appended: false,
            failure: None,
        })))
    }

    /// None means transient contention. Poison is never treated as a free
    /// slot or completion; the same Arc still retains the original owner.
    pub(super) fn try_lock(
        &self,
    ) -> Result<Option<MutexGuard<'_, PolicyRowPipelineV3>>, &'static str> {
        match self.0.try_lock() {
            Ok(guard) => Ok(Some(guard)),
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Poisoned(_)) => Err("V3 policy pipeline owner poisoned"),
        }
    }

    pub(super) fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    pub(super) fn output_owner(&self) -> PolicyOutputOwnerV3 {
        PolicyOutputOwnerV3(self.clone())
    }

    /// Return an immutable receipt observation for the same original game.
    /// The original receipt and finished writer stay in this pipeline owner.
    pub(super) fn try_read_receipt<S>(&self, guard: &GamePublishGuardV3<S>) -> PolicyReceiptReadV3 {
        let mut state = match self.try_lock() {
            Ok(Some(state)) => state,
            Ok(None) => return PolicyReceiptReadV3::Busy,
            Err(_) => return PolicyReceiptReadV3::Unknown,
        };
        if state.is_uncertain() {
            return PolicyReceiptReadV3::Unknown;
        }
        let Some(origin) = state.output_origin.as_ref() else {
            return PolicyReceiptReadV3::NotReady;
        };
        if !guard.matches_original_game(origin) {
            return PolicyReceiptReadV3::Foreign;
        }
        let Some(receipt) = state.output_publication.as_ref() else {
            return PolicyReceiptReadV3::NotReady;
        };
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| receipt.clone())) {
            Ok(receipt) => PolicyReceiptReadV3::Available(receipt),
            Err(payload) => {
                state.failure = Some(PolicyPipelineFailureV3::ReceiptReadPanic(payload));
                PolicyReceiptReadV3::Unknown
            }
        }
    }

    /// Advance only this original pipeline after the real game owner has
    /// issued its publication guard. The caller retains both owners and the
    /// guard across Busy; no result here proves worker or lease termination.
    pub(super) fn try_finalize<S>(
        &self,
        guard: &mut GamePublishGuardV3<S>,
        until: std::time::Instant,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> PolicyFinalizationV3 {
        let mut state = match self.try_lock() {
            Ok(Some(state)) => state,
            Ok(None) => return PolicyFinalizationV3::Pending,
            Err(_) => return PolicyFinalizationV3::Unknown,
        };
        if state
            .output_origin
            .as_ref()
            .is_some_and(|origin| !guard.matches_original_game(origin))
        {
            return PolicyFinalizationV3::Failed;
        }
        if state.is_uncertain() {
            return PolicyFinalizationV3::Unknown;
        }
        if state.output_publication.is_some() {
            return PolicyFinalizationV3::Published;
        }
        if state.output_no_data {
            return PolicyFinalizationV3::NoData;
        }
        if state.output_aborted {
            return PolicyFinalizationV3::Aborted;
        }
        if state.failure.is_some() {
            return PolicyFinalizationV3::Failed;
        }
        let Some(output) = state.output_directory.as_ref().cloned() else {
            return PolicyFinalizationV3::Pending;
        };
        match guard.try_matches_output_pipeline(&output, self) {
            Ok(Some(true)) => {}
            Ok(None) => return PolicyFinalizationV3::Pending,
            Ok(Some(false)) => return PolicyFinalizationV3::Failed,
            Err(error) => {
                state.failure = Some(PolicyPipelineFailureV3::PublicationUnknown(error.into()));
                return PolicyFinalizationV3::Unknown;
            }
        }
        if state.output_origin.is_none() {
            state.output_origin = Some(guard.original_game_identity());
        }
        if state.output_finished.is_some() {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                output.checked_publish_corpus(
                    state
                        .output_finished
                        .as_ref()
                        .expect("checked finished owner"),
                    self,
                    guard,
                    until,
                    cancel,
                )
            }));
            return match result {
                Ok(Ok(receipt)) => {
                    state.output_publication = Some(receipt);
                    PolicyFinalizationV3::Published
                }
                Ok(Err(ManagedOutputWriteErrorV3::Busy)) => PolicyFinalizationV3::Pending,
                Ok(Err(ManagedOutputWriteErrorV3::BeforeWrite(error))) => {
                    state.failure = Some(PolicyPipelineFailureV3::PublicationBeforeWrite(error));
                    PolicyFinalizationV3::Failed
                }
                Ok(Err(ManagedOutputWriteErrorV3::AuthorityUnknown(error))) => {
                    state.failure = Some(PolicyPipelineFailureV3::PublicationUnknown(error));
                    PolicyFinalizationV3::Unknown
                }
                Ok(Err(ManagedOutputWriteErrorV3::AfterAttempt(error))) => {
                    state.failure = Some(PolicyPipelineFailureV3::PublicationUnknown(error));
                    PolicyFinalizationV3::Unknown
                }
                Err(payload) => {
                    state.failure = Some(PolicyPipelineFailureV3::PublicationPanic(payload));
                    PolicyFinalizationV3::Unknown
                }
            };
        }
        if !state.idle() {
            return PolicyFinalizationV3::Pending;
        }
        if state.output_chunk.is_some() {
            return match state.persist_chunk() {
                PolicyChunkPersistenceV3::Persisted => PolicyFinalizationV3::Progress,
                PolicyChunkPersistenceV3::NotReady => PolicyFinalizationV3::Pending,
                PolicyChunkPersistenceV3::Failed => PolicyFinalizationV3::Failed,
                PolicyChunkPersistenceV3::Unknown => PolicyFinalizationV3::Unknown,
            };
        }
        match state.flush_into() {
            PolicyFlushOutcomeV3::Packet => PolicyFinalizationV3::Progress,
            PolicyFlushOutcomeV3::NotReady => PolicyFinalizationV3::Pending,
            PolicyFlushOutcomeV3::Failed => PolicyFinalizationV3::Failed,
            PolicyFlushOutcomeV3::Unknown => PolicyFinalizationV3::Unknown,
            PolicyFlushOutcomeV3::Empty => {
                if state
                    .writer
                    .as_ref()
                    .is_some_and(|writer| writer.is_pristine())
                {
                    match guard.try_finish_no_data() {
                        GameStepV3::Ready => {
                            state.output_no_data = true;
                            PolicyFinalizationV3::NoData
                        }
                        GameStepV3::Busy => PolicyFinalizationV3::Pending,
                        GameStepV3::Aborted => {
                            state.output_aborted = true;
                            PolicyFinalizationV3::Aborted
                        }
                        _ => {
                            state.failure = Some(PolicyPipelineFailureV3::PublicationUnknown(
                                "game no-data termination unknown".into(),
                            ));
                            PolicyFinalizationV3::Unknown
                        }
                    }
                } else if state.finish_into() {
                    PolicyFinalizationV3::Progress
                } else if state.is_uncertain() {
                    PolicyFinalizationV3::Unknown
                } else if state.failure.is_some() {
                    PolicyFinalizationV3::Failed
                } else {
                    PolicyFinalizationV3::Pending
                }
            }
        }
    }
}

impl PolicyRootRetainedOwnerV3 {
    /// `None` is temporary pipeline contention; no root owner is created.
    pub(super) fn try_new(
        pipeline: &PolicyRowPipelineOwnerV3,
        admission: RoventCollectionRootAdmissionV3,
        maximum: usize,
    ) -> Result<Option<Self>, &'static str> {
        if maximum == 0 || maximum > MAX_CORPUS_ROWS as usize {
            return Err("V3 policy retention capacity invalid");
        }
        let Some(owner) = pipeline.try_lock()? else {
            return Ok(None);
        };
        Ok(Some(Self(Arc::new(Mutex::new(PolicyRootRetainedV3 {
            pipeline_identity: Arc::clone(&owner.identity),
            admission,
            maximum,
            completed: Vec::new(),
            rejected: Vec::new(),
        })))))
    }

    pub(super) fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Checks only identity of the pre-registered owners. None is Busy;
    /// neither result proves worker or teacher completion.
    pub(super) fn try_matches(
        &self,
        pipeline: &PolicyRowPipelineOwnerV3,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<Option<bool>, &'static str> {
        // Follow the same pipeline -> root retention nonblocking order as
        // transfer. No caller may hold retention and wait for pipeline.
        let Some(pipeline) = pipeline.try_lock()? else {
            return Ok(None);
        };
        let retained = match self.0.try_lock() {
            Ok(retained) => retained,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(TryLockError::Poisoned(_)) => return Err("V3 policy root retention poisoned"),
        };
        Ok(Some(
            Arc::ptr_eq(&pipeline.identity, &retained.pipeline_identity)
                && retained.admission.same_original_admission(admission),
        ))
    }
}

impl PolicyOutputOwnerV3 {
    pub(super) fn same_owner(&self, pipeline: &PolicyRowPipelineOwnerV3) -> bool {
        self.0.same_owner(pipeline)
    }

    pub(super) fn try_observe(&self) -> Result<Option<PolicyOutputStatusV3>, &'static str> {
        let Some(owner) = self.0.try_lock()? else {
            return Ok(None);
        };
        Ok(Some(PolicyOutputStatusV3 {
            chunk_held: owner.output_chunk.is_some(),
            completed_held: owner.output_completed.is_some(),
            finished_held: owner.output_finished.is_some(),
            chunk_write_attempted: owner.chunk_attempted,
            synced_chunks: owner.chunk_receipts.len(),
        }))
    }
}

impl PolicyRowPipelineV3 {
    fn idle(&self) -> bool {
        self.writer.is_some()
            && self.row.is_none()
            && self.prepared.is_none()
            && self.pending.is_none()
            && self.capability.is_none()
            && self.output_completed.is_none()
            && !self.appended
            && self.failure.is_none()
    }

    /// Caller owns both slots. On rejection no value is removed from either.
    pub(super) fn install_row(&mut self, source: &mut Option<RoventRetiredLockV3Record>) -> bool {
        if !self.idle() || source.is_none() {
            return false;
        }
        self.row = source.take();
        true
    }

    pub(super) fn install_prepared(
        &mut self,
        source: &mut Option<PreparedTeacherPolicyLockV3>,
    ) -> bool {
        if self.row.is_none()
            || self.pending.is_some()
            || self.prepared.is_some()
            || self.failure.is_some()
            || source.is_none()
        {
            return false;
        }
        self.prepared = source.take();
        true
    }

    /// The same caller-owned sealed slot is not moved until its accepted
    /// original row has passed all checks and the prepared slot is free.
    pub(super) fn try_prepare_sealed(
        &mut self,
        sealed: &mut Option<RoventSealedTeacherPolicyV3>,
    ) -> PolicyPrepareOutcomeV3 {
        if self.is_uncertain() {
            return PolicyPrepareOutcomeV3::Unknown;
        }
        if self.failure.is_some() {
            return PolicyPrepareOutcomeV3::Failed;
        }
        if self.row.is_none()
            || sealed.is_none()
            || self.prepared.is_some()
            || self.pending.is_some()
        {
            return PolicyPrepareOutcomeV3::NotReady;
        }
        let row = self.row.as_ref().expect("checked original row");
        let original = sealed.as_ref().expect("checked original sealed slot");
        if !original
            .capsule()
            .admission()
            .same_original_admission(row.admission())
            || original.capsule().ticket().id != row.ticket().id
        {
            // A game pipeline can temporarily hold another root/ticket's
            // legitimate row; do not label that as malformed evidence.
            return PolicyPrepareOutcomeV3::NotReady;
        }
        match prepare_teacher_policy_lock_v3_in_slots(row, sealed, &mut self.prepared) {
            Ok(()) => PolicyPrepareOutcomeV3::Prepared,
            Err(failure) => {
                self.failure = Some(if failure.policy_unavailable {
                    PolicyPipelineFailureV3::PolicyUnavailable(failure.error)
                } else {
                    PolicyPipelineFailureV3::Prepare(failure.error, failure.panic)
                });
                PolicyPrepareOutcomeV3::Failed
            }
        }
    }

    /// Exact ticket/route identity is checked by pending.bind, not by a
    /// caller-provided digest or by installing this opaque owner in a slot.
    pub(super) fn empty_capability_slot(&mut self) -> Option<&mut Option<TargetPublicationV3>> {
        if self
            .pending
            .as_ref()
            .is_none_or(|pending| pending.target_capability().is_some())
            || self.capability.is_some()
            || self.failure.is_some()
            || self.appended
        {
            return None;
        }
        Some(&mut self.capability)
    }

    /// Transfer only the capability owned by this exact root/ticket directly
    /// into the original supervisor-retained slot; never return a free cap.
    pub(super) fn try_take_target_capability(
        &mut self,
        route: &SpeculativeRootV3,
        ticket: &rz_contracts::pals_adaptive::RoleTicketEnvelope,
    ) -> Result<Option<TargetCapabilityTakeV3>, ArenaError> {
        let Some(pending) = self.pending.as_ref() else {
            return Ok(None);
        };
        if !route.matches_admission(pending.original_admission()) {
            // A different root may currently own the shared game pipeline.
            return Ok(None);
        }
        if pending.ticket() != *ticket {
            return Ok(None);
        }
        let Some(slot) = self.empty_capability_slot() else {
            return Ok(None);
        };
        route
            .try_take_target_capability_in_slot(ticket, slot)
            .map(Some)
    }

    pub(super) fn failure(&self) -> Option<&PolicyPipelineFailureV3> {
        self.failure.as_ref()
    }

    /// `None` means no root/ticket owner is present (for example, a writer
    /// failure before a row). An unrelated live row is ordinary NotReady.
    pub(super) fn scope_matches(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
        ticket: &rz_contracts::pals_adaptive::RoleTicketEnvelope,
    ) -> Option<bool> {
        if let Some(pending) = &self.pending {
            return Some(
                pending
                    .original_admission()
                    .same_original_admission(admission)
                    && pending.ticket() == *ticket,
            );
        }
        self.row.as_ref().map(|row| {
            row.admission().same_original_admission(admission) && row.ticket() == *ticket
        })
    }

    /// Unknown never authorizes retry, dropping the writer, or publication.
    pub(super) fn is_uncertain(&self) -> bool {
        matches!(
            self.failure.as_ref(),
            Some(
                PolicyPipelineFailureV3::AppendUncertain
                    | PolicyPipelineFailureV3::FlushPanic(_)
                    | PolicyPipelineFailureV3::ChunkIo(_)
                    | PolicyPipelineFailureV3::ChunkPanic(_)
                    | PolicyPipelineFailureV3::PublicationUnknown(_)
                    | PolicyPipelineFailureV3::PublicationPanic(_)
                    | PolicyPipelineFailureV3::ReceiptReadPanic(_)
            ) | Some(PolicyPipelineFailureV3::GateRejected(
                TargetBeginRejectionV3::Uncertain
                    | TargetBeginRejectionV3::Poisoned
                    | TargetBeginRejectionV3::AlreadyCommitted
            ))
        )
    }

    /// Bind the same original plan and managed directory owner before any
    /// row is handled. A pathname, equivalent bytes, or a separately loaded
    /// manifest is not an original-owner proof.
    pub(super) fn install_output_directory(
        &mut self,
        inputs: &Arc<PreparedPcCollectionInputsV3>,
        output: &Arc<ManagedOutputOwnerV3>,
    ) -> Result<bool, ArenaError> {
        super::check(
            self.idle()
                && self.output_directory.is_none()
                && self.output_chunk.is_none()
                && self.chunk_receipts.is_empty()
                && output.matches_inputs(inputs)
                && self.writer.as_ref().is_some_and(|writer| {
                    writer.is_pristine() && writer.matches_original_manifest(inputs.manifest())
                }),
            "V3 managed output plan, manifest or writer owner differs",
        )?;
        let identity = self
            .writer
            .as_ref()
            .expect("validated original writer")
            .original_identity();
        if !output
            .try_register_original_writer(&identity)
            .map_err(super::invalid)?
        {
            return Ok(false);
        }
        self.output_directory = Some(Arc::clone(output));
        Ok(true)
    }

    /// Persist the exact retained chunk through the original managed chunks
    /// directory capability. Any attempted write error/panic retains packet,
    /// owner and failure; the same path is never retried automatically.
    pub(super) fn persist_chunk(&mut self) -> PolicyChunkPersistenceV3 {
        if self.is_uncertain() || self.chunk_attempted {
            return PolicyChunkPersistenceV3::Unknown;
        }
        if self.failure.is_some() {
            return PolicyChunkPersistenceV3::Failed;
        }
        let (Some(packet), Some(output)) =
            (self.output_chunk.as_ref(), self.output_directory.as_ref())
        else {
            return PolicyChunkPersistenceV3::NotReady;
        };
        let prepare = || -> Result<_, ArenaError> {
            let path = std::path::Path::new(&packet.declaration.path);
            let mut components = path.components();
            let leaf = match (components.next(), components.next(), components.next()) {
                (Some(Component::Normal(first)), Some(Component::Normal(leaf)), None)
                    if first == "chunks" =>
                {
                    leaf.to_owned()
                }
                _ => return Err(super::invalid("V3 chunk path is not a chunks leaf")),
            };
            super::check(
                !path.is_absolute()
                    && packet.bytes.len() as u64 == packet.declaration.bytes
                    && super::hex(Digest(Sha256::digest(&packet.bytes).into()))
                        == packet.declaration.sha256,
                "V3 retained chunk size or hash differs",
            )?;
            Ok((leaf, packet.declaration.clone()))
        };
        let (leaf, receipt) = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(prepare))
        {
            Ok(Ok(value)) => value,
            Ok(Err(error)) => {
                self.failure = Some(PolicyPipelineFailureV3::ChunkPrepare(error));
                return PolicyChunkPersistenceV3::Failed;
            }
            Err(payload) => {
                self.failure = Some(PolicyPipelineFailureV3::ChunkPreparePanic(payload));
                return PolicyChunkPersistenceV3::Failed;
            }
        };
        if self.chunk_receipts.try_reserve(1).is_err() {
            self.failure = Some(PolicyPipelineFailureV3::ChunkPrepare(super::invalid(
                "V3 chunk receipt allocation",
            )));
            return PolicyChunkPersistenceV3::Failed;
        }
        self.chunk_attempted = true;
        let write = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            output.checked_write_chunk(&leaf, &packet.bytes)
        }));
        match write {
            Ok(Ok(())) => {
                self.chunk_receipts.push(receipt);
                self.output_chunk = None;
                self.chunk_attempted = false;
                PolicyChunkPersistenceV3::Persisted
            }
            Ok(Err(super::launch::ManagedOutputWriteErrorV3::Busy)) => {
                self.chunk_attempted = false;
                PolicyChunkPersistenceV3::NotReady
            }
            Ok(Err(super::launch::ManagedOutputWriteErrorV3::BeforeWrite(error))) => {
                self.chunk_attempted = false;
                self.failure = Some(PolicyPipelineFailureV3::ChunkPrepare(super::invalid(error)));
                PolicyChunkPersistenceV3::Failed
            }
            Ok(Err(super::launch::ManagedOutputWriteErrorV3::AuthorityUnknown(error))) => {
                self.chunk_attempted = false;
                self.failure = Some(PolicyPipelineFailureV3::PublicationUnknown(error));
                PolicyChunkPersistenceV3::Unknown
            }
            Ok(Err(super::launch::ManagedOutputWriteErrorV3::AfterAttempt(error))) => {
                self.failure = Some(PolicyPipelineFailureV3::ChunkIo(super::invalid(error)));
                PolicyChunkPersistenceV3::Unknown
            }
            Err(payload) => {
                self.failure = Some(PolicyPipelineFailureV3::ChunkPanic(payload));
                PolicyChunkPersistenceV3::Unknown
            }
        }
    }

    /// Called outside the native manager. The producer belongs to the same
    /// original observer and is only borrowed while the supervisor retains it.
    pub(super) fn prepare(
        &mut self,
        producer: &mut RoventCorpusProducerLockV3,
        admission: &RoventCollectionRootAdmissionV3,
        ticket: &rz_contracts::pals_adaptive::RoleTicketEnvelope,
    ) -> bool {
        if self.failure.is_some() || self.appended {
            return false;
        }
        if let Some(pending) = &self.pending {
            return pending
                .original_admission()
                .same_original_admission(admission)
                && pending.ticket() == *ticket;
        }
        // A teacher completion and retired row may arrive in either order.
        // Missing prepared evidence is ordinary asynchronous NotReady.
        if self.prepared.is_none() {
            return false;
        }
        let Some(row) = self.row.as_mut() else {
            return false;
        };
        if !row.admission().same_original_admission(admission) || row.ticket() != *ticket {
            return false;
        }
        if let Err(error) = producer.stage_prepared_teacher_policy(row, &mut self.prepared) {
            self.failure = Some(PolicyPipelineFailureV3::Stage(error));
            return false;
        }
        match producer.prepare_policy_publication_in_slots(&mut self.row, &mut self.pending) {
            Ok(()) => true,
            Err(failure) => {
                self.failure = Some(PolicyPipelineFailureV3::Prepare(
                    failure.error,
                    failure.panic_payload,
                ));
                false
            }
        }
    }

    /// Runs only in the retained worker. It never takes an observer or
    /// registration lock while the target gate is held by the writer.
    pub(super) fn try_append(
        &mut self,
        admission: &RoventCollectionRootAdmissionV3,
        ticket: &rz_contracts::pals_adaptive::RoleTicketEnvelope,
    ) -> PolicyPipelineProgressV3 {
        if self.is_uncertain() {
            return PolicyPipelineProgressV3::Unknown;
        }
        if self.pending.as_ref().is_some_and(|pending| {
            !pending
                .original_admission()
                .same_original_admission(admission)
                || pending.ticket() != *ticket
        }) {
            // Another root may legitimately occupy this game pipeline.
            return PolicyPipelineProgressV3::NotReady;
        }
        if self.failure.is_some() || self.appended {
            return PolicyPipelineProgressV3::Failed;
        }
        let needs_flush = match (self.writer.as_ref(), self.pending.as_ref()) {
            (Some(writer), Some(pending)) => {
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    writer.policy_chunk_flush_required(pending)
                })) {
                    Ok(Ok(value)) => value,
                    Ok(Err(error)) => {
                        self.failure = Some(PolicyPipelineFailureV3::Flush(error));
                        return PolicyPipelineProgressV3::Failed;
                    }
                    Err(payload) => {
                        self.failure = Some(PolicyPipelineFailureV3::BeforeAppendPanic(payload));
                        return PolicyPipelineProgressV3::Failed;
                    }
                }
            }
            _ => return PolicyPipelineProgressV3::NotReady,
        };
        if needs_flush {
            return match self.flush_into() {
                PolicyFlushOutcomeV3::Packet => PolicyPipelineProgressV3::ChunkReady,
                PolicyFlushOutcomeV3::NotReady => PolicyPipelineProgressV3::NotReady,
                PolicyFlushOutcomeV3::Unknown => PolicyPipelineProgressV3::Unknown,
                PolicyFlushOutcomeV3::Empty | PolicyFlushOutcomeV3::Failed => {
                    PolicyPipelineProgressV3::Failed
                }
            };
        }
        let Some(pending) = self.pending.as_mut() else {
            return PolicyPipelineProgressV3::NotReady;
        };
        if pending.target_capability().is_none() {
            if self.capability.is_none() {
                return PolicyPipelineProgressV3::NotReady;
            }
            if let Err(error) = pending.bind_target_capability(&mut self.capability) {
                self.failure = Some(PolicyPipelineFailureV3::Bind(error));
                return PolicyPipelineProgressV3::Failed;
            }
        }
        let Some(writer) = self.writer.as_mut() else {
            return PolicyPipelineProgressV3::Failed;
        };
        match writer.try_append_policy(pending) {
            PolicyAppendResultV3::Appended => {
                self.appended = true;
                PolicyPipelineProgressV3::Appended
            }
            PolicyAppendResultV3::Busy => PolicyPipelineProgressV3::Busy,
            PolicyAppendResultV3::Rejected(error) => {
                self.failure = Some(PolicyPipelineFailureV3::AppendRejected(error));
                PolicyPipelineProgressV3::Failed
            }
            PolicyAppendResultV3::GateRejected(error) => {
                self.failure = Some(PolicyPipelineFailureV3::GateRejected(error));
                if self.is_uncertain() {
                    PolicyPipelineProgressV3::Unknown
                } else {
                    PolicyPipelineProgressV3::Failed
                }
            }
            PolicyAppendResultV3::PreparationPanicked(payload) => {
                self.failure = Some(PolicyPipelineFailureV3::BeforeAppendPanic(payload));
                PolicyPipelineProgressV3::Failed
            }
            PolicyAppendResultV3::Uncertain => {
                self.failure = Some(PolicyPipelineFailureV3::AppendUncertain);
                PolicyPipelineProgressV3::Unknown
            }
        }
    }

    /// The supervisor must retain this completed original evidence until its
    /// own separate game/teacher lifecycle permits release. This does not
    /// claim durable output or physical retirement.
    pub(super) fn transfer_appended(&mut self) -> bool {
        if !self.appended
            || self.failure.is_some()
            || self.output_completed.is_some()
            || self.pending.is_none()
        {
            return false;
        }
        self.output_completed = self.pending.take();
        self.appended = false;
        true
    }

    /// Move completed evidence only to the supervisor's root-bound owner,
    /// checking both the original game pipeline and original root. Both
    /// owners remain separately retained; no physical completion is inferred.
    pub(super) fn try_retain_completed(
        &mut self,
        target: &PolicyRootRetainedOwnerV3,
    ) -> Result<PolicyRetentionOutcomeV3, ArenaError> {
        let Some(completed) = self.output_completed.as_ref() else {
            return Ok(PolicyRetentionOutcomeV3::NotReady);
        };
        let mut target = match target.0.try_lock() {
            Ok(target) => target,
            Err(TryLockError::WouldBlock) => return Ok(PolicyRetentionOutcomeV3::Busy),
            Err(TryLockError::Poisoned(_)) => {
                return Err(super::invalid("V3 policy root retention poisoned"));
            }
        };
        if !Arc::ptr_eq(&self.identity, &target.pipeline_identity) {
            return Err(super::invalid("V3 policy retention pipeline differs"));
        }
        if !completed
            .original_admission()
            .same_original_admission(&target.admission)
        {
            return Ok(PolicyRetentionOutcomeV3::OtherRoot);
        }
        if target.completed.len().saturating_add(target.rejected.len()) >= target.maximum {
            return Ok(PolicyRetentionOutcomeV3::Capacity);
        }
        target
            .completed
            .try_reserve(1)
            .map_err(|_| super::invalid("V3 policy retention allocation"))?;
        let original = self
            .output_completed
            .take()
            .expect("validated original output");
        target.completed.push(original);
        Ok(PolicyRetentionOutcomeV3::Transferred)
    }

    /// Isolate a known row-scoped rejection from the shared game pipeline
    /// without dropping the row, capability, evidence or failure. Unknown
    /// writer state stays quarantined on the original pipeline instead.
    pub(super) fn try_retain_known_rejection(
        &mut self,
        target: &PolicyRootRetainedOwnerV3,
    ) -> Result<PolicyRetentionOutcomeV3, ArenaError> {
        if self.is_uncertain() {
            return Ok(PolicyRetentionOutcomeV3::NotReady);
        }
        let Some(failure) = self.failure.as_ref() else {
            return Ok(PolicyRetentionOutcomeV3::NotReady);
        };
        if !matches!(
            failure,
            PolicyPipelineFailureV3::Stage(_)
                | PolicyPipelineFailureV3::Prepare(_, _)
                | PolicyPipelineFailureV3::Bind(_)
                | PolicyPipelineFailureV3::AppendRejected(_)
                | PolicyPipelineFailureV3::GateRejected(TargetBeginRejectionV3::Revoked)
                | PolicyPipelineFailureV3::PolicyUnavailable(_)
                | PolicyPipelineFailureV3::BeforeAppendPanic(_)
        ) {
            return Ok(PolicyRetentionOutcomeV3::NotReady);
        }
        let admission = if let Some(pending) = &self.pending {
            pending.original_admission()
        } else if let Some(row) = &self.row {
            row.admission()
        } else {
            return Ok(PolicyRetentionOutcomeV3::NotReady);
        };
        let mut target = match target.0.try_lock() {
            Ok(target) => target,
            Err(TryLockError::WouldBlock) => return Ok(PolicyRetentionOutcomeV3::Busy),
            Err(TryLockError::Poisoned(_)) => {
                return Err(super::invalid("V3 policy root retention poisoned"));
            }
        };
        if !Arc::ptr_eq(&self.identity, &target.pipeline_identity) {
            return Err(super::invalid("V3 policy retention pipeline differs"));
        }
        if !admission.same_original_admission(&target.admission) {
            return Ok(PolicyRetentionOutcomeV3::OtherRoot);
        }
        if target.completed.len().saturating_add(target.rejected.len()) >= target.maximum {
            return Ok(PolicyRetentionOutcomeV3::Capacity);
        }
        target
            .rejected
            .try_reserve(1)
            .map_err(|_| super::invalid("V3 policy rejection retention allocation"))?;
        let original = RetainedPolicyRejectionV3 {
            row: self.row.take(),
            prepared: self.prepared.take(),
            pending: self.pending.take(),
            capability: self.capability.take(),
            failure: self.failure.take().expect("validated original failure"),
        };
        target.rejected.push(original);
        Ok(PolicyRetentionOutcomeV3::Transferred)
    }

    /// A successful packet remains in this same original output owner.
    pub(super) fn flush_into(&mut self) -> PolicyFlushOutcomeV3 {
        if self.is_uncertain() {
            return PolicyFlushOutcomeV3::Unknown;
        }
        if self.failure.is_some() {
            return PolicyFlushOutcomeV3::Failed;
        }
        if self.output_chunk.is_some() {
            return PolicyFlushOutcomeV3::NotReady;
        }
        let Some(writer) = self.writer.as_mut() else {
            return PolicyFlushOutcomeV3::NotReady;
        };
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| writer.flush())) {
            Ok(Ok(Some(packet))) => {
                self.output_chunk = Some(packet);
                PolicyFlushOutcomeV3::Packet
            }
            Ok(Ok(None)) => PolicyFlushOutcomeV3::Empty,
            Ok(Err(error)) => {
                self.failure = Some(PolicyPipelineFailureV3::Flush(error));
                PolicyFlushOutcomeV3::Failed
            }
            Err(payload) => {
                self.failure = Some(PolicyPipelineFailureV3::FlushPanic(payload));
                PolicyFlushOutcomeV3::Unknown
            }
        }
    }

    /// Final assembly moves the original writer only after successful
    /// validation and into this same output owner. The caller must retain
    /// the actual flushed chunk packets and separate lifecycle evidence.
    pub(super) fn finish_into(&mut self) -> bool {
        if !self.idle()
            || self.output_finished.is_some()
            || self.output_directory.is_none()
            || self.output_chunk.is_some()
            || self.chunk_attempted
            || !self
                .writer
                .as_ref()
                .is_some_and(|writer| writer.matches_synced_chunks(&self.chunk_receipts))
        {
            return false;
        }
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            RoventCorpusWriterLockV3::finish_in_slots(&mut self.writer, &mut self.output_finished)
        })) {
            Ok(Ok(())) => true,
            Ok(Err(error)) => {
                self.failure = Some(PolicyPipelineFailureV3::Flush(error));
                false
            }
            Err(payload) => {
                self.failure = Some(PolicyPipelineFailureV3::FlushPanic(payload));
                false
            }
        }
    }
}
