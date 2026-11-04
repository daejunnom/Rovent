//! Shared original custody for one V3 policy row and its game writer.
//!
//! A supervisor must retain the same owner before handing a clone to any
//! worker and through that worker's actual join. This is not a worker-join,
//! GPU fence, durable publication, or training-eligibility proof.

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
use super::{ArenaError, RoventChunkPacket};
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

pub(super) enum PolicyPrepareOutcomeV3 {
    Prepared,
    NotReady,
    Failed,
    Unknown,
}

pub(super) struct PolicyRowPipelineV3 {
    row: Option<RoventRetiredLockV3Record>,
    prepared: Option<PreparedTeacherPolicyLockV3>,
    pending: Option<PendingPolicyPublicationV3>,
    capability: Option<TargetPublicationV3>,
    writer: Option<RoventCorpusWriterLockV3>,
    output_chunk: Option<RoventChunkPacket>,
    output_completed: Option<PendingPolicyPublicationV3>,
    output_finished: Option<FinishedRoventCorpusLockV3>,
    appended: bool,
    failure: Option<PolicyPipelineFailureV3>,
}

#[derive(Clone)]
pub(super) struct PolicyRowPipelineOwnerV3(Arc<Mutex<PolicyRowPipelineV3>>);

/// A view of the very same game-scoped pipeline owner. It cannot be created
/// from a digest or from another root's arbitrary output slot.
#[derive(Clone)]
pub(super) struct PolicyOutputOwnerV3(PolicyRowPipelineOwnerV3);

/// Retention observation only; none of these flags proves durable output,
/// target validity, worker join, or permission to release an owner.
pub(super) struct PolicyOutputStatusV3 {
    pub(super) chunk_held: bool,
    pub(super) completed_held: bool,
    pub(super) finished_held: bool,
}

impl PolicyRowPipelineOwnerV3 {
    pub(super) fn new(writer: RoventCorpusWriterLockV3) -> Self {
        Self(Arc::new(Mutex::new(PolicyRowPipelineV3 {
            row: None,
            prepared: None,
            pending: None,
            capability: None,
            writer: Some(writer),
            output_chunk: None,
            output_completed: None,
            output_finished: None,
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
    ) -> Result<Option<TargetCapabilityTakeV3>, ArenaError> {
        let Some(pending) = self.pending.as_ref() else {
            return Ok(None);
        };
        if !route.matches_admission(pending.original_admission()) {
            return Err(super::invalid(
                "policy route differs from original admission",
            ));
        }
        let ticket = pending.ticket();
        let Some(slot) = self.empty_capability_slot() else {
            return Ok(None);
        };
        route
            .try_take_target_capability_in_slot(&ticket, slot)
            .map(Some)
    }

    pub(super) fn failure(&self) -> Option<&PolicyPipelineFailureV3> {
        self.failure.as_ref()
    }

    /// Unknown never authorizes retry, dropping the writer, or publication.
    pub(super) fn is_uncertain(&self) -> bool {
        matches!(
            self.failure.as_ref(),
            Some(PolicyPipelineFailureV3::AppendUncertain | PolicyPipelineFailureV3::FlushPanic(_))
                | Some(PolicyPipelineFailureV3::GateRejected(
                    TargetBeginRejectionV3::Uncertain
                        | TargetBeginRejectionV3::Poisoned
                        | TargetBeginRejectionV3::AlreadyCommitted
                ))
        )
    }

    /// Called outside the native manager. The producer belongs to the same
    /// original observer and is only borrowed while the supervisor retains it.
    pub(super) fn prepare(&mut self, producer: &mut RoventCorpusProducerLockV3) -> bool {
        if self.failure.is_some() || self.appended {
            return false;
        }
        if self.pending.is_some() {
            return true;
        }
        // A teacher completion and retired row may arrive in either order.
        // Missing prepared evidence is ordinary asynchronous NotReady.
        if self.prepared.is_none() {
            return false;
        }
        let Some(row) = self.row.as_mut() else {
            return false;
        };
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
    pub(super) fn try_append(&mut self) -> PolicyPipelineProgressV3 {
        if self.is_uncertain() {
            return PolicyPipelineProgressV3::Unknown;
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
        if !self.idle() || self.output_finished.is_some() {
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
