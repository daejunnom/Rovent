//! Shared original custody for one V3 policy row and its game writer.
//!
//! A supervisor must retain the same owner before handing a clone to any
//! worker and through that worker's actual join. This is not a worker-join,
//! GPU fence, durable publication, or training-eligibility proof.

use super::producer_v3::{
    PendingPolicyPublicationV3, RoventCorpusProducerLockV3, RoventRetiredLockV3Record,
};
use super::speculative_route_v3::TargetPublicationV3;
use super::teacher::policy_prepare_v3::PreparedTeacherPolicyLockV3;
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
    BeforeAppendPanic(Box<dyn std::any::Any + Send>),
    AppendUncertain,
    Flush(ArenaError),
    FlushPanic(Box<dyn std::any::Any + Send>),
}

pub(super) enum PolicyPipelineProgressV3 {
    Appended,
    Busy,
    NotReady,
    Failed,
}

pub(super) enum PolicyFlushOutcomeV3 {
    Packet,
    Empty,
    NotReady,
    Failed,
}

pub(super) struct PolicyRowPipelineV3 {
    row: Option<RoventRetiredLockV3Record>,
    prepared: Option<PreparedTeacherPolicyLockV3>,
    pending: Option<PendingPolicyPublicationV3>,
    capability: Option<TargetPublicationV3>,
    writer: Option<RoventCorpusWriterLockV3>,
    appended: bool,
    failure: Option<PolicyPipelineFailureV3>,
}

#[derive(Clone)]
pub(super) struct PolicyRowPipelineOwnerV3(Arc<Mutex<PolicyRowPipelineV3>>);

impl PolicyRowPipelineOwnerV3 {
    pub(super) fn new(writer: RoventCorpusWriterLockV3) -> Self {
        Self(Arc::new(Mutex::new(PolicyRowPipelineV3 {
            row: None,
            prepared: None,
            pending: None,
            capability: None,
            writer: Some(writer),
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
}

impl PolicyRowPipelineV3 {
    fn idle(&self) -> bool {
        self.writer.is_some()
            && self.row.is_none()
            && self.prepared.is_none()
            && self.pending.is_none()
            && self.capability.is_none()
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

    pub(super) fn failure(&self) -> Option<&PolicyPipelineFailureV3> {
        self.failure.as_ref()
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
        if self.failure.is_some() || self.appended {
            return PolicyPipelineProgressV3::Failed;
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
            PolicyAppendResultV3::PreparationPanicked(payload) => {
                self.failure = Some(PolicyPipelineFailureV3::BeforeAppendPanic(payload));
                PolicyPipelineProgressV3::Failed
            }
            PolicyAppendResultV3::Uncertain => {
                self.failure = Some(PolicyPipelineFailureV3::AppendUncertain);
                PolicyPipelineProgressV3::Failed
            }
        }
    }

    /// The supervisor must retain this completed original evidence until its
    /// own separate game/teacher lifecycle permits release. This does not
    /// claim durable output or physical retirement.
    pub(super) fn transfer_appended(
        &mut self,
        target: &mut Option<PendingPolicyPublicationV3>,
    ) -> bool {
        if !self.appended || self.failure.is_some() || target.is_some() || self.pending.is_none() {
            return false;
        }
        *target = self.pending.take();
        self.appended = false;
        true
    }

    /// A successful packet moves only to another caller-owned retained slot.
    pub(super) fn flush_into(
        &mut self,
        output: &mut Option<RoventChunkPacket>,
    ) -> PolicyFlushOutcomeV3 {
        if self.failure.is_some() {
            return PolicyFlushOutcomeV3::Failed;
        }
        if output.is_some() {
            return PolicyFlushOutcomeV3::NotReady;
        }
        let Some(writer) = self.writer.as_mut() else {
            return PolicyFlushOutcomeV3::NotReady;
        };
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| writer.flush())) {
            Ok(Ok(Some(packet))) => {
                *output = Some(packet);
                PolicyFlushOutcomeV3::Packet
            }
            Ok(Ok(None)) => PolicyFlushOutcomeV3::Empty,
            Ok(Err(error)) => {
                self.failure = Some(PolicyPipelineFailureV3::Flush(error));
                PolicyFlushOutcomeV3::Failed
            }
            Err(payload) => {
                self.failure = Some(PolicyPipelineFailureV3::FlushPanic(payload));
                PolicyFlushOutcomeV3::Failed
            }
        }
    }

    /// Final assembly moves the original writer only after successful
    /// validation and into another caller-owned slot. The caller must retain
    /// the actual flushed chunk packets and separate lifecycle evidence.
    pub(super) fn finish_into(&mut self, output: &mut Option<FinishedRoventCorpusLockV3>) -> bool {
        if !self.idle() || output.is_some() {
            return false;
        }
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            RoventCorpusWriterLockV3::finish_in_slots(&mut self.writer, output)
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
