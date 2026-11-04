//! Lightweight historical capture identity, activated by the original worker.
//!
//! This capability keeps no model, Bound, View, input, Rules snapshot, State
//! mutex, or Submission alive. It proves only the original activation and can
//! never submit, resume, evaluate, consume feedback, or issue a training target.

use super::super::review::{BoundInstanceAuthority, RequestIdentity};
use super::{CaptureError, NnuecCaptureReservation, NnuecCaptureRetirement};
use crate::pals::coordinator::FrameId;
use rz_contracts::pals_adaptive::{CpuJobEnvelope, CpuWorkerId};
use rz_contracts::{Deadline, Digest};
use std::sync::{Arc, OnceLock};

struct Activation {
    original_job: OnceLock<(CpuJobEnvelope, CpuWorkerId)>,
}

/// Clone shares one private original reservation identity and one immutable
/// activation. Equal public metadata cannot substitute another reservation.
#[derive(Clone)]
pub struct NnuecCaptureLineage {
    authority: Arc<Activation>,
    bound: BoundInstanceAuthority,
    request: RequestIdentity,
    frame: FrameId,
    question: Digest,
    capture_sequence: u64,
    original_deadline: Deadline,
    binding: Digest,
    full_view: Digest,
    view_seal: Digest,
}
impl std::fmt::Debug for NnuecCaptureLineage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NnuecCaptureLineage")
            .field("frame", &self.frame)
            .field("question", &self.question)
            .field("capture_sequence", &self.capture_sequence)
            .field("activated", &self.authority.original_job.get().is_some())
            .finish_non_exhaustive()
    }
}
impl NnuecCaptureLineage {
    // Each original identity and lease is checked independently at this authority boundary.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        bound: BoundInstanceAuthority,
        request: RequestIdentity,
        frame: FrameId,
        question: Digest,
        capture_sequence: u64,
        original_deadline: Deadline,
        binding: Digest,
        full_view: Digest,
        view_seal: Digest,
    ) -> Self {
        Self {
            authority: Arc::new(Activation {
                original_job: OnceLock::new(),
            }),
            bound,
            request,
            frame,
            question,
            capture_sequence,
            original_deadline,
            binding,
            full_view,
            view_seal,
        }
    }
    pub fn frame(&self) -> FrameId {
        self.frame
    }
    pub fn question_digest(&self) -> Digest {
        self.question
    }
    pub fn capture_sequence(&self) -> u64 {
        self.capture_sequence
    }
    pub fn original_deadline(&self) -> Deadline {
        self.original_deadline
    }
    /// Historical worker activation only, not an enqueue/terminal/usage
    /// proof. Absence alone cannot attest a manager's unsubmitted capture.
    pub fn activation_observed(&self) -> bool {
        self.authority.original_job.get().is_some()
    }
    pub fn binding_digest(&self) -> Digest {
        self.binding
    }
    pub fn full_view_digest(&self) -> Digest {
        self.full_view
    }
    pub fn view_seal_digest(&self) -> Digest {
        self.view_seal
    }

    /// Only capture::bind_job calls this after validating the original live
    /// Bound, immutable prepared Job, exact Rules snapshot and clock origin.
    pub(super) fn activate(
        &self,
        job: CpuJobEnvelope,
        owner: CpuWorkerId,
    ) -> Result<(), CaptureError> {
        if job.id.authority != self.frame.authority
            || job.question != self.question
            || job.evaluation_namespace != self.binding
            || job.hard_deadline.clock != self.original_deadline.clock
            || job.hard_deadline.at > self.original_deadline.at
        {
            return Err(CaptureError::Stale);
        }
        if let Some(original) = self.authority.original_job.get() {
            return if *original == (job, owner) {
                Ok(())
            } else {
                Err(CaptureError::Duplicate)
            };
        }
        self.authority
            .original_job
            .set((job, owner))
            .map_err(|_| CaptureError::Duplicate)
    }

    /// Preparation/attachment alone cannot pass. This remains a historical
    /// identity check after cancellation/retirement, not live execution rights.
    pub fn matches_activated_job(&self, job: CpuJobEnvelope, owner: CpuWorkerId) -> bool {
        self.authority.original_job.get() == Some(&(job, owner))
    }
    pub(super) fn same_authority(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.authority, &other.authority)
    }
    pub(crate) fn matches_original_bound(
        &self,
        bound: &BoundInstanceAuthority,
        request: &RequestIdentity,
    ) -> bool {
        self.bound.matches(bound) && self.request.same_request(request)
    }
    pub fn matches_reservation(&self, reservation: &NnuecCaptureReservation) -> bool {
        reservation.matches_lineage(self)
    }
    pub fn matches_retirement(&self, retirement: &NnuecCaptureRetirement) -> bool {
        retirement.matches_lineage(self)
    }
}
