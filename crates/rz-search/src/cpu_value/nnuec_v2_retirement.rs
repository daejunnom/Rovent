//! Cleanup-only ownership of an actual installed V2 evaluator.
//!
//! A revoked/expired Bound can still be physically released. This seal grants
//! no score, resume, new delta, source conversion, or view consumption rights.

use super::Model;
use super::review::{BoundNnuecReviewContext, Provenance, RequestIdentity};
use crate::cpu_value::CpuValueIdentity;
use rz_contracts::Digest;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

/// A cache-clear request from one actual actor. The claim bit prevents
/// repeated polling/Clone from issuing a second cache-release acknowledgement.
#[derive(Clone)]
pub struct NnuecCacheInvalidation {
    authority: Arc<AtomicU64>,
    epoch: u64,
    claimed: Arc<AtomicBool>,
}
impl std::fmt::Debug for NnuecCacheInvalidation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NnuecCacheInvalidation")
            .field("epoch", &self.epoch)
            .finish_non_exhaustive()
    }
}
impl NnuecCacheInvalidation {
    pub(super) fn new(authority: Arc<AtomicU64>, epoch: u64) -> Self {
        Self {
            authority,
            epoch,
            claimed: Arc::new(AtomicBool::new(false)),
        }
    }
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
    pub(super) fn matches_actor(&self, authority: &Arc<AtomicU64>) -> bool {
        Arc::ptr_eq(&self.authority, authority)
    }
    pub(super) fn claim(&self) -> Result<NnuecCacheReleaseAck, super::review::PreparationError> {
        self.claimed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| {
                super::review::PreparationError::InvalidView("cache release already claimed")
            })?;
        Ok(NnuecCacheReleaseAck {
            authority: Arc::clone(&self.authority),
            epoch: self.epoch,
        })
    }
}

/// Actual cache release observation, not engine/evaluator release. No public
/// constructor, Deserialize or Clone can turn a number into this witness.
pub struct NnuecCacheReleaseAck {
    authority: Arc<AtomicU64>,
    epoch: u64,
}
impl std::fmt::Debug for NnuecCacheReleaseAck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NnuecCacheReleaseAck")
            .field("epoch", &self.epoch)
            .finish_non_exhaustive()
    }
}
impl NnuecCacheReleaseAck {
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
    pub(crate) fn clears(&self, authority: &Arc<AtomicU64>, original_epoch: u64) -> bool {
        Arc::ptr_eq(&self.authority, authority) && self.epoch > original_epoch
    }
    /// Exact actor/epoch only. A prelude issuer must additionally consume this
    /// ACK before submitting its unique Request and verify the cache is empty.
    pub(crate) fn prepares(&self, authority: &Arc<AtomicU64>, preparation_epoch: u64) -> bool {
        Arc::ptr_eq(&self.authority, authority) && self.epoch == preparation_epoch
    }
}

/// Constructed only by the actual V2 evaluator; numeric/profile declarations
/// cannot construct the original model and preparation-actor authority.
#[derive(Clone)]
pub struct NnuecValueRetirementOwner {
    model: Arc<Model>,
    bound: Arc<BoundNnuecReviewContext>,
    binding: Digest,
}
impl NnuecValueRetirementOwner {
    pub(super) fn new(
        model: Arc<Model>,
        bound: Arc<BoundNnuecReviewContext>,
        binding: Digest,
    ) -> Self {
        Self {
            model,
            bound,
            binding,
        }
    }
    pub fn model_identity(&self) -> &CpuValueIdentity {
        self.model.identity()
    }
    pub fn binding_digest(&self) -> Digest {
        self.binding
    }
    /// Cleanup-only readback from this same retained original Bound. It
    /// issues no new score, binding, deadline, worker or resume authority.
    pub(crate) fn authority_revocation(
        &self,
    ) -> Option<crate::cpu_worker::NnuecAuthorityRevocation> {
        self.bound.authority_revocation()
    }
    /// The admitted original clock boundary, read for cleanup classification
    /// only. This never extends, replaces or revives the Bound's deadline.
    pub(crate) fn original_hard_deadline(&self) -> std::time::Instant {
        self.bound.original_hard_deadline()
    }
    pub fn provenance(&self) -> &Provenance {
        self.bound.provenance()
    }
    pub(crate) fn preparation_epoch(&self) -> u64 {
        self.bound.preparation_epoch()
    }
    pub(crate) fn request_identity(&self) -> RequestIdentity {
        self.bound.request_identity()
    }
    pub(crate) fn matches_model(&self, model: &Arc<Model>) -> bool {
        Arc::ptr_eq(&self.model, model)
    }
    pub(crate) fn matches_bound(&self, bound: &Arc<BoundNnuecReviewContext>) -> bool {
        Arc::ptr_eq(&self.bound, bound)
    }
    pub(crate) fn matches_preparation_actor(
        &self,
        model: &Arc<Model>,
        stopping: &Arc<AtomicBool>,
        invalidations: &Arc<AtomicU64>,
    ) -> bool {
        self.matches_model(model)
            && self
                .bound
                .matches_preparation_actor(stopping, invalidations)
    }
}
