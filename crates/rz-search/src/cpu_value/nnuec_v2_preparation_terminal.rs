//! Actor-issued C terminal custody without retaining neural or View owners.
//!
//! This acknowledgement is move-only and concerns physical preparation only.
//! It grants no evaluation/worker/semantic acceptance/training authority. A
//! later actual materialization permanently disqualifies C-only cleanup.

use super::{BoundInstanceAuthority, PreparationStatus, Provenance, RequestIdentity};
use crate::cpu_value::CpuValueIdentity;
use crate::cpu_value::nnuec::v2::Model;
use crate::cpu_value::nnuec::v2::retirement::NnuecCacheReleaseAck;
use crate::pals::review_context::ReviewView;
use std::sync::{Arc, Weak, atomic::AtomicU64};

pub struct NnuecPreparationTerminalAck {
    provenance: Provenance,
    request: RequestIdentity,
    model: CpuValueIdentity,
    original_model: Weak<Model>,
    original_view: Weak<ReviewView>,
    status: PreparationStatus,
    instance: Option<BoundInstanceAuthority>,
    actor_authority: Arc<AtomicU64>,
    preparation_epoch: u64,
}
impl std::fmt::Debug for NnuecPreparationTerminalAck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NnuecPreparationTerminalAck")
            .field("provenance", &self.provenance)
            .field("status", &self.status)
            .field("preparation_epoch", &self.preparation_epoch)
            .finish_non_exhaustive()
    }
}
impl NnuecPreparationTerminalAck {
    // Independent request, model, view and actor authorities form this terminal proof.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        provenance: Provenance,
        request: RequestIdentity,
        model: CpuValueIdentity,
        original_model: Weak<Model>,
        original_view: Weak<ReviewView>,
        status: PreparationStatus,
        instance: Option<BoundInstanceAuthority>,
        actor_authority: Arc<AtomicU64>,
        preparation_epoch: u64,
    ) -> Self {
        Self {
            provenance,
            request,
            model,
            original_model,
            original_view,
            status,
            instance,
            actor_authority,
            preparation_epoch,
        }
    }
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }
    pub(crate) fn matches_request(&self, request: &RequestIdentity) -> bool {
        self.request.same_request(request)
    }
    pub fn model_identity(&self) -> &CpuValueIdentity {
        &self.model
    }
    /// Weak original-instance identity does not retain model weights or the
    /// View's Store pin. Equal public metadata cannot replace these owners.
    pub fn matches_original(&self, model: &Arc<Model>, view: &Arc<ReviewView>) -> bool {
        Weak::ptr_eq(&self.original_model, &Arc::downgrade(model))
            && Weak::ptr_eq(&self.original_view, &Arc::downgrade(view))
    }
    pub(crate) fn matches_original_weak(
        &self,
        model: &Weak<Model>,
        view: &Weak<ReviewView>,
    ) -> bool {
        Weak::ptr_eq(&self.original_model, model) && Weak::ptr_eq(&self.original_view, view)
    }
    pub fn status(&self) -> PreparationStatus {
        self.status
    }
    pub fn preparation_epoch(&self) -> u64 {
        self.preparation_epoch
    }
    pub fn allows_c_only_cleanup(&self) -> bool {
        self.instance
            .as_ref()
            .is_none_or(|instance| !instance.was_materialized())
    }
    pub fn matches_cache_release(&self, ack: &NnuecCacheReleaseAck) -> bool {
        ack.clears(&self.actor_authority, self.preparation_epoch)
    }
}
