//! One-use actual joined-worker custody and actor-side evaluator release.
//!
//! Only an actual known joined owner can transfer its same unique engine.
//! No metadata-only constructor can provide an engine release.

use super::{CpuWorker, CpuWorkerError, CpuWorkerProfile, DeferredNnuecRSource};
use crate::cpu::CpuEngine;
use crate::cpu_value::CpuValueIdentity;
use crate::cpu_value::nnuec::v2::Model;
use crate::cpu_value::nnuec::v2::retirement::{NnuecCacheReleaseAck, NnuecValueRetirementOwner};
use crate::cpu_value::nnuec::v2::review::{Provenance, RequestIdentity};
use rz_contracts::Digest;
use rz_contracts::pals_adaptive::CpuWorkerId;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64},
};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct NnuecEngineRetirementId {
    authority: Arc<()>,
    sequence: u64,
}
impl PartialEq for NnuecEngineRetirementId {
    fn eq(&self, other: &Self) -> bool {
        self.sequence == other.sequence && Arc::ptr_eq(&self.authority, &other.authority)
    }
}
impl Eq for NnuecEngineRetirementId {}
impl NnuecEngineRetirementId {
    pub(crate) fn new(authority: Arc<()>, sequence: u64) -> Self {
        Self {
            authority,
            sequence,
        }
    }
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
}

#[derive(Debug)]
pub struct NnuecEngineRetirementCompletion {
    id: NnuecEngineRetirementId,
    result: Result<NnuecEngineReleaseAck, (RecoveredNnuecEngine, CpuWorkerError)>,
}
impl NnuecEngineRetirementCompletion {
    pub(crate) fn new(
        id: NnuecEngineRetirementId,
        result: Result<NnuecEngineReleaseAck, (RecoveredNnuecEngine, CpuWorkerError)>,
    ) -> Self {
        Self { id, result }
    }
    pub fn request_id(&self) -> &NnuecEngineRetirementId {
        &self.id
    }
    #[allow(clippy::result_large_err)]
    pub fn into_result(
        self,
    ) -> Result<NnuecEngineReleaseAck, (RecoveredNnuecEngine, CpuWorkerError)> {
        self.result
    }
}

/// The exact joined worker transfers its same unique engine. Do not drop this
/// on the manager loop: submit it to the same preparation actor, or preserve
/// it as cleanup/quarantine custody if that owner is not available.
#[must_use = "retain the actual engine until its actor release acknowledgement"]
pub struct RecoveredNnuecEngine {
    source: DeferredNnuecRSource,
    engine: CpuEngine,
    value_owner: NnuecValueRetirementOwner,
    profile: CpuWorkerProfile,
    stack_owner_id: u64,
}
impl std::fmt::Debug for RecoveredNnuecEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecoveredNnuecEngine")
            .field("profile", &self.profile)
            .field("stack_owner_id", &self.stack_owner_id)
            .field("binding", &self.binding_digest())
            .finish_non_exhaustive()
    }
}
impl RecoveredNnuecEngine {
    pub fn owner(&self) -> CpuWorkerId {
        self.profile.owner
    }
    pub fn binding_digest(&self) -> Digest {
        self.value_owner.binding_digest()
    }
    pub fn actual_stack_owner_id(&self) -> u64 {
        self.stack_owner_id
    }
    pub fn profile(&self) -> CpuWorkerProfile {
        self.profile
    }
    pub fn model_identity(&self) -> &CpuValueIdentity {
        self.value_owner.model_identity()
    }
    pub fn provenance(&self) -> &Provenance {
        self.value_owner.provenance()
    }
    /// Scalar readback of this exact joined engine only. This is not a root
    /// aggregate, token authority, admission closure or release acknowledgement.
    pub fn paused_stack_snapshot(&self) -> Option<crate::cpu::CpuPausedStackSnapshot> {
        self.engine.paused_stack_snapshot()
    }
    pub fn matches_source(&self, source: &DeferredNnuecRSource) -> bool {
        self.source.same_authority(source)
    }
    /// Actual actor-only release. Original cancellation/expiry does not
    /// prevent cleanup; unrelated actor/source/namespace never owns it.
    #[allow(clippy::result_large_err)]
    pub(crate) fn release_on_actor(
        self,
        model: &Arc<Model>,
        stopping: &Arc<AtomicBool>,
        invalidations: &Arc<AtomicU64>,
    ) -> Result<NnuecEngineReleaseAck, (Self, CpuWorkerError)> {
        if !self
            .value_owner
            .matches_preparation_actor(model, stopping, invalidations)
            || !Arc::ptr_eq(self.source.model(), model)
            || self.profile.evaluation_namespace != self.value_owner.binding_digest()
            || self.model_identity() != model.identity()
        {
            return Err((
                self,
                CpuWorkerError::Invalid("foreign NNUEC retirement actor/source"),
            ));
        }
        let RecoveredNnuecEngine {
            source,
            engine,
            value_owner,
            profile,
            stack_owner_id,
        } = self;
        let binding = value_owner.binding_digest();
        let request = value_owner.request_identity();
        let provenance = value_owner.provenance().clone();
        let preparation_epoch = value_owner.preparation_epoch();
        let started = Instant::now();
        drop(engine);
        drop(value_owner);
        let release_time = started.elapsed();
        Ok(NnuecEngineReleaseAck {
            source,
            profile,
            stack_owner_id,
            binding,
            request,
            provenance,
            release_time,
            actor_authority: Arc::clone(invalidations),
            preparation_epoch,
        })
    }
}

impl CpuWorker {
    /// Known join and actual original source are mandatory. Validation happens
    /// before moving anything; a mismatch preserves the same owner in self.
    pub fn take_nnuec_retirement(&mut self) -> Result<RecoveredNnuecEngine, CpuWorkerError> {
        if self.join.is_some() {
            return Err(CpuWorkerError::AlreadyReserved);
        }
        if !self.joined_known {
            return Err(CpuWorkerError::Quarantined);
        }
        let source = self
            .deferred_nnuec_source
            .as_ref()
            .ok_or(CpuWorkerError::Invalid(
                "worker has no original deferred NNUEC source",
            ))?;
        let engine = self
            .recovered_engine
            .as_ref()
            .ok_or(CpuWorkerError::Duplicate)?;
        let value_owner = engine
            .nnuec_retirement_owner()?
            .ok_or(CpuWorkerError::Invalid(
                "actual evaluator has no NNUEC retirement owner",
            ))?;
        let paused = engine
            .paused_stack_snapshot()
            .filter(|value| value.complete && value.owner_id != 0)
            .ok_or(CpuWorkerError::Quarantined)?;
        let plan = source.cpu_plan();
        let actual = engine.config();
        let expected = plan.config();
        if !value_owner.matches_model(source.model())
            || value_owner.model_identity() != source.model().identity()
            || value_owner.binding_digest() != self.evaluation_namespace
            || self.search_conditions != super::cpu_search_conditions_digest(engine)
            || actual.profile != expected.profile
            || actual.tt_entries != expected.tt_entries
            || actual.max_depth != expected.max_depth
            || actual.quiescence_ply != expected.quiescence_ply
            || engine.ordering_policy() != plan.ordering_policy()
            || engine.paused_stack_max_bytes() != plan.paused_bytes_limit()
            || engine.paused_side_scope_reserved_bytes().is_some()
                != plan.preserves_paused_side_scope()
            || paused.tokens_retained != 0
            || paused.bytes_current != 0
            || engine.paused_stack_bytes().is_some()
            || engine.pending_reciprocal_request().is_some()
        {
            return Err(CpuWorkerError::Invalid(
                "joined NNUEC engine/source/profile mismatch",
            ));
        }
        let profile = self.profile();
        let source = source.clone();
        let engine = self
            .recovered_engine
            .take()
            .expect("validated same recovered engine");
        Ok(RecoveredNnuecEngine {
            source,
            engine,
            value_owner,
            profile,
            stack_owner_id: paused.owner_id,
        })
    }
}

/// Issued after this actor physically drops that exact recovered engine and
/// its evaluator owner. It does not assert release of actor cache, pending
/// Work/capture views, or success/reusability of the preceding search.
pub struct NnuecEngineReleaseAck {
    source: DeferredNnuecRSource,
    profile: CpuWorkerProfile,
    stack_owner_id: u64,
    binding: Digest,
    request: RequestIdentity,
    provenance: Provenance,
    release_time: Duration,
    actor_authority: Arc<AtomicU64>,
    preparation_epoch: u64,
}
impl std::fmt::Debug for NnuecEngineReleaseAck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NnuecEngineReleaseAck")
            .field("profile", &self.profile)
            .field("stack_owner_id", &self.stack_owner_id)
            .field("binding", &self.binding)
            .field("release_time", &self.release_time)
            .finish_non_exhaustive()
    }
}
impl NnuecEngineReleaseAck {
    pub(crate) fn matches_request(&self, request: &RequestIdentity) -> bool {
        self.request.same_request(request)
    }
    pub fn matches_source(&self, source: &DeferredNnuecRSource) -> bool {
        self.source.same_authority(source)
    }
    pub fn matches_cache_release(&self, ack: &NnuecCacheReleaseAck) -> bool {
        ack.clears(&self.actor_authority, self.preparation_epoch)
    }
    pub fn preparation_epoch(&self) -> u64 {
        self.preparation_epoch
    }
    pub fn owner(&self) -> CpuWorkerId {
        self.profile.owner
    }
    pub fn profile(&self) -> CpuWorkerProfile {
        self.profile
    }
    pub fn actual_stack_owner_id(&self) -> u64 {
        self.stack_owner_id
    }
    pub fn binding_digest(&self) -> Digest {
        self.binding
    }
    pub fn model_identity(&self) -> &CpuValueIdentity {
        self.source.model().identity()
    }
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }
    pub fn release_time(&self) -> Duration {
        self.release_time
    }
}
