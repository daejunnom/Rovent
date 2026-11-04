//! Same-actor release of a unique materialized engine before worker creation.
//!
//! No worker, Ready gate, Job, join or native execution is fabricated here.
//! The original materialization is returned unchanged on a foreign owner or
//! namespace failure, and only its preparation actor destroys the actual TT.

use super::{DeferredNnuecRSource, MaterializedNnuecEngine};
use crate::cpu_value::CpuValueIdentity;
use crate::cpu_value::nnuec::v2::Model;
use crate::cpu_value::nnuec::v2::retirement::NnuecCacheReleaseAck;
use crate::cpu_value::nnuec::v2::review::{Provenance, RequestIdentity};
use crate::cpu_worker::CpuWorkerError;
use rz_contracts::Digest;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64},
};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct NnuecMaterializationReleaseId {
    authority: Arc<()>,
    sequence: u64,
}
impl PartialEq for NnuecMaterializationReleaseId {
    fn eq(&self, other: &Self) -> bool {
        self.sequence == other.sequence && Arc::ptr_eq(&self.authority, &other.authority)
    }
}
impl Eq for NnuecMaterializationReleaseId {}
impl NnuecMaterializationReleaseId {
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
pub struct NnuecMaterializationReleaseCompletion {
    id: NnuecMaterializationReleaseId,
    result: Result<NnuecMaterializationReleaseAck, (MaterializedNnuecEngine, CpuWorkerError)>,
}
impl NnuecMaterializationReleaseCompletion {
    pub(crate) fn new(
        id: NnuecMaterializationReleaseId,
        result: Result<NnuecMaterializationReleaseAck, (MaterializedNnuecEngine, CpuWorkerError)>,
    ) -> Self {
        Self { id, result }
    }
    pub fn request_id(&self) -> &NnuecMaterializationReleaseId {
        &self.id
    }
    #[allow(clippy::result_large_err)]
    pub fn into_result(
        self,
    ) -> Result<NnuecMaterializationReleaseAck, (MaterializedNnuecEngine, CpuWorkerError)> {
        self.result
    }
}

pub struct NnuecMaterializationReleaseAck {
    source: DeferredNnuecRSource,
    binding: Digest,
    request: RequestIdentity,
    provenance: Provenance,
    actor_authority: Arc<AtomicU64>,
    preparation_epoch: u64,
    release_time: Duration,
}
impl std::fmt::Debug for NnuecMaterializationReleaseAck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NnuecMaterializationReleaseAck")
            .field("binding", &self.binding)
            .field("preparation_epoch", &self.preparation_epoch)
            .field("release_time", &self.release_time)
            .finish_non_exhaustive()
    }
}
impl NnuecMaterializationReleaseAck {
    pub(crate) fn matches_request(&self, request: &RequestIdentity) -> bool {
        self.request.same_request(request)
    }
    pub fn matches_source(&self, source: &DeferredNnuecRSource) -> bool {
        self.source.same_authority(source)
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
    pub fn preparation_epoch(&self) -> u64 {
        self.preparation_epoch
    }
    pub fn release_time(&self) -> Duration {
        self.release_time
    }
    pub fn matches_cache_release(&self, ack: &NnuecCacheReleaseAck) -> bool {
        ack.clears(&self.actor_authority, self.preparation_epoch)
    }
}

impl MaterializedNnuecEngine {
    #[allow(clippy::result_large_err)]
    pub(crate) fn release_on_actor(
        self,
        model: &Arc<Model>,
        stopping: &Arc<AtomicBool>,
        invalidations: &Arc<AtomicU64>,
    ) -> Result<NnuecMaterializationReleaseAck, (Self, CpuWorkerError)> {
        let value = match self.engine.nnuec_retirement_owner() {
            Ok(Some(owner)) => owner,
            Ok(None) => {
                return Err((
                    self,
                    CpuWorkerError::Invalid("materialization lost its V2 owner"),
                ));
            }
            Err(error) => return Err((self, error.into())),
        };
        if !Arc::ptr_eq(&self.declaration.model, model)
            || !self
                .bound
                .matches_preparation_actor(stopping, invalidations)
            || !value.matches_model(model)
            || !value.matches_bound(&self.bound)
            || value.binding_digest() != self.binding
            || value.model_identity() != model.identity()
        {
            return Err((
                self,
                CpuWorkerError::Invalid("foreign unstarted NNUEC materialization"),
            ));
        }
        let Self {
            declaration,
            engine,
            bound,
            binding,
        } = self;
        let request = bound.request_identity();
        let provenance = bound.provenance().clone();
        let preparation_epoch = bound.preparation_epoch();
        let started = Instant::now();
        drop(engine);
        drop(bound);
        drop(value);
        Ok(NnuecMaterializationReleaseAck {
            source: DeferredNnuecRSource { declaration },
            binding,
            request,
            provenance,
            actor_authority: Arc::clone(invalidations),
            preparation_epoch,
            release_time: started.elapsed(),
        })
    }
}
