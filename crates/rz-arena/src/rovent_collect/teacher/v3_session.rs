//! Root-bound V3 teacher roster and physical worker custody.
//!
//! This is a session building block, not a corpus label or publication. The
//! caller must independently retain and verify the registered manifest file;
//! an in-memory manifest is not evidence of original descriptor custody.

use rz_contracts::Digest;
use rz_experiments::rovent_corpus_lock_v3::record::RoventCorpusManifestLockV3;
use rz_experiments::rovent_corpus_v2::RoventCorpusManifestV2;
use rz_search::external_cpu::collection_teacher::TeacherWorkerRegistration;
use rz_search::external_cpu::process_scope::{CollectionProcessPermit, CollectionProcessScope};
use rz_uci::pals_adaptive::{
    AdaptiveCollectionError, RoventCollectionRootAdmissionV3, RoventTeacherScopeCapabilityV3,
    RoventTeacherScopeClaimFailureV3, RoventTeacherScopePhysicalAckV3,
    RoventTeacherScopeWorkerAckV3,
};
use rz_uci::rovent_bootstrap::RoventStartupControl;
use rz_uci::rovent_external_launch::{
    BoundCollectionTeacherActorV3, BoundTeacherSpawnCauseV3, BoundTeacherSpawnFailureV3,
    RoventCollectionTeacherLifecycleV3, RoventTeacherScopeAggregateAckV3,
    RoventTeacherScopeAggregateFailureV3, RoventTeacherWorkerTerminalTokenV3,
    TeacherActorBindingErrorV3, TeacherRecoveryCustodianV3, TeacherRecoveryKeyV3,
    bind_collection_teacher_actor_v3, spawn_bound_collection_teacher_v3,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

pub(crate) struct TeacherRosterEntryV3 {
    pub actor_id: String,
    pub source_index: usize,
    pub collection_owner: u32,
}

struct ActorV3 {
    bound: Option<BoundCollectionTeacherActorV3>,
    registration: TeacherWorkerRegistration,
    permit: Option<CollectionProcessPermit>,
    lifecycle: Option<RoventCollectionTeacherLifecycleV3>,
    recovery_key: Option<TeacherRecoveryKeyV3>,
    terminal: Option<RoventTeacherWorkerTerminalTokenV3>,
    spawn_attempted: bool,
}

/// A failed one-shot claim returns the original scope or capability.
pub(crate) enum TeacherPrepareOwnerV3 {
    Unclaimed(CollectionProcessScope),
    Claimed(RoventTeacherScopeCapabilityV3),
}

pub(crate) enum TeacherPrepareErrorV3 {
    Invalid(String),
    Binding(TeacherActorBindingErrorV3),
    Claim {
        error: AdaptiveCollectionError,
        owner: TeacherPrepareOwnerV3,
    },
}

pub(crate) enum TeacherStartErrorV3 {
    Invalid(String),
    BeforeSpawn(BoundTeacherSpawnCauseV3),
    /// The lifecycle is retained inside the session and this same key is
    /// already registered with the caller's recovery custodian.
    AfterSpawn { cause: BoundTeacherSpawnCauseV3, recovery_key: TeacherRecoveryKeyV3 },
}

pub(crate) struct RoventCollectionTeacherSessionV3 {
    admission: RoventCollectionRootAdmissionV3,
    // Retain the same caller-provided owner throughout the session. This does
    // not attest that the registration was loaded from a pinned file.
    manifest: Arc<RoventCorpusManifestLockV3>,
    registration: Arc<RoventCorpusManifestV2>,
    manifest_digest: Digest,
    scope: RoventTeacherScopeCapabilityV3,
    actors: BTreeMap<String, ActorV3>,
    closing: bool,
    closed: bool,
    physical: Option<RoventTeacherScopePhysicalAckV3>,
    workers: Option<RoventTeacherScopeWorkerAckV3>,
    aggregate_failure: Option<RoventTeacherScopeAggregateFailureV3>,
    aggregate_emitted: bool,
}

impl RoventCollectionTeacherSessionV3 {
    pub(crate) fn prepare(
        admission: &RoventCollectionRootAdmissionV3,
        startup: &RoventStartupControl,
        manifest: &Arc<RoventCorpusManifestLockV3>,
        roster: Vec<TeacherRosterEntryV3>,
    ) -> Result<Self, TeacherPrepareErrorV3> {
        let lock = admission.native_origin().original_input_lock();
        manifest.validate(lock).map_err(|e| TeacherPrepareErrorV3::Invalid(e.to_string()))?;
        let registration = Arc::new(manifest.registration.clone());
        let digest = registration.sha256().map_err(|e| TeacherPrepareErrorV3::Invalid(e.to_string()))?;
        let manifest_digest = Digest(rz_eval::asset::parse_sha256(&digest)
            .map_err(|e| TeacherPrepareErrorV3::Invalid(e.to_string()))?);
        if roster.is_empty() || roster.len() != registration.teachers.len() {
            return Err(TeacherPrepareErrorV3::Invalid("teacher roster is incomplete".into()));
        }
        let root = admission.root_admission();
        let root_digest = root.descriptor().digest()
            .map_err(|e| TeacherPrepareErrorV3::Invalid(e.to_string()))?;
        let mut actors = BTreeMap::new();
        let mut indices = BTreeSet::new();
        let mut owners = BTreeSet::new();
        for entry in roster {
            if !registration.teachers.contains_key(&entry.actor_id)
                || actors.contains_key(&entry.actor_id)
                || !indices.insert(entry.source_index)
                || entry.collection_owner == 0
                || !owners.insert(entry.collection_owner)
            {
                return Err(TeacherPrepareErrorV3::Invalid("duplicate or unknown teacher roster entry".into()));
            }
            let bound = bind_collection_teacher_actor_v3(
                admission, startup, &registration, &entry.actor_id,
                entry.source_index, entry.collection_owner,
            ).map_err(TeacherPrepareErrorV3::Binding)?;
            let worker = TeacherWorkerRegistration {
                collection_owner: entry.collection_owner,
                authority: root.authority(),
                original_deadline: root.hard_deadline(),
                input_lock: root.startup_lock_digest(),
                root_admission: root_digest,
                root_budget: admission.root_budget(),
                frozen_actor: bound.actor_digest(),
            };
            actors.insert(entry.actor_id, ActorV3 {
                bound: Some(bound), registration: worker, permit: None,
                lifecycle: None, recovery_key: None, terminal: None,
                spawn_attempted: false,
            });
        }
        let scope = CollectionProcessScope::new(manifest_digest, actors.values().map(|a| a.registration).collect())
            .map_err(|e| TeacherPrepareErrorV3::Invalid(e.to_string()))?;
        let scope = admission.claim_teacher_scope(scope).map_err(|failure| match failure {
            RoventTeacherScopeClaimFailureV3::BeforeClaim { error, scope } => TeacherPrepareErrorV3::Claim {
                error, owner: TeacherPrepareOwnerV3::Unclaimed(scope),
            },
            RoventTeacherScopeClaimFailureV3::AfterClaim { error, capability } => TeacherPrepareErrorV3::Claim {
                error, owner: TeacherPrepareOwnerV3::Claimed(capability),
            },
        })?;
        Ok(Self { admission: admission.clone(), manifest: Arc::clone(manifest), registration,
            manifest_digest, scope, actors, closing: false, closed: false,
            physical: None, workers: None, aggregate_failure: None, aggregate_emitted: false })
    }

    pub(crate) fn start_actor(
        &mut self, actor_id: &str, startup_deadline: Instant, cleanup_ceiling: Instant,
        custodian: &mut TeacherRecoveryCustodianV3,
    ) -> Result<TeacherRecoveryKeyV3, TeacherStartErrorV3> {
        if self.closing { return Err(TeacherStartErrorV3::Invalid("teacher admission is closed".into())); }
        let actor = self.actors.get_mut(actor_id)
            .ok_or_else(|| TeacherStartErrorV3::Invalid("teacher actor absent".into()))?;
        if actor.spawn_attempted { return Err(TeacherStartErrorV3::Invalid("teacher actor already attempted".into())); }
        if actor.permit.is_none() {
            actor.permit = Some(self.scope.permit(self.manifest_digest, actor.registration)
                .map_err(|e| TeacherStartErrorV3::Invalid(e.to_string()))?);
        }
        let bound = actor.bound.take().ok_or_else(|| TeacherStartErrorV3::Invalid("teacher bound actor absent".into()))?;
        actor.spawn_attempted = true;
        match spawn_bound_collection_teacher_v3(bound, actor.permit.as_ref().expect("retained permit"),
            &self.scope, startup_deadline, cleanup_ceiling, custodian) {
            Ok((lifecycle, key)) => {
                actor.lifecycle = Some(lifecycle);
                actor.recovery_key = Some(key.clone());
                Ok(key)
            }
            Err(BoundTeacherSpawnFailureV3::BeforeSpawn { cause, actor: bound }) => {
                actor.bound = Some(bound);
                Err(TeacherStartErrorV3::BeforeSpawn(cause))
            }
            Err(BoundTeacherSpawnFailureV3::AfterSpawn { cause, lifecycle, recovery_key }) => {
                actor.lifecycle = Some(lifecycle);
                actor.recovery_key = Some(recovery_key.clone());
                Err(TeacherStartErrorV3::AfterSpawn { cause, recovery_key })
            }
        }
    }

    pub(crate) fn lifecycle_mut(&mut self, actor_id: &str) -> Option<&mut RoventCollectionTeacherLifecycleV3> {
        self.actors.get_mut(actor_id)?.lifecycle.as_mut()
    }

    pub(crate) fn request_shutdown(&mut self, until: Instant) -> Result<(), AdaptiveCollectionError> {
        self.closing = true;
        let result = if self.closed { Ok(()) } else { self.scope.close() };
        if result.is_ok() { self.closed = true; }
        for actor in self.actors.values_mut() {
            if let Some(worker) = actor.lifecycle.as_mut() { worker.request_shutdown(until); }
        }
        result
    }

    /// Only after caller drains every completion and observes diagnostics.
    pub(crate) fn finish_worker(&mut self, actor_id: &str) -> Result<bool, &'static str> {
        let actor = self.actors.get_mut(actor_id).ok_or("teacher actor absent")?;
        if actor.terminal.is_some() { return Ok(true); }
        let Some(lifecycle) = actor.lifecycle.take() else { return Ok(false); };
        match lifecycle.try_finish_worker() {
            Ok(token) => { actor.terminal = Some(token); Ok(true) }
            Err(lifecycle) => { actor.lifecycle = Some(lifecycle); Ok(false) }
        }
    }

    /// Each scope acknowledgement is one-shot. Keep any independently
    /// completed evidence here until all workers and both acks are ready.
    /// A failed aggregate retains every original ack and token for retry.
    pub(crate) fn try_aggregate(&mut self) -> Result<Option<RoventTeacherScopeAggregateAckV3>, String> {
        if self.aggregate_emitted { return Err("teacher aggregate already emitted".into()); }
        if !self.closed || self.actors.values().any(|actor| actor.lifecycle.is_some()) {
            return Ok(None);
        }
        if let Some(failure) = self.aggregate_failure.take() {
            return match failure.retry() {
                Ok(aggregate) => { self.aggregate_emitted = true; Ok(Some(aggregate)) }
                Err(failure) => { self.aggregate_failure = Some(failure); Ok(None) }
            };
        }
        if self.physical.is_none() {
            self.physical = self.scope.try_physical_retirement().map_err(|e| e.to_string())?;
        }
        if self.workers.is_none() {
            self.workers = self.scope.try_worker_retirement().map_err(|e| e.to_string())?;
        }
        if self.physical.is_none() || self.workers.is_none() { return Ok(None); }
        let physical = self.physical.take().expect("physical ack retained");
        let workers = self.workers.take().expect("worker ack retained");
        let tokens = self.actors.values_mut().filter_map(|actor| actor.terminal.take()).collect();
        match RoventTeacherScopeAggregateAckV3::try_build(physical, Some(workers), tokens) {
            Ok(aggregate) => { self.aggregate_emitted = true; Ok(Some(aggregate)) }
            Err(failure) => { self.aggregate_failure = Some(failure); Ok(None) }
        }
    }
}
