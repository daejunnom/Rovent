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
    RoventTeacherScopeClaimStageV3, RoventTeacherScopePhysicalAckV3, RoventTeacherScopeWorkerAckV3,
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
use std::sync::{Arc, Mutex, Weak};
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
    spawn_accepted: bool,
    dispatch_claimed: bool,
    effect_identity: Arc<()>,
    effect_source_issued: bool,
    effect_attempted: bool,
}

/// Original session/actor identity only; source issuance is not spawn or
/// physical completion evidence.
pub(crate) struct TeacherActorEffectSourceV3 {
    session_identity: Arc<()>,
    actor_identity: Arc<()>,
    admission: RoventCollectionRootAdmissionV3,
    actor_id: String,
}
pub(crate) struct TeacherActorEffectOutcomeV3 {
    session_identity: Arc<()>,
    actor_identity: Arc<()>,
    admission: RoventCollectionRootAdmissionV3,
    actor_id: String,
}
impl TeacherActorEffectSourceV3 {
    /// Identity observations only; neither grants actor start authority.
    pub(crate) fn session_identity_weak(&self) -> Weak<()> {
        Arc::downgrade(&self.session_identity)
    }
    pub(crate) fn session_identity(&self) -> &Arc<()> {
        &self.session_identity
    }
    pub(crate) fn actor_identity(&self) -> &Arc<()> {
        &self.actor_identity
    }
    pub(crate) fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV3) -> bool {
        self.admission.same_original_admission(admission)
    }
    pub(crate) fn matches_session_identity(&self, identity: &Arc<()>) -> bool {
        Arc::ptr_eq(&self.session_identity, identity)
    }
    pub(crate) fn matches_actor_identity(&self, identity: &Arc<()>) -> bool {
        Arc::ptr_eq(&self.actor_identity, identity)
    }
    pub(crate) fn matches_outcome(&self, outcome: &TeacherActorEffectOutcomeV3) -> bool {
        Arc::ptr_eq(&self.session_identity, &outcome.session_identity)
            && Arc::ptr_eq(&self.actor_identity, &outcome.actor_identity)
            && self.admission.same_original_admission(&outcome.admission)
            && self.actor_id == outcome.actor_id
    }
}

pub(crate) enum TeacherActorEffectFailureV3 {
    Known {
        error: TeacherStartErrorV3,
        outcome: TeacherActorEffectOutcomeV3,
    },
    RejectedBeforeEntry(TeacherStartErrorV3),
    Unknown(Box<dyn std::any::Any + Send>),
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

/// Publish this owner to the original root's recovery slot *before* `claim`.
/// A panic inside UCI's consuming claim is explicitly unknown: this stage
/// cannot recover an issuer that the lower-level call unwound through.
pub(crate) struct TeacherSessionStagingV3 {
    preflight: Option<TeacherSessionPreflightV3>,
    state: TeacherSessionStageStateV3,
    ready: Option<Arc<Mutex<RoventCollectionTeacherSessionV3>>>,
    ready_effect_identity: Option<Weak<()>>,
    panic: Option<Box<dyn std::any::Any + Send>>,
    scope_original_admission: RoventCollectionRootAdmissionV3,
    scope_effect_identity: Arc<()>,
    scope_effect_source_issued: bool,
    scope_effect_attempted: bool,
    scope_preclaim_sealed: bool,
}

/// One original staged teacher scope; this does not attest dispatch receipt,
/// game authority, actor startup, or physical completion.
pub(crate) struct TeacherScopeEffectSourceV3 {
    identity: Arc<()>,
    admission: RoventCollectionRootAdmissionV3,
}

/// The exact staged issuer never entered its original claim and was sealed
/// against future claim. This is not a game abort, process/GPU completion,
/// or the separate one-shot no-scope acknowledgement.
pub(crate) struct TeacherScopeUnclaimedProofV3 {
    identity: Arc<()>,
    admission: RoventCollectionRootAdmissionV3,
}

impl TeacherScopeUnclaimedProofV3 {
    pub(crate) fn matches_original(
        &self,
        stage: &TeacherSessionStagingV3,
        source: &TeacherScopeEffectSourceV3,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> bool {
        Arc::ptr_eq(&self.identity, &stage.scope_effect_identity)
            && Arc::ptr_eq(&self.identity, &source.identity)
            && self.admission.same_original_admission(admission)
            && source.admission.same_original_admission(admission)
            && stage
                .scope_original_admission
                .same_original_admission(admission)
            && stage.scope_preclaim_sealed
            && stage.scope_effect_source_issued
            && !stage.scope_effect_attempted
            && stage.state == TeacherSessionStageStateV3::Prepared
            && stage.ready.is_none()
            && stage.panic.is_none()
            && stage
                .preflight
                .as_ref()
                .is_some_and(|preflight| !preflight.scope.is_claimed())
    }
}

/// Known return from the same staged scope claim. The stage and any acquired
/// scope owner remain in their original recovery slot.
pub(crate) struct TeacherScopeEffectOutcomeV3 {
    identity: Arc<()>,
    admission: RoventCollectionRootAdmissionV3,
    state: TeacherSessionStageStateV3,
}
impl TeacherScopeEffectSourceV3 {
    /// Identity only; source issuance does not prove dispatch receipt.
    pub(crate) fn identity_weak(&self) -> Weak<()> {
        Arc::downgrade(&self.identity)
    }
    /// Retaining this identity alone does not grant a stage or game claim.
    pub(crate) fn identity(&self) -> &Arc<()> {
        &self.identity
    }
    pub(crate) fn matches_stage_identity(&self, identity: &Arc<()>) -> bool {
        Arc::ptr_eq(&self.identity, identity)
    }
    pub(crate) fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV3) -> bool {
        self.admission.same_original_admission(admission)
    }
    pub(crate) fn matches_outcome(&self, outcome: &TeacherScopeEffectOutcomeV3) -> bool {
        Arc::ptr_eq(&self.identity, &outcome.identity)
            && self.admission.same_original_admission(&outcome.admission)
    }
}
impl TeacherScopeEffectOutcomeV3 {
    pub(crate) fn state(&self) -> TeacherSessionStageStateV3 {
        self.state
    }
}

pub(crate) enum TeacherScopeEffectClaimFailureV3 {
    Known {
        error: TeacherSessionClaimErrorV3,
        outcome: TeacherScopeEffectOutcomeV3,
    },
    RejectedBeforeEntry(TeacherSessionClaimErrorV3),
    Unknown(TeacherSessionClaimErrorV3),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TeacherSessionStageStateV3 {
    Prepared,
    ClaimingUnknown,
    BeforeClaimRejected,
    AfterClaimRejected,
    Claimed,
    Ready,
}

struct TeacherSessionPreflightV3 {
    admission: RoventCollectionRootAdmissionV3,
    manifest: Arc<RoventCorpusManifestLockV3>,
    registration: Arc<RoventCorpusManifestV2>,
    manifest_digest: Digest,
    actors: BTreeMap<String, ActorV3>,
    scope: RoventTeacherScopeClaimStageV3,
}

pub(crate) enum TeacherSessionClaimErrorV3 {
    InvalidState,
    Rejected(AdaptiveCollectionError),
    /// The original lower-level claim did not return; no no-scope assertion
    /// or recovery of its consumed opaque issuer may be inferred.
    Unknown,
}

impl TeacherSessionStagingV3 {
    /// The original stage owns this identity before any scope claim.
    pub(crate) fn scope_effect_identity_weak(&self) -> Weak<()> {
        Arc::downgrade(&self.scope_effect_identity)
    }

    /// Identity of the actual session this stage assembled, if Ready. This
    /// does not prove an actor was started or a worker physically completed.
    pub(crate) fn ready_session_effect_identity_weak(&self) -> Option<Weak<()>> {
        self.ready_effect_identity.clone()
    }
    pub(crate) fn state(&self) -> TeacherSessionStageStateV3 {
        self.state
    }

    pub(crate) fn panic_payload(&self) -> Option<&(dyn std::any::Any + Send)> {
        self.panic.as_deref()
    }

    /// The same stage issues this once, only before the real scope claim.
    /// Issuance does not establish that a game dispatch was received.
    pub(crate) fn issue_scope_effect_source(
        &mut self,
        admission: &RoventCollectionRootAdmissionV3,
        destination: &mut Option<TeacherScopeEffectSourceV3>,
    ) -> Result<(), TeacherSessionClaimErrorV3> {
        if self.scope_effect_source_issued
            || self.scope_preclaim_sealed
            || destination.is_some()
            || self.state != TeacherSessionStageStateV3::Prepared
            || !self
                .scope_original_admission
                .same_original_admission(admission)
        {
            return Err(TeacherSessionClaimErrorV3::InvalidState);
        }
        let source = TeacherScopeEffectSourceV3 {
            identity: Arc::clone(&self.scope_effect_identity),
            admission: self.scope_original_admission.clone(),
        };
        self.scope_effect_source_issued = true;
        *destination = Some(source);
        Ok(())
    }

    /// Retain the same stage/source proof directly in the caller's recovery
    /// slot. The real issuer remains in this stage; no no-scope claim occurs.
    pub(crate) fn seal_unclaimed_scope_after_source(
        &mut self,
        source: &TeacherScopeEffectSourceV3,
        destination: &mut Option<TeacherScopeUnclaimedProofV3>,
    ) -> bool {
        if destination.is_some()
            || self.scope_preclaim_sealed
            || !self.scope_effect_source_issued
            || self.scope_effect_attempted
            || self.state != TeacherSessionStageStateV3::Prepared
            || self.ready.is_some()
            || self.panic.is_some()
            || !Arc::ptr_eq(&self.scope_effect_identity, &source.identity)
            || !self
                .scope_original_admission
                .same_original_admission(&source.admission)
            || !self
                .preflight
                .as_ref()
                .is_some_and(|preflight| !preflight.scope.is_claimed())
        {
            return false;
        }
        let proof = TeacherScopeUnclaimedProofV3 {
            identity: Arc::clone(&self.scope_effect_identity),
            admission: self.scope_original_admission.clone(),
        };
        self.scope_preclaim_sealed = true;
        *destination = Some(proof);
        true
    }

    /// The caller must have published this stage before invoking the claim.
    pub(crate) fn claim(&mut self) -> Result<(), TeacherSessionClaimErrorV3> {
        if self.scope_effect_source_issued {
            return Err(TeacherSessionClaimErrorV3::InvalidState);
        }
        self.claim_original()
    }

    /// Once an effect source is issued, only that exact stage source may
    /// enter the original claim. Unknown/unwind never creates an outcome.
    pub(crate) fn claim_with_effect_source<S>(
        &mut self,
        source: &TeacherScopeEffectSourceV3,
        guard: &super::super::game_authority_v3::GameEffectClaimV3<S>,
        destination: &mut Option<
            Result<TeacherScopeEffectOutcomeV3, TeacherScopeEffectClaimFailureV3>,
        >,
    ) -> bool {
        if destination.is_some() || self.scope_effect_attempted {
            return false;
        }
        if !self.scope_effect_source_issued
            || self.scope_preclaim_sealed
            || !Arc::ptr_eq(&self.scope_effect_identity, &source.identity)
            || !self
                .scope_original_admission
                .same_original_admission(&source.admission)
            || !guard.authorizes_teacher_scope(source)
        {
            *destination = Some(Err(TeacherScopeEffectClaimFailureV3::RejectedBeforeEntry(
                TeacherSessionClaimErrorV3::InvalidState,
            )));
            return true;
        }
        // Prepare both retained identities before the actual claim; after it
        // returns no callback or fallible operation precedes the slot write.
        let identity = Arc::clone(&source.identity);
        let admission = source.admission.clone();
        self.scope_effect_attempted = true;
        let original_result = self.claim_original();
        *destination = Some(match original_result {
            Ok(()) => Ok(TeacherScopeEffectOutcomeV3 {
                identity,
                admission,
                state: self.state,
            }),
            Err(error @ TeacherSessionClaimErrorV3::Rejected(_))
                if matches!(
                    self.state,
                    TeacherSessionStageStateV3::BeforeClaimRejected
                        | TeacherSessionStageStateV3::AfterClaimRejected
                ) =>
            {
                Err(TeacherScopeEffectClaimFailureV3::Known {
                    error,
                    outcome: TeacherScopeEffectOutcomeV3 {
                        identity,
                        admission,
                        state: self.state,
                    },
                })
            }
            Err(error @ TeacherSessionClaimErrorV3::InvalidState)
                if self.state != TeacherSessionStageStateV3::ClaimingUnknown =>
            {
                Err(TeacherScopeEffectClaimFailureV3::RejectedBeforeEntry(error))
            }
            Err(error) => Err(TeacherScopeEffectClaimFailureV3::Unknown(error)),
        });
        true
    }

    fn claim_original(&mut self) -> Result<(), TeacherSessionClaimErrorV3> {
        if self.state != TeacherSessionStageStateV3::Prepared || self.scope_preclaim_sealed {
            return Err(TeacherSessionClaimErrorV3::InvalidState);
        }
        let Some(preflight) = self.preflight.as_mut() else {
            return Err(TeacherSessionClaimErrorV3::InvalidState);
        };
        self.state = TeacherSessionStageStateV3::ClaimingUnknown;
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| preflight.scope.claim()));
        match outcome {
            Ok(Ok(())) => {
                self.state = TeacherSessionStageStateV3::Claimed;
                Ok(())
            }
            Ok(Err(error)) => {
                self.state = if preflight.scope.is_claimed() {
                    TeacherSessionStageStateV3::AfterClaimRejected
                } else {
                    TeacherSessionStageStateV3::BeforeClaimRejected
                };
                Err(TeacherSessionClaimErrorV3::Rejected(error))
            }
            Err(payload) => {
                self.panic = Some(payload);
                Err(TeacherSessionClaimErrorV3::Unknown)
            }
        }
    }

    /// Move the already-staged capability into the staged session without
    /// releasing root custody. Caller keeps this stage as the recovery owner.
    pub(crate) fn finish(&mut self) -> Result<(), TeacherSessionClaimErrorV3> {
        if self.state != TeacherSessionStageStateV3::Claimed
            || self.preflight.is_none()
            || !self
                .preflight
                .as_ref()
                .is_some_and(|preflight| preflight.scope.is_claimed())
        {
            return Err(TeacherSessionClaimErrorV3::InvalidState);
        }
        let Some(mut preflight) = self.preflight.take() else {
            return Err(TeacherSessionClaimErrorV3::InvalidState);
        };
        let Some(scope) = preflight.scope.take_claimed_capability() else {
            self.preflight = Some(preflight);
            return Err(TeacherSessionClaimErrorV3::InvalidState);
        };
        let session_identity = Arc::new(());
        self.ready = Some(Arc::new(Mutex::new(RoventCollectionTeacherSessionV3 {
            admission: preflight.admission,
            effect_identity: Arc::clone(&session_identity),
            manifest: preflight.manifest,
            registration: preflight.registration,
            manifest_digest: preflight.manifest_digest,
            scope,
            actors: preflight.actors,
            closing: false,
            closed: false,
            physical: None,
            workers: None,
            aggregate_failure: None,
            aggregate_emitted: false,
        })));
        self.ready_effect_identity = Some(Arc::downgrade(&session_identity));
        self.state = TeacherSessionStageStateV3::Ready;
        Ok(())
    }

    pub(crate) fn session(&self) -> Option<Arc<Mutex<RoventCollectionTeacherSessionV3>>> {
        self.ready.as_ref().map(Arc::clone)
    }
    pub(crate) fn issuer_claimed(&self) -> Option<bool> {
        self.preflight
            .as_ref()
            .map(|preflight| preflight.scope.is_claimed())
    }

    /// Return the exact failed claim's owner for root-scoped recovery. The
    /// caller must retain it; this does not prove physical retirement or
    /// that an unclaimed scope is reusable. An unwind remains unknown.
    pub(crate) fn take_rejected_owner(&mut self) -> Option<TeacherPrepareOwnerV3> {
        let issuer = &mut self.preflight.as_mut()?.scope;
        match self.state {
            TeacherSessionStageStateV3::BeforeClaimRejected => issuer
                .take_unclaimed_scope()
                .map(TeacherPrepareOwnerV3::Unclaimed),
            TeacherSessionStageStateV3::AfterClaimRejected => issuer
                .take_claimed_capability()
                .map(TeacherPrepareOwnerV3::Claimed),
            _ => None,
        }
    }
}

pub(crate) enum TeacherStartErrorV3 {
    Invalid(String),
    BeforeSpawn(BoundTeacherSpawnCauseV3),
    /// The lifecycle is retained inside the session and this same key is
    /// already registered with the caller's recovery custodian.
    AfterSpawn {
        cause: BoundTeacherSpawnCauseV3,
        recovery_key: TeacherRecoveryKeyV3,
    },
}

pub(crate) struct RoventCollectionTeacherSessionV3 {
    admission: RoventCollectionRootAdmissionV3,
    effect_identity: Arc<()>,
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
    /// Identity observation only, while the caller holds this original
    /// session. It is not proof of game dispatch, actor start or retirement.
    pub(crate) fn matches_actor_effect_source(&self, source: &TeacherActorEffectSourceV3) -> bool {
        self.admission.same_original_admission(&source.admission)
            && Arc::ptr_eq(&self.effect_identity, &source.session_identity)
            && self.actors.get(&source.actor_id).is_some_and(|actor| {
                actor.effect_source_issued
                    && Arc::ptr_eq(&actor.effect_identity, &source.actor_identity)
            })
    }
    pub(crate) fn original_admission(&self) -> &RoventCollectionRootAdmissionV3 {
        &self.admission
    }

    pub(crate) fn manifest_lock(&self) -> &Arc<RoventCorpusManifestLockV3> {
        &self.manifest
    }

    pub(crate) fn registration(&self) -> &Arc<RoventCorpusManifestV2> {
        &self.registration
    }

    pub(crate) fn actor_registration(&self, actor_id: &str) -> Option<TeacherWorkerRegistration> {
        self.actors.get(actor_id).map(|actor| actor.registration)
    }

    pub(crate) fn is_closing(&self) -> bool {
        self.closing
    }

    /// Each registered actor has its original accepted spawn owner. This is
    /// not worker readiness, successful inference, or physical retirement.
    pub(crate) fn all_actor_owners_started(&self) -> bool {
        !self.actors.is_empty()
            && self.actors.values().all(|actor| {
                actor.spawn_accepted && actor.lifecycle.is_some() && actor.recovery_key.is_some()
            })
    }

    /// A physical actor has one dispatcher and one sequence namespace for the
    /// whole original root. Dropping a dispatcher does not permit re-issuance.
    pub(crate) fn claim_dispatch_owner(&mut self, actor_id: &str) -> bool {
        let Some(actor) = self.actors.get_mut(actor_id) else {
            return false;
        };
        if self.closing || actor.dispatch_claimed {
            return false;
        }
        actor.dispatch_claimed = true;
        true
    }

    pub(crate) fn prepare_staged(
        admission: &RoventCollectionRootAdmissionV3,
        startup: &RoventStartupControl,
        manifest: &Arc<RoventCorpusManifestLockV3>,
        roster: Vec<TeacherRosterEntryV3>,
    ) -> Result<TeacherSessionStagingV3, TeacherPrepareErrorV3> {
        let lock = admission.native_origin().original_input_lock();
        manifest
            .validate(lock)
            .map_err(|e| TeacherPrepareErrorV3::Invalid(e.to_string()))?;
        let registration = Arc::new(manifest.registration.clone());
        let digest = registration
            .sha256()
            .map_err(|e| TeacherPrepareErrorV3::Invalid(e.to_string()))?;
        let manifest_digest = Digest(
            rz_eval::asset::parse_sha256(&digest)
                .map_err(|e| TeacherPrepareErrorV3::Invalid(e.to_string()))?,
        );
        if roster.is_empty() || roster.len() != registration.teachers.len() {
            return Err(TeacherPrepareErrorV3::Invalid(
                "teacher roster is incomplete".into(),
            ));
        }
        let root = admission.root_admission();
        let root_digest = root
            .descriptor()
            .digest()
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
                return Err(TeacherPrepareErrorV3::Invalid(
                    "duplicate or unknown teacher roster entry".into(),
                ));
            }
            let bound = bind_collection_teacher_actor_v3(
                admission,
                startup,
                &registration,
                &entry.actor_id,
                entry.source_index,
                entry.collection_owner,
            )
            .map_err(TeacherPrepareErrorV3::Binding)?;
            let worker = TeacherWorkerRegistration {
                collection_owner: entry.collection_owner,
                authority: root.authority(),
                original_deadline: root.hard_deadline(),
                input_lock: root.startup_lock_digest(),
                root_admission: root_digest,
                root_budget: admission.root_budget(),
                frozen_actor: bound.actor_digest(),
            };
            actors.insert(
                entry.actor_id,
                ActorV3 {
                    bound: Some(bound),
                    registration: worker,
                    permit: None,
                    lifecycle: None,
                    recovery_key: None,
                    terminal: None,
                    spawn_attempted: false,
                    spawn_accepted: false,
                    dispatch_claimed: false,
                    effect_identity: Arc::new(()),
                    effect_source_issued: false,
                    effect_attempted: false,
                },
            );
        }
        let scope = CollectionProcessScope::new(
            manifest_digest,
            actors.values().map(|a| a.registration).collect(),
        )
        .map_err(|e| TeacherPrepareErrorV3::Invalid(e.to_string()))?;
        Ok(TeacherSessionStagingV3 {
            preflight: Some(TeacherSessionPreflightV3 {
                admission: admission.clone(),
                manifest: Arc::clone(manifest),
                registration,
                manifest_digest,
                actors,
                scope: admission.stage_teacher_scope(scope),
            }),
            state: TeacherSessionStageStateV3::Prepared,
            ready: None,
            ready_effect_identity: None,
            panic: None,
            scope_original_admission: admission.clone(),
            scope_effect_identity: Arc::new(()),
            scope_effect_source_issued: false,
            scope_effect_attempted: false,
            scope_preclaim_sealed: false,
        })
    }

    /// Pins the exact original session/actor in a caller-owned source slot.
    /// Dispatch receipt and a game effect claim remain independent.
    pub(crate) fn issue_actor_effect_source(
        &mut self,
        admission: &RoventCollectionRootAdmissionV3,
        actor_id: &str,
        destination: &mut Option<TeacherActorEffectSourceV3>,
    ) -> Result<(), TeacherStartErrorV3> {
        if destination.is_some()
            || self.closing
            || !self.admission.same_original_admission(admission)
        {
            return Err(TeacherStartErrorV3::Invalid(
                "teacher effect source owner/state differs".into(),
            ));
        }
        let actor = self
            .actors
            .get_mut(actor_id)
            .ok_or_else(|| TeacherStartErrorV3::Invalid("teacher actor absent".into()))?;
        if actor.effect_source_issued || actor.spawn_attempted {
            return Err(TeacherStartErrorV3::Invalid(
                "teacher effect source already issued or attempted".into(),
            ));
        }
        let source = TeacherActorEffectSourceV3 {
            session_identity: Arc::clone(&self.effect_identity),
            actor_identity: Arc::clone(&actor.effect_identity),
            admission: self.admission.clone(),
            actor_id: actor_id.to_owned(),
        };
        actor.effect_source_issued = true;
        *destination = Some(source);
        Ok(())
    }

    pub(crate) fn start_actor(
        &mut self,
        actor_id: &str,
        startup_deadline: Instant,
        cleanup_ceiling: Instant,
        custodian: &mut TeacherRecoveryCustodianV3,
    ) -> Result<TeacherRecoveryKeyV3, TeacherStartErrorV3> {
        if self
            .actors
            .get(actor_id)
            .is_some_and(|actor| actor.effect_source_issued)
        {
            return Err(TeacherStartErrorV3::Invalid(
                "teacher actor effect source requires guarded start".into(),
            ));
        }
        self.start_actor_original(actor_id, startup_deadline, cleanup_ceiling, custodian)
    }

    /// Once a source is issued, invoke the original start only with the same
    /// game claim. Result/error/panic are deposited directly in caller-owned
    /// custody. Unknown never manufactures a start outcome or physical join.
    pub(crate) fn start_actor_with_effect_source<S>(
        &mut self,
        source: &TeacherActorEffectSourceV3,
        guard: &super::super::game_authority_v3::GameEffectClaimV3<S>,
        startup_deadline: Instant,
        cleanup_ceiling: Instant,
        custodian: &mut TeacherRecoveryCustodianV3,
        destination: &mut Option<
            Result<
                (TeacherRecoveryKeyV3, TeacherActorEffectOutcomeV3),
                TeacherActorEffectFailureV3,
            >,
        >,
    ) -> bool {
        if destination.is_some() {
            return false;
        }
        let Some(actor) = self.actors.get(&source.actor_id) else {
            *destination = Some(Err(TeacherActorEffectFailureV3::RejectedBeforeEntry(
                TeacherStartErrorV3::Invalid("teacher actor absent".into()),
            )));
            return true;
        };
        if actor.effect_attempted {
            return false;
        }
        if !actor.effect_source_issued
            || !Arc::ptr_eq(&self.effect_identity, &source.session_identity)
            || !Arc::ptr_eq(&actor.effect_identity, &source.actor_identity)
            || !self.admission.same_original_admission(&source.admission)
            || !guard.authorizes_teacher_actor(source)
        {
            *destination = Some(Err(TeacherActorEffectFailureV3::RejectedBeforeEntry(
                TeacherStartErrorV3::Invalid("teacher actor effect authority differs".into()),
            )));
            return true;
        }
        let outcome = TeacherActorEffectOutcomeV3 {
            session_identity: Arc::clone(&source.session_identity),
            actor_identity: Arc::clone(&source.actor_identity),
            admission: source.admission.clone(),
            actor_id: source.actor_id.clone(),
        };
        self.actors
            .get_mut(&source.actor_id)
            .expect("validated actor")
            .effect_attempted = true;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.start_actor_original(
                &source.actor_id,
                startup_deadline,
                cleanup_ceiling,
                custodian,
            )
        }));
        *destination = Some(match result {
            Ok(Ok(key)) => Ok((key, outcome)),
            Ok(Err(error)) => Err(TeacherActorEffectFailureV3::Known { error, outcome }),
            Err(payload) => Err(TeacherActorEffectFailureV3::Unknown(payload)),
        });
        true
    }

    fn start_actor_original(
        &mut self,
        actor_id: &str,
        startup_deadline: Instant,
        cleanup_ceiling: Instant,
        custodian: &mut TeacherRecoveryCustodianV3,
    ) -> Result<TeacherRecoveryKeyV3, TeacherStartErrorV3> {
        if self.closing {
            return Err(TeacherStartErrorV3::Invalid(
                "teacher admission is closed".into(),
            ));
        }
        let actor = self
            .actors
            .get_mut(actor_id)
            .ok_or_else(|| TeacherStartErrorV3::Invalid("teacher actor absent".into()))?;
        if actor.spawn_attempted {
            return Err(TeacherStartErrorV3::Invalid(
                "teacher actor already attempted".into(),
            ));
        }
        if actor.permit.is_none() {
            actor.permit = Some(
                self.scope
                    .permit(self.manifest_digest, actor.registration)
                    .map_err(|e| TeacherStartErrorV3::Invalid(e.to_string()))?,
            );
        }
        let bound = actor
            .bound
            .take()
            .ok_or_else(|| TeacherStartErrorV3::Invalid("teacher bound actor absent".into()))?;
        actor.spawn_attempted = true;
        match spawn_bound_collection_teacher_v3(
            bound,
            actor.permit.as_ref().expect("retained permit"),
            &self.scope,
            startup_deadline,
            cleanup_ceiling,
            custodian,
        ) {
            Ok((lifecycle, key)) => {
                actor.lifecycle = Some(lifecycle);
                actor.recovery_key = Some(key.clone());
                actor.spawn_accepted = true;
                Ok(key)
            }
            Err(BoundTeacherSpawnFailureV3::BeforeSpawn {
                cause,
                actor: bound,
            }) => {
                actor.bound = Some(bound);
                Err(TeacherStartErrorV3::BeforeSpawn(cause))
            }
            Err(BoundTeacherSpawnFailureV3::AfterSpawn {
                cause,
                lifecycle,
                recovery_key,
            }) => {
                actor.lifecycle = Some(lifecycle);
                actor.recovery_key = Some(recovery_key.clone());
                Err(TeacherStartErrorV3::AfterSpawn {
                    cause,
                    recovery_key,
                })
            }
        }
    }

    pub(crate) fn lifecycle_mut(
        &mut self,
        actor_id: &str,
    ) -> Option<&mut RoventCollectionTeacherLifecycleV3> {
        self.actors.get_mut(actor_id)?.lifecycle.as_mut()
    }

    pub(crate) fn request_shutdown(
        &mut self,
        until: Instant,
    ) -> Result<(), AdaptiveCollectionError> {
        self.closing = true;
        let result = if self.closed {
            Ok(())
        } else {
            self.scope.close()
        };
        if result.is_ok() {
            self.closed = true;
        }
        for actor in self.actors.values_mut() {
            if let Some(worker) = actor.lifecycle.as_mut() {
                worker.request_shutdown(until);
            }
        }
        result
    }

    /// Only after caller drains every completion and observes diagnostics.
    pub(crate) fn finish_worker(&mut self, actor_id: &str) -> Result<bool, &'static str> {
        let actor = self
            .actors
            .get_mut(actor_id)
            .ok_or("teacher actor absent")?;
        if actor.terminal.is_some() {
            return Ok(true);
        }
        let Some(lifecycle) = actor.lifecycle.take() else {
            return Ok(false);
        };
        match lifecycle.try_finish_worker() {
            Ok(token) => {
                actor.terminal = Some(token);
                Ok(true)
            }
            Err(lifecycle) => {
                actor.lifecycle = Some(lifecycle);
                Ok(false)
            }
        }
    }

    /// Each scope acknowledgement is one-shot. Keep any independently
    /// completed evidence here until all workers and both acks are ready.
    /// A failed aggregate retains every original ack and token for retry.
    pub(crate) fn try_aggregate(
        &mut self,
    ) -> Result<Option<RoventTeacherScopeAggregateAckV3>, String> {
        if self.aggregate_emitted {
            return Err("teacher aggregate already emitted".into());
        }
        if !self.closed || self.actors.values().any(|actor| actor.lifecycle.is_some()) {
            return Ok(None);
        }
        if let Some(failure) = self.aggregate_failure.take() {
            return match failure.retry() {
                Ok(aggregate) => {
                    self.aggregate_emitted = true;
                    Ok(Some(aggregate))
                }
                Err(failure) => {
                    self.aggregate_failure = Some(failure);
                    Ok(None)
                }
            };
        }
        if self.physical.is_none() {
            self.physical = self
                .scope
                .try_physical_retirement()
                .map_err(|e| e.to_string())?;
        }
        if self.workers.is_none() {
            self.workers = self
                .scope
                .try_worker_retirement()
                .map_err(|e| e.to_string())?;
        }
        if self.physical.is_none() || self.workers.is_none() {
            return Ok(None);
        }
        let physical = self.physical.take().expect("physical ack retained");
        let workers = self.workers.take().expect("worker ack retained");
        let tokens = self
            .actors
            .values_mut()
            .filter_map(|actor| actor.terminal.take())
            .collect();
        match RoventTeacherScopeAggregateAckV3::try_build(physical, Some(workers), tokens) {
            Ok(aggregate) => {
                self.aggregate_emitted = true;
                Ok(Some(aggregate))
            }
            Err(failure) => {
                self.aggregate_failure = Some(failure);
                Ok(None)
            }
        }
    }
}
