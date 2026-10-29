//! V2 Collection teacher roster and original resource-scope custody.
//!
//! This is not yet a running teacher session. The UCI owner must consume a
//! typed actor/source binding before any worker can be started. In particular,
//! a process permit or matching digest alone does not authorize a teacher go,
//! completion, label, or publication.

use super::v2_actor_preflight::{TeacherActorSourcePreflightV2, preflight_teacher_actor_source_v2};
use super::{ArenaError, check, invalid};
use rz_contracts::Digest;
use rz_experiments::rovent_corpus_v2::RoventCorpusManifestV2;
use rz_search::external_cpu::collection_teacher::TeacherWorkerRegistration;
use rz_search::external_cpu::process_scope::{CollectionProcessPermit, CollectionProcessScope};
use rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2;
use rz_uci::rovent_bootstrap::RoventStartupControl;
use rz_uci::rovent_external_launch::{
    BoundCollectionTeacherActorV2, BoundTeacherSpawnCauseV2, BoundTeacherSpawnFailureV2,
    RoventCollectionTeacherLifecycleV2, RoventCollectionTeacherOwnerV2,
    RoventTeacherRetirementObservationV2, RoventTeacherRetirementTokenV2,
    TeacherActorBindingErrorV2, TeacherRecoveryCustodianV2, TeacherRecoveryKeyV2,
    bind_collection_teacher_actor_v2, spawn_bound_collection_teacher_v2,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

/// An explicit actor-to-original-source slot assignment, not an authority.
pub(crate) struct RoventTeacherRosterEntryV2 {
    pub actor_id: String,
    pub source_index: usize,
    pub collection_owner: u32,
}

struct ActorSourceV2 {
    original: TeacherActorSourcePreflightV2,
    source_index: usize,
    registration: TeacherWorkerRegistration,
    bound: Option<BoundCollectionTeacherActorV2>,
    permit: Option<CollectionProcessPermit>,
    running: Option<RunningTeacherV2>,
    spawn_attempted: bool,
}

struct RunningTeacherV2 {
    lifecycle: RoventCollectionTeacherLifecycleV2,
    // The external custodian receives the same key before worker startup.
    recovery_key: TeacherRecoveryKeyV2,
}

/// Proof for one particular session, not for other sessions that may have
/// been prepared with the same root. Retains all original actor tokens.
#[allow(dead_code)]
pub(crate) struct TeacherSessionRetirementSealV2 {
    session: RoventCollectionTeacherSessionV2,
    actors: BTreeMap<String, RoventTeacherRetirementTokenV2>,
}

#[allow(dead_code)]
impl TeacherSessionRetirementSealV2 {
    pub(crate) fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV2) -> bool {
        self.session.admission.same_original_admission(admission)
    }

    pub(crate) fn actor_count(&self) -> usize {
        self.actors.len()
    }
}

/// A failed seal preserves both the original session and every handed-in
/// token; the caller must not treat the rejected roster as complete.
#[allow(dead_code)]
pub(crate) struct TeacherSessionRetirementSealRejectionV2 {
    pub(crate) session: RoventCollectionTeacherSessionV2,
    pub(crate) actors: BTreeMap<String, RoventTeacherRetirementTokenV2>,
}

/// A started worker remains owned by the session even on `AfterSpawn`.
/// Its recovery key is already registered with the external custodian.
pub(crate) enum TeacherSessionStartErrorV2 {
    Admission(ArenaError),
    Binding(TeacherActorBindingErrorV2),
    BeforeSpawn(BoundTeacherSpawnCauseV2),
    AfterSpawn {
        cause: BoundTeacherSpawnCauseV2,
        recovery_key: TeacherRecoveryKeyV2,
    },
}

/// One prospective session owns the original root, manifest and preflight
/// sources while it exists. The future worker factory must separately retain
/// the actual source custody through physical retirement. Its roster and scope
/// do not prove a loaded worker or actual execution.
#[allow(dead_code)]
pub(crate) struct RoventCollectionTeacherSessionV2 {
    admission: RoventCollectionRootAdmissionV2,
    startup_control: RoventStartupControl,
    manifest: Arc<RoventCorpusManifestV2>,
    manifest_digest: Digest,
    actors: BTreeMap<String, ActorSourceV2>,
    scope: Option<CollectionProcessScope>,
}

#[allow(dead_code)]
impl RoventCollectionTeacherSessionV2 {
    /// Seal retirement only for this exact roster, after scope closure. No
    /// token is fabricated for an unstarted, failed-before-spawn, or unknown
    /// actor. An actor's actual worker registration must match the roster.
    /// This does not prove a different session with the same root is closed.
    pub(crate) fn try_seal_retirement(
        self,
        actors: BTreeMap<String, RoventTeacherRetirementTokenV2>,
    ) -> Result<TeacherSessionRetirementSealV2, Box<TeacherSessionRetirementSealRejectionV2>> {
        let complete = self.scope.is_none()
            && !self.actors.is_empty()
            && actors.len() == self.actors.len()
            && self.actors.iter().all(|(actor_id, actor)| {
                actor.spawn_attempted
                    && actor.running.is_none()
                    && actors.get(actor_id).is_some_and(|token| {
                        token.matches_admission(&self.admission)
                            && token.owner().source_index() == actor.source_index
                            && token.owner().worker().registration() == actor.registration
                    })
            });
        if complete {
            Ok(TeacherSessionRetirementSealV2 { session: self, actors })
        } else {
            Err(Box::new(TeacherSessionRetirementSealRejectionV2 {
                session: self,
                actors,
            }))
        }
    }

    pub(crate) fn prepare(
        admission: &RoventCollectionRootAdmissionV2,
        startup_control: &RoventStartupControl,
        manifest: &Arc<RoventCorpusManifestV2>,
        roster: &[RoventTeacherRosterEntryV2],
    ) -> Result<Self, ArenaError> {
        manifest.validate()?;
        let manifest_sha = manifest.sha256()?;
        let manifest_digest = Digest(
            rz_eval::asset::parse_sha256(&manifest_sha)
                .map_err(|error| invalid(error.to_string()))?,
        );
        let root = admission.root_admission();
        check(
            !roster.is_empty()
                && roster.len() <= 16
                && roster.len() == manifest.teachers.len()
                && root.original_context().accepts()?,
            "V2 teacher roster or live root is invalid",
        )?;
        let root_digest = root.descriptor().digest()?;
        let mut actors = BTreeMap::new();
        let mut indices = BTreeSet::new();
        let mut owners = BTreeSet::new();
        for item in roster {
            check(
                manifest.teachers.contains_key(&item.actor_id)
                    && !actors.contains_key(&item.actor_id)
                    && indices.insert(item.source_index)
                    && item.collection_owner != 0
                    && owners.insert(item.collection_owner),
                "V2 teacher actor, source slot or owner is duplicated or absent",
            )?;
            let original = preflight_teacher_actor_source_v2(
                admission,
                startup_control,
                manifest,
                &item.actor_id,
                item.source_index,
                item.collection_owner,
            )?;
            let registration = TeacherWorkerRegistration {
                collection_owner: original.collection_owner(),
                authority: root.authority(),
                original_deadline: root.hard_deadline(),
                input_lock: root.startup_lock_digest(),
                root_admission: root_digest,
                root_budget: admission.root_budget(),
                frozen_actor: original.actor_digest(),
            };
            actors.insert(
                item.actor_id.clone(),
                ActorSourceV2 {
                    original,
                    source_index: item.source_index,
                    registration,
                    bound: None,
                    permit: None,
                    running: None,
                    spawn_attempted: false,
                },
            );
        }
        let registrations = actors.values().map(|actor| actor.registration).collect();
        let scope = CollectionProcessScope::new(manifest_digest, registrations)
            .map_err(|error| invalid(error.to_string()))?;
        check(
            root.original_context().accepts()?,
            "V2 teacher root closed during roster preparation",
        )?;
        Ok(Self {
            admission: admission.clone(),
            startup_control: startup_control.clone(),
            manifest: Arc::clone(manifest),
            manifest_digest,
            actors,
            scope: Some(scope),
        })
    }

    /// Bind and start exactly one roster member. A failed pre-spawn attempt
    /// retains its one-use permit and original bound actor for inspection;
    /// a post-spawn failure retains the live worker and returns its key. The
    /// external custodian receives the original reservation before spawn.
    /// The permit may already have been consumed even on a pre-spawn failure,
    /// so this session does not retry it. This starts no job and creates no
    /// label or publication authority.
    pub(crate) fn start_actor(
        &mut self,
        actor_id: &str,
        startup_deadline: Instant,
        cleanup_ceiling: Instant,
        custodian: &mut TeacherRecoveryCustodianV2,
    ) -> Result<TeacherRecoveryKeyV2, TeacherSessionStartErrorV2> {
        // A previously allocated permit must not permit a new spawn after
        // this session has closed its process scope for retirement.
        if self.scope.is_none() {
            return Err(TeacherSessionStartErrorV2::Admission(invalid(
                "V2 teacher session admission is closed",
            )));
        }
        self.require_live_root()
            .map_err(TeacherSessionStartErrorV2::Admission)?;
        let actor = self.actors.get_mut(actor_id).ok_or_else(|| {
            TeacherSessionStartErrorV2::Admission(invalid("V2 teacher actor absent"))
        })?;
        if actor.spawn_attempted {
            return Err(TeacherSessionStartErrorV2::Admission(invalid(
                "V2 teacher actor already attempted",
            )));
        }
        if actor.bound.is_none() {
            let bound = bind_collection_teacher_actor_v2(
                &self.admission,
                &self.startup_control,
                &self.manifest,
                actor_id,
                actor.source_index,
                actor.registration.collection_owner,
            )
            .map_err(TeacherSessionStartErrorV2::Binding)?;
            if bound.actor_digest() != actor.registration.frozen_actor
                || bound.source_index() != actor.source_index
                || bound.collection_owner() != actor.registration.collection_owner
            {
                return Err(TeacherSessionStartErrorV2::Admission(invalid(
                    "V2 teacher bound actor differs from original roster",
                )));
            }
            actor.bound = Some(bound);
        }
        if actor.permit.is_none() {
            let scope = self.scope.as_ref().ok_or_else(|| {
                TeacherSessionStartErrorV2::Admission(invalid("V2 teacher scope closed"))
            })?;
            actor.permit = Some(
                scope
                    .permit(self.manifest_digest, actor.registration)
                    .map_err(|error| {
                        TeacherSessionStartErrorV2::Admission(invalid(error.to_string()))
                    })?,
            );
        }
        let bound = actor.bound.take().expect("bound actor retained");
        let permit = actor.permit.as_ref().expect("process permit retained");
        actor.spawn_attempted = true;
        match spawn_bound_collection_teacher_v2(
            bound,
            permit,
            startup_deadline,
            cleanup_ceiling,
            custodian,
        ) {
            Ok((lifecycle, recovery_key)) => {
                actor.running = Some(RunningTeacherV2 {
                    lifecycle,
                    recovery_key: recovery_key.clone(),
                });
                Ok(recovery_key)
            }
            Err(BoundTeacherSpawnFailureV2::BeforeSpawn {
                cause,
                actor: bound,
            }) => {
                actor.bound = Some(bound);
                Err(TeacherSessionStartErrorV2::BeforeSpawn(cause))
            }
            Err(BoundTeacherSpawnFailureV2::AfterSpawn {
                cause,
                lifecycle,
                recovery_key,
            }) => {
                actor.running = Some(RunningTeacherV2 {
                    lifecycle,
                    recovery_key: recovery_key.clone(),
                });
                Err(TeacherSessionStartErrorV2::AfterSpawn {
                    cause,
                    recovery_key,
                })
            }
        }
    }

    /// Access the same started worker; a dispatch or completion still needs
    /// the lifecycle's live-root validation and separate target provenance.
    pub(crate) fn worker_mut(
        &mut self,
        actor_id: &str,
    ) -> Option<&mut RoventCollectionTeacherLifecycleV2> {
        self.actors
            .get_mut(actor_id)?
            .running
            .as_mut()
            .map(|run| &mut run.lifecycle)
    }

    /// Original preflight registration for this roster entry. Callers must
    /// still bind the actual worker, dispatch and completion before labeling.
    pub(crate) fn registered(&self, actor_id: &str) -> Option<&super::RoventRegisteredTeacher> {
        self.actors.get(actor_id).map(|actor| actor.original.registered())
    }

    /// Access a clone of the key already registered with the custodian.
    pub(crate) fn recovery_key(&self, actor_id: &str) -> Option<TeacherRecoveryKeyV2> {
        self.actors
            .get(actor_id)?
            .running
            .as_ref()
            .map(|run| run.recovery_key.clone())
    }

    /// Close the process scope before requesting each original worker's
    /// bounded shutdown. Requests are not physical-completion evidence.
    pub(crate) fn request_shutdown(&mut self, until: Instant) {
        self.close_admission();
        for actor in self.actors.values_mut() {
            if let Some(run) = actor.running.as_mut() {
                run.lifecycle.request_shutdown(until);
            }
        }
    }

    /// Poll only the original actor owner. This can observe the root mutex,
    /// so the caller must not hold a manager or other ownership lock.
    pub(crate) fn poll_actor(
        &mut self,
        actor_id: &str,
    ) -> Option<RoventTeacherRetirementObservationV2> {
        self.worker_mut(actor_id)
            .map(RoventCollectionTeacherLifecycleV2::poll)
    }

    /// Return the same actor owner only when the lifecycle confirms physical
    /// closure, diagnostics and delivery of every submitted terminal. An
    /// unresolved owner is restored to the session so it can be polled again
    /// or parked by the existing recovery path. This does not certify label
    /// consumption or authorize publication.
    pub(crate) fn try_finish_actor(
        &mut self,
        actor_id: &str,
    ) -> Result<Option<(RoventCollectionTeacherOwnerV2, RoventTeacherRetirementObservationV2)>, ArenaError>
    {
        let actor = self
            .actors
            .get_mut(actor_id)
            .ok_or_else(|| invalid("V2 teacher actor absent"))?;
        let Some(run) = actor.running.take() else {
            return Ok(None);
        };
        match run.lifecycle.try_finish() {
            Ok((owner, observation)) => Ok(Some((owner, observation))),
            Err(lifecycle) => {
                actor.running = Some(RunningTeacherV2 {
                    lifecycle,
                    recovery_key: run.recovery_key,
                });
                Ok(None)
            }
        }
    }

    /// Recover an opaque per-actor retirement token through the original
    /// lifecycle. A non-started or unresolved actor never yields a token.
    /// The token alone is not a proof that the entire roster is retired.
    pub(crate) fn try_retire_actor(
        &mut self,
        actor_id: &str,
    ) -> Result<Option<RoventTeacherRetirementTokenV2>, ArenaError> {
        let actor = self
            .actors
            .get_mut(actor_id)
            .ok_or_else(|| invalid("V2 teacher actor absent"))?;
        let Some(run) = actor.running.take() else {
            return Ok(None);
        };
        match run.lifecycle.try_retire() {
            Ok(token) => Ok(Some(token)),
            Err(lifecycle) => {
                actor.running = Some(RunningTeacherV2 {
                    lifecycle,
                    recovery_key: run.recovery_key,
                });
                Ok(None)
            }
        }
    }

    /// The root observation can block on its mutex; callers must not hold a
    /// manager/other owner lock while invoking the roster or permit methods.
    fn require_live_root(&mut self) -> Result<(), ArenaError> {
        match self.admission.root_admission().original_context().accepts() {
            Ok(true) => Ok(()),
            Ok(false) => {
                self.close_admission();
                Err(invalid("V2 teacher root is no longer accepting"))
            }
            Err(error) => {
                self.close_admission();
                Err(ArenaError::Contract(error))
            }
        }
    }

    /// Closes admission, not worker processes. The eventual owner must request
    /// and observe each worker's shutdown and physical drain separately.
    pub(crate) fn close_admission(&mut self) {
        if let Some(scope) = self.scope.take() {
            scope.close_admission();
        }
    }
}
