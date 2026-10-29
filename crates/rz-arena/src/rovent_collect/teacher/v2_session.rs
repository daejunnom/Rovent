//! V2 Collection teacher roster and original resource-scope custody.
//!
//! This is not yet a running teacher session. The UCI owner must consume a
//! typed actor/source binding before any worker can be started. In particular,
//! a process permit or matching digest alone does not authorize a teacher go,
//! completion, label, or publication.

use super::super::{exact_rules_anchor, RoventRetiredLockV2Record, RoventSubmittedInputSeal};
use super::v2_actor_preflight::{preflight_teacher_actor_source_v2, TeacherActorSourcePreflightV2};
use super::{check, invalid, ArenaError};
use rz_contracts::pals::Role;
use rz_contracts::Digest;
use rz_experiments::pals_data_v3::GenerationRoleV3;
use rz_experiments::rovent_corpus::MAX_CORPUS_ROWS;
use rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2;
use rz_experiments::rovent_corpus_v2::RoventCorpusManifestV2;
use rz_search::external_cpu::collection_teacher::{
    TeacherCompletion, TeacherDispatch, TeacherInputScope, TeacherJob, TeacherJobId,
    TeacherWorkerRegistration,
};
use rz_search::external_cpu::process_scope::{CollectionProcessPermit, CollectionProcessScope};
use rz_search::external_cpu::ExternalGoConstraint;
use rz_uci::pals_adaptive::{
    AdaptiveCollectionError, RoventCollectionRootAdmissionV2, RoventTeacherScopeCapabilityV2,
    RoventTeacherScopeClaimFailureV2, RoventTeacherScopePhysicalAckV2,
    RoventTeacherScopeWorkerAckV2,
};
use rz_uci::rovent_bootstrap::RoventStartupControl;
use rz_uci::rovent_external_launch::{
    bind_collection_teacher_actor_v2, spawn_bound_collection_teacher_v2,
    BoundCollectionTeacherActorV2, BoundTeacherSpawnCauseV2, BoundTeacherSpawnFailureV2,
    RoventCollectionTeacherLifecycleV2, RoventCollectionTeacherOwnerV2,
    RoventTeacherPolicyCustodyV2, RoventTeacherPolicyEvidenceV2, RoventTeacherPolicySealErrorV2,
    RoventTeacherPolicySealFailureKindV2, RoventTeacherRetirementObservationV2,
    RoventTeacherRetirementTokenV2, RoventTeacherScopeAggregateAckV2,
    RoventTeacherWorkerTerminalTokenV2, TeacherActorBindingErrorV2, TeacherRecoveryCustodianV2,
    TeacherRecoveryKeyV2, VerifiedRoventExternalStartup,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

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
    worker_state: ActorWorkerStateV2,
    next_sequence: u64,
    pending: Option<PendingTeacherV2>,
}

/// Session-owned history of the actual spawn branch. This records neither
/// OS child state nor an aggregate proof; a started worker needs its original
/// opaque join/terminal token and the scope member outcome separately.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ActorWorkerStateV2 {
    NotAttempted,
    SpawnInProgress,
    BeforeSpawn,
    Started,
    Joined,
    Retired,
}

struct PendingTeacherV2 {
    actor_id: String,
    input: RoventSubmittedInputSeal,
    registered: super::RoventRegisteredTeacher,
    startup: VerifiedRoventExternalStartup,
    soft_quantum: Duration,
    dispatch: TeacherDispatch,
}

/// The caller keeps the original row and decides whether to re-observe the
/// same bounded deadline. A pending state is not a submitted teacher job.
pub(crate) enum TeacherSubmitOutcomeV2 {
    Submitted(TeacherJobId),
    Pending(TeacherSubmitPendingV2),
}

pub(crate) enum TeacherSubmitPendingV2 {
    Startup,
    ActorBusy,
}

/// The original submitted input and physical completion remain together.
/// This is not a verified teacher target or publication authority.
pub(crate) struct CompletedTeacherV2 {
    pub(crate) actor_id: String,
    pub(crate) input: RoventSubmittedInputSeal,
    pub(crate) registered: super::RoventRegisteredTeacher,
    pub(crate) startup: VerifiedRoventExternalStartup,
    pub(crate) soft_quantum: Duration,
    pub(crate) dispatch: TeacherDispatch,
    pub(crate) completion: TeacherCompletion,
}

/// This preserves the originally submitted metadata and the non-cloneable
/// UCI authority witness. It is not itself a valid policy target.
pub(crate) struct SealedTeacherPolicyV2 {
    pub(crate) manifest_owner: Arc<RoventCorpusManifestLockV2>,
    pub(crate) actor_id: String,
    pub(crate) input: RoventSubmittedInputSeal,
    pub(crate) registered: super::RoventRegisteredTeacher,
    pub(crate) startup: VerifiedRoventExternalStartup,
    pub(crate) soft_quantum: Duration,
    pub(crate) evidence: RoventTeacherPolicyEvidenceV2,
}

/// A panic is a distinct unknown outcome, never a transient Busy retry.
pub(crate) enum TeacherPolicySealFailureV2 {
    Validation(ArenaError),
    Native {
        kind: RoventTeacherPolicySealFailureKindV2,
        error: rz_search::cpu_checker::CheckerError,
    },
    Panicked(Box<dyn std::any::Any + Send>),
}

pub(crate) struct TeacherPolicyQuarantineV2 {
    pub(crate) manifest_owner: Arc<RoventCorpusManifestLockV2>,
    pub(crate) actor_id: String,
    pub(crate) input: RoventSubmittedInputSeal,
    pub(crate) registered: super::RoventRegisteredTeacher,
    pub(crate) startup: VerifiedRoventExternalStartup,
    pub(crate) soft_quantum: Duration,
    pub(crate) custody: RoventTeacherPolicyCustodyV2,
    pub(crate) panic: Option<Box<dyn std::any::Any + Send>>,
    pub(crate) native_failure: Option<RoventTeacherPolicySealErrorV2>,
}

pub(crate) enum TeacherPolicySealRejectionV2 {
    Rejected {
        cause: TeacherPolicySealFailureV2,
        completed: CompletedTeacherV2,
    },
    Quarantined(TeacherPolicyQuarantineV2),
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
    pub(crate) fn try_physical_retirement(
        &self,
    ) -> Result<Option<RoventTeacherScopePhysicalAckV2>, AdaptiveCollectionError> {
        self.session.try_physical_retirement()
    }
    pub(crate) fn try_worker_retirement(
        &self,
    ) -> Result<Option<RoventTeacherScopeWorkerAckV2>, AdaptiveCollectionError> {
        self.session.try_worker_retirement()
    }
    pub(crate) fn release_retired_permits(
        &mut self,
        aggregate: &RoventTeacherScopeAggregateAckV2,
    ) -> Result<(), ArenaError> {
        self.session.release_retired_permits(aggregate)
    }
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

/// The original resource survives a rejected one-shot claim. An unclaimed
/// issuer has never started an actor; an already claimed capability retains
/// the only scope able to issue physical retirement evidence for that root.
pub(crate) enum TeacherSessionPrepareOwnerV2 {
    Unclaimed(CollectionProcessScope),
    Claimed(RoventTeacherScopeCapabilityV2),
}

pub(crate) struct TeacherSessionPrepareFailureV2 {
    pub(crate) error: ArenaError,
    pub(crate) owner: Option<Box<TeacherSessionPrepareOwnerV2>>,
}

impl From<ArenaError> for TeacherSessionPrepareFailureV2 {
    fn from(error: ArenaError) -> Self {
        Self { error, owner: None }
    }
}
impl From<rz_contracts::ContractError> for TeacherSessionPrepareFailureV2 {
    fn from(error: rz_contracts::ContractError) -> Self {
        ArenaError::from(error).into()
    }
}
impl From<rz_experiments::ManifestError> for TeacherSessionPrepareFailureV2 {
    fn from(error: rz_experiments::ManifestError) -> Self {
        ArenaError::from(error).into()
    }
}

/// One prospective session owns the original root, manifest and preflight
/// sources while it exists. The future worker factory must separately retain
/// the actual source custody through physical retirement. Its roster and scope
/// do not prove a loaded worker or actual execution.
#[allow(dead_code)]
pub(crate) struct RoventCollectionTeacherSessionV2 {
    pump_owner: Arc<()>,
    pump_claimed: bool,
    admission: RoventCollectionRootAdmissionV2,
    startup_control: RoventStartupControl,
    lock_manifest: Arc<RoventCorpusManifestLockV2>,
    lock_manifest_sha256: String,
    manifest: Arc<RoventCorpusManifestV2>,
    manifest_digest: Digest,
    actors: BTreeMap<String, ActorSourceV2>,
    root_scope: RoventTeacherScopeCapabilityV2,
    admission_closing: bool,
    root_scope_closed: bool,
    issued_inputs: BTreeSet<(u32, rz_contracts::RequestId)>,
}

/// Non-cloneable authority for one session's collection data plane.
pub(crate) struct TeacherPumpPermitV2 {
    owner: Arc<()>,
}

#[allow(dead_code)]
impl RoventCollectionTeacherSessionV2 {
    /// Historical worker startup observation, not a current liveness or
    /// teacher-label authority. The actual worker owns the report.
    pub(crate) fn actor_startup_report(
        &self,
        actor_id: &str,
    ) -> Option<rz_search::external_cpu::worker::ExternalWorkerStartup> {
        self.actors.get(actor_id)?.running.as_ref()?.lifecycle.owner().worker().startup_report()
    }
    /// One pump per original session. Dropping the token is not retirement.
    pub(crate) fn claim_pump_owner(&mut self) -> Result<TeacherPumpPermitV2, ArenaError> {
        check(
            !self.pump_claimed,
            "V2 teacher session pump already claimed",
        )?;
        self.pump_claimed = true;
        Ok(TeacherPumpPermitV2 {
            owner: Arc::clone(&self.pump_owner),
        })
    }
    pub(crate) fn validate_pump_owner(
        &self,
        permit: &TeacherPumpPermitV2,
    ) -> Result<(), ArenaError> {
        check(
            self.pump_claimed && Arc::ptr_eq(&self.pump_owner, &permit.owner),
            "V2 teacher pump permit belongs to another session",
        )
    }
    pub(crate) fn original_admission(&self) -> &RoventCollectionRootAdmissionV2 {
        &self.admission
    }
    /// Seal retirement only for this exact roster, after scope closure. No
    /// token is fabricated for an unstarted, failed-before-spawn, or unknown
    /// actor. An actor's actual worker registration must match the roster.
    /// This does not prove a different session with the same root is closed.
    pub(crate) fn try_seal_retirement(
        self,
        actors: BTreeMap<String, RoventTeacherRetirementTokenV2>,
    ) -> Result<TeacherSessionRetirementSealV2, Box<TeacherSessionRetirementSealRejectionV2>> {
        let complete = self.root_scope_closed
            && !self.actors.is_empty()
            && actors.len() == self.actors.len()
            && self.actors.iter().all(|(actor_id, actor)| {
                actor.spawn_attempted
                    && actor.running.is_none()
                    && actor.pending.is_none()
                    && actors.get(actor_id).is_some_and(|token| {
                        token.matches_admission(&self.admission)
                            && token.owner().source_index() == actor.source_index
                            && token.owner().worker().registration() == actor.registration
                    })
            });
        if complete {
            Ok(TeacherSessionRetirementSealV2 {
                session: self,
                actors,
            })
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
        lock_manifest: &Arc<RoventCorpusManifestLockV2>,
        roster: &[RoventTeacherRosterEntryV2],
    ) -> Result<Self, TeacherSessionPrepareFailureV2> {
        let lock = admission.native_origin().original_input_lock();
        lock_manifest.validate(lock)?;
        let lock_manifest_sha256 = lock_manifest.sha256(lock)?;
        let manifest = Arc::new(lock_manifest.registration.clone());
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
                &manifest,
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
                    worker_state: ActorWorkerStateV2::NotAttempted,
                    next_sequence: 1,
                    pending: None,
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
        // Claim only after all reversible roster/source/scope preparation has
        // succeeded. A claim is never reissued after failure or unknown work.
        let root_scope = admission.claim_teacher_scope(scope).map_err(|failure| {
            let (error, owner) = match failure {
                RoventTeacherScopeClaimFailureV2::BeforeClaim { error, scope } => {
                    (error, TeacherSessionPrepareOwnerV2::Unclaimed(scope))
                }
                RoventTeacherScopeClaimFailureV2::AfterClaim { error, capability } => {
                    (error, TeacherSessionPrepareOwnerV2::Claimed(capability))
                }
            };
            TeacherSessionPrepareFailureV2 {
                error: invalid(error.to_string()),
                owner: Some(Box::new(owner)),
            }
        })?;
        Ok(Self {
            pump_owner: Arc::new(()),
            pump_claimed: false,
            admission: admission.clone(),
            startup_control: startup_control.clone(),
            lock_manifest: Arc::clone(lock_manifest),
            lock_manifest_sha256,
            manifest,
            manifest_digest,
            actors,
            root_scope,
            admission_closing: false,
            root_scope_closed: false,
            issued_inputs: BTreeSet::new(),
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
        if self.admission_closing {
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
            actor.permit = Some(
                self.root_scope
                    .permit(self.manifest_digest, actor.registration)
                    .map_err(|error| {
                        TeacherSessionStartErrorV2::Admission(invalid(error.to_string()))
                    })?,
            );
        }
        let bound = actor.bound.take().expect("bound actor retained");
        let permit = actor.permit.as_ref().expect("process permit retained");
        actor.spawn_attempted = true;
        // An unwind across the spawn call is not evidence of no worker.
        actor.worker_state = ActorWorkerStateV2::SpawnInProgress;
        match spawn_bound_collection_teacher_v2(
            bound,
            permit,
            &self.root_scope,
            startup_deadline,
            cleanup_ceiling,
            custodian,
        ) {
            Ok((lifecycle, recovery_key)) => {
                actor.worker_state = ActorWorkerStateV2::Started;
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
                actor.worker_state = ActorWorkerStateV2::BeforeSpawn;
                actor.bound = Some(bound);
                Err(TeacherSessionStartErrorV2::BeforeSpawn(cause))
            }
            Err(BoundTeacherSpawnFailureV2::AfterSpawn {
                cause,
                lifecycle,
                recovery_key,
            }) => {
                actor.worker_state = ActorWorkerStateV2::Started;
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
    fn worker_mut(&mut self, actor_id: &str) -> Option<&mut RoventCollectionTeacherLifecycleV2> {
        self.actors
            .get_mut(actor_id)?
            .running
            .as_mut()
            .map(|run| &mut run.lifecycle)
    }

    /// Retrieve diagnostics only; this does not expose the data-plane owner.
    pub(crate) fn take_actor_diagnostics_after_join(
        &mut self,
        actor_id: &str,
    ) -> Option<rz_search::external_cpu::collection_teacher::TeacherDiagnostics> {
        self.worker_mut(actor_id)?.take_diagnostics_after_join()
    }

    pub(crate) fn try_physical_retirement(
        &self,
    ) -> Result<Option<RoventTeacherScopePhysicalAckV2>, AdaptiveCollectionError> {
        self.root_scope.try_physical_retirement()
    }

    pub(crate) fn try_worker_retirement(
        &self,
    ) -> Result<Option<RoventTeacherScopeWorkerAckV2>, AdaptiveCollectionError> {
        self.root_scope.try_worker_retirement()
    }

    /// Release this session's original permits only after a complete opaque
    /// aggregate matches this admission, this exact roster and each retained
    /// member. Validation is read-only and precedes all permit releases.
    /// Physical retirement does not imply teacher or Collection success.
    pub(crate) fn release_retired_permits(
        &mut self,
        aggregate: &RoventTeacherScopeAggregateAckV2,
    ) -> Result<(), ArenaError> {
        use rz_search::external_cpu::process_scope::{
            CollectionMemberPhysicalState as P, CollectionMemberWorkerState as W,
        };
        check(
            aggregate.matches_admission(&self.admission) && self.root_scope_closed,
            "V2 retired permit aggregate admission/scope mismatch",
        )?;
        let physical = aggregate
            .physical()
            .scope_outcome()
            .ok_or_else(|| invalid("V2 session cannot retire using no-scope proof"))?;
        let workers = aggregate
            .workers()
            .filter(|value| value.matches_admission(&self.admission))
            .ok_or_else(|| invalid("V2 retired worker roster absent"))?
            .outcome();
        check(
            physical.same_issuer_as_worker(workers)
                && physical.members().len() == self.actors.len()
                && workers.members().len() == self.actors.len(),
            "V2 retired permit issuer or roster mismatch",
        )?;
        for actor in self.actors.values() {
            let matches: Vec<_> = physical
                .members()
                .iter()
                .enumerate()
                .filter_map(|(index, (registration, _))| {
                    (*registration == actor.registration).then_some(index)
                })
                .collect();
            check(
                matches.len() == 1 && actor.running.is_none() && actor.pending.is_none(),
                "V2 retired actor roster or pending work mismatch",
            )?;
            let index = matches[0];
            let (_, p) = physical.members()[index];
            let (worker_registration, w) = workers.members()[index];
            check(
                worker_registration == actor.registration,
                "V2 retired worker registration mismatch",
            )?;
            check(
                match actor.worker_state {
                    ActorWorkerStateV2::NotAttempted | ActorWorkerStateV2::BeforeSpawn => {
                        w == W::NoThread
                    }
                    ActorWorkerStateV2::Joined | ActorWorkerStateV2::Retired => w == W::Joined,
                    ActorWorkerStateV2::SpawnInProgress | ActorWorkerStateV2::Started => false,
                },
                "V2 retired actor worker state mismatch",
            )?;
            check(
                matches!(
                    p,
                    P::NeverIssuedNoChild
                        | P::IssuedNoChild
                        | P::AcquiredNoChild
                        | P::PhysicallyConfirmed
                ),
                "V2 retired actor physical state unresolved",
            )?;
            if let Some(permit) = &actor.permit {
                check(
                    physical.state_for_permit(permit) == Some(p)
                        && workers.matches_member_at(index, &permit.member_witness()),
                    "V2 retired actor permit belongs to another issuer/member",
                )?;
            } else {
                // An issued-without-returned-permit race is safe only when
                // the actual worker roster proves this member has no thread.
                check(
                    w == W::NoThread,
                    "V2 retired joined actor lacks original permit",
                )?;
            }
        }
        for actor in self.actors.values_mut() {
            actor.permit = None;
        }
        Ok(())
    }

    /// Original preflight registration for this roster entry. Callers must
    /// still bind the actual worker, dispatch and completion before labeling.
    pub(crate) fn registered(&self, actor_id: &str) -> Option<&super::RoventRegisteredTeacher> {
        self.actors
            .get(actor_id)
            .map(|actor| actor.original.registered())
    }

    /// Start one real teacher job from the original retired row. This runs on
    /// the separate collection worker, not inside the manager or producer
    /// lock. The borrowed row remains with the caller on every error.
    pub(crate) fn submit_retired(
        &mut self,
        permit: &TeacherPumpPermitV2,
        row: &RoventRetiredLockV2Record,
        constraint: ExternalGoConstraint,
        soft_quantum: Duration,
    ) -> Result<TeacherSubmitOutcomeV2, ArenaError> {
        self.validate_pump_owner(permit)?;
        if self.admission_closing {
            return Err(invalid("V2 teacher session admission is closed"));
        }
        check(
            row.admission.same_original_admission(&self.admission)
                && row.matches_manifest_owner(&self.lock_manifest)
                && row.manifest_sha256 == self.lock_manifest_sha256
                && !row.record.0.input.legal_moves.is_empty(),
            "V2 teacher row differs from original root/manifest or has no policy candidates",
        )?;
        self.require_live_root()?;
        let input = row.submitted_input_seal();
        self.admission
            .validate_ticket(&input.ticket)
            .map_err(|error| invalid(error.to_string()))?;
        let actor_id = input
            .teacher_actor_id
            .as_deref()
            .ok_or_else(|| invalid("V2 row has no frozen teacher actor"))?;
        let position = row.original_teacher_query_position()?;
        let root = self.admission.root_admission();
        let actor = self
            .actors
            .get_mut(actor_id)
            .ok_or_else(|| invalid("V2 teacher actor absent"))?;
        if actor.pending.is_some() {
            return Ok(TeacherSubmitOutcomeV2::Pending(
                TeacherSubmitPendingV2::ActorBusy,
            ));
        }
        let registered = actor.original.registered();
        let run = actor
            .running
            .as_mut()
            .ok_or_else(|| invalid("V2 teacher actor was not started"))?;
        let owner = run.lifecycle.owner();
        let startup = match owner
            .verified_startup()
            .map_err(|error| invalid(error.to_string()))?
        {
            Some(startup) => startup,
            None => {
                return Ok(TeacherSubmitOutcomeV2::Pending(
                    TeacherSubmitPendingV2::Startup,
                ));
            }
        };
        check(
            owner.admission().same_original_admission(&self.admission)
                && owner.source_index() == actor.source_index
                && owner.worker().registration() == actor.registration
                && actor.registration.frozen_actor == registered.frozen_actor_digest()?
                && input.ticket.authority == root.authority()
                && input.ticket.original_deadline == root.hard_deadline()
                && exact_rules_anchor(&position)? == input.anchor
                && startup.canonical_sha256() == registered.pin.canonical_sha256
                && startup.capability_sha256() == registered.pin.capability_sha256
                && startup.input_lock_sha256() == super::super::hex(root.startup_lock_digest())
                && owner.launch().binary().sha256 == registered.pin.binary_sha256
                && owner.launch().binary().bytes == registered.pin.binary_bytes,
            "V2 teacher original input/actor/root/startup mismatch",
        )?;
        let role = match input.role {
            GenerationRoleV3::Proposer => Role::Proposer,
            GenerationRoleV3::Critic => Role::Critic,
            GenerationRoleV3::Validator => return Err(invalid("V2 teacher policy is P/C only")),
        };
        check(
            input.ticket.role == role,
            "V2 teacher input and ticket role differ",
        )?;
        let parse = |value: &str| {
            rz_eval::asset::parse_sha256(value)
                .map(Digest)
                .map_err(|error| invalid(error.to_string()))
        };
        let scope = TeacherInputScope {
            authority: input.ticket.authority,
            input_sha256: parse(&input.input_sha256)?,
            view_sha256: parse(&input.selected_view_sha256)?,
            source_sha256: parse(&self.manifest.source_sha256)?,
            role,
            rules_semantic: parse(&input.anchor.rules_state_sha256)?,
            input_lock: root.startup_lock_digest(),
            root_admission: root.descriptor().digest()?,
            root_budget: self.admission.root_budget(),
            frozen_actor: actor.registration.frozen_actor,
        };
        let id = TeacherJobId {
            collection_owner: actor.registration.collection_owner,
            sequence: actor.next_sequence,
        };
        let next = actor
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| invalid("V2 teacher sequence exhausted"))?;
        // Allocate/copy all receipt metadata before the real worker admission.
        let pending_actor = actor_id.to_owned();
        let pending_registered = registered.clone();
        let input_key = (id.collection_owner, input.ticket.id);
        check(
            self.issued_inputs.len() < MAX_CORPUS_ROWS,
            "V2 teacher input ledger full",
        )?;
        check(
            self.issued_inputs.insert(input_key),
            "V2 teacher input already submitted",
        )?;
        let job = TeacherJob {
            id,
            input_scope: scope,
            position,
            root_moves: None,
            constraint,
            original_deadline: input.ticket.original_deadline,
            soft_quantum,
            control: owner.worker().control_mailbox(id),
        };
        let dispatch = match run.lifecycle.submit(job) {
            Ok(dispatch) => dispatch,
            Err(error) => {
                self.issued_inputs.remove(&input_key);
                return Err(invalid(error.to_string()));
            }
        };
        actor.next_sequence = next;
        actor.pending = Some(PendingTeacherV2 {
            actor_id: pending_actor,
            input,
            registered: pending_registered,
            startup,
            soft_quantum,
            dispatch,
        });
        Ok(TeacherSubmitOutcomeV2::Submitted(id))
    }

    /// Retrieve only the terminal from the same original actor lifecycle.
    /// The caller retains the row and must separately validate live authority
    /// before considering any future target.
    pub(crate) fn take_completed_actor(
        &mut self,
        permit: &TeacherPumpPermitV2,
        actor_id: &str,
    ) -> Result<Option<CompletedTeacherV2>, ArenaError> {
        self.validate_pump_owner(permit)?;
        let actor = self
            .actors
            .get_mut(actor_id)
            .ok_or_else(|| invalid("V2 teacher actor absent"))?;
        if actor.pending.is_none() {
            return Ok(None);
        }
        let run = actor
            .running
            .as_mut()
            .ok_or_else(|| invalid("V2 pending teacher owner absent"))?;
        let Some(completion) = run.lifecycle.take_completion() else {
            return Ok(None);
        };
        let pending = actor.pending.take().expect("checked pending teacher job");
        Ok(Some(CompletedTeacherV2 {
            actor_id: pending.actor_id,
            input: pending.input,
            registered: pending.registered,
            startup: pending.startup,
            soft_quantum: pending.soft_quantum,
            dispatch: pending.dispatch,
            completion,
        }))
    }

    /// Bind this original completed job to the exact retired row, then claim
    /// its one-shot UCI evidence at the original live authority boundary.
    /// A rejection returns every original input and worker result unchanged.
    pub(crate) fn seal_completed_policy(
        &mut self,
        permit: &TeacherPumpPermitV2,
        row: &RoventRetiredLockV2Record,
        completed: CompletedTeacherV2,
    ) -> Result<SealedTeacherPolicyV2, Box<TeacherPolicySealRejectionV2>> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.validate_pump_owner(permit)
        })) {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                return Err(Box::new(TeacherPolicySealRejectionV2::Rejected {
                    cause: TeacherPolicySealFailureV2::Validation(error),
                    completed,
                }))
            }
            Err(payload) => {
                return Err(Box::new(TeacherPolicySealRejectionV2::Rejected {
                    cause: TeacherPolicySealFailureV2::Panicked(payload),
                    completed,
                }))
            }
        }
        let validation =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_, ArenaError> {
                let original = row.submitted_input_seal();
                check(
                    row.admission.same_original_admission(&self.admission)
                        && row.matches_manifest_owner(&self.lock_manifest)
                        && row.manifest_sha256 == self.lock_manifest_sha256
                        && original.ticket == completed.input.ticket
                        && original.manifest_sha256 == completed.input.manifest_sha256
                        && original.input_sha256 == completed.input.input_sha256
                        && original.selected_view_sha256 == completed.input.selected_view_sha256
                        && original.anchor == completed.input.anchor
                        && original.role == completed.input.role
                        && original.teacher_actor_id.as_deref()
                            == Some(completed.actor_id.as_str()),
                    "V2 teacher completed input differs from original retired row",
                )?;
                let actor = self
                    .actors
                    .get(&completed.actor_id)
                    .ok_or_else(|| invalid("V2 completed teacher actor absent"))?;
                check(
                    actor.original.registered().frozen_actor_digest()?
                        == completed.registered.frozen_actor_digest()?
                        && actor.registration.frozen_actor
                            == completed.registered.frozen_actor_digest()?,
                    "V2 completed teacher registration differs from original actor",
                )?;
                let run = actor
                    .running
                    .as_ref()
                    .ok_or_else(|| invalid("V2 completed teacher owner absent"))?;
                let owner = run.lifecycle.owner();
                let startup = owner
                    .verified_startup()
                    .map_err(|error| invalid(error.to_string()))?
                    .ok_or_else(|| invalid("V2 completed teacher startup absent"))?;
                check(
                    owner.admission().same_original_admission(&self.admission)
                        && owner.worker().registration() == actor.registration
                        && startup.process_identity() == completed.startup.process_identity()
                        && startup.canonical_sha256() == completed.startup.canonical_sha256()
                        && startup.capability_sha256() == completed.startup.capability_sha256()
                        && startup.input_lock_sha256() == completed.startup.input_lock_sha256(),
                    "V2 completed teacher owner/startup differs from original",
                )?;
                Ok(row.original_teacher_query_position()?.snapshot())
            }));
        let query = match validation {
            Ok(Ok(query)) => query,
            Ok(Err(error)) => {
                return Err(Box::new(TeacherPolicySealRejectionV2::Rejected {
                    cause: TeacherPolicySealFailureV2::Validation(error),
                    completed,
                }));
            }
            Err(payload) => {
                return Err(Box::new(TeacherPolicySealRejectionV2::Rejected {
                    cause: TeacherPolicySealFailureV2::Panicked(payload),
                    completed,
                }));
            }
        };
        let Some(actor) = self.actors.get_mut(&completed.actor_id) else {
            return Err(Box::new(TeacherPolicySealRejectionV2::Rejected {
                cause: TeacherPolicySealFailureV2::Validation(invalid(
                    "V2 completed teacher actor disappeared",
                )),
                completed,
            }));
        };
        let Some(run) = actor.running.as_mut() else {
            return Err(Box::new(TeacherPolicySealRejectionV2::Rejected {
                cause: TeacherPolicySealFailureV2::Validation(invalid(
                    "V2 completed teacher owner disappeared",
                )),
                completed,
            }));
        };
        let manifest_owner = Arc::clone(&self.lock_manifest);
        let CompletedTeacherV2 {
            actor_id,
            input,
            registered,
            startup,
            soft_quantum,
            dispatch,
            completion,
        } = completed;
        let mut custody = RoventTeacherPolicyCustodyV2::new(dispatch, completion);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run.lifecycle
                .seal_policy_custody(&mut custody, &query, None)
        }));
        match result {
            Ok(Ok(evidence)) => Ok(SealedTeacherPolicyV2 {
                manifest_owner,
                actor_id,
                input,
                registered,
                startup,
                soft_quantum,
                evidence,
            }),
            Ok(Err(rejected)) => match custody.into_originals() {
                Ok((dispatch, completion)) => {
                    Err(Box::new(TeacherPolicySealRejectionV2::Rejected {
                        cause: TeacherPolicySealFailureV2::Native {
                            kind: rejected.kind,
                            error: rejected.error,
                        },
                        completed: CompletedTeacherV2 {
                            actor_id,
                            input,
                            registered,
                            startup,
                            soft_quantum,
                            dispatch,
                            completion,
                        },
                    }))
                }
                Err(custody) => Err(Box::new(TeacherPolicySealRejectionV2::Quarantined(
                    TeacherPolicyQuarantineV2 {
                        manifest_owner,
                        actor_id,
                        input,
                        registered,
                        startup,
                        soft_quantum,
                        custody,
                        panic: None,
                        native_failure: Some(rejected),
                    },
                ))),
            },
            Err(payload) => Err(Box::new(TeacherPolicySealRejectionV2::Quarantined(
                TeacherPolicyQuarantineV2 {
                    manifest_owner,
                    actor_id,
                    input,
                    registered,
                    startup,
                    soft_quantum,
                    custody,
                    panic: Some(payload),
                    native_failure: None,
                },
            ))),
        }
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
    pub(crate) fn request_shutdown(
        &mut self,
        until: Instant,
    ) -> Result<(), AdaptiveCollectionError> {
        let close_result = self.close_admission();
        for actor in self.actors.values_mut() {
            if let Some(run) = actor.running.as_mut() {
                run.lifecycle.request_shutdown(until);
            }
        }
        close_result
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
    ) -> Result<
        Option<(
            RoventCollectionTeacherOwnerV2,
            RoventTeacherRetirementObservationV2,
        )>,
        ArenaError,
    > {
        let actor = self
            .actors
            .get_mut(actor_id)
            .ok_or_else(|| invalid("V2 teacher actor absent"))?;
        if actor.pending.is_some() {
            return Ok(None);
        }
        let Some(run) = actor.running.take() else {
            return Ok(None);
        };
        match run.lifecycle.try_finish() {
            Ok((owner, observation)) => {
                // Preserve the original permit for later same-member
                // physical proof and release it only with session custody.
                actor.worker_state = ActorWorkerStateV2::Retired;
                Ok(Some((owner, observation)))
            }
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
        if actor.pending.is_some() {
            return Ok(None);
        }
        let Some(run) = actor.running.take() else {
            return Ok(None);
        };
        match run.lifecycle.try_retire() {
            Ok(token) => {
                // Preserve the original permit for same-member proof.
                actor.worker_state = ActorWorkerStateV2::Retired;
                Ok(Some(token))
            }
            Err(lifecycle) => {
                actor.running = Some(RunningTeacherV2 {
                    lifecycle,
                    recovery_key: run.recovery_key,
                });
                Ok(None)
            }
        }
    }

    /// Take the original joined worker even if it never created an OS child.
    /// The token retains lifecycle and recovery custody; it is not by itself
    /// a process, whole-roster, or semantic-success proof.
    pub(crate) fn try_finish_worker_actor(
        &mut self,
        actor_id: &str,
    ) -> Result<Option<RoventTeacherWorkerTerminalTokenV2>, ArenaError> {
        let actor = self
            .actors
            .get_mut(actor_id)
            .ok_or_else(|| invalid("V2 teacher actor absent"))?;
        if actor.pending.is_some() {
            return Ok(None);
        }
        let Some(run) = actor.running.take() else {
            return Ok(None);
        };
        match run.lifecycle.try_finish_worker() {
            Ok(token) => {
                actor.worker_state = ActorWorkerStateV2::Joined;
                Ok(Some(token))
            }
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
                if let Err(close_error) = self.close_admission() {
                    return Err(invalid(format!(
                        "V2 teacher root is no longer accepting; scope closure failed: {close_error}"
                    )));
                }
                Err(invalid("V2 teacher root is no longer accepting"))
            }
            Err(error) => {
                if let Err(close_error) = self.close_admission() {
                    return Err(invalid(format!(
                        "V2 teacher root observation failed: {error}; scope closure failed: {close_error}"
                    )));
                }
                Err(ArenaError::Contract(error))
            }
        }
    }

    /// Closes admission, not worker processes. The eventual owner must request
    /// and observe each worker's shutdown and physical drain separately.
    pub(crate) fn close_admission(&mut self) -> Result<(), AdaptiveCollectionError> {
        // Block new work immediately, including when the capability's
        // nonblocking close is temporarily unavailable and must be retried.
        self.admission_closing = true;
        let result = if self.root_scope_closed {
            Ok(())
        } else {
            self.root_scope.close()
        };
        if result.is_ok() {
            self.root_scope_closed = true;
        }
        result
    }
}
