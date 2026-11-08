//! Original Own /3 root preparation custody, shared with its supervisor.
//! This is not a worker completion, teacher result, or publication proof.

use super::capture_router_v3::CaptureRouterV3;
use super::launch::PreparedPcCollectionInputsV3;
use super::observer_lock_v3::RoventCollectionOwnerLockV3;
use super::own_hooks_lock_v3::{
    OwnPrepareOutcomeV3, OwnPreparedRootV3, OwnReserveOutcomeV3, OwnRootRequestV3,
    RoventOwnCollectionBridgeV3,
};
use super::policy_row_pipeline_v3::{
    PolicyChunkPersistenceV3, PolicyRetentionOutcomeV3, PolicyRootRetainedOwnerV3,
    PolicyRowPipelineOwnerV3,
};
use super::speculative_route_v3::{SpeculativeReceiverV3, SpeculativeRootV3};
use super::teacher::v3_policy_consumer::{
    TeacherPolicyConsumerOwnerV3, TeacherPolicyConsumerProgressV3, TeacherPolicyConsumerStartV3,
    TeacherProcessedOwnerV3,
};
use super::teacher::v3_pump::{
    TeacherClosedCustodyStartV3, TeacherClosedCustodyV3, TeacherClosedDrainV3,
    TeacherClosedHandoffV3, TeacherPumpSetupErrorV3, TeacherPumpV3,
};
use super::teacher::v3_session::{
    RoventCollectionTeacherSessionV3, TeacherPrepareErrorV3, TeacherSessionClaimErrorV3, TeacherSessionStageStateV3,
    TeacherSessionStagingV3, TeacherStartErrorV3,
};
use super::writer_lock_v3::RoventCorpusWriterLockV3;
use super::{ArenaError, invalid};
use rz_uci::pals_adaptive::{
    AdaptiveCollectionError, RoventBoardCollectionWorkerKeyV3, RoventBoardCollectionWorkerStageV3,
    RoventBoardCollectionWorkerV3, RoventBoardRootHandoffReceiverV3, RoventBoardRootHandoffV3,
    RoventBoardWorkerStagePrepareFailureV3, RoventBoardWorkerStageStartRejectionV3,
    RoventBoardWorkerStageStartV3, RoventCollectionRootAdmissionV3,
};
use rz_uci::rovent_external_launch::{
    RoventTeacherScopeAggregateAckV3, TeacherRecoveryCustodianV3,
};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

struct OwnScopeEffectSlotsV3 {
    proof: Option<OwnReceivedStageProofV3>,
    teacher_source: Option<super::teacher::v3_session::TeacherScopeEffectSourceV3>,
    game_source: Option<super::game_authority_v3::GameEffectSourceV3<OwnWorkerSupervisorV3>>,
    claim: Option<super::game_authority_v3::GameEffectClaimV3<OwnWorkerSupervisorV3>>,
    outcome: Option<
        Result<
            super::teacher::v3_session::TeacherScopeEffectOutcomeV3,
            super::teacher::v3_session::TeacherScopeEffectClaimFailureV3,
        >,
    >,
    unclaimed_proof: Option<super::teacher::v3_session::TeacherScopeUnclaimedProofV3>,
    before_source_proof: Option<super::teacher::v3_session::TeacherScopeBeforeSourceProofV3>,
    issue_error: Option<TeacherSessionClaimErrorV3>,
    source_error: Option<&'static str>,
    claim_error: Option<&'static str>,
    completion_error: Option<&'static str>,
    source_step: Option<super::game_authority_v3::GameStepV3>,
    claim_step: Option<super::game_authority_v3::GameStepV3>,
    completion_step: Option<super::game_authority_v3::GameStepV3>,
    finished: bool,
}
impl OwnScopeEffectSlotsV3 {
    fn new() -> Self {
        Self {
            proof: None,
            teacher_source: None,
            game_source: None,
            claim: None,
            outcome: None,
            unclaimed_proof: None,
            before_source_proof: None,
            issue_error: None,
            source_error: None,
            claim_error: None,
            completion_error: None,
            source_step: None,
            claim_step: None,
            completion_step: None,
            finished: false,
        }
    }
}

/// A separate owner for one original actor's acquisition state. It is
/// registered with the root before source issuance or process creation.
struct OwnActorEffectSlotsV3 {
    actor_id: String,
    proof: Option<OwnReceivedActorProofV3>,
    teacher_source: Option<super::teacher::v3_session::TeacherActorEffectSourceV3>,
    game_source: Option<super::game_authority_v3::GameEffectSourceV3<OwnWorkerSupervisorV3>>,
    claim: Option<super::game_authority_v3::GameEffectClaimV3<OwnWorkerSupervisorV3>>,
    outcome: Option<
        Result<
            (
                rz_uci::rovent_external_launch::TeacherRecoveryKeyV3,
                super::teacher::v3_session::TeacherActorEffectOutcomeV3,
            ),
            super::teacher::v3_session::TeacherActorEffectFailureV3,
        >,
    >,
    issue_error: Option<TeacherStartErrorV3>,
    source_error: Option<&'static str>,
    claim_error: Option<&'static str>,
    completion_error: Option<&'static str>,
    source_step: Option<super::game_authority_v3::GameStepV3>,
    claim_step: Option<super::game_authority_v3::GameStepV3>,
    completion_step: Option<super::game_authority_v3::GameStepV3>,
    finished: bool,
}
impl OwnActorEffectSlotsV3 {
    fn new(actor_id: String) -> Self {
        Self {
            actor_id,
            proof: None,
            teacher_source: None,
            game_source: None,
            claim: None,
            outcome: None,
            issue_error: None,
            source_error: None,
            claim_error: None,
            completion_error: None,
            source_step: None,
            claim_step: None,
            completion_step: None,
            finished: false,
        }
    }
}


/// Publish this Arc to the bridge before claiming a teacher scope, opening a
/// route, or starting a worker. All uncertain original owners stay together.
pub(crate) struct OwnRootRecoveryV3 {
    original_self: Weak<Mutex<OwnRootRecoveryV3>>,
    scope_effects: Arc<Mutex<OwnScopeEffectSlotsV3>>,
    actor_effects: Vec<Arc<Mutex<OwnActorEffectSlotsV3>>>,
    original_scope_identity: Option<Weak<()>>,
    recovery_published: bool,
    pub(super) request: OwnRootRequestV3,
    pub(super) inputs: Arc<PreparedPcCollectionInputsV3>,
    pub(super) stage: Option<Arc<Mutex<TeacherSessionStagingV3>>>,
    pub(super) route: Option<SpeculativeRootV3>,
    pub(super) receiver: Option<Arc<Mutex<SpeculativeReceiverV3>>>,
    pub(super) owner: Option<RoventCollectionOwnerLockV3>,
    pub(super) pump: Option<Arc<Mutex<TeacherPumpV3>>>,
    pub(super) closed_custody: Option<TeacherClosedCustodyV3>,
    pub(super) closed_custody_setup_unknown: bool,
    pub(super) consumer: Option<TeacherPolicyConsumerOwnerV3>,
    // One root-owned phase; only a successful step advances this cursor.
    pub(super) consumer_phase: u8,
    pub(super) processed: Option<TeacherProcessedOwnerV3>,
    pub(super) consumer_failure: Option<ArenaError>,
    // Only a row-scoped rejection may request retention for this exact
    // original completion ticket. Seal failures and Unknown have no ticket.
    pub(super) rejection_scope: Option<(
        RoventCollectionRootAdmissionV3,
        rz_contracts::pals_adaptive::RoleTicketEnvelope,
    )>,
    pub(super) rejection_retained: bool,
    pub(super) rejection_retention_failure: Option<ArenaError>,
    // The pipeline owns the original packet and detailed failure. This flag
    // prevents an automatic retry after a failed or uncertain write.
    pub(super) persistence_failure: Option<&'static str>,
    pub(super) pipeline: PolicyRowPipelineOwnerV3,
    pub(super) retained_policy: Option<PolicyRootRetainedOwnerV3>,
    pub(super) pump_error: Option<TeacherPumpSetupErrorV3>,
    pub(super) pump_unknown: bool,
    pub(super) worker_key: Option<RoventBoardCollectionWorkerKeyV3>,
    pub(super) worker_stage: Option<RoventBoardCollectionWorkerStageV3>,
    pub(super) worker_prepare_failure: Option<RoventBoardWorkerStagePrepareFailureV3>,
    pub(super) worker_start_rejection: Option<RoventBoardWorkerStageStartRejectionV3>,
    pub(super) custodian: TeacherRecoveryCustodianV3,
    pub(super) panic: Option<Box<dyn std::any::Any + Send>>,
    pub(super) additional_panics: Vec<Box<dyn std::any::Any + Send>>,
    pub(super) preflight_failure: Option<TeacherPrepareErrorV3>,
    pub(super) actor_failure: Option<TeacherStartErrorV3>,
    pub(super) worker: Option<RoventBoardCollectionWorkerV3>,
    pub(super) prepared: Option<OwnPreparedRootV3>,
    pub(super) aggregate: Option<RoventTeacherScopeAggregateAckV3>,
    pub(super) diagnostics: Vec<(
        String,
        rz_search::external_cpu::collection_teacher::TeacherDiagnostics,
    )>,
    pub(super) next_actor: usize,
    pub(super) published: bool,
    pub(super) route_unlinked: bool,
    pub(super) failure: Option<String>,
    pub(super) additional_failures: Vec<String>,
    pub(super) shutdown_failure: Option<AdaptiveCollectionError>,
    pub(super) shutdown_failure_count: u64,
    pub(super) pump_failure: Option<String>,
    pub(super) pump_failure_count: u64,
    pub(super) pump_admission_closed: bool,
    pub(super) closed_input_pending: bool,
    pub(super) closed_input_error: Option<String>,
    pub(super) closed_handoff_pending: bool,
    pub(super) closed_handoff_error: Option<String>,
    pub(super) target_revoke_pending: bool,
    pub(super) target_revoke_error: Option<String>,
}

impl OwnRootRecoveryV3 {
    pub(crate) fn new(
        request: OwnRootRequestV3,
        inputs: Arc<PreparedPcCollectionInputsV3>,
        pipeline: PolicyRowPipelineOwnerV3,
    ) -> Arc<Mutex<Self>> {
        Arc::new_cyclic(|original_self| Mutex::new(Self {
            original_self: original_self.clone(),
            scope_effects: Arc::new(Mutex::new(OwnScopeEffectSlotsV3::new())),
            actor_effects: Vec::new(),
            original_scope_identity: None,
            recovery_published: false,
            request,
            inputs,
            pipeline,
            retained_policy: None,
            stage: None,
            route: None,
            receiver: None,
            owner: None,
            pump: None,
            closed_custody: None,
            closed_custody_setup_unknown: false,
            consumer: None,
            consumer_phase: 0,
            processed: None,
            consumer_failure: None,
            rejection_scope: None,
            rejection_retained: false,
            rejection_retention_failure: None,
            persistence_failure: None,
            pump_error: None,
            pump_unknown: false,
            worker_key: None,
            worker_stage: None,
            worker_prepare_failure: None,
            worker_start_rejection: None,
            custodian: TeacherRecoveryCustodianV3::new(),
            panic: None,
            additional_panics: Vec::new(),
            preflight_failure: None,
            actor_failure: None,
            worker: None,
            prepared: None,
            aggregate: None,
            diagnostics: Vec::new(),
            next_actor: 0,
            published: false,
            route_unlinked: false,
            failure: None,
            additional_failures: Vec::new(),
            shutdown_failure: None,
            shutdown_failure_count: 0,
            pump_failure: None,
            pump_failure_count: 0,
            pump_admission_closed: false,
            closed_input_pending: false,
            closed_input_error: None,
            closed_handoff_pending: false,
            closed_handoff_error: None,
            target_revoke_pending: false,
            target_revoke_error: None,
        }))
    }

    /// False or Busy retains the launch; effect issuance is not completion.
    pub(super) fn try_effects_settled(&self) -> Result<Option<bool>, &'static str> {
        let scope = match self.scope_effects.try_lock() {
            Ok(value) => value,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("Own /3 scope slots poisoned"),
        };
        if !scope.finished && (scope.proof.is_some() || scope.teacher_source.is_some()
            || scope.game_source.is_some() || scope.claim.is_some() || scope.outcome.is_some()
            || scope.issue_error.is_some() || scope.source_error.is_some()
            || scope.claim_error.is_some() || scope.completion_error.is_some()) {
            return Ok(Some(false));
        }
        drop(scope);
        for owner in &self.actor_effects {
            let actor = match owner.try_lock() {
                Ok(value) => value,
                Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
                Err(_) => return Err("Own /3 actor slots poisoned"),
            };
            if !actor.finished && (actor.proof.is_some() || actor.teacher_source.is_some()
                || actor.game_source.is_some() || actor.claim.is_some() || actor.outcome.is_some()
                || actor.issue_error.is_some() || actor.source_error.is_some()
                || actor.claim_error.is_some() || actor.completion_error.is_some()) {
                return Ok(Some(false));
            }
        }
        Ok(Some(true))
    }

    pub(crate) fn matches_admission(&self, original: &RoventCollectionRootAdmissionV3) -> bool {
        self.request.admission.same_original_admission(original)
    }

    pub(crate) fn try_matches_prepared(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
        route: &SpeculativeRootV3,
        owner: &RoventCollectionOwnerLockV3,
        stage: &Arc<Mutex<TeacherSessionStagingV3>>,
        pump: &Arc<Mutex<TeacherPumpV3>>,
        pipeline: &PolicyRowPipelineOwnerV3,
        key: &RoventBoardCollectionWorkerKeyV3,
    ) -> Result<Option<bool>, &'static str> {
        let basic = !self.request.is_revoked()
            && self.pipeline.same_owner(pipeline)
            && self.matches_admission(admission)
            && self
                .route
                .as_ref()
                .is_some_and(|original| original.same_original_route(route))
            && self
                .owner
                .as_ref()
                .is_some_and(|original| original.same_original_owner(owner))
            && self
                .stage
                .as_ref()
                .is_some_and(|original| Arc::ptr_eq(original, stage))
            && self
                .pump
                .as_ref()
                .is_some_and(|original| Arc::ptr_eq(original, pump))
            && self
                .worker_key
                .as_ref()
                .is_some_and(|original| original.same_original(key));
        if !basic {
            return Ok(Some(false));
        }
        let stage = match stage.try_lock() {
            Ok(value) => value,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err("Own /3 staged session poisoned");
            }
        };
        let Some(session) = stage.session() else {
            return Ok(Some(false));
        };
        let pump = match pump.try_lock() {
            Ok(value) => value,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err("Own /3 teacher pump poisoned");
            }
        };
        if !Arc::ptr_eq(pump.session(), &session)
            || !self
                .receiver
                .as_ref()
                .is_some_and(|original| Arc::ptr_eq(original, pump.receiver()))
            || !self
                .closed_custody
                .as_ref()
                .is_some_and(|original| pump.owns_closed_custody(original))
        {
            return Ok(Some(false));
        }
        let receiver = match pump.receiver().try_lock() {
            Ok(value) => value,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err("Own /3 route receiver poisoned");
            }
        };
        if !receiver.matches_original_route(route) {
            return Ok(Some(false));
        }
        let session = match session.try_lock() {
            Ok(value) => value,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err("Own /3 teacher session poisoned");
            }
        };
        Ok(Some(
            session
                .original_admission()
                .same_original_admission(admission)
                && session.all_actor_owners_started(),
        ))
    }
}

pub(crate) enum OwnPrepareStepV3 {
    Pending,
    Published,
    Failed,
}

pub(super) struct OwnReceivedStageProofV3 {
    supervisor: Weak<Mutex<OwnWorkerSupervisorV3>>,
    bridge: Weak<RoventOwnCollectionBridgeV3>,
    recovery: Weak<Mutex<OwnRootRecoveryV3>>,
    stage: Weak<Mutex<TeacherSessionStagingV3>>,
    scope_identity: Weak<()>,
    admission: RoventCollectionRootAdmissionV3,
}

impl OwnReceivedStageProofV3 {
    pub(super) fn matches_game(
        &self,
        game: &Arc<super::game_authority_v3::GameAuthorityV3<OwnWorkerSupervisorV3>>,
    ) -> Option<bool> {
        self.bridge
            .upgrade()
            .map(|bridge| Arc::ptr_eq(bridge.game(), game))
    }
    pub(super) fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV3) -> bool {
        self.admission.same_original_admission(admission)
    }
    pub(super) fn matches_recovery(&self, recovery: &Arc<Mutex<OwnRootRecoveryV3>>) -> Option<bool> {
        self.recovery
            .upgrade()
            .map(|original| Arc::ptr_eq(&original, recovery))
    }
    pub(super) fn stage(&self) -> Option<Arc<Mutex<TeacherSessionStagingV3>>> {
        self.stage.upgrade()
    }
    pub(super) fn matches_scope_source(
        &self,
        source: &super::teacher::v3_session::TeacherScopeEffectSourceV3,
    ) -> bool {
        Weak::ptr_eq(&self.scope_identity, &source.identity_weak())
            && source.matches_admission(&self.admission)
    }

    /// The caller holds the exact supervisor, recovery and stage guards, and
    /// supplies the same stage Arc from which its guard was obtained.
    pub(super) fn try_from_guards(
        supervisor: &OwnWorkerSupervisorV3,
        recovery: &OwnRootRecoveryV3,
        stage_owner: &Arc<Mutex<TeacherSessionStagingV3>>,
        stage: &TeacherSessionStagingV3,
    ) -> Result<Option<Self>, &'static str> {
        let Some(supervisor_identity) = supervisor.original_self.get() else {
            return Err("Own /3 supervisor identity is not bound");
        };
        if !recovery
            .stage
            .as_ref()
            .is_some_and(|original| Arc::ptr_eq(original, stage_owner))
        {
            return Err("Own /3 stage owner differs");
        }
        let Some(scope_identity) = &recovery.original_scope_identity else {
            return Err("Own /3 original scope identity absent");
        };
        if !Weak::ptr_eq(scope_identity, &stage.scope_effect_identity_weak()) {
            return Err("Own /3 guarded stage identity differs");
        }
        let proof = Self {
            supervisor: supervisor_identity.clone(),
            bridge: Arc::downgrade(&supervisor.bridge),
            recovery: recovery.original_self.clone(),
            stage: Arc::downgrade(stage_owner),
            scope_identity: scope_identity.clone(),
            admission: recovery.request.admission.clone(),
        };
        match proof.try_revalidate_with_guards(supervisor, recovery)? {
            None => Ok(None),
            Some(true) => Ok(Some(proof)),
            Some(false) => Err("Own /3 original guarded stage differs"),
        }
    }

    /// Revalidate while the caller already holds both original guards. The
    /// two Weak identities were sealed at construction/binding; no guard is
    /// reacquired and no raw address or copied identifier is authority.
    pub(super) fn try_revalidate_with_guards(
        &self,
        supervisor: &OwnWorkerSupervisorV3,
        recovery: &OwnRootRecoveryV3,
    ) -> Result<Option<bool>, &'static str> {
        let Some(supervisor_identity) = supervisor.original_self.get() else {
            return Err("Own /3 supervisor identity is not bound");
        };
        if !Weak::ptr_eq(supervisor_identity, &self.supervisor)
            || !Weak::ptr_eq(&recovery.original_self, &self.recovery)
        {
            return Ok(Some(false));
        }
        let bridge = self
            .bridge
            .upgrade()
            .ok_or("Own /3 bridge owner unavailable")?;
        let recovery_owner = self
            .recovery
            .upgrade()
            .ok_or("Own /3 recovery owner unavailable")?;
        let stage = self
            .stage
            .upgrade()
            .ok_or("Own /3 stage owner unavailable")?;
        if !supervisor.matches_original_bridge(&bridge)
            || !supervisor.owns_recovery(&recovery_owner)
            || !recovery.matches_admission(&self.admission)
            || !recovery
                .stage
                .as_ref()
                .is_some_and(|original| Arc::ptr_eq(original, &stage))
            || !recovery
                .original_scope_identity
                .as_ref()
                .is_some_and(|original| Weak::ptr_eq(original, &self.scope_identity))
        {
            return Ok(Some(false));
        }
        bridge.try_matches_registered_recovery(&self.admission, &recovery_owner, &recovery.request.revocation_flag())
    }


}

/// Same received root, original ready session and exact actor slot. A source
/// identity is not an actor start, a terminal result or a physical fence.
pub(super) struct OwnReceivedActorProofV3 {
    stage_proof: OwnReceivedStageProofV3,
    session: Weak<Mutex<RoventCollectionTeacherSessionV3>>,
    session_identity: Weak<()>,
    actor_identity: Weak<()>,
}

impl OwnReceivedActorProofV3 {
    pub(super) fn matches_game(
        &self,
        game: &Arc<super::game_authority_v3::GameAuthorityV3<OwnWorkerSupervisorV3>>,
    ) -> Option<bool> {
        self.stage_proof.matches_game(game)
    }
    pub(super) fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV3) -> bool {
        self.stage_proof.matches_admission(admission)
    }
    pub(super) fn matches_actor_source(
        &self,
        source: &super::teacher::v3_session::TeacherActorEffectSourceV3,
    ) -> bool {
        Weak::ptr_eq(&self.session_identity, &source.session_identity_weak())
            && Weak::ptr_eq(
                &self.actor_identity,
                &Arc::downgrade(source.actor_identity()),
            )
            && source.matches_admission(&self.stage_proof.admission)
    }
    pub(super) fn try_from_guards(
        supervisor: &OwnWorkerSupervisorV3,
        recovery: &OwnRootRecoveryV3,
        stage_owner: &Arc<Mutex<TeacherSessionStagingV3>>,
        stage: &TeacherSessionStagingV3,
        session_owner: &Arc<Mutex<RoventCollectionTeacherSessionV3>>,
        session: &RoventCollectionTeacherSessionV3,
        source: &super::teacher::v3_session::TeacherActorEffectSourceV3,
    ) -> Result<Option<Self>, &'static str> {
        let Some(stage_proof) =
            OwnReceivedStageProofV3::try_from_guards(supervisor, recovery, stage_owner, stage)?
        else {
            return Ok(None);
        };
        let Some(original_session) = stage.session() else {
            return Ok(None);
        };
        if !Arc::ptr_eq(&original_session, session_owner)
            || !stage
                .ready_session_effect_identity_weak()
                .is_some_and(|id| Weak::ptr_eq(&id, &source.session_identity_weak()))
            || !session.matches_actor_effect_source(source)
        {
            return Err("Own /3 original actor session differs");
        }
        Ok(Some(Self {
            stage_proof,
            session: Arc::downgrade(session_owner),
            session_identity: source.session_identity_weak(),
            actor_identity: Arc::downgrade(source.actor_identity()),
        }))
    }
    pub(super) fn try_revalidate_with_guards(
        &self,
        supervisor: &OwnWorkerSupervisorV3,
        recovery: &OwnRootRecoveryV3,
        stage: &TeacherSessionStagingV3,
        session: &RoventCollectionTeacherSessionV3,
        source: &super::teacher::v3_session::TeacherActorEffectSourceV3,
    ) -> Result<Option<bool>, &'static str> {
        match self
            .stage_proof
            .try_revalidate_with_guards(supervisor, recovery)?
        {
            Some(true) => {}
            other => return Ok(other),
        }
        let original_session = self
            .session
            .upgrade()
            .ok_or("Own /3 actor session owner unavailable")?;
        if !Weak::ptr_eq(
            &self.stage_proof.scope_identity,
            &stage.scope_effect_identity_weak(),
        ) || !stage
            .session()
            .is_some_and(|original| Arc::ptr_eq(&original, &original_session))
            || !stage
                .ready_session_effect_identity_weak()
                .is_some_and(|id| Weak::ptr_eq(&id, &self.session_identity))
            || !self.matches_actor_source(source)
            || !session.matches_actor_effect_source(source)
        {
            return Ok(Some(false));
        }
        Ok(Some(true))
    }
}


/// One supervisor owns the pinned input and every root recovery. Its caller
/// chooses finite deadlines and must retain this owner across any failure.
pub(crate) struct OwnWorkerSupervisorV3 {
    original_self: OnceLock<Weak<Mutex<OwnWorkerSupervisorV3>>>,
    bridge: Arc<RoventOwnCollectionBridgeV3>,
    router: Arc<CaptureRouterV3>,
    pipeline: Option<PolicyRowPipelineOwnerV3>,
    pending_request: Option<OwnRootRequestV3>,
    pending_request_error: Option<String>,
    roots: Vec<Arc<Mutex<OwnRootRecoveryV3>>>,
    admissions: Vec<RoventCollectionRootAdmissionV3>,
    transfers: Vec<OwnPublishTransferV3>,
    deferred: Vec<Option<OwnDeferredOutcomeV3>>,
    poisoned_roots: Vec<bool>,
    startup_deadline: Instant,
    cleanup_deadline: Instant,
    abort_revoke_pending: bool,
    abort_revoke_error: Option<String>,
    dispatch_drain_pending: bool,
    dispatch_drain_error: Option<String>,
}

enum OwnPublishTransferV3 {
    Empty,
    Publishing,
    Returned(Result<OwnPrepareOutcomeV3, Box<dyn std::any::Any + Send>>),
}

enum OwnDeferredOutcomeV3 {
    Failure(String),
    Panic(Box<dyn std::any::Any + Send>),
}

impl OwnWorkerSupervisorV3 {
    fn matches_original_bridge(&self, bridge: &Arc<RoventOwnCollectionBridgeV3>) -> bool {
        Arc::ptr_eq(&self.bridge, bridge)
    }
    fn owns_recovery(&self, recovery: &Arc<Mutex<OwnRootRecoveryV3>>) -> bool {
        self.roots.iter().any(|owner| Arc::ptr_eq(owner, recovery))
    }

    /// Recovery-only identity check while the caller holds this supervisor's
    /// original guard. It gives no permission to start or complete an effect.
    pub(super) fn matches_bound_recovery(
        &self,
        bound: &Weak<Mutex<OwnWorkerSupervisorV3>>,
        bridge: &Arc<RoventOwnCollectionBridgeV3>,
        recovery: &Arc<Mutex<OwnRootRecoveryV3>>,
    ) -> bool {
        self.original_self
            .get()
            .is_some_and(|original| Weak::ptr_eq(original, bound) && original.upgrade().is_some())
            && self.matches_original_bridge(bridge)
            && self.owns_recovery(recovery)
    }
    /// Bind this exact externally retained supervisor to its game authority
    /// before starting the supervisor thread or admitting native roots.
    pub(crate) fn try_bind_game(supervisor: &Arc<Mutex<Self>>) -> Result<bool, &'static str> {
        let game = match supervisor.try_lock() {
            Ok(owner) => {
                let identity = Arc::downgrade(supervisor);
                if let Some(original) = owner.original_self.get() {
                    if !Weak::ptr_eq(original, &identity) { return Err("Own /3 supervisor identity differs"); }
                } else if owner.original_self.set(identity).is_err() {
                    return Err("Own /3 supervisor identity binding raced");
                }
                Arc::clone(owner.bridge.game())
            },
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err("Own /3 supervisor poisoned"),
        };
        match game.try_bind_supervisor(supervisor) {
            super::game_authority_v3::GameStepV3::Ready
            | super::game_authority_v3::GameStepV3::Existing => Ok(true),
            super::game_authority_v3::GameStepV3::Busy => Ok(false),
            _ => Err("Own /3 game supervisor binding rejected"),
        }
    }
    pub(crate) fn new(
        bridge: Arc<RoventOwnCollectionBridgeV3>,
        router: Arc<CaptureRouterV3>,
        startup_deadline: Instant,
        cleanup_deadline: Instant,
    ) -> Result<Self, &'static str> {
        if startup_deadline > cleanup_deadline
            || !router
                .registration()
                .same_original_owner(bridge.capture_registration())
        {
            return Err("Own /3 supervisor deadline or capture owner mismatch");
        }
        let mut roots = Vec::new();
        let mut admissions = Vec::new();
        let mut transfers = Vec::new();
        let mut deferred = Vec::new();
        let mut poisoned_roots = Vec::new();
        roots
            .try_reserve_exact(usize::from(bridge.inputs().plan().maximum_roots))
            .map_err(|_| "Own /3 supervisor root allocation")?;
        admissions
            .try_reserve_exact(usize::from(bridge.inputs().plan().maximum_roots))
            .map_err(|_| "Own /3 admission allocation")?;
        transfers
            .try_reserve_exact(usize::from(bridge.inputs().plan().maximum_roots))
            .map_err(|_| "Own /3 transfer allocation")?;
        deferred
            .try_reserve_exact(usize::from(bridge.inputs().plan().maximum_roots))
            .map_err(|_| "Own /3 deferred outcome allocation")?;
        poisoned_roots
            .try_reserve_exact(usize::from(bridge.inputs().plan().maximum_roots))
            .map_err(|_| "Own /3 poison observation allocation")?;
        Ok(Self {
            original_self: OnceLock::new(),
            bridge,
            router,
            pipeline: None,
            pending_request: None,
            pending_request_error: None,
            roots,
            admissions,
            transfers,
            deferred,
            poisoned_roots,
            startup_deadline,
            cleanup_deadline,
            abort_revoke_pending: false,
            abort_revoke_error: None,
            dispatch_drain_pending: false,
            dispatch_drain_error: None,
        })
    }

    /// One bounded supervisor pass, never invoked from the native manager.
    /// Failures retain every original root and do not skip its cleanup.
    pub(crate) fn step(&mut self) -> Result<(), String> {
        if self.bridge.game().abort_requested() {
            match self.bridge.try_request_game_abort() {
                Ok(done) => self.abort_revoke_pending = !done,
                Err(error) => {
                    self.abort_revoke_pending = true;
                    if self.abort_revoke_error.is_none() {
                        self.abort_revoke_error = Some(error);
                    }
                }
            }
        }
        match self.bridge.try_drain_existing_dispatch() {
            Ok(done) => self.dispatch_drain_pending = !done,
            Err(error) => {
                self.dispatch_drain_pending = true;
                if self.dispatch_drain_error.is_none() {
                    self.dispatch_drain_error = Some(error);
                }
            }
        }
        let pending_result =
            (|| -> Result<(), String> {
                if self.pending_request_error.is_some() {
                    return Ok(());
                }
                if self.pending_request.is_none()
                    && self.roots.len() < usize::from(self.bridge.inputs().plan().maximum_roots)
                {
                    self.pending_request = self.bridge.try_next_root().map_err(str::to_owned)?;
                }
                if let Some(request) = self.pending_request.as_ref() {
                    if self.pipeline.is_none() {
                        let writer = RoventCorpusWriterLockV3::new_shared(
                            self.bridge.inputs().manifest(),
                            &request.admission,
                        )
                        .map_err(|error| error.to_string())?;
                        let pipeline = PolicyRowPipelineOwnerV3::new(writer);
                        self.bridge
                            .register_pipeline(&pipeline)
                            .map_err(str::to_owned)?;
                        self.pipeline = Some(pipeline);
                    }
                    match self.bridge.game().try_bind_pipeline(
                        self.pipeline.as_ref().expect("original pipeline retained"),
                    ) {
                        super::game_authority_v3::GameStepV3::Ready
                        | super::game_authority_v3::GameStepV3::Existing => {}
                        super::game_authority_v3::GameStepV3::Busy => return Ok(()),
                        _ => return Err("Own /3 game pipeline binding rejected".into()),
                    }
                    if self
                        .admissions
                        .iter()
                        .any(|root| root.same_original_admission(&request.admission))
                    {
                        return Err("Own /3 duplicate original root".into());
                    }
                    let original = request.clone();
                    let original_admission = original.admission.clone();
                    let recovery = OwnRootRecoveryV3::new(
                        original,
                        Arc::clone(self.bridge.inputs()),
                        self.pipeline.as_ref().expect("pipeline retained").clone(),
                    );
                    self.roots.push(recovery);
                    self.admissions.push(original_admission);
                    self.transfers.push(OwnPublishTransferV3::Empty);
                    self.deferred.push(None);
                    self.poisoned_roots.push(false);
                    self.pending_request.take();
                }
                Ok(())
            })();
        if let Err(error) = pending_result {
            // Retain the actual request and its first preparation error while
            // continuing cleanup of independent, already admitted roots.
            self.pending_request_error = Some(error);
        }
        for index in 0..self.roots.len() {
            let original = Arc::clone(&self.roots[index]);
            if self.poisoned_roots[index] {
                continue;
            }
            if self.deferred[index].is_some() {
                let mut root = match original.try_lock() {
                    Ok(root) => root,
                    // Poison is unknown, not permission to take or discard
                    // the original deferred payload.
                    Err(std::sync::TryLockError::WouldBlock) => continue,
                    Err(std::sync::TryLockError::Poisoned(_)) => {
                        self.poisoned_roots[index] = true;
                        continue;
                    }
                };
                if matches!(self.deferred[index], Some(OwnDeferredOutcomeV3::Panic(_)))
                    && root.panic.is_some()
                    && root.additional_panics.try_reserve(1).is_err()
                {
                    continue;
                }
                if matches!(self.deferred[index], Some(OwnDeferredOutcomeV3::Failure(_)))
                    && root.failure.is_some()
                    && root.additional_failures.try_reserve(1).is_err()
                {
                    continue;
                }
                match self.deferred[index]
                    .take()
                    .expect("deferred outcome retained")
                {
                    OwnDeferredOutcomeV3::Failure(error) => {
                        if root.failure.is_none() {
                            root.failure = Some(error);
                        } else {
                            root.additional_failures.push(error);
                        }
                    }
                    OwnDeferredOutcomeV3::Panic(payload) => {
                        root.retain_panic(payload);
                        if root.failure.is_none() {
                            root.failure = Some(
                                "Own /3 supervisor step panicked; original state unknown".into(),
                            );
                        }
                    }
                }
                root.try_request_teacher_shutdown(self.cleanup_deadline);
                continue;
            }
            // Keep the exact transfer and poison state across an unwind while
            // allowing proof checks to borrow the whole supervisor.
            let mut transfer = std::mem::replace(&mut self.transfers[index], OwnPublishTransferV3::Empty);
            let mut poisoned = self.poisoned_roots[index];
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if let Ok(mut root) = original.try_lock() {
                    if root.consumer_failure.is_some() {
                        root.try_retain_known_rejection();
                    }
                    root.try_drain_closed_input();
                }
                prepare_own_root_step_v3(
                    self,
                    &self.bridge,
                    &self.router,
                    &original,
                    &mut transfer,
                    &mut poisoned,
                    self.startup_deadline,
                    self.cleanup_deadline,
                )
            }));
            self.transfers[index] = transfer;
            self.poisoned_roots[index] = poisoned;
            match outcome {
                Ok(Ok(OwnPrepareStepV3::Published)) => {}
                Ok(Ok(OwnPrepareStepV3::Pending)) => continue,
                Ok(Ok(OwnPrepareStepV3::Failed)) => {
                    match original.try_lock() {
                        Ok(mut root) => root.try_request_teacher_shutdown(self.cleanup_deadline),
                        Err(std::sync::TryLockError::WouldBlock) => {}
                        Err(std::sync::TryLockError::Poisoned(_)) => {
                            self.poisoned_roots[index] = true
                        }
                    }
                    continue;
                }
                Ok(Err(error)) => {
                    self.deferred[index] = Some(OwnDeferredOutcomeV3::Failure(error));
                    continue;
                }
                Err(payload) => {
                    self.deferred[index] = Some(OwnDeferredOutcomeV3::Panic(payload));
                    continue;
                }
            }
            let admission = match original.try_lock() {
                Ok(root) => root.request.admission.clone(),
                Err(std::sync::TryLockError::WouldBlock) => continue,
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    self.poisoned_roots[index] = true;
                    continue;
                }
            };
            // Joining this one original worker is observed separately from
            // teacher aggregate, target publication and native closure.
            if let Err(error) = self.bridge.try_join_root(&admission) {
                self.deferred[index] = Some(OwnDeferredOutcomeV3::Failure(error.into()));
                continue;
            }
            let already_unlinked = match original.try_lock() {
                Ok(root) => root.route_unlinked,
                Err(std::sync::TryLockError::WouldBlock) => continue,
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    self.poisoned_roots[index] = true;
                    continue;
                }
            };
            if !already_unlinked {
                match self.bridge.try_retire_route(&admission) {
                    Ok(Some(true)) => {}
                    Ok(Some(false) | None) => {}
                    Err(error) => {
                        self.deferred[index] =
                            Some(OwnDeferredOutcomeV3::Failure(error.to_string()));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Resume the same root's staged acquisition. No stage is manufactured from
/// a digest, and every failed/uncertain owner remains in the same recovery.
enum OwnScopeDriveV3 {
    Pending,
    Finished,
    Failed,
    Aborted,
}

/// All mutable effect slots were installed with this recovery before entry.
/// The caller holds the original supervisor/recovery/stage guards already.
fn drive_own_scope_effect_v3(
    supervisor: &OwnWorkerSupervisorV3,
    recovery: &mut OwnRootRecoveryV3,
    stage_owner: &Arc<Mutex<TeacherSessionStagingV3>>,
    stage: &mut TeacherSessionStagingV3,
) -> OwnScopeDriveV3 {
    use super::game_authority_v3::GameStepV3 as G;
    use super::teacher::v3_session::TeacherScopeEffectClaimFailureV3 as F;
    let owner = Arc::clone(&recovery.scope_effects);
    let mut slots = match owner.try_lock() {
        Ok(slots) => slots,
        Err(std::sync::TryLockError::WouldBlock) => return OwnScopeDriveV3::Pending,
        Err(_) => {
            recovery
                .failure
                .get_or_insert_with(|| "Own /3 scope slots poisoned".into());
            return OwnScopeDriveV3::Failed;
        }
    };
    let OwnScopeEffectSlotsV3 {
        proof,
        teacher_source,
        game_source,
        claim,
        outcome,
        issue_error,
        source_error,
        claim_error,
        completion_error,
        source_step,
        claim_step,
        completion_step,
        finished,
        ..
    } = &mut *slots;
    if *finished {
        // Effect settlement and a successful scope claim are separate. Keep
        // a known rejected original outcome rejected on every re-observation.
        return if matches!(outcome.as_ref(), Some(Ok(_))) {
            OwnScopeDriveV3::Finished
        } else {
            OwnScopeDriveV3::Failed
        };
    }
    if issue_error.is_some()
        || source_error.is_some()
        || claim_error.is_some()
        || completion_error.is_some()
    {
        return OwnScopeDriveV3::Failed;
    }
    let game = supervisor.bridge.game();
    if game.abort_requested() && claim.is_none() {
        return OwnScopeDriveV3::Aborted;
    }
    if game_source.is_none() {
        if proof.is_none() {
            match OwnReceivedStageProofV3::try_from_guards(supervisor, recovery, stage_owner, stage)
            {
                Ok(Some(original)) => *proof = Some(original),
                Ok(None) => return OwnScopeDriveV3::Pending,
                Err(error) => {
                    *source_error = Some(error);
                    return OwnScopeDriveV3::Failed;
                }
            }
        }
        if teacher_source.is_none() {
            if let Err(error) = stage.issue_scope_effect_source(&recovery.request.admission, teacher_source)
            {
                *issue_error = Some(error);
                return OwnScopeDriveV3::Failed;
            }
        }
        let step = game.try_issue_own_scope_source_in_slot(
            proof,
            teacher_source,
            supervisor,
            recovery,
            &recovery.request.admission,
            game_source,
            source_error,
        );
        *source_step = Some(step);
        match step {
            G::Ready => {}
            G::Busy => return OwnScopeDriveV3::Pending,
            G::Aborted | G::Closed => return OwnScopeDriveV3::Aborted,
            _ => return OwnScopeDriveV3::Failed,
        }
    }
    let Some(source) = game_source.as_ref() else {
        return OwnScopeDriveV3::Failed;
    };
    if claim.is_none() {
        let step = game.try_claim_own_scope(source, supervisor, recovery, claim, claim_error);
        *claim_step = Some(step);
        match step {
            G::Ready => {}
            G::Busy => return OwnScopeDriveV3::Pending,
            G::Aborted | G::Closed => return OwnScopeDriveV3::Aborted,
            _ => return OwnScopeDriveV3::Failed,
        }
    }
    let Some(original_claim) = claim.as_mut() else {
        return OwnScopeDriveV3::Failed;
    };
    if outcome.is_none() {
        let Some(teacher) = source.own_scope_source() else {
            return OwnScopeDriveV3::Failed;
        };
        let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            stage.claim_with_effect_source(teacher, original_claim, outcome)
        }));
        match attempted {
            Ok(true) => {}
            Ok(false) => {
                *completion_error = Some("Own /3 scope claim was not entered");
                return OwnScopeDriveV3::Failed;
            }
            Err(payload) => {
                recovery.retain_panic(payload);
                return OwnScopeDriveV3::Failed;
            }
        }
    }
    let (original_outcome, succeeded) = match outcome.as_ref() {
        Some(Ok(value)) => (value, true),
        Some(Err(F::Known { outcome, .. })) => (outcome, false),
        Some(Err(F::RejectedBeforeEntry(_))) | Some(Err(F::Unknown(_))) | None => {
            return OwnScopeDriveV3::Failed;
        }
    };
    let step = original_claim.try_complete_own_scope(
        source,
        original_outcome,
        supervisor,
        recovery,
        completion_error,
    );
    *completion_step = Some(step);
    match step {
        G::Ready | G::Existing => {
            *finished = true;
            if succeeded {
                OwnScopeDriveV3::Finished
            } else {
                OwnScopeDriveV3::Failed
            }
        }
        G::Busy => OwnScopeDriveV3::Pending,
        _ => OwnScopeDriveV3::Failed,
    }
}

/// One already-registered actor acquisition, separate from physical shutdown.
/// All four original guards are held by the caller; none is reacquired here.
fn drive_own_actor_effect_v3(
    supervisor: &OwnWorkerSupervisorV3,
    recovery: &mut OwnRootRecoveryV3,
    stage_owner: &Arc<Mutex<TeacherSessionStagingV3>>,
    stage: &TeacherSessionStagingV3,
    session_owner: &Arc<Mutex<RoventCollectionTeacherSessionV3>>,
    session: &mut RoventCollectionTeacherSessionV3,
    actor_slot: &Arc<Mutex<OwnActorEffectSlotsV3>>,
    startup_deadline: Instant,
    cleanup_deadline: Instant,
) -> OwnScopeDriveV3 {
    use super::game_authority_v3::GameStepV3 as G;
    use super::teacher::v3_session::TeacherActorEffectFailureV3 as F;
    let mut slots = match actor_slot.try_lock() {
        Ok(slots) => slots,
        Err(std::sync::TryLockError::WouldBlock) => return OwnScopeDriveV3::Pending,
        Err(_) => {
            recovery
                .failure
                .get_or_insert_with(|| "Own /3 actor slots poisoned".into());
            return OwnScopeDriveV3::Failed;
        }
    };
    let OwnActorEffectSlotsV3 {
        actor_id,
        proof,
        teacher_source,
        game_source,
        claim,
        outcome,
        issue_error,
        source_error,
        claim_error,
        completion_error,
        source_step,
        claim_step,
        completion_step,
        finished,
    } = &mut *slots;
    if *finished {
        return if matches!(outcome.as_ref(), Some(Ok(_))) {
            OwnScopeDriveV3::Finished
        } else {
            OwnScopeDriveV3::Failed
        };
    }
    if issue_error.is_some()
        || source_error.is_some()
        || claim_error.is_some()
        || completion_error.is_some()
    {
        return OwnScopeDriveV3::Failed;
    }
    let game = supervisor.bridge.game();
    if game.abort_requested() && claim.is_none() {
        return OwnScopeDriveV3::Aborted;
    }
    if game_source.is_none() {
        if teacher_source.is_none() {
            if let Err(error) =
                session.issue_actor_effect_source(&recovery.request.admission, actor_id, teacher_source)
            {
                *issue_error = Some(error);
                return OwnScopeDriveV3::Failed;
            }
        }
        if proof.is_none() {
            let Some(source) = teacher_source.as_ref() else {
                return OwnScopeDriveV3::Failed;
            };
            match OwnReceivedActorProofV3::try_from_guards(
                supervisor,
                recovery,
                stage_owner,
                stage,
                session_owner,
                session,
                source,
            ) {
                Ok(Some(original)) => *proof = Some(original),
                Ok(None) => return OwnScopeDriveV3::Pending,
                Err(error) => {
                    *source_error = Some(error);
                    return OwnScopeDriveV3::Failed;
                }
            }
        }
        let step = game.try_issue_own_actor_source_in_slot(
            proof,
            teacher_source,
            supervisor,
            recovery,
            stage,
            session,
            &recovery.request.admission,
            game_source,
            source_error,
        );
        *source_step = Some(step);
        match step {
            G::Ready => {}
            G::Busy => return OwnScopeDriveV3::Pending,
            G::Aborted | G::Closed => return OwnScopeDriveV3::Aborted,
            _ => return OwnScopeDriveV3::Failed,
        }
    }
    let Some(source) = game_source.as_ref() else {
        return OwnScopeDriveV3::Failed;
    };
    if claim.is_none() {
        let step = game.try_claim_own_actor(
            source,
            supervisor,
            recovery,
            stage,
            session,
            claim,
            claim_error,
        );
        *claim_step = Some(step);
        match step {
            G::Ready => {}
            G::Busy => return OwnScopeDriveV3::Pending,
            G::Aborted | G::Closed => return OwnScopeDriveV3::Aborted,
            _ => return OwnScopeDriveV3::Failed,
        }
    }
    let Some(original_claim) = claim.as_mut() else {
        return OwnScopeDriveV3::Failed;
    };
    if outcome.is_none() {
        let Some(teacher) = source.own_actor_source() else {
            return OwnScopeDriveV3::Failed;
        };
        if !session.start_actor_with_effect_source(
            teacher,
            original_claim,
            startup_deadline,
            cleanup_deadline,
            &mut recovery.custodian,
            outcome,
        ) {
            *completion_error = Some("Own /3 actor start was not entered");
            return OwnScopeDriveV3::Failed;
        }
    }
    let (original_outcome, succeeded) = match outcome.as_ref() {
        Some(Ok((_, value))) => (value, true),
        Some(Err(F::Known { outcome, .. })) => (outcome, false),
        Some(Err(F::RejectedBeforeEntry(_))) | Some(Err(F::Unknown(_))) | None => {
            return OwnScopeDriveV3::Failed;
        }
    };
    let step = original_claim.try_complete_own_actor(
        source,
        original_outcome,
        supervisor,
        recovery,
        stage,
        session,
        completion_error,
    );
    *completion_step = Some(step);
    match step {
        G::Ready | G::Existing => {
            *finished = true;
            if succeeded {
                OwnScopeDriveV3::Finished
            } else {
                OwnScopeDriveV3::Failed
            }
        }
        G::Busy => OwnScopeDriveV3::Pending,
        _ => OwnScopeDriveV3::Failed,
    }
}


fn prepare_own_root_step_v3(
    supervisor: &OwnWorkerSupervisorV3,
    bridge: &Arc<RoventOwnCollectionBridgeV3>,
    router: &Arc<CaptureRouterV3>,
    recovery: &Arc<Mutex<OwnRootRecoveryV3>>,
    transfer: &mut OwnPublishTransferV3,
    recovery_poisoned: &mut bool,
    startup_deadline: Instant,
    cleanup_deadline: Instant,
) -> Result<OwnPrepareStepV3, String> {
    if matches!(transfer, OwnPublishTransferV3::Publishing) {
        // An interrupted call can never be reissued as if no publication ran.
        return Ok(OwnPrepareStepV3::Pending);
    }
    if let OwnPublishTransferV3::Returned(outcome) = transfer {
        let mut state = match recovery.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnPrepareStepV3::Pending),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                *recovery_poisoned = true;
                return Ok(OwnPrepareStepV3::Pending);
            }
        };
        let has_panic = matches!(outcome, Err(_) | Ok(OwnPrepareOutcomeV3::Panicked { .. }));
        if has_panic && state.panic.is_some() && state.additional_panics.try_reserve(1).is_err() {
            return Ok(OwnPrepareStepV3::Pending);
        }
        if matches!(outcome, Ok(OwnPrepareOutcomeV3::Rejected { .. }))
            && state.failure.is_some()
            && state.additional_failures.try_reserve(1).is_err()
        {
            return Ok(OwnPrepareStepV3::Pending);
        }
        let OwnPublishTransferV3::Returned(outcome) =
            std::mem::replace(transfer, OwnPublishTransferV3::Empty)
        else {
            unreachable!("returned transfer was checked");
        };
        return match outcome {
            Ok(OwnPrepareOutcomeV3::Prepared) => {
                state.published = true;
                Ok(OwnPrepareStepV3::Published)
            }
            Ok(OwnPrepareOutcomeV3::Busy(original)) => {
                state.prepared = Some(original);
                Ok(OwnPrepareStepV3::Pending)
            }
            Ok(OwnPrepareOutcomeV3::Rejected { reason, original }) => {
                state.prepared = Some(original);
                if state.failure.is_none() {
                    state.failure = Some(reason);
                } else {
                    state.additional_failures.push(reason);
                }
                Ok(OwnPrepareStepV3::Failed)
            }
            Ok(OwnPrepareOutcomeV3::Panicked { original, payload }) => {
                state.prepared = Some(original);
                state.retain_panic(payload);
                if state.failure.is_none() {
                    state.failure = Some("Own /3 root preparation panicked".into());
                }
                Ok(OwnPrepareStepV3::Failed)
            }
            Err(payload) => {
                state.retain_panic(payload);
                if state.failure.is_none() {
                    state.failure = Some("Own /3 prepared publication outcome unknown".into());
                }
                Ok(OwnPrepareStepV3::Failed)
            }
        };
    }
    let (admission, registered) = match recovery.try_lock() {
        Ok(root) => (root.request.admission.clone(), root.recovery_published),
        Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnPrepareStepV3::Pending),
        Err(std::sync::TryLockError::Poisoned(_)) => {
            *recovery_poisoned = true;
            return Ok(OwnPrepareStepV3::Pending);
        }
    };
    if !registered {
        let attached = if bridge.game().abort_requested() {
            bridge.try_publish_aborted_recovery(supervisor, &admission, recovery)
        } else {
            bridge.try_publish_recovery(&admission, recovery)
        }
        .map_err(str::to_owned)?;
        if !attached {
            return Ok(OwnPrepareStepV3::Pending);
        }
    }
    let mut state = match recovery.try_lock() {
        Ok(state) => state,
        Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnPrepareStepV3::Pending),
        Err(std::sync::TryLockError::Poisoned(_)) => {
            *recovery_poisoned = true;
            return Ok(OwnPrepareStepV3::Pending);
        }
    };
    state.recovery_published = true;
    if state.published {
        return Ok(OwnPrepareStepV3::Published);
    }
    // Finish each previously acquired claim before root observation: an
    // observation error or expiry cannot erase an already issued effect.
    let scope_active = match state.scope_effects.try_lock() {
        Ok(slots) => slots.claim.is_some() && !slots.finished,
        Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnPrepareStepV3::Pending),
        Err(_) => return Err("Own /3 scope slots poisoned".into()),
    };
    if scope_active {
        let stage_owner = state.stage.as_ref().map(Arc::clone)
            .ok_or("Own /3 active scope lost stage owner")?;
        let mut stage = match stage_owner.try_lock() {
            Ok(stage) => stage,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnPrepareStepV3::Pending),
            Err(_) => return Err("Own /3 stage poisoned".into()),
        };
        if matches!(drive_own_scope_effect_v3(supervisor, &mut state, &stage_owner, &mut stage), OwnScopeDriveV3::Failed) {
            state.failure.get_or_insert_with(|| "Own /3 active scope outcome unresolved".into());
        }
        return Ok(OwnPrepareStepV3::Pending);
    }
    if let Some(actor_slot) = state.actor_effects.get(state.next_actor).cloned() {
        let active = match actor_slot.try_lock() {
            Ok(slots) => slots.claim.is_some() && !slots.finished,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnPrepareStepV3::Pending),
            Err(_) => return Err("Own /3 actor slots poisoned".into()),
        };
        if active {
            let stage_owner = state.stage.as_ref().map(Arc::clone)
                .ok_or("Own /3 active actor lost stage owner")?;
            let stage = match stage_owner.try_lock() {
                Ok(stage) => stage,
                Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnPrepareStepV3::Pending),
                Err(_) => return Err("Own /3 stage poisoned".into()),
            };
            let session_owner = stage.session().ok_or("Own /3 active actor lost session owner")?;
            let mut session = match session_owner.try_lock() {
                Ok(session) => session,
                Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnPrepareStepV3::Pending),
                Err(_) => return Err("Own /3 session poisoned".into()),
            };
            if matches!(drive_own_actor_effect_v3(supervisor, &mut state, &stage_owner, &stage, &session_owner, &mut session, &actor_slot, startup_deadline, cleanup_deadline), OwnScopeDriveV3::Failed) {
                state.failure.get_or_insert_with(|| "Own /3 active actor outcome unresolved".into());
            }
            return Ok(OwnPrepareStepV3::Pending);
        }
    }
    let root_expired = match admission.root_admission().try_accepts_original() {
        Ok(value) => !value,
        Err(error) if error.code == rz_contracts::ErrorCode::ResourceExhausted => return Ok(OwnPrepareStepV3::Pending),
        Err(error) => { state.failure = Some(format!("Own /3 root observation failed: {error}")); return Ok(OwnPrepareStepV3::Failed); }
    };
    if bridge.game().abort_requested() || state.request.is_revoked() || root_expired {
        state.failure.get_or_insert_with(|| "Own /3 original request revoked; retain staged owners".into());
        return Ok(OwnPrepareStepV3::Failed);
    }
    if state.failure.is_some() {
        return Ok(OwnPrepareStepV3::Failed);
    }
    if state.request.is_revoked() {
        state.failure = Some("Own /3 original request revoked; retain staged owners".into());
        return Ok(OwnPrepareStepV3::Failed);
    }
    if !router
        .registration()
        .same_original_owner(bridge.capture_registration())
    {
        state.failure = Some("Own /3 capture owner changed".into());
        return Ok(OwnPrepareStepV3::Failed);
    }
    for teacher in [false, true] {
        match bridge.try_reserve_root(&admission, recovery, teacher) {
            Ok(OwnReserveOutcomeV3::Reserved | OwnReserveOutcomeV3::AlreadyReserved) => {}
            Ok(OwnReserveOutcomeV3::Busy | OwnReserveOutcomeV3::Exhausted) => {
                return Ok(OwnPrepareStepV3::Pending);
            }
            Err(error) => {
                state.failure = Some(error.into());
                return Ok(OwnPrepareStepV3::Failed);
            }
        }
    }
    let Some(startup) = bridge.startup() else {
        return Ok(OwnPrepareStepV3::Pending);
    };
    if state.processed.is_none() {
        let capacity = state.inputs.plan().teacher_tickets_per_root as usize;
        match TeacherProcessedOwnerV3::new(admission.clone(), capacity) {
            Ok(owner) => state.processed = Some(owner),
            Err(error) => {
                state.consumer_failure = Some(error);
                state.failure = Some("Own /3 processed completion owner unavailable".into());
                return Ok(OwnPrepareStepV3::Failed);
            }
        }
    }
    if state.retained_policy.is_none() {
        let capacity = usize::try_from(state.inputs.plan().maximum_records)
            .map_err(|_| "Own /3 policy retention capacity overflow")?;
        match PolicyRootRetainedOwnerV3::try_new(&state.pipeline, admission.clone(), capacity) {
            Ok(Some(owner)) => state.retained_policy = Some(owner),
            Ok(None) => return Ok(OwnPrepareStepV3::Pending),
            Err(error) => {
                state.failure = Some(error.into());
                return Ok(OwnPrepareStepV3::Failed);
            }
        }
    }
    if state.stage.is_none() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            RoventCollectionTeacherSessionV3::prepare_staged(
                &admission,
                startup,
                state.inputs.manifest(),
                state.inputs.roster(),
            )
        }));
        match result {
            Ok(Ok(stage)) => {
                state.original_scope_identity = Some(stage.scope_effect_identity_weak());
                state.stage = Some(Arc::new(Mutex::new(stage)));
            },
            Ok(Err(error)) => {
                state.preflight_failure = Some(error);
                state.failure = Some("Own /3 teacher preflight rejected".into());
                return Ok(OwnPrepareStepV3::Failed);
            }
            Err(payload) => {
                state.retain_panic(payload);
                state.failure = Some("Own /3 teacher preflight panicked; state unknown".into());
                return Ok(OwnPrepareStepV3::Failed);
            }
        }
    }
    let stage = Arc::clone(state.stage.as_ref().expect("original stage retained"));
    {
        let mut staged = match stage.try_lock() {
            Ok(staged) => staged,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnPrepareStepV3::Pending),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err("Own /3 teacher stage poisoned".into());
            }
        };
        let scope_pending = match state.scope_effects.try_lock() {
            Ok(slots) => slots.claim.is_some() && !slots.finished,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnPrepareStepV3::Pending),
            Err(_) => return Err("Own /3 scope slots poisoned".into()),
        };
        if staged.state() == TeacherSessionStageStateV3::Prepared || scope_pending {
            match drive_own_scope_effect_v3(supervisor, &mut state, &stage, &mut staged) {
                OwnScopeDriveV3::Finished => {}
                OwnScopeDriveV3::Pending | OwnScopeDriveV3::Aborted => return Ok(OwnPrepareStepV3::Pending),
                OwnScopeDriveV3::Failed => {
                    state.failure.get_or_insert_with(|| "Own /3 teacher scope effect unresolved".into());
                    return Ok(OwnPrepareStepV3::Failed);
                }
            }
        }
        if !matches!(staged.state(), TeacherSessionStageStateV3::Claimed | TeacherSessionStageStateV3::Ready) {
            state.failure = Some("Own /3 teacher scope unresolved".into());
            return Ok(OwnPrepareStepV3::Failed);
        }
        if staged.state() == TeacherSessionStageStateV3::Claimed && staged.finish().is_err() {
            state.failure = Some("Own /3 teacher session assembly failed".into());
            return Ok(OwnPrepareStepV3::Failed);
        }
    }
    let session = match stage.try_lock() {
        Ok(staged) => staged.session().ok_or("Own /3 session absent")?,
        Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnPrepareStepV3::Pending),
        Err(std::sync::TryLockError::Poisoned(_)) => {
            return Err("Own /3 stage poisoned".into());
        }
    };
    if state.next_actor < state.inputs.plan().teacher_slots.len() {
        if state.actor_effects.len() == state.next_actor {
            if state.actor_effects.try_reserve(1).is_err() { return Ok(OwnPrepareStepV3::Pending); }
            let actor = state.inputs.plan().teacher_slots[state.next_actor].actor_id.clone();
            state.actor_effects.push(Arc::new(Mutex::new(OwnActorEffectSlotsV3::new(actor))));
        }
        let actor_slot = state.actor_effects.get(state.next_actor).cloned().ok_or("Own /3 original actor slot absent")?;
        let stage_guard = match stage.try_lock() {
            Ok(value) => value,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnPrepareStepV3::Pending),
            Err(_) => return Err("Own /3 stage poisoned".into()),
        };
        let mut owner = match session.try_lock() {
            Ok(value) => value,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnPrepareStepV3::Pending),
            Err(_) => return Err("Own /3 session poisoned".into()),
        };
        match drive_own_actor_effect_v3(supervisor, &mut state, &stage, &stage_guard, &session, &mut owner, &actor_slot, startup_deadline, cleanup_deadline) {
            OwnScopeDriveV3::Finished => state.next_actor += 1,
            OwnScopeDriveV3::Pending | OwnScopeDriveV3::Aborted => {},
            OwnScopeDriveV3::Failed => {
                state.failure.get_or_insert_with(|| "Own /3 actor effect unresolved".into());
                return Ok(OwnPrepareStepV3::Failed);
            }
        }
        return Ok(OwnPrepareStepV3::Pending);
    }
    if state.request.is_revoked() || bridge.game().abort_requested() {
        state.failure = Some("Own /3 root revoked during teacher preparation".into());
        return Ok(OwnPrepareStepV3::Failed);
    }
    if state.route.is_none() {
        let installed = bridge
            .capture_registration()
            .try_install_speculative_root_v3(
                &admission,
                state.inputs.plan().teacher_mailbox_capacity as usize,
                state.inputs.plan().teacher_tickets_per_root as usize,
                state.request.revocation_flag(),
            )
            .map_err(|error| error.to_string())?;
        let Some((route, receiver)) = installed else {
            return Ok(OwnPrepareStepV3::Pending);
        };
        state.route = Some(route);
        state.receiver = Some(Arc::new(Mutex::new(receiver)));
    }
    if state.owner.is_none() {
        state.owner = Some(
            RoventCollectionOwnerLockV3::new_shared(
                Arc::clone(state.inputs.manifest()),
                state.inputs.plan().metadata.clone(),
                Arc::clone(router),
            )
            .map_err(|error| error.to_string())?,
        );
    }
    if state.pump.is_none() {
        let receiver = Arc::clone(state.receiver.as_ref().expect("original receiver retained"));
        state.pump_unknown = true;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            TeacherPumpV3::new(
                Arc::clone(&session),
                receiver,
                state.inputs.plan().teacher_constraint,
                Duration::from_nanos(state.inputs.plan().teacher_soft_quantum_ns),
            )
        }));
        match result {
            Ok(Ok(pump)) => {
                state.pump = Some(pump);
                state.pump_unknown = false;
            }
            Ok(Err(error)) => {
                state.pump_error = Some(error);
                state.pump_unknown = false;
                state.failure = Some("Own /3 teacher pump setup rejected".into());
                return Ok(OwnPrepareStepV3::Failed);
            }
            Err(payload) => {
                state.retain_panic(payload);
                state.failure = Some("Own /3 teacher pump setup unknown".into());
                return Ok(OwnPrepareStepV3::Failed);
            }
        }
    }
    if state.closed_custody.is_none() {
        if state.closed_custody_setup_unknown {
            state.failure = Some("Own /3 closed custody setup outcome unknown".into());
            return Ok(OwnPrepareStepV3::Failed);
        }
        state.closed_custody_setup_unknown = true;
        let pump = state.pump.as_ref().expect("original pump retained");
        let route = state.route.as_ref().expect("original route retained");
        match TeacherClosedCustodyV3::try_new(
            pump,
            route,
            state.inputs.plan().teacher_tickets_per_root as usize,
        ) {
            Ok(TeacherClosedCustodyStartV3::Ready(custody)) => {
                state.closed_custody = Some(custody);
                state.closed_custody_setup_unknown = false;
            }
            Ok(TeacherClosedCustodyStartV3::Busy) => {
                state.closed_custody_setup_unknown = false;
                return Ok(OwnPrepareStepV3::Pending);
            }
            Err(error) => {
                state.closed_custody_setup_unknown = false;
                state.failure = Some(error.to_string());
                return Ok(OwnPrepareStepV3::Failed);
            }
        }
    }
    if state.worker.is_none() && state.prepared.is_none() {
        if state.worker_stage.is_none() {
            match RoventBoardCollectionWorkerStageV3::prepare(admission.clone()) {
                Ok(stage) => {
                    state.worker_key = Some(stage.recovery_key());
                    state.worker_stage = Some(stage);
                    // The original stage and key are retained before spawn.
                    return Ok(OwnPrepareStepV3::Pending);
                }
                Err(error) => {
                    state.worker_prepare_failure = Some(error);
                    state.failure = Some("Own /3 root worker preparation failed".into());
                    return Ok(OwnPrepareStepV3::Failed);
                }
            }
        }
        // A failed spawn retains its closure in this same recovery. Use a
        // Weak back-reference so that retained task cannot form an Arc cycle.
        let root = Arc::downgrade(recovery);
        let task = Box::new(move |receiver: RoventBoardRootHandoffReceiverV3| {
            let mut handoff = None;
            while Instant::now() < cleanup_deadline {
                let Some(original_root) = root.upgrade() else {
                    break;
                };
                if let Ok(mut original) = original_root.try_lock() {
                    let pump = if original.pump_failure.is_none() {
                        original.pump.as_ref().map(Arc::clone)
                    } else {
                        None
                    };
                    let failure = if let Some(pump) = pump {
                        let value = match pump.try_lock() {
                            Ok(mut owner) => owner.try_pump().err(),
                            Err(_) => None,
                        };
                        value
                    } else {
                        None
                    };
                    if let Some(error) = failure {
                        original.pump_failure_count = original.pump_failure_count.saturating_add(1);
                        if original.pump_failure.is_none() {
                            original.pump_failure = Some(error);
                        }
                    }
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        original.try_drive_consumer()
                    }));
                    if let Err(payload) = outcome {
                        original.retain_panic(payload);
                        if original.consumer_failure.is_none() {
                            original.consumer_failure = Some(invalid(
                                "Own /3 policy consumer outcome unknown after panic",
                            ));
                        }
                    }
                }
                match receiver.recv_timeout(Duration::from_millis(10)) {
                    Ok(Some(value)) => {
                        handoff = Some(value);
                        break;
                    }
                    Ok(None) => {}
                    Err(_) => break,
                }
            }
            if let Some(handoff) = handoff {
                while Instant::now() < cleanup_deadline {
                    let Some(original_root) = root.upgrade() else {
                        break;
                    };
                    let complete = match original_root.try_lock() {
                        Ok(mut original) => original.try_drain_teacher(&handoff, cleanup_deadline),
                        Err(_) => false,
                    };
                    if complete {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
            Box::new(root) as Box<dyn std::any::Any + Send>
        });
        let stage = state
            .worker_stage
            .as_mut()
            .expect("original worker stage retained");
        match stage.start(task) {
            Ok(RoventBoardWorkerStageStartV3::Started) => {
                let Some(worker) = stage.take_worker() else {
                    state.failure = Some("Own /3 started worker custody unknown".into());
                    return Ok(OwnPrepareStepV3::Failed);
                };
                state.worker = Some(worker);
            }
            Ok(RoventBoardWorkerStageStartV3::Failed) => {
                // The stage retains the original spawn error and task.
                state.failure = Some("Own /3 root worker spawn failed".into());
                return Ok(OwnPrepareStepV3::Failed);
            }
            Err(rejection) => {
                state.worker_start_rejection = Some(rejection);
                state.failure = Some("Own /3 root worker start rejected".into());
                return Ok(OwnPrepareStepV3::Failed);
            }
        }
    }
    if state.prepared.is_none() {
        state.prepared = Some(OwnPreparedRootV3 {
            admission: admission.clone(),
            owner: state
                .owner
                .as_ref()
                .expect("original observer owner")
                .clone(),
            route: state.route.as_ref().expect("original route").clone(),
            worker: state.worker.take().expect("original worker"),
            stage,
            pump: Arc::clone(state.pump.as_ref().expect("original pump")),
            pipeline: state.pipeline.clone(),
            recovery: Arc::downgrade(recovery),
        });
    }
    let prepared = state.prepared.take().expect("original prepared root");
    *transfer = OwnPublishTransferV3::Publishing;
    drop(state);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        bridge.try_publish_prepared(prepared)
    }));
    *transfer = OwnPublishTransferV3::Returned(outcome);
    Ok(OwnPrepareStepV3::Pending)
}

impl OwnRootRecoveryV3 {
    fn retain_panic(&mut self, payload: Box<dyn std::any::Any + Send>) {
        if self.panic.is_none() {
            self.panic = Some(payload);
        } else {
            self.additional_panics.push(payload);
        }
    }

    /// Run only on the retained root worker. Busy and not-ready observations
    /// preserve the original consumer, completion, target and output owners.
    fn try_drive_consumer(&mut self) {
        if !self.published {
            return;
        }
        if self.consumer_failure.is_some() {
            self.try_retain_known_rejection();
            return;
        }
        let Some(stage) = self.stage.as_ref().map(Arc::clone) else {
            return;
        };
        let session = match stage.try_lock() {
            Ok(stage) => stage.session(),
            Err(_) => None,
        };
        let (
            Some(session),
            Some(pump),
            Some(route),
            Some(observer),
            Some(processed),
            Some(retained),
        ) = (
            session,
            self.pump.as_ref().map(Arc::clone),
            self.route.as_ref(),
            self.owner.as_ref(),
            self.processed.as_ref(),
            self.retained_policy.as_ref(),
        )
        else {
            return;
        };
        if self.consumer.is_none() {
            match TeacherPolicyConsumerOwnerV3::try_new(
                &session,
                &pump,
                route,
                observer,
                &self.pipeline,
                processed,
                retained,
            ) {
                Ok(TeacherPolicyConsumerStartV3::Ready(consumer)) => self.consumer = Some(consumer),
                Ok(TeacherPolicyConsumerStartV3::Busy | TeacherPolicyConsumerStartV3::NotReady) => {
                    return;
                }
                Err(error) => {
                    self.consumer_failure = Some(error);
                    return;
                }
            }
        }
        let consumer = self.consumer.as_ref().expect("consumer retained").clone();
        match consumer.try_matches_registered_owners(
            &session,
            &pump,
            route,
            observer,
            &self.pipeline,
            processed,
            retained,
        ) {
            Ok(Some(true)) => {}
            Ok(None) => return,
            Ok(Some(false)) => {
                self.consumer_failure =
                    Some(invalid("Own /3 original policy consumer owner changed"));
                return;
            }
            Err(error) => {
                self.consumer_failure = Some(invalid(error));
                return;
            }
        }
        let mut consumer = match consumer.try_lock() {
            Ok(Some(consumer)) => consumer,
            Ok(None) => return,
            Err(error) => {
                self.consumer_failure = Some(invalid(error));
                return;
            }
        };
        use super::teacher::v3_policy_consumer::TeacherPolicyConsumerV3;
        let steps: [fn(
            &mut TeacherPolicyConsumerV3,
        ) -> Result<TeacherPolicyConsumerProgressV3, ArenaError>; 9] = [
            TeacherPolicyConsumerV3::try_transfer_completion,
            TeacherPolicyConsumerV3::try_seal,
            TeacherPolicyConsumerV3::try_transfer_retired,
            TeacherPolicyConsumerV3::try_prepare_sealed,
            TeacherPolicyConsumerV3::try_prepare_candidate,
            TeacherPolicyConsumerV3::try_transfer_capability,
            TeacherPolicyConsumerV3::try_append_policy,
            TeacherPolicyConsumerV3::try_retain_appended,
            TeacherPolicyConsumerV3::try_retain_processed_completion,
        ];
        let Some(step) = steps.get(usize::from(self.consumer_phase)) else {
            self.consumer_failure = Some(invalid("Own /3 invalid consumer phase"));
            return;
        };
        match step(&mut consumer) {
            Ok(TeacherPolicyConsumerProgressV3::ChunkReady) => {
                if self.persistence_failure.is_some() {
                    return;
                }
                let mut pipeline = match self.pipeline.try_lock() {
                    Ok(Some(pipeline)) => pipeline,
                    Ok(None) => return,
                    Err(_) => {
                        self.persistence_failure = Some("Own /3 output owner poisoned");
                        return;
                    }
                };
                match pipeline.persist_chunk() {
                    PolicyChunkPersistenceV3::Persisted | PolicyChunkPersistenceV3::NotReady => {}
                    PolicyChunkPersistenceV3::Failed => {
                        self.persistence_failure = Some("Own /3 output write failed");
                    }
                    PolicyChunkPersistenceV3::Unknown => {
                        self.persistence_failure = Some("Own /3 output write outcome unknown");
                    }
                }
                // Re-observe append on the next worker pass after the
                // original pipeline has retained the persistence result.
            }
            Ok(TeacherPolicyConsumerProgressV3::Rejected) => {
                if (3..=6).contains(&self.consumer_phase) {
                    self.rejection_scope =
                        consumer
                            .original_completion_scope()
                            .and_then(|(admission, ticket)| {
                                admission
                                    .same_original_admission(&self.request.admission)
                                    .then(|| (admission.clone(), ticket))
                            });
                }
                self.consumer_failure = Some(invalid(
                    "Own /3 policy consumer rejected; original owner retained",
                ));
            }
            Ok(TeacherPolicyConsumerProgressV3::Unknown) => {
                self.consumer_failure = Some(invalid(
                    "Own /3 policy consumer outcome unknown; original owner retained",
                ));
            }
            Ok(
                TeacherPolicyConsumerProgressV3::Busy
                | TeacherPolicyConsumerProgressV3::NotReady
                | TeacherPolicyConsumerProgressV3::OutputHeld,
            ) => {}
            Ok(progress) => {
                use TeacherPolicyConsumerProgressV3 as Progress;
                let advanced = matches!(
                    (self.consumer_phase, progress),
                    (0, Progress::CompletionStored)
                        | (1, Progress::Sealed)
                        | (2, Progress::RowTransferred)
                        | (3, Progress::Prepared)
                        | (4, Progress::CandidateReady)
                        | (5, Progress::CapabilityStored)
                        | (6, Progress::Appended)
                        | (7, Progress::AppendedEvidenceStored)
                        | (8, Progress::CompletionStored)
                );
                if advanced {
                    self.consumer_phase = (self.consumer_phase + 1) % 9;
                } else {
                    self.consumer_failure =
                        Some(invalid("Own /3 unexpected consumer phase outcome"));
                }
            }
            Err(error) => {
                self.consumer_failure = Some(error);
            }
        }
    }

    /// Preserve a known row rejection in its original bounded root owner so
    /// another root can continue. An uncertain writer is never moved here.
    fn try_retain_known_rejection(&mut self) {
        if self.rejection_retained || self.rejection_retention_failure.is_some() {
            return;
        }
        let Some(retained) = &self.retained_policy else {
            return;
        };
        let Some((admission, ticket)) = &self.rejection_scope else {
            return;
        };
        let mut pipeline = match self.pipeline.try_lock() {
            Ok(Some(pipeline)) => pipeline,
            Ok(None) => return,
            Err(error) => {
                self.rejection_retention_failure = Some(invalid(error));
                return;
            }
        };
        match pipeline.try_retain_known_rejection(retained, admission, ticket) {
            Ok(PolicyRetentionOutcomeV3::Transferred) => self.rejection_retained = true,
            Ok(
                PolicyRetentionOutcomeV3::Busy
                | PolicyRetentionOutcomeV3::NotReady
                | PolicyRetentionOutcomeV3::OtherRoot
                | PolicyRetentionOutcomeV3::Capacity,
            ) => {}
            Err(error) => self.rejection_retention_failure = Some(error),
        }
    }
    /// Best-effort cancellation of this same root after preparation failure.
    /// This is only a request: owner, workers and unresolved results remain
    /// retained until their own physical and handoff evidence is collected.
    fn try_request_teacher_shutdown(&mut self, deadline: Instant) {
        if let Some(route) = &self.route {
            route.cancel_all();
            match route.try_revoke_all_targets() {
                Ok(Some(())) => self.target_revoke_pending = false,
                Ok(None) => self.target_revoke_pending = true,
                Err(error) => {
                    self.target_revoke_pending = true;
                    if self.target_revoke_error.is_none() {
                        self.target_revoke_error = Some(error.to_string());
                    }
                }
            }
        }
        if let Some(pump) = &self.pump {
            if let Ok(mut pump) = pump.try_lock() {
                pump.close_admission();
                self.pump_admission_closed = true;
            }
        }
        let Some(stage) = self.stage.as_ref() else {
            return;
        };
        let session = match stage.try_lock() {
            Ok(stage) => stage.session(),
            Err(_) => None,
        };
        if let Some(session) = session {
            if let Ok(mut session) = session.try_lock() {
                if let Err(error) = session.request_shutdown(deadline) {
                    self.shutdown_failure_count = self.shutdown_failure_count.saturating_add(1);
                    if self.shutdown_failure.is_none() {
                        self.shutdown_failure = Some(error);
                    }
                }
            }
        }
    }

    /// Retain at most one undispatched original capsule per pass in the same
    /// pump. Empty is only a queue observation; sender fencing, target revoke
    /// and physical completion remain independent.
    fn try_drain_closed_input(&mut self) {
        if !self.pump_admission_closed {
            return;
        }
        let Some(pump) = self.pump.as_ref().map(Arc::clone) else {
            return;
        };
        let mut pump = match pump.try_lock() {
            Ok(pump) => pump,
            Err(std::sync::TryLockError::WouldBlock) => {
                self.closed_input_pending = true;
                return;
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                self.closed_input_pending = true;
                self.closed_input_error = Some("Own /3 closed pump poisoned".into());
                return;
            }
        };
        if self.closed_input_error.is_none() {
            match pump.try_closed_drain() {
                Ok(TeacherClosedDrainV3::EmptyNow | TeacherClosedDrainV3::Disconnected) => {
                    self.closed_input_pending = false;
                }
                Ok(
                    TeacherClosedDrainV3::Retained
                    | TeacherClosedDrainV3::InputHeld
                    | TeacherClosedDrainV3::Busy
                    | TeacherClosedDrainV3::PendingDispatch
                    | TeacherClosedDrainV3::Capacity,
                ) => self.closed_input_pending = true,
                Err(error) => {
                    self.closed_input_pending = true;
                    self.closed_input_error = Some(error);
                }
            }
        }
        if self.closed_handoff_error.is_none() {
            let (Some(route), Some(custody)) = (&self.route, &self.closed_custody) else {
                self.closed_handoff_pending = true;
                return;
            };
            match pump.try_handoff_closed(route, custody) {
                Ok(TeacherClosedHandoffV3::Empty) => self.closed_handoff_pending = false,
                Ok(
                    TeacherClosedHandoffV3::Transferred
                    | TeacherClosedHandoffV3::Busy
                    | TeacherClosedHandoffV3::Capacity
                    | TeacherClosedHandoffV3::RevokeRequired,
                ) => {
                    self.closed_handoff_pending = true;
                }
                Err(error) => {
                    self.closed_handoff_pending = true;
                    self.closed_handoff_error = Some(error.to_string());
                }
            }
        }
    }

    /// Retire the actual teacher scope only after all physical workers join;
    /// a pending query, unconsumed completion or missing aggregate stays held.
    fn try_drain_teacher(&mut self, handoff: &RoventBoardRootHandoffV3, deadline: Instant) -> bool {
        self.try_drive_consumer();
        self.try_request_teacher_shutdown(deadline);
        self.try_drain_closed_input();
        let Some(stage) = self.stage.as_ref().map(Arc::clone) else {
            return false;
        };
        let session = match stage.try_lock() {
            Ok(stage) => stage.session(),
            Err(_) => None,
        };
        let Some(session) = session else {
            return false;
        };
        if let Some(route) = &self.route {
            route.cancel_all();
        }
        let Some(pump) = self.pump.as_ref().map(Arc::clone) else {
            return false;
        };
        {
            let Ok(mut pump) = pump.try_lock() else {
                return false;
            };
            pump.close_admission();
            match pump.try_pump() {
                Err(error) => {
                    self.pump_failure_count = self.pump_failure_count.saturating_add(1);
                    if self.pump_failure.is_none() {
                        self.pump_failure = Some(error);
                    }
                }
                Ok(_) => {}
            }
            // Semantic output may be held while other actors terminate.
            // Keep their original terminals in the bounded pump backlog and
            // independently attempt the physical worker join below. This does
            // not retire the route or mark the held output as consumed.
            if let Err(error) = pump.try_drain_pending() {
                self.pump_failure_count = self.pump_failure_count.saturating_add(1);
                if self.pump_failure.is_none() {
                    self.pump_failure = Some(error);
                }
            }
        }
        let Ok(mut session) = session.try_lock() else {
            return false;
        };
        if let Err(error) = session.request_shutdown(deadline) {
            self.shutdown_failure_count = self.shutdown_failure_count.saturating_add(1);
            if self.shutdown_failure.is_none() {
                self.shutdown_failure = Some(error);
            }
            return false;
        }
        let actors: Vec<String> = session.registration().teachers.keys().cloned().collect();
        for actor in actors {
            if self.diagnostics.try_reserve(1).is_err() {
                return false;
            }
            if let Some(lifecycle) = session.lifecycle_mut(&actor) {
                let _ = lifecycle.poll();
                if let Some(diagnostics) = lifecycle.take_diagnostics_after_join() {
                    self.diagnostics.push((actor.clone(), diagnostics));
                }
            }
            if !matches!(session.finish_worker(&actor), Ok(true)) {
                return false;
            }
        }
        if self.aggregate.is_none() {
            match session.try_aggregate() {
                Ok(Some(value)) => self.aggregate = Some(value),
                _ => return false,
            }
        }
        let Ok(mut original) = handoff.try_lock() else {
            return false;
        };
        let Some(aggregate) = self.aggregate.take() else {
            return false;
        };
        match original.submit_teacher_aggregate_ack(aggregate) {
            Ok(()) => true,
            Err(aggregate) => {
                self.aggregate = Some(aggregate);
                false
            }
        }
    }
}
