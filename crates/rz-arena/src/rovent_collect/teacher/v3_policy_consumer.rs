//! Original-owner staging for a V3 teacher completion and later policy work.
//!
//! The root supervisor retains this same Arc before the worker starts and
//! through actual worker join, including Busy, errors, or uncertain claims.
//! Transferring a completion here is not a target, publication or fence.

use super::super::observer_lock_v3::{
    OriginalRouteMatchV3, PipelineCapabilityV3, PipelinePreparationV3, RetiredTransferV3,
    RoventCollectionOwnerLockV3,
};
use super::super::policy_row_pipeline_v3::{
    PolicyPipelineProgressV3, PolicyPrepareOutcomeV3, PolicyRetentionOutcomeV3,
    PolicyRootRetainedOwnerV3, PolicyRowPipelineOwnerV3,
};
use super::super::speculative_route_v3::{SpeculativeRootV3, TargetCapabilityTakeV3};
use super::super::{ArenaError, MAX_CORPUS_ROWS, invalid};
use super::v3_dispatch::TeacherCompletedV3;
use super::v3_policy::{
    RoventSealedTeacherPolicyV3, TeacherPolicySealRejectionV3, TeacherPolicySealStepV3,
    seal_teacher_policy_in_slots_v3,
};
use super::v3_pump::TeacherPumpV3;
use super::v3_session::RoventCollectionTeacherSessionV3;
use rz_uci::pals_adaptive::RoventCollectionRootAdmissionV3;
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};

pub(crate) enum TeacherPolicyConsumerProgressV3 {
    Busy,
    NotReady,
    CompletionStored,
    OutputHeld,
    Sealed,
    Prepared,
    RowTransferred,
    CandidateReady,
    CapabilityStored,
    Appended,
    AppendedEvidenceStored,
    ChunkReady,
    /// The original revoked row and capability are held by this root's
    /// retained owner; the original completion is still held here.
    CanceledRetained,
    Rejected,
    Unknown,
}

pub(crate) struct TeacherPolicyConsumerV3 {
    session: Arc<Mutex<RoventCollectionTeacherSessionV3>>,
    pump: Arc<Mutex<TeacherPumpV3>>,
    route: SpeculativeRootV3,
    observer: RoventCollectionOwnerLockV3,
    pipeline: PolicyRowPipelineOwnerV3,
    completed: Option<TeacherCompletedV3>,
    sealed: Option<RoventSealedTeacherPolicyV3>,
    rejection: Option<Box<TeacherPolicySealRejectionV3>>,
    panic: Option<Box<dyn std::any::Any + Send>>,
    seal_state: TeacherPolicySealStepV3,
    prepared_transferred: bool,
    policy_appended: bool,
    appended_retained: bool,
    appended_in_output: bool,
    processed: TeacherProcessedOwnerV3,
    retained: PolicyRootRetainedOwnerV3,
    pc_revoked_retention_pending: bool,
    pc_revoked_retained: bool,
    authority_unknown: bool,
}

#[derive(Clone)]
pub(crate) struct TeacherPolicyConsumerOwnerV3(Arc<Mutex<TeacherPolicyConsumerV3>>);

struct TeacherProcessedV3 {
    admission: RoventCollectionRootAdmissionV3,
    capacity: usize,
    originals: Vec<TeacherCompletedV3>,
    /// Canceled tickets are not successful policy publications.
    revoked: Vec<TeacherCompletedV3>,
}

fn completion_retention_has_space(successful: usize, revoked: usize, capacity: usize) -> bool {
    successful.saturating_add(revoked) < capacity
}

#[cfg(test)]
mod completion_retention_tests {
    use super::completion_retention_has_space;

    #[test]
    fn successful_and_revoked_completions_share_the_root_bound() {
        assert!(completion_retention_has_space(1, 1, 3));
        assert!(!completion_retention_has_space(2, 1, 3));
        assert!(!completion_retention_has_space(usize::MAX, 1, 3));
    }
}

/// The same bounded original-completion owner must be held by root recovery
/// before a worker starts. Storing here is not teacher-worker retirement.
#[derive(Clone)]
pub(crate) struct TeacherProcessedOwnerV3(Arc<Mutex<TeacherProcessedV3>>);

impl TeacherProcessedOwnerV3 {
    pub(crate) fn new(
        admission: RoventCollectionRootAdmissionV3,
        capacity: usize,
    ) -> Result<Self, ArenaError> {
        if !(1..=MAX_CORPUS_ROWS).contains(&capacity) {
            return Err(invalid("V3 retained completion capacity invalid"));
        }
        Ok(Self(Arc::new(Mutex::new(TeacherProcessedV3 {
            admission,
            capacity,
            originals: Vec::new(),
            revoked: Vec::new(),
        }))))
    }
    pub(crate) fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

pub(crate) enum TeacherPolicyConsumerStartV3 {
    Ready(TeacherPolicyConsumerOwnerV3),
    Busy,
    NotReady,
}

/// Failure-only observation of the same retained consumer; none of these
/// fields confers permission to retry, release owners, or claim completion.
pub(crate) struct TeacherPolicyConsumerRetentionV3 {
    pub(crate) poisoned: bool,
    pub(crate) completed_held: bool,
    pub(crate) sealed_held: bool,
    pub(crate) rejection_held: bool,
    pub(crate) panic_held: bool,
    pub(crate) seal_state: TeacherPolicySealStepV3,
    pub(crate) authority_unknown: bool,
}

impl TeacherPolicyConsumerOwnerV3 {
    /// Verify the exact pump/session/receiver/root/pipeline owners before
    /// constructing a consumer. None is transient lock contention. The
    /// borrowed inputs remain caller-owned on every failure.
    pub(crate) fn try_new(
        session: &Arc<Mutex<RoventCollectionTeacherSessionV3>>,
        pump: &Arc<Mutex<TeacherPumpV3>>,
        route: &SpeculativeRootV3,
        observer: &RoventCollectionOwnerLockV3,
        pipeline: &PolicyRowPipelineOwnerV3,
        processed: &TeacherProcessedOwnerV3,
        retained: &PolicyRootRetainedOwnerV3,
    ) -> Result<TeacherPolicyConsumerStartV3, ArenaError> {
        let receiver = {
            let pump = match pump.try_lock() {
                Ok(guard) => guard,
                Err(TryLockError::WouldBlock) => return Ok(TeacherPolicyConsumerStartV3::Busy),
                Err(TryLockError::Poisoned(_)) => return Err(invalid("V3 teacher pump poisoned")),
            };
            if !Arc::ptr_eq(pump.session(), session) {
                return Err(invalid("V3 consumer pump/session owner differs"));
            }
            Arc::clone(pump.receiver())
        };
        {
            let receiver = match receiver.try_lock() {
                Ok(guard) => guard,
                Err(TryLockError::WouldBlock) => return Ok(TeacherPolicyConsumerStartV3::Busy),
                Err(TryLockError::Poisoned(_)) => {
                    return Err(invalid("V3 teacher receiver poisoned"));
                }
            };
            if !receiver.matches_original_route(route) {
                return Err(invalid("V3 consumer receiver/root route differs"));
            }
        }
        {
            let session = match session.try_lock() {
                Ok(guard) => guard,
                Err(TryLockError::WouldBlock) => return Ok(TeacherPolicyConsumerStartV3::Busy),
                Err(TryLockError::Poisoned(_)) => {
                    return Err(invalid("V3 teacher session poisoned"));
                }
            };
            if !route.matches_admission(session.original_admission()) {
                return Err(invalid("V3 consumer session/root admission differs"));
            }
        }
        {
            let processed = match processed.0.try_lock() {
                Ok(guard) => guard,
                Err(TryLockError::WouldBlock) => return Ok(TeacherPolicyConsumerStartV3::Busy),
                Err(TryLockError::Poisoned(_)) => {
                    return Err(invalid("V3 processed completion owner poisoned"));
                }
            };
            if !route.matches_admission(&processed.admission) {
                return Err(invalid("V3 processed completion root differs"));
            }
        }
        match retained
            .try_matches(pipeline, route.admission())
            .map_err(invalid)?
        {
            Some(true) => {}
            Some(false) => return Err(invalid("V3 policy retention root or pipeline differs")),
            None => return Ok(TeacherPolicyConsumerStartV3::Busy),
        }
        match observer.try_matches_original_route(route)? {
            OriginalRouteMatchV3::Match => {}
            OriginalRouteMatchV3::Different => {
                return Err(invalid("V3 consumer observer/root route differs"));
            }
            OriginalRouteMatchV3::Busy => return Ok(TeacherPolicyConsumerStartV3::Busy),
            OriginalRouteMatchV3::NotReady => return Ok(TeacherPolicyConsumerStartV3::NotReady),
        }
        if observer.try_bind_policy_pipeline(pipeline)?.is_none() {
            return Ok(TeacherPolicyConsumerStartV3::Busy);
        }
        Ok(TeacherPolicyConsumerStartV3::Ready(Self(Arc::new(
            Mutex::new(TeacherPolicyConsumerV3 {
                session: Arc::clone(session),
                pump: Arc::clone(pump),
                route: route.clone(),
                observer: observer.clone(),
                pipeline: pipeline.clone(),
                completed: None,
                sealed: None,
                rejection: None,
                panic: None,
                seal_state: TeacherPolicySealStepV3::Ready,
                prepared_transferred: false,
                policy_appended: false,
                appended_retained: false,
                appended_in_output: false,
                processed: processed.clone(),
                retained: retained.clone(),
                pc_revoked_retention_pending: false,
                pc_revoked_retained: false,
                authority_unknown: false,
            }),
        ))))
    }

    pub(crate) fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Identity only: the recovery owner must compare this consumer with its
    /// originally registered root, actor, observer, game pipeline and output
    /// owners. A match does not establish a worker join or any fence.
    pub(crate) fn try_matches_registered_owners(
        &self,
        session: &Arc<Mutex<RoventCollectionTeacherSessionV3>>,
        pump: &Arc<Mutex<TeacherPumpV3>>,
        route: &SpeculativeRootV3,
        observer: &RoventCollectionOwnerLockV3,
        pipeline: &PolicyRowPipelineOwnerV3,
        processed: &TeacherProcessedOwnerV3,
        retained: &PolicyRootRetainedOwnerV3,
    ) -> Result<Option<bool>, &'static str> {
        let Some(owner) = self.try_lock()? else {
            return Ok(None);
        };
        Ok(Some(
            Arc::ptr_eq(&owner.session, session)
                && Arc::ptr_eq(&owner.pump, pump)
                && owner.route.same_original_route(route)
                && owner.observer.same_original_owner(observer)
                && owner.pipeline.same_owner(pipeline)
                && owner.processed.same_owner(processed)
                && owner.retained.same_owner(retained),
        ))
    }

    pub(crate) fn try_lock(
        &self,
    ) -> Result<Option<MutexGuard<'_, TeacherPolicyConsumerV3>>, &'static str> {
        match self.0.try_lock() {
            Ok(guard) => Ok(Some(guard)),
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Poisoned(_)) => Err("V3 teacher consumer owner poisoned"),
        }
    }

    /// Even a poisoned mutex retains its original slots. Observe them without
    /// clearing poison or handing out mutable recovery authority.
    pub(crate) fn try_observe_retention(&self) -> Option<TeacherPolicyConsumerRetentionV3> {
        let (owner, poisoned) = match self.0.try_lock() {
            Ok(guard) => (guard, false),
            Err(TryLockError::WouldBlock) => return None,
            Err(TryLockError::Poisoned(error)) => (error.into_inner(), true),
        };
        Some(TeacherPolicyConsumerRetentionV3 {
            poisoned,
            completed_held: owner.completed.is_some(),
            sealed_held: owner.sealed.is_some(),
            rejection_held: owner.rejection.is_some(),
            panic_held: owner.panic.is_some(),
            seal_state: owner.seal_state,
            authority_unknown: owner.authority_unknown,
        })
    }
}

impl TeacherPolicyConsumerV3 {
    /// Transfer only into the still-live original consumer slot. The pump's
    /// output remains in the pump on contention or when this slot is full.
    pub(crate) fn try_transfer_completion(
        &mut self,
    ) -> Result<TeacherPolicyConsumerProgressV3, ArenaError> {
        if self.authority_unknown {
            return Ok(TeacherPolicyConsumerProgressV3::Unknown);
        }
        if self.completed.is_some() || self.sealed.is_some() || self.rejection.is_some() {
            return Ok(TeacherPolicyConsumerProgressV3::OutputHeld);
        }
        let mut pump = match self.pump.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return Ok(TeacherPolicyConsumerProgressV3::Busy),
            Err(TryLockError::Poisoned(_)) => return Err(invalid("V3 teacher pump poisoned")),
        };
        if !Arc::ptr_eq(pump.session(), &self.session) {
            return Err(invalid("V3 consumer original session changed"));
        }
        if pump.transfer_completed(&mut self.completed) {
            Ok(TeacherPolicyConsumerProgressV3::CompletionStored)
        } else {
            Ok(TeacherPolicyConsumerProgressV3::NotReady)
        }
    }

    /// No pump/observer/registration lock is held while checking the same
    /// session. Busy leaves every original slot in this retained owner.
    pub(crate) fn try_seal(&mut self) -> Result<TeacherPolicyConsumerProgressV3, ArenaError> {
        if self.authority_unknown {
            return Ok(TeacherPolicyConsumerProgressV3::Unknown);
        }
        if self.completed.is_none() {
            return Ok(TeacherPolicyConsumerProgressV3::NotReady);
        }
        if self.prepared_transferred {
            return Ok(TeacherPolicyConsumerProgressV3::OutputHeld);
        }
        let mut session = match self.session.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return Ok(TeacherPolicyConsumerProgressV3::Busy),
            Err(TryLockError::Poisoned(_)) => return Err(invalid("V3 teacher session poisoned")),
        };
        if !self.route.matches_admission(session.original_admission()) {
            return Err(invalid("V3 consumer original admission changed"));
        }
        let step = seal_teacher_policy_in_slots_v3(
            &mut session,
            &mut self.completed,
            &mut self.sealed,
            &mut self.rejection,
            &mut self.panic,
            &mut self.seal_state,
        );
        Ok(match step {
            TeacherPolicySealStepV3::Ready => TeacherPolicyConsumerProgressV3::NotReady,
            TeacherPolicySealStepV3::Busy => TeacherPolicyConsumerProgressV3::Busy,
            TeacherPolicySealStepV3::Sealed => TeacherPolicyConsumerProgressV3::Sealed,
            TeacherPolicySealStepV3::Rejected => TeacherPolicyConsumerProgressV3::Rejected,
            TeacherPolicySealStepV3::Unknown => TeacherPolicyConsumerProgressV3::Unknown,
        })
    }

    /// Observe the ticket of the original completion retained in this owner.
    /// This does not grant target authority or establish a completed row.
    /// Callers acquire this consumer through its nonblocking `try_lock`.
    pub(crate) fn original_completion_ticket(
        &self,
    ) -> Option<rz_contracts::pals_adaptive::RoleTicketEnvelope> {
        self.completed
            .as_ref()
            .map(|completed| completed.capsule.ticket())
    }

    /// Read the original completion's admission and ticket while this
    /// consumer guard remains held. This does not authorize publication.
    pub(crate) fn original_completion_scope(
        &self,
    ) -> Option<(
        &RoventCollectionRootAdmissionV3,
        rz_contracts::pals_adaptive::RoleTicketEnvelope,
    )> {
        self.completed
            .as_ref()
            .map(|completed| (completed.capsule.admission(), completed.capsule.ticket()))
    }

    /// The exact ticket selects its original retired row; a digest or the
    /// teacher completion alone does not establish retirement.
    pub(crate) fn sealed_ticket(&self) -> Option<rz_contracts::pals_adaptive::RoleTicketEnvelope> {
        self.sealed.as_ref().map(|sealed| sealed.capsule().ticket())
    }

    /// Transfer only the actual retired row for this same sealed ticket. An
    /// absent row or occupied pipeline stays a pending observation.
    pub(crate) fn try_transfer_retired(
        &mut self,
    ) -> Result<TeacherPolicyConsumerProgressV3, ArenaError> {
        if self.authority_unknown {
            return Ok(TeacherPolicyConsumerProgressV3::Unknown);
        }
        let Some(ticket) = self.sealed_ticket() else {
            return Ok(TeacherPolicyConsumerProgressV3::NotReady);
        };
        Ok(
            match self
                .observer
                .try_transfer_retired_to_pipeline(&ticket, &self.pipeline)?
            {
                RetiredTransferV3::Transferred => TeacherPolicyConsumerProgressV3::RowTransferred,
                RetiredTransferV3::Busy => TeacherPolicyConsumerProgressV3::Busy,
                RetiredTransferV3::NotFound => TeacherPolicyConsumerProgressV3::NotReady,
                RetiredTransferV3::SlotOccupied => TeacherPolicyConsumerProgressV3::OutputHeld,
            },
        )
    }

    /// Called only on the root's retained worker, after the original observer
    /// transferred the matching accepted row into this same pipeline owner.
    pub(crate) fn try_prepare_sealed(
        &mut self,
    ) -> Result<TeacherPolicyConsumerProgressV3, ArenaError> {
        if self.authority_unknown {
            return Ok(TeacherPolicyConsumerProgressV3::Unknown);
        }
        if self.prepared_transferred {
            return Ok(TeacherPolicyConsumerProgressV3::OutputHeld);
        }
        if self.sealed.is_none() {
            return Ok(TeacherPolicyConsumerProgressV3::NotReady);
        }
        let mut pipeline = match self.pipeline.try_lock().map_err(invalid)? {
            Some(guard) => guard,
            None => return Ok(TeacherPolicyConsumerProgressV3::Busy),
        };
        Ok(match pipeline.try_prepare_sealed(&mut self.sealed) {
            PolicyPrepareOutcomeV3::Prepared => {
                self.prepared_transferred = true;
                TeacherPolicyConsumerProgressV3::Prepared
            }
            PolicyPrepareOutcomeV3::NotReady => TeacherPolicyConsumerProgressV3::NotReady,
            PolicyPrepareOutcomeV3::Failed => TeacherPolicyConsumerProgressV3::Rejected,
            PolicyPrepareOutcomeV3::Unknown => {
                self.authority_unknown = true;
                TeacherPolicyConsumerProgressV3::Unknown
            }
        })
    }

    /// The observer takes its own lock before the already-registered pipeline
    /// lock. It preserves the original producer and does not acquire the gate.
    pub(crate) fn try_prepare_candidate(
        &mut self,
    ) -> Result<TeacherPolicyConsumerProgressV3, ArenaError> {
        if self.authority_unknown {
            return Ok(TeacherPolicyConsumerProgressV3::Unknown);
        }
        if !self.prepared_transferred {
            return Ok(TeacherPolicyConsumerProgressV3::NotReady);
        }
        let Some(completed) = self.completed.as_ref() else {
            return Ok(TeacherPolicyConsumerProgressV3::NotReady);
        };
        let ticket = completed.capsule.ticket();
        Ok(
            match self
                .observer
                .try_prepare_pipeline(&self.pipeline, &ticket)?
            {
                PipelinePreparationV3::Ready => TeacherPolicyConsumerProgressV3::CandidateReady,
                PipelinePreparationV3::Busy => TeacherPolicyConsumerProgressV3::Busy,
                PipelinePreparationV3::NotReady => TeacherPolicyConsumerProgressV3::NotReady,
                PipelinePreparationV3::Failed => TeacherPolicyConsumerProgressV3::Rejected,
                PipelinePreparationV3::Unknown => {
                    self.authority_unknown = true;
                    TeacherPolicyConsumerProgressV3::Unknown
                }
            },
        )
    }

    /// Move the exact ticket's unique target capability directly from its
    /// original route slot into the still-retained pipeline slot.
    pub(crate) fn try_transfer_capability(
        &mut self,
    ) -> Result<TeacherPolicyConsumerProgressV3, ArenaError> {
        if self.authority_unknown {
            return Ok(TeacherPolicyConsumerProgressV3::Unknown);
        }
        let Some(completed) = self.completed.as_ref() else {
            return Ok(TeacherPolicyConsumerProgressV3::NotReady);
        };
        let ticket = completed.capsule.ticket();
        if !self.prepared_transferred {
            return Ok(TeacherPolicyConsumerProgressV3::NotReady);
        }
        Ok(
            match self
                .observer
                .try_take_pipeline_capability(&self.pipeline, &ticket)?
            {
                PipelineCapabilityV3::Busy => TeacherPolicyConsumerProgressV3::Busy,
                PipelineCapabilityV3::NotReady => TeacherPolicyConsumerProgressV3::NotReady,
                PipelineCapabilityV3::Outcome(outcome) => match outcome {
                    TargetCapabilityTakeV3::Transferred => {
                        TeacherPolicyConsumerProgressV3::CapabilityStored
                    }
                    TargetCapabilityTakeV3::Busy => TeacherPolicyConsumerProgressV3::Busy,
                    TargetCapabilityTakeV3::AlreadyTaken => {
                        self.authority_unknown = true;
                        TeacherPolicyConsumerProgressV3::Unknown
                    }
                    TargetCapabilityTakeV3::Occupied => TeacherPolicyConsumerProgressV3::OutputHeld,
                },
            },
        )
    }

    /// The same game pipeline retains the chunk output slot. An append does
    /// not prove durable storage or physical shutdown.
    pub(crate) fn try_append_policy(
        &mut self,
    ) -> Result<TeacherPolicyConsumerProgressV3, ArenaError> {
        if self.authority_unknown {
            return Ok(TeacherPolicyConsumerProgressV3::Unknown);
        }
        if !self.prepared_transferred {
            return Ok(TeacherPolicyConsumerProgressV3::NotReady);
        }
        if self.policy_appended {
            return Ok(TeacherPolicyConsumerProgressV3::OutputHeld);
        }
        let Some(completed) = self.completed.as_ref() else {
            return Ok(TeacherPolicyConsumerProgressV3::NotReady);
        };
        let ticket = completed.capsule.ticket();
        let admission = completed.capsule.admission();
        let mut pipeline = match self.pipeline.try_lock().map_err(invalid)? {
            Some(guard) => guard,
            None => return Ok(TeacherPolicyConsumerProgressV3::Busy),
        };
        Ok(match pipeline.try_append(admission, &ticket) {
            PolicyPipelineProgressV3::Appended => {
                self.policy_appended = true;
                TeacherPolicyConsumerProgressV3::Appended
            }
            PolicyPipelineProgressV3::ChunkReady => TeacherPolicyConsumerProgressV3::ChunkReady,
            PolicyPipelineProgressV3::Busy => TeacherPolicyConsumerProgressV3::Busy,
            PolicyPipelineProgressV3::NotReady => TeacherPolicyConsumerProgressV3::NotReady,
            PolicyPipelineProgressV3::Failed => TeacherPolicyConsumerProgressV3::Rejected,
            PolicyPipelineProgressV3::Unknown => {
                self.authority_unknown = true;
                TeacherPolicyConsumerProgressV3::Unknown
            }
        })
    }

    /// P-C only: quarantine an exact revoked ticket in its original root
    /// while holding the same shared-pipeline guard that observed rejection.
    /// A busy/full retained owner leaves the original shared slot untouched;
    /// later calls retry retention and never append the rejected ticket again.
    pub(crate) fn try_append_policy_pc(
        &mut self,
    ) -> Result<TeacherPolicyConsumerProgressV3, ArenaError> {
        if self.authority_unknown {
            return Ok(TeacherPolicyConsumerProgressV3::Unknown);
        }
        if !self.prepared_transferred || self.policy_appended {
            return Ok(TeacherPolicyConsumerProgressV3::NotReady);
        }
        let Some(completed) = self.completed.as_ref() else {
            return Ok(TeacherPolicyConsumerProgressV3::NotReady);
        };
        let ticket = completed.capsule.ticket();
        let admission = completed.capsule.admission();
        let mut pipeline = match self.pipeline.try_lock().map_err(invalid)? {
            Some(guard) => guard,
            None => return Ok(TeacherPolicyConsumerProgressV3::Busy),
        };
        if !self.pc_revoked_retention_pending {
            match pipeline.try_append(admission, &ticket) {
                PolicyPipelineProgressV3::Appended => {
                    self.policy_appended = true;
                    return Ok(TeacherPolicyConsumerProgressV3::Appended);
                }
                PolicyPipelineProgressV3::ChunkReady => {
                    return Ok(TeacherPolicyConsumerProgressV3::ChunkReady);
                }
                PolicyPipelineProgressV3::Busy => return Ok(TeacherPolicyConsumerProgressV3::Busy),
                PolicyPipelineProgressV3::NotReady => {
                    return Ok(TeacherPolicyConsumerProgressV3::NotReady);
                }
                PolicyPipelineProgressV3::Unknown => {
                    self.authority_unknown = true;
                    return Ok(TeacherPolicyConsumerProgressV3::Unknown);
                }
                PolicyPipelineProgressV3::Failed => {
                    if !pipeline.known_revocation_matches(admission, &ticket) {
                        return Ok(TeacherPolicyConsumerProgressV3::Rejected);
                    }
                    self.pc_revoked_retention_pending = true;
                }
            }
        }
        if !pipeline.known_revocation_matches(admission, &ticket) {
            // The original rejection changed or vanished unexpectedly.
            self.authority_unknown = true;
            return Ok(TeacherPolicyConsumerProgressV3::Unknown);
        }
        Ok(
            match pipeline.try_retain_known_rejection(&self.retained, admission, &ticket)? {
                PolicyRetentionOutcomeV3::Transferred => {
                    self.pc_revoked_retention_pending = false;
                    self.pc_revoked_retained = true;
                    TeacherPolicyConsumerProgressV3::CanceledRetained
                }
                PolicyRetentionOutcomeV3::Busy => TeacherPolicyConsumerProgressV3::Busy,
                PolicyRetentionOutcomeV3::NotReady | PolicyRetentionOutcomeV3::OtherRoot => {
                    TeacherPolicyConsumerProgressV3::NotReady
                }
                PolicyRetentionOutcomeV3::Capacity => TeacherPolicyConsumerProgressV3::OutputHeld,
            },
        )
    }

    /// Transfer original appended evidence only into the same pipeline's
    /// typed game-output owner; no arbitrary destination is accepted.
    pub(crate) fn try_retain_appended(
        &mut self,
    ) -> Result<TeacherPolicyConsumerProgressV3, ArenaError> {
        if self.authority_unknown {
            return Ok(TeacherPolicyConsumerProgressV3::Unknown);
        }
        if !self.policy_appended {
            return Ok(TeacherPolicyConsumerProgressV3::NotReady);
        }
        if self.appended_retained {
            return Ok(TeacherPolicyConsumerProgressV3::OutputHeld);
        }
        let Some(completed) = self.completed.as_ref() else {
            return Ok(TeacherPolicyConsumerProgressV3::NotReady);
        };
        let admission = completed.capsule.admission();
        let ticket = completed.capsule.ticket();
        let mut pipeline = match self.pipeline.try_lock().map_err(invalid)? {
            Some(guard) => guard,
            None => return Ok(TeacherPolicyConsumerProgressV3::Busy),
        };
        if pipeline.is_uncertain() {
            self.authority_unknown = true;
            return Ok(TeacherPolicyConsumerProgressV3::Unknown);
        }
        if !self.appended_in_output {
            if !pipeline.transfer_appended(admission, &ticket) {
                return Ok(TeacherPolicyConsumerProgressV3::OutputHeld);
            }
            self.appended_in_output = true;
        }
        Ok(
            match pipeline.try_retain_completed(&self.retained, admission, &ticket)? {
                PolicyRetentionOutcomeV3::Transferred => {
                    self.appended_retained = true;
                    TeacherPolicyConsumerProgressV3::AppendedEvidenceStored
                }
                PolicyRetentionOutcomeV3::Busy => TeacherPolicyConsumerProgressV3::Busy,
                PolicyRetentionOutcomeV3::NotReady => TeacherPolicyConsumerProgressV3::NotReady,
                PolicyRetentionOutcomeV3::OtherRoot => TeacherPolicyConsumerProgressV3::NotReady,
                PolicyRetentionOutcomeV3::Capacity => TeacherPolicyConsumerProgressV3::OutputHeld,
            },
        )
    }

    /// Move the original completion only into the pre-registered root's
    /// bounded recovery owner after the same policy was appended. Busy or
    /// full leaves this consumer's original completion intact.
    pub(crate) fn try_retain_processed_completion(
        &mut self,
    ) -> Result<TeacherPolicyConsumerProgressV3, ArenaError> {
        if self.authority_unknown {
            return Ok(TeacherPolicyConsumerProgressV3::Unknown);
        }
        if !self.prepared_transferred
            || !self.policy_appended
            || !self.appended_retained
            || self.sealed.is_some()
            || self.rejection.is_some()
            || self.panic.is_some()
            || self.completed.is_none()
        {
            return Ok(TeacherPolicyConsumerProgressV3::NotReady);
        }
        let mut processed = match self.processed.0.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return Ok(TeacherPolicyConsumerProgressV3::Busy),
            Err(TryLockError::Poisoned(_)) => {
                return Err(invalid("V3 processed completion owner poisoned"));
            }
        };
        if !self.route.matches_admission(&processed.admission) {
            return Err(invalid("V3 processed completion original root changed"));
        }
        if !completion_retention_has_space(
            processed.originals.len(),
            processed.revoked.len(),
            processed.capacity,
        ) {
            return Ok(TeacherPolicyConsumerProgressV3::OutputHeld);
        }
        processed
            .originals
            .try_reserve(1)
            .map_err(|_| invalid("V3 processed completion storage allocation"))?;
        let original = self
            .completed
            .take()
            .expect("validated original completion");
        processed.originals.push(original);
        self.prepared_transferred = false;
        self.policy_appended = false;
        self.appended_retained = false;
        self.appended_in_output = false;
        self.seal_state = TeacherPolicySealStepV3::Ready;
        Ok(TeacherPolicyConsumerProgressV3::CompletionStored)
    }

    /// Preserve a revoked ticket's original completion separately from
    /// successfully appended policy completions. A row/capability retention
    /// transfer must already have succeeded for this same consumer ticket.
    pub(crate) fn try_retain_revoked_completion_pc(
        &mut self,
    ) -> Result<TeacherPolicyConsumerProgressV3, ArenaError> {
        if self.authority_unknown {
            return Ok(TeacherPolicyConsumerProgressV3::Unknown);
        }
        if !self.pc_revoked_retained
            || self.pc_revoked_retention_pending
            || !self.prepared_transferred
            || self.policy_appended
            || self.appended_retained
            || self.appended_in_output
            || self.sealed.is_some()
            || self.rejection.is_some()
            || self.panic.is_some()
            || self.completed.is_none()
        {
            return Ok(TeacherPolicyConsumerProgressV3::NotReady);
        }
        let mut processed = match self.processed.0.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return Ok(TeacherPolicyConsumerProgressV3::Busy),
            Err(TryLockError::Poisoned(_)) => {
                return Err(invalid("V3 revoked completion owner poisoned"));
            }
        };
        let completed = self.completed.as_ref().expect("checked completion");
        if !self.route.matches_admission(&processed.admission)
            || !completed
                .capsule
                .admission()
                .same_original_admission(&processed.admission)
        {
            return Err(invalid("V3 revoked completion original root changed"));
        }
        if !completion_retention_has_space(
            processed.originals.len(),
            processed.revoked.len(),
            processed.capacity,
        ) {
            return Ok(TeacherPolicyConsumerProgressV3::OutputHeld);
        }
        processed
            .revoked
            .try_reserve(1)
            .map_err(|_| invalid("V3 revoked completion storage allocation"))?;
        let original = self
            .completed
            .take()
            .expect("validated original completion");
        processed.revoked.push(original);
        self.pc_revoked_retained = false;
        self.prepared_transferred = false;
        self.seal_state = TeacherPolicySealStepV3::Ready;
        Ok(TeacherPolicyConsumerProgressV3::CompletionStored)
    }
}
