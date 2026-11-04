//! Original-owner staging for a V3 teacher completion and later policy work.
//!
//! The root supervisor retains this same Arc before the worker starts and
//! through actual worker join, including Busy, errors, or uncertain claims.
//! Transferring a completion here is not a target, publication or fence.

use super::super::observer_lock_v3::{
    OriginalRouteMatchV3, PipelinePreparationV3, RetiredTransferV3, RoventCollectionOwnerLockV3,
};
use super::super::policy_row_pipeline_v3::{
    PolicyPipelineProgressV3, PolicyPrepareOutcomeV3, PolicyRowPipelineOwnerV3,
};
use super::super::speculative_route_v3::{SpeculativeRootV3, TargetCapabilityTakeV3};
use super::super::{ArenaError, MAX_CORPUS_ROWS, RoventChunkPacket, invalid};
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
    ChunkReady,
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
    processed: TeacherProcessedOwnerV3,
    authority_unknown: bool,
}

#[derive(Clone)]
pub(crate) struct TeacherPolicyConsumerOwnerV3(Arc<Mutex<TeacherPolicyConsumerV3>>);

struct TeacherProcessedV3 {
    admission: RoventCollectionRootAdmissionV3,
    capacity: usize,
    originals: Vec<TeacherCompletedV3>,
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
                processed: processed.clone(),
                authority_unknown: false,
            }),
        ))))
    }

    pub(crate) fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
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
        Ok(match self.observer.try_prepare_pipeline(&self.pipeline)? {
            PipelinePreparationV3::Ready => TeacherPolicyConsumerProgressV3::CandidateReady,
            PipelinePreparationV3::Busy => TeacherPolicyConsumerProgressV3::Busy,
            PipelinePreparationV3::NotReady => TeacherPolicyConsumerProgressV3::NotReady,
            PipelinePreparationV3::Failed => TeacherPolicyConsumerProgressV3::Rejected,
        })
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
        if !self.prepared_transferred {
            return Ok(TeacherPolicyConsumerProgressV3::NotReady);
        }
        let ticket = completed.capsule.ticket();
        let mut pipeline = match self.pipeline.try_lock().map_err(invalid)? {
            Some(guard) => guard,
            None => return Ok(TeacherPolicyConsumerProgressV3::Busy),
        };
        let Some(destination) = pipeline.empty_capability_slot() else {
            return Ok(TeacherPolicyConsumerProgressV3::OutputHeld);
        };
        Ok(
            match self
                .route
                .try_take_target_capability_in_slot(&ticket, destination)?
            {
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
        )
    }

    /// The caller retains the chunk output slot and this consumer owner. A
    /// returned append does not prove durable storage or physical shutdown.
    pub(crate) fn try_append_policy(
        &mut self,
        chunk: &mut Option<RoventChunkPacket>,
    ) -> Result<TeacherPolicyConsumerProgressV3, ArenaError> {
        if self.authority_unknown {
            return Ok(TeacherPolicyConsumerProgressV3::Unknown);
        }
        if !self.prepared_transferred {
            return Ok(TeacherPolicyConsumerProgressV3::NotReady);
        }
        let mut pipeline = match self.pipeline.try_lock().map_err(invalid)? {
            Some(guard) => guard,
            None => return Ok(TeacherPolicyConsumerProgressV3::Busy),
        };
        Ok(match pipeline.try_append(chunk) {
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
        if processed.originals.len() >= processed.capacity {
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
        self.seal_state = TeacherPolicySealStepV3::Ready;
        Ok(TeacherPolicyConsumerProgressV3::CompletionStored)
    }
}
