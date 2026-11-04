//! Original Own /3 root preparation custody, shared with its supervisor.
//! This is not a worker completion, teacher result, or publication proof.

use super::observer_lock_v3::RoventCollectionOwnerLockV3;
use super::own_hooks_lock_v3::OwnRootRequestV3;
use super::speculative_route_v3::{SpeculativeReceiverV3, SpeculativeRootV3};
use super::teacher::v3_pump::{TeacherPumpSetupErrorV3, TeacherPumpV3};
use super::teacher::v3_session::TeacherSessionStagingV3;
use rz_uci::pals_adaptive::{
    RoventBoardCollectionWorkerKeyV3, RoventBoardCollectionWorkerSpawnFailureV3,
    RoventCollectionRootAdmissionV3,
};
use rz_uci::rovent_external_launch::TeacherRecoveryCustodianV3;
use std::sync::{Arc, Mutex};

/// Publish this Arc to the bridge before claiming a teacher scope, opening a
/// route, or starting a worker. All uncertain original owners stay together.
pub(crate) struct OwnRootRecoveryV3 {
    pub(super) request: OwnRootRequestV3,
    pub(super) stage: Option<Arc<Mutex<TeacherSessionStagingV3>>>,
    pub(super) route: Option<SpeculativeRootV3>,
    pub(super) receiver: Option<Arc<Mutex<SpeculativeReceiverV3>>>,
    pub(super) owner: Option<RoventCollectionOwnerLockV3>,
    pub(super) pump: Option<Arc<Mutex<TeacherPumpV3>>>,
    pub(super) pump_error: Option<TeacherPumpSetupErrorV3>,
    pub(super) pump_unknown: bool,
    pub(super) worker_key: Option<RoventBoardCollectionWorkerKeyV3>,
    pub(super) worker_failure: Option<RoventBoardCollectionWorkerSpawnFailureV3>,
    pub(super) custodian: TeacherRecoveryCustodianV3,
    pub(super) panic: Option<Box<dyn std::any::Any + Send>>,
}

impl OwnRootRecoveryV3 {
    pub(crate) fn new(request: OwnRootRequestV3) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            request,
            stage: None,
            route: None,
            receiver: None,
            owner: None,
            pump: None,
            pump_error: None,
            pump_unknown: false,
            worker_key: None,
            worker_failure: None,
            custodian: TeacherRecoveryCustodianV3::new(),
            panic: None,
        }))
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
        key: &RoventBoardCollectionWorkerKeyV3,
    ) -> Result<Option<bool>, &'static str> {
        let basic = !self.request.is_revoked()
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
