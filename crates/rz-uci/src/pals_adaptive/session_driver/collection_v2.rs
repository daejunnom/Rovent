//! Board-backed PCR collection uses the original loaded /2 owner and frontier.
//! These hooks are admission/coordination only; they are not a teacher fence.

use super::*;
use crate::pals_adaptive::AdaptiveSearchDriver as _;
use crate::pals_adaptive::{
    AdaptiveCollectionError, AdaptiveCollectionObserverBoxV2, AdaptiveCollectionV2ObserverStatus,
    RoventCollectionRootAdmissionV2, RoventNativeCollectionOriginV2,
};
use std::any::Any;

/// The collection owner must not perform blocking teacher or filesystem work
/// from these callbacks, which run while the actual session manager is held.
pub trait RoventBoardCollectionHooksV2<B>: Send + Sync
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    /// Bind the exact startup control without spawning or waiting.
    fn register_startup_control(&self, control: &crate::rovent_bootstrap::RoventStartupControl)
        -> Result<(), AdaptiveManagerError>;
    fn admit(
        &self,
        port: &mut RoventBoardCollectionAdmissionPortV2<'_, B>,
        admission: RoventCollectionRootAdmissionV2,
    ) -> Result<(), RoventBoardCollectionAdmissionFailureV2>;
    fn pump(&self) -> Result<(), AdaptiveManagerError>;
    fn revoke(&self);
    fn handoff_sender(&self) -> std::sync::mpsc::SyncSender<
        Arc<Mutex<RoventBoardCollectionHandoffV2<B::Output>>>,
    >;
}

/// Native root completion and the original observer; this does not attest
/// retirement of any separate teacher process.
pub struct RoventBoardCollectionHandoffV2<O> {
    pub(super) admission: RoventCollectionRootAdmissionV2,
    pub(super) observer: AdaptiveCollectionObserverBoxV2<AdaptivePortQuestion, O>,
    pub(super) settlement: Result<(), AdaptiveCollectionError>,
    pub(super) native_fence: AdaptiveShutdownReceipt,
}

impl<O> RoventBoardCollectionHandoffV2<O> {
    pub fn parts_mut(&mut self) -> (
        &RoventCollectionRootAdmissionV2,
        &mut AdaptiveCollectionObserverBoxV2<AdaptivePortQuestion, O>,
        &Result<(), AdaptiveCollectionError>,
        &AdaptiveShutdownReceipt,
    ) {
        (&self.admission, &mut self.observer, &self.settlement, &self.native_fence)
    }

    pub fn into_parts(self) -> (
        RoventCollectionRootAdmissionV2,
        AdaptiveCollectionObserverBoxV2<AdaptivePortQuestion, O>,
        Result<(), AdaptiveCollectionError>,
        AdaptiveShutdownReceipt,
    ) {
        (self.admission, self.observer, self.settlement, self.native_fence)
    }
}

/// Preserve the original admission and every acquired owner on a failed hook.
pub struct RoventBoardCollectionAdmissionFailureV2 {
    pub error: AdaptiveManagerError,
    pub admission: RoventCollectionRootAdmissionV2,
    pub owners: Box<dyn Any + Send>,
}

pub struct RoventBoardCollectionObserverRejectionV2<O> {
    pub error: AdaptiveManagerError,
    pub admission: RoventCollectionRootAdmissionV2,
    pub observer: AdaptiveCollectionObserverBoxV2<AdaptivePortQuestion, O>,
    pub observer_status: AdaptiveCollectionV2ObserverStatus,
    pub panic_payload: Option<Box<dyn Any + Send>>,
}

pub struct RoventBoardCollectionAdmissionPortV2<'a, B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    pub(super) manager: &'a mut AdaptiveSessionManager<B>,
}

impl<B> RoventBoardCollectionAdmissionPortV2<'_, B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    pub fn install(
        &mut self,
        admission: RoventCollectionRootAdmissionV2,
        observer: AdaptiveCollectionObserverBoxV2<AdaptivePortQuestion, B::Output>,
    ) -> Result<(), RoventBoardCollectionObserverRejectionV2<B::Output>> {
        self.manager
            .install_collection_observer_v2(admission, observer)
            .map_err(|failure| RoventBoardCollectionObserverRejectionV2 {
                error: failure.error,
                admission: failure.admission,
                observer: failure.observer,
                observer_status: failure.observer_status,
                panic_payload: failure.panic_payload,
            })
    }
}

/// Registration returns both original owners on rejection, rather than
/// dropping a hook that may itself own a teacher worker.
pub struct RoventBoardCollectionRegistrationRejectionV2<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    pub error: SearchSessionFailure,
    pub origin: RoventNativeCollectionOriginV2,
    pub hooks: Arc<dyn RoventBoardCollectionHooksV2<B>>,
}

impl<B> AdaptiveSessionDriver<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    /// Installs only the actual loaded /2 owner on the still-idle session.
    /// A copied digest or a separately loaded model cannot stand in for it.
    pub fn register_rovent_collection_v2(
        &mut self,
        origin: RoventNativeCollectionOriginV2,
        hooks: Arc<dyn RoventBoardCollectionHooksV2<B>>,
    ) -> Result<(), Box<RoventBoardCollectionRegistrationRejectionV2<B>>> {
        let valid = self.rovent_root_policy_v2.as_ref().is_some_and(|policy| {
            policy.owner.v2().is_some_and(|lock| {
                Arc::ptr_eq(lock, origin.sources().lock_owner())
                    && matches!(
                        lock.manifest.engine.researcher.as_ref(),
                        Some(rz_experiments::RoventResearcher::Own { .. })
                    )
            })
        }) && self.rovent_collection_origin.is_none()
            && self.rovent_collection_origin_v2.is_none()
            && self.rovent_collection_hooks_v2.is_none()
            && !self.closed.load(Ordering::Acquire)
            && self.state.try_lock().is_ok_and(|state| {
                matches!(&*state, SessionState::Idle(owners)
                    if owners.continuation.is_none() && owners.native_finished.is_none())
            });
        if !valid {
            return Err(Box::new(RoventBoardCollectionRegistrationRejectionV2 {
                error: failure("RoventBoardCollectionRegistrationV2", "original /2 lock or idle owner differs"),
                origin,
                hooks,
            }));
        }
        self.rovent_collection_origin_v2 = Some(origin);
        self.rovent_collection_hooks_v2 = Some(hooks);
        Ok(())
    }

    pub(super) fn admit_board_collection_v2(
        &self,
        manager: &mut AdaptiveSessionManager<B>,
        root: &RoventPcrRootAdmission,
        profile: &RoventPcrSessionProfile,
    ) -> Result<bool, AdaptiveManagerError> {
        let Some(origin) = &self.rovent_collection_origin_v2 else {
            return Ok(false);
        };
        let hooks = self.rovent_collection_hooks_v2.as_ref().ok_or(
            AdaptiveManagerError::Invalid("V2 board collection hook owner is absent"),
        )?;
        let original = root.original_v2.as_ref().ok_or(AdaptiveManagerError::Invalid(
            "V2 board collection requires the original root admission",
        ))?;
        let frontier = manager.driver().port().and_then(|port| port.rovent_frontier()).ok_or(
            AdaptiveManagerError::Invalid("actual V2 board collection frontier absent"),
        )?;
        let admission = RoventCollectionRootAdmissionV2::from_attached_frontier(
            origin, frontier, Arc::clone(original), root.budget, profile.selective_policy,
        )?;
        let expected = admission.clone();
        let mut port = RoventBoardCollectionAdmissionPortV2 { manager };
        if let Err(rejected) = hooks.admit(&mut port, admission) {
            let error = if rejected.admission.same_original_admission(&expected) {
                rejected.error.clone()
            } else {
                AdaptiveManagerError::Invalid("V2 board rejected a different root owner")
            };
            *self.rovent_collection_failure_v2.lock().unwrap_or_else(|p| p.into_inner()) =
                Some(rejected);
            return Err(error);
        }
        if !port.manager.collection_root_admission_v2().is_some_and(|installed| {
            installed.same_original_admission(&expected)
        }) {
            return Err(AdaptiveManagerError::Invalid(
                "V2 board hook did not install the original collection admission",
            ));
        }
        Ok(true)
    }
}
