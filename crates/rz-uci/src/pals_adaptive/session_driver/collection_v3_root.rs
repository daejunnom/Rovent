//! Tick-driven Own Collection root lifecycle. The original manager, native
//! fence, observer and separately joined teacher owner are never substituted.

use super::collection_v3::{
    RoventBoardAdmissionStepV3, RoventBoardCollectionHandoffV3, RoventBoardCollectionStateV3,
    RoventBoardCustodyReturnV3, RoventBoardDeferredCleanupFailureV3, RoventBoardDeferredCleanupV3,
    RoventBoardRootAdmissionAttemptV3,
};
use super::*;
use crate::pals_adaptive::AdaptiveCollectionBoundaryV2;
use crate::rovent_root_admission::source_seal::RoventRootAdmissionV3;

/// This frame must be retained with the original session across Pending,
/// errors and cancellations; none of its flags represents physical evidence.
#[derive(Default)]
pub(super) struct OwnCollectionRootRunV3 {
    attempt: Option<RoventBoardRootAdmissionAttemptV3>,
    installed: bool,
    deferred: bool,
    revoked: bool,
    handoff_sent: bool,
    custody_take_uncertain: bool,
    custody_return_uncertain: bool,
    custody_returned: bool,
    callback_failed: bool,
    settled: bool,
}

impl OwnCollectionRootRunV3 {
    /// Make one nonblocking admission attempt and retain the same root on
    /// Pending. The session owns deadline and cancellation scheduling.
    pub(super) fn admit_tick<B>(
        &mut self,
        state: &RoventBoardCollectionStateV3<B>,
        manager: &mut AdaptiveSessionManager<B>,
        root: Arc<RoventRootAdmissionV3>,
        budget: Digest,
        policy: Digest,
    ) -> Result<bool, AdaptiveManagerError>
    where
        B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
        B::Output: ProposerRankingOutput,
    {
        if self.installed || self.deferred || self.settled {
            return Err(AdaptiveManagerError::Invalid(
                "V3 own root already admitted",
            ));
        }
        if self.attempt.is_none() {
            self.attempt = Some(state.prepare_root(manager, Arc::clone(&root), budget, policy)?);
        }
        let attempt = self.attempt.as_mut().expect("original attempt retained");
        if !Arc::ptr_eq(attempt.admission.root_admission(), &root)
            || attempt.admission.root_budget() != budget
            || attempt.admission.selective_policy() != policy
        {
            return Err(AdaptiveManagerError::Invalid(
                "V3 own admission retry differs",
            ));
        }
        match state.try_admit_root(manager, attempt)? {
            RoventBoardAdmissionStepV3::Installed => {
                self.installed = true;
                Ok(true)
            }
            RoventBoardAdmissionStepV3::DeferredNoCapture => {
                self.deferred = true;
                Ok(true)
            }
            RoventBoardAdmissionStepV3::Pending => Ok(false),
        }
    }

    /// Pump or revoke only the exact admitted root; callbacks are nonblocking.
    pub(super) fn pump_or_revoke<B>(
        &mut self,
        state: &RoventBoardCollectionStateV3<B>,
        manager: &AdaptiveSessionManager<B>,
        context: &SearchSessionContext,
    ) -> Result<(), AdaptiveManagerError>
    where
        B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
        B::Output: ProposerRankingOutput,
    {
        if !self.installed || self.revoked {
            return Ok(());
        }
        let admission = &self
            .attempt
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid(
                "V3 own collection admission missing",
            ))?
            .admission;
        if !context.accepts().unwrap_or(false)
            || context.soft_expired(Instant::now())
            || manager.phase() != ManagerPhase::Running
        {
            self.revoke_tick(state)
        } else {
            match catch_unwind(AssertUnwindSafe(|| state.hooks.pump_root(admission))) {
                Ok(Ok(())) => Ok(()),
                Ok(Err(AdaptiveManagerError::AuxiliaryPending)) => {
                    Err(AdaptiveManagerError::AuxiliaryPending)
                }
                Ok(Err(error)) => {
                    self.callback_failed = true;
                    Err(error)
                }
                Err(payload) => {
                    self.callback_failed = true;
                    state
                        .panics
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .push(payload);
                    Err(AdaptiveManagerError::Invalid(
                        "V3 own collection pump panicked",
                    ))
                }
            }
        }
    }

    /// A Pending/uncertain callback can already hold a teacher owner, so
    /// revoke it even when observer installation never completed.
    pub(super) fn revoke_tick<B>(
        &mut self,
        state: &RoventBoardCollectionStateV3<B>,
    ) -> Result<(), AdaptiveManagerError>
    where
        B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
        B::Output: ProposerRankingOutput,
    {
        if self.revoked || !state.outstanding.load(Ordering::Acquire) || self.deferred {
            return Ok(());
        }
        let admission = &self
            .attempt
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid(
                "V3 own collection revoke root missing",
            ))?
            .admission;
        match catch_unwind(AssertUnwindSafe(|| state.hooks.revoke_root(admission))) {
            Ok(Ok(())) => {
                self.revoked = true;
                Ok(())
            }
            Ok(Err(AdaptiveManagerError::AuxiliaryPending)) => {
                Err(AdaptiveManagerError::AuxiliaryPending)
            }
            Ok(Err(error)) => {
                self.callback_failed = true;
                Err(error)
            }
            Err(payload) => {
                self.callback_failed = true;
                state
                    .panics
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .push(payload);
                Err(AdaptiveManagerError::Invalid(
                    "V3 own collection revoke panicked",
                ))
            }
        }
    }

    /// One settlement step after the actual manager's root fence. The caller
    /// retries Pending until its registered deadline while retaining this frame.
    pub(super) fn settle_tick<B>(
        &mut self,
        state: &RoventBoardCollectionStateV3<B>,
        manager: &mut AdaptiveSessionManager<B>,
    ) -> Result<bool, AdaptiveManagerError>
    where
        B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
        B::Output: ProposerRankingOutput,
    {
        if self.settled {
            return Ok(true);
        }
        let physical = manager.shutdown_receipt()?;
        if !physical.permits_root_reuse() {
            return Err(AdaptiveManagerError::Invalid(
                "V3 own native root fence unavailable",
            ));
        }
        if self.deferred {
            return self.settle_no_capture(state, manager);
        }
        if !self.installed {
            // An installation that raced a PendingUncertain response can be
            // recovered only by observing this same manager's original root.
            if let Some(installed) = manager.collection_root_admission_v3() {
                let expected = &self
                    .attempt
                    .as_ref()
                    .ok_or(AdaptiveManagerError::Invalid("V3 own pending root missing"))?
                    .admission;
                if !installed.same_original_admission(expected) {
                    return Err(AdaptiveManagerError::Invalid(
                        "V3 own installed pending root differs",
                    ));
                }
                self.installed = true;
            }
        }
        if !self.installed {
            // An uncertain admission remains owned even if the native root
            // drained; native completion cannot settle a teacher owner.
            if state.outstanding.load(Ordering::Acquire) {
                return Err(AdaptiveManagerError::AuxiliaryPending);
            }
            self.settled = true;
            return Ok(true);
        }
        if !self.revoked {
            self.revoke_tick(state)?;
            if !self.revoked {
                return Ok(false);
            }
        }
        let original = &self
            .attempt
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid("V3 own handoff root missing"))?
            .admission;
        let existing = state
            .handoff
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .cloned();
        let handoff = if let Some(original) = existing {
            original
        } else {
            // Obtain a destination before removing the unique observer from
            // the same manager; publish its retained handoff before delivery.
            let sender = match catch_unwind(AssertUnwindSafe(|| {
                state.hooks.handoff_sender_root(original)
            })) {
                Ok(result) => result?,
                Err(payload) => {
                    state
                        .panics
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .push(payload);
                    return Err(AdaptiveManagerError::Invalid(
                        "V3 own handoff callback panicked",
                    ));
                }
            };
            let (admission, observer, settlement) =
                manager
                    .take_collection_observer_v3()?
                    .ok_or(AdaptiveManagerError::Invalid(
                        "V3 own original observer missing",
                    ))?;
            let handoff = Arc::new(Mutex::new(RoventBoardCollectionHandoffV3 {
                admission,
                observer,
                settlement,
                native_fence: physical,
                teacher_physical_ack: None,
                teacher_aggregate_ack: None,
            }));
            *state.handoff.lock().unwrap_or_else(|p| p.into_inner()) = Some(Arc::clone(&handoff));
            if !handoff
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .admission
                .same_original_admission(original)
            {
                return Err(AdaptiveManagerError::Invalid("V3 own handoff root differs"));
            }
            match sender.try_send(Arc::clone(&handoff)) {
                Ok(()) => self.handoff_sent = true,
                Err(std::sync::mpsc::TrySendError::Full(_)) => return Ok(false),
                Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                    return Err(AdaptiveManagerError::Invalid(
                        "V3 own handoff channel disconnected",
                    ));
                }
            }
            handoff
        };
        let handoff_matches = match handoff.try_lock() {
            Ok(guard) => guard.admission.same_original_admission(original),
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(AdaptiveManagerError::Invalid("V3 own handoff poisoned"));
            }
        };
        if !handoff_matches {
            return Err(AdaptiveManagerError::Invalid(
                "V3 own retained handoff differs",
            ));
        }
        if !self.handoff_sent {
            let sender = match catch_unwind(AssertUnwindSafe(|| {
                state.hooks.handoff_sender_root(original)
            })) {
                Ok(result) => result?,
                Err(payload) => {
                    state
                        .panics
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .push(payload);
                    return Err(AdaptiveManagerError::Invalid(
                        "V3 own handoff callback panicked",
                    ));
                }
            };
            match sender.try_send(Arc::clone(&handoff)) {
                Ok(()) => self.handoff_sent = true,
                Err(std::sync::mpsc::TrySendError::Full(_)) => return Ok(false),
                Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                    return Err(AdaptiveManagerError::Invalid(
                        "V3 own handoff channel disconnected",
                    ));
                }
            }
        }
        let existing = state
            .custody
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .cloned();
        let custody = if let Some(original) = existing {
            original
        } else {
            if self.custody_take_uncertain {
                return Err(AdaptiveManagerError::Invalid(
                    "V3 own custody transfer unresolved",
                ));
            }
            let owner = match catch_unwind(AssertUnwindSafe(|| {
                state.hooks.try_take_root_custody(&handoff)
            })) {
                Ok(Ok(Some(owner))) => owner,
                Ok(Ok(None)) | Ok(Err(AdaptiveManagerError::AuxiliaryPending)) => return Ok(false),
                Ok(Err(error)) => {
                    self.custody_take_uncertain = true;
                    return Err(error);
                }
                Err(payload) => {
                    self.custody_take_uncertain = true;
                    state
                        .panics
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .push(payload);
                    return Err(AdaptiveManagerError::Invalid(
                        "V3 own custody callback panicked",
                    ));
                }
            };
            let matches = owner.matches_handoff(&handoff);
            let owner = Arc::new(Mutex::new(owner));
            *state.custody.lock().unwrap_or_else(|p| p.into_inner()) = Some(Arc::clone(&owner));
            if !matches {
                return Err(AdaptiveManagerError::Invalid("V3 own joined root differs"));
            }
            owner
        };
        let custody_matches = match custody.try_lock() {
            Ok(guard) => guard.matches_handoff(&handoff),
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(AdaptiveManagerError::Invalid("V3 own custody poisoned"));
            }
        };
        if !custody_matches {
            return Err(AdaptiveManagerError::Invalid("V3 own custody differs"));
        }
        let (teacher_done, settlement_error) = match handoff.try_lock() {
            Ok(guard) => (
                guard.teacher_aggregate_ack().is_some(),
                guard.settlement.as_ref().err().cloned(),
            ),
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(AdaptiveManagerError::Invalid(
                    "V3 own teacher handoff poisoned",
                ));
            }
        };
        if !teacher_done {
            return Ok(false);
        }
        if self.custody_return_uncertain {
            return Err(AdaptiveManagerError::Invalid(
                "V3 own custody return unresolved",
            ));
        }
        if !self.custody_returned {
            match catch_unwind(AssertUnwindSafe(|| {
                state.hooks.return_root_custody(Arc::clone(&custody))
            })) {
                Ok(Ok(RoventBoardCustodyReturnV3::Accepted)) => self.custody_returned = true,
                Ok(Ok(RoventBoardCustodyReturnV3::Busy))
                | Ok(Err(AdaptiveManagerError::AuxiliaryPending)) => return Ok(false),
                Ok(Err(error)) => {
                    self.custody_return_uncertain = true;
                    return Err(error);
                }
                Err(payload) => {
                    self.custody_return_uncertain = true;
                    state
                        .panics
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .push(payload);
                    return Err(AdaptiveManagerError::Invalid(
                        "V3 own return callback panicked",
                    ));
                }
            }
        }
        if let Some(error) = settlement_error {
            // Physical teacher retirement cannot turn an observer failure
            // into semantic success. Retain the original handoff and error.
            return Err(error.into());
        }
        if self.callback_failed {
            return Err(AdaptiveManagerError::Invalid(
                "V3 own collection callback failed",
            ));
        }
        state
            .custody
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take();
        state
            .handoff
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take();
        self.settled = true;
        state.outstanding.store(false, Ordering::Release);
        Ok(true)
    }

    fn settle_no_capture<B>(
        &mut self,
        state: &RoventBoardCollectionStateV3<B>,
        manager: &mut AdaptiveSessionManager<B>,
    ) -> Result<bool, AdaptiveManagerError>
    where
        B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
        B::Output: ProposerRankingOutput,
    {
        let capability = state
            .no_capture
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|item| item.0.clone())
            .ok_or(AdaptiveManagerError::Invalid(
                "V3 own marker capability missing",
            ))?;
        let fence = manager.own_no_capture_fence_v3(&capability)?;
        if !capability.matches_own_fence(&fence) {
            return Err(AdaptiveManagerError::Invalid("V3 own marker fence differs"));
        }
        let result = catch_unwind(AssertUnwindSafe(|| {
            state.hooks.finish_deferred(&capability, &fence)
        }));
        let failure = match result {
            Ok(Ok(AdaptiveCollectionBoundaryV2::Ready)) => None,
            Ok(Ok(AdaptiveCollectionBoundaryV2::Busy))
            | Ok(Err(AdaptiveManagerError::AuxiliaryPending)) => return Ok(false),
            Ok(Err(error)) => Some(RoventBoardDeferredCleanupFailureV3::Error(error)),
            Err(payload) => Some(RoventBoardDeferredCleanupFailureV3::Panic(payload)),
        };
        if let Some(failure) = failure {
            let reported = match &failure {
                RoventBoardDeferredCleanupFailureV3::Busy => AdaptiveManagerError::AuxiliaryPending,
                RoventBoardDeferredCleanupFailureV3::Error(error) => error.clone(),
                RoventBoardDeferredCleanupFailureV3::Panic(_) => {
                    AdaptiveManagerError::Invalid("V3 own marker cleanup panicked")
                }
            };
            let reason = state
                .no_capture
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .as_ref()
                .ok_or(AdaptiveManagerError::Invalid("V3 own marker owner missing"))?
                .1
                .clone();
            let owner = Arc::new(Mutex::new(RoventBoardDeferredCleanupV3 {
                capability,
                reason,
                fence,
                failure,
                retention_failure: None,
            }));
            let mut active = state
                .deferred_cleanup
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            if let Some(previous) = active.replace(Arc::clone(&owner)) {
                state
                    .prior_cleanup_failures
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .push(previous);
            }
            return Err(reported);
        }
        manager.confirm_no_capture_v3_cleanup(&capability, &fence)?;
        state
            .no_capture
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take();
        if let Some(previous) = state
            .deferred_cleanup
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take()
        {
            state
                .prior_cleanup_failures
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(previous);
        }
        state.outstanding.store(false, Ordering::Release);
        self.settled = true;
        Ok(true)
    }
}
