//! Root-scoped V3 collection lifecycle. The original manager and hook own all
//! physical resources; errors leave unresolved owners in `state`.
use super::pc_collection_v3::{
    RoventPcCollectionAdmissionFailureV3, RoventPcCollectionAdmissionOutcomeV3,
    RoventPcCollectionAdmissionPortV3, RoventPcCollectionHandoffV3, RoventPcCollectionStateV3,
    RoventPcCustodyReturnV3, RoventPcDeferredCleanupFailureV3, RoventPcDeferredCleanupV3,
};
use super::pc_session::{RoventPcRevokeObservationV2, RoventPcSessionManager};
use crate::pals_adaptive::{
    AdaptiveCollectionBoundaryV2, AdaptiveManagerError, ManagerPhase,
    RoventCollectionRootAdmissionV3, RoventV3FrontierAttachFailure,
};
use crate::rovent_root_admission::source_seal::RoventRootAdmissionV3;
use crate::search_driver::SearchSessionContext;
use rz_contracts::Digest;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError, atomic::Ordering};
use std::time::{Duration, Instant};

/// Observe the original clock and authority without waiting on either owner.
/// Busy does not grant admission, revoke a root, or release any custody.
fn try_admission_live(
    context: &SearchSessionContext,
    phase: ManagerPhase,
) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveManagerError> {
    let now = Instant::now();
    if context.cancellation.is_canceled() || context.control.cancellation.load(Ordering::Acquire) {
        return Err(AdaptiveManagerError::Contract(
            rz_contracts::ContractError::new(
                rz_contracts::ErrorCode::Canceled,
                rz_contracts::Stage::Admission,
                "P/C V3 admission canceled",
            ),
        ));
    }
    if now >= context.control.deadline {
        return Err(AdaptiveManagerError::Deadline);
    }
    if phase != ManagerPhase::Running {
        return Err(AdaptiveManagerError::AdmissionClosed);
    }
    let soft_expired = if let Some(clock) = &context.effective_clock {
        let Some(snapshot) = clock
            .try_snapshot(now)
            .map_err(AdaptiveManagerError::Contract)?
        else {
            return Ok(AdaptiveCollectionBoundaryV2::Busy);
        };
        if snapshot.is_closed() {
            return Err(AdaptiveManagerError::AdmissionClosed);
        }
        if !snapshot.is_valid_observation() {
            return Err(AdaptiveManagerError::Contract(
                rz_contracts::ContractError::new(
                    rz_contracts::ErrorCode::BackendFailure,
                    rz_contracts::Stage::Admission,
                    "P/C V3 clock observation invalid",
                ),
            ));
        }
        snapshot.soft_expired(now)
    } else {
        now >= context.control.soft_deadline
    };
    if soft_expired {
        return Err(AdaptiveManagerError::Deadline);
    }
    match context.try_accepts() {
        Ok(true) => Ok(AdaptiveCollectionBoundaryV2::Ready),
        Ok(false) => {
            if context.cancellation.is_canceled()
                || context.control.cancellation.load(Ordering::Acquire)
            {
                Err(AdaptiveManagerError::Contract(
                    rz_contracts::ContractError::new(
                        rz_contracts::ErrorCode::Canceled,
                        rz_contracts::Stage::Admission,
                        "P/C V3 admission canceled",
                    ),
                ))
            } else if Instant::now() >= context.control.deadline {
                Err(AdaptiveManagerError::Deadline)
            } else {
                Err(AdaptiveManagerError::AdmissionClosed)
            }
        }
        Err(error) if error.code == rz_contracts::ErrorCode::ResourceExhausted => {
            Ok(AdaptiveCollectionBoundaryV2::Busy)
        }
        Err(error) => Err(AdaptiveManagerError::Contract(error)),
    }
}

/// Handoff and custody may be held by the teacher worker. Never wait on
/// these locks while driving root completion or while transferring ownership.
fn try_shared_owner<T>(owner: &Mutex<T>) -> Result<MutexGuard<'_, T>, AdaptiveManagerError> {
    match owner.try_lock() {
        Ok(guard) => Ok(guard),
        Err(TryLockError::WouldBlock) => Err(AdaptiveManagerError::AuxiliaryPending),
        Err(TryLockError::Poisoned(_)) => Err(AdaptiveManagerError::Invalid(
            "P/C V3 shared owner lock poisoned",
        )),
    }
}

#[derive(Default)]
pub(crate) struct PcCollectionRootRunV3 {
    admission: Option<RoventCollectionRootAdmissionV3>,
    started: bool,
    revoked: bool,
    uncertain: bool,
    handoff_sent: bool,
    custody_take_uncertain: bool,
    custody_return_uncertain: bool,
    custody_returned: bool,
    settled: bool,
}

impl PcCollectionRootRunV3 {
    pub(crate) fn admit(
        &mut self,
        state: &RoventPcCollectionStateV3,
        manager: &mut RoventPcSessionManager,
        root: Arc<RoventRootAdmissionV3>,
        budget: Digest,
        policy: Digest,
        context: &SearchSessionContext,
    ) -> Result<(), AdaptiveManagerError> {
        let admission =
            loop {
                match try_admission_live(context, manager.phase())? {
                    AdaptiveCollectionBoundaryV2::Ready => {}
                    AdaptiveCollectionBoundaryV2::Busy => {
                        std::thread::sleep(Duration::from_millis(1));
                        continue;
                    }
                }
                let frontier = manager.driver().port().rovent_frontier().ok_or(
                    AdaptiveManagerError::Invalid("P/C V3 actual frontier absent"),
                )?;
                match RoventCollectionRootAdmissionV3::try_from_attached_frontier(
                    &state.origin,
                    frontier,
                    &root,
                    budget,
                    policy,
                ) {
                    Ok(admission) => break admission,
                    Err(RoventV3FrontierAttachFailure::Busy) => {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    Err(RoventV3FrontierAttachFailure::Rejected(error)) => return Err(error.into()),
                }
            };
        self.admission = Some(admission.clone());
        loop {
            match try_admission_live(context, manager.phase()) {
                Ok(AdaptiveCollectionBoundaryV2::Ready) => {}
                Ok(AdaptiveCollectionBoundaryV2::Busy) => {
                    std::thread::sleep(Duration::from_millis(1));
                    continue;
                }
                Err(error) => {
                    if self.started {
                        *state
                            .admission_failure
                            .lock()
                            .unwrap_or_else(|p| p.into_inner()) =
                            Some(RoventPcCollectionAdmissionFailureV3 {
                                error: error.clone(),
                                admission: admission.clone(),
                                owners: Box::new(()),
                            });
                    }
                    return Err(error);
                }
            }
            // The hook may panic after taking ownership. Publish the root and
            // its outstanding lease before calling it.
            self.started = true;
            state.outstanding.store(true, Ordering::Release);
            let mut port = RoventPcCollectionAdmissionPortV3 { manager };
            let result = match catch_unwind(AssertUnwindSafe(|| {
                state.hooks.admit(&mut port, admission.clone())
            })) {
                Ok(result) => result,
                Err(payload) => {
                    self.uncertain = true;
                    *state
                        .admission_failure
                        .lock()
                        .unwrap_or_else(|p| p.into_inner()) =
                        Some(RoventPcCollectionAdmissionFailureV3 {
                            error: AdaptiveManagerError::Invalid(
                                "P/C V3 admission callback panicked",
                            ),
                            admission: admission.clone(),
                            owners: Box::new(()),
                        });
                    state.retain_panic(payload);
                    return Err(AdaptiveManagerError::Invalid(
                        "P/C V3 admission callback panicked",
                    ));
                }
            };
            match result {
                Ok(RoventPcCollectionAdmissionOutcomeV3::Installed) => {
                    let actual = manager.collection_root_admission_v3().ok_or(
                        AdaptiveManagerError::Invalid("P/C V3 observer not installed"),
                    )?;
                    if !actual.same_original_admission(&admission) {
                        return Err(AdaptiveManagerError::Invalid(
                            "P/C V3 installed root differs",
                        ));
                    }
                    return Ok(());
                }
                Ok(RoventPcCollectionAdmissionOutcomeV3::DeferredNoCapture {
                    capability,
                    reason,
                }) => {
                    if !capability.same_original_admission(&admission)
                        || manager.collection_root_admission_v3().is_some()
                        || self.uncertain
                    {
                        *state
                            .admission_failure
                            .lock()
                            .unwrap_or_else(|p| p.into_inner()) =
                            Some(RoventPcCollectionAdmissionFailureV3 {
                                error: AdaptiveManagerError::Invalid(
                                    "P/C V3 no-capture root mismatch",
                                ),
                                admission: admission.clone(),
                                owners: Box::new((capability, reason)),
                            });
                        return Err(AdaptiveManagerError::Invalid(
                            "P/C V3 no-capture root mismatch",
                        ));
                    }
                    *state.no_capture.lock().unwrap_or_else(|p| p.into_inner()) =
                        Some((capability, reason));
                    self.started = false;
                    return Ok(());
                }
                Ok(RoventPcCollectionAdmissionOutcomeV3::PendingUncertain {
                    admission: returned,
                }) => {
                    if !returned.same_original_admission(&admission) {
                        *state
                            .admission_failure
                            .lock()
                            .unwrap_or_else(|p| p.into_inner()) =
                            Some(RoventPcCollectionAdmissionFailureV3 {
                                error: AdaptiveManagerError::Invalid(
                                    "P/C V3 uncertain root differs",
                                ),
                                admission: returned,
                                owners: Box::new(()),
                            });
                        return Err(AdaptiveManagerError::Invalid(
                            "P/C V3 uncertain root differs",
                        ));
                    }
                    self.uncertain = true;
                }
                Ok(RoventPcCollectionAdmissionOutcomeV3::Pending {
                    admission: returned,
                }) => {
                    if !returned.same_original_admission(&admission)
                        || manager.collection_root_admission_v3().is_some()
                    {
                        *state
                            .admission_failure
                            .lock()
                            .unwrap_or_else(|p| p.into_inner()) =
                            Some(RoventPcCollectionAdmissionFailureV3 {
                                error: AdaptiveManagerError::Invalid("P/C V3 pending root differs"),
                                admission: returned,
                                owners: Box::new(()),
                            });
                        return Err(AdaptiveManagerError::Invalid("P/C V3 pending root differs"));
                    }
                    // A later Pending does not seal an earlier uncertain attempt.
                    self.started = self.uncertain || manager.original_no_capture_v3().is_some();
                    if !self.started {
                        state.outstanding.store(false, Ordering::Release);
                    }
                }
                Err(failure) => {
                    let error = if failure.admission.same_original_admission(&admission) {
                        failure.error.clone()
                    } else {
                        AdaptiveManagerError::Invalid("P/C V3 rejected root differs")
                    };
                    *state
                        .admission_failure
                        .lock()
                        .unwrap_or_else(|p| p.into_inner()) = Some(failure);
                    return Err(error);
                }
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// The session supplies its existing read-only timing observation.
    pub(crate) fn pump_or_revoke(
        &mut self,
        state: &RoventPcCollectionStateV3,
        manager: &RoventPcSessionManager,
        context: &SearchSessionContext,
        observation: RoventPcRevokeObservationV2,
    ) -> Result<(), AdaptiveManagerError> {
        if !self.started || self.revoked {
            return Ok(());
        }
        let admission = self
            .admission
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid("P/C V3 original root absent"))?;
        if !context.accepts().unwrap_or(false)
            || context.soft_expired(Instant::now())
            || manager.phase() != ManagerPhase::Running
        {
            state.hooks.revoke_root_observed(admission, observation)?;
            self.revoked = true;
        } else {
            state.hooks.pump_root(admission)?;
        }
        Ok(())
    }

    /// Call after the root run even if it failed. The caller retains the same
    /// manager and root state on every error or panic.
    pub(crate) fn revoke_after_run(
        &mut self,
        state: &RoventPcCollectionStateV3,
        context: &SearchSessionContext,
        observation: RoventPcRevokeObservationV2,
    ) -> Result<(), AdaptiveManagerError> {
        if !self.started || self.revoked {
            return Ok(());
        }
        let admission = self
            .admission
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid("P/C V3 original root absent"))?;
        loop {
            match catch_unwind(AssertUnwindSafe(|| {
                state.hooks.revoke_root_observed(admission, observation)
            })) {
                Ok(Ok(())) => {
                    self.revoked = true;
                    return Ok(());
                }
                Ok(Err(AdaptiveManagerError::AuxiliaryPending))
                    if Instant::now() < context.shutdown_deadline =>
                {
                    std::thread::sleep(Duration::from_millis(1))
                }
                Ok(Err(error)) => return Err(error),
                Err(payload) => {
                    state.retain_panic(payload);
                    return Err(AdaptiveManagerError::Invalid(
                        "P/C V3 revoke callback panicked",
                    ));
                }
            }
        }
    }

    /// Execute only after the original manager's shutdown. On failure the state
    /// retains original handoff, joined worker, marker and callback payloads.
    pub(crate) fn settle_after_shutdown(
        &mut self,
        state: &RoventPcCollectionStateV3,
        manager: &mut RoventPcSessionManager,
        context: &SearchSessionContext,
    ) -> Result<(), AdaptiveManagerError> {
        let physical = manager.shutdown_receipt()?;
        if !physical.permits_semantic_closure() {
            return Err(AdaptiveManagerError::Invalid(
                "P/C V3 native fence unavailable",
            ));
        }
        let no_capture = state
            .no_capture
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|v| v.0.clone());
        if let Some(capability) = no_capture {
            let fence = manager.role_only_no_capture_fence_v3(&capability)?;
            if !capability.matches_role_only_fence(&fence) {
                return Err(AdaptiveManagerError::Invalid(
                    "P/C V3 no-capture worker differs",
                ));
            }
            let signal = manager.activity_signal();
            let failure = loop {
                let version = signal.version();
                match catch_unwind(AssertUnwindSafe(|| {
                    state.hooks.finish_deferred(&capability, &fence)
                })) {
                    Ok(Ok(AdaptiveCollectionBoundaryV2::Ready)) => break None,
                    Ok(Ok(AdaptiveCollectionBoundaryV2::Busy))
                        if Instant::now() < context.shutdown_deadline =>
                    {
                        signal.wait_changed(
                            version,
                            context
                                .shutdown_deadline
                                .saturating_duration_since(Instant::now())
                                .min(Duration::from_millis(10)),
                        );
                    }
                    Ok(Ok(AdaptiveCollectionBoundaryV2::Busy)) => {
                        break Some(RoventPcDeferredCleanupFailureV3::Busy);
                    }
                    Ok(Err(error)) => break Some(RoventPcDeferredCleanupFailureV3::Error(error)),
                    Err(payload) => break Some(RoventPcDeferredCleanupFailureV3::Panic(payload)),
                }
            };
            if let Some(failure) = failure {
                let reported = match &failure {
                    RoventPcDeferredCleanupFailureV3::Busy => {
                        AdaptiveManagerError::AuxiliaryPending
                    }
                    RoventPcDeferredCleanupFailureV3::Error(error) => error.clone(),
                    RoventPcDeferredCleanupFailureV3::Panic(_) => {
                        AdaptiveManagerError::Invalid("P/C V3 marker cleanup callback panicked")
                    }
                };
                // The hook has not proven marker removal. Keep both the
                // original capability and its first failure under the session;
                // handing an Arc to another owner is not cleanup completion.
                let mut retained = state
                    .deferred_cleanup
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                if let Some(existing) = retained.as_ref() {
                    match failure {
                        RoventPcDeferredCleanupFailureV3::Error(error) => {
                            existing
                                .lock()
                                .unwrap_or_else(|p| p.into_inner())
                                .retention_failure = Some(error);
                        }
                        RoventPcDeferredCleanupFailureV3::Panic(payload) => {
                            state.retain_panic(payload);
                        }
                        RoventPcDeferredCleanupFailureV3::Busy => {}
                    }
                } else {
                    let reason = state
                        .no_capture
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .as_ref()
                        .ok_or(AdaptiveManagerError::Invalid("P/C V3 marker owner absent"))?
                        .1
                        .clone();
                    *retained = Some(Arc::new(Mutex::new(RoventPcDeferredCleanupV3 {
                        capability: capability.clone(),
                        reason,
                        fence,
                        failure,
                        retention_failure: None,
                    })));
                }
                return Err(reported);
            }
            // Only the original hook's Ready result authorizes confirmation on
            // this manager. On confirmation failure neither owner is released.
            manager.confirm_no_capture_v3_cleanup(&capability, &fence)?;
            state
                .no_capture
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .take();
            // Keep the original error and panic payload after a later Ready.
            // Publish the history reference before releasing the active slot.
            let mut active = state
                .deferred_cleanup
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            if let Some(failure) = active.as_ref() {
                state
                    .prior_cleanup_failures
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .push(Arc::clone(failure));
                active.take();
            }
            state.outstanding.store(false, Ordering::Release);
        }
        if !self.started {
            return Ok(());
        }
        let original = self
            .admission
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid(
                "P/C V3 handoff original root absent",
            ))?;
        if self.settled {
            return Ok(());
        }
        // A taken observer cannot be recreated from the manager. Reuse the
        // exact handoff retained before sending it, even after a failed send.
        let existing = state
            .handoff
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .cloned();
        let handoff = if let Some(handoff) = existing {
            handoff
        } else {
            // Obtain the sender before taking the observer. A failing or
            // panicking callback therefore leaves it in the original manager.
            let sender = match catch_unwind(AssertUnwindSafe(|| {
                state.hooks.handoff_sender_root(original)
            })) {
                Ok(result) => result?,
                Err(payload) => {
                    state.retain_panic(payload);
                    return Err(AdaptiveManagerError::Invalid(
                        "P/C V3 handoff callback panicked",
                    ));
                }
            };
            let (admission, observer, settlement) =
                manager
                    .take_collection_observer_v3()?
                    .ok_or(AdaptiveManagerError::Invalid(
                        "P/C V3 original observer unavailable",
                    ))?;
            let handoff = Arc::new(Mutex::new(RoventPcCollectionHandoffV3::new(
                admission, observer, settlement, physical,
            )));
            *state.handoff.lock().unwrap_or_else(|p| p.into_inner()) = Some(Arc::clone(&handoff));
            if !try_shared_owner(&handoff)?
                .admission
                .same_original_admission(original)
            {
                return Err(AdaptiveManagerError::Invalid("P/C V3 handoff root differs"));
            }
            if sender.try_send(Arc::clone(&handoff)).is_err() {
                return Err(AdaptiveManagerError::AuxiliaryPending);
            }
            self.handoff_sent = true;
            handoff
        };
        if !try_shared_owner(&handoff)?
            .admission
            .same_original_admission(original)
        {
            return Err(AdaptiveManagerError::Invalid(
                "P/C V3 retained handoff root differs",
            ));
        }
        if !self.handoff_sent {
            let sender = match catch_unwind(AssertUnwindSafe(|| {
                state.hooks.handoff_sender_root(original)
            })) {
                Ok(result) => result?,
                Err(payload) => {
                    state.retain_panic(payload);
                    return Err(AdaptiveManagerError::Invalid(
                        "P/C V3 handoff callback panicked",
                    ));
                }
            };
            if sender.try_send(Arc::clone(&handoff)).is_err() {
                return Err(AdaptiveManagerError::AuxiliaryPending);
            }
            self.handoff_sent = true;
        }
        let signal = manager.activity_signal();
        let existing = state
            .custody
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .cloned();
        let custody = if let Some(owner) = existing {
            if !try_shared_owner(&owner)?.matches_handoff(&handoff) {
                return Err(AdaptiveManagerError::Invalid(
                    "P/C V3 retained joined root differs",
                ));
            }
            owner
        } else {
            if self.custody_take_uncertain {
                return Err(AdaptiveManagerError::Invalid(
                    "P/C V3 worker custody transfer unresolved",
                ));
            }
            loop {
                let version = signal.version();
                match catch_unwind(AssertUnwindSafe(|| {
                    state.hooks.try_take_root_custody(&handoff)
                })) {
                    Ok(Ok(Some(owner))) => {
                        let matches = owner.matches_handoff(&handoff);
                        let owner = Arc::new(Mutex::new(owner));
                        *state.custody.lock().unwrap_or_else(|p| p.into_inner()) =
                            Some(Arc::clone(&owner));
                        if !matches {
                            return Err(AdaptiveManagerError::Invalid(
                                "P/C V3 joined root differs",
                            ));
                        }
                        break owner;
                    }
                    Ok(Ok(None)) if Instant::now() < context.shutdown_deadline => {
                        signal.wait_changed(
                            version,
                            context
                                .shutdown_deadline
                                .saturating_duration_since(Instant::now())
                                .min(Duration::from_millis(10)),
                        );
                    }
                    Ok(Ok(None)) => return Err(AdaptiveManagerError::AuxiliaryPending),
                    Ok(Err(error)) => {
                        self.custody_take_uncertain = true;
                        return Err(error);
                    }
                    Err(payload) => {
                        self.custody_take_uncertain = true;
                        state.retain_panic(payload);
                        return Err(AdaptiveManagerError::Invalid(
                            "P/C V3 custody callback panicked",
                        ));
                    }
                }
            }
        };
        if try_shared_owner(&handoff)?
            .teacher_aggregate_ack()
            .is_none()
        {
            return Err(AdaptiveManagerError::AuxiliaryPending);
        }
        if self.custody_return_uncertain {
            return Err(AdaptiveManagerError::Invalid(
                "P/C V3 worker return unresolved",
            ));
        }
        if !self.custody_returned {
            loop {
                let version = signal.version();
                match catch_unwind(AssertUnwindSafe(|| {
                    state.hooks.return_root_custody(Arc::clone(&custody))
                })) {
                    Ok(Ok(RoventPcCustodyReturnV3::Accepted)) => {
                        self.custody_returned = true;
                        break;
                    }
                    Ok(Ok(RoventPcCustodyReturnV3::Busy))
                        if Instant::now() < context.shutdown_deadline =>
                    {
                        signal.wait_changed(
                            version,
                            context
                                .shutdown_deadline
                                .saturating_duration_since(Instant::now())
                                .min(Duration::from_millis(10)),
                        );
                    }
                    Ok(Ok(RoventPcCustodyReturnV3::Busy)) => {
                        return Err(AdaptiveManagerError::AuxiliaryPending);
                    }
                    Ok(Err(error)) => {
                        self.custody_return_uncertain = true;
                        return Err(error);
                    }
                    Err(payload) => {
                        self.custody_return_uncertain = true;
                        state.retain_panic(payload);
                        return Err(AdaptiveManagerError::Invalid(
                            "P/C V3 return callback panicked",
                        ));
                    }
                }
            }
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
        Ok(())
    }
}

#[cfg(test)]
mod admission_observation_tests {
    use super::*;
    use crate::contracts::SessionScope;
    use rz_contracts::{
        CancelToken, GameGeneration, ProcessEpoch, RootGeneration, pals::SearchAuthority,
    };
    use rz_search::driver::SearchControl;

    fn context() -> SearchSessionContext {
        let authority = SearchAuthority {
            epoch: ProcessEpoch(1),
            game: GameGeneration(1),
            root: RootGeneration(1),
            implementation: Digest([0; 32]),
        };
        let until = Instant::now() + Duration::from_secs(60);
        SearchSessionContext {
            authority,
            original_go: None,
            original_time_budget: None,
            original_start: None,
            effective_clock: None,
            root_moves: None,
            goal: rz_contracts::rovent::SearchGoal::default(),
            requested_nodes: None,
            control: SearchControl::new(until, 1),
            cancellation: CancelToken::new(),
            shutdown_deadline: until,
            current: Arc::new(Mutex::new(SessionScope::Search(authority))),
        }
    }

    #[test]
    fn authority_contention_is_pending_and_cancellation_is_observed_while_locked() {
        let context = context();
        let lock = context.current.lock().expect("original authority");
        assert!(matches!(
            try_admission_live(&context, ManagerPhase::Running),
            Ok(AdaptiveCollectionBoundaryV2::Busy)
        ));
        context.cancellation.cancel();
        assert!(
            matches!(try_admission_live(&context, ManagerPhase::Running), Err(AdaptiveManagerError::Contract(error)) if error.code == rz_contracts::ErrorCode::Canceled)
        );
        drop(lock);
    }

    #[test]
    fn frontier_attach_distinguishes_contention_and_cancellation() {
        let context = context();
        let lock = context.current.lock().expect("original authority");
        assert!(matches!(
            crate::pals_adaptive::try_v3_frontier_attach_accepts(&context),
            Err(RoventV3FrontierAttachFailure::Busy)
        ));
        context.cancellation.cancel();
        assert!(matches!(
            crate::pals_adaptive::try_v3_frontier_attach_accepts(&context),
            Err(RoventV3FrontierAttachFailure::Rejected(error)) if error.code == "RootCanceled"
        ));
        drop(lock);
    }

    #[test]
    fn frontier_attach_checks_deadline_before_contended_authority() {
        let mut context = context();
        context.control.deadline = Instant::now() - Duration::from_secs(1);
        let lock = context.current.lock().expect("original authority");
        assert!(matches!(
            crate::pals_adaptive::try_v3_frontier_attach_accepts(&context),
            Err(RoventV3FrontierAttachFailure::Rejected(error)) if error.code == "RootDeadline"
        ));
        drop(lock);
    }

    #[test]
    fn frontier_attach_checks_actual_authority_and_control_cancellation() {
        let context = context();
        assert!(matches!(
            crate::pals_adaptive::try_v3_frontier_attach_accepts(&context),
            Ok(true)
        ));
        context.control.cancellation.store(true, Ordering::Release);
        assert!(matches!(
            crate::pals_adaptive::try_v3_frontier_attach_accepts(&context),
            Err(RoventV3FrontierAttachFailure::Rejected(error)) if error.code == "RootCanceled"
        ));
        context.control.cancellation.store(false, Ordering::Release);
        let mut replacement = context.authority;
        replacement.root = RootGeneration(context.authority.root.0 + 1);
        *context.current.lock().expect("original authority") = SessionScope::Search(replacement);
        assert!(matches!(
            crate::pals_adaptive::try_v3_frontier_attach_accepts(&context),
            Err(RoventV3FrontierAttachFailure::Rejected(error)) if error.code == "RootAuthorityClosed"
        ));
    }

    #[test]
    fn frontier_attach_does_not_treat_poison_as_contention() {
        let context = context();
        let authority = Arc::clone(&context.current);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _guard = authority.lock().unwrap();
            panic!("exercise authority-owner poisoning");
        }));
        assert!(matches!(
            crate::pals_adaptive::try_v3_frontier_attach_accepts(&context),
            Err(RoventV3FrontierAttachFailure::Rejected(_))
        ));
    }

    #[test]
    fn original_deadline_wins_over_authority_contention() {
        let mut context = context();
        context.control.deadline = Instant::now() - Duration::from_secs(1);
        let lock = context.current.lock().expect("original authority");
        assert!(matches!(
            try_admission_live(&context, ManagerPhase::Running),
            Err(AdaptiveManagerError::Deadline)
        ));
        drop(lock);
    }

    #[test]
    fn original_authority_must_remain_current() {
        let context = context();
        assert!(matches!(
            try_admission_live(&context, ManagerPhase::Running),
            Ok(AdaptiveCollectionBoundaryV2::Ready)
        ));
        let mut replacement = context.authority;
        replacement.root = RootGeneration(context.authority.root.0 + 1);
        *context.current.lock().expect("original authority") = SessionScope::Search(replacement);
        assert!(matches!(
            try_admission_live(&context, ManagerPhase::Running),
            Err(AdaptiveManagerError::AdmissionClosed)
        ));
    }
}
