//! Read-only same-Start clock control. Never rewrite a submitted ticket/job,
//! its original deadline, a CPU node reservation, or a root credit limit.
use super::*;
use crate::rovent_ponder::RoventPonderClockHandle;

pub(super) fn admit(
    handle: &RoventPonderClockHandle,
    authority: SearchAuthority,
    clock: &ContractSystemClock,
    deadline: Deadline,
    cleanup: Option<CleanupBoundary>,
) -> Result<(), AdaptiveManagerError> {
    let cleanup = cleanup.ok_or(AdaptiveManagerError::Invalid(
        "effective clock requires original cleanup registration",
    ))?;
    if !handle.matches_registration(authority, clock, deadline, cleanup.original()) {
        return Err(AdaptiveManagerError::Invalid(
            "effective clock differs from original Start/root/clock/deadline",
        ));
    }
    Ok(())
}

impl<Q, B, D> AdaptiveExecutionManager<Q, B, D>
where
    Q: Clone + Send + Sync + 'static,
    B: TicketNativeBackend<ReviewView, Q>,
    B::Output: ProposerRankingOutput,
    D: AdaptiveSearchDriver<Q>,
{
    pub fn register_effective_clock(
        &mut self,
        handle: RoventPonderClockHandle,
    ) -> Result<(), AdaptiveManagerError> {
        if self.phase != ManagerPhase::Running
            || self.root_started
            || !self.roles.is_empty()
            || self.effective_clock.is_some()
        {
            return Err(AdaptiveManagerError::Invalid(
                "effective clock startup order",
            ));
        }
        admit(
            &handle,
            self.registration.authority,
            &self.clock,
            self.deadline,
            self.cleanup_boundary,
        )?;
        self.effective_clock = Some(handle);
        Ok(())
    }
    pub fn can_admit_work(&self) -> bool {
        let now = Instant::now();
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if self.cohort_action_expired(now) {
            return false;
        }
        self.effective_clock
            .as_ref()
            .is_none_or(|handle| handle.snapshot(now).can_admit(now))
    }
    pub(super) fn effective_hard_expired(&self) -> bool {
        let now = Instant::now();
        self.effective_clock
            .as_ref()
            .is_some_and(|handle| handle.snapshot(now).hard_expired(now))
    }
}
