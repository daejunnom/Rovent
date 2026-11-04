//! Explicit bounded copies at original continuation source boundaries.
//! No callback, work/acceptance permit, native/control owner or model payload.
use super::{AdaptivePendingQuestionSnapshot, AdaptivePendingSnapshotUnavailable};
use rz_contracts::pals::SearchAuthority;
use rz_contracts::{ClockDomain, MonotonicTick, RequestId};
use rz_search::pals::coordinator::{FrameCompletionScope, FrameId, RefinementStage, StepEvidence};
use rz_search::pals::engine::adaptive_port::{
    AdaptiveRoleConsumptionEvent, AdaptiveRoleConsumptionObservation,
    AdaptiveRoleConsumptionUnavailable, MAX_ROLE_CONSUMPTION_EVENTS, ProfiledFeedbackConsumedChain,
    ProfiledFeedbackConsumptionObservation,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::time::Instant;

pub const CONTINUATION_PENDING_SNAPSHOTS_MAX: usize = 32;
pub const CONTINUATION_RESOLUTION_RECORDS_MAX: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContinuationDiagnosticOptions {
    /// Zero disables this channel. Each whole original snapshot is bounded.
    pub pending_snapshots_max: usize,
    /// Zero disables this channel; accepted and publication events are separate.
    pub resolution_records_max: usize,
    pub role_consumption: bool,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContinuationDiagnosticError {
    Capacity,
    Allocation,
    Busy,
    Poisoned,
    AdmissionAlreadyStarted,
    AlreadyInstalled,
}
impl std::fmt::Display for ContinuationDiagnosticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "continuation diagnostic: {self:?}")
    }
}
impl std::error::Error for ContinuationDiagnosticError {}

/// Sticky losses, distinct from original semantic/runtime errors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u64)]
pub enum ContinuationDiagnosticUnavailable {
    Contended = 1,
    Poisoned = 2,
    ScopeMismatch = 4,
    PendingCapacity = 8,
    ResolutionCapacity = 16,
    PendingSourceUnavailable = 32,
    PortUnavailable = 64,
    RoleInstallRejected = 128,
    RoleJournalUnavailable = 256,
    ReadbackRepeated = 512,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContinuationPendingSource {
    /// Original execute_with_admission_clock returned Ok and binding was updated.
    QueueAdmitted,
    /// Original service completion custody/validation and cost updates finished.
    CompletionProcessed,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContinuationPendingObservation {
    pub source: ContinuationPendingSource,
    pub trigger: RequestId,
    /// Only QueueAdmitted carries the original native queue admission timestamp.
    pub queue_admitted_at: Option<Instant>,
    pub snapshot: Result<AdaptivePendingQuestionSnapshot, AdaptivePendingSnapshotUnavailable>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContinuationTransitionKind {
    Continue { next_stage: RefinementStage },
    Complete { scope: FrameCompletionScope },
    Unknown { reason: &'static str },
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContinuationResolutionSource {
    MachineTransition {
        checked_rules_pair: bool,
        accepted: bool,
    },
    ClosedFramePublication {
        published: bool,
    },
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContinuationResolutionObservation {
    pub authority: SearchAuthority,
    pub evidence: StepEvidence,
    pub transition: ContinuationTransitionKind,
    /// Original running check's time, not a new publication/end timestamp.
    pub checked_at: MonotonicTick,
    pub clock: ClockDomain,
    pub source: ContinuationResolutionSource,
    pub closed_frame: Option<FrameId>,
    pub closed_scope: Option<FrameCompletionScope>,
    pub closed_unknown: Option<&'static str>,
}
#[derive(Clone, Debug)]
pub struct ContinuationDiagnosticSnapshot {
    pub authority: Option<SearchAuthority>,
    pub unavailable_bits: u64,
    pub pending: Vec<ContinuationPendingObservation>,
    pub resolutions: Vec<ContinuationResolutionObservation>,
    pub role_install_observed: Option<bool>,
    pub role_events: Vec<AdaptiveRoleConsumptionEvent>,
    pub role_unavailable: Option<AdaptiveRoleConsumptionUnavailable>,
    /// True only at the original permits_root_reuse/Finished fence.
    pub role_readback_root_fence: Option<bool>,
    pub profiled_feedback: Option<ProfiledFeedbackConsumptionObservation>,
}
#[derive(Debug)]
struct State {
    authority: Option<SearchAuthority>,
    pending: Vec<ContinuationPendingObservation>,
    resolutions: Vec<ContinuationResolutionObservation>,
    role_install_observed: Option<bool>,
    role_events: Vec<AdaptiveRoleConsumptionEvent>,
    role_unavailable: Option<AdaptiveRoleConsumptionUnavailable>,
    role_readback_root_fence: Option<bool>,
    profiled_feedback: Option<ProfiledFeedbackConsumptionObservation>,
}
#[derive(Debug)]
struct Inner {
    options: ContinuationDiagnosticOptions,
    state: Mutex<State>,
    unavailable: AtomicU64,
}
/// Clone shares metadata only; the original Search manager is the sole producer.
#[derive(Clone, Debug)]
pub struct ContinuationDiagnostic {
    inner: Arc<Inner>,
}
impl ContinuationDiagnostic {
    pub fn new(
        options: ContinuationDiagnosticOptions,
    ) -> Result<Self, ContinuationDiagnosticError> {
        if options.pending_snapshots_max > CONTINUATION_PENDING_SNAPSHOTS_MAX
            || options.resolution_records_max > CONTINUATION_RESOLUTION_RECORDS_MAX
            || (options.pending_snapshots_max == 0
                && options.resolution_records_max == 0
                && !options.role_consumption)
        {
            return Err(ContinuationDiagnosticError::Capacity);
        }
        let mut pending = Vec::new();
        let mut resolutions = Vec::new();
        let mut role_events = Vec::new();
        pending
            .try_reserve_exact(options.pending_snapshots_max)
            .map_err(|_| ContinuationDiagnosticError::Allocation)?;
        resolutions
            .try_reserve_exact(options.resolution_records_max)
            .map_err(|_| ContinuationDiagnosticError::Allocation)?;
        if options.role_consumption {
            role_events
                .try_reserve_exact(MAX_ROLE_CONSUMPTION_EVENTS)
                .map_err(|_| ContinuationDiagnosticError::Allocation)?;
        }
        Ok(Self {
            inner: Arc::new(Inner {
                options,
                state: Mutex::new(State {
                    authority: None,
                    pending,
                    resolutions,
                    role_install_observed: None,
                    role_events,
                    role_unavailable: None,
                    role_readback_root_fence: None,
                    profiled_feedback: None,
                }),
                unavailable: AtomicU64::new(0),
            }),
        })
    }
    pub fn options(&self) -> ContinuationDiagnosticOptions {
        self.inner.options
    }
    pub(crate) fn unavailable(&self, why: ContinuationDiagnosticUnavailable) {
        self.inner
            .unavailable
            .fetch_or(why as u64, Ordering::AcqRel);
    }
    fn copy_into(&self, copy: impl FnOnce(&mut State)) {
        match self.inner.state.try_lock() {
            Ok(mut state) => copy(&mut state),
            Err(TryLockError::WouldBlock) => {
                self.unavailable(ContinuationDiagnosticUnavailable::Contended)
            }
            Err(TryLockError::Poisoned(_)) => {
                self.unavailable(ContinuationDiagnosticUnavailable::Poisoned)
            }
        }
    }
    pub(crate) fn bind(&self, authority: SearchAuthority) {
        self.copy_into(|state| {
            if state.authority.is_some() {
                self.unavailable(ContinuationDiagnosticUnavailable::ScopeMismatch);
            } else {
                state.authority = Some(authority);
            }
        });
    }
    pub(crate) fn pending_enabled(&self) -> bool {
        self.inner.options.pending_snapshots_max != 0
    }
    pub(crate) fn pending(
        &self,
        authority: SearchAuthority,
        event: ContinuationPendingObservation,
    ) {
        self.copy_into(|state| {
            if state.authority != Some(authority)
                || event
                    .snapshot
                    .as_ref()
                    .is_ok_and(|snapshot| snapshot.authority != authority)
            {
                self.unavailable(ContinuationDiagnosticUnavailable::ScopeMismatch);
                return;
            }
            if state.pending.len() >= self.inner.options.pending_snapshots_max {
                self.unavailable(ContinuationDiagnosticUnavailable::PendingCapacity);
                return;
            }
            if event.snapshot.is_err() {
                self.unavailable(ContinuationDiagnosticUnavailable::PendingSourceUnavailable);
            }
            state.pending.push(event);
        });
    }
    pub(crate) fn resolution(&self, event: ContinuationResolutionObservation) {
        if self.inner.options.resolution_records_max == 0 {
            return;
        }
        self.copy_into(|state| {
            if state.authority != Some(event.authority) {
                self.unavailable(ContinuationDiagnosticUnavailable::ScopeMismatch);
                return;
            }
            if state.resolutions.len() >= self.inner.options.resolution_records_max {
                self.unavailable(ContinuationDiagnosticUnavailable::ResolutionCapacity);
                return;
            }
            state.resolutions.push(event);
        });
    }
    pub(crate) fn role_install(&self, authority: SearchAuthority, installed: bool) {
        self.copy_into(|state| {
            if state.authority != Some(authority) || state.role_install_observed.is_some() {
                self.unavailable(ContinuationDiagnosticUnavailable::ScopeMismatch);
                return;
            }
            state.role_install_observed = Some(installed);
            if !installed {
                self.unavailable(ContinuationDiagnosticUnavailable::RoleInstallRejected);
            }
        });
    }
    pub(crate) fn role_readback(
        &self,
        authority: SearchAuthority,
        journal: Option<&AdaptiveRoleConsumptionObservation>,
        root_fence: bool,
    ) {
        if !self.inner.options.role_consumption {
            return;
        }
        self.copy_into(|state| {
            if state.authority != Some(authority) {
                self.unavailable(ContinuationDiagnosticUnavailable::ScopeMismatch);
                return;
            }
            if state.role_readback_root_fence.is_some() {
                self.unavailable(ContinuationDiagnosticUnavailable::ReadbackRepeated);
                return;
            }
            state.role_readback_root_fence = Some(root_fence);
            let Some(journal) = journal else {
                self.unavailable(ContinuationDiagnosticUnavailable::PortUnavailable);
                return;
            };
            if journal.authority() != authority
                || journal.events().len() > MAX_ROLE_CONSUMPTION_EVENTS
            {
                self.unavailable(ContinuationDiagnosticUnavailable::ScopeMismatch);
                return;
            }
            state.role_events.extend_from_slice(journal.events());
            state.role_unavailable = journal.unavailable();
            if state.role_unavailable.is_some() {
                self.unavailable(ContinuationDiagnosticUnavailable::RoleJournalUnavailable);
            }
        });
    }
    /// Read after original quiesce. It does not drain/reset, certify a case, or
    /// reinterpret missing/prefix records as full coverage.
    pub fn snapshot(&self) -> Result<ContinuationDiagnosticSnapshot, ContinuationDiagnosticError> {
        let state = match self.inner.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Err(ContinuationDiagnosticError::Busy),
            Err(TryLockError::Poisoned(_)) => {
                self.unavailable(ContinuationDiagnosticUnavailable::Poisoned);
                return Err(ContinuationDiagnosticError::Poisoned);
            }
        };
        Ok(ContinuationDiagnosticSnapshot {
            authority: state.authority,
            unavailable_bits: self.inner.unavailable.load(Ordering::Acquire),
            pending: state.pending.clone(),
            resolutions: state.resolutions.clone(),
            role_install_observed: state.role_install_observed,
            role_events: state.role_events.clone(),
            role_unavailable: state.role_unavailable,
            role_readback_root_fence: state.role_readback_root_fence,
            profiled_feedback: state.profiled_feedback,
        })
    }
    pub(crate) fn profiled_feedback_readback(
        &self,
        authority: SearchAuthority,
        observed: Option<ProfiledFeedbackConsumptionObservation>,
    ) {
        if !self.inner.options.role_consumption {
            return;
        }
        self.copy_into(|state| {
            if state.authority != Some(authority)
                || state.profiled_feedback.is_some()
                || observed
                    .as_ref()
                    .is_some_and(|value| value.authority() != authority)
            {
                self.unavailable(ContinuationDiagnosticUnavailable::ScopeMismatch);
                return;
            }
            state.profiled_feedback = observed;
        });
    }
}
impl ContinuationDiagnosticSnapshot {
    /// Accepted copies of original semantic facts, not an execution/fence
    /// permit. Missing, duplicate, unknown or cross-clock resolutions fail.
    pub fn accepted_profiled_feedback_chain(
        &self,
    ) -> Result<&ProfiledFeedbackConsumedChain, &'static str> {
        if self.unavailable_bits != 0
            || self.role_unavailable.is_some()
            || self.role_readback_root_fence != Some(true)
        {
            return Err("original_root_fence_or_journal_unavailable");
        }
        let observed = self
            .profiled_feedback
            .as_ref()
            .ok_or("profiled_source_unavailable")?;
        if self.authority != Some(observed.authority()) || observed.unavailable().is_some() {
            return Err("profiled_source_or_chain_incomplete");
        }
        let chain = observed
            .chain()
            .ok_or("nonzero_repair_critic_fresh_chain_unavailable")?;
        let mut original_clock = None;
        for expected in chain.steps() {
            let mut accepted = self.resolutions.iter().filter(|event| {
                event.authority == observed.authority()
                    && event.evidence == *expected
                    && matches!(
                        event.source,
                        ContinuationResolutionSource::MachineTransition { accepted: true, .. }
                    )
                    && !matches!(event.transition, ContinuationTransitionKind::Unknown { .. })
            });
            let original = accepted
                .next()
                .ok_or("original_machine_acceptance_missing")?;
            if accepted.next().is_some() {
                return Err("original_machine_acceptance_duplicate");
            }
            if original_clock.is_some_and(|clock| clock != original.clock) {
                return Err("original_machine_clock_mismatch");
            }
            original_clock = Some(original.clock);
        }
        Ok(chain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::{Digest, GameGeneration, ProcessEpoch, RootGeneration};
    fn authority(root: u64) -> SearchAuthority {
        SearchAuthority {
            epoch: ProcessEpoch(1),
            game: GameGeneration(1),
            root: RootGeneration(root),
            implementation: Digest([1; 32]),
        }
    }
    fn diagnostic(max: usize) -> ContinuationDiagnostic {
        ContinuationDiagnostic::new(ContinuationDiagnosticOptions {
            pending_snapshots_max: max,
            resolution_records_max: 0,
            role_consumption: false,
        })
        .unwrap()
    }
    #[test]
    fn source_failure_and_overflow_keep_bounded_original_prefix() {
        let d = diagnostic(2);
        let a = authority(1);
        d.bind(a);
        for seq in 1..=3 {
            d.pending(
                a,
                ContinuationPendingObservation {
                    source: ContinuationPendingSource::CompletionProcessed,
                    trigger: RequestId::new(a.epoch, seq),
                    queue_admitted_at: None,
                    snapshot: Err(AdaptivePendingSnapshotUnavailable::RegistryBindingMismatch),
                },
            );
        }
        let s = d.snapshot().unwrap();
        assert_eq!(s.pending.len(), 2);
        assert_eq!(s.pending[0].trigger, RequestId::new(a.epoch, 1));
        assert_eq!(s.pending[1].trigger, RequestId::new(a.epoch, 2));
        assert_ne!(
            s.unavailable_bits & ContinuationDiagnosticUnavailable::PendingCapacity as u64,
            0
        );
        assert_ne!(
            s.unavailable_bits & ContinuationDiagnosticUnavailable::PendingSourceUnavailable as u64,
            0
        );
        assert!(s.pending.iter().all(|e| e.snapshot.is_err()));
    }
    #[test]
    fn rebinding_does_not_replace_scope_or_reconstruct_a_root() {
        let d = diagnostic(1);
        d.bind(authority(1));
        d.bind(authority(2));
        let s = d.snapshot().unwrap();
        assert_eq!(s.authority, Some(authority(1)));
        assert_ne!(
            s.unavailable_bits & ContinuationDiagnosticUnavailable::ScopeMismatch as u64,
            0
        );
    }
    #[test]
    fn live_contention_is_missing_observation_without_waiting() {
        let d = diagnostic(1);
        let held = d.inner.state.lock().unwrap();
        d.bind(authority(1));
        assert_eq!(d.snapshot().unwrap_err(), ContinuationDiagnosticError::Busy);
        drop(held);
        let s = d.snapshot().unwrap();
        assert_eq!(s.authority, None);
        assert_ne!(
            s.unavailable_bits & ContinuationDiagnosticUnavailable::Contended as u64,
            0
        );
    }
}
