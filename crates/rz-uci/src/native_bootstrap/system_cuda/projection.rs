//! Read-only observations of one original LC0 session; never a release proof.

use serde::Serialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Lc0ObservedService {
    NotStarted,
    Serving,
    Succeeded,
    Failed,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Lc0ObservedRelease {
    NotObserved,
    Pending,
    Released,
    Failed,
}

/// Aggregate observations; mixed never discards an earlier failure.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Lc0ObservedPublication {
    #[default]
    NotAttempted,
    Succeeded,
    Failed,
    Mixed,
}

impl Lc0ObservedPublication {
    pub(super) fn record(&mut self, succeeded: bool) {
        *self = match (*self, succeeded) {
            (Self::NotAttempted, true) | (Self::Succeeded, true) => Self::Succeeded,
            (Self::NotAttempted, false) | (Self::Failed, false) => Self::Failed,
            _ => Self::Mixed,
        };
    }
}

#[derive(Default)]
pub(super) struct PublicationObservations {
    pub(super) runtime_start: Lc0ObservedPublication,
    pub(super) runtime_finish: Lc0ObservedPublication,
    pub(super) original_failure: Lc0ObservedPublication,
    pub(super) finish: Lc0ObservedPublication,
    pub(super) reobservation: Lc0ObservedPublication,
    pub(super) endpoint_retirement: Lc0ObservedPublication,
}

/// Generated from the original supervisor and its own publication attempts.
/// This is diagnostic data, not a capability or permission to exit. In
/// particular, `scope_settled` may be true before native entry, and session
/// release is neither a device-wide fence nor complete output preservation.
#[derive(Debug, Serialize)]
pub struct Lc0OriginalSessionObservation {
    pub(super) schema: &'static str,
    pub(super) process_id: u32,
    pub(super) process_epoch: u64,
    pub(super) lock_sha256: String,
    pub(super) cuda_ordinal: i32,
    pub(super) ort_entry_observed: bool,
    pub(super) service: Lc0ObservedService,
    pub(super) original_release: Lc0ObservedRelease,
    pub(super) latest_release: Lc0ObservedRelease,
    pub(super) scope_settled: bool,
    pub(super) runtime_start_receipt: Lc0ObservedPublication,
    pub(super) runtime_finish_receipt: Lc0ObservedPublication,
    pub(super) original_failure_receipt: Lc0ObservedPublication,
    pub(super) finish_receipt: Lc0ObservedPublication,
    pub(super) reobservation_receipt: Lc0ObservedPublication,
    pub(super) endpoint_retirement: Lc0ObservedPublication,
    // UCI output transport is outside this supervisor and has not been joined.
    pub(super) output_preservation: &'static str,
    pub(super) physical_completion: &'static str,
}

impl Lc0OriginalSessionObservation {
    pub fn service(&self) -> Lc0ObservedService {
        self.service
    }

    pub fn original_release(&self) -> Lc0ObservedRelease {
        self.original_release
    }

    pub fn latest_release(&self) -> Lc0ObservedRelease {
        self.latest_release
    }

    /// This can also be true before the native entry; it is not a success flag.
    pub fn scope_settled(&self) -> bool {
        self.scope_settled
    }

    pub fn ort_entry_observed(&self) -> bool {
        self.ort_entry_observed
    }

    pub fn endpoint_retirement(&self) -> Lc0ObservedPublication {
        self.endpoint_retirement
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publication_history_never_erases_an_earlier_error() {
        let mut state = Lc0ObservedPublication::default();
        state.record(false);
        assert_eq!(state, Lc0ObservedPublication::Failed);
        state.record(true);
        assert_eq!(state, Lc0ObservedPublication::Mixed);
        state.record(true);
        assert_eq!(state, Lc0ObservedPublication::Mixed);
    }
}
