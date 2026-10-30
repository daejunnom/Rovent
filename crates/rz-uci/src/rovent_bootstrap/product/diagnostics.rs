//! Bounded operational observations of the existing product owners. These
//! values are deliberately Serialize-only: no decoder can reconstruct a root,
//! fence, actor, acceptance permit or physical-completion authority from them.
use super::{ActualProductDriver, RoventNativeProductDriver};
use crate::pals_adaptive::{
    AdaptiveManagerCounters, AdaptiveShutdownReceipt, RoleOnlyShutdownReceipt,
};
use crate::rovent_root_admission::RoventRootAdmission;
use crate::search_driver::{
    DriverPhysicalCompletion, SearchSessionContext, SearchSessionFailure, SearchSessionReport,
};
use rz_contracts::Digest;
use rz_contracts::pals::SearchAuthority;
use rz_contracts::rovent::ExecutionMode;
use rz_search::cpu_value::nnuec::v2::review::PreparationStatus;
use rz_search::pals::rovent_frontier::{FrontierGoalStatus, FrontierSnapshot, FrontierStop};
use serde::Serialize;
use std::io::{self, Write};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

pub const ROVENT_LIFECYCLE_DIAGNOSTIC_DOMAIN: &str = "rz-rovent-product-lifecycle-observation/1";
pub const ROVENT_STORAGE_DIAGNOSTIC_DOMAIN: &str = "rz-rovent-storage-maintenance-observation/1";
mod initial_stop;
#[cfg(feature = "pals-adaptive-diagnostics")]
mod nnuec;
#[cfg(feature = "pals-adaptive-diagnostics")]
mod nnuec_cpu_slices;
// Operational output limits, independent of search/node/role credit. Keep one
// final row available even after an incomplete root stream reaches its bound.
const MAX_ROOT_ROWS: u64 = 256;
const MAX_ROW_BYTES: usize = 8192;
const MAX_STREAM_BYTES: usize = 1 << 20;

#[derive(Clone, Debug, Serialize)]
struct AuthorityObservation {
    epoch: u64,
    game: u64,
    root: u64,
    implementation_sha256: String,
}
impl From<SearchAuthority> for AuthorityObservation {
    fn from(value: SearchAuthority) -> Self {
        Self {
            epoch: value.epoch.0,
            game: value.game.0,
            root: value.root.0,
            implementation_sha256: hex(value.implementation),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct FailureObservation {
    code: String,
    code_truncated: bool,
    detail: String,
    detail_truncated: bool,
    physical_completion: &'static str,
}
impl From<&SearchSessionFailure> for FailureObservation {
    fn from(value: &SearchSessionFailure) -> Self {
        Self {
            code: bounded(value.code, 128),
            code_truncated: value.code.len() > 128,
            detail: bounded(&value.detail, 512),
            detail_truncated: value.detail.len() > 512,
            physical_completion: physical(value.physical_completion),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct CountersObservation {
    submitted_role_tickets: u64,
    physical_role_completions: u64,
    accepted_nn_inputs: Option<u64>,
    consumed_role_outputs: u64,
    completed_nn_inputs: Option<u64>,
    consumed_nn_inputs: Option<u64>,
    unused_role_results: u64,
    cpu_slice_receipts: u64,
    validator_ticket_admissions: u64,
}
impl From<&AdaptiveManagerCounters> for CountersObservation {
    fn from(value: &AdaptiveManagerCounters) -> Self {
        Self {
            submitted_role_tickets: value.submitted_role_tickets,
            physical_role_completions: value.physical_role_completions,
            accepted_nn_inputs: (value.accepted_nn_input_count_unknown == 0)
                .then_some(value.accepted_nn_inputs),
            consumed_role_outputs: value.consumed_role_outputs,
            completed_nn_inputs: (value.nn_input_count_unknown == 0)
                .then_some(value.completed_nn_inputs),
            consumed_nn_inputs: (value.consumed_nn_input_count_unknown == 0)
                .then_some(value.consumed_nn_inputs),
            unused_role_results: value.unused_role_results,
            cpu_slice_receipts: value.cpu_slice_receipts,
            validator_ticket_admissions: value.validator_ticket_admissions,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct RootPhysicalObservation {
    source: &'static str,
    external_root_fence_present: Option<bool>,
    external_root_fence_reusable: Option<bool>,
    external_process_joined: Option<bool>,
    external_process_close_proof_present: Option<bool>,
    auxiliary_shutdown_confirmed: Option<bool>,
    reported_external_outstanding_questions: Option<usize>,
    semantic_closure_confirmed: bool,
    native_root_fence_present: Option<bool>,
    native_root_fence_reusable: bool,
    native_process_joined: bool,
    native_physical_shutdown_confirmed: bool,
    pending_tickets: usize,
    retained_outputs: Option<usize>,
    quarantined: bool,
    own_cpu_joined: Option<usize>,
    expected_own_cpu_owners: Option<usize>,
    own_cpu_panicked: Option<usize>,
    native_cpu_reserved: Option<u64>,
    shared_nnuec_closure_present: Option<bool>,
    shared_nnuec_issued_requests: Option<usize>,
    shared_nnuec_submitted_requests: Option<usize>,
    shared_nnuec_completed_requests: Option<usize>,
    shared_nnuec_cancelled_requests: Option<usize>,
    shared_nnuec_expired_requests: Option<usize>,
    shared_nnuec_failed_requests: Option<usize>,
    shared_nnuec_semantic_consumed_requests: Option<usize>,
}
impl RootPhysicalObservation {
    // Called only after the original producer-only witness matched this
    // root's admission/clock/ledger and actual pre-materialization retirement.
    // Inapplicable manager/native shutdown fields remain false/None. This
    // source never substitutes counter0 for a manager fence or process join.
    fn pre_materialization() -> Self {
        Self {
            source: "original_pre_materialization_retirement",
            external_root_fence_present: None,
            external_root_fence_reusable: None,
            external_process_joined: None,
            external_process_close_proof_present: None,
            auxiliary_shutdown_confirmed: None,
            reported_external_outstanding_questions: None,
            semantic_closure_confirmed: true,
            native_root_fence_present: Some(false),
            native_root_fence_reusable: false,
            native_process_joined: false,
            native_physical_shutdown_confirmed: false,
            pending_tickets: 0,
            retained_outputs: None,
            quarantined: false,
            own_cpu_joined: None,
            expected_own_cpu_owners: None,
            own_cpu_panicked: None,
            native_cpu_reserved: None,
            shared_nnuec_closure_present: None,
            shared_nnuec_issued_requests: None,
            shared_nnuec_submitted_requests: None,
            shared_nnuec_completed_requests: None,
            shared_nnuec_cancelled_requests: None,
            shared_nnuec_expired_requests: None,
            shared_nnuec_failed_requests: None,
            shared_nnuec_semantic_consumed_requests: None,
        }
    }

    fn pc(value: &RoleOnlyShutdownReceipt) -> Self {
        let closure = value.shared_nnuec_closure();
        Self {
            source: "original_role_only_shutdown_receipt",
            external_root_fence_present: None,
            external_root_fence_reusable: None,
            external_process_joined: None,
            external_process_close_proof_present: None,
            auxiliary_shutdown_confirmed: None,
            reported_external_outstanding_questions: None,
            semantic_closure_confirmed: value.permits_semantic_closure(),
            native_root_fence_present: Some(value.root_fence().is_some()),
            native_root_fence_reusable: value
                .root_fence()
                .is_some_and(|fence| fence.authority() == value.authority && fence.permits_reuse()),
            native_process_joined: value.native_process_joined,
            native_physical_shutdown_confirmed: value.native_physical_shutdown_confirmed,
            pending_tickets: value.pending_tickets,
            retained_outputs: Some(value.retained_outputs),
            quarantined: value.quarantined,
            own_cpu_joined: None,
            expected_own_cpu_owners: None,
            own_cpu_panicked: None,
            native_cpu_reserved: None,
            shared_nnuec_closure_present: Some(closure.is_some()),
            shared_nnuec_issued_requests: closure.map(|c| c.issued_requests()),
            shared_nnuec_submitted_requests: closure.map(|c| c.submitted_requests()),
            shared_nnuec_completed_requests: closure
                .map(|c| c.terminal_status_count(PreparationStatus::Completed)),
            shared_nnuec_cancelled_requests: closure
                .map(|c| c.terminal_status_count(PreparationStatus::Cancelled)),
            shared_nnuec_expired_requests: closure
                .map(|c| c.terminal_status_count(PreparationStatus::Expired)),
            shared_nnuec_failed_requests: closure
                .map(|c| c.terminal_status_count(PreparationStatus::Failed)),
            shared_nnuec_semantic_consumed_requests: closure
                .map(|c| c.semantic_consumed_requests()),
        }
    }
    fn pcr(value: &AdaptiveShutdownReceipt) -> Self {
        // The original predicate validates its private fence. No new fence is
        // inferred from thread counts or from a successful protocol write.
        Self {
            source: "original_adaptive_shutdown_receipt",
            external_root_fence_present: None,
            external_root_fence_reusable: None,
            external_process_joined: None,
            external_process_close_proof_present: None,
            auxiliary_shutdown_confirmed: None,
            reported_external_outstanding_questions: None,
            semantic_closure_confirmed: value.permits_semantic_closure(),
            native_root_fence_present: None,
            native_root_fence_reusable: value.permits_root_reuse(),
            native_process_joined: value.native_owner_joined,
            native_physical_shutdown_confirmed: value.native_physical_shutdown_confirmed,
            pending_tickets: value.pending_tickets,
            retained_outputs: None,
            quarantined: value.quarantined,
            own_cpu_joined: Some(value.own_cpu_joined),
            expected_own_cpu_owners: Some(value.expected_own_cpu_owners),
            own_cpu_panicked: Some(value.own_cpu_panicked),
            native_cpu_reserved: Some(value.native_cpu_reserved),
            shared_nnuec_closure_present: None,
            shared_nnuec_issued_requests: None,
            shared_nnuec_submitted_requests: None,
            shared_nnuec_completed_requests: None,
            shared_nnuec_cancelled_requests: None,
            shared_nnuec_expired_requests: None,
            shared_nnuec_failed_requests: None,
            shared_nnuec_semantic_consumed_requests: None,
        }
    }
}

impl RootPhysicalObservation {
    fn profiled(value: &crate::pals_adaptive::ProfiledExternalShutdownReceipt) -> Self {
        Self {
            source: "original_profiled_external_shutdown_receipt",
            external_root_fence_present: Some(value.external_root_fence().is_some()),
            external_root_fence_reusable: Some(value.external_root_fence().is_some_and(|fence| {
                fence.authority() == value.authority()
                    && fence.source().same_source(value.source())
                    && fence.permits_reuse()
            })),
            external_process_joined: Some(value.external_process_joined()),
            external_process_close_proof_present: Some(value.process_close_proof().is_some()),
            auxiliary_shutdown_confirmed: Some(value.auxiliary_shutdown_confirmed()),
            reported_external_outstanding_questions: value
                .reported_work()
                .map(|work| work.outstanding_questions),
            semantic_closure_confirmed: value.permits_semantic_closure(),
            native_root_fence_present: Some(value.native_root_fence().is_some()),
            native_root_fence_reusable: value.native_root_fence().is_some_and(|fence| {
                fence.authority() == value.authority() && fence.permits_reuse()
            }),
            native_process_joined: value.native_process_joined(),
            native_physical_shutdown_confirmed: value.native_physical_shutdown_confirmed(),
            pending_tickets: value.pending_tickets(),
            retained_outputs: Some(value.retained_outputs()),
            quarantined: value.quarantined(),
            own_cpu_joined: None,
            expected_own_cpu_owners: None,
            own_cpu_panicked: None,
            native_cpu_reserved: None,
            shared_nnuec_closure_present: None,
            shared_nnuec_issued_requests: None,
            shared_nnuec_submitted_requests: None,
            shared_nnuec_completed_requests: None,
            shared_nnuec_cancelled_requests: None,
            shared_nnuec_expired_requests: None,
            shared_nnuec_failed_requests: None,
            shared_nnuec_semantic_consumed_requests: None,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct RetainedSessionObservation {
    observation_point: &'static str,
    stage: &'static str,
    original_authority: Option<AuthorityObservation>,
    next_authority: Option<AuthorityObservation>,
    manager_phase: Option<&'static str>,
    manager_first_retained_failure: Option<FailureObservation>,
    manager_retained_failure_count: Option<usize>,
    archive_operation: Option<&'static str>,
    archive_quarantine_reason: Option<FailureObservation>,
    physical: Option<RootPhysicalObservation>,
    physical_read_failure: Option<FailureObservation>,
}

#[derive(Clone, Debug, Serialize)]
struct SharedNnuecObservation {
    spawned_runtime_helpers: u8,
    worker_thread_id_observed: bool,
    worker_joined: bool,
    // No per-thread affinity readback is manufactured from the process quota.
    worker_thread_affinity_known: bool,
}

#[derive(Clone, Debug, Serialize)]
struct InitialObservation {
    receipt_domain: &'static str,
    source_seal_sha256: String,
    previous_clean_root_present: Option<bool>,
    used_original_startup_ready: Option<bool>,
    stopped_before_research_execution: Option<bool>,
    deadline_before_frontier: Option<bool>,
    original_nnuec_retirement_present: bool,
    original_researcher_release_present: Option<bool>,
    original_cache_release_epoch: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
struct RootObservation {
    receipt_domain: &'static str,
    root_admission_sha256: String,
    root_budget_sha256: String,
    deadline_seal_sha256: String,
    state_and_history_sha256: String,
    goal_status: &'static str,
    completed_depth: Option<u16>,
    maximum_examined_ply: u16,
    native_usage_known: bool,
    native_nodes: Option<u64>,
    native_qsearch_nodes: Option<u64>,
    reserved_native_nodes: u64,
    outstanding_native_executions: usize,
    counters: CountersObservation,
    physical: Option<RootPhysicalObservation>,
    // Only the concrete RoleOnly driver can supply this structural observation.
    // None means not observed/not applicable, never an inferred zero.
    r_free_confirmed: Option<bool>,
    shared_nnuec: Option<SharedNnuecObservation>,
}
impl RootObservation {
    fn new(
        domain: &'static str,
        admission: &RoventRootAdmission,
        frontier: &FrontierSnapshot,
        counters: &AdaptiveManagerCounters,
        physical: Option<RootPhysicalObservation>,
    ) -> Self {
        Self {
            receipt_domain: domain,
            root_admission_sha256: hex(admission.digest()),
            root_budget_sha256: hex(admission.root_budget_digest()),
            deadline_seal_sha256: hex(admission.deadline_seal_digest()),
            state_and_history_sha256: hex(admission.state_and_history_digest()),
            goal_status: goal(frontier.goal),
            completed_depth: frontier.horizon.completed_depth.map(|depth| depth.0),
            maximum_examined_ply: frontier.horizon.maximum_examined_ply.0,
            native_usage_known: frontier.work.native_usage_known,
            native_nodes: frontier
                .work
                .native_usage_known
                .then_some(frontier.work.native_nodes),
            native_qsearch_nodes: frontier
                .work
                .native_usage_known
                .then_some(frontier.work.native_qsearch_nodes),
            reserved_native_nodes: frontier.work.reserved_native_nodes,
            outstanding_native_executions: frontier.work.outstanding_native_executions,
            counters: counters.into(),
            physical,
            r_free_confirmed: None,
            shared_nnuec: None,
        }
    }
}

/// An observational copy only. Its private fields have no parser or conversion
/// into live search authority. A root fence and whole-session finish stay apart.
/// `diagnostics_complete` describes coverage before this row is written. It is
/// not a delivery acknowledgement: consumers must also observe successful
/// process termination and both stream EOFs, since the final flush can fail.
#[derive(Clone, Debug, Serialize)]
pub struct RoventProductLifecycleObservation {
    domain: &'static str,
    event: &'static str,
    sequence: u64,
    input_lock_sha256: String,
    implementation_sha256: String,
    mode: &'static str,
    authority: Option<AuthorityObservation>,
    source_status: &'static str,
    root: Option<RootObservation>,
    initial: Option<InitialObservation>,
    run_failure: Option<FailureObservation>,
    retained_session: Option<RetainedSessionObservation>,
    retained_session_after_finish: Option<RetainedSessionObservation>,
    finish_observed: bool,
    finish_result: &'static str,
    finish_failure: Option<FailureObservation>,
    shared_nnuec_session: Option<SharedNnuecObservation>,
    whole_session_physical: &'static str,
    protocol_succeeded: Option<bool>,
    root_runs_observed: u64,
    root_run_error_results: u64,
    root_rows_emitted: u64,
    diagnostics_complete: bool,
    scope_incomplete: bool,
    diagnostic_failure: Option<&'static str>,
}

#[derive(Default)]
struct State {
    root_runs: u64,
    initial_stop_rows: u64,
    root_run_errors: u64,
    root_rows: u64,
    bytes: usize,
    finish_observed: bool,
    finish_result: Option<&'static str>,
    finish_failure: Option<FailureObservation>,
    shared_nnuec_session: Option<SharedNnuecObservation>,
    first_issue: Option<&'static str>,
    scope_incomplete: bool,
    latest: Option<RoventProductLifecycleObservation>,
    retained_session: Option<RetainedSessionObservation>,
    retained_session_after_finish: Option<RetainedSessionObservation>,
    storage_sequence: u64,
    storage_first_issue: Option<&'static str>,
    storage_scope_incomplete: bool,
    #[cfg(feature = "pals-adaptive-diagnostics")]
    profiled_feedback: Option<crate::pals_adaptive::ContinuationDiagnostic>,
    #[cfg(feature = "pals-adaptive-diagnostics")]
    profiled_feedback_issue: Option<&'static str>,
    #[cfg(feature = "pals-adaptive-diagnostics")]
    profiled_feedback_emitted: bool,
}

/// A read-only handle to this product's diagnostic state. Keeping the handle
/// does not retain a ReviewView, worker, model or shutdown capability.
#[derive(Clone)]
pub struct RoventLifecycleDiagnostics {
    state: Arc<Mutex<State>>,
    poisoned: Arc<AtomicBool>,
    physical_close: Arc<AtomicBool>,
    initial_stop_source: Arc<AtomicBool>,
    lock: Digest,
    implementation: Digest,
    mode: ExecutionMode,
}
impl RoventLifecycleDiagnostics {
    pub(super) fn enable_initial_stop_source(&self) {
        self.initial_stop_source.store(true, Ordering::Release);
    }
    /// This separate Serialize-only domain never changes the established
    /// lifecycle row shape, result, source authority or physical fence.
    pub(super) fn observe_storage(
        &self,
        product: &RoventNativeProductDriver,
        authority: Option<SearchAuthority>,
    ) {
        if product.storage_registration.is_none() {
            return;
        }
        let observed = match product.inner.as_ref() {
            ActualProductDriver::Pc(driver) | ActualProductDriver::PcNnuec(driver) => {
                driver.storage_observation()
            }
            ActualProductDriver::Nnuec(driver) => driver.storage_observation(),
            ActualProductDriver::Pcr(driver) => driver.storage_observation(),
        };
        let Ok(mut state) = self.state.lock() else {
            self.poisoned.store(true, Ordering::Release);
            return;
        };
        let (mut storage, source_status) = match observed {
            Ok(Some(value)) => (Some(value), "original_storage_custody"),
            Ok(None) => {
                state
                    .storage_first_issue
                    .get_or_insert("selected_storage_owner_absent");
                (None, "selected_storage_owner_absent")
            }
            Err(cause) => {
                state.storage_first_issue.get_or_insert(cause);
                (None, "original_storage_readback_unavailable")
            }
        };
        if let Some(storage) = storage.as_mut() {
            match authority {
                Some(authority) => storage.require_after_root(authority),
                None => storage.require_after_finish(),
            }
        }
        if storage.as_ref().is_some_and(|value| !value.complete) {
            state
                .storage_first_issue
                .get_or_insert("original_storage_observation_incomplete");
        }
        match state.storage_sequence.checked_add(1) {
            Some(sequence) => state.storage_sequence = sequence,
            None => {
                state
                    .storage_first_issue
                    .get_or_insert("storage_sequence_exhausted");
                return;
            }
        }
        if state.storage_sequence > MAX_ROOT_ROWS + 1 {
            state.storage_first_issue.get_or_insert("storage_row_limit");
            return;
        }
        let row = StorageObservationRow {
            domain: ROVENT_STORAGE_DIAGNOSTIC_DOMAIN,
            event: if authority.is_some() {
                "after_root"
            } else {
                "after_finish"
            },
            sequence: state.storage_sequence,
            input_lock_sha256: hex(self.lock),
            implementation_sha256: hex(self.implementation),
            mode: mode(self.mode),
            observed_after_authority: authority.map(Into::into),
            source_status,
            storage: storage.as_ref(),
            diagnostics_complete: state.storage_first_issue.is_none()
                && !state.storage_scope_incomplete,
            diagnostic_failure: state.storage_first_issue,
            physical_completion_attested: false,
            capabilities_reissued: false,
        };
        match write_storage_row(&mut io::stderr().lock(), row, &mut state) {
            Ok(true) => {}
            Ok(false) => {
                state.storage_scope_incomplete = true;
                state
                    .storage_first_issue
                    .get_or_insert("storage_row_scope_incomplete");
            }
            Err(cause) => {
                state.storage_scope_incomplete = true;
                state.storage_first_issue.get_or_insert(cause);
            }
        }
    }
    #[cfg(feature = "pals-adaptive-diagnostics")]
    pub(super) fn install_profiled_feedback(
        &self,
        value: Result<
            crate::pals_adaptive::ContinuationDiagnostic,
            crate::pals_adaptive::ContinuationDiagnosticError,
        >,
    ) {
        let Ok(mut state) = self.state.lock() else {
            self.poisoned.store(true, Ordering::Release);
            return;
        };
        match value {
            Ok(value) if state.profiled_feedback.is_none() => state.profiled_feedback = Some(value),
            _ => state.profiled_feedback_issue = Some("profiled_feedback_install_unavailable"),
        }
    }
    /// One original root only. The existing diagnostic bind is once-only;
    /// later roots cannot reset/relabel that opaque journal into a new proof.
    #[cfg(feature = "pals-adaptive-diagnostics")]
    pub(super) fn observe_profiled_feedback(
        &self,
        product: &RoventNativeProductDriver,
        context: &SearchSessionContext,
    ) {
        let Ok(mut state) = self.state.lock() else {
            self.poisoned.store(true, Ordering::Release);
            return;
        };
        if state.profiled_feedback_emitted
            || (state.profiled_feedback.is_none() && state.profiled_feedback_issue.is_none())
        {
            return;
        }
        state.profiled_feedback_emitted = true;
        let snapshot = state
            .profiled_feedback
            .as_ref()
            .and_then(|d| d.snapshot().ok());
        let accepted = snapshot
            .as_ref()
            .ok_or("profiled_feedback_snapshot_unavailable")
            .and_then(|s| {
                if s.authority != Some(context.authority) {
                    return Err("profiled_feedback_wrong_root");
                }
                s.accepted_profiled_feedback_chain()
            });
        let commits = match product.inner.as_ref() {
            ActualProductDriver::Pcr(driver) => driver.profiled_retirement_commits(),
            _ => return,
        };
        let mut row = serde_json::json!({
            "domain":"rz-rovent-profiled-feedback-consumption-observation/1",
            "event":"first_original_root_readback",
            "input_lock_sha256":hex(self.lock),
            "implementation_sha256":hex(self.implementation),
            "authority":AuthorityObservation::from(context.authority),
            "maximum_root_observations":1,
            "semantic_chain_observed":accepted.is_ok(),
            "chain_unavailable":accepted.as_ref().err().copied(),
            "source_unavailable":snapshot.as_ref().and_then(|s| s.profiled_feedback.as_ref())
                .and_then(|v| v.unavailable()).map(|v| format!("{v:?}")),
            "role_readback_root_fence":snapshot.as_ref().and_then(|s| s.role_readback_root_fence),
            "journal_unavailable_bits":snapshot.as_ref().map(|s| s.unavailable_bits),
            "diagnostic_install_issue":state.profiled_feedback_issue,
            "original_archive_commits":commits.as_ref().ok(),
            "archive_observation_unavailable":commits.is_err(),
            "physical_completion_attested":false,"capabilities_reissued":false,
            "execution_authorized":false,"performance_evidence":false,
            "full_acceptance_issued":false,
        });
        if let Ok(chain) = accepted {
            let steps: Vec<_> = chain
                .steps()
                .iter()
                .map(|value| {
                    serde_json::json!({
                        "authority":AuthorityObservation::from(value.frame.authority),
                        "frame_sequence":value.frame.sequence,
                        "stage":format!("{:?}", value.stage),
                        "original_execution":format!("{:?}", value.execution),
                        "question_sha256":hex(value.submitted_question),
                        "view_sha256":hex(value.submitted_view.view_digest),
                        "view_sequence":value.submitted_view.view_sequence,
                        "public_context_policy_sha256":hex(value.submitted_view.context_policy),
                        "completed_evidence_sha256":hex(value.completed_evidence),
                    })
                })
                .collect();
            let repair = chain.repair();
            let pair = chain.fresh_pair();
            row["chain"] = serde_json::json!({
                "original_foreign_observation":chain.foreign_observation().0,
                "steps_source_r_p_repair_c_fresh_repaired_fresh_counter":steps,
                "prefix_before_plies":chain.prefix_before_plies(),
                "committed_repair_plies":repair.published_line.committed_plies,
                "repair_line":repair.published_line.line.0,
                "repair_observation":repair.published_line.observation.0,
                "record_revision":repair.published_line.record_revision,
                "parent_revision":repair.published_line.parent_revision,
                "supersedes_revision":repair.published_line.supersedes_revision,
                "frozen_namespace_sha256":hex(chain.namespace()),
                "fresh_public_context_sha256":hex(pair.public_context),
                "fresh_model_sha256":hex(pair.model),"fresh_encoding_sha256":hex(pair.encoding),
                "fresh_precision":format!("{:?}",pair.precision),"fresh_epoch":pair.frozen_epoch,
                "fresh_perspective":format!("{:?}",pair.perspective),
            });
        }
        if let Err(why) = write_profiled_feedback_row(&mut io::stderr().lock(), &row, &mut state) {
            state.first_issue.get_or_insert(why);
            state.scope_incomplete = true;
        }
    }
    pub(super) fn new_from_sources(
        lock: Digest,
        implementation: Digest,
        mode: ExecutionMode,
        physical_close: Arc<AtomicBool>,
    ) -> Self {
        Self {
            state: Arc::new(Mutex::new(State::default())),
            poisoned: Arc::new(AtomicBool::new(false)),
            physical_close,
            initial_stop_source: Arc::new(AtomicBool::new(false)),
            lock,
            implementation,
            mode,
        }
    }
    /// Last attempted diagnostic row, not a new live root or shutdown receipt.
    pub fn latest_observation(
        &self,
    ) -> Result<Option<RoventProductLifecycleObservation>, &'static str> {
        self.state
            .lock()
            .map(|state| state.latest.clone())
            .map_err(|_| "diagnostic_state_poisoned")
    }
    pub(super) fn observe_root(
        &self,
        product: &RoventNativeProductDriver,
        context: &SearchSessionContext,
        result: Result<&SearchSessionReport, &SearchSessionFailure>,
    ) {
        let failure = result.as_ref().err().copied();
        let (mut source_status, root) = project_root(product, context);
        let initial = project_initial(product, context);
        let retained = project_retained_session(product, "after_run");
        if root.is_none() && initial.is_some() {
            source_status = "original_initial_stop";
        }
        let Ok(mut state) = self.state.lock() else {
            self.poisoned.store(true, Ordering::Release);
            return;
        };
        match retained {
            Ok(value) => state.retained_session = value,
            Err(cause) => {
                state.first_issue.get_or_insert(cause);
            }
        }
        state.root_runs = state.root_runs.saturating_add(1);
        if failure.is_some() {
            state.root_run_errors = state.root_run_errors.saturating_add(1);
        }
        if state.root_runs == u64::MAX {
            state.first_issue.get_or_insert("root_sequence_overflow");
        }
        if !matches!(
            source_status,
            "original_current_root" | "original_initial_stop"
        ) {
            state
                .first_issue
                .get_or_insert("original_current_root_receipt_unavailable");
        }
        if state.root_rows >= MAX_ROOT_ROWS {
            state.first_issue.get_or_insert("root_row_limit");
        }
        let mut row = self.row(&state, "root", Some(context.authority), source_status, root);
        row.initial = initial;
        row.run_failure = failure.map(Into::into);
        if state.root_rows >= MAX_ROOT_ROWS {
            state.latest = Some(row);
            return;
        }
        if let Err(cause) = write_row(&mut io::stderr().lock(), &row, &mut state, false) {
            state.first_issue.get_or_insert(cause);
        } else {
            state.root_rows += 1;
        }
        row.diagnostics_complete = state.first_issue.is_none();
        row.scope_incomplete |= state.scope_incomplete;
        row.diagnostic_failure = state.first_issue;
        state.latest = Some(row);
        if self.initial_stop_source.load(Ordering::Acquire)
            && source_status == "original_initial_stop"
        {
            let companion = initial_stop::project(product, context, result, state.root_runs);
            let written = companion
                .and_then(|row| initial_stop::write(&mut io::stderr().lock(), &row, &mut state));
            if let Err(cause) = written {
                state.first_issue.get_or_insert(cause);
                state.scope_incomplete = true;
                let issue = state.first_issue;
                if let Some(latest) = &mut state.latest {
                    latest.diagnostics_complete = false;
                    latest.scope_incomplete = true;
                    latest.diagnostic_failure = issue;
                }
            }
        }
    }
    pub(super) fn observe_finish(
        &self,
        product: &RoventNativeProductDriver,
        result: &Result<Option<DriverPhysicalCompletion>, SearchSessionFailure>,
    ) {
        // This is after the original finish call has returned. Do not replace
        // the earlier root-cause snapshot, or infer Closed from its OS result.
        let retained = project_retained_session(product, "after_finish");
        let shared = match product.inner.as_ref() {
            ActualProductDriver::Pc(driver) | ActualProductDriver::PcNnuec(driver) => {
                driver.shared_nnuec_readback()
            }
            _ => Ok(None),
        };
        let Ok(mut state) = self.state.lock() else {
            self.poisoned.store(true, Ordering::Release);
            return;
        };
        match retained {
            Ok(value) => state.retained_session_after_finish = value,
            Err(cause) => {
                state.first_issue.get_or_insert(cause);
            }
        }
        state.finish_observed = true;
        state.finish_result = Some(match result {
            Ok(Some(value)) => physical(*value),
            Ok(None) => "unavailable",
            Err(_) => "failed",
        });
        state.finish_failure = result.as_ref().err().map(Into::into);
        match shared {
            Ok(value) => state.shared_nnuec_session = value.map(shared_observation),
            Err(_) => {
                state
                    .first_issue
                    .get_or_insert("shared_nnuec_session_readback_failed");
            }
        }
    }
    // Called after the existing UCI owner has completed its original cleanup.
    // Diagnostic failure cannot skip, replace or repeat that physical cleanup.
    pub(crate) fn complete_session(&self, protocol_succeeded: bool) -> Result<(), &'static str> {
        let Ok(mut state) = self.state.lock() else {
            self.poisoned.store(true, Ordering::Release);
            return Err("diagnostic_state_poisoned");
        };
        let mut row = self.row(
            &state,
            "session",
            None,
            "whole_session_finish_observation",
            None,
        );
        row.protocol_succeeded = Some(protocol_succeeded);
        if let Err(cause) = write_row(&mut io::stderr().lock(), &row, &mut state, true) {
            state.first_issue.get_or_insert(cause);
        }
        row.diagnostics_complete =
            state.first_issue.is_none() && !self.poisoned.load(Ordering::Acquire);
        row.scope_incomplete |= state.scope_incomplete;
        row.diagnostic_failure = state.first_issue;
        state.latest = Some(row);
        match state.first_issue {
            Some(cause) => Err(cause),
            None => Ok(()),
        }
    }
    fn row(
        &self,
        state: &State,
        event: &'static str,
        authority: Option<SearchAuthority>,
        source_status: &'static str,
        root: Option<RootObservation>,
    ) -> RoventProductLifecycleObservation {
        RoventProductLifecycleObservation {
            domain: ROVENT_LIFECYCLE_DIAGNOSTIC_DOMAIN,
            event,
            sequence: state
                .root_runs
                .saturating_add(u64::from(event == "session")),
            input_lock_sha256: hex(self.lock),
            mode: mode(self.mode),
            implementation_sha256: hex(self.implementation),
            authority: authority.map(Into::into),
            source_status,
            root,
            initial: None,
            run_failure: None,
            retained_session: state.retained_session.clone(),
            retained_session_after_finish: state.retained_session_after_finish.clone(),
            finish_observed: state.finish_observed,
            finish_result: state.finish_result.unwrap_or("not_observed"),
            finish_failure: state.finish_failure.clone(),
            shared_nnuec_session: state.shared_nnuec_session.clone(),
            whole_session_physical: if self.physical_close.load(Ordering::Acquire) {
                "confirmed"
            } else {
                "unknown"
            },
            protocol_succeeded: None,
            scope_incomplete: state.scope_incomplete,
            root_runs_observed: state.root_runs,
            root_run_error_results: state.root_run_errors,
            root_rows_emitted: state.root_rows,
            diagnostics_complete: state.first_issue.is_none()
                && !self.poisoned.load(Ordering::Acquire),
            diagnostic_failure: state.first_issue,
        }
    }
}

fn project_initial(
    product: &RoventNativeProductDriver,
    context: &SearchSessionContext,
) -> Option<InitialObservation> {
    let lock = product.sources.lock_digest();
    match product.inner.as_ref() {
        ActualProductDriver::Pc(driver) | ActualProductDriver::PcNnuec(driver) => {
            let receipt = driver.latest_initial().ok()??;
            if receipt.authority() != context.authority || receipt.input_lock_digest() != lock {
                return None;
            }
            Some(InitialObservation {
                receipt_domain: receipt.domain(),
                source_seal_sha256: hex(receipt.source().digest()),
                previous_clean_root_present: Some(receipt.previous_clean_root().is_some()),
                used_original_startup_ready: Some(receipt.used_original_startup_ready()),
                stopped_before_research_execution: None,
                deadline_before_frontier: None,
                original_nnuec_retirement_present: false,
                original_researcher_release_present: None,
                original_cache_release_epoch: None,
            })
        }
        ActualProductDriver::Nnuec(driver) => {
            if let Some(receipt) = driver.latest_root().ok()? {
                if receipt.authority == context.authority
                    && receipt.input_lock == lock
                    && receipt.frontier.is_none()
                    && receipt.root_admission.authority() == context.authority
                    && receipt.root_admission.startup_lock_digest() == lock
                    && same_original_start(&receipt.root_admission, context)
                {
                    return Some(InitialObservation {
                        receipt_domain: receipt.domain,
                        source_seal_sha256: hex(receipt.root_admission.digest()),
                        previous_clean_root_present: None,
                        used_original_startup_ready: None,
                        stopped_before_research_execution: Some(
                            receipt.stopped_before_research_execution,
                        ),
                        deadline_before_frontier: Some(receipt.deadline_before_frontier),
                        original_nnuec_retirement_present: receipt.initial_retirement.is_some(),
                        original_researcher_release_present: receipt
                            .initial_retirement
                            .as_ref()
                            .map(|value| value.release().is_some()),
                        original_cache_release_epoch: receipt
                            .initial_retirement
                            .as_ref()
                            .map(|value| value.cache_release().epoch()),
                    });
                }
            }
            let receipt = driver.latest_initial().ok()??;
            if receipt.authority() != context.authority || receipt.input_lock_digest() != lock {
                return None;
            }
            Some(InitialObservation {
                receipt_domain: receipt.domain(),
                source_seal_sha256: hex(receipt.source().digest()),
                previous_clean_root_present: Some(receipt.previous_clean_root().is_some()),
                used_original_startup_ready: Some(receipt.used_original_startup_ready()),
                stopped_before_research_execution: None,
                deadline_before_frontier: None,
                original_nnuec_retirement_present: false,
                original_researcher_release_present: None,
                original_cache_release_epoch: None,
            })
        }
        ActualProductDriver::Pcr(driver) => {
            let receipt = driver.latest_rovent_initial().ok()??;
            if receipt.authority() != context.authority
                || receipt.source().startup_lock_digest() != lock
            {
                return None;
            }
            Some(InitialObservation {
                receipt_domain: receipt.domain(),
                source_seal_sha256: hex(receipt.source().digest()),
                previous_clean_root_present: Some(
                    receipt.previous_clean_root().is_some()
                        || receipt.previous_clean_profiled_root().is_some(),
                ),
                used_original_startup_ready: Some(receipt.used_original_startup_ready()),
                stopped_before_research_execution: None,
                deadline_before_frontier: None,
                original_nnuec_retirement_present: false,
                original_researcher_release_present: None,
                original_cache_release_epoch: None,
            })
        }
    }
}

fn original_scope(
    admission: &RoventRootAdmission,
    authority: SearchAuthority,
    lock: Digest,
    context: &SearchSessionContext,
    frontier: &FrontierSnapshot,
) -> bool {
    authority == context.authority
        && admission.authority() == context.authority
        && admission.startup_lock_digest() == lock
        && frontier.authority == context.authority
        && frontier.original_deadline == admission.hard_deadline()
        && same_original_start(admission, context)
}
fn same_original_start(admission: &RoventRootAdmission, context: &SearchSessionContext) -> bool {
    admission
        .original_context()
        .original_start
        .as_ref()
        .zip(context.original_start.as_ref())
        .is_some_and(|(original, current)| Arc::ptr_eq(original, current))
}

/// Same current-root projection for the concrete driver and private producer
/// regression; neither the helper nor its serialized row can issue authority.
fn project_nnuec_root_receipt(
    receipt: &super::super::nnuec_session::RoventNnuecRootReceipt,
    lock: Digest,
    context: &SearchSessionContext,
) -> (&'static str, Option<RootObservation>) {
    let Some(frontier) = &receipt.frontier else {
        return ("original_frontier_not_admitted", None);
    };
    if receipt.input_lock != lock
        || !original_scope(
            &receipt.root_admission,
            receipt.authority,
            lock,
            context,
            frontier,
        )
        || receipt
            .physical
            .as_ref()
            .is_some_and(|value| value.authority != context.authority)
    {
        return ("original_root_scope_mismatch", None);
    }
    (
        "original_current_root",
        Some(RootObservation::new(
            receipt.domain,
            &receipt.root_admission,
            frontier,
            &receipt.counters,
            receipt
                .physical
                .as_ref()
                .map(RootPhysicalObservation::pcr)
                .or_else(|| {
                    receipt
                        .pre_materialization_retirement_matches()
                        .then(RootPhysicalObservation::pre_materialization)
                }),
        )),
    )
}

fn project_root(
    product: &RoventNativeProductDriver,
    context: &SearchSessionContext,
) -> (&'static str, Option<RootObservation>) {
    let lock = product.sources.lock_digest();
    match product.inner.as_ref() {
        ActualProductDriver::Pc(driver) | ActualProductDriver::PcNnuec(driver) => {
            let receipt = match driver.latest_root() {
                Ok(Some(receipt)) => receipt,
                Ok(None) => return ("no_original_root_receipt", None),
                Err(_) => return ("original_root_receipt_read_failed", None),
            };
            if receipt.input_lock != lock
                || !original_scope(
                    &receipt.root_admission,
                    receipt.authority,
                    lock,
                    context,
                    &receipt.frontier,
                )
                || receipt.physical.authority != context.authority
                || receipt.physical.root_budget != receipt.root_admission.root_budget_digest()
            {
                return ("original_root_scope_mismatch", None);
            }
            let mut row = RootObservation::new(
                receipt.domain,
                &receipt.root_admission,
                &receipt.frontier,
                &receipt.physical.counters,
                Some(RootPhysicalObservation::pc(&receipt.physical)),
            );
            row.r_free_confirmed = Some(
                !receipt.physical.mode.has_native_r()
                    && !product.sources.lock().manifest.engine.mode.has_native_r()
                    && receipt.frontier.work.native_usage_known
                    && receipt.frontier.work.native_nodes == 0
                    && receipt.frontier.work.native_qsearch_nodes == 0
                    && receipt.frontier.work.reserved_native_nodes == 0
                    && receipt.frontier.work.outstanding_native_executions == 0
                    && receipt.physical.counters.cpu_slice_receipts == 0,
            );
            row.shared_nnuec = match driver.shared_nnuec_readback() {
                Ok(value) => value.map(shared_observation),
                Err(_) => return ("shared_nnuec_readback_failed", None),
            };
            ("original_current_root", Some(row))
        }
        ActualProductDriver::Nnuec(driver) => {
            let receipt = match driver.latest_root() {
                Ok(Some(receipt)) => receipt,
                Ok(None) => return ("no_original_root_receipt", None),
                Err(_) => return ("original_root_receipt_read_failed", None),
            };
            project_nnuec_root_receipt(&receipt, lock, context)
        }
        ActualProductDriver::Pcr(driver) => {
            let receipt = match driver.latest_rovent_root() {
                Ok(Some(receipt)) => receipt,
                Ok(None) => return ("no_original_root_receipt", None),
                Err(_) => return ("original_root_receipt_read_failed", None),
            };
            let (Some(admission), Some(frontier)) = (&receipt.root_admission, &receipt.frontier)
            else {
                return ("original_frontier_not_admitted", None);
            };
            if receipt.input_lock != lock
                || !original_scope(admission, receipt.authority, lock, context, frontier)
                || receipt
                    .physical
                    .as_ref()
                    .is_some_and(|value| value.authority != context.authority)
            {
                return ("original_root_scope_mismatch", None);
            }
            let physical = match select_pcr_physical(
                receipt.domain,
                receipt.physical.as_ref().map(RootPhysicalObservation::pcr),
                receipt
                    .profiled_physical
                    .as_ref()
                    .map(RootPhysicalObservation::profiled),
            ) {
                Ok(value) => value,
                Err(cause) => return (cause, None),
            };
            if receipt
                .profiled_physical
                .as_ref()
                .is_some_and(|value| value.authority() != context.authority)
            {
                return ("original_profiled_root_scope_mismatch", None);
            }
            (
                "original_current_root",
                Some(RootObservation::new(
                    receipt.domain,
                    admission,
                    frontier,
                    &receipt.counters,
                    physical,
                )),
            )
        }
    }
}

// A missing external receipt remains missing. Never fall back to a native
// zero-ledger receipt, and never turn a metadata domain into a live proof.
fn select_pcr_physical<T>(
    domain: &str,
    native: Option<T>,
    profiled: Option<T>,
) -> Result<Option<T>, &'static str> {
    match domain {
        crate::pals_adaptive::ROVENT_NATIVE_ROOT_DOMAIN if profiled.is_none() => Ok(native),
        crate::pals_adaptive::ROVENT_PROFILED_EXTERNAL_ROOT_DOMAIN if native.is_none() => {
            Ok(profiled)
        }
        _ => Err("original_researcher_receipt_kind_mismatch"),
    }
}

fn project_retained_session(
    product: &RoventNativeProductDriver,
    observation_point: &'static str,
) -> Result<Option<RetainedSessionObservation>, &'static str> {
    let ActualProductDriver::Pcr(driver) = product.inner.as_ref() else {
        return Ok(None);
    };
    let value = driver
        .retained_session_observation()
        .map_err(|_| "retained_session_readback_unavailable")?;
    project_retained_observation(value, observation_point).map(Some)
}
fn project_retained_observation(
    value: crate::pals_adaptive::AdaptiveRetainedSessionObservation,
    observation_point: &'static str,
) -> Result<RetainedSessionObservation, &'static str> {
    // The retained manager/archive authority is explicitly separate from the
    // outer attempted root. An old new-game source must not be relabeled.
    let physical = match (&value.native_physical, &value.profiled_physical) {
        (Some(native), None) => Some(RootPhysicalObservation::pcr(native)),
        (None, Some(profiled)) => Some(RootPhysicalObservation::profiled(profiled)),
        (None, None) => None,
        (Some(_), Some(_)) => return Err("retained_researcher_receipt_kind_mismatch"),
    };
    Ok(RetainedSessionObservation {
        observation_point,
        stage: value.stage,
        original_authority: value.authority.map(Into::into),
        next_authority: value.next_authority.map(Into::into),
        manager_phase: value.manager_phase,
        manager_first_retained_failure: value
            .manager_first_retained_failure
            .as_ref()
            .map(Into::into),
        manager_retained_failure_count: value.manager_retained_failure_count,
        archive_operation: value.archive_operation,
        archive_quarantine_reason: value.archive_quarantine_reason.as_ref().map(Into::into),
        physical,
        physical_read_failure: value.physical_read_failure.as_ref().map(Into::into),
    })
}

// Allocate only the registered row bound before serialization. serde's writer
// cannot grow this buffer, even if a future DTO accidentally gains a long field.
struct RowBuffer(Vec<u8>);
impl Write for RowBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > MAX_ROW_BYTES.saturating_sub(self.0.len()) {
            return Err(io::Error::other("lifecycle diagnostic row byte limit"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
// Compact only observational detail. Keep original stage/authority/cause,
// and explicitly revoke completeness; this creates no physical proof. The
// same 8192-byte RowBuffer is reused, before any OS byte has been written.
fn compact_failure(value: &FailureObservation) -> FailureObservation {
    FailureObservation {
        code: bounded(&value.code, 32),
        code_truncated: value.code_truncated || value.code.len() > 32,
        detail: bounded(&value.detail, 64),
        detail_truncated: value.detail_truncated || value.detail.len() > 64,
        physical_completion: value.physical_completion,
    }
}
fn compact_row(value: &RoventProductLifecycleObservation) -> RoventProductLifecycleObservation {
    let mut row = value.clone();
    row.root = None;
    row.initial = None;
    row.shared_nnuec_session = None;
    row.run_failure = row.run_failure.as_ref().map(compact_failure);
    row.finish_failure = row.finish_failure.as_ref().map(compact_failure);
    for retained in [
        &mut row.retained_session,
        &mut row.retained_session_after_finish,
    ]
    .into_iter()
    .flatten()
    {
        retained.physical = None;
        retained.manager_first_retained_failure = retained
            .manager_first_retained_failure
            .as_ref()
            .map(compact_failure);
        retained.archive_quarantine_reason = retained
            .archive_quarantine_reason
            .as_ref()
            .map(compact_failure);
        retained.physical_read_failure =
            retained.physical_read_failure.as_ref().map(compact_failure);
    }
    row.scope_incomplete = true;
    row.diagnostics_complete = false;
    row.diagnostic_failure
        .get_or_insert("diagnostic_row_scope_incomplete");
    row
}
fn write_row(
    output: &mut impl Write,
    row: &RoventProductLifecycleObservation,
    state: &mut State,
    final_row: bool,
) -> Result<(), &'static str> {
    let mut buffer = RowBuffer(Vec::new());
    buffer
        .0
        .try_reserve_exact(MAX_ROW_BYTES)
        .map_err(|_| "diagnostic_allocation_failed")?;
    let row_exceeded = match serde_json::to_writer(&mut buffer, row) {
        Ok(()) => buffer.write_all(b"\n").is_err(),
        // RowBuffer's only IO error is its finite byte limit. Other codec
        // failures are not repaired into a successful diagnostic row.
        Err(error) if error.is_io() => true,
        Err(_) => return Err("diagnostic_row_encoding_or_byte_limit"),
    };
    if row_exceeded {
        buffer.0.clear();
        let compacted = compact_row(row);
        serde_json::to_writer(&mut buffer, &compacted)
            .map_err(|_| "diagnostic_compact_row_encoding_or_byte_limit")?;
        buffer
            .write_all(b"\n")
            .map_err(|_| "diagnostic_compact_row_byte_limit")?;
        state.scope_incomplete = true;
        state
            .first_issue
            .get_or_insert("diagnostic_row_scope_incomplete");
    }
    let limit = if final_row {
        MAX_STREAM_BYTES
    } else {
        MAX_STREAM_BYTES - MAX_ROW_BYTES
    };
    if buffer.0.len() > limit.saturating_sub(state.bytes) {
        return Err("diagnostic_stream_byte_limit");
    }
    // Charge a partial/failed OS write once as a full attempted row. The final
    // receipt remains incomplete; retrying a torn JSONL record is prohibited.
    state.bytes += buffer.0.len();
    output
        .write_all(&buffer.0)
        .map_err(|_| "diagnostic_write_failed")?;
    output.flush().map_err(|_| "diagnostic_flush_failed")
}

#[derive(Serialize)]
struct StorageObservationRow<'a> {
    domain: &'static str,
    event: &'static str,
    sequence: u64,
    input_lock_sha256: String,
    implementation_sha256: String,
    mode: &'static str,
    observed_after_authority: Option<AuthorityObservation>,
    source_status: &'static str,
    storage: Option<&'a super::super::storage_lifecycle::StorageLifecycleObservation>,
    diagnostics_complete: bool,
    diagnostic_failure: Option<&'static str>,
    physical_completion_attested: bool,
    capabilities_reissued: bool,
}

fn write_storage_row(
    output: &mut impl Write,
    mut row: StorageObservationRow<'_>,
    state: &mut State,
) -> Result<bool, &'static str> {
    let mut buffer = RowBuffer(Vec::new());
    buffer
        .0
        .try_reserve_exact(MAX_ROW_BYTES)
        .map_err(|_| "storage_diagnostic_allocation_failed")?;
    let exceeded = match serde_json::to_writer(&mut buffer, &row) {
        Ok(()) => buffer.write_all(b"\n").is_err(),
        Err(error) if error.is_io() => true,
        Err(_) => return Err("storage_diagnostic_encoding_failed"),
    };
    if exceeded {
        // Do not turn truncated counters or a missing receipt into success.
        buffer.0.clear();
        row.storage = None;
        row.source_status = "bounded_storage_row_incomplete";
        row.diagnostics_complete = false;
        row.diagnostic_failure = Some("storage_row_scope_incomplete");
        serde_json::to_writer(&mut buffer, &row)
            .map_err(|_| "storage_diagnostic_fallback_failed")?;
        buffer
            .write_all(b"\n")
            .map_err(|_| "storage_diagnostic_fallback_byte_limit")?;
    }
    if buffer.0.len() > (MAX_STREAM_BYTES - MAX_ROW_BYTES).saturating_sub(state.bytes) {
        return Err("storage_diagnostic_stream_byte_limit");
    }
    // Same global stream accounting/final lifecycle reserve. Never retry a
    // partially written JSONL row and never alter product cleanup on failure.
    state.bytes += buffer.0.len();
    output
        .write_all(&buffer.0)
        .map_err(|_| "storage_diagnostic_write_failed")?;
    output
        .flush()
        .map_err(|_| "storage_diagnostic_flush_failed")?;
    Ok(!exceeded)
}
#[cfg(feature = "pals-adaptive-diagnostics")]
fn write_profiled_feedback_row(
    output: &mut impl Write,
    row: &serde_json::Value,
    state: &mut State,
) -> Result<(), &'static str> {
    let mut buffer = RowBuffer(Vec::new());
    buffer
        .0
        .try_reserve_exact(MAX_ROW_BYTES)
        .map_err(|_| "profiled_feedback_allocation_failed")?;
    if serde_json::to_writer(&mut buffer, row).is_err() || buffer.write_all(b"\n").is_err() {
        // Never truncate a positive into a seemingly complete chain.
        buffer.0.clear();
        serde_json::to_writer(
            &mut buffer,
            &serde_json::json!({
                "domain":"rz-rovent-profiled-feedback-consumption-observation/1",
                "event":"first_original_root_readback","semantic_chain_observed":false,
                "chain_unavailable":"bounded_row_incomplete","physical_completion_attested":false,
                "capabilities_reissued":false,"full_acceptance_issued":false,
            }),
        )
        .map_err(|_| "profiled_feedback_fallback_encoding_failed")?;
        buffer
            .write_all(b"\n")
            .map_err(|_| "profiled_feedback_fallback_capacity")?;
        state.scope_incomplete = true;
        state
            .first_issue
            .get_or_insert("profiled_feedback_row_incomplete");
    }
    if buffer.0.len() > (MAX_STREAM_BYTES - MAX_ROW_BYTES).saturating_sub(state.bytes) {
        return Err("profiled_feedback_stream_capacity");
    }
    // Same final-row reserve and one charged attempt as lifecycle output.
    state.bytes += buffer.0.len();
    output
        .write_all(&buffer.0)
        .map_err(|_| "profiled_feedback_write_failed")?;
    output.flush().map_err(|_| "profiled_feedback_flush_failed")
}
fn physical(value: DriverPhysicalCompletion) -> &'static str {
    match value {
        DriverPhysicalCompletion::Confirmed => "confirmed",
        DriverPhysicalCompletion::Unknown => "unknown",
    }
}
fn shared_observation(
    value: super::super::pc_session::RoventPcSharedNnuecReadback,
) -> SharedNnuecObservation {
    SharedNnuecObservation {
        spawned_runtime_helpers: value.spawned_runtime_helpers,
        worker_thread_id_observed: value.worker_thread_id.is_some(),
        worker_joined: value.worker_joined,
        worker_thread_affinity_known: value.worker_thread_affinity.is_some(),
    }
}
fn mode(value: ExecutionMode) -> &'static str {
    match value {
        ExecutionMode::CpuPc => "cpu_pc",
        ExecutionMode::CpuPcr => "cpu_pcr",
        ExecutionMode::HybridPc => "hybrid_pc",
        ExecutionMode::HybridPcr => "hybrid_pcr",
    }
}
fn goal(value: FrontierGoalStatus) -> &'static str {
    match value {
        FrontierGoalStatus::Pending => "pending",
        FrontierGoalStatus::DepthSatisfied => "depth_satisfied",
        FrontierGoalStatus::MateSatisfied => "mate_satisfied",
        FrontierGoalStatus::SelectiveFrontier => "selective_frontier",
        FrontierGoalStatus::GoalUnresolved => "goal_unresolved",
        FrontierGoalStatus::Stopped(reason) => match reason {
            FrontierStop::Canceled => "stopped_canceled",
            FrontierStop::Deadline => "stopped_deadline",
            FrontierStop::WorkBudget => "stopped_work_budget",
            FrontierStop::RoleBudget => "stopped_role_budget",
            FrontierStop::Capacity => "stopped_capacity",
            FrontierStop::NativeUnknown => "stopped_native_unknown",
            FrontierStop::EvidenceReclaimed => "stopped_evidence_reclaimed",
            FrontierStop::Failed => "stopped_failed",
        },
    }
}
fn bounded(value: &str, limit: usize) -> String {
    let mut end = value.len().min(limit);
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].into()
}
fn hex(value: Digest) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(64);
    for byte in value.0 {
        result.push(DIGITS[(byte >> 4) as usize] as char);
        result.push(DIGITS[(byte & 15) as usize] as char);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nnuec_original_retirement_producer_matcher_and_projection_keep_scope() {
        crate::rovent_bootstrap::nnuec_session::pre_materialization_tests::check_projection(
            |receipt, lock, context| {
                let (status, root) = project_nnuec_root_receipt(receipt, lock, context);
                serde_json::json!({"observation_status": status, "root": root})
            },
        );
    }

    fn diagnostic() -> RoventLifecycleDiagnostics {
        RoventLifecycleDiagnostics::new_from_sources(
            Digest([3; 32]),
            Digest([4; 32]),
            ExecutionMode::CpuPc,
            Arc::new(AtomicBool::new(false)),
        )
    }

    #[test]
    fn external_physical_projection_never_substitutes_native_receipts() {
        let external = crate::pals_adaptive::ROVENT_PROFILED_EXTERNAL_ROOT_DOMAIN;
        let native = crate::pals_adaptive::ROVENT_NATIVE_ROOT_DOMAIN;
        assert_eq!(select_pcr_physical::<u8>(external, None, None), Ok(None));
        assert_eq!(select_pcr_physical(external, None, Some(7)), Ok(Some(7)));
        assert!(select_pcr_physical(external, Some(3), None).is_err());
        assert!(select_pcr_physical(native, None, Some(7)).is_err());
        assert!(select_pcr_physical(native, Some(3), Some(7)).is_err());
        assert!(select_pcr_physical::<u8>("unknown", None, None).is_err());
    }

    #[test]
    fn protocol_success_and_root_rows_cannot_manufacture_a_session_finish() {
        let diagnostics = diagnostic();
        let state = State::default();
        let mut row = diagnostics.row(
            &state,
            "session",
            None,
            "whole_session_finish_observation",
            None,
        );
        row.protocol_succeeded = Some(true);
        let mut output = Vec::new();
        let mut accounting = State::default();
        write_row(&mut output, &row, &mut accounting, true).unwrap();
        let decoded: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(decoded["protocol_succeeded"], true);
        assert_eq!(decoded["whole_session_physical"], "unknown");
        assert_eq!(decoded["finish_observed"], false);
        assert_eq!(decoded["finish_result"], "not_observed");
        assert!(decoded["root"].is_null());
        assert!(decoded["initial"].is_null());
        assert_eq!(output.last(), Some(&b'\n'));
    }

    #[test]
    fn storage_domain_uses_original_stream_budget_without_issuing_physical_authority() {
        let row = StorageObservationRow {
            domain: ROVENT_STORAGE_DIAGNOSTIC_DOMAIN,
            event: "after_finish",
            sequence: 1,
            input_lock_sha256: "1".repeat(64),
            implementation_sha256: "2".repeat(64),
            mode: "cpu_pc",
            observed_after_authority: None,
            source_status: "original_storage_readback_unavailable",
            storage: None,
            diagnostics_complete: false,
            diagnostic_failure: Some("fixture_readback_unknown"),
            physical_completion_attested: false,
            capabilities_reissued: false,
        };
        let mut state = State::default();
        let mut output = Vec::new();
        assert!(write_storage_row(&mut output, row, &mut state).unwrap());
        let decoded: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(decoded["domain"], ROVENT_STORAGE_DIAGNOSTIC_DOMAIN);
        assert!(decoded["storage"].is_null());
        assert_eq!(decoded["diagnostics_complete"], false);
        assert_eq!(decoded["physical_completion_attested"], false);
        assert_eq!(state.bytes, output.len());
        assert!(state.first_issue.is_none());
        assert!(state.latest.is_none());
    }

    #[test]
    fn bounded_stream_retains_the_final_incomplete_observation() {
        let diagnostics = diagnostic();
        let mut raw_cause =
            SearchSessionFailure::debug("RetainedManagerEarliestFailure", &"원인".repeat(1024));
        raw_cause.physical_completion = DriverPhysicalCompletion::Unknown;
        let projected_cause = FailureObservation::from(&raw_cause);
        assert!(raw_cause.detail.len() <= 512 + " [cause truncated]".len());
        assert!(raw_cause.detail.ends_with(" [cause truncated]"));
        assert!(projected_cause.detail.len() <= 512);
        assert!(projected_cause.detail_truncated);
        let mut state = State {
            retained_session: Some(RetainedSessionObservation {
                observation_point: "after_run",
                stage: "manager_retained",
                original_authority: None,
                next_authority: None,
                manager_phase: Some("quarantined"),
                manager_first_retained_failure: Some(projected_cause),
                manager_retained_failure_count: Some(1),
                archive_operation: None,
                archive_quarantine_reason: None,
                physical: None,
                physical_read_failure: None,
            }),
            root_runs: MAX_ROOT_ROWS + 1,
            root_rows: MAX_ROOT_ROWS,
            bytes: MAX_STREAM_BYTES - MAX_ROW_BYTES,
            first_issue: Some("root_row_limit"),
            ..State::default()
        };
        let row = diagnostics.row(
            &state,
            "session",
            None,
            "whole_session_finish_observation",
            None,
        );
        let mut output = Vec::new();
        assert_eq!(
            write_row(&mut output, &row, &mut state, false),
            Err("diagnostic_stream_byte_limit")
        );
        assert!(output.is_empty());
        write_row(&mut output, &row, &mut state, true).unwrap();
        let decoded: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(decoded["diagnostics_complete"], false);
        assert_eq!(decoded["diagnostic_failure"], "root_row_limit");
        assert_eq!(decoded["root_runs_observed"], MAX_ROOT_ROWS + 1);
        assert_eq!(decoded["root_rows_emitted"], MAX_ROOT_ROWS);
        let cause = &decoded["retained_session"]["manager_first_retained_failure"];
        assert!(cause["detail"].as_str().unwrap().len() <= 512);
        assert_eq!(cause["detail_truncated"], true);
        assert_eq!(cause["physical_completion"], "unknown");
        assert_eq!(decoded["whole_session_physical"], "unknown");
        assert!(state.bytes <= MAX_STREAM_BYTES);
    }

    fn maximum_retained_projection(observation_point: &'static str) -> RetainedSessionObservation {
        use rz_contracts::{GameGeneration, ProcessEpoch, RootGeneration};
        let authority = SearchAuthority {
            epoch: ProcessEpoch(u64::MAX),
            game: GameGeneration(u64::MAX),
            root: RootGeneration(u64::MAX),
            implementation: Digest([255; 32]),
        };
        let mut cause =
            SearchSessionFailure::debug("RetainedManagerEarliestFailure", &"\0".repeat(1024));
        cause.physical_completion = DriverPhysicalCompletion::Unknown;
        project_retained_observation(
            crate::pals_adaptive::AdaptiveRetainedSessionObservation {
                stage: "manager_retained",
                authority: Some(authority),
                next_authority: None,
                manager_phase: Some("quarantined"),
                manager_first_retained_failure: Some(cause.clone()),
                manager_retained_failure_count: Some(64),
                archive_operation: None,
                archive_quarantine_reason: None,
                native_physical: None,
                profiled_physical: None,
                physical_read_failure: Some(cause),
            },
            observation_point,
        )
        .unwrap()
    }
    #[test]
    fn maximum_retained_projection_keeps_root_and_reserved_final_cause() {
        let diagnostics = diagnostic();
        let retained = maximum_retained_projection("after_run");
        assert_eq!(retained.stage, "manager_retained");
        assert_eq!(retained.original_authority.as_ref().unwrap().root, u64::MAX);
        assert!(
            retained
                .manager_first_retained_failure
                .as_ref()
                .unwrap()
                .detail_truncated
        );
        let mut after_finish = maximum_retained_projection("after_finish");
        // A failed finish may retain this same quarantined manager; a copied
        // after-run observation must not be relabeled as an after-finish read.
        let mut state = State {
            root_runs: MAX_ROOT_ROWS,
            root_rows: MAX_ROOT_ROWS - 1,
            root_run_errors: MAX_ROOT_ROWS,
            retained_session: Some(retained),
            ..State::default()
        };
        let mut row = diagnostics.row(&state, "root", None, "original_current_root", None);
        row.run_failure = state
            .retained_session
            .as_ref()
            .unwrap()
            .manager_first_retained_failure
            .clone();
        // Largest root observation scalar/identity fields, without inventing
        // a live admission or a fence. No absent profiled receipt is filled in.
        row.root = Some(RootObservation {
            receipt_domain: crate::pals_adaptive::ROVENT_PROFILED_EXTERNAL_ROOT_DOMAIN,
            root_admission_sha256: "f".repeat(64),
            root_budget_sha256: "f".repeat(64),
            deadline_seal_sha256: "f".repeat(64),
            state_and_history_sha256: "f".repeat(64),
            goal_status: "stopped_native_unknown",
            completed_depth: Some(u16::MAX),
            maximum_examined_ply: u16::MAX,
            native_usage_known: false,
            native_nodes: None,
            native_qsearch_nodes: None,
            reserved_native_nodes: u64::MAX,
            outstanding_native_executions: usize::MAX,
            counters: CountersObservation {
                submitted_role_tickets: u64::MAX,
                physical_role_completions: u64::MAX,
                accepted_nn_inputs: Some(u64::MAX),
                consumed_role_outputs: u64::MAX,
                completed_nn_inputs: Some(u64::MAX),
                consumed_nn_inputs: Some(u64::MAX),
                unused_role_results: u64::MAX,
                cpu_slice_receipts: u64::MAX,
                validator_ticket_admissions: u64::MAX,
            },
            physical: None,
            r_free_confirmed: None,
            shared_nnuec: None,
        });
        let mut root_output = Vec::new();
        write_row(&mut root_output, &row, &mut state, false).unwrap();
        assert!(root_output.len() <= MAX_ROW_BYTES);
        let root: serde_json::Value = serde_json::from_slice(&root_output).unwrap();
        assert_eq!(root["retained_session"]["stage"], "manager_retained");
        assert_eq!(
            root["retained_session"]["original_authority"]["root"],
            u64::MAX
        );
        assert_eq!(root["retained_session"]["observation_point"], "after_run");
        assert!(root["retained_session_after_finish"].is_null());
        // Adversarial escape expansion inside the already bounded DTO schema.
        // This stress case is not evidence of a live root or authority.
        let mut expanded = row.clone();
        let extreme = FailureObservation {
            code: "F".repeat(128),
            code_truncated: false,
            detail: "\0".repeat(512),
            detail_truncated: false,
            physical_completion: "unknown",
        };
        state.first_issue = Some("first_original_readback_failure");
        expanded.diagnostic_failure = state.first_issue;
        expanded.diagnostics_complete = false;
        expanded.run_failure = Some(extreme.clone());
        expanded.finish_failure = Some(extreme.clone());
        let held = expanded.retained_session.as_mut().unwrap();
        held.manager_first_retained_failure = Some(extreme.clone());
        held.physical_read_failure = Some(extreme.clone());
        assert!(serde_json::to_vec(&expanded).unwrap().len() > MAX_ROW_BYTES);
        let mut expanded_output = Vec::new();
        write_row(&mut expanded_output, &expanded, &mut state, false).unwrap();
        let compacted: serde_json::Value = serde_json::from_slice(&expanded_output).unwrap();
        assert!(expanded_output.len() <= MAX_ROW_BYTES);
        assert_eq!(compacted["scope_incomplete"], true);
        assert_eq!(compacted["diagnostics_complete"], false);
        assert_eq!(compacted["retained_session"]["stage"], "manager_retained");
        assert_eq!(
            compacted["retained_session"]["original_authority"]["root"],
            u64::MAX
        );
        assert_eq!(
            compacted["retained_session"]["manager_first_retained_failure"]["detail_truncated"],
            true
        );
        assert_eq!(compacted["whole_session_physical"], "unknown");
        assert!(compacted["retained_session_after_finish"].is_null());
        assert_eq!(
            compacted["diagnostic_failure"],
            "first_original_readback_failure"
        );
        // Preserve the original projected cause in State, not just the first
        // compact copy. The reserved final output must survive this stress.
        state.retained_session = expanded.retained_session;
        after_finish.manager_first_retained_failure = Some(FailureObservation {
            code: "AfterFinishPhysicalReceipt".into(),
            ..extreme.clone()
        });
        after_finish.physical_read_failure = Some(extreme.clone());
        state.retained_session_after_finish = Some(after_finish);
        state.finish_observed = true;
        state.finish_result = Some("failed");
        state.finish_failure = Some(extreme);
        state.bytes = MAX_STREAM_BYTES - MAX_ROW_BYTES;
        let final_row = diagnostics.row(
            &state,
            "session",
            None,
            "whole_session_finish_observation",
            None,
        );
        let mut final_output = Vec::new();
        write_row(&mut final_output, &final_row, &mut state, true).unwrap();
        let final_value: serde_json::Value = serde_json::from_slice(&final_output).unwrap();
        assert!(final_output.len() <= MAX_ROW_BYTES);
        assert!(state.bytes <= MAX_STREAM_BYTES);
        assert_eq!(final_value["scope_incomplete"], true);
        assert_eq!(final_value["retained_session"]["stage"], "manager_retained");
        assert_eq!(
            final_value["retained_session"]["original_authority"]["root"],
            u64::MAX
        );
        assert_eq!(
            final_value["retained_session"]["manager_first_retained_failure"]["code"],
            "F".repeat(32)
        );
        assert_eq!(
            final_value["retained_session"]["manager_first_retained_failure"]["code_truncated"],
            true
        );
        assert_eq!(
            final_value["retained_session"]["manager_first_retained_failure"]["detail"],
            "\0".repeat(64)
        );
        assert_eq!(
            final_value["retained_session"]["manager_first_retained_failure"]["detail_truncated"],
            true
        );
        assert_eq!(
            final_value["retained_session"]["observation_point"],
            "after_run"
        );
        assert_eq!(
            final_value["retained_session_after_finish"]["observation_point"],
            "after_finish"
        );
        assert_eq!(
            final_value["retained_session_after_finish"]["stage"],
            "manager_retained"
        );
        assert_eq!(
            final_value["retained_session_after_finish"]["original_authority"]["root"],
            u64::MAX
        );
        assert!(final_value["retained_session_after_finish"]["next_authority"].is_null());
        assert_eq!(
            final_value["retained_session_after_finish"]["manager_first_retained_failure"]["code"],
            "AfterFinishPhysicalReceipt"
        );
        assert_eq!(
            final_value["retained_session_after_finish"]["physical_read_failure"]["detail_truncated"],
            true
        );
        assert_eq!(
            final_value["diagnostic_failure"],
            "first_original_readback_failure"
        );
        assert_eq!(final_value["finish_observed"], true);
        assert_eq!(final_value["finish_result"], "failed");
        assert_eq!(
            final_value["retained_session"]["manager_first_retained_failure"]["physical_completion"],
            "unknown"
        );
        assert_eq!(final_value["whole_session_physical"], "unknown");
    }

    #[test]
    fn partial_diagnostic_io_cannot_advance_a_root_or_physical_completion() {
        struct TornOutput(usize);
        impl Write for TornOutput {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                if self.0 == 0 {
                    return Err(io::Error::other("original FD rejected write"));
                }
                let count = bytes.len().min(self.0);
                self.0 -= count;
                Ok(count)
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let diagnostics = diagnostic();
        let mut state = State::default();
        let row = diagnostics.row(&state, "root", None, "no_original_root_receipt", None);
        assert_eq!(
            write_row(&mut TornOutput(7), &row, &mut state, false),
            Err("diagnostic_write_failed")
        );
        assert!(state.bytes > 7); // Conservatively retain the entire attempted write charge.
        assert_eq!(state.root_rows, 0);
        assert!(!state.finish_observed);
        assert!(!diagnostics.physical_close.load(Ordering::Acquire));
    }
}
