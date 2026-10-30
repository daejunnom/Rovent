//! The existing settled-slice diagnostic owns its original bounded handle.
//! This adapter copies scalar receipts after root return; it cannot request
//! Yield, resume a token, change credit or certify semantic consumption.
use super::*;
use crate::pals_adaptive::{
    AdaptiveCpuSliceDiagnostic, AdaptiveCpuSliceDiagnosticOptions,
    AdaptiveCpuSliceDiagnosticSnapshot, AdaptiveCpuSliceObservation,
    AdaptiveCpuSliceResultObservation,
};

const RECORDS_MAX: usize = 32;

#[derive(Default)]
struct Original {
    authority: Option<SearchAuthority>,
    root_admission: Option<Digest>,
    installation_attempted: bool,
    installed: bool,
    installation_error: Option<&'static str>,
    handle: Option<AdaptiveCpuSliceDiagnostic>,
}
#[derive(Default)]
pub(super) struct CpuSliceJournal {
    original: Mutex<Original>,
    unavailable: AtomicBool,
}
impl CpuSliceJournal {
    pub(super) fn begin(&self, authority: SearchAuthority, root_admission: Digest) {
        match self.original.try_lock() {
            Ok(mut original) => {
                *original = Original {
                    authority: Some(authority),
                    root_admission: Some(root_admission),
                    ..Original::default()
                }
            }
            Err(_) => self.unavailable.store(true, Ordering::Release),
        }
    }
    pub(super) fn install(&self, manager: &Manager) {
        let Ok(mut original) = self.original.try_lock() else {
            self.unavailable.store(true, Ordering::Release);
            return;
        };
        if original.installation_attempted || original.authority.is_none() {
            original.installation_error = Some("original_slice_scope_missing_or_repeated");
            return;
        }
        original.installation_attempted = true;
        let handle = match AdaptiveCpuSliceDiagnostic::new(AdaptiveCpuSliceDiagnosticOptions {
            records_max: RECORDS_MAX,
        }) {
            Ok(handle) => handle,
            Err(_) => {
                original.installation_error = Some("original_slice_handle_allocation_failed");
                return;
            }
        };
        // The exact same handle is retained even on installation rejection.
        // No retry, alternate manager, fresh window or shadow records exist.
        original.handle = Some(handle.clone());
        match manager.install_cpu_slice_diagnostic(handle) {
            Ok(()) => original.installed = true,
            Err(_) => original.installation_error = Some("original_slice_installation_rejected"),
        }
    }
    pub(super) fn snapshot(&self) -> Result<Option<CpuSliceOwnerObservation>, &'static str> {
        if self.unavailable.load(Ordering::Acquire) {
            return Err("original_slice_scope_unavailable");
        }
        let original = self
            .original
            .try_lock()
            .map_err(|_| "original_slice_scope_busy_or_poisoned")?;
        let (Some(authority), Some(root_admission)) = (original.authority, original.root_admission)
        else {
            return Ok(None);
        };
        let snapshot = original
            .handle
            .as_ref()
            .map(AdaptiveCpuSliceDiagnostic::snapshot)
            .transpose()
            .map_err(|_| "original_slice_snapshot_busy_or_poisoned")?;
        project(
            authority,
            root_admission,
            original.installation_attempted,
            original.installed,
            original.installation_error,
            snapshot,
        )
        .map(Some)
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct CpuSliceOwnerObservation {
    pub summary: CpuSliceSummary,
    pub records: Vec<SettledSlice>,
    // A separate versioned writer consumes these; existing slice JSON stays
    // unchanged. Missing handle never becomes an empty observed chain.
    #[serde(skip)]
    pub nnuec_chain: Vec<crate::pals_adaptive::NnuecChainObservation>,
    #[serde(skip)]
    pub nnuec_chain_dropped: Option<u64>,
}
#[derive(Clone, Debug, Serialize)]
pub(crate) struct CpuSliceSummary {
    pub authority: Authority,
    pub root_admission_sha256: String,
    pub installation_attempted: bool,
    pub original_handle_installed: bool,
    pub installation_error: Option<&'static str>,
    pub records_max: usize,
    pub retained_prefix_records: Option<usize>,
    pub total_settled_records: Option<usize>,
    pub omitted_records: Option<usize>,
    pub unavailable: Option<String>,
    pub original_handle_authority: Option<Authority>,
    pub original_root_scope_matches: Option<bool>,
    pub scope: &'static str,
}

fn project(
    authority: SearchAuthority,
    root_admission: Digest,
    attempted: bool,
    installed: bool,
    error: Option<&'static str>,
    snapshot: Option<AdaptiveCpuSliceDiagnosticSnapshot>,
) -> Result<CpuSliceOwnerObservation, &'static str> {
    if snapshot
        .as_ref()
        .is_some_and(|s| s.records.len().saturating_add(s.nnuec_chain.len()) > RECORDS_MAX)
    {
        return Err("original_slice_snapshot_bound_exceeded");
    }
    let scope_matches = snapshot.as_ref().map(|s| s.authority == Some(authority));
    let complete = installed
        && scope_matches == Some(true)
        && snapshot.as_ref().is_some_and(|s| s.unavailable.is_none());
    let records = snapshot
        .as_ref()
        .map(|s| s.records.iter().map(SettledSlice::from).collect())
        .unwrap_or_default();
    Ok(CpuSliceOwnerObservation {
        summary: CpuSliceSummary {
            authority: Authority::from_original(authority),
            root_admission_sha256: hex(root_admission),
            installation_attempted: attempted,
            original_handle_installed: installed,
            installation_error: error,
            records_max: RECORDS_MAX,
            retained_prefix_records: snapshot.as_ref().map(|s| s.records.len()),
            total_settled_records: complete.then(|| {
                snapshot
                    .as_ref()
                    .expect("observed original scope")
                    .records
                    .len()
            }),
            omitted_records: complete.then_some(0),
            unavailable: snapshot
                .as_ref()
                .and_then(|s| s.unavailable)
                .map(|v| format!("{v:?}")),
            original_handle_authority: snapshot
                .as_ref()
                .and_then(|s| s.authority)
                .map(Authority::from_original),
            original_root_scope_matches: scope_matches,
            scope: "original_manager_settled_receipt_prefix_not_token_or_semantic_authority",
        },
        records,
        nnuec_chain: snapshot
            .as_ref()
            .map(|s| s.nnuec_chain.clone())
            .unwrap_or_default(),
        nnuec_chain_dropped: snapshot.as_ref().map(|s| s.nnuec_chain_dropped),
    })
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct WorkObservation {
    nodes: u64,
    quiescence_nodes: u64,
    tt_hits: u64,
}
impl From<rz_search::cpu::CpuWork> for WorkObservation {
    fn from(value: rz_search::cpu::CpuWork) -> Self {
        Self {
            nodes: value.nodes,
            quiescence_nodes: value.quiescence_nodes,
            tt_hits: value.tt_hits,
        }
    }
}
#[derive(Clone, Debug, Serialize)]
pub(crate) struct ControlsObservation {
    cancel_sequence: u64,
    yield_sequence: u64,
    publish_sequence: u64,
    cancel_ack: u64,
    yield_ack: u64,
    publish_ack: u64,
    terminal: bool,
}
impl From<rz_runtime::adaptive::ControlSnapshot> for ControlsObservation {
    fn from(value: rz_runtime::adaptive::ControlSnapshot) -> Self {
        Self {
            cancel_sequence: value.cancel_sequence,
            yield_sequence: value.yield_sequence,
            publish_sequence: value.publish_sequence,
            cancel_ack: value.cancel_ack,
            yield_ack: value.yield_ack,
            publish_ack: value.publish_ack,
            terminal: value.terminal,
        }
    }
}
#[derive(Clone, Debug, Serialize)]
pub(crate) struct SettlementObservation {
    global_spent: u64,
    global_reserved: u64,
    job_spent: u64,
    job_reserved: u64,
    job_quarantined: bool,
}
#[derive(Clone, Debug, Serialize)]
pub(crate) struct ResultObservation {
    completion: Option<String>,
    error_kind: Option<String>,
    completed_depth: Option<u16>,
    reported_work: Option<WorkObservation>,
    score_scope: Option<String>,
    reused_completed_depth: Option<u16>,
    yield_cause: Option<String>,
    yield_ack_sequence: Option<u64>,
    publish_ack_sequence: Option<u64>,
    elapsed_ns: Option<u64>,
}
impl From<AdaptiveCpuSliceResultObservation> for ResultObservation {
    fn from(value: AdaptiveCpuSliceResultObservation) -> Self {
        match value {
            AdaptiveCpuSliceResultObservation::Report {
                completion,
                completed_depth,
                reported_work,
                score_scope,
                reused_completed_depth,
                yield_cause,
                yield_ack_sequence,
                publish_ack_sequence,
                elapsed,
            } => Self {
                completion: Some(format!("{completion:?}")),
                error_kind: None,
                completed_depth: Some(completed_depth),
                reported_work: Some(reported_work.into()),
                score_scope: Some(format!("{score_scope:?}")),
                reused_completed_depth: Some(reused_completed_depth),
                yield_cause: yield_cause.map(|v| format!("{v:?}")),
                yield_ack_sequence,
                publish_ack_sequence: Some(publish_ack_sequence),
                elapsed_ns: ns(elapsed),
            },
            AdaptiveCpuSliceResultObservation::Error(error) => Self {
                completion: None,
                error_kind: Some(format!("{error:?}")),
                completed_depth: None,
                reported_work: None,
                score_scope: None,
                reused_completed_depth: None,
                yield_cause: None,
                yield_ack_sequence: None,
                publish_ack_sequence: None,
                elapsed_ns: None,
            },
        }
    }
}
#[derive(Clone, Debug, Serialize)]
pub(crate) struct SettledSlice {
    pub sequence: u64,
    authority: Authority,
    frame_authority: Authority,
    frame_sequence: u64,
    stage: String,
    question_sha256: String,
    view_authority: Authority,
    view_sha256: String,
    view_sequence: u64,
    view_context_policy_sha256: String,
    view_root_state_semantic_sha256: String,
    public_context_sha256: String,
    job_authority: Authority,
    job_sequence: u64,
    job_question_sha256: String,
    state_owner: u64,
    state_revision: u64,
    state_semantic_sha256: String,
    root_restriction_sha256: String,
    evaluation_namespace_sha256: String,
    search_conditions_sha256: String,
    total_native_node_credit: u64,
    hard_deadline_clock_epoch: u64,
    hard_deadline_tick_ns: u64,
    grant_job_authority: Authority,
    grant_job_sequence: u64,
    grant_execution_epoch: u64,
    grant_execution_sequence: u64,
    grant_owner: u16,
    grant_node_credit: u64,
    grant_soft_lease_ns: Option<u64>,
    grant_control_sequence: u64,
    requested_yield_sequence: Option<u64>,
    prior_yield_ack: u64,
    binding_progress_sequence: u64,
    progress_since_slice: bool,
    feedback_already_forwarded: bool,
    actual_work: Option<WorkObservation>,
    result: ResultObservation,
    cumulative_nodes: u64,
    report_progress_sequence: u64,
    continuation_retained: bool,
    terminal: bool,
    admission_rejected: bool,
    controls: Option<ControlsObservation>,
    accounting_after_settlement: Option<SettlementObservation>,
    worker_resume_admission_and_restore_ns: Option<u64>,
    worker_return_processing_ns: Option<u64>,
    #[cfg(feature = "native-validation-diagnostics")]
    #[serde(skip)]
    nnuec_scalar: Option<serde_json::Value>,
}
impl From<&AdaptiveCpuSliceObservation> for SettledSlice {
    fn from(value: &AdaptiveCpuSliceObservation) -> Self {
        let scope = value.scope;
        let job = scope.job;
        let grant = value.grant;
        Self {
            sequence: value.sequence,
            authority: Authority::from_original(scope.authority),
            frame_authority: Authority::from_original(scope.frame.authority),
            frame_sequence: scope.frame.sequence,
            stage: format!("{:?}", scope.stage),
            question_sha256: hex(scope.question),
            view_authority: Authority::from_original(scope.view.authority),
            view_sha256: hex(scope.view.view_digest),
            view_sequence: scope.view.view_sequence,
            view_context_policy_sha256: hex(scope.view.context_policy),
            view_root_state_semantic_sha256: hex(scope.view.root_state.semantic),
            public_context_sha256: hex(scope.public_context),
            job_authority: Authority::from_original(job.id.authority),
            job_sequence: job.id.sequence,
            job_question_sha256: hex(job.question),
            state_owner: job.state_identity.owner.0,
            state_revision: job.state_identity.revision.0,
            state_semantic_sha256: hex(job.state_identity.semantic),
            root_restriction_sha256: hex(job.root_restriction),
            evaluation_namespace_sha256: hex(job.evaluation_namespace),
            search_conditions_sha256: hex(job.search_conditions),
            total_native_node_credit: job.total_native_node_credit,
            hard_deadline_clock_epoch: job.hard_deadline.clock.0.0,
            hard_deadline_tick_ns: job.hard_deadline.at.0,
            grant_job_authority: Authority::from_original(grant.job.authority),
            grant_job_sequence: grant.job.sequence,
            grant_execution_epoch: grant.execution.epoch.0,
            grant_execution_sequence: grant.execution.sequence,
            grant_owner: grant.owner.0,
            grant_node_credit: grant.node_credit,
            grant_soft_lease_ns: ns(grant.soft_lease),
            grant_control_sequence: grant.control_sequence,
            requested_yield_sequence: scope.requested_yield_sequence,
            prior_yield_ack: scope.prior_yield_ack,
            binding_progress_sequence: scope.binding_progress_sequence,
            progress_since_slice: scope.progress_since_slice,
            feedback_already_forwarded: scope.feedback_already_forwarded,
            actual_work: value.actual_work.map(Into::into),
            result: value.result.into(),
            cumulative_nodes: value.cumulative_nodes,
            report_progress_sequence: value.report_progress_sequence,
            continuation_retained: value.continuation_retained,
            terminal: value.terminal,
            admission_rejected: value.admission_rejected,
            controls: value.controls.map(Into::into),
            accounting_after_settlement: value.accounting_after_settlement.map(|a| {
                SettlementObservation {
                    global_spent: a.global_spent,
                    global_reserved: a.global_reserved,
                    job_spent: a.job_spent,
                    job_reserved: a.job_reserved,
                    job_quarantined: a.job_quarantined,
                }
            }),
            worker_resume_admission_and_restore_ns: value
                .worker_timing
                .and_then(|t| t.resume_admission_and_restore())
                .and_then(ns),
            worker_return_processing_ns: value
                .worker_timing
                .and_then(|t| ns(t.return_processing())),
            #[cfg(feature = "native-validation-diagnostics")]
            nnuec_scalar: value
                .worker_timing
                .and_then(|timing| timing.nnuec_scalar())
                .map(scalar_observation),
        }
    }
}
#[cfg(feature = "native-validation-diagnostics")]
impl SettledSlice {
    pub(crate) fn nnuec_scalar(&self) -> Option<&serde_json::Value> {
        self.nnuec_scalar.as_ref()
    }
}
#[cfg(feature = "native-validation-diagnostics")]
fn scalar_observation(
    value: rz_search::cpu_value::nnuec::v2::scalar_diagnostic::SliceObservation,
) -> serde_json::Value {
    use serde_json::json;
    let sample = |value: rz_search::cpu_value::nnuec::v2::scalar_diagnostic::Sample| {
        json!({
        "binding_sha256":hex(value.binding),"full_view_sha256":hex(Digest(value.full_view)),
        "view_seal_sha256":hex(Digest(value.view_seal)),"record_scope_sha256":hex(Digest(value.record_scope)),
        "generation":value.generation,"raw_value_f32_bits":value.raw_value_bits,
        "scaled_value_f32_bits":value.scaled_value_bits,"returned_frontier_score_i32":value.returned_frontier_score})
    };
    let stack = |value: rz_search::cpu::CpuPausedStackSnapshot| {
        json!({
        "owner_id":value.owner_id,"tokens_resumed":value.tokens_resumed,
        "tokens_retained":value.tokens_retained,"replayed_consumed_work":value.replayed_consumed_work,
        "complete":value.complete})
    };
    json!({"scope":"actual_private_worker_score_returns_not_semantic_acceptance",
        "successful_score_returns":value.scalar.map(|value|value.successful_score_returns),
        "diagnostic_incomplete":value.scalar.map(|value|value.incomplete),
        "first":value.scalar.and_then(|value|value.first).map(sample),
        "last":value.scalar.and_then(|value|value.last).map(sample),
        "resume_command":value.resume_command,"prior_token_present":value.prior_token_present,
        "stack_before":value.before.map(stack),"stack_after":value.after.map(stack),
        "new_forward_inferred_from_resume":false,"token_or_job_authority_issued":false})
}
fn ns(value: Duration) -> Option<u64> {
    u64::try_from(value.as_nanos()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn authority() -> SearchAuthority {
        SearchAuthority {
            epoch: rz_contracts::ProcessEpoch(1),
            game: rz_contracts::GameGeneration(2),
            root: rz_contracts::RootGeneration(3),
            implementation: Digest([4; 32]),
        }
    }
    #[test]
    fn missing_slice_handle_is_not_an_observed_zero() {
        let projected = project(authority(), Digest([5; 32]), false, false, None, None).unwrap();
        assert!(projected.summary.retained_prefix_records.is_none());
        assert!(projected.summary.total_settled_records.is_none());
        assert!(projected.summary.omitted_records.is_none());
        assert!(!projected.summary.original_handle_installed);
    }
    #[test]
    fn overflow_keeps_prefix_but_total_and_omitted_count_unknown() {
        let projected = project(
            authority(),
            Digest([5; 32]),
            true,
            true,
            None,
            Some(AdaptiveCpuSliceDiagnosticSnapshot {
                authority: Some(authority()),
                records_max: RECORDS_MAX,
                unavailable: Some(
                    crate::pals_adaptive::AdaptiveCpuSliceDiagnosticUnavailable::Overflow,
                ),
                records: Vec::new(),
                nnuec_chain: Vec::new(),
                nnuec_chain_dropped: 0,
            }),
        )
        .unwrap();
        assert_eq!(projected.summary.retained_prefix_records, Some(0));
        assert!(projected.summary.total_settled_records.is_none());
        assert!(projected.summary.omitted_records.is_none());
        assert_eq!(projected.summary.unavailable.as_deref(), Some("Overflow"));
    }
    #[test]
    fn exact_empty_original_scope_can_report_zero_without_reconstructing_work() {
        let projected = project(
            authority(),
            Digest([5; 32]),
            true,
            true,
            None,
            Some(AdaptiveCpuSliceDiagnosticSnapshot {
                authority: Some(authority()),
                records_max: RECORDS_MAX,
                unavailable: None,
                records: Vec::new(),
                nnuec_chain: Vec::new(),
                nnuec_chain_dropped: 0,
            }),
        )
        .unwrap();
        assert_eq!(projected.summary.total_settled_records, Some(0));
        assert_eq!(projected.summary.omitted_records, Some(0));
    }
    #[test]
    fn different_original_handle_root_is_preserved_without_known_completion() {
        let mut other = authority();
        other.root = rz_contracts::RootGeneration(9);
        let projected = project(
            authority(),
            Digest([5; 32]),
            true,
            true,
            None,
            Some(AdaptiveCpuSliceDiagnosticSnapshot {
                authority: Some(other),
                records_max: RECORDS_MAX,
                unavailable: None,
                records: Vec::new(),
                nnuec_chain: Vec::new(),
                nnuec_chain_dropped: 0,
            }),
        )
        .unwrap();
        assert_eq!(projected.summary.original_root_scope_matches, Some(false));
        assert!(
            projected
                .summary
                .original_handle_authority
                .as_ref()
                .unwrap()
                .matches(other)
        );
        assert!(projected.summary.total_settled_records.is_none());
        assert!(projected.summary.omitted_records.is_none());
    }
}
