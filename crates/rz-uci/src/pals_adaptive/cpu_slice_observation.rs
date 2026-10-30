//! Explicit own-CPU diagnostic only. A copied settled receipt is not semantic
//! consumption, a resume token, a control request, or a physical owner fence.
use rz_contracts::Digest;
use rz_contracts::pals::SearchAuthority;
use rz_contracts::pals_adaptive::{CpuJobEnvelope, ReviewViewSeal, SliceGrant};
use rz_runtime::adaptive::ControlSnapshot;
use rz_search::cpu::{CpuCompletion, CpuError, CpuScoreScope, CpuWork, CpuYieldCause};
use rz_search::cpu_worker::{CpuSliceReport, CpuWorkerSliceTiming};
use rz_search::pals::coordinator::{FrameId, RefinementStage};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::time::Duration;

#[cfg(feature = "pals-adaptive-diagnostics")]
mod nnuec_chain;
#[cfg(feature = "pals-adaptive-diagnostics")]
pub use nnuec_chain::*;

pub const CPU_SLICE_DIAGNOSTIC_RECORDS_MAX: usize = 256;

/// Explicit diagnostic opt-in. The ordinary manager allocates no journal.
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveCpuSliceDiagnosticOptions {
    pub records_max: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveCpuSliceDiagnosticError {
    InvalidCapacity,
    Allocation,
    AlreadyBound,
    AlreadyInstalled,
    AdmissionAlreadyStarted,
    Busy,
    Poisoned,
}
impl std::fmt::Display for AdaptiveCpuSliceDiagnosticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CPU slice diagnostic: {self:?}")
    }
}
impl std::error::Error for AdaptiveCpuSliceDiagnosticError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum AdaptiveCpuSliceDiagnosticUnavailable {
    Overflow = 1,
    Contended = 2,
    Poisoned = 3,
    OriginalReceiptRejected = 4,
    AccountingUnavailable = 5,
    ScopeMismatch = 6,
    InternalUnavailable = 7,
    InstallationRejected = 8,
}
impl AdaptiveCpuSliceDiagnosticUnavailable {
    fn decode(value: u8) -> Option<Self> {
        match value {
            0 => None,
            1 => Some(Self::Overflow),
            2 => Some(Self::Contended),
            3 => Some(Self::Poisoned),
            4 => Some(Self::OriginalReceiptRejected),
            5 => Some(Self::AccountingUnavailable),
            6 => Some(Self::ScopeMismatch),
            8 => Some(Self::InstallationRejected),
            _ => Some(Self::InternalUnavailable),
        }
    }
}

/// Classification only. The original CpuError stays on its existing report
/// and error path; no dynamic backend text or evaluator payload is retained.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveCpuSliceErrorKind {
    InvalidConfig,
    InvalidLimits,
    InvalidRootMoves,
    ResumeMismatch,
    PausedStackMemoryLimit,
    Unsupported,
    Allocation,
    Rules,
    Value,
}
impl AdaptiveCpuSliceErrorKind {
    fn from_original(error: &CpuError) -> Self {
        match error {
            CpuError::InvalidConfig(_) => Self::InvalidConfig,
            CpuError::InvalidLimits(_) => Self::InvalidLimits,
            CpuError::InvalidRootMoves(_) => Self::InvalidRootMoves,
            CpuError::ResumeMismatch(_) => Self::ResumeMismatch,
            CpuError::PausedStackMemoryLimit { .. } => Self::PausedStackMemoryLimit,
            CpuError::Unsupported(_) => Self::Unsupported,
            CpuError::Allocation => Self::Allocation,
            CpuError::Rules(_) => Self::Rules,
            CpuError::Value(_) => Self::Value,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveCpuSliceResultObservation {
    Report {
        completion: CpuCompletion,
        completed_depth: u16,
        reported_work: CpuWork,
        score_scope: CpuScoreScope,
        reused_completed_depth: u16,
        yield_cause: Option<CpuYieldCause>,
        yield_ack_sequence: Option<u64>,
        publish_ack_sequence: u64,
        elapsed: Duration,
    },
    Error(AdaptiveCpuSliceErrorKind),
}

/// The original CPU question survives a logical frame's first feedback. These
/// are that binding's seals, not a newly published or reconstructed view.
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveCpuSliceScopeObservation {
    pub authority: SearchAuthority,
    pub frame: FrameId,
    pub stage: RefinementStage,
    pub question: Digest,
    pub view: ReviewViewSeal,
    pub public_context: Digest,
    pub job: CpuJobEnvelope,
    pub requested_yield_sequence: Option<u64>,
    pub prior_yield_ack: u64,
    /// Original binding watermark; this does not prove semantic forwarding.
    pub binding_progress_sequence: u64,
    pub progress_since_slice: bool,
    pub feedback_already_forwarded: bool,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptiveCpuSliceAccountingObservation {
    pub global_spent: u64,
    pub global_reserved: u64,
    pub job_spent: u64,
    pub job_reserved: u64,
    pub job_quarantined: bool,
}
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveCpuSliceObservation {
    pub sequence: u64,
    pub scope: AdaptiveCpuSliceScopeObservation,
    pub grant: SliceGrant,
    pub actual_work: Option<CpuWork>,
    pub result: AdaptiveCpuSliceResultObservation,
    /// Copied raw. For admission_rejected this is not settlement evidence.
    pub cumulative_nodes: u64,
    pub report_progress_sequence: u64,
    pub continuation_retained: bool,
    pub terminal: bool,
    pub admission_rejected: bool,
    pub controls: Option<ControlSnapshot>,
    pub accounting_after_settlement: Option<AdaptiveCpuSliceAccountingObservation>,
    /// Missing for receipts recovered during final join. No new clock sample.
    pub worker_timing: Option<CpuWorkerSliceTiming>,
}

/// A retained prefix only. Even a fault-free snapshot is not a root/process
/// fence or proof of semantic consumption. Empty means NOT_OBSERVED for Y1.
#[derive(Clone, Debug)]
pub struct AdaptiveCpuSliceDiagnosticSnapshot {
    pub authority: Option<SearchAuthority>,
    pub records_max: usize,
    pub unavailable: Option<AdaptiveCpuSliceDiagnosticUnavailable>,
    pub records: Vec<AdaptiveCpuSliceObservation>,
    #[cfg(feature = "pals-adaptive-diagnostics")]
    pub nnuec_chain: Vec<NnuecChainObservation>,
    #[cfg(feature = "pals-adaptive-diagnostics")]
    pub nnuec_chain_dropped: u64,
}
#[derive(Debug)]
struct State {
    authority: Option<SearchAuthority>,
    records: Vec<AdaptiveCpuSliceObservation>,
    #[cfg(feature = "pals-adaptive-diagnostics")]
    nnuec_chain: Vec<NnuecChainObservation>,
}
#[derive(Debug)]
struct Inner {
    records_max: usize,
    state: Mutex<State>,
    unavailable: AtomicU8,
    #[cfg(feature = "pals-adaptive-diagnostics")]
    nnuec_chain_dropped: std::sync::atomic::AtomicU64,
}
/// Explicit bounded metadata collector. It owns no worker, controls, native
/// credit, Rules proof, tensor, callback, or I/O handle. Clone shares only this
/// journal; its sole producer is the original manager's settled-report path.
#[derive(Clone, Debug)]
pub struct AdaptiveCpuSliceDiagnostic {
    inner: Arc<Inner>,
}
impl AdaptiveCpuSliceDiagnostic {
    pub fn new(
        options: AdaptiveCpuSliceDiagnosticOptions,
    ) -> Result<Self, AdaptiveCpuSliceDiagnosticError> {
        if options.records_max == 0 || options.records_max > CPU_SLICE_DIAGNOSTIC_RECORDS_MAX {
            return Err(AdaptiveCpuSliceDiagnosticError::InvalidCapacity);
        }
        let mut records = Vec::new();
        records
            .try_reserve_exact(options.records_max)
            .map_err(|_| AdaptiveCpuSliceDiagnosticError::Allocation)?;
        #[cfg(feature = "pals-adaptive-diagnostics")]
        let mut nnuec_chain = Vec::new();
        #[cfg(feature = "pals-adaptive-diagnostics")]
        nnuec_chain
            .try_reserve_exact(options.records_max)
            .map_err(|_| AdaptiveCpuSliceDiagnosticError::Allocation)?;
        Ok(Self {
            inner: Arc::new(Inner {
                records_max: options.records_max,
                state: Mutex::new(State {
                    authority: None,
                    records,
                    #[cfg(feature = "pals-adaptive-diagnostics")]
                    nnuec_chain,
                }),
                unavailable: AtomicU8::new(0),
                #[cfg(feature = "pals-adaptive-diagnostics")]
                nnuec_chain_dropped: std::sync::atomic::AtomicU64::new(0),
            }),
        })
    }
    /// Read after the original root/process has quiesced. Reading while live
    /// may contend with its producer and makes that missing event unavailable.
    /// This never drains or overwrites a prefix to manufacture a complete run.
    pub fn snapshot(
        &self,
    ) -> Result<AdaptiveCpuSliceDiagnosticSnapshot, AdaptiveCpuSliceDiagnosticError> {
        let state = match self.inner.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Err(AdaptiveCpuSliceDiagnosticError::Busy),
            Err(TryLockError::Poisoned(_)) => {
                self.mark_unavailable(AdaptiveCpuSliceDiagnosticUnavailable::Poisoned);
                return Err(AdaptiveCpuSliceDiagnosticError::Poisoned);
            }
        };
        Ok(AdaptiveCpuSliceDiagnosticSnapshot {
            authority: state.authority,
            records_max: self.inner.records_max,
            records: state.records.clone(),
            #[cfg(feature = "pals-adaptive-diagnostics")]
            nnuec_chain: state.nnuec_chain.clone(),
            #[cfg(feature = "pals-adaptive-diagnostics")]
            nnuec_chain_dropped: self.inner.nnuec_chain_dropped.load(Ordering::Acquire),
            unavailable: AdaptiveCpuSliceDiagnosticUnavailable::decode(
                self.inner.unavailable.load(Ordering::Acquire),
            ),
        })
    }
    pub(crate) fn bind(
        &self,
        authority: SearchAuthority,
    ) -> Result<(), AdaptiveCpuSliceDiagnosticError> {
        let mut state = match self.inner.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Err(AdaptiveCpuSliceDiagnosticError::Busy),
            Err(TryLockError::Poisoned(_)) => {
                return Err(AdaptiveCpuSliceDiagnosticError::Poisoned);
            }
        };
        if state.authority.is_some() {
            return Err(AdaptiveCpuSliceDiagnosticError::AlreadyBound);
        }
        state.authority = Some(authority);
        Ok(())
    }
    pub(crate) fn mark_unavailable(&self, reason: AdaptiveCpuSliceDiagnosticUnavailable) {
        // The first lost/invalid event is sticky. It cannot alter the original
        // manager Result or release/refund a physical reservation.
        let _ = self.inner.unavailable.compare_exchange(
            0,
            reason as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
    pub(crate) fn record_settled(
        &self,
        scope: AdaptiveCpuSliceScopeObservation,
        report: &CpuSliceReport,
        accounting: Option<AdaptiveCpuSliceAccountingObservation>,
        worker_timing: Option<CpuWorkerSliceTiming>,
    ) {
        if self.inner.unavailable.load(Ordering::Acquire) != 0 {
            return;
        }
        let mut state = match self.inner.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => {
                self.mark_unavailable(AdaptiveCpuSliceDiagnosticUnavailable::Contended);
                return;
            }
            Err(TryLockError::Poisoned(_)) => {
                self.mark_unavailable(AdaptiveCpuSliceDiagnosticUnavailable::Poisoned);
                return;
            }
        };
        if self.inner.unavailable.load(Ordering::Acquire) != 0 {
            return;
        }
        if state.authority != Some(scope.authority)
            || scope.frame.authority != scope.authority
            || scope.view.authority != scope.authority
            || scope.job.id.authority != scope.authority
            || report.grant.job != scope.job.id
        {
            self.mark_unavailable(AdaptiveCpuSliceDiagnosticUnavailable::ScopeMismatch);
            return;
        }
        let retained_records = state.records.len();
        #[cfg(feature = "pals-adaptive-diagnostics")]
        let retained_records = retained_records.saturating_add(state.nnuec_chain.len());
        if retained_records >= self.inner.records_max {
            self.mark_unavailable(AdaptiveCpuSliceDiagnosticUnavailable::Overflow);
            return;
        }
        let result = match &report.result {
            Ok(actual) => AdaptiveCpuSliceResultObservation::Report {
                completion: actual.report.completion,
                completed_depth: actual.report.completed_depth,
                reported_work: CpuWork {
                    nodes: actual.report.nodes,
                    quiescence_nodes: actual.report.quiescence_nodes,
                    tt_hits: actual.report.tt_hits,
                },
                score_scope: actual.report.score_scope,
                reused_completed_depth: actual.report.reused_completed_depth,
                yield_cause: actual.yield_cause,
                yield_ack_sequence: actual.yield_ack_sequence,
                publish_ack_sequence: actual.publish_ack_sequence,
                elapsed: actual.report.elapsed,
            },
            Err(error) => AdaptiveCpuSliceResultObservation::Error(
                AdaptiveCpuSliceErrorKind::from_original(error),
            ),
        };
        // records_max <=256; this adds no allocation after explicit opt-in.
        let sequence = state.records.len() as u64 + 1;
        state.records.push(AdaptiveCpuSliceObservation {
            sequence,
            scope,
            grant: report.grant,
            actual_work: report.actual_work,
            result,
            cumulative_nodes: report.cumulative_nodes,
            report_progress_sequence: report.progress_sequence,
            continuation_retained: report.continuation_retained,
            terminal: report.terminal,
            admission_rejected: report.admission_rejected,
            controls: report.controls,
            accounting_after_settlement: accounting,
            worker_timing,
        });
        if accounting.is_none() {
            self.mark_unavailable(AdaptiveCpuSliceDiagnosticUnavailable::AccountingUnavailable);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::pals_adaptive::{CpuJobId, CpuWorkerId};
    use rz_contracts::{
        ClockDomain, Deadline, ExecutionId, GameGeneration, MonotonicTick, OwnerId, ProcessEpoch,
        RootGeneration, StateIdentity, StateRevision,
    };

    // Authored metadata for bounded-queue tests only; never actual CPU evidence.
    fn sample() -> (
        AdaptiveCpuSliceScopeObservation,
        CpuSliceReport,
        AdaptiveCpuSliceAccountingObservation,
    ) {
        let authority = SearchAuthority {
            epoch: ProcessEpoch(7),
            game: GameGeneration(1),
            root: RootGeneration(1),
            implementation: Digest([1; 32]),
        };
        let state = StateIdentity {
            owner: OwnerId(1),
            revision: StateRevision(1),
            semantic: Digest([2; 32]),
        };
        let job = CpuJobEnvelope {
            id: CpuJobId {
                authority,
                sequence: 1,
            },
            state_identity: state,
            question: Digest([3; 32]),
            root_restriction: Digest([4; 32]),
            evaluation_namespace: Digest([5; 32]),
            search_conditions: Digest([6; 32]),
            total_native_node_credit: 128,
            hard_deadline: Deadline {
                clock: ClockDomain(authority.epoch),
                at: MonotonicTick(1000),
            },
        };
        let scope = AdaptiveCpuSliceScopeObservation {
            authority,
            frame: FrameId {
                authority,
                sequence: 1,
            },
            stage: RefinementStage::CpuValidation,
            question: Digest([7; 32]),
            public_context: Digest([8; 32]),
            job,
            view: ReviewViewSeal {
                authority,
                view_sequence: 1,
                context_policy: Digest([9; 32]),
                view_digest: Digest([10; 32]),
                root_state: state,
                actual_tokens: 8,
                selected_records: 0,
                anchors: 1,
                length_bucket: 8,
            },
            requested_yield_sequence: Some(5),
            prior_yield_ack: 3,
            binding_progress_sequence: 0,
            progress_since_slice: false,
            feedback_already_forwarded: false,
        };
        let report = CpuSliceReport {
            reciprocal_request: None,
            grant: SliceGrant {
                job: job.id,
                execution: ExecutionId {
                    epoch: authority.epoch,
                    sequence: 1,
                },
                owner: CpuWorkerId(0),
                node_credit: 1,
                soft_lease: Duration::from_millis(8),
                control_sequence: 1,
            },
            actual_work: Some(CpuWork::default()),
            result: Err(CpuError::Allocation),
            cumulative_nodes: 0,
            progress_sequence: 0,
            continuation_retained: false,
            terminal: true,
            admission_rejected: false,
            controls: Some(ControlSnapshot {
                cancel_sequence: 0,
                yield_sequence: 5,
                publish_sequence: 6,
                cancel_ack: 0,
                yield_ack: 5,
                publish_ack: 6,
                terminal: true,
            }),
        };
        let accounting = AdaptiveCpuSliceAccountingObservation {
            global_spent: 0,
            global_reserved: 0,
            job_spent: 0,
            job_reserved: 0,
            job_quarantined: false,
        };
        (scope, report, accounting)
    }
    fn journal(capacity: usize) -> AdaptiveCpuSliceDiagnostic {
        AdaptiveCpuSliceDiagnostic::new(AdaptiveCpuSliceDiagnosticOptions {
            records_max: capacity,
        })
        .unwrap()
    }
    #[test]
    fn explicit_caps_and_one_root_binding_do_not_mint_observations() {
        for records_max in [0, CPU_SLICE_DIAGNOSTIC_RECORDS_MAX + 1] {
            assert!(matches!(
                AdaptiveCpuSliceDiagnostic::new(AdaptiveCpuSliceDiagnosticOptions { records_max }),
                Err(AdaptiveCpuSliceDiagnosticError::InvalidCapacity)
            ));
        }
        let diagnostic = journal(1);
        let (scope, _, _) = sample();
        assert!(diagnostic.snapshot().unwrap().authority.is_none());
        diagnostic.bind(scope.authority).unwrap();
        assert_eq!(
            diagnostic.bind(scope.authority),
            Err(AdaptiveCpuSliceDiagnosticError::AlreadyBound)
        );
        let snapshot = diagnostic.snapshot().unwrap();
        assert_eq!(snapshot.authority, Some(scope.authority));
        assert!(snapshot.records.is_empty());
        assert_eq!(snapshot.unavailable, None);
    }
    #[test]
    fn overflow_keeps_original_prefix_and_raw_error_class_without_overwrite() {
        let diagnostic = journal(1);
        let (scope, report, accounting) = sample();
        diagnostic.bind(scope.authority).unwrap();
        diagnostic.record_settled(scope, &report, Some(accounting), None);
        diagnostic.record_settled(scope, &report, Some(accounting), None);
        diagnostic.record_settled(scope, &report, None, None);
        let snapshot = diagnostic.snapshot().unwrap();
        assert_eq!(
            snapshot.unavailable,
            Some(AdaptiveCpuSliceDiagnosticUnavailable::Overflow)
        );
        assert_eq!(snapshot.records.len(), 1);
        let recorded = snapshot.records[0];
        assert_eq!(recorded.sequence, 1);
        assert_eq!(recorded.grant, report.grant);
        assert_eq!(recorded.actual_work, report.actual_work);
        assert_eq!(recorded.controls, report.controls);
        assert_eq!(recorded.scope.requested_yield_sequence, Some(5));
        assert_eq!(recorded.scope.prior_yield_ack, 3);
        assert_eq!(
            recorded.result,
            AdaptiveCpuSliceResultObservation::Error(AdaptiveCpuSliceErrorKind::Allocation)
        );
        assert_eq!(recorded.accounting_after_settlement, Some(accounting));
        assert!(recorded.worker_timing.is_none());
    }
    #[test]
    fn missing_accounting_and_foreign_scope_are_unavailable_instead_of_zero() {
        let diagnostic = journal(2);
        let (scope, report, _) = sample();
        diagnostic.bind(scope.authority).unwrap();
        diagnostic.record_settled(scope, &report, None, None);
        let snapshot = diagnostic.snapshot().unwrap();
        assert_eq!(
            snapshot.unavailable,
            Some(AdaptiveCpuSliceDiagnosticUnavailable::AccountingUnavailable)
        );
        assert!(snapshot.records[0].accounting_after_settlement.is_none());
        let diagnostic = journal(2);
        diagnostic.bind(scope.authority).unwrap();
        let mut foreign = scope;
        foreign.authority.root = RootGeneration(2);
        diagnostic.record_settled(foreign, &report, None, None);
        let snapshot = diagnostic.snapshot().unwrap();
        assert_eq!(
            snapshot.unavailable,
            Some(AdaptiveCpuSliceDiagnosticUnavailable::ScopeMismatch)
        );
        assert!(snapshot.records.is_empty());
    }
    #[test]
    fn producer_contention_and_poison_are_sticky_without_blocking_original_owner() {
        let diagnostic = journal(1);
        let (scope, report, accounting) = sample();
        diagnostic.bind(scope.authority).unwrap();
        let guard = diagnostic.inner.state.lock().unwrap();
        diagnostic.record_settled(scope, &report, Some(accounting), None);
        drop(guard);
        assert_eq!(
            diagnostic.snapshot().unwrap().unavailable,
            Some(AdaptiveCpuSliceDiagnosticUnavailable::Contended)
        );
        let diagnostic = journal(1);
        let poisoned = diagnostic.clone();
        assert!(
            std::thread::spawn(move || {
                let _guard = poisoned.inner.state.lock().unwrap();
                panic!("only the diagnostic metadata owner is poisoned");
            })
            .join()
            .is_err()
        );
        diagnostic.record_settled(scope, &report, Some(accounting), None);
        assert!(matches!(
            diagnostic.snapshot(),
            Err(AdaptiveCpuSliceDiagnosticError::Poisoned)
        ));
    }
    #[cfg(feature = "pals-adaptive-diagnostics")]
    #[test]
    fn chain_and_settled_slices_share_the_original_record_budget() {
        let diagnostic = journal(1);
        let (scope, report, accounting) = sample();
        diagnostic.bind(scope.authority).unwrap();
        // Authored metadata only, not an actual installation/Begin receipt.
        let event = NnuecChainEvent::NnuecJobBound(NnuecJobBindingObservation {
            frame: scope.frame,
            question: scope.question,
            view: scope.view,
            full_view: Digest([9; 32]),
            job: scope.job,
            first_grant: report.grant,
            installation_sequence: 1,
            installation_succeeded: true,
            binding: scope.job.evaluation_namespace,
            installed_owner: Some(report.grant.owner),
            actual_stack_owner_id: None,
        });
        diagnostic.record_nnuec_chain(event);
        diagnostic.record_settled(scope, &report, Some(accounting), None);
        diagnostic.record_nnuec_chain(event);
        let snapshot = diagnostic.snapshot().unwrap();
        assert_eq!(snapshot.nnuec_chain.len(), 1);
        assert!(snapshot.records.is_empty());
        assert_eq!(snapshot.nnuec_chain_dropped, 1);
        assert_eq!(
            snapshot.unavailable,
            Some(AdaptiveCpuSliceDiagnosticUnavailable::Overflow)
        );
    }
}
