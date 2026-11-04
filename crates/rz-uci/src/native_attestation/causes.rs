//! Public, bounded projections of privately owned errors.
//!
//! This is a view of an error, not a replacement for its original ownership.
//! Text is represented only by a bounded SHA-256 fingerprint; raw native text,
//! private tickets, paths, and formatted error/debug values never enter the wire.

use super::{CompletionReceiptV1, EpochSequenceV1, hex};
use crate::{
    SearchCompletion, ServeFailure,
    engine::{EngineError, RejectedCompletionKind, WorkerDiagnosticReceipt, WorkerFailureSource},
};
use rz_contracts::{
    ContractError, ErrorCode as ContractCode, EvalFailure, RecoveryOutcome, Stage as ContractStage,
};
use rz_eval::{
    contracts::PhysicalFailure,
    error::{
        BackendError, CauseCode, ExternalCause, FailureKind, FailureStage, OutputCause, OutputHead,
    },
    native_runtime_bridge::{NativeDiagnosticKind, NativeDiagnosticReceipt},
};
use rz_search::{contracts::ContractSearchFailure, tree::SearchError};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

const MAX_DEPTH: u8 = 8;
const MAX_NODES: u16 = 128;
const MAX_TEXT_PREFIX_BYTES: usize = 1024;

/// Every projected error/diagnostic node consumes one slot. Omitted links and
/// list tails are explicit, and do not invent an absent original/cleanup cause.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CauseReceiptV1 {
    pub projection: CauseNodeV1,
    pub projection_truncated: bool,
    pub max_depth: u8,
    pub max_nodes: u16,
    pub projected_nodes: u16,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionLimitV1 {
    Depth,
    Nodes,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum CauseLinkV1 {
    Present { cause: Box<CauseNodeV1> },
    Omitted { reason: ProjectionLimitV1 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CauseListV1 {
    pub items: Vec<CauseNodeV1>,
    pub omitted_count: usize,
    pub omission: Option<ProjectionLimitV1>,
}

/// This hashes at most 1024 source bytes, without retaining the text. The
/// truncated flag distinguishes a prefix fingerprint from a whole-text hash.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextFingerprintV1 {
    pub sha256: String,
    pub hashed_bytes: u16,
    pub truncated: bool,
}
impl TextFingerprintV1 {
    fn capture(text: &str) -> Self {
        let count = text.len().min(MAX_TEXT_PREFIX_BYTES);
        Self {
            sha256: hex(&Sha256::digest(&text.as_bytes()[..count]).into()),
            hashed_bytes: count as u16,
            truncated: count != text.len(),
        }
    }
}

macro_rules! wire_enum {
    ($wire:ident, $source:ident, [$($variant:ident),+ $(,)?]) => {
        #[derive(Clone, Copy, Debug, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        pub enum $wire { $($variant),+ }
        impl From<$source> for $wire {
            fn from(value: $source) -> Self {
                match value { $($source::$variant => Self::$variant),+ }
            }
        }
    };
}

wire_enum!(
    ContractCodeV1,
    ContractCode,
    [
        InvalidInput,
        UnsupportedContract,
        BackendUnavailable,
        ResourceExhausted,
        NumericalFailure,
        IdentityMismatch,
        BackendFailure,
        Canceled,
        Expired,
        Stale,
    ]
);
wire_enum!(
    ContractStageV1,
    ContractStage,
    [Contract, Admission, Backend, Output, Backup]
);
wire_enum!(
    BackendKindV1,
    FailureKind,
    [
        InvalidInput,
        UnsupportedModel,
        IdentityMismatch,
        BackendUnavailable,
        ResourceExhausted,
        NumericalFailure,
        BackendFailure,
        Io,
    ]
);
wire_enum!(
    BackendStageV1,
    FailureStage,
    [Asset, Admission, Backend, Output]
);
wire_enum!(
    CauseCodeV1,
    CauseCode,
    [
        RuntimePath,
        RuntimeInitialize,
        RuntimePanic,
        RuntimeLibraryLoad,
        OrtNative,
        SessionBuilder,
        SessionConfiguration,
        ProviderQuery,
        ProviderRegistration,
        ProfilingStart,
        ModelLoad,
        ProfilingFinish,
        InputAllocation,
        TensorCreate,
        OrtRun,
        PolicyExtract,
        WdlExtract,
        OutputValidation,
        ProfileParse,
        MockCallback,
        MockLifecycle,
    ]
);
wire_enum!(OutputHeadV1, OutputHead, [Policy, Wdl]);
wire_enum!(
    RecoveryV1,
    RecoveryOutcome,
    [NotAttempted, Failed, Completed]
);
wire_enum!(
    RejectedOutcomeV1,
    RejectedCompletionKind,
    [Completed, Canceled, Expired, Stale]
);
wire_enum!(
    NativeKindV1,
    NativeDiagnosticKind,
    [DispatchRefused, PhysicalFailure, Quarantined]
);

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum OutputCauseV1 {
    WrongShape {
        head: OutputHeadV1,
        actual: usize,
        expected: usize,
    },
    NonFinite {
        head: OutputHeadV1,
        index: usize,
    },
    WdlOutOfRange {
        index: usize,
    },
    InvalidWdlSum {
        actual_f64_bits: u64,
    },
    EmptyLegalPolicy,
    TooManyLegalIndices,
    InvalidPolicyIndex {
        index: usize,
    },
    DuplicatePolicyIndex {
        index: usize,
    },
    AllocationFailed,
    InvalidPolicyNormalization,
}
impl From<OutputCause> for OutputCauseV1 {
    fn from(value: OutputCause) -> Self {
        match value {
            OutputCause::WrongShape {
                head,
                actual,
                expected,
            } => Self::WrongShape {
                head: head.into(),
                actual,
                expected,
            },
            OutputCause::NonFinite { head, index } => Self::NonFinite {
                head: head.into(),
                index,
            },
            OutputCause::WdlOutOfRange { index } => Self::WdlOutOfRange { index },
            OutputCause::InvalidWdlSum { actual_f64_bits } => {
                Self::InvalidWdlSum { actual_f64_bits }
            }
            OutputCause::EmptyLegalPolicy => Self::EmptyLegalPolicy,
            OutputCause::TooManyLegalIndices => Self::TooManyLegalIndices,
            OutputCause::InvalidPolicyIndex(index) => Self::InvalidPolicyIndex { index },
            OutputCause::DuplicatePolicyIndex(index) => Self::DuplicatePolicyIndex { index },
            OutputCause::AllocationFailed => Self::AllocationFailed,
            OutputCause::InvalidPolicyNormalization => Self::InvalidPolicyNormalization,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalCauseV1 {
    pub code: CauseCodeV1,
    pub prefix_sha256: String,
    pub hashed_bytes: u16,
    pub truncated: bool,
    pub formatting_failed: bool,
    pub output: Option<OutputCauseV1>,
}
impl From<ExternalCause> for ExternalCauseV1 {
    fn from(value: ExternalCause) -> Self {
        Self {
            code: value.code.into(),
            prefix_sha256: hex(&value.prefix_sha256),
            hashed_bytes: value.hashed_bytes,
            truncated: value.truncated,
            formatting_failed: value.formatting_failed,
            output: value.output.map(Into::into),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TreeCodeV1 {
    InvalidConfiguration,
    InvalidStatistics,
    InvalidValue,
    InvalidPolicy,
    DuplicateMove,
    NoLegalEdges,
    Busy,
    Canceled,
    Expired,
    DepthLimit,
    NodeLimit,
    EdgeLimit,
    AllocationFailed,
    CounterOverflow,
    InvalidPolicySelection,
    InconsistentLeaf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum CompletionDispositionV1 {
    Completed { bestmove_present: bool },
    Failed { code: TextFingerprintV1 },
}
impl From<&SearchCompletion> for CompletionDispositionV1 {
    fn from(value: &SearchCompletion) -> Self {
        match value {
            SearchCompletion::Completed { bestmove } => Self::Completed {
                bestmove_present: bestmove.is_some(),
            },
            SearchCompletion::Failed { code, .. } => Self::Failed {
                code: TextFingerprintV1::capture(code),
            },
        }
    }
}

/// Node variants retain cause relationships rather than flattening them into
/// messages. A missing optional cleanup means the source had no such cleanup;
/// an existing cause that exceeds the budget is a CauseLinkV1::Omitted instead.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum CauseNodeV1 {
    Contract {
        code: ContractCodeV1,
        stage: ContractStageV1,
        detail: TextFingerprintV1,
    },
    Backend {
        kind: BackendKindV1,
        stage: BackendStageV1,
        detail: TextFingerprintV1,
        cause: Option<ExternalCauseV1>,
        native_diagnostic_present: bool,
    },
    PhysicalFailure {
        contract: CauseLinkV1,
        backend: Option<CauseLinkV1>,
    },
    NativeDiagnostic {
        ticket_execution: EpochSequenceV1,
        ticket_request: EpochSequenceV1,
        context: CompletionReceiptV1,
        kind: NativeKindV1,
        failure: CauseLinkV1,
    },
    EvaluationFailure {
        context: CompletionReceiptV1,
        recovery: RecoveryV1,
        original: CauseLinkV1,
    },
    SearchBoundary {
        original: CauseLinkV1,
    },
    SearchEvaluation {
        original: CauseLinkV1,
    },
    SearchTree {
        original: CauseLinkV1,
    },
    SearchCleanup {
        primary: CauseLinkV1,
        runtime: Option<CauseLinkV1>,
        tree: Option<CauseLinkV1>,
    },
    Tree {
        code: TreeCodeV1,
        detail: Option<TextFingerprintV1>,
    },
    EngineContract {
        original: CauseLinkV1,
    },
    EngineSession {
        detail: TextFingerprintV1,
    },
    /// Additive closure-failure projection. Historical cause variants retain
    /// their exact fields; the original typed owner failure stays in EngineError.
    EngineSearchDriverClose {
        physical_completion_unknown: bool,
        code: TextFingerprintV1,
        detail: TextFingerprintV1,
    },
    EngineTransport {
        failure: CauseLinkV1,
        cleanup: CauseListV1,
        diagnostics: CauseListV1,
    },
    /// The code is formatted exclusively from std::io::ErrorKind, never from
    /// io::Error's Display, source, or custom error payload.
    TransportIo {
        error_kind: String,
    },
    TransportHandler {
        original: CauseLinkV1,
    },
    PresentationDiagnostic {
        code: TextFingerprintV1,
    },
    EngineWorkerPanic,
    EngineDrainTimeout,
    EngineUndeliveredDiagnostics {
        diagnostics: CauseListV1,
    },
    DiagnosticOccurrence {
        occurrences: u64,
        original: CauseLinkV1,
    },
    WorkerBoundary {
        original: CauseLinkV1,
    },
    RejectedFailure {
        rejection: CauseLinkV1,
        original: CauseLinkV1,
    },
    RejectedCompletion {
        rejection: CauseLinkV1,
        outcome: RejectedOutcomeV1,
        context: CompletionReceiptV1,
    },
    EngineUndeliveredWorkerFailure {
        completion: CompletionDispositionV1,
        source: CauseLinkV1,
    },
    WorkerFactory {
        original: CauseLinkV1,
    },
    WorkerSearchConstructor {
        original: CauseLinkV1,
    },
    WorkerAuthority {
        original: CauseLinkV1,
    },
    WorkerSearch {
        original: CauseLinkV1,
    },
    WorkerDiagnosticRetention {
        error: CauseLinkV1,
        original: CauseLinkV1,
    },
    EngineCleanup {
        primary: CauseLinkV1,
        cleanup: CauseListV1,
    },
}

type Projection = Result<CauseNodeV1, ProjectionLimitV1>;

#[derive(Default)]
struct Budget {
    nodes: u16,
    truncated: bool,
}
impl Budget {
    fn enter(&mut self, depth: u8) -> Result<(), ProjectionLimitV1> {
        let reason = if depth > MAX_DEPTH {
            Some(ProjectionLimitV1::Depth)
        } else if self.nodes >= MAX_NODES {
            Some(ProjectionLimitV1::Nodes)
        } else {
            None
        };
        if let Some(reason) = reason {
            self.truncated = true;
            return Err(reason);
        }
        self.nodes += 1;
        Ok(())
    }
    fn link(&mut self, project: impl FnOnce(&mut Self) -> Projection) -> CauseLinkV1 {
        match project(self) {
            Ok(cause) => CauseLinkV1::Present {
                cause: Box::new(cause),
            },
            Err(reason) => CauseLinkV1::Omitted { reason },
        }
    }
    fn list<T>(
        &mut self,
        source: &[T],
        project: impl Fn(&mut Self, &T) -> Projection,
    ) -> CauseListV1 {
        let mut items = Vec::new();
        for (index, value) in source.iter().enumerate() {
            match project(self, value) {
                Ok(value) => items.push(value),
                Err(reason) => {
                    return CauseListV1 {
                        items,
                        omitted_count: source.len() - index,
                        omission: Some(reason),
                    };
                }
            }
        }
        CauseListV1 {
            items,
            omitted_count: 0,
            omission: None,
        }
    }
    fn contract(&mut self, error: &ContractError, depth: u8) -> Projection {
        self.enter(depth)?;
        Ok(CauseNodeV1::Contract {
            code: error.code.into(),
            stage: error.stage.into(),
            detail: TextFingerprintV1::capture(error.detail),
        })
    }
    fn backend(&mut self, error: &BackendError, depth: u8) -> Projection {
        self.enter(depth)?;
        Ok(CauseNodeV1::Backend {
            kind: error.kind.into(),
            stage: error.stage.into(),
            detail: TextFingerprintV1::capture(error.detail),
            cause: error.cause.map(Into::into),
            native_diagnostic_present: error.native.is_some(),
        })
    }
    fn physical(&mut self, error: &PhysicalFailure, depth: u8) -> Projection {
        self.enter(depth)?;
        Ok(CauseNodeV1::PhysicalFailure {
            contract: self.link(|budget| budget.contract(&error.contract, depth + 1)),
            backend: error
                .backend
                .as_ref()
                .map(|error| self.link(|budget| budget.backend(error, depth + 1))),
        })
    }
    fn native(&mut self, receipt: &NativeDiagnosticReceipt, depth: u8) -> Projection {
        self.enter(depth)?;
        Ok(CauseNodeV1::NativeDiagnostic {
            ticket_execution: EpochSequenceV1 {
                epoch: receipt.ticket.0.epoch.0,
                sequence: receipt.ticket.0.sequence,
            },
            ticket_request: EpochSequenceV1 {
                epoch: receipt.ticket.1.epoch.0,
                sequence: receipt.ticket.1.sequence,
            },
            context: CompletionReceiptV1::from_context(receipt.context),
            kind: receipt.kind.into(),
            failure: self.link(|budget| budget.physical(&receipt.failure, depth + 1)),
        })
    }
    fn evaluation(&mut self, error: &EvalFailure, depth: u8) -> Projection {
        self.enter(depth)?;
        Ok(CauseNodeV1::EvaluationFailure {
            context: CompletionReceiptV1::from_context(error.context),
            recovery: error.recovery.into(),
            original: self.link(|budget| budget.contract(&error.error, depth + 1)),
        })
    }
    fn tree(&mut self, error: &SearchError, depth: u8) -> Projection {
        self.enter(depth)?;
        let (code, detail) = match error {
            SearchError::InvalidConfiguration(detail) => (
                TreeCodeV1::InvalidConfiguration,
                Some(TextFingerprintV1::capture(detail)),
            ),
            SearchError::InvalidStatistics => (TreeCodeV1::InvalidStatistics, None),
            SearchError::InvalidValue => (TreeCodeV1::InvalidValue, None),
            SearchError::InvalidPolicy => (TreeCodeV1::InvalidPolicy, None),
            SearchError::DuplicateMove => (TreeCodeV1::DuplicateMove, None),
            SearchError::NoLegalEdges => (TreeCodeV1::NoLegalEdges, None),
            SearchError::Busy => (TreeCodeV1::Busy, None),
            SearchError::Canceled => (TreeCodeV1::Canceled, None),
            SearchError::Expired => (TreeCodeV1::Expired, None),
            SearchError::DepthLimit => (TreeCodeV1::DepthLimit, None),
            SearchError::NodeLimit => (TreeCodeV1::NodeLimit, None),
            SearchError::EdgeLimit => (TreeCodeV1::EdgeLimit, None),
            SearchError::AllocationFailed => (TreeCodeV1::AllocationFailed, None),
            SearchError::CounterOverflow => (TreeCodeV1::CounterOverflow, None),
            SearchError::InvalidPolicySelection => (TreeCodeV1::InvalidPolicySelection, None),
            SearchError::InconsistentLeaf => (TreeCodeV1::InconsistentLeaf, None),
        };
        Ok(CauseNodeV1::Tree { code, detail })
    }
    fn search(&mut self, error: &ContractSearchFailure, depth: u8) -> Projection {
        self.enter(depth)?;
        Ok(match error {
            ContractSearchFailure::Boundary(error) => CauseNodeV1::SearchBoundary {
                original: self.link(|budget| budget.contract(error, depth + 1)),
            },
            ContractSearchFailure::Evaluation(error) => CauseNodeV1::SearchEvaluation {
                original: self.link(|budget| budget.evaluation(error, depth + 1)),
            },
            ContractSearchFailure::Tree(error) => CauseNodeV1::SearchTree {
                original: self.link(|budget| budget.tree(error, depth + 1)),
            },
            ContractSearchFailure::Cleanup {
                primary,
                runtime,
                tree,
            } => CauseNodeV1::SearchCleanup {
                primary: self.link(|budget| budget.search(primary, depth + 1)),
                runtime: runtime
                    .as_ref()
                    .map(|error| self.link(|budget| budget.contract(error, depth + 1))),
                tree: tree
                    .as_ref()
                    .map(|error| self.link(|budget| budget.tree(error, depth + 1))),
            },
        })
    }
    fn diagnostic(&mut self, receipt: &WorkerDiagnosticReceipt, depth: u8) -> Projection {
        self.enter(depth)?;
        Ok(match receipt {
            // V1 belongs to the unchanged LC0 evaluation path. Search-driver
            // accounting is retained by its owner and has its own receipt;
            // projecting it as NN/evaluation work would fabricate evidence.
            WorkerDiagnosticReceipt::DriverWork { .. } => CauseNodeV1::WorkerBoundary {
                original: self.link(|budget| {
                    budget.contract(
                        &ContractError::new(
                            ContractCode::UnsupportedContract,
                            ContractStage::Output,
                            "search-driver work cannot be represented by LC0 native receipt V1",
                        ),
                        depth + 1,
                    )
                }),
            },
            WorkerDiagnosticReceipt::Boundary(error) => CauseNodeV1::WorkerBoundary {
                original: self.link(|budget| budget.contract(error, depth + 1)),
            },
            WorkerDiagnosticReceipt::RejectedFailure { rejection, failure } => {
                CauseNodeV1::RejectedFailure {
                    rejection: self.link(|budget| budget.contract(rejection, depth + 1)),
                    original: self.link(|budget| budget.evaluation(failure, depth + 1)),
                }
            }
            WorkerDiagnosticReceipt::RejectedCompletion {
                rejection,
                kind,
                context,
            } => CauseNodeV1::RejectedCompletion {
                rejection: self.link(|budget| budget.contract(rejection, depth + 1)),
                outcome: (*kind).into(),
                context: CompletionReceiptV1::from_context(**context),
            },
        })
    }
    fn worker_source(&mut self, source: &WorkerFailureSource, depth: u8) -> Projection {
        self.enter(depth)?;
        Ok(match source {
            WorkerFailureSource::Driver(_) => CauseNodeV1::WorkerBoundary {
                original: self.link(|budget| {
                    budget.contract(
                        &ContractError::new(
                            ContractCode::UnsupportedContract,
                            ContractStage::Output,
                            "search-driver failure retains a separate typed owner receipt",
                        ),
                        depth + 1,
                    )
                }),
            },
            WorkerFailureSource::RejectedDriverReport { error, .. } => {
                CauseNodeV1::WorkerBoundary {
                    // The original report and work remain in the typed worker
                    // owner; this legacy graph cannot relabel them as V5 work.
                    original: self.link(|budget| budget.contract(error, depth + 1)),
                }
            }
            WorkerFailureSource::Factory(error) => CauseNodeV1::WorkerFactory {
                original: self.link(|budget| budget.contract(error, depth + 1)),
            },
            WorkerFailureSource::SearchConstructor(error) => CauseNodeV1::WorkerSearchConstructor {
                original: self.link(|budget| budget.contract(error, depth + 1)),
            },
            WorkerFailureSource::Authority(error) => CauseNodeV1::WorkerAuthority {
                original: self.link(|budget| budget.contract(error, depth + 1)),
            },
            WorkerFailureSource::Search(error) => CauseNodeV1::WorkerSearch {
                original: self.link(|budget| budget.search(error, depth + 1)),
            },
            WorkerFailureSource::DiagnosticRetention { error, receipt } => {
                CauseNodeV1::WorkerDiagnosticRetention {
                    error: self.link(|budget| budget.contract(error, depth + 1)),
                    original: self.link(|budget| budget.diagnostic(receipt, depth + 1)),
                }
            }
        })
    }
    fn engine(&mut self, error: &EngineError, depth: u8) -> Projection {
        self.enter(depth)?;
        Ok(match error {
            EngineError::Contract(error) => CauseNodeV1::EngineContract {
                original: self.link(|budget| budget.contract(error, depth + 1)),
            },
            EngineError::SearchDriverClose(error) => CauseNodeV1::EngineSearchDriverClose {
                physical_completion_unknown: error.physical_completion
                    == crate::search_driver::DriverPhysicalCompletion::Unknown,
                code: TextFingerprintV1::capture(error.code),
                detail: TextFingerprintV1::capture(&error.detail),
            },
            EngineError::Session(error) => CauseNodeV1::EngineSession {
                detail: TextFingerprintV1::capture(&error.0),
            },
            EngineError::Transport(error) => CauseNodeV1::EngineTransport {
                failure: self.link(|budget| {
                    budget.enter(depth + 1)?;
                    Ok(match &error.failure {
                        ServeFailure::Io(error) => CauseNodeV1::TransportIo {
                            error_kind: format!("{:?}", error.kind()),
                        },
                        ServeFailure::Handler(error) => CauseNodeV1::TransportHandler {
                            original: budget.link(|budget| budget.engine(error, depth + 2)),
                        },
                    })
                }),
                cleanup: self.list(&error.cleanup_failures, |budget, error| {
                    budget.engine(error, depth + 1)
                }),
                diagnostics: self.list(&error.diagnostics, |budget, diagnostic| {
                    budget.enter(depth + 1)?;
                    Ok(CauseNodeV1::PresentationDiagnostic {
                        code: TextFingerprintV1::capture(diagnostic.code),
                    })
                }),
            },
            EngineError::WorkerPanic => CauseNodeV1::EngineWorkerPanic,
            EngineError::DrainTimeout => CauseNodeV1::EngineDrainTimeout,
            EngineError::UndeliveredDiagnostics(receipts) => {
                CauseNodeV1::EngineUndeliveredDiagnostics {
                    diagnostics: self.list(receipts, |budget, (receipt, occurrences)| {
                        budget.enter(depth + 1)?;
                        Ok(CauseNodeV1::DiagnosticOccurrence {
                            occurrences: *occurrences,
                            original: budget.link(|budget| budget.diagnostic(receipt, depth + 2)),
                        })
                    }),
                }
            }
            EngineError::UndeliveredWorkerFailure(failure) => {
                CauseNodeV1::EngineUndeliveredWorkerFailure {
                    completion: (&failure.completion).into(),
                    source: self.link(|budget| budget.worker_source(&failure.source, depth + 1)),
                }
            }
            EngineError::Cleanup { primary, cleanup } => CauseNodeV1::EngineCleanup {
                primary: self.link(|budget| budget.engine(primary, depth + 1)),
                cleanup: self.list(cleanup, |budget, error| budget.engine(error, depth + 1)),
            },
        })
    }
}

fn project(run: impl FnOnce(&mut Budget) -> Projection) -> CauseReceiptV1 {
    let mut budget = Budget::default();
    // The root consumes one of the nonzero, compile-time node/depth budgets.
    let projection = run(&mut budget).expect("root projection is inside its fixed budget");
    CauseReceiptV1 {
        projection,
        projection_truncated: budget.truncated,
        max_depth: MAX_DEPTH,
        max_nodes: MAX_NODES,
        projected_nodes: budget.nodes,
    }
}

pub(crate) fn contract(error: &ContractError) -> CauseReceiptV1 {
    project(|budget| budget.contract(error, 1))
}
#[cfg(any(test, feature = "onnx-cuda"))]
pub(crate) fn backend(error: &BackendError) -> CauseReceiptV1 {
    project(|budget| budget.backend(error, 1))
}
pub(crate) fn engine(error: &EngineError) -> CauseReceiptV1 {
    project(|budget| budget.engine(error, 1))
}
pub(crate) fn native(receipt: &NativeDiagnosticReceipt) -> CauseReceiptV1 {
    project(|budget| budget.native(receipt, 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Diagnostic, ServeError, SessionError};
    use rz_contracts::*;

    const PRIVATE_TEXT: &str = "C:\\private-person\\native-runtime\\credentials-never-publish";

    #[test]
    fn driver_close_keeps_unknown_completion_and_redacts_dynamic_detail() {
        let original = EngineError::SearchDriverClose(crate::search_driver::SearchSessionFailure {
            physical_completion: crate::search_driver::DriverPhysicalCompletion::Unknown,
            code: "PhysicalCompletionUnknown",
            detail: PRIVATE_TEXT.into(),
        });
        let projected = engine(&original);
        assert!(matches!(
            &projected.projection,
            CauseNodeV1::EngineSearchDriverClose {
                physical_completion_unknown: true,
                ..
            }
        ));
        let bytes = serde_json::to_vec(&projected).unwrap();
        assert!(!String::from_utf8_lossy(&bytes).contains("private-person"));
        let decoded: CauseReceiptV1 = serde_json::from_slice(&bytes).unwrap();
        assert!(matches!(
            decoded.projection,
            CauseNodeV1::EngineSearchDriverClose {
                physical_completion_unknown: true,
                ..
            }
        ));
    }

    fn context() -> CompletionContext {
        let epoch = ProcessEpoch(41);
        CompletionContext {
            request: EvalContext {
                revision: CONTRACT_REVISION,
                request: RequestId::new(epoch, 19),
                selection: SelectionId::new(epoch, 23),
                game: GameGeneration(7),
                root: RootGeneration(11),
                state: StateIdentity {
                    owner: OwnerId(3),
                    revision: StateRevision(5),
                    semantic: Digest([1; 32]),
                },
                legal_order: LegalOrderIdentity(Digest([2; 32])),
                input: EvalInputKey(Digest([3; 32])),
                model: ModelHandle {
                    owner: OwnerId(13),
                    slot: 17,
                    generation: SlotGeneration(29),
                    manifest: Digest([4; 32]),
                },
                encoding: EncodingHandle {
                    owner: OwnerId(31),
                    slot: 37,
                    generation: SlotGeneration(43),
                    manifest: Digest([5; 32]),
                },
                precision: PrecisionProfile::Fp32,
                compute: ComputeBudget {
                    min_steps: 1,
                    max_steps: 1,
                    require_full: true,
                },
                backend: Digest([6; 32]),
            },
            execution: Some(ExecutionId::new(epoch, 47)),
        }
    }
    fn failure() -> EvalFailure {
        EvalFailure {
            context: context(),
            error: ContractError::new(ErrorCode::BackendFailure, Stage::Backend, PRIVATE_TEXT),
            recovery: RecoveryOutcome::Failed,
        }
    }
    fn child(link: &CauseLinkV1) -> &CauseNodeV1 {
        match link {
            CauseLinkV1::Present { cause } => cause,
            CauseLinkV1::Omitted { .. } => panic!("expected source within budget"),
        }
    }

    #[test]
    fn compound_rejected_failure_preserves_original_context_recovery_and_cleanup() {
        let error = EngineError::Cleanup {
            primary: Box::new(EngineError::UndeliveredDiagnostics(vec![(
                WorkerDiagnosticReceipt::RejectedFailure {
                    rejection: ContractError::new(ErrorCode::Stale, Stage::Output, "foreign root"),
                    failure: Box::new(failure()),
                },
                3,
            )])),
            cleanup: vec![EngineError::DrainTimeout],
        };
        let receipt = engine(&error);
        assert!(!receipt.projection_truncated);
        let CauseNodeV1::EngineCleanup { primary, cleanup } = &receipt.projection else {
            panic!("compound cause")
        };
        assert!(matches!(
            cleanup.items.as_slice(),
            [CauseNodeV1::EngineDrainTimeout]
        ));
        let CauseNodeV1::EngineUndeliveredDiagnostics { diagnostics } = child(primary) else {
            panic!("retained source")
        };
        let CauseNodeV1::DiagnosticOccurrence {
            occurrences,
            original,
        } = &diagnostics.items[0]
        else {
            panic!("occurrence receipt")
        };
        assert_eq!(*occurrences, 3);
        let CauseNodeV1::RejectedFailure {
            rejection,
            original,
        } = child(original)
        else {
            panic!("rejected original")
        };
        assert!(matches!(
            child(rejection),
            CauseNodeV1::Contract {
                code: ContractCodeV1::Stale,
                ..
            }
        ));
        let CauseNodeV1::EvaluationFailure {
            context,
            recovery,
            original,
        } = child(original)
        else {
            panic!("evaluation provenance")
        };
        assert!(matches!(recovery, RecoveryV1::Failed));
        assert_eq!(context.request.epoch, 41);
        assert_eq!(context.request.sequence, 19);
        assert_eq!(context.selection.sequence, 23);
        assert_eq!((context.game, context.root), (7, 11));
        assert_eq!(
            context
                .execution
                .as_ref()
                .map(|value| (value.epoch, value.sequence)),
            Some((41, 47))
        );
        assert!(matches!(
            child(original),
            CauseNodeV1::Contract {
                code: ContractCodeV1::BackendFailure,
                stage: ContractStageV1::Backend,
                ..
            }
        ));
        let json = serde_json::to_string(&receipt).unwrap();
        assert!(!json.contains(PRIVATE_TEXT));
        let restored: CauseReceiptV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.projected_nodes, receipt.projected_nodes);
    }

    #[test]
    fn search_cleanup_keeps_primary_runtime_and_tree_separate() {
        let error = ContractSearchFailure::Cleanup {
            primary: Box::new(ContractSearchFailure::Evaluation(Box::new(failure()))),
            runtime: Some(ContractError::new(
                ErrorCode::Expired,
                Stage::Backend,
                "drain deadline",
            )),
            tree: Some(SearchError::CounterOverflow),
        };
        let receipt = project(|budget| budget.search(&error, 1));
        let CauseNodeV1::SearchCleanup {
            primary,
            runtime,
            tree,
        } = &receipt.projection
        else {
            panic!("search cleanup")
        };
        assert!(matches!(
            child(primary),
            CauseNodeV1::SearchEvaluation { .. }
        ));
        assert!(matches!(
            child(runtime.as_ref().unwrap()),
            CauseNodeV1::Contract {
                code: ContractCodeV1::Expired,
                ..
            }
        ));
        assert!(matches!(
            child(tree.as_ref().unwrap()),
            CauseNodeV1::Tree {
                code: TreeCodeV1::CounterOverflow,
                ..
            }
        ));
        assert!(!receipt.projection_truncated);
    }

    #[test]
    fn native_and_transport_projection_never_emit_raw_codes_messages_or_paths() {
        let source = BackendError::new(
            FailureKind::BackendFailure,
            FailureStage::Backend,
            PRIVATE_TEXT,
        )
        .with_external_cause(CauseCode::OrtRun, &PRIVATE_TEXT)
        .with_diagnostic(PRIVATE_TEXT, PRIVATE_TEXT);
        let native_receipt = NativeDiagnosticReceipt {
            ticket: (
                ExecutionId::new(ProcessEpoch(41), 53),
                RequestId::new(ProcessEpoch(41), 19),
            ),
            context: context(),
            kind: NativeDiagnosticKind::PhysicalFailure,
            failure: PhysicalFailure {
                contract: failure().error,
                backend: Some(source.clone()),
            },
        };
        let native_projection = native(&native_receipt);
        let json = serde_json::to_value(&native_projection).unwrap();
        assert_eq!(json["projection"]["ticket_execution"]["sequence"], 53);
        assert_eq!(json["projection"]["context"]["execution"]["sequence"], 47);
        let backend_projection = backend(&source);
        let encoded = serde_json::to_string(&backend_projection).unwrap();
        assert!(!encoded.contains("private-person"));
        assert!(!encoded.contains("credentials-never-publish"));
        assert!(encoded.contains("ort_run"));
        assert!(encoded.contains("native_diagnostic_present"));
        let transport = EngineError::Transport(Box::new(ServeError {
            failure: ServeFailure::Io(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                PRIVATE_TEXT,
            )),
            cleanup_failures: vec![EngineError::Session(SessionError(PRIVATE_TEXT.into()))],
            diagnostics: vec![Diagnostic {
                code: PRIVATE_TEXT,
                message: PRIVATE_TEXT.into(),
            }],
        }));
        let value = serde_json::to_value(engine(&transport)).unwrap();
        assert_eq!(
            value["projection"]["failure"]["cause"]["error_kind"],
            "BrokenPipe"
        );
        let text = serde_json::to_string(&value).unwrap();
        assert!(!text.contains("private-person"));
        assert!(!text.contains("credentials-never-publish"));
        assert!(!text.contains("message"));
    }

    #[test]
    fn depth_and_node_limits_mark_omissions_without_fabricating_absence() {
        let mut deep = EngineError::DrainTimeout;
        for _ in 0..20 {
            deep = EngineError::Cleanup {
                primary: Box::new(deep),
                cleanup: vec![EngineError::WorkerPanic],
            };
        }
        let deep_receipt = engine(&deep);
        assert!(deep_receipt.projection_truncated);
        let deep_json = serde_json::to_string(&deep_receipt).unwrap();
        assert!(deep_json.contains("\"state\":\"omitted\",\"reason\":\"depth\""));
        assert!(deep_receipt.projected_nodes <= MAX_NODES);
        let wide = EngineError::Cleanup {
            primary: Box::new(EngineError::DrainTimeout),
            cleanup: (0..200).map(|_| EngineError::WorkerPanic).collect(),
        };
        let receipt = engine(&wide);
        assert!(receipt.projection_truncated);
        assert_eq!(receipt.projected_nodes, MAX_NODES);
        let CauseNodeV1::EngineCleanup { cleanup, .. } = receipt.projection else {
            panic!("wide cleanup")
        };
        assert_eq!(cleanup.items.len() + cleanup.omitted_count, 200);
        assert!(matches!(cleanup.omission, Some(ProjectionLimitV1::Nodes)));
        assert!(!cleanup.items.is_empty());
    }

    #[test]
    fn detail_fingerprints_are_bounded_and_unknown_fields_are_rejected() {
        let fingerprint = TextFingerprintV1::capture(&"x".repeat(MAX_TEXT_PREFIX_BYTES + 1));
        assert_eq!(fingerprint.hashed_bytes as usize, MAX_TEXT_PREFIX_BYTES);
        assert!(fingerprint.truncated);
        assert_eq!(
            fingerprint.sha256,
            hex(&Sha256::digest(vec![b'x'; MAX_TEXT_PREFIX_BYTES]).into())
        );
        let error = ContractError::new(
            ErrorCode::InvalidInput,
            Stage::Contract,
            "short fixed detail",
        );
        let mut value = serde_json::to_value(contract(&error)).unwrap();
        value["unrecognized"] = true.into();
        assert!(serde_json::from_value::<CauseReceiptV1>(value).is_err());
    }
}
