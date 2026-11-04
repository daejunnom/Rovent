//! Copies actual search and execution events for the V4 consumer. These facts
//! are independent of selected policies and own no native tensors or leases.

use super::super::store::{ArchiveOwnerSnapshot, StoreError};
use super::*;
use crate::cpu::CpuPausedStackSnapshot;
use std::collections::VecDeque;

pub const PALS_FOLLOWUP_EXECUTION_VERSION: &str = "rz-pals-followup-lifecycle-v4/1";
pub const PALS_FOLLOWUP_EXECUTION_TRACE_MAX: usize = 64;
const VALUE_BINDINGS_MAX: usize = 128;

/// Fixed-size copy of the actual logical context, including its ordered-input
/// digests. Store handles remain Store handles; they are not Runtime IDs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoleExecutionContext {
    pub game_generation: u64,
    pub search_generation: u64,
    pub situation: SituationId,
    pub state: StateId,
    pub focus: LineId,
    pub purpose: RoleQueryPurpose,
    pub focus_sha256: [u8; 32],
    pub prefix_sha256: [u8; 32],
    pub proposal_sha256: [u8; 32],
    pub refutation_sha256: Option<[u8; 32]>,
    pub divergence_sha256: [u8; 32],
    pub public_revision: u64,
    pub situation_revision: u64,
}
impl From<&RoleLogicalContext> for RoleExecutionContext {
    fn from(value: &RoleLogicalContext) -> Self {
        Self {
            game_generation: value.game_generation,
            search_generation: value.search_generation,
            situation: value.situation,
            state: value.state,
            focus: value.focus,
            purpose: value.purpose,
            focus_sha256: value.focus_sha256,
            prefix_sha256: value.prefix_sha256,
            proposal_sha256: value.proposal_sha256,
            refutation_sha256: value.refutation_sha256,
            divergence_sha256: value.divergence_sha256,
            public_revision: value.public_revision,
            situation_revision: value.situation_revision,
        }
    }
}

/// A provider's actual submitted Lease and physical completion observation.
/// This getter value precedes logical acceptance. Unknown backend input counts
/// are never replaced with a successful callback count or a policy declaration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoleValueExecutionEvidence {
    pub request: rz_contracts::RequestId,
    pub execution: rz_contracts::ExecutionId,
    pub prepared_state: rz_contracts::StateIdentity,
    pub prepared_perspective: rz_contracts::Color,
    pub input_sha256: [u8; 32],
    pub model_epoch: [u8; 32],
    pub context: RoleExecutionContext,
    pub fresh: Option<bool>,
    pub physically_completed: Option<bool>,
    pub completed_nn_inputs: Option<u64>,
    pub complete: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PalsFollowupExecutionScope {
    pub game_generation: u64,
    pub search_attempt_sequence: u64,
    pub store_root_generation: u64,
    pub root: SituationId,
    pub root_state: StateId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PalsFollowupObservationIssue {
    SearchSequenceExhausted,
    TraceOverflow,
    Allocation,
    MissingRepairContext,
    MissingValueExecution,
    ValueExecutionMismatch,
    IncompleteValueExecution,
    DuplicateValueExecution,
    TraceAuthorityMismatch,
    IncompleteRepairTrace,
}

#[derive(Clone, Debug)]
pub struct PalsFrozenValueExecution {
    pub identity: ModelValueIdentity,
    pub input_sha256: [u8; 32],
    pub perspective: Color,
    pub wdl_bits: [u32; 3],
    pub context_revision: u64,
    pub observation: ObservationId,
    /// Present only after the Engine's checked acceptance and Node publication.
    pub logical_accepted: bool,
    pub execution: Option<RoleValueExecutionEvidence>,
}

#[derive(Clone, Debug)]
pub enum PalsExecutionEndpointValue {
    Unknown,
    FrozenWdl(Box<PalsFrozenValueExecution>),
    RulesTerminal {
        reason: TerminalReason,
        value: i32,
        perspective: Color,
    },
}

/// Exact immutable Rules/history is retained by a bounded snapshot, so the
/// Native owner can export its real Rules digest without an invented OwnerId.
#[derive(Clone, Debug)]
pub struct PalsExecutionEndpointSnapshot {
    pub state: StateId,
    pub situation: SituationId,
    pub snapshot: PositionSnapshot,
    pub value: PalsExecutionEndpointValue,
}

#[derive(Clone, Debug)]
pub struct PalsFrozenComparisonExecution {
    pub before: PalsExecutionEndpointSnapshot,
    pub after: Option<PalsExecutionEndpointSnapshot>,
    pub comparison_perspective: Color,
    pub context_revision: Option<u64>,
    pub comparison_attempted: bool,
    pub comparable: bool,
    /// Request count and physical backend inputs are different units. Two
    /// Fresh requests can complete two through four physical NN inputs.
    pub fresh_requests_completed: Option<u32>,
    pub completed_nn_inputs: Option<u64>,
    pub publication: Option<ObservationId>,
    pub original_error: Option<PalsError>,
    pub complete: bool,
}

#[derive(Clone, Debug)]
pub struct PalsRepairRecheckExecution {
    pub reply_context: RoleExecutionContext,
    pub prepared_accepted: bool,
    pub reply_call_attempted: bool,
    pub reply_accepted: bool,
    pub selected_response: Option<BoardMove>,
    pub counterline: Vec<BoardMove>,
    pub counterline_completed: bool,
    pub disposition: RecheckDisposition,
    pub comparison: PalsFrozenComparisonExecution,
    pub original_error: Option<PalsError>,
}

/// One actual accepted Repair record. Its parent/supersedes values are copied
/// from the record, including prior-go predecessors and a first None value.
#[derive(Clone, Debug)]
pub struct PalsRepairExecutionTrace {
    pub scope: PalsFollowupExecutionScope,
    pub first_move: BoardMove,
    pub lineage_root_record: u64,
    pub parent_revision: Option<u64>,
    pub supersedes_revision: Option<u64>,
    pub record_revision: u64,
    pub repair_ordinal: u32,
    pub repaired_line: LineId,
    pub repaired: Vec<BoardMove>,
    pub accepted_question: Option<RoleExecutionContext>,
    pub iterative_repair_comparison: Option<PalsFrozenComparisonExecution>,
    pub recheck: Option<PalsRepairRecheckExecution>,
}

#[derive(Clone, Debug)]
pub struct PalsFollowupExecutionSnapshot {
    pub version: &'static str,
    pub game_generation: Option<u64>,
    pub search_attempt_sequence: Option<u64>,
    pub requested_position: PositionSnapshot,
    pub scope: Option<PalsFollowupExecutionScope>,
    /// Selected fields are declarations, separate from the event observations.
    pub selected_recheck: PostRepairRecheckPolicy,
    pub selected_resolver: ResolverPolicy,
    pub model_identity: Option<ModelValueIdentity>,
    pub pending_questions_peak: Option<u32>,
    pub repair_admissions_peak: Option<u32>,
    pub repair_traces: Vec<PalsRepairExecutionTrace>,
    pub repair_trace_total: u64,
    pub overflow: bool,
    pub complete: bool,
    pub issues: Vec<PalsFollowupObservationIssue>,
    pub closure: Option<RoleSearchClosure>,
    pub original_error: Option<PalsError>,
    pub archive_start: Result<Option<ArchiveOwnerSnapshot>, StoreError>,
    pub archive_end: Result<Option<ArchiveOwnerSnapshot>, StoreError>,
    pub previous_closed_archive_owner: Option<ArchiveOwnerSnapshot>,
    pub paused_stack_start: Option<CpuPausedStackSnapshot>,
    pub paused_stack_end: Option<CpuPausedStackSnapshot>,
}

#[derive(Clone, Debug)]
pub struct PalsFollowupOwnerSnapshot {
    pub version: &'static str,
    pub game_generation: Option<u64>,
    pub search_attempt_sequence: Option<u64>,
    pub store_root_generation: u64,
    pub root: Option<SituationId>,
    pub archive: Result<Option<ArchiveOwnerSnapshot>, StoreError>,
    pub previous_closed_archive_owner: Option<ArchiveOwnerSnapshot>,
    pub paused_stack: Option<CpuPausedStackSnapshot>,
    pub new_game_error: Option<PalsError>,
}

pub(super) struct FollowupExecutionRecorder {
    sequence: Option<u64>,
    snapshot: Option<PalsFollowupExecutionSnapshot>,
    value_bindings: VecDeque<(ObservationId, RoleValueExecutionEvidence)>,
    accepted_repair_question: Option<RoleExecutionContext>,
}
impl Default for FollowupExecutionRecorder {
    fn default() -> Self {
        Self {
            sequence: Some(0),
            snapshot: None,
            value_bindings: VecDeque::new(),
            accepted_repair_question: None,
        }
    }
}
impl FollowupExecutionRecorder {
    fn issue(&mut self, issue: PalsFollowupObservationIssue) {
        if let Some(snapshot) = &mut self.snapshot {
            snapshot.complete = false;
            if !snapshot.issues.contains(&issue) && snapshot.issues.try_reserve(1).is_ok() {
                snapshot.issues.push(issue);
            }
        }
    }
    fn binding(&self, observation: ObservationId) -> Option<RoleValueExecutionEvidence> {
        self.value_bindings
            .iter()
            .rev()
            .find(|(id, _)| *id == observation)
            .map(|(_, evidence)| *evidence)
    }
}

impl<M: RoleModel> PalsEngine<M> {
    /// Read only after the entered search returns. No I/O, inference, event or
    /// cumulative counter is produced by this query.
    pub fn last_followup_execution_snapshot(&self) -> Option<&PalsFollowupExecutionSnapshot> {
        self.followup_execution.snapshot.as_ref()
    }

    /// Native calls this again after its actual shutdown fences. Idle search is
    /// not release evidence: these are unchanged actual owner getters.
    pub fn followup_owner_snapshot(&self) -> PalsFollowupOwnerSnapshot {
        PalsFollowupOwnerSnapshot {
            version: PALS_FOLLOWUP_EXECUTION_VERSION,
            game_generation: self.semantic.game_generation,
            search_attempt_sequence: self.followup_execution.sequence,
            store_root_generation: self.semantic.stores.generation(),
            root: self.semantic.stores.root(),
            archive: self.engine_archive_owner_snapshot(),
            previous_closed_archive_owner: self.semantic.archive_state.previous_closed_owner,
            paused_stack: self.researcher.cpu.paused_stack_snapshot(),
            new_game_error: self.semantic.new_game_transition_error.clone(),
        }
    }

    pub(super) fn begin_followup_execution(&mut self, position: &Position) {
        self.followup_execution.sequence = self
            .followup_execution
            .sequence
            .and_then(|sequence| sequence.checked_add(1));
        self.followup_execution.value_bindings.clear();
        self.followup_execution.accepted_repair_question = None;
        let archive = self.engine_archive_owner_snapshot();
        let paused = self.researcher.cpu.paused_stack_snapshot();
        self.followup_execution.snapshot = Some(PalsFollowupExecutionSnapshot {
            version: PALS_FOLLOWUP_EXECUTION_VERSION,
            game_generation: self.semantic.game_generation,
            search_attempt_sequence: self.followup_execution.sequence,
            requested_position: position.snapshot(),
            scope: None,
            selected_recheck: self.post_repair_recheck,
            selected_resolver: self.resolver,
            model_identity: self.semantic.model_registered_value.clone(),
            // An entered search really observes the empty queue/admission table.
            pending_questions_peak: Some(0),
            repair_admissions_peak: Some(0),
            repair_traces: Vec::new(),
            repair_trace_total: 0,
            overflow: false,
            complete: true,
            issues: Vec::new(),
            closure: None,
            original_error: None,
            archive_start: archive.clone(),
            archive_end: archive,
            previous_closed_archive_owner: self.semantic.archive_state.previous_closed_owner,
            paused_stack_start: paused,
            paused_stack_end: paused,
        });
        if self.followup_execution.sequence.is_none() {
            self.followup_execution
                .issue(PalsFollowupObservationIssue::SearchSequenceExhausted);
        }
    }

    pub(super) fn observe_followup_root(&mut self, root: usize) {
        let Some((game_generation, search_attempt_sequence)) = self
            .semantic
            .game_generation
            .zip(self.followup_execution.sequence)
        else {
            return;
        };
        if let Some(snapshot) = &mut self.followup_execution.snapshot {
            snapshot.scope = Some(PalsFollowupExecutionScope {
                game_generation,
                search_attempt_sequence,
                store_root_generation: self.semantic.stores.generation(),
                root: self.semantic.nodes[root].situation,
                root_state: self.semantic.nodes[root].state,
            });
        }
    }

    /// Search attempts and owner counters retain their lifetime scopes. Only
    /// per-game facts are discarded after the actual old-owner reset succeeds.
    pub(super) fn clear_followup_game(&mut self) {
        self.followup_execution.snapshot = None;
        self.followup_execution.value_bindings.clear();
        self.followup_execution.accepted_repair_question = None;
    }

    pub(super) fn finish_followup_execution(
        &mut self,
        closure: RoleSearchClosure,
        error: Option<&PalsError>,
        counters: PalsCounters,
    ) {
        let archive = self.engine_archive_owner_snapshot();
        let paused = self.researcher.cpu.paused_stack_snapshot();
        if let Some(snapshot) = &mut self.followup_execution.snapshot {
            snapshot.closure = Some(closure);
            snapshot.original_error = error.cloned();
            snapshot.archive_end = archive;
            snapshot.paused_stack_end = paused;
            snapshot.pending_questions_peak = u32::try_from(counters.repair_queue_peak).ok();
            snapshot.repair_admissions_peak = Some(
                self.semantic
                    .repair_counts
                    .values()
                    .copied()
                    .max()
                    .map_or(0, u32::from),
            );
            if snapshot.pending_questions_peak.is_none() {
                snapshot.complete = false;
            }
        }
        if self
            .followup_execution
            .snapshot
            .as_ref()
            .is_some_and(|snapshot| {
                snapshot.repair_traces.iter().any(|trace| {
                    trace
                        .recheck
                        .as_ref()
                        .is_none_or(|recheck| !recheck.comparison.complete)
                        || trace
                            .iterative_repair_comparison
                            .as_ref()
                            .is_some_and(|comparison| !comparison.complete)
                })
            })
        {
            self.followup_execution
                .issue(PalsFollowupObservationIssue::IncompleteRepairTrace);
        }
    }

    pub(super) fn observe_repair_question(&mut self, context: &RoleLogicalContext) {
        self.followup_execution.accepted_repair_question = Some(context.into());
    }

    pub(super) fn observe_value_execution(
        &mut self,
        observation: ObservationId,
        input_sha256: [u8; 32],
        model_epoch: [u8; 32],
        output_perspective: Color,
        context: &RoleLogicalContext,
        evidence: Option<RoleValueExecutionEvidence>,
    ) {
        let Some(evidence) = evidence else {
            return;
        };
        let perspective = match output_perspective {
            Color::White => rz_contracts::Color::White,
            Color::Black => rz_contracts::Color::Black,
        };
        if evidence.input_sha256 != input_sha256
            || evidence.model_epoch != model_epoch
            || evidence.context != RoleExecutionContext::from(context)
            || evidence.context.purpose != RoleQueryPurpose::ValueFresh
            || evidence.prepared_perspective != perspective
            || evidence.request.epoch != evidence.execution.epoch
        {
            self.followup_execution
                .issue(PalsFollowupObservationIssue::ValueExecutionMismatch);
            return;
        }
        if self.followup_execution.value_bindings.len() == VALUE_BINDINGS_MAX {
            self.followup_execution.value_bindings.pop_front();
        }
        if self
            .followup_execution
            .value_bindings
            .try_reserve(1)
            .is_err()
        {
            self.followup_execution
                .issue(PalsFollowupObservationIssue::Allocation);
            return;
        }
        self.followup_execution
            .value_bindings
            .push_back((observation, evidence));
    }

    pub(super) fn observe_accepted_repair(
        &mut self,
        root: usize,
        first_move: BoardMove,
        repaired_line: LineId,
        record_revision: u64,
    ) {
        if !self.post_repair_recheck.uses_frozen_model_wdl() {
            return;
        }
        let Some(snapshot) = &mut self.followup_execution.snapshot else {
            return;
        };
        if snapshot
            .repair_traces
            .iter()
            .any(|trace| trace.record_revision == record_revision)
        {
            self.followup_execution
                .issue(PalsFollowupObservationIssue::TraceAuthorityMismatch);
            return;
        }
        let Some(total) = snapshot.repair_trace_total.checked_add(1) else {
            snapshot.overflow = true;
            self.followup_execution
                .issue(PalsFollowupObservationIssue::TraceOverflow);
            return;
        };
        snapshot.repair_trace_total = total;
        if snapshot.repair_traces.len() >= PALS_FOLLOWUP_EXECUTION_TRACE_MAX {
            snapshot.overflow = true;
            self.followup_execution
                .issue(PalsFollowupObservationIssue::TraceOverflow);
            return;
        }
        let Some(scope) = snapshot.scope else {
            self.followup_execution
                .issue(PalsFollowupObservationIssue::TraceAuthorityMismatch);
            return;
        };
        let Some(record) = self.semantic.records.iter().find(|record| {
            record.revision == record_revision
                && record.kind == RecordKind::Repair
                && record.origin_state == self.semantic.nodes[root].state
                && record.line.first() == Some(&first_move)
        }) else {
            self.followup_execution
                .issue(PalsFollowupObservationIssue::TraceAuthorityMismatch);
            return;
        };
        let previous = snapshot
            .repair_traces
            .iter()
            .rev()
            .find(|trace| trace.first_move == first_move);
        let lineage_root_record =
            previous.map_or(record_revision, |trace| trace.lineage_root_record);
        let repair_ordinal = previous.map_or(1, |trace| trace.repair_ordinal + 1);
        if snapshot.repair_traces.try_reserve(1).is_err() {
            self.followup_execution
                .issue(PalsFollowupObservationIssue::Allocation);
            return;
        }
        let mut repaired = Vec::new();
        if repaired.try_reserve_exact(record.line.len()).is_err() {
            self.followup_execution
                .issue(PalsFollowupObservationIssue::Allocation);
            return;
        }
        repaired.extend_from_slice(&record.line);
        let question = self.followup_execution.accepted_repair_question;
        snapshot.repair_traces.push(PalsRepairExecutionTrace {
            scope,
            first_move,
            lineage_root_record,
            parent_revision: record.parent_revision,
            supersedes_revision: record.supersedes_revision,
            record_revision,
            repair_ordinal,
            repaired_line,
            repaired,
            accepted_question: question,
            iterative_repair_comparison: None,
            recheck: None,
        });
        if question.is_none() {
            self.followup_execution
                .issue(PalsFollowupObservationIssue::MissingRepairContext);
        }
    }

    fn execution_endpoint(&self, node: usize) -> PalsExecutionEndpointSnapshot {
        let node = &self.semantic.nodes[node];
        let value = if let Some((reason, value)) = node.terminal {
            PalsExecutionEndpointValue::RulesTerminal {
                reason,
                value,
                perspective: node.position.side_to_move(),
            }
        } else {
            node.model_value
                .as_ref()
                .zip(node.model_observation.zip(node.model_value_revision))
                .and_then(|(output, (observation, revision))| {
                    let raw = self.semantic.stores.observations.get(observation).ok()?;
                    if raw.state != node.state
                        || raw.model_value_identity.as_ref() != Some(&output.identity)
                        || raw.model_value_input != Some(output.input_sha256)
                        || !matches!(raw.score, RawScore::ContextWdl { win, draw, loss, perspective, context_revision }
                            if [win.to_bits(), draw.to_bits(), loss.to_bits()] == output.wdl.map(f32::to_bits)
                                && perspective == output.perspective && context_revision == revision)
                    {
                        return None;
                    }
                    Some(PalsExecutionEndpointValue::FrozenWdl(Box::new(
                        PalsFrozenValueExecution {
                            identity: output.identity.clone(),
                            input_sha256: output.input_sha256,
                            perspective: output.perspective,
                            wdl_bits: output.wdl.map(f32::to_bits),
                            context_revision: revision,
                            observation,
                            logical_accepted: true,
                            execution: self.followup_execution.binding(observation),
                        },
                    )))
                })
                .unwrap_or(PalsExecutionEndpointValue::Unknown)
        };
        PalsExecutionEndpointSnapshot {
            state: node.state,
            situation: node.situation,
            snapshot: node.position.snapshot(),
            value,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn observe_frozen_comparison(
        &mut self,
        before: usize,
        after: Option<usize>,
        comparison_perspective: Color,
        comparison_attempted: bool,
        comparable: bool,
        publication: Option<ObservationId>,
        error: Option<&PalsError>,
    ) -> PalsFrozenComparisonExecution {
        let before = self.execution_endpoint(before);
        let after = after.map(|node| self.execution_endpoint(node));
        let (requests, inputs, context_revision, issue) = if comparison_attempted {
            execution_pair(&before, after.as_ref())
        } else {
            (None, None, None, None)
        };
        if let Some(issue) = issue {
            self.followup_execution.issue(issue);
        }
        PalsFrozenComparisonExecution {
            before,
            after,
            comparison_perspective,
            context_revision,
            comparison_attempted,
            comparable,
            fresh_requests_completed: requests,
            completed_nn_inputs: inputs,
            publication,
            original_error: error.cloned(),
            complete: issue.is_none() && requests.is_some() && inputs.is_some(),
        }
    }

    pub(super) fn observe_repair_recheck(
        &mut self,
        record_revision: u64,
        recheck: PalsRepairRecheckExecution,
    ) {
        if let Some(trace) = self
            .followup_execution
            .snapshot
            .as_mut()
            .and_then(|snapshot| {
                snapshot
                    .repair_traces
                    .iter_mut()
                    .find(|trace| trace.record_revision == record_revision)
            })
        {
            if trace.recheck.is_some() {
                self.followup_execution
                    .issue(PalsFollowupObservationIssue::TraceAuthorityMismatch);
            } else {
                trace.recheck = Some(recheck);
            }
        }
    }

    pub(super) fn observe_iterative_repair_comparison(
        &mut self,
        record_revision: u64,
        comparison: PalsFrozenComparisonExecution,
    ) {
        if let Some(trace) = self
            .followup_execution
            .snapshot
            .as_mut()
            .and_then(|snapshot| {
                snapshot
                    .repair_traces
                    .iter_mut()
                    .find(|trace| trace.record_revision == record_revision)
            })
        {
            trace.iterative_repair_comparison = Some(comparison);
        }
    }
}

/// Counts only actual physically completed bindings. A repeated input hash is
/// allowed; repeated request or physical execution IDs cannot certify two calls.
fn execution_pair(
    before: &PalsExecutionEndpointSnapshot,
    after: Option<&PalsExecutionEndpointSnapshot>,
) -> (
    Option<u32>,
    Option<u64>,
    Option<u64>,
    Option<PalsFollowupObservationIssue>,
) {
    let Some(after) = after else {
        return (None, None, None, None);
    };
    let mut requests = 0u32;
    let mut inputs = 0u64;
    let mut context_revision = None;
    let mut first = None;
    for endpoint in [before, after] {
        match &endpoint.value {
            PalsExecutionEndpointValue::RulesTerminal { .. } => {}
            PalsExecutionEndpointValue::Unknown => return (None, None, None, None),
            PalsExecutionEndpointValue::FrozenWdl(value) => {
                let Some(binding) = value.execution else {
                    return (
                        None,
                        None,
                        None,
                        Some(PalsFollowupObservationIssue::MissingValueExecution),
                    );
                };
                if !value.logical_accepted
                    || !binding.complete
                    || binding.fresh != Some(true)
                    || binding.physically_completed != Some(true)
                    || binding.completed_nn_inputs.is_none()
                {
                    return (
                        None,
                        None,
                        None,
                        Some(PalsFollowupObservationIssue::IncompleteValueExecution),
                    );
                }
                if first.is_some_and(|(request, execution)| {
                    request == binding.request || execution == binding.execution
                }) {
                    return (
                        None,
                        None,
                        None,
                        Some(PalsFollowupObservationIssue::DuplicateValueExecution),
                    );
                }
                if context_revision.is_some_and(|revision| revision != value.context_revision) {
                    return (
                        None,
                        None,
                        None,
                        Some(PalsFollowupObservationIssue::ValueExecutionMismatch),
                    );
                }
                context_revision = Some(value.context_revision);
                first = Some((binding.request, binding.execution));
                requests += 1;
                let Some(sum) = inputs.checked_add(binding.completed_nn_inputs.unwrap_or(0)) else {
                    return (
                        None,
                        None,
                        None,
                        Some(PalsFollowupObservationIssue::IncompleteValueExecution),
                    );
                };
                inputs = sum;
            }
        }
    }
    (Some(requests), Some(inputs), context_revision, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuConfig;

    /// An explicit synchronous fixture supplies its own distinct typed Runtime
    /// namespace. Its counters exercise the join; no native/NN work is claimed.
    #[derive(Default)]
    struct BoundValueFixture {
        mock: LegalOrderRoleMock,
        sequence: u64,
        inputs: Option<u64>,
        last: Option<RoleValueExecutionEvidence>,
        mismatch_context: bool,
    }
    impl RoleModel for BoundValueFixture {
        fn identity(&self) -> &str {
            self.mock.identity()
        }
        fn value_identity(&self) -> Option<&ModelValueIdentity> {
            self.mock.value_identity()
        }
        fn last_value_execution_evidence(&self) -> Option<RoleValueExecutionEvidence> {
            self.last
        }
        fn evaluate_value_with_context(
            &mut self,
            query: RoleQuery<'_>,
            context: &RoleLogicalContext,
        ) -> Result<ModelValueOutput, RoleError> {
            let output = self.mock.evaluate_value(query)?;
            self.sequence += 1;
            let epoch = rz_contracts::ProcessEpoch(77);
            let mut observed_context = RoleExecutionContext::from(context);
            if self.mismatch_context {
                observed_context.public_revision += 1;
            }
            self.last = Some(RoleValueExecutionEvidence {
                request: rz_contracts::RequestId::new(epoch, self.sequence),
                execution: rz_contracts::ExecutionId::new(epoch, self.sequence),
                prepared_state: rz_contracts::StateIdentity {
                    owner: rz_contracts::OwnerId(77),
                    revision: rz_contracts::StateRevision(self.sequence),
                    semantic: rz_contracts::Digest(output.input_sha256),
                },
                prepared_perspective: match output.perspective {
                    Color::White => rz_contracts::Color::White,
                    Color::Black => rz_contracts::Color::Black,
                },
                input_sha256: output.input_sha256,
                model_epoch: output.identity.model_epoch,
                context: observed_context,
                fresh: Some(true),
                physically_completed: Some(true),
                completed_nn_inputs: self.inputs,
                complete: self.inputs.is_some(),
            });
            Ok(output)
        }
        fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            self.mock.propose(query)
        }
        fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            self.mock.reply(query)
        }
        fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            self.mock.repair(query)
        }
        fn divergences(&mut self, query: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
            self.mock.divergences(query)
        }
    }

    fn engine(policy: PostRepairRecheckPolicy) -> PalsEngine<BoundValueFixture> {
        PalsEngine::new_with_boxed_checker_and_policies(
            PalsConfig {
                line_plies: 6,
                max_nodes: 257,
                max_records: 128,
                ..PalsConfig::default()
            },
            BoundValueFixture {
                inputs: Some(2),
                ..BoundValueFixture::default()
            },
            Box::new(
                OwnedCpuChecker::new(Box::new(
                    CpuEngine::new(CpuConfig {
                        tt_entries: 64,
                        max_depth: 2,
                        ..CpuConfig::default()
                    })
                    .unwrap(),
                ))
                .unwrap(),
            ),
            ResolverPolicy::ModelWdlRestricted,
            policy,
        )
        .unwrap()
    }
    fn limits() -> PalsLimits {
        PalsLimits {
            deadline: Instant::now() + Duration::from_secs(20),
            max_rounds: 1,
            max_cpu_nodes: 1024,
            cpu_depth: 1,
        }
    }
    fn entered_root(engine: &mut PalsEngine<BoundValueFixture>) -> usize {
        let position = Position::startpos();
        engine.begin_followup_execution(&position);
        engine
            .semantic
            .stores
            .focus_actual_moves(position.snapshot())
            .unwrap();
        let root = engine.intern(position).unwrap();
        engine.observe_followup_root(root);
        root
    }

    #[test]
    fn actual_value_acceptance_joins_distinct_bindings_for_repeated_input_and_four_inputs() {
        let mut engine = engine(PostRepairRecheckPolicy::FrozenModelWdlV2);
        assert!(engine.last_followup_execution_snapshot().is_none());
        let root = entered_root(&mut engine);
        let limits = limits();
        let cancel = AtomicBool::new(false);
        let mut counters = PalsCounters::default();
        engine
            .evaluate_model_value(root, &[], limits, &cancel, &mut counters)
            .unwrap();
        let first = engine.execution_endpoint(root);
        engine
            .evaluate_model_value(root, &[], limits, &cancel, &mut counters)
            .unwrap();
        let second = engine.execution_endpoint(root);
        let (PalsExecutionEndpointValue::FrozenWdl(a), PalsExecutionEndpointValue::FrozenWdl(b)) =
            (&first.value, &second.value)
        else {
            panic!("published WDL evidence missing")
        };
        assert!(a.logical_accepted && b.logical_accepted);
        assert_eq!(a.input_sha256, b.input_sha256);
        assert_ne!(a.observation, b.observation);
        assert_ne!(a.execution.unwrap().request, b.execution.unwrap().request);
        assert_eq!(
            a.execution.unwrap().prepared_state.owner,
            rz_contracts::OwnerId(77)
        );
        assert_eq!(
            execution_pair(&first, Some(&second)),
            (Some(2), Some(4), Some(0), None)
        );
        assert_eq!(
            (counters.value_calls, counters.accepted_value_outputs),
            (2, 2)
        );
        assert_eq!(counters.cpu_tasks, 0);
        assert_eq!(
            engine.semantic.nodes[root].model_observation,
            Some(b.observation)
        );
    }

    #[test]
    fn new_go_clears_fresh_bindings_without_resetting_owner_or_attempt_sequence() {
        let mut engine = engine(PostRepairRecheckPolicy::FrozenModelWdlV2);
        let root = entered_root(&mut engine);
        engine
            .evaluate_model_value(
                root,
                &[],
                limits(),
                &AtomicBool::new(false),
                &mut PalsCounters::default(),
            )
            .unwrap();
        assert!(matches!(engine.execution_endpoint(root).value,
            PalsExecutionEndpointValue::FrozenWdl(ref value) if value.execution.is_some()));
        let generation = engine.semantic.game_generation;
        engine.begin_followup_execution(&Position::startpos());
        let stale = engine.execution_endpoint(root);
        assert_eq!(engine.semantic.game_generation, generation);
        assert_eq!(
            engine
                .last_followup_execution_snapshot()
                .unwrap()
                .search_attempt_sequence,
            Some(2)
        );
        assert!(
            matches!(stale.value, PalsExecutionEndpointValue::FrozenWdl(ref value) if value.execution.is_none())
        );
        assert_eq!(
            execution_pair(&stale, Some(&stale)).3,
            Some(PalsFollowupObservationIssue::MissingValueExecution)
        );
        engine.clear_followup_game();
        assert!(engine.last_followup_execution_snapshot().is_none());
        assert_eq!(
            engine.followup_owner_snapshot().search_attempt_sequence,
            Some(2)
        );
    }

    #[test]
    fn mismatched_observer_binding_keeps_valid_value_and_marks_actual_receipt_incomplete() {
        let mut engine = engine(PostRepairRecheckPolicy::FrozenModelWdlV2);
        let root = entered_root(&mut engine);
        engine.semantic.model.mismatch_context = true;
        let mut counters = PalsCounters::default();
        engine
            .evaluate_model_value(root, &[], limits(), &AtomicBool::new(false), &mut counters)
            .unwrap();
        assert!(engine.semantic.nodes[root].model_value.is_some());
        assert_eq!(counters.accepted_value_outputs, 1);
        assert!(matches!(engine.execution_endpoint(root).value,
            PalsExecutionEndpointValue::FrozenWdl(ref value) if value.execution.is_none()));
        let snapshot = engine.last_followup_execution_snapshot().unwrap();
        assert!(!snapshot.complete);
        assert!(
            snapshot
                .issues
                .contains(&PalsFollowupObservationIssue::ValueExecutionMismatch)
        );
    }

    #[test]
    fn unknown_or_duplicate_physical_evidence_is_not_two_calls_and_rules_terminals_are_zero() {
        let mut engine = engine(PostRepairRecheckPolicy::FrozenModelWdlV2);
        let root = entered_root(&mut engine);
        engine
            .evaluate_model_value(
                root,
                &[],
                limits(),
                &AtomicBool::new(false),
                &mut PalsCounters::default(),
            )
            .unwrap();
        let first = engine.execution_endpoint(root);
        assert_eq!(
            execution_pair(&first, Some(&first)).3,
            Some(PalsFollowupObservationIssue::DuplicateValueExecution)
        );
        engine.semantic.model.inputs = None;
        engine
            .evaluate_model_value(
                root,
                &[],
                limits(),
                &AtomicBool::new(false),
                &mut PalsCounters::default(),
            )
            .unwrap();
        let unknown = engine.execution_endpoint(root);
        assert_eq!(execution_pair(&first, Some(&unknown)).0, None);
        assert_eq!(
            execution_pair(&first, Some(&unknown)).3,
            Some(PalsFollowupObservationIssue::IncompleteValueExecution)
        );
        let terminal = engine
            .intern(Position::from_fen("7k/8/5KQ1/8/8/8/8/8 b - - 0 1").unwrap())
            .unwrap();
        let exact = engine.execution_endpoint(terminal);
        assert!(matches!(
            exact.value,
            PalsExecutionEndpointValue::RulesTerminal {
                reason: TerminalReason::Stalemate,
                ..
            }
        ));
        assert_eq!(
            execution_pair(&exact, Some(&exact)),
            (Some(0), Some(0), None, None)
        );
    }

    #[test]
    fn accepted_records_preserve_actual_parent_supersedes_and_bound_trace_overflow() {
        let mut engine = engine(PostRepairRecheckPolicy::FrozenModelWdlV2);
        let root = entered_root(&mut engine);
        let line: Vec<_> = ["e2e4", "e7e5", "g1f3"]
            .map(|value| BoardMove::from_uci(value).unwrap())
            .into();
        engine
            .record(RecordKind::Counterexample, &line, None, 0, None, None)
            .unwrap();
        let parent = engine.semantic.revision;
        let mut previous = None;
        for ordinal in 1..=65 {
            let question = engine
                .role_context(
                    root,
                    RoleQuestion {
                        purpose: RoleQueryPurpose::RepairPolicy,
                        prefix: &[],
                        proposal: &line,
                        refutation: Some(&line),
                        divergences: &[],
                    },
                )
                .unwrap();
            engine.observe_repair_question(&question);
            let (id, _) = engine
                .record(RecordKind::Repair, &line, None, 0, None, None)
                .unwrap()
                .unwrap();
            engine.observe_accepted_repair(root, line[0], id, engine.semantic.revision);
            if ordinal <= 64 {
                let snapshot = engine.last_followup_execution_snapshot().unwrap();
                let trace = snapshot.repair_traces.last().unwrap();
                assert_eq!(trace.parent_revision, Some(parent));
                assert_eq!(trace.supersedes_revision, previous);
                assert_eq!(trace.repair_ordinal, ordinal);
                assert_eq!(trace.lineage_root_record, parent + 1);
            }
            previous = Some(engine.semantic.revision);
        }
        let snapshot = engine.last_followup_execution_snapshot().unwrap();
        assert_eq!(
            (snapshot.repair_traces.len(), snapshot.repair_trace_total),
            (64, 65)
        );
        assert!(snapshot.overflow && !snapshot.complete);
        assert!(
            snapshot
                .issues
                .contains(&PalsFollowupObservationIssue::TraceOverflow)
        );
        assert_eq!(engine.semantic.records.len(), 66);
        assert_eq!(engine.semantic.revision, parent + 65);
    }

    #[test]
    fn iterative_admission_peak_includes_failed_admissions_separately_from_accepted_records() {
        let mut engine = engine(PostRepairRecheckPolicy::IterativeFrozenModelWdlV2);
        entered_root(&mut engine);
        let movement = BoardMove::from_uci("e2e4").unwrap();
        for _ in 0..3 {
            assert!(engine.admit_iterative_repair(movement));
        }
        assert!(!engine.admit_iterative_repair(movement));
        engine.finish_followup_execution(
            RoleSearchClosure::Failed,
            Some(&PalsError::Role(RoleError::InvalidOutput)),
            PalsCounters::default(),
        );
        let snapshot = engine.last_followup_execution_snapshot().unwrap();
        assert_eq!(snapshot.repair_admissions_peak, Some(3));
        assert_eq!(snapshot.pending_questions_peak, Some(0));
        assert_eq!(snapshot.repair_trace_total, 0);
        assert!(snapshot.repair_traces.is_empty());
        assert!(matches!(
            snapshot.original_error,
            Some(PalsError::Role(RoleError::InvalidOutput))
        ));
    }
}
