//! Search-owner commit for the private foreign CPU worker.
//!
//! A UCI node request is a hint. Neither `bestmove`, reported CP/mate, nor a
//! stop/readiness barrier is a native minimax/Rules proof. Semantic and physical
//! question namespaces are registered separately and never substituted.

use super::*;
use crate::external_cpu::worker::{
    ExternalWorkerReport, external_evaluation_namespace, external_question_digest,
    external_root_restriction_digest, external_search_conditions,
};
use crate::external_cpu::{
    ExternalCooperativeReport, ExternalCpuConfig, ExternalGoIssue, ExternalSlicePolicy,
};
use crate::pals::coordinator::CheckedExternalCpuBinding;
use crate::pals::store::{ForeignAppendPermit, ForeignGoIssuedProof};
use rz_contracts::pals_adaptive::{ExternalCpuJobEnvelope, ExternalSliceGrant};
use rz_contracts::{ClockDomain, MonotonicTick, StateIdentity};

const FOREIGN_QUESTION_CAPACITY: usize = 64;

pub struct AdaptivePortExternalQuestion {
    pub position: ContractPosition,
    pub root_moves: Option<Vec<BoardMove>>,
    pub target_depth: u16,
    pub policy: ExternalSlicePolicy,
}

pub struct AdaptiveForeignPublication {
    pub observation: Observation,
    pub issued: ForeignGoIssuedProof,
}

#[derive(Clone)]
pub(super) struct ExternalPortAdmission {
    node: usize,
    position: Position,
    state_identity: StateIdentity,
    restriction: Option<Vec<BoardMove>>,
    target_depth: u16,
    config: ExternalCpuConfig,
    policy: ExternalSlicePolicy,
    physical_question: ContractDigest,
    generation: u64,
    root: SituationId,
    root_state: StateId,
    root_revision: u64,
    root_snapshot: PositionSnapshot,
    situation: SituationId,
    situation_revision: u64,
    task_key: TaskKey,
    execution: Option<ExecutionId>,
    consumer: Option<u64>,
    original_consumer: Option<TaskConsumer>,
    job: Option<ExternalCpuJobEnvelope>,
    grant: Option<ExternalSliceGrant>,
    received: Option<ContractDigest>,
    physical_accounted: bool,
    published: Option<ObservationId>,
    consumed_observation: Option<ObservationId>,
    prepared_line: Option<Option<LineId>>,
    task_settled: bool,
    frame: crate::pals::coordinator::FrameId,
    question: Arc<AdaptivePortQuestion>,
    view: Arc<ReviewView>,
    view_seal: ReviewViewSeal,
    context: RoleLogicalContext,
    first_move: rz_contracts::pals::Move16,
    terminal: bool,
    quarantined: bool,
}

impl ExternalPortAdmission {
    /// Conservative observable admission payload backing, not an RSS/allocator/
    /// physical peak. Arc control blocks and Store-owned pin registry storage
    /// are outside this scope. Shared question/view/history backing is charged
    /// per retained reference; unknown BTreeMap allocation stays unknown.
    pub(super) fn retained_host_bytes(&self) -> Option<u64> {
        let mut bytes = std::mem::size_of::<Self>();
        for backing in [
            usize::try_from(self.question.owned_host_bytes()?).ok()?,
            usize::try_from(self.view.owned_host_bytes()?).ok()?,
            self.position.snapshot().retained_history_bytes()?,
            self.root_snapshot.retained_history_bytes()?,
            self.restriction
                .as_ref()
                .map_or(0, Vec::capacity)
                .checked_mul(std::mem::size_of::<BoardMove>())?,
            self.context
                .prefix
                .capacity()
                .checked_mul(std::mem::size_of::<BoardMove>())?,
            self.task_key
                .root_moves
                .capacity()
                .checked_mul(std::mem::size_of::<Move16>())?,
            self.task_key
                .cpu_condition
                .as_ref()
                .map_or(0, String::capacity),
            self.config.program.capacity(),
            self.config.working_directory.capacity(),
            self.config
                .arguments
                .capacity()
                .checked_mul(std::mem::size_of::<std::ffi::OsString>())?,
            self.config
                .expected_uci_name
                .as_ref()
                .map_or(0, String::capacity),
            foreign_identity_owned_backing(&self.config.identity)?,
        ] {
            bytes = bytes.checked_add(backing)?;
        }
        for argument in &self.config.arguments {
            bytes = bytes.checked_add(argument.capacity())?;
        }
        if self.task_key.value_identity.is_some() {
            // This is not a supported foreign task; do not silently ignore an
            // unexpected owned value namespace's retained allocation.
            return None;
        }
        if let Some(checker) = &self.task_key.checker_identity {
            let CheckerIdentity::ExternalUci(identity) = checker else {
                return None;
            };
            bytes = bytes.checked_add(foreign_identity_owned_backing(identity)?)?;
        }
        u64::try_from(bytes).ok()
    }
}

fn foreign_identity_owned_backing(
    identity: &crate::cpu_checker::ExternalCheckerIdentity,
) -> Option<usize> {
    // Public BTreeMap APIs expose entry count, not allocated node capacity.
    // A nonempty map cannot be reported as zero or an invented node size.
    if !identity.options.is_empty() {
        return None;
    }
    let mut bytes = identity
        .assets
        .capacity()
        .checked_mul(std::mem::size_of::<crate::cpu_checker::ExternalAssetIdentity>())?;
    for text in [
        &identity.adapter_semantics,
        &identity.binary_sha256,
        &identity.launch_arguments_sha256,
        &identity.declared_name,
        &identity.declared_version,
        &identity.declared_source,
        &identity.declared_license,
    ] {
        bytes = bytes.checked_add(text.capacity())?;
    }
    for asset in &identity.assets {
        bytes = bytes.checked_add(asset.purpose.capacity())?;
        bytes = bytes.checked_add(asset.sha256.capacity())?;
    }
    for text in [
        &identity.model_metadata.weights_sha256,
        &identity.model_metadata.declared_rights,
        &identity.model_metadata.precision,
    ]
    .into_iter()
    .flatten()
    {
        bytes = bytes.checked_add(text.capacity())?;
    }
    Some(bytes)
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// A legal coherent external tuple may forecast early stop for the
    /// existing P/C continuation. It remains raw selective-engine feedback:
    /// neither an own completed iteration nor a conditional proof.
    pub fn published_external_continuation_eligible(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        progress: &crate::external_cpu::worker::ExternalWorkerProgress,
    ) -> bool {
        let Some(admission) = self.external_admissions.get(&work.input.question_digest) else {
            return false;
        };
        let (Some(job), Some(grant), Some(best)) = (
            admission.job,
            admission.grant,
            progress.opinion.pv.first().copied(),
        ) else {
            return false;
        };
        let query = &work.input.question;
        let proposal = if query.proposal.is_empty() {
            &query.context.prefix
        } else {
            &query.proposal
        };
        if self.check_work(work).is_err()
            || !self.external_context_current(work, admission)
            || !admission
                .execution
                .zip(admission.original_consumer)
                .is_some_and(|(execution, original)| {
                    admission.consumer == Some(original.id)
                        && self
                            .engine
                            .semantic
                            .stores
                            .hot_validate_pending_cpu_feedback(
                                execution,
                                original,
                                &admission.task_key,
                                &admission.position.snapshot(),
                                self.engine.tick_at(Instant::now()),
                            )
                            .is_ok()
                })
            || !Arc::ptr_eq(&work.input.question, &admission.question)
            || !Arc::ptr_eq(&work.input.view, &admission.view)
            || work.input.view_seal != admission.view_seal
            || work.original_first_move != admission.first_move
            || query.node != admission.node
            || query.context != admission.context
            || !query
                .position
                .snapshot()
                .same_state(&admission.position.snapshot())
            || !foreign_root_restriction(query)
                .is_ok_and(|restriction| restriction == admission.restriction)
            || work.budget.role_submissions_remaining == 0
            || admission.terminal
            || admission.quarantined
            || query.stage != RefinementStage::CpuValidation
            || self.engine.semantic.nodes[query.node].terminal.is_some()
            || !self
                .divergence_indices(proposal)
                .is_ok_and(|indices| !indices.is_empty())
            || job.id != progress.scope.job
            || grant.job != job.id
            || grant.execution != progress.scope.execution
            || grant.owner != progress.scope.owner
            || progress.strict_native_nodes.is_some()
            || progress.completed_iteration_observed.is_some()
            || progress.opinion.rank != 1
            || progress.opinion.depth == 0
            || progress.opinion.depth > admission.target_depth
            || !admission.position.legal_moves().contains(&best)
            || admission
                .restriction
                .as_ref()
                .is_some_and(|moves| !moves.contains(&best))
        {
            return false;
        }
        let mut position = admission.position.clone();
        progress
            .opinion
            .pv
            .iter()
            .all(|move_| position.make_move(*move_).is_ok())
    }

    /// Observable admission payload backing retained by the foreign endpoint.
    /// HashMap allocator/control, Arc control blocks and the Store-owned pin
    /// registry are outside this scope: this is neither whole-port ownership
    /// nor RSS/physical peak. Shared backing is charged conservatively per
    /// admission. Unknown allocation/overflow remains unknown; callers must
    /// not turn it into zero for resource admission.
    pub fn external_retained_host_bytes(&self) -> Option<u64> {
        self.external_admissions
            .values()
            .try_fold(0u64, |sum, admission| {
                sum.checked_add(admission.retained_host_bytes()?)
            })
    }

    /// No file verification/spawn/pipe drain occurs here. The private worker's
    /// startup attestation remains independent of bounded declaration checks.
    pub fn external_question(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        config: &ExternalCpuConfig,
        mut policy: ExternalSlicePolicy,
    ) -> Result<AdaptivePortExternalQuestion, PalsError> {
        self.check_work(work)?;
        config.validate()?;
        self.engine.validate_checker_namespace()?;
        if !foreign_stage(work.input.stage)
            || self.closed.contains(&work.frame)
            || self.engine.semantic.nodes[work.input.question.node]
                .terminal
                .is_some()
            || self.engine.researcher.checker_registered_identity
                != CheckerIdentity::ExternalUci(config.identity.clone())
            || !self.engine.uses_model_resolver()
        {
            return Err(CheckerError::Invalid("foreign question namespace or resolver").into());
        }
        let digest = work.input.question_digest;
        let restriction = foreign_root_restriction(&work.input.question)?;
        if let Some(old) = self.external_admissions.get(&digest) {
            if old.terminal
                || old.quarantined
                || !Arc::ptr_eq(&work.input.question, &old.question)
                || !Arc::ptr_eq(&work.input.view, &old.view)
                || !self.external_context_current(work, old)
                || old.config.identity != config.identity
                || old.restriction != restriction
                || external_search_conditions(config, old.target_depth, policy)
                    != external_search_conditions(&old.config, old.target_depth, old.policy)
                || policy.quantum != old.policy.quantum
                || policy.stop_drain_reserve != old.policy.stop_drain_reserve
                || policy.multipv != old.policy.multipv
                || policy.requested_reported_nodes < old.policy.requested_reported_nodes
            {
                return Err(CheckerError::Invalid("foreign question cannot be replaced").into());
            }
            return Ok(AdaptivePortExternalQuestion {
                // Re-export only the same immutable owner/position, never issue
                // a fresh owner that would replace this physical question.
                position: ContractPosition::new(old.state_identity.owner, old.position.clone()),
                root_moves: old.restriction.clone(),
                target_depth: old.target_depth,
                policy: old.policy,
            });
        }
        self.external_admissions.retain(|_, value| !value.terminal);
        if self.external_admissions.len() >= FOREIGN_QUESTION_CAPACITY {
            return Err(PalsError::Capacity);
        }
        let question = &work.input.question;
        let node = &self.engine.semantic.nodes[question.node];
        let remaining = self
            .limits
            .max_cpu_nodes
            .checked_sub(self.engine.cpu_budget_used(&self.counters))
            .ok_or(PalsError::InvalidLimits)?;
        policy.requested_reported_nodes = policy
            .requested_reported_nodes
            .min(remaining)
            .min(self.engine.config.cpu_nodes_per_task);
        policy.validate(
            restriction
                .as_ref()
                .map_or_else(|| question.position.legal_moves().len(), Vec::len),
        )?;
        if self.limits.cpu_depth > config.max_depth {
            return Err(PalsError::InvalidLimits);
        }
        let position = question.position.clone();
        let rules_position = self.issue_rules_position(position.clone())?;
        let exported = rules_position
            .export()
            .map_err(|_| RoleError::InvalidOutput)?;
        let state_identity = exported.snapshot().identity();
        let roots = external_root_restriction_digest(restriction.as_deref());
        let evaluation = external_evaluation_namespace(&config.identity)?;
        let conditions = external_search_conditions(config, self.limits.cpu_depth, policy);
        let physical_question =
            external_question_digest(state_identity, roots, evaluation, conditions);
        let task_question = match question.stage {
            _ if restriction.is_some() => TaskQuestion::AnalyzeRootMoves,
            RefinementStage::CpuDiscovery => TaskQuestion::FindAlternative {
                divergence_ply: u16::try_from(question.context.prefix.len())
                    .map_err(|_| PalsError::InvalidLimits)?,
            },
            _ => TaskQuestion::AnalyzePosition,
        };
        let root_moves: Vec<_> = restriction
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .copied()
            .map(Move16::pack)
            .collect::<Result<_, _>>()?;
        let semantic_hex = hex_digest(digest);
        let physical_hex = hex_digest(physical_question);
        let condition = format!(
            "{};adaptive-semantic={semantic_hex};adaptive-physical={physical_hex};question={task_question:?};nodes={};depth={};deadline-tick={}",
            self.engine.researcher.cpu_registered_condition,
            policy.requested_reported_nodes,
            self.limits.cpu_depth,
            self.deadline.at.0
        );
        if condition.len() > 8192 {
            return Err(PalsError::InvalidConfig);
        }
        let root = self
            .engine
            .semantic
            .stores
            .root()
            .ok_or(StoreError::StaleConsumer)?;
        let root_record = self.engine.semantic.stores.situations.get(root)?;
        let situation = node.situation;
        let admission = ExternalPortAdmission {
            node: question.node,
            position,
            state_identity,
            restriction: restriction.clone(),
            target_depth: self.limits.cpu_depth,
            config: config.clone(),
            policy,
            physical_question,
            generation: self.engine.semantic.stores.generation(),
            root,
            root_state: root_record.state,
            root_revision: root_record.revision,
            root_snapshot: self
                .engine
                .semantic
                .stores
                .states
                .get(root_record.state)?
                .clone(),
            situation,
            situation_revision: self
                .engine
                .semantic
                .stores
                .situations
                .get(situation)?
                .revision,
            task_key: TaskKey {
                state: node.state,
                line: None,
                question: task_question,
                root_moves,
                value_identity: None,
                checker_identity: Some(CheckerIdentity::ExternalUci(config.identity.clone())),
                cpu_condition: Some(condition.clone()),
                model: 0,
                epoch: 0,
                profile: 0,
                condition: stable_id(&condition),
                input_revision: question.context.prefix.len() as u64,
                requested_depth: self.limits.cpu_depth,
                node_budget: policy.requested_reported_nodes,
            },
            execution: None,
            consumer: None,
            original_consumer: None,
            job: None,
            grant: None,
            received: None,
            physical_accounted: false,
            published: None,
            consumed_observation: None,
            prepared_line: None,
            task_settled: false,
            frame: work.frame,
            question: Arc::clone(&work.input.question),
            view: Arc::clone(&work.input.view),
            view_seal: work.input.view_seal,
            context: question.context.clone(),
            first_move: work.original_first_move,
            terminal: false,
            quarantined: false,
        };
        self.external_admissions.insert(digest, admission);
        Ok(AdaptivePortExternalQuestion {
            position: rules_position,
            root_moves: restriction,
            target_depth: self.limits.cpu_depth,
            policy,
        })
    }

    pub fn external_admitted(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        job: &ExternalCpuJobEnvelope,
        grant: &ExternalSliceGrant,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<CheckedExternalCpuBinding, PalsError> {
        self.check_work(work)?;
        let admission = self
            .external_admissions
            .get(&work.input.question_digest)
            .ok_or(CheckerError::Invalid("foreign question was not prepared"))?
            .clone();
        let exported =
            ContractPosition::new(admission.state_identity.owner, admission.position.clone())
                .export()
                .map_err(|_| RoleError::InvalidOutput)?;
        if admission.job.is_some()
            || admission.terminal
            || admission.quarantined
            || !self.external_context_current(work, &admission)
            || job.id.authority != self.authority
            || job.id.sequence == 0
            || job.state_identity != exported.snapshot().identity()
            || job.question != admission.physical_question
            || job.root_restriction
                != external_root_restriction_digest(admission.restriction.as_deref())
            || job.evaluation_namespace
                != external_evaluation_namespace(&admission.config.identity)?
            || job.search_conditions
                != external_search_conditions(
                    &admission.config,
                    admission.target_depth,
                    admission.policy,
                )
            || job.requested_reported_node_hint != admission.policy.requested_reported_nodes
            || job.hard_deadline != self.deadline
            || grant.job != job.id
            || grant.execution.epoch != self.authority.epoch
            || grant.execution.sequence == 0
            || grant.control_sequence == 0
            || grant.requested_reported_node_hint != admission.policy.requested_reported_nodes
            || grant.soft_quantum != admission.policy.quantum
        {
            return Err(
                CheckerError::Invalid("foreign grant differs from immutable question").into(),
            );
        }
        let proof = CheckedExternalCpuBinding::attest(
            work,
            job,
            grant,
            &exported,
            admission.restriction.as_deref(),
            admission.target_depth,
            admission.policy,
            &admission.config,
            clock,
            now,
        )
        .map_err(|_| CheckerError::Invalid("foreign semantic/physical binding proof"))?;
        let reserved = self
            .counters
            .external_checker_node_budget_reserved
            .checked_add(admission.policy.requested_reported_nodes)
            .ok_or(PalsError::Capacity)?;
        let tasks = self
            .counters
            .external_checker_tasks
            .checked_add(1)
            .ok_or(PalsError::Capacity)?;
        self.engine.semantic.consumer_id = self
            .engine
            .semantic
            .consumer_id
            .checked_add(1)
            .ok_or(PalsError::Capacity)?;
        let consumer = self.engine.semantic.consumer_id;
        let original_consumer = TaskConsumer {
            id: consumer,
            situation: admission.situation,
            revision: admission.situation_revision,
            generation: admission.generation,
            deadline_tick: self.engine.tick_at(self.limits.deadline),
        };
        let task = self.engine.semantic.stores.hot_request_task(
            admission.task_key.clone(),
            original_consumer,
            self.engine.tick_at(Instant::now()),
        )?;
        let execution = match task {
            TaskAdmission::Start(execution) => execution,
            TaskAdmission::Join(execution) | TaskAdmission::Reuse { execution, .. } => {
                self.engine
                    .semantic
                    .stores
                    .tasks
                    .cancel_consumer(execution, consumer)?;
                return Err(
                    CheckerError::Invalid("foreign dispatch cannot reuse another go").into(),
                );
            }
            TaskAdmission::Resume { execution, .. } => {
                self.engine.semantic.stores.tasks.fail(execution)?;
                return Err(CheckerError::Unsupported("foreign exact stack resume").into());
            }
        };
        let stored = self
            .external_admissions
            .get_mut(&work.input.question_digest)
            .expect("prepared foreign");
        stored.execution = Some(execution);
        stored.consumer = Some(consumer);
        stored.original_consumer = Some(original_consumer);
        stored.job = Some(*job);
        stored.grant = Some(*grant);
        self.counters.external_checker_node_budget_reserved = reserved;
        self.counters.external_checker_tasks = tasks;
        Ok(proof)
    }

    /// Only a proven submit rejection can release an unused foreign hint. A
    /// worker's ambiguous write/failure follows the physical receipt instead.
    pub fn external_rejected(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        job: &ExternalCpuJobEnvelope,
        grant: &ExternalSliceGrant,
    ) -> Result<(), PalsError> {
        let admission = self
            .external_admissions
            .get(&work.input.question_digest)
            .ok_or(CheckerError::Invalid("foreign rejection lacks admission"))?
            .clone();
        if admission.job.as_ref() != Some(job)
            || admission.grant.as_ref() != Some(grant)
            || admission.physical_accounted
            || admission.received.is_some()
            || admission.terminal
        {
            return Err(CheckerError::Invalid("foreign rejection scope or repeated refund").into());
        }
        self.engine
            .semantic
            .stores
            .tasks
            .fail(admission.execution.ok_or(PalsError::InvalidConfig)?)?;
        self.counters.external_checker_node_budget_reserved = self
            .counters
            .external_checker_node_budget_reserved
            .checked_sub(admission.policy.requested_reported_nodes)
            .ok_or(PalsError::InvalidConfig)?;
        self.counters.external_checker_tasks = self
            .counters
            .external_checker_tasks
            .checked_sub(1)
            .ok_or(PalsError::InvalidConfig)?;
        self.external_admissions
            .get_mut(&work.input.question_digest)
            .expect("checked foreign")
            .terminal = true;
        Ok(())
    }

    /// Manager hot publication. The bool describes reported request coverage,
    /// never current frame authority; follow-up work additionally requires the
    /// private consumed-observation seal. Cold publication requires a worker-
    /// issued uniqueness permit and returns the ordinary typed barrier.
    pub fn commit_external_report(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        receipt: &ExternalWorkerReport,
    ) -> Result<(ObservationId, bool), PalsError> {
        self.commit_external_inner(work, receipt, None, true)
    }

    /// Shutdown observes physical work in its original immutable namespace.
    /// It neither advances a frame nor consumes a task as new search evidence.
    /// Expiry/revocation cannot hide already executed foreign work.
    pub fn observe_external_report(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        receipt: &ExternalWorkerReport,
    ) -> Result<(), PalsError> {
        let admission = self.external_receipt_admission(work, receipt)?;
        let fingerprint = receipt_digest(receipt);
        if admission.physical_accounted {
            if admission.received != Some(fingerprint) {
                return Err(CheckerError::Invalid(
                    "foreign shutdown receipt changed physical evidence",
                )
                .into());
            }
            if admission.quarantined {
                return Err(
                    CheckerError::Invalid("foreign physical receipt is quarantined").into(),
                );
            }
            if admission.terminal {
                return match &receipt.result {
                    Ok(_) if admission.published.is_some() => Ok(()),
                    Err(error) => Err(error.clone().into()),
                    _ => {
                        Err(CheckerError::Invalid("foreign terminal lacks raw publication").into())
                    }
                };
            }
        }
        self.commit_external_inner(work, receipt, None, false)
            .map(|_| ())
    }

    /// A V action may refer only to an actually published terminal foreign
    /// execution, not admission, hinted nodes, or an unknown/quarantined go.
    pub(super) fn external_action_committed(
        &self,
        question: ContractDigest,
        job: rz_contracts::pals_adaptive::CpuJobId,
        execution: rz_contracts::ExecutionId,
    ) -> bool {
        self.external_admissions
            .get(&question)
            .is_some_and(|admission| {
                admission.terminal
                    && !admission.quarantined
                    && admission.physical_accounted
                    && admission.received.is_some()
                    && admission.published.is_some()
                    && admission.job.is_some_and(|issued| issued.id == job)
                    && admission
                        .grant
                        .is_some_and(|grant| grant.job == job && grant.execution == execution)
            })
    }

    /// Only the original question whose live consumer actually accepted this
    /// exact raw observation may create follow-up model/line work. Physical
    /// coverage, publication, or terminal readiness alone grants no authority.
    pub(super) fn external_observation_consumed(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        observation: ObservationId,
    ) -> bool {
        self.external_admissions
            .get(&work.input.question_digest)
            .is_some_and(|admission| {
                admission.frame == work.frame
                    && admission.first_move == work.original_first_move
                    && Arc::ptr_eq(&admission.question, &work.input.question)
                    && Arc::ptr_eq(&admission.view, &work.input.view)
                    && work.input.question.question_digest == work.input.question_digest
                    && work.input.stage == admission.question.stage
                    && work.input.view_seal == admission.view_seal
                    && self.seal(&work.input.view) == admission.view_seal
                    && admission.context == work.input.question.context
                    && self.external_context_current(work, admission)
                    && self
                        .external_feedback_still_authorized(work.input.question_digest, observation)
            })
    }

    /// Read-only materialization gate for a feedback already consumed by the
    /// exact original Work. No Work clone or new view lease is needed: the
    /// private admission retains the original authority and physical binding.
    /// Cancellation after after_cpu must still revoke follow-up publication.
    pub(super) fn external_feedback_still_authorized(
        &self,
        question: ContractDigest,
        observation: ObservationId,
    ) -> bool {
        self.external_admissions
            .get(&question)
            .is_some_and(|admission| {
                admission.terminal
                    && !admission.quarantined
                    && admission.physical_accounted
                    && admission.task_settled
                    && admission.consumed_observation == Some(observation)
                    && admission.published == Some(observation)
                    && admission.question.question_digest == question
                    && admission.question.node == admission.node
                    && admission.question.context == admission.context
                    && self.seal(&admission.view) == admission.view_seal
                    && admission
                        .job
                        .zip(admission.grant)
                        .is_some_and(|(job, grant)| {
                            job.id.authority == admission.frame.authority
                                && job.state_identity == admission.state_identity
                                && job.question == admission.physical_question
                                && job.hard_deadline == self.deadline
                                && grant.job == job.id
                                && grant.execution.epoch == job.id.authority.epoch
                        })
                    && self.external_admission_context_current(admission)
                    && admission
                        .execution
                        .zip(admission.original_consumer)
                        .is_some_and(|(execution, original)| {
                            admission.consumer == Some(original.id)
                                && self
                                    .engine
                                    .semantic
                                    .stores
                                    .hot_validate_consumed_foreign_feedback(
                                        execution,
                                        original,
                                        &admission.task_key,
                                        &admission.position.snapshot(),
                                        observation,
                                        self.engine.tick_at(Instant::now()),
                                    )
                                    .is_ok_and(|validated| validated == observation)
                        })
                    && self
                        .engine
                        .semantic
                        .stores
                        .observations
                        .get(observation)
                        .is_ok_and(|raw| {
                            raw.execution == admission.execution
                                && raw.state == admission.task_key.state
                                && raw.checker_identity == admission.task_key.checker_identity
                                && raw.cpu_condition == admission.task_key.cpu_condition
                                && raw.kind == ObservationKind::ExternalCpuAnalysis
                        })
            })
    }

    /// Manager prepares a raw payload; the archive worker only scans its
    /// uniqueness. A scan failure never authorizes another payload or deadline.
    pub fn prepare_external_publication(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        receipt: &ExternalWorkerReport,
    ) -> Result<AdaptiveForeignPublication, PalsError> {
        let (admission, result) = self.checked_external_receipt(work, receipt)?;
        if !self.engine.semantic.stores.archive_enabled() || admission.published.is_some() {
            return Err(CheckerError::Invalid(
                "foreign permit is only for unpublished cold-on receipt",
            )
            .into());
        }
        let observation =
            self.external_observation(work.input.question_digest, &admission, &result.report)?;
        let issued = ForeignGoIssuedProof::checked_worker_report(
            receipt,
            &observation,
            self.limits.deadline,
        )?;
        Ok(AdaptiveForeignPublication {
            observation,
            issued,
        })
    }

    pub fn commit_external_permitted(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        receipt: &ExternalWorkerReport,
        permit: ForeignAppendPermit,
    ) -> Result<(ObservationId, bool), PalsError> {
        self.commit_external_inner(work, receipt, Some(permit), true)
    }

    /// The same one-use cold uniqueness capability, with search consumption
    /// explicitly disabled for shutdown. The original permit deadline remains.
    pub fn observe_external_permitted(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        receipt: &ExternalWorkerReport,
        permit: ForeignAppendPermit,
    ) -> Result<(), PalsError> {
        self.commit_external_inner(work, receipt, Some(permit), false)
            .map(|_| ())
    }

    fn commit_external_inner(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        receipt: &ExternalWorkerReport,
        permit: Option<ForeignAppendPermit>,
        consume_current: bool,
    ) -> Result<(ObservationId, bool), PalsError> {
        let (admission, cooperative) = self.checked_external_receipt(work, receipt)?;
        let semantic = work.input.question_digest;
        // No allocation before the cold-scan admission barrier, so repeated
        // notification does not create a new PV line or a new observation.
        if self.engine.semantic.stores.archive_enabled()
            && admission.published.is_none()
            && permit.is_none()
        {
            return Err(StoreError::ArchiveBudgetRequired.into());
        }
        let report = &cooperative.report;
        let execution = admission.execution.expect("checked execution");
        let observation = if let Some(published) = admission.published {
            if permit.is_some() {
                return Err(CheckerError::Invalid("already-published foreign permit").into());
            }
            published
        } else {
            let expected = self.external_observation(semantic, &admission, report)?;
            let published = if let Some(permit) = permit {
                if permit.observation() != &expected
                    || permit.native_job() != receipt.job.id
                    || permit.native_execution() != receipt.scope.execution
                {
                    return Err(CheckerError::Invalid(
                        "foreign permit belongs to another raw publication",
                    )
                    .into());
                }
                self.engine
                    .semantic
                    .stores
                    .hot_append_foreign_permitted(permit)?
            } else {
                self.engine
                    .semantic
                    .stores
                    .hot_append_observation(expected)?
            };
            // Preserve the append across a later dependency/capacity failure.
            // Retrying the same physical receipt must not publish twice.
            self.external_admissions
                .get_mut(&semantic)
                .expect("admitted foreign")
                .published = Some(published);
            published
        };
        self.engine
            .semantic
            .stores
            .hot_add_dependency(observation, admission.situation)?;
        self.engine
            .semantic
            .stores
            .hot_add_dependency(observation, admission.root)?;
        self.engine.pin_archive_active_observation(observation)?;
        let coverage = report.completion == ExternalCompletion::BestMove
            && report.best_move.is_some()
            && report.work.nodes.is_some();
        let reusable = coverage && !cooperative.reported_node_overshoot;
        let current = self.external_context_current(work, &admission);
        let next_consumed = if consume_current && current && reusable {
            Some(
                self.counters
                    .consumed_external_checker_tasks
                    .checked_add(1)
                    .ok_or(PalsError::Capacity)?,
            )
        } else {
            None
        };
        if !admission.task_settled {
            if reusable {
                self.engine
                    .semantic
                    .stores
                    .complete_task(execution, observation)?;
            } else {
                // Reported overshoot is raw evidence and cannot satisfy a
                // cached bounded request. No strict native credit is charged.
                self.engine.semantic.stores.tasks.fail(execution)?;
            }
            // Completion survives a later consumption rejection. Repeating a
            // terminal TaskTable mutation would hide the retained observation.
            self.external_admissions
                .get_mut(&semantic)
                .expect("admitted foreign")
                .task_settled = true;
        }
        let followup = matches!(
            report.completion,
            ExternalCompletion::BestMove | ExternalCompletion::StoppedYield
        );
        if consume_current && current && followup {
            let original = admission
                .original_consumer
                .expect("admitted original consumer");
            let snapshot = admission.position.snapshot();
            let now = self.engine.tick_at(Instant::now());
            let consumed = if reusable {
                self.engine.semantic.stores.hot_consume_sealed_cpu_task(
                    execution,
                    original,
                    &admission.task_key,
                    &snapshot,
                    now,
                )
            } else {
                // Partial/hint-overshoot feedback has a one-use original
                // consumer authority but never becomes a completed/cacheable
                // bounded task. It can ask the model for Fresh WDL only.
                self.engine
                    .semantic
                    .stores
                    .hot_consume_sealed_foreign_feedback(
                        execution,
                        original,
                        &admission.task_key,
                        &snapshot,
                        observation,
                        now,
                    )
            };
            match consumed {
                Ok(_) => {
                    if reusable {
                        self.counters.consumed_external_checker_tasks =
                            next_consumed.expect("preflighted consumption count");
                    }
                    self.external_admissions
                        .get_mut(&semantic)
                        .expect("admitted foreign")
                        .consumed_observation = Some(observation);
                }
                Err(
                    StoreError::ExpiredConsumer
                    | StoreError::CancelledConsumer
                    | StoreError::StaleConsumer,
                ) => {
                    // A normal revocation race keeps raw completed evidence,
                    // with no search consumption or replacement permission.
                }
                Err(error) => {
                    let stored = self
                        .external_admissions
                        .get_mut(&semantic)
                        .expect("admitted foreign");
                    stored.terminal = true;
                    stored.quarantined = true;
                    return Err(error.into());
                }
            }
        }
        self.external_admissions
            .get_mut(&semantic)
            .expect("admitted foreign")
            .terminal = true;
        Ok((observation, coverage))
    }

    fn checked_external_receipt<'r>(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        receipt: &'r ExternalWorkerReport,
    ) -> Result<(ExternalPortAdmission, &'r ExternalCooperativeReport), PalsError> {
        let semantic = work.input.question_digest;
        let admission = self.external_receipt_admission(work, receipt)?;
        if admission.terminal || admission.quarantined {
            return Err(CheckerError::Invalid("duplicate or quarantined foreign receipt").into());
        }
        let fingerprint = receipt_digest(receipt);
        if admission.received.is_some_and(|old| old != fingerprint) {
            return Err(CheckerError::Invalid(
                "foreign publication retry changed physical receipt",
            )
            .into());
        }
        if !admission.physical_accounted {
            self.capture_external_receipt(semantic, receipt)?;
        }
        if let Some(attempt) = receipt.attempt.as_ref().and_then(|a| a.external.as_ref()) {
            if attempt.partial_report.as_ref().is_some_and(|partial| {
                partial.request_id != attempt.request_id
                    || partial.identity != admission.config.identity
                    || partial.requested_depth != admission.target_depth
                    || partial.root_restricted != admission.restriction.is_some()
                    || partial.perspective != admission.position.side_to_move()
                    || admission
                        .config
                        .expected_uci_name
                        .as_ref()
                        .is_some_and(|name| *name != partial.observed_uci.name)
            }) {
                // Preserve actual raw work before rejecting its attribution.
                // Failed partial reports cannot bypass the frozen source pin.
                self.engine
                    .semantic
                    .stores
                    .tasks
                    .fail(admission.execution.expect("checked execution"))?;
                self.external_admissions
                    .get_mut(&semantic)
                    .expect("admitted foreign")
                    .quarantined = true;
                return Err(CheckerError::Invalid(
                    "foreign retained attempt changed source or scope",
                )
                .into());
            }
        }
        let cooperative = match &receipt.result {
            Ok(result) => result,
            Err(error) => {
                let stored = self
                    .external_admissions
                    .get_mut(&semantic)
                    .expect("admitted foreign");
                if receipt.go_issue == ExternalGoIssue::Unknown {
                    stored.quarantined = true;
                } else {
                    self.engine
                        .semantic
                        .stores
                        .tasks
                        .fail(admission.execution.expect("checked execution"))?;
                    stored.terminal = true;
                }
                return Err(error.clone().into());
            }
        };
        if let Err(error) = self.validate_cooperative_receipt(&admission, receipt, cooperative) {
            self.engine
                .semantic
                .stores
                .tasks
                .fail(admission.execution.expect("checked execution"))?;
            self.external_admissions
                .get_mut(&semantic)
                .expect("admitted foreign")
                .quarantined = true;
            return Err(error);
        }
        Ok((admission, cooperative))
    }

    /// Original scope checks remain valid after expiry, closed admission, or
    /// completed publication. Liveness is checked only for search consumption.
    fn external_receipt_admission(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        receipt: &ExternalWorkerReport,
    ) -> Result<ExternalPortAdmission, PalsError> {
        let semantic = work.input.question_digest;
        let admission = self
            .external_admissions
            .get(&semantic)
            .ok_or(CheckerError::Invalid("foreign receipt has no admission"))?
            .clone();
        let job = admission
            .job
            .ok_or(CheckerError::Invalid("foreign receipt was not dispatched"))?;
        let grant = admission
            .grant
            .ok_or(CheckerError::Invalid("foreign receipt lacks actual grant"))?;
        validate_receipt_header(receipt, job, grant)?;
        if work.frame != admission.frame
            || !Arc::ptr_eq(&work.input.question, &admission.question)
            || !Arc::ptr_eq(&work.input.view, &admission.view)
            || work.input.stage != admission.question.stage
            || work.frame.authority != job.id.authority
            || work.original_deadline != job.hard_deadline
            || work.input.view_seal != admission.view_seal
            || work.original_first_move != admission.first_move
            || work.input.question.question_digest != semantic
            || work.input.question.node != admission.node
            || work.input.question.context != admission.context
            || foreign_root_restriction(&work.input.question)? != admission.restriction
            || !work
                .input
                .question
                .position
                .snapshot()
                .same_state(&admission.position.snapshot())
            || self
                .engine
                .semantic
                .stores
                .tasks
                .get(admission.execution.ok_or(PalsError::InvalidConfig)?)?
                .key
                != admission.task_key
        {
            return Err(CheckerError::Invalid("mismatched original foreign receipt scope").into());
        }
        Ok(admission)
    }

    fn external_observation(
        &mut self,
        semantic: ContractDigest,
        admission: &ExternalPortAdmission,
        report: &ExternalCheckerReport,
    ) -> Result<Observation, PalsError> {
        let cpu_pv = if let Some(line) = admission.prepared_line {
            line
        } else {
            let projected: Vec<_> = report
                .pv
                .iter()
                .copied()
                .take(self.continuation_limit())
                .collect();
            let line = if projected.is_empty() {
                None
            } else {
                Some(
                    self.engine
                        .semantic
                        .stores
                        .hot_append_cpu_pv(&admission.position, &projected)?,
                )
            };
            self.external_admissions
                .get_mut(&semantic)
                .expect("admitted foreign")
                .prepared_line = Some(line);
            line
        };
        Ok(Observation {
            state: admission.task_key.state,
            line: None,
            source: stable_id(&report.identity.adapter_semantics),
            epoch: 0,
            value_identity: None,
            checker_identity: admission.task_key.checker_identity.clone(),
            checker_work: Some(report.work),
            external_report: Some(Box::new(report.clone())),
            model_value_identity: None,
            model_value_input: None,
            cpu_condition: admission.task_key.cpu_condition.clone(),
            cpu_pv,
            scope: EvidenceScope::ExternalUci {
                requested_depth: report.requested_depth,
                reported_depth: report.reported_depth,
                seldepth: report.seldepth,
                bound: report.bound,
            },
            score: RawScore::ExternalUci {
                value: report.score,
                bound: report.bound,
                perspective: report.perspective,
                wdl_per_mille: report.wdl_per_mille,
            },
            budget: report.work.nodes.unwrap_or(0),
            kind: ObservationKind::ExternalCpuAnalysis,
            supersedes: None,
            execution: admission.execution,
        })
    }
    fn capture_external_receipt(
        &mut self,
        semantic: ContractDigest,
        receipt: &ExternalWorkerReport,
    ) -> Result<(), PalsError> {
        let mut next = self.counters;
        if let Some(attempt) = &receipt.attempt {
            let external = attempt.external.as_ref().ok_or(CheckerError::Invalid(
                "foreign physical attempt has own work namespace",
            ))?;
            if external.request_id == 0
                || self
                    .engine
                    .researcher
                    .last_external_attempt
                    .is_some_and(|old| external.request_id <= old)
            {
                return Err(CheckerError::Invalid("foreign physical go ID reused").into());
            }
            if let Some(partial) = &external.partial_report {
                partial.identity.validate()?;
                if partial.pv.len() > 256
                    || partial.pv.capacity() > 256
                    || partial.observed_uci.name.len() > 256
                    || partial.observed_uci.name.capacity() > 256
                    || partial
                        .observed_uci
                        .author
                        .as_ref()
                        .is_some_and(|value| value.len() > 256 || value.capacity() > 256)
                {
                    return Err(
                        CheckerError::Invalid("foreign retained attempt exceeds bounds").into(),
                    );
                }
            }
            if self.engine.researcher.external_attempts.len() >= self.engine.config.max_records {
                return Err(PalsError::Capacity);
            }
            if let Some(nodes) = attempt.work.nodes {
                next.external_checker_nodes_observed = next
                    .external_checker_nodes_observed
                    .checked_add(nodes)
                    .ok_or(PalsError::Capacity)?;
            }
            next.external_checker_work_incomplete |= attempt.work.nodes.is_none()
                || attempt.work.qnodes.is_none()
                || attempt.work.tt_hits.is_none();
        } else {
            next.external_checker_work_incomplete |= receipt.go_issue != ExternalGoIssue::NotIssued;
        }
        if receipt.result.is_ok() {
            next.external_checker_reports = next
                .external_checker_reports
                .checked_add(1)
                .ok_or(PalsError::Capacity)?;
        } else if receipt.go_issue == ExternalGoIssue::NotIssued && receipt.attempt.is_none() {
            let admission = &self.external_admissions[&semantic];
            next.external_checker_node_budget_reserved = next
                .external_checker_node_budget_reserved
                .checked_sub(admission.policy.requested_reported_nodes)
                .ok_or(PalsError::InvalidConfig)?;
            next.external_checker_tasks = next
                .external_checker_tasks
                .checked_sub(1)
                .ok_or(PalsError::InvalidConfig)?;
        }
        // Every fallible budget/counter check precedes publication. A failed
        // preflight must not advance the go high-water or append one half of
        // an unaccounted receipt; the same retained worker report remains
        // retryable by shutdown without a fabricated "go ID reused" failure.
        if let Some(attempt) = &receipt.attempt {
            self.engine
                .researcher
                .external_attempts
                .push(attempt.clone());
            self.engine.researcher.last_external_attempt = Some(
                attempt
                    .external
                    .as_ref()
                    .expect("checked foreign namespace")
                    .request_id,
            );
        }
        let admission = self
            .external_admissions
            .get_mut(&semantic)
            .expect("admitted foreign");
        admission.received = Some(receipt_digest(receipt));
        admission.physical_accounted = true;
        self.counters = next;
        Ok(())
    }

    fn validate_cooperative_receipt(
        &self,
        admission: &ExternalPortAdmission,
        receipt: &ExternalWorkerReport,
        result: &ExternalCooperativeReport,
    ) -> Result<(), PalsError> {
        validate_cooperative_control(receipt, result)?;
        let report = &result.report;
        if result.reported_node_request != admission.policy.requested_reported_nodes
            || result.requested_multipv != admission.policy.multipv
            || result.reported_node_overshoot
                != report
                    .work
                    .nodes
                    .is_some_and(|nodes| nodes > admission.policy.requested_reported_nodes)
            || report.identity != admission.config.identity
            || admission
                .config
                .expected_uci_name
                .as_ref()
                .is_some_and(|name| *name != report.observed_uci.name)
            || receipt
                .attempt
                .as_ref()
                .and_then(|attempt| attempt.external.as_ref())
                .is_none_or(|attempt| {
                    attempt.request_id != report.request_id
                        || attempt.partial_report.as_ref() != Some(report)
                })
            || receipt.attempt.as_ref().is_none_or(|attempt| {
                attempt.work != report.work || attempt.elapsed != report.elapsed
            })
        {
            return Err(CheckerError::Invalid(
                "foreign result differs from actual issue/profile/attempt",
            )
            .into());
        }
        self.engine.validate_external_report(
            &admission.position,
            report,
            admission.restriction.as_deref(),
            CpuLimits {
                max_depth: admission.target_depth,
                // Cooperative reported node hints are not physical hard caps.
                max_nodes: u64::MAX,
                deadline: Some(self.limits.deadline),
            },
        )?;
        if result.opinions.len() > usize::from(admission.policy.multipv)
            || result.opinions.capacity() > 4
        {
            return Err(CheckerError::Invalid("foreign conditional policy opinion cap").into());
        }
        let mut ranks = HashSet::new();
        let mut roots = HashSet::new();
        for opinion in &result.opinions {
            if opinion.rank == 0
                || opinion.rank > admission.policy.multipv
                || !ranks.insert(opinion.rank)
                || opinion.pv.is_empty()
                || opinion.pv.len() > 256
                || opinion.pv.capacity() > 256
                || !roots.insert(opinion.pv[0])
                || admission
                    .restriction
                    .as_ref()
                    .is_some_and(|allowed| !allowed.contains(&opinion.pv[0]))
            {
                return Err(CheckerError::Invalid("foreign rank/PV/ordered support").into());
            }
            let mut replay = admission.position.clone();
            for mv in &opinion.pv {
                replay.make_move(*mv)?;
            }
        }
        if result.policy_target_known
            && (result.opinions.len() != usize::from(admission.policy.multipv)
                || report.completion != ExternalCompletion::BestMove
                || result.opinions.iter().enumerate().any(|(at, opinion)| {
                    opinion.rank != at as u16 + 1
                        || Some(opinion.depth) != result.opinions.first().map(|first| first.depth)
                        || opinion.bound != crate::cpu_checker::ExternalBound::ExactReported
                })
                || result
                    .opinions
                    .first()
                    .and_then(|opinion| opinion.pv.first())
                    .copied()
                    != report.best_move)
        {
            return Err(
                CheckerError::Invalid("foreign policy target lacks complete rank set").into(),
            );
        }
        Ok(())
    }

    fn external_context_current(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        admission: &ExternalPortAdmission,
    ) -> bool {
        work.frame == admission.frame
            && work.frame.authority == self.authority
            && work.original_deadline == self.deadline
            && self.external_admission_context_current(admission)
    }

    fn external_admission_context_current(&self, admission: &ExternalPortAdmission) -> bool {
        Instant::now() < self.limits.deadline
            && !self.closed.contains(&admission.frame)
            && admission.frame.authority == self.authority
            && self.engine.semantic.game_generation == Some(self.authority.game.0)
            && self.engine.semantic.stores.generation() == admission.generation
            && self.engine.semantic.stores.root() == Some(admission.root)
            && self
                .engine
                .semantic
                .stores
                .situations
                .get(admission.root)
                .is_ok_and(|root| {
                    root.state == admission.root_state
                        && root.revision >= admission.root_revision
                        && self
                            .engine
                            .semantic
                            .stores
                            .states
                            .get(root.state)
                            .is_ok_and(|snapshot| snapshot.same_state(&admission.root_snapshot))
                })
            && self
                .engine
                .semantic
                .stores
                .situations
                .get(admission.situation)
                .is_ok_and(|situation| {
                    situation.revision >= admission.situation_revision
                        && situation.state == admission.task_key.state
                })
            && self
                .engine
                .semantic
                .nodes
                .get(admission.node)
                .is_some_and(|node| {
                    node.state == admission.task_key.state
                        && node.situation == admission.situation
                        && node
                            .position
                            .snapshot()
                            .same_state(&admission.position.snapshot())
                })
    }
}

pub(super) fn foreign_stage(stage: RefinementStage) -> bool {
    matches!(
        stage,
        RefinementStage::CpuValidation
            | RefinementStage::CpuDiscovery
            | RefinementStage::CounterValidation
            | RefinementStage::RepairValidation
            | RefinementStage::RecheckCounterValidation
    )
}

pub(super) fn foreign_root_restriction(
    question: &AdaptivePortQuestion,
) -> Result<Option<Vec<BoardMove>>, PalsError> {
    let legal = question.position.legal_moves();
    if let Some(roots) = &question.validator_root_moves {
        // Explicit V action restrictions take precedence over the historical
        // discovery exclusion. Preserve the advertised candidate order.
        let mut seen = HashSet::new();
        if roots.is_empty()
            || roots.len() > 256
            || roots.capacity() > 256
            || roots
                .iter()
                .any(|movement| !legal.contains(movement) || !seen.insert(*movement))
        {
            return Err(CheckerError::Invalid("foreign explicit root restriction").into());
        }
        return Ok(Some(roots.clone()));
    }
    if question.stage != RefinementStage::CpuDiscovery || !question.discovery_restricted {
        return Ok(None);
    }
    let attack = question.attack_ply.ok_or(RoleError::InvalidOutput)?;
    let old = question
        .proposal
        .get(attack)
        .ok_or(RoleError::InvalidOutput)?;
    if question.context.prefix != question.proposal[..attack] {
        return Err(RoleError::InvalidOutput.into());
    }
    let alternatives: Vec<_> = legal
        .into_iter()
        .filter(|movement| movement != old)
        .collect();
    if alternatives.is_empty() {
        return Err(CheckerError::Invalid("foreign discovery has no alternative").into());
    }
    Ok(Some(alternatives))
}

fn validate_receipt_header(
    receipt: &ExternalWorkerReport,
    job: ExternalCpuJobEnvelope,
    grant: ExternalSliceGrant,
) -> Result<(), PalsError> {
    if receipt.job != job
        || receipt.grant != grant
        || receipt.scope.job != job.id
        || receipt.scope.execution != grant.execution
        || receipt.scope.owner != grant.owner
        || receipt.exact_resume_supported
        || !receipt.controls.terminal
        || receipt.controls.cancel_ack > receipt.controls.cancel_sequence
        || receipt.controls.yield_ack > receipt.controls.yield_sequence
        || receipt.controls.publish_ack > receipt.controls.publish_sequence
    {
        return Err(CheckerError::Invalid("foreign old-go/owner/terminal/control seal").into());
    }
    Ok(())
}

fn validate_cooperative_control(
    receipt: &ExternalWorkerReport,
    result: &ExternalCooperativeReport,
) -> Result<(), PalsError> {
    use crate::external_cpu::ExternalControlAction;
    if result.go_issue != ExternalGoIssue::Issued
        || receipt.go_issue != result.go_issue
        || result.strict_native_nodes.is_some()
        || !result.readiness_barrier_completed
        || result.report.completion == ExternalCompletion::Pending
    {
        return Err(
            CheckerError::Invalid("foreign success lacks actual issue and ready drain").into(),
        );
    }
    match (result.report.completion, result.stop_control) {
        (ExternalCompletion::StoppedYield, Some(control))
            if control.action == ExternalControlAction::Yield
                && control.sequence != 0
                && control.sequence == receipt.controls.yield_sequence
                && control.sequence == receipt.controls.yield_ack => {}
        (ExternalCompletion::StoppedCanceled, Some(control))
            if control.action == ExternalControlAction::Cancel
                && control.sequence != 0
                && control.sequence == receipt.controls.cancel_sequence
                && control.sequence == receipt.controls.cancel_ack => {}
        (ExternalCompletion::StoppedCanceled, Some(control))
            if control.action == ExternalControlAction::Revoked
                && control.sequence == 0
                && receipt.controls.cancel_sequence == 0
                && receipt.controls.cancel_ack == 0 => {}
        // Natural soft-quantum Yield and a shutdown cancellation carry no
        // fabricated mailbox request/ack. Both still require actual stop/drain.
        (ExternalCompletion::StoppedYield, None) if receipt.controls.yield_ack == 0 => {}
        (ExternalCompletion::StoppedCanceled, None)
            if receipt.controls.cancel_sequence == 0 && receipt.controls.cancel_ack == 0 => {}
        (ExternalCompletion::StoppedDeadline | ExternalCompletion::BestMove, None) => {}
        _ => {
            return Err(
                CheckerError::Invalid("foreign stop completion and control ack disagree").into(),
            );
        }
    }
    Ok(())
}

fn hex_digest(digest: ContractDigest) -> String {
    digest.0.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Retry identity only, not a wire codec or execution/result acceptance proof.
/// Streaming Debug formatting avoids allocating a second retained raw report.
fn receipt_digest(receipt: &ExternalWorkerReport) -> ContractDigest {
    struct HashWriter(Sha256);
    impl std::fmt::Write for HashWriter {
        fn write_str(&mut self, text: &str) -> std::fmt::Result {
            self.0.update(text.as_bytes());
            Ok(())
        }
    }
    use std::fmt::Write;
    let mut writer = HashWriter(Sha256::new());
    writer.0.update(b"rz-pals-foreign-publication-retry/1\0");
    write!(&mut writer, "{receipt:?}").expect("infallible hash formatting");
    ContractDigest(writer.0.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_checker::{
        CheckerCapabilities, CheckerWork, ExternalAttemptEvidence, ExternalBound,
        ExternalCheckerIdentity, ExternalModelMetadata, ExternalRawScore,
        ExternalTrainingKnowledge, ExternalUciIdentity,
    };
    use crate::external_cpu::worker::{ExternalWorkerProgress, ExternalWorkerScope};
    use crate::pals::coordinator::FrameId;
    use crate::pals::review_context::{ReviewIndexLimits, ReviewSelectionPolicy};
    use rz_contracts::pals_adaptive::{CpuJobId, CpuWorkerId};
    use rz_contracts::{GameGeneration, ProcessEpoch, RootGeneration};
    use rz_runtime::adaptive::ControlSnapshot;
    use std::sync::atomic::AtomicU64;

    /// Metadata only. A mistaken manager-side search fails this source test
    /// immediately; fixtures are not actual UCI/physical execution evidence.
    struct MetadataChecker(CheckerIdentity);
    impl CpuChecker for MetadataChecker {
        fn identity(&self) -> &CheckerIdentity {
            &self.0
        }
        fn conditions(&self) -> &str {
            "fixture-foreign-private-worker/1"
        }
        fn capabilities(&self) -> CheckerCapabilities {
            CheckerCapabilities {
                max_depth: 8,
                max_prefix_plies: 64,
                max_root_moves: 256,
                root_moves: true,
                divergence: true,
                resume: false,
                selective_search: None,
            }
        }
        fn last_attempt(&self) -> Option<&CheckerAttempt> {
            None
        }
        fn reset_attempt(&mut self) {}
        fn analyze(
            &mut self,
            _: &Position,
            _: CpuLimits,
            _: &AtomicBool,
        ) -> Result<CheckerReport, CheckerError> {
            panic!("manager executed foreign CPU")
        }
        fn new_game(&mut self, _: Instant, _: &AtomicBool) -> Result<(), CheckerError> {
            Ok(())
        }
        fn shutdown(&mut self, _: Instant) -> Result<CheckerShutdown, CheckerError> {
            Err(CheckerError::Unsupported(
                "source fixture has no physical process",
            ))
        }
    }

    type Port = AdaptiveEnginePort<LegalOrderRoleMock>;
    type Work = CoordinatorWork<ReviewView, AdaptivePortQuestion>;

    fn fixture() -> (Port, Work, ExternalCpuConfig) {
        let directory = std::env::temp_dir();
        let identity = ExternalCheckerIdentity {
            adapter_semantics: crate::external_cpu::EXTERNAL_UCI_SCORE_SEMANTICS.into(),
            binary_sha256: "a".repeat(64),
            launch_arguments_sha256: crate::external_cpu::arguments_sha256(&[]),
            declared_name: "fixture".into(),
            declared_version: "1".into(),
            declared_source: "inert-source-fixture".into(),
            declared_license: "MIT".into(),
            options: Default::default(),
            assets: Vec::new(),
            model_metadata: ExternalModelMetadata {
                weights_sha256: None,
                training: ExternalTrainingKnowledge::Unknown,
                declared_rights: None,
                precision: None,
            },
        };
        let config = ExternalCpuConfig {
            program: directory.join("rz-inert-foreign-source-fixture"),
            arguments: Vec::new(),
            working_directory: directory,
            identity: identity.clone(),
            expected_uci_name: Some("fixture".into()),
            max_depth: 8,
            max_prefix_plies: 64,
            handshake_timeout: Duration::from_secs(1),
            max_task_wall_time: Duration::from_secs(2),
            stop_grace: Duration::from_millis(500),
            shutdown_grace: Duration::from_secs(1),
            max_output_bytes: 65536,
            max_line_bytes: 8192,
        };
        let engine = PalsEngine::new_with_checker(
            PalsConfig {
                // Match the product's registered root plus maximum 256 legal
                // candidates. This fixture examines foreign receipt paths;
                // it must not fail earlier at invalid Store configuration.
                max_nodes: 257,
                max_records: 16,
                beam_width: 1,
                line_plies: 2,
                cpu_nodes_per_task: 32,
                ..PalsConfig::default()
            },
            LegalOrderRoleMock,
            MetadataChecker(CheckerIdentity::ExternalUci(identity)),
        )
        .unwrap()
        .with_selected_review_context(
            ReviewSelectionPolicy::default(),
            ReviewIndexLimits::default(),
        )
        .unwrap();
        let authority = SearchAuthority {
            epoch: ProcessEpoch(29),
            game: GameGeneration(0),
            root: RootGeneration(1),
            implementation: ContractDigest([9; 32]),
        };
        let deadline = Deadline {
            clock: ClockDomain(authority.epoch),
            at: MonotonicTick(20_000_000_000),
        };
        // Test registry owns issued IDs; roots/questions cannot synthesize them
        // from generations. This callback models the actual UCI registry seam.
        let sequence = Arc::new(AtomicU64::new(1));
        let issuer: AdaptiveRulesOwnerIssuer =
            Arc::new(move || Ok(OwnerId(sequence.fetch_add(1, Ordering::Relaxed))));
        let mut port = Port::new(
            engine,
            Position::startpos(),
            authority,
            deadline,
            PalsLimits {
                deadline: Instant::now() + Duration::from_secs(20),
                max_rounds: 1,
                max_cpu_nodes: 128,
                cpu_depth: 2,
            },
            issuer,
        )
        .unwrap();
        let first = BoardMove::from_uci("e2e4").unwrap();
        let node = port.connect_hot(port.root, first).unwrap();
        let input = port
            .input(
                node,
                RefinementStage::CpuValidation,
                Some(first),
                vec![first],
                vec![first],
                None,
                None,
                None,
                false,
                None,
                None,
            )
            .unwrap();
        let work = Work {
            frame: FrameId {
                authority,
                sequence: 1,
            },
            original_first_move: rz_contracts::pals::Move16::encode(
                rz_contracts::Move::try_from(first).unwrap(),
            ),
            original_deadline: deadline,
            budget: FrameBudget {
                original_cpu_credit: port.limits.max_cpu_nodes,
                role_submissions_remaining: port.engine.config.max_role_calls,
                // The fixture's registered logical hot-store ceiling; it is
                // not an observed resident-memory or external-node credit.
                original_store_byte_limit: 8 * 1024 * 1024,
            },
            input,
        };
        (port, work, config)
    }

    fn dispatched(
        port: &mut Port,
        work: &Work,
        config: &ExternalCpuConfig,
    ) -> ExternalWorkerReport {
        let prepared = port
            .external_question(
                work,
                config,
                ExternalSlicePolicy {
                    requested_reported_nodes: 16,
                    ..ExternalSlicePolicy::default()
                },
            )
            .unwrap();
        let state = prepared.position.export().unwrap();
        let roots = external_root_restriction_digest(prepared.root_moves.as_deref());
        let evaluation = external_evaluation_namespace(&config.identity).unwrap();
        let conditions = external_search_conditions(config, prepared.target_depth, prepared.policy);
        let job = ExternalCpuJobEnvelope {
            id: CpuJobId {
                authority: port.authority,
                sequence: 1,
            },
            state_identity: state.snapshot().identity(),
            question: external_question_digest(
                state.snapshot().identity(),
                roots,
                evaluation,
                conditions,
            ),
            root_restriction: roots,
            evaluation_namespace: evaluation,
            search_conditions: conditions,
            requested_reported_node_hint: 16,
            hard_deadline: port.deadline,
        };
        let grant = ExternalSliceGrant {
            job: job.id,
            execution: rz_contracts::ExecutionId::new(port.authority.epoch, 77),
            owner: CpuWorkerId(1),
            requested_reported_node_hint: 16,
            soft_quantum: Duration::from_millis(100),
            control_sequence: 1,
        };
        let proof = port
            .external_admitted(work, &job, &grant, port.deadline.clock, MonotonicTick(1))
            .unwrap();
        assert_eq!(proof.semantic_question(), work.input.question_digest);
        assert_eq!(proof.physical_question(), job.question);
        assert_ne!(proof.semantic_question(), proof.physical_question());
        let raw = ExternalCheckerReport {
            identity: config.identity.clone(),
            observed_uci: ExternalUciIdentity {
                name: "fixture".into(),
                author: None,
            },
            request_id: 1,
            best_move: Some(BoardMove::from_uci("e7e5").unwrap()),
            pv: vec![BoardMove::from_uci("e7e5").unwrap()],
            score: ExternalRawScore::MateMoves(2),
            bound: ExternalBound::ExactReported,
            wdl_per_mille: None,
            perspective: Color::Black,
            requested_depth: 2,
            reported_depth: Some(1),
            seldepth: None,
            root_restricted: prepared.root_moves.is_some(),
            completion: ExternalCompletion::BestMove,
            work: CheckerWork {
                nodes: Some(8),
                qnodes: None,
                tt_hits: None,
            },
            elapsed: Duration::from_millis(1),
        };
        ExternalWorkerReport {
            job,
            grant,
            scope: ExternalWorkerScope {
                job: job.id,
                execution: grant.execution,
                owner: grant.owner,
            },
            result: Ok(ExternalCooperativeReport {
                report: raw.clone(),
                opinions: Vec::new(),
                stop_control: None,
                strict_native_nodes: None,
                reported_node_request: 16,
                requested_multipv: 1,
                option_application_observed: None,
                reported_node_overshoot: false,
                readiness_barrier_completed: true,
                policy_target_known: false,
                go_issue: ExternalGoIssue::Issued,
            }),
            attempt: Some(CheckerAttempt {
                work: raw.work,
                elapsed: raw.elapsed,
                external: Some(ExternalAttemptEvidence {
                    request_id: 1,
                    partial_report: Some(raw),
                    process: CheckerShutdown::default(),
                }),
            }),
            controls: ControlSnapshot {
                terminal: true,
                ..Default::default()
            },
            exact_resume_supported: false,
            go_issue: ExternalGoIssue::Issued,
        }
    }

    fn published_continuation_fixture() -> (Port, Work, ExternalWorkerProgress) {
        let (mut port, mut work, config) = fixture();
        let first = BoardMove::from_uci("e2e4").unwrap();
        let reply = BoardMove::from_uci("e7e5").unwrap();
        // The submitted proposal includes a real opponent continuation point.
        // The ordinary one-ply receipt fixture is otherwise insufficient to
        // authorize a P/C continuation forecast.
        work.input = port
            .input(
                work.input.question.node,
                RefinementStage::CpuValidation,
                Some(first),
                vec![first],
                vec![first, reply],
                None,
                None,
                None,
                false,
                None,
                None,
            )
            .unwrap();
        let receipt = dispatched(&mut port, &work, &config);
        let raw = &receipt.result.as_ref().unwrap().report;
        let progress = ExternalWorkerProgress {
            scope: receipt.scope,
            opinion: crate::external_cpu::ExternalPolicyOpinion {
                rank: 1,
                depth: raw.reported_depth.unwrap(),
                seldepth: raw.seldepth,
                pv: raw.pv.clone(),
                score: raw.score,
                bound: raw.bound,
                reported_wdl: raw.wdl_per_mille,
            },
            reported_nodes: raw.work.nodes,
            strict_native_nodes: None,
            control_sequence: 0,
            completed_iteration_observed: None,
        };
        (port, work, progress)
    }

    #[test]
    fn published_foreign_continuation_allows_only_the_original_legal_raw_question() {
        let (port, work, progress) = published_continuation_fixture();
        let before = port
            .engine
            .semantic
            .stores
            .observations
            .next_observation_sequence();
        assert!(port.published_external_continuation_eligible(&work, &progress));
        // Eligibility is read-only scheduling advice, not observation/target
        // publication, search consumption or completed native work.
        assert_eq!(
            port.engine
                .semantic
                .stores
                .observations
                .next_observation_sequence(),
            before
        );
        assert!(port.engine.semantic.records.is_empty());
        assert_eq!(port.counters.consumed_external_checker_tasks, 0);
        assert_eq!(port.counters.external_checker_nodes_observed, 0);
    }

    #[test]
    fn published_foreign_continuation_rejects_old_go_and_native_completion_fabrication() {
        let (port, work, progress) = published_continuation_fixture();
        let mut changed = progress.clone();
        changed.scope.job.sequence += 1;
        assert!(!port.published_external_continuation_eligible(&work, &changed));
        let mut changed = progress.clone();
        changed.scope.execution.sequence += 1;
        assert!(!port.published_external_continuation_eligible(&work, &changed));
        let mut changed = progress.clone();
        changed.scope.owner.0 += 1;
        assert!(!port.published_external_continuation_eligible(&work, &changed));
        for completed in [false, true] {
            let mut changed = progress.clone();
            changed.completed_iteration_observed = Some(completed);
            assert!(!port.published_external_continuation_eligible(&work, &changed));
        }
        for nodes in [0, 8] {
            let mut changed = progress.clone();
            changed.strict_native_nodes = Some(nodes);
            assert!(!port.published_external_continuation_eligible(&work, &changed));
        }
        let mut changed = progress.clone();
        changed.opinion.pv.clear();
        assert!(!port.published_external_continuation_eligible(&work, &changed));
        let mut changed = progress.clone();
        changed.opinion.pv = vec![BoardMove::from_uci("e2e3").unwrap()];
        assert!(!port.published_external_continuation_eligible(&work, &changed));
        let mut changed = progress.clone();
        changed
            .opinion
            .pv
            .push(BoardMove::from_uci("e2e4").unwrap());
        assert!(!port.published_external_continuation_eligible(&work, &changed));
    }

    #[test]
    fn published_foreign_continuation_rejects_anchor_query_view_and_first_move_drift() {
        let (mut port, work, progress) = published_continuation_fixture();
        let mut changed = work.clone();
        Arc::make_mut(&mut changed.input.question).position = Position::startpos();
        assert!(!port.published_external_continuation_eligible(&changed, &progress));
        let mut changed = work.clone();
        // Equal fields/digest in another Arc cannot replace the private query.
        let _ = Arc::make_mut(&mut changed.input.question);
        assert!(!port.published_external_continuation_eligible(&changed, &progress));
        let query = &work.input.question;
        let replacement = port
            .input(
                query.node,
                query.stage,
                query.first,
                query.context.prefix.clone(),
                query.proposal.clone(),
                None,
                None,
                None,
                false,
                None,
                None,
            )
            .unwrap();
        let mut changed = work.clone();
        changed.input.view = replacement.view;
        changed.input.view_seal = replacement.view_seal;
        changed.input.public_context = replacement.public_context;
        assert!(!port.published_external_continuation_eligible(&changed, &progress));
        let mut changed = work.clone();
        changed.original_first_move = rz_contracts::pals::Move16::encode(
            rz_contracts::Move::try_from(BoardMove::from_uci("d2d4").unwrap()).unwrap(),
        );
        assert!(!port.published_external_continuation_eligible(&changed, &progress));
        assert!(port.published_external_continuation_eligible(&work, &progress));
    }

    fn assert_partial_foreign_feedback_is_one_use(
        port: &mut Port,
        work: &Work,
        observation: ObservationId,
    ) {
        let admission = port.external_admissions[&work.input.question_digest].clone();
        assert_eq!(
            port.engine
                .semantic
                .stores
                .tasks
                .get(admission.execution.unwrap())
                .unwrap()
                .status,
            TaskStatus::Failed,
        );
        assert_eq!(port.counters.consumed_external_checker_tasks, 0);
        assert!(port.external_observation_consumed(work, observation));
        let now = port.engine.tick_at(Instant::now());
        assert!(matches!(
            port.engine
                .semantic
                .stores
                .hot_consume_sealed_foreign_feedback(
                    admission.execution.unwrap(),
                    admission.original_consumer.unwrap(),
                    &admission.task_key,
                    &admission.position.snapshot(),
                    observation,
                    now,
                ),
            Err(StoreError::AlreadyConsumed),
        ));
    }

    #[test]
    fn foreign_hot_commit_preserves_mate_as_raw_and_charges_no_native_credit() {
        let (mut port, work, config) = fixture();
        let receipt = dispatched(&mut port, &work, &config);
        let (observation, coverage) = port.commit_external_report(&work, &receipt).unwrap();
        assert!(coverage);
        let raw = port
            .engine
            .semantic
            .stores
            .observations
            .get(observation)
            .unwrap();
        assert!(matches!(
            raw.score,
            RawScore::ExternalUci {
                value: ExternalRawScore::MateMoves(2),
                ..
            }
        ));
        assert!(raw.value_identity.is_none());
        assert!(raw.model_value_identity.is_none());
        assert_eq!(port.counters.cpu_nodes, 0);
        assert_eq!(port.counters.external_checker_nodes_observed, 8);
        assert_eq!(port.counters.consumed_external_checker_tasks, 1);
        assert!(port.external_observation_consumed(&work, observation));
        assert!(
            port.engine.semantic.nodes[work.input.question.node]
                .evidence
                .is_none()
        );
        let before = port
            .engine
            .semantic
            .stores
            .observations
            .next_observation_sequence();
        assert!(port.commit_external_report(&work, &receipt).is_err());
        assert_eq!(
            port.engine
                .semantic
                .stores
                .observations
                .next_observation_sequence(),
            before
        );
    }

    #[test]
    fn foreign_verification_advances_to_fresh_without_cpu_verification_projection() {
        for score in [
            ExternalRawScore::MateMoves(2),
            ExternalRawScore::Centipawns(17),
        ] {
            let (mut port, work, config) = fixture();
            let mut receipt = dispatched(&mut port, &work, &config);
            let result = receipt.result.as_mut().unwrap();
            result.report.score = score;
            receipt
                .attempt
                .as_mut()
                .unwrap()
                .external
                .as_mut()
                .unwrap()
                .partial_report = Some(result.report.clone());
            let public_before = port.engine.semantic.records.len();
            let (observation, coverage) = port.commit_external_report(&work, &receipt).unwrap();
            let step = port.after_cpu(&work, observation, coverage).unwrap();
            let step = port.materialize_step(step).unwrap();
            let FrameTransition::Continue(next) = step.transition else {
                panic!("foreign verification must ask the frozen model for Fresh WDL");
            };
            assert_eq!(next.stage, RefinementStage::ProposalValueFresh);
            assert_eq!(port.engine.semantic.records.len(), public_before);
            assert!(
                port.engine
                    .semantic
                    .records
                    .iter()
                    .all(|record| { record.kind != RecordKind::CpuVerification })
            );
            assert!(
                next.view
                    .records()
                    .iter()
                    .all(|record| { record.kind != RecordKind::CpuVerification })
            );
            let raw = port
                .engine
                .semantic
                .stores
                .observations
                .get(observation)
                .unwrap();
            assert!(matches!(raw.score, RawScore::ExternalUci { value, .. } if value == score));
            assert!(raw.value_identity.is_none());
            assert!(raw.model_value_identity.is_none());
            assert!(
                port.engine.semantic.nodes[work.input.question.node]
                    .evidence
                    .is_none()
            );
        }
    }

    #[test]
    fn foreign_discovery_publishes_only_raw_source_counterexample_before_reply() {
        let (mut port, mut work, config) = fixture();
        let first = BoardMove::from_uci("e2e4").unwrap();
        let proposal = vec![first, BoardMove::from_uci("e7e5").unwrap()];
        port.record_hot(RecordKind::Proposal, &proposal, None)
            .unwrap();
        work.input = port
            .input(
                work.input.question.node,
                RefinementStage::CpuDiscovery,
                Some(first),
                vec![first],
                proposal,
                None,
                None,
                Some(1),
                false,
                None,
                None,
            )
            .unwrap();
        let mut receipt = dispatched(&mut port, &work, &config);
        let alternative = BoardMove::from_uci("c7c5").unwrap();
        let result = receipt.result.as_mut().unwrap();
        result.report.best_move = Some(alternative);
        result.report.pv = vec![alternative];
        receipt
            .attempt
            .as_mut()
            .unwrap()
            .external
            .as_mut()
            .unwrap()
            .partial_report = Some(result.report.clone());
        let (observation, coverage) = port.commit_external_report(&work, &receipt).unwrap();
        let step = port.after_cpu(&work, observation, coverage).unwrap();
        let step = port.materialize_step(step).unwrap();
        let FrameTransition::Continue(next) = step.transition else {
            panic!("foreign discovery must continue with critic reply");
        };
        assert_eq!(next.stage, RefinementStage::CriticReply);
        assert_eq!(next.question.counter_source, Some(observation));
        let record = port.engine.semantic.records.last().unwrap();
        assert_eq!(record.kind, RecordKind::Counterexample);
        assert_eq!(record.line, vec![first, alternative]);
        assert_eq!(record.value, None);
        assert_eq!(record.completed_depth, 0);
        assert_eq!(record.score_scope, None);
        assert_eq!(record.cpu_observation, Some(observation));
        assert!(
            port.engine
                .semantic
                .records
                .iter()
                .all(|record| { record.kind != RecordKind::CpuVerification })
        );
    }

    #[test]
    fn foreign_reported_overshoot_is_retained_without_reusable_task_or_native_charge() {
        let (mut port, work, config) = fixture();
        let mut receipt = dispatched(&mut port, &work, &config);
        let result = receipt.result.as_mut().unwrap();
        result.report.work.nodes = Some(17);
        result.reported_node_overshoot = true;
        let attempt = receipt.attempt.as_mut().unwrap();
        attempt.work = result.report.work;
        attempt.external.as_mut().unwrap().partial_report = Some(result.report.clone());
        let (observation, coverage) = port.commit_external_report(&work, &receipt).unwrap();
        assert!(coverage);
        assert_eq!(
            port.engine
                .semantic
                .stores
                .observations
                .get(observation)
                .unwrap()
                .budget,
            17
        );
        assert_eq!(port.counters.external_checker_nodes_observed, 17);
        assert_eq!(port.counters.cpu_nodes, 0);
        assert_eq!(port.counters.consumed_external_checker_tasks, 0);
        assert!(port.external_observation_consumed(&work, observation));
        assert_partial_foreign_feedback_is_one_use(&mut port, &work, observation);
        let execution = port.external_admissions[&work.input.question_digest]
            .execution
            .unwrap();
        assert!(matches!(
            port.engine
                .semantic
                .stores
                .tasks
                .get(execution)
                .unwrap()
                .status,
            TaskStatus::Failed
        ));
    }

    #[test]
    fn foreign_quantum_yield_is_partial_without_fabricated_ack_or_exact_resume() {
        let (mut port, work, config) = fixture();
        let mut receipt = dispatched(&mut port, &work, &config);
        let result = receipt.result.as_mut().unwrap();
        result.report.completion = ExternalCompletion::StoppedYield;
        receipt
            .attempt
            .as_mut()
            .unwrap()
            .external
            .as_mut()
            .unwrap()
            .partial_report = Some(result.report.clone());
        assert!(validate_cooperative_control(&receipt, receipt.result.as_ref().unwrap()).is_ok());
        let (observation, coverage) = port.commit_external_report(&work, &receipt).unwrap();
        assert!(!coverage);
        assert!(port.external_observation_consumed(&work, observation));
        assert_eq!(port.counters.consumed_external_checker_tasks, 0);
        assert_partial_foreign_feedback_is_one_use(&mut port, &work, observation);
        let execution = port.external_admissions[&work.input.question_digest]
            .execution
            .unwrap();
        assert_eq!(
            port.engine
                .semantic
                .stores
                .tasks
                .get(execution)
                .unwrap()
                .status,
            TaskStatus::Failed
        );
        let step = port.after_cpu(&work, observation, coverage).unwrap();
        let step = port.materialize_step(step).unwrap();
        assert!(
            matches!(step.transition, FrameTransition::Continue(ref next)
            if next.stage == RefinementStage::ProposalValueFresh)
        );
        assert_eq!(
            port.engine
                .semantic
                .stores
                .observations
                .get(observation)
                .unwrap()
                .external_report
                .as_ref()
                .unwrap()
                .completion,
            ExternalCompletion::StoppedYield
        );
        assert_eq!(port.counters.consumed_external_checker_tasks, 0);
    }

    #[test]
    fn foreign_old_go_and_native_credit_claim_fail_before_search_mutation() {
        let (mut port, work, config) = fixture();
        let receipt = dispatched(&mut port, &work, &config);
        let mut stale = receipt.clone();
        stale.scope.execution.sequence += 1;
        let before = port
            .engine
            .semantic
            .stores
            .observations
            .next_observation_sequence();
        assert!(port.commit_external_report(&work, &stale).is_err());
        assert_eq!(port.counters.external_checker_nodes_observed, 0);
        assert_eq!(
            port.engine
                .semantic
                .stores
                .observations
                .next_observation_sequence(),
            before
        );
        let mut false_native = receipt;
        false_native.result.as_mut().unwrap().strict_native_nodes = Some(8);
        assert!(port.commit_external_report(&work, &false_native).is_err());
        assert_eq!(port.counters.cpu_nodes, 0);
        assert_eq!(
            port.engine
                .semantic
                .stores
                .observations
                .next_observation_sequence(),
            before
        );
    }

    #[test]
    fn submit_rejection_refunds_once_but_ambiguous_physical_issue_does_not() {
        let (mut port, work, config) = fixture();
        let receipt = dispatched(&mut port, &work, &config);
        assert_eq!(port.counters.external_checker_node_budget_reserved, 16);
        port.external_rejected(&work, &receipt.job, &receipt.grant)
            .unwrap();
        assert_eq!(port.counters.external_checker_node_budget_reserved, 0);
        assert!(
            port.external_rejected(&work, &receipt.job, &receipt.grant)
                .is_err()
        );

        let (mut port, work, config) = fixture();
        let mut unknown = dispatched(&mut port, &work, &config);
        unknown.go_issue = ExternalGoIssue::Unknown;
        unknown.result = Err(CheckerError::External {
            stage: "write",
            code: "partial_issue",
        });
        assert!(port.commit_external_report(&work, &unknown).is_err());
        assert_eq!(port.counters.external_checker_node_budget_reserved, 16);
        assert!(port.external_admissions[&work.input.question_digest].quarantined);
        assert_eq!(port.counters.cpu_nodes, 0);
    }

    #[test]
    fn shutdown_observes_expired_closed_receipt_once_without_search_consumption() {
        let (mut port, work, config) = fixture();
        let receipt = dispatched(&mut port, &work, &config);
        port.closed.insert(work.frame);
        port.limits.deadline = Instant::now();
        port.observe_external_report(&work, &receipt).unwrap();
        let before = port
            .engine
            .semantic
            .stores
            .observations
            .next_observation_sequence();
        assert_eq!(port.counters.external_checker_nodes_observed, 8);
        assert_eq!(port.counters.consumed_external_checker_tasks, 0);
        assert!(port.external_action_committed(
            work.input.question_digest,
            receipt.job.id,
            receipt.grant.execution,
        ));
        assert!(!port.external_action_committed(
            work.input.question_digest,
            receipt.job.id,
            rz_contracts::ExecutionId::new(receipt.grant.execution.epoch, 78),
        ));
        port.observe_external_report(&work, &receipt).unwrap();
        assert_eq!(port.counters.external_checker_nodes_observed, 8);
        assert_eq!(
            port.engine
                .semantic
                .stores
                .observations
                .next_observation_sequence(),
            before,
        );
        assert!(
            port.engine.semantic.nodes[work.input.question.node]
                .evidence
                .is_none()
        );
    }

    #[test]
    fn shutdown_preserves_unknown_issue_and_never_refunds_or_marks_action_committed() {
        let (mut port, work, config) = fixture();
        let mut unknown = dispatched(&mut port, &work, &config);
        unknown.go_issue = ExternalGoIssue::Unknown;
        unknown.result = Err(CheckerError::External {
            stage: "write",
            code: "partial_issue",
        });
        port.closed.insert(work.frame);
        port.limits.deadline = Instant::now();
        assert!(port.observe_external_report(&work, &unknown).is_err());
        let retained = port.engine.researcher.external_attempts.len();
        assert!(port.observe_external_report(&work, &unknown).is_err());
        assert_eq!(port.engine.researcher.external_attempts.len(), retained);
        assert_eq!(port.counters.external_checker_node_budget_reserved, 16);
        assert_eq!(port.counters.external_checker_nodes_observed, 8);
        assert!(!port.external_action_committed(
            work.input.question_digest,
            unknown.job.id,
            unknown.grant.execution,
        ));
    }

    #[test]
    fn canceled_consumer_retains_completed_raw_without_repeating_terminal_task_mutation() {
        let (mut port, work, config) = fixture();
        let receipt = dispatched(&mut port, &work, &config);
        let admission = port.external_admissions[&work.input.question_digest].clone();
        port.engine
            .semantic
            .stores
            .tasks
            .cancel_consumer(admission.execution.unwrap(), admission.consumer.unwrap())
            .unwrap();
        let (observation, coverage) = port.commit_external_report(&work, &receipt).unwrap();
        assert!(coverage);
        assert!(!port.external_observation_consumed(&work, observation));
        let public_before = port.engine.semantic.records.len();
        let step = port.after_cpu(&work, observation, coverage).unwrap();
        assert!(matches!(
            step.transition,
            AdaptivePortTransition::Unknown { .. }
        ));
        assert_eq!(port.engine.semantic.records.len(), public_before);
        assert_eq!(port.counters.consumed_external_checker_tasks, 0);
        assert!(matches!(
            port.engine.semantic.stores.tasks.get(admission.execution.unwrap()).unwrap().status,
            TaskStatus::Completed(id) if id == observation,
        ));
        assert!(port.external_admissions[&work.input.question_digest].task_settled);
        port.observe_external_report(&work, &receipt).unwrap();
        assert_eq!(port.counters.external_checker_nodes_observed, 8);
    }

    #[test]
    fn foreign_unknown_work_can_request_fresh_without_completed_task_or_zero_work_claim() {
        let (mut port, work, config) = fixture();
        let mut receipt = dispatched(&mut port, &work, &config);
        let result = receipt.result.as_mut().unwrap();
        result.report.work.nodes = None;
        let attempt = receipt.attempt.as_mut().unwrap();
        attempt.work = result.report.work;
        attempt.external.as_mut().unwrap().partial_report = Some(result.report.clone());
        let (observation, coverage) = port.commit_external_report(&work, &receipt).unwrap();
        assert!(!coverage);
        assert!(port.external_observation_consumed(&work, observation));
        assert!(port.counters.external_checker_work_incomplete);
        assert_eq!(port.counters.cpu_nodes, 0);
        assert_eq!(port.counters.consumed_external_checker_tasks, 0);
        assert_eq!(port.counters.external_checker_node_budget_reserved, 16);
        assert_partial_foreign_feedback_is_one_use(&mut port, &work, observation);
        let step = port.after_cpu(&work, observation, coverage).unwrap();
        let step = port.materialize_step(step).unwrap();
        assert!(
            matches!(step.transition, FrameTransition::Continue(ref next)
            if next.stage == RefinementStage::ProposalValueFresh)
        );
    }

    #[test]
    fn foreign_canceled_stop_is_raw_only_even_before_deadline() {
        let (mut port, work, config) = fixture();
        let mut receipt = dispatched(&mut port, &work, &config);
        let result = receipt.result.as_mut().unwrap();
        result.report.completion = ExternalCompletion::StoppedCanceled;
        receipt
            .attempt
            .as_mut()
            .unwrap()
            .external
            .as_mut()
            .unwrap()
            .partial_report = Some(result.report.clone());
        let (observation, coverage) = port.commit_external_report(&work, &receipt).unwrap();
        assert!(!coverage);
        assert!(!port.external_observation_consumed(&work, observation));
        let step = port.after_cpu(&work, observation, coverage).unwrap();
        assert!(matches!(
            step.transition,
            AdaptivePortTransition::Unknown { .. }
        ));
    }

    #[test]
    fn revoked_or_expired_partial_consumer_cannot_create_followup() {
        for expired in [false, true] {
            let (mut port, work, config) = fixture();
            let mut receipt = dispatched(&mut port, &work, &config);
            let result = receipt.result.as_mut().unwrap();
            result.report.completion = ExternalCompletion::StoppedYield;
            receipt
                .attempt
                .as_mut()
                .unwrap()
                .external
                .as_mut()
                .unwrap()
                .partial_report = Some(result.report.clone());
            let admission = port.external_admissions[&work.input.question_digest].clone();
            if expired {
                port.limits.deadline = Instant::now();
            } else {
                port.engine
                    .semantic
                    .stores
                    .tasks
                    .cancel_consumer(admission.execution.unwrap(), admission.consumer.unwrap())
                    .unwrap();
            }
            let (observation, coverage) = port.commit_external_report(&work, &receipt).unwrap();
            assert!(!coverage);
            assert!(!port.external_observation_consumed(&work, observation));
            assert_eq!(port.counters.consumed_external_checker_tasks, 0);
            assert_eq!(port.counters.external_checker_node_budget_reserved, 16);
            assert_eq!(
                port.engine
                    .semantic
                    .stores
                    .tasks
                    .get(admission.execution.unwrap())
                    .unwrap()
                    .status,
                TaskStatus::Failed
            );
            assert!(port.engine.semantic.records.is_empty());
            if !expired {
                let step = port.after_cpu(&work, observation, coverage).unwrap();
                assert!(matches!(
                    step.transition,
                    AdaptivePortTransition::Unknown { .. }
                ));
            }
        }
    }

    #[test]
    fn foreign_consumption_seal_does_not_outlive_explicit_original_consumer_revocation() {
        for partial in [false, true] {
            let (mut port, work, config) = fixture();
            let mut receipt = dispatched(&mut port, &work, &config);
            if partial {
                let result = receipt.result.as_mut().unwrap();
                result.report.completion = ExternalCompletion::StoppedYield;
                receipt
                    .attempt
                    .as_mut()
                    .unwrap()
                    .external
                    .as_mut()
                    .unwrap()
                    .partial_report = Some(result.report.clone());
            }
            let (observation, coverage) = port.commit_external_report(&work, &receipt).unwrap();
            assert!(port.external_observation_consumed(&work, observation));
            let admission = port.external_admissions[&work.input.question_digest].clone();
            port.engine
                .semantic
                .stores
                .tasks
                .cancel_consumer(
                    admission.execution.unwrap(),
                    admission.original_consumer.unwrap().id,
                )
                .unwrap();
            assert!(!port.external_observation_consumed(&work, observation));
            let step = port.after_cpu(&work, observation, coverage).unwrap();
            assert!(matches!(
                step.transition,
                AdaptivePortTransition::Unknown { .. }
            ));
            assert!(port.engine.semantic.records.is_empty());
            // The already observed physical report and its private historical
            // consumption are retained; revocation creates no zero work.
            assert_eq!(port.counters.external_checker_nodes_observed, 8);
            assert_eq!(admission.consumed_observation, Some(observation));
        }
    }

    #[test]
    fn foreign_materialization_rechecks_original_consumer_after_after_cpu() {
        for partial in [false, true] {
            let (mut port, work, config) = fixture();
            let mut receipt = dispatched(&mut port, &work, &config);
            if partial {
                let result = receipt.result.as_mut().unwrap();
                result.report.completion = ExternalCompletion::StoppedYield;
                receipt
                    .attempt
                    .as_mut()
                    .unwrap()
                    .external
                    .as_mut()
                    .unwrap()
                    .partial_report = Some(result.report.clone());
            }
            let (observation, coverage) = port.commit_external_report(&work, &receipt).unwrap();
            let raw_sequence = port
                .engine
                .semantic
                .stores
                .observations
                .next_observation_sequence();
            let questions = port.next_question;
            let leases = port.engine.semantic.stores.review_pin_count().unwrap();
            let step = port.after_cpu(&work, observation, coverage).unwrap();
            assert!(matches!(
                step.transition,
                AdaptivePortTransition::Continue(_)
            ));
            assert!(
                port.external_feedback_still_authorized(work.input.question_digest, observation)
            );
            assert_eq!(port.next_question, questions);
            assert_eq!(
                port.engine.semantic.stores.review_pin_count().unwrap(),
                leases
            );

            let admission = &port.external_admissions[&work.input.question_digest];
            let execution = admission.execution.unwrap();
            let original = admission.original_consumer.unwrap();
            port.engine
                .semantic
                .stores
                .tasks
                .cancel_consumer(execution, original.id)
                .unwrap();
            assert!(
                !port.external_feedback_still_authorized(work.input.question_digest, observation)
            );
            assert!(matches!(
                port.materialize_step(step),
                Err(PalsError::Store(StoreError::StaleConsumer))
            ));
            assert_eq!(port.next_question, questions);
            assert_eq!(
                port.engine.semantic.stores.review_pin_count().unwrap(),
                leases
            );
            assert_eq!(
                port.engine
                    .semantic
                    .stores
                    .observations
                    .next_observation_sequence(),
                raw_sequence
            );
            assert_eq!(port.counters.external_checker_nodes_observed, 8);
            assert_eq!(
                port.external_admissions[&work.input.question_digest].consumed_observation,
                Some(observation)
            );
            assert!(
                port.engine
                    .semantic
                    .stores
                    .observations
                    .get(observation)
                    .is_ok()
            );
            assert!(port.engine.semantic.records.is_empty());
        }
    }

    #[test]
    fn foreign_recheck_validation_is_general_endpoint_question_without_discovery_exclusion() {
        let (mut port, mut work, config) = fixture();
        let first = BoardMove::from_uci("e2e4").unwrap();
        work.input = port
            .input(
                work.input.question.node,
                RefinementStage::RecheckCounterValidation,
                Some(first),
                vec![first],
                vec![first],
                Some(vec![first]),
                Some(vec![first]),
                Some(1),
                true,
                None,
                None,
            )
            .unwrap();
        let question = port
            .external_question(&work, &config, ExternalSlicePolicy::default())
            .unwrap();
        assert!(question.root_moves.is_none());
        assert_eq!(
            port.external_admissions[&work.input.question_digest]
                .task_key
                .question,
            TaskQuestion::AnalyzePosition
        );
    }

    #[test]
    fn consume_counter_overflow_can_be_settled_by_shutdown_without_duplicate_publication() {
        let (mut port, work, config) = fixture();
        let receipt = dispatched(&mut port, &work, &config);
        port.counters.consumed_external_checker_tasks = u64::MAX;
        assert!(matches!(
            port.commit_external_report(&work, &receipt),
            Err(PalsError::Capacity),
        ));
        let admission = &port.external_admissions[&work.input.question_digest];
        assert!(admission.published.is_some());
        assert!(!admission.task_settled);
        let before = port
            .engine
            .semantic
            .stores
            .observations
            .next_observation_sequence();
        port.observe_external_report(&work, &receipt).unwrap();
        assert!(port.external_admissions[&work.input.question_digest].task_settled);
        assert_eq!(port.counters.consumed_external_checker_tasks, u64::MAX);
        assert_eq!(port.counters.external_checker_nodes_observed, 8);
        assert_eq!(
            port.engine
                .semantic
                .stores
                .observations
                .next_observation_sequence(),
            before
        );
    }

    #[test]
    fn physical_receipt_counter_overflow_does_not_partially_advance_go_ownership() {
        let (mut port, work, config) = fixture();
        let receipt = dispatched(&mut port, &work, &config);
        port.counters.external_checker_reports = u64::MAX;
        for observed in [false, true] {
            let result = if observed {
                port.observe_external_report(&work, &receipt)
            } else {
                port.commit_external_report(&work, &receipt).map(|_| ())
            };
            assert!(matches!(result, Err(PalsError::Capacity)));
            assert!(port.engine.researcher.external_attempts.is_empty());
            assert_eq!(port.engine.researcher.last_external_attempt, None);
            let admission = &port.external_admissions[&work.input.question_digest];
            assert!(!admission.physical_accounted);
            assert!(admission.received.is_none());
            assert_eq!(port.counters.external_checker_node_budget_reserved, 16);
        }
        // Only the source fixture changes the forced counter boundary; the
        // product never clears actual work to make a capacity failure pass.
        port.counters.external_checker_reports = 0;
        port.observe_external_report(&work, &receipt).unwrap();
        assert_eq!(port.engine.researcher.external_attempts.len(), 1);
        assert_eq!(port.counters.external_checker_reports, 1);
        assert_eq!(port.counters.external_checker_nodes_observed, 8);
    }

    #[test]
    fn explicit_foreign_root_mask_preserves_order_and_rejects_illegal_or_duplicate_moves() {
        let (_, mut work, _) = fixture();
        let roots = vec![
            BoardMove::from_uci("c7c5").unwrap(),
            BoardMove::from_uci("e7e5").unwrap(),
        ];
        Arc::make_mut(&mut work.input.question).validator_root_moves = Some(roots.clone());
        // The explicit action is not interpreted as the historical exclusion.
        Arc::make_mut(&mut work.input.question).discovery_restricted = true;
        Arc::make_mut(&mut work.input.question).stage = RefinementStage::CpuDiscovery;
        assert_eq!(
            foreign_root_restriction(&work.input.question).unwrap(),
            Some(roots.clone()),
        );
        Arc::make_mut(&mut work.input.question).validator_root_moves =
            Some(vec![roots[0], roots[0]]);
        assert!(foreign_root_restriction(&work.input.question).is_err());
        Arc::make_mut(&mut work.input.question).validator_root_moves =
            Some(vec![BoardMove::from_uci("e2e3").unwrap()]);
        assert!(foreign_root_restriction(&work.input.question).is_err());
        Arc::make_mut(&mut work.input.question).validator_root_moves = Some(Vec::new());
        assert!(foreign_root_restriction(&work.input.question).is_err());
    }

    #[test]
    fn explicit_non_discovery_action_uses_root_restricted_task_and_frozen_query() {
        let (mut port, mut work, config) = fixture();
        let roots = vec![
            BoardMove::from_uci("c7c5").unwrap(),
            BoardMove::from_uci("e7e5").unwrap(),
        ];
        Arc::make_mut(&mut work.input.question).validator_root_moves = Some(roots.clone());
        let receipt = dispatched(&mut port, &work, &config);
        let admission = &port.external_admissions[&work.input.question_digest];
        assert_eq!(admission.task_key.question, TaskQuestion::AnalyzeRootMoves);
        assert_eq!(admission.restriction, Some(roots));
        // Replacing a private stage/query with a fresh Arc cannot substitute
        // this original submission while echoing its semantic digest.
        let mut changed = work.clone();
        Arc::make_mut(&mut changed.input.question).recheck = true;
        assert!(port.observe_external_report(&changed, &receipt).is_err());
        assert_eq!(port.counters.external_checker_nodes_observed, 0);
        let (_, coverage) = port.commit_external_report(&work, &receipt).unwrap();
        assert!(coverage);
    }

    #[test]
    fn foreign_pending_task_survives_independent_conclusion_revisions_with_original_consumer() {
        let (mut port, work, config) = fixture();
        let receipt = dispatched(&mut port, &work, &config);
        let admission = port.external_admissions[&work.input.question_digest].clone();
        for situation in [admission.root, admission.situation] {
            let current = port
                .engine
                .semantic
                .stores
                .situations
                .get(situation)
                .unwrap()
                .clone();
            // Independent metadata fixture uses the ordinary hot conclusion
            // writer. It changes neither Rules state nor submitted CPU query.
            let evidence = port
                .engine
                .semantic
                .stores
                .hot_append_observation(Observation {
                    state: current.state,
                    line: Some(current.focus),
                    source: 42,
                    epoch: 0,
                    value_identity: None,
                    checker_identity: None,
                    checker_work: None,
                    external_report: None,
                    model_value_identity: None,
                    model_value_input: None,
                    cpu_condition: None,
                    cpu_pv: None,
                    scope: EvidenceScope::Model {
                        model: 999,
                        encoding: 0,
                        input: 0,
                    },
                    score: RawScore::Unknown,
                    budget: 0,
                    kind: ObservationKind::Refutation,
                    supersedes: None,
                    execution: None,
                })
                .unwrap();
            port.engine
                .semantic
                .stores
                .hot_refute_continuation(situation, current.focus, evidence)
                .unwrap();
        }
        assert!(
            port.engine
                .semantic
                .stores
                .situations
                .get(admission.root)
                .unwrap()
                .revision
                > admission.root_revision
        );
        assert!(
            port.engine
                .semantic
                .stores
                .situations
                .get(admission.situation)
                .unwrap()
                .revision
                > admission.situation_revision
        );
        let (observation, coverage) = port.commit_external_report(&work, &receipt).unwrap();
        assert!(coverage);
        assert_eq!(port.counters.consumed_external_checker_tasks, 1);
        assert!(port.external_observation_consumed(&work, observation));
        let mut replaced = work.clone();
        // Equal value/digest is still not the original private question seal.
        let _ = Arc::make_mut(&mut replaced.input.question);
        assert!(!port.external_observation_consumed(&replaced, observation));
        let retained = &port.external_admissions[&work.input.question_digest];
        assert_eq!(retained.original_consumer, admission.original_consumer);
        assert_eq!(retained.context, admission.context);
        assert_eq!(retained.view_seal, admission.view_seal);
    }

    #[test]
    fn external_retained_host_accounting_counts_backing_and_keeps_map_unknown() {
        let (mut port, work, config) = fixture();
        assert_eq!(port.external_retained_host_bytes(), Some(0));
        let _ = dispatched(&mut port, &work, &config);
        assert_eq!(
            port.external_retained_host_bytes(),
            port.external_admissions[&work.input.question_digest].retained_host_bytes(),
        );
        let mut admission = port.external_admissions[&work.input.question_digest].clone();
        let before = admission.retained_host_bytes().unwrap();
        assert!(
            before
                >= u64::try_from(std::mem::size_of::<ExternalPortAdmission>()).unwrap()
                    + admission.question.owned_host_bytes().unwrap()
                    + admission.view.owned_host_bytes().unwrap()
        );
        admission
            .config
            .expected_uci_name
            .as_mut()
            .unwrap()
            .reserve_exact(512);
        assert!(admission.retained_host_bytes().unwrap() > before);
        admission
            .config
            .identity
            .options
            .insert("Threads".into(), "1".into());
        assert!(admission.retained_host_bytes().is_none());
        port.external_admissions
            .insert(work.input.question_digest, admission);
        assert!(port.external_retained_host_bytes().is_none());
    }
}
