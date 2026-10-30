//! Manager-only admission and publication of worker-owned native CPU slices.
//! No CPU search, neural evaluation, encoding or archive recovery occurs here.
//! Store execution IDs and physical grant IDs are different namespaces; every
//! admitted grant keeps an explicit mapping until its one-use receipt arrives.

use super::*;
use crate::cpu::{CpuCompletion, CpuWork};
use crate::cpu_worker::{
    CPU_JOB_CAPACITY, CpuAdmissionRejection, CpuJobTermination, CpuSliceReport, CpuWorkerProfile,
    cpu_root_restriction_digest,
};
use rz_contracts::pals_adaptive::{CpuJobEnvelope, CpuJobId, SliceGrant};

impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// Read-only authorization for an independent PublishProgress control.
    /// The worker keeps running and owns completed-iteration acknowledgement;
    /// no credit, result or new consumer is issued by this query.
    pub fn cpu_progress_request_eligible(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        job: &CpuJobEnvelope,
        grant: &SliceGrant,
    ) -> bool {
        self.cpu_admissions
            .get(&work.input.question_digest)
            .is_some_and(|admission| {
                self.check_work(work).is_ok()
                    && self.cpu_context_current(work, admission)
                    && !admission.terminal
                    && !admission.quarantined
                    && admission.job.as_ref() == Some(job)
                    && admission.grant.as_ref() == Some(grant)
                    && admission.owner == Some(grant.owner)
                    && admission
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
            })
    }

    /// An adaptive S-policy may feed a completed iteration to the existing
    /// P/C path while the original physical CPU question keeps its token.
    /// This is not requested-depth coverage or a conditional refutation.
    pub fn published_cpu_continuation_eligible(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        progress: &crate::cpu_worker::CpuWorkerProgress,
    ) -> bool {
        let Some(admission) = self.cpu_admissions.get(&work.input.question_digest) else {
            return false;
        };
        let (Some(job), Some(grant), Some(best)) =
            (admission.job, admission.grant, progress.iteration.best_move)
        else {
            return false;
        };
        let query = &work.input.question;
        let proposal = if query.proposal.is_empty() {
            &query.context.prefix
        } else {
            &query.proposal
        };
        self.check_work(work).is_ok()
            && self.cpu_context_current(work, admission)
            && admission
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
            && work.budget.role_submissions_remaining > 0
            && !admission.terminal
            && !admission.quarantined
            && query.stage == RefinementStage::CpuValidation
            && self.engine.semantic.nodes[query.node].terminal.is_none()
            && self
                .divergence_indices(proposal)
                .is_ok_and(|indices| !indices.is_empty())
            && job.id == progress.job
            && job.question == work.input.question_digest
            && grant.job == job.id
            && grant.execution == progress.execution
            && grant.owner == progress.owner
            && admission.owner == Some(progress.owner)
            && progress.iteration.completed_depth > 0
            && progress.iteration.completed_depth <= admission.target_depth
            && progress.iteration.nodes <= grant.node_credit
            && admission.position.legal_moves().contains(&best)
            && admission
                .restriction
                .as_ref()
                .is_none_or(|moves| moves.contains(&best))
    }

    pub fn yielded_cpu_continuation_eligible(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        receipt: &CpuSliceReport,
    ) -> bool {
        let Ok(actual) = &receipt.result else {
            return false;
        };
        if receipt.terminal
            || receipt.admission_rejected
            || !receipt.continuation_retained
            || actual.report.completion != CpuCompletion::Yielded
            || actual.report.score_scope != crate::cpu::CpuScoreScope::CompletedIteration
            || receipt.actual_work.is_none()
        {
            return false;
        }
        let progress = crate::cpu_worker::CpuWorkerProgress {
            job: receipt.grant.job,
            execution: receipt.grant.execution,
            owner: receipt.grant.owner,
            sequence: receipt.progress_sequence,
            iteration: crate::cpu::CpuIterationProgress {
                best_move: actual.report.best_move,
                score: actual.report.score,
                completed_depth: actual.report.completed_depth,
                nodes: actual.report.nodes,
            },
        };
        self.published_cpu_continuation_eligible(work, &progress)
    }

    /// Only a worker-issued proof of definitive pre-enqueue rejection permits
    /// rollback. An unknown worker response never reaches this path.
    pub fn reject_cpu_admission(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        job: &CpuJobEnvelope,
        proof: &CpuAdmissionRejection,
    ) -> Result<(), PalsError> {
        let grant = proof.grant();
        let digest = work.input.question_digest;
        let admission = self
            .cpu_admissions
            .get(&digest)
            .ok_or(StoreError::InvalidConditions(
                "unadmitted CPU grant has no admission",
            ))?
            .clone();
        let execution = admission.execution.ok_or(StoreError::InvalidConditions(
            "unadmitted CPU grant has no Store execution",
        ))?;
        if !proof.is_verified_unqueued()
            || admission.quarantined
            || admission.terminal
            || admission.grant != Some(grant)
            || admission.job.as_ref() != Some(job)
            || job.question != digest
            || grant.job != job.id
            || admission.owner != Some(grant.owner)
            || work.frame != admission.frame
            || work.frame.authority != job.id.authority
            || job.id.authority != self.authority
            || work.original_deadline != job.hard_deadline
            || work.original_first_move != admission.first_move
            || work.input.stage != admission.stage
            || work.input.question.stage != admission.stage
            || work.input.question.question_digest != digest
            || work.input.question.context != admission.context
            || work.input.view_seal != admission.view_seal
            || self.seal(&work.input.view) != admission.view_seal
            || work.input.question.node != admission.node
            || !work
                .input
                .question
                .position
                .snapshot()
                .same_state(&admission.position.snapshot())
            || self.engine.semantic.stores.tasks.get(execution)?.key != admission.task_key
            || self.engine.semantic.stores.tasks.get(execution)?.status != TaskStatus::InFlight
        {
            return Err(StoreError::InvalidConditions(
                "unadmitted CPU proof differs from original grant",
            )
            .into());
        }
        if admission.continuation_retained {
            self.engine
                .semantic
                .stores
                .pause_task(execution, job.id.sequence, None)?;
            self.cpu_admissions
                .get_mut(&digest)
                .expect("checked rejection")
                .grant = None;
        } else {
            self.engine.semantic.stores.tasks.fail(execution)?;
            if let Some(consumer) = admission.consumer {
                self.engine
                    .semantic
                    .stores
                    .tasks
                    .cancel_consumer(execution, consumer)?;
            }
            self.cpu_admissions.remove(&digest);
        }
        Ok(())
    }

    /// Latest settled raw observation for this exact question, original job
    /// and physical execution. Reservations and diagnostic zero-work reports
    /// cannot qualify as an actually committed V action outcome.
    pub(super) fn cpu_action_committed(
        &self,
        question: ContractDigest,
        job: CpuJobId,
        execution: rz_contracts::ExecutionId,
    ) -> bool {
        let Some(admission) = self.cpu_admissions.get(&question) else {
            return false;
        };
        if admission.quarantined
            || admission.grant.is_some()
            || admission
                .job
                .is_none_or(|original| original.id != job || original.question != question)
            || execution.epoch != job.authority.epoch
        {
            return false;
        }
        let Some((actual, stored_execution, observation)) = admission.committed_execution else {
            return false;
        };
        actual == execution
            && self
                .engine
                .semantic
                .stores
                .observations
                .get(observation)
                .is_ok_and(|raw| {
                    raw.kind == ObservationKind::CpuAnalysis
                        && raw.execution == Some(stored_execution)
                        && raw.state == admission.task_key.state
                        && raw.value_identity.as_ref() == Some(&admission.value_identity)
                        && raw.cpu_condition == admission.task_key.cpu_condition
                        && self
                            .engine
                            .semantic
                            .stores
                            .tasks
                            .get(stored_execution)
                            .is_ok_and(|task| task.key == admission.task_key)
                })
    }

    /// A settled raw observation is not automatically accepted feedback. Only
    /// the original live consumer and actually consumed receipt may advance a
    /// frame; deadline/cancel/stale raw-only reports leave this seal empty.
    pub(super) fn cpu_observation_consumed(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        observation: ObservationId,
    ) -> bool {
        let Some(admission) = self.cpu_admissions.get(&work.input.question_digest) else {
            return false;
        };
        self.cpu_context_current(work, admission)
            && self.cpu_feedback_receipt_current(work.input.question_digest, admission, observation)
    }

    /// An opaque CPU-origin continuation gate retains only its original
    /// question and actually consumed observation. Recheck that existing
    /// admission at materialization without cloning Work or another view lease.
    pub(super) fn cpu_feedback_still_authorized(
        &self,
        question: ContractDigest,
        observation: ObservationId,
    ) -> bool {
        let Some(admission) = self.cpu_admissions.get(&question) else {
            return false;
        };
        self.cpu_admission_current(admission)
            && self.cpu_feedback_receipt_current(question, admission, observation)
    }

    /// A fixed deferred recheck retains its original accepted CPU feedback
    /// after that exact parent closes. Only the port-issued queued closure
    /// permits this exception; ordinary closed-frame feedback stays invalid.
    pub(super) fn cpu_deferred_feedback_still_authorized(
        &self,
        parent: crate::pals::coordinator::FrameId,
        question: ContractDigest,
        observation: ObservationId,
    ) -> bool {
        let Some(first) = self.queued_recheck_closures.get(&(parent, question)) else {
            return false;
        };
        let Some(admission) = self.cpu_admissions.get(&question) else {
            return false;
        };
        let Ok(first) = rz_contracts::Move::try_from(*first) else {
            return false;
        };
        admission.frame == parent
            && admission.first_move == rz_contracts::pals::Move16::encode(first)
            && self.closed.contains(&parent)
            && self.cpu_admission_identity_current(admission)
            && self.cpu_feedback_receipt_current(question, admission, observation)
    }

    fn cpu_feedback_receipt_current(
        &self,
        question: ContractDigest,
        admission: &CpuPortAdmission,
        observation: ObservationId,
    ) -> bool {
        if admission.quarantined
            || admission.grant.is_some()
            || admission.consumed_observation != Some(observation)
            || admission.job.is_none_or(|job| {
                job.id.authority != self.authority
                    || job.question != question
                    || job.hard_deadline != self.deadline
            })
        {
            return false;
        }
        admission
            .committed_execution
            .is_some_and(|(physical, execution, actual)| {
                actual == observation
                    && admission.execution == Some(execution)
                    && admission
                        .settled_receipt
                        .is_some_and(|(settled, stored, _)| {
                            settled == physical && stored == execution
                        })
                    && admission.original_consumer.is_some_and(|original| {
                        self.engine
                            .semantic
                            .stores
                            .hot_validate_consumed_cpu_feedback(
                                execution,
                                original,
                                &admission.task_key,
                                &admission.position.snapshot(),
                                observation,
                                self.engine.tick_at(Instant::now()),
                            )
                            .is_ok_and(|accepted| accepted == observation)
                    })
                    && self
                        .engine
                        .semantic
                        .stores
                        .observations
                        .get(observation)
                        .is_ok_and(|raw| {
                            raw.execution == Some(execution)
                                && raw.state == admission.task_key.state
                                && raw.value_identity.as_ref() == Some(&admission.value_identity)
                                && raw.cpu_condition == admission.task_key.cpu_condition
                        })
            })
    }

    /// A concrete worker attests that its paused stack was discarded. Every
    /// preceding slice must already have known, settled usage. This cleanup is
    /// legal after the deadline/root admission closes and never refunds work.
    pub fn retire_cpu_token(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        termination: &CpuJobTermination,
    ) -> Result<(), PalsError> {
        let digest = work.input.question_digest;
        let admission = self
            .cpu_admissions
            .get(&digest)
            .ok_or(StoreError::InvalidConditions("CPU token has no admission"))?
            .clone();
        let job = admission.job.ok_or(StoreError::InvalidConditions(
            "CPU token has no admitted job",
        ))?;
        let execution = admission.execution.ok_or(StoreError::InvalidConditions(
            "CPU token has no Store execution",
        ))?;
        let controls = termination.controls;
        if admission.quarantined
            || admission.terminal
            || admission.grant.is_some()
            || !admission.continuation_retained
            || job.id != termination.job
            || admission.owner != Some(termination.owner)
            || termination.last_execution.epoch != job.id.authority.epoch
            || termination.last_execution.sequence != admission.last_started_sequence
            || job.id.authority != self.authority
            || work.frame != admission.frame
            || work.frame.authority != job.id.authority
            || work.original_deadline != job.hard_deadline
            || work.original_first_move != admission.first_move
            || work.input.stage != admission.stage
            || work.input.question.stage != admission.stage
            || work.input.question.question_digest != digest
            || work.input.question.context != admission.context
            || work.input.question.node != admission.node
            || work.input.view_seal != admission.view_seal
            || self.seal(&work.input.view) != admission.view_seal
            || !work
                .input
                .question
                .position
                .snapshot()
                .same_state(&admission.position.snapshot())
            || self.engine.semantic.game_generation != Some(job.id.authority.game.0)
            || admission.cumulative_nodes > admission.credit
            || termination.retained_bytes != 0
            || !controls.terminal
            || controls.cancel_ack > controls.cancel_sequence
            || controls.yield_ack > controls.yield_sequence
            || controls.publish_ack > controls.publish_sequence
            || !matches!(
                termination.completion,
                CpuCompletion::Canceled | CpuCompletion::Deadline
            )
            || (termination.completion == CpuCompletion::Deadline
                && Instant::now() < self.limits.deadline)
            || (termination.completion == CpuCompletion::Canceled
                && !termination.shutdown
                && (controls.cancel_sequence == 0
                    || controls.cancel_ack != controls.cancel_sequence))
        {
            return Err(StoreError::InvalidConditions(
                "CPU token discard differs from its actual owner or settled job",
            )
            .into());
        }
        // Current root/revision/deadline acceptance does not authorize disposal
        // of another job. Verify the original immutable Store facts instead.
        let root = self.engine.semantic.stores.situations.get(admission.root)?;
        let task = self.engine.semantic.stores.tasks.get(execution)?;
        if !self
            .engine
            .semantic
            .stores
            .states
            .get(root.state)?
            .same_state(&admission.root_snapshot)
            || task.key != admission.task_key
            || !self
                .engine
                .semantic
                .stores
                .states
                .get(task.key.state)?
                .same_state(&admission.position.snapshot())
            || !matches!(task.status, TaskStatus::Paused { checkpoint, .. } if checkpoint == job.id.sequence)
        {
            return Err(StoreError::InvalidConditions(
                "CPU token discard original Store identity changed",
            )
            .into());
        }
        self.engine.semantic.stores.hot_retire_paused(execution)?;
        let stored = self
            .cpu_admissions
            .get_mut(&digest)
            .expect("checked CPU token admission");
        stored.continuation_retained = false;
        stored.terminal = true;
        Ok(())
    }

    /// Failure-only metadata: semantic work and declarations are not the
    /// manager's physical spent/reserved ledger. No Rules/PV/view body is copied.
    pub fn cpu_question_failure_detail(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        error: &PalsError,
    ) -> String {
        let checker_max_depth = self
            .engine
            .owned_descriptor()
            .ok()
            .map(|descriptor| descriptor.config.max_depth);
        format!(
            "CPU question stage={:?}; original_cpu_limit={}; semantic_cpu_nodes={}; per_task_cpu_limit={}; target_depth={}; checker_max_depth={checker_max_depth:?}; source_error={error}; frame_sequence={}; question={:?}; view={:?}",
            work.input.stage,
            self.limits.max_cpu_nodes,
            self.counters.cpu_nodes,
            self.engine.config.cpu_nodes_per_task,
            self.limits.cpu_depth,
            work.frame.sequence,
            work.input.question_digest,
            work.input.view_seal.view_digest,
        )
    }

    /// Semantic ledger only. Physical settlement is owned and checked by the
    /// coordinator before this pair may authorize a normal root budget end.
    pub fn semantic_cpu_budget(&self) -> (u64, u64) {
        (self.limits.max_cpu_nodes, self.counters.cpu_nodes)
    }

    /// Prepare immutable conditions without claiming that a worker accepted or
    /// executed anything. The ledger remains the owner of outstanding credit.
    pub fn cpu_question(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        profiles: &[CpuWorkerProfile],
    ) -> Result<AdaptivePortCpuQuestion, PalsError> {
        self.prepare_cpu_question(work, profiles, false)?
            .ok_or(PalsError::InvalidLimits)
    }

    /// None is a validated native-budget boundary, not a CPU report or frame
    /// completion. The manager still owns the physical ledger and root drain.
    /// Logical admission checks precede this boundary so exhausted work cannot
    /// hide stale/depth/namespace/overrun failures.
    pub fn prepare_cpu_question(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        profiles: &[CpuWorkerProfile],
        physical_budget_exhausted: bool,
    ) -> Result<Option<AdaptivePortCpuQuestion>, PalsError> {
        self.check_work(work)?;
        let question = &work.input.question;
        if !is_cpu_stage(question.stage) || self.closed.contains(&work.frame) {
            return Err(StoreError::InvalidConditions("not an open CPU frame").into());
        }
        let installed = self.installed_cpu_binding(work)?;
        if (self.engine.owned_identity()?.semantics == crate::cpu_value::nnuec::v2::SEMANTICS
            && installed.is_none())
            || installed.is_some_and(|profile| profiles != [profile])
        {
            return Err(StoreError::InvalidConditions(
                "CPU question has no exact installed NNUEC owner",
            )
            .into());
        }
        let node = &self.engine.semantic.nodes[question.node];
        if node.terminal.is_some()
            || node.state != question.context.state
            || node.situation != question.context.situation
            || !node
                .position
                .snapshot()
                .same_state(&question.position.snapshot())
        {
            return Err(
                StoreError::InvalidConditions("CPU question differs from Rules node").into(),
            );
        }
        if let Some(admission) = self
            .cpu_admissions
            .get(&work.input.question_digest)
            .cloned()
        {
            if admission.terminal
                || admission.quarantined
                || admission.job.is_some()
                || !self.cpu_context_current(work, &admission)
                || profiles != admission.profiles.as_slice()
            {
                return Err(
                    StoreError::InvalidConditions("CPU question cannot be replaced").into(),
                );
            }
            if self.counters.cpu_nodes > self.limits.max_cpu_nodes {
                return Err(PalsError::InvalidLimits);
            }
            if physical_budget_exhausted || self.counters.cpu_nodes == self.limits.max_cpu_nodes {
                return Ok(None);
            }
            let position = self.issue_rules_position(admission.position.clone())?;
            let identity = position
                .export()
                .map_err(|_| RoleError::InvalidOutput)?
                .snapshot()
                .identity();
            self.cpu_admissions
                .get_mut(&work.input.question_digest)
                .expect("prepared CPU question")
                .rules_identity = identity;
            return Ok(Some(AdaptivePortCpuQuestion {
                position,
                root_moves: admission.restriction.clone(),
                target_depth: admission.target_depth,
                total_native_node_credit: admission.credit,
            }));
        }
        // Retired questions are not reusable cache entries. Removing metadata
        // never removes Store evidence or a still-retained worker continuation.
        self.cpu_admissions.retain(|_, value| !value.terminal);
        if self.cpu_admissions.len() >= CPU_JOB_CAPACITY
            || profiles.is_empty()
            || profiles.len() > CPU_JOB_CAPACITY
        {
            return Err(PalsError::Capacity);
        }
        let descriptor = self.engine.owned_descriptor()?;
        let target_depth = if let Some(horizon) = self.original_goal_horizon() {
            let remaining = horizon
                .checked_sub(question.context.prefix.len())
                .filter(|remaining| *remaining != 0)
                .ok_or(crate::pals::rovent_frontier::FrontierError::DepthHorizon)?;
            self.limits
                .cpu_depth
                .min(u16::try_from(remaining).map_err(|_| PalsError::InvalidLimits)?)
        } else {
            self.limits.cpu_depth
        };
        if target_depth > descriptor.config.max_depth {
            return Err(PalsError::InvalidLimits);
        }
        let value_identity = self.engine.owned_identity()?.clone();
        let (legacy_namespace, search_conditions) = self.cpu_namespace_digests()?;
        let evaluation_namespace =
            installed.map_or(legacy_namespace, |profile| profile.evaluation_namespace);
        let mut owners = HashSet::new();
        for profile in profiles {
            if !owners.insert(profile.owner)
                || profile.max_depth != descriptor.config.max_depth
                || profile.evaluation_namespace != evaluation_namespace
                || profile.search_conditions != search_conditions
            {
                return Err(StoreError::InvalidConditions(
                    "CPU worker profile differs from registered owner",
                )
                .into());
            }
        }
        let remaining = self
            .limits
            .max_cpu_nodes
            .checked_sub(self.counters.cpu_nodes)
            .ok_or(PalsError::InvalidLimits)?;
        let credit = remaining.min(self.engine.config.cpu_nodes_per_task);
        if self.engine.config.cpu_nodes_per_task == 0 {
            return Err(PalsError::InvalidLimits);
        }
        let restriction = if let Some(moves) = &question.validator_root_moves {
            let legal = question.position.legal_moves();
            let mut unique = HashSet::new();
            if question.stage != RefinementStage::CpuDiscovery
                || moves.is_empty()
                || moves.len() > 256
                || moves
                    .iter()
                    .any(|movement| !legal.contains(movement) || !unique.insert(*movement))
            {
                return Err(CpuError::InvalidRootMoves("invalid validator action roots").into());
            }
            Some(moves.clone())
        } else if question.stage == RefinementStage::CpuDiscovery && question.discovery_restricted {
            let attack = question.attack_ply.ok_or(RoleError::InvalidOutput)?;
            let old = question
                .proposal
                .get(attack)
                .copied()
                .ok_or(RoleError::InvalidOutput)?;
            if question.context.prefix != question.proposal[..attack] {
                return Err(RoleError::InvalidOutput.into());
            }
            let alternatives: Vec<_> = question
                .position
                .legal_moves()
                .into_iter()
                .filter(|movement| *movement != old)
                .collect();
            if alternatives.is_empty() {
                return Err(
                    CpuError::InvalidRootMoves("no restricted discovery alternative").into(),
                );
            }
            Some(alternatives)
        } else {
            None
        };
        let task_question = if question.stage == RefinementStage::CpuDiscovery {
            if restriction.is_some() {
                TaskQuestion::AnalyzeRootMoves
            } else {
                TaskQuestion::FindAlternative {
                    divergence_ply: u16::try_from(question.context.prefix.len())
                        .map_err(|_| PalsError::InvalidLimits)?,
                }
            }
        } else {
            TaskQuestion::AnalyzePosition
        };
        let root_moves: Vec<_> = restriction
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .copied()
            .map(Move16::pack)
            .collect::<Result<_, _>>()?;
        let ordered_mask: String = root_moves
            .iter()
            .map(|movement| format!("{:04x}", movement.bits()))
            .collect();
        let base_condition = self.engine.cpu_condition();
        // Keep the original calculation namespace. Logical question/view/job
        // authority belongs to the private admission and physical dispatch
        // handle, not the CPU implementation's exact evaluation condition.
        let task_condition = if task_question == TaskQuestion::AnalyzePosition {
            base_condition.clone()
        } else {
            format!(
                "{};question={:?};ordered-root-mask={};prefix-plies={}",
                base_condition,
                task_question,
                ordered_mask,
                question.context.prefix.len(),
            )
        };
        let task_condition = if installed.is_some() {
            let binding: String = evaluation_namespace
                .0
                .iter()
                .map(|value| format!("{value:02x}"))
                .collect();
            format!("{task_condition};nnuec-context-binding={binding}")
        } else {
            task_condition
        };
        if physical_budget_exhausted || remaining == 0 {
            return Ok(None);
        }
        let profile = stable_id(descriptor.config.profile.identity());
        let condition = stable_id(&format!(
            "{};q={}",
            descriptor.search_identity, descriptor.config.quiescence_ply
        ));
        let root = self
            .engine
            .semantic
            .stores
            .root()
            .ok_or(StoreError::StaleConsumer)?;
        let root_record = self.engine.semantic.stores.situations.get(root)?;
        let situation = node.situation;
        let situation_revision = self
            .engine
            .semantic
            .stores
            .situations
            .get(situation)?
            .revision;
        let position = question.position.clone();
        let rules_position = self.issue_rules_position(position.clone())?;
        let rules_identity = rules_position
            .export()
            .map_err(|_| RoleError::InvalidOutput)?
            .snapshot()
            .identity();
        let result = AdaptivePortCpuQuestion {
            position: rules_position,
            root_moves: restriction.clone(),
            target_depth,
            total_native_node_credit: credit,
        };
        self.cpu_admissions.insert(
            work.input.question_digest,
            CpuPortAdmission {
                node: question.node,
                frame: work.frame,
                stage: work.input.stage,
                view_seal: work.input.view_seal,
                context: question.context.clone(),
                first_move: work.original_first_move,
                execution: None,
                dispatch: None,
                consumer: None,
                original_consumer: None,
                target_depth,
                credit,
                restriction,
                generation: self.engine.semantic.stores.generation(),
                root,
                root_revision: root_record.revision,
                root_snapshot: self
                    .engine
                    .semantic
                    .stores
                    .states
                    .get(root_record.state)?
                    .clone(),
                position,
                rules_identity,
                situation,
                situation_revision,
                task_key: TaskKey {
                    state: node.state,
                    line: None,
                    question: task_question,
                    root_moves,
                    value_identity: Some(value_identity.clone()),
                    checker_identity: None,
                    cpu_condition: Some(task_condition),
                    model: 0,
                    epoch: 0,
                    profile,
                    condition,
                    input_revision: if task_question == TaskQuestion::AnalyzePosition {
                        0
                    } else {
                        question.context.prefix.len() as u64
                    },
                    requested_depth: target_depth,
                    node_budget: credit,
                },
                value_identity,
                base_condition,
                profiles: profiles.to_vec(),
                evaluation_namespace,
                search_conditions,
                job: None,
                grant: None,
                owner: None,
                cumulative_nodes: 0,
                progress_sequence: 0,
                control_sequence: 0,
                execution_sequence: 0,
                last_started_sequence: 0,
                committed_execution: None,
                consumed_observation: None,
                settled_receipt: None,
                continuation_retained: false,
                terminal: false,
                quarantined: false,
            },
        );
        Ok(Some(result))
    }

    /// Call only after ledger reservation and before dispatch. Store admission
    /// is hot-only; a capacity error must be handled by the archive worker.
    pub fn cpu_admitted(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        job: &CpuJobEnvelope,
        grant: &SliceGrant,
    ) -> Result<(), PalsError> {
        self.check_work(work)?;
        let admission = self
            .cpu_admissions
            .get(&work.input.question_digest)
            .ok_or(StoreError::InvalidConditions(
                "CPU question was not prepared",
            ))?
            .clone();
        let evaluation_namespace = admission.evaluation_namespace;
        let search_conditions = admission.search_conditions;
        if self.engine.owned_identity()?.semantics == crate::cpu_value::nnuec::v2::SEMANTICS
            && self.installed_cpu_binding(work)?.is_none_or(|profile| {
                admission.profiles.as_slice() != [profile]
                    || profile.evaluation_namespace != evaluation_namespace
                    || profile.search_conditions != search_conditions
            })
        {
            return Err(StoreError::InvalidConditions(
                "CPU grant lost original NNUEC installation",
            )
            .into());
        }
        if admission.terminal
            || admission.quarantined
            || admission.grant.is_some()
            || !self.cpu_context_current(work, &admission)
            || job.id.sequence == 0
            || job.id.authority != self.authority
            || job.question != work.input.question_digest
            || job.state_identity != admission.rules_identity
            || job.root_restriction != cpu_root_restriction_digest(admission.restriction.as_deref())
            || job.evaluation_namespace != evaluation_namespace
            || job.search_conditions != search_conditions
            || job.total_native_node_credit != admission.credit
            || job.hard_deadline != self.deadline
            || admission.job.is_some_and(|old| old != *job)
            || grant.job != job.id
            || grant.execution.epoch != self.authority.epoch
            || grant.execution.sequence == 0
            || grant.execution.sequence <= admission.execution_sequence
            || grant.control_sequence == 0
            || grant.control_sequence <= admission.control_sequence
            || grant.node_credit == 0
            || grant.soft_lease.is_zero()
            || grant.soft_lease
                > self
                    .limits
                    .deadline
                    .saturating_duration_since(Instant::now())
            || admission
                .cumulative_nodes
                .checked_add(grant.node_credit)
                .is_none_or(|sum| sum > admission.credit)
            || admission.owner.is_some_and(|owner| owner != grant.owner)
            || !admission.profiles.iter().any(|profile| {
                profile.owner == grant.owner
                    && profile.max_depth >= admission.target_depth
                    && profile.evaluation_namespace == evaluation_namespace
                    && profile.search_conditions == search_conditions
            })
        {
            return Err(StoreError::InvalidConditions(
                "CPU grant differs from immutable question or owner",
            )
            .into());
        }
        self.engine.semantic.consumer_id = self
            .engine
            .semantic
            .consumer_id
            .checked_add(1)
            .ok_or(PalsError::Capacity)?;
        let consumer = self.engine.semantic.consumer_id;
        let deadline_tick = self.engine.tick_at(self.limits.deadline);
        let now_tick = self.engine.tick_at(Instant::now());
        let next_consumer = TaskConsumer {
            id: consumer,
            situation: admission.situation,
            revision: admission.situation_revision,
            generation: admission.generation,
            deadline_tick,
        };
        let (store_admission, dispatch) = if admission.continuation_retained {
            let previous = admission.execution.ok_or(StoreError::InvalidConditions(
                "CPU resume has no original Store execution",
            ))?;
            let original = admission
                .original_consumer
                .ok_or(StoreError::InvalidConditions(
                    "CPU resume has no original consumer authority",
                ))?;
            if original.situation != next_consumer.situation
                || original.revision != next_consumer.revision
                || original.generation != next_consumer.generation
                || original.deadline_tick != next_consumer.deadline_tick
            {
                return Err(StoreError::StaleConsumer.into());
            }
            let dispatch = admission
                .dispatch
                .as_ref()
                .ok_or(StoreError::InvalidConditions(
                    "CPU resume has no original dispatch custody",
                ))?;
            if !dispatch.identifies(previous, original) {
                return Err(StoreError::InvalidConditions(
                    "CPU resume dispatch differs from original execution and consumer",
                )
                .into());
            }
            self.engine.semantic.stores.hot_resume_cpu_instance(
                dispatch,
                admission.task_key.clone(),
                consumer,
                &admission.position.snapshot(),
                now_tick,
            )?
        } else {
            self.engine.semantic.stores.hot_request_cpu_instance(
                admission.task_key.clone(),
                next_consumer,
                now_tick,
            )?
        };
        let execution = match store_admission {
            TaskAdmission::Start(execution) if !admission.continuation_retained => execution,
            TaskAdmission::Resume {
                execution,
                previous,
                checkpoint,
            } if admission.continuation_retained
                && admission.execution == Some(previous)
                && checkpoint == job.id.sequence
                && admission.owner == Some(grant.owner) =>
            {
                self.engine.semantic.stores.hot_retire_paused(previous)?;
                execution
            }
            TaskAdmission::Join(execution) | TaskAdmission::Reuse { execution, .. } => {
                self.engine
                    .semantic
                    .stores
                    .tasks
                    .cancel_consumer(execution, consumer)?;
                return Err(StoreError::InvalidConditions(
                    "CPU physical dispatch cannot join or reuse a different execution",
                )
                .into());
            }
            TaskAdmission::Start(execution) | TaskAdmission::Resume { execution, .. } => {
                self.engine.semantic.stores.tasks.fail(execution)?;
                return Err(StoreError::InvalidConditions(
                    "CPU resume checkpoint differs from its actual owner",
                )
                .into());
            }
        };
        let stored = self
            .cpu_admissions
            .get_mut(&work.input.question_digest)
            .expect("prepared CPU admission");
        stored.execution = Some(execution);
        stored.dispatch = Some(dispatch);
        stored.consumer = Some(consumer);
        stored.original_consumer = Some(next_consumer);
        stored.job = Some(*job);
        stored.grant = Some(*grant);
        stored.consumed_observation = None;
        stored.owner = Some(grant.owner);
        stored.control_sequence = grant.control_sequence;
        stored.execution_sequence = grant.execution.sequence;
        self.counters.cpu_tasks_requested = self
            .counters
            .cpu_tasks_requested
            .checked_add(1)
            .ok_or(PalsError::Capacity)?;
        Ok(())
    }

    /// Accepts physical usage separately from logical search authority. Late
    /// receipts retain raw evidence but cannot update the current node value.
    /// The returned bool describes completed requested-depth coverage only.
    pub fn commit_cpu_report(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        receipt: &CpuSliceReport,
    ) -> Result<(ObservationId, bool), PalsError> {
        self.commit_cpu_report_inner(work, receipt, true)
    }

    /// Shutdown settlement is distinct from successful search consumption.
    /// An identical already-settled receipt is idempotent; unknown usage and
    /// physical quarantine remain errors rather than being called complete.
    pub fn observe_cpu_report(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        receipt: &CpuSliceReport,
    ) -> Result<(), PalsError> {
        if self.cpu_receipt_settled(work, receipt) {
            return Ok(());
        }
        match self.commit_cpu_report_inner(work, receipt, false) {
            Ok(_) => Ok(()),
            Err(_) if self.cpu_receipt_settled(work, receipt) => Ok(()),
            Err(error) => Err(error),
        }
    }

    fn cpu_receipt_settled(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        receipt: &CpuSliceReport,
    ) -> bool {
        let Some(admission) = self.cpu_admissions.get(&work.input.question_digest) else {
            return false;
        };
        let Some((actual, stored_execution, seal)) = admission.settled_receipt else {
            return false;
        };
        let Some(job) = admission.job else {
            return false;
        };
        !admission.quarantined
            && admission.grant.is_none()
            && actual == receipt.grant.execution
            && seal == cpu_receipt_digest(receipt)
            && receipt.grant.job == job.id
            && admission
                .owner
                .is_none_or(|owner| receipt.grant.owner == owner)
            && work.frame == admission.frame
            && work.frame.authority == job.id.authority
            && work.original_deadline == job.hard_deadline
            && work.original_first_move == admission.first_move
            && work.input.stage == admission.stage
            && work.input.question.stage == admission.stage
            && work.input.question.question_digest == work.input.question_digest
            && work.input.question.context == admission.context
            && work.input.question.node == admission.node
            && work.input.view_seal == admission.view_seal
            && self.seal(&work.input.view) == admission.view_seal
            && work
                .input
                .question
                .position
                .snapshot()
                .same_state(&admission.position.snapshot())
            && self
                .engine
                .semantic
                .stores
                .tasks
                .get(stored_execution)
                .is_ok_and(|task| {
                    task.key == admission.task_key
                        && matches!(
                            task.status,
                            TaskStatus::Completed(_)
                                | TaskStatus::Paused { .. }
                                | TaskStatus::RetiredPaused { .. }
                                | TaskStatus::Failed
                        )
                })
    }

    /// Settle a real no-value terminal without publishing an ObservationId,
    /// score, PV, feedback, coverage or a successful task. The returned seal is
    /// the original receipt equality seal already retained by this admission.
    pub fn observe_cpu_no_value_termination(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        receipt: &CpuSliceReport,
    ) -> Result<ContractDigest, PalsError> {
        let soft_stop = receipt
            .result
            .as_ref()
            .err()
            .and_then(crate::cpu::CpuError::value_authority_stop);
        if soft_stop.is_none()
            || receipt.actual_work.is_none()
            || !receipt.terminal
            || receipt.continuation_retained
            || receipt.admission_rejected
            || receipt.reciprocal_request.is_some()
            || receipt.controls.is_none_or(|controls| {
                !controls.terminal
                    || controls.cancel_ack > controls.cancel_sequence
                    || controls.yield_ack > controls.yield_sequence
                    || controls.publish_ack > controls.publish_sequence
            })
        {
            return Err(StoreError::InvalidEvidence(
                "CPU no-value termination requires an original known soft revocation",
            )
            .into());
        }
        // This exact observer performs original question/Store/grant validation
        // and charges the known delta once, including when admission is revoked.
        // Its Err result is retained in the receipt, never replaced by a value.
        self.observe_cpu_report(work, receipt)?;
        let admission = self.cpu_admissions.get(&work.input.question_digest).ok_or(
            StoreError::InvalidConditions("CPU no-value admission missing"),
        )?;
        let (_, execution, seal) = admission
            .settled_receipt
            .ok_or(StoreError::InvalidEvidence(
                "CPU no-value original receipt was not settled",
            ))?;
        if !admission.terminal
            || admission.quarantined
            || admission.continuation_retained
            || admission.grant.is_some()
            || !self.cpu_receipt_settled(work, receipt)
            || self.engine.semantic.stores.tasks.get(execution)?.status != TaskStatus::Failed
        {
            return Err(StoreError::InvalidEvidence(
                "CPU no-value termination did not retain its original failed task",
            )
            .into());
        }
        Ok(seal)
    }

    fn commit_cpu_report_inner(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        receipt: &CpuSliceReport,
        consume_current: bool,
    ) -> Result<(ObservationId, bool), PalsError> {
        // V5 never issued a reciprocal scope. Keep an owned question/token
        // outside the legacy Yield/terminal codec instead of retiring it as
        // ordinary completed work. Rovent's ticket consumer owns this receipt.
        if receipt.reciprocal_request.is_some()
            || receipt
                .result
                .as_ref()
                .is_ok_and(|actual| actual.report.completion == CpuCompletion::AwaitReciprocal)
        {
            return Err(StoreError::InvalidConditions(
                "reciprocal CPU wait requires the Rovent question consumer",
            )
            .into());
        }
        let digest = work.input.question_digest;
        let admission = self
            .cpu_admissions
            .get(&digest)
            .ok_or(StoreError::InvalidConditions(
                "CPU receipt has no admission",
            ))?
            .clone();
        let job = admission.job.ok_or(StoreError::InvalidConditions(
            "CPU receipt was not dispatched",
        ))?;
        let execution = admission.execution.ok_or(StoreError::InvalidConditions(
            "CPU receipt has no Store execution",
        ))?;
        if admission.quarantined
            || admission.terminal
            || admission.grant.as_ref() != Some(&receipt.grant)
            || admission.owner != Some(receipt.grant.owner)
            || receipt.grant.job != job.id
            || work.frame.authority != job.id.authority
            || work.original_deadline != job.hard_deadline
            || work.input.question.question_digest != digest
            || work.input.question.stage != work.input.stage
            || work.frame != admission.frame
            || work.input.stage != admission.stage
            || work.original_first_move != admission.first_move
            || work.input.question.context != admission.context
            || work.input.view_seal != admission.view_seal
            || self.seal(&work.input.view) != admission.view_seal
            || work.input.question.node != admission.node
            || !work
                .input
                .question
                .position
                .snapshot()
                .same_state(&admission.position.snapshot())
            || self.engine.semantic.stores.tasks.get(execution)?.key != admission.task_key
        {
            return Err(
                StoreError::InvalidConditions("duplicate or mismatched CPU slice receipt").into(),
            );
        }
        let Some(actual) = receipt.actual_work else {
            self.counters.cpu_work_observation_incomplete = true;
            if receipt.result.is_ok() {
                self.counters.cpu_tasks = self
                    .counters
                    .cpu_tasks
                    .checked_add(1)
                    .ok_or(PalsError::Capacity)?;
            }
            self.cpu_admissions
                .get_mut(&digest)
                .expect("checked CPU admission")
                .quarantined = true;
            return Err(StoreError::InvalidEvidence(
                "CPU usage unknown; outstanding task is quarantined",
            )
            .into());
        };
        let expected_total = admission.cumulative_nodes.checked_add(actual.nodes);
        let invalid_result = match &receipt.result {
            Ok(result) => {
                result.report.nodes != actual.nodes
                    || result.report.quiescence_nodes != actual.quiescence_nodes
                    || result.report.tt_hits != actual.tt_hits
                    || result.report.resume.is_some()
                    || (result.report.completion == CpuCompletion::Yielded
                        && (receipt.terminal || !receipt.continuation_retained))
                    || (result.report.completion != CpuCompletion::Yielded
                        && (!receipt.terminal || receipt.continuation_retained))
            }
            Err(_) => !receipt.admission_rejected && !receipt.terminal,
        };
        if actual.nodes > receipt.grant.node_credit
            || actual.quiescence_nodes > actual.nodes
            || expected_total.is_none_or(|sum| sum > admission.credit)
            || (!receipt.admission_rejected && expected_total != Some(receipt.cumulative_nodes))
            || invalid_result
            || (receipt.terminal && receipt.continuation_retained)
            || receipt.progress_sequence < admission.progress_sequence
            || (receipt.admission_rejected
                && (actual != CpuWork::default() || receipt.result.is_ok() || receipt.terminal))
        {
            self.counters.cpu_work_observation_incomplete = true;
            if receipt.result.is_ok() {
                self.counters.cpu_tasks = self
                    .counters
                    .cpu_tasks
                    .checked_add(1)
                    .ok_or(PalsError::Capacity)?;
            }
            self.cpu_admissions
                .get_mut(&digest)
                .expect("checked CPU admission")
                .quarantined = true;
            return Err(StoreError::InvalidEvidence(
                "CPU usage or completion violates its reserved slice",
            )
            .into());
        }
        let stored = self
            .cpu_admissions
            .get_mut(&digest)
            .expect("checked CPU admission");
        stored.grant = None;
        stored.settled_receipt = Some((
            receipt.grant.execution,
            execution,
            cpu_receipt_digest(receipt),
        ));
        if !receipt.admission_rejected {
            stored.last_started_sequence = receipt.grant.execution.sequence;
            stored.cumulative_nodes = expected_total.expect("checked cumulative native usage");
            stored.progress_sequence = receipt.progress_sequence;
            stored.continuation_retained = receipt.continuation_retained;
            stored.terminal = receipt.terminal;
        }
        let controlled = match &receipt.result {
            Ok(report) => report,
            Err(error) => {
                self.account_cpu_failure(actual)?;
                if receipt.admission_rejected && admission.continuation_retained {
                    self.engine
                        .semantic
                        .stores
                        .pause_task(execution, job.id.sequence, None)?;
                } else {
                    self.engine.semantic.stores.tasks.fail(execution)?;
                    if receipt.admission_rejected {
                        let stored = self
                            .cpu_admissions
                            .get_mut(&digest)
                            .expect("checked admission");
                        stored.execution = None;
                        stored.consumer = None;
                        stored.owner = None;
                    }
                }
                return Err(error.clone().into());
            }
        };
        let report = &controlled.report;
        let cpu_limits = CpuLimits {
            max_depth: admission.target_depth,
            max_nodes: receipt.grant.node_credit,
            deadline: Some(self.limits.deadline),
        };
        let validation =
            if report.pv.is_empty() && work.input.stage == RefinementStage::CpuDiscovery {
                self.validate_empty_discovery(&admission, report, cpu_limits)
            } else {
                self.engine.validate_cpu_report(
                    &admission.position,
                    report,
                    &admission.value_identity,
                    &admission.base_condition,
                    admission.restriction.is_some(),
                    cpu_limits,
                )
            }
            .and_then(|()| {
                if admission.restriction.as_ref().is_some_and(|moves| {
                    report.best_move.is_some_and(|best| !moves.contains(&best))
                }) {
                    return Err(StoreError::InvalidEvidence(
                        "CPU report violates its ordered root restriction",
                    )
                    .into());
                }
                Ok(())
            });
        if let Err(error) = validation {
            self.account_cpu_failure(actual)?;
            self.counters.cpu_tasks = self
                .counters
                .cpu_tasks
                .checked_add(1)
                .ok_or(PalsError::Capacity)?;
            self.engine.semantic.stores.tasks.fail(execution)?;
            self.cpu_admissions
                .get_mut(&digest)
                .expect("checked admission")
                .quarantined = true;
            return Err(error);
        }
        if self
            .counters
            .cpu_nodes
            .checked_add(actual.nodes)
            .is_none_or(|total| total > self.limits.max_cpu_nodes)
        {
            self.account_cpu_failure(actual)?;
            self.counters.cpu_tasks = self
                .counters
                .cpu_tasks
                .checked_add(1)
                .ok_or(PalsError::Capacity)?;
            self.engine.semantic.stores.tasks.fail(execution)?;
            self.cpu_admissions
                .get_mut(&digest)
                .expect("checked admission")
                .quarantined = true;
            return Err(
                StoreError::InvalidEvidence("CPU receipt exceeds global native credit").into(),
            );
        }
        let complete = PalsEngine::<M>::observe_cpu_report(
            &mut self.counters,
            report,
            admission.target_depth,
            true,
        );
        let projected: Vec<_> = report
            .pv
            .iter()
            .copied()
            .take(
                self.continuation_limit()
                    .saturating_sub(work.input.question.context.prefix.len()),
            )
            .collect();
        let observation = self.publish_cpu_observation(&admission, report, execution, &projected);
        let observation = match observation {
            Ok(observation) => observation,
            Err(error) => {
                self.engine.semantic.stores.tasks.fail(execution)?;
                self.cpu_admissions
                    .get_mut(&digest)
                    .expect("checked admission")
                    .quarantined = true;
                return Err(error);
            }
        };
        self.cpu_admissions
            .get_mut(&digest)
            .expect("checked CPU admission")
            .committed_execution = Some((receipt.grant.execution, execution, observation));
        // A retained physical stack remains paused even if an unusual yield
        // report already covers target_depth. Coverage cannot release its owner.
        let aborted = matches!(
            report.completion,
            CpuCompletion::Canceled | CpuCompletion::Deadline
        );
        if receipt.continuation_retained {
            self.engine.semantic.stores.pause_task(
                execution,
                job.id.sequence,
                Some(observation),
            )?;
        } else if complete && !aborted {
            self.engine
                .semantic
                .stores
                .complete_task(execution, observation)?;
        } else {
            self.engine.semantic.stores.tasks.fail(execution)?;
        }
        if consume_current && !aborted && self.cpu_context_current(work, &admission) {
            let prepared_context = self.prepare_cpu_context_seal(work, &admission)?;
            let now_tick = self.engine.tick_at(Instant::now());
            let original_consumer =
                admission
                    .original_consumer
                    .ok_or(StoreError::InvalidConditions(
                        "CPU original consumer missing",
                    ))?;
            let accepted = if receipt.continuation_retained {
                self.engine
                    .semantic
                    .stores
                    .hot_validate_sealed_paused_cpu_feedback(
                        execution,
                        original_consumer,
                        &admission.task_key,
                        &admission.position.snapshot(),
                        observation,
                        now_tick,
                    )
            } else if complete {
                self.engine.semantic.stores.hot_consume_sealed_cpu_task(
                    execution,
                    original_consumer,
                    &admission.task_key,
                    &admission.position.snapshot(),
                    now_tick,
                )
            } else {
                self.engine.semantic.stores.hot_consume_sealed_cpu_feedback(
                    execution,
                    original_consumer,
                    &admission.task_key,
                    &admission.position.snapshot(),
                    observation,
                    now_tick,
                )
            };
            match accepted {
                Ok(actual) if actual == observation => {}
                Ok(_) => {
                    return Err(
                        StoreError::InvalidEvidence("CPU consumed another observation").into(),
                    );
                }
                Err(
                    StoreError::ExpiredConsumer
                    | StoreError::StaleConsumer
                    | StoreError::CancelledConsumer,
                ) => return Ok((observation, complete)),
                Err(error) => return Err(error.into()),
            }
            // This remains true if a subsequent public projection append fails;
            // the returned error still blocks the semantic next-stage commit.
            self.cpu_admissions
                .get_mut(&digest)
                .expect("checked CPU consumer")
                .consumed_observation = Some(observation);
            if complete && !receipt.continuation_retained {
                self.counters.consumed_cpu_tasks += 1;
            } else if report.score_scope == CpuScoreScope::CompletedIteration {
                self.counters.consumed_partial_cpu_values += 1;
            } else if report.score_scope == CpuScoreScope::FrontierOnly {
                self.counters.consumed_frontier_cpu_values += 1;
            }
            // Discovery estimates describe the divergence question, including
            // unrestricted discovery. They never enter generic node evidence.
            if work.input.stage != RefinementStage::CpuDiscovery {
                let context = self.consumed_cpu_context_seal(
                    work,
                    &admission,
                    prepared_context,
                    execution,
                    observation,
                )?;
                self.engine.semantic.nodes[admission.node].evidence = Some(CpuEvidence {
                    score: report.score,
                    depth: report.completed_depth,
                    scope: report.score_scope,
                    value_identity: report.value_identity.clone(),
                    provenance: Some((observation, execution)),
                    context,
                });
            }
            if work.input.stage != RefinementStage::CpuDiscovery {
                self.record_cpu_verification_hot(
                    &admission.context.prefix,
                    observation,
                    report.score,
                    report.completed_depth,
                    report.score_scope,
                    admission.position.side_to_move(),
                )?;
            }
        }
        Ok((observation, complete))
    }

    pub(super) fn cpu_namespace_digests(
        &self,
    ) -> Result<(ContractDigest, ContractDigest), PalsError> {
        let bytes = serde_json::to_vec(self.engine.owned_identity()?)
            .map_err(|_| PalsError::InvalidConfig)?;
        let mut value = Sha256::new();
        value.update(b"rz-cpu-evaluation-namespace/1");
        value.update(bytes);
        let mut conditions = Sha256::new();
        conditions.update(b"rz-cpu-search-conditions/1");
        conditions.update(self.engine.owned_descriptor()?.search_conditions.as_bytes());
        Ok((
            ContractDigest(value.finalize().into()),
            ContractDigest(conditions.finalize().into()),
        ))
    }

    pub(super) fn cpu_context_current(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        admission: &CpuPortAdmission,
    ) -> bool {
        self.cpu_admission_current(admission)
            && work.frame == admission.frame
            && work.input.stage == admission.stage
            && work.input.question.context == admission.context
            && work.original_first_move == admission.first_move
            && work.input.view_seal == admission.view_seal
            && self.seal(&work.input.view) == admission.view_seal
            && work.frame.authority == self.authority
            && work.original_deadline == self.deadline
    }

    fn cpu_admission_current(&self, admission: &CpuPortAdmission) -> bool {
        !self.closed.contains(&admission.frame) && self.cpu_admission_identity_current(admission)
    }

    /// Immutable admission authority shared with the narrowly registered
    /// deferred recheck path. Frame openness is checked by each caller.
    fn cpu_admission_identity_current(&self, admission: &CpuPortAdmission) -> bool {
        self.check_live().is_ok()
            && admission.frame.authority == self.authority
            && admission.view_seal.authority == self.authority
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
                    root.revision >= admission.root_revision
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

    pub(super) fn account_cpu_failure(&mut self, work: CpuWork) -> Result<(), PalsError> {
        let nodes = self
            .counters
            .cpu_nodes
            .checked_add(work.nodes)
            .ok_or(PalsError::Capacity)?;
        let qnodes = self
            .counters
            .cpu_quiescence_nodes
            .checked_add(work.quiescence_nodes)
            .ok_or(PalsError::Capacity)?;
        let hits = self
            .counters
            .cpu_tt_hits
            .checked_add(work.tt_hits)
            .ok_or(PalsError::Capacity)?;
        self.counters.cpu_nodes = nodes;
        self.counters.cpu_quiescence_nodes = qnodes;
        self.counters.cpu_tt_hits = hits;
        Ok(())
    }

    fn publish_cpu_observation(
        &mut self,
        admission: &CpuPortAdmission,
        report: &CpuReport,
        execution: ExecutionId,
        projected: &[BoardMove],
    ) -> Result<ObservationId, PalsError> {
        if !self
            .engine
            .semantic
            .stores
            .states
            .get(admission.task_key.state)?
            .same_state(&admission.position.snapshot())
        {
            return Err(StoreError::InvalidEvidence("CPU observation anchor changed").into());
        }
        let cpu_pv = if projected.is_empty() {
            None
        } else {
            Some(
                self.engine
                    .semantic
                    .stores
                    .hot_append_cpu_pv(&admission.position, projected)?,
            )
        };
        let observation = self
            .engine
            .semantic
            .stores
            .hot_append_observation(Observation {
                state: admission.task_key.state,
                line: None,
                source: stable_id(report.score_provenance),
                epoch: 0,
                value_identity: Some(report.value_identity.clone()),
                checker_identity: None,
                checker_work: None,
                external_report: None,
                model_value_identity: None,
                model_value_input: None,
                cpu_condition: admission.task_key.cpu_condition.clone(),
                cpu_pv,
                scope: EvidenceScope::DepthLimited {
                    depth: report.completed_depth,
                    profile: admission.task_key.profile,
                    condition: admission.task_key.condition,
                },
                score: if report.score_scope == CpuScoreScope::FrontierOnly {
                    RawScore::Estimate {
                        value: report.score as f32,
                        perspective: admission.position.side_to_move(),
                    }
                } else {
                    RawScore::Cpu {
                        value: report.score,
                        perspective: admission.position.side_to_move(),
                        bound: BoundKind::ExactWithinSearch,
                    }
                },
                budget: report.nodes,
                kind: ObservationKind::CpuAnalysis,
                supersedes: None,
                execution: Some(execution),
            })?;
        self.engine
            .semantic
            .stores
            .hot_add_dependency(observation, admission.situation)?;
        self.engine
            .semantic
            .stores
            .hot_add_dependency(observation, admission.root)?;
        Ok(observation)
    }

    /// Discovery may yield before publishing a first PV. Retain that strict
    /// frontier-only report without inventing a move or certifying coverage.
    fn validate_empty_discovery(
        &self,
        admission: &CpuPortAdmission,
        report: &CpuReport,
        limits: CpuLimits,
    ) -> Result<(), PalsError> {
        let descriptor = self.engine.owned_descriptor()?;
        if report.value_identity != admission.value_identity
            || self.engine.owned_identity()? != &admission.value_identity
            || self.engine.cpu_condition() != admission.base_condition
            || report.search_version != descriptor.search_identity
            || report.profile != descriptor.config.profile
            || report.root_restricted != admission.restriction.is_some()
            || report.best_move.is_some()
            || !report.pv.is_empty()
            || report.completed_depth != 0
            || report.reused_completed_depth != 0
            || report.score_scope != CpuScoreScope::FrontierOnly
            || report.resume.is_some()
            || report.nodes > limits.max_nodes
            || report.quiescence_nodes > report.nodes
            || report.score.unsigned_abs() > crate::cpu::CPU_FRONTIER_SCORE_LIMIT as u32
        {
            return Err(StoreError::InvalidEvidence(
                "empty Discovery is not a valid immutable frontier report",
            )
            .into());
        }
        Ok(())
    }
}

fn is_cpu_stage(stage: RefinementStage) -> bool {
    matches!(
        stage,
        RefinementStage::CpuValidation
            | RefinementStage::CpuDiscovery
            | RefinementStage::CounterValidation
            | RefinementStage::RepairValidation
            | RefinementStage::RecheckCounterValidation
    )
}

/// Internal complete-receipt equality for retries, not a physical or
/// performance attestation. Debug is streamed into SHA-256 without cloning
/// the PV or allocating a serialized report; seals never leave this process.
pub(super) fn cpu_receipt_digest(receipt: &CpuSliceReport) -> ContractDigest {
    struct DigestWriter(Sha256);
    impl std::fmt::Write for DigestWriter {
        fn write_str(&mut self, value: &str) -> std::fmt::Result {
            self.0.update(value.as_bytes());
            Ok(())
        }
    }
    use std::fmt::Write as _;
    let mut hash = Sha256::new();
    hash.update(b"rz-pals-in-process-CPU-receipt-equality/1\0");
    let mut writer = DigestWriter(hash);
    write!(&mut writer, "{receipt:?}").expect("digest writer cannot fail");
    ContractDigest(writer.0.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pals::coordinator::FrameId;
    use crate::pals::review_context::{ReviewIndexLimits, ReviewSelectionPolicy};
    use rz_contracts::pals_adaptive::{CpuJobId, CpuWorkerId};
    use rz_contracts::{ClockDomain, GameGeneration, MonotonicTick, ProcessEpoch, RootGeneration};

    type Port = AdaptiveEnginePort<LegalOrderRoleMock>;
    type Work = CoordinatorWork<ReviewView, AdaptivePortQuestion>;

    /// Receipts below are contract fixtures, not physical search or performance
    /// evidence. The test only exercises manager hot admission and acceptance.
    fn port_and_work(stage: RefinementStage, target_depth: u16) -> (Port, Work, CpuWorkerProfile) {
        let cpu = CpuEngine::new(crate::cpu::CpuConfig {
            tt_entries: 16,
            max_depth: 8,
            ..crate::cpu::CpuConfig::default()
        })
        .unwrap();
        let engine = PalsEngine::new(PalsConfig::default(), LegalOrderRoleMock, cpu)
            .unwrap()
            .with_selected_review_context(
                ReviewSelectionPolicy::default(),
                ReviewIndexLimits::default(),
            )
            .unwrap();
        let authority = SearchAuthority {
            epoch: ProcessEpoch(23),
            game: GameGeneration(0),
            root: RootGeneration(1),
            implementation: ContractDigest([7; 32]),
        };
        let deadline = Deadline {
            clock: ClockDomain(authority.epoch),
            at: MonotonicTick(20_000_000_000),
        };
        let limits = PalsLimits {
            deadline: Instant::now() + Duration::from_secs(20),
            max_rounds: 1,
            max_cpu_nodes: 2000,
            cpu_depth: target_depth,
        };
        let owner_sequence = Arc::new(std::sync::atomic::AtomicU64::new(1));
        let rules_owner_issuer: AdaptiveRulesOwnerIssuer =
            Arc::new(move || Ok(OwnerId(owner_sequence.fetch_add(1, Ordering::Relaxed))));
        let mut port = Port::new(
            engine,
            Position::startpos(),
            authority,
            deadline,
            limits,
            rules_owner_issuer,
        )
        .unwrap();
        let first = BoardMove::from_uci("e2e4").unwrap();
        let node = port.connect_hot(port.root, first).unwrap();
        let proposal = if stage == RefinementStage::CpuDiscovery {
            vec![first, BoardMove::from_uci("e7e5").unwrap()]
        } else {
            vec![first]
        };
        let input = port
            .input(
                node,
                stage,
                Some(first),
                vec![first],
                proposal,
                None,
                None,
                (stage == RefinementStage::CpuDiscovery).then_some(1),
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
                original_store_byte_limit: 8 * 1024 * 1024,
            },
            input,
        };
        let (evaluation_namespace, search_conditions) = port.cpu_namespace_digests().unwrap();
        let profile = CpuWorkerProfile {
            owner: CpuWorkerId(1),
            max_depth: 8,
            evaluation_namespace,
            search_conditions,
        };
        (port, work, profile)
    }

    #[test]
    fn cpu_question_failure_detail_preserves_exhausted_credit_and_original_error() {
        let (mut port, work, profile) = port_and_work(RefinementStage::CpuValidation, 2);
        port.counters.cpu_nodes = port.limits.max_cpu_nodes;
        let before = port.counters();
        let Err(error) = port.cpu_question(&work, &[profile]) else {
            panic!("exhausted credit must retain the original failure");
        };
        assert!(matches!(error, PalsError::InvalidLimits));
        let detail = port.cpu_question_failure_detail(&work, &error);
        assert!(detail.contains("stage=CpuValidation"));
        assert!(detail.contains("original_cpu_limit=2000; semantic_cpu_nodes=2000"));
        assert!(detail.contains("target_depth=2; checker_max_depth=Some(8)"));
        assert!(detail.contains("source_error=PALS: InvalidLimits"));
        assert_eq!(port.counters(), before);
        assert!(port.cpu_admissions.is_empty());
    }

    #[test]
    fn cpu_question_failure_detail_distinguishes_depth_from_credit() {
        let (mut port, work, profile) = port_and_work(RefinementStage::CpuValidation, 9);
        let Err(error) = port.cpu_question(&work, &[profile]) else {
            panic!("unsupported depth must retain the original failure");
        };
        assert!(matches!(error, PalsError::InvalidLimits));
        let detail = port.cpu_question_failure_detail(&work, &error);
        assert!(detail.contains("original_cpu_limit=2000; semantic_cpu_nodes=0"));
        assert!(detail.contains("target_depth=9; checker_max_depth=Some(8)"));
        assert!(detail.contains("source_error=PALS: InvalidLimits"));
        assert!(port.cpu_admissions.is_empty());
    }

    #[test]
    fn independent_publish_progress_gate_preserves_original_pending_consumer() {
        let (mut port, work, profile) = port_and_work(RefinementStage::CpuValidation, 2);
        let (job, grant) = first_grant(&mut port, &work, profile);
        assert!(port.cpu_progress_request_eligible(&work, &job, &grant));
        let mut different_owner = grant;
        different_owner.owner = CpuWorkerId(grant.owner.0 + 1);
        assert!(!port.cpu_progress_request_eligible(&work, &job, &different_owner));
        let mut different_execution = grant;
        different_execution.execution.sequence += 1;
        assert!(!port.cpu_progress_request_eligible(&work, &job, &different_execution));
        let mut different_root = work.clone();
        different_root.frame.authority.root.0 += 1;
        assert!(!port.cpu_progress_request_eligible(&different_root, &job, &grant));
        let original = port.cpu_admissions[&work.input.question_digest].clone();
        let execution = original.execution.unwrap();
        let consumer = original.original_consumer.unwrap();
        assert_eq!(
            port.engine
                .semantic
                .stores
                .tasks
                .get(execution)
                .unwrap()
                .status,
            TaskStatus::InFlight
        );
        port.engine
            .semantic
            .stores
            .tasks
            .cancel_consumer(execution, consumer.id)
            .unwrap();
        assert!(!port.cpu_progress_request_eligible(&work, &job, &grant));
        assert_eq!(port.counters.consumed_cpu_tasks, 0);
        assert_eq!(port.counters.consumed_partial_cpu_values, 0);
        assert!(
            port.cpu_admissions[&work.input.question_digest]
                .consumed_observation
                .is_none()
        );
    }

    fn first_grant(
        port: &mut Port,
        work: &Work,
        profile: CpuWorkerProfile,
    ) -> (CpuJobEnvelope, SliceGrant) {
        let prepared = port.cpu_question(work, &[profile]).unwrap();
        let job = CpuJobEnvelope {
            id: CpuJobId {
                authority: port.authority,
                sequence: 1,
            },
            state_identity: prepared.position.export().unwrap().snapshot().identity(),
            question: work.input.question_digest,
            root_restriction: cpu_root_restriction_digest(prepared.root_moves.as_deref()),
            evaluation_namespace: profile.evaluation_namespace,
            search_conditions: profile.search_conditions,
            total_native_node_credit: prepared.total_native_node_credit,
            hard_deadline: port.deadline,
        };
        let grant = SliceGrant {
            job: job.id,
            execution: rz_contracts::ExecutionId::new(port.authority.epoch, 1),
            owner: profile.owner,
            node_credit: 100,
            soft_lease: Duration::from_millis(8),
            control_sequence: 1,
        };
        port.cpu_admitted(work, &job, &grant).unwrap();
        (job, grant)
    }

    fn receipt(
        port: &Port,
        work: &Work,
        grant: SliceGrant,
        depth: u16,
        delta: u64,
        cumulative: u64,
        completion: CpuCompletion,
    ) -> CpuSliceReport {
        let admission = port
            .cpu_admissions
            .get(&work.input.question_digest)
            .unwrap();
        let movement = admission
            .restriction
            .as_ref()
            .and_then(|moves| moves.first().copied())
            .unwrap_or_else(|| admission.position.legal_moves()[0]);
        let descriptor = port.engine.owned_descriptor().unwrap();
        let yielded = completion == CpuCompletion::Yielded;
        CpuSliceReport {
            grant,
            reciprocal_request: None,
            actual_work: Some(CpuWork {
                nodes: delta,
                quiescence_nodes: 1,
                tt_hits: 0,
            }),
            result: Ok(crate::cpu::CpuControlledReport {
                report: CpuReport {
                    best_move: Some(movement),
                    pv: vec![movement],
                    score: 17,
                    completed_depth: depth,
                    nodes: delta,
                    quiescence_nodes: 1,
                    tt_hits: 0,
                    completion,
                    score_scope: if depth == 0 {
                        CpuScoreScope::FrontierOnly
                    } else {
                        CpuScoreScope::CompletedIteration
                    },
                    profile: descriptor.config.profile,
                    score_provenance: "manager-contract-fixture",
                    value_identity: admission.value_identity.clone(),
                    search_version: descriptor.search_identity,
                    root_restricted: admission.restriction.is_some(),
                    elapsed: Duration::from_millis(1),
                    reused_completed_depth: 0,
                    resume: None,
                },
                yield_cause: yielded.then_some(crate::cpu::CpuYieldCause::NodeCredit),
                yield_ack_sequence: None,
                publish_ack_sequence: 0,
            }),
            cumulative_nodes: cumulative,
            progress_sequence: grant.execution.sequence,
            continuation_retained: yielded,
            terminal: !yielded,
            admission_rejected: false,
            controls: None,
        }
    }

    #[test]
    fn completed_report_hot_commits_and_duplicate_changes_nothing() {
        let (mut port, work, profile) = port_and_work(RefinementStage::CpuValidation, 1);
        let (job, grant) = first_grant(&mut port, &work, profile);
        assert!(!port.cpu_action_committed(work.input.question_digest, job.id, grant.execution));
        let report = receipt(&port, &work, grant, 1, 12, 12, CpuCompletion::DepthLimit);
        let (observation, complete) = port.commit_cpu_report(&work, &report).unwrap();
        assert!(port.cpu_observation_consumed(&work, observation));
        assert!(port.cpu_action_committed(work.input.question_digest, job.id, grant.execution));
        assert!(!port.cpu_action_committed(
            work.input.question_digest,
            job.id,
            rz_contracts::ExecutionId::new(port.authority.epoch, grant.execution.sequence + 1)
        ));
        assert!(complete);
        let admission = port
            .cpu_admissions
            .get(&work.input.question_digest)
            .unwrap();
        assert_eq!(
            port.engine
                .semantic
                .stores
                .observations
                .get(observation)
                .unwrap()
                .budget,
            12
        );
        assert_eq!(
            port.engine
                .semantic
                .stores
                .tasks
                .get(admission.execution.unwrap())
                .unwrap()
                .status,
            TaskStatus::Completed(observation)
        );
        assert!(
            port.engine.semantic.nodes[work.input.question.node]
                .evidence
                .is_some()
        );
        assert_eq!(port.counters.cpu_nodes, 12);
        assert_eq!(port.counters.consumed_cpu_tasks, 1);
        let observed = port.engine.semantic.stores.observations.len();
        assert!(port.commit_cpu_report(&work, &report).is_err());
        assert_eq!(port.engine.semantic.stores.observations.len(), observed);
        assert_eq!(port.counters.cpu_nodes, 12);
        assert_eq!(port.counters.consumed_cpu_tasks, 1);
    }

    #[test]
    fn independent_conclusion_revision_does_not_revoke_exact_cpu_question() {
        let (mut port, work, profile) = port_and_work(RefinementStage::CpuValidation, 1);
        let (_, grant) = first_grant(&mut port, &work, profile);
        let admission = port.cpu_admissions[&work.input.question_digest].clone();
        for node in [port.root, admission.node] {
            let situation = port.engine.semantic.nodes[node].situation;
            let focus = port
                .engine
                .semantic
                .stores
                .situations
                .get(situation)
                .unwrap()
                .focus;
            let evidence = port
                .engine
                .conclusion_observation(node, focus, ObservationKind::Refutation, -1, None)
                .unwrap();
            port.engine
                .semantic
                .stores
                .hot_refute_continuation(situation, focus, evidence)
                .unwrap();
        }
        let report = receipt(&port, &work, grant, 1, 12, 12, CpuCompletion::DepthLimit);
        assert!(port.cpu_context_current(&work, &admission));
        let (_, complete) = port.commit_cpu_report(&work, &report).unwrap();
        assert!(complete);
        assert_eq!(port.counters.consumed_cpu_tasks, 1);
        assert!(
            port.engine.semantic.nodes[admission.node]
                .evidence
                .is_some()
        );
        // No latest-revision consumer was substituted for the original one.
        assert_eq!(
            port.cpu_admissions[&work.input.question_digest]
                .original_consumer
                .unwrap()
                .revision,
            admission.situation_revision
        );
    }

    #[test]
    fn shutdown_observer_is_idempotent_and_never_consumes_search_result() {
        let (mut port, work, profile) = port_and_work(RefinementStage::CpuValidation, 1);
        let (_, grant) = first_grant(&mut port, &work, profile);
        let report = receipt(&port, &work, grant, 1, 12, 12, CpuCompletion::DepthLimit);
        port.observe_cpu_report(&work, &report).unwrap();
        let observed = port.engine.semantic.stores.observations.len();
        port.observe_cpu_report(&work, &report).unwrap();
        assert_eq!(port.engine.semantic.stores.observations.len(), observed);
        assert_eq!(port.counters.cpu_nodes, 12);
        assert_eq!(port.counters.consumed_cpu_tasks, 0);
        assert!(
            port.engine.semantic.nodes[work.input.question.node]
                .evidence
                .is_none()
        );
        let mut changed = report.clone();
        changed.progress_sequence += 1;
        assert!(port.observe_cpu_report(&work, &changed).is_err());
    }

    #[test]
    fn known_worker_failure_is_settled_once_but_unknown_usage_stays_quarantined() {
        let (mut port, work, profile) = port_and_work(RefinementStage::CpuValidation, 1);
        let (_, grant) = first_grant(&mut port, &work, profile);
        let mut report = receipt(&port, &work, grant, 1, 12, 12, CpuCompletion::Canceled);
        report.result = Err(CpuError::Allocation);
        port.observe_cpu_report(&work, &report).unwrap();
        port.observe_cpu_report(&work, &report).unwrap();
        assert_eq!(port.counters.cpu_nodes, 12);
        assert_eq!(port.counters.consumed_cpu_tasks, 0);
        let mut other = report.clone();
        other.actual_work = None;
        assert!(port.observe_cpu_report(&work, &other).is_err());
    }

    #[test]
    fn yielded_same_owner_resume_charges_delta_once_and_preserves_job_key() {
        let (mut port, work, profile) = port_and_work(RefinementStage::CpuValidation, 2);
        let (job, first) = first_grant(&mut port, &work, profile);
        let report = receipt(&port, &work, first, 1, 10, 10, CpuCompletion::Yielded);
        let (partial, complete) = port.commit_cpu_report(&work, &report).unwrap();
        assert!(!complete);
        assert!(port.cpu_observation_consumed(&work, partial));
        let before = port
            .cpu_admissions
            .get(&work.input.question_digest)
            .unwrap();
        let previous = before.execution.unwrap();
        let key = before.task_key.clone();
        assert_eq!(
            port.engine
                .semantic
                .stores
                .tasks
                .get(previous)
                .unwrap()
                .status,
            TaskStatus::Paused {
                checkpoint: job.id.sequence,
                evidence: Some(partial)
            }
        );
        let second = SliceGrant {
            execution: rz_contracts::ExecutionId::new(port.authority.epoch, 2),
            control_sequence: 2,
            ..first
        };
        let foreign = SliceGrant {
            owner: CpuWorkerId(2),
            ..second
        };
        assert!(port.cpu_admitted(&work, &job, &foreign).is_err());
        port.cpu_admitted(&work, &job, &second).unwrap();
        assert!(!port.cpu_observation_consumed(&work, partial));
        let resumed = port
            .cpu_admissions
            .get(&work.input.question_digest)
            .unwrap();
        assert_eq!(resumed.task_key, key);
        assert_eq!(
            port.engine
                .semantic
                .stores
                .tasks
                .get(previous)
                .unwrap()
                .status,
            TaskStatus::RetiredPaused {
                checkpoint: job.id.sequence,
                evidence: Some(partial)
            }
        );
        assert_eq!(
            port.engine
                .semantic
                .stores
                .tasks
                .get(resumed.execution.unwrap())
                .unwrap()
                .resumed_from,
            Some(previous)
        );
        let report = receipt(&port, &work, second, 2, 7, 17, CpuCompletion::DepthLimit);
        let (observation, complete) = port.commit_cpu_report(&work, &report).unwrap();
        assert!(complete);
        assert!(port.cpu_observation_consumed(&work, observation));
        assert_eq!(
            port.engine
                .semantic
                .stores
                .observations
                .get(observation)
                .unwrap()
                .budget,
            7
        );
        assert_eq!(port.counters.cpu_nodes, 17);
        assert_eq!(port.counters.cpu_tasks_requested, 2);
        assert_eq!(port.counters.consumed_partial_cpu_values, 1);
        assert_eq!(port.counters.consumed_cpu_tasks, 1);
    }

    #[test]
    fn soft_no_value_terminal_retains_original_receipt_without_score_or_observation() {
        use crate::cpu_value::{CpuValueAuthorityRevocation, CpuValueError};
        for reason in [
            CpuValueAuthorityRevocation::Cancelled,
            CpuValueAuthorityRevocation::Expired,
        ] {
            let (mut port, work, profile) = port_and_work(RefinementStage::CpuValidation, 2);
            let (_, grant) = first_grant(&mut port, &work, profile);
            let report = CpuSliceReport {
                grant,
                reciprocal_request: None,
                actual_work: Some(CpuWork {
                    nodes: 7,
                    quiescence_nodes: 2,
                    tt_hits: 1,
                }),
                result: Err(crate::cpu::CpuError::Value(
                    CpuValueError::AuthorityRevoked(reason),
                )),
                cumulative_nodes: 7,
                progress_sequence: 1,
                continuation_retained: false,
                terminal: true,
                admission_rejected: false,
                controls: Some(rz_runtime::adaptive::ControlSnapshot {
                    terminal: true,
                    ..Default::default()
                }),
            };
            let observations = port.engine.semantic.stores.observations.len();
            let seal = port
                .observe_cpu_no_value_termination(&work, &report)
                .unwrap();
            assert_eq!(seal, cpu_receipt_digest(&report));
            assert_eq!(
                port.observe_cpu_no_value_termination(&work, &report)
                    .unwrap(),
                seal
            );
            assert_eq!(port.counters.cpu_nodes, 7);
            assert_eq!(port.counters.cpu_quiescence_nodes, 2);
            assert_eq!(port.counters.consumed_cpu_tasks, 0);
            assert_eq!(port.counters.consumed_partial_cpu_values, 0);
            assert_eq!(port.counters.consumed_frontier_cpu_values, 0);
            assert_eq!(port.engine.semantic.stores.observations.len(), observations);
            assert!(
                port.engine.semantic.nodes[work.input.question.node]
                    .evidence
                    .is_none()
            );
            let held = &port.cpu_admissions[&work.input.question_digest];
            assert!(held.consumed_observation.is_none());
            assert!(held.committed_execution.is_none());
            assert!(held.terminal && held.grant.is_none());
            assert_eq!(
                port.engine
                    .semantic
                    .stores
                    .tasks
                    .get(held.execution.unwrap())
                    .unwrap()
                    .status,
                TaskStatus::Failed
            );
            let mut changed = report.clone();
            changed.progress_sequence += 1;
            assert!(
                port.observe_cpu_no_value_termination(&work, &changed)
                    .is_err()
            );
        }
    }

    #[test]
    fn no_value_terminal_rejects_hard_errors_unknown_work_and_retained_tokens() {
        use crate::cpu_value::{CpuValueAuthorityRevocation, CpuValueError};
        for error in [
            CpuValueError::StateMismatch,
            CpuValueError::NonFiniteForward,
            CpuValueError::AuthorityRevoked(CpuValueAuthorityRevocation::ActorStopped),
            CpuValueError::AuthorityRevoked(CpuValueAuthorityRevocation::Invalidated),
        ] {
            let (mut port, work, profile) = port_and_work(RefinementStage::CpuValidation, 2);
            let (_, grant) = first_grant(&mut port, &work, profile);
            let mut report = receipt(&port, &work, grant, 1, 0, 0, CpuCompletion::Canceled);
            report.result = Err(crate::cpu::CpuError::Value(error));
            report.controls = Some(rz_runtime::adaptive::ControlSnapshot {
                terminal: true,
                ..Default::default()
            });
            assert!(
                port.observe_cpu_no_value_termination(&work, &report)
                    .is_err()
            );
            assert!(
                port.cpu_admissions[&work.input.question_digest]
                    .grant
                    .is_some()
            );
        }
        let (mut port, work, profile) = port_and_work(RefinementStage::CpuValidation, 2);
        let (_, grant) = first_grant(&mut port, &work, profile);
        // No CpuReport or numeric score exists in this original terminal.
        let report = CpuSliceReport {
            grant,
            reciprocal_request: None,
            actual_work: Some(CpuWork::default()),
            result: Err(crate::cpu::CpuError::Value(
                CpuValueError::AuthorityRevoked(CpuValueAuthorityRevocation::Cancelled),
            )),
            cumulative_nodes: 0,
            progress_sequence: 0,
            continuation_retained: false,
            terminal: true,
            admission_rejected: false,
            controls: Some(rz_runtime::adaptive::ControlSnapshot {
                terminal: true,
                ..Default::default()
            }),
        };
        let mut unknown = report.clone();
        unknown.actual_work = None;
        assert!(
            port.observe_cpu_no_value_termination(&work, &unknown)
                .is_err()
        );
        let mut retained = report.clone();
        retained.continuation_retained = true;
        assert!(
            port.observe_cpu_no_value_termination(&work, &retained)
                .is_err()
        );
        assert!(
            port.cpu_admissions[&work.input.question_digest]
                .grant
                .is_some()
        );
        let before = port.engine.semantic.stores.observations.len();
        port.observe_cpu_no_value_termination(&work, &report)
            .unwrap();
        assert_eq!(port.counters.cpu_nodes, 0);
        assert_eq!(port.engine.semantic.stores.observations.len(), before);
    }

    #[test]
    fn accepted_terminal_partial_is_one_use_feedback_without_completed_task() {
        let (mut port, work, profile) = port_and_work(RefinementStage::CpuValidation, 2);
        let (_, grant) = first_grant(&mut port, &work, profile);
        let report = receipt(&port, &work, grant, 1, 100, 100, CpuCompletion::NodeLimit);
        let (observation, complete) = port.commit_cpu_report(&work, &report).unwrap();
        assert!(!complete);
        assert!(port.cpu_observation_consumed(&work, observation));
        let admission = port.cpu_admissions[&work.input.question_digest].clone();
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
        assert_eq!(port.counters.completed_cpu_tasks, 0);
        assert_eq!(port.counters.consumed_cpu_tasks, 0);
        assert_eq!(port.counters.consumed_partial_cpu_values, 1);
        assert_eq!(
            port.engine.semantic.stores.hot_consume_sealed_cpu_feedback(
                admission.execution.unwrap(),
                admission.original_consumer.unwrap(),
                &admission.task_key,
                &admission.position.snapshot(),
                observation,
                0,
            ),
            Err(StoreError::AlreadyConsumed)
        );
        let counters = port.counters;
        port.observe_cpu_report(&work, &report).unwrap();
        assert_eq!(port.counters, counters);
    }

    #[test]
    fn revoked_consumer_keeps_partial_raw_but_never_advances_frame() {
        let (mut port, work, profile) = port_and_work(RefinementStage::CpuValidation, 2);
        let (_, grant) = first_grant(&mut port, &work, profile);
        let admission = port.cpu_admissions[&work.input.question_digest].clone();
        port.engine
            .semantic
            .stores
            .tasks
            .cancel_consumer(admission.execution.unwrap(), admission.consumer.unwrap())
            .unwrap();
        let report = receipt(&port, &work, grant, 1, 100, 100, CpuCompletion::NodeLimit);
        let (observation, complete) = port.commit_cpu_report(&work, &report).unwrap();
        assert!(!complete);
        assert!(!port.cpu_observation_consumed(&work, observation));
        assert!(
            port.engine.semantic.nodes[admission.node]
                .evidence
                .is_none()
        );
        assert_eq!(port.counters.cpu_nodes, 100);
        assert_eq!(port.counters.consumed_partial_cpu_values, 0);
        assert!(
            port.engine
                .semantic
                .stores
                .observations
                .get(observation)
                .is_ok()
        );
        port.observe_cpu_report(&work, &report).unwrap();
        assert!(!port.cpu_observation_consumed(&work, observation));
    }

    #[test]
    fn accepted_own_feedback_loses_followup_authority_after_original_consumer_revoke() {
        for completion in [
            CpuCompletion::DepthLimit,
            CpuCompletion::NodeLimit,
            CpuCompletion::Yielded,
        ] {
            let depth = if completion == CpuCompletion::DepthLimit {
                1
            } else {
                2
            };
            let (mut port, work, profile) = port_and_work(RefinementStage::CpuValidation, depth);
            let (_, grant) = first_grant(&mut port, &work, profile);
            let report = receipt(&port, &work, grant, 1, 12, 12, completion);
            let (observation, _) = port.commit_cpu_report(&work, &report).unwrap();
            assert!(port.cpu_observation_consumed(&work, observation));
            let admission = port.cpu_admissions[&work.input.question_digest].clone();
            let execution = admission.execution.unwrap();
            let original = admission.original_consumer.unwrap();
            let status = port
                .engine
                .semantic
                .stores
                .tasks
                .get(execution)
                .unwrap()
                .status;
            let counters = port.counters;
            let observations = port.engine.semantic.stores.observations.len();
            // Retained paused feedback used the real receipt but did not
            // consume the checkpoint's original resume permission. Terminal
            // feedback did consume it. Both still obey an explicit revoke.
            assert_eq!(
                port.engine
                    .semantic
                    .stores
                    .hot_validate_consumed_cpu_feedback(
                        execution,
                        original,
                        &admission.task_key,
                        &admission.position.snapshot(),
                        observation,
                        0,
                    ),
                Ok(observation)
            );
            assert_eq!(
                port.engine
                    .semantic
                    .stores
                    .hot_validate_consumed_cpu_feedback(
                        execution,
                        original,
                        &admission.task_key,
                        &admission.position.snapshot(),
                        observation,
                        original.deadline_tick,
                    ),
                Err(StoreError::ExpiredConsumer)
            );
            port.engine
                .semantic
                .stores
                .tasks
                .cancel_consumer(execution, original.id)
                .unwrap();
            assert_eq!(
                port.engine
                    .semantic
                    .stores
                    .hot_validate_consumed_cpu_feedback(
                        execution,
                        original,
                        &admission.task_key,
                        &admission.position.snapshot(),
                        observation,
                        0,
                    ),
                Err(StoreError::CancelledConsumer)
            );
            assert!(!port.cpu_observation_consumed(&work, observation));
            assert_eq!(
                port.engine
                    .semantic
                    .stores
                    .tasks
                    .get(execution)
                    .unwrap()
                    .status,
                status
            );
            assert!(
                port.engine
                    .semantic
                    .stores
                    .observations
                    .get(observation)
                    .is_ok()
            );
            assert_eq!(port.engine.semantic.stores.observations.len(), observations);
            assert_eq!(port.counters, counters);
            port.observe_cpu_report(&work, &report).unwrap();
            assert_eq!(port.counters, counters);
        }
    }

    #[test]
    fn cpu_origin_step_rechecks_original_consumer_before_materialization() {
        for continuing in [false, true] {
            let (mut port, mut work, profile) = port_and_work(RefinementStage::CpuValidation, 1);
            if continuing {
                // The actual registered proposal has an opponent divergence.
                // This remains an explicit CPU/mock receipt authority fixture.
                let proposal = vec![
                    BoardMove::from_uci("e2e4").unwrap(),
                    BoardMove::from_uci("e7e5").unwrap(),
                ];
                port.record_hot(RecordKind::Proposal, &proposal, None)
                    .unwrap();
                let node = port.replay_node(&proposal).unwrap();
                work.input = port
                    .input(
                        node,
                        RefinementStage::CpuValidation,
                        Some(proposal[0]),
                        proposal.clone(),
                        proposal,
                        None,
                        None,
                        None,
                        false,
                        None,
                        None,
                    )
                    .unwrap();
            }
            let (_, grant) = first_grant(&mut port, &work, profile);
            let report = receipt(&port, &work, grant, 1, 12, 12, CpuCompletion::DepthLimit);
            let (observation, complete) = port.commit_cpu_report(&work, &report).unwrap();
            assert!(complete);
            assert!(port.cpu_feedback_still_authorized(work.input.question_digest, observation));
            let step = port.after_cpu(&work, observation, complete).unwrap();
            assert_eq!(
                step.feedback_gate,
                Some((work.input.question_digest, observation))
            );
            assert_eq!(
                matches!(step.transition, AdaptivePortTransition::Continue(_)),
                continuing
            );
            let admission = port.cpu_admissions[&work.input.question_digest].clone();
            let execution = admission.execution.unwrap();
            let original = admission.original_consumer.unwrap();
            let counters = port.counters;
            let observations = port.engine.semantic.stores.observations.len();
            let public_revision = port.engine.semantic.revision;
            port.engine
                .semantic
                .stores
                .tasks
                .cancel_consumer(execution, original.id)
                .unwrap();
            assert!(!port.cpu_feedback_still_authorized(work.input.question_digest, observation));
            assert!(matches!(
                port.materialize_step(step),
                Err(PalsError::Store(StoreError::StaleConsumer))
            ));
            assert_eq!(port.engine.semantic.stores.observations.len(), observations);
            assert_eq!(port.engine.semantic.revision, public_revision);
            assert_eq!(port.counters, counters);
            assert_eq!(
                port.engine
                    .semantic
                    .stores
                    .tasks
                    .get(execution)
                    .unwrap()
                    .status,
                TaskStatus::Completed(observation)
            );
        }
    }

    #[test]
    fn closed_parent_feedback_requires_its_exact_live_queued_recheck_closure() {
        for revoke in [None, Some("cancel"), Some("expire")] {
            let (mut port, work, profile) = port_and_work(RefinementStage::CpuValidation, 1);
            let (_, grant) = first_grant(&mut port, &work, profile);
            let report = receipt(&port, &work, grant, 1, 12, 12, CpuCompletion::DepthLimit);
            let (observation, _) = port.commit_cpu_report(&work, &report).unwrap();
            let question = work.input.question_digest;
            let first = work.input.question.first.unwrap();
            let admission = port.cpu_admissions[&question].clone();
            let execution = admission.execution.unwrap();
            let original = admission.original_consumer.unwrap();

            // This CPU/mock authority fixture models the known entry that
            // exists only after the opaque queue producer commits. Merely
            // closing a QueuedRecheck frame does not issue this certificate;
            // actual queue production is covered by its separate path tests.
            port.closed.insert(work.frame);
            assert!(!port.cpu_feedback_still_authorized(question, observation));
            assert!(!port.cpu_deferred_feedback_still_authorized(
                work.frame,
                question,
                observation
            ));
            port.queued_recheck_closures
                .insert((work.frame, ContractDigest([91; 32])), first);
            assert!(!port.cpu_deferred_feedback_still_authorized(
                work.frame,
                question,
                observation
            ));
            port.queued_recheck_closures
                .insert((work.frame, question), BoardMove::from_uci("d2d4").unwrap());
            assert!(!port.cpu_deferred_feedback_still_authorized(
                work.frame,
                question,
                observation
            ));
            port.queued_recheck_closures
                .insert((work.frame, question), first);
            assert!(port.cpu_deferred_feedback_still_authorized(work.frame, question, observation));
            assert!(!port.cpu_feedback_still_authorized(question, observation));
            port.closed.remove(&work.frame);
            assert!(!port.cpu_deferred_feedback_still_authorized(
                work.frame,
                question,
                observation
            ));
            assert!(port.cpu_feedback_still_authorized(question, observation));
            port.closed.insert(work.frame);
            let other = FrameId {
                sequence: work.frame.sequence + 1,
                ..work.frame
            };
            port.closed.insert(other);
            port.queued_recheck_closures
                .insert((other, question), first);
            assert!(!port.cpu_deferred_feedback_still_authorized(other, question, observation));

            let counters = port.counters;
            let observations = port.engine.semantic.stores.observations.len();
            let public_revision = port.engine.semantic.revision;
            match revoke {
                Some("cancel") => port
                    .engine
                    .semantic
                    .stores
                    .tasks
                    .cancel_consumer(execution, original.id)
                    .unwrap(),
                Some("expire") => {
                    port.limits.deadline = Instant::now();
                }
                None => {}
                _ => unreachable!(),
            }
            assert_eq!(
                port.cpu_deferred_feedback_still_authorized(work.frame, question, observation),
                revoke.is_none()
            );
            assert!(!port.cpu_feedback_still_authorized(question, observation));
            assert_eq!(port.counters, counters);
            assert_eq!(port.engine.semantic.stores.observations.len(), observations);
            assert_eq!(port.engine.semantic.revision, public_revision);
            assert_eq!(
                port.engine
                    .semantic
                    .stores
                    .tasks
                    .get(execution)
                    .unwrap()
                    .status,
                TaskStatus::Completed(observation)
            );
        }
    }

    #[test]
    fn independent_conclusions_preserve_materialize_authority_and_paused_resume_permission() {
        for completion in [CpuCompletion::DepthLimit, CpuCompletion::Yielded] {
            let depth = if completion == CpuCompletion::Yielded {
                2
            } else {
                1
            };
            let (mut port, work, profile) = port_and_work(RefinementStage::CpuValidation, depth);
            let (_, grant) = first_grant(&mut port, &work, profile);
            let report = receipt(&port, &work, grant, 1, 12, 12, completion);
            let (observation, complete) = port.commit_cpu_report(&work, &report).unwrap();
            let step = port.after_cpu(&work, observation, complete).unwrap();
            let admission = port.cpu_admissions[&work.input.question_digest].clone();
            let execution = admission.execution.unwrap();
            let original = admission.original_consumer.unwrap();
            let counters = port.counters;
            for node in [port.root, admission.node] {
                let situation = port.engine.semantic.nodes[node].situation;
                let focus = port
                    .engine
                    .semantic
                    .stores
                    .situations
                    .get(situation)
                    .unwrap()
                    .focus;
                let conclusion = port
                    .engine
                    .conclusion_observation(node, focus, ObservationKind::Refutation, -1, None)
                    .unwrap();
                port.engine
                    .semantic
                    .stores
                    .hot_refute_continuation(situation, focus, conclusion)
                    .unwrap();
            }
            assert!(port.cpu_feedback_still_authorized(work.input.question_digest, observation));
            let materialized = port.materialize_step(step).unwrap();
            assert!(matches!(
                materialized.transition,
                FrameTransition::Complete {
                    scope: FrameCompletionScope::NoDivergence
                }
            ));
            assert_eq!(port.counters, counters);
            if completion == CpuCompletion::Yielded {
                assert_eq!(
                    port.engine
                        .semantic
                        .stores
                        .hot_validate_sealed_paused_cpu_feedback(
                            execution,
                            original,
                            &admission.task_key,
                            &admission.position.snapshot(),
                            observation,
                            0,
                        ),
                    Ok(observation)
                );
                assert!(
                    matches!(port.engine.semantic.stores.tasks.get(execution).unwrap().status,
                    TaskStatus::Paused { evidence: Some(id), .. } if id == observation)
                );
            } else {
                assert_eq!(
                    port.engine.semantic.stores.hot_consume_sealed_cpu_task(
                        execution,
                        original,
                        &admission.task_key,
                        &admission.position.snapshot(),
                        0,
                    ),
                    Err(StoreError::AlreadyConsumed)
                );
            }
        }
    }

    #[test]
    fn canceled_and_expired_worker_reports_remain_raw_only_under_live_wall_clock() {
        for completion in [CpuCompletion::Canceled, CpuCompletion::Deadline] {
            let (mut port, work, profile) = port_and_work(RefinementStage::CpuValidation, 2);
            let (_, grant) = first_grant(&mut port, &work, profile);
            let report = receipt(&port, &work, grant, 1, 12, 12, completion);
            let (observation, _) = port.commit_cpu_report(&work, &report).unwrap();
            assert!(!port.cpu_observation_consumed(&work, observation));
            let admission = port.cpu_admissions[&work.input.question_digest].clone();
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
            assert!(
                port.engine.semantic.nodes[admission.node]
                    .evidence
                    .is_none()
            );
            assert_eq!(port.counters.cpu_nodes, 12);
            assert_eq!(port.counters.consumed_cpu_tasks, 0);
            assert_eq!(port.counters.consumed_partial_cpu_values, 0);
            assert!(port.engine.semantic.records.is_empty());
            port.observe_cpu_report(&work, &report).unwrap();
            assert_eq!(port.counters.cpu_nodes, 12);
        }
    }

    #[test]
    fn independent_conclusions_preserve_paused_job_resume_and_original_authority() {
        let (mut port, work, profile) = port_and_work(RefinementStage::CpuValidation, 2);
        let (job, first) = first_grant(&mut port, &work, profile);
        let report = receipt(&port, &work, first, 1, 10, 10, CpuCompletion::Yielded);
        let (_, complete) = port.commit_cpu_report(&work, &report).unwrap();
        assert!(!complete);
        let original = port.cpu_admissions[&work.input.question_digest].clone();
        let previous = original.execution.unwrap();
        // An independent frame may publish conclusions at either shared
        // situation while this exact worker-owned job is temporarily paused.
        for node in [port.root, original.node] {
            let situation = port.engine.semantic.nodes[node].situation;
            let focus = port
                .engine
                .semantic
                .stores
                .situations
                .get(situation)
                .unwrap()
                .focus;
            let evidence = port
                .engine
                .conclusion_observation(node, focus, ObservationKind::Refutation, -1, None)
                .unwrap();
            port.engine
                .semantic
                .stores
                .hot_refute_continuation(situation, focus, evidence)
                .unwrap();
        }
        let second = SliceGrant {
            execution: rz_contracts::ExecutionId::new(port.authority.epoch, 2),
            control_sequence: 2,
            ..first
        };
        assert!(port.cpu_context_current(&work, &original));
        port.cpu_admitted(&work, &job, &second).unwrap();
        let resumed = port.cpu_admissions[&work.input.question_digest].clone();
        let authority = resumed.original_consumer.unwrap();
        assert_eq!(
            authority.revision,
            original.original_consumer.unwrap().revision
        );
        assert_eq!(
            authority.deadline_tick,
            original.original_consumer.unwrap().deadline_tick
        );
        assert_eq!(
            authority.generation,
            original.original_consumer.unwrap().generation
        );
        assert_ne!(authority.id, original.original_consumer.unwrap().id);
        assert_eq!(
            port.engine
                .semantic
                .stores
                .tasks
                .get(resumed.execution.unwrap())
                .unwrap()
                .resumed_from,
            Some(previous)
        );
        let report = receipt(&port, &work, second, 2, 7, 17, CpuCompletion::DepthLimit);
        let (_, complete) = port.commit_cpu_report(&work, &report).unwrap();
        assert!(complete);
        assert_eq!(port.counters.cpu_nodes, 17);
        assert_eq!(port.counters.consumed_cpu_tasks, 1);
        assert_eq!(
            port.engine.semantic.stores.hot_consume_sealed_cpu_task(
                resumed.execution.unwrap(),
                resumed.original_consumer.unwrap(),
                &resumed.task_key,
                &resumed.position.snapshot(),
                0,
            ),
            Err(StoreError::AlreadyConsumed)
        );
    }

    #[test]
    fn two_pending_frames_keep_original_cpu_seals_through_conclusion_and_resume() {
        let (mut port, first_work, first_profile) =
            port_and_work(RefinementStage::CpuValidation, 2);
        let (first_job, first_grant) = first_grant(&mut port, &first_work, first_profile);
        let first_move = BoardMove::from_uci("d2d4").unwrap();
        let second_node = port.connect_hot(port.root, first_move).unwrap();
        let second_input = port
            .input(
                second_node,
                RefinementStage::CpuValidation,
                Some(first_move),
                vec![first_move],
                vec![first_move],
                None,
                None,
                None,
                false,
                None,
                None,
            )
            .unwrap();
        let second_work = Work {
            frame: FrameId {
                authority: port.authority,
                sequence: 2,
            },
            original_first_move: rz_contracts::pals::Move16::encode(
                rz_contracts::Move::try_from(first_move).unwrap(),
            ),
            original_deadline: port.deadline,
            budget: FrameBudget {
                original_cpu_credit: port.limits.max_cpu_nodes,
                role_submissions_remaining: port.engine.config.max_role_calls,
                original_store_byte_limit: 8 * 1024 * 1024,
            },
            input: second_input,
        };
        let second_profile = CpuWorkerProfile {
            owner: CpuWorkerId(2),
            ..first_profile
        };
        let prepared = port.cpu_question(&second_work, &[second_profile]).unwrap();
        let second_job = CpuJobEnvelope {
            id: CpuJobId {
                authority: port.authority,
                sequence: 2,
            },
            state_identity: prepared.position.export().unwrap().snapshot().identity(),
            question: second_work.input.question_digest,
            root_restriction: cpu_root_restriction_digest(prepared.root_moves.as_deref()),
            evaluation_namespace: second_profile.evaluation_namespace,
            search_conditions: second_profile.search_conditions,
            total_native_node_credit: prepared.total_native_node_credit,
            hard_deadline: port.deadline,
        };
        let second_grant = SliceGrant {
            job: second_job.id,
            owner: second_profile.owner,
            execution: rz_contracts::ExecutionId::new(port.authority.epoch, 2),
            ..first_grant
        };
        port.cpu_admitted(&second_work, &second_job, &second_grant)
            .unwrap();
        let yielded = receipt(
            &port,
            &first_work,
            first_grant,
            1,
            10,
            10,
            CpuCompletion::Yielded,
        );
        port.commit_cpu_report(&first_work, &yielded).unwrap();
        let first = port.cpu_admissions[&first_work.input.question_digest].clone();
        for node in [port.root, first.node] {
            let situation = port.engine.semantic.nodes[node].situation;
            let focus = port
                .engine
                .semantic
                .stores
                .situations
                .get(situation)
                .unwrap()
                .focus;
            let evidence = port
                .engine
                .conclusion_observation(node, focus, ObservationKind::Refutation, -1, None)
                .unwrap();
            port.engine
                .semantic
                .stores
                .hot_refute_continuation(situation, focus, evidence)
                .unwrap();
        }
        // Both pending questions retain their independently sealed authority.
        let second_report = receipt(
            &port,
            &second_work,
            second_grant,
            2,
            8,
            8,
            CpuCompletion::DepthLimit,
        );
        assert!(
            port.commit_cpu_report(&second_work, &second_report)
                .unwrap()
                .1
        );
        let resumed_grant = SliceGrant {
            execution: rz_contracts::ExecutionId::new(port.authority.epoch, 3),
            control_sequence: 2,
            ..first_grant
        };
        port.cpu_admitted(&first_work, &first_job, &resumed_grant)
            .unwrap();
        let resumed_report = receipt(
            &port,
            &first_work,
            resumed_grant,
            2,
            7,
            17,
            CpuCompletion::DepthLimit,
        );
        assert!(
            port.commit_cpu_report(&first_work, &resumed_report)
                .unwrap()
                .1
        );
        assert_eq!(port.counters.cpu_nodes, 25);
        assert_eq!(port.counters.consumed_cpu_tasks, 2);
        assert_eq!(
            port.cpu_admissions[&first_work.input.question_digest]
                .original_consumer
                .unwrap()
                .revision,
            first.original_consumer.unwrap().revision
        );
        assert_ne!(
            first_work.input.question_digest,
            second_work.input.question_digest
        );
    }

    #[test]
    fn discovery_keeps_raw_pv_without_generic_node_evidence() {
        let (mut port, work, profile) = port_and_work(RefinementStage::CpuDiscovery, 1);
        let (_, grant) = first_grant(&mut port, &work, profile);
        let report = receipt(&port, &work, grant, 1, 11, 11, CpuCompletion::DepthLimit);
        let (observation, complete) = port.commit_cpu_report(&work, &report).unwrap();
        assert!(complete);
        assert!(
            port.engine
                .semantic
                .stores
                .observations
                .get(observation)
                .unwrap()
                .cpu_pv
                .is_some()
        );
        assert!(
            port.engine.semantic.nodes[work.input.question.node]
                .evidence
                .is_none()
        );
    }

    #[test]
    fn predispatch_retry_issues_fresh_rules_owner_and_resume_cannot_reissue_it() {
        let (mut port, work, profile) = port_and_work(RefinementStage::CpuValidation, 1);
        let first = port.cpu_question(&work, &[profile]).unwrap();
        let original = first.position.export().unwrap().snapshot().identity();
        let retry = port.cpu_question(&work, &[profile]).unwrap();
        let current = retry.position.export().unwrap().snapshot().identity();
        assert_ne!(original, current);
        let mut job = CpuJobEnvelope {
            id: CpuJobId {
                authority: port.authority,
                sequence: 1,
            },
            state_identity: original,
            question: work.input.question_digest,
            root_restriction: cpu_root_restriction_digest(retry.root_moves.as_deref()),
            evaluation_namespace: profile.evaluation_namespace,
            search_conditions: profile.search_conditions,
            total_native_node_credit: retry.total_native_node_credit,
            hard_deadline: port.deadline,
        };
        let grant = SliceGrant {
            job: job.id,
            execution: rz_contracts::ExecutionId::new(port.authority.epoch, 1),
            owner: profile.owner,
            node_credit: 100,
            soft_lease: Duration::from_millis(8),
            control_sequence: 1,
        };
        assert!(port.cpu_admitted(&work, &job, &grant).is_err());
        job.state_identity = current;
        port.cpu_admitted(&work, &job, &grant).unwrap();
        assert!(port.cpu_question(&work, &[profile]).is_err());
        assert_eq!(
            port.cpu_admissions
                .get(&work.input.question_digest)
                .unwrap()
                .rules_identity,
            current
        );
    }

    #[test]
    fn restricted_discovery_cannot_accept_excluded_legal_move() {
        let (mut port, mut work, profile) = port_and_work(RefinementStage::CpuDiscovery, 1);
        Arc::make_mut(&mut work.input.question).discovery_restricted = true;
        let (_, grant) = first_grant(&mut port, &work, profile);
        let old = BoardMove::from_uci("e7e5").unwrap();
        let mut report = receipt(&port, &work, grant, 1, 11, 11, CpuCompletion::DepthLimit);
        let controlled = report.result.as_mut().unwrap();
        controlled.report.best_move = Some(old);
        controlled.report.pv = vec![old];
        assert!(port.commit_cpu_report(&work, &report).is_err());
        assert_eq!(port.counters.cpu_nodes, 11);
        assert!(
            port.engine.semantic.nodes[work.input.question.node]
                .evidence
                .is_none()
        );
        assert!(
            port.cpu_admissions
                .get(&work.input.question_digest)
                .unwrap()
                .quarantined
        );
    }

    #[test]
    fn expired_report_preserves_actual_work_and_raw_observation_without_consumption() {
        let (mut port, work, profile) = port_and_work(RefinementStage::CpuValidation, 1);
        let (_, grant) = first_grant(&mut port, &work, profile);
        let report = receipt(&port, &work, grant, 1, 13, 13, CpuCompletion::DepthLimit);
        port.limits.deadline = Instant::now() - Duration::from_millis(1);
        let (observation, complete) = port.commit_cpu_report(&work, &report).unwrap();
        assert!(complete);
        assert_eq!(
            port.engine
                .semantic
                .stores
                .observations
                .get(observation)
                .unwrap()
                .budget,
            13
        );
        assert_eq!(port.counters.cpu_nodes, 13);
        assert_eq!(port.counters.consumed_cpu_tasks, 0);
        assert!(
            port.engine.semantic.nodes[work.input.question.node]
                .evidence
                .is_none()
        );
    }

    #[test]
    fn unknown_usage_retains_inflight_task_and_quarantines_admission() {
        let (mut port, work, profile) = port_and_work(RefinementStage::CpuValidation, 1);
        let (_, grant) = first_grant(&mut port, &work, profile);
        let mut report = receipt(&port, &work, grant, 1, 12, 12, CpuCompletion::DepthLimit);
        report.actual_work = None;
        assert!(port.commit_cpu_report(&work, &report).is_err());
        let admission = port
            .cpu_admissions
            .get(&work.input.question_digest)
            .unwrap();
        assert!(admission.quarantined);
        assert!(admission.grant.is_some());
        assert_eq!(
            port.engine
                .semantic
                .stores
                .tasks
                .get(admission.execution.unwrap())
                .unwrap()
                .status,
            TaskStatus::InFlight
        );
        assert!(port.counters.cpu_work_observation_incomplete);
        assert_eq!(port.counters.cpu_nodes, 0);
        assert_eq!(port.counters.cpu_tasks, 1);
        assert!(
            port.engine.semantic.nodes[work.input.question.node]
                .evidence
                .is_none()
        );
    }

    #[test]
    fn paused_owner_discard_after_deadline_preserves_work_and_evidence() {
        let (mut port, work, profile) = port_and_work(RefinementStage::CpuValidation, 2);
        let (job, grant) = first_grant(&mut port, &work, profile);
        let yielded = receipt(&port, &work, grant, 1, 10, 10, CpuCompletion::Yielded);
        let (evidence, _) = port.commit_cpu_report(&work, &yielded).unwrap();
        let execution = port.cpu_admissions[&work.input.question_digest]
            .execution
            .unwrap();
        port.limits.deadline = Instant::now();
        let termination = CpuJobTermination {
            job: job.id,
            owner: grant.owner,
            last_execution: grant.execution,
            completion: CpuCompletion::Deadline,
            controls: rz_runtime::adaptive::ControlSnapshot {
                terminal: true,
                ..Default::default()
            },
            retained_bytes: 0,
            shutdown: false,
        };
        let observed = port.engine.semantic.stores.observations.len();
        port.retire_cpu_token(&work, &termination).unwrap();
        assert_eq!(port.counters.cpu_nodes, 10);
        assert_eq!(port.engine.semantic.stores.observations.len(), observed);
        assert_eq!(
            port.engine
                .semantic
                .stores
                .tasks
                .get(execution)
                .unwrap()
                .status,
            TaskStatus::RetiredPaused {
                checkpoint: job.id.sequence,
                evidence: Some(evidence)
            }
        );
        assert!(port.cpu_admissions[&work.input.question_digest].terminal);
        assert!(port.retire_cpu_token(&work, &termination).is_err());
    }

    #[test]
    fn paused_owner_discard_rejects_foreign_ack_and_quarantined_usage() {
        let (mut port, work, profile) = port_and_work(RefinementStage::CpuValidation, 2);
        let (job, grant) = first_grant(&mut port, &work, profile);
        let yielded = receipt(&port, &work, grant, 1, 10, 10, CpuCompletion::Yielded);
        port.commit_cpu_report(&work, &yielded).unwrap();
        let execution = port.cpu_admissions[&work.input.question_digest]
            .execution
            .unwrap();
        let mut termination = CpuJobTermination {
            job: job.id,
            owner: CpuWorkerId(99),
            last_execution: grant.execution,
            completion: CpuCompletion::Canceled,
            controls: rz_runtime::adaptive::ControlSnapshot {
                cancel_sequence: 2,
                cancel_ack: 2,
                terminal: true,
                ..Default::default()
            },
            retained_bytes: 0,
            shutdown: false,
        };
        assert!(port.retire_cpu_token(&work, &termination).is_err());
        termination.owner = grant.owner;
        termination.controls.cancel_ack = 1;
        assert!(port.retire_cpu_token(&work, &termination).is_err());
        termination.controls.cancel_ack = 2;
        port.cpu_admissions
            .get_mut(&work.input.question_digest)
            .unwrap()
            .quarantined = true;
        assert!(port.retire_cpu_token(&work, &termination).is_err());
        assert!(matches!(
            port.engine
                .semantic
                .stores
                .tasks
                .get(execution)
                .unwrap()
                .status,
            TaskStatus::Paused { .. }
        ));
        assert_eq!(port.counters.cpu_nodes, 10);
    }

    #[test]
    fn mismatched_cumulative_usage_cannot_publish_or_double_charge() {
        let (mut port, work, profile) = port_and_work(RefinementStage::CpuValidation, 1);
        let (_, grant) = first_grant(&mut port, &work, profile);
        let report = receipt(&port, &work, grant, 1, 12, 24, CpuCompletion::DepthLimit);
        let observed = port.engine.semantic.stores.observations.len();
        assert!(port.commit_cpu_report(&work, &report).is_err());
        assert!(port.commit_cpu_report(&work, &report).is_err());
        assert_eq!(port.engine.semantic.stores.observations.len(), observed);
        assert_eq!(port.counters.cpu_nodes, 0);
        assert!(port.counters.cpu_work_observation_incomplete);
        assert_eq!(port.counters.cpu_tasks, 1);
    }
}
