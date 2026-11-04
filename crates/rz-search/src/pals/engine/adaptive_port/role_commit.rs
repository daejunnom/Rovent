//! Pure checked role consumption and continuation construction. The physical
//! owner must acknowledge the ORIGINAL ticket before calling commit_role.
use super::checked_replay::PreparedRoleChoice;
use super::*;

impl<M: RoleModel> AdaptiveEnginePort<M> {
    pub fn commit_role(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
        physical: Option<AdaptivePortRoleEvidence>,
    ) -> Result<AdaptivePortStep, PalsError> {
        self.role_consumption_observation_started = true;
        let prior_semantic_attempt = self
            .role_consumption_observation
            .as_ref()
            .map(|_| self.attempted_role_commits.contains(&ticket.id));
        let mut published_repair_line = None;
        let repair_publication_sink = self
            .role_consumption_observation
            .is_some()
            .then_some(&mut published_repair_line);
        let mut returned =
            self.commit_role_unobserved(work, ticket, output, physical, repair_publication_sink);
        if let (Some(published_line), Ok(step)) = (published_repair_line, &mut returned) {
            step.transition
                .carry_diagnostic_repair_line(Some(AdaptiveRepairLineProvenance {
                    producer_frame: work.frame,
                    producer_request: ticket.id,
                    submitted_question: work.input.question_digest,
                    submitted_view: work.input.view_seal.view_digest,
                    completed_evidence: step.evidence,
                    original_deadline: work.original_deadline,
                    published_line,
                }));
        }
        // Copy only a successful original semantic result. The separate
        // first ConsumedView and coordinator acceptance are joined at readback.
        let feedback_edge = self.role_consumption_observation.as_ref().and_then(|_| {
            returned.as_ref().ok().map(|step| {
                super::profiled_external_port::feedback_observation::ProfiledFeedbackRoleEdge {
                    evidence: crate::pals::coordinator::StepEvidence {
                        frame: work.frame,
                        stage: work.input.stage,
                        execution: crate::pals::coordinator::StepExecution::Role(ticket.id),
                        submitted_question: work.input.question_digest,
                        submitted_view: work.input.view_seal,
                        completed_evidence: step.evidence,
                    },
                    request: ticket.id,
                    role: ticket.role,
                    ticket_question: ticket.question,
                    ticket_view: ticket.view,
                    namespace: self
                        .model_namespace()
                        .ok()
                        .map(|v| v.utility_cohort_digest()),
                    original_deadline: work.original_deadline,
                    prefix_before_plies: work.input.question.context.prefix.len(),
                    counter_source: work.input.question.counter_source,
                    published_repair_line,
                    repair_input: work.input.question.diagnostic_repair_line,
                    fresh_pair: step.fresh_pair,
                }
            })
        });
        if let Some(journal) = &mut self.role_consumption_observation {
            if let Some(edge) = feedback_edge {
                journal.record_feedback_role(edge);
            }
            let (outcome, completed_evidence) = match &returned {
                Ok(step) => (
                    AdaptiveRoleConsumptionOutcome::SemanticResult,
                    Some(step.evidence),
                ),
                Err(error) => (
                    AdaptiveRoleConsumptionOutcome::Rejected(
                        AdaptiveRoleConsumptionRejection::from_error(error),
                    ),
                    None,
                ),
            };
            journal.record(AdaptiveRoleConsumptionEvent {
                authority: ticket.authority,
                request: ticket.id,
                role: ticket.role,
                question: ticket.question,
                submitted_question: Some(work.input.question_digest),
                ticket_view: ticket.view.view_digest,
                submitted_view: work.input.view_seal.view_digest,
                frame: Some(work.frame),
                stage: AdaptiveRoleConsumptionStage::SemanticCommit,
                outcome,
                completed_evidence,
                prior_semantic_attempt,
                published_repair_line,
                repair_line_input: work
                    .input
                    .question
                    .diagnostic_repair_line
                    .map(|provenance| AdaptiveRepairLineInput {
                        refinement_stage: work.input.stage,
                        provenance,
                        model_proposal_sha256: ContractDigest(
                            work.input.question.context.proposal_sha256,
                        ),
                        model_proposal_plies: work.input.view.query().proposal.len(),
                        repaired_plies: work.input.question.repaired.as_ref().map(Vec::len),
                    }),
            });
        }
        returned
    }

    fn commit_role_unobserved(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
        physical: Option<AdaptivePortRoleEvidence>,
        repair_publication_sink: Option<&mut Option<AdaptivePublishedRepairLine>>,
    ) -> Result<AdaptivePortStep, PalsError> {
        self.validate_role(work, ticket, output)?;
        if let Some(failure) = self.reciprocal_source_failure(work, ticket, output) {
            return Err(failure);
        }
        let source_preconsumed = self.reciprocal_source_consumed(work, ticket, output);
        let observations_before = self
            .engine
            .semantic
            .stores
            .observations
            .next_observation_sequence();
        if !source_preconsumed && self.counters.role_calls >= self.engine.config.max_role_calls {
            return Err(PalsError::RoleCallLimit);
        }
        let query = &work.input.question;
        if matches!(
            query.stage,
            RefinementStage::ProposerRanking
                | RefinementStage::ValidatorTaskRanking
                | RefinementStage::CpuValidation
                | RefinementStage::CpuDiscovery
                | RefinementStage::CounterValidation
                | RefinementStage::RepairValidation
                | RefinementStage::RecheckCounterValidation
        ) {
            return Err(RoleError::Unsupported("stage is not a role completion").into());
        }
        if query.context.purpose == RoleQueryPurpose::ValueFresh {
            // Missing physical completion or a mismatched value identity is a
            // pure admission failure. It must not consume the one semantic
            // attempt reserved for a fully checked response.
            self.checked_role_value(work, ticket, output, physical)?;
        }
        let prepared_fresh_pair = if query.stage == RefinementStage::FreshRepairedEndpoint {
            let next = self
                .fresh_pending
                .get(&query.question_digest)
                .ok_or(RoleError::InvalidOutput)?;
            // The legacy full-context baseline freezes the original raw rows
            // once for both endpoints. Equal selected views do not authorize
            // substituting a later raw snapshot for the second Fresh value.
            let same_raw_context = match (
                &query.legacy_full_records,
                &next.question.legacy_full_records,
            ) {
                (None, None) => true,
                (Some(first), Some(second)) => {
                    Arc::ptr_eq(first, second)
                        && first.authority() == self.authority
                        && second.authority() == self.authority
                        && first.digest() == second.digest()
                        && first.validates(&self.engine.semantic.stores)
                        && second.validates(&self.engine.semantic.stores)
                }
                _ => false,
            };
            if !same_raw_context {
                return Err(StoreError::InvalidEvidence(
                    "Fresh pair legacy raw context differs from frozen submission",
                )
                .into());
            }
            let seal = FreshPairSeal::from_ticket(
                ticket,
                work.input.public_context,
                self.engine.semantic.nodes[self.root]
                    .position
                    .side_to_move()
                    .into(),
            );
            if !seal.validates_public_view(&next.view_seal, next.public_context) {
                return Err(RoleError::InvalidOutput.into());
            }
            Some(seal)
        } else {
            None
        };
        if self.attempted_role_commits.contains(&ticket.id) {
            return Err(
                StoreError::InvalidEvidence("role semantic commit was already attempted").into(),
            );
        }
        let choice = self.prepare_role_choice(work, output)?;
        // Keep the original head and graph choice intact while an actual
        // native reservation blocks the next Rules edge. No Store publication
        // or semantic-attempt marker has occurred when this returns Reserved.
        self.preflight_role_choice(work, choice)?;
        self.attempted_role_commits.insert(ticket.id);
        let evidence = role_output_digest(ticket, output);
        let mut fresh_pair = None;
        let rules_pair = None;
        let transition = match query.stage {
            RefinementStage::ProposalContinuation
            | RefinementStage::CounterContinuation
            | RefinementStage::Repair => {
                let PreparedRoleChoice::Continuation(movement) = choice else {
                    return Err(RoleError::InvalidOutput.into());
                };
                let mut prefix = query.context.prefix.clone();
                prefix.push(movement);
                let node = self.replay_node(&prefix)?;
                let done = prefix.len() >= self.continuation_limit()
                    || self.engine.semantic.nodes[node].terminal.is_some();
                let next = if done {
                    let (kind, stage) = match query.stage {
                        RefinementStage::ProposalContinuation => {
                            (RecordKind::Proposal, RefinementStage::CpuValidation)
                        }
                        RefinementStage::CounterContinuation => (
                            RecordKind::Counterexample,
                            RefinementStage::CounterValidation,
                        ),
                        _ => (RecordKind::Repair, RefinementStage::RepairValidation),
                    };
                    self.record_hot_with_cpu_value(
                        kind,
                        &prefix,
                        if kind == RecordKind::Counterexample {
                            query.counter_source
                        } else {
                            None
                        },
                        None,
                        repair_publication_sink,
                    )?;
                    match kind {
                        RecordKind::Proposal => self.counters.proposals += 1,
                        RecordKind::Counterexample => self.counters.refutations += 1,
                        RecordKind::Repair => self.counters.repairs += 1,
                        _ => {}
                    }
                    stage
                } else {
                    query.stage
                };
                let proposal = if query.stage == RefinementStage::ProposalContinuation && done {
                    prefix.clone()
                } else {
                    query.proposal.clone()
                };
                let counter = if query.stage == RefinementStage::CounterContinuation && done {
                    Some(prefix.clone())
                } else {
                    query.counter.clone()
                };
                let repaired = if query.stage == RefinementStage::Repair && done {
                    Some(prefix.clone())
                } else {
                    query.repaired.clone()
                };
                let repair_selection = if query.stage == RefinementStage::Repair && done {
                    self.prepare_completed_repair_goal_selection(work, &prefix)?
                } else {
                    None
                };
                let mut next = self.blueprint(
                    node,
                    next,
                    query.first,
                    prefix,
                    proposal,
                    counter,
                    repaired,
                    query.attack_ply,
                    false,
                    None,
                    None,
                )?;
                next = next
                    .with_counter_source(query.counter_source)
                    .with_goal_lane_selection(repair_selection)?;
                AdaptivePortTransition::Continue(next)
            }
            RefinementStage::CriticDivergence => {
                let PreparedRoleChoice::Divergence(ply) = choice else {
                    return Err(RoleError::InvalidOutput.into());
                };
                if let Some(ply) = ply {
                    let prefix = query.proposal[..ply].to_vec();
                    let node = self.replay_node(&prefix)?;
                    AdaptivePortTransition::Continue(self.blueprint(
                        node,
                        RefinementStage::CpuDiscovery,
                        query.first,
                        prefix,
                        query.proposal.clone(),
                        None,
                        None,
                        Some(ply),
                        false,
                        None,
                        None,
                    )?)
                } else {
                    AdaptivePortTransition::Complete {
                        scope: FrameCompletionScope::NoDivergence,
                    }
                }
            }
            RefinementStage::CriticReply => {
                let ply = query.attack_ply.ok_or(RoleError::InvalidOutput)?;
                let cpu_response = query
                    .counter
                    .as_ref()
                    .and_then(|line| line.get(ply))
                    .copied();
                let PreparedRoleChoice::CriticReply(movement) = choice else {
                    return Err(RoleError::InvalidOutput.into());
                };
                let critic = if let Some(movement) = movement {
                    let mut prefix = query.context.prefix.clone();
                    prefix.push(movement);
                    Some(Self::pending_counter(query, prefix, None)?)
                } else {
                    None
                };
                if let Some(cpu) = query.counter.as_ref().filter(|_| cpu_response.is_some()) {
                    if let Some(critic) = critic {
                        self.enqueue_counter(critic)?;
                    }
                    self.begin_counter(
                        work,
                        Self::pending_counter(query, cpu.clone(), query.counter_source)?,
                    )?
                } else if let Some(critic) = critic {
                    self.begin_counter(work, critic)?
                } else {
                    AdaptivePortTransition::Complete {
                        scope: FrameCompletionScope::NoAlternative,
                    }
                }
            }
            RefinementStage::ProposalValueFresh
            | RefinementStage::CounterValueFresh
            | RefinementStage::RepairValueFresh => {
                self.commit_value(work, ticket, output, physical)?;
                self.after_validation(query, Some(work))?
            }
            RefinementStage::RecheckRepairedValueFresh => {
                self.commit_value(work, ticket, output, physical)?;
                self.critic_recheck_after_repaired_fresh(query)?
            }
            RefinementStage::CriticRecheck => {
                let repaired = query.repaired.as_ref().ok_or(RoleError::InvalidOutput)?;
                let PreparedRoleChoice::CriticRecheck(movement) = choice else {
                    return Err(RoleError::InvalidOutput.into());
                };
                if let Some(movement) = movement {
                    let mut counterline = query.context.prefix.clone();
                    counterline.push(movement);
                    let node = self.replay_node(&counterline)?;
                    if self.engine.semantic.nodes[node].terminal.is_some()
                        || counterline.len() >= repaired.len()
                    {
                        if counterline.len() != repaired.len() {
                            AdaptivePortTransition::Unknown {
                                reason: "incomplete conditional recheck line",
                            }
                        } else {
                            self.record_hot(RecordKind::Counterexample, &counterline, None)?;
                            AdaptivePortTransition::Continue(self.blueprint(
                                node,
                                RefinementStage::RecheckCounterValidation,
                                query.first,
                                counterline.clone(),
                                query.proposal.clone(),
                                Some(counterline),
                                Some(repaired.clone()),
                                query.attack_ply,
                                true,
                                None,
                                None,
                            )?)
                        }
                    } else {
                        AdaptivePortTransition::Continue(self.blueprint(
                            node,
                            RefinementStage::CriticRecheck,
                            query.first,
                            counterline,
                            repaired.clone(),
                            query.counter.clone(),
                            Some(repaired.clone()),
                            query.attack_ply,
                            true,
                            None,
                            None,
                        )?)
                    }
                } else {
                    AdaptivePortTransition::Complete {
                        scope: FrameCompletionScope::NoAlternative,
                    }
                }
            }
            RefinementStage::FreshRepairedEndpoint => {
                let seal = prepared_fresh_pair.ok_or(RoleError::InvalidOutput)?;
                let observation = self.commit_value(work, ticket, output, physical)?;
                let mut next = self
                    .fresh_pending
                    .remove(&query.question_digest)
                    .ok_or(RoleError::InvalidOutput)?;
                let mut next_query = next.question.as_ref().clone();
                next_query.paired_repaired = Some(observation);
                next_query.paired_fresh_seal = Some(seal);
                next.question = Arc::new(next_query);
                fresh_pair = Some(seal);
                AdaptivePortTransition::Continue(self.blueprint_ready(next)?)
            }
            RefinementStage::FreshCounterEndpoint => {
                let counter = self.commit_value(work, ticket, output, physical)?;
                let repaired = query.paired_repaired.ok_or(RoleError::InvalidOutput)?;
                if self.publish_fresh_comparison(query, repaired, counter)? {
                    self.queue_strict_repair(work, query.context.prefix.clone())?;
                }
                fresh_pair = Some(FreshPairSeal::from_ticket(
                    ticket,
                    work.input.public_context,
                    self.engine.semantic.nodes[self.root]
                        .position
                        .side_to_move()
                        .into(),
                ));
                AdaptivePortTransition::Complete {
                    scope: FrameCompletionScope::CompletedFreshPair,
                }
            }
            _ => return Err(RoleError::Unsupported("stage is not a role completion").into()),
        };
        let mut transition = self.finish_branch(work, transition)?;
        transition.bind_role_cause(work, ticket)?;
        self.bind_deferred_recheck_role_cause(work, ticket)?;
        self.accepted.insert(ticket.id);
        self.committed_role_questions
            .insert(ticket.id, ticket.question);
        self.last_role_commit = Some((
            work.input.question_digest,
            ticket.id,
            observations_before,
            self.engine
                .semantic
                .stores
                .observations
                .next_observation_sequence(),
        ));
        if !source_preconsumed {
            self.counters.role_calls += 1;
            self.counters.consumed_role_outputs += 1;
            match query.context.purpose {
                RoleQueryPurpose::ValueFresh => {
                    self.counters.value_calls += 1;
                    self.counters.completed_value_calls += 1;
                    self.counters.accepted_value_outputs += 1;
                }
                RoleQueryPurpose::RepairPolicy => {
                    self.counters.repair_calls += 1;
                    self.counters.completed_repair_calls += 1;
                    self.counters.accepted_repair_outputs += 1;
                }
                RoleQueryPurpose::ReplyPolicy | RoleQueryPurpose::DivergencePolicy => {
                    self.counters.critic_calls += 1;
                    self.counters.completed_critic_calls += 1;
                    self.counters.accepted_critic_outputs += 1;
                }
                _ => {
                    self.counters.proposer_calls += 1;
                    self.counters.completed_proposer_calls += 1;
                    self.counters.accepted_proposer_outputs += 1;
                }
            }
        }
        self.retire_reciprocal_source(ticket.id);
        Ok(AdaptivePortStep {
            evidence,
            transition,
            fresh_pair,
            rules_pair,
            consumption_gate: Some((ticket.id, ticket.question, ticket.view.view_digest)),
            feedback_gate: None,
            queue_question: Some((work.frame, work.input.question_digest)),
        })
    }

    /// CPU report itself was already checked/committed by cpu_commit. This
    /// helper creates only the next exact question, never another CPU call.
    pub fn after_cpu(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        observation: ObservationId,
        _complete: bool,
    ) -> Result<AdaptivePortStep, PalsError> {
        self.check_work(work)?;
        let query = &work.input.question;
        let mut hash = Sha256::new();
        hash.update(b"rz-pals-accepted-cpu-evidence/1\0");
        hash.update(work.input.question_digest.0);
        hash.update((observation.0 as u64).to_le_bytes());
        let evidence = ContractDigest(hash.finalize().into());
        if !self.cpu_observation_consumed(work, observation)
            && !self.external_observation_consumed(work, observation)
            && !self.profiled_external_observation_consumed(work, observation)
        {
            return Ok(AdaptivePortStep {
                evidence,
                transition: self.finish_branch(
                    work,
                    AdaptivePortTransition::Unknown {
                        reason: "raw CPU observation was not consumed by this exact question",
                    },
                )?,
                fresh_pair: None,
                rules_pair: None,
                consumption_gate: None,
                feedback_gate: None,
                queue_question: None,
            });
        }
        let transition = if query.stage == RefinementStage::CpuDiscovery {
            let raw = self.engine.semantic.stores.observations.get(observation)?;
            let pv = raw
                .cpu_pv
                .map(|line| self.engine.semantic.stores.lines.moves(line))
                .transpose()?
                .unwrap_or_default();
            // A discovered restricted score is never assigned to its leaf. Only
            // exact legal line and immutable source observation enter memory.
            let alternative = !pv.is_empty()
                && query
                    .attack_ply
                    .and_then(|ply| query.proposal.get(ply))
                    .copied()
                    != pv.first().copied();
            if !alternative
                && !query.discovery_restricted
                && query.validator_root_moves.is_none()
                && query.position.legal_moves().iter().any(|movement| {
                    Some(*movement)
                        != query
                            .attack_ply
                            .and_then(|ply| query.proposal.get(ply))
                            .copied()
                })
            {
                let mut next = self.blueprint(
                    query.node,
                    RefinementStage::CpuDiscovery,
                    query.first,
                    query.context.prefix.clone(),
                    query.proposal.clone(),
                    None,
                    None,
                    query.attack_ply,
                    false,
                    None,
                    None,
                )?;
                next = next.with_restricted_discovery();
                AdaptivePortTransition::Continue(next)
            } else {
                let cpu_line = if alternative {
                    let mut line = query.context.prefix.clone();
                    let available = self.continuation_limit().saturating_sub(line.len());
                    line.extend(pv.iter().take(available));
                    self.replay_node(&line)?;
                    self.record_hot(RecordKind::Counterexample, &line, Some(observation))?;
                    Some(line)
                } else {
                    None
                };
                let mut next = self.blueprint(
                    query.node,
                    RefinementStage::CriticReply,
                    query.first,
                    query.context.prefix.clone(),
                    query.proposal.clone(),
                    cpu_line,
                    query.repaired.clone(),
                    query.attack_ply,
                    false,
                    None,
                    None,
                )?;
                next = next.with_counter_source(if alternative { Some(observation) } else { None });
                AdaptivePortTransition::Continue(next)
            }
        } else if query.stage == RefinementStage::RecheckCounterValidation {
            let pair = self.prepare_final_pair(
                work,
                query.repaired.clone().ok_or(RoleError::InvalidOutput)?,
                query.counter.clone().ok_or(RoleError::InvalidOutput)?,
            )?;
            let transition = self.finish_branch(work, pair.0)?;
            return Ok(AdaptivePortStep {
                evidence,
                transition,
                fresh_pair: None,
                rules_pair: pair.1,
                consumption_gate: None,
                feedback_gate: Some((work.input.question_digest, observation)),
                queue_question: Some((work.frame, work.input.question_digest)),
            });
        } else if self.engine.uses_model_resolver() {
            let stage = match query.stage {
                RefinementStage::CpuValidation => RefinementStage::ProposalValueFresh,
                RefinementStage::CounterValidation => RefinementStage::CounterValueFresh,
                RefinementStage::RepairValidation => RefinementStage::RepairValueFresh,
                _ => return Err(RoleError::InvalidOutput.into()),
            };
            let selection = self.prepare_zero_step_repair_feedback_selection(work)?;
            let mut blueprint = self.blueprint(
                query.node,
                stage,
                query.first,
                query.context.prefix.clone(),
                query.proposal.clone(),
                query.counter.clone(),
                query.repaired.clone(),
                query.attack_ply,
                false,
                None,
                None,
            )?;
            if selection.is_some() {
                blueprint = blueprint
                    .with_counter_source(query.counter_source)
                    .with_goal_lane_selection(selection)?;
            }
            AdaptivePortTransition::Continue(blueprint)
        } else {
            self.after_validation(query, Some(work))?
        };
        if matches!(
            transition,
            AdaptivePortTransition::Complete {
                scope: FrameCompletionScope::QueuedRecheck
            }
        ) {
            self.bind_deferred_recheck_feedback_cause(work, observation)?;
        }
        let transition = self.finish_branch(work, transition)?;
        Ok(AdaptivePortStep {
            evidence,
            transition,
            fresh_pair: None,
            rules_pair: None,
            consumption_gate: None,
            feedback_gate: Some((work.input.question_digest, observation)),
            queue_question: Some((work.frame, work.input.question_digest)),
        })
    }

    pub(super) fn after_validation(
        &mut self,
        query: &AdaptivePortQuestion,
        parent: Option<&CoordinatorWork<ReviewView, AdaptivePortQuestion>>,
    ) -> Result<AdaptivePortTransition, PalsError> {
        match query.stage {
            RefinementStage::CpuValidation | RefinementStage::ProposalValueFresh => {
                let proposal = if query.proposal.is_empty() {
                    query.context.prefix.clone()
                } else {
                    query.proposal.clone()
                };
                if self.divergence_indices(&proposal)?.is_empty() {
                    Ok(AdaptivePortTransition::Complete {
                        scope: FrameCompletionScope::NoDivergence,
                    })
                } else {
                    Ok(AdaptivePortTransition::Continue(self.blueprint(
                        self.root,
                        RefinementStage::CriticDivergence,
                        query.first,
                        Vec::new(),
                        proposal,
                        None,
                        None,
                        None,
                        false,
                        None,
                        None,
                    )?))
                }
            }
            RefinementStage::CounterValidation | RefinementStage::CounterValueFresh => {
                self.publish_supported_counter(query)?;
                let first = query.first.ok_or(RoleError::InvalidOutput)?;
                if !self.engine.admit_iterative_repair(first) {
                    return Ok(AdaptivePortTransition::Complete {
                        scope: FrameCompletionScope::RepairNotAdmitted,
                    });
                }
                let ply = query.attack_ply.ok_or(RoleError::InvalidOutput)?;
                let counter = query.counter.as_ref().ok_or(RoleError::InvalidOutput)?;
                let prefix = counter
                    .get(..=ply)
                    .ok_or(RoleError::InvalidOutput)?
                    .to_vec();
                let node = self.replay_node(&prefix)?;
                let mut transition = self.begin_repair_continuation(
                    node,
                    query.first,
                    prefix,
                    query.proposal.clone(),
                    counter.clone(),
                    ply,
                    query.counter_source,
                )?;
                if let Some(original) = parent {
                    self.bind_zero_step_repair_goal_transition(original, &mut transition)?;
                }
                Ok(transition)
            }
            RefinementStage::RepairValidation | RefinementStage::RepairValueFresh => {
                self.publish_supported_repair(query)?;
                if self.engine.post_repair_recheck == PostRepairRecheckPolicy::Disabled {
                    return Ok(AdaptivePortTransition::Complete {
                        scope: FrameCompletionScope::NoAlternative,
                    });
                }
                let checked = self.critic_recheck_after_repaired_fresh(query)?;
                if !matches!(checked, AdaptivePortTransition::Continue(_)) {
                    return Ok(checked);
                }
                let repaired = query.repaired.as_ref().ok_or(RoleError::InvalidOutput)?;
                let repaired_node = self.replay_node(repaired)?;
                let fixed_iterative = self.root_fanout == AdaptiveRootFanoutPolicy::SequentialFixed
                    && self.engine.post_repair_recheck
                        == PostRepairRecheckPolicy::IterativeFrozenModelWdlV2;
                if self.engine.semantic.nodes[repaired_node].terminal.is_some() && !fixed_iterative
                {
                    return Ok(checked);
                }
                let blueprint = self.blueprint(
                    repaired_node,
                    RefinementStage::RecheckRepairedValueFresh,
                    query.first,
                    repaired.clone(),
                    repaired.clone(),
                    query.counter.clone(),
                    Some(repaired.clone()),
                    query.attack_ply,
                    false,
                    None,
                    None,
                )?;
                if fixed_iterative {
                    let work = parent.ok_or(RoleError::InvalidOutput)?;
                    self.defer_fixed_recheck(work, blueprint)
                } else {
                    Ok(AdaptivePortTransition::Continue(blueprint))
                }
            }
            _ => Err(RoleError::InvalidOutput.into()),
        }
    }

    /// Mirrors synchronous `follow(Call::Repair)`: a full or terminal prefix
    /// is a zero-step Repair, not permission to append one more model move.
    /// The caller already admitted this Repair under the per-first-move cap.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn begin_repair_continuation(
        &mut self,
        node: usize,
        first: Option<BoardMove>,
        prefix: Vec<BoardMove>,
        proposal: Vec<BoardMove>,
        counter: Vec<BoardMove>,
        attack: usize,
        source: Option<ObservationId>,
    ) -> Result<AdaptivePortTransition, PalsError> {
        self.check_live()?;
        let first = first.ok_or(RoleError::InvalidOutput)?;
        let continuation_limit = self.continuation_limit();
        if prefix.len() > continuation_limit
            || proposal.len() > continuation_limit
            || counter.len() > continuation_limit
            || prefix.first() != Some(&first)
            || proposal.first() != Some(&first)
            || counter.first() != Some(&first)
            || counter.get(..=attack) != Some(prefix.as_slice())
            || proposal.get(..attack) != prefix.get(..attack)
            || proposal.get(attack) == counter.get(attack)
            || self.replay_node(&prefix)? != node
        {
            return Err(ReviewError::AnchorMismatch.into());
        }
        let zero_step = prefix.len() == continuation_limit
            || self.engine.semantic.nodes[node].terminal.is_some();
        if !zero_step {
            return Ok(AdaptivePortTransition::Continue(
                self.blueprint(
                    node,
                    RefinementStage::Repair,
                    Some(first),
                    prefix,
                    proposal,
                    Some(counter),
                    None,
                    Some(attack),
                    false,
                    None,
                    None,
                )?
                .with_counter_source(source),
            ));
        }

        // Retain the unchanged line's actual Repair record and genealogy. No
        // accepted Repair output, revised own move or new witness is invented.
        self.record_hot(RecordKind::Repair, &prefix, None)?;
        self.counters.repairs = self
            .counters
            .repairs
            .checked_add(1)
            .ok_or(PalsError::Capacity)?;
        if self.engine.semantic.nodes[node].terminal.is_some() {
            return Ok(AdaptivePortTransition::Complete {
                scope: FrameCompletionScope::NoRepairContinuation,
            });
        }
        let cached_cpu = self.has_reusable_completed_cpu(node, prefix.len())?;
        if cached_cpu {
            // One new semantic verification lookup consumes the already
            // accepted exact estimate. It adds no physical execution or record.
            self.counters.evidence_cache_hits = self
                .counters
                .evidence_cache_hits
                .checked_add(1)
                .ok_or(PalsError::Capacity)?;
            self.counters.consumed_cached_cpu_values = self
                .counters
                .consumed_cached_cpu_values
                .checked_add(1)
                .ok_or(PalsError::Capacity)?;
            if !self.engine.uses_model_resolver() {
                return Ok(AdaptivePortTransition::Complete {
                    scope: FrameCompletionScope::NoRepairContinuation,
                });
            }
        }
        let next = if cached_cpu {
            // A model-resolver value must still be Fresh for the new record
            // revision, even though own CPU verification used exact evidence.
            RefinementStage::RepairValueFresh
        } else {
            RefinementStage::RepairValidation
        };
        Ok(AdaptivePortTransition::Continue(
            self.blueprint(
                node,
                next,
                Some(first),
                prefix.clone(),
                proposal,
                Some(counter),
                Some(prefix),
                Some(attack),
                false,
                None,
                None,
            )?
            .with_counter_source(source),
        ))
    }

    /// The extra Fresh endpoint was already accepted. Continue to the actual
    /// opponent anchor without publishing Repair or enqueueing it a second time.
    pub(super) fn critic_recheck_after_repaired_fresh(
        &mut self,
        query: &AdaptivePortQuestion,
    ) -> Result<AdaptivePortTransition, PalsError> {
        let repaired = query.repaired.as_ref().ok_or(RoleError::InvalidOutput)?;
        let counter = query.counter.as_ref().ok_or(RoleError::InvalidOutput)?;
        let attack = query.attack_ply.ok_or(RoleError::InvalidOutput)?;
        let root_color = self.engine.semantic.nodes[self.root]
            .position
            .side_to_move();
        let checked = if self.rovent_frontier().is_some() {
            Some(self.existing_checked_frontier_line(repaired)?.ok_or(
                crate::pals::rovent_frontier::FrontierError::Invalid(
                    "repaired source has no original charged Rules path",
                ),
            )?)
        } else {
            None
        };
        let mut position = self.engine.semantic.nodes[self.root].position.clone();
        let mut changed = None;
        let mut anchor = None;
        for (ply, movement) in repaired.iter().enumerate() {
            if ply > attack
                && position.side_to_move() == root_color
                && counter.get(ply) != Some(movement)
                && changed.is_none()
            {
                changed = Some(ply);
            }
            if changed.is_some_and(|change| ply > change)
                && position.side_to_move() != root_color
                && matches!(
                    position.classify_position()?.play_status,
                    PlayStatus::Ongoing
                )
            {
                anchor = Some(ply);
                break;
            }
            if let Some(checked) = &checked {
                position =
                    Position::from_snapshot(checked.steps()[ply].snapshot(), Default::default())?;
            } else {
                position.make_move(*movement)?;
            }
        }
        if let Some(anchor) = anchor {
            let prefix = repaired[..anchor].to_vec();
            let node = self.replay_node(&prefix)?;
            Ok(AdaptivePortTransition::Continue(self.blueprint(
                node,
                RefinementStage::CriticRecheck,
                query.first,
                prefix,
                repaired.clone(),
                Some(counter.clone()),
                Some(repaired.clone()),
                Some(attack),
                false,
                None,
                None,
            )?))
        } else {
            Ok(AdaptivePortTransition::Complete {
                scope: FrameCompletionScope::NoAlternative,
            })
        }
    }

    pub(super) fn record_hot(
        &mut self,
        kind: RecordKind,
        line: &[BoardMove],
        cpu: Option<ObservationId>,
    ) -> Result<Option<(LineId, ObservationId)>, PalsError> {
        self.record_hot_with_cpu_value(kind, line, cpu, None, None)
    }

    /// Publish the same model-facing verification projection as the legacy
    /// own checker, after its independently retained raw task was consumed.
    pub(super) fn record_cpu_verification_hot(
        &mut self,
        line: &[BoardMove],
        observation: ObservationId,
        score: i32,
        depth: u16,
        scope: CpuScoreScope,
        perspective: Color,
    ) -> Result<(), PalsError> {
        self.record_hot_with_cpu_value(
            RecordKind::CpuVerification,
            line,
            Some(observation),
            Some((score, depth, scope, perspective)),
            None,
        )?;
        Ok(())
    }

    fn record_hot_with_cpu_value(
        &mut self,
        kind: RecordKind,
        line: &[BoardMove],
        cpu: Option<ObservationId>,
        value: Option<(i32, u16, CpuScoreScope, Color)>,
        repair_publication_sink: Option<&mut Option<AdaptivePublishedRepairLine>>,
    ) -> Result<Option<(LineId, ObservationId)>, PalsError> {
        if value.is_some() && kind != RecordKind::CpuVerification {
            return Err(StoreError::InvalidEvidence("CPU verification projection kind").into());
        }
        if self.engine.semantic.records.len() >= self.engine.config.max_records {
            // Raw/failed evidence is not deleted on the manager to keep running.
            if let Some(frontier) = self.engine.semantic.rovent_frontier.as_deref_mut() {
                return Err(frontier.refuse_public_record_capacity().into());
            }
            return Err(
                StoreError::PinSaturated("adaptive public records need archive worker").into(),
            );
        }
        let root_state = self.engine.semantic.nodes[self.root].state;
        let revision = self
            .engine
            .semantic
            .revision
            .checked_add(1)
            .ok_or(PalsError::Capacity)?;
        let supersedes_revision = self
            .engine
            .semantic
            .records
            .iter()
            .rev()
            .find(|record| {
                record.origin_state == root_state
                    && record.kind == kind
                    && record.line.first() == line.first()
            })
            .map(|record| record.revision);
        let parent_revision = self
            .engine
            .semantic
            .records
            .iter()
            .rev()
            .find(|record| {
                record.origin_state == root_state
                    && record.line.first() == line.first()
                    && match kind {
                        RecordKind::Repair => record.kind == RecordKind::Counterexample,
                        RecordKind::Counterexample => {
                            matches!(record.kind, RecordKind::Proposal | RecordKind::Repair)
                        }
                        RecordKind::CpuVerification => record.line == line,
                        RecordKind::Proposal => false,
                    }
            })
            .map(|record| record.revision);
        if self.rovent_frontier().is_some() {
            self.replay_node(line)?;
        } else {
            let mut replay = self.engine.semantic.nodes[self.root].position.clone();
            for movement in line {
                replay.make_move(*movement)?;
            }
        }
        let record = RoleRecord {
            revision,
            parent_revision,
            supersedes_revision,
            origin_state: root_state,
            kind,
            line: line.to_vec(),
            value: value.map(|value| value.0),
            completed_depth: value.map_or(0, |value| value.1),
            score_scope: value.map(|value| value.2),
            cpu_observation: cpu,
            perspective: value.map_or(
                self.engine.semantic.nodes[self.root]
                    .position
                    .side_to_move(),
                |value| value.3,
            ),
            critical: true,
        };
        if cpu.is_some() || kind == RecordKind::CpuVerification {
            self.publish_hot_record(record, None)?;
            return Ok(None);
        }
        let focus = self
            .engine
            .semantic
            .stores
            .situations
            .get(self.engine.semantic.nodes[self.root].situation)?
            .focus;
        let line = self.engine.semantic.stores.hot_append_line(focus, line)?;
        let observation = self
            .engine
            .semantic
            .stores
            .hot_append_observation(Observation {
                state: root_state,
                line: Some(line),
                source: stable_id(self.engine.semantic.model.identity()),
                epoch: 0,
                scope: EvidenceScope::Model {
                    model: stable_id(self.engine.semantic.model.identity()),
                    encoding: stable_id("pals-selected-review-v3"),
                    input: revision,
                },
                value_identity: None,
                checker_identity: None,
                checker_work: None,
                external_report: None,
                model_value_identity: None,
                model_value_input: None,
                cpu_condition: None,
                cpu_pv: None,
                score: RawScore::Unknown,
                budget: 0,
                kind: match kind {
                    RecordKind::Proposal => ObservationKind::Proposal,
                    RecordKind::Counterexample => ObservationKind::Refutation,
                    RecordKind::Repair => ObservationKind::Repair,
                    RecordKind::CpuVerification => ObservationKind::Proposal,
                },
                supersedes: None,
                execution: None,
            })?;
        self.engine.pin_archive_active_record(line, observation)?;
        // Copy only scalar fields of the record that this owner is about to
        // publish. No additional line clone, lookup/replay or digest is made.
        let copied_repair_line = (repair_publication_sink.is_some() && kind == RecordKind::Repair)
            .then(|| AdaptivePublishedRepairLine {
                line,
                observation,
                record_revision: record.revision,
                parent_revision: record.parent_revision,
                supersedes_revision: record.supersedes_revision,
                origin_state: record.origin_state,
                committed_plies: record.line.len(),
                first_move: record.line.first().copied(),
            });
        self.publish_hot_record(record, Some(observation))?;
        // Only the successful original publication releases this observation.
        // A later commit error still returns unchanged, alongside Rejected.
        if let Some(destination) = repair_publication_sink {
            *destination = copied_repair_line;
        }
        Ok(Some((line, observation)))
    }

    /// Only the original bounded bundle producer can close this root's
    /// review-index admission. Failed observations/lines remain with their
    /// original Store and output owners until the existing physical drain.
    fn publish_hot_record(
        &mut self,
        record: RoleRecord,
        evidence: Option<ObservationId>,
    ) -> Result<(), PalsError> {
        match self.engine.publish_record(record, None, evidence) {
            Err(PalsError::Review(ReviewError::BundleRecordCapacity)) => {
                match self.engine.semantic.rovent_frontier.as_deref_mut() {
                    Some(frontier) => Err(frontier.refuse_public_review_index_capacity().into()),
                    None => Err(ReviewError::BundleRecordCapacity.into()),
                }
            }
            result => result,
        }
    }

    pub(super) fn checked_role_value(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
        physical: Option<AdaptivePortRoleEvidence>,
    ) -> Result<ModelValueOutput, PalsError> {
        self.checked_value_input(&work.input, ticket, output, physical)
    }

    pub(super) fn checked_value_input(
        &self,
        input: &FrameInput<ReviewView, AdaptivePortQuestion>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
        physical: Option<AdaptivePortRoleEvidence>,
    ) -> Result<ModelValueOutput, PalsError> {
        AdaptivePortQuestion::validate_frontier_input(input, &self.engine.semantic.stores)?;
        self.verify_registered_model_namespace(ticket, output)?;
        let physical = physical.ok_or(RoleError::InvalidOutput)?;
        if ticket.authority != self.authority
            || ticket.original_deadline != self.deadline
            || ticket.view != input.view_seal
            || output.id != ticket.id
            || output.authority != ticket.authority
            || ticket.role != Role::Proposer
            || ticket.question != input.question_digest
            || ticket.representation != output.key
            || input.question.context.purpose != RoleQueryPurpose::ValueFresh
            || physical.physical_fresh != Some(true)
            || !physical.measurement_complete
            // Fresh WDL needs a known completed evaluation. Its physical graph
            // input count belongs to the producer, not this semantic output:
            // e.g. public cache miss + role graph completes two NN inputs.
            || physical.completed_nn_inputs.is_none_or(|inputs| inputs == 0)
            || physical.execution.is_none()
        {
            return Err(RoleError::InvalidOutput.into());
        }
        validate_payload(RoleQueryPurpose::ValueFresh, &input.view, output)?;
        let query = &input.question;
        let RolePayload::Proposal { wdl, .. } = output.payload else {
            return Err(RoleError::InvalidOutput.into());
        };
        let identity = self
            .engine
            .semantic
            .model_registered_value
            .clone()
            .ok_or(RoleError::Unavailable)?;
        if self
            .engine
            .semantic
            .model
            .registered_representation_namespace()
            .is_some_and(|namespace| namespace.value_identity() != &identity)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let value = ModelValueOutput {
            identity: identity.clone(),
            input_sha256: output.key.input.0,
            state: query.position.position_identity(),
            perspective: query.position.side_to_move(),
            wdl: wdl.probabilities(),
        };
        value.validate(&query.position, &identity)?;
        Ok(value)
    }

    fn commit_value(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
        physical: Option<AdaptivePortRoleEvidence>,
    ) -> Result<ObservationId, PalsError> {
        self.commit_value_input(&work.input, ticket, output, physical)
    }

    pub(super) fn commit_value_input(
        &mut self,
        input: &FrameInput<ReviewView, AdaptivePortQuestion>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
        physical: Option<AdaptivePortRoleEvidence>,
    ) -> Result<ObservationId, PalsError> {
        // Recheck the immutable producer namespace at the actual Store commit
        // boundary, even though admission already validated it before attempt.
        let value = self.checked_value_input(input, ticket, output, physical)?;
        let identity = value.identity.clone();
        let query = &input.question;
        let observation = self
            .engine
            .semantic
            .stores
            .hot_append_observation(Observation {
                state: query.context.state,
                line: None,
                source: stable_id(&identity.semantics),
                epoch: 0,
                scope: EvidenceScope::Model {
                    model: stable_id(&identity.model),
                    encoding: stable_id(&identity.encoding),
                    input: u64::from_le_bytes(
                        output.key.input.0[..8]
                            .try_into()
                            .map_err(|_| RoleError::InvalidOutput)?,
                    ),
                },
                value_identity: None,
                checker_identity: None,
                checker_work: None,
                external_report: None,
                model_value_identity: Some(identity),
                model_value_input: Some(output.key.input.0),
                cpu_condition: None,
                cpu_pv: None,
                score: RawScore::ContextWdl {
                    win: value.wdl[0],
                    draw: value.wdl[1],
                    loss: value.wdl[2],
                    perspective: value.perspective,
                    context_revision: query.context.public_revision,
                },
                budget: 1,
                kind: ObservationKind::Proposal,
                supersedes: None,
                execution: None,
            })?;
        self.engine
            .semantic
            .stores
            .hot_add_dependency(observation, query.context.situation)?;
        self.engine
            .semantic
            .stores
            .hot_add_dependency(observation, self.engine.semantic.nodes[self.root].situation)?;
        self.engine.semantic.nodes[query.node].model_value = Some(value);
        self.engine.semantic.nodes[query.node].model_observation = Some(observation);
        self.engine.semantic.nodes[query.node].model_value_revision =
            Some(query.context.public_revision);
        Ok(observation)
    }

    pub(super) fn prepare_final_pair(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        repaired: Vec<BoardMove>,
        counter: Vec<BoardMove>,
    ) -> Result<(AdaptivePortTransition, Option<AdaptiveRulesPairProof>), PalsError> {
        let query = &work.input.question;
        let repaired_node = self.replay_node(&repaired)?;
        let counter_node = self.replay_node(&counter)?;
        match (
            self.engine.semantic.nodes[repaired_node].terminal,
            self.engine.semantic.nodes[counter_node].terminal,
        ) {
            (Some(_), Some(_)) => {
                // Exact endpoints can be compared through Rules, but this alone
                // does not certify every defense or fabricate NN executions.
                let perspective = self.engine.semantic.nodes[self.root]
                    .position
                    .side_to_move();
                let old = self.engine.endpoint_expectation(repaired_node, perspective);
                let new = self.engine.endpoint_expectation(counter_node, perspective);
                if matches!((old,new),(Some(old),Some(new)) if new<old) {
                    self.record_hot(RecordKind::Counterexample, &counter, None)?;
                    self.queue_strict_repair(work, counter.clone())?;
                }
                let proof = AdaptiveRulesPairProof {
                    authority: self.authority,
                    root_state: work.input.view_seal.root_state,
                    public_context: work.input.public_context,
                    submitted_question: work.input.question_digest,
                    repaired_line: repaired,
                    counter_line: counter,
                    repaired: Arc::new(
                        self.issue_rules_position(
                            self.engine.semantic.nodes[repaired_node].position.clone(),
                        )?
                        .export()
                        .map_err(|_| RoleError::InvalidOutput)?,
                    ),
                    counter: Arc::new(
                        self.issue_rules_position(
                            self.engine.semantic.nodes[counter_node].position.clone(),
                        )?
                        .export()
                        .map_err(|_| RoleError::InvalidOutput)?,
                    ),
                    perspective: perspective.into(),
                };
                proof.validate(work)?;
                Ok((
                    AdaptivePortTransition::Complete {
                        scope: FrameCompletionScope::CompletedRulesPair,
                    },
                    Some(proof),
                ))
            }
            (Some(_), None) | (None, Some(_)) => Ok((
                AdaptivePortTransition::Unknown {
                    reason: "mixed terminal and model Fresh pair",
                },
                None,
            )),
            (None, None) => {
                let blueprint = self.blueprint_fresh_pair(
                    query,
                    repaired_node,
                    repaired,
                    counter_node,
                    counter,
                )?;
                Ok((AdaptivePortTransition::Continue(blueprint), None))
            }
        }
    }
    fn publish_fresh_comparison(
        &mut self,
        query: &AdaptivePortQuestion,
        repaired: ObservationId,
        counter: ObservationId,
    ) -> Result<bool, PalsError> {
        let before = self.engine.semantic.stores.observations.get(repaired)?;
        let after = self.engine.semantic.stores.observations.get(counter)?;
        if before.model_value_identity != after.model_value_identity
            || before.model_value_identity != self.engine.semantic.model_registered_value
            || before.model_value_input.is_none()
            || after.model_value_input.is_none()
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let perspective = self.engine.semantic.nodes[self.root]
            .position
            .side_to_move();
        let expected = |observation: &Observation| -> Result<(f32, u64), PalsError> {
            if let RawScore::ContextWdl {
                win,
                loss,
                perspective: side,
                context_revision,
                ..
            } = observation.score
            {
                Ok((
                    if side == perspective {
                        win - loss
                    } else {
                        loss - win
                    },
                    context_revision,
                ))
            } else {
                Err(RoleError::InvalidOutput.into())
            }
        };
        let (before_value, before_revision) = expected(before)?;
        let (after_value, after_revision) = expected(after)?;
        if before_revision != after_revision {
            return Err(RoleError::InvalidOutput.into());
        }
        if after_value >= before_value {
            self.counters.frozen_wdl_unresolved += 1;
            return Ok(false);
        }
        if self.conditional_witnesses.len() >= 64 {
            return Err(PalsError::Capacity);
        }
        let repaired_line = query.repaired.as_ref().ok_or(RoleError::InvalidOutput)?;
        let root = self.engine.semantic.nodes[self.root].situation;
        let focus = self.engine.semantic.stores.situations.get(root)?.focus;
        let line = self
            .engine
            .semantic
            .stores
            .hot_append_line(focus, repaired_line)?;
        let evidence = self
            .engine
            .semantic
            .stores
            .hot_append_observation(Observation {
                state: self.engine.semantic.nodes[self.root].state,
                line: Some(line),
                source: stable_id(self.engine.search_identity()),
                epoch: 0,
                scope: EvidenceScope::Model {
                    model: stable_id(self.engine.semantic.model.identity()),
                    encoding: stable_id("pals-frozen-pair-v3"),
                    input: before_revision,
                },
                value_identity: None,
                checker_identity: None,
                checker_work: None,
                external_report: None,
                model_value_identity: None,
                model_value_input: None,
                cpu_condition: None,
                cpu_pv: None,
                score: RawScore::ConditionalWdl {
                    expectation: after_value,
                    perspective,
                    context_revision: before_revision,
                    repaired,
                    counter,
                },
                budget: 0,
                kind: ObservationKind::Refutation,
                supersedes: None,
                execution: None,
            })?;
        self.engine
            .semantic
            .stores
            .hot_refute_continuation(root, line, evidence)?;
        self.conditional_witnesses.insert(
            evidence,
            (
                ObservationKind::Refutation,
                query.context.prefix.clone(),
                None,
            ),
        );
        self.counters.supported_refutations += 1;
        Ok(true)
    }

    pub fn publish_closed(&mut self, closed: &ClosedFrame) -> Result<(), PalsError> {
        if closed.id.authority != self.authority || self.closed.contains(&closed.id) {
            return Err(StoreError::StaleConsumer.into());
        }
        if let Some((movement, _)) = self
            .round_candidates
            .values()
            .find(|(_, frame)| *frame == Some(closed.id))
        {
            let first = BoardMove::try_from(
                closed
                    .original_first_move
                    .decode()
                    .map_err(|_| RoleError::InvalidOutput)?,
            )
            .map_err(|_| RoleError::InvalidOutput)?;
            if first != *movement {
                return Err(RoleError::InvalidOutput.into());
            }
            self.last_main_closed = Some(closed.id);
        }
        self.closed.insert(closed.id);
        Ok(())
    }
    pub fn result(&self, completion: PalsCompletion) -> Result<PalsResult, PalsError> {
        let position = &self.engine.semantic.nodes[self.root].position;
        if self.engine.semantic.nodes[self.root].terminal.is_some() {
            return self
                .engine
                .unsearched_result(position, PalsCompletion::Terminal, self.started);
        }
        let legal: Vec<_> = position
            .legal_moves()
            .into_iter()
            .filter(|movement| self.root_move_allowed(*movement))
            .collect();
        let mut counters = self.counters;
        if self.engine.uses_model_resolver() {
            let result = self.engine.model_result(
                self.root,
                &legal,
                completion,
                self.started,
                &mut counters,
            )?;
            return self.apply_rovent_mate_strategy(result);
        }
        let mut best = legal.first().copied();
        let mut score = None;
        let mut scope = PalsValueScope::Unknown;
        let mut root_values = Vec::with_capacity(legal.len());
        for movement in legal {
            let edge = self.engine.semantic.nodes[self.root]
                .edges
                .iter()
                .find(|edge| edge.movement == movement);
            let value = edge
                .and_then(|edge| {
                    self.engine
                        .value(edge.child, self.engine.config.line_plies.saturating_sub(1))
                })
                .map(|value| -value);
            let child_scope = edge.map_or(PalsValueScope::Unknown, |edge| {
                if self.engine.semantic.nodes[edge.child].terminal.is_some() {
                    PalsValueScope::RulesTerminal
                } else if value.is_some() {
                    PalsValueScope::RestrictedEstimate
                } else {
                    PalsValueScope::Unknown
                }
            });
            if let Some(value) = value
                && better_root_choice(value, child_scope, score, scope)
            {
                best = Some(movement);
                score = Some(value);
                scope = child_scope;
            }
            let resolved_value = value.map_or(PalsResolvedValue::Unknown, |value| {
                PalsResolvedValue::OwnedRaw {
                    value,
                    perspective: position.side_to_move(),
                }
            });
            root_values.push(PalsRootValue {
                movement,
                score: value,
                resolved_value,
                scope: child_scope,
                examined_replies: edge
                    .map_or(0, |edge| self.engine.semantic.nodes[edge.child].edges.len()),
                unexplored_replies: edge.map(|edge| {
                    self.engine.semantic.nodes[edge.child]
                        .position
                        .legal_moves()
                        .len()
                        .saturating_sub(self.engine.semantic.nodes[edge.child].edges.len())
                }),
            });
        }
        counters.retained_situations = self.engine.semantic.nodes.len();
        counters.root_scope_observation_complete = true;
        self.apply_rovent_mate_strategy(PalsResult {
            best_move: best,
            score,
            resolved_value: score.map_or(PalsResolvedValue::Unknown, |value| {
                PalsResolvedValue::OwnedRaw {
                    value,
                    perspective: position.side_to_move(),
                }
            }),
            model_value_identity: self.engine.semantic.model_registered_value.clone(),
            checker_identity: self.engine.researcher.checker_registered_identity.clone(),
            value_scope: scope,
            terminal: None,
            completion,
            counters,
            root_values,
            elapsed: self.started.elapsed(),
            model_identity: self.engine.semantic.model.identity().to_owned(),
            resolver_version: self.engine.resolver_version(),
        })
    }
}
pub(super) fn rank_moves(
    legal: &[BoardMove],
    output: &RoleOutput,
) -> Result<Vec<BoardMove>, PalsError> {
    let logits = match &output.payload {
        RolePayload::Proposal {
            candidate_logits, ..
        }
        | RolePayload::Counterexample {
            candidate_logits, ..
        } => candidate_logits,
        _ => return Err(RoleError::InvalidOutput.into()),
    };
    if logits.len() != legal.len() {
        return Err(RoleError::InvalidOutput.into());
    }
    let mut indices = (0..legal.len()).collect::<Vec<_>>();
    indices.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]).then(a.cmp(b)));
    Ok(indices.into_iter().map(|index| legal[index]).collect())
}
fn role_output_digest(ticket: &RoleTicketEnvelope, output: &RoleOutput) -> ContractDigest {
    let mut hash = Sha256::new();
    hash.update(b"rz-pals-consumed-role-proof/1\0");
    hash.update(ticket.question.0);
    hash.update(ticket.view.view_digest.0);
    hash.update(output.key.input.0);
    hash.update(output.id.sequence.to_le_bytes());
    let mut values = Vec::new();
    match &output.payload {
        RolePayload::Proposal {
            candidate_logits,
            wdl,
        } => {
            values.extend_from_slice(candidate_logits);
            values.extend_from_slice(&wdl.probabilities());
        }
        RolePayload::Counterexample {
            candidate_logits,
            divergence_logits,
            wdl,
        } => {
            values.extend_from_slice(candidate_logits);
            values.extend_from_slice(divergence_logits);
            values.extend_from_slice(&wdl.probabilities());
        }
        RolePayload::TaskRanking { task_logits } => values.extend_from_slice(task_logits),
    }
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    ContractDigest(hash.finalize().into())
}
