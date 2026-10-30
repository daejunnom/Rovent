//! Bounded post-Repair work with one frozen Fresh-WDL namespace.
//! CPU checker reports discover lines; they never supply this comparison.
use super::*;

pub(super) const MAX_REPAIRS_PER_FIRST: u8 = 3;
pub(super) const MAX_PENDING_REPAIRS: usize = 64;

#[derive(Clone)]
pub(super) struct RepairWork {
    pub root: usize,
    pub original_first: BoardMove,
    pub attack_ply: usize,
    pub repaired_line: LineId,
    pub repair_record_revision: u64,
    pub repaired_leaf: usize,
    pub repaired: Vec<BoardMove>,
    pub refutation: Vec<BoardMove>,
}

struct NewFrontier {
    prefix: Vec<BoardMove>,
    response_node: usize,
    counterline: Vec<BoardMove>,
    attack_ply: usize,
    counter_leaf: usize,
    publication: Option<ObservationId>,
}

#[derive(Default)]
struct IterativeRepairObservation {
    comparison_attempted: bool,
    comparable: bool,
    publication: Option<ObservationId>,
}

impl<M: RoleModel> PalsEngine<M> {
    pub(super) fn admit_iterative_repair(&mut self, movement: BoardMove) -> bool {
        if self.post_repair_recheck != PostRepairRecheckPolicy::IterativeFrozenModelWdlV2 {
            return true;
        }
        let count = self.semantic.repair_counts.entry(movement).or_default();
        if *count >= MAX_REPAIRS_PER_FIRST {
            return false;
        }
        *count += 1;
        true
    }

    pub(super) fn queue_or_recheck_frozen(
        &mut self,
        work: RepairWork,
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
        progress: &mut ObservedProgress<'_>,
    ) -> Result<(), PalsError> {
        if self.post_repair_recheck == PostRepairRecheckPolicy::IterativeFrozenModelWdlV2 {
            let scope = (work.root, work.original_first);
            if self.enqueue_frozen_repair(work, counters)?
                && self.stopped(limits, cancel).is_none()
                && self.held_cpu_repair_priority(scope.0, scope.1, limits)?
            {
                // This selected S lane observes the accepted Repair's C tail,
                // actual original-task resume and finish before another branch
                // can publish P. Other pending first moves retain their order.
                self.drain_repair_queue_scope(Some(scope), limits, cancel, counters, progress)?;
            }
        } else {
            self.frozen_recheck_once(&work, limits, cancel, counters, progress)?;
        }
        Ok(())
    }

    /// Queue admission never invokes a role or drains recursively. In
    /// particular, iterative followups share the current bounded drain loop.
    fn enqueue_frozen_repair(
        &mut self,
        work: RepairWork,
        counters: &mut PalsCounters,
    ) -> Result<bool, PalsError> {
        let question = (
            work.original_first,
            work.repair_record_revision,
            work.repaired.clone(),
        );
        if !self.semantic.repair_questions.insert(question) {
            counters.repair_queue_deduplicated += 1;
            return Ok(false);
        }
        if self.semantic.repair_queue.len() >= MAX_PENDING_REPAIRS {
            return Err(PalsError::Capacity);
        }
        self.semantic.repair_queue.push_back(work);
        counters.repair_queue_enqueued += 1;
        counters.repair_queue_peak = counters
            .repair_queue_peak
            .max(self.semantic.repair_queue.len());
        Ok(true)
    }

    pub(super) fn drain_repair_queue(
        &mut self,
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
        progress: &mut ObservedProgress<'_>,
    ) -> Result<(), PalsError> {
        self.drain_repair_queue_scope(None, limits, cancel, counters, progress)
    }

    fn drain_repair_queue_scope(
        &mut self,
        priority: Option<(usize, BoardMove)>,
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
        progress: &mut ObservedProgress<'_>,
    ) -> Result<(), PalsError> {
        while self.stopped(limits, cancel).is_none() {
            let next = if let Some((root, first)) = priority {
                if !self.held_cpu_repair_priority(root, first, limits)? {
                    break;
                }
                self.semantic
                    .repair_queue
                    .iter()
                    .position(|work| work.root == root && work.original_first == first)
                    .and_then(|index| self.semantic.repair_queue.remove(index))
            } else {
                self.semantic.repair_queue.pop_front()
            };
            let Some(work) = next else {
                break;
            };
            let Some(frontier) =
                self.frozen_recheck_once(&work, limits, cancel, counters, progress)?
            else {
                counters.repair_queue_without_new_evidence += 1;
                continue;
            };
            if !self.admit_iterative_repair(work.original_first) {
                continue;
            }
            if self.cpu_budget_used(counters) >= limits.max_cpu_nodes {
                break;
            }
            let before = counters.accepted_repair_outputs;
            let mut repaired = frontier.prefix.clone();
            let repaired_leaf = self.follow(
                frontier.response_node,
                &mut repaired,
                &work.repaired,
                Some(&frontier.counterline),
                Call::Repair,
                limits,
                cancel,
                counters,
            )?;
            if repaired.first() != Some(&work.original_first) {
                return Err(StoreError::InvalidEvidence("queued Repair changed first move").into());
            }
            counters.repairs += 1;
            let Some((repaired_line, _)) = self.record_checked(
                RecordKind::Repair,
                &repaired,
                None,
                0,
                None,
                None,
                limits,
                cancel,
            )?
            else {
                continue;
            };
            let revision = self.semantic.revision;
            if counters.accepted_repair_outputs > before {
                self.observe_accepted_repair(
                    work.root,
                    work.original_first,
                    repaired_line,
                    revision,
                );
            }
            self.verify(repaired_leaf, &repaired, limits, cancel, counters)?;
            self.publish_iterative_repair(
                &work,
                &frontier,
                repaired_line,
                repaired_leaf,
                &repaired,
                limits,
                cancel,
                counters,
            )?;
            self.publish_choice(work.root, limits, cancel, progress)?;
            if counters.accepted_repair_outputs > before && repaired != work.repaired {
                self.enqueue_frozen_repair(
                    RepairWork {
                        root: work.root,
                        original_first: work.original_first,
                        attack_ply: frontier.attack_ply,
                        repaired_line,
                        repair_record_revision: revision,
                        repaired_leaf,
                        repaired,
                        refutation: frontier.counterline,
                    },
                    counters,
                )?;
            } else {
                counters.repair_queue_without_new_evidence += 1;
            }
        }
        Ok(())
    }

    /// A new P line is not a repair proof by itself. Compare its endpoint with
    /// the actual C counter under one Fresh context, then supersede only the
    /// still-active conditional refutation. CPU CP/mate is never converted here.
    #[allow(clippy::too_many_arguments)]
    fn publish_iterative_repair(
        &mut self,
        work: &RepairWork,
        frontier: &NewFrontier,
        repaired_line: LineId,
        repaired_leaf: usize,
        repaired: &[BoardMove],
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
    ) -> Result<bool, PalsError> {
        let record_revision = self.semantic.revision;
        let mut observed = IterativeRepairObservation::default();
        let result = self.publish_iterative_repair_impl(
            work,
            frontier,
            repaired_line,
            repaired_leaf,
            repaired,
            limits,
            cancel,
            counters,
            &mut observed,
        );
        let comparison = self.observe_frozen_comparison(
            frontier.counter_leaf,
            Some(repaired_leaf),
            self.semantic.nodes[work.root].position.side_to_move(),
            observed.comparison_attempted,
            observed.comparable,
            observed.publication,
            result.as_ref().err(),
        );
        self.observe_iterative_repair_comparison(record_revision, comparison);
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn publish_iterative_repair_impl(
        &mut self,
        work: &RepairWork,
        frontier: &NewFrontier,
        repaired_line: LineId,
        repaired_leaf: usize,
        repaired: &[BoardMove],
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
        observed: &mut IterativeRepairObservation,
    ) -> Result<bool, PalsError> {
        let Some(previous) = frontier.publication else {
            return Ok(false);
        };
        if self.stopped(limits, cancel).is_some() || repaired_line == work.repaired_line {
            return Ok(false);
        }
        self.validate_checker_namespace()?;
        let root = &self.semantic.nodes[work.root];
        let root_state = root.state;
        let root_situation = root.situation;
        let root_color = root.position.side_to_move();
        let generation = self.semantic.stores.generation();
        let current = self.semantic.stores.situations.get(root_situation)?;
        let root_revision = current.revision;
        if self.semantic.stores.root() != Some(root_situation)
            || repaired.first() != Some(&work.original_first)
            || self.semantic.stores.lines.get(repaired_line)?.start_state != root_state
            || self.semantic.stores.lines.moves(repaired_line)? != repaired
            || current
                .conclusions
                .get(work.repaired_line)
                .is_none_or(|conclusion| {
                    conclusion.status != ContinuationStatus::Refuted
                        || conclusion.evidence != Some(previous)
                })
        {
            return Err(StoreError::StaleConsumer.into());
        }
        let context_revision = self.semantic.revision;
        let terminals = (
            self.semantic.nodes[frontier.counter_leaf].terminal,
            self.semantic.nodes[repaired_leaf].terminal,
        );
        match terminals {
            (None, None) => {
                observed.comparison_attempted = true;
                self.evaluate_model_value_pair(
                    frontier.counter_leaf,
                    &frontier.counterline,
                    repaired_leaf,
                    repaired,
                    limits,
                    cancel,
                    counters,
                )?;
                if self.semantic.nodes[frontier.counter_leaf].model_value_revision
                    != Some(context_revision)
                    || self.semantic.nodes[repaired_leaf].model_value_revision
                        != Some(context_revision)
                {
                    counters.frozen_wdl_unresolved += 1;
                    return Ok(false);
                }
            }
            (Some(_), Some(_)) => observed.comparison_attempted = true,
            _ => {
                counters.frozen_wdl_unresolved += 1;
                return Ok(false);
            }
        }
        if self.stopped(limits, cancel).is_some() {
            return Ok(false);
        }
        if self.semantic.revision != context_revision
            || self.semantic.stores.root() != Some(root_situation)
            || self.semantic.stores.generation() != generation
            || self
                .semantic
                .stores
                .situations
                .get(root_situation)?
                .revision
                != root_revision
        {
            return Err(StoreError::StaleConsumer.into());
        }
        self.validate_checker_namespace()?;
        let Some((before, after)) = self
            .endpoint_expectation(frontier.counter_leaf, root_color)
            .zip(self.endpoint_expectation(repaired_leaf, root_color))
        else {
            counters.frozen_wdl_unresolved += 1;
            return Ok(false);
        };
        observed.comparable = true;
        counters.frozen_wdl_comparisons += 1;
        if after <= before {
            return Ok(false);
        }
        let score = if terminals.0.is_some() {
            // Both endpoints are exact Rules terminals. This derived line
            // conclusion remains conditional Model scope, not a mate proof.
            RawScore::Estimate {
                value: after,
                perspective: root_color,
            }
        } else {
            RawScore::ConditionalRepairWdl {
                expectation: after,
                perspective: root_color,
                before: self.semantic.nodes[frontier.counter_leaf]
                    .model_observation
                    .ok_or(RoleError::Unavailable)?,
                after: self.semantic.nodes[repaired_leaf]
                    .model_observation
                    .ok_or(RoleError::Unavailable)?,
                context_revision,
            }
        };
        let evidence = self.append_engine_observation_with_controls(
            Observation {
                state: root_state,
                line: Some(repaired_line),
                source: stable_id(self.search_identity()),
                epoch: 0,
                scope: EvidenceScope::Model {
                    model: stable_id(self.semantic.model.identity()),
                    encoding: stable_id("pals-frozen-wdl-conditional-repair/2"),
                    input: context_revision,
                },
                score,
                value_identity: None,
                checker_identity: None,
                checker_work: None,
                external_report: None,
                model_value_identity: None,
                model_value_input: None,
                cpu_condition: None,
                cpu_pv: None,
                budget: 0,
                kind: ObservationKind::Repair,
                supersedes: Some(previous),
                execution: None,
            },
            Some((limits.deadline, cancel)),
        )?;
        self.archive_repair(
            root_situation,
            work.repaired_line,
            repaired_line,
            evidence,
            limits,
            cancel,
        )?;
        observed.publication = Some(evidence);
        counters.supported_repairs += 1;
        Ok(true)
    }

    fn frozen_endpoint<'a>(
        node: &'a Node,
        stores: &'a PalsStores,
        snapshot: &'a PositionSnapshot,
        _revision: u64,
    ) -> RecheckEndpoint<'a> {
        let evidence = if let Some((reason, value)) = node.terminal {
            RecheckEndpointEvidence::RulesTerminal {
                reason,
                value,
                perspective: node.position.side_to_move(),
            }
        } else if let (Some(output), Some(id)) = (&node.model_value, node.model_observation) {
            match stores.observations.get(id) {
                Ok(observation)
                    if observation.state == node.state
                        && observation.model_value_identity.as_ref() == Some(&output.identity)
                        && observation.model_value_input == Some(output.input_sha256)
                        && matches!(observation.score,
                            RawScore::ContextWdl { win, draw, loss, perspective, context_revision }
                            if [win,draw,loss] == output.wdl
                                && perspective == output.perspective
                                && Some(context_revision) == node.model_value_revision) =>
                {
                    RecheckEndpointEvidence::ModelWdl {
                        observation_id: id,
                        observation,
                        output,
                        context_revision: node
                            .model_value_revision
                            .expect("validated raw revision"),
                    }
                }
                _ => RecheckEndpointEvidence::InvalidProvenance {
                    error: StoreError::InvalidEvidence("Fresh WDL observation mismatch"),
                },
            }
        } else {
            RecheckEndpointEvidence::Unobserved
        };
        RecheckEndpoint {
            state: node.state,
            situation: node.situation,
            snapshot,
            evidence,
        }
    }

    pub(super) fn endpoint_expectation(&self, node: usize, root_color: Color) -> Option<f32> {
        let node = &self.semantic.nodes[node];
        let value = if let Some((_, value)) = node.terminal {
            value.signum() as f32
        } else {
            let output = node.model_value.as_ref()?;
            output
                .validate(
                    &node.position,
                    self.semantic.model_registered_value.as_ref()?,
                )
                .ok()?;
            output.wdl[0] - output.wdl[2]
        };
        Some(if node.position.side_to_move() == root_color {
            value
        } else {
            -value
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn frozen_recheck_once(
        &mut self,
        work: &RepairWork,
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
        progress: &mut ObservedProgress<'_>,
    ) -> Result<Option<NewFrontier>, PalsError> {
        if self.stopped(limits, cancel).is_some() {
            return Ok(None);
        }
        self.validate_checker_namespace()?;
        let root = self
            .semantic
            .nodes
            .get(work.root)
            .ok_or(StoreError::InvalidHandle("WDL root"))?;
        let root_state = root.state;
        let root_situation = root.situation;
        let root_color = root.position.side_to_move();
        let generation = self.semantic.stores.generation();
        let root_revision = self
            .semantic
            .stores
            .situations
            .get(root_situation)?
            .revision;
        if self.semantic.stores.root() != Some(root_situation)
            || work.repaired.first() != Some(&work.original_first)
            || work.attack_ply >= work.repaired.len()
            || work.attack_ply >= work.refutation.len()
            || work.repaired[..=work.attack_ply] != work.refutation[..=work.attack_ply]
            || self
                .semantic
                .stores
                .lines
                .get(work.repaired_line)?
                .start_state
                != root_state
            || self.semantic.stores.lines.moves(work.repaired_line)? != work.repaired
        {
            return Err(
                StoreError::InvalidEvidence("frozen recheck line authority mismatch").into(),
            );
        }
        let record = self
            .semantic
            .records
            .iter()
            .find(|record| {
                record.revision == work.repair_record_revision
                    && record.origin_state == root_state
                    && record.kind == RecordKind::Repair
                    && record.line == work.repaired
            })
            .cloned()
            .ok_or(StoreError::InvalidEvidence("WDL Repair record missing"))?;
        let mut path = Vec::with_capacity(work.repaired.len() + 1);
        path.push(work.root);
        for movement in &work.repaired {
            let parent = *path.last().ok_or(StoreError::InvalidHandle("WDL path"))?;
            path.push(
                self.semantic.nodes[parent]
                    .edges
                    .iter()
                    .find(|edge| edge.movement == *movement)
                    .ok_or(StoreError::InvalidHandle("WDL path edge"))?
                    .child,
            );
        }
        if path.last() != Some(&work.repaired_leaf) {
            return Err(StoreError::InvalidEvidence("WDL repaired endpoint mismatch").into());
        }
        let changed = (work.attack_ply + 1..work.repaired.len()).find(|&ply| {
            self.semantic.nodes[path[ply]].position.side_to_move() == root_color
                && work.refutation.get(ply) != Some(&work.repaired[ply])
        });
        let Some(anchor_ply) = changed.and_then(|ply| {
            (ply + 1..work.repaired.len()).find(|&index| {
                self.semantic.nodes[path[index]].position.side_to_move() != root_color
            })
        }) else {
            counters.frozen_wdl_unresolved += 1;
            return Ok(None);
        };
        let anchor = path[anchor_ply];
        self.evaluate_model_value(work.repaired_leaf, &work.repaired, limits, cancel, counters)?;
        let reply_context = self.role_context(
            anchor,
            RoleQuestion {
                purpose: RoleQueryPurpose::ReplyPolicy,
                prefix: &work.repaired[..anchor_ply],
                proposal: &work.repaired,
                refutation: Some(&work.refutation),
                divergences: &[],
            },
        )?;
        let identity = RecheckIdentity {
            game_generation: self.semantic.game_generation.ok_or(PalsError::Capacity)?,
            search_generation: generation,
            root: root_situation,
            root_revision,
            repair_record_revision: work.repair_record_revision,
            repaired_line: work.repaired_line,
        };
        let root_snapshot = self.semantic.nodes[work.root].position.snapshot();
        let anchor_snapshot = self.semantic.nodes[anchor].position.snapshot();
        let repaired_snapshot = self.semantic.nodes[work.repaired_leaf].position.snapshot();
        let examined: Vec<_> = self.semantic.nodes[anchor]
            .edges
            .iter()
            .map(|edge| edge.movement)
            .collect();
        let deadline_tick = self.tick_at(limits.deadline);
        let previous = self
            .semantic
            .stores
            .situations
            .get(root_situation)?
            .conclusions
            .get(work.repaired_line)
            .and_then(|c| c.evidence);
        let mut trace = RecheckProgress::default();
        let mut comparison_attempted = false;
        let prepared = self.semantic.model.recheck_prepared(RecheckPrepared {
            policy: self.post_repair_recheck,
            identity,
            root_state,
            root_snapshot: &root_snapshot,
            anchor_state: self.semantic.nodes[anchor].state,
            anchor_situation: self.semantic.nodes[anchor].situation,
            anchor_snapshot: &anchor_snapshot,
            anchor_ply,
            examined_responses: &examined,
            reply_context: &reply_context,
            repair_record: &record,
            repaired: &work.repaired,
            refutation: &work.refutation,
            repaired_endpoint: Self::frozen_endpoint(
                &self.semantic.nodes[work.repaired_leaf],
                &self.semantic.stores,
                &repaired_snapshot,
                self.semantic.revision,
            ),
            limits,
            deadline_tick,
            cancel,
            config: &self.config,
            checker_identity: &self.researcher.checker_registered_identity,
            cpu_condition: &self.researcher.cpu_registered_condition,
        });
        let mut new_frontier = None;
        let mut result = if let Err(error) = prepared {
            trace.disposition = RecheckDisposition::PreparedRejected;
            Err(PalsError::Role(error))
        } else {
            trace.prepared_accepted = true;
            (|| {
                let calls = counters.critic_calls;
                let accepted = counters.accepted_critic_outputs;
                let replies = self.ranked(
                    anchor,
                    Call::Reply,
                    &work.repaired[..anchor_ply],
                    &work.repaired,
                    Some(&work.refutation),
                    limits,
                    cancel,
                    counters,
                );
                trace.reply_call_attempted = counters.critic_calls > calls;
                trace.reply_accepted = counters.accepted_critic_outputs > accepted;
                let replies = replies?;
                let old = work.repaired[anchor_ply];
                let response = replies
                    .iter()
                    .copied()
                    .find(|m| *m != old && !examined.contains(m))
                    .or_else(|| replies.iter().copied().find(|m| *m != old));
                let Some(response) = response else {
                    trace.disposition = RecheckDisposition::NoAlternativeResponse;
                    return Ok(());
                };
                trace.selected_response = Some(response);
                trace
                    .counterline
                    .extend_from_slice(&work.repaired[..anchor_ply]);
                let mut state = self.semantic.nodes[anchor].position.clone();
                state.make_move(response)?;
                let response_node =
                    self.connect_checked(anchor, response, state, counters, limits, cancel)?;
                trace.counterline.push(response);
                let repair_prefix = trace.counterline.clone();
                let mut leaf = response_node;
                trace.counter_leaf = Some(leaf);
                while trace.counterline.len() < work.repaired.len()
                    && self.semantic.nodes[leaf].terminal.is_none()
                    && self.stopped(limits, cancel).is_none()
                {
                    let ranked = self.ranked(
                        leaf,
                        Call::Reply,
                        &trace.counterline,
                        &work.repaired,
                        Some(&work.refutation),
                        limits,
                        cancel,
                        counters,
                    )?;
                    let Some(movement) = ranked.first().copied() else {
                        break;
                    };
                    let mut position = self.semantic.nodes[leaf].position.clone();
                    position.make_move(movement)?;
                    leaf =
                        self.connect_checked(leaf, movement, position, counters, limits, cancel)?;
                    trace.counter_leaf = Some(leaf);
                    trace.counterline.push(movement);
                }
                trace.counterline_completed = trace.counterline.len() == work.repaired.len();
                if !trace.counterline_completed {
                    trace.disposition = RecheckDisposition::IncompleteCounterline;
                    return Ok(());
                }
                if trace.counterline == work.refutation || trace.counterline == work.repaired {
                    trace.disposition = RecheckDisposition::NoAlternativeResponse;
                    return Ok(());
                }
                // Only this accepted Repair/C continuation barrier can select
                // the opt-in held parent. It resumes the original task/state,
                // never the new counter leaf or a token from another namespace.
                self.resume_cpu_after_repair_c(limits, cancel, counters)?;
                counters.refutations += 1;
                self.record_checked(
                    RecordKind::Counterexample,
                    &trace.counterline,
                    None,
                    0,
                    None,
                    None,
                    limits,
                    cancel,
                )?;
                self.verify_checked_cpu(leaf, &trace.counterline, limits, cancel, counters)?;
                let comparison_revision = self.semantic.revision;
                comparison_attempted = true;
                // Both Fresh invocations see exactly the same public-record revision.
                // They have different target states, not a shared raw input/cache key.
                self.evaluate_model_value_pair(
                    work.repaired_leaf,
                    &work.repaired,
                    leaf,
                    &trace.counterline,
                    limits,
                    cancel,
                    counters,
                )?;
                if self.semantic.revision != comparison_revision
                    || self.semantic.stores.root() != Some(root_situation)
                    || self.semantic.stores.generation() != generation
                    || self
                        .semantic
                        .stores
                        .situations
                        .get(root_situation)?
                        .revision
                        != root_revision
                {
                    return Err(StoreError::StaleConsumer.into());
                }
                self.validate_checker_namespace()?;
                trace.comparable = match (
                    self.semantic.nodes[work.repaired_leaf].terminal,
                    self.semantic.nodes[leaf].terminal,
                ) {
                    (Some(_), Some(_)) => true,
                    (None, None) => {
                        self.semantic.nodes[work.repaired_leaf].model_value_revision
                            == Some(comparison_revision)
                            && self.semantic.nodes[leaf].model_value_revision
                                == Some(comparison_revision)
                            && self.semantic.nodes[work.repaired_leaf]
                                .model_value
                                .as_ref()
                                .zip(self.semantic.nodes[leaf].model_value.as_ref())
                                .is_some_and(|(a, b)| {
                                    a.identity == b.identity
                                        && Some(&a.identity)
                                            == self.semantic.model_registered_value.as_ref()
                                })
                    }
                    _ => false,
                };
                new_frontier = Some(NewFrontier {
                    prefix: repair_prefix,
                    response_node,
                    counterline: trace.counterline.clone(),
                    attack_ply: anchor_ply,
                    counter_leaf: leaf,
                    publication: None,
                });
                if !trace.comparable {
                    counters.frozen_wdl_unresolved += 1;
                    trace.disposition = RecheckDisposition::IncomparableEvidence;
                    return Ok(());
                }
                counters.frozen_wdl_comparisons += 1;
                let Some((old, new)) = self
                    .endpoint_expectation(work.repaired_leaf, root_color)
                    .zip(self.endpoint_expectation(leaf, root_color))
                else {
                    counters.frozen_wdl_unresolved += 1;
                    trace.disposition = RecheckDisposition::MissingCompletedCounterValue;
                    return Ok(());
                };
                if new >= old {
                    trace.disposition = RecheckDisposition::CounterNotLower;
                    return Ok(());
                }
                if self.cpu_interleave == CpuInterleavePolicy::ReplaceOnFresh {
                    self.publish_choice(work.root, limits, cancel, progress)?;
                }
                let evidence =
                    self.frozen_refutation_observation(work, leaf, new, previous, limits, cancel)?;
                self.archive_refute_continuation(
                    root_situation,
                    work.repaired_line,
                    evidence,
                    limits,
                    cancel,
                )?;
                counters.supported_refutations += 1;
                trace.publication = Some(evidence);
                if let Some(frontier) = new_frontier.as_mut() {
                    frontier.publication = Some(evidence);
                }
                trace.disposition = RecheckDisposition::ConditionalRefutationPublished;
                Ok(())
            })()
        };
        let counter_snapshot = trace
            .counter_leaf
            .map(|node| self.semantic.nodes[node].position.snapshot());
        let publication = match trace
            .publication
            .map(|id| -> Result<_, PalsError> {
                Ok(RecheckPublication {
                    observation_id: id,
                    observation: self.semantic.stores.observations.get(id)?,
                    conclusion: *self
                        .semantic
                        .stores
                        .situations
                        .get(root_situation)?
                        .conclusions
                        .get(work.repaired_line)
                        .ok_or(StoreError::InvalidEvidence("WDL publication absent"))?,
                })
            })
            .transpose()
        {
            Ok(publication) => publication,
            Err(error) => {
                if result.is_ok() {
                    result = Err(error);
                }
                None
            }
        };
        let finish = self.semantic.model.recheck_finished(RecheckFinished {
            policy: self.post_repair_recheck,
            identity,
            prepared_accepted: trace.prepared_accepted,
            reply_call_attempted: trace.reply_call_attempted,
            reply_accepted: trace.reply_accepted,
            reply_context: &reply_context,
            selected_response: trace.selected_response,
            counterline: &trace.counterline,
            full_suffix_replayed: false,
            counterline_completed: trace.counterline_completed,
            repaired_endpoint: Self::frozen_endpoint(
                &self.semantic.nodes[work.repaired_leaf],
                &self.semantic.stores,
                &repaired_snapshot,
                self.semantic.revision,
            ),
            counter_endpoint: trace.counter_leaf.zip(counter_snapshot.as_ref()).map(
                |(node, snapshot)| {
                    Self::frozen_endpoint(
                        &self.semantic.nodes[node],
                        &self.semantic.stores,
                        snapshot,
                        self.semantic.revision,
                    )
                },
            ),
            comparable: trace.comparable,
            publication,
            disposition: trace.disposition,
            original_error: result.as_ref().err(),
            limits,
            deadline_tick,
            cancel,
        });
        if let Err(error) = finish {
            self.semantic
                .last_recheck_observer_error
                .get_or_insert_with(|| error.clone());
            if result.is_ok() {
                result = Err(error.into());
            }
        }
        let comparison = self.observe_frozen_comparison(
            work.repaired_leaf,
            trace.counter_leaf,
            root_color,
            comparison_attempted,
            trace.comparable,
            trace.publication,
            result.as_ref().err(),
        );
        self.observe_repair_recheck(
            work.repair_record_revision,
            PalsRepairRecheckExecution {
                reply_context: (&reply_context).into(),
                prepared_accepted: trace.prepared_accepted,
                reply_call_attempted: trace.reply_call_attempted,
                reply_accepted: trace.reply_accepted,
                selected_response: trace.selected_response,
                counterline: trace.counterline,
                counterline_completed: trace.counterline_completed,
                disposition: trace.disposition,
                original_error: result.as_ref().err().cloned(),
                comparison,
            },
        );
        result?;
        if self.cpu_interleave == CpuInterleavePolicy::ResumeAfterAcceptedRepairC
            && trace.prepared_accepted
            && trace.reply_accepted
            && trace.counterline_completed
        {
            // recheck_finished and its immutable TaskTable/record witness are
            // now complete. Publish before a subsequent P call can replace its
            // resident seed. This is a read-only observer, not invented work.
            self.publish_choice(work.root, limits, cancel, progress)?;
        }
        if let Some(stop) = self.stopped(limits, cancel) {
            return Err(match stop {
                PalsCompletion::Canceled => RoleError::Canceled,
                _ => RoleError::Deadline,
            }
            .into());
        }
        Ok(new_frontier)
    }

    fn frozen_refutation_observation(
        &mut self,
        work: &RepairWork,
        counter_leaf: usize,
        expectation: f32,
        supersedes: Option<ObservationId>,
        limits: PalsLimits,
        cancel: &AtomicBool,
    ) -> Result<ObservationId, PalsError> {
        let score = if self.semantic.nodes[counter_leaf].terminal.is_some() {
            RawScore::Estimate {
                value: expectation,
                perspective: self.semantic.nodes[work.root].position.side_to_move(),
            }
        } else {
            RawScore::ConditionalWdl {
                expectation,
                perspective: self.semantic.nodes[work.root].position.side_to_move(),
                repaired: self.semantic.nodes[work.repaired_leaf]
                    .model_observation
                    .ok_or(RoleError::Unavailable)?,
                counter: self.semantic.nodes[counter_leaf]
                    .model_observation
                    .ok_or(RoleError::Unavailable)?,
                context_revision: self.semantic.revision,
            }
        };
        self.append_engine_observation_with_controls(
            Observation {
                state: self.semantic.nodes[work.root].state,
                line: Some(work.repaired_line),
                source: stable_id(self.search_identity()),
                epoch: 0,
                scope: EvidenceScope::Model {
                    model: stable_id(self.semantic.model.identity()),
                    encoding: stable_id("pals-frozen-wdl-conditional-line/2"),
                    input: self.semantic.revision,
                },
                score,
                value_identity: None,
                checker_identity: None,
                checker_work: None,
                external_report: None,
                model_value_identity: None,
                model_value_input: None,
                cpu_condition: None,
                cpu_pv: None,
                budget: 0,
                kind: ObservationKind::Refutation,
                supersedes,
                execution: None,
            },
            Some((limits.deadline, cancel)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuConfig;

    struct WdlFixture {
        identity: ModelValueIdentity,
        finished: Vec<(RecheckDisposition, bool, Option<u64>, Option<u64>)>,
        finished_lines: Vec<LineId>,
        revisions: Vec<u64>,
        replies: Option<[BoardMove; 3]>,
        finished_notice: Option<std::sync::Arc<AtomicBool>>,
    }
    impl WdlFixture {
        fn new() -> Self {
            Self {
                identity: ModelValueIdentity {
                    semantics: MODEL_WDL_VALUE_SEMANTICS.into(),
                    model: "frozen-WDL-fixture/2".into(),
                    encoding: "exact-typed-mock/2".into(),
                    precision: "fp32".into(),
                    model_epoch: [3; 32],
                },
                finished: Vec::new(),
                finished_lines: Vec::new(),
                revisions: Vec::new(),
                replies: None,
                finished_notice: None,
            }
        }
    }
    impl RoleModel for WdlFixture {
        fn identity(&self) -> &str {
            &self.identity.model
        }
        fn value_identity(&self) -> Option<&ModelValueIdentity> {
            Some(&self.identity)
        }
        fn evaluate_value(&mut self, query: RoleQuery<'_>) -> Result<ModelValueOutput, RoleError> {
            query.check_control()?;
            self.revisions.push(query.revision);
            let mut digest = Sha256::new();
            digest.update(query.position.to_fen().as_bytes());
            digest.update(query.revision.to_le_bytes());
            for movement in query.prefix {
                digest.update(
                    Move16::pack(*movement)
                        .map_err(|_| RoleError::InvalidOutput)?
                        .bits()
                        .to_le_bytes(),
                );
            }
            let lower = query.prefix.last() == Some(&BoardMove::from_uci("e5d4").unwrap());
            Ok(ModelValueOutput {
                identity: self.identity.clone(),
                input_sha256: digest.finalize().into(),
                state: query.position.position_identity(),
                perspective: query.position.side_to_move(),
                wdl: if lower {
                    [0.1, 0.2, 0.7]
                } else {
                    [0.7, 0.2, 0.1]
                },
            })
        }
        fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            LegalOrderRoleMock.propose(query)
        }
        fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            let wanted = BoardMove::from_uci(match query.prefix.len() {
                4 => "d2d3",
                5 => "d7d6",
                _ => "e2e4",
            })
            .unwrap();
            let selected = query.legal.iter().position(|movement| *movement == wanted);
            let mut output = LegalOrderRoleMock.repair(query)?;
            if let Some(index) = selected {
                output.logits[index] = 100.0;
            }
            Ok(output)
        }
        fn divergences(&mut self, query: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
            LegalOrderRoleMock.divergences(query)
        }
        fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            let default = BoardMove::from_uci(match query.prefix.len() {
                4 => "d2d4",
                5 => "e5d4",
                _ => "g8f6",
            })
            .unwrap();
            let wanted = self
                .replies
                .as_ref()
                .and_then(|replies| {
                    query
                        .prefix
                        .len()
                        .checked_sub(3)
                        .and_then(|index| replies.get(index))
                })
                .copied()
                .unwrap_or(default);
            let selected = query.legal.iter().position(|m| *m == wanted);
            let mut output = LegalOrderRoleMock.reply(query)?;
            if let Some(index) = selected {
                output.logits[index] = 100.0;
            }
            Ok(output)
        }
        fn recheck_finished(&mut self, event: RecheckFinished<'_>) -> Result<(), RoleError> {
            self.finished_lines.push(event.identity.repaired_line);
            if let Some(notice) = &self.finished_notice {
                notice.store(true, Ordering::Release);
            }
            let revision = |endpoint: &RecheckEndpoint<'_>| match &endpoint.evidence {
                RecheckEndpointEvidence::ModelWdl {
                    context_revision, ..
                } => Some(*context_revision),
                _ => None,
            };
            self.finished.push((
                event.disposition,
                event.comparable,
                revision(&event.repaired_endpoint),
                event.counter_endpoint.as_ref().and_then(revision),
            ));
            Ok(())
        }
    }
    fn engine(policy: PostRepairRecheckPolicy, resolver: ResolverPolicy) -> PalsEngine<WdlFixture> {
        PalsEngine::new_with_boxed_checker_and_policies(
            PalsConfig {
                line_plies: 6,
                ..PalsConfig::default()
            },
            WdlFixture::new(),
            Box::new(
                OwnedCpuChecker::new(Box::new(
                    CpuEngine::new(CpuConfig {
                        max_depth: 2,
                        tt_entries: 128,
                        ..CpuConfig::default()
                    })
                    .unwrap(),
                ))
                .unwrap(),
            ),
            resolver,
            policy,
        )
        .unwrap()
    }
    fn limits() -> PalsLimits {
        PalsLimits {
            deadline: Instant::now() + Duration::from_secs(10),
            max_rounds: 1,
            max_cpu_nodes: 100_000,
            cpu_depth: 1,
        }
    }
    fn line(text: &[&str]) -> Vec<BoardMove> {
        text.iter()
            .map(|m| BoardMove::from_uci(m).unwrap())
            .collect()
    }

    fn held_engine() -> PalsEngine<WdlFixture> {
        use crate::cpu::CpuOrderingPolicy;
        use crate::cpu_value::BootstrapCpuValue;
        let cpu = CpuEngine::with_paused_side_scope(
            CpuConfig {
                max_depth: 2,
                tt_entries: 64,
                ..CpuConfig::default()
            },
            std::sync::Arc::new(BootstrapCpuValue::default()),
            CpuOrderingPolicy::LegacyMvvLvaV1,
        )
        .unwrap();
        PalsEngine::new_with_boxed_checker_and_policies_and_cpu_interleave(
            PalsConfig {
                line_plies: 6,
                cpu_nodes_per_task: 8,
                ..PalsConfig::default()
            },
            WdlFixture::new(),
            Box::new(OwnedCpuChecker::new(cpu).unwrap()),
            ResolverPolicy::ModelWdlRestricted,
            PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
            CpuInterleavePolicy::ResumeAfterAcceptedRepairC,
        )
        .unwrap()
    }

    fn other_first_repair(
        engine: &mut PalsEngine<WdlFixture>,
        counters: &mut PalsCounters,
    ) -> RepairWork {
        repair_work_for_lines(
            engine,
            counters,
            line(&["d2d4", "d7d5", "c1f4", "g8f6", "g1f3", "e7e6"]),
            line(&["d2d4", "d7d5", "b1c3", "g8f6", "g1f3", "e7e6"]),
        )
    }

    #[test]
    fn held_parent_priority_rechecks_before_other_first_and_retains_pending_order() {
        let mut selected = held_engine();
        let mut counters = PalsCounters::default();
        let work = accepted_repair(&mut selected, &mut counters);
        let other = other_first_repair(&mut selected, &mut counters);
        let later_other = other_first_repair(&mut selected, &mut counters);
        let pending = [other.repaired_line, later_other.repaired_line];
        let cancel = AtomicBool::new(false);
        let budget = PalsLimits {
            max_cpu_nodes: 256,
            cpu_depth: 2,
            ..limits()
        };
        // Registered CPU/mock branches are already accepted fixture records.
        // Admit their initial Repair and keep unrelated followups out of this
        // queue-order check; the selected first still exercises real followups.
        assert!(selected.admit_iterative_repair(work.original_first));
        selected
            .semantic
            .repair_counts
            .insert(other.original_first, MAX_REPAIRS_PER_FIRST);
        selected
            .enqueue_frozen_repair(other, &mut counters)
            .unwrap();
        selected
            .enqueue_frozen_repair(later_other, &mut counters)
            .unwrap();
        selected
            .verify_checked_cpu(work.root, &[], budget, &cancel, &mut counters)
            .unwrap();
        let held = selected.held_cpu_task.as_ref().unwrap();
        assert!(held.prefix.is_empty());
        let previous = held.execution;
        let original_key = selected
            .semantic
            .stores
            .tasks
            .get(previous)
            .unwrap()
            .key
            .clone();
        let mut resumed_before_return = false;
        let mut progress = |_: BoardMove, stores: &PalsStores, _: &[RoleRecord]| {
            resumed_before_return |= (0..stores.tasks.len()).any(|index| {
                let task = stores.tasks.get(ExecutionId(index)).unwrap();
                task.resumed_from == Some(previous) && task.key == original_key
            });
        };
        selected
            .queue_or_recheck_frozen(work.clone(), budget, &cancel, &mut counters, &mut progress)
            .unwrap();
        assert!(resumed_before_return);
        assert_eq!(
            selected.semantic.model.finished_lines.first(),
            Some(&work.repaired_line)
        );
        assert!(
            selected
                .semantic
                .model
                .finished_lines
                .iter()
                .all(|line| !pending.contains(line))
        );
        assert_eq!(
            selected
                .semantic
                .repair_queue
                .iter()
                .map(|work| work.repaired_line)
                .collect::<Vec<_>>(),
            pending
        );
        assert!(counters.accepted_repair_outputs > 0);
        assert!(selected.semantic.repair_counts[&work.original_first] <= MAX_REPAIRS_PER_FIRST);
        let enqueued = counters.repair_queue_enqueued;
        let role_calls = counters.role_calls;
        selected
            .queue_or_recheck_frozen(work, budget, &cancel, &mut counters, &mut |_, _, _| {})
            .unwrap();
        assert_eq!(counters.repair_queue_deduplicated, 1);
        assert_eq!(counters.repair_queue_enqueued, enqueued);
        assert_eq!(counters.role_calls, role_calls);
        let before_public_drain = selected.semantic.model.finished_lines.len();
        selected
            .drain_repair_queue(budget, &cancel, &mut counters, &mut |_, _, _| {})
            .unwrap();
        assert_eq!(
            &selected.semantic.model.finished_lines[before_public_drain..],
            &pending
        );
        assert!(selected.semantic.repair_queue.is_empty());
        assert!(counters.cpu_nodes <= budget.max_cpu_nodes);
        assert_eq!(
            selected
                .researcher
                .cpu
                .paused_stack_snapshot()
                .unwrap()
                .tokens_peak,
            1
        );
        selected.discard_canceled_cpu_checkpoints().unwrap();
    }

    #[test]
    fn held_parent_priority_requires_matching_nonempty_prefix_first() {
        let mut selected = held_engine();
        let mut counters = PalsCounters::default();
        let work = accepted_repair(&mut selected, &mut counters);
        let other = other_first_repair(&mut selected, &mut counters);
        selected
            .semantic
            .repair_counts
            .insert(work.original_first, MAX_REPAIRS_PER_FIRST);
        let budget = PalsLimits {
            max_cpu_nodes: 128,
            cpu_depth: 2,
            ..limits()
        };
        let cancel = AtomicBool::new(false);
        selected
            .verify_checked_cpu(
                work.repaired_leaf,
                &work.repaired,
                budget,
                &cancel,
                &mut counters,
            )
            .unwrap();
        let original = selected.held_cpu_task.as_ref().unwrap().execution;
        assert_eq!(
            selected.held_cpu_task.as_ref().unwrap().prefix.first(),
            Some(&work.original_first)
        );
        let pending_line = other.repaired_line;
        selected
            .queue_or_recheck_frozen(other, budget, &cancel, &mut counters, &mut |_, _, _| {})
            .unwrap();
        assert_eq!(counters.critic_calls, 0);
        assert_eq!(counters.cpu_nodes, 8);
        assert_eq!(selected.held_cpu_task.as_ref().unwrap().execution, original);
        selected
            .queue_or_recheck_frozen(
                work.clone(),
                budget,
                &cancel,
                &mut counters,
                &mut |_, _, _| {},
            )
            .unwrap();
        assert_eq!(selected.semantic.model.finished_lines, [work.repaired_line]);
        assert_eq!(selected.semantic.repair_queue.len(), 1);
        assert_eq!(
            selected.semantic.repair_queue[0].repaired_line,
            pending_line
        );
        assert!((0..selected.semantic.stores.tasks.len()).any(|index| {
            selected
                .semantic
                .stores
                .tasks
                .get(ExecutionId(index))
                .unwrap()
                .resumed_from
                == Some(original)
        }));
        selected.discard_canceled_cpu_checkpoints().unwrap();
    }

    #[test]
    fn held_parent_priority_cancel_deadline_duplicate_and_no_owner_do_not_execute() {
        for canceled in [true, false] {
            let mut selected = held_engine();
            let mut counters = PalsCounters::default();
            let work = accepted_repair(&mut selected, &mut counters);
            let budget = PalsLimits {
                max_cpu_nodes: 128,
                cpu_depth: 2,
                ..limits()
            };
            let cancel = AtomicBool::new(false);
            selected
                .verify_checked_cpu(work.root, &[], budget, &cancel, &mut counters)
                .unwrap();
            let token = selected.semantic.nodes[work.root].resume.clone().unwrap();
            cancel.store(canceled, Ordering::Release);
            let stopped_budget = if canceled {
                budget
            } else {
                PalsLimits {
                    deadline: Instant::now(),
                    ..budget
                }
            };
            for _ in 0..2 {
                selected
                    .queue_or_recheck_frozen(
                        work.clone(),
                        stopped_budget,
                        &cancel,
                        &mut counters,
                        &mut |_, _, _| {},
                    )
                    .unwrap();
            }
            selected
                .drain_repair_queue(stopped_budget, &cancel, &mut counters, &mut |_, _, _| {})
                .unwrap();
            assert_eq!(selected.semantic.repair_queue.len(), 1);
            assert_eq!(counters.repair_queue_deduplicated, 1);
            assert_eq!(counters.cpu_nodes, 8);
            assert_eq!(counters.critic_calls, 0);
            assert!(selected.researcher.cpu.token_is_current(&token));
            selected.discard_canceled_cpu_checkpoints().unwrap();
            assert!(!selected.researcher.cpu.token_is_current(&token));
            assert!(selected.held_cpu_task.is_none());
            cancel.store(false, Ordering::Release);
            let next = accepted_repair(&mut selected, &mut counters);
            selected
                .queue_or_recheck_frozen(next, budget, &cancel, &mut counters, &mut |_, _, _| {})
                .unwrap();
            assert_eq!(selected.semantic.repair_queue.len(), 2);
            assert_eq!(counters.critic_calls, 0);
            assert_eq!(counters.cpu_nodes, 8);
        }
    }

    #[test]
    fn default_repair_queue_waits_and_public_drain_retains_fifo() {
        let mut selected = engine(
            PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
            ResolverPolicy::ModelWdlRestricted,
        );
        let mut counters = PalsCounters::default();
        let work = accepted_repair(&mut selected, &mut counters);
        let other = other_first_repair(&mut selected, &mut counters);
        let pending = [work.repaired_line, other.repaired_line];
        for first in [work.original_first, other.original_first] {
            selected
                .semantic
                .repair_counts
                .insert(first, MAX_REPAIRS_PER_FIRST);
        }
        let cancel = AtomicBool::new(false);
        let budget = limits();
        for work in [work, other] {
            selected
                .queue_or_recheck_frozen(work, budget, &cancel, &mut counters, &mut |_, _, _| {})
                .unwrap();
        }
        assert_eq!(selected.cpu_interleave, CpuInterleavePolicy::ReplaceOnFresh);
        assert_eq!(counters.critic_calls, 0);
        assert!(selected.semantic.model.finished_lines.is_empty());
        assert_eq!(
            selected
                .semantic
                .repair_queue
                .iter()
                .map(|work| work.repaired_line)
                .collect::<Vec<_>>(),
            pending
        );
        selected
            .drain_repair_queue(budget, &cancel, &mut counters, &mut |_, _, _| {})
            .unwrap();
        assert_eq!(selected.semantic.model.finished_lines, pending);
        assert!(selected.semantic.repair_queue.is_empty());
    }

    #[test]
    fn held_parent_repair_c_barrier_resumes_original_task_with_shared_budget() {
        use crate::cpu::{CpuOrderingPolicy, CpuResumePolicy};
        use crate::cpu_value::BootstrapCpuValue;
        let finished = std::sync::Arc::new(AtomicBool::new(false));
        let mut model = WdlFixture::new();
        model.finished_notice = Some(finished.clone());
        let cpu = CpuEngine::with_paused_side_scope(
            CpuConfig {
                max_depth: 2,
                tt_entries: 64,
                ..CpuConfig::default()
            },
            std::sync::Arc::new(BootstrapCpuValue::default()),
            CpuOrderingPolicy::LegacyMvvLvaV1,
        )
        .unwrap();
        assert_eq!(cpu.resume_policy(), CpuResumePolicy::PausedStack);
        let mut engine = PalsEngine::new_with_boxed_checker_and_policies_and_cpu_interleave(
            PalsConfig {
                line_plies: 6,
                cpu_nodes_per_task: 8,
                ..PalsConfig::default()
            },
            model,
            Box::new(OwnedCpuChecker::new(cpu).unwrap()),
            ResolverPolicy::ModelWdlRestricted,
            PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
            CpuInterleavePolicy::ResumeAfterAcceptedRepairC,
        )
        .unwrap();
        let cancel = AtomicBool::new(false);
        let budget = PalsLimits {
            max_cpu_nodes: 32,
            cpu_depth: 2,
            ..limits()
        };
        let mut counters = PalsCounters::default();
        // Explicit registered CPU/mock Repair fixture, never GPU evidence.
        let work = accepted_repair(&mut engine, &mut counters);
        engine
            .verify_checked_cpu(
                work.repaired_leaf,
                &work.repaired,
                budget,
                &cancel,
                &mut counters,
            )
            .unwrap();
        let token = engine.semantic.nodes[work.repaired_leaf]
            .resume
            .clone()
            .unwrap();
        let (observation, previous) = engine.semantic.nodes[work.repaired_leaf]
            .evidence
            .as_ref()
            .unwrap()
            .provenance
            .unwrap();
        let raw = engine
            .semantic
            .stores
            .observations
            .get(observation)
            .unwrap()
            .clone();
        let root = work.root;
        // A same-node frontier visit is not the actual Repair/C barrier and
        // must neither resume nor charge the held original task early.
        let requested = counters.cpu_tasks_requested;
        engine
            .verify_checked_cpu(
                work.repaired_leaf,
                &work.repaired,
                budget,
                &cancel,
                &mut counters,
            )
            .unwrap();
        assert_eq!(counters.cpu_nodes, 8);
        assert_eq!(counters.cpu_tasks_requested, requested);
        assert_eq!(engine.held_cpu_task.as_ref().unwrap().execution, previous);
        assert!(engine.researcher.cpu.token_is_current(&token));
        // A different fresh state runs in the real one-shot side owner before
        // C; it cannot invalidate the original task, token, TT or path.
        engine
            .verify_checked_cpu(root, &[], budget, &cancel, &mut counters)
            .unwrap();
        assert!(engine.researcher.cpu.token_is_current(&token));
        assert_eq!(counters.cpu_nodes, 16);
        let mut observed = false;
        engine
            .frozen_recheck_once(
                &work,
                budget,
                &cancel,
                &mut counters,
                &mut |_, stores, records| {
                    assert!(finished.load(Ordering::Acquire));
                    assert!(!records.is_empty());
                    observed = (0..stores.tasks.len()).any(|index| {
                        let task = stores.tasks.get(ExecutionId(index)).unwrap();
                        task.resumed_from == Some(previous)
                            && task.key.state == stores.tasks.get(previous).unwrap().key.state
                    });
                },
            )
            .unwrap();
        assert!(observed);
        assert_eq!(engine.search_identity(), PALS_CPU_INTERLEAVE_SEARCH_VERSION);
        assert_eq!(counters.cpu_nodes, 32);
        assert_eq!(counters.cpu_tasks_requested, 4);
        assert!(!engine.researcher.cpu.token_is_current(&token));
        let renewed = engine.semantic.nodes[work.repaired_leaf]
            .resume
            .as_ref()
            .unwrap();
        assert!(engine.researcher.cpu.token_is_current(renewed));
        assert_eq!(renewed.cumulative_work().unwrap().nodes, 16);
        assert_eq!(
            engine
                .semantic
                .stores
                .observations
                .get(observation)
                .unwrap(),
            &raw
        );
        let side = engine.cpu_side_scope_snapshot().unwrap();
        assert_eq!((side.attempts, side.work.nodes), (2, 16));
        assert!(side.complete);
        let owner = engine.researcher.cpu.paused_stack_snapshot().unwrap();
        assert_eq!(owner.tokens_peak, 1);
        assert_eq!(owner.replayed_consumed_work, 0);
        assert_eq!(owner.tokens_resumed, 1);
        engine.discard_canceled_cpu_checkpoints().unwrap();
        assert_eq!(
            engine
                .researcher
                .cpu
                .paused_stack_snapshot()
                .unwrap()
                .tokens_retained,
            0
        );
    }

    #[test]
    fn held_parent_c_barrier_with_smaller_task_chunk_keeps_original_owner() {
        use crate::cpu::CpuOrderingPolicy;
        use crate::cpu_value::BootstrapCpuValue;
        let cpu = CpuEngine::with_paused_side_scope(
            CpuConfig {
                max_depth: 2,
                tt_entries: 64,
                ..CpuConfig::default()
            },
            std::sync::Arc::new(BootstrapCpuValue::default()),
            CpuOrderingPolicy::LegacyMvvLvaV1,
        )
        .unwrap();
        let mut engine = PalsEngine::new_with_boxed_checker_and_policies_and_cpu_interleave(
            PalsConfig {
                line_plies: 6,
                cpu_nodes_per_task: 8,
                ..PalsConfig::default()
            },
            WdlFixture::new(),
            Box::new(OwnedCpuChecker::new(cpu).unwrap()),
            ResolverPolicy::ModelWdlRestricted,
            PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
            CpuInterleavePolicy::ResumeAfterAcceptedRepairC,
        )
        .unwrap();
        let cancel = AtomicBool::new(false);
        let budget = PalsLimits {
            max_cpu_nodes: 23,
            cpu_depth: 2,
            ..limits()
        };
        let mut counters = PalsCounters::default();
        let work = accepted_repair(&mut engine, &mut counters);
        engine
            .verify_checked_cpu(
                work.repaired_leaf,
                &work.repaired,
                budget,
                &cancel,
                &mut counters,
            )
            .unwrap();
        let token = engine.semantic.nodes[work.repaired_leaf]
            .resume
            .clone()
            .unwrap();
        let previous = engine.held_cpu_task.as_ref().unwrap().execution;
        engine
            .verify_checked_cpu(work.root, &[], budget, &cancel, &mut counters)
            .unwrap();
        engine
            .frozen_recheck_once(&work, budget, &cancel, &mut counters, &mut |_, _, _| {})
            .unwrap();
        assert!(engine.researcher.cpu.token_is_current(&token));
        assert_eq!(engine.held_cpu_task.as_ref().unwrap().execution, previous);
        assert_eq!(
            engine
                .researcher
                .cpu
                .paused_stack_snapshot()
                .unwrap()
                .tokens_resumed,
            0
        );
        assert_eq!(counters.cpu_nodes, 23);
        assert_eq!(engine.cpu_side_scope_snapshot().unwrap().work.nodes, 15);
        engine.discard_canceled_cpu_checkpoints().unwrap();
    }

    #[test]
    fn held_parent_profile_rejects_replace_or_wrong_refinement_owner() {
        use crate::cpu::{CpuOrderingPolicy, CpuResumePolicy};
        use crate::cpu_value::BootstrapCpuValue;
        for policy in [
            PostRepairRecheckPolicy::Disabled,
            PostRepairRecheckPolicy::FrozenModelWdlV2,
            PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
        ] {
            let cpu =
                CpuEngine::with_resume_policy(CpuConfig::default(), CpuResumePolicy::PausedStack)
                    .unwrap();
            assert!(
                PalsEngine::new_with_boxed_checker_and_policies_and_cpu_interleave(
                    PalsConfig::default(),
                    WdlFixture::new(),
                    Box::new(OwnedCpuChecker::new(cpu).unwrap()),
                    ResolverPolicy::ModelWdlRestricted,
                    policy,
                    CpuInterleavePolicy::ResumeAfterAcceptedRepairC,
                )
                .is_err()
            );
        }
        let cpu = CpuEngine::with_paused_side_scope(
            CpuConfig::default(),
            std::sync::Arc::new(BootstrapCpuValue::default()),
            CpuOrderingPolicy::LegacyMvvLvaV1,
        )
        .unwrap();
        assert!(
            PalsEngine::new_with_boxed_checker_and_policies_and_cpu_interleave(
                PalsConfig::default(),
                WdlFixture::new(),
                Box::new(OwnedCpuChecker::new(cpu).unwrap()),
                ResolverPolicy::ModelWdlRestricted,
                PostRepairRecheckPolicy::Disabled,
                CpuInterleavePolicy::ResumeAfterAcceptedRepairC,
            )
            .is_err()
        );
    }

    fn accepted_repair(
        engine: &mut PalsEngine<WdlFixture>,
        counters: &mut PalsCounters,
    ) -> RepairWork {
        repair_work_for_lines(
            engine,
            counters,
            line(&["e2e4", "e7e5", "g1f3", "b8c6", "d2d3", "d7d6"]),
            line(&["e2e4", "e7e5", "f1c4", "b8c6", "d2d3", "d7d6"]),
        )
    }
    fn repair_work_for_lines(
        engine: &mut PalsEngine<WdlFixture>,
        counters: &mut PalsCounters,
        refutation: Vec<BoardMove>,
        repaired: Vec<BoardMove>,
    ) -> RepairWork {
        let root_position = Position::startpos();
        engine
            .semantic
            .stores
            .focus_actual_moves(root_position.snapshot())
            .unwrap();
        let root = engine.intern(root_position).unwrap();
        engine
            .record(RecordKind::Counterexample, &refutation, None, 0, None, None)
            .unwrap();
        let mut leaf = root;
        for movement in &repaired {
            let mut state = engine.semantic.nodes[leaf].position.clone();
            state.make_move(*movement).unwrap();
            leaf = engine.connect(leaf, *movement, state, counters).unwrap();
        }
        let (repaired_line, _) = engine
            .record(RecordKind::Repair, &repaired, None, 0, None, None)
            .unwrap()
            .unwrap();
        RepairWork {
            root,
            original_first: repaired[0],
            attack_ply: 1,
            repaired_line,
            repair_record_revision: engine.semantic.revision,
            repaired_leaf: leaf,
            repaired,
            refutation,
        }
    }

    /// Explicit CPU/mock API-span observation, separate from regular checks.
    /// The enqueue/dedup spans can be queue-only; drain includes actual fixture
    /// rechecks and repairs. Different observed work never yields a cost ratio.
    #[test]
    #[ignore = "opt-in bounded CPU/mock Repair queue cost observation"]
    fn cost_fixed_repair_actions() {
        use serde_json::{Value, json};
        const WARMUPS: usize = 3;
        const SAMPLES: usize = 5;
        const MAX_OUTPUT_BYTES: usize = 256 * 1024;

        fn raw_counters(counters: PalsCounters) -> Value {
            macro_rules! fields {
                ($($field:ident),* $(,)?) => {{
                    let mut result = serde_json::Map::new();
                    $(result.insert(stringify!($field).into(), json!(counters.$field));)*
                    Value::Object(result)
                }};
            }
            fields!(
                rounds,
                proposals,
                refutations,
                repairs,
                supported_refutations,
                supported_repairs,
                role_calls,
                proposer_calls,
                critic_calls,
                repair_calls,
                completed_proposer_calls,
                completed_critic_calls,
                completed_repair_calls,
                consumed_role_outputs,
                accepted_proposer_outputs,
                accepted_critic_outputs,
                accepted_repair_outputs,
                value_calls,
                completed_value_calls,
                accepted_value_outputs,
                external_checker_tasks,
                external_checker_reports,
                external_checker_nodes_observed,
                external_checker_work_incomplete,
                external_checker_node_budget_reserved,
                consumed_external_checker_tasks,
                cpu_tasks_requested,
                cpu_tasks,
                cpu_work_observation_incomplete,
                cpu_nodes,
                cpu_quiescence_nodes,
                cpu_tt_hits,
                completed_cpu_tasks,
                partial_cpu_iterations,
                consumed_cpu_tasks,
                reused_completed_cpu_tasks_consumed,
                consumed_partial_cpu_values,
                consumed_frontier_cpu_values,
                consumed_cached_cpu_values,
                evidence_cache_hits,
                examined_edges,
                retained_situations,
                root_scope_observation_complete,
                unknown_root_children,
                repair_queue_enqueued,
                repair_queue_deduplicated,
                repair_queue_without_new_evidence,
                repair_queue_peak,
                frozen_wdl_comparisons,
                frozen_wdl_unresolved,
            )
        }
        fn queue_state(selected: &PalsEngine<WdlFixture>, first: BoardMove) -> Value {
            json!({
                "pending": selected.semantic.repair_queue.len(),
                "dedup_questions": selected.semantic.repair_questions.len(),
                "first_move_repairs": selected.semantic.repair_counts.get(&first).copied().unwrap_or(0),
                "record_revision": selected.semantic.revision,
                "records": selected.semantic.records.len(),
                "mock_value_revisions": selected.semantic.model.revisions,
            })
        }
        fn observed_work(
            selected: &PalsEngine<WdlFixture>,
            mut counters: PalsCounters,
        ) -> (PalsCounters, Vec<u64>, u64, usize) {
            // Exclude queue bookkeeping only from the workload comparison;
            // raw before/after counters below retain all four actual fields.
            counters.repair_queue_enqueued = 0;
            counters.repair_queue_deduplicated = 0;
            counters.repair_queue_without_new_evidence = 0;
            counters.repair_queue_peak = 0;
            (
                counters,
                selected.semantic.model.revisions.clone(),
                selected.semantic.revision,
                selected.semantic.records.len(),
            )
        }

        let stop_at = Instant::now() + Duration::from_secs(45);
        let cancel = AtomicBool::new(false);
        let mut rows = Vec::with_capacity(SAMPLES * 2 * 3);
        let mut comparisons = Vec::with_capacity(SAMPLES);
        let mut complete = true;
        let mut warmups_completed = 0;
        let mut samples_completed = 0;
        for trial in 0..WARMUPS + SAMPLES {
            if Instant::now() >= stop_at {
                complete = false;
                break;
            }
            // Both independent owners and their accepted frontier are prepared
            // before either policy's timer. No new backend or search is added.
            let mut immediate = engine(
                PostRepairRecheckPolicy::FrozenModelWdlV2,
                ResolverPolicy::ModelWdlRestricted,
            );
            let mut bounded = engine(
                PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
                ResolverPolicy::ModelWdlRestricted,
            );
            let mut immediate_counters = PalsCounters::default();
            let mut bounded_counters = PalsCounters::default();
            let immediate_work = accepted_repair(&mut immediate, &mut immediate_counters);
            let bounded_work = accepted_repair(&mut bounded, &mut bounded_counters);
            assert_eq!(immediate_work.original_first, bounded_work.original_first);
            assert_eq!(immediate_work.repaired, bounded_work.repaired);
            assert_eq!(immediate_work.refutation, bounded_work.refutation);
            assert!(
                immediate.semantic.nodes[immediate_work.root]
                    .position
                    .snapshot()
                    .same_state(
                        &bounded.semantic.nodes[bounded_work.root]
                            .position
                            .snapshot()
                    )
            );
            assert!(
                immediate.semantic.nodes[immediate_work.repaired_leaf]
                    .position
                    .snapshot()
                    .same_state(
                        &bounded.semantic.nodes[bounded_work.repaired_leaf]
                            .position
                            .snapshot()
                    )
            );
            assert_eq!(
                immediate.semantic.model.identity,
                bounded.semantic.model.identity
            );
            assert_eq!(
                immediate.config.max_role_calls,
                bounded.config.max_role_calls
            );
            assert_eq!(
                immediate_work.repair_record_revision,
                bounded_work.repair_record_revision
            );
            let mut budget = limits();
            budget.deadline = budget.deadline.min(stop_at);
            let measured = trial >= WARMUPS;
            let mut workloads = Vec::with_capacity(2);
            for (policy, selected, counters, work) in [
                (
                    "FrozenModelWdlV2",
                    &mut immediate,
                    &mut immediate_counters,
                    immediate_work,
                ),
                (
                    "IterativeFrozenModelWdlV2",
                    &mut bounded,
                    &mut bounded_counters,
                    bounded_work,
                ),
            ] {
                let question = json!({
                    "first_move": work.original_first.to_string(),
                    "record_revision": work.repair_record_revision,
                    "attack_ply": work.attack_ply,
                    "repaired": work.repaired.iter().map(ToString::to_string).collect::<Vec<_>>(),
                    "refutation": work.refutation.iter().map(ToString::to_string).collect::<Vec<_>>(),
                    "root_fen": selected.semantic.nodes[work.root].position.to_fen(),
                });
                let first_move = work.original_first;
                let first = work.clone();
                let duplicate = work.clone();
                let mut progress = |_: BoardMove, _: &PalsStores, _: &[RoleRecord]| {};
                for (action, queued) in [
                    ("accepted_work", Some(first)),
                    ("same_work_duplicate", Some(duplicate)),
                    ("drain", None),
                ] {
                    if Instant::now() >= budget.deadline {
                        complete = false;
                        break;
                    }
                    let before = *counters;
                    let queue_before = queue_state(selected, first_move);
                    let work_before = observed_work(selected, before);
                    // Only the actual API entry-to-return span is timed. Work
                    // cloning, snapshots, assertions and JSON stay outside.
                    let (outcome, elapsed) = if let Some(work) = queued {
                        let started = Instant::now();
                        let outcome = selected.queue_or_recheck_frozen(
                            work,
                            budget,
                            &cancel,
                            counters,
                            &mut progress,
                        );
                        (outcome, started.elapsed())
                    } else {
                        let started = Instant::now();
                        let outcome =
                            selected.drain_repair_queue(budget, &cancel, counters, &mut progress);
                        (outcome, started.elapsed())
                    };
                    let after = *counters;
                    let work_after = observed_work(selected, after);
                    let stopped = selected.stopped(budget, &cancel);
                    if measured || outcome.is_err() || stopped.is_some() {
                        rows.push(json!({
                            "phase": if measured { "measurement" } else { "warmup" },
                            "trial": trial,
                            "policy": policy,
                            "action": action,
                            "question": question,
                            "max_rounds": budget.max_rounds,
                            "max_cpu_nodes": budget.max_cpu_nodes,
                            "cpu_depth": budget.cpu_depth,
                            "max_role_calls": selected.config.max_role_calls,
                            "external_evidence_added_between_actions": 0,
                            "api_ns": u64::try_from(elapsed.as_nanos()).ok(),
                            "span_scope": if work_before == work_after {
                                "queue_management_only"
                            } else {
                                "cpu_mock_recheck_or_repair_included"
                            },
                            "raw_counters_before": raw_counters(before),
                            "raw_counters_after": raw_counters(after),
                            "queue_before": queue_before,
                            "queue_after": queue_state(selected, first_move),
                            "original_error": outcome.as_ref().err().map(|e| format!("{e:?}")),
                            "stop_cause": stopped.map(|s| format!("{s:?}")),
                        }));
                    }
                    complete &= outcome.is_ok()
                        && stopped.is_none()
                        && selected.semantic.repair_queue.len() <= MAX_PENDING_REPAIRS
                        && selected
                            .semantic
                            .repair_counts
                            .values()
                            .all(|count| *count <= MAX_REPAIRS_PER_FIRST)
                        && selected
                            .semantic
                            .records
                            .iter()
                            .filter(|record| record.kind == RecordKind::Repair)
                            .all(|record| record.line.first() == Some(&first_move));
                    if !complete {
                        break;
                    }
                }
                workloads.push(observed_work(selected, *counters));
                if !complete {
                    break;
                }
            }
            if measured {
                let work_matches = workloads.len() == 2 && workloads[0] == workloads[1];
                comparisons.push(json!({
                    "trial": trial,
                    "same_input_script": true,
                    "same_original_deadline": true,
                    "same_node_limit": budget.max_cpu_nodes,
                    "same_depth": budget.cpu_depth,
                    "observed_work_matches": work_matches,
                    "comparability": if !complete || !work_matches {
                        "not_comparable"
                    } else {
                        "matched_cpu_mock_api_work_only"
                    },
                    "cost_effect_ratio": null,
                }));
            }
            if !complete {
                break;
            }
            if measured {
                samples_completed += 1;
            } else {
                warmups_completed += 1;
            }
        }
        let report = json!({
            "schema": "rz-pals-fixed-repair-api-cost/1",
            "scope": "CPU/mock queue API spans; drain includes actual fixture recheck/repair work",
            "product_disabled": "not_called: production Disabled never enters these queue APIs",
            "backend": "existing WdlFixture and owned CPU checker",
            "physical_nn_inputs": null,
            "limits": { "warmups": WARMUPS, "samples": SAMPLES, "wall_ms": 45000,
                "output_bytes_max": MAX_OUTPUT_BYTES, "pending": MAX_PENDING_REPAIRS,
                "repairs_per_first_move": MAX_REPAIRS_PER_FIRST },
            "warmups_completed": warmups_completed,
            "samples_completed": samples_completed,
            "complete": complete && Instant::now() < stop_at,
            "comparisons": comparisons,
            "api_spans": rows,
        });
        let encoded = serde_json::to_string(&report).unwrap();
        assert!(encoded.len() <= MAX_OUTPUT_BYTES);
        println!("PALS_QUEUE_COST_JSON {encoded}");
        // Failures retain their raw report above; they are never successful
        // samples or a reason to change the selected policy/budgets silently.
        assert_eq!(report["complete"], json!(true));
        assert_eq!(warmups_completed, WARMUPS);
        assert_eq!(samples_completed, SAMPLES);
    }

    #[test]
    fn own_checker_and_model_resolver_are_independent_and_legacy_constructor_is_preserved() {
        let mut selected = engine(
            PostRepairRecheckPolicy::Disabled,
            ResolverPolicy::ModelWdlRestricted,
        );
        let position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let report = selected.search(&position, limits(), &cancel).unwrap();
        assert_eq!(report.resolver_version, MODEL_WDL_RESOLVER_VERSION);
        assert!(matches!(report.checker_identity, CheckerIdentity::Owned(_)));
        assert!(report.counters.value_calls > 0);
        assert_eq!(report.score, None);
        assert_eq!(selected.search_identity(), PALS_FOLLOWUP_SEARCH_VERSION);
        let legacy = PalsEngine::new(
            PalsConfig::default(),
            LegalOrderRoleMock,
            CpuEngine::new(CpuConfig::default()).unwrap(),
        )
        .unwrap();
        assert_eq!(legacy.resolver_policy(), ResolverPolicy::OwnRawRestricted);
        assert_eq!(legacy.search_identity(), PALS_SEARCH_VERSION);
    }
    #[test]
    fn common_recheck_uses_actual_c_continuation_and_same_fresh_record_revision() {
        for resolver in [
            ResolverPolicy::OwnRawRestricted,
            ResolverPolicy::ModelWdlRestricted,
        ] {
            let mut selected = engine(PostRepairRecheckPolicy::FrozenModelWdlV2, resolver);
            let mut counters = PalsCounters::default();
            let work = accepted_repair(&mut selected, &mut counters);
            let cancel = AtomicBool::new(false);
            let frontier = selected
                .frozen_recheck_once(&work, limits(), &cancel, &mut counters, &mut |_, _, _| {})
                .unwrap()
                .unwrap();
            assert_eq!(
                frontier.counterline,
                line(&["e2e4", "e7e5", "f1c4", "g8f6", "d2d4", "e5d4"])
            );
            assert_eq!(counters.frozen_wdl_comparisons, 1);
            assert_eq!(counters.supported_refutations, 1);
            let final_event = selected.semantic.model.finished.last().unwrap();
            assert_eq!(
                final_event.0,
                RecheckDisposition::ConditionalRefutationPublished
            );
            assert!(final_event.1);
            assert!(final_event.2.is_some());
            assert_eq!(final_event.2, final_event.3);
            let revisions = &selected.semantic.model.revisions;
            assert_eq!(
                revisions[revisions.len() - 1],
                revisions[revisions.len() - 2]
            );
            let conclusion = selected
                .semantic
                .stores
                .situations
                .get(selected.semantic.nodes[work.root].situation)
                .unwrap()
                .conclusions
                .get(work.repaired_line)
                .unwrap();
            let observation = selected
                .semantic
                .stores
                .observations
                .get(conclusion.evidence.unwrap())
                .unwrap();
            assert!(matches!(observation.score, RawScore::ConditionalWdl { .. }));
            assert!(observation.value_identity.is_none());
            assert!(observation.model_value_identity.is_none());
            assert!(observation.model_value_input.is_none());
        }
    }
    #[test]
    fn iterative_queue_bounds_questions_and_never_changes_original_first_move() {
        let mut selected = engine(
            PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
            ResolverPolicy::ModelWdlRestricted,
        );
        let mut counters = PalsCounters::default();
        let work = accepted_repair(&mut selected, &mut counters);
        let cancel = AtomicBool::new(false);
        let budget = limits();
        selected
            .queue_or_recheck_frozen(
                work.clone(),
                budget,
                &cancel,
                &mut counters,
                &mut |_, _, _| {},
            )
            .unwrap();
        selected
            .queue_or_recheck_frozen(
                work.clone(),
                budget,
                &cancel,
                &mut counters,
                &mut |_, _, _| {},
            )
            .unwrap();
        assert_eq!(selected.semantic.repair_queue.len(), 1);
        assert_eq!(counters.repair_queue_deduplicated, 1);
        for _ in 0..MAX_REPAIRS_PER_FIRST {
            assert!(selected.admit_iterative_repair(work.original_first));
        }
        assert!(!selected.admit_iterative_repair(work.original_first));
        selected
            .drain_repair_queue(budget, &cancel, &mut counters, &mut |_, _, _| {})
            .unwrap();
        assert!(selected.semantic.repair_queue.is_empty());
        assert_eq!(selected.semantic.repair_counts[&work.original_first], 3);
        assert!(
            selected
                .semantic
                .records
                .iter()
                .filter(|r| r.kind == RecordKind::Repair)
                .all(|r| r.line.first() == Some(&work.original_first))
        );
        selected.new_game();
        assert!(selected.semantic.repair_counts.is_empty());
        assert!(selected.semantic.repair_questions.is_empty());
    }
    #[test]
    fn endpoint_perspective_and_missing_or_different_revision_never_mint_completion() {
        let mut selected = engine(
            PostRepairRecheckPolicy::FrozenModelWdlV2,
            ResolverPolicy::ModelWdlRestricted,
        );
        let position = Position::startpos();
        selected
            .semantic
            .stores
            .focus_actual_moves(position.snapshot())
            .unwrap();
        let root = selected.intern(position).unwrap();
        assert_eq!(selected.endpoint_expectation(root, Color::White), None);
        let mut counters = PalsCounters::default();
        selected
            .evaluate_model_value(root, &[], limits(), &AtomicBool::new(false), &mut counters)
            .unwrap();
        let white = selected.endpoint_expectation(root, Color::White).unwrap();
        assert_eq!(
            selected.endpoint_expectation(root, Color::Black),
            Some(-white)
        );
        let snapshot = selected.semantic.nodes[root].position.snapshot();
        let endpoint = PalsEngine::<WdlFixture>::frozen_endpoint(
            &selected.semantic.nodes[root],
            &selected.semantic.stores,
            &snapshot,
            9999,
        );
        assert!(matches!(
            endpoint.evidence,
            RecheckEndpointEvidence::ModelWdl {
                context_revision: 0,
                ..
            }
        ));
        selected.semantic.nodes[root].model_value_revision = Some(9999);
        let endpoint = PalsEngine::<WdlFixture>::frozen_endpoint(
            &selected.semantic.nodes[root],
            &selected.semantic.stores,
            &snapshot,
            9999,
        );
        assert!(matches!(
            endpoint.evidence,
            RecheckEndpointEvidence::InvalidProvenance { .. }
        ));
        selected.semantic.nodes[root].model_observation = None;
        let endpoint = PalsEngine::<WdlFixture>::frozen_endpoint(
            &selected.semantic.nodes[root],
            &selected.semantic.stores,
            &snapshot,
            0,
        );
        assert!(matches!(
            endpoint.evidence,
            RecheckEndpointEvidence::Unobserved
        ));
    }

    #[test]
    fn actual_iterative_repair_supersedes_conditional_refutation_with_fresh_pair() {
        let mut selected = engine(
            PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
            ResolverPolicy::ModelWdlRestricted,
        );
        let mut counters = PalsCounters::default();
        let work = accepted_repair(&mut selected, &mut counters);
        let cancel = AtomicBool::new(false);
        let budget = limits();
        selected
            .queue_or_recheck_frozen(
                work.clone(),
                budget,
                &cancel,
                &mut counters,
                &mut |_, _, _| {},
            )
            .unwrap();
        selected
            .drain_repair_queue(budget, &cancel, &mut counters, &mut |_, _, _| {})
            .unwrap();
        assert!(counters.accepted_repair_outputs > 0);
        assert!(counters.supported_repairs > 0);
        assert!(selected.semantic.repair_counts[&work.original_first] <= MAX_REPAIRS_PER_FIRST);
        assert!(selected.semantic.repair_queue.is_empty());
        let conclusion = selected
            .semantic
            .stores
            .situations
            .get(selected.semantic.nodes[work.root].situation)
            .unwrap()
            .conclusions
            .get(work.repaired_line)
            .unwrap();
        let ContinuationStatus::RepairedBy(repaired_line) = conclusion.status else {
            panic!("an accepted new P line must replace its actual conditional refutation");
        };
        assert_eq!(
            selected
                .semantic
                .stores
                .lines
                .moves(repaired_line)
                .unwrap()
                .first(),
            Some(&work.original_first)
        );
        let evidence = selected
            .semantic
            .stores
            .observations
            .get(conclusion.evidence.unwrap())
            .unwrap();
        assert_eq!(evidence.kind, ObservationKind::Repair);
        let refutation = selected
            .semantic
            .stores
            .observations
            .get(evidence.supersedes.unwrap())
            .unwrap();
        assert_eq!(refutation.kind, ObservationKind::Refutation);
        assert_eq!(refutation.line, Some(work.repaired_line));
        let RawScore::ConditionalRepairWdl {
            before,
            after,
            context_revision,
            ..
        } = evidence.score
        else {
            panic!("model repair requires immutable paired raw observations");
        };
        for observation in [before, after] {
            assert!(
                matches!(selected.semantic.stores.observations.get(observation).unwrap().score,
                RawScore::ContextWdl { context_revision: actual, .. } if actual == context_revision)
            );
        }
    }
    #[test]
    fn pending_queue_accepts_64_rejects_65_and_deduplicates_existing_work() {
        let mut selected = engine(
            PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
            ResolverPolicy::ModelWdlRestricted,
        );
        let mut counters = PalsCounters::default();
        let work = accepted_repair(&mut selected, &mut counters);
        let cancel = AtomicBool::new(false);
        let budget = limits();
        for offset in 0..MAX_PENDING_REPAIRS {
            let mut question = work.clone();
            if offset > 0 {
                question.repaired_line = selected
                    .record(RecordKind::Repair, &work.repaired, None, 0, None, None)
                    .unwrap()
                    .unwrap()
                    .0;
                question.repair_record_revision = selected.semantic.revision;
            }
            selected
                .queue_or_recheck_frozen(
                    question,
                    budget,
                    &cancel,
                    &mut counters,
                    &mut |_, _, _| {},
                )
                .unwrap();
        }
        assert_eq!(selected.semantic.repair_queue.len(), MAX_PENDING_REPAIRS);
        assert_eq!(counters.repair_queue_enqueued, MAX_PENDING_REPAIRS as u64);
        assert_eq!(counters.repair_queue_peak, MAX_PENDING_REPAIRS);
        selected
            .queue_or_recheck_frozen(
                work.clone(),
                budget,
                &cancel,
                &mut counters,
                &mut |_, _, _| {},
            )
            .unwrap();
        assert_eq!(counters.repair_queue_deduplicated, 1);
        let mut overflow = work;
        overflow.repaired_line = selected
            .record(RecordKind::Repair, &overflow.repaired, None, 0, None, None)
            .unwrap()
            .unwrap()
            .0;
        overflow.repair_record_revision = selected.semantic.revision;
        assert!(matches!(
            selected.queue_or_recheck_frozen(
                overflow,
                budget,
                &cancel,
                &mut counters,
                &mut |_, _, _| {}
            ),
            Err(PalsError::Capacity)
        ));
        assert_eq!(selected.semantic.repair_queue.len(), MAX_PENDING_REPAIRS);
        assert_eq!(counters.repair_queue_enqueued, MAX_PENDING_REPAIRS as u64);
        assert_eq!(counters.critic_calls, 0);
        assert!(selected.semantic.repair_counts.is_empty());
    }
    #[test]
    fn frozen_v2_actual_mixed_rules_and_model_endpoints_remain_unresolved() {
        for resolver in [
            ResolverPolicy::OwnRawRestricted,
            ResolverPolicy::ModelWdlRestricted,
        ] {
            for repaired_terminal in [false, true] {
                let mut selected = engine(PostRepairRecheckPolicy::FrozenModelWdlV2, resolver);
                let mut counters = PalsCounters::default();
                let terminal = line(&["h2h3", "a7a6", "f2f3", "e7e5", "g2g4", "d8h4"]);
                let ongoing = line(&["h2h3", "a7a6", "f2f3", "b8c6", "g2g4", "e7e5"]);
                let (repaired, counter) = if repaired_terminal {
                    (terminal, ongoing)
                } else {
                    (ongoing, terminal)
                };
                let mut refutation = repaired.clone();
                refutation[2] = BoardMove::from_uci("g1f3").unwrap();
                selected.semantic.model.replies = Some([counter[3], counter[4], counter[5]]);
                let work =
                    repair_work_for_lines(&mut selected, &mut counters, refutation, repaired);
                assert_eq!(
                    selected.semantic.nodes[work.repaired_leaf]
                        .terminal
                        .is_some(),
                    repaired_terminal
                );
                let frontier = selected
                    .frozen_recheck_once(
                        &work,
                        limits(),
                        &AtomicBool::new(false),
                        &mut counters,
                        &mut |_, _, _| {},
                    )
                    .unwrap()
                    .unwrap();
                assert_eq!(frontier.counterline, counter);
                assert_eq!(
                    selected.semantic.nodes[frontier.counter_leaf]
                        .terminal
                        .is_some(),
                    !repaired_terminal
                );
                let ongoing_leaf = if repaired_terminal {
                    frontier.counter_leaf
                } else {
                    work.repaired_leaf
                };
                assert!(selected.semantic.nodes[ongoing_leaf].model_value.is_some());
                assert!(
                    selected.semantic.nodes[ongoing_leaf]
                        .model_observation
                        .is_some()
                );
                assert!(frontier.publication.is_none());
                assert_eq!(counters.frozen_wdl_comparisons, 0);
                assert_eq!(counters.frozen_wdl_unresolved, 1);
                assert_eq!(counters.supported_refutations, 0);
                let event = selected.semantic.model.finished.last().unwrap();
                assert_eq!(event.0, RecheckDisposition::IncomparableEvidence);
                assert!(!event.1);
                assert!(
                    selected
                        .semantic
                        .stores
                        .situations
                        .get(selected.semantic.nodes[work.root].situation)
                        .unwrap()
                        .conclusions
                        .get(work.repaired_line)
                        .is_none()
                );
            }
        }
    }
}
