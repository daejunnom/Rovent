//! Conditional support uses completed own-CPU estimates or two exact Rules
//! terminals. Foreign CP and independently submitted model WDL never acquire
//! authority through this path. The manager performs hot commits only.
use super::*;

const SUPPORTED_COUNTER_LIMIT: usize = 64;

/// Immutable baseline of one conditional refutation. The active conclusion is
/// checked again before Repair, so another counter cannot silently replace it.
#[derive(Clone)]
pub(super) struct SupportedCounter {
    proposal_line: LineId,
    refutation: ObservationId,
    counter_value: ConditionalEndpoint,
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use rz_contracts::{ClockDomain, GameGeneration, MonotonicTick, ProcessEpoch, RootGeneration};

    type Port = AdaptiveEnginePort<LegalOrderRoleMock>;

    fn line(moves: &[&str]) -> Vec<BoardMove> {
        moves
            .iter()
            .map(|text| BoardMove::from_uci(text).unwrap())
            .collect()
    }

    pub(in crate::pals::engine::adaptive_port) fn port(position: Position) -> Port {
        port_with_line_plies(position, 4)
    }

    pub(in crate::pals::engine::adaptive_port) fn port_with_line_plies(
        position: Position,
        line_plies: usize,
    ) -> Port {
        let cpu = CpuEngine::new(crate::cpu::CpuConfig {
            tt_entries: 16,
            max_depth: 8,
            ..Default::default()
        })
        .unwrap();
        let engine = PalsEngine::new(
            PalsConfig {
                line_plies,
                ..Default::default()
            },
            LegalOrderRoleMock,
            cpu,
        )
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
        let limits = PalsLimits {
            max_rounds: 1,
            max_cpu_nodes: 2000,
            cpu_depth: 2,
            deadline: Instant::now() + Duration::from_secs(20),
        };
        let owners = Arc::new(std::sync::atomic::AtomicU64::new(1));
        Port::new(
            engine,
            position,
            authority,
            Deadline {
                clock: ClockDomain(authority.epoch),
                at: MonotonicTick(20_000_000_000),
            },
            limits,
            Arc::new(move || Ok(OwnerId(owners.fetch_add(1, Ordering::Relaxed)))),
        )
        .unwrap()
    }

    // Contract fixtures below fabricate no physical search claims. They seal
    // checked hot task/observation inputs to exercise the semantic consumer.
    pub(in crate::pals::engine::adaptive_port) fn accepted_raw(
        port: &mut Port,
        moves: &[BoardMove],
        score: i32,
        depth: u16,
    ) -> ObservationId {
        raw_fixture(port, moves, score, depth, false).0
    }

    pub(in crate::pals::engine::adaptive_port) fn accepted_raw_source(
        port: &mut Port,
        moves: &[BoardMove],
        score: i32,
        depth: u16,
    ) -> (ObservationId, TaskConsumer) {
        raw_fixture(port, moves, score, depth, false)
    }

    fn raw_fixture(
        port: &mut Port,
        moves: &[BoardMove],
        score: i32,
        depth: u16,
        paused: bool,
    ) -> (ObservationId, TaskConsumer) {
        let node = port.replay_node(moves).unwrap();
        let state = port.engine.semantic.nodes[node].state;
        let situation = port.engine.semantic.nodes[node].situation;
        let identity = port.engine.researcher.cpu_registered_value.clone().unwrap();
        let descriptor = port.engine.owned_descriptor().unwrap();
        let profile = stable_id(descriptor.config.profile.identity());
        let condition = stable_id(&format!(
            "{};q={}",
            descriptor.search_identity, descriptor.config.quiescence_ply
        ));
        let key = TaskKey {
            state,
            line: None,
            question: TaskQuestion::AnalyzePosition,
            root_moves: Vec::new(),
            model: 0,
            epoch: 0,
            value_identity: Some(identity.clone()),
            checker_identity: None,
            cpu_condition: Some(port.engine.cpu_condition()),
            profile,
            condition,
            input_revision: 0,
            requested_depth: port.limits.cpu_depth,
            node_budget: 100,
        };
        port.engine.semantic.consumer_id += 1;
        let consumer = TaskConsumer {
            id: port.engine.semantic.consumer_id,
            situation,
            revision: port
                .engine
                .semantic
                .stores
                .situations
                .get(situation)
                .unwrap()
                .revision,
            generation: port.engine.semantic.stores.generation(),
            deadline_tick: 100,
        };
        let TaskAdmission::Start(execution) = port
            .engine
            .semantic
            .stores
            .hot_request_task(key.clone(), consumer, 0)
            .unwrap()
        else {
            panic!("fresh fixture task")
        };
        let observation = port
            .engine
            .semantic
            .stores
            .hot_append_observation(Observation {
                state,
                line: None,
                source: 1,
                epoch: 0,
                value_identity: Some(identity.clone()),
                checker_identity: None,
                checker_work: None,
                external_report: None,
                model_value_identity: None,
                model_value_input: None,
                cpu_condition: key.cpu_condition.clone(),
                cpu_pv: None,
                scope: EvidenceScope::DepthLimited {
                    depth,
                    profile,
                    condition,
                },
                score: RawScore::Cpu {
                    value: score,
                    perspective: port.engine.semantic.nodes[node].position.side_to_move(),
                    bound: BoundKind::ExactWithinSearch,
                },
                budget: 1,
                kind: ObservationKind::CpuAnalysis,
                supersedes: None,
                execution: Some(execution),
            })
            .unwrap();
        if paused {
            port.engine
                .semantic
                .stores
                .pause_task(execution, 7, Some(observation))
                .unwrap();
        } else {
            port.engine
                .semantic
                .stores
                .complete_task(execution, observation)
                .unwrap();
            port.engine
                .semantic
                .stores
                .consume_task(execution, consumer.id, 0)
                .unwrap();
        }
        port.engine.semantic.nodes[node].evidence = Some(CpuEvidence {
            score,
            depth,
            scope: CpuScoreScope::CompletedIteration,
            value_identity: identity,
            provenance: Some((observation, execution)),
            context: None,
        });
        (observation, consumer)
    }

    fn question(
        port: &mut Port,
        stage: RefinementStage,
        proposal: &[BoardMove],
        counter: &[BoardMove],
        repaired: Option<&[BoardMove]>,
    ) -> AdaptivePortQuestion {
        let endpoint = repaired.unwrap_or(counter);
        let node = port.replay_node(endpoint).unwrap();
        // Actual producer uses this same input constructor after publishing its
        // completed public line; the synthetic fixture keeps that relation.
        port.record_hot(RecordKind::Proposal, proposal, None)
            .unwrap();
        port.record_hot(RecordKind::Counterexample, counter, None)
            .unwrap();
        if let Some(repaired) = repaired {
            port.record_hot(RecordKind::Repair, repaired, None).unwrap();
        }
        port.input(
            node,
            stage,
            proposal.first().copied(),
            endpoint.to_vec(),
            proposal.to_vec(),
            Some(counter.to_vec()),
            repaired.map(<[BoardMove]>::to_vec),
            Some(1),
            false,
            None,
            None,
        )
        .unwrap()
        .question
        .as_ref()
        .clone()
    }

    fn branches() -> (Vec<BoardMove>, Vec<BoardMove>, Vec<BoardMove>) {
        (
            line(&["e2e4", "e7e5", "g1f3", "b8c6"]),
            line(&["e2e4", "c7c5", "g1f3", "b8c6"]),
            line(&["e2e4", "c7c5", "d2d4", "b8c6"]),
        )
    }

    #[test]
    fn full_zero_step_repair_reuses_checked_cpu_without_new_execution_or_witness() {
        let mut port = port_with_line_plies(Position::startpos(), 2);
        let proposal = line(&["e2e4", "e7e5"]);
        let counter = line(&["e2e4", "c7c5"]);
        accepted_raw(&mut port, &counter, -100, 2);
        port.record_hot(RecordKind::Proposal, &proposal, None)
            .unwrap();
        port.record_hot(RecordKind::Counterexample, &counter, None)
            .unwrap();
        let node = port.replay_node(&counter).unwrap();
        let observations = port.engine.semantic.stores.observations.len();
        let executions = port.engine.semantic.stores.tasks.len();
        assert!(port.engine.admit_iterative_repair(counter[0]));
        let transition = port
            .begin_repair_continuation(
                node,
                Some(counter[0]),
                counter.clone(),
                proposal,
                counter.clone(),
                1,
                None,
            )
            .unwrap();
        assert!(matches!(
            transition,
            AdaptivePortTransition::Complete {
                scope: FrameCompletionScope::NoRepairContinuation
            }
        ));
        assert_eq!(port.engine.semantic.stores.tasks.len(), executions);
        assert_eq!(
            port.engine.semantic.stores.observations.len(),
            observations + 1
        );
        let repair = port.engine.semantic.records.last().unwrap();
        assert_eq!(repair.kind, RecordKind::Repair);
        assert_eq!(repair.line, counter);
        assert!(repair.parent_revision.is_some());
        assert_eq!(port.counters.repairs, 1);
        assert_eq!(port.counters.evidence_cache_hits, 1);
        assert_eq!(port.counters.consumed_cached_cpu_values, 1);
        assert_eq!(port.counters.cpu_tasks_requested, 0);
        assert_eq!(port.counters.role_calls, 0);
        assert_eq!(port.counters.accepted_repair_outputs, 0);
        assert_eq!(port.counters.supported_repairs, 0);
    }

    #[test]
    fn zero_step_model_resolver_still_requires_fresh_value_for_new_revision() {
        let mut port = port_with_line_plies(Position::startpos(), 2);
        let proposal = line(&["e2e4", "e7e5"]);
        let counter = line(&["e2e4", "c7c5"]);
        accepted_raw(&mut port, &counter, -100, 2);
        port.record_hot(RecordKind::Proposal, &proposal, None)
            .unwrap();
        port.record_hot(RecordKind::Counterexample, &counter, None)
            .unwrap();
        // Authored metadata variation checks the consumer boundary; no model
        // output, physical execution or WDL value is created in this fixture.
        port.engine.resolver = ResolverPolicy::ModelWdlRestricted;
        let node = port.replay_node(&counter).unwrap();
        let transition = port
            .begin_repair_continuation(
                node,
                Some(counter[0]),
                counter.clone(),
                proposal,
                counter,
                1,
                None,
            )
            .unwrap();
        let materialized = port
            .materialize_step(AdaptivePortStep {
                evidence: ContractDigest([8; 32]),
                transition,
                fresh_pair: None,
                rules_pair: None,
                consumption_gate: None,
                feedback_gate: None,
                queue_question: None,
            })
            .unwrap();
        let FrameTransition::Continue(next) = materialized.transition else {
            panic!("zero-step model value requires actual Fresh continuation")
        };
        assert_eq!(next.stage, RefinementStage::RepairValueFresh);
        assert!(next.question.repaired.is_some());
        assert_eq!(port.counters.evidence_cache_hits, 1);
        assert_eq!(port.counters.value_calls, 0);
        assert_eq!(port.counters.accepted_repair_outputs, 0);
    }

    #[test]
    fn zero_step_retained_stack_does_not_reuse_completed_iteration_as_completion() {
        let mut port = port_with_line_plies(Position::startpos(), 2);
        let proposal = line(&["e2e4", "e7e5"]);
        let counter = line(&["e2e4", "c7c5"]);
        raw_fixture(&mut port, &counter, -100, 2, true);
        port.record_hot(RecordKind::Proposal, &proposal, None)
            .unwrap();
        port.record_hot(RecordKind::Counterexample, &counter, None)
            .unwrap();
        let node = port.replay_node(&counter).unwrap();
        let transition = port
            .begin_repair_continuation(
                node,
                Some(counter[0]),
                counter.clone(),
                proposal,
                counter,
                1,
                None,
            )
            .unwrap();
        let materialized = port
            .materialize_step(AdaptivePortStep {
                evidence: ContractDigest([9; 32]),
                transition,
                fresh_pair: None,
                rules_pair: None,
                consumption_gate: None,
                feedback_gate: None,
                queue_question: None,
            })
            .unwrap();
        let FrameTransition::Continue(next) = materialized.transition else {
            panic!("retained stack is not settled cache authority")
        };
        assert_eq!(next.stage, RefinementStage::RepairValidation);
        assert_eq!(port.counters.evidence_cache_hits, 0);
        assert_eq!(port.counters.consumed_cached_cpu_values, 0);
    }

    #[test]
    fn terminal_zero_step_repair_keeps_rules_endpoint_without_model_or_cpu_work() {
        let mut position = Position::startpos();
        for movement in line(&["f2f3", "e7e5"]) {
            position.make_move(movement).unwrap();
        }
        let mut port = port_with_line_plies(position, 4);
        let proposal = line(&["g2g4", "a7a6"]);
        let counter = line(&["g2g4", "d8h4"]);
        let node = port.replay_node(&counter).unwrap();
        assert!(port.engine.semantic.nodes[node].terminal.is_some());
        let transition = port
            .begin_repair_continuation(
                node,
                Some(counter[0]),
                counter.clone(),
                proposal,
                counter,
                1,
                None,
            )
            .unwrap();
        assert!(matches!(
            transition,
            AdaptivePortTransition::Complete {
                scope: FrameCompletionScope::NoRepairContinuation
            }
        ));
        assert_eq!(port.counters.repairs, 1);
        assert_eq!(port.counters.role_calls, 0);
        assert_eq!(port.counters.cpu_tasks_requested, 0);
        assert_eq!(port.counters.evidence_cache_hits, 0);
    }

    #[test]
    fn strict_counter_and_own_repair_keep_conditional_estimate_scope() {
        let mut port = port(Position::startpos());
        let (proposal, counter, repaired) = branches();
        accepted_raw(&mut port, &proposal, 100, 2);
        accepted_raw(&mut port, &counter, -100, 2);
        let query = question(
            &mut port,
            RefinementStage::CounterValidation,
            &proposal,
            &counter,
            None,
        );
        assert!(port.publish_supported_counter(&query).unwrap());
        let before = port.engine.semantic.stores.observations.len();
        assert!(!port.publish_supported_counter(&query).unwrap());
        assert_eq!(before, port.engine.semantic.stores.observations.len());
        let support = port.supported_counters.values().next().unwrap().clone();
        assert_eq!(
            port.conditional_witness_kind(support.refutation),
            Some(ObservationKind::Refutation)
        );
        assert_eq!(
            port.conditional_witness_line(support.refutation),
            Some(counter.as_slice())
        );
        assert!(
            matches!(port.engine.semantic.stores.observations.get(support.refutation).unwrap().score,
            RawScore::Estimate { value, perspective: Color::White } if value == -100.0)
        );
        accepted_raw(&mut port, &repaired, 0, 2);
        let query = question(
            &mut port,
            RefinementStage::RepairValidation,
            &proposal,
            &counter,
            Some(&repaired),
        );
        assert!(port.publish_supported_repair(&query).unwrap());
        assert!(port.supported_counters.is_empty());
        let current = port
            .engine
            .semantic
            .stores
            .situations
            .get(port.engine.semantic.nodes[port.root].situation)
            .unwrap();
        let conclusion = current.conclusions.get(support.proposal_line).unwrap();
        assert!(matches!(
            conclusion.status,
            ContinuationStatus::RepairedBy(_)
        ));
        let observation = port
            .engine
            .semantic
            .stores
            .observations
            .get(conclusion.evidence.unwrap())
            .unwrap();
        assert_eq!(observation.supersedes, Some(support.refutation));
        assert!(matches!(
            observation.score,
            RawScore::Estimate { value: 0.0, .. }
        ));
        assert!(observation.execution.is_none());
        assert_eq!(
            port.conditional_witness_kind(conclusion.evidence.unwrap()),
            Some(ObservationKind::Repair)
        );
        assert_eq!(
            port.conditional_witness_line(conclusion.evidence.unwrap()),
            Some(repaired.as_slice())
        );
        assert_eq!(port.counters.supported_refutations, 1);
        assert_eq!(port.counters.supported_repairs, 1);
    }

    #[test]
    fn equal_value_or_different_completed_depth_does_not_refute() {
        for (score, depth) in [(100, 2), (-100, 3)] {
            let mut port = port(Position::startpos());
            let (proposal, counter, _) = branches();
            accepted_raw(&mut port, &proposal, 100, 2);
            accepted_raw(&mut port, &counter, score, depth);
            let query = question(
                &mut port,
                RefinementStage::CounterValidation,
                &proposal,
                &counter,
                None,
            );
            let before = port.engine.semantic.stores.observations.len();
            assert!(!port.publish_supported_counter(&query).unwrap());
            assert_eq!(before, port.engine.semantic.stores.observations.len());
            assert!(port.supported_counters.is_empty());
        }
    }

    #[test]
    fn retained_or_resumed_completed_iteration_is_progress_without_refutation() {
        let mut port = port(Position::startpos());
        let (proposal, counter, _) = branches();
        accepted_raw(&mut port, &proposal, 100, 2);
        let observation = raw_fixture(&mut port, &counter, -100, 2, true).0;
        let query = question(
            &mut port,
            RefinementStage::CounterValidation,
            &proposal,
            &counter,
            None,
        );
        assert!(!port.publish_supported_counter(&query).unwrap());
        assert!(port.supported_counters.is_empty());
        let execution = port
            .engine
            .semantic
            .stores
            .observations
            .get(observation)
            .unwrap()
            .execution
            .unwrap();
        port.engine
            .semantic
            .stores
            .hot_retire_paused(execution)
            .unwrap();
        assert!(!port.publish_supported_counter(&query).unwrap());
        assert!(port.supported_counters.is_empty());
        assert_eq!(port.counters.supported_refutations, 0);
    }

    #[test]
    fn changed_opponent_only_line_does_not_count_as_repair() {
        let mut port = port(Position::startpos());
        let (proposal, counter, _) = branches();
        let optimistic_opponent = line(&["e2e4", "c7c5", "g1f3", "d7d6"]);
        accepted_raw(&mut port, &proposal, 100, 2);
        accepted_raw(&mut port, &counter, -100, 2);
        let query = question(
            &mut port,
            RefinementStage::CounterValidation,
            &proposal,
            &counter,
            None,
        );
        assert!(port.publish_supported_counter(&query).unwrap());
        accepted_raw(&mut port, &optimistic_opponent, 200, 2);
        let query = question(
            &mut port,
            RefinementStage::RepairValidation,
            &proposal,
            &counter,
            Some(&optimistic_opponent),
        );
        let before = port.engine.semantic.stores.observations.len();
        assert!(!port.publish_supported_repair(&query).unwrap());
        assert_eq!(before, port.engine.semantic.stores.observations.len());
        assert_eq!(port.counters.supported_repairs, 0);
    }

    #[test]
    fn wrong_root_history_rejects_same_fen_endpoint() {
        let mut port = port(Position::startpos());
        let (proposal, counter, _) = branches();
        accepted_raw(&mut port, &proposal, 100, 2);
        accepted_raw(&mut port, &counter, -100, 2);
        let mut query = question(
            &mut port,
            RefinementStage::CounterValidation,
            &proposal,
            &counter,
            None,
        );
        // FEN preserves the board/counters but omits the actual move history.
        query.position = Position::from_fen(&query.position.to_fen()).unwrap();
        assert!(matches!(
            port.publish_supported_counter(&query),
            Err(PalsError::Review(ReviewError::AnchorMismatch))
        ));
        assert_eq!(port.counters.supported_refutations, 0);
    }

    #[test]
    fn source_observation_from_different_exact_conditions_is_rejected() {
        let mut port = port(Position::startpos());
        let (proposal, counter, _) = branches();
        accepted_raw(&mut port, &proposal, 100, 2);
        accepted_raw(&mut port, &counter, -100, 2);
        let node = port.replay_node(&counter).unwrap();
        let accepted = port.engine.semantic.nodes[node].evidence.as_mut().unwrap();
        // An unproven front-end value must not piggyback on a raw task receipt.
        accepted.score = -300;
        let query = question(
            &mut port,
            RefinementStage::CounterValidation,
            &proposal,
            &counter,
            None,
        );
        assert!(matches!(
            port.publish_supported_counter(&query),
            Err(PalsError::Store(StoreError::InvalidEvidence(_)))
        ));
        assert_eq!(port.counters.supported_refutations, 0);
    }

    #[test]
    fn two_rules_endpoints_compare_without_cpu_or_model_authority() {
        let mate = port(Position::from_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1").unwrap());
        let draw = port(Position::from_fen("7k/5Q2/5K2/8/8/8/8/8 b - - 0 1").unwrap());
        let mate_value = mate.conditional_endpoint(mate.root, 0).unwrap().unwrap();
        let draw_value = draw.conditional_endpoint(draw.root, 0).unwrap().unwrap();
        assert!(mate_value.comparable(&draw_value));
        assert_eq!(mate_value.root_value, -CPU_MATE_SCORE);
        assert_eq!(draw_value.root_value, 0);
        assert_eq!(
            mate.conditional_endpoint(mate.root, 1)
                .unwrap()
                .unwrap()
                .root_value,
            CPU_MATE_SCORE
        );
        assert_eq!(mate.counters.cpu_nodes, 0);
        assert_eq!(draw.counters.cpu_nodes, 0);
    }

    #[test]
    fn arbitrary_estimate_is_not_an_owner_issued_conditional_witness() {
        let mut port = port(Position::startpos());
        let (proposal, _, _) = branches();
        let focus = port
            .engine
            .semantic
            .stores
            .situations
            .get(port.engine.semantic.nodes[port.root].situation)
            .unwrap()
            .focus;
        let line = port
            .engine
            .semantic
            .stores
            .hot_append_line(focus, &proposal)
            .unwrap();
        let uncommitted = port
            .conditional_observation(line, ObservationKind::Refutation, -100, None)
            .unwrap();
        assert_eq!(port.conditional_witness_kind(uncommitted), None);
        assert_eq!(port.conditional_witness_line(uncommitted), None);
        assert_eq!(port.counters.supported_refutations, 0);
    }

    #[test]
    fn different_contextual_binding_is_not_a_comparable_raw_basis() {
        let mut port = port(Position::startpos());
        let (proposal, counter, _) = branches();
        accepted_raw(&mut port, &proposal, 100, 2);
        accepted_raw(&mut port, &counter, -100, 2);
        let proposal_node = port.replay_node(&proposal).unwrap();
        let counter_node = port.replay_node(&counter).unwrap();
        let mut old = port
            .conditional_endpoint(proposal_node, proposal.len())
            .unwrap()
            .unwrap();
        let mut new = port
            .conditional_endpoint(counter_node, counter.len())
            .unwrap()
            .unwrap();
        assert!(old.comparable(&new));
        let EndpointBasis::OwnCpu {
            base_condition: old_condition,
            ..
        } = &mut old.basis
        else {
            panic!("actual accepted own CPU fixture")
        };
        old_condition.push_str(";nnuec-context-binding=original-view");
        assert!(!old.comparable(&new));
        let EndpointBasis::OwnCpu {
            base_condition: new_condition,
            ..
        } = &mut new.basis
        else {
            panic!("actual accepted own CPU fixture")
        };
        new_condition.push_str(";nnuec-context-binding=other-view");
        assert!(!old.comparable(&new));
    }
}

#[derive(Clone)]
pub(super) struct ConditionalEndpoint {
    pub(super) root_value: i32,
    pub(super) basis: EndpointBasis,
}

#[derive(Clone)]
pub(super) enum EndpointBasis {
    Rules,
    OwnCpu {
        depth: u16,
        plies: usize,
        identity: CpuValueIdentity,
        base_condition: String,
        profile: u64,
        condition: u64,
        observation: ObservationId,
        execution: ExecutionId,
    },
}

impl ConditionalEndpoint {
    fn comparable(&self, other: &Self) -> bool {
        match (&self.basis, &other.basis) {
            (EndpointBasis::Rules, EndpointBasis::Rules) => true,
            (
                EndpointBasis::OwnCpu {
                    depth: a_depth,
                    plies: a_plies,
                    identity: a_identity,
                    base_condition: a_base,
                    profile: a_profile,
                    condition: a_condition,
                    ..
                },
                EndpointBasis::OwnCpu {
                    depth: b_depth,
                    plies: b_plies,
                    identity: b_identity,
                    base_condition: b_base,
                    profile: b_profile,
                    condition: b_condition,
                    ..
                },
            ) => {
                a_depth == b_depth
                    && a_plies == b_plies
                    && a_identity == b_identity
                    && a_base == b_base
                    && a_profile == b_profile
                    && a_condition == b_condition
            }
            _ => false,
        }
    }
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// Refute only this exact proposal continuation under comparable finite
    /// evidence. This does not blacklist the first move or certify a mate.
    pub(super) fn publish_supported_counter(
        &mut self,
        query: &AdaptivePortQuestion,
    ) -> Result<bool, PalsError> {
        self.check_live()?;
        if !matches!(
            query.stage,
            RefinementStage::CounterValidation | RefinementStage::CounterValueFresh
        ) {
            return Err(RoleError::InvalidOutput.into());
        }
        let (key, counter) = self.conditional_question(query, false)?;
        if self.supported_counters.contains_key(&key) {
            return Ok(false);
        }
        let proposal_node = self.replay_node(&query.proposal)?;
        let counter_node = self.replay_node(counter)?;
        self.check_conditional_endpoint(query, counter_node, counter)?;
        let Some(old) = self.conditional_endpoint(proposal_node, query.proposal.len())? else {
            return Ok(false);
        };
        let Some(new) = self.conditional_endpoint(counter_node, counter.len())? else {
            return Ok(false);
        };
        if !old.comparable(&new) || new.root_value >= old.root_value {
            return Ok(false);
        }
        if self.supported_counters.len() >= SUPPORTED_COUNTER_LIMIT {
            return Err(StoreError::PinSaturated("adaptive supported counters").into());
        }
        if self.conditional_witnesses.len() >= SUPPORTED_COUNTER_LIMIT {
            return Err(StoreError::PinSaturated("adaptive conditional witness registry").into());
        }
        // Preserve only original, already checked comparison sources when an
        // actual utility owner requested history. This creates no conclusion.
        let comparison = if self.utility_continuations.enabled() {
            Some(self.capture_utility_conditional_comparison(
                &query.proposal,
                &old,
                counter,
                &new,
                None,
            )?)
        } else {
            None
        };
        // All source references are accepted hot observations. Pin them before
        // publication; neither this operation nor the conclusion invokes I/O.
        self.pin_conditional_endpoint(&old)?;
        self.pin_conditional_endpoint(&new)?;
        let root = self.engine.semantic.nodes[self.root].situation;
        let focus = self.engine.semantic.stores.situations.get(root)?.focus;
        let proposal_line = self
            .engine
            .semantic
            .stores
            .hot_append_line(focus, &query.proposal)?;
        let refutation = self.conditional_observation(
            proposal_line,
            ObservationKind::Refutation,
            new.root_value,
            None,
        )?;
        self.engine
            .pin_archive_active_record(proposal_line, refutation)?;
        self.check_live()?;
        self.engine
            .semantic
            .stores
            .hot_refute_continuation(root, proposal_line, refutation)?;
        self.conditional_witnesses.insert(
            refutation,
            (ObservationKind::Refutation, counter.to_vec(), comparison),
        );
        self.supported_counters.insert(
            key,
            SupportedCounter {
                proposal_line,
                refutation,
                counter_value: new,
            },
        );
        self.counters.supported_refutations += 1;
        Ok(true)
    }

    /// Repair is strictly better than the retained counter estimate and must
    /// supersede the same still-active refutation. Unknown remains unresolved.
    pub(super) fn publish_supported_repair(
        &mut self,
        query: &AdaptivePortQuestion,
    ) -> Result<bool, PalsError> {
        self.check_live()?;
        if !matches!(
            query.stage,
            RefinementStage::RepairValidation | RefinementStage::RepairValueFresh
        ) {
            return Err(RoleError::InvalidOutput.into());
        }
        let (key, _) = self.conditional_question(query, true)?;
        let Some(previous) = self.supported_counters.get(&key).cloned() else {
            return Ok(false);
        };
        let root = self.engine.semantic.nodes[self.root].situation;
        let active = self
            .engine
            .semantic
            .stores
            .situations
            .get(root)?
            .conclusions
            .get(previous.proposal_line);
        if !active.is_some_and(|conclusion| {
            conclusion.status == ContinuationStatus::Refuted
                && conclusion.evidence == Some(previous.refutation)
        }) {
            // Raw evidence remains in the store. Only the no-longer-active
            // support registry entry is retired; no witness is deleted.
            self.supported_counters.remove(&key);
            return Ok(false);
        }
        let repaired = query.repaired.as_ref().ok_or(RoleError::InvalidOutput)?;
        if repaired == &query.proposal || !self.is_own_repair(query)? {
            return Ok(false);
        }
        let repaired_node = self.replay_node(repaired)?;
        self.check_conditional_endpoint(query, repaired_node, repaired)?;
        let Some(after) = self.conditional_endpoint(repaired_node, repaired.len())? else {
            return Ok(false);
        };
        if !previous.counter_value.comparable(&after)
            || after.root_value <= previous.counter_value.root_value
        {
            return Ok(false);
        }
        if self.conditional_witnesses.len() >= SUPPORTED_COUNTER_LIMIT {
            return Err(StoreError::PinSaturated("adaptive conditional witness registry").into());
        }
        let comparison = if self.utility_continuations.enabled() {
            Some(self.capture_utility_conditional_comparison(
                query.counter.as_deref().ok_or(RoleError::InvalidOutput)?,
                &previous.counter_value,
                repaired,
                &after,
                Some(previous.refutation),
            )?)
        } else {
            None
        };
        self.pin_conditional_endpoint(&previous.counter_value)?;
        self.pin_conditional_endpoint(&after)?;
        let focus = self.engine.semantic.stores.situations.get(root)?.focus;
        let repaired_line = self
            .engine
            .semantic
            .stores
            .hot_append_line(focus, repaired)?;
        if repaired_line == previous.proposal_line {
            return Ok(false);
        }
        let evidence = self.conditional_observation(
            repaired_line,
            ObservationKind::Repair,
            after.root_value,
            Some(previous.refutation),
        )?;
        self.engine
            .pin_archive_active_record(repaired_line, evidence)?;
        self.check_live()?;
        self.engine.semantic.stores.hot_repair(
            root,
            previous.proposal_line,
            repaired_line,
            evidence,
        )?;
        self.conditional_witnesses.insert(
            evidence,
            (ObservationKind::Repair, repaired.to_vec(), comparison),
        );
        self.supported_counters.remove(&key);
        self.counters.supported_repairs += 1;
        Ok(true)
    }

    /// Only successful semantic commits mint this authority. A caller cannot
    /// qualify an arbitrary finite Estimate as a newly supported V witness.
    pub(super) fn conditional_witness_kind(
        &self,
        observation: ObservationId,
    ) -> Option<ObservationKind> {
        self.conditional_witnesses
            .get(&observation)
            .map(|(kind, _, _)| *kind)
    }

    /// The line is the actual successful counter or revised own continuation,
    /// not the proposal line on which a conditional refutation is recorded.
    pub(super) fn conditional_witness_line(
        &self,
        observation: ObservationId,
    ) -> Option<&[BoardMove]> {
        self.conditional_witnesses
            .get(&observation)
            .map(|(_, line, _)| line.as_slice())
    }

    /// The digest is acceptance metadata, never a numeric model feature. It
    /// keeps independent counter branches and exact root generations separate.
    fn conditional_question<'a>(
        &self,
        query: &'a AdaptivePortQuestion,
        repairing: bool,
    ) -> Result<(ContractDigest, &'a [BoardMove]), PalsError> {
        let counter = query.counter.as_deref().ok_or(RoleError::InvalidOutput)?;
        let first = query.first.ok_or(RoleError::InvalidOutput)?;
        let attack = query.attack_ply.ok_or(RoleError::InvalidOutput)?;
        if query.proposal.first() != Some(&first)
            || counter.first() != Some(&first)
            || attack >= query.proposal.len()
            || attack >= counter.len()
            || query.proposal[..attack] != counter[..attack]
            || query.proposal[attack] == counter[attack]
            || query.proposal.len() > self.engine.config.line_plies
            || counter.len() > self.engine.config.line_plies
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let mut divergence = self.engine.semantic.nodes[self.root].position.clone();
        for movement in &query.proposal[..attack] {
            divergence.make_move(*movement)?;
        }
        if divergence.side_to_move()
            == self.engine.semantic.nodes[self.root]
                .position
                .side_to_move()
            || !matches!(
                divergence.classify_position()?.play_status,
                PlayStatus::Ongoing
            )
        {
            return Err(RoleError::InvalidOutput.into());
        }
        if repairing {
            let repaired = query.repaired.as_deref().ok_or(RoleError::InvalidOutput)?;
            if repaired.first() != Some(&first)
                || repaired.len() > self.engine.config.line_plies
                || repaired.get(..=attack) != counter.get(..=attack)
            {
                return Err(RoleError::InvalidOutput.into());
            }
        }
        let mut hash = Sha256::new();
        hash.update(b"rz-pals-conditional-counter/1\0");
        for value in [
            self.authority.epoch.0,
            self.authority.game.0,
            self.authority.root.0,
            self.engine.semantic.nodes[self.root].state.0 as u64,
            attack as u64,
        ] {
            hash.update(value.to_le_bytes());
        }
        hash.update(self.authority.implementation.0);
        for line in [&query.proposal[..], counter] {
            hash.update((line.len() as u64).to_le_bytes());
            for movement in line {
                hash.update(Move16::pack(*movement)?.bits().to_le_bytes());
            }
        }
        Ok((ContractDigest(hash.finalize().into()), counter))
    }

    fn check_conditional_endpoint(
        &self,
        query: &AdaptivePortQuestion,
        node: usize,
        line: &[BoardMove],
    ) -> Result<(), PalsError> {
        if query.node != node
            || query.context.prefix != line
            || query.context.state != self.engine.semantic.nodes[node].state
            || query.context.situation != self.engine.semantic.nodes[node].situation
            || !query
                .position
                .snapshot()
                .same_state(&self.engine.semantic.nodes[node].position.snapshot())
        {
            return Err(ReviewError::AnchorMismatch.into());
        }
        Ok(())
    }

    fn is_own_repair(&self, query: &AdaptivePortQuestion) -> Result<bool, PalsError> {
        let counter = query.counter.as_deref().ok_or(RoleError::InvalidOutput)?;
        let repaired = query.repaired.as_deref().ok_or(RoleError::InvalidOutput)?;
        let mut position = self.engine.semantic.nodes[self.root].position.clone();
        let root_color = position.side_to_move();
        for (old, new) in counter.iter().zip(repaired) {
            if old != new {
                return Ok(position.side_to_move() == root_color);
            }
            position.make_move(*old)?;
        }
        // Extending an unchanged prefix is not an actual revised own choice.
        Ok(false)
    }

    fn conditional_endpoint(
        &self,
        node: usize,
        plies: usize,
    ) -> Result<Option<ConditionalEndpoint>, PalsError> {
        let endpoint = &self.engine.semantic.nodes[node];
        if !self
            .engine
            .semantic
            .stores
            .states
            .get(endpoint.state)?
            .same_state(&endpoint.position.snapshot())
            || self
                .engine
                .semantic
                .stores
                .situations
                .get(endpoint.situation)?
                .state
                != endpoint.state
        {
            return Err(ReviewError::AnchorMismatch.into());
        }
        // Conditional support uses exact automatic Rules termination only.
        // Intended-move draw-claim previews need an extra history slot and do
        // not change whether this exact endpoint is a terminal or ongoing.
        let legal = endpoint.position.ordered_legal_moves();
        let status = endpoint.position.play_status_from_view(&legal)?;
        if let Some((reason, value)) = endpoint.terminal {
            let PlayStatus::Terminal {
                reason: actual_reason,
                winner,
            } = status
            else {
                return Err(ReviewError::AnchorMismatch.into());
            };
            let actual_value = match winner {
                Some(color) if color == endpoint.position.side_to_move() => CPU_MATE_SCORE,
                Some(_) => -CPU_MATE_SCORE,
                None => 0,
            };
            if actual_reason != reason || value != actual_value {
                return Err(ReviewError::AnchorMismatch.into());
            }
            return Ok(Some(ConditionalEndpoint {
                root_value: if plies % 2 == 0 { value } else { -value },
                basis: EndpointBasis::Rules,
            }));
        }
        // Foreign raw observations and model-resolver estimates are distinct
        // namespaces. Only a separately sealed Fresh pair can compare WDL.
        if status != PlayStatus::Ongoing {
            return Err(StoreError::InvalidEvidence(
                "adaptive CPU endpoint is an actual Rules terminal",
            )
            .into());
        }
        if self.engine.is_external() || self.engine.uses_model_resolver() {
            return Ok(None);
        }
        self.checked_completed_cpu_endpoint(node, plies)
    }

    /// Exact accepted own-CPU evidence can skip another physical verification
    /// at the same endpoint. This does not reuse model WDL, count a new CPU
    /// execution, or turn that finite estimate into a game result.
    pub(super) fn has_reusable_completed_cpu(
        &self,
        node: usize,
        plies: usize,
    ) -> Result<bool, PalsError> {
        if self.engine.is_external() {
            return Ok(false);
        }
        Ok(self.checked_completed_cpu_endpoint(node, plies)?.is_some())
    }

    fn checked_completed_cpu_endpoint(
        &self,
        node: usize,
        plies: usize,
    ) -> Result<Option<ConditionalEndpoint>, PalsError> {
        let endpoint = &self.engine.semantic.nodes[node];
        if endpoint.terminal.is_some() {
            return Ok(None);
        }
        if !self
            .engine
            .semantic
            .stores
            .states
            .get(endpoint.state)?
            .same_state(&endpoint.position.snapshot())
            || self
                .engine
                .semantic
                .stores
                .situations
                .get(endpoint.situation)?
                .state
                != endpoint.state
        {
            return Err(ReviewError::AnchorMismatch.into());
        }
        let Some(accepted) = &endpoint.evidence else {
            return Ok(None);
        };
        if accepted.scope != CpuScoreScope::CompletedIteration
            || accepted.depth < self.limits.cpu_depth
            || Some(&accepted.value_identity)
                != self.engine.researcher.cpu_registered_value.as_ref()
        {
            return Ok(None);
        }
        let root_value = if plies % 2 == 0 {
            accepted.score
        } else {
            -accepted.score
        };
        let Some((observation_id, execution)) = accepted.provenance else {
            return Ok(None);
        };
        let observation = self
            .engine
            .semantic
            .stores
            .observations
            .get(observation_id)?;
        let task = self.engine.semantic.stores.tasks.get(execution)?;
        let base_condition = task
            .key
            .cpu_condition
            .clone()
            .ok_or(StoreError::InvalidEvidence(
                "adaptive conditional CPU condition missing",
            ))?;
        let descriptor = self.engine.owned_descriptor()?;
        let profile = stable_id(descriptor.config.profile.identity());
        let condition = stable_id(&format!(
            "{};q={}",
            descriptor.search_identity, descriptor.config.quiescence_ply
        ));
        let scoped = matches!(observation.scope,
            EvidenceScope::DepthLimited { depth, profile, condition }
            if depth == accepted.depth && profile == task.key.profile && condition == task.key.condition);
        let scored = matches!(observation.score,
            RawScore::Cpu { value, perspective, bound: BoundKind::ExactWithinSearch }
            if value == accepted.score && perspective == endpoint.position.side_to_move());
        if accepted.scope != CpuScoreScope::CompletedIteration
            || observation.state != endpoint.state
            || observation.line.is_some()
            || observation.kind != ObservationKind::CpuAnalysis
            || observation.execution != Some(execution)
            || observation.value_identity.as_ref() != Some(&accepted.value_identity)
            || observation.checker_identity.is_some()
            || observation.model_value_identity.is_some()
            || observation.cpu_condition.is_none()
            || observation.cpu_condition != task.key.cpu_condition
            || !self.conditional_cpu_task_condition_matches(
                accepted,
                &task.key,
                execution,
                observation_id,
            )
            || task.key.state != endpoint.state
            || task.key.line.is_some()
            || task.key.question != TaskQuestion::AnalyzePosition
            || !task.key.root_moves.is_empty()
            || task.key.input_revision != 0
            || task.key.profile != profile
            || task.key.condition != condition
            || task.key.requested_depth != self.limits.cpu_depth
            || task.key.value_identity.as_ref() != Some(&accepted.value_identity)
            || task.key.checker_identity.is_some()
            || task.key.node_budget == 0
            || observation.epoch != task.key.epoch
            || observation.budget > task.key.node_budget
            || !scoped
            || !scored
        {
            return Err(StoreError::InvalidEvidence(
                "adaptive conditional accepted CPU provenance",
            )
            .into());
        }
        match task.status {
            TaskStatus::Completed(id) if id == observation_id => {}
            TaskStatus::Paused {
                evidence: Some(id), ..
            }
            | TaskStatus::RetiredPaused {
                evidence: Some(id), ..
            } if id == observation_id => {
                // A completed iteration from a still-retained or resumed
                // stack is valid progress, but not this question's completion.
                return Ok(None);
            }
            _ => {
                return Err(StoreError::InvalidEvidence(
                    "adaptive conditional CPU completion is not accepted",
                )
                .into());
            }
        }
        Ok(Some(ConditionalEndpoint {
            root_value,
            basis: EndpointBasis::OwnCpu {
                depth: accepted.depth,
                plies,
                identity: accepted.value_identity.clone(),
                base_condition,
                profile: task.key.profile,
                condition: task.key.condition,
                observation: observation_id,
                execution,
            },
        }))
    }

    fn pin_conditional_endpoint(
        &mut self,
        endpoint: &ConditionalEndpoint,
    ) -> Result<(), PalsError> {
        if let EndpointBasis::OwnCpu {
            observation,
            execution,
            ..
        } = &endpoint.basis
        {
            // Retain both immutable raw evidence and its checked task lineage.
            if self
                .engine
                .semantic
                .stores
                .observations
                .get(*observation)?
                .execution
                != Some(*execution)
            {
                return Err(
                    StoreError::InvalidEvidence("conditional CPU evidence task changed").into(),
                );
            }
            self.engine.pin_archive_active_observation(*observation)?;
        }
        Ok(())
    }

    /// Same restricted-summary meaning as conclusion_observation: even Rules
    /// endpoints yield a conditional Estimate here, never a whole-game proof.
    fn conditional_observation(
        &mut self,
        line: LineId,
        kind: ObservationKind,
        value: i32,
        supersedes: Option<ObservationId>,
    ) -> Result<ObservationId, PalsError> {
        self.check_live()?;
        self.engine
            .semantic
            .stores
            .hot_append_observation(Observation {
                state: self.engine.semantic.nodes[self.root].state,
                line: Some(line),
                source: stable_id(self.engine.search_identity()),
                epoch: 0,
                scope: EvidenceScope::Model {
                    model: stable_id(self.engine.semantic.model.identity()),
                    encoding: stable_id("pals-restricted-summary-v1"),
                    input: self.engine.semantic.revision,
                },
                value_identity: None,
                checker_identity: None,
                checker_work: None,
                external_report: None,
                model_value_identity: None,
                model_value_input: None,
                cpu_condition: None,
                cpu_pv: None,
                score: RawScore::Estimate {
                    value: value as f32,
                    perspective: self.engine.semantic.nodes[self.root]
                        .position
                        .side_to_move(),
                },
                budget: 0,
                kind,
                supersedes,
                execution: None,
            })
            .map_err(PalsError::from)
    }
}
