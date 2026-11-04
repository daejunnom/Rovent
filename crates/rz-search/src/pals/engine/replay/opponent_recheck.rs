//! Opt-in 4N replay of an opponent's actual post-Repair model continuation.
//!
//! This search-local lane reuses Rules, role acceptance, CPU admission and the
//! one-shot closure. It never splices the Repair suffix after a new response.
//! Actual stage observations remain distinct from native/whole-cost/utility
//! authority, and the old 2N/3N consumers do not select this lane implicitly.

use super::*;

pub const FRESH_REPLAY_OPPONENT_SCOPE: &str =
    "rz-pals-frozen-parent-repair-opponent-continuation-replay/1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RepairOpponentReplayRequirements {
    cpu_nodes: u64,
    role_calls: u64,
    nodes: usize,
    observations: usize,
    line_chunks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pals::store::TaskStatus;

    // Source-level deterministic RoleModel only: no native requests, provider
    // work, physical completion, external supervisor or utility authority.
    #[derive(Clone, Copy, Default)]
    enum Fault {
        #[default]
        None,
        Failure,
        Invalid,
        CancelReturned,
        CancelAccepted,
        PhysicalUnknown,
    }
    struct Roles {
        proposal: Vec<BoardMove>,
        counter: Vec<BoardMove>,
        repaired: Vec<BoardMove>,
        opponent: Vec<BoardMove>,
        fault: Fault,
        fault_ply: usize,
        contexts: Vec<RoleLogicalContext>,
        accepted_contexts: Vec<RoleLogicalContext>,
        finish_calls: usize,
        closure: Option<RoleSearchClosure>,
    }
    impl Roles {
        fn evaluate(
            query: &RoleQuery<'_>,
            movement: BoardMove,
        ) -> Result<RoleEvaluation, RoleError> {
            let index = query
                .legal
                .iter()
                .position(|candidate| *candidate == movement)
                .ok_or(RoleError::InvalidOutput)?;
            let mut logits = vec![0.0; query.legal.len()];
            logits[index] = 1.0;
            Ok(RoleEvaluation {
                logits,
                wdl: [0.25, 0.5, 0.25],
            })
        }
        fn is_opponent(&self, query: &RoleQuery<'_>) -> bool {
            query
                .records
                .iter()
                .any(|record| record.kind == RecordKind::Repair)
        }
    }
    impl RoleModel for Roles {
        fn identity(&self) -> &str {
            "synthetic-actual-opponent-continuation/1"
        }
        fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            Self::evaluate(&query, self.proposal[query.prefix.len()])
        }
        fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            if !self.is_opponent(&query) {
                return Self::evaluate(&query, self.counter[query.prefix.len()]);
            }
            let record = query
                .records
                .iter()
                .find(|record| record.kind == RecordKind::Repair)
                .ok_or(RoleError::InvalidOutput)?;
            if query.proposal != self.repaired
                || query.counterexample != Some(self.counter.as_slice())
                || record.line != self.repaired
                || record.cpu_observation.is_some()
                || record.value.is_some()
                || query.prefix != &self.opponent[..query.prefix.len()]
                || query.revision != record.revision
            {
                return Err(RoleError::InvalidOutput);
            }
            if query.prefix.len() == self.fault_ply {
                match self.fault {
                    Fault::Failure => {
                        return Err(RoleError::Backend("new C continuation failure".into()));
                    }
                    Fault::Invalid => {
                        return Ok(RoleEvaluation {
                            logits: vec![],
                            wdl: [0.25, 0.5, 0.25],
                        });
                    }
                    Fault::CancelReturned => query.cancel.store(true, Ordering::Release),
                    Fault::PhysicalUnknown => return Err(RoleError::PhysicalCompletionUnknown),
                    Fault::None | Fault::CancelAccepted => (),
                }
            }
            Self::evaluate(&query, self.opponent[query.prefix.len()])
        }
        fn reply_with_context(
            &mut self,
            query: RoleQuery<'_>,
            context: &RoleLogicalContext,
        ) -> Result<RoleEvaluation, RoleError> {
            if self.is_opponent(&query) {
                self.contexts.push(context.clone());
            }
            self.reply(query)
        }
        fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            if query.counterexample != Some(self.counter.as_slice()) {
                return Err(RoleError::InvalidOutput);
            }
            Self::evaluate(&query, self.repaired[query.prefix.len()])
        }
        fn divergences(&mut self, _: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
            Err(RoleError::Unavailable)
        }
        fn accepted_output_checked(
            &mut self,
            acceptance: RoleAcceptance<'_>,
        ) -> Result<(), RoleError> {
            acceptance.check_control()?;
            if self.contexts.last() == Some(acceptance.context) {
                self.accepted_contexts.push(acceptance.context.clone());
                if matches!(self.fault, Fault::CancelAccepted)
                    && acceptance.context.prefix.len() == self.fault_ply
                {
                    acceptance.cancel.store(true, Ordering::Release);
                }
            }
            Ok(())
        }
        fn finish_search(&mut self, reason: RoleSearchClosure) {
            self.finish_calls += 1;
            self.closure = Some(reason);
        }
    }
    fn mv(text: &str) -> BoardMove {
        BoardMove::from_uci(text).unwrap()
    }
    fn line(moves: &[&str]) -> Vec<BoardMove> {
        moves.iter().map(|text| mv(text)).collect()
    }
    fn fixture() -> FreshReplayOwner<Roles> {
        fixture_with_quiescence(16)
    }
    // Constructor-selected finite synthetic test condition, identical across
    // all four stages. No partial result triggers an automatic setting change.
    fn fixture_with_quiescence(quiescence_ply: u16) -> FreshReplayOwner<Roles> {
        let root = Position::startpos();
        let mut target = root.clone();
        target.make_move(mv("e2e4")).unwrap();
        let proposal = line(&["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]);
        let roles = Roles {
            proposal: proposal.clone(),
            counter: line(&["e2e4", "e7e6", "d2d4", "d7d5", "e4e5"]),
            repaired: line(&["e2e4", "e7e6", "d2d3", "d7d5", "g1f3"]),
            opponent: line(&["e2e4", "e7e6", "d2d3", "c7c5", "b1c3"]),
            fault: Fault::None,
            fault_ply: 3,
            contexts: vec![],
            accepted_contexts: vec![],
            finish_calls: 0,
            closure: None,
        };
        FreshReplayOwner::new(
            PalsConfig {
                line_plies: 5,
                max_records: 5,
                cpu_nodes_per_task: 100_000,
                ..PalsConfig::default()
            },
            roles,
            root.clone(),
            DefendResponseReplayPlan {
                historical: ReplayHistoricalPins {
                    parent_input_sha256: [1; 32],
                    current_view_sha256: [2; 32],
                    frozen_admission_sha256: [3; 32],
                    query_sha256: [4; 32],
                    catalogue_sha256: [5; 32],
                    before_result_sha256: [6; 32],
                    semantic_input_sha256: [7; 32],
                    cpu_request_sha256: [8; 32],
                },
                expected_root: root.snapshot(),
                expected_target: target.snapshot(),
                root_legal_order: root.legal_moves(),
                target_legal_order: target.legal_moves(),
                prefix: vec![mv("e2e4")],
                response_restriction: vec![mv("c7c5")],
                claimed_line: proposal,
                baseline_depth: 1,
                requested_depth: 2,
                nodes_per_check: 100_000,
                cpu: CpuConfig {
                    profile: CpuProfile::PlanAssisted,
                    tt_entries: 16,
                    max_depth: 2,
                    quiescence_ply,
                },
            },
        )
        .unwrap()
    }
    fn limits() -> PalsLimits {
        PalsLimits {
            deadline: Instant::now() + Duration::from_secs(10),
            max_rounds: 1,
            max_cpu_nodes: 400_000,
            cpu_depth: 2,
        }
    }
    fn is_role_error(
        result: Result<ReplayOpponentOutcome, ReplayError>,
        expected: RoleError,
    ) -> bool {
        matches!(result, Err(ReplayError::Search(error)) if matches!(*error, PalsError::Role(ref error) if *error == expected))
    }

    #[test]
    fn repair_origin_borrows_actual_record_and_recomputes_anchor_without_new_work() {
        let mut owner = fixture();
        let limits = limits();
        assert!(
            owner
                .opponent_repair_origin(limits.deadline, &AtomicBool::new(false))
                .unwrap()
                .is_none()
        );
        owner
            .run_with_opponent_recheck(limits, &AtomicBool::new(false))
            .unwrap();
        let work = (
            owner.counters().cpu_nodes,
            owner.counters().cpu_tasks_requested,
            owner.engine.semantic.model.contexts.len(),
            owner.engine.semantic.model.finish_calls,
        );
        let origin = owner
            .opponent_repair_origin(limits.deadline, &AtomicBool::new(false))
            .unwrap()
            .unwrap();
        assert_eq!(origin.anchor_ply(), 3);
        assert_eq!(origin.model_counterline(), owner.model_counterline());
        assert_eq!(origin.repaired_line(), owner.repaired_line());
        assert_eq!(
            Some(origin.repair_record_revision()),
            owner.repair_record_revision
        );
        assert_eq!(
            work,
            (
                owner.counters().cpu_nodes,
                owner.counters().cpu_tasks_requested,
                owner.engine.semantic.model.contexts.len(),
                owner.engine.semantic.model.finish_calls
            )
        );
    }

    #[test]
    fn repair_origin_refuses_record_revision_anchor_or_original_line_substitution() {
        for case in 0..4 {
            let mut owner = fixture();
            let limits = limits();
            owner
                .run_with_opponent_recheck(limits, &AtomicBool::new(false))
                .unwrap();
            match case {
                0 => owner
                    .engine
                    .semantic
                    .records
                    .retain(|record| record.kind != RecordKind::Repair),
                1 => owner.repair_record_revision = owner.repair_record_revision.map(|r| r + 1),
                2 => owner.opponent_recheck.as_mut().unwrap().anchor_ply = Some(4),
                _ => owner.model_counterline[2] = owner.repaired_line[2],
            }
            assert!(
                owner
                    .opponent_repair_origin(limits.deadline, &AtomicBool::new(false))
                    .is_err(),
                "case {case}"
            );
        }
    }

    #[test]
    fn repair_origin_result_verification_obeys_cancellation_and_its_original_window() {
        let owner = fixture();
        assert!(
            owner
                .opponent_repair_origin(limits().deadline, &AtomicBool::new(true))
                .is_err()
        );
        assert!(
            owner
                .opponent_repair_origin(Instant::now(), &AtomicBool::new(false))
                .is_err()
        );
    }

    #[test]
    fn actual_c_generates_response_and_remaining_tail_not_repair_suffix() {
        let mut owner = fixture();
        let outcome = owner
            .run_with_opponent_recheck(limits(), &AtomicBool::new(false))
            .unwrap();
        let ReplayOpponentOutcome::CompletedOpponentEndpoint {
            counterline,
            counter_record_revision,
            endpoint_state,
            execution,
            observation,
            repair_record_revision,
            ..
        } = outcome
        else {
            panic!("{outcome:?}");
        };
        assert_eq!(owner.opponent_anchor_ply(), Some(3));
        assert_eq!(
            owner.opponent_counterline(),
            owner.engine.semantic.model.opponent
        );
        assert_eq!(
            &owner.opponent_counterline()[..3],
            &owner.repaired_line()[..3]
        );
        assert_ne!(owner.opponent_counterline()[3], owner.repaired_line()[3]);
        assert_ne!(owner.opponent_counterline()[4], owner.repaired_line()[4]);
        assert_eq!(owner.opponent_role_steps().len(), 2);
        assert_eq!(
            owner.engine.semantic.model.contexts,
            owner.engine.semantic.model.accepted_contexts
        );
        for (index, step) in owner.opponent_role_steps().iter().enumerate() {
            assert_eq!(step.context(), &owner.engine.semantic.model.contexts[index]);
            assert_eq!(
                step.context().prefix,
                owner.opponent_counterline()[..3 + index]
            );
            assert_eq!(step.context().public_revision, repair_record_revision);
            assert_eq!(step.context().purpose, RoleQueryPurpose::ReplyPolicy);
            assert!(step.accepted());
            assert_eq!(
                step.selected(),
                Some(owner.opponent_counterline()[3 + index])
            );
        }
        let mut checked = owner.root.clone();
        for &movement in owner.opponent_counterline() {
            checked.make_move(movement).unwrap();
        }
        let endpoint = owner.opponent_endpoint().unwrap();
        assert!(endpoint.snapshot.same_state(&checked.snapshot()));
        assert_eq!(endpoint.state, endpoint_state);
        assert!(endpoint.terminal.is_none());
        assert_eq!(endpoint.stage.unwrap().execution, execution);
        assert_eq!(
            owner
                .engine
                .semantic
                .stores
                .lines
                .moves(counterline)
                .unwrap()
                .as_slice(),
            owner.opponent_counterline()
        );
        let record = owner
            .records()
            .iter()
            .find(|record| record.revision == counter_record_revision)
            .unwrap();
        assert_eq!(record.kind, RecordKind::Counterexample);
        assert_eq!(record.line, owner.opponent_counterline());
        assert_eq!(record.cpu_observation, None);
        assert_eq!(record.value, None);
        assert_eq!(record.completed_depth, 0);
        assert_eq!(record.score_scope, None);
        assert_eq!(
            owner.task(execution).unwrap().status,
            TaskStatus::Completed(observation)
        );
        assert_eq!(owner.engine.semantic.model.finish_calls, 1);
        assert_eq!(
            owner.engine.semantic.model.closure,
            Some(RoleSearchClosure::Completed)
        );
    }

    #[test]
    fn fourth_is_fresh_exact_h1_start_with_own_scope_and_unrestricted_legal_moves() {
        let mut owner = fixture();
        owner
            .run_with_opponent_recheck(limits(), &AtomicBool::new(false))
            .unwrap();
        assert_eq!(owner.stages.len(), 4);
        let fourth = &owner.stages[3];
        assert_eq!(fourth.phase, ReplayCpuPhase::RepairOpponentEndpoint);
        assert!(fourth.exact_completed && fourth.attempt.is_some());
        assert_eq!(fourth.requested_depth, 2);
        assert_eq!(fourth.node_budget, 100_000);
        let report = fourth.report.as_ref().unwrap();
        assert_eq!(report.reused_completed_depth, 0);
        assert_eq!(report.completed_depth, 2);
        assert_eq!(report.score_scope, CpuScoreScope::CompletedIteration);
        let task = owner.task(fourth.execution).unwrap();
        assert_eq!(task.key.question, TaskQuestion::AnalyzePosition);
        assert!(task.key.root_moves.is_empty());
        assert_eq!(task.key.input_revision, 0);
        assert_eq!(task.resumed_from, None);
        assert!(fourth.task_condition.contains(FRESH_REPLAY_OPPONENT_SCOPE));
        assert!(
            fourth
                .task_condition
                .contains("phase=repair-opponent-endpoint")
        );
        for prior in &owner.stages[..3] {
            assert_ne!(fourth.execution, prior.execution);
            assert_ne!(fourth.observation, prior.observation);
        }
        assert_eq!(
            fourth.registered_condition,
            owner.stages[2].registered_condition
        );
        assert_eq!(owner.counters.cpu_tasks_requested, 4);
        assert_eq!(owner.counters.consumed_cpu_tasks, 4);
        assert_eq!(
            owner.counters.role_calls,
            repair_opponent_replay_requirements(5, 1, 100_000)
                .unwrap()
                .role_calls()
        );
        assert_eq!(owner.counters.supported_repairs, 0);
        assert_eq!(owner.counters.supported_refutations, 0);
        assert!(!owner.utility_authority() && !owner.training_target_created());
        assert!(!owner.native_freshness_verified() && !owner.strict_query_admission_verified());
        assert!(!owner.physical_closure_verified() && !owner.whole_cost_verified());
    }

    #[test]
    fn third_endpoint_evidence_remains_third_after_fourth_execution() {
        let mut owner = fixture();
        owner
            .run_with_opponent_recheck(limits(), &AtomicBool::new(false))
            .unwrap();
        let third = &owner.stages[2];
        assert!(
            matches!(owner.repair_endpoint().unwrap().evidence, RecheckEndpointEvidence::OwnCpu {
            observation_id, execution_id, ..
        } if Some(observation_id) == third.observation && execution_id == third.execution)
        );
        assert_eq!(
            owner.opponent_endpoint().unwrap().stage.unwrap().phase,
            ReplayCpuPhase::RepairOpponentEndpoint
        );
    }

    #[test]
    fn original_three_n_mode_never_implicitly_selects_new_opponent_lane() {
        let mut owner = fixture();
        let result = owner
            .run_with_repair(limits(), &AtomicBool::new(false))
            .unwrap();
        assert!(matches!(
            result,
            ReplayRepairOutcome::CompletedRepairEndpoint { .. }
        ));
        assert_eq!(owner.stages.len(), 3);
        assert!(owner.opponent_recheck.is_none());
        assert!(owner.opponent_role_steps().is_empty() && owner.opponent_counterline().is_empty());
        assert!(owner.engine.semantic.model.contexts.is_empty());
        assert!(matches!(
            owner.run_with_opponent_recheck(limits(), &AtomicBool::new(false)),
            Err(ReplayError::AlreadyUsed)
        ));
        assert_eq!(owner.engine.semantic.model.finish_calls, 1);
    }

    #[test]
    fn unchanged_own_repair_has_no_eligible_opponent_anchor_or_fourth_cpu() {
        let mut owner = fixture();
        owner.engine.semantic.model.repaired = owner.engine.semantic.model.counter.clone();
        let result = owner
            .run_with_opponent_recheck(limits(), &AtomicBool::new(false))
            .unwrap();
        assert_eq!(result, ReplayOpponentOutcome::NoEligibleOpponentAnchor);
        assert_eq!(owner.stages.len(), 3);
        assert!(owner.opponent_anchor_ply().is_none() && owner.opponent_role_steps().is_empty());
        assert!(owner.opponent_endpoint().is_none());
        assert_eq!(owner.engine.semantic.model.finish_calls, 1);
    }

    #[test]
    fn incomplete_initial_cpu_never_enters_repair_or_opponent() {
        let mut owner = fixture();
        owner.plan.nodes_per_check = 1;
        owner.engine.config.cpu_nodes_per_task = 1;
        let mut bound = limits();
        bound.max_cpu_nodes = 4;
        let result = owner
            .run_with_opponent_recheck(bound, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(
            result,
            ReplayOpponentOutcome::RepairNotReady(ReplayRepairOutcome::ReplyNotApplied(
                ReplayOutcome::Partial(ReplayCpuPhase::Baseline)
            ))
        );
        assert_eq!(owner.stages.len(), 1);
        assert!(owner.opponent_role_steps().is_empty());
    }

    #[test]
    fn original_short_quiescence_repair_stays_partial_and_never_enters_new_c() {
        let mut owner = fixture_with_quiescence(4);
        let outcome = owner
            .run_with_opponent_recheck(limits(), &AtomicBool::new(false))
            .unwrap();
        let ReplayOpponentOutcome::RepairNotReady(ReplayRepairOutcome::PartialRepairEndpoint {
            execution,
            observation,
        }) = outcome
        else {
            panic!("{outcome:?}; stages={:?}", owner.stages);
        };
        assert_eq!(owner.stages.len(), 3);
        let third = &owner.stages[2];
        assert_eq!(third.execution, execution);
        assert_eq!(third.observation, observation);
        assert_eq!(
            third.report.as_ref().unwrap().completion,
            CpuCompletion::QuiescenceLimit
        );
        assert!(!third.exact_completed);
        assert_eq!(owner.task(execution).unwrap().status, TaskStatus::Failed);
        assert_eq!(owner.counters.cpu_tasks_requested, 3);
        assert!(
            owner.opponent_role_steps().is_empty()
                && owner.engine.semantic.model.contexts.is_empty()
        );
        assert!(owner.opponent_endpoint().is_none());
        assert_eq!(owner.plan.cpu.quiescence_ply, 4);
        assert_eq!(owner.engine.semantic.model.finish_calls, 1);
    }

    #[test]
    fn four_n_role_node_and_record_shortages_refuse_before_any_work() {
        let required = repair_opponent_replay_requirements(5, 1, 100_000).unwrap();
        for shortage in 0..4 {
            let mut owner = fixture();
            let mut bound = limits();
            match shortage {
                0 => bound.max_cpu_nodes = required.cpu_nodes() - 1,
                1 => owner.engine.config.max_role_calls = required.role_calls() - 1,
                2 => owner.engine.config.max_nodes = required.nodes() - 1,
                _ => owner.engine.config.max_records = required.records() - 1,
            }
            assert!(matches!(
                owner.run_with_opponent_recheck(bound, &AtomicBool::new(false)),
                Err(ReplayError::InvalidPlan(_))
            ));
            assert!(owner.stages.is_empty() && owner.opponent_recheck.is_none());
            assert_eq!(owner.counters.role_calls, 0);
            assert_eq!(owner.counters.cpu_tasks_requested, 0);
            assert_eq!(owner.engine.semantic.model.finish_calls, 1);
        }
    }

    #[test]
    fn canceled_or_expired_original_window_refuses_before_dispatch() {
        let mut owner = fixture();
        assert!(is_role_error(
            owner.run_with_opponent_recheck(limits(), &AtomicBool::new(true)),
            RoleError::Canceled
        ));
        assert!(owner.stages.is_empty());
        assert_eq!(
            owner.engine.semantic.model.closure,
            Some(RoleSearchClosure::Canceled)
        );
        let mut owner = fixture();
        let mut bound = limits();
        bound.deadline = Instant::now();
        assert!(is_role_error(
            owner.run_with_opponent_recheck(bound, &AtomicBool::new(false)),
            RoleError::Deadline
        ));
        assert!(owner.stages.is_empty());
        assert_eq!(
            owner.engine.semantic.model.closure,
            Some(RoleSearchClosure::Deadline)
        );
    }

    #[test]
    fn new_c_failure_invalid_and_physical_unknown_keep_completed_three_stages() {
        for (fault, error, closure) in [
            (
                Fault::Failure,
                RoleError::Backend("new C continuation failure".into()),
                RoleSearchClosure::Failed,
            ),
            (
                Fault::Invalid,
                RoleError::InvalidOutput,
                RoleSearchClosure::Failed,
            ),
            (
                Fault::PhysicalUnknown,
                RoleError::PhysicalCompletionUnknown,
                RoleSearchClosure::PhysicalCompletionUnknown,
            ),
        ] {
            let mut owner = fixture();
            owner.engine.semantic.model.fault = fault;
            assert!(is_role_error(
                owner.run_with_opponent_recheck(limits(), &AtomicBool::new(false)),
                error
            ));
            assert_eq!(owner.stages.len(), 3);
            assert!(owner.stages.iter().all(|stage| stage.exact_completed));
            assert_eq!(owner.opponent_counterline(), &owner.repaired_line()[..3]);
            assert_eq!(owner.opponent_role_steps().len(), 1);
            assert!(!owner.opponent_role_steps()[0].accepted());
            assert_eq!(owner.opponent_role_steps()[0].selected(), None);
            assert!(owner.opponent_endpoint().is_none());
            assert_eq!(owner.records().len(), 4);
            assert_eq!(owner.engine.semantic.model.finish_calls, 1);
            assert_eq!(owner.engine.semantic.model.closure, Some(closure));
        }
    }

    #[test]
    fn cancellation_on_new_c_return_or_acceptance_never_dispatches_fourth_cpu() {
        for (fault, accepted) in [
            (Fault::CancelReturned, false),
            (Fault::CancelAccepted, true),
        ] {
            let mut owner = fixture();
            owner.engine.semantic.model.fault = fault;
            let cancel = AtomicBool::new(false);
            assert!(is_role_error(
                owner.run_with_opponent_recheck(limits(), &cancel),
                RoleError::Canceled
            ));
            assert_eq!(owner.stages.len(), 3);
            assert_eq!(owner.counters.cpu_tasks_requested, 3);
            let step = &owner.opponent_role_steps()[0];
            assert_eq!(step.accepted(), accepted);
            assert_eq!(step.selected(), None);
            assert_eq!(
                owner.engine.semantic.model.accepted_contexts.len(),
                usize::from(accepted)
            );
            assert_eq!(
                owner.engine.semantic.model.closure,
                Some(RoleSearchClosure::Canceled)
            );
            assert_eq!(owner.engine.semantic.model.finish_calls, 1);
        }
    }

    #[test]
    fn later_c_failure_preserves_actual_partial_new_continuation() {
        let mut owner = fixture();
        owner.engine.semantic.model.fault = Fault::Failure;
        owner.engine.semantic.model.fault_ply = 4;
        assert!(is_role_error(
            owner.run_with_opponent_recheck(limits(), &AtomicBool::new(false)),
            RoleError::Backend("new C continuation failure".into())
        ));
        assert_eq!(
            owner.opponent_counterline(),
            &owner.engine.semantic.model.opponent[..4]
        );
        assert_eq!(owner.opponent_role_steps().len(), 2);
        assert!(owner.opponent_role_steps()[0].accepted());
        assert_eq!(owner.opponent_role_steps()[0].selected(), Some(mv("c7c5")));
        assert!(!owner.opponent_role_steps()[1].accepted());
        assert_eq!(owner.opponent_role_steps()[1].selected(), None);
        assert_eq!(owner.stages.len(), 3);
        assert!(owner.opponent_endpoint().is_none());
    }

    #[test]
    fn requirements_are_finite_checked_and_charge_all_remaining_model_plies() {
        let base = repair_replay_requirements(5, 1, 100_000).unwrap();
        let required = repair_opponent_replay_requirements(5, 1, 100_000).unwrap();
        assert_eq!(required.cpu_nodes(), 400_000);
        assert_eq!(required.role_calls(), base.role_calls() + 2);
        assert_eq!(required.nodes(), base.nodes() + 2);
        assert_eq!(required.observations(), base.observations() + 3);
        assert_eq!(required.line_chunks(), base.line_chunks() + 2 * 6 + 5);
        assert_eq!(required.records(), 5);
        assert_eq!(required.stages(), 4);
        for (extent, prefix, nodes) in [
            (3, 1, 1),
            (5, 5, 1),
            (0, 0, 1),
            (5, 1, 0),
            (5, 1, u64::MAX / 4 + 1),
            (usize::MAX, 1, 1),
        ] {
            assert!(matches!(
                repair_opponent_replay_requirements(extent, prefix, nodes),
                Err(ReplayError::InvalidPlan(_))
            ));
        }
        assert!(repair_opponent_replay_requirements(4, 1, 1).is_ok());
    }
}
impl RepairOpponentReplayRequirements {
    pub const fn cpu_nodes(self) -> u64 {
        self.cpu_nodes
    }
    pub const fn role_calls(self) -> u64 {
        self.role_calls
    }
    pub const fn nodes(self) -> usize {
        self.nodes
    }
    pub const fn observations(self) -> usize {
        self.observations
    }
    pub const fn line_chunks(self) -> usize {
        self.line_chunks
    }
    pub const fn records(self) -> usize {
        5
    }
    pub const fn stages(self) -> usize {
        4
    }
}

/// Finite declarations, not measured peaks or execution support. An eligible
/// opponent anchor requires a changed own move and a later opponent move after
/// the original attack. Every remaining model continuation ply is charged.
pub fn repair_opponent_replay_requirements(
    line_plies: usize,
    prefix_plies: usize,
    nodes_per_check: u64,
) -> Result<RepairOpponentReplayRequirements, ReplayError> {
    let base = repair_replay_requirements(line_plies, prefix_plies, nodes_per_check)?;
    let extra = line_plies
        .checked_sub(prefix_plies)
        .and_then(|n| n.checked_sub(2))
        .filter(|n| *n > 0)
        .ok_or(ReplayError::InvalidPlan(
            "opponent continuation needs L-P >= 3",
        ))?;
    let cpu_nodes = nodes_per_check
        .checked_mul(4)
        .ok_or(ReplayError::InvalidPlan("4N overflow"))?;
    let role_calls = base
        .role_calls()
        .checked_add(
            u64::try_from(extra)
                .map_err(|_| ReplayError::InvalidPlan("opponent role extent conversion"))?,
        )
        .ok_or(ReplayError::InvalidPlan("opponent role bound overflow"))?;
    let nodes = base
        .nodes()
        .checked_add(extra)
        .ok_or(ReplayError::InvalidPlan("opponent node bound overflow"))?;
    let observations = base
        .observations()
        .checked_add(extra)
        .and_then(|n| n.checked_add(1))
        .ok_or(ReplayError::InvalidPlan(
            "opponent observation bound overflow",
        ))?;
    let line_chunks = extra
        .checked_mul(
            line_plies
                .checked_add(1)
                .ok_or(ReplayError::InvalidPlan("opponent line extent overflow"))?,
        )
        .and_then(|n| n.checked_add(line_plies))
        .and_then(|n| n.checked_add(base.line_chunks()))
        .ok_or(ReplayError::InvalidPlan(
            "opponent line chunk bound overflow",
        ))?;
    Ok(RepairOpponentReplayRequirements {
        cpu_nodes,
        role_calls,
        nodes,
        observations,
        line_chunks,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayOpponentOutcome {
    RepairNotReady(ReplayRepairOutcome),
    NoEligibleOpponentAnchor,
    NoAlternativeResponse,
    PartialOpponentEndpoint {
        execution: ExecutionId,
        observation: Option<ObservationId>,
    },
    CompletedOpponentEndpoint {
        repaired_line: LineId,
        repair_record_revision: u64,
        counterline: LineId,
        counter_record_revision: u64,
        endpoint_state: StateId,
        endpoint_situation: SituationId,
        execution: ExecutionId,
        observation: ObservationId,
    },
    RulesTerminalOpponentEndpoint {
        counterline: LineId,
        counter_record_revision: u64,
        endpoint_state: StateId,
        endpoint_situation: SituationId,
        reason: TerminalReason,
    },
}

/// Search-local accepted context and selected legal move. This does not attest
/// a native RequestId, provider execution, physical completion or model epoch.
#[derive(Debug)]
pub struct ReplayOpponentRoleStep {
    context: RoleLogicalContext,
    accepted: bool,
    selected: Option<BoardMove>,
}
impl ReplayOpponentRoleStep {
    pub fn context(&self) -> &RoleLogicalContext {
        &self.context
    }
    pub fn accepted(&self) -> bool {
        self.accepted
    }
    pub fn selected(&self) -> Option<BoardMove> {
        self.selected
    }
}

pub(super) struct OpponentRecheckState {
    counterline: Vec<BoardMove>,
    steps: Vec<ReplayOpponentRoleStep>,
    anchor_ply: Option<usize>,
    endpoint_node: Option<usize>,
    endpoint_snapshot: Option<PositionSnapshot>,
    line_id: Option<LineId>,
    record_revision: Option<u64>,
}

/// Borrowed actual endpoint and its own fourth-stage ledger, including partial
/// or failed observations. A terminal endpoint has no fabricated CPU stage.
pub struct ReplayOpponentEndpoint<'a> {
    pub state: StateId,
    pub situation: SituationId,
    pub snapshot: &'a PositionSnapshot,
    pub terminal: Option<TerminalReason>,
    pub stage: Option<&'a ReplayStageReport>,
}

/// Borrowed actual accepted Repair record/path and recomputed anchor. Local
/// revision is provenance, never a portable record handle or strategic feature.
/// Only the live owner can construct it; no Clone/Deserialize/owned constructor.
pub struct ReplayOpponentRepairOrigin<'a> {
    model_counterline: &'a [BoardMove],
    repaired_line: &'a [BoardMove],
    repair_record_revision: u64,
    anchor_ply: usize,
}
impl ReplayOpponentRepairOrigin<'_> {
    pub fn model_counterline(&self) -> &[BoardMove] {
        self.model_counterline
    }
    pub fn repaired_line(&self) -> &[BoardMove] {
        self.repaired_line
    }
    pub fn repair_record_revision(&self) -> u64 {
        self.repair_record_revision
    }
    pub fn anchor_ply(&self) -> usize {
        self.anchor_ply
    }
}

impl<M: RoleModel> FreshReplayOwner<M> {
    /// Readonly result verification under the supplied original W. It reuses the
    /// actual work's anchor predicate and store checks, without CPU/model work,
    /// role counters, reservations, new records or an extension of work's E.
    pub fn opponent_repair_origin(
        &self,
        verification_deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<Option<ReplayOpponentRepairOrigin<'_>>, ReplayError> {
        let control = || {
            check_role_control(verification_deadline, cancel)
                .map_err(|error| PalsError::from(error).into())
        };
        control()?;
        let Some(anchor) = self.opponent_anchor_ply() else {
            return Ok(None);
        };
        let (Some(line), Some(revision)) = (self.repair_line_id, self.repair_record_revision)
        else {
            return Err(
                StoreError::InvalidEvidence("opponent anchor lacks actual Repair record").into(),
            );
        };
        let recomputed = self
            .opponent_anchor_with_control(line, revision, &control)?
            .ok_or(StoreError::InvalidEvidence(
                "record has no eligible opponent anchor",
            ))?;
        if recomputed.2 != anchor {
            return Err(StoreError::InvalidEvidence(
                "stored opponent anchor differs from actual Repair path",
            )
            .into());
        }
        control()?;
        Ok(Some(ReplayOpponentRepairOrigin {
            model_counterline: &self.model_counterline,
            repaired_line: &self.repaired_line,
            repair_record_revision: revision,
            anchor_ply: anchor,
        }))
    }
    pub fn opponent_counterline(&self) -> &[BoardMove] {
        self.opponent_recheck
            .as_ref()
            .map_or(&[], |state| state.counterline.as_slice())
    }
    pub fn opponent_role_steps(&self) -> &[ReplayOpponentRoleStep] {
        self.opponent_recheck
            .as_ref()
            .map_or(&[], |state| state.steps.as_slice())
    }
    pub fn opponent_anchor_ply(&self) -> Option<usize> {
        self.opponent_recheck
            .as_ref()
            .and_then(|state| state.anchor_ply)
    }
    pub fn opponent_endpoint(&self) -> Option<ReplayOpponentEndpoint<'_>> {
        let state = self.opponent_recheck.as_ref()?;
        let node = self.engine.semantic.nodes.get(state.endpoint_node?)?;
        Some(ReplayOpponentEndpoint {
            state: node.state,
            situation: node.situation,
            snapshot: state.endpoint_snapshot.as_ref()?,
            terminal: node.terminal.map(|(reason, _)| reason),
            stage: self
                .stages
                .get(3)
                .filter(|stage| stage.phase == ReplayCpuPhase::RepairOpponentEndpoint),
        })
    }
    pub(super) fn endpoint_snapshot(&self, kind: EndpointKind) -> Option<&PositionSnapshot> {
        match kind {
            EndpointKind::Repair => self.repair_endpoint_snapshot.as_ref(),
            EndpointKind::Opponent => self
                .opponent_recheck
                .as_ref()
                .and_then(|state| state.endpoint_snapshot.as_ref()),
        }
    }

    /// Explicit 4N selection. Preparation, the old three stages and the actual
    /// new opponent continuation share the original deadline and one finish hook.
    /// No supported-Repair conclusion, utility target or native authority is issued.
    pub fn run_with_opponent_recheck(
        &mut self,
        limits: PalsLimits,
        cancel: &AtomicBool,
    ) -> Result<ReplayOpponentOutcome, ReplayError> {
        self.run_once(limits, cancel, |owner, counters| {
            owner.prepare_opponent_recheck(limits, cancel)?;
            let reply = owner.run_inner(limits, cancel, counters)?;
            let repair = owner.repair_tail(reply, limits, cancel, counters)?;
            owner.opponent_recheck_tail(repair, limits, cancel, counters)
        })
    }

    fn prepare_opponent_recheck(
        &mut self,
        limits: PalsLimits,
        cancel: &AtomicBool,
    ) -> Result<(), ReplayError> {
        let extent = self.engine.config.line_plies;
        let required = repair_opponent_replay_requirements(
            extent,
            self.plan.prefix.len(),
            self.plan.nodes_per_check,
        )?;
        let reserved = PalsEngine::<M>::store_limits(&self.engine.config);
        if limits.max_rounds == 0
            || limits.cpu_depth != self.plan.requested_depth
            || limits.max_cpu_nodes < required.cpu_nodes()
            || self.engine.config.cpu_nodes_per_task != self.plan.nodes_per_check
            || self.engine.config.max_role_calls < required.role_calls()
            || self.engine.config.max_records < required.records()
            || self.engine.config.max_nodes < required.nodes()
            || reserved.observations < required.observations()
            || reserved.line_chunks < required.line_chunks()
            || reserved.executions < required.stages()
            || reserved.consumers < required.stages()
            || !self.stages.is_empty()
            || self.opponent_recheck.is_some()
        {
            return Err(ReplayError::InvalidPlan(
                "limits do not support fixed H1/4N and opponent continuation reservations",
            ));
        }
        self.check_control(limits, cancel)?;
        self.stages
            .try_reserve_exact(required.stages())
            .map_err(|_| PalsError::Capacity)?;
        let mut steps = Vec::new();
        steps
            .try_reserve_exact(extent - self.plan.prefix.len() - 2)
            .map_err(|_| PalsError::Capacity)?;
        self.opponent_recheck = Some(Box::new(OpponentRecheckState {
            counterline: bounded_moves(&[], extent)?,
            steps,
            anchor_ply: None,
            endpoint_node: None,
            endpoint_snapshot: None,
            line_id: None,
            record_revision: None,
        }));
        self.prepare_repair(limits, cancel)
    }

    fn opponent_anchor(
        &self,
        repaired_line: LineId,
        record_revision: u64,
        limits: PalsLimits,
        cancel: &AtomicBool,
    ) -> Result<Option<(usize, usize, usize)>, ReplayError> {
        self.opponent_anchor_with_control(repaired_line, record_revision, &|| {
            self.check_control(limits, cancel)
        })
    }

    fn opponent_anchor_with_control(
        &self,
        repaired_line: LineId,
        record_revision: u64,
        control: &dyn Fn() -> Result<(), ReplayError>,
    ) -> Result<Option<(usize, usize, usize)>, ReplayError> {
        let active_root = self
            .engine
            .semantic
            .stores
            .root()
            .ok_or(StoreError::StaleConsumer)?;
        let root = self
            .engine
            .semantic
            .nodes
            .iter()
            .position(|node| node.situation == active_root)
            .ok_or(StoreError::StaleConsumer)?;
        let attack = self.plan.prefix.len();
        if self.repaired_line.len() > self.engine.config.line_plies
            || attack >= self.repaired_line.len()
            || attack >= self.model_counterline.len()
            || self.repaired_line.first() != self.proposal.first()
            || self.repaired_line[..=attack] != self.model_counterline[..=attack]
            || self
                .engine
                .semantic
                .stores
                .lines
                .get(repaired_line)?
                .start_state
                != self.engine.semantic.nodes[root].state
            || self
                .engine
                .semantic
                .stores
                .lines
                .moves(repaired_line)?
                .as_slice()
                != self.repaired_line
            || !self.engine.semantic.nodes[root]
                .position
                .snapshot()
                .same_state(&self.root.snapshot())
            || !self.engine.semantic.records.iter().any(|record| {
                record.origin_state == self.engine.semantic.nodes[root].state
                    && record.kind == RecordKind::Repair
                    && record.revision == record_revision
                    && record.line == self.repaired_line
            })
        {
            return Err(StoreError::InvalidEvidence(
                "opponent recheck lacks the actual accepted Repair line/revision",
            )
            .into());
        }
        let color = self.root.side_to_move();
        let mut node = root;
        let mut changed_own = false;
        for (ply, movement) in self.repaired_line.iter().enumerate() {
            control()?;
            self.check_node_state(node)?;
            if changed_own
                && self.engine.semantic.nodes[node].position.side_to_move() != color
                && self.engine.semantic.nodes[node].terminal.is_none()
            {
                return Ok(Some((root, node, ply)));
            }
            if ply > attack
                && self.engine.semantic.nodes[node].position.side_to_move() == color
                && self.model_counterline.get(ply) != Some(movement)
            {
                changed_own = true;
            }
            node = self.engine.semantic.nodes[node]
                .edges
                .iter()
                .find(|edge| edge.movement == *movement)
                .ok_or(StoreError::InvalidEvidence(
                    "accepted Repair path disappeared",
                ))?
                .child;
        }
        Ok(None)
    }

    fn check_node_state(&self, node: usize) -> Result<(), ReplayError> {
        let node = self
            .engine
            .semantic
            .nodes
            .get(node)
            .ok_or(StoreError::InvalidHandle("opponent recheck path node"))?;
        if !self
            .engine
            .semantic
            .stores
            .states
            .get(node.state)?
            .same_state(&node.position.snapshot())
        {
            return Err(StoreError::InvalidEvidence("opponent recheck path Rules state").into());
        }
        Ok(())
    }

    fn opponent_recheck_tail(
        &mut self,
        repair: ReplayRepairOutcome,
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
    ) -> Result<ReplayOpponentOutcome, ReplayError> {
        let ReplayRepairOutcome::CompletedRepairEndpoint {
            repaired_line,
            repair_record_revision,
            execution,
            observation,
            ..
        } = repair
        else {
            return Ok(ReplayOpponentOutcome::RepairNotReady(repair));
        };
        self.check_control(limits, cancel)?;
        let stage = self.stages.get(2).ok_or(PalsError::Capacity)?;
        if stage.phase != ReplayCpuPhase::RepairEndpoint
            || !stage.exact_completed
            || stage.execution != execution
            || stage.observation != Some(observation)
            || self.task(execution)?.status != TaskStatus::Completed(observation)
        {
            return Err(StoreError::InvalidEvidence(
                "opponent recheck lacks completed third-stage Repair evidence",
            )
            .into());
        }
        let Some((root, anchor, anchor_ply)) =
            self.opponent_anchor(repaired_line, repair_record_revision, limits, cancel)?
        else {
            return Ok(ReplayOpponentOutcome::NoEligibleOpponentAnchor);
        };
        self.opponent_recheck
            .as_mut()
            .ok_or(PalsError::Capacity)?
            .anchor_ply = Some(anchor_ply);
        let generation = self.engine.semantic.stores.generation();
        let active_root = self
            .engine
            .semantic
            .stores
            .root()
            .ok_or(StoreError::StaleConsumer)?;
        let root_revision = self
            .engine
            .semantic
            .stores
            .situations
            .get(active_root)?
            .revision;
        let mut counterline = std::mem::take(
            &mut self
                .opponent_recheck
                .as_mut()
                .ok_or(PalsError::Capacity)?
                .counterline,
        );
        counterline.extend_from_slice(&self.repaired_line[..anchor_ply]);
        let generated = self.generate_opponent_continuation(
            anchor,
            anchor_ply,
            &mut counterline,
            limits,
            cancel,
            counters,
        );
        self.opponent_recheck
            .as_mut()
            .ok_or(PalsError::Capacity)?
            .counterline = counterline;
        let Some(endpoint) = generated? else {
            return Ok(ReplayOpponentOutcome::NoAlternativeResponse);
        };
        self.check_control(limits, cancel)?;
        if self.engine.semantic.stores.generation() != generation
            || self.engine.semantic.stores.root() != Some(active_root)
            || self
                .engine
                .semantic
                .stores
                .situations
                .get(active_root)?
                .revision
                != root_revision
        {
            return Err(StoreError::StaleConsumer.into());
        }
        let mut checked = self.root.clone();
        for &movement in self.opponent_counterline() {
            self.check_control(limits, cancel)?;
            if !matches!(
                checked.classify_position()?.play_status,
                PlayStatus::Ongoing
            ) {
                return Err(StoreError::InvalidEvidence(
                    "opponent continuation follows Rules terminal",
                )
                .into());
            }
            checked.make_move(movement)?;
        }
        let snapshot = checked.snapshot();
        if !snapshot.same_state(&self.engine.semantic.nodes[endpoint].position.snapshot())
            || checked.legal_moves() != self.engine.semantic.nodes[endpoint].position.legal_moves()
        {
            return Err(StoreError::InvalidEvidence(
                "opponent continuation full Rules history/order",
            )
            .into());
        }
        let mut line = std::mem::take(
            &mut self
                .opponent_recheck
                .as_mut()
                .ok_or(PalsError::Capacity)?
                .counterline,
        );
        let publication = self.engine.record_checked(
            RecordKind::Counterexample,
            &line,
            None,
            0,
            None,
            None,
            limits,
            cancel,
        );
        self.opponent_recheck
            .as_mut()
            .ok_or(PalsError::Capacity)?
            .counterline = std::mem::take(&mut line);
        let (line_id, _) = publication?.ok_or(StoreError::InvalidEvidence(
            "opponent counterline publication absent",
        ))?;
        counters.refutations += 1;
        let revision = self.engine.semantic.revision;
        let state = self.opponent_recheck.as_mut().ok_or(PalsError::Capacity)?;
        state.endpoint_node = Some(endpoint);
        state.endpoint_snapshot = Some(snapshot);
        state.line_id = Some(line_id);
        state.record_revision = Some(revision);
        self.check_control(limits, cancel)?;
        let endpoint_state = self.engine.semantic.nodes[endpoint].state;
        let endpoint_situation = self.engine.semantic.nodes[endpoint].situation;
        if let Some((reason, _)) = self.engine.semantic.nodes[endpoint].terminal {
            return Ok(ReplayOpponentOutcome::RulesTerminalOpponentEndpoint {
                counterline: line_id,
                counter_record_revision: revision,
                endpoint_state,
                endpoint_situation,
                reason,
            });
        }
        if self.opponent_counterline().len() != self.engine.config.line_plies {
            return Err(StoreError::InvalidEvidence(
                "opponent continuation did not reach declared horizon",
            )
            .into());
        }
        if self
            .engine
            .cpu_budget_used(counters)
            .checked_add(self.plan.nodes_per_check)
            .is_none_or(|n| n > limits.max_cpu_nodes)
        {
            return Err(ReplayError::InvalidPlan(
                "original remaining CPU budget lacks the exact fourth N",
            ));
        }
        self.engine.researcher.cpu = Box::new(OwnedCpuChecker::new(CpuEngine::new(
            self.plan.cpu.clone(),
        )?)?);
        self.engine.start_checker(limits.deadline, cancel)?;
        let completed = self.unrestricted_endpoint_stage(
            StageRequest {
                root,
                target: endpoint,
                phase: ReplayCpuPhase::RepairOpponentEndpoint,
                depth: self.plan.requested_depth,
                limits,
                cancel,
            },
            counters,
            EndpointKind::Opponent,
        )?;
        self.check_control(limits, cancel)?;
        let fourth = self.stages.get(3).ok_or(PalsError::Capacity)?;
        let Some(observation) = completed else {
            return Ok(ReplayOpponentOutcome::PartialOpponentEndpoint {
                execution: fourth.execution,
                observation: fourth.observation,
            });
        };
        if !fourth.exact_completed
            || fourth.observation != Some(observation)
            || self.task(fourth.execution)?.status != TaskStatus::Completed(observation)
        {
            return Err(
                StoreError::InvalidEvidence("opponent fourth-stage completion provenance").into(),
            );
        }
        Ok(ReplayOpponentOutcome::CompletedOpponentEndpoint {
            repaired_line,
            repair_record_revision,
            counterline: line_id,
            counter_record_revision: revision,
            endpoint_state,
            endpoint_situation,
            execution: fourth.execution,
            observation,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_opponent_continuation(
        &mut self,
        anchor: usize,
        anchor_ply: usize,
        prefix: &mut Vec<BoardMove>,
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
    ) -> Result<Option<usize>, ReplayError> {
        let mut node = anchor;
        while prefix.len() < self.engine.config.line_plies
            && self.engine.semantic.nodes[node].terminal.is_none()
        {
            self.check_control(limits, cancel)?;
            let context = self.engine.role_context(
                node,
                RoleQuestion {
                    purpose: RoleQueryPurpose::ReplyPolicy,
                    prefix,
                    proposal: &self.repaired_line,
                    refutation: Some(&self.model_counterline),
                    divergences: &[],
                },
            )?;
            let steps = &mut self
                .opponent_recheck
                .as_mut()
                .ok_or(PalsError::Capacity)?
                .steps;
            if steps.len() == steps.capacity() {
                return Err(PalsError::Capacity.into());
            }
            steps.push(ReplayOpponentRoleStep {
                context,
                accepted: false,
                selected: None,
            });
            let accepted_before = counters.accepted_critic_outputs;
            let ranked = self.engine.ranked(
                node,
                Call::Reply,
                prefix,
                &self.repaired_line,
                Some(&self.model_counterline),
                limits,
                cancel,
                counters,
            );
            let steps = &mut self
                .opponent_recheck
                .as_mut()
                .ok_or(PalsError::Capacity)?
                .steps;
            steps.last_mut().ok_or(PalsError::Capacity)?.accepted =
                counters.accepted_critic_outputs != accepted_before;
            let ranked = ranked?;
            self.check_control(limits, cancel)?;
            let movement = if prefix.len() == anchor_ply {
                ranked
                    .into_iter()
                    .find(|movement| *movement != self.repaired_line[anchor_ply])
            } else {
                ranked.first().copied()
            };
            let Some(movement) = movement else {
                return Ok(None);
            };
            let mut position = self.engine.semantic.nodes[node].position.clone();
            position.make_move(movement)?;
            node = self
                .engine
                .connect_checked(node, movement, position, counters, limits, cancel)?;
            prefix.push(movement);
            self.opponent_recheck
                .as_mut()
                .ok_or(PalsError::Capacity)?
                .steps
                .last_mut()
                .ok_or(PalsError::Capacity)?
                .selected = Some(movement);
        }
        self.check_control(limits, cancel)?;
        Ok(Some(node))
    }
}
