//! Actual NoResearcher port and manager, with authored native-service heads.
//! These are lifecycle/Rules/goal checks, never neural or GPU evidence.
use super::*;
use crate::pals_adaptive::PcPortDriver;
use rz_contracts::rovent::{ExecutionMode, SearchGoal};
use rz_runtime::contracts::ContractClock;
use rz_search::contract_time::InstantClock;
use rz_search::pals::coordinator::AcceptedRoleStep;
use rz_search::pals::engine::adaptive_port::{AdaptiveModelNamespace, AdaptivePortModel};
use rz_search::pals::engine::pc_port::{PcFrontierStartup, PcPortQuestion, PcSemanticPort};
use rz_search::pals::engine::{
    LegalOrderRoleMock, NoResearcher, PalsSemanticConfig, PcSemanticOwner, PcSemanticProfile,
    RoleModel,
};
use rz_search::pals::review_context::ReviewIndexLimits;
use rz_search::pals::rovent_frontier::{FrontierGoalStatus, FrontierLimits, FrontierRegistration};
use std::num::NonZeroU16;

/// Preserve the real semantic question's public revision in this authored
/// backend. The generic manager fixture deliberately uses revision zero;
/// these actual PC goals publish lines and subsequently prepare newer heads.
struct GoalNativeFixture {
    inner: NativeFixture,
}
impl TicketNativeBackend<ReviewView, PcPortQuestion> for GoalNativeFixture {
    type Prepared = RepresentationKey;
    type Output = RoleOutput;

    fn clock(&self) -> ContractSystemClock {
        <NativeFixture as TicketNativeBackend<ReviewView, PcPortQuestion>>::clock(&self.inner)
    }
    fn prepare(
        &mut self,
        question: &TicketPreparation<ReviewView, PcPortQuestion>,
        canceled: &AtomicBool,
    ) -> Result<(Self::Prepared, RepresentationKey), RoleFailure> {
        let (mut prepared, mut key) = <NativeFixture as TicketNativeBackend<
            ReviewView,
            PcPortQuestion,
        >>::prepare(&mut self.inner, question, canceled)?;
        let revision = question.query.context().public_revision;
        prepared.record_revision = revision;
        key.record_revision = revision;
        Ok((prepared, key))
    }
    fn prepared_host_bytes(&self, prepared: &Self::Prepared) -> u64 {
        <NativeFixture as TicketNativeBackend<ReviewView, PcPortQuestion>>::prepared_host_bytes(
            &self.inner,
            prepared,
        )
    }
    fn run(
        &mut self,
        ticket: &RoleTicketEnvelope,
        question: &TicketPreparation<ReviewView, PcPortQuestion>,
        prepared: &Self::Prepared,
        canceled: &AtomicBool,
    ) -> TicketPhysicalResult<Self::Output> {
        <NativeFixture as TicketNativeBackend<ReviewView, PcPortQuestion>>::run(
            &mut self.inner,
            ticket,
            question,
            prepared,
            canceled,
        )
    }
    fn output_host_bytes(&self, output: &Self::Output) -> u64 {
        <NativeFixture as TicketNativeBackend<ReviewView, PcPortQuestion>>::output_host_bytes(
            &self.inner,
            output,
        )
    }
    fn acceptance_output_digest(&self, output: &Self::Output) -> Option<Digest> {
        <NativeFixture as TicketNativeBackend<ReviewView, PcPortQuestion>>::acceptance_output_digest(
            &self.inner,
            output,
        )
    }
    fn execution_evidence(
        &mut self,
        ticket: &RoleTicketEnvelope,
    ) -> Option<TicketExecutionEvidence> {
        <NativeFixture as TicketNativeBackend<ReviewView, PcPortQuestion>>::execution_evidence(
            &mut self.inner,
            ticket,
        )
    }
    fn accept(
        &mut self,
        ticket: &RoleTicketEnvelope,
        question: &TicketPreparation<ReviewView, PcPortQuestion>,
        prepared: &Self::Prepared,
        output: &Self::Output,
    ) -> Result<(), RoleFailure> {
        <NativeFixture as TicketNativeBackend<ReviewView, PcPortQuestion>>::accept(
            &mut self.inner,
            ticket,
            question,
            prepared,
            output,
        )
    }
    fn discard(&mut self, ticket: &RoleTicketEnvelope) -> Result<(), RoleFailure> {
        <NativeFixture as TicketNativeBackend<ReviewView, PcPortQuestion>>::discard(
            &mut self.inner,
            ticket,
        )
    }
    fn quiesce(&mut self, until: Instant) -> bool {
        <NativeFixture as TicketNativeBackend<ReviewView, PcPortQuestion>>::quiesce(
            &mut self.inner,
            until,
        )
    }
    fn finish(&mut self, until: Instant) -> bool {
        <NativeFixture as TicketNativeBackend<ReviewView, PcPortQuestion>>::finish(
            &mut self.inner,
            until,
        )
    }
}

struct ActualGoalDriver {
    inner: PcPortDriver<AdaptivePortModel>,
    submitted: Vec<(RoleTicketEnvelope, FrameInput<ReviewView, PcPortQuestion>)>,
    consumed: Vec<RoleTicketEnvelope>,
    consumed_missing_fresh: Vec<RoleTicketEnvelope>,
}
impl RoleOnlySearchDriver<PcPortQuestion> for ActualGoalDriver {
    fn researcher(&self) -> &NoResearcher {
        self.inner.researcher()
    }
    fn validate_registration(
        &self,
        authority: SearchAuthority,
        mode: ExecutionMode,
        until: Instant,
        budget: Digest,
    ) -> Result<(), AdaptiveManagerError> {
        self.inner
            .validate_registration(authority, mode, until, budget)
    }
    fn next_role(
        &mut self,
    ) -> Result<Option<FrameInput<ReviewView, PcPortQuestion>>, AdaptiveManagerError> {
        self.inner.next_role()
    }
    fn role_submitted(
        &mut self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(), AdaptiveManagerError> {
        self.inner.role_submitted(input, ticket)?;
        // Retain only the first imposed input and first imposed Fresh input.
        // Keeping every submitted view would itself consume Store lease seats
        // and turn the fixture into a different memory/admission workload.
        if input.question.goal_local_origin_digest().is_some()
            && (self.submitted.is_empty()
                || (input.question.context().purpose == RoleQueryPurpose::ValueFresh
                    && !self.submitted.iter().any(|(_, original)| {
                        original.question.context().purpose == RoleQueryPurpose::ValueFresh
                    })))
        {
            self.submitted.push((*ticket, input.clone()));
        }
        Ok(())
    }
    fn validate_role(
        &self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
    ) -> Result<(), AdaptiveManagerError> {
        self.inner.validate_role(input, ticket, output)
    }
    fn consume_accepted(
        &mut self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: Option<TicketExecutionEvidence>,
    ) -> Result<(), AdaptiveManagerError> {
        self.inner
            .consume_accepted(input, accepted, output, physical)?;
        assert!(accepted.validates_output(output));
        self.consumed.push(*accepted.ticket());
        if input.question.goal_local_origin_digest().is_some()
            && input.question.context().purpose == RoleQueryPurpose::ValueFresh
        {
            self.consumed_missing_fresh.push(*accepted.ticket());
        }
        Ok(())
    }
    fn role_discarded(
        &mut self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        ticket: Option<&RoleTicketEnvelope>,
    ) -> Result<(), AdaptiveManagerError> {
        self.inner.role_discarded(input, ticket)
    }
    fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }
    fn close_role_call_budget(
        &mut self,
        admitted: u64,
        limit: u64,
    ) -> Result<(), AdaptiveManagerError> {
        self.inner.close_role_call_budget(admitted, limit)
    }
    fn role_consumption_recorded(
        &self,
        ticket: &RoleTicketEnvelope,
        input: &FrameInput<ReviewView, PcPortQuestion>,
    ) -> Result<bool, AdaptiveManagerError> {
        self.inner.role_consumption_recorded(ticket, input)
    }
    fn quiesce_satisfied_goal(&mut self) -> Result<(), AdaptiveManagerError> {
        self.inner.quiesce_satisfied_goal()
    }
    fn revoke(&mut self) -> Result<(), AdaptiveManagerError> {
        self.inner.revoke()
    }
    fn review_lease_admission(&self) -> Result<AdaptiveReviewLeaseAdmission, AdaptiveManagerError> {
        self.inner.review_lease_admission()
    }
}

type ActualGoalManager =
    RoleOnlyExecutionManager<PcPortQuestion, GoalNativeFixture, ActualGoalDriver>;

fn fixture(goal: SearchGoal) -> (ActualGoalManager, Arc<NativeFacts>, Instant) {
    fixture_with_role_limit(goal, 128)
}

fn fixture_with_role_limit(
    goal: SearchGoal,
    role_limit: u64,
) -> (ActualGoalManager, Arc<NativeFacts>, Instant) {
    let clock = ContractSystemClock::new(ProcessEpoch(717));
    let until = clock.origin() + WINDOW;
    let authority = SearchAuthority {
        epoch: ProcessEpoch(717),
        // A freshly constructed semantic owner is in its actual initial game.
        game: GameGeneration(0),
        root: RootGeneration(1),
        implementation: Digest([91; 32]),
    };
    let deadline = rz_contracts::Deadline {
        clock: clock.domain(),
        at: MonotonicTick(WINDOW.as_nanos() as u64),
    };
    let value = LegalOrderRoleMock.value_identity().unwrap().clone();
    let namespace = AdaptiveModelNamespace::new(
        value.clone(),
        Digest([5; 32]),
        Digest([6; 32]),
        PrecisionProfile::Fp32,
        1,
    )
    .unwrap();
    // Only metadata is registered; every inline physical model call rejects.
    let model =
        AdaptivePortModel::new_with_namespace(value.model.clone(), Some(value), namespace.clone())
            .unwrap();
    let mut owner = PcSemanticOwner::new_selected_rovent(
        PcSemanticProfile {
            semantic: PalsSemanticConfig {
                beam_width: 1,
                line_plies: 2,
                max_nodes: 257,
                max_records: 128,
                max_role_calls: role_limit,
            },
            maximum_root_prefix_plies: 16,
        },
        model,
        ReviewSelectionPolicy::default(),
        ReviewIndexLimits::default(),
    )
    .unwrap();
    let position = Position::startpos();
    let registration = owner.register_adaptive_root(&position, authority).unwrap();
    let next_owner = Arc::new(std::sync::atomic::AtomicU64::new(717));
    let issuer: rz_search::pals::engine::adaptive_port::AdaptiveRulesOwnerIssuer =
        Arc::new(move || Ok(OwnerId(next_owner.fetch_add(1, Ordering::Relaxed))));
    let roots = [BoardMove::from_uci("e2e4").unwrap()];
    let mut port = PcSemanticPort::new_registered_retaining(
        owner,
        position,
        authority,
        deadline,
        until,
        ExecutionMode::CpuPc,
        issuer,
        registration,
        Some(&roots),
    )
    .unwrap_or_else(|(_, error)| panic!("actual PC port: {error:?}"));
    port.attach_rovent_frontier(
        FrontierRegistration {
            authority,
            mode: ExecutionMode::CpuPc,
            original_deadline: deadline,
            root_budget: Digest([92; 32]),
            goal,
            nodes: Some(2000),
            selective_policy: Digest([93; 32]),
            root_rules: port.root_rules(),
            root_moves: Arc::clone(port.root_move_domain()),
            root_state: port.root_state_id(),
            model_namespace: Arc::new(namespace),
            native_ledger: None,
        },
        FrontierLimits {
            maximum_goal_plies: NonZeroU16::new(16).unwrap(),
            maximum_frames: 64,
            maximum_paths: 128,
            maximum_mate_nodes: 256,
            maximum_native_receipts: 1,
        },
        PcFrontierStartup {
            root_budget: Digest([92; 32]),
            clock: InstantClock::new(clock.domain(), clock.origin()),
        },
    )
    .unwrap();
    let facts = Arc::new(NativeFacts::default());
    facts.child_gate.store(true, Ordering::Release);
    let owned_facts = Arc::clone(&facts);
    let native_clock = clock.clone();
    let native = TicketedNativeService::spawn(
        move || {
            Ok(GoalNativeFixture {
                inner: NativeFixture {
                    clock: native_clock,
                    facts: owned_facts,
                    finish_ok: true,
                },
            })
        },
        TicketServiceLimits::default(),
    )
    .unwrap();
    let driver = ActualGoalDriver {
        inner: PcPortDriver::new(port),
        submitted: Vec::new(),
        consumed: Vec::new(),
        consumed_missing_fresh: Vec::new(),
    };
    let manager = RoleOnlyExecutionManager::new_retaining(
        RoleOnlyManagerRegistration {
            authority,
            mode: ExecutionMode::CpuPc,
            root_budget: Digest([92; 32]),
            hard_deadline: until,
            token_budget: 512,
            first_request_sequence: 1,
            role_tickets_max: 2,
            role_call_limit: role_limit,
            event_order: CoordinatorPolicy::Fixed,
        },
        clock,
        RoleOnlyManagerSources { native, driver },
    )
    .unwrap_or_else(|rejected| panic!("actual PC manager: {}", rejected.error));
    (manager, facts, until)
}

fn close_and_recover(mut manager: ActualGoalManager, until: Instant) -> RoleOnlyShutdownReceipt {
    manager.begin_root_quiesce(until).unwrap();
    let receipt = manager
        .run_root_until(until, &AtomicBool::new(false))
        .unwrap();
    assert!(receipt.permits_semantic_closure(), "{receipt:?}");
    assert!(
        manager.failures().next().is_none(),
        "actual PC quiesce failures: {:?}",
        manager.failures().collect::<Vec<_>>()
    );
    assert_eq!(receipt.counters.cpu_slice_receipts, 0);
    assert_eq!(receipt.counters.validator_ticket_admissions, 0);
    let mut sources = manager
        .into_sources()
        .unwrap_or_else(|(_, error)| panic!("actual PC fence: {error}"));
    // These fixture-only observations retain original views. Drop them only
    // after the real native fence, before checking semantic pin retirement.
    sources.driver.submitted.clear();
    let mut port = sources
        .driver
        .inner
        .into_port(&receipt)
        .unwrap_or_else(|(_, error)| panic!("PC after actual owner fence: {error}"));
    assert_eq!(std::mem::size_of_val(port.researcher()), 0);
    let repair = port.take_repair_report().unwrap();
    if let Some(report) = &repair {
        assert_eq!(report.authority(), receipt.authority);
        assert_eq!(report.root_budget(), receipt.root_budget);
    }
    let frontier = port.retire_rovent_frontier().unwrap().unwrap();
    assert_eq!(frontier.snapshot.work.native_nodes, 0);
    assert_eq!(frontier.snapshot.work.reserved_native_nodes, 0);
    let mut owner = port.into_owner();
    assert!(owner.retired_repair_report().is_none());
    assert_eq!(owner.stores().review_pin_count().unwrap(), 0);
    let next = SearchAuthority {
        root: RootGeneration(receipt.authority.root.0 + 1),
        ..receipt.authority
    };
    let registration = owner
        .register_adaptive_root(&Position::startpos(), next)
        .unwrap();
    owner.abandon_unconsumed_adaptive_root(next).unwrap();
    drop(registration);
    // A recovered report is separate historical custody; it did not grant the
    // actual fence or the next root and must not retain an old review lease.
    drop(repair);
    sources.native.begin_shutdown(until);
    while !sources.native.try_join().unwrap() {
        assert!(Instant::now() < until, "actual native owner join timed out");
        sources.native.poll();
        thread::sleep(Duration::from_millis(1));
    }
    receipt
}

#[test]
fn no_researcher_manager_consumes_real_missing_origin_through_retired_fresh_head() {
    let (mut manager, facts, until) = fixture(SearchGoal {
        depth_plies: None,
        mate_moves: NonZeroU16::new(2),
    });
    while manager.driver().consumed_missing_fresh.is_empty() {
        assert!(
            Instant::now() < until,
            "actual Missing/Fresh consumer timed out; failures={:?}",
            manager.failures().collect::<Vec<_>>()
        );
        manager.tick().unwrap();
        assert!(
            manager.failures().next().is_none(),
            "{:?}",
            manager.failures().collect::<Vec<_>>()
        );
        thread::sleep(Duration::from_millis(1));
    }
    let source = manager.driver().consumed_missing_fresh[0];
    let (_, input) = manager
        .driver()
        .submitted
        .iter()
        .find(|(ticket, _)| *ticket == source)
        .unwrap();
    assert!(input.question.goal_local_origin_digest().is_some());
    assert!(input.question.context().public_revision > 0);
    assert_eq!(
        source.representation.record_revision,
        input.question.context().public_revision
    );
    let (_, original) = manager
        .driver()
        .submitted
        .iter()
        .find(|(_, input)| input.question.goal_local_origin_digest().is_some())
        .unwrap();
    assert_eq!(
        input.question.goal_local_origin_digest(),
        original.question.goal_local_origin_digest()
    );
    assert!(original.view.query().proposal.is_empty());
    assert!(original.view.query().counterexample.is_none());
    assert_eq!(
        facts
            .accepted
            .lock()
            .unwrap()
            .iter()
            .filter(|ticket| **ticket == source)
            .count(),
        1
    );
    assert_eq!(
        manager
            .driver()
            .consumed
            .iter()
            .filter(|ticket| **ticket == source)
            .count(),
        1
    );
    let snapshot = manager
        .driver()
        .inner
        .port()
        .rovent_frontier_receipt()
        .unwrap()
        .snapshot;
    assert!(snapshot.work.checked_transitions > 2);
    assert_eq!(snapshot.work.native_nodes, 0);
    assert!(manager.driver().inner.result().counters.accepted_fresh > 0);
    // This finite check stops after the actual source consumer. It does not
    // claim that every missing defense or the mate goal has been completed.
    close_and_recover(manager, until);
}

#[test]
fn actual_no_researcher_depth_completion_survives_fence_and_frontier_retirement() {
    let (mut manager, _, until) = fixture(SearchGoal {
        depth_plies: NonZeroU16::new(5),
        mate_moves: None,
    });
    let receipt = manager
        .run_root_until(until, &AtomicBool::new(false))
        .unwrap();
    assert!(receipt.permits_semantic_closure(), "{receipt:?}");
    assert!(manager.failures().next().is_none());
    let snapshot = manager
        .driver()
        .inner
        .port()
        .rovent_frontier_receipt()
        .unwrap()
        .snapshot;
    assert_eq!(snapshot.goal, FrontierGoalStatus::DepthSatisfied);
    assert_eq!(snapshot.horizon.completed_depth.unwrap().0, 5);
    assert_eq!(snapshot.completed_frames, snapshot.registered_frames);
    assert!(manager.driver().inner.result().counters.accepted_fresh > 0);
    assert!(
        manager
            .driver()
            .consumed
            .iter()
            .any(|ticket| ticket.representation.record_revision > 0)
    );
    close_and_recover(manager, until);
}

#[test]
fn actual_no_goal_selected_completion_survives_the_original_native_fence() {
    let (mut manager, _, until) = fixture(SearchGoal::default());
    let receipt = manager
        .run_root_until(until, &AtomicBool::new(false))
        .unwrap();
    assert!(receipt.permits_semantic_closure(), "{receipt:?}");
    assert!(manager.failures().next().is_none());
    let snapshot = manager
        .driver()
        .inner
        .port()
        .rovent_frontier_receipt()
        .unwrap()
        .snapshot;
    assert_eq!(snapshot.goal, FrontierGoalStatus::SelectiveFrontier);
    assert!(snapshot.selection_sealed && snapshot.registered_frames > 0);
    assert_eq!(snapshot.completed_frames, snapshot.registered_frames);
    assert_eq!(snapshot.horizon.completed_depth, None);
    assert!(!snapshot.horizon.target_satisfied);
    assert_eq!(snapshot.work.native_nodes, 0);
    assert_eq!(snapshot.work.outstanding_native_executions, 0);
    assert!(manager.driver().inner.result().counters.accepted_fresh > 0);
    assert_eq!(receipt.counters.validator_ticket_admissions, 0);
    assert_eq!(receipt.counters.cpu_slice_receipts, 0);
    close_and_recover(manager, until);
}

#[test]
fn actual_role_limit_closes_only_after_the_original_head_is_settled() {
    use rz_search::pals::rovent_frontier::FrontierStop;
    let (mut manager, _, until) = fixture_with_role_limit(SearchGoal::default(), 1);
    let receipt = manager
        .run_root_until(until, &AtomicBool::new(false))
        .unwrap();
    assert!(receipt.permits_semantic_closure(), "{receipt:?}");
    assert!(manager.failures().next().is_none());
    let port = manager.driver().inner.port();
    let snapshot = port.rovent_frontier_receipt().unwrap().snapshot;
    assert_eq!(
        snapshot.goal,
        FrontierGoalStatus::Stopped(FrontierStop::RoleBudget)
    );
    assert_eq!(port.finish_materialized().counters.submitted_role_calls, 1);
    assert_eq!(port.finish_materialized().counters.consumed_role_outputs, 1);
    assert_eq!(receipt.pending_tickets, 0);
    assert_eq!(snapshot.horizon.completed_depth, None);
    close_and_recover(manager, until);
}
