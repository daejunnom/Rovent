//! Authored CPU native-service outputs, real registry/acceptance/fences, and
//! structurally absent R owners. These tests execute no neural model or GPU.
use super::*;
use rz_contracts::RequestId;
use rz_search::pals::coordinator::AcceptedRoleStep;
use rz_search::pals::engine::NoResearcher;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, Default)]
enum ConsumeFault {
    #[default]
    None,
    ErrorAfterCommit,
    ReadbackUnknownAfterCommit,
    DeadlineBeforeCommit,
}

struct RoleOnlyFixture {
    no_r: NoResearcher,
    registration: RoleOnlyManagerRegistration,
    ready: VecDeque<FrameInput<ReviewView, u64>>,
    stores: PalsStores,
    submitted: BTreeMap<RequestId, (RoleTicketEnvelope, FrameInput<ReviewView, u64>)>,
    consumed: Vec<(RoleTicketEnvelope, u64)>,
    discarded: usize,
    strict_discard: bool,
    consume_fault: ConsumeFault,
    revoked: bool,
}
impl RoleOnlySearchDriver<u64> for RoleOnlyFixture {
    fn researcher(&self) -> &NoResearcher {
        &self.no_r
    }
    fn validate_registration(
        &self,
        authority: SearchAuthority,
        mode: rz_contracts::rovent::ExecutionMode,
        hard_deadline: Instant,
        root_budget: Digest,
    ) -> Result<(), AdaptiveManagerError> {
        assert_eq!(authority, self.registration.authority);
        assert_eq!(mode, self.registration.mode);
        assert_eq!(hard_deadline, self.registration.hard_deadline);
        assert_eq!(root_budget, self.registration.root_budget);
        Ok(())
    }
    fn next_role(&mut self) -> Result<Option<FrameInput<ReviewView, u64>>, AdaptiveManagerError> {
        Ok(self.ready.pop_front())
    }
    fn role_submitted(
        &mut self,
        input: &FrameInput<ReviewView, u64>,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(), AdaptiveManagerError> {
        assert!(
            self.submitted
                .insert(ticket.id, (*ticket, input.clone()))
                .is_none()
        );
        Ok(())
    }
    fn validate_role(
        &self,
        input: &FrameInput<ReviewView, u64>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
    ) -> Result<(), AdaptiveManagerError> {
        let (original, submitted) = self.submitted.get(&ticket.id).unwrap();
        assert_eq!(original, ticket);
        assert_eq!(submitted.question_digest, input.question_digest);
        assert_eq!(submitted.view_seal, input.view_seal);
        assert!(Arc::ptr_eq(&submitted.view, &input.view));
        assert!(input.view.validates(&self.stores));
        assert_eq!(output.id, ticket.id);
        assert_eq!(output.key, ticket.representation);
        Ok(())
    }
    fn consume_accepted(
        &mut self,
        input: &FrameInput<ReviewView, u64>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        _: Option<TicketExecutionEvidence>,
    ) -> Result<(), AdaptiveManagerError> {
        self.validate_role(input, accepted.ticket(), output)?;
        assert!(accepted.validates_output(output));
        if matches!(self.consume_fault, ConsumeFault::DeadlineBeforeCommit) {
            return Err(AdaptiveManagerError::Deadline);
        }
        self.submitted.remove(&accepted.ticket().id).unwrap();
        self.consumed
            .push((*accepted.ticket(), input.view.sequence()));
        if matches!(
            self.consume_fault,
            ConsumeFault::ErrorAfterCommit | ConsumeFault::ReadbackUnknownAfterCommit
        ) {
            return Err(AdaptiveManagerError::Invalid(
                "fixture error after actual consume",
            ));
        }
        Ok(())
    }
    fn role_discarded(
        &mut self,
        _: &FrameInput<ReviewView, u64>,
        ticket: Option<&RoleTicketEnvelope>,
    ) -> Result<(), AdaptiveManagerError> {
        if let Some(ticket) = ticket {
            let original = self.submitted.remove(&ticket.id);
            if self.strict_discard {
                assert!(
                    original.is_some(),
                    "consumed ticket cannot be discarded again"
                );
            }
        }
        self.discarded += 1;
        Ok(())
    }
    fn role_consumption_recorded(
        &self,
        ticket: &RoleTicketEnvelope,
        _: &FrameInput<ReviewView, u64>,
    ) -> Result<bool, AdaptiveManagerError> {
        if matches!(self.consume_fault, ConsumeFault::ReadbackUnknownAfterCommit) {
            return Err(AdaptiveManagerError::Invalid(
                "fixture consumption readback unknown",
            ));
        }
        Ok(self.consumed.iter().any(|(recorded, _)| recorded == ticket))
    }
    fn is_complete(&self) -> bool {
        self.ready.is_empty() && self.submitted.is_empty()
    }
    fn revoke(&mut self) -> Result<(), AdaptiveManagerError> {
        self.revoked = true;
        self.ready.clear();
        Ok(())
    }
    fn review_lease_admission(
        &self,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::AdaptiveReviewLeaseAdmission,
        AdaptiveManagerError,
    > {
        let current = self.stores.review_pin_count().unwrap();
        let capacity = self.stores.review_pin_capacity();
        Ok(
            rz_search::pals::engine::adaptive_port::AdaptiveReviewLeaseAdmission {
                current,
                capacity,
                required_new_leases: 1,
                frontier_replacement_reserve: 0,
                frontier_owner_present: false,
                may_prepare: current < capacity,
            },
        )
    }
}

fn role_only_fixture(
    block: bool,
) -> (
    RoleOnlyExecutionManager<u64, NativeFixture, RoleOnlyFixture>,
    Arc<NativeFacts>,
    Instant,
) {
    let clock = ContractSystemClock::new(ProcessEpoch(517));
    role_only_fixture_on_clock(block, clock)
}

fn role_only_fixture_on_clock(
    block: bool,
    clock: ContractSystemClock,
) -> (
    RoleOnlyExecutionManager<u64, NativeFixture, RoleOnlyFixture>,
    Arc<NativeFacts>,
    Instant,
) {
    let until = clock.origin() + WINDOW;
    let authority = SearchAuthority {
        epoch: ProcessEpoch(517),
        game: GameGeneration(1),
        root: RootGeneration(1),
        implementation: Digest([91; 32]),
    };
    let registration = RoleOnlyManagerRegistration {
        authority,
        mode: rz_contracts::rovent::ExecutionMode::CpuPc,
        root_budget: Digest([92; 32]),
        hard_deadline: until,
        token_budget: 512,
        first_request_sequence: 1,
        role_tickets_max: 2,
        role_call_limit: 2,
        event_order: CoordinatorPolicy::Fixed,
    };
    let root = Position::startpos();
    let rules = ContractPosition::new(OwnerId(517), root.clone())
        .export()
        .unwrap();
    let mut stores = PalsStores::new(StoreLimits::default());
    let situation = stores.focus_actual_moves(root.snapshot()).unwrap();
    let state = stores.situations.get(situation).unwrap().state;
    let mut index = ReviewContextIndex::default();
    let mut ready = VecDeque::new();
    for (number, stage, purpose) in [
        (
            1,
            RefinementStage::ProposalContinuation,
            RoleQueryPurpose::ProposePolicy,
        ),
        (
            2,
            RefinementStage::CriticReply,
            RoleQueryPurpose::ReplyPolicy,
        ),
    ] {
        let mut endpoint = root.clone();
        let prefix = if block {
            vec![root.legal_moves()[number as usize]]
        } else {
            Vec::new()
        };
        for movement in &prefix {
            endpoint.make_move(*movement).unwrap();
        }
        let legal = endpoint.legal_moves();
        let view = Arc::new(
            index
                .select(
                    &stores,
                    state,
                    ReviewQuerySequences {
                        purpose,
                        prefix: &prefix,
                        proposal: &[],
                        counterexample: None,
                        legal: &legal,
                        divergences: &[],
                    },
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        let seal = ReviewViewSeal {
            authority,
            view_sequence: view.sequence(),
            context_policy: Digest(view.policy_digest()),
            view_digest: Digest(view.seal_digest()),
            root_state: rules.snapshot().identity(),
            actual_tokens: view.token_count(),
            selected_records: view.records().len() as u16,
            anchors: view.anchors().len() as u8,
            length_bucket: view.length_bucket(),
        };
        ready.push_back(FrameInput {
            stage,
            question: Arc::new(number),
            question_digest: Digest(view.digest()),
            public_context: Digest(view.record_scope_digest()),
            view_seal: seal,
            view,
        });
    }
    let driver = RoleOnlyFixture {
        no_r: NoResearcher,
        registration,
        ready,
        stores,
        submitted: BTreeMap::new(),
        consumed: Vec::new(),
        discarded: 0,
        strict_discard: false,
        consume_fault: ConsumeFault::None,
        revoked: false,
    };
    let facts = Arc::new(NativeFacts::default());
    if !block {
        facts.child_gate.store(true, Ordering::Release);
    }
    let owned_facts = Arc::clone(&facts);
    let native_clock = clock.clone();
    let native = TicketedNativeService::spawn(
        move || {
            Ok(NativeFixture {
                clock: native_clock,
                facts: owned_facts,
                finish_ok: true,
            })
        },
        TicketServiceLimits::default(),
    )
    .unwrap();
    let manager = RoleOnlyExecutionManager::new_retaining(
        registration,
        clock,
        RoleOnlyManagerSources { native, driver },
    )
    .unwrap_or_else(|rejected| panic!("NoR admission: {}", rejected.error));
    (manager, facts, until)
}

type PonderRoleFixture = (
    RoleOnlyExecutionManager<u64, NativeFixture, RoleOnlyFixture>,
    Arc<NativeFacts>,
    crate::rovent_ponder::RoventPonderClockOwner,
    crate::rovent_ponder::RoventPonderClockHandle,
    crate::SearchTicket,
    crate::engine::ProcessClock,
    Instant,
);

/// The actual Session issues this source's Start. The manager/native fixture
/// uses that same process origin; no readonly handle can mint a new root.
fn ponder_role_fixture() -> PonderRoleFixture {
    use crate::bridge::BuildSearchSettings;
    use crate::engine::{ProcessClock, driver_start_fixture};
    use crate::parser::GoLimits;
    let clock = ProcessClock::new(ProcessEpoch(517));
    let runtime = clock.runtime_clock();
    let (mut manager, facts, until) = role_only_fixture_on_clock(true, runtime.clone());
    let start = match driver_start_fixture::issue_start(
        &Position::startpos(),
        GoLimits {
            ponder: true,
            ..GoLimits::default()
        },
        manager.driver().registration.authority,
        &clock,
        runtime.origin(),
        BuildSearchSettings {
            untimed_limit: WINDOW + Duration::from_millis(10),
            time_config: rz_search::time::TimeBudgetConfig::default(),
            max_simulations: 1024,
        },
        Duration::from_secs(1),
    )
    .unwrap()
    {
        driver_start_fixture::FixtureStart::Started(context) => context,
        _ => panic!("nonterminal actual Rules must issue Start"),
    };
    let original = start.original_time_budget.unwrap();
    assert_eq!(original.hard_deadline, until);
    let ticket = start.original_start.as_ref().unwrap().ticket().clone();
    let (owner, handle) = crate::rovent_ponder::RoventPonderClockOwner::new(
        ticket.clone(),
        manager.driver().registration.authority,
        original,
        start.shutdown_deadline,
        &clock,
    )
    .unwrap();
    manager
        .register_cleanup_deadline(start.shutdown_deadline)
        .unwrap();
    manager.register_effective_clock(handle.clone()).unwrap();
    (manager, facts, owner, handle, ticket, clock, until)
}

#[test]
fn role_only_ponder_admission_close_keeps_original_pending_ticket_acceptance() {
    use crate::bridge::{
        BuildSearchSettings, NodeBudgetSemantics, PreparedSearchBinding, SideToMove,
    };
    use crate::contracts::SessionScope;
    use crate::parser::GoLimits;
    let (mut manager, facts, mut owner, handle, ticket, clock, until) = ponder_role_fixture();
    while (manager.pending_tickets() < 2 || facts.started.lock().unwrap().is_empty())
        && Instant::now() < until
    {
        manager.tick().unwrap();
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(manager.pending_tickets(), 2);
    let hit_at = Instant::now();
    let normal = PreparedSearchBinding::prepare(
        &GoLimits {
            movetime_ms: Some(1000),
            ..GoLimits::default()
        },
        SideToMove::White,
        hit_at,
        BuildSearchSettings {
            time_config: rz_search::time::TimeBudgetConfig::default(),
            max_simulations: 1024,
            untimed_limit: WINDOW,
        },
        None,
        NodeBudgetSemantics::RoventPositionVisits,
    )
    .unwrap();
    owner
        .hit(
            &ticket,
            SessionScope::Search(manager.driver().registration.authority),
            hit_at,
            &normal,
            hit_at,
        )
        .unwrap();
    let gate = handle.snapshot(hit_at).budget().admission_deadline;
    assert!(!handle.snapshot(gate).can_admit(gate));
    assert!(!handle.snapshot(gate).hard_expired(gate));
    assert!(!manager.can_admit_work());
    facts.child_gate.store(true, Ordering::Release);
    while manager.driver().consumed.len() != 2 && Instant::now() < until {
        manager.tick().unwrap();
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(manager.driver().consumed.len(), 2);
    let original = clock.deadline(until).unwrap();
    assert!(
        facts
            .accepted
            .lock()
            .unwrap()
            .iter()
            .all(|ticket| ticket.original_deadline == original)
    );
    assert_eq!(facts.accepted.lock().unwrap().len(), 2);
    // Consuming the last ticket has already issued actual root quiescence.
    // Escalation may race its historical RootQuiesced packet below.
    assert_eq!(manager.phase(), ManagerPhase::Draining);
    manager
        .begin_shutdown(until + Duration::from_secs(1))
        .unwrap();
    let receipt = manager
        .run_root_until(until + Duration::from_secs(1), &AtomicBool::new(false))
        .unwrap();
    assert!(receipt.permits_semantic_closure(), "{receipt:?}");
    assert!(receipt.native_process_joined);
    assert!(receipt.native_physical_shutdown_confirmed);
    assert!(
        manager.failures().next().is_none(),
        "{:?}",
        manager.failures().collect::<Vec<_>>()
    );
}

#[test]
fn role_only_ponder_effective_hard_closes_before_any_late_head_consumption() {
    let (mut manager, facts, _owner, handle, _ticket, _clock, until) = ponder_role_fixture();
    while (manager.pending_tickets() < 2 || facts.started.lock().unwrap().is_empty())
        && Instant::now() < until
    {
        manager.tick().unwrap();
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(manager.pending_tickets(), 2);
    // A real handle's monotonic observation closes this boundary for every
    // later manager read, including a read with an older event timestamp.
    assert!(handle.snapshot(until).hard_expired(until));
    manager.tick().unwrap();
    facts.child_gate.store(true, Ordering::Release);
    let receipt = manager
        .run_root_until(until + Duration::from_secs(1), &AtomicBool::new(false))
        .unwrap();
    assert!(receipt.permits_semantic_closure(), "{receipt:?}");
    assert_eq!(manager.driver().consumed.len(), 0);
    assert_eq!(facts.accepted.lock().unwrap().len(), 0);
    assert!(manager.driver().revoked);
}

#[test]
fn role_only_native_two_pending_keep_their_original_views_and_return_no_researcher_owner() {
    let (mut manager, facts, until) = role_only_fixture(false);
    let receipt = manager
        .run_root_until(until, &AtomicBool::new(false))
        .unwrap();
    assert!(receipt.permits_semantic_closure(), "{receipt:?}");
    assert!(manager.failures().next().is_none());
    assert_eq!(manager.driver().consumed.len(), 2);
    assert_ne!(
        manager.driver().consumed[0].1,
        manager.driver().consumed[1].1
    );
    assert_eq!(facts.accepted.lock().unwrap().len(), 2);
    assert_eq!(receipt.counters.submitted_role_tickets, 2);
    assert_eq!(receipt.counters.consumed_role_outputs, 2);
    assert_eq!(receipt.counters.cpu_slice_receipts, 0);
    assert_eq!(receipt.counters.validator_ticket_admissions, 0);
    assert!(!manager.driver().revoked);
    let mut sources = manager
        .into_sources()
        .unwrap_or_else(|(_, error)| panic!("NoR physical closure: {error}"));
    sources.native.begin_shutdown(Instant::now() + WINDOW);
    let finish = Instant::now() + WINDOW;
    while !sources.native.try_join().unwrap() && Instant::now() < finish {
        sources.native.poll();
        thread::sleep(Duration::from_millis(1));
    }
    assert!(sources.native.try_join().unwrap());
}

#[test]
fn role_only_stop_drains_pending_native_inputs_without_consuming_late_heads() {
    let (mut manager, facts, until) = role_only_fixture(true);
    while (manager.pending_tickets() < 2 || facts.started.lock().unwrap().is_empty())
        && Instant::now() < until
    {
        manager.tick().unwrap();
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(manager.pending_tickets(), 2);
    manager.begin_shutdown(Instant::now() + WINDOW).unwrap();
    let receipt = manager
        .run_root_until(Instant::now() + WINDOW, &AtomicBool::new(false))
        .unwrap();
    assert!(receipt.permits_semantic_closure(), "{receipt:?}");
    assert_eq!(manager.driver().consumed.len(), 0);
    assert_eq!(facts.accepted.lock().unwrap().len(), 0);
    assert!(manager.driver().revoked);
    assert_eq!(receipt.counters.cpu_slice_receipts, 0);
    assert_eq!(receipt.pending_tickets, 0);
}

/// This fixture installs only an observer, without fabricating a Collection
/// admission. Actual tickets, ACKs, acceptance proofs and fences still come
/// from the existing native service and registry fixture.
struct RoleOnlyObserverFixture {
    events: Arc<Mutex<Vec<&'static str>>>,
    reject_accepted: bool,
}
impl AdaptiveCollectionObserver<u64, RoleOutput> for RoleOnlyObserverFixture {
    fn admitted(
        &mut self,
        _: &RoventCollectionRootAdmission,
    ) -> Result<(), AdaptiveCollectionError> {
        panic!("fixture does not issue Collection admission")
    }
    fn dispatch(
        &mut self,
        _: &RoleTicketEnvelope,
        _: &FrameInput<ReviewView, u64>,
    ) -> Result<(), AdaptiveCollectionError> {
        Ok(())
    }
    fn native_event(
        &mut self,
        _: &TicketNativeEvent<RoleOutput>,
    ) -> Result<(), AdaptiveCollectionError> {
        Ok(())
    }
    fn accepted(
        &mut self,
        accepted: &AcceptedRoleStep,
        _: &FrameInput<ReviewView, u64>,
        output: &RoleOutput,
    ) -> Result<(), AdaptiveCollectionError> {
        assert!(accepted.validates_output(output));
        self.events.lock().unwrap().push("accepted");
        if self.reject_accepted {
            return Err(AdaptiveCollectionError::new(
                AdaptiveCollectionStage::Accepted,
                "fixture",
                "injected observer failure",
            ));
        }
        Ok(())
    }
    fn root_ranking(
        &mut self,
        _: &RoventRootRankingAcceptance,
        _: &FrameInput<ReviewView, u64>,
        _: &RoleOutput,
    ) -> Result<(), AdaptiveCollectionError> {
        Ok(())
    }
    fn abandoned(&mut self, _: &RoleTicketEnvelope) -> Result<(), AdaptiveCollectionError> {
        self.events.lock().unwrap().push("abandoned");
        Ok(())
    }
    fn settled(&mut self, _: &[RoleTicketEnvelope]) -> Result<(), AdaptiveCollectionError> {
        self.events.lock().unwrap().push("settled");
        Ok(())
    }
}

#[test]
fn missing_v1_collection_admission_stops_adaptive_ranking_fanout() {
    use rz_contracts::rovent::{ExecutionMode, SearchRecipient};

    // This injects only an observer into the existing CPU native/registry
    // fixture. No Collection admission or observer ranking callback is made:
    // the failure is the missing original admission, not a fabricated proof.
    let (mut manager, facts, root, until) = fixture_with_registered_role_limit_window(
        FixtureScope::OwnCpuReciprocal,
        true,
        4096,
        None,
        None,
        Duration::from_secs(15),
    );
    manager
        .configure_rovent_reciprocal(RoventReciprocalPolicy {
            mode: ExecutionMode::CpuPcr,
            root_budget: Digest([92; 32]),
            recipient: SearchRecipient::Proposer,
            max_questions_per_job: 1,
            node_interval: 1,
            first_question_sequence: 1,
        })
        .unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    manager.collection_observer = Some(Box::new(RoleOnlyObserverFixture {
        events,
        reject_accepted: false,
    }));
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |manager| {
        manager.failures().next().is_some()
    });

    assert!(manager.failures().any(|error| matches!(error,
        AdaptiveManagerError::Collection(error)
            if error.code == "MissingRoventRootAdmission")));
    // Registry acceptance and the original root consumption happened, but
    // no candidate frame or subsequent CPU/NN work may be admitted.
    assert!(manager.root_ranking_proof.is_some());
    assert_eq!(manager.counters().submitted_role_tickets, 1);
    assert_eq!(manager.counters().consumed_role_outputs, 1);
    assert_eq!(manager.driver().consumed.len(), 1);
    assert_eq!(facts.accepted.lock().unwrap().len(), 1);
    assert_eq!(manager.machine.frames().count(), 0);
    assert!(manager.driver().actual_cpu.is_empty());

    let receipt = manager.run_until(until, &AtomicBool::new(false)).unwrap();
    assert!(receipt.permits_archive_close(), "{receipt:?}");
    assert!(facts.finish_called.load(Ordering::Acquire));
    assert_eq!(receipt.pending_tickets, 0);
}

fn finish_role_only_test_native(
    manager: RoleOnlyExecutionManager<u64, NativeFixture, RoleOnlyFixture>,
    facts: &NativeFacts,
) -> RoleOnlyFixture {
    let mut sources = manager
        .into_sources()
        .unwrap_or_else(|(_, error)| panic!("fixture closure: {error}"));
    let cleanup_until = Instant::now() + WINDOW;
    sources.native.begin_shutdown(cleanup_until);
    let mut physical_shutdown_confirmed = None;
    loop {
        while let Some(event) = sources.native.poll() {
            if let TicketNativeEvent::Finished {
                physical_shutdown_confirmed: confirmed,
            } = event
            {
                assert!(physical_shutdown_confirmed.replace(confirmed).is_none());
            }
        }
        if sources.native.try_join().unwrap() {
            while let Some(event) = sources.native.poll() {
                if let TicketNativeEvent::Finished {
                    physical_shutdown_confirmed: confirmed,
                } = event
                {
                    assert!(physical_shutdown_confirmed.replace(confirmed).is_none());
                }
            }
            break;
        }
        assert!(Instant::now() < cleanup_until);
        thread::sleep(Duration::from_millis(1));
    }
    assert!(facts.finish_called.load(Ordering::Acquire));
    assert_eq!(physical_shutdown_confirmed, Some(true));
    sources.driver
}

#[test]
fn role_only_observer_accept_failure_keeps_real_acceptance_without_search_consumption() {
    let (mut manager, facts, until) = role_only_fixture(false);
    manager.driver_mut().ready.truncate(1);
    manager.driver_mut().strict_discard = true;
    let events = Arc::new(Mutex::new(Vec::new()));
    manager
        .install_collection_observer_fixture(Box::new(RoleOnlyObserverFixture {
            events: Arc::clone(&events),
            reject_accepted: true,
        }))
        .unwrap();
    let receipt = manager
        .run_root_until(until, &AtomicBool::new(false))
        .unwrap();
    assert!(receipt.permits_semantic_closure());
    assert!(
        manager
            .failures()
            .any(|e| matches!(e, AdaptiveManagerError::Collection(_)))
    );
    assert_eq!(facts.accepted.lock().unwrap().len(), 1);
    assert_eq!(receipt.counters.consumed_role_outputs, 0);
    assert_eq!(
        *events.lock().unwrap(),
        ["accepted", "abandoned", "settled"]
    );
    let driver = finish_role_only_test_native(manager, &facts);
    assert!(driver.consumed.is_empty());
    assert_eq!(driver.discarded, 1);
}

#[test]
fn role_only_consume_error_with_positive_readback_is_not_discarded_twice() {
    let (mut manager, facts, until) = role_only_fixture(false);
    manager.driver_mut().ready.truncate(1);
    manager.driver_mut().strict_discard = true;
    manager.driver_mut().consume_fault = ConsumeFault::ErrorAfterCommit;
    let events = Arc::new(Mutex::new(Vec::new()));
    manager
        .install_collection_observer_fixture(Box::new(RoleOnlyObserverFixture {
            events: Arc::clone(&events),
            reject_accepted: false,
        }))
        .unwrap();
    let receipt = manager
        .run_root_until(until, &AtomicBool::new(false))
        .unwrap();
    assert!(receipt.permits_semantic_closure());
    assert!(manager.failures().any(|e| matches!(
        e,
        AdaptiveManagerError::Invalid("fixture error after actual consume")
    )));
    assert_eq!(facts.accepted.lock().unwrap().len(), 1);
    assert_eq!(receipt.counters.consumed_role_outputs, 1);
    assert_eq!(*events.lock().unwrap(), ["accepted", "settled"]);
    let driver = finish_role_only_test_native(manager, &facts);
    assert_eq!(driver.consumed.len(), 1);
    assert_eq!(driver.discarded, 0);
}

#[test]
fn role_only_unknown_consumption_is_retained_without_duplicate_discard() {
    let (mut manager, facts, until) = role_only_fixture(false);
    manager.driver_mut().ready.truncate(1);
    manager.driver_mut().strict_discard = true;
    manager.driver_mut().consume_fault = ConsumeFault::ReadbackUnknownAfterCommit;
    let events = Arc::new(Mutex::new(Vec::new()));
    manager
        .install_collection_observer_fixture(Box::new(RoleOnlyObserverFixture {
            events: Arc::clone(&events),
            reject_accepted: false,
        }))
        .unwrap();
    while manager.failures().next().is_none() {
        assert!(Instant::now() < until);
        manager.tick().unwrap();
        thread::sleep(Duration::from_millis(1));
    }
    let cleanup_until = Instant::now() + WINDOW;
    manager.begin_shutdown(cleanup_until).unwrap();
    loop {
        manager.tick().unwrap();
        let receipt = manager.shutdown_receipt().unwrap();
        if receipt.quarantined && receipt.native_physical_shutdown_confirmed {
            assert!(receipt.retained_outputs > 0);
            assert_eq!(receipt.counters.consumed_role_outputs, 0);
            assert_eq!(receipt.counters.unused_role_results, 0);
            break;
        }
        assert!(Instant::now() < cleanup_until);
        thread::sleep(Duration::from_millis(1));
    }
    assert!(facts.finish_called.load(Ordering::Acquire));
    assert_eq!(manager.driver().consumed.len(), 1);
    assert_eq!(manager.driver().discarded, 0);
    assert!(
        !events
            .lock()
            .unwrap()
            .iter()
            .any(|event| *event == "abandoned")
    );
    assert!(manager.failures().any(|error| matches!(
        error,
        AdaptiveManagerError::Invalid("fixture consumption readback unknown")
    )));
}

#[test]
fn role_only_driver_deadline_after_admission_abandons_without_consumption() {
    // Deterministic deadline rejection injection after manager admission;
    // this is not evidence of a measured wall-clock race.
    let (mut manager, facts, until) = role_only_fixture(false);
    manager.driver_mut().ready.truncate(1);
    manager.driver_mut().strict_discard = true;
    manager.driver_mut().consume_fault = ConsumeFault::DeadlineBeforeCommit;
    let events = Arc::new(Mutex::new(Vec::new()));
    manager
        .install_collection_observer_fixture(Box::new(RoleOnlyObserverFixture {
            events: Arc::clone(&events),
            reject_accepted: false,
        }))
        .unwrap();
    let receipt = manager
        .run_root_until(until, &AtomicBool::new(false))
        .unwrap();
    assert!(receipt.permits_semantic_closure());
    assert!(
        manager
            .failures()
            .any(|e| matches!(e, AdaptiveManagerError::Deadline))
    );
    assert_eq!(facts.accepted.lock().unwrap().len(), 1);
    assert_eq!(receipt.counters.consumed_role_outputs, 0);
    assert_eq!(
        *events.lock().unwrap(),
        ["accepted", "abandoned", "settled"]
    );
    let driver = finish_role_only_test_native(manager, &facts);
    assert!(driver.consumed.is_empty());
    assert_eq!(driver.discarded, 1);
}

#[test]
fn role_only_close_consumes_original_prepared_after_native_controls_retire() {
    let (mut manager, facts, until) = role_only_fixture(false);
    // Stop polling after both original preparations were queued. Neither is
    // admitted to a role ticket, and neither native run can start yet.
    while manager.counters().native_preparations < 2 {
        assert!(Instant::now() < until);
        manager.tick().unwrap();
        thread::sleep(Duration::from_millis(1));
    }
    while facts.prepared.load(Ordering::Acquire) < 2 {
        assert!(Instant::now() < until);
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(manager.pending_tickets(), 0);
    let cleanup_until = Instant::now() + WINDOW;
    manager.begin_root_quiesce(cleanup_until).unwrap();
    // The actual owner removes canceled unsubmitted controls before calling
    // backend.quiesce. Its original Prepared events are still waiting for the
    // manager, so re-canceling those IDs here would return UnknownPreparation.
    while !facts.quiesce_called.load(Ordering::Acquire) {
        assert!(Instant::now() < cleanup_until);
        thread::sleep(Duration::from_millis(1));
    }
    let receipt = manager
        .run_root_until(cleanup_until, &AtomicBool::new(false))
        .unwrap();
    assert!(receipt.permits_semantic_closure(), "{receipt:?}");
    assert!(
        manager.failures().next().is_none(),
        "{:?}",
        manager.failures().collect::<Vec<_>>()
    );
    assert_eq!(receipt.counters.canceled_preparations, 2);
    assert_eq!(receipt.counters.submitted_role_tickets, 0);
    assert_eq!(receipt.counters.physical_role_completions, 0);
    assert_eq!(receipt.counters.consumed_role_outputs, 0);
    assert_eq!(receipt.retained_outputs, 0);
    let mut sources = manager
        .into_sources()
        .unwrap_or_else(|(_, error)| panic!("NoR prepared cancellation physical closure: {error}"));
    assert_eq!(sources.driver.discarded, 2);
    assert_eq!(sources.driver.stores.review_pin_count().unwrap(), 0);
    sources.native.begin_shutdown(cleanup_until);
    while !sources.native.try_join().unwrap() {
        assert!(Instant::now() < cleanup_until);
        sources.native.poll();
        thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn role_only_stop_keeps_canceled_issued_acceptance_until_the_actual_fence() {
    for process_shutdown in [false, true] {
        let (mut manager, facts, until) = role_only_fixture(true);
        facts.block_after_first_run.store(true, Ordering::Release);
        // Dispatch both native runs without delivering the first completion.
        // The second run then occupies the actual owner while the first ACK
        // is enqueued. No fabricated Retired event or copied permit is used.
        while manager.pending_tickets() < 2 {
            assert!(Instant::now() < until);
            manager.tick().unwrap();
            thread::sleep(Duration::from_millis(1));
        }
        facts.release_first_run.store(true, Ordering::Release);
        while facts.started.lock().unwrap().len() < 2 {
            assert!(
                Instant::now() < until,
                "second native fixture run did not start"
            );
            thread::sleep(Duration::from_millis(1));
        }
        manager.tick().unwrap();
        assert_eq!(manager.driver().consumed.len(), 0);
        assert_eq!(facts.accepted.lock().unwrap().len(), 0);
        assert_eq!(
            manager
                .shutdown_receipt()
                .unwrap()
                .counters
                .physical_role_completions,
            1
        );

        let cleanup_until = Instant::now() + WINDOW;
        if process_shutdown {
            manager.begin_shutdown(cleanup_until).unwrap();
        } else {
            manager.begin_root_quiesce(cleanup_until).unwrap();
        }
        let receipt = manager
            .run_root_until(cleanup_until, &AtomicBool::new(false))
            .unwrap();
        assert!(receipt.permits_semantic_closure(), "{receipt:?}");
        assert!(
            manager.failures().next().is_none(),
            "{:?}",
            manager.failures().collect::<Vec<_>>()
        );
        assert_eq!(receipt.canceled_acceptance_acks, 1);
        assert_eq!(receipt.expired_acceptance_acks, 0);
        assert_eq!(receipt.counters.submitted_role_tickets, 2);
        assert_eq!(receipt.counters.accepted_nn_inputs, 0);
        assert_eq!(receipt.counters.consumed_role_outputs, 0);
        assert_eq!(receipt.counters.unused_role_results, 2);
        assert_eq!(receipt.pending_tickets, 0);
        assert_eq!(receipt.retained_outputs, 0);
        assert_eq!(manager.driver().consumed.len(), 0);
        assert_eq!(facts.accepted.lock().unwrap().len(), 0);
        let mut sources = manager
            .into_sources()
            .unwrap_or_else(|(_, error)| panic!("NoR canceled ACK physical closure: {error}"));
        assert_eq!(sources.driver.discarded, 2);
        assert_eq!(sources.driver.stores.review_pin_count().unwrap(), 0);
        sources.native.begin_shutdown(cleanup_until);
        while !sources.native.try_join().unwrap() {
            assert!(Instant::now() < cleanup_until);
            sources.native.poll();
            thread::sleep(Duration::from_millis(1));
        }
    }
}

#[test]
fn role_only_non_cancel_acceptance_rejection_is_not_hidden_by_cleanup() {
    let (mut manager, facts, until) = role_only_fixture(false);
    facts.reject_acceptance.store(true, Ordering::Release);
    let receipt = manager
        .run_root_until(until, &AtomicBool::new(false))
        .unwrap();
    assert!(receipt.permits_semantic_closure(), "{receipt:?}");
    assert!(manager.failures().any(|failure| matches!(
        failure,
        AdaptiveManagerError::Invalid("P/C native retirement/ack differs")
    )));
    assert_eq!(receipt.counters.accepted_nn_inputs, 0);
    assert_eq!(receipt.counters.consumed_role_outputs, 0);
    assert_eq!(manager.driver().consumed.len(), 0);
    assert_eq!(facts.accepted.lock().unwrap().len(), 0);
    let mut sources = manager
        .into_sources()
        .unwrap_or_else(|(_, error)| panic!("NoR rejected ACK physical closure: {error}"));
    assert_eq!(sources.driver.stores.review_pin_count().unwrap(), 0);
    sources.native.begin_shutdown(until);
    while !sources.native.try_join().unwrap() {
        assert!(Instant::now() < until);
        sources.native.poll();
        thread::sleep(Duration::from_millis(1));
    }
}
