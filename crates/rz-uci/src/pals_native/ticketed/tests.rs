use super::*;
use rz_contracts::pals::{Role, SearchAuthority};
use rz_contracts::{
    ClockDomain, GameGeneration, MonotonicTick, OwnerId, PrecisionProfile, ProcessEpoch, RequestId,
    RootGeneration, StateIdentity, StateRevision,
};

struct View {
    value: u64,
    drops: Arc<std::sync::atomic::AtomicUsize>,
}
impl Drop for View {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::AcqRel);
    }
}
struct Facts {
    started: Mutex<Vec<(u64, u64)>>,
    accepted: Mutex<Vec<u64>>,
    run_gate: AtomicBool,
    shutdown: AtomicBool,
    fail_accept: AtomicBool,
    initialized: AtomicUsize,
    quiesced: AtomicUsize,
    fixture_evidence_produced: Mutex<Vec<(RequestId, TicketExecutionEvidence)>>,
    fixture_evidence_taken: Mutex<Vec<RequestId>>,
    expected_dispatch: Mutex<
        Option<(
            NativeAcceptanceVerifier,
            NativeAcceptanceVerifier,
            Arc<View>,
        )>,
    >,
    owned_dispatch: Mutex<Vec<(RoleTicketEnvelope, u64, u64, bool, bool, bool)>>,
    run_observation: Mutex<Option<(TicketNativeObservationDisposition, RoleFailure)>>,
    accept_observation: Mutex<Option<(TicketNativeObservationDisposition, RoleFailure)>>,
    accept_failure: Mutex<Option<RoleFailure>>,
    pending_observations: Mutex<std::collections::VecDeque<TicketNativeObservationFailure>>,
}
struct Fake {
    clock: ContractSystemClock,
    facts: Arc<Facts>,
    fixture_evidence: Option<(RequestId, TicketExecutionEvidence)>,
}
impl TicketNativeBackend<View, u64> for Fake {
    type Prepared = u64;
    type Output = (u64, u64);
    fn clock(&self) -> ContractSystemClock {
        self.clock.clone()
    }
    fn prepare(
        &mut self,
        question: &TicketPreparation<View, u64>,
        _: &AtomicBool,
    ) -> Result<(u64, RepresentationKey), RoleFailure> {
        let value = question.query;
        let mut representation = key(value);
        representation.role = question.role;
        Ok((value, representation))
    }
    fn prepared_host_bytes(&self, _: &u64) -> u64 {
        8
    }
    fn run(
        &mut self,
        ticket: &RoleTicketEnvelope,
        question: &TicketPreparation<View, u64>,
        prepared: &u64,
        _: &AtomicBool,
    ) -> TicketPhysicalResult<(u64, u64)> {
        self.facts
            .started
            .lock()
            .unwrap()
            .push((ticket.id.sequence, ticket.authority.root.0));
        let timeout = Instant::now() + Duration::from_secs(2);
        while !self.facts.run_gate.load(Ordering::Acquire) && Instant::now() < timeout {
            thread::sleep(Duration::from_millis(1));
        }
        // These are deterministic fake-backend observations, never evidence of
        // actual ORT/NN execution. Only this fixture's run produces them.
        let evidence = fixture_execution_evidence(ticket, *prepared);
        if let Some((disposition, cause)) = self.facts.run_observation.lock().unwrap().take() {
            self.facts.pending_observations.lock().unwrap().push_back(
                TicketNativeObservationFailure {
                    ticket: *ticket,
                    stage: TicketNativeObservationStage::Completed,
                    disposition,
                    cause,
                },
            );
        }
        self.fixture_evidence = Some((ticket.id, evidence));
        self.facts
            .fixture_evidence_produced
            .lock()
            .unwrap()
            .push((ticket.id, evidence));
        TicketPhysicalResult {
            physical: if *prepared == 99 {
                PhysicalRoleCompletion::Unknown
            } else {
                PhysicalRoleCompletion::Confirmed
            },
            output: Ok((*prepared, question.view.value)),
        }
    }
    fn run_owned(
        &mut self,
        context: TicketOwnerDispatchContext<'_, View, u64, u64>,
        canceled: &AtomicBool,
    ) -> TicketPhysicalResult<(u64, u64)> {
        if let Some((expected, other, view)) = self.facts.expected_dispatch.lock().unwrap().as_ref()
        {
            self.facts.owned_dispatch.lock().unwrap().push((
                *context.ticket(),
                context.question().query,
                *context.prepared(),
                context.same_owner(expected),
                context.same_owner(other),
                Arc::ptr_eq(&context.question().view, view),
            ));
        }
        self.run(
            context.ticket(),
            context.question(),
            context.prepared(),
            canceled,
        )
    }
    fn output_host_bytes(&self, _: &(u64, u64)) -> u64 {
        16
    }
    fn acceptance_output_digest(&self, output: &(u64, u64)) -> Option<Digest> {
        rz_search::pals::coordinator::native_acceptance_output_digest(&proposal_head(
            None, output.0,
        ))
        .ok()
    }
    fn execution_evidence(
        &mut self,
        ticket: &RoleTicketEnvelope,
    ) -> Option<TicketExecutionEvidence> {
        if self
            .fixture_evidence
            .as_ref()
            .is_none_or(|(id, _)| *id != ticket.id)
        {
            return None;
        }
        let (id, evidence) = self.fixture_evidence.take().unwrap();
        self.facts.fixture_evidence_taken.lock().unwrap().push(id);
        Some(evidence)
    }
    fn take_observation_failure(&mut self) -> Option<TicketNativeObservationFailure> {
        self.facts.pending_observations.lock().unwrap().pop_front()
    }
    fn accept(
        &mut self,
        ticket: &RoleTicketEnvelope,
        _: &TicketPreparation<View, u64>,
        _: &u64,
        _: &(u64, u64),
    ) -> Result<(), RoleFailure> {
        self.facts.accepted.lock().unwrap().push(ticket.id.sequence);
        if let Some((disposition, cause)) = self.facts.accept_observation.lock().unwrap().take() {
            self.facts.pending_observations.lock().unwrap().push_back(
                TicketNativeObservationFailure {
                    ticket: *ticket,
                    stage: TicketNativeObservationStage::Retired,
                    disposition,
                    cause,
                },
            );
        }
        if let Some(failure) = self.facts.accept_failure.lock().unwrap().take() {
            return Err(failure);
        }
        if self.facts.fail_accept.load(Ordering::Acquire) {
            Err(RoleFailure::Backend(
                "fixture observer failed after a partial acknowledgement".into(),
            ))
        } else {
            Ok(())
        }
    }
    fn discard(&mut self, _: &RoleTicketEnvelope) -> Result<(), RoleFailure> {
        Ok(())
    }
    fn quiesce(&mut self, until: Instant) -> bool {
        self.facts.quiesced.fetch_add(1, Ordering::AcqRel);
        Instant::now() < until && !self.facts.shutdown.load(Ordering::Acquire)
    }
    fn finish(&mut self, _: Instant) -> bool {
        self.facts.shutdown.store(true, Ordering::Release);
        true
    }
}
fn fixture_execution_evidence(ticket: &RoleTicketEnvelope, value: u64) -> TicketExecutionEvidence {
    if value == 99 {
        return TicketExecutionEvidence {
            execution: Some(ExecutionId::new(ticket.id.epoch, ticket.id.sequence)),
            completed_nn_inputs: None,
            measurement_complete: false,
            physical_fresh: None,
            accounting: None,
        };
    }
    let miss = u64::from(value & 1 == 0);
    let validator = u64::from(ticket.role == Role::Validator);
    TicketExecutionEvidence {
        execution: Some(ExecutionId::new(ticket.id.epoch, ticket.id.sequence)),
        completed_nn_inputs: Some(1 + miss),
        measurement_complete: true,
        physical_fresh: Some(true),
        accounting: Some(TicketNativeAccountingEvidence {
            public_cache_hits: 1 - miss,
            public_cache_misses: miss,
            public_nn_runs_attempted: miss,
            public_nn_runs_completed: miss,
            public_nn_runs_failed_known: 0,
            role_nn_runs_attempted: 1,
            role_nn_runs_completed: 1,
            role_nn_runs_failed_known: 0,
            validator_nn_runs_attempted: validator,
            validator_nn_runs_completed: validator,
            validator_nn_runs_failed_known: 0,
        }),
    }
}
fn fixture_facts(gate: bool) -> Arc<Facts> {
    Arc::new(Facts {
        started: Mutex::new(Vec::new()),
        accepted: Mutex::new(Vec::new()),
        run_gate: AtomicBool::new(gate),
        shutdown: AtomicBool::new(false),
        fail_accept: AtomicBool::new(false),
        initialized: AtomicUsize::new(0),
        quiesced: AtomicUsize::new(0),
        fixture_evidence_produced: Mutex::new(Vec::new()),
        fixture_evidence_taken: Mutex::new(Vec::new()),
        expected_dispatch: Mutex::new(None),
        owned_dispatch: Mutex::new(Vec::new()),
        run_observation: Mutex::new(None),
        accept_observation: Mutex::new(None),
        accept_failure: Mutex::new(None),
        pending_observations: Mutex::new(std::collections::VecDeque::new()),
    })
}
fn authority(root: u64) -> SearchAuthority {
    SearchAuthority {
        epoch: ProcessEpoch(17),
        game: GameGeneration(1),
        root: RootGeneration(root),
        implementation: Digest([1; 32]),
    }
}
fn key(value: u64) -> RepresentationKey {
    RepresentationKey {
        input: Digest([value as u8; 32]),
        history: Digest([2; 32]),
        candidates: Digest([3; 32]),
        mask_positions: Digest([4; 32]),
        model: Digest([5; 32]),
        encoding: Digest([6; 32]),
        precision: PrecisionProfile::Fp32,
        frozen_epoch: 1,
        record_revision: value,
        role: Role::Proposer,
    }
}
fn question(
    value: u64,
    root: u64,
    clock: &ContractSystemClock,
    drops: Arc<std::sync::atomic::AtomicUsize>,
) -> TicketPreparation<View, u64> {
    TicketPreparation {
        question: Digest([value as u8; 32]),
        seal: ReviewViewSeal {
            authority: authority(root),
            view_sequence: value,
            context_policy: Digest([7; 32]),
            view_digest: Digest([value as u8; 32]),
            root_state: StateIdentity {
                owner: OwnerId(3),
                revision: StateRevision(value),
                semantic: Digest([8; 32]),
            },
            actual_tokens: 0,
            selected_records: 0,
            anchors: 1,
            length_bucket: 8,
        },
        role: Role::Proposer,
        deadline: Deadline {
            clock: ClockDomain(ProcessEpoch(17)),
            at: MonotonicTick(clock.now().0 + 60_000_000_000),
        },
        view: Arc::new(View {
            value: value + 100,
            drops,
        }),
        query: value,
    }
}
fn ticket(seal: PreparedTicketSeal, sequence: u64) -> RoleTicketEnvelope {
    RoleTicketEnvelope {
        id: RequestId::new(ProcessEpoch(17), sequence),
        authority: seal.view.authority,
        question: seal.question,
        view: seal.view,
        representation: seal.representation,
        role: seal.role,
        original_deadline: seal.original_deadline,
    }
}
fn proposal_head(ticket: Option<RoleTicketEnvelope>, value: u64) -> rz_contracts::pals::RoleOutput {
    rz_contracts::pals::RoleOutput {
        id: ticket.map_or(RequestId::new(ProcessEpoch(17), 1), |ticket| ticket.id),
        authority: ticket.map_or(authority(7), |ticket| ticket.authority),
        key: ticket.map_or(key(value), |ticket| ticket.representation),
        payload: rz_contracts::pals::RolePayload::Proposal {
            candidate_logits: vec![value as f32],
            wdl: rz_contracts::Wdl::try_new(0.5, 0.0, 0.5, 1e-4).unwrap(),
        },
        private_latent: vec![0.0; 16 * 384],
    }
}
fn issued_acceptance(
    service: &mut TicketedNativeService<View, u64, Fake>,
    clock: &ContractSystemClock,
    actual_head: bool,
) -> (
    PreparationId,
    RoleTicketEnvelope,
    NativeAcceptancePermit,
    rz_search::pals::coordinator::RoleTicketRegistry<View, rz_contracts::pals::RoleOutput>,
) {
    use rz_search::pals::coordinator::{RoleSubmission, RoleTicketRegistry};
    let q = question(1, 7, clock, Arc::new(AtomicUsize::new(0)));
    let view = Arc::clone(&q.view);
    let id = service.prepare(q).map_err(|error| error.0).unwrap();
    let seal = loop {
        if let TicketNativeEvent::Prepared(seal) = next(service) {
            break seal;
        }
    };
    let mut registry = RoleTicketRegistry::new(
        authority(7),
        512,
        rz_runtime::adaptive::CoordinatorPolicy::Fixed,
        1,
    )
    .unwrap();
    registry
        .register_native_acceptance_verifier(service.native_acceptance_verifier().unwrap())
        .unwrap();
    let ticket = registry
        .admit(
            RoleSubmission {
                question: seal.question,
                view: seal.view,
                representation: seal.representation,
                role: seal.role,
                original_deadline: seal.original_deadline,
            },
            view,
            clock.domain(),
            clock.now(),
        )
        .unwrap();
    assert_eq!(registry.take_dispatch().unwrap().ticket, ticket);
    registry.physical_started(&ticket).unwrap();
    service.execute(id, ticket).unwrap();
    let physical = completed(service);
    assert_eq!(*physical.result.unwrap(), (1, 101));
    // A fake backend fixture supplies the registered head. Changing that head
    // simulates a caller trying to substitute a completion before publication.
    registry
        .complete_proposer_ranking(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(proposal_head(Some(ticket), if actual_head { 1 } else { 2 })),
        })
        .unwrap();
    assert!(
        registry
            .take_ready(clock.domain(), clock.now())
            .unwrap()
            .logically_eligible
    );
    let permit = registry
        .issue_native_acceptance(&ticket, clock.domain(), clock.now())
        .unwrap();
    (id, ticket, permit, registry)
}

#[test]
fn native_owner_receipt_confirms_only_after_the_original_output_acknowledgement() {
    let (mut service, facts, clock) = spawn(true);
    let (id, ticket, permit, mut registry) = issued_acceptance(&mut service, &clock, true);
    service.accept_with_permit(id, ticket, permit).unwrap();
    let receipt = loop {
        if let TicketNativeEvent::Retired {
            accepted,
            result,
            native_receipt,
            rejected_permit,
            ..
        } = next(&service)
        {
            assert!(accepted);
            assert!(result.is_ok());
            assert!(rejected_permit.is_none());
            break native_receipt.unwrap();
        }
    };
    assert_eq!(facts.accepted.lock().unwrap().as_slice(), &[1]);
    registry
        .confirm_native_acceptance(receipt, clock.domain(), clock.now())
        .unwrap();
    assert_eq!(service.occupancy(), 0);
    finish(service);
}

#[test]
fn required_retirement_observation_does_not_rewrite_original_owner_acceptance() {
    let (mut service, facts, clock) = spawn(true);
    let (id, original, permit, mut registry) = issued_acceptance(&mut service, &clock, true);
    let cause = RoleFailure::Backend("required retirement observation failed".into());
    *facts.accept_observation.lock().unwrap() = Some((
        TicketNativeObservationDisposition::RequiredAfterPhysical,
        cause.clone(),
    ));
    service.accept_with_permit(id, original, permit).unwrap();
    let receipt = match next(&service) {
        TicketNativeEvent::Retired {
            ticket,
            accepted,
            result,
            native_receipt,
            rejected_permit,
            observation_failures,
        } => {
            assert_eq!(ticket, original);
            assert!(accepted);
            assert_eq!(result, Ok(()));
            assert!(rejected_permit.is_none());
            let reports: Vec<_> = observation_failures.iter().collect();
            assert_eq!(reports.len(), 1);
            assert_eq!(reports[0].ticket, original);
            assert_eq!(reports[0].stage, TicketNativeObservationStage::Retired);
            assert_eq!(
                reports[0].disposition,
                TicketNativeObservationDisposition::RequiredAfterPhysical
            );
            assert_eq!(reports[0].cause, cause);
            native_receipt.expect("actual successful acceptance retains original receipt")
        }
        _ => panic!("expected original retirement with observation"),
    };
    registry
        .confirm_native_acceptance(receipt, clock.domain(), clock.now())
        .unwrap();
    service.accept(id, original).unwrap();
    assert!(matches!(next(&service), TicketNativeEvent::Rejected {
        preparation, error: TicketServiceError::UnknownPreparation
    } if preparation == id));
    assert_eq!(facts.accepted.lock().unwrap().as_slice(), &[1]);
    finish(service);
}

#[test]
fn unknown_acceptance_preserves_failure_and_original_permit_without_replay() {
    let (mut service, facts, clock) = spawn(true);
    let (id, original, permit, _registry) = issued_acceptance(&mut service, &clock, true);
    *facts.accept_failure.lock().unwrap() = Some(RoleFailure::PhysicalUnknown);
    let cause = RoleFailure::Backend("retirement observation failed".into());
    *facts.accept_observation.lock().unwrap() = Some((
        TicketNativeObservationDisposition::RequiredAfterPhysical,
        cause.clone(),
    ));
    service.accept_with_permit(id, original, permit).unwrap();
    let permit = match next(&service) {
        TicketNativeEvent::Retired {
            ticket,
            accepted,
            result,
            native_receipt,
            rejected_permit,
            observation_failures,
        } => {
            assert_eq!(ticket, original);
            assert!(!accepted);
            assert_eq!(result, Err(RoleFailure::PhysicalUnknown));
            assert!(native_receipt.is_none());
            let reports: Vec<_> = observation_failures.iter().collect();
            assert_eq!(reports.len(), 1);
            assert_eq!(reports[0].ticket, original);
            assert_eq!(reports[0].stage, TicketNativeObservationStage::Retired);
            assert_eq!(reports[0].cause, cause);
            let permit = rejected_permit.expect("unconfirmed acceptance returns original permit");
            assert_eq!(permit.ticket(), &original);
            permit
        }
        _ => panic!("expected physical-unknown retirement"),
    };
    service.accept_with_permit(id, original, permit).unwrap();
    match next(&service) {
        TicketNativeEvent::AcceptanceRejected {
            preparation,
            ticket,
            error,
            permit,
        } => {
            assert_eq!(preparation, id);
            assert_eq!(ticket, original);
            assert_eq!(error, TicketServiceError::Duplicate);
            assert_eq!(permit.ticket(), &original);
        }
        _ => panic!("expected duplicate rejection retaining permit"),
    }
    assert_eq!(facts.accepted.lock().unwrap().as_slice(), &[1]);
    // This fake has no device-backed resources. Real PhysicalUnknown is not
    // resolved by this fixture's explicit discard or fake finish callback.
    service.discard(id, original).unwrap();
    assert!(matches!(
        next(&service),
        TicketNativeEvent::Retired { result: Ok(()), .. }
    ));
    finish(service);
}

#[test]
fn completed_observation_is_one_terminal_and_preserves_actual_physical_result() {
    for physical_unknown in [false, true] {
        let (mut service, facts, clock) = spawn(true);
        let value = if physical_unknown { 99 } else { 2 };
        let seal = prepare(
            &mut service,
            value,
            4,
            &clock,
            Arc::new(AtomicUsize::new(0)),
        );
        let original = ticket(seal, 1);
        let cause = RoleFailure::Backend("completed observation failed".into());
        *facts.run_observation.lock().unwrap() = Some((
            TicketNativeObservationDisposition::RequiredAfterPhysical,
            cause.clone(),
        ));
        service.execute(seal.preparation, original).unwrap();
        let (completion, reports) = loop {
            if let TicketNativeEvent::Completed {
                completion,
                observation_failures,
                ..
            } = next(&service)
            {
                break (completion, observation_failures);
            }
        };
        assert_eq!(completion.ticket, original);
        let entries: Vec<_> = reports.iter().collect();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].ticket, original);
        assert_eq!(entries[0].stage, TicketNativeObservationStage::Completed);
        assert_eq!(
            entries[0].disposition,
            TicketNativeObservationDisposition::RequiredAfterPhysical
        );
        assert_eq!(entries[0].cause, cause);
        if physical_unknown {
            assert_eq!(completion.physical, PhysicalRoleCompletion::Unknown);
            assert!(matches!(
                completion.result,
                Err(RoleFailure::PhysicalUnknown)
            ));
            assert!(matches!(
                next(&service),
                TicketNativeEvent::Finished {
                    physical_shutdown_confirmed: false
                }
            ));
            assert!(service.is_quarantined());
            drop(service);
        } else {
            assert_eq!(completion.physical, PhysicalRoleCompletion::Confirmed);
            assert_eq!(*completion.result.unwrap(), (2, 102));
            service.discard(seal.preparation, original).unwrap();
            assert!(matches!(
                next(&service),
                TicketNativeEvent::Retired { result: Ok(()), .. }
            ));
            finish(service);
        }
    }
}

#[test]
fn substituted_output_head_cannot_obtain_an_owner_receipt_or_accept_callback() {
    let (mut service, facts, clock) = spawn(true);
    let (id, ticket, permit, _) = issued_acceptance(&mut service, &clock, false);
    service.accept_with_permit(id, ticket, permit).unwrap();
    loop {
        if let TicketNativeEvent::Retired {
            accepted,
            result,
            native_receipt,
            rejected_permit,
            ..
        } = next(&service)
        {
            assert!(!accepted);
            assert_eq!(result, Err(RoleFailure::InvalidOutput));
            assert!(native_receipt.is_none());
            assert_eq!(rejected_permit.unwrap().ticket(), &ticket);
            break;
        }
    }
    assert!(facts.accepted.lock().unwrap().is_empty());
    assert_eq!(service.occupancy(), 1);
    service.discard(id, ticket).unwrap();
    loop {
        if let TicketNativeEvent::Retired {
            accepted, result, ..
        } = next(&service)
        {
            assert!(!accepted);
            assert!(result.is_ok());
            break;
        }
    }
    finish(service);
}
fn spawn(
    gate: bool,
) -> (
    TicketedNativeService<View, u64, Fake>,
    Arc<Facts>,
    ContractSystemClock,
) {
    let clock = ContractSystemClock::new(ProcessEpoch(17));
    let backend_clock = clock.clone();
    let facts = fixture_facts(gate);
    let backend_facts = Arc::clone(&facts);
    let service = TicketedNativeService::spawn(
        move || {
            backend_facts.initialized.fetch_add(1, Ordering::AcqRel);
            Ok(Fake {
                clock: backend_clock,
                facts: backend_facts,
                fixture_evidence: None,
            })
        },
        TicketServiceLimits::default(),
    )
    .unwrap();
    (service, facts, clock)
}
fn spawn_limited(
    gate: bool,
    pending_capacity: usize,
    command_capacity: usize,
) -> (
    TicketedNativeService<View, u64, Fake>,
    Arc<Facts>,
    ContractSystemClock,
) {
    let clock = ContractSystemClock::new(ProcessEpoch(17));
    let backend_clock = clock.clone();
    let facts = fixture_facts(gate);
    let backend_facts = Arc::clone(&facts);
    let service = TicketedNativeService::spawn_with_capacity(
        move || {
            backend_facts.initialized.fetch_add(1, Ordering::AcqRel);
            Ok(Fake {
                clock: backend_clock,
                facts: backend_facts,
                fixture_evidence: None,
            })
        },
        TicketServiceLimits::default(),
        pending_capacity,
        command_capacity,
    )
    .unwrap();
    (service, facts, clock)
}

#[test]
fn two_roots_reuse_one_loaded_backend_and_only_process_shutdown_joins() {
    let (mut service, facts, clock) = spawn(true);
    let drops = Arc::new(AtomicUsize::new(0));
    service.replace_root(authority(7)).unwrap();
    let first = prepare(&mut service, 1, 7, &clock, Arc::clone(&drops));
    service
        .execute(first.preparation, ticket(first, 1))
        .unwrap();
    drop(completed(&service));
    service
        .begin_root_quiesce(Instant::now() + Duration::from_secs(1))
        .unwrap();
    assert_eq!(
        service
            .prepare(question(3, 7, &clock, Arc::clone(&drops)))
            .unwrap_err()
            .0,
        TicketServiceError::Closed
    );
    let fence = loop {
        if let TicketNativeEvent::RootQuiesced(fence) = next(&service) {
            break fence;
        }
    };
    assert_eq!(fence.authority(), authority(7));
    assert!(fence.permits_reuse());
    assert_eq!(service.root_fence(), Some(fence));
    assert_eq!(service.occupancy(), 0);
    assert!(service.ready_clock().is_some());
    assert_eq!(facts.initialized.load(Ordering::Acquire), 1);
    assert!(!facts.shutdown.load(Ordering::Acquire));
    assert_eq!(drops.load(Ordering::Acquire), 2);
    assert_eq!(
        service.replace_root(authority(7)),
        Err(TicketServiceError::Closed)
    );
    service.replace_root(authority(8)).unwrap();
    let second = prepare(&mut service, 2, 8, &clock, Arc::clone(&drops));
    service
        .execute(second.preparation, ticket(second, 1))
        .unwrap();
    loop {
        if let TicketNativeEvent::Rejected { error, .. } = next(&service) {
            assert_eq!(error, TicketServiceError::IdentityMismatch);
            break;
        }
    }
    service
        .execute(second.preparation, ticket(second, 2))
        .unwrap();
    let second_output = completed(&service);
    assert_eq!(second_output.ticket.authority.root.0, 8);
    assert_eq!(*second_output.result.unwrap(), (2, 102));
    assert_eq!(facts.initialized.load(Ordering::Acquire), 1);
    assert_eq!(facts.quiesced.load(Ordering::Acquire), 1);
    finish(service);
    assert!(facts.shutdown.load(Ordering::Acquire));
}

#[test]
fn root_reuse_cannot_bypass_report_capacity_or_block_process_shutdown() {
    let (mut service, facts, _) = spawn(true);
    let whole = Instant::now() + Duration::from_secs(3);
    let mut capacity_observed = false;
    for root in 1..=REPORT_HIGH_WATER as u64 + 2 {
        match service.replace_root(authority(root)) {
            Ok(()) => {}
            Err(TicketServiceError::Capacity) => {
                capacity_observed = true;
                break;
            }
            Err(error) => panic!("unexpected root replacement: {error:?}"),
        }
        service.begin_root_quiesce(whole).unwrap();
        while service
            .root_fence()
            .is_none_or(|fence| fence.authority() != authority(root) || !fence.permits_reuse())
        {
            assert!(Instant::now() < whole);
            thread::sleep(Duration::from_millis(1));
        }
        // Deliberately no event polling: a readonly fence is not permission to
        // produce an unbounded stream of terminal reports across roots.
    }
    assert!(capacity_observed);
    service.begin_shutdown(whole);
    while !service.try_join().unwrap() {
        assert!(Instant::now() < whole);
        thread::sleep(Duration::from_millis(1));
    }
    assert!(facts.shutdown.load(Ordering::Acquire));
    assert!(!service.is_quarantined());
}
fn next(service: &TicketedNativeService<View, u64, Fake>) -> TicketNativeEvent<(u64, u64)> {
    let until = Instant::now() + Duration::from_secs(3);
    loop {
        if let Some(event) = service.poll() {
            return event;
        }
        assert!(Instant::now() < until, "bounded fixture event deadline");
        thread::sleep(Duration::from_millis(1));
    }
}
fn prepare(
    service: &mut TicketedNativeService<View, u64, Fake>,
    value: u64,
    root: u64,
    clock: &ContractSystemClock,
    drops: Arc<std::sync::atomic::AtomicUsize>,
) -> PreparedTicketSeal {
    service
        .prepare(question(value, root, clock, drops))
        .map_err(|error| error.0)
        .unwrap();
    loop {
        if let TicketNativeEvent::Prepared(seal) = next(service) {
            return seal;
        }
    }
}
fn completed(service: &TicketedNativeService<View, u64, Fake>) -> RoleCompletion<Arc<(u64, u64)>> {
    completed_with_evidence(service).0
}
type FakeCompletionWithEvidence = (
    RoleCompletion<Arc<(u64, u64)>>,
    Option<TicketExecutionEvidence>,
);
type FakeCompletionWithOwnerWitness = (
    RoleCompletion<Arc<(u64, u64)>>,
    Option<TicketExecutionEvidence>,
    TicketOwnerRunWitness,
);
fn completed_with_evidence(
    service: &TicketedNativeService<View, u64, Fake>,
) -> FakeCompletionWithEvidence {
    let (completion, evidence, _) = completed_with_owner_witness(service);
    (completion, evidence)
}
fn completed_with_owner_witness(
    service: &TicketedNativeService<View, u64, Fake>,
) -> FakeCompletionWithOwnerWitness {
    loop {
        if let TicketNativeEvent::Completed {
            completion,
            evidence,
            owner_run,
            ..
        } = next(service)
        {
            return (completion, evidence, owner_run);
        }
    }
}
fn finish(mut service: TicketedNativeService<View, u64, Fake>) {
    service.begin_shutdown(Instant::now() + Duration::from_secs(1));
    loop {
        if let TicketNativeEvent::Finished {
            physical_shutdown_confirmed,
        } = next(&service)
        {
            assert!(physical_shutdown_confirmed);
            break;
        }
    }
    let until = Instant::now() + Duration::from_secs(1);
    while !service.try_join().unwrap() {
        assert!(Instant::now() < until);
        thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn real_worker_dispatch_preserves_original_owner_ticket_and_prepared_input() {
    let (mut service, facts, clock) = spawn(true);
    let q = question(23, 7, &clock, Arc::new(AtomicUsize::new(0)));
    let original_view = Arc::clone(&q.view);
    let preparation = service.prepare(q).map_err(|error| error.0).unwrap();
    let seal = loop {
        if let TicketNativeEvent::Prepared(seal) = next(&service) {
            assert_eq!(seal.preparation, preparation);
            break seal;
        }
    };
    let expected = service.native_acceptance_verifier().unwrap();
    let (_other_issuer, other) = NativeAcceptanceIssuer::new_pair();
    *facts.expected_dispatch.lock().unwrap() = Some((expected, other, original_view));
    let original = ticket(seal, 1);
    service.execute(preparation, original).unwrap();
    let completion = completed(&service);
    assert_eq!(completion.ticket, original);
    assert_eq!(*completion.result.unwrap(), (23, 123));
    assert_eq!(
        facts.owned_dispatch.lock().unwrap().as_slice(),
        &[(original, 23, 23, true, false, true)]
    );
    service.discard(preparation, original).unwrap();
    assert!(matches!(
        next(&service),
        TicketNativeEvent::Retired { result: Ok(()), .. }
    ));
    finish(service);
}

#[test]
fn owner_context_projection_requires_the_original_view_and_question_contract() {
    let clock = ContractSystemClock::new(ProcessEpoch(17));
    let source = question(5, 7, &clock, Arc::new(AtomicUsize::new(0)));
    let seal = PreparedTicketSeal {
        preparation: PreparationId(1),
        question: source.question,
        view: source.seal,
        representation: key(5),
        role: source.role,
        original_deadline: source.deadline,
    };
    let original = ticket(seal, 1);
    let (_issuer, owner) = NativeAcceptanceIssuer::new_pair();
    let (_other_issuer, other) = NativeAcceptanceIssuer::new_pair();
    let original_prepared = 5_u64;
    let projected_prepared = 9_u64;
    let original_cancellation = Arc::new(AtomicBool::new(false));
    let projected = || TicketPreparation {
        question: source.question,
        seal: source.seal,
        role: source.role,
        deadline: source.deadline,
        view: Arc::clone(&source.view),
        query: 9_u64,
    };
    let context = || TicketOwnerDispatchContext {
        verifier: &owner,
        ticket: &original,
        question: &source,
        prepared: &original_prepared,
        canceled: &original_cancellation,
    };

    let matching = projected();
    let result = context().project(&matching, &projected_prepared).unwrap();
    assert!(result.same_owner(&owner));
    assert!(!result.same_owner(&other));
    assert_eq!(result.ticket(), &original);
    assert_eq!(*result.prepared(), projected_prepared);
    assert!(Arc::ptr_eq(&result.question().view, &source.view));
    // Projection must retain the real preparation's cancellation word, not
    // a new flag or acceptance/fence authority.
    let original_cancel = context().original_cancellation();
    let projected_cancel = result.original_cancellation();
    assert!(original_cancel.try_claim_submission());
    original_cancellation.store(true, Ordering::Release);
    assert!(projected_cancel.is_canceled());
    assert!(!projected_cancel.try_claim_submission());

    let mut changed = projected();
    changed.question = Digest([42; 32]);
    assert!(matches!(
        context().project(&changed, &projected_prepared),
        Err(RoleFailure::InvalidOutput)
    ));
    let mut changed = projected();
    changed.seal.view_sequence += 1;
    assert!(matches!(
        context().project(&changed, &projected_prepared),
        Err(RoleFailure::InvalidOutput)
    ));
    let mut changed = projected();
    changed.role = Role::Critic;
    assert!(matches!(
        context().project(&changed, &projected_prepared),
        Err(RoleFailure::InvalidOutput)
    ));
    let mut changed = projected();
    changed.deadline.at = MonotonicTick(changed.deadline.at.0 + 1);
    assert!(matches!(
        context().project(&changed, &projected_prepared),
        Err(RoleFailure::InvalidOutput)
    ));
    let mut changed = projected();
    changed.view = Arc::new(View {
        value: source.view.value,
        drops: Arc::clone(&source.view.drops),
    });
    assert!(matches!(
        context().project(&changed, &projected_prepared),
        Err(RoleFailure::InvalidOutput)
    ));
}

#[test]
fn two_fresh_pending_keep_original_view_and_complete_before_first_accept() {
    let (mut service, facts, clock) = spawn(true);
    let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let first = prepare(&mut service, 1, 4, &clock, Arc::clone(&drops));
    let second = prepare(&mut service, 2, 4, &clock, Arc::clone(&drops));
    let first_ticket = ticket(first, 1);
    let second_ticket = ticket(second, 2);
    service.execute(first.preparation, first_ticket).unwrap();
    service.execute(second.preparation, second_ticket).unwrap();
    let a = completed(&service);
    let b = completed(&service);
    assert_eq!(a.ticket, first_ticket);
    assert_eq!(b.ticket, second_ticket);
    assert_eq!(*a.result.unwrap(), (1, 101));
    assert_eq!(*b.result.unwrap(), (2, 102));
    assert_eq!(*facts.started.lock().unwrap(), [(1, 4), (2, 4)]);
    assert!(facts.accepted.lock().unwrap().is_empty());
    assert_eq!(drops.load(Ordering::Acquire), 0);
    service.accept(second.preparation, second_ticket).unwrap();
    service.accept(first.preparation, first_ticket).unwrap();
    let mut retired = 0;
    while retired < 2 {
        if let TicketNativeEvent::Retired {
            accepted, result, ..
        } = next(&service)
        {
            assert!(accepted);
            result.unwrap();
            retired += 1;
        }
    }
    assert_eq!(*facts.accepted.lock().unwrap(), [2, 1]);
    assert_eq!(service.occupancy(), 0);
    assert_eq!(drops.load(Ordering::Acquire), 2);
    finish(service);
}

#[test]
fn duplicate_and_mutated_envelope_never_reexecute_or_double_accept() {
    let (mut service, facts, clock) = spawn(true);
    let seal = prepare(
        &mut service,
        1,
        4,
        &clock,
        Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    );
    let original = ticket(seal, 1);
    let mut changed = original;
    changed.question = Digest([9; 32]);
    service.execute(seal.preparation, changed).unwrap();
    assert!(matches!(
        next(&service),
        TicketNativeEvent::Rejected {
            error: TicketServiceError::IdentityMismatch,
            ..
        }
    ));
    assert!(facts.started.lock().unwrap().is_empty());
    service.execute(seal.preparation, original).unwrap();
    completed(&service);
    service.execute(seal.preparation, original).unwrap();
    assert!(matches!(
        next(&service),
        TicketNativeEvent::Rejected {
            error: TicketServiceError::Duplicate,
            ..
        }
    ));
    service.accept(seal.preparation, original).unwrap();
    assert!(matches!(
        next(&service),
        TicketNativeEvent::Retired { accepted: true, .. }
    ));
    service.accept(seal.preparation, original).unwrap();
    assert!(matches!(
        next(&service),
        TicketNativeEvent::Rejected {
            error: TicketServiceError::UnknownPreparation,
            ..
        }
    ));
    assert_eq!(*facts.accepted.lock().unwrap(), [1]);
    finish(service);
}

#[test]
fn root_replacement_revokes_old_output_without_resetting_request_sequence() {
    let (mut service, facts, clock) = spawn(true);
    let seal = prepare(
        &mut service,
        1,
        4,
        &clock,
        Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    );
    let old = ticket(seal, 1);
    service.execute(seal.preparation, old).unwrap();
    completed(&service);
    service.replace_root(authority(5)).unwrap();
    service.accept(seal.preparation, old).unwrap();
    assert!(matches!(
        next(&service),
        TicketNativeEvent::Retired {
            accepted: false,
            result: Err(RoleFailure::Canceled),
            ..
        }
    ));
    service.discard(seal.preparation, old).unwrap();
    next(&service);
    let new = prepare(
        &mut service,
        2,
        5,
        &clock,
        Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    );
    service.execute(new.preparation, ticket(new, 1)).unwrap();
    assert!(matches!(
        next(&service),
        TicketNativeEvent::Rejected {
            error: TicketServiceError::IdentityMismatch,
            ..
        }
    ));
    service.execute(new.preparation, ticket(new, 2)).unwrap();
    assert_eq!(completed(&service).ticket.authority.root, RootGeneration(5));
    assert_eq!(*facts.started.lock().unwrap(), [(1, 4), (2, 5)]);
    finish(service);
}

#[test]
fn cancellation_does_not_release_pin_before_physical_ready_even_when_queue_full() {
    let (mut service, facts, clock) = spawn(false);
    let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let first = prepare(&mut service, 1, 4, &clock, Arc::clone(&drops));
    service
        .execute(first.preparation, ticket(first, 1))
        .unwrap();
    assert!(matches!(next(&service), TicketNativeEvent::Started(_)));
    for value in 2..=16 {
        service
            .prepare(question(value, 4, &clock, Arc::clone(&drops)))
            .map_err(|error| error.0)
            .unwrap();
    }
    let error = service
        .prepare(question(17, 4, &clock, Arc::clone(&drops)))
        .unwrap_err();
    assert_eq!(error.0, TicketServiceError::Capacity);
    drop(error.1);
    service.cancel(first.preparation).unwrap();
    assert_eq!(drops.load(Ordering::Acquire), 1); // only the never-admitted seventeenth view
    facts.run_gate.store(true, Ordering::Release);
    let output = completed(&service);
    assert_eq!(output.physical, PhysicalRoleCompletion::Confirmed);
    assert!(matches!(output.result, Err(RoleFailure::Canceled)));
    service
        .discard(first.preparation, ticket(first, 1))
        .unwrap();
    finish(service);
    assert_eq!(drops.load(Ordering::Acquire), 17);
}

#[test]
fn physical_unknown_quarantines_backend_input_output_and_view_without_acceptance() {
    let (mut service, facts, clock) = spawn(true);
    let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seal = prepare(&mut service, 99, 4, &clock, Arc::clone(&drops));
    service.execute(seal.preparation, ticket(seal, 1)).unwrap();
    let (output, evidence, owner_run) = completed_with_owner_witness(&service);
    assert_eq!(owner_run.ticket(), &output.ticket);
    assert!(owner_run.backend_run_invoked());
    assert_eq!(output.physical, PhysicalRoleCompletion::Unknown);
    assert!(matches!(output.result, Err(RoleFailure::PhysicalUnknown)));
    assert!(evidence.is_none());
    assert!(matches!(
        next(&service),
        TicketNativeEvent::Finished {
            physical_shutdown_confirmed: false
        }
    ));
    assert!(service.is_quarantined());
    assert!(facts.accepted.lock().unwrap().is_empty());
    assert_eq!(facts.fixture_evidence_produced.lock().unwrap().len(), 1);
    assert!(facts.fixture_evidence_taken.lock().unwrap().is_empty());
    drop(service);
    assert_eq!(drops.load(Ordering::Acquire), 0);
    assert!(!facts.shutdown.load(Ordering::Acquire));
}

#[test]
fn report_backpressure_reserves_terminal_space_and_shutdown_joins_without_polling() {
    let (mut service, facts, clock) = spawn(true);
    let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let until = Instant::now() + Duration::from_secs(3);
    // Invalid preparation never runs. A manager that stops draining receives
    // Capacity instead of growing reports forever or blocking the owner fence.
    while service.shared.pending_reports.load(Ordering::Acquire) < REPORT_HIGH_WATER {
        assert!(Instant::now() < until);
        match service.prepare(question(0, 4, &clock, Arc::clone(&drops))) {
            Ok(_) => {}
            Err((TicketServiceError::Capacity, rejected)) => {
                drop(rejected);
                thread::sleep(Duration::from_millis(1));
            }
            Err((error, _)) => panic!("unexpected admission failure: {error:?}"),
        }
    }
    let (error, rejected) = service
        .prepare(question(1, 4, &clock, Arc::clone(&drops)))
        .unwrap_err();
    assert_eq!(error, TicketServiceError::Capacity);
    drop(rejected);
    service.begin_shutdown(Instant::now() + Duration::from_secs(1));
    while !service.try_join().unwrap() {
        assert!(Instant::now() < until);
        thread::sleep(Duration::from_millis(1));
    }
    assert!(facts.shutdown.load(Ordering::Acquire));
    assert!(facts.started.lock().unwrap().is_empty());
    let mut finished = false;
    while let Some(event) = service.poll() {
        if let TicketNativeEvent::Finished {
            physical_shutdown_confirmed,
        } = event
        {
            assert!(physical_shutdown_confirmed);
            finished = true;
        }
    }
    assert!(finished);
    assert_eq!(service.shared.pending_reports.load(Ordering::Acquire), 0);
}

#[test]
fn failed_acknowledgement_cannot_replay_partial_side_effects() {
    let (mut service, facts, clock) = spawn(true);
    facts.fail_accept.store(true, Ordering::Release);
    let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seal = prepare(&mut service, 1, 4, &clock, Arc::clone(&drops));
    let original = ticket(seal, 1);
    service.execute(seal.preparation, original).unwrap();
    completed(&service);
    service.accept(seal.preparation, original).unwrap();
    assert!(matches!(
        next(&service),
        TicketNativeEvent::Retired {
            accepted: false,
            result: Err(_),
            ..
        }
    ));
    assert_eq!(drops.load(Ordering::Acquire), 0);
    service.accept(seal.preparation, original).unwrap();
    assert!(matches!(
        next(&service),
        TicketNativeEvent::Rejected {
            error: TicketServiceError::Duplicate,
            ..
        }
    ));
    assert_eq!(*facts.accepted.lock().unwrap(), [1]);
    service.discard(seal.preparation, original).unwrap();
    assert!(matches!(
        next(&service),
        TicketNativeEvent::Retired {
            accepted: false,
            result: Ok(()),
            ..
        }
    ));
    assert_eq!(drops.load(Ordering::Acquire), 1);
    finish(service);
}

#[test]
fn fixture_accounting_keeps_two_pending_ticket_observations_through_reverse_ack() {
    let (mut service, facts, clock) = spawn(true);
    let first = prepare(&mut service, 1, 4, &clock, Arc::new(AtomicUsize::new(0)));
    let mut second_question = question(2, 4, &clock, Arc::new(AtomicUsize::new(0)));
    second_question.role = Role::Validator;
    service
        .prepare(second_question)
        .map_err(|error| error.0)
        .unwrap();
    let second = loop {
        if let TicketNativeEvent::Prepared(seal) = next(&service) {
            break seal;
        }
    };
    let a = ticket(first, 1);
    let b = ticket(second, 2);
    service.execute(first.preparation, a).unwrap();
    service.execute(second.preparation, b).unwrap();
    let (first_completion, first_evidence) = completed_with_evidence(&service);
    let (second_completion, second_evidence) = completed_with_evidence(&service);
    assert_eq!(first_completion.ticket, a);
    assert_eq!(second_completion.ticket, b);
    assert!(first_completion.result.is_ok());
    assert!(second_completion.result.is_ok());
    let first_evidence = first_evidence.unwrap();
    let second_evidence = second_evidence.unwrap();
    assert_eq!(first_evidence, fixture_execution_evidence(&a, 1));
    assert_eq!(second_evidence, fixture_execution_evidence(&b, 2));
    assert_eq!(first_evidence.completed_nn_inputs, Some(1));
    assert_eq!(second_evidence.completed_nn_inputs, Some(2));
    let first_accounting = first_evidence.accounting.unwrap();
    let second_accounting = second_evidence.accounting.unwrap();
    assert_eq!(first_accounting.public_cache_hits(), 1);
    assert_eq!(first_accounting.public_cache_misses(), 0);
    assert_eq!(first_accounting.public_nn_runs_attempted(), 0);
    assert_eq!(first_accounting.public_nn_runs_completed(), 0);
    assert_eq!(first_accounting.public_nn_runs_failed_known(), 0);
    assert_eq!(first_accounting.validator_nn_runs_attempted(), 0);
    assert_eq!(second_accounting.public_cache_hits(), 0);
    assert_eq!(second_accounting.public_cache_misses(), 1);
    assert_eq!(second_accounting.public_nn_runs_attempted(), 1);
    assert_eq!(second_accounting.public_nn_runs_completed(), 1);
    assert_eq!(second_accounting.public_nn_runs_failed_known(), 0);
    assert_eq!(second_accounting.role_nn_runs_attempted(), 1);
    assert_eq!(second_accounting.role_nn_runs_completed(), 1);
    assert_eq!(second_accounting.role_nn_runs_failed_known(), 0);
    assert_eq!(second_accounting.validator_nn_runs_attempted(), 1);
    assert_eq!(second_accounting.validator_nn_runs_completed(), 1);
    assert_eq!(second_accounting.validator_nn_runs_failed_known(), 0);
    assert!(facts.accepted.lock().unwrap().is_empty());
    assert_eq!(
        facts.fixture_evidence_taken.lock().unwrap().as_slice(),
        &[a.id, b.id]
    );
    service.accept(second.preparation, b).unwrap();
    service.accept(first.preparation, a).unwrap();
    for expected in [b, a] {
        assert!(matches!(
            next(&service),
            TicketNativeEvent::Retired { ticket, accepted: true, result: Ok(()), .. }
                if ticket == expected
        ));
    }
    assert_eq!(facts.accepted.lock().unwrap().as_slice(), &[2, 1]);
    assert_eq!(facts.fixture_evidence_produced.lock().unwrap().len(), 2);
    assert_eq!(facts.fixture_evidence_taken.lock().unwrap().len(), 2);
    finish(service);
}

#[test]
fn fixture_accounting_preserves_completed_work_after_cancel_without_acceptance() {
    let (mut service, facts, clock) = spawn(false);
    let seal = prepare(&mut service, 2, 4, &clock, Arc::new(AtomicUsize::new(0)));
    let original = ticket(seal, 1);
    service.execute(seal.preparation, original).unwrap();
    assert!(matches!(next(&service), TicketNativeEvent::Started(ticket) if ticket == original));
    service.cancel(seal.preparation).unwrap();
    facts.run_gate.store(true, Ordering::Release);
    let (completion, evidence, owner_run) = completed_with_owner_witness(&service);
    assert_eq!(owner_run.ticket(), &original);
    assert!(owner_run.backend_run_invoked());
    assert_eq!(completion.physical, PhysicalRoleCompletion::Confirmed);
    assert!(matches!(completion.result, Err(RoleFailure::Canceled)));
    assert_eq!(evidence, Some(fixture_execution_evidence(&original, 2)));
    service.accept(seal.preparation, original).unwrap();
    assert!(matches!(
        next(&service),
        TicketNativeEvent::Retired {
            accepted: false,
            result: Err(RoleFailure::Canceled),
            native_receipt: None,
            ..
        }
    ));
    assert!(facts.accepted.lock().unwrap().is_empty());
    assert_eq!(facts.fixture_evidence_produced.lock().unwrap().len(), 1);
    assert_eq!(
        facts.fixture_evidence_taken.lock().unwrap().as_slice(),
        &[original.id]
    );
    service.discard(seal.preparation, original).unwrap();
    assert!(matches!(
        next(&service),
        TicketNativeEvent::Retired {
            accepted: false,
            result: Ok(()),
            ..
        }
    ));
    finish(service);
}

#[test]
fn fixture_accounting_is_not_created_or_taken_for_a_pre_run_cancellation() {
    let (mut service, facts, clock) = spawn(false);
    let first = prepare(&mut service, 1, 4, &clock, Arc::new(AtomicUsize::new(0)));
    let second = prepare(&mut service, 2, 4, &clock, Arc::new(AtomicUsize::new(0)));
    let a = ticket(first, 1);
    let b = ticket(second, 2);
    service.execute(first.preparation, a).unwrap();
    assert!(matches!(next(&service), TicketNativeEvent::Started(ticket) if ticket == a));
    service.execute(second.preparation, b).unwrap();
    service.cancel(second.preparation).unwrap();
    assert_eq!(
        service
            .shared
            .controls
            .lock()
            .unwrap()
            .get(&second.preparation)
            .unwrap()
            .queued_executions,
        1,
        "admitted Execute stays submitted while the first run owns the worker"
    );
    facts.run_gate.store(true, Ordering::Release);
    let (first_completion, first_evidence, first_owner_run) =
        completed_with_owner_witness(&service);
    let (second_completion, second_evidence, second_owner_run) =
        completed_with_owner_witness(&service);
    assert_eq!(first_owner_run.ticket(), &a);
    assert!(first_owner_run.backend_run_invoked());
    assert_eq!(second_owner_run.ticket(), &b);
    assert!(!second_owner_run.backend_run_invoked());
    assert_eq!(first_completion.ticket, a);
    assert!(first_completion.result.is_ok());
    assert_eq!(first_evidence, Some(fixture_execution_evidence(&a, 1)));
    assert_eq!(second_completion.ticket, b);
    assert!(matches!(
        second_completion.result,
        Err(RoleFailure::Canceled)
    ));
    assert_eq!(
        second_completion.physical,
        PhysicalRoleCompletion::Confirmed
    );
    assert!(second_evidence.is_none());
    assert_eq!(facts.started.lock().unwrap().as_slice(), &[(1, 4)]);
    assert_eq!(facts.fixture_evidence_produced.lock().unwrap().len(), 1);
    assert_eq!(
        facts.fixture_evidence_taken.lock().unwrap().as_slice(),
        &[a.id]
    );
    for (seal, ticket) in [(first, a), (second, b)] {
        service.discard(seal.preparation, ticket).unwrap();
        assert!(matches!(
            next(&service),
            TicketNativeEvent::Retired {
                accepted: false,
                result: Ok(()),
                ..
            }
        ));
    }
    finish(service);
}

#[test]
fn prepared_cancellation_without_execute_retires_without_owner_run_or_accounting() {
    let (mut service, facts, clock) = spawn(true);
    let drops = Arc::new(AtomicUsize::new(0));
    let seal = prepare(&mut service, 1, 4, &clock, Arc::clone(&drops));
    service.cancel(seal.preparation).unwrap();
    assert!(matches!(
        next(&service),
        TicketNativeEvent::PreparationFailed {
            preparation,
            failure: RoleFailure::Canceled,
        } if preparation == seal.preparation
    ));
    assert_eq!(service.occupancy(), 0);
    assert_eq!(drops.load(Ordering::Acquire), 1);
    assert!(facts.started.lock().unwrap().is_empty());
    assert!(facts.fixture_evidence_produced.lock().unwrap().is_empty());
    assert!(facts.fixture_evidence_taken.lock().unwrap().is_empty());
    finish(service);
}

#[test]
fn fixture_accounting_wrong_request_cannot_consume_the_original_take_once_evidence() {
    let clock = ContractSystemClock::new(ProcessEpoch(17));
    let facts = fixture_facts(true);
    let q = question(1, 4, &clock, Arc::new(AtomicUsize::new(0)));
    let seal = PreparedTicketSeal {
        preparation: PreparationId(1),
        question: q.question,
        view: q.seal,
        representation: key(1),
        role: q.role,
        original_deadline: q.deadline,
    };
    let original = ticket(seal, 1);
    let evidence = fixture_execution_evidence(&original, 1);
    let mut backend = Fake {
        clock,
        facts: Arc::clone(&facts),
        fixture_evidence: Some((original.id, evidence)),
    };
    for wrong in [
        RoleTicketEnvelope {
            id: RequestId::new(ProcessEpoch(17), 2),
            ..original
        },
        RoleTicketEnvelope {
            id: RequestId::new(ProcessEpoch(18), 1),
            ..original
        },
    ] {
        assert!(backend.execution_evidence(&wrong).is_none());
        assert_eq!(backend.fixture_evidence, Some((original.id, evidence)));
        assert!(facts.fixture_evidence_taken.lock().unwrap().is_empty());
    }
    assert_eq!(backend.execution_evidence(&original), Some(evidence));
    assert!(backend.execution_evidence(&original).is_none());
    assert_eq!(
        facts.fixture_evidence_taken.lock().unwrap().as_slice(),
        &[original.id]
    );
}

#[test]
fn fixture_accounting_failed_and_duplicate_ack_never_recharge_completion() {
    let (mut service, facts, clock) = spawn(true);
    facts.fail_accept.store(true, Ordering::Release);
    let seal = prepare(&mut service, 2, 4, &clock, Arc::new(AtomicUsize::new(0)));
    let original = ticket(seal, 1);
    service.execute(seal.preparation, original).unwrap();
    let (completion, evidence) = completed_with_evidence(&service);
    assert!(completion.result.is_ok());
    assert_eq!(evidence, Some(fixture_execution_evidence(&original, 2)));
    service.accept(seal.preparation, original).unwrap();
    assert!(matches!(
        next(&service),
        TicketNativeEvent::Retired {
            accepted: false,
            result: Err(_),
            native_receipt: None,
            ..
        }
    ));
    facts.fail_accept.store(false, Ordering::Release);
    service.accept(seal.preparation, original).unwrap();
    assert!(matches!(
        next(&service),
        TicketNativeEvent::Rejected {
            error: TicketServiceError::Duplicate,
            ..
        }
    ));
    assert_eq!(facts.accepted.lock().unwrap().as_slice(), &[1]);
    assert_eq!(
        facts.fixture_evidence_produced.lock().unwrap().as_slice(),
        &[(original.id, evidence.unwrap())]
    );
    assert_eq!(
        facts.fixture_evidence_taken.lock().unwrap().as_slice(),
        &[original.id]
    );
    service.discard(seal.preparation, original).unwrap();
    assert!(matches!(
        next(&service),
        TicketNativeEvent::Retired {
            accepted: false,
            result: Ok(()),
            ..
        }
    ));
    finish(service);
    assert_eq!(facts.fixture_evidence_produced.lock().unwrap().len(), 1);
    assert_eq!(facts.fixture_evidence_taken.lock().unwrap().len(), 1);
}

#[test]
fn owner_run_witness_for_an_original_ticket_never_authorizes_a_substituted_ack() {
    let (mut service, facts, clock) = spawn(true);
    let seal = prepare(&mut service, 2, 4, &clock, Arc::new(AtomicUsize::new(0)));
    let original = ticket(seal, 1);
    service.execute(seal.preparation, original).unwrap();
    let (completion, evidence, owner_run) = completed_with_owner_witness(&service);
    assert_eq!(completion.ticket, original);
    assert!(completion.result.is_ok());
    assert_eq!(owner_run.ticket(), &original);
    assert!(owner_run.backend_run_invoked());
    assert_eq!(evidence, Some(fixture_execution_evidence(&original, 2)));
    // A run witness describes the owner's original call. It is neither a
    // native acceptance receipt nor permission to substitute another ticket.
    assert!(facts.accepted.lock().unwrap().is_empty());
    let substituted = RoleTicketEnvelope {
        id: RequestId::new(original.id.epoch, original.id.sequence + 1),
        ..original
    };
    service.accept(seal.preparation, substituted).unwrap();
    assert!(matches!(
        next(&service),
        TicketNativeEvent::Rejected {
            error: TicketServiceError::IdentityMismatch,
            ..
        }
    ));
    assert_eq!(service.occupancy(), 1);
    assert!(facts.accepted.lock().unwrap().is_empty());
    assert_eq!(facts.fixture_evidence_produced.lock().unwrap().len(), 1);
    assert_eq!(facts.fixture_evidence_taken.lock().unwrap().len(), 1);
    assert_eq!(owner_run.ticket(), &original);
    service.accept(seal.preparation, original).unwrap();
    assert!(matches!(
        next(&service),
        TicketNativeEvent::Retired {
            accepted: true,
            result: Ok(()),
            ..
        }
    ));
    assert_eq!(
        facts.accepted.lock().unwrap().as_slice(),
        &[original.id.sequence]
    );
    assert_eq!(facts.fixture_evidence_produced.lock().unwrap().len(), 1);
    assert_eq!(facts.fixture_evidence_taken.lock().unwrap().len(), 1);
    finish(service);
}

#[test]
fn explicit_capacity_rejects_invalid_limits_before_factory_start() {
    for (pending_capacity, command_capacity) in [(0, 1), (17, 1), (1, 0), (1, 9)] {
        let clock = ContractSystemClock::new(ProcessEpoch(17));
        let facts = fixture_facts(true);
        let backend_facts = Arc::clone(&facts);
        let result = TicketedNativeService::<View, u64, Fake>::spawn_with_capacity(
            move || {
                backend_facts.initialized.fetch_add(1, Ordering::AcqRel);
                Ok(Fake {
                    clock,
                    facts: backend_facts,
                    fixture_evidence: None,
                })
            },
            TicketServiceLimits::default(),
            pending_capacity,
            command_capacity,
        );
        assert!(matches!(result, Err(TicketServiceError::Capacity)));
        assert_eq!(facts.initialized.load(Ordering::Acquire), 0);
    }
}

#[test]
fn legacy_spawn_preserves_capacity_and_explicit_pending_slot_lasts_until_retirement() {
    let (legacy, _, _) = spawn(true);
    assert_eq!(legacy.pending_capacity(), 16);
    assert_eq!(legacy.command_capacity(), 16);
    finish(legacy);

    let (mut service, facts, clock) = spawn_limited(false, 1, 1);
    assert_eq!(service.pending_capacity(), 1);
    assert_eq!(service.command_capacity(), 1);
    let drops = Arc::new(AtomicUsize::new(0));
    let first = prepare(&mut service, 1, 4, &clock, Arc::clone(&drops));
    let original = ticket(first, 1);
    // Encoding/preparation has finished, but it still owns the single slot.
    let (error, returned) = service
        .prepare(question(2, 4, &clock, Arc::clone(&drops)))
        .unwrap_err();
    assert_eq!(error, TicketServiceError::Capacity);
    assert_eq!(returned.query, 2);
    assert_eq!(returned.view.value, 102);
    assert_eq!(service.occupancy(), 1);
    drop(returned);

    service.execute(first.preparation, original).unwrap();
    assert!(matches!(next(&service), TicketNativeEvent::Started(ticket) if ticket == original));
    // A running owner also keeps its pending reservation.
    let (error, returned) = service
        .prepare(question(3, 4, &clock, Arc::clone(&drops)))
        .unwrap_err();
    assert_eq!(error, TicketServiceError::Capacity);
    assert_eq!(service.occupancy(), 1);
    drop(returned);
    facts.run_gate.store(true, Ordering::Release);
    let completion = completed(&service);
    assert!(completion.result.is_ok());
    // Physical completion alone does not retire the immutable output's slot.
    let (error, returned) = service
        .prepare(question(4, 4, &clock, Arc::clone(&drops)))
        .unwrap_err();
    assert_eq!(error, TicketServiceError::Capacity);
    assert_eq!(service.occupancy(), 1);
    drop(returned);
    service.accept(first.preparation, original).unwrap();
    assert!(matches!(
        next(&service),
        TicketNativeEvent::Retired {
            accepted: true,
            result: Ok(()),
            ..
        }
    ));
    assert_eq!(service.occupancy(), 0);

    let second = prepare(&mut service, 2, 4, &clock, Arc::clone(&drops));
    let next_ticket = ticket(second, 2);
    service.execute(second.preparation, next_ticket).unwrap();
    assert!(completed(&service).result.is_ok());
    service.discard(second.preparation, next_ticket).unwrap();
    assert!(matches!(
        next(&service),
        TicketNativeEvent::Retired {
            accepted: false,
            result: Ok(()),
            ..
        }
    ));
    assert_eq!(service.occupancy(), 0);
    assert_eq!(facts.initialized.load(Ordering::Acquire), 1);
    finish(service);
}

#[test]
fn command_capacity_is_actual_and_full_prepare_rolls_back_without_blocking_cancel() {
    let clock = ContractSystemClock::new(ProcessEpoch(17));
    let backend_clock = clock.clone();
    let facts = fixture_facts(true);
    let backend_facts = Arc::clone(&facts);
    let factory_gate = Arc::new(AtomicBool::new(false));
    let owner_gate = Arc::clone(&factory_gate);
    let factory_entered = Arc::new(AtomicBool::new(false));
    let owner_entered = Arc::clone(&factory_entered);
    let mut service = TicketedNativeService::<View, u64, Fake>::spawn_with_capacity(
        move || {
            backend_facts.initialized.fetch_add(1, Ordering::AcqRel);
            owner_entered.store(true, Ordering::Release);
            let until = Instant::now() + Duration::from_secs(5);
            while !owner_gate.load(Ordering::Acquire) {
                if Instant::now() >= until {
                    return Err(RoleFailure::Backend(
                        "bounded fixture factory gate expired".into(),
                    ));
                }
                thread::sleep(Duration::from_millis(1));
            }
            Ok(Fake {
                clock: backend_clock,
                facts: backend_facts,
                fixture_evidence: None,
            })
        },
        TicketServiceLimits::default(),
        2,
        1,
    )
    .unwrap();
    let until = Instant::now() + Duration::from_secs(3);
    while !factory_entered.load(Ordering::Acquire) {
        assert!(
            Instant::now() < until,
            "bounded fixture factory-start deadline"
        );
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(service.pending_capacity(), 2);
    assert_eq!(service.command_capacity(), 1);
    let drops = Arc::new(AtomicUsize::new(0));
    let first = service
        .prepare(question(1, 4, &clock, Arc::clone(&drops)))
        .map_err(|error| error.0)
        .unwrap();
    assert_eq!(service.occupancy(), 1);
    let (error, returned) = service
        .prepare(question(2, 4, &clock, Arc::clone(&drops)))
        .unwrap_err();
    assert_eq!(error, TicketServiceError::Capacity);
    assert_eq!(returned.query, 2);
    assert_eq!(returned.view.value, 102);
    assert_eq!(drops.load(Ordering::Acquire), 0);
    // Pending capacity is two. Only the actual one-command queue caused this
    // rejection, and its temporary control reservation has been rolled back.
    assert_eq!(service.occupancy(), 1);
    service.cancel(first).unwrap();
    assert!(
        service
            .shared
            .controls
            .lock()
            .unwrap()
            .get(&first)
            .unwrap()
            .canceled
            .load(Ordering::Acquire)
    );
    assert!(!factory_gate.load(Ordering::Acquire));
    assert!(facts.started.lock().unwrap().is_empty());
    factory_gate.store(true, Ordering::Release);
    loop {
        if let TicketNativeEvent::PreparationFailed {
            preparation,
            failure,
        } = next(&service)
        {
            assert_eq!(preparation, first);
            assert_eq!(failure, RoleFailure::Canceled);
            break;
        }
    }
    assert_eq!(service.occupancy(), 0);
    assert_eq!(drops.load(Ordering::Acquire), 1);
    // The rejected original question is still usable after the queue drains.
    let second_id = service.prepare(returned).map_err(|error| error.0).unwrap();
    let second = loop {
        if let TicketNativeEvent::Prepared(seal) = next(&service) {
            assert_eq!(seal.preparation, second_id);
            break seal;
        }
    };
    let original = ticket(second, 1);
    service.execute(second_id, original).unwrap();
    assert!(completed(&service).result.is_ok());
    service.discard(second_id, original).unwrap();
    assert!(matches!(
        next(&service),
        TicketNativeEvent::Retired {
            accepted: false,
            result: Ok(()),
            ..
        }
    ));
    assert_eq!(service.occupancy(), 0);
    assert_eq!(facts.initialized.load(Ordering::Acquire), 1);
    assert_eq!(facts.fixture_evidence_produced.lock().unwrap().len(), 1);
    finish(service);
}
