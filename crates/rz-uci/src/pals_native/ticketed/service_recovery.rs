//! Recoverable ticket worker loop. The caller retains the original state
//! outside the unwind boundary and owns the same worker and recovery key.
use super::*;

pub(super) fn run_worker_loop<
    V: Send + Sync + 'static,
    Q: Send + 'static,
    B: TicketNativeBackend<V, Q>,
>(
    state: &mut WorkerState<V, Q, B>,
    shared: &Arc<Shared>,
    limits: TicketServiceLimits,
) -> bool {
    let WorkerState {
        backend,
        held,
        commands,
        acks,
        events,
        active,
        active_ack,
        active_prepared,
        active_output,
        active_panics,
        active_observations,
        active_ack_result,
        recovery_commands,
        recovery_root_fence,
        ..
    } = state;

    let clock = backend.clock();
    let (acceptance_issuer, acceptance_verifier) = NativeAcceptanceIssuer::new_pair();
    *shared
        .acceptance_verifier
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = Some(acceptance_verifier.clone());
    *shared.ready_clock.lock().unwrap_or_else(|e| e.into_inner()) = Some(clock.clone());
    *shared.ready_cpu.lock().unwrap_or_else(|e| e.into_inner()) =
        Some(rz_runtime::adaptive::observe_current_cpu_thread(
            rz_runtime::adaptive::CpuActor::NativeCaller,
        ));
    let mut request_high_water = 0;
    if events
        .send(TicketNativeEvent::Ready {
            clock: clock.clone(),
        })
        .is_err()
    {
        shared.closed.store(true, Ordering::Release);
    }
    loop {
        if !events.undelivered.borrow().is_empty() {
            // The observer cannot receive terminals. Preserve the queue and
            // stop issuing work; the original supervisor retrieves events.
            return false;
        }
        let observed_signal = shared.signal.version();
        for _ in 0..shared.pending_capacity {
            let Ok(ack) = acks.try_recv() else {
                break;
            };
            *active_ack = Some(ack);
            if !acknowledge_recoverable(
                backend,
                held,
                active_ack,
                events,
                shared,
                &clock,
                limits.review_token_budget,
                &acceptance_issuer,
                active_panics,
                active_observations,
                active_ack_result,
            ) {
                return false;
            }
            if !events.undelivered.borrow().is_empty() {
                return false;
            }
        }
        if !retire_canceled_recoverable(held, events, shared) {
            return false;
        }
        #[cfg(test)]
        {
            let gate = {
                let mut state = shared
                    .cancellation_scan_test
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                if state.pending.as_ref().is_some_and(|gate| {
                    held.get(&gate.preparation)
                        .is_some_and(|value| value.ticket.is_none())
                }) {
                    state.pending.take()
                } else {
                    None
                }
            };
            if let Some(gate) = gate {
                let _ = gate.entered.send(());
                let _ = gate
                    .release
                    .recv_timeout(gate.until.saturating_duration_since(Instant::now()));
            }
        }
        let until = *shared
            .shutdown_until
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(until) = until {
            // After this point no producer that observed an open recoverable
            // admission can still enqueue a command or an acknowledgement.
            shared.seal_admission();
            let report = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                backend.finish_report(until)
            })) {
                Ok(report) => report,
                Err(payload) => {
                    active_panics.push(payload);
                    TicketNativeFinishReport {
                        physical_shutdown_confirmed: false,
                        physical_failure: None,
                        semantic_failure: None,
                        session_release: None,
                        runtime_mapping_status: TicketRuntimeMappingStatus::NotObserved,
                        system_placement_status: TicketSystemPlacementStatus::NotObserved,
                    }
                }
            };
            let known = report.physical_shutdown_confirmed;
            *shared
                .finish_report_snapshot
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(report.clone());
            *shared
                .finish_report
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(report);
            if !known {
                shared.quarantine.store(true, Ordering::Release);
            } else {
                shared
                    .controls
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clear();
            }
            let _ = events.send(TicketNativeEvent::Finished {
                physical_shutdown_confirmed: known,
            });
            if known && active.is_none() {
                // Preserve an original admitted but unprocessed command in
                // state; the recovery owner can explicitly transfer it.
                *active = commands.try_recv().ok();
            }
            if known && active_ack.is_none() {
                // A producer may have passed the admission gate after this
                // loop's ack scan but before shutdown sealed it. Preserve
                // its exact permit and the remaining receiver for recovery.
                *active_ack = acks.try_recv().ok();
            }
            return known;
        }
        let root_quiesce = *shared
            .root_quiesce
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some((authority, until)) = root_quiesce {
            let cleanup_known = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                held.values().all(|value| {
                    value
                        .ticket
                        .is_none_or(|ticket| value.completed && backend.discard(&ticket).is_ok())
                })
            })) {
                Ok(known) => known,
                Err(payload) => {
                    active_panics.push(payload);
                    false
                }
            };
            let physical_known = cleanup_known
                && match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    backend.quiesce(until)
                })) {
                    Ok(known) => known,
                    Err(payload) => {
                        active_panics.push(payload);
                        false
                    }
                };
            let fence = NativeRootFence {
                authority,
                physical_completion_confirmed: physical_known,
                ticket_cleanup_confirmed: cleanup_known,
            };
            if !fence.permits_reuse() {
                shared.quarantine.store(true, Ordering::Release);
                let _ = events.send(TicketNativeEvent::RootQuiesced(fence));
                let _ = events.send(TicketNativeEvent::Finished {
                    physical_shutdown_confirmed: false,
                });
                return false;
            }
            // Keep the original queue intact until the fence is known. Once
            // the owner proves quiescence, queued commands never entered a
            // physical call and can be canceled before fence publication.
            for command in commands.try_iter() {
                let delivered = match &command {
                    Command::Prepare(id, _, _) => {
                        shared.retire(*id);
                        events
                            .send(TicketNativeEvent::PreparationFailed {
                                preparation: *id,
                                failure: RoleFailure::Canceled,
                            })
                            .is_ok()
                    }
                    Command::Execute(id, _, _) => events
                        .send(TicketNativeEvent::Rejected {
                            preparation: *id,
                            error: TicketServiceError::Closed,
                        })
                        .is_ok(),
                };
                if !delivered {
                    recovery_commands.push(super::super::recovery::TicketUndeliveredEvent::new(
                        TicketPendingCommand::from(command),
                    ));
                    // Remaining original commands stay in their channel.
                    break;
                }
            }
            if !events.undelivered.borrow().is_empty() {
                // This proves only the old root fence. The full session has
                // not finished, so preserve its owner and all held records.
                *recovery_root_fence = Some(fence);
                return false;
            }
            if !retire_canceled_recoverable(held, events, shared) {
                *recovery_root_fence = Some(fence);
                return false;
            }
            // Successful root fence can release this root's pins/host outputs,
            // while the backend, model, session and owner thread remain alive.
            held.clear();
            shared
                .controls
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clear();
            // This guard makes old-root fence publication/request removal one
            // transition against replace_root and begin_root_quiesce. A new
            // root's request cannot be overwritten by old-root cleanup.
            let _authority_transition = shared
                .root_authority
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            *shared.root_fence.lock().unwrap_or_else(|e| e.into_inner()) = Some(fence);
            *shared
                .root_quiesce
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = None;
            drop(_authority_transition);
            let _ = events.send(TicketNativeEvent::RootQuiesced(fence));
            continue;
        }
        let command = match commands.try_recv() {
            Ok(command) => command,
            Err(mpsc::TryRecvError::Empty) => {
                shared
                    .signal
                    .wait_changed(observed_signal, Duration::from_millis(50));
                continue;
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                *shared
                    .shutdown_until
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
                continue;
            }
        };
        *active = Some(command);
        if let Some(Command::Execute(id, _, true)) = active.as_ref() {
            if let Some(control) = shared
                .controls
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get_mut(id)
            {
                control.queued_executions = control
                    .queued_executions
                    .checked_sub(1)
                    .expect("accepted Execute commands have one queued registration");
            }
        }
        match active.as_ref().expect("received command retained") {
            Command::Prepare(id, question, canceled) => {
                let id = *id;
                let valid = question
                    .seal
                    .validate(question.seal.authority, limits.review_token_budget)
                    .and_then(|_| question.deadline.accepts(clock.domain(), clock.now()));
                let prepared = if valid.is_err() {
                    Err(RoleFailure::InvalidOutput)
                } else if canceled.load(Ordering::Acquire) {
                    Err(RoleFailure::Canceled)
                } else {
                    backend.prepare(question, canceled)
                };
                *active_prepared = Some(prepared);
                let accepted = match active_prepared.as_ref().expect("prepared retained") {
                    Ok((prepared, representation)) => {
                        backend
                            .prepared_host_bytes(prepared)
                            .checked_add(backend.question_host_bytes(question))
                            .is_some_and(|bytes| bytes <= limits.prepared_host_bytes_per_ticket)
                            && representation.role == question.role
                    }
                    Err(_) => false,
                };
                if accepted {
                    let Command::Prepare(_, question, canceled) =
                        active.take().expect("active retained")
                    else {
                        unreachable!()
                    };
                    let Ok((prepared, representation)) =
                        active_prepared.take().expect("prepared retained")
                    else {
                        unreachable!()
                    };
                    let seal = PreparedTicketSeal {
                        preparation: id,
                        question: question.question,
                        view: question.seal,
                        representation,
                        role: question.role,
                        original_deadline: question.deadline,
                    };
                    held.insert(
                        id,
                        Held {
                            question,
                            prepared,
                            seal,
                            canceled,
                            ticket: None,
                            output: None,
                            completed: false,
                            ack_attempted: false,
                            evidence: None,
                        },
                    );
                    let _ = events.send(TicketNativeEvent::Prepared(seal));
                } else {
                    let failure = active_prepared
                        .take()
                        .expect("prepared retained")
                        .err()
                        .unwrap_or(RoleFailure::InvalidOutput);
                    *active = None;
                    shared.retire(id);
                    let _ = events.send(TicketNativeEvent::PreparationFailed {
                        preparation: id,
                        failure,
                    });
                }
            }
            Command::Execute(id, ticket, _) => {
                let (id, ticket) = (*id, *ticket);
                let mut authority = shared
                    .root_authority
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                let error = match held.get(&id) {
                    None => Some(TicketServiceError::UnknownPreparation),
                    Some(value) if value.ticket.is_some() => Some(TicketServiceError::Duplicate),
                    Some(value)
                        if !value.seal.matches(&ticket)
                            || ticket.id.sequence <= request_high_water
                            || authority.is_some_and(|old| old != ticket.authority)
                            || ticket
                                .validate(clock.domain(), clock.now(), limits.review_token_budget)
                                .is_err() =>
                    {
                        Some(TicketServiceError::IdentityMismatch)
                    }
                    _ => None,
                };
                if let Some(error) = error {
                    let _ = events.send(TicketNativeEvent::Rejected {
                        preparation: id,
                        error,
                    });
                    *active = None;
                    continue;
                }
                authority.get_or_insert(ticket.authority);
                drop(authority);
                request_high_water = ticket.id.sequence;
                let value = held.get_mut(&id).expect("checked preparation");
                value.ticket = Some(ticket);
                let mut service_started = None;
                let result = if value.canceled.load(Ordering::Acquire) {
                    TicketPhysicalResult {
                        physical: PhysicalRoleCompletion::Confirmed,
                        output: Err(RoleFailure::Canceled),
                    }
                } else {
                    if events.send(TicketNativeEvent::Started(ticket)).is_err() {
                        return false;
                    }
                    let started_at = Instant::now();
                    service_started = Some(started_at);
                    if events
                        .send(TicketNativeEvent::OwnerRunStarted { ticket, started_at })
                        .is_err()
                    {
                        return false;
                    }
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        backend.run_owned(
                            TicketOwnerDispatchContext {
                                verifier: &acceptance_verifier,
                                ticket: &ticket,
                                question: &value.question,
                                prepared: &value.prepared,
                                canceled: &value.canceled,
                            },
                            &value.canceled,
                        )
                    })) {
                        Ok(result) => result,
                        Err(payload) => {
                            active_panics.push(payload);
                            TicketPhysicalResult {
                                physical: PhysicalRoleCompletion::Unknown,
                                output: Err(RoleFailure::PhysicalUnknown),
                            }
                        }
                    }
                };
                *active_output = Some(result);
                let physical = active_output
                    .as_ref()
                    .expect("returned result retained")
                    .physical;
                // A pre-run cancellation has no current owner invocation.
                // An unwound/unknown physical call cannot publish a complete
                // accounting observation even if its backend retained one.
                value.evidence =
                    if service_started.is_some() && physical == PhysicalRoleCompletion::Confirmed {
                        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            backend.execution_evidence(&ticket)
                        })) {
                            Ok(evidence) => evidence,
                            Err(payload) => {
                                active_panics.push(payload);
                                return false;
                            }
                        }
                    } else {
                        None
                    };
                let within_budget = match &active_output
                    .as_ref()
                    .expect("returned result retained")
                    .output
                {
                    Ok(_) if physical == PhysicalRoleCompletion::Unknown => false,
                    Ok(output) => {
                        backend.output_host_bytes(output) <= limits.output_host_bytes_per_ticket
                    }
                    Err(_) => true,
                };
                let observation_failures = if service_started.is_some() {
                    if let Err(payload) =
                        collect_observations::<V, Q, B>(backend, active_observations)
                    {
                        active_panics.push(payload);
                        return false;
                    }
                    std::mem::take(active_observations)
                } else {
                    TicketNativeObservationFailures::default()
                };
                let mut output = if physical == PhysicalRoleCompletion::Unknown {
                    // Preserve the whole original result, including an Err
                    // cause or oversized native output, in the same owner.
                    Err(RoleFailure::PhysicalUnknown)
                } else {
                    match active_output
                        .take()
                        .expect("returned result retained")
                        .output
                    {
                        Ok(output) if within_budget => {
                            let output = Arc::new(output);
                            value.output = Some(Arc::clone(&output));
                            Ok(output)
                        }
                        Ok(_) => Err(RoleFailure::InvalidOutput),
                        Err(failure) => Err(failure),
                    }
                };
                if physical == PhysicalRoleCompletion::Unknown {
                    output = Err(RoleFailure::PhysicalUnknown);
                } else if value.canceled.load(Ordering::Acquire)
                    || shared.closed.load(Ordering::Acquire)
                    || shared.root_closed.load(Ordering::Acquire)
                    || shared
                        .root_authority
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .is_some_and(|authority| authority != ticket.authority)
                {
                    output = Err(RoleFailure::Canceled);
                } else if ticket
                    .validate(clock.domain(), clock.now(), limits.review_token_budget)
                    .is_err()
                {
                    output = Err(RoleFailure::Expired);
                }
                value.completed = physical == PhysicalRoleCompletion::Confirmed;
                let _ = events.send(TicketNativeEvent::Completed {
                    completion: RoleCompletion {
                        ticket,
                        physical,
                        result: output,
                    },
                    evidence: value.evidence,
                    owner_run: TicketOwnerRunWitness {
                        ticket,
                        invoked: service_started.is_some(),
                    },
                    owner_service_elapsed: service_started.map(|started| started.elapsed()),
                    observation_failures,
                });
                if physical == PhysicalRoleCompletion::Unknown {
                    shared.quarantine.store(true, Ordering::Release);
                    let _ = events.send(TicketNativeEvent::Finished {
                        physical_shutdown_confirmed: false,
                    });
                    return false;
                }
                *active = None;
                // No logical acknowledgement barrier here. A second Fresh
                // execution begins while this immutable output awaits accept.
            }
        }
    }
}

/// Cancel an unsubmitted preparation only after its terminal was delivered.
/// Controls remain locked through this nonblocking send so an Execute cannot
/// enter between the queued check and retirement. Removed owners are dropped
/// outside that lock, and a failed send leaves the original held state intact.
fn retire_canceled_recoverable<V, Q, P, O: Send + Sync + 'static>(
    held: &mut HeldTickets<V, Q, P, O>,
    events: &EventPublisher<O>,
    shared: &Shared,
) -> bool {
    let mut controls = shared.controls.lock().unwrap_or_else(|e| e.into_inner());
    let ids: Vec<_> = held
        .iter()
        .filter_map(|(id, value)| {
            (value.ticket.is_none()
                && value.canceled.load(Ordering::Acquire)
                && controls
                    .get(id)
                    .is_none_or(|control| control.queued_executions == 0))
            .then_some(*id)
        })
        .collect();
    let mut removed = Vec::with_capacity(ids.len());
    for id in ids {
        if events
            .send(TicketNativeEvent::PreparationFailed {
                preparation: id,
                failure: RoleFailure::Canceled,
            })
            .is_err()
        {
            drop(controls);
            drop(removed);
            return false;
        }
        if let Some(value) = held.remove(&id) {
            removed.push(value);
        }
        controls.remove(&id);
    }
    drop(controls);
    drop(removed);
    true
}

/// Keep the exact linear permit in the parked state across any callback
/// unwind. Move it only after all backend callbacks have returned.
fn acknowledge_recoverable<V, Q, B: TicketNativeBackend<V, Q>>(
    backend: &mut B,
    held: &mut HeldTickets<V, Q, B::Prepared, B::Output>,
    active: &mut Option<Ack>,
    events: &EventPublisher<B::Output>,
    shared: &Shared,
    clock: &ContractSystemClock,
    token_budget: u16,
    issuer: &NativeAcceptanceIssuer,
    panics: &mut Vec<Box<dyn std::any::Any + Send>>,
    observations: &mut TicketNativeObservationFailures,
    original_result: &mut Option<Result<(), RoleFailure>>,
) -> bool {
    let (id, ticket, accepted, permit) = match active.as_ref().expect("received ack retained") {
        Ack::Accept(id, ticket, permit) => (*id, *ticket, true, permit.as_ref()),
        Ack::Discard(id, ticket) => (*id, *ticket, false, None),
    };
    let rejection = match held.get(&id) {
        None => Some(TicketServiceError::UnknownPreparation),
        Some(value) if !value.completed || value.ticket != Some(ticket) => {
            Some(TicketServiceError::IdentityMismatch)
        }
        Some(value) if accepted && value.ack_attempted => Some(TicketServiceError::Duplicate),
        _ => None,
    };
    if let Some(error) = rejection {
        let permit = match active.take().expect("received ack retained") {
            Ack::Accept(_, _, permit) => permit,
            Ack::Discard(_, _) => None,
        };
        reject_acceptance(events, id, ticket, error, permit);
        return true;
    }
    let value = held.get_mut(&id).expect("ack checked");
    if accepted {
        value.ack_attempted = true;
    }
    let mut backend_called = false;
    let result = if accepted {
        if value.canceled.load(Ordering::Acquire)
            || shared.closed.load(Ordering::Acquire)
            || shared.root_closed.load(Ordering::Acquire)
        {
            Err(RoleFailure::Canceled)
        } else if ticket
            .validate(clock.domain(), clock.now(), token_budget)
            .is_err()
        {
            Err(RoleFailure::Expired)
        } else if let Some(output) = &value.output {
            if permit.is_some_and(|permit| {
                permit.ticket() != &ticket
                    || backend.acceptance_output_digest(output) != Some(permit.output_digest())
            }) {
                Err(RoleFailure::InvalidOutput)
            } else {
                backend_called = true;
                backend.accept(&ticket, &value.question, &value.prepared, output)
            }
        } else {
            Err(RoleFailure::InvalidOutput)
        }
    } else {
        backend_called = true;
        backend.discard(&ticket)
    };
    *original_result = Some(result);
    let failures = if backend_called {
        if let Err(payload) = collect_observations::<V, Q, B>(backend, observations) {
            panics.push(payload);
            return false;
        }
        std::mem::take(observations)
    } else {
        TicketNativeObservationFailures::default()
    };
    let result = original_result
        .take()
        .expect("ack result retained through callbacks");
    let mut permit = match active.take().expect("ack retained through callbacks") {
        Ack::Accept(_, _, permit) => permit,
        Ack::Discard(_, _) => None,
    };
    let retire = !accepted || result.is_ok();
    let event = TicketNativeEvent::Retired {
        ticket,
        accepted: accepted && result.is_ok(),
        native_receipt: if accepted && result.is_ok() {
            permit.take().map(|permit| issuer.issue_success(permit))
        } else {
            None
        },
        rejected_permit: permit,
        result,
        observation_failures: failures,
    };
    let delivered = events.send(event).is_ok();
    // An unconsumed terminal stays associated with its original held ticket
    // until the quarantined owner hands the event to a recovery caller.
    if retire && delivered {
        held.remove(&id);
        shared.retire(id);
    }
    true
}

fn collect_observations<V, Q, B: TicketNativeBackend<V, Q>>(
    backend: &mut B,
    original: &mut TicketNativeObservationFailures,
) -> Result<(), Box<dyn std::any::Any + Send>> {
    for entry in &mut original.entries {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            backend.take_observation_failure()
        }))? {
            Some(failure) => *entry = Some(failure),
            None => break,
        }
    }
    Ok(())
}
