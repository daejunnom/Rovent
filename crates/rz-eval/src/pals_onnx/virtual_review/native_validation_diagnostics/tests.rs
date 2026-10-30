//! 순수 합성 reducer 검사다. ORT, 모델, GPU, worker 실행 증거를 만들지 않는다.
//! 부모 진단 모듈의 `#[cfg(test)] mod tests;` 아래 파일로 배치한다.
//! 시간은 같은 origin에서 계산하며 실제 sleep이나 대국을 사용하지 않는다.
use super::*;
use crate::error::{BackendError, FailureKind, FailureStage};
use rz_contracts::pals::{RepresentationKey, Role};
use rz_contracts::pals_adaptive::ReviewViewSeal;
use rz_contracts::{
    ClockDomain, Deadline, GameGeneration, MonotonicTick, OwnerId, PrecisionProfile, ProcessEpoch,
    RequestId, RootGeneration, StateIdentity, StateRevision,
};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Duration;

const PREPARATION: u64 = 7;

fn ticket() -> RoleTicketEnvelope {
    let authority = SearchAuthority {
        epoch: ProcessEpoch(41),
        game: GameGeneration(2),
        root: RootGeneration(3),
        implementation: Digest([4; 32]),
    };
    RoleTicketEnvelope {
        id: RequestId::new(authority.epoch, 1),
        authority,
        question: Digest([11; 32]),
        view: ReviewViewSeal {
            authority,
            view_sequence: 1,
            context_policy: Digest([12; 32]),
            view_digest: Digest([13; 32]),
            root_state: StateIdentity {
                owner: OwnerId(5),
                revision: StateRevision(6),
                semantic: Digest([7; 32]),
            },
            actual_tokens: 1,
            selected_records: 0,
            anchors: 1,
            length_bucket: 8,
        },
        representation: RepresentationKey {
            input: Digest([14; 32]),
            history: Digest([15; 32]),
            candidates: Digest([16; 32]),
            mask_positions: Digest([17; 32]),
            model: Digest([21; 32]),
            encoding: Digest([18; 32]),
            precision: PrecisionProfile::Fp32,
            frozen_epoch: 9,
            record_revision: 10,
            role: Role::Proposer,
        },
        role: Role::Proposer,
        original_deadline: Deadline {
            clock: ClockDomain(authority.epoch),
            at: MonotonicTick(1_000_000),
        },
    }
}

fn at(origin: Instant, step: u64) -> Instant {
    origin
        .checked_add(Duration::from_micros(step))
        .expect("finite synthetic timestamp")
}

fn fixture() -> (
    GpuLifecycleTrace,
    RoleTicketEnvelope,
    ExecutionId,
    Vec<Event>,
) {
    let trace = GpuLifecycleTrace::new();
    let ticket = ticket();
    let execution = ExecutionId::new(ticket.authority.epoch, 17);
    let origin = trace.0.origin;
    trace.bind_actual_cuda_owner(ticket.representation.model.0, [22; 32], [23; 32], [24; 32]);
    let events = vec![
        Event::Admitted(PREPARATION, ticket, at(origin, 1)),
        Event::Enter(ticket, execution, RunGraph::Private, at(origin, 2)),
        Event::Cancel(PREPARATION, at(origin, 3), at(origin, 4), true),
        Event::Return(
            ticket,
            execution,
            RunGraph::Private,
            at(origin, 2),
            at(origin, 5),
            Some(true),
        ),
        Event::Backend(
            ticket,
            execution,
            Some(ticket.representation.input),
            None,
            true,
            Some(1),
            true,
            at(origin, 6),
        ),
        Event::Logical(ticket, true, true, at(origin, 7)),
        Event::Retired(ticket, false, true, at(origin, 8)),
        Event::Fence(ticket.authority, true, at(origin, 9)),
        Event::Placement(at(origin, 10)),
        Event::Finished(true, at(origin, 11)),
        Event::Joined(at(origin, 12)),
    ];
    (trace, ticket, execution, events)
}

fn changed_receipt(
    change: impl FnOnce(&mut Vec<Event>, RoleTicketEnvelope, ExecutionId, Instant),
) -> Value {
    let (trace, ticket, execution, mut events) = fixture();
    change(&mut events, ticket, execution, trace.0.origin);
    for event in events {
        trace.push(event);
    }
    trace.receipt()
}

fn assert_rejected(receipt: &Value, label: &str) {
    assert_ne!(receipt["status"], "Observed", "{label}: {receipt}");
    assert!(
        receipt["cases"].as_array().expect("cases array").is_empty(),
        "{label}: rejected trace must not publish a positive case: {receipt}",
    );
}

#[test]
fn complete_same_ticket_host_envelope_is_observed_without_gpu_claims() {
    let ticket = ticket();
    assert!(ticket
        .validate(ticket.original_deadline.clock, MonotonicTick(0), 128)
        .is_ok());
    let receipt = changed_receipt(|_, _, _, _| {});
    assert_eq!(receipt["status"], "Observed");
    assert_eq!(receipt["cases"].as_array().expect("cases").len(), 1);
    assert_eq!(receipt["trace_complete"], true);
    assert_eq!(receipt["native_drain_and_join_confirmed"], true);
    assert_eq!(receipt["gpu_kernel_timing_attested"], false);
    assert_eq!(receipt["numerical_acceptance"], false);
    assert_eq!(receipt["performance_evidence"], false);
    assert_eq!(receipt["new_execution_authority_issued"], false);
    assert_eq!(receipt["cases"][0]["native_retired_accepted"], false);
}

#[test]
fn cancellation_must_strictly_overlap_the_recorded_envelope() {
    for (label, before, after, first) in [
        ("before submission", 0, 0, true),
        ("before Run", 1, 1, true),
        ("at Run entry", 2, 3, true),
        ("at Run return", 3, 5, true),
        ("after Run return", 6, 7, true),
        ("reversed store bracket", 4, 3, true),
        ("already canceled", 3, 4, false),
    ] {
        let receipt = changed_receipt(|events, _, _, origin| {
            events[2] = Event::Cancel(PREPARATION, at(origin, before), at(origin, after), first);
        });
        assert_rejected(&receipt, label);
    }
}

#[test]
fn evidence_from_another_ticket_execution_or_invocation_does_not_join() {
    for mutation in 0..7 {
        let receipt = changed_receipt(|events, ticket, execution, origin| {
            let mut other_ticket = ticket;
            other_ticket.id.sequence += 1;
            let other_execution = ExecutionId::new(execution.epoch, execution.sequence + 1);
            match mutation {
                0 => {
                    events[4] = Event::Backend(
                        other_ticket,
                        execution,
                        Some(ticket.representation.input),
                        None,
                        true,
                        Some(1),
                        true,
                        at(origin, 6),
                    )
                }
                1 => {
                    events[4] = Event::Backend(
                        ticket,
                        other_execution,
                        Some(ticket.representation.input),
                        None,
                        true,
                        Some(1),
                        true,
                        at(origin, 6),
                    )
                }
                2 => {
                    events[4] = Event::Backend(
                        ticket,
                        execution,
                        Some(Digest([99; 32])),
                        None,
                        true,
                        Some(1),
                        true,
                        at(origin, 6),
                    )
                }
                3 => {
                    events[4] = Event::Backend(
                        ticket,
                        execution,
                        None,
                        None,
                        true,
                        Some(1),
                        true,
                        at(origin, 6),
                    )
                }
                4 => {
                    events[3] = Event::Return(
                        other_ticket,
                        execution,
                        RunGraph::Private,
                        at(origin, 2),
                        at(origin, 5),
                        Some(true),
                    )
                }
                5 => {
                    events[3] = Event::Return(
                        ticket,
                        other_execution,
                        RunGraph::Private,
                        at(origin, 2),
                        at(origin, 5),
                        Some(true),
                    )
                }
                6 => events[2] = Event::Cancel(PREPARATION + 1, at(origin, 3), at(origin, 4), true),
                _ => unreachable!(),
            }
        });
        assert_rejected(&receipt, &format!("lineage mutation {mutation}"));
    }
}

#[test]
fn warm_seed_and_unknown_or_zero_measurement_are_not_fresh_completion() {
    for (label, seed, fresh, count, known) in [
        ("Warm seed", Some(Digest([31; 32])), false, Some(1), true),
        (
            "seed contradicts Fresh",
            Some(Digest([31; 32])),
            true,
            Some(1),
            true,
        ),
        ("not Fresh", None, false, Some(1), true),
        ("unknown count", None, true, None, true),
        ("zero inputs", None, true, Some(0), true),
        ("incomplete measurement", None, true, Some(1), false),
    ] {
        let receipt = changed_receipt(|events, ticket, execution, origin| {
            events[4] = Event::Backend(
                ticket,
                execution,
                Some(ticket.representation.input),
                seed,
                fresh,
                count,
                known,
                at(origin, 6),
            );
        });
        assert_rejected(&receipt, label);
    }
}

#[test]
fn unknown_physical_or_unconfirmed_drain_is_never_repaired_by_later_markers() {
    for mutation in 0..6 {
        let receipt = changed_receipt(|events, ticket, execution, origin| match mutation {
            0 => events[5] = Event::Logical(ticket, false, true, at(origin, 7)),
            1 => {
                events[3] = Event::Return(
                    ticket,
                    execution,
                    RunGraph::Private,
                    at(origin, 2),
                    at(origin, 5),
                    None,
                )
            }
            2 => {
                events[3] = Event::Return(
                    ticket,
                    execution,
                    RunGraph::Private,
                    at(origin, 2),
                    at(origin, 5),
                    Some(false),
                )
            }
            3 => events[7] = Event::Fence(ticket.authority, false, at(origin, 9)),
            4 => events[9] = Event::Finished(false, at(origin, 11)),
            5 => events[6] = Event::Retired(ticket, false, false, at(origin, 8)),
            _ => unreachable!(),
        });
        assert_rejected(&receipt, &format!("unknown or failed stage {mutation}"));
    }
}

#[test]
fn late_acceptance_or_ordinary_logical_success_is_rejected() {
    for mutation in 0..3 {
        let receipt = changed_receipt(|events, ticket, _, origin| match mutation {
            0 => events[6] = Event::Retired(ticket, true, true, at(origin, 8)),
            1 => events[5] = Event::Logical(ticket, true, false, at(origin, 7)),
            2 => events.push(Event::Retired(ticket, true, true, at(origin, 8))),
            _ => unreachable!(),
        });
        assert_rejected(&receipt, &format!("late acceptance mutation {mutation}"));
    }
}

fn terminal_at(event: Event) -> Instant {
    match event {
        Event::Return(_, _, _, _, at, _)
        | Event::Backend(_, _, _, _, _, _, _, at)
        | Event::Logical(_, _, _, at)
        | Event::Retired(_, _, _, at)
        | Event::Fence(_, _, at)
        | Event::Placement(at)
        | Event::Finished(_, at)
        | Event::Joined(at) => at,
        _ => panic!("fixture terminal stage expected"),
    }
}

fn retime(event: &mut Event, at: Instant) {
    *event = match *event {
        Event::Return(t, x, g, enter, _, ok) => Event::Return(t, x, g, enter, at, ok),
        Event::Backend(t, x, key, seed, fresh, n, known, _) => {
            Event::Backend(t, x, key, seed, fresh, n, known, at)
        }
        Event::Logical(t, physical, canceled, _) => Event::Logical(t, physical, canceled, at),
        Event::Retired(t, accepted, ok, _) => Event::Retired(t, accepted, ok, at),
        Event::Fence(a, reusable, _) => Event::Fence(a, reusable, at),
        Event::Placement(_) => Event::Placement(at),
        Event::Finished(ok, _) => Event::Finished(ok, at),
        Event::Joined(_) => Event::Joined(at),
        _ => panic!("fixture terminal stage expected"),
    };
}

#[test]
fn every_adjacent_terminal_stage_must_follow_the_original_completion() {
    for pair in 0..7 {
        let receipt = changed_receipt(|events, _, _, _| {
            let left = 3 + pair;
            let right = left + 1;
            let left_at = terminal_at(events[left]);
            let right_at = terminal_at(events[right]);
            retime(&mut events[left], right_at);
            retime(&mut events[right], left_at);
        });
        assert_rejected(&receipt, &format!("terminal inversion {pair}"));
    }
}

#[test]
fn duplicate_success_or_conflicting_terminal_fact_is_rejected() {
    for index in [0, 2, 4, 5, 6, 7, 8, 9, 10] {
        let receipt = changed_receipt(|events, _, _, _| events.push(events[index]));
        assert_rejected(&receipt, &format!("duplicate terminal fact {index}"));
    }
    let receipt = changed_receipt(|events, ticket, execution, origin| {
        events.push(Event::Backend(
            ticket,
            execution,
            None,
            None,
            true,
            None,
            false,
            at(origin, 6),
        ));
    });
    assert_rejected(
        &receipt,
        "a successful backend event cannot hide an unknown event",
    );
}

#[test]
fn ort_intervals_require_exactly_one_matching_entry_and_return() {
    // source 001의 Return만 고르는 reducer 결함을 재현한다.
    // 대응 Enter 부재, 중복, 미복귀에서 성공 사실만 골라내면 안 된다.
    for mutation in 0..8 {
        let receipt = changed_receipt(|events, ticket, execution, origin| {
            let other_execution = ExecutionId::new(execution.epoch, execution.sequence + 1);
            match mutation {
                0 => {
                    events.remove(1);
                }
                1 => {
                    events[1] =
                        Event::Enter(ticket, other_execution, RunGraph::Private, at(origin, 2))
                }
                2 => events[1] = Event::Enter(ticket, execution, RunGraph::Public, at(origin, 2)),
                3 => events[1] = Event::Enter(ticket, execution, RunGraph::Private, at(origin, 1)),
                4 => events.push(events[1]),
                5 => events.push(events[3]),
                6 => events.push(Event::Enter(
                    ticket,
                    execution,
                    RunGraph::Public,
                    at(origin, 6),
                )),
                7 => events.push(Event::Return(
                    ticket,
                    execution,
                    RunGraph::Private,
                    at(origin, 2),
                    at(origin, 5),
                    Some(false),
                )),
                _ => unreachable!(),
            }
        });
        assert_rejected(
            &receipt,
            &format!("unmatched or duplicate ORT interval {mutation}"),
        );
    }
}

#[test]
fn bounded_overflow_stays_incomplete_and_never_grows_the_event_buffer() {
    let (trace, _, _, events) = fixture();
    for event in events {
        trace.push(event);
    }
    assert_eq!(trace.receipt()["status"], "Observed");
    for _ in 0..CAPACITY {
        trace.push(Event::Cancel(999, trace.0.origin, trace.0.origin, false));
    }
    assert_eq!(
        trace.0.state.lock().expect("test state").events.len(),
        CAPACITY
    );
    let receipt = trace.receipt();
    assert_eq!(receipt["status"], "TraceIncomplete");
    assert_rejected(&receipt, "overflow");
}

#[test]
fn contention_and_poison_are_sticky_incomplete_without_wait_or_panic() {
    let (trace, ticket, _, events) = fixture();
    for event in events {
        trace.push(event);
    }
    let held = trace.0.state.lock().expect("test holds diagnostic lock");
    trace.logical_completed(ticket, true, true);
    drop(held);
    trace.joined();
    assert_eq!(trace.receipt()["status"], "TraceIncomplete");

    let (poisoned, _, _, events) = fixture();
    for event in events {
        poisoned.push(event);
    }
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _held = poisoned.0.state.lock().expect("test poison guard");
        panic!("synthetic diagnostic lock poison");
    }));
    assert!(panic.is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| poisoned.joined())).is_ok());
    assert_eq!(poisoned.receipt()["status"], "TraceIncomplete");
    assert_rejected(&poisoned.receipt(), "poison");
}

#[test]
fn query_contention_loses_no_event_and_poison_is_still_sticky() {
    let (trace, ticket, _, events) = fixture();
    for event in events {
        trace.push(event);
    }
    let before = trace.receipt();
    assert_eq!(before["status"], "Observed");
    let held = trace.0.state.lock().expect("test holds diagnostic state");
    assert!(!trace.has_open_ort_run(&ticket));
    assert!(!trace.0.incomplete.load(Ordering::Acquire));
    drop(held);
    assert_eq!(trace.receipt(), before);

    let (poisoned, ticket, _, events) = fixture();
    for event in events {
        poisoned.push(event);
    }
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _held = poisoned.0.state.lock().expect("test poisons query state");
        panic!("synthetic query lock poison");
    }));
    assert!(panic.is_err());
    assert!(!poisoned.0.incomplete.load(Ordering::Acquire));
    assert!(!poisoned.has_open_ort_run(&ticket));
    assert!(poisoned.0.incomplete.load(Ordering::Acquire));
    assert_eq!(poisoned.receipt()["status"], "TraceIncomplete");
}

#[test]
fn native_primary_error_and_unreturned_scope_are_preserved() {
    // 원 Result를 그대로 반환하는 실제 삽입 형태의 작은 합성 함수다.
    // 오류의 kind/stage/detail/native 진단을 trace 결과로 덮지 않는다.
    fn retain_primary(
        trace: &GpuLifecycleTrace,
        ticket: RoleTicketEnvelope,
        execution: ExecutionId,
        primary: BackendError,
    ) -> Result<(), BackendError> {
        let _ticket = trace.enter_ticket(ticket, execution);
        let run = trace.enter_ort(RunGraph::Private);
        let result = Err(primary);
        if let Some(run) = run {
            run.returned(result.is_ok());
        }
        result
    }

    let primary = BackendError::new(
        FailureKind::BackendFailure,
        FailureStage::Backend,
        "synthetic original native error",
    )
    .with_diagnostic("SyntheticOrtRun", "original bounded diagnostic");
    for contended in [false, true] {
        let trace = GpuLifecycleTrace::new();
        let ticket = ticket();
        let execution = ExecutionId::new(ticket.authority.epoch, 1);
        let guard = contended.then(|| trace.0.state.lock().expect("test contention"));
        let observed = catch_unwind(AssertUnwindSafe(|| {
            retain_primary(&trace, ticket, execution, primary.clone())
        }))
        .expect("diagnostics must not replace the primary with a panic");
        drop(guard);
        assert_eq!(observed, Err(primary.clone()));
        if contended {
            assert!(trace.0.incomplete.load(Ordering::Acquire));
        }
    }

    let trace = GpuLifecycleTrace::new();
    let ticket = ticket();
    let scope = trace.enter_ticket(ticket, ExecutionId::new(ticket.authority.epoch, 1));
    let run = trace.enter_ort(RunGraph::Private).expect("synthetic span");
    drop(run);
    drop(scope);
    let receipt = trace.receipt();
    assert_eq!(receipt["status"], "TraceIncomplete");
    assert!(receipt["events"]
        .as_array()
        .expect("events")
        .iter()
        .any(|event| event["kind"] == "ort_return" && event["ok"].is_null()));
    assert_rejected(
        &receipt,
        "missing Run return is unknown rather than success",
    );
}
