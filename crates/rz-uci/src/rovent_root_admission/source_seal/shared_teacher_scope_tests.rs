//! Source-only tests: two materializations must share the real Start's claim.
//! This module is nested under `source_seal::pure_tests` to reuse its original
//! UCI Start, V2 lock and Rules fixtures; it starts no process or neural worker.

use super::*;

#[test]
fn same_original_start_two_v2_admissions_share_one_teacher_claim() {
    let position = Position::startpos();
    let clock = ProcessClock::new(ProcessEpoch(7711));
    let original = context(&clock, false);
    let owner = input_lock_v2(&position);
    let materialize = || PreparedSourceV2::prepare(
        &owner,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &original,
        &clock,
    ).unwrap().try_into_live().unwrap();

    let first = materialize();
    let second = materialize();
    assert!(first.claim_teacher_scope_once());
    assert!(!second.claim_teacher_scope_once());
    assert!(!first.claim_teacher_scope_once());
}

#[test]
fn distinct_original_start_has_an_independent_teacher_claim() {
    let position = Position::startpos();
    let clock = ProcessClock::new(ProcessEpoch(7712));
    let first_start = context(&clock, false);
    // A second call to the real Start fixture makes a distinct original Start,
    // even though this source-only fixture uses the same visible authority.
    let second_start = context(&clock, false);
    let owner = input_lock_v2(&position);
    let materialize = |start: &SearchSessionContext| PreparedSourceV2::prepare(
        &owner,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        start,
        &clock,
    ).unwrap().try_into_live().unwrap();

    let first = materialize(&first_start);
    let second = materialize(&second_start);
    assert!(first.claim_teacher_scope_once());
    assert!(second.claim_teacher_scope_once());
    assert!(!first.claim_teacher_scope_once());
    assert!(!second.claim_teacher_scope_once());
}
