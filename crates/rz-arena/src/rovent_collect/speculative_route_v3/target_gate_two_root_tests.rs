//! Production gate transitions with test-owned locks and ticket authorities.
//! No admission, route, shared writer, consumer, or physical fence is issued.

use super::*;

struct TwoRoots {
    a_gate: Arc<Mutex<()>>,
    a_revoked: TargetRevocationV3,
    a_first_authority: SpeculativeTicketAuthorityV3,
    a_second_authority: SpeculativeTicketAuthorityV3,
    a_first: TargetGateV3,
    a_second: TargetGateV3,
    b_revoked: TargetRevocationV3,
    b_authority: SpeculativeTicketAuthorityV3,
    b_target: TargetGateV3,
}

fn two_roots() -> TwoRoots {
    let a_gate = Arc::new(Mutex::new(()));
    let a_revoked = TargetRevocationV3::new();
    let a_first_authority = SpeculativeTicketAuthorityV3::new();
    let a_second_authority = SpeculativeTicketAuthorityV3::new();
    let a_first = TargetGateV3::new(
        Arc::clone(&a_gate),
        a_revoked.clone(),
        a_first_authority.clone(),
    );
    let a_second = TargetGateV3::new(
        Arc::clone(&a_gate),
        a_revoked.clone(),
        a_second_authority.clone(),
    );
    let b_gate = Arc::new(Mutex::new(()));
    let b_revoked = TargetRevocationV3::new();
    let b_authority = SpeculativeTicketAuthorityV3::new();
    let b_target = TargetGateV3::new(Arc::clone(&b_gate), b_revoked.clone(), b_authority.clone());
    TwoRoots {
        a_gate,
        a_revoked,
        a_first_authority,
        a_second_authority,
        a_first,
        a_second,
        b_revoked,
        b_authority,
        b_target,
    }
}

#[test]
fn ticket_cancel_waits_for_gate_and_leaves_other_tickets_available() {
    let fixture = two_roots();
    let held = fixture.a_first.try_begin().unwrap().expect("A1 gate");
    assert!(matches!(
        try_cancel_ticket(&fixture.a_gate, &fixture.a_first_authority, || {}),
        Ok(None)
    ));
    assert!(!fixture.a_first_authority.canceled());
    held.known_not_written();
    assert!(matches!(
        try_cancel_ticket(&fixture.a_gate, &fixture.a_first_authority, || {}),
        Ok(Some(()))
    ));
    assert!(matches!(
        fixture.a_first.try_begin(),
        Err(TargetBeginRejectionV3::Revoked)
    ));
    assert!(!fixture.a_second_authority.canceled());
    assert!(!fixture.b_authority.canceled());
    fixture
        .a_second
        .try_begin()
        .unwrap()
        .expect("A2 gate")
        .committed();
    fixture
        .b_target
        .try_begin()
        .unwrap()
        .expect("B gate")
        .committed();
}

#[test]
fn commit_before_ticket_cancel_remains_committed_and_other_root_is_open() {
    let fixture = two_roots();
    fixture
        .a_first
        .try_begin()
        .unwrap()
        .expect("A1 gate")
        .committed();
    assert!(matches!(
        try_cancel_ticket(&fixture.a_gate, &fixture.a_first_authority, || {}),
        Ok(Some(()))
    ));
    assert!(matches!(
        fixture.a_first.try_begin(),
        Err(TargetBeginRejectionV3::AlreadyCommitted)
    ));
    assert!(!fixture.b_revoked.is_revoked());
    fixture
        .b_target
        .try_begin()
        .unwrap()
        .expect("B gate")
        .committed();
}

#[test]
fn root_revoke_waits_for_append_and_does_not_revoke_other_root() {
    let fixture = two_roots();
    let canceled = AtomicBool::new(false);
    let held = fixture.a_first.try_begin().unwrap().expect("A1 gate");
    assert!(matches!(
        try_revoke_root(&fixture.a_gate, &fixture.a_revoked, &canceled),
        Ok(None)
    ));
    assert!(!canceled.load(Ordering::Acquire));
    assert!(!fixture.a_revoked.is_revoked());
    held.known_not_written();
    assert!(matches!(
        try_revoke_root(&fixture.a_gate, &fixture.a_revoked, &canceled),
        Ok(Some(()))
    ));
    assert!(canceled.load(Ordering::Acquire));
    assert!(matches!(
        fixture.a_first.try_begin(),
        Err(TargetBeginRejectionV3::Revoked)
    ));
    assert!(matches!(
        fixture.a_second.try_begin(),
        Err(TargetBeginRejectionV3::Revoked)
    ));
    assert!(!fixture.b_revoked.is_revoked());
    fixture
        .b_target
        .try_begin()
        .unwrap()
        .expect("B gate")
        .committed();
}

#[test]
fn uncertain_target_and_poisoned_root_gate_do_not_change_other_root() {
    let fixture = two_roots();
    let held = fixture.a_first.try_begin().unwrap().expect("A1 gate");
    drop(held);
    assert!(matches!(
        fixture.a_first.try_begin(),
        Err(TargetBeginRejectionV3::Uncertain)
    ));
    fixture
        .a_second
        .try_begin()
        .unwrap()
        .expect("A2 gate")
        .known_not_written();
    let gate = Arc::clone(&fixture.a_gate);
    assert!(
        std::thread::spawn(move || {
            let _held = gate.lock().unwrap();
            panic!("test poisoned root A gate");
        })
        .join()
        .is_err()
    );
    assert!(matches!(
        fixture.a_second.try_begin(),
        Err(TargetBeginRejectionV3::Poisoned)
    ));
    assert!(!fixture.b_revoked.is_revoked());
    assert!(!fixture.b_authority.canceled());
    fixture
        .b_target
        .try_begin()
        .unwrap()
        .expect("B gate")
        .committed();
}
