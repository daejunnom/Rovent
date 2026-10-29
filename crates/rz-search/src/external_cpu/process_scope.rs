//! Bounded Collection process ownership; this is neither go nor root authority.
//!
//! The legacy singleton remains exclusive with one Collection scope. Only the
//! registered members of that same scope may hold multiple idle processes.
//! Closed scope metadata can survive physical release; a quarantined process
//! keeps the original global gate occupied, just as the legacy owner did.

use super::collection_teacher::TeacherWorkerRegistration;
use super::{CheckerError, error};
use rz_contracts::Digest;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, OnceLock};

const UNISSUED: u8 = 0;
const ISSUED: u8 = 1;
#[cfg(any(target_os = "linux", test))]
const ACQUIRED: u8 = 2;
const CLOSED: u8 = 3;
#[cfg(any(target_os = "linux", test))]
const QUARANTINED: u8 = 4;
const MAX_MEMBERS: usize = 16;

#[derive(Debug, Default)]
struct Gate {
    occupied: AtomicBool,
}
fn global_gate() -> Arc<Gate> {
    static GATE: OnceLock<Arc<Gate>> = OnceLock::new();
    Arc::clone(GATE.get_or_init(|| Arc::new(Gate::default())))
}

#[derive(Debug)]
struct GlobalLease {
    gate: Arc<Gate>,
    quarantined: AtomicBool,
}
impl GlobalLease {
    fn acquire(gate: Arc<Gate>) -> Result<Self, CheckerError> {
        gate.occupied
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| error("spawn", "process_slot_unavailable"))?;
        Ok(Self {
            gate,
            quarantined: AtomicBool::new(false),
        })
    }
    #[cfg(any(target_os = "linux", test))]
    fn quarantine(&self) {
        self.quarantined.store(true, Ordering::Release);
    }
}
impl Drop for GlobalLease {
    fn drop(&mut self) {
        if !self.quarantined.load(Ordering::Acquire) {
            self.gate.occupied.store(false, Ordering::Release);
        }
    }
}

#[derive(Debug)]
struct Member {
    registration: TeacherWorkerRegistration,
    state: AtomicU8,
}
#[derive(Debug)]
struct Inner {
    manifest: Digest,
    members: Vec<Member>,
    closed: AtomicBool,
    _global: GlobalLease,
}

/// One original Collection session owns this issuer. It must close/take this
/// issuer at shutdown; worker permits then retain the gate until their real
/// physical owners retire. This value is intentionally not Clone/Deserialize.
#[derive(Debug)]
pub struct CollectionProcessScope {
    inner: Arc<Inner>,
}
impl CollectionProcessScope {
    /// Admission requires an unshared, unused issuer for exactly one root.
    /// Copied digests alone are not a permit; the issuer remains an opaque
    /// object and later permit checks compare its actual Arc identity.
    pub fn is_unissued_for_root(
        &self,
        authority: rz_contracts::pals::SearchAuthority,
        deadline: rz_contracts::Deadline,
        input_lock: Digest,
        root_admission: Digest,
        root_budget: Digest,
    ) -> bool {
        Arc::strong_count(&self.inner) == 1
            && !self.inner.closed.load(Ordering::Acquire)
            && !self.inner.members.is_empty()
            && self.inner.members.iter().all(|member| {
                let r = member.registration;
                r.authority == authority && r.original_deadline == deadline
                    && r.input_lock == input_lock && r.root_admission == root_admission
                    && r.root_budget == root_budget
                    && member.state.load(Ordering::Acquire) == UNISSUED
            })
    }

    /// The upper owner retains and validates the opaque original root cap.
    /// Digests here bind a physical resource scope, never recreate that cap.
    pub fn new(
        manifest: Digest,
        registrations: Vec<TeacherWorkerRegistration>,
    ) -> Result<Self, CheckerError> {
        Self::in_gate(manifest, registrations, global_gate())
    }
    fn in_gate(
        manifest: Digest,
        registrations: Vec<TeacherWorkerRegistration>,
        gate: Arc<Gate>,
    ) -> Result<Self, CheckerError> {
        let first = registrations
            .first()
            .ok_or(CheckerError::Invalid("empty Collection process roster"))?;
        if manifest == Digest([0; 32])
            || registrations.len() > MAX_MEMBERS
            || first.authority.epoch.0 == 0
            || first.authority.game.0 == 0
            || first.authority.root.0 == 0
            || first.authority.implementation == Digest([0; 32])
            || first.original_deadline.clock.0 != first.authority.epoch
            || first.original_deadline.at.0 == 0
            || [first.input_lock, first.root_admission, first.root_budget]
                .contains(&Digest([0; 32]))
        {
            return Err(CheckerError::Invalid("Collection original process scope"));
        }
        for (index, registration) in registrations.iter().enumerate() {
            if registration.collection_owner == 0
                || registration.frozen_actor == Digest([0; 32])
                || registration.authority != first.authority
                || registration.original_deadline != first.original_deadline
                || registration.input_lock != first.input_lock
                || registration.root_admission != first.root_admission
                || registration.root_budget != first.root_budget
                || registrations[..index].iter().any(|old| {
                    old.collection_owner == registration.collection_owner
                        || old.frozen_actor == registration.frozen_actor
                })
            {
                return Err(CheckerError::Invalid(
                    "Collection process roster mismatch/duplicate",
                ));
            }
        }
        let global = GlobalLease::acquire(gate)?;
        Ok(Self {
            inner: Arc::new(Inner {
                manifest,
                members: registrations
                    .into_iter()
                    .map(|registration| Member {
                        registration,
                        state: AtomicU8::new(UNISSUED),
                    })
                    .collect(),
                closed: AtomicBool::new(false),
                _global: global,
            }),
        })
    }
    pub fn permit(
        &self,
        manifest: Digest,
        registration: TeacherWorkerRegistration,
    ) -> Result<CollectionProcessPermit, CheckerError> {
        if self.inner.closed.load(Ordering::Acquire) || manifest != self.inner.manifest {
            return Err(CheckerError::Invalid(
                "Collection process issuer closed/manifest mismatch",
            ));
        }
        let index = self
            .inner
            .members
            .iter()
            .position(|member| member.registration == registration)
            .ok_or(CheckerError::Invalid(
                "unregistered Collection process member",
            ))?;
        self.inner.members[index]
            .state
            .compare_exchange(UNISSUED, ISSUED, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| CheckerError::Invalid("Collection process member already issued"))?;
        // Closing admission never turns a late permit into startup permission.
        if self.inner.closed.load(Ordering::Acquire) {
            self.inner.members[index]
                .state
                .store(CLOSED, Ordering::Release);
            return Err(CheckerError::Invalid("Collection process issuer closed"));
        }
        Ok(CollectionProcessPermit {
            inner: Arc::clone(&self.inner),
            index,
        })
    }
    pub fn close_admission(&self) {
        self.inner.closed.store(true, Ordering::Release);
    }
}
impl Drop for CollectionProcessScope {
    fn drop(&mut self) {
        self.close_admission();
    }
}

/// Clone aliases the same one-use member, not another process reservation.
#[derive(Clone, Debug)]
pub struct CollectionProcessPermit {
    inner: Arc<Inner>,
    index: usize,
}
impl CollectionProcessPermit {
    /// Compare opaque issuer identity, not copied roster or digest fields.
    pub fn belongs_to(&self, scope: &CollectionProcessScope) -> bool {
        Arc::ptr_eq(&self.inner, &scope.inner)
    }

    pub(crate) fn validate(
        &self,
        registration: TeacherWorkerRegistration,
    ) -> Result<(), CheckerError> {
        if self.inner.closed.load(Ordering::Acquire)
            || self.inner.members[self.index].registration != registration
            || self.inner.members[self.index].state.load(Ordering::Acquire) != ISSUED
        {
            return Err(CheckerError::Invalid(
                "Collection process permit root/member/closed",
            ));
        }
        Ok(())
    }
}

/// Moved into the actual OS owner immediately after successful spawn. A failed
/// spawn before any child exists is distinct from an unconfirmed live owner.
#[cfg(any(target_os = "linux", test))]
pub(super) struct ProcessLease {
    kind: LeaseKind,
    started: bool,
    complete: bool,
}
#[cfg(any(target_os = "linux", test))]
enum LeaseKind {
    Legacy(Option<GlobalLease>),
    Collection(CollectionProcessPermit),
}
#[cfg(any(target_os = "linux", test))]
impl ProcessLease {
    #[cfg(target_os = "linux")]
    pub(super) fn acquire(permit: Option<&CollectionProcessPermit>) -> Result<Self, CheckerError> {
        Self::acquire_in_gate(permit, global_gate())
    }
    fn acquire_in_gate(
        permit: Option<&CollectionProcessPermit>,
        gate: Arc<Gate>,
    ) -> Result<Self, CheckerError> {
        let kind = match permit {
            None => LeaseKind::Legacy(Some(GlobalLease::acquire(gate)?)),
            Some(permit) => {
                if permit.inner.closed.load(Ordering::Acquire) {
                    return Err(error("spawn", "collection_process_scope_closed"));
                }
                permit.inner.members[permit.index]
                    .state
                    .compare_exchange(ISSUED, ACQUIRED, Ordering::AcqRel, Ordering::Acquire)
                    .map_err(|_| error("spawn", "collection_process_member_unavailable"))?;
                if permit.inner.closed.load(Ordering::Acquire) {
                    permit.inner.members[permit.index]
                        .state
                        .store(CLOSED, Ordering::Release);
                    return Err(error("spawn", "collection_process_scope_closed"));
                }
                LeaseKind::Collection(permit.clone())
            }
        };
        Ok(Self {
            kind,
            started: false,
            complete: false,
        })
    }
    pub(super) fn mark_started(&mut self) {
        self.started = true;
    }
    /// Recheck immediately before the original OS spawn, after bounded hashing.
    /// A reservation already racing the close remains an owned in-flight start;
    /// the original worker cancellation/shutdown still drains that owner.
    pub(super) fn ensure_spawn_allowed(&self) -> Result<(), CheckerError> {
        if self.started || self.complete {
            return Err(error("spawn", "process_reservation_already_consumed"));
        }
        if let LeaseKind::Collection(permit) = &self.kind {
            if permit.inner.closed.load(Ordering::Acquire)
                || permit.inner.members[permit.index]
                    .state
                    .load(Ordering::Acquire)
                    != ACQUIRED
            {
                return Err(error("spawn", "collection_process_scope_closed"));
            }
        }
        Ok(())
    }
    /// Called only at the existing exit/group/pipe/reap confirmation boundary.
    pub(super) fn complete_physical(&mut self) {
        self.complete = true;
        match &mut self.kind {
            LeaseKind::Legacy(lease) => {
                drop(lease.take());
            }
            LeaseKind::Collection(permit) => {
                permit.inner.members[permit.index]
                    .state
                    .store(CLOSED, Ordering::Release);
            }
        }
    }
    fn quarantine(&self) {
        match &self.kind {
            LeaseKind::Legacy(Some(lease)) => lease.quarantine(),
            LeaseKind::Collection(permit) => {
                permit.inner.members[permit.index]
                    .state
                    .store(QUARANTINED, Ordering::Release);
                permit.inner._global.quarantine();
            }
            LeaseKind::Legacy(None) => {}
        }
    }
}
#[cfg(any(target_os = "linux", test))]
impl Drop for ProcessLease {
    fn drop(&mut self) {
        if self.started && !self.complete {
            self.quarantine();
        } else if !self.started {
            // There was no OS child. This does not fabricate exit/drain proof.
            if let LeaseKind::Collection(permit) = &self.kind {
                permit.inner.members[permit.index]
                    .state
                    .store(CLOSED, Ordering::Release);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::pals::SearchAuthority;
    use rz_contracts::{
        ClockDomain, Deadline, GameGeneration, MonotonicTick, ProcessEpoch, RootGeneration,
    };

    fn registration(owner: u32) -> TeacherWorkerRegistration {
        TeacherWorkerRegistration {
            collection_owner: owner,
            authority: SearchAuthority {
                epoch: ProcessEpoch(1),
                game: GameGeneration(1),
                root: RootGeneration(1),
                implementation: Digest([1; 32]),
            },
            original_deadline: Deadline {
                clock: ClockDomain(ProcessEpoch(1)),
                at: MonotonicTick(10),
            },
            input_lock: Digest([2; 32]),
            root_admission: Digest([3; 32]),
            root_budget: Digest([4; 32]),
            frozen_actor: Digest([u8::try_from(owner).unwrap(); 32]),
        }
    }
    fn scope(gate: &Arc<Gate>) -> CollectionProcessScope {
        CollectionProcessScope::in_gate(
            Digest([9; 32]),
            vec![registration(5), registration(6)],
            Arc::clone(gate),
        )
        .unwrap()
    }
    fn acquire(
        permit: Option<&CollectionProcessPermit>,
        gate: &Arc<Gate>,
    ) -> Result<ProcessLease, CheckerError> {
        ProcessLease::acquire_in_gate(permit, Arc::clone(gate))
    }
    #[test]
    fn collection_scope_root_and_issuer_identity_are_not_interchangeable() {
        let first = scope(&Arc::new(Gate::default()));
        let second = scope(&Arc::new(Gate::default()));
        let registration = registration(5);
        assert!(first.is_unissued_for_root(
            registration.authority, registration.original_deadline,
            registration.input_lock, registration.root_admission, registration.root_budget,
        ));
        assert!(!first.is_unissued_for_root(
            registration.authority, registration.original_deadline,
            registration.input_lock, registration.root_admission, Digest([99; 32]),
        ));
        let permit = first.permit(Digest([9; 32]), registration).unwrap();
        assert!(permit.belongs_to(&first));
        assert!(!permit.belongs_to(&second));
        assert!(!first.is_unissued_for_root(
            registration.authority, registration.original_deadline,
            registration.input_lock, registration.root_admission, registration.root_budget,
        ));
        drop(permit);
        assert!(!first.is_unissued_for_root(
            registration.authority, registration.original_deadline,
            registration.input_lock, registration.root_admission, registration.root_budget,
        ));
    }
    #[test]
    fn legacy_remains_singleton_and_not_started_release_is_not_physical() {
        let gate = Arc::new(Gate::default());
        let first = acquire(None, &gate).unwrap();
        assert!(!first.started && !first.complete);
        assert!(acquire(None, &gate).is_err());
        assert!(
            CollectionProcessScope::in_gate(
                Digest([9; 32]),
                vec![registration(5)],
                Arc::clone(&gate)
            )
            .is_err()
        );
        drop(first);
        assert!(acquire(None, &gate).is_ok());
    }
    #[test]
    fn two_registered_members_fit_one_exclusive_scope_not_a_third() {
        let gate = Arc::new(Gate::default());
        let scope = scope(&gate);
        let a = scope.permit(Digest([9; 32]), registration(5)).unwrap();
        let b = scope.permit(Digest([9; 32]), registration(6)).unwrap();
        let _first = acquire(Some(&a), &gate).unwrap();
        let _second = acquire(Some(&b), &gate).unwrap();
        assert!(scope.permit(Digest([9; 32]), registration(7)).is_err());
        assert!(acquire(None, &gate).is_err());
        assert!(
            CollectionProcessScope::in_gate(
                Digest([8; 32]),
                vec![registration(7)],
                Arc::clone(&gate)
            )
            .is_err()
        );
    }
    #[test]
    fn member_alias_cannot_duplicate_or_reopen_a_closed_process() {
        let gate = Arc::new(Gate::default());
        let scope = scope(&gate);
        let permit = scope.permit(Digest([9; 32]), registration(5)).unwrap();
        let alias = permit.clone();
        let mut first = acquire(Some(&permit), &gate).unwrap();
        first.mark_started();
        assert!(acquire(Some(&alias), &gate).is_err());
        first.complete_physical();
        assert!(acquire(Some(&alias), &gate).is_err());
        assert!(scope.permit(Digest([9; 32]), registration(5)).is_err());
    }
    #[test]
    fn wrong_manifest_root_budget_owner_and_actor_are_rejected() {
        let gate = Arc::new(Gate::default());
        let scope = scope(&gate);
        assert!(scope.permit(Digest([8; 32]), registration(5)).is_err());
        for change in 0..5 {
            let mut changed = registration(5);
            match change {
                0 => changed.root_admission = Digest([8; 32]),
                1 => changed.root_budget = Digest([8; 32]),
                2 => changed.input_lock = Digest([8; 32]),
                3 => changed.collection_owner = 8,
                _ => changed.frozen_actor = Digest([8; 32]),
            }
            assert!(scope.permit(Digest([9; 32]), changed).is_err());
        }
        let permit = scope.permit(Digest([9; 32]), registration(5)).unwrap();
        assert!(permit.validate(registration(6)).is_err());
    }
    #[test]
    fn close_prevents_late_spawn_and_issuer_drop_does_not_release_live_member() {
        let gate = Arc::new(Gate::default());
        let scope = scope(&gate);
        let a = scope.permit(Digest([9; 32]), registration(5)).unwrap();
        let b = scope.permit(Digest([9; 32]), registration(6)).unwrap();
        let mut first = acquire(Some(&a), &gate).unwrap();
        first.mark_started();
        scope.close_admission();
        assert!(scope.permit(Digest([9; 32]), registration(6)).is_err());
        assert!(acquire(Some(&b), &gate).is_err());
        drop(scope);
        drop(a);
        drop(b);
        assert!(acquire(None, &gate).is_err());
        first.complete_physical();
        drop(first);
        assert!(acquire(None, &gate).is_ok());
    }
    #[test]
    fn one_closed_member_cannot_refund_another_live_member() {
        let gate = Arc::new(Gate::default());
        let scope = scope(&gate);
        let a = scope.permit(Digest([9; 32]), registration(5)).unwrap();
        let b = scope.permit(Digest([9; 32]), registration(6)).unwrap();
        let mut first = acquire(Some(&a), &gate).unwrap();
        let mut second = acquire(Some(&b), &gate).unwrap();
        first.mark_started();
        second.mark_started();
        scope.close_admission();
        drop(scope);
        drop(a);
        drop(b);
        first.complete_physical();
        drop(first);
        assert!(acquire(None, &gate).is_err());
        second.complete_physical();
        drop(second);
        let replacement = acquire(None, &gate).unwrap();
        assert!(gate.occupied.load(Ordering::Acquire));
        drop(replacement);
    }
    #[test]
    fn quarantine_survives_all_handle_drops_and_blocks_replacement() {
        let gate = Arc::new(Gate::default());
        let scope = scope(&gate);
        let permit = scope.permit(Digest([9; 32]), registration(5)).unwrap();
        let mut owner = acquire(Some(&permit), &gate).unwrap();
        owner.mark_started();
        scope.close_admission();
        drop(scope);
        drop(permit);
        drop(owner);
        assert!(acquire(None, &gate).is_err());
        assert!(
            CollectionProcessScope::in_gate(Digest([8; 32]), vec![registration(7)], gate).is_err()
        );
    }
    #[test]
    fn never_started_member_returns_only_its_reservation_not_exit_evidence() {
        let gate = Arc::new(Gate::default());
        let scope = scope(&gate);
        let permit = scope.permit(Digest([9; 32]), registration(5)).unwrap();
        let owner = acquire(Some(&permit), &gate).unwrap();
        assert!(!owner.started && !owner.complete);
        assert!(owner.ensure_spawn_allowed().is_ok());
        scope.close_admission();
        assert!(owner.ensure_spawn_allowed().is_err());
        drop(scope);
        drop(permit);
        drop(owner);
        assert!(acquire(None, &gate).is_ok());
    }
    #[test]
    fn invalid_roster_is_rejected_before_global_reservation() {
        let gate = Arc::new(Gate::default());
        assert!(
            CollectionProcessScope::in_gate(Digest([9; 32]), vec![], Arc::clone(&gate)).is_err()
        );
        assert!(
            CollectionProcessScope::in_gate(
                Digest([9; 32]),
                vec![registration(5), registration(5)],
                Arc::clone(&gate)
            )
            .is_err()
        );
        let mut other_root = registration(6);
        other_root.root_admission = Digest([8; 32]);
        assert!(
            CollectionProcessScope::in_gate(
                Digest([9; 32]),
                vec![registration(5), other_root],
                Arc::clone(&gate)
            )
            .is_err()
        );
        assert!(acquire(None, &gate).is_ok());
    }
}
