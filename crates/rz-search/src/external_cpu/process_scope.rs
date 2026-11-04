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
const ACQUIRED: u8 = 2;
const CLOSED_NEVER_ISSUED: u8 = 3;
const QUARANTINED: u8 = 4;
const CLOSED_ISSUED_NO_CHILD: u8 = 5;
const CLOSED_ACQUIRED_NO_CHILD: u8 = 6;
const PHYSICALLY_CONFIRMED: u8 = 7;
const WORKER_UNSTARTED: u8 = 0;
const WORKER_STARTING: u8 = 1;
const WORKER_STARTED: u8 = 2;
const WORKER_JOINED: u8 = 3;
const WORKER_NO_THREAD: u8 = 4;
const WORKER_UNKNOWN: u8 = 5;
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
    worker_state: AtomicU8,
}

/// A distinct identity is retained by proof snapshots without extending the
/// OS process-gate lease held by `Inner`.
#[derive(Debug)]
struct ScopeIdentity;

#[derive(Debug)]
struct Inner {
    identity: Arc<ScopeIdentity>,
    manifest: Digest,
    members: Vec<Member>,
    closed: AtomicBool,
    physical_outcome_issued: AtomicBool,
    worker_outcome_issued: AtomicBool,
    _global: GlobalLease,
}

/// A single roster member's observed lifecycle. `NoChild` variants establish
/// absence of a child for this member; they are not process exit evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollectionMemberPhysicalState {
    Unissued,
    Issued,
    AcquisitionPending,
    NeverIssuedNoChild,
    IssuedNoChild,
    AcquiredNoChild,
    PhysicallyConfirmed,
    Quarantined,
}

/// Thread ownership is independent of OS child and pipe retirement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollectionMemberWorkerState {
    Unstarted,
    Starting,
    Started,
    Joined,
    NoThread,
    Unknown,
}

/// Original scope's immutable joined/no-thread roster, issued at most once.
/// It retains identity only, never the global process gate.
#[derive(Debug)]
pub struct CollectionScopeWorkerOutcome {
    identity: Arc<ScopeIdentity>,
    members: Vec<(TeacherWorkerRegistration, CollectionMemberWorkerState)>,
}
impl CollectionScopeWorkerOutcome {
    pub fn members(&self) -> &[(TeacherWorkerRegistration, CollectionMemberWorkerState)] {
        &self.members
    }
    /// Match an original member at its exact roster position; metadata alone
    /// never establishes membership in this issuer.
    pub fn matches_member_at(&self, index: usize, member: &CollectionProcessMemberWitness) -> bool {
        index < self.members.len()
            && index == member.index
            && Arc::ptr_eq(&self.identity, &member.identity)
    }
    pub fn belongs_to(&self, scope: &CollectionProcessScope) -> bool {
        Arc::ptr_eq(&self.identity, &scope.inner.identity)
    }
    pub fn state_for_permit(
        &self,
        permit: &CollectionProcessPermit,
    ) -> Option<CollectionMemberWorkerState> {
        if !Arc::ptr_eq(&self.identity, &permit.inner.identity) {
            return None;
        }
        let (registration, state) = self.members.get(permit.index)?;
        (permit.inner.members.get(permit.index)?.registration == *registration).then_some(*state)
    }
}

/// Proof that the original member's actual JoinHandle was joined. Join success
/// is not inferred: its result is retained separately by the worker owner.
#[derive(Debug)]
pub struct CollectionWorkerJoinedWitness {
    identity: Arc<ScopeIdentity>,
    index: usize,
}

/// Identity-only member reference. This does not assert that a worker was
/// spawned or joined; callers must also possess original lifecycle evidence.
#[derive(Debug)]
pub struct CollectionProcessMemberWitness {
    identity: Arc<ScopeIdentity>,
    index: usize,
}
impl CollectionProcessMemberWitness {
    pub fn same_member(&self, permit: &CollectionProcessPermit) -> bool {
        Arc::ptr_eq(&self.identity, &permit.inner.identity) && self.index == permit.index
    }
    pub fn same_joined_member(&self, joined: &CollectionWorkerJoinedWitness) -> bool {
        Arc::ptr_eq(&self.identity, &joined.identity) && self.index == joined.index
    }
    pub fn state_in(
        &self,
        outcome: &CollectionScopeWorkerOutcome,
    ) -> Option<CollectionMemberWorkerState> {
        if !Arc::ptr_eq(&self.identity, &outcome.identity) {
            return None;
        }
        outcome.members.get(self.index).map(|(_, state)| *state)
    }
    pub fn physical_state_in(
        &self,
        outcome: &CollectionScopePhysicalOutcome,
    ) -> Option<CollectionMemberPhysicalState> {
        if !Arc::ptr_eq(&self.identity, &outcome.identity) {
            return None;
        }
        outcome.members.get(self.index).map(|(_, state)| *state)
    }
}
impl CollectionWorkerJoinedWitness {
    pub fn same_member(&self, permit: &CollectionProcessPermit) -> bool {
        Arc::ptr_eq(&self.identity, &permit.inner.identity) && self.index == permit.index
    }
}

/// Same-issuer, non-cloneable closed-roster snapshot. Private fields prevent a
/// copied registration or externally constructed state from becoming proof.
/// Retaining it does not itself retain the global process lease. This does not
/// certify worker joins, terminal delivery, successful labels, or publication.
#[derive(Debug)]
pub struct CollectionScopePhysicalOutcome {
    identity: Arc<ScopeIdentity>,
    manifest: Digest,
    members: Vec<(TeacherWorkerRegistration, CollectionMemberPhysicalState)>,
}

impl CollectionScopePhysicalOutcome {
    pub fn manifest(&self) -> Digest {
        self.manifest
    }

    /// Both outcomes must come from the same original, opaque scope issuer.
    pub fn same_issuer_as_worker(&self, worker: &CollectionScopeWorkerOutcome) -> bool {
        Arc::ptr_eq(&self.identity, &worker.identity)
    }

    pub fn members(&self) -> &[(TeacherWorkerRegistration, CollectionMemberPhysicalState)] {
        &self.members
    }

    pub fn belongs_to(&self, scope: &CollectionProcessScope) -> bool {
        Arc::ptr_eq(&self.identity, &scope.inner.identity)
    }

    /// A copied registration alone is not proof of the original member.
    pub fn state_for_permit(
        &self,
        permit: &CollectionProcessPermit,
    ) -> Option<CollectionMemberPhysicalState> {
        if !Arc::ptr_eq(&self.identity, &permit.inner.identity) {
            return None;
        }
        let (registration, state) = self.members.get(permit.index)?;
        (permit.inner.members.get(permit.index)?.registration == *registration).then_some(*state)
    }
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
                r.authority == authority
                    && r.original_deadline == deadline
                    && r.input_lock == input_lock
                    && r.root_admission == root_admission
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
                identity: Arc::new(ScopeIdentity),
                manifest,
                members: registrations
                    .into_iter()
                    .map(|registration| Member {
                        registration,
                        state: AtomicU8::new(UNISSUED),
                        worker_state: AtomicU8::new(WORKER_UNSTARTED),
                    })
                    .collect(),
                closed: AtomicBool::new(false),
                physical_outcome_issued: AtomicBool::new(false),
                worker_outcome_issued: AtomicBool::new(false),
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
                .store(CLOSED_ISSUED_NO_CHILD, Ordering::Release);
            return Err(CheckerError::Invalid("Collection process issuer closed"));
        }
        Ok(CollectionProcessPermit {
            inner: Arc::clone(&self.inner),
            index,
        })
    }
    pub fn close_admission(&self) {
        self.inner.closed.store(true, Ordering::Release);
        // CAS makes closing race safely with both permit issuance and actual
        // acquisition. ACQUIRED stays pending until its original owner either
        // confirms there was no child, reaps it, or quarantines it.
        for member in &self.inner.members {
            // An in-flight spawn stays pending until its actual owner settles it.
            let _ = member.worker_state.compare_exchange(
                WORKER_UNSTARTED,
                WORKER_NO_THREAD,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            let mut observed = member.state.load(Ordering::Acquire);
            loop {
                let no_child = match observed {
                    UNISSUED => CLOSED_NEVER_ISSUED,
                    ISSUED => CLOSED_ISSUED_NO_CHILD,
                    _ => break,
                };
                match member.state.compare_exchange_weak(
                    observed,
                    no_child,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => break,
                    Err(actual) => observed = actual,
                }
            }
        }
    }

    /// Read-only member states. They do not by themselves certify closure.
    pub fn physical_states(
        &self,
    ) -> Vec<(TeacherWorkerRegistration, CollectionMemberPhysicalState)> {
        self.inner
            .members
            .iter()
            .map(|member| {
                (
                    member.registration,
                    physical_state(member.state.load(Ordering::Acquire)),
                )
            })
            .collect()
    }

    pub fn worker_states(&self) -> Vec<(TeacherWorkerRegistration, CollectionMemberWorkerState)> {
        self.inner
            .members
            .iter()
            .map(|member| {
                let state = match member.worker_state.load(Ordering::Acquire) {
                    WORKER_UNSTARTED => CollectionMemberWorkerState::Unstarted,
                    WORKER_STARTING => CollectionMemberWorkerState::Starting,
                    WORKER_STARTED => CollectionMemberWorkerState::Started,
                    WORKER_JOINED => CollectionMemberWorkerState::Joined,
                    WORKER_NO_THREAD => CollectionMemberWorkerState::NoThread,
                    _ => CollectionMemberWorkerState::Unknown,
                };
                (member.registration, state)
            })
            .collect()
    }

    pub fn try_worker_retirement(
        &self,
    ) -> Result<Option<CollectionScopeWorkerOutcome>, CheckerError> {
        if !self.inner.closed.load(Ordering::Acquire) {
            return Ok(None);
        }
        let members = self.worker_states();
        if members
            .iter()
            .any(|(_, state)| *state == CollectionMemberWorkerState::Unknown)
        {
            return Err(CheckerError::Invalid("Collection worker ownership unknown"));
        }
        if members.iter().any(|(_, state)| {
            !matches!(
                state,
                CollectionMemberWorkerState::Joined | CollectionMemberWorkerState::NoThread
            )
        }) {
            return Ok(None);
        }
        self.inner
            .worker_outcome_issued
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| CheckerError::Invalid("Collection worker outcome already issued"))?;
        Ok(Some(CollectionScopeWorkerOutcome {
            identity: Arc::clone(&self.inner.identity),
            members,
        }))
    }

    /// Issue the original closed roster only after every member has a stable
    /// no-child or physical completion outcome. Pending or quarantined members
    /// never become a successful proof. At most one snapshot can be issued.
    pub fn try_physical_retirement(
        &self,
    ) -> Result<Option<CollectionScopePhysicalOutcome>, CheckerError> {
        if !self.inner.closed.load(Ordering::Acquire) {
            return Ok(None);
        }
        let members = self.physical_states();
        if members
            .iter()
            .any(|(_, state)| *state == CollectionMemberPhysicalState::Quarantined)
        {
            return Err(CheckerError::Invalid(
                "Collection process member is quarantined",
            ));
        }
        if members.iter().any(|(_, state)| {
            matches!(
                state,
                CollectionMemberPhysicalState::Unissued
                    | CollectionMemberPhysicalState::Issued
                    | CollectionMemberPhysicalState::AcquisitionPending
            )
        }) {
            return Ok(None);
        }
        self.inner
            .physical_outcome_issued
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| CheckerError::Invalid("Collection physical outcome already issued"))?;
        Ok(Some(CollectionScopePhysicalOutcome {
            identity: Arc::clone(&self.inner.identity),
            manifest: self.inner.manifest,
            members,
        }))
    }
}

fn physical_state(state: u8) -> CollectionMemberPhysicalState {
    match state {
        UNISSUED => CollectionMemberPhysicalState::Unissued,
        ISSUED => CollectionMemberPhysicalState::Issued,
        ACQUIRED => CollectionMemberPhysicalState::AcquisitionPending,
        CLOSED_NEVER_ISSUED => CollectionMemberPhysicalState::NeverIssuedNoChild,
        CLOSED_ISSUED_NO_CHILD => CollectionMemberPhysicalState::IssuedNoChild,
        CLOSED_ACQUIRED_NO_CHILD => CollectionMemberPhysicalState::AcquiredNoChild,
        PHYSICALLY_CONFIRMED => CollectionMemberPhysicalState::PhysicallyConfirmed,
        QUARANTINED => CollectionMemberPhysicalState::Quarantined,
        // Unknown byte values are never treated as a successful retirement.
        _ => CollectionMemberPhysicalState::Quarantined,
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
    /// Compare the original issuer and roster slot, not copied registration data.
    pub fn same_member(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner) && self.index == other.index
    }

    pub fn member_witness(&self) -> CollectionProcessMemberWitness {
        CollectionProcessMemberWitness {
            identity: Arc::clone(&self.inner.identity),
            index: self.index,
        }
    }

    /// The unique reservation precedes thread creation. Closing the scope and
    /// reservation race through the same slot; neither can invent no-thread
    /// while an actual spawn attempt may still be in flight.
    pub(super) fn reserve_worker(&self) -> Result<CollectionWorkerReservation, CheckerError> {
        if self.inner.closed.load(Ordering::Acquire) {
            return Err(CheckerError::Invalid("Collection worker scope closed"));
        }
        self.inner.members[self.index]
            .worker_state
            .compare_exchange(
                WORKER_UNSTARTED,
                WORKER_STARTING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map_err(|_| CheckerError::Invalid("Collection worker already reserved"))?;
        let reservation = CollectionWorkerReservation {
            permit: Some(self.clone()),
        };
        if self.inner.closed.load(Ordering::Acquire) {
            reservation.cancel_before_spawn();
            return Err(CheckerError::Invalid("Collection worker scope closed"));
        }
        Ok(reservation)
    }

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

/// Owned by the spawn caller until it has an explicit spawn outcome. Dropping
/// on an uncertain unwind cannot be interpreted as no thread.
pub(super) struct CollectionWorkerReservation {
    permit: Option<CollectionProcessPermit>,
}
impl CollectionWorkerReservation {
    pub(super) fn spawn_allowed(&self) -> bool {
        self.permit
            .as_ref()
            .is_some_and(|permit| !permit.inner.closed.load(Ordering::Acquire))
    }
    pub(super) fn cancel_before_spawn(mut self) {
        self.finish_no_thread();
    }
    pub(super) fn spawn_failed(mut self) {
        self.finish_no_thread();
    }
    fn finish_no_thread(&mut self) {
        if let Some(permit) = self.permit.take() {
            if permit.inner.members[permit.index]
                .worker_state
                .compare_exchange(
                    WORKER_STARTING,
                    WORKER_NO_THREAD,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
            {
                quarantine_worker(&permit);
            }
        }
    }
    /// Always returns ownership to the caller after a successful OS spawn.
    /// An impossible state mismatch is quarantined and still retains the lease.
    pub(super) fn spawn_succeeded(mut self) -> CollectionWorkerLease {
        let permit = self.permit.take().expect("live worker reservation");
        if permit.inner.members[permit.index]
            .worker_state
            .compare_exchange(
                WORKER_STARTING,
                WORKER_STARTED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            quarantine_worker(&permit);
        }
        CollectionWorkerLease {
            permit: Some(permit),
        }
    }
}
impl Drop for CollectionWorkerReservation {
    fn drop(&mut self) {
        if let Some(permit) = self.permit.take() {
            quarantine_worker(&permit);
        }
    }
}

/// Moves with the actual JoinHandle owner until an actual join, including a
/// joined panic. Its error remains distinct from this ownership evidence.
pub(super) struct CollectionWorkerLease {
    permit: Option<CollectionProcessPermit>,
}
impl CollectionWorkerLease {
    pub(super) fn joined(mut self) -> Result<CollectionWorkerJoinedWitness, CheckerError> {
        let permit = self.permit.take().expect("live worker lease");
        if permit.inner.members[permit.index]
            .worker_state
            .compare_exchange(
                WORKER_STARTED,
                WORKER_JOINED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            quarantine_worker(&permit);
            return Err(CheckerError::Invalid(
                "Collection worker join state changed",
            ));
        }
        Ok(CollectionWorkerJoinedWitness {
            identity: Arc::clone(&permit.inner.identity),
            index: permit.index,
        })
    }
}
impl Drop for CollectionWorkerLease {
    fn drop(&mut self) {
        if let Some(permit) = self.permit.take() {
            quarantine_worker(&permit);
        }
    }
}

fn quarantine_worker(permit: &CollectionProcessPermit) {
    permit.inner.members[permit.index]
        .worker_state
        .store(WORKER_UNKNOWN, Ordering::Release);
    permit.inner._global.quarantine();
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
                        .store(CLOSED_ACQUIRED_NO_CHILD, Ordering::Release);
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
        if !self.started {
            // This method is reserved for a reaped OS child. Treat misuse as
            // uncertain ownership, rather than reporting success or silently
            // converting it to the normal no-child release path.
            self.quarantine();
            self.started = true;
            panic!("process physical completion attempted before start");
        }
        self.complete = true;
        match &mut self.kind {
            LeaseKind::Legacy(lease) => {
                drop(lease.take());
            }
            LeaseKind::Collection(permit) => {
                permit.inner.members[permit.index]
                    .state
                    .store(PHYSICALLY_CONFIRMED, Ordering::Release);
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
                    .store(CLOSED_ACQUIRED_NO_CHILD, Ordering::Release);
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

    #[test]
    fn worker_spawn_racing_close_remains_pending_until_join() {
        let gate = Arc::new(Gate::default());
        let member = registration(1);
        let scope =
            CollectionProcessScope::in_gate(Digest([9; 32]), vec![member], Arc::clone(&gate))
                .unwrap();
        let permit = scope.permit(Digest([9; 32]), member).unwrap();
        let id = permit.member_witness();
        let reservation = permit.reserve_worker().unwrap();
        assert!(permit.reserve_worker().is_err());
        scope.close_admission();
        assert!(!reservation.spawn_allowed());
        assert!(scope.try_worker_retirement().unwrap().is_none());
        // Models the case where the thread was created before close won.
        let lease = reservation.spawn_succeeded();
        assert!(scope.try_worker_retirement().unwrap().is_none());
        let joined = lease.joined().unwrap();
        let roster = scope.try_worker_retirement().unwrap().unwrap();
        assert!(id.same_joined_member(&joined));
        assert_eq!(
            id.state_in(&roster),
            Some(CollectionMemberWorkerState::Joined)
        );
        assert!(scope.try_worker_retirement().is_err());
    }

    #[test]
    fn worker_only_explicit_failure_is_no_thread_and_abandoned_spawn_is_unknown() {
        let gate = Arc::new(Gate::default());
        let member = registration(1);
        let scope =
            CollectionProcessScope::in_gate(Digest([9; 32]), vec![member], Arc::clone(&gate))
                .unwrap();
        let permit = scope.permit(Digest([9; 32]), member).unwrap();
        permit.reserve_worker().unwrap().spawn_failed();
        scope.close_admission();
        assert_eq!(
            scope
                .try_worker_retirement()
                .unwrap()
                .unwrap()
                .state_for_permit(&permit),
            Some(CollectionMemberWorkerState::NoThread)
        );
        drop(permit);
        drop(scope);
        let next =
            CollectionProcessScope::in_gate(Digest([9; 32]), vec![member], Arc::clone(&gate))
                .unwrap();
        let permit = next.permit(Digest([9; 32]), member).unwrap();
        drop(permit.reserve_worker().unwrap());
        next.close_admission();
        assert!(next.try_worker_retirement().is_err());
        drop(permit);
        drop(next);
        assert!(gate.occupied.load(Ordering::Acquire));
    }

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
            registration.authority,
            registration.original_deadline,
            registration.input_lock,
            registration.root_admission,
            registration.root_budget,
        ));
        assert!(!first.is_unissued_for_root(
            registration.authority,
            registration.original_deadline,
            registration.input_lock,
            registration.root_admission,
            Digest([99; 32]),
        ));
        let permit = first.permit(Digest([9; 32]), registration).unwrap();
        assert!(permit.belongs_to(&first));
        assert!(!permit.belongs_to(&second));
        assert!(!first.is_unissued_for_root(
            registration.authority,
            registration.original_deadline,
            registration.input_lock,
            registration.root_admission,
            registration.root_budget,
        ));
        drop(permit);
        assert!(!first.is_unissued_for_root(
            registration.authority,
            registration.original_deadline,
            registration.input_lock,
            registration.root_admission,
            registration.root_budget,
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
    fn closed_roster_separates_no_child_outcomes_from_real_reap_and_keeps_identity() {
        let gate = Arc::new(Gate::default());
        let scope = CollectionProcessScope::in_gate(
            Digest([9; 32]),
            vec![
                registration(5),
                registration(6),
                registration(7),
                registration(8),
            ],
            Arc::clone(&gate),
        )
        .unwrap();
        let issued = scope.permit(Digest([9; 32]), registration(5)).unwrap();
        let acquired = scope.permit(Digest([9; 32]), registration(6)).unwrap();
        let confirmed = scope.permit(Digest([9; 32]), registration(7)).unwrap();
        let unstarted_owner = acquire(Some(&acquired), &gate).unwrap();
        let mut finished_owner = acquire(Some(&confirmed), &gate).unwrap();
        finished_owner.mark_started();
        scope.close_admission();
        assert!(scope.try_physical_retirement().unwrap().is_none());
        drop(unstarted_owner);
        assert!(scope.try_physical_retirement().unwrap().is_none());
        finished_owner.complete_physical();
        drop(finished_owner);
        let outcome = scope
            .try_physical_retirement()
            .unwrap()
            .expect("closed roster");
        assert!(outcome.belongs_to(&scope));
        assert_eq!(
            outcome.state_for_permit(&issued),
            Some(CollectionMemberPhysicalState::IssuedNoChild)
        );
        assert_eq!(
            outcome.state_for_permit(&acquired),
            Some(CollectionMemberPhysicalState::AcquiredNoChild)
        );
        assert_eq!(
            outcome.state_for_permit(&confirmed),
            Some(CollectionMemberPhysicalState::PhysicallyConfirmed)
        );
        assert_eq!(
            outcome.members()[3].1,
            CollectionMemberPhysicalState::NeverIssuedNoChild
        );
        assert!(scope.try_physical_retirement().is_err());
        drop(scope);
        drop(issued);
        drop(acquired);
        drop(confirmed);
        // The immutable outcome does not retain the physical process gate.
        let next = self::scope(&gate);
        assert!(!outcome.belongs_to(&next));
        let next_permit = next.permit(Digest([9; 32]), registration(5)).unwrap();
        assert_eq!(outcome.state_for_permit(&next_permit), None);
    }

    #[test]
    fn quarantined_started_member_cannot_issue_successful_closed_outcome() {
        let gate = Arc::new(Gate::default());
        let scope = scope(&gate);
        let permit = scope.permit(Digest([9; 32]), registration(5)).unwrap();
        let mut owner = acquire(Some(&permit), &gate).unwrap();
        owner.mark_started();
        scope.close_admission();
        assert!(scope.try_physical_retirement().unwrap().is_none());
        drop(owner);
        assert!(scope.try_physical_retirement().is_err());
        assert_eq!(
            scope.physical_states()[0].1,
            CollectionMemberPhysicalState::Quarantined
        );
        drop(scope);
        drop(permit);
        assert!(acquire(None, &gate).is_err());
    }

    #[test]
    fn completion_before_start_is_failure_and_keeps_process_gate_quarantined() {
        let gate = Arc::new(Gate::default());
        let scope = scope(&gate);
        let permit = scope.permit(Digest([9; 32]), registration(5)).unwrap();
        let mut owner = acquire(Some(&permit), &gate).unwrap();
        scope.close_admission();
        let failure =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| owner.complete_physical()));
        assert!(failure.is_err());
        assert!(scope.try_physical_retirement().is_err());
        drop(owner);
        drop(scope);
        drop(permit);
        assert!(acquire(None, &gate).is_err());
    }

    #[test]
    fn close_racing_acquisition_never_turns_a_live_owner_into_no_child() {
        use std::sync::{Barrier, mpsc};
        let gate = Arc::new(Gate::default());
        let scope = Arc::new(scope(&gate));
        let permit = scope.permit(Digest([9; 32]), registration(5)).unwrap();
        let attempt = permit.clone();
        let barrier = Arc::new(Barrier::new(2));
        let worker_barrier = Arc::clone(&barrier);
        let worker_gate = Arc::clone(&gate);
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            worker_barrier.wait();
            match acquire(Some(&attempt), &worker_gate) {
                Ok(mut owner) => {
                    owner.mark_started();
                    started_tx.send(true).unwrap();
                    release_rx.recv().unwrap();
                    owner.complete_physical();
                }
                Err(_) => {
                    started_tx.send(false).unwrap();
                }
            }
        });
        barrier.wait();
        scope.close_admission();
        let started = started_rx.recv().unwrap();
        let early = scope.try_physical_retirement().unwrap();
        if started {
            assert!(early.is_none());
        }
        release_tx.send(()).unwrap_or(());
        worker.join().unwrap();
        let outcome = match early {
            Some(outcome) => outcome,
            None => scope
                .try_physical_retirement()
                .unwrap()
                .expect("finished member"),
        };
        let state = outcome.state_for_permit(&permit).unwrap();
        if started {
            assert_eq!(state, CollectionMemberPhysicalState::PhysicallyConfirmed);
        } else {
            assert!(matches!(
                state,
                CollectionMemberPhysicalState::IssuedNoChild
                    | CollectionMemberPhysicalState::AcquiredNoChild
            ));
        }
        assert!(acquire(Some(&permit), &gate).is_err());
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
