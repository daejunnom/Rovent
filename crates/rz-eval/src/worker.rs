//! One bounded physical worker, suitable for a runtime Backend lease adapter.
//! No queue policy, deadlines, request IDs, cancellation or logical finalization.

use crate::error::{BackendError, CauseCode, FailureKind as K, FailureStage as S};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::task::Poll;
use std::thread::JoinHandle;
use std::time::Instant;

/// Scope of actual physical work, distinct from a later recovery observation.
/// A value does not itself prove that any physical operation completed.
#[cfg(feature = "contracts")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhysicalWorkScope {
    Ticket {
        ticket: rz_contracts::pals_adaptive::RoleTicketEnvelope,
        execution: rz_contracts::ExecutionId,
    },
    Control {
        epoch: rz_contracts::ProcessEpoch,
        sequence: u64,
        kind: PhysicalControlKind,
    },
    Startup {
        epoch: rz_contracts::ProcessEpoch,
    },
}

#[cfg(feature = "contracts")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhysicalControlKind {
    Reset,
    Verify,
    Quiesce,
    Discard,
    Other,
}

/// Scoped submit derives identity from the actual command, rather than
/// accepting a separate potentially inconsistent caller-supplied claim.
#[cfg(feature = "contracts")]
pub trait PhysicalScopedInput {
    fn physical_work_scope(&self) -> Result<PhysicalWorkScope, BackendError>;
}

#[cfg(feature = "contracts")]
impl PhysicalWorkScope {
    fn validate(self) -> Result<Self, BackendError> {
        if let Self::Ticket { ticket, execution } = self {
            if execution.epoch != ticket.id.epoch
                || ticket.id.epoch != ticket.authority.epoch
                || ticket.view.authority != ticket.authority
            {
                return Err(BackendError::new(
                    K::IdentityMismatch,
                    S::Admission,
                    "physical work scope epoch or root differs from its ticket",
                ));
            }
        }
        Ok(self)
    }
}

const IDLE: u8 = 0;
const BUSY: u8 = 1;
const CLOSED: u8 = 2;
const QUARANTINED: u8 = 3;

type PhysicalHandle = Arc<Mutex<Option<JoinHandle<Result<(), BackendError>>>>>;
type ShutdownSender = mpsc::SyncSender<Result<(), BackendError>>;

// Bounded process-local custody. This is not cross-process device admission,
// GPU completion, or permission to issue a new execution.
const RECOVERY_SLOTS: usize = 16;
struct RecoverySlot {
    identity: Arc<()>,
    control: Option<RecoveryControl>,
}
struct RecoveryControl {
    sender: mpsc::SyncSender<RecoveryCommand>,
    pending: Arc<AtomicBool>,
    state: Arc<AtomicU8>,
    handle: PhysicalHandle,
    confirmation: Arc<Mutex<Option<PhysicalRecoveryConfirmation>>>,
}
enum RecoveryCommand {
    Observe {
        deadline: Instant,
        request: Arc<()>,
        reply: mpsc::SyncSender<PhysicalRecoveryObservation>,
    },
    Release {
        deadline: Instant,
        request: Arc<()>,
        reply: mpsc::SyncSender<PhysicalRecoveryObservation>,
    },
}
fn recovery_slots() -> &'static Mutex<[Option<RecoverySlot>; RECOVERY_SLOTS]> {
    static SLOTS: OnceLock<Mutex<[Option<RecoverySlot>; RECOVERY_SLOTS]>> = OnceLock::new();
    SLOTS.get_or_init(|| Mutex::new(std::array::from_fn(|_| None)))
}

#[derive(Clone)]
pub struct PhysicalRecoveryKey {
    index: usize,
    identity: Arc<()>,
}
pub struct PhysicalRecoveryReservation {
    key: PhysicalRecoveryKey,
    published: bool,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhysicalRecoveryObservation {
    Unknown,
    Confirmed,
}
/// Only CPU fixture code can produce a positive result until a concrete,
/// same-owner production fence is implemented and independently validated.
/// Opaque evidence produced by the original backend after successful
/// synchronous work. It is not evidence for another owner or for recovery.
#[derive(Clone, Debug)]
pub struct SynchronousOwnerProof { owner: Arc<()>, completed_synchronous_runs: u64 }
impl SynchronousOwnerProof {
    pub(crate) fn new(owner: Arc<()>, completed_synchronous_runs: u64) -> Self { Self { owner, completed_synchronous_runs } }
}

/// Native session release after a successful join; this is not a GPU fence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SessionReleased { pub completed_synchronous_runs: u64 }

enum ShutdownEvidence {
    Recovery(PhysicalRecoveryFence),
    Session(SynchronousOwnerProof),
}

pub(crate) mod synchronous_owner_sealed {
    pub trait Sealed {}
}
/// Only crate-owned physical backends can bind a proof to their original owner.
pub trait SynchronousPhysicalOwner: synchronous_owner_sealed::Sealed {
    fn synchronous_owner_identity(&self) -> Arc<()>;
}

#[derive(Clone, Debug)]
pub enum PhysicalRecoveryFence {
    Unknown,
    #[cfg(test)]
    FixtureOwnerCompleted,
}
impl PhysicalRecoveryFence {
    fn confirmed(&self) -> bool {
        match self {
            Self::Unknown => false,
            #[cfg(test)]
            Self::FixtureOwnerCompleted => true,
        }
    }
}
#[derive(Clone)]
pub struct PhysicalRecoveryConfirmation {
    owner: Arc<()>,
    request: Arc<()>,
    late: bool,
    #[cfg(feature = "contracts")]
    work_scope: Option<PhysicalWorkScope>,
}
impl PhysicalRecoveryConfirmation {
    pub fn was_late(&self) -> bool {
        self.late
    }
    /// Original active physical job identity, not the request that asked for
    /// this recovery observation. Unscoped legacy/startup work stays None.
    #[cfg(feature = "contracts")]
    pub fn work_scope(&self) -> Option<PhysicalWorkScope> {
        self.work_scope
    }
}
pub struct PhysicalRecoveryPending {
    reply: mpsc::Receiver<PhysicalRecoveryObservation>,
    deadline: Instant,
    result: Option<PhysicalRecoveryObservation>,
}
impl PhysicalRecoveryReservation {
    /// Reserve before the original native entry. It protects only this local
    /// slot; a supervisor must separately reserve and identify the device.
    pub fn reserve() -> Result<(Self, PhysicalRecoveryKey), BackendError> {
        let mut slots = recovery_slots().lock().unwrap_or_else(|e| e.into_inner());
        let index = slots.iter().position(Option::is_none).ok_or_else(|| {
            BackendError::new(
                K::ResourceExhausted,
                S::Admission,
                "physical recovery slots full",
            )
        })?;
        let identity = Arc::new(());
        slots[index] = Some(RecoverySlot {
            identity: Arc::clone(&identity),
            control: None,
        });
        let key = PhysicalRecoveryKey { index, identity };
        Ok((
            Self {
                key: key.clone(),
                published: false,
            },
            key,
        ))
    }
    /// Retire only this still-unpublished process-local slot. The caller must
    /// independently establish that native entry did not occur: this method
    /// proves neither native non-entry nor GPU completion.
    pub fn cancel_unpublished(self) -> Result<(), Self> {
        let mut slots = recovery_slots().lock().unwrap_or_else(|e| e.into_inner());
        let matches = !self.published
            && slots.get(self.key.index).and_then(Option::as_ref).is_some_and(|slot| {
                Arc::ptr_eq(&slot.identity, &self.key.identity) && slot.control.is_none()
            });
        if matches {
            slots[self.key.index] = None;
        }
        drop(slots);
        if matches { Ok(()) } else { Err(self) }
    }
    fn publish(&mut self, control: RecoveryControl) -> Result<(), RecoveryControl> {
        let mut slots = recovery_slots().lock().unwrap_or_else(|e| e.into_inner());
        let Some(slot) = slots.get_mut(self.key.index).and_then(Option::as_mut) else {
            return Err(control);
        };
        if self.published
            || !Arc::ptr_eq(&slot.identity, &self.key.identity)
            || slot.control.is_some()
        {
            return Err(control);
        }
        slot.control = Some(control);
        self.published = true;
        Ok(())
    }
    fn unpublish_before_spawn(&mut self) {
        let mut slots = recovery_slots().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = slots.get_mut(self.key.index).and_then(Option::as_mut) {
            if Arc::ptr_eq(&slot.identity, &self.key.identity) {
                slot.control = None;
                self.published = false;
            }
        }
    }
}
impl Drop for PhysicalRecoveryReservation {
    fn drop(&mut self) {
        if !self.published {
            let mut slots = recovery_slots().lock().unwrap_or_else(|e| e.into_inner());
            if slots
                .get(self.key.index)
                .and_then(Option::as_ref)
                .is_some_and(|slot| {
                    Arc::ptr_eq(&slot.identity, &self.key.identity) && slot.control.is_none()
                })
            {
                slots[self.key.index] = None;
            }
        }
    }
}
impl PhysicalRecoveryKey {
    /// Identity checks are local to the exact registered owner, and do not
    /// themselves create a physical fence or release permission.
    pub fn same_owner(&self, other: &Self) -> bool {
        self.index == other.index && Arc::ptr_eq(&self.identity, &other.identity)
    }
    pub fn confirmation_owner_matches(&self, proof: &PhysicalRecoveryConfirmation) -> bool {
        Arc::ptr_eq(&proof.owner, &self.identity)
    }
    pub fn owns_confirmation(&self, proof: &PhysicalRecoveryConfirmation) -> bool {
        if !Arc::ptr_eq(&proof.owner, &self.identity) {
            return false;
        }
        let slots = recovery_slots().lock().unwrap_or_else(|e| e.into_inner());
        let Some(control) = slots
            .get(self.index)
            .and_then(Option::as_ref)
            .filter(|slot| Arc::ptr_eq(&slot.identity, &self.identity))
            .and_then(|slot| slot.control.as_ref())
        else {
            return false;
        };
        if control.state.load(Ordering::Acquire) != QUARANTINED
            || control.pending.load(Ordering::Acquire)
        {
            return false;
        }
        let matches = control
            .confirmation
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|last| {
                Arc::ptr_eq(&last.request, &proof.request)
                    && {
                        #[cfg(feature = "contracts")]
                        { last.work_scope == proof.work_scope }
                        #[cfg(not(feature = "contracts"))]
                        { true }
                    }
            });
        matches
    }
    /// The latest proof must be from this exact owner and this exact original
    /// command scope. A recovery-observation ID is never treated as work ID.
    #[cfg(feature = "contracts")]
    pub fn confirms_work_scope(
        &self,
        proof: &PhysicalRecoveryConfirmation,
        expected: PhysicalWorkScope,
    ) -> bool {
        proof.work_scope == Some(expected) && self.owns_confirmation(proof)
    }
    pub fn recoverable_keys() -> Vec<Self> {
        let slots = recovery_slots().lock().unwrap_or_else(|e| e.into_inner());
        slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| {
                let slot = slot.as_ref()?;
                let control = slot.control.as_ref()?;
                matches!(control.state.load(Ordering::Acquire), QUARANTINED | CLOSED).then(|| {
                    Self {
                        index,
                        identity: Arc::clone(&slot.identity),
                    }
                })
            })
            .collect()
    }
    /// One outstanding observation per original owner. This is nonblocking;
    /// timeout or an absent fence is Unknown and never releases the slot.
    pub fn begin_reobserve(
        &self,
        deadline: Instant,
    ) -> Result<PhysicalRecoveryPending, BackendError> {
        if Instant::now() >= deadline {
            return Err(BackendError::new(
                K::BackendUnavailable,
                S::Admission,
                "physical reobservation deadline expired",
            ));
        }
        let slots = recovery_slots().lock().unwrap_or_else(|e| e.into_inner());
        let control = slots
            .get(self.index)
            .and_then(Option::as_ref)
            .filter(|slot| Arc::ptr_eq(&slot.identity, &self.identity))
            .and_then(|slot| slot.control.as_ref())
            .ok_or_else(|| {
                BackendError::new(
                    K::InvalidInput,
                    S::Admission,
                    "physical recovery owner unavailable",
                )
            })?;
        if control.state.load(Ordering::Acquire) != QUARANTINED
            || control
                .pending
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return Err(BackendError::new(
                K::BackendUnavailable,
                S::Admission,
                "physical reobservation is unavailable or pending",
            ));
        }
        let (reply, receiver) = mpsc::sync_channel(1);
        let request = Arc::new(());
        if control
            .sender
            .try_send(RecoveryCommand::Observe {
                deadline,
                request,
                reply,
            })
            .is_err()
        {
            control.pending.store(false, Ordering::Release);
            return Err(BackendError::new(
                K::BackendUnavailable,
                S::Backend,
                "physical owner control unavailable",
            ));
        }
        Ok(PhysicalRecoveryPending {
            reply: receiver,
            deadline,
            result: None,
        })
    }
    /// The supervisor may recover the same owner's last proven observation,
    /// including a late result that the original caller observed as Unknown.
    pub fn latest_confirmation(&self) -> Option<PhysicalRecoveryConfirmation> {
        let slots = recovery_slots().lock().unwrap_or_else(|e| e.into_inner());
        let control = slots
            .get(self.index)
            .and_then(Option::as_ref)
            .filter(|slot| Arc::ptr_eq(&slot.identity, &self.identity))?
            .control
            .as_ref()?;
        let result = control
            .confirmation
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        result
    }
    /// Release is a separate explicit command bound to the exact owner and
    /// last successful observation. It never requests a fresh fence itself.
    pub fn begin_release(
        &self,
        proof: &PhysicalRecoveryConfirmation,
        deadline: Instant,
    ) -> Result<PhysicalRecoveryPending, BackendError> {
        if Instant::now() >= deadline || !Arc::ptr_eq(&proof.owner, &self.identity) {
            return Err(BackendError::new(
                K::InvalidInput,
                S::Admission,
                "physical release proof/deadline invalid",
            ));
        }
        let slots = recovery_slots().lock().unwrap_or_else(|e| e.into_inner());
        let control = slots
            .get(self.index)
            .and_then(Option::as_ref)
            .filter(|slot| Arc::ptr_eq(&slot.identity, &self.identity))
            .and_then(|slot| slot.control.as_ref())
            .ok_or_else(|| {
                BackendError::new(
                    K::InvalidInput,
                    S::Admission,
                    "physical recovery owner unavailable",
                )
            })?;
        let valid = control
            .confirmation
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|last| Arc::ptr_eq(&last.request, &proof.request));
        if !valid
            || control.state.load(Ordering::Acquire) != QUARANTINED
            || control
                .pending
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return Err(BackendError::new(
                K::BackendUnavailable,
                S::Admission,
                "physical release is unavailable or pending",
            ));
        }
        let (reply, receiver) = mpsc::sync_channel(1);
        if control
            .sender
            .try_send(RecoveryCommand::Release {
                deadline,
                request: Arc::clone(&proof.request),
                reply,
            })
            .is_err()
        {
            control.pending.store(false, Ordering::Release);
            return Err(BackendError::new(
                K::BackendUnavailable,
                S::Backend,
                "physical owner control unavailable",
            ));
        }
        Ok(PhysicalRecoveryPending {
            reply: receiver,
            deadline,
            result: None,
        })
    }
    /// Retire a physically confirmed and already exited original worker. The
    /// registry is never locked during thread join or owner destruction.
    pub fn finish_confirmed(&self) -> Poll<Result<(), BackendError>> {
        let handle = {
            let slots = recovery_slots().lock().unwrap_or_else(|e| e.into_inner());
            let Some(control) = slots
                .get(self.index)
                .and_then(Option::as_ref)
                .filter(|slot| Arc::ptr_eq(&slot.identity, &self.identity))
                .and_then(|slot| slot.control.as_ref())
            else {
                return Poll::Ready(Err(BackendError::new(
                    K::InvalidInput,
                    S::Admission,
                    "physical recovery owner unavailable",
                )));
            };
            if control.state.load(Ordering::Acquire) != CLOSED {
                return Poll::Pending;
            }
            Arc::clone(&control.handle)
        };
        let finished = {
            let mut guard = handle.lock().unwrap_or_else(|e| e.into_inner());
            if !guard.as_ref().is_some_and(JoinHandle::is_finished) {
                return Poll::Pending;
            }
            guard.take().expect("finished handle exists")
        };
        let result = match finished.join() {
            Ok(result) => result,
            Err(payload) => Err(shutdown_panic_failure(payload.as_ref(), "WorkerJoinPanic")),
        };
        if result.is_ok() {
            let mut slots = recovery_slots().lock().unwrap_or_else(|e| e.into_inner());
            if slots
                .get(self.index)
                .and_then(Option::as_ref)
                .is_some_and(|slot| Arc::ptr_eq(&slot.identity, &self.identity))
            {
                slots[self.index] = None;
            }
        }
        Poll::Ready(result)
    }
}
impl PhysicalRecoveryPending {
    pub fn poll(&mut self) -> Poll<PhysicalRecoveryObservation> {
        if let Some(result) = self.result {
            return Poll::Ready(result);
        }
        let result = match self.reply.try_recv() {
            Ok(observation) => observation,
            Err(mpsc::TryRecvError::Disconnected) => PhysicalRecoveryObservation::Unknown,
            Err(mpsc::TryRecvError::Empty) if Instant::now() >= self.deadline => {
                PhysicalRecoveryObservation::Unknown
            }
            Err(mpsc::TryRecvError::Empty) => return Poll::Pending,
        };
        self.result = Some(result);
        Poll::Ready(result)
    }
}

struct Job<J, R> {
    input: Arc<J>,
    completion: mpsc::SyncSender<R>,
    quarantine_cause: Arc<Mutex<Option<BackendError>>>,
    #[cfg(feature = "contracts")]
    work_scope: Option<PhysicalWorkScope>,
}

pub struct SingleWorker<J, R> {
    #[cfg(feature = "experimental-notify")]
    signal: rz_runtime::CompletionSignal,
    sender: Option<mpsc::SyncSender<Job<J, R>>>,
    state: Arc<AtomicU8>,
    // The slot, rather than the spawn closure, owns the handle until a reaper
    // actually starts. A failed reaper spawn must not detach the native thread.
    physical_handle: PhysicalHandle,
    shutdown_completion: Option<mpsc::Receiver<Result<(), BackendError>>>,
    shutdown_result: Option<Result<(), BackendError>>,
    retained_recovery: Option<PhysicalRecoveryKey>,
    released_session: Arc<Mutex<Option<SessionReleased>>>,
}

pub struct PhysicalLease<J, R> {
    input: Arc<J>,
    completion: mpsc::Receiver<R>,
    consumed: bool,
    quarantine_cause: Arc<Mutex<Option<BackendError>>>,
    #[cfg(feature = "contracts")]
    work_scope: Option<PhysicalWorkScope>,
}

pub enum PhysicalRun<R> {
    /// Native execution has physically completed and owns its output.
    Complete(R),
    /// Physical completion is unknown. Retain this job and backend until process
    /// exit and publish the original typed failure without permitting reuse.
    Quarantined(BackendError),
}

pub enum PhysicalPoll<R> {
    Pending,
    Ready(R),
    /// Native completion is unknown. The job and backend remain pinned; runtime
    /// must quarantine this lease and must not release/reuse its reservations.
    Quarantined,
    Consumed,
}

impl<J: Send + Sync + 'static, R: Send + 'static> SingleWorker<J, R> {
    pub fn physical_recovery_key(&self) -> Option<PhysicalRecoveryKey> {
        self.retained_recovery.clone()
    }
    /// Only available after same-owner destruction and successful worker join.
    pub fn session_release(&self) -> Option<SessionReleased> {
        if !matches!(self.shutdown_result.as_ref(), Some(Ok(()))) { return None; }
        *self.released_session.lock().unwrap_or_else(|e| e.into_inner())
    }
    /// Transfer the actual caller-owned state only after preserving a recovery
    /// slot for thread creation failure. The shared lock is used for this
    /// one-time transfer, not in the physical request loop. Neither successful
    /// spawn nor thread join is itself a native/GPU completion fence.
    ///
    /// `shutdown_fence` must attest completion for this very state; false or
    /// unwind retains it and reports unknown rather than releasing it.
    pub fn spawn_with_retained_state<T, F, G>(
        original: &mut Option<T>,
        reservation: &mut PhysicalRecoveryReservation,
        run: F,
        mut shutdown_fence: G,
    ) -> Result<Self, BackendError>
    where
        T: Send + 'static,
        F: FnMut(&mut T, &J) -> PhysicalRun<R> + Send + 'static,
        G: FnMut(&mut T, Instant) -> PhysicalRecoveryFence + Send + 'static,
    {
        Self::spawn_with_retained_state_inner(original, reservation, None, run, move |owner, until| ShutdownEvidence::Recovery(shutdown_fence(owner, until)))
    }

    /// Register the identity of the same original owner before moving it to
    /// the worker. A synchronous proof is valid only for normal shutdown;
    /// it is never accepted as a recovery fence after a physical failure.
    pub fn spawn_with_retained_synchronous_owner<T, F, G>(
        original: &mut Option<T>,
        reservation: &mut PhysicalRecoveryReservation,
        run: F,
        mut shutdown_fence: G,
    ) -> Result<Self, BackendError>
    where
        T: SynchronousPhysicalOwner + Send + 'static,
        F: FnMut(&mut T, &J) -> PhysicalRun<R> + Send + 'static,
        G: FnMut(&mut T, Instant) -> Option<SynchronousOwnerProof> + Send + 'static,
    {
        let owner_identity = original.as_ref().ok_or_else(|| BackendError::new(
            K::InvalidInput, S::Admission, "missing original physical state",
        ))?.synchronous_owner_identity();
        Self::spawn_with_retained_state_inner(original, reservation, Some(owner_identity), run, move |owner, until| match shutdown_fence(owner, until) {
            Some(proof) => ShutdownEvidence::Session(proof),
            None => ShutdownEvidence::Recovery(PhysicalRecoveryFence::Unknown),
        })
    }

    fn spawn_with_retained_state_inner<T, F, G>(
        original: &mut Option<T>,
        reservation: &mut PhysicalRecoveryReservation,
        synchronous_owner: Option<Arc<()>>,
        mut run: F,
        mut shutdown_fence: G,
    ) -> Result<Self, BackendError>
    where
        T: Send + 'static,
        F: FnMut(&mut T, &J) -> PhysicalRun<R> + Send + 'static,
        G: FnMut(&mut T, Instant) -> ShutdownEvidence + Send + 'static,
    {
        if original.is_none() {
            return Err(BackendError::new(
                K::InvalidInput,
                S::Admission,
                "missing original physical state",
            ));
        }
        let (sender, receiver) = mpsc::sync_channel::<Job<J, R>>(1);
        let state = Arc::new(AtomicU8::new(IDLE));
        let worker_state = Arc::clone(&state);
        #[cfg(feature = "experimental-notify")]
        let signal = rz_runtime::CompletionSignal::default();
        #[cfg(feature = "experimental-notify")]
        let worker_signal = signal.clone();
        let (control_sender, control_receiver) = mpsc::sync_channel::<RecoveryCommand>(1);
        let pending = Arc::new(AtomicBool::new(false));
        let worker_pending = Arc::clone(&pending);
        let confirmation = Arc::new(Mutex::new(None));
        let worker_confirmation = Arc::clone(&confirmation);
        let worker_identity = Arc::clone(&reservation.key.identity);
        let released_session = Arc::new(Mutex::new(None));
        let worker_released_session = Arc::clone(&released_session);
        let handle_slot: PhysicalHandle = Arc::new(Mutex::new(None));
        reservation
            .publish(RecoveryControl {
                sender: control_sender,
                pending,
                state: Arc::clone(&state),
                handle: Arc::clone(&handle_slot),
                confirmation,
            })
            .map_err(|_| {
                BackendError::new(
                    K::BackendUnavailable,
                    S::Admission,
                    "physical recovery slot unavailable",
                )
            })?;
        let transfer = Arc::new(Mutex::new(original.take()));
        let worker_transfer = Arc::clone(&transfer);
        let started = std::thread::Builder::new()
            .name("rz-maia-physical".into())
            .spawn(move || {
                // The guard exists before callbacks or native operations can run.
                // Poison does not grant permission to abandon the actual owner.
                let original = worker_transfer
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .take();
                let Some(original) = original else {
                    worker_state.store(QUARANTINED, Ordering::Release);
                    return Err(BackendError::new(
                        K::BackendFailure,
                        S::Backend,
                        "physical startup owner missing",
                    ));
                };
                // Both the actual backend and the current job are outside the
                // unwind boundary. Nothing substitutes a reconstructed owner.
                let mut retained = original;
                let mut active: Option<Job<J, R>> = None;
                let execution = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    while let Ok(job) = receiver.recv() {
                        active = Some(job);
                        let outcome =
                            run(&mut retained, &active.as_ref().expect("active job").input);
                        match outcome {
                            PhysicalRun::Complete(output) => {
                                worker_state.store(IDLE, Ordering::Release);
                                let job = active.take().expect("active job");
                                let _ = job.completion.send(output);
                                #[cfg(feature = "experimental-notify")]
                                worker_signal.notify();
                            }
                            PhysicalRun::Quarantined(cause) => return Some(cause),
                        }
                    }
                    None
                }));
                let cause = match execution {
                    Ok(Some(cause)) => Some(cause),
                    Ok(None) => None,
                    Err(payload) => Some(panic_failure(payload.as_ref())),
                };
                if let Some(cause) = &cause {
                    if let Some(job) = &active {
                        *job.quarantine_cause
                            .lock()
                            .unwrap_or_else(|e| e.into_inner()) = Some(cause.clone());
                    }
                }
                // Normal sender closure can also encounter an unproven CUDA
                // startup. The callback must be a real same-owner fence.
                let mut session_release = None;
                let known = cause.is_none()
                    && std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        match shutdown_fence(&mut retained, Instant::now()) {
                            ShutdownEvidence::Recovery(fence) => fence.confirmed(),
                            ShutdownEvidence::Session(proof) if synchronous_owner.as_ref().is_some_and(|owner| Arc::ptr_eq(owner, &proof.owner)) => {
                                session_release = Some(SessionReleased { completed_synchronous_runs: proof.completed_synchronous_runs });
                                true
                            }
                            ShutdownEvidence::Session(_) => false,
                        }
                    }))
                    .unwrap_or(false);
                if !known {
                    worker_state.store(QUARANTINED, Ordering::Release);
                    #[cfg(feature = "experimental-notify")]
                    worker_signal.notify();
                    // Registry retains the sender and JoinHandle independently
                    // of a logical caller. Keep the exact original values on
                    // this physical owner thread until its fence confirms.
                    loop {
                        let Ok(command) = control_receiver.recv() else {
                            // A disconnected registry violates ownership; do
                            // not claim completion or release original values.
                            std::thread::park();
                            continue;
                        };
                        match command {
                            RecoveryCommand::Observe {
                                deadline,
                                request,
                                reply,
                            } => {
                                // The new observation may itself enter a
                                // native path. No proof from the prior request
                                // remains authority after that entry.
                                *worker_confirmation
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner()) = None;
                                let confirmed = Instant::now() < deadline
                                    && std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                                        || matches!(shutdown_fence(&mut retained, deadline), ShutdownEvidence::Recovery(fence) if fence.confirmed()),
                                    ))
                                    .unwrap_or(false);
                                let late = Instant::now() >= deadline;
                                if confirmed {
                                    *worker_confirmation
                                        .lock()
                                        .unwrap_or_else(|e| e.into_inner()) =
                                        Some(PhysicalRecoveryConfirmation {
                                            owner: Arc::clone(&worker_identity),
                                            request,
                                            late,
                                            #[cfg(feature = "contracts")]
                                            work_scope: active.as_ref().and_then(|job| job.work_scope),
                                        });
                                }
                                worker_pending.store(false, Ordering::Release);
                                let _ = reply.send(if confirmed && !late {
                                    PhysicalRecoveryObservation::Confirmed
                                } else {
                                    PhysicalRecoveryObservation::Unknown
                                });
                            }
                            RecoveryCommand::Release {
                                deadline,
                                request,
                                reply,
                            } => {
                                let valid = Instant::now() < deadline
                                    && worker_confirmation
                                        .lock()
                                        .unwrap_or_else(|e| e.into_inner())
                                        .as_ref()
                                        .is_some_and(|proof| Arc::ptr_eq(&proof.request, &request));
                                if valid {
                                    drop(active.take());
                                    drop(retained);
                                    worker_state.store(CLOSED, Ordering::Release);
                                    #[cfg(feature = "experimental-notify")]
                                    worker_signal.notify();
                                    let _ = reply.send(PhysicalRecoveryObservation::Confirmed);
                                    worker_pending.store(false, Ordering::Release);
                                    return Ok(());
                                }
                                worker_pending.store(false, Ordering::Release);
                                let _ = reply.send(PhysicalRecoveryObservation::Unknown);
                            }
                        }
                    }
                }
                // Drop the same original active job and backend only after the
                // physical owner has confirmed its own completion.
                drop(active);
                drop(retained);
                *worker_released_session.lock().unwrap_or_else(|e| e.into_inner()) = session_release;
                worker_state.store(CLOSED, Ordering::Release);
                #[cfg(feature = "experimental-notify")]
                worker_signal.notify();
                Ok(())
            });
        let physical_handle = match started {
            Ok(handle) => handle,
            Err(error) => {
                // spawn failure means the closure never executed. Recover the
                // same owner from the original slot, without substituting one.
                *original = transfer.lock().unwrap_or_else(|e| e.into_inner()).take();
                reservation.unpublish_before_spawn();
                return Err(BackendError::new(
                    K::BackendUnavailable,
                    S::Backend,
                    "cannot create retained physical worker",
                )
                .with_external_cause(CauseCode::RuntimeInitialize, &error));
            }
        };
        *handle_slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(physical_handle);
        Ok(Self {
            #[cfg(feature = "experimental-notify")]
            signal,
            sender: Some(sender),
            state,
            physical_handle: handle_slot,
            shutdown_completion: None,
            shutdown_result: None,
            retained_recovery: Some(reservation.key.clone()),
            released_session,
        })
    }
    /// run must return only after physical completion, with owned outputs. There
    /// is one admitted job and no hidden waiting queue. Dropping this handle
    /// closes admission; an active worker continues owning its job until done.
    pub fn spawn<F>(mut run: F) -> Result<Self, BackendError>
    where
        F: FnMut(&J) -> R + Send + 'static,
    {
        Self::spawn_with_outcome(move |input| PhysicalRun::Complete(run(input)))
    }

    /// Complete must attest physical completion, including owned output. An
    /// explicit Quarantined outcome or unwind permanently closes admission and
    /// retains this job and run closure, preserving the job's original cause.
    /// Resources still in native use must already be owned by J or captured
    /// closure state: callback-local resources are not retained after return.
    pub fn spawn_with_outcome<F>(mut run: F) -> Result<Self, BackendError>
    where
        F: FnMut(&J) -> PhysicalRun<R> + Send + 'static,
    {
        let (sender, receiver) = mpsc::sync_channel::<Job<J, R>>(1);
        let state = Arc::new(AtomicU8::new(IDLE));
        let worker_state = Arc::clone(&state);
        #[cfg(feature = "experimental-notify")]
        let signal = rz_runtime::CompletionSignal::default();
        #[cfg(feature = "experimental-notify")]
        let worker_signal = signal.clone();
        let physical_handle = std::thread::Builder::new()
            .name("rz-maia-physical".into())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    let outcome =
                        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            run(&job.input)
                        })) {
                            Ok(outcome) => outcome,
                            Err(payload) => {
                                PhysicalRun::Quarantined(panic_failure(payload.as_ref()))
                            }
                        };
                    match outcome {
                        PhysicalRun::Complete(output) => {
                            // Even a dropped logical consumer does not cancel native
                            // execution. At this point run has physically completed.
                            worker_state.store(IDLE, Ordering::Release);
                            let _ = job.completion.send(output);
                            #[cfg(feature = "experimental-notify")]
                            worker_signal.notify();
                        }
                        PhysicalRun::Quarantined(cause) => {
                            // Neither unwinding nor a native failure alone attests
                            // physical completion or GPU synchronization.
                            // Deliberately retain backend/inputs until process exit;
                            // no retry, no Ready, no reservation-release permission.
                            // Per-job receipt: a later job's failure must not be
                            // attributed to an earlier completed physical lease.
                            match job.quarantine_cause.lock() {
                                Ok(mut slot) => *slot = Some(cause.clone()),
                                Err(poisoned) => *poisoned.into_inner() = Some(cause.clone()),
                            }
                            std::mem::forget(job);
                            std::mem::forget(run);
                            worker_state.store(QUARANTINED, Ordering::Release);
                            #[cfg(feature = "experimental-notify")]
                            worker_signal.notify();
                            // Thread termination does not prove the quarantined
                            // execution completed, nor release its leaked pins.
                            return Err(cause);
                        }
                    }
                }
                // Session/closure destruction is part of process shutdown,
                // not of an individual physical lease's Ready transition.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(run)))
                    .map_err(|payload| {
                        shutdown_panic_failure(payload.as_ref(), "WorkerShutdownPanic")
                    });
                worker_state.store(CLOSED, Ordering::Release);
                #[cfg(feature = "experimental-notify")]
                worker_signal.notify();
                result
            })
            .map_err(|_| {
                BackendError::new(
                    K::BackendUnavailable,
                    S::Backend,
                    "cannot create physical worker",
                )
            })?;
        Ok(Self {
            #[cfg(feature = "experimental-notify")]
            signal,
            sender: Some(sender),
            state,
            physical_handle: Arc::new(Mutex::new(Some(physical_handle))),
            shutdown_completion: None,
            shutdown_result: None,
            retained_recovery: None,
            released_session: Arc::new(Mutex::new(None)),
        })
    }

    #[cfg(feature = "experimental-notify")]
    pub fn completion_signal(&self) -> rz_runtime::CompletionSignal {
        self.signal.clone()
    }

    /// A rejection guarantees this input was not handed to native execution.
    pub fn submit(&mut self, input: J) -> Result<PhysicalLease<J, R>, BackendError> {
        self.submit_inner(input, #[cfg(feature = "contracts")] None)
    }

    /// The original command is the only source of the work scope. The scope
    /// is copied to both the queued job and its matching lease in one submit.
    #[cfg(feature = "contracts")]
    pub fn submit_scoped(&mut self, input: J) -> Result<PhysicalLease<J, R>, BackendError>
    where
        J: PhysicalScopedInput,
    {
        let scope = input.physical_work_scope()?.validate()?;
        self.submit_inner(input, Some(scope))
    }

    fn submit_inner(
        &mut self,
        input: J,
        #[cfg(feature = "contracts")] work_scope: Option<PhysicalWorkScope>,
    ) -> Result<PhysicalLease<J, R>, BackendError> {
        let sender = self.sender.as_ref().ok_or_else(|| {
            BackendError::new(
                K::BackendUnavailable,
                S::Admission,
                "physical worker process shutdown has closed admission",
            )
        })?;
        self.state
            .compare_exchange(IDLE, BUSY, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|state| {
                BackendError::new(
                    if state == BUSY {
                        K::ResourceExhausted
                    } else {
                        K::BackendUnavailable
                    },
                    S::Admission,
                    "physical worker is busy, closed or quarantined",
                )
            })?;
        let (completion, receiver) = mpsc::sync_channel(1);
        let input = Arc::new(input);
        let quarantine_cause = Arc::new(Mutex::new(None));
        if sender
            .try_send(Job {
                input: Arc::clone(&input),
                completion,
                quarantine_cause: Arc::clone(&quarantine_cause),
                #[cfg(feature = "contracts")]
                work_scope,
            })
            .is_err()
        {
            self.state.store(CLOSED, Ordering::Release);
            return Err(BackendError::new(
                K::BackendUnavailable,
                S::Admission,
                "physical worker disconnected before launch",
            ));
        }
        Ok(PhysicalLease {
            input,
            completion: receiver,
            consumed: false,
            quarantine_cause,
            #[cfg(feature = "contracts")]
            work_scope,
        })
    }

    /// Terminal, process-only shutdown. Close future admission once, then poll a
    /// single reaper's result without blocking this caller on native destruction
    /// or thread TLS. A successful join is reported only after the native thread
    /// has fully exited; quarantined execution never becomes successful here.
    /// Dropping this owner or reaching a caller deadline does not terminate the
    /// physical thread, invalidate retained inputs, or make a quarantine Ready.
    pub fn try_shutdown(&mut self) -> Poll<Result<(), BackendError>> {
        if let Some(result) = &self.shutdown_result {
            return Poll::Ready(result.clone());
        }
        if let Some(key) = &self.retained_recovery {
            // Do not hand the original JoinHandle to a reaper while the
            // process-local registry needs it for an unknown owner.
            self.sender.take();
            match self.state.load(Ordering::Acquire) {
                QUARANTINED => {
                    return Poll::Ready(Err(BackendError::new(
                        K::BackendFailure,
                        S::Backend,
                        "physical completion unknown; recovery owner retained",
                    )));
                }
                CLOSED => {
                    let result = key.finish_confirmed();
                    if let Poll::Ready(value) = &result {
                        self.shutdown_result = Some(value.clone());
                    }
                    return result;
                }
                _ => return Poll::Pending,
            }
        }
        self.try_shutdown_with_start(|slot, completed| {
            std::thread::Builder::new()
                .name("rz-maia-reaper".into())
                .spawn(move || reap_physical(slot, completed))
        })
    }

    fn try_shutdown_with_start<F>(&mut self, start: F) -> Poll<Result<(), BackendError>>
    where
        F: FnOnce(PhysicalHandle, ShutdownSender) -> std::io::Result<JoinHandle<()>>,
    {
        if let Some(result) = &self.shutdown_result {
            return Poll::Ready(result.clone());
        }
        if self.shutdown_completion.is_none() {
            self.sender.take();
            let slot = Arc::clone(&self.physical_handle);
            let (completed, completion) = mpsc::sync_channel(1);
            let started = start(slot, completed);
            match started {
                Ok(reaper) => {
                    // This small reaper has no native state after it publishes
                    // the joined result. It may detach; the physical handle may
                    // not, and is owned by the retained slot/reaper until joined.
                    drop(reaper);
                    self.shutdown_completion = Some(completion);
                }
                Err(error) => {
                    let error = BackendError::new(
                        K::BackendUnavailable,
                        S::Backend,
                        "cannot create physical worker shutdown reaper",
                    )
                    .with_external_cause(CauseCode::RuntimeInitialize, &error);
                    // The failed spawn closure held only an Arc clone. The
                    // original JoinHandle remains in physical_handle, with no
                    // retry, forced drop or false completion acknowledgement.
                    self.shutdown_result = Some(Err(error.clone()));
                    return Poll::Ready(Err(error));
                }
            }
        }
        let result = match self.shutdown_completion.as_ref().unwrap().try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => return Poll::Pending,
            Err(mpsc::TryRecvError::Disconnected) => Err(BackendError::new(
                K::BackendFailure,
                S::Backend,
                "physical worker shutdown reaper disconnected without a result",
            )),
        };
        self.shutdown_result = Some(result.clone());
        Poll::Ready(result)
    }
}

fn reap_physical(slot: PhysicalHandle, completed: ShutdownSender) {
    // Release the slot guard before joining. The caller never waits for this
    // mutex or invokes JoinHandle::join itself, including after is_finished.
    let handle = match slot.lock() {
        Ok(mut slot) => slot.take(),
        Err(error) => {
            let _ = completed.send(Err(BackendError::new(
                K::BackendFailure,
                S::Backend,
                "physical worker join ownership lock is poisoned",
            )
            .with_external_cause(CauseCode::RuntimePanic, &error)));
            return;
        }
    };
    let result = match handle {
        Some(handle) => match handle.join() {
            Ok(result) => result,
            Err(payload) => Err(shutdown_panic_failure(payload.as_ref(), "WorkerJoinPanic")),
        },
        None => Err(BackendError::new(
            K::BackendFailure,
            S::Backend,
            "physical worker join ownership is missing",
        )),
    };
    let _ = completed.send(result);
}

impl<J, R> PhysicalLease<J, R> {
    pub fn input(&self) -> &J {
        &self.input
    }

    /// Same command-derived scope handed to the original physical worker.
    #[cfg(feature = "contracts")]
    pub fn work_scope(&self) -> Option<PhysicalWorkScope> {
        self.work_scope
    }

    /// Bounded per-job evidence, not completion or permission to release pins.
    /// The failure receipt is published before the QUARANTINED Release transition.
    /// Reading raw NativeDiagnostic text remains an explicit local operation.
    pub fn quarantine_cause(&self) -> Result<Option<BackendError>, BackendError> {
        self.quarantine_cause
            .lock()
            .map(|slot| slot.clone())
            .map_err(|error| {
                BackendError::new(
                    K::BackendFailure,
                    S::Backend,
                    "physical worker quarantine receipt lock is poisoned",
                )
                .with_external_cause(CauseCode::RuntimePanic, &error)
            })
    }

    /// Nonblocking. Each owned output can be obtained exactly once, independently
    /// of any logical request cancellation performed by the runtime.
    pub fn poll(&mut self) -> PhysicalPoll<R> {
        if self.consumed {
            return PhysicalPoll::Consumed;
        }
        match self.completion.try_recv() {
            Ok(result) => {
                self.consumed = true;
                PhysicalPoll::Ready(result)
            }
            Err(mpsc::TryRecvError::Empty) => self.poll_empty(),
            // Unexpected disconnection is also not proof of physical completion.
            Err(mpsc::TryRecvError::Disconnected) => PhysicalPoll::Quarantined,
        }
    }

    fn poll_empty(&self) -> PhysicalPoll<R> {
        // The worker's admission state can belong to a later job by now. Only
        // this job's own receipt may quarantine an observed Empty completion.
        // Its output may also have arrived since try_recv: Pending is safe and
        // the next poll will consume it; an unrelated job's panic is irrelevant.
        match self.quarantine_cause.lock() {
            Ok(slot) if slot.is_none() => PhysicalPoll::Pending,
            _ => PhysicalPoll::Quarantined,
        }
    }
}

fn panic_failure(payload: &(dyn std::any::Any + Send)) -> BackendError {
    panic_with_context(
        payload,
        "physical worker unwound; completion is unknown",
        "WorkerPanic",
    )
}

fn shutdown_panic_failure(
    payload: &(dyn std::any::Any + Send),
    code: &'static str,
) -> BackendError {
    panic_with_context(payload, "physical worker process shutdown unwound", code)
}

fn panic_with_context(
    payload: &(dyn std::any::Any + Send),
    detail: &'static str,
    code: &'static str,
) -> BackendError {
    let error = BackendError::new(K::BackendFailure, S::Backend, detail);
    if let Some(message) = payload.downcast_ref::<&str>() {
        error
            .with_external_cause(CauseCode::RuntimePanic, message)
            .with_diagnostic(code, message)
    } else if let Some(message) = payload.downcast_ref::<String>() {
        error
            .with_external_cause(CauseCode::RuntimePanic, message)
            .with_diagnostic(code, message)
    } else {
        // Any need not implement Display/Debug; do not invent its original text.
        let unknown = "non-string physical worker panic payload";
        error
            .with_external_cause(CauseCode::RuntimePanic, &unknown)
            .with_diagnostic(
                if code == "WorkerPanic" {
                    "WorkerPanicNonString"
                } else {
                    code
                },
                unknown,
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::time::{Duration, Instant};

    struct TrackedState(Arc<AtomicUsize>);
    impl Drop for TrackedState {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct SyncTrackedState {
        _tracked: TrackedState,
        identity: Arc<()>,
    }
    impl synchronous_owner_sealed::Sealed for SyncTrackedState {}
    impl SynchronousPhysicalOwner for SyncTrackedState {
        fn synchronous_owner_identity(&self) -> Arc<()> {
            Arc::clone(&self.identity)
        }
    }

    fn wait_until(mut check: impl FnMut() -> bool) {
        let stop = Instant::now() + Duration::from_secs(2);
        while !check() {
            assert!(
                Instant::now() < stop,
                "worker race fixture exceeded its budget"
            );
            std::thread::yield_now();
        }
    }

    #[test]
    fn unpublished_reservation_cancel_requires_original_empty_slot() {
        let (reservation, key) = PhysicalRecoveryReservation::reserve().expect("reserve");
        let original_identity = {
            let mut slots = recovery_slots().lock().unwrap_or_else(|e| e.into_inner());
            let slot = slots[key.index].as_mut().expect("registered");
            std::mem::replace(&mut slot.identity, Arc::new(()))
        };
        let reservation = match reservation.cancel_unpublished() {
            Err(owner) => owner,
            Ok(()) => panic!("different owner was removed"),
        };
        {
            let mut slots = recovery_slots().lock().unwrap_or_else(|e| e.into_inner());
            let slot = slots[key.index].as_mut().expect("mismatched slot preserved");
            slot.identity = original_identity;
            let (sender, _receiver) = mpsc::sync_channel(1);
            slot.control = Some(RecoveryControl {
                sender,
                pending: Arc::new(AtomicBool::new(false)),
                state: Arc::new(AtomicU8::new(0)),
                handle: Arc::new(Mutex::new(None)),
                confirmation: Arc::new(Mutex::new(None)),
            });
        }
        let mut reservation = match reservation.cancel_unpublished() {
            Err(owner) => owner,
            Ok(()) => panic!("slot with control was removed"),
        };
        reservation.published = true;
        {
            let mut slots = recovery_slots().lock().unwrap_or_else(|e| e.into_inner());
            slots[key.index].as_mut().expect("registered").control = None;
        }
        let mut reservation = match reservation.cancel_unpublished() {
            Err(owner) => owner,
            Ok(()) => panic!("published reservation was removed"),
        };
        reservation.published = false;
        assert!(reservation.cancel_unpublished().is_ok());
        let slots = recovery_slots().lock().unwrap_or_else(|e| e.into_inner());
        assert!(!slots[key.index].as_ref().is_some_and(|slot| Arc::ptr_eq(&slot.identity, &key.identity)));
    }

    #[cfg(feature = "contracts")]
    fn scoped_ticket() -> rz_contracts::pals_adaptive::RoleTicketEnvelope {
        use rz_contracts::pals::{RepresentationKey, Role, SearchAuthority};
        use rz_contracts::pals_adaptive::{ReviewViewSeal, RoleTicketEnvelope};
        use rz_contracts::{ClockDomain, Deadline, Digest, GameGeneration, MonotonicTick,
            OwnerId, PrecisionProfile, ProcessEpoch, RequestId, RootGeneration, StateIdentity,
            StateRevision};
        let epoch = ProcessEpoch(17);
        let authority = SearchAuthority { epoch, game: GameGeneration(2), root: RootGeneration(3), implementation: Digest([1; 32]) };
        let role = Role::Proposer;
        RoleTicketEnvelope {
            id: RequestId::new(epoch, 4), authority, question: Digest([2; 32]),
            view: ReviewViewSeal {
                authority, view_sequence: 1, context_policy: Digest([3; 32]), view_digest: Digest([4; 32]),
                root_state: StateIdentity { owner: OwnerId(5), revision: StateRevision(6), semantic: Digest([7; 32]) },
                actual_tokens: 0, selected_records: 0, anchors: 1, length_bucket: 8,
            },
            representation: RepresentationKey {
                input: Digest([8; 32]), history: Digest([9; 32]), candidates: Digest([10; 32]),
                mask_positions: Digest([11; 32]), model: Digest([12; 32]), encoding: Digest([13; 32]),
                precision: PrecisionProfile::Fp32, frozen_epoch: 1, record_revision: 1, role,
            },
            role, original_deadline: Deadline { clock: ClockDomain(epoch), at: MonotonicTick(1_000_000) },
        }
    }

    #[cfg(feature = "contracts")]
    #[test]
    fn scoped_recovery_proof_matches_actual_ticket_execution_and_root() {
        use rz_contracts::{ExecutionId, ProcessEpoch, RootGeneration};
        struct ScopedCommand(PhysicalWorkScope);
        impl PhysicalScopedInput for ScopedCommand {
            fn physical_work_scope(&self) -> Result<PhysicalWorkScope, BackendError> { Ok(self.0) }
        }
        let ticket = scoped_ticket();
        let execution = ExecutionId::new(ticket.id.epoch, 9);
        let expected = PhysicalWorkScope::Ticket { ticket, execution };
        let mut original = Some(());
        let (mut reservation, key) = PhysicalRecoveryReservation::reserve().unwrap();
        let mut worker = SingleWorker::<ScopedCommand, ()>::spawn_with_retained_state(
            &mut original, &mut reservation,
            |_, _| PhysicalRun::Quarantined(BackendError::new(K::BackendFailure, S::Backend, "fixture work unknown")),
            |_, _| PhysicalRecoveryFence::FixtureOwnerCompleted,
        ).unwrap();
        let mut invalid_ticket = ticket;
        invalid_ticket.id.epoch = ProcessEpoch(18);
        assert!(worker.submit_scoped(ScopedCommand(PhysicalWorkScope::Ticket { ticket: invalid_ticket, execution })).is_err());
        let mut lease = worker.submit_scoped(ScopedCommand(expected)).unwrap();
        assert_eq!(lease.work_scope(), Some(expected));
        wait_until(|| matches!(lease.poll(), PhysicalPoll::Quarantined));
        let mut pending = key.begin_reobserve(Instant::now() + Duration::from_secs(2)).unwrap();
        wait_until(|| matches!(pending.poll(), Poll::Ready(PhysicalRecoveryObservation::Confirmed)));
        let proof = key.latest_confirmation().expect("actual owner observation");
        assert_eq!(proof.work_scope(), Some(expected));
        assert!(key.confirms_work_scope(&proof, expected));
        let mut other_ticket = ticket;
        other_ticket.id.sequence += 1;
        assert!(!key.confirms_work_scope(&proof, PhysicalWorkScope::Ticket { ticket: other_ticket, execution }));
        assert!(!key.confirms_work_scope(&proof, PhysicalWorkScope::Ticket { ticket, execution: ExecutionId::new(ticket.id.epoch, 10) }));
        let mut other_root = ticket;
        other_root.authority.root = RootGeneration(99);
        other_root.view.authority = other_root.authority;
        assert!(!key.confirms_work_scope(&proof, PhysicalWorkScope::Ticket { ticket: other_root, execution }));
        assert!(!key.confirms_work_scope(&proof, PhysicalWorkScope::Startup { epoch: ticket.id.epoch }));
        let mut release = key.begin_release(&proof, Instant::now() + Duration::from_secs(2)).unwrap();
        wait_until(|| matches!(release.poll(), Poll::Ready(PhysicalRecoveryObservation::Confirmed)));
        wait_until(|| matches!(worker.try_shutdown(), Poll::Ready(Ok(()))));
    }

    #[test]
    fn synchronous_owner_proof_only_confirms_matching_normal_shutdown() {
      for count in [0, 7] {
        let drops = Arc::new(AtomicUsize::new(0));
        let owner = Arc::new(());
        let mut original = Some(SyncTrackedState { _tracked: TrackedState(Arc::clone(&drops)), identity: Arc::clone(&owner) });
        let (mut reservation, _) = PhysicalRecoveryReservation::reserve().unwrap();
        let callback_owner = Arc::clone(&owner);
        let mut worker = SingleWorker::<u8, u8>::spawn_with_retained_synchronous_owner(
            &mut original, &mut reservation,
            |_, input| PhysicalRun::Complete(*input),
            move |_, _| Some(SynchronousOwnerProof::new(Arc::clone(&callback_owner), count)),
        ).unwrap();
        assert!(worker.session_release().is_none());
        if count != 0 { let mut job = worker.submit(7).unwrap(); wait_until(|| matches!(job.poll(), PhysicalPoll::Ready(7))); }
        wait_until(|| matches!(worker.try_shutdown(), Poll::Ready(Ok(()))));
        assert_eq!(worker.session_release(), Some(SessionReleased { completed_synchronous_runs: count }));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
      }
    }

    #[test]
    fn synchronous_owner_proof_does_not_confirm_foreign_owner_or_quarantined_work() {
        for quarantined in [false, true] {
            let drops = Arc::new(AtomicUsize::new(0));
            let owner = Arc::new(());
            let mut original = Some(SyncTrackedState { _tracked: TrackedState(Arc::clone(&drops)), identity: Arc::clone(&owner) });
            let (mut reservation, key) = PhysicalRecoveryReservation::reserve().unwrap();
            let callback_owner = if quarantined { Arc::clone(&owner) } else { Arc::new(()) };
            let fixture_release = Arc::new(AtomicBool::new(false));
            let callback_release = Arc::clone(&fixture_release);
            let mut worker = SingleWorker::<u8, u8>::spawn_with_retained_state_inner(
                &mut original, &mut reservation, Some(owner),
                move |_, input| if quarantined {
                    PhysicalRun::Quarantined(BackendError::new(K::BackendFailure, S::Backend, "fixture"))
                } else { PhysicalRun::Complete(*input) },
                move |_, _| if callback_release.load(Ordering::Acquire) {
                    ShutdownEvidence::Recovery(PhysicalRecoveryFence::FixtureOwnerCompleted)
                } else {
                    ShutdownEvidence::Session(SynchronousOwnerProof::new(Arc::clone(&callback_owner), 1))
                },
            ).unwrap();
            let mut job = worker.submit(7).unwrap();
            if quarantined {
                wait_until(|| matches!(job.poll(), PhysicalPoll::Quarantined));
            } else {
                wait_until(|| matches!(job.poll(), PhysicalPoll::Ready(7)));
            }
            wait_until(|| matches!(worker.try_shutdown(), Poll::Ready(Err(_))));
            assert!(worker.session_release().is_none());
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            let mut observation = key.begin_reobserve(Instant::now() + Duration::from_secs(2)).unwrap();
            wait_until(|| matches!(observation.poll(), Poll::Ready(PhysicalRecoveryObservation::Unknown)));
            assert!(key.latest_confirmation().is_none());
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            // The positive fixture is test-only; release this exact owner so the
            // test does not leave a worker or recovery slot running.
            fixture_release.store(true, Ordering::Release);
            let mut observation = key.begin_reobserve(Instant::now() + Duration::from_secs(2)).unwrap();
            wait_until(|| matches!(observation.poll(), Poll::Ready(PhysicalRecoveryObservation::Confirmed)));
            let proof = key.latest_confirmation().expect("fixture confirmation");
            let mut release = key.begin_release(&proof, Instant::now() + Duration::from_secs(2)).unwrap();
            wait_until(|| matches!(release.poll(), Poll::Ready(PhysicalRecoveryObservation::Confirmed)));
            wait_until(|| matches!(worker.try_shutdown(), Poll::Ready(Ok(()))));
            assert!(worker.session_release().is_none());
            assert_eq!(drops.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn synchronous_session_release_waits_for_drop_and_join() {
        struct BlockingDrop { identity: Arc<()>, entered: mpsc::SyncSender<()>, resume: mpsc::Receiver<()> }
        impl synchronous_owner_sealed::Sealed for BlockingDrop {}
        impl SynchronousPhysicalOwner for BlockingDrop {
            fn synchronous_owner_identity(&self) -> Arc<()> { Arc::clone(&self.identity) }
        }
        impl Drop for BlockingDrop {
            fn drop(&mut self) {
                self.entered.send(()).unwrap();
                self.resume.recv_timeout(Duration::from_secs(2)).unwrap();
            }
        }
        let identity = Arc::new(());
        let (entered, entered_rx) = mpsc::sync_channel(1);
        let (resume, resume_rx) = mpsc::sync_channel(1);
        let mut original = Some(BlockingDrop { identity: Arc::clone(&identity), entered, resume: resume_rx });
        let (mut reservation, _) = PhysicalRecoveryReservation::reserve().unwrap();
        let mut worker = SingleWorker::<u8,u8>::spawn_with_retained_synchronous_owner(
            &mut original, &mut reservation, |_, input| PhysicalRun::Complete(*input),
            move |_, _| Some(SynchronousOwnerProof::new(Arc::clone(&identity), 0)),
        ).unwrap();
        assert!(matches!(worker.try_shutdown(), Poll::Pending));
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(worker.session_release().is_none());
        assert!(matches!(worker.try_shutdown(), Poll::Pending));
        resume.send(()).unwrap();
        wait_until(|| matches!(worker.try_shutdown(), Poll::Ready(Ok(()))));
        assert_eq!(worker.session_release(), Some(SessionReleased { completed_synchronous_runs: 0 }));
    }

    #[test]
    fn synchronous_session_release_is_not_published_if_drop_panics() {
        struct PanicDrop(Arc<()>);
        impl synchronous_owner_sealed::Sealed for PanicDrop {}
        impl SynchronousPhysicalOwner for PanicDrop {
            fn synchronous_owner_identity(&self) -> Arc<()> { Arc::clone(&self.0) }
        }
        impl Drop for PanicDrop { fn drop(&mut self) { panic!("fixture drop failure"); } }
        let identity = Arc::new(());
        let mut original = Some(PanicDrop(Arc::clone(&identity)));
        let (mut reservation, key) = PhysicalRecoveryReservation::reserve().unwrap();
        let mut worker = SingleWorker::<u8,u8>::spawn_with_retained_synchronous_owner(
            &mut original, &mut reservation, |_, input| PhysicalRun::Complete(*input),
            move |_, _| Some(SynchronousOwnerProof::new(Arc::clone(&identity), 0)),
        ).unwrap();
        let _ = worker.try_shutdown();
        wait_until(|| worker.physical_handle.lock().unwrap().as_ref().is_some_and(JoinHandle::is_finished));
        assert!(worker.session_release().is_none());
        // Test-only cleanup: this fixture owns no native data and the exact
        // original worker has terminated; join and free its matching slot.
        let handle = worker.physical_handle.lock().unwrap().take().unwrap();
        assert!(handle.join().is_err());
        let mut slots = recovery_slots().lock().unwrap_or_else(|e| e.into_inner());
        assert!(slots[key.index].as_ref().is_some_and(|slot| Arc::ptr_eq(&slot.identity, &key.identity)));
        slots[key.index] = None;
    }

    #[test]
    fn retained_state_releases_only_after_the_supplied_fence() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut original = Some(TrackedState(Arc::clone(&drops)));
        let (mut reservation, _key) = PhysicalRecoveryReservation::reserve().unwrap();
        let mut worker = SingleWorker::<u8, u8>::spawn_with_retained_state(
            &mut original,
            &mut reservation,
            |_, input| PhysicalRun::Complete(*input),
            |_, _| PhysicalRecoveryFence::FixtureOwnerCompleted,
        )
        .unwrap();
        assert!(original.is_none());
        let mut job = worker.submit(7).unwrap();
        wait_until(|| matches!(job.poll(), PhysicalPoll::Ready(7)));
        wait_until(|| match worker.try_shutdown() {
            Poll::Ready(Ok(())) => true,
            Poll::Pending => false,
            Poll::Ready(Err(error)) => panic!("unexpected shutdown: {error}"),
        });
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn retained_state_unknown_shutdown_does_not_drop_original() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut original = Some(TrackedState(Arc::clone(&drops)));
        let (mut reservation, key) = PhysicalRecoveryReservation::reserve().unwrap();
        let fence = Arc::new(AtomicBool::new(false));
        let test_fence = Arc::clone(&fence);
        let mut worker = SingleWorker::<u8, u8>::spawn_with_retained_state(
            &mut original,
            &mut reservation,
            |_, input| PhysicalRun::Complete(*input),
            move |_, _| {
                if test_fence.load(Ordering::Acquire) {
                    PhysicalRecoveryFence::FixtureOwnerCompleted
                } else {
                    PhysicalRecoveryFence::Unknown
                }
            },
        )
        .unwrap();
        wait_until(|| match worker.try_shutdown() {
            Poll::Ready(Err(_)) => true,
            Poll::Pending => false,
            Poll::Ready(Ok(())) => panic!("unknown completion must not be confirmed"),
        });
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        let mut observation = key
            .begin_reobserve(Instant::now() + Duration::from_secs(2))
            .unwrap();
        wait_until(|| {
            matches!(
                observation.poll(),
                Poll::Ready(PhysicalRecoveryObservation::Unknown)
            )
        });
        assert!(matches!(
            observation.poll(),
            Poll::Ready(PhysicalRecoveryObservation::Unknown)
        ));
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        fence.store(true, Ordering::Release);
        let mut observation = key
            .begin_reobserve(Instant::now() + Duration::from_secs(2))
            .unwrap();
        wait_until(|| {
            matches!(
                observation.poll(),
                Poll::Ready(PhysicalRecoveryObservation::Confirmed)
            )
        });
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        let old_proof = key.latest_confirmation().expect("fixture confirmation");
        fence.store(false, Ordering::Release);
        let mut observation = key
            .begin_reobserve(Instant::now() + Duration::from_secs(2))
            .unwrap();
        wait_until(|| {
            matches!(
                observation.poll(),
                Poll::Ready(PhysicalRecoveryObservation::Unknown)
            )
        });
        assert!(key.latest_confirmation().is_none());
        assert!(
            key.begin_release(&old_proof, Instant::now() + Duration::from_secs(2))
                .is_err()
        );
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        fence.store(true, Ordering::Release);
        let mut observation = key
            .begin_reobserve(Instant::now() + Duration::from_secs(2))
            .unwrap();
        wait_until(|| {
            matches!(
                observation.poll(),
                Poll::Ready(PhysicalRecoveryObservation::Confirmed)
            )
        });
        let proof = key.latest_confirmation().expect("new fixture confirmation");
        let mut release = key
            .begin_release(&proof, Instant::now() + Duration::from_secs(2))
            .unwrap();
        wait_until(|| {
            matches!(
                release.poll(),
                Poll::Ready(PhysicalRecoveryObservation::Confirmed)
            )
        });
        wait_until(|| matches!(worker.try_shutdown(), Poll::Ready(Ok(()))));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn retained_state_unwind_keeps_original_and_job() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut original = Some(TrackedState(Arc::clone(&drops)));
        let (mut reservation, key) = PhysicalRecoveryReservation::reserve().unwrap();
        let mut worker = SingleWorker::<u8, u8>::spawn_with_retained_state(
            &mut original,
            &mut reservation,
            |_, _| panic!("authored physical callback panic"),
            |_, _| PhysicalRecoveryFence::FixtureOwnerCompleted,
        )
        .unwrap();
        let mut job = worker.submit(1).unwrap();
        wait_until(|| matches!(job.poll(), PhysicalPoll::Quarantined));
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert!(worker.submit(2).is_err());
        let mut observation = key
            .begin_reobserve(Instant::now() + Duration::from_secs(2))
            .unwrap();
        wait_until(|| {
            matches!(
                observation.poll(),
                Poll::Ready(PhysicalRecoveryObservation::Confirmed)
            )
        });
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        let proof = key.latest_confirmation().expect("fixture confirmation");
        let mut release = key
            .begin_release(&proof, Instant::now() + Duration::from_secs(2))
            .unwrap();
        wait_until(|| {
            matches!(
                release.poll(),
                Poll::Ready(PhysicalRecoveryObservation::Confirmed)
            )
        });
        wait_until(|| matches!(worker.try_shutdown(), Poll::Ready(Ok(()))));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn retained_state_late_fence_stays_unknown_until_explicit_release() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut original = Some(TrackedState(Arc::clone(&drops)));
        let (mut reservation, key) = PhysicalRecoveryReservation::reserve().unwrap();
        let (entered, entered_rx) = mpsc::sync_channel(1);
        let (continue_tx, continue_rx) = mpsc::sync_channel(1);
        let mut worker = SingleWorker::<u8, u8>::spawn_with_retained_state(
            &mut original,
            &mut reservation,
            |_, input| PhysicalRun::Complete(*input),
            move |_, until| {
                // The ordinary initial shutdown has no usable fence window.
                if Instant::now() >= until {
                    return PhysicalRecoveryFence::Unknown;
                }
                entered.send(()).unwrap();
                continue_rx.recv_timeout(Duration::from_secs(2)).unwrap();
                PhysicalRecoveryFence::FixtureOwnerCompleted
            },
        )
        .unwrap();
        wait_until(|| matches!(worker.try_shutdown(), Poll::Ready(Err(_))));
        let deadline = Instant::now() + Duration::from_secs(1);
        let mut pending = key.begin_reobserve(deadline).unwrap();
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        // The channel holds the owner inside its fence callback until the
        // original request expires, without relying on a guessed sleep.
        while Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert!(matches!(
            pending.poll(),
            Poll::Ready(PhysicalRecoveryObservation::Unknown)
        ));
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        continue_tx.send(()).unwrap();
        let mut late = None;
        wait_until(|| {
            late = key.latest_confirmation();
            late.as_ref()
                .is_some_and(|proof| key.owns_confirmation(proof))
        });
        let proof = late.unwrap();
        assert!(proof.was_late());
        assert!(matches!(
            pending.poll(),
            Poll::Ready(PhysicalRecoveryObservation::Unknown)
        ));
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        let mut release = key
            .begin_release(&proof, Instant::now() + Duration::from_secs(2))
            .unwrap();
        wait_until(|| {
            matches!(
                release.poll(),
                Poll::Ready(PhysicalRecoveryObservation::Confirmed)
            )
        });
        wait_until(|| matches!(worker.try_shutdown(), Poll::Ready(Ok(()))));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn reaper_spawn_failure_retains_physical_handle_and_never_retries() {
        let mut worker = SingleWorker::spawn(|input: &u8| *input).unwrap();
        let original = match worker.try_shutdown_with_start(|_, _| {
            Err(std::io::Error::other("authored reaper spawn failure"))
        }) {
            Poll::Ready(Err(error)) => error,
            _ => panic!("failed spawn must publish its original typed failure"),
        };
        assert_eq!(original.kind, K::BackendUnavailable);
        assert_eq!(original.cause.unwrap().code, CauseCode::RuntimeInitialize);
        assert!(worker.physical_handle.lock().unwrap().is_some());
        assert!(worker.submit(1).is_err());
        assert!(matches!(worker.try_shutdown_with_start(|_, _| {
            panic!("a sticky spawn failure must never launch another reaper")
        }), Poll::Ready(Err(error)) if error == original));
        assert!(worker.physical_handle.lock().unwrap().is_some());
        // Test cleanup runs off the production caller path. The owned handle
        // stayed in its slot through failure and no production join blocked.
        let handle = worker.physical_handle.lock().unwrap().take().unwrap();
        handle.join().unwrap().unwrap();
    }

    #[test]
    fn an_empty_observation_stays_pending_when_a_later_job_quarantines() {
        let (entered, entering) = mpsc::sync_channel(1);
        let (release, released) = mpsc::sync_channel(1);
        let mut worker = SingleWorker::spawn(move |input: &u8| {
            if *input == 0 {
                entered.send(()).unwrap();
                released.recv_timeout(Duration::from_secs(2)).unwrap();
                return 0;
            }
            panic!("later job's injected unwind");
        })
        .unwrap();
        let mut first = worker.submit(0).unwrap();
        entering.recv_timeout(Duration::from_secs(2)).unwrap();
        // Freeze the exact interleaving between production poll's Empty receipt
        // and its Empty classification, without timing sleeps or a public hook.
        assert!(matches!(
            first.completion.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
        release.send(()).unwrap();
        let mut next = None;
        wait_until(|| match worker.submit(1) {
            Ok(lease) => {
                next = Some(lease);
                true
            }
            Err(error) => {
                assert_eq!(error.kind, K::ResourceExhausted);
                false
            }
        });
        let mut next = next.unwrap();
        wait_until(|| matches!(next.poll(), PhysicalPoll::Quarantined));
        wait_until(|| worker.state.load(Ordering::Acquire) == QUARANTINED);
        assert!(next.quarantine_cause().unwrap().is_some());
        assert!(first.quarantine_cause().unwrap().is_none());
        assert!(matches!(first.poll_empty(), PhysicalPoll::Pending));
        assert!(matches!(first.poll(), PhysicalPoll::Ready(0)));
        assert!(matches!(first.poll(), PhysicalPoll::Consumed));
    }
}
