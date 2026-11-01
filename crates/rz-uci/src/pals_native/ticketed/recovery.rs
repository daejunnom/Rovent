//! Process-local recovery of the original ticket worker. A key is not a GPU
//! fence, an external supervisor identity, or a cross-process device lease.
use std::any::Any;
use std::cell::RefCell;
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use rz_eval::worker::{PhysicalRecoveryConfirmation, PhysicalRecoveryKey, PhysicalWorkScope};
use super::service::{Shared, TicketNativeFinishReport, TicketNativeRecoveryProbe, TicketNativeRelease};
use super::service::TicketNativeEvent;

const SLOT_LIMIT: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TicketRecoveryStatus {
    Reserved,
    Running,
    BeforeNative,
    /// The worker never received its unique start token; no factory ran.
    StartNotDelivered,
    Unknown,
    Confirmed,
    Released,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TicketRecoveryError {
    Capacity,
    InvalidKey,
    WrongState,
    TimedOut,
    Disconnected,
    WrongOwnerType,
    WorkerPanicked,
    RetainedStatePending,
}

pub(crate) enum TicketRecoveryCommand {
    Observe { until: Instant, reply: mpsc::SyncSender<TicketRecoveryStatus> },
    Release { until: Instant, reply: mpsc::SyncSender<TicketRecoveryStatus> },
    TakeEvent { until: Instant, reply: mpsc::SyncSender<Option<TicketUndeliveredEvent>> },
    TakeAck { until: Instant, reply: mpsc::SyncSender<Option<TicketUndeliveredEvent>> },
    TakeState { until: Instant, reply: mpsc::SyncSender<Option<TicketUndeliveredEvent>> },
}

/// The owner and reader retain the same original event. A timeout or dropped
/// reply never removes it from the worker's custody.
#[derive(Clone)]
pub struct TicketUndeliveredEvent {
    event: Arc<Mutex<Option<Box<dyn Any + Send>>>>,
}
impl TicketUndeliveredEvent {
    pub(crate) fn new<T: Any + Send>(event: T) -> Self {
        Self { event: Arc::new(Mutex::new(Some(Box::new(event)))) }
    }
    pub(crate) fn is_pending(&self) -> bool {
        self.event.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }
    pub fn take<T: Any + Send>(&self) -> Result<Option<T>, TicketRecoveryError> {
        let mut original = self.event.lock().unwrap_or_else(|e| e.into_inner());
        let Some(event) = original.take() else { return Ok(None); };
        match event.downcast::<T>() {
            Ok(event) => Ok(Some(*event)),
            Err(event) => { *original = Some(event); Err(TicketRecoveryError::WrongOwnerType) }
        }
    }
}

enum JoinState {
    NotInstalled,
    Running(JoinHandle<()>),
    Joining,
    JoinedOk,
    JoinedErr,
}

/// Same single join slot is used by the service and recovery key. Failure is
/// sticky and cannot be reinterpreted as an already successful join.
pub(crate) struct TicketOwnerJoin {
    state: Mutex<JoinState>,
}
impl TicketOwnerJoin {
    pub(crate) fn new() -> Self { Self { state: Mutex::new(JoinState::NotInstalled) } }
    pub(crate) fn install(&self, thread: JoinHandle<()>) -> Result<(), JoinHandle<()>> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !matches!(*state, JoinState::NotInstalled) { return Err(thread); }
        *state = JoinState::Running(thread);
        Ok(())
    }
    pub(crate) fn try_join(&self) -> Result<bool, TicketRecoveryError> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match &*state {
            JoinState::NotInstalled | JoinState::Joining => return Ok(false),
            JoinState::Running(thread) if !thread.is_finished() => return Ok(false),
            JoinState::JoinedOk => return Ok(true),
            JoinState::JoinedErr => return Err(TicketRecoveryError::WorkerPanicked),
            JoinState::Running(_) => {}
        }
        let JoinState::Running(thread) = std::mem::replace(&mut *state, JoinState::Joining) else { unreachable!() };
        let joined = thread.join().is_ok();
        *state = if joined { JoinState::JoinedOk } else { JoinState::JoinedErr };
        if joined { Ok(true) } else { Err(TicketRecoveryError::WorkerPanicked) }
    }
    fn not_installed(&self) -> bool {
        matches!(*self.state.lock().unwrap_or_else(|e| e.into_inner()), JoinState::NotInstalled)
    }
}

struct Slot {
    identity: Arc<()>,
    status: TicketRecoveryStatus,
    sender: Option<mpsc::SyncSender<TicketRecoveryCommand>>,
    thread: Option<Arc<TicketOwnerJoin>>,
    before_native_owner: Option<Box<dyn Any + Send>>,
    before_native_events: Vec<TicketUndeliveredEvent>,
    observation_pending: bool,
    expected_inner: Option<PhysicalRecoveryKey>,
    original_epoch: Option<u64>,
    expected_work: Option<PhysicalWorkScope>,
    confirmation: Option<TicketRecoveryConfirmation>,
    queued_receiver: Option<Box<dyn OriginalEventReceiver>>,
    queued_event: Option<TicketUndeliveredEvent>,
    receiver_released: Arc<AtomicBool>,
    clean_finish: bool,
    release_in_progress: bool,
}

struct ReleaseGuard(TicketNativeRecoveryKey);
impl Drop for ReleaseGuard {
    fn drop(&mut self) {
        let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = slots.get_mut(self.0.index).and_then(Option::as_mut)
            .filter(|slot| Arc::ptr_eq(&slot.identity, &self.0.identity)) {
            slot.release_in_progress = false;
        }
    }
}

trait OriginalEventReceiver: Send {
    fn try_recv(&self) -> Option<TicketUndeliveredEvent>;
    fn take_finish_report(&self) -> Option<TicketNativeFinishReport>;
    fn has_finish_report(&self) -> bool;
}
struct TypedEventReceiver<O> {
    receiver: Arc<Mutex<mpsc::Receiver<TicketNativeEvent<O>>>>,
    shared: Arc<Shared>,
}
impl<O: Send + Sync + 'static> OriginalEventReceiver for TypedEventReceiver<O> {
    fn take_finish_report(&self) -> Option<TicketNativeFinishReport> { self.shared.take_finish_report() }
    fn has_finish_report(&self) -> bool { self.shared.has_finish_report() }
    fn try_recv(&self) -> Option<TicketUndeliveredEvent> {
        self.receiver.lock().unwrap_or_else(|e| e.into_inner()).try_recv().ok().map(|event| {
            self.shared.recovery_dequeued();
            TicketUndeliveredEvent::new(event)
        })
    }
}

/// Issued by this parked worker only after validating the matching original
/// inner owner's proof. Cloning never grants a second worker or reservation.
#[derive(Clone)]
pub struct TicketRecoveryConfirmation {
    owner: Arc<()>,
    request: Arc<()>,
    epoch: Option<u64>,
    inner: Option<PhysicalRecoveryConfirmation>,
    late: bool,
    work: Option<PhysicalWorkScope>,
}
impl TicketRecoveryConfirmation {
    pub fn was_late(&self) -> bool { self.late }
}

pub(crate) enum TicketRecoveryOperation<'a> {
    Observe(Instant),
    Release(&'a PhysicalRecoveryConfirmation, Instant),
    TakeAck,
    TakeState { inner_released: bool },
    CanRelease,
}
pub(crate) enum TicketRecoveryOperationResult {
    Observe(TicketNativeRecoveryProbe),
    Release(TicketNativeRelease),
    Ack(Option<TicketUndeliveredEvent>),
    State(Option<TicketUndeliveredEvent>),
    CanRelease(bool),
}

fn registry() -> &'static Mutex<[Option<Slot>; SLOT_LIMIT]> {
    static REGISTRY: OnceLock<Mutex<[Option<Slot>; SLOT_LIMIT]>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(std::array::from_fn(|_| None)))
}

/// An identity capability for the same original process-local slot.
#[derive(Clone)]
pub struct TicketNativeRecoveryKey {
    index: usize,
    identity: Arc<()>,
    receiver_released: Arc<AtomicBool>,
}

impl TicketNativeRecoveryKey {
    /// Transfer the original worker's report after its service receiver has
    /// been released. This is the same single-take slot used by the service.
    pub fn take_finish_report(&self) -> Result<Option<TicketNativeFinishReport>, TicketRecoveryError> {
        if !self.receiver_released.load(Ordering::Acquire) { return Err(TicketRecoveryError::WrongState); }
        let slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        let slot = self.slot(&slots)?;
        let receiver = slot.queued_receiver.as_ref().ok_or(TicketRecoveryError::WrongState)?;
        Ok(receiver.take_finish_report())
    }
    pub(super) fn handover_receiver<O: Send + Sync + 'static>(&self, receiver: mpsc::Receiver<TicketNativeEvent<O>>, shared: Arc<Shared>)
        -> Result<(), mpsc::Receiver<TicketNativeEvent<O>>> {
        let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        let Some(slot) = slots.get_mut(self.index).and_then(Option::as_mut)
            .filter(|slot| Arc::ptr_eq(&slot.identity, &self.identity) && slot.queued_receiver.is_none()) else {
            return Err(receiver);
        };
        slot.queued_receiver = Some(Box::new(TypedEventReceiver { receiver: Arc::new(Mutex::new(receiver)), shared }));
        self.receiver_released.store(true, Ordering::Release);
        Ok(())
    }

    /// Register the exact receiver before the factory starts. Service and
    /// recovery share the same receiver lock; no fallible Drop-time transfer.
    pub(super) fn install_shared_receiver<O: Send + Sync + 'static>(&self, receiver: &Arc<Mutex<mpsc::Receiver<TicketNativeEvent<O>>>>, shared: Arc<Shared>) -> Result<(), TicketRecoveryError> {
        let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        let slot = slots.get_mut(self.index).and_then(Option::as_mut)
            .filter(|slot| Arc::ptr_eq(&slot.identity, &self.identity) && slot.queued_receiver.is_none())
            .ok_or(TicketRecoveryError::InvalidKey)?;
        slot.queued_receiver = Some(Box::new(TypedEventReceiver { receiver: Arc::clone(receiver), shared }));
        Ok(())
    }

    /// The flag is tied directly to the key, so dropping the service does not
    /// need a registry lookup or a fallible receiver move.
    pub(crate) fn release_shared_receiver(&self) {
        self.receiver_released.store(true, Ordering::Release);
    }

    /// Opportunistic O(1) cleanup after a normal finished worker and receiver
    /// handover. It never waits, scans files or consumes an unreturned event.
    pub(crate) fn try_retire_clean(&self) {
        let thread = {
            let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
            let Some(slot) = slots.get_mut(self.index).and_then(Option::as_mut)
                .filter(|slot| Arc::ptr_eq(&slot.identity, &self.identity)) else { return; };
            if slot.release_in_progress || !slot.receiver_released.load(Ordering::Acquire)
                || !matches!(slot.status, TicketRecoveryStatus::Released | TicketRecoveryStatus::StartNotDelivered) || slot.queued_receiver.is_none() { return; }
            if slot.queued_receiver.as_ref().is_some_and(|receiver| receiver.has_finish_report()) { return; }
            if slot.queued_event.as_ref().is_some_and(|event| !event.is_pending()) { slot.queued_event = None; }
            if slot.queued_event.is_none() {
                slot.queued_event = slot.queued_receiver.as_ref().and_then(|receiver| receiver.try_recv());
            }
            if slot.queued_event.is_some() || slot.before_native_events.iter().any(TicketUndeliveredEvent::is_pending) { return; }
            slot.thread.clone()
        };
        if !thread.is_some_and(|thread| matches!(thread.try_join(), Ok(true))) { return; }
        let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        if self.slot(&slots).is_ok_and(|slot| !slot.release_in_progress && matches!(slot.status, TicketRecoveryStatus::Released | TicketRecoveryStatus::StartNotDelivered) && slot.queued_event.is_none() && !slot.queued_receiver.as_ref().is_some_and(|receiver| receiver.has_finish_report())) {
            slots[self.index] = None;
        }
    }

    /// Nonblocking retirement for a worker that never received its start
    /// token. The caller may retry after the original thread exits.
    pub fn try_retire_not_started(&self) -> Result<bool, TicketRecoveryError> {
        if self.status()? != TicketRecoveryStatus::StartNotDelivered { return Err(TicketRecoveryError::WrongState); }
        self.try_retire_clean();
        match self.status() {
            Err(TicketRecoveryError::InvalidKey) => Ok(true),
            Ok(TicketRecoveryStatus::StartNotDelivered) => Ok(false),
            _ => Err(TicketRecoveryError::WrongState),
        }
    }
    pub(crate) fn bind_inner_before_native(&self, inner: &PhysicalRecoveryKey, epoch: u64) -> Result<(), TicketRecoveryError> {
        let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        let slot = slots.get_mut(self.index).and_then(Option::as_mut)
            .filter(|slot| Arc::ptr_eq(&slot.identity, &self.identity))
            .ok_or(TicketRecoveryError::InvalidKey)?;
        if !matches!(slot.status, TicketRecoveryStatus::Reserved | TicketRecoveryStatus::Running)
            || slot.expected_inner.is_some() || epoch == 0 {
            return Err(TicketRecoveryError::WrongState);
        }
        slot.expected_inner = Some(inner.clone());
        slot.original_epoch = Some(epoch);
        Ok(())
    }
    /// Discovers only retained unknown/confirmed owners in this process.
    pub fn recoverable_keys() -> Vec<Self> {
        let slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        slots.iter().enumerate().filter_map(|(index, slot)| {
            let slot = slot.as_ref()?;
            matches!(slot.status, TicketRecoveryStatus::BeforeNative | TicketRecoveryStatus::StartNotDelivered | TicketRecoveryStatus::Unknown | TicketRecoveryStatus::Confirmed | TicketRecoveryStatus::Released)
                .then(|| Self { index, identity: Arc::clone(&slot.identity), receiver_released: Arc::clone(&slot.receiver_released) })
        }).collect()
    }

    pub fn status(&self) -> Result<TicketRecoveryStatus, TicketRecoveryError> {
        let slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        let slot = self.slot(&slots)?;
        Ok(slot.status)
    }

    pub fn latest_confirmation(&self) -> Option<TicketRecoveryConfirmation> {
        let slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        self.slot(&slots).ok()?.confirmation.clone()
    }

    fn slot<'a>(&self, slots: &'a [Option<Slot>; SLOT_LIMIT]) -> Result<&'a Slot, TicketRecoveryError> {
        slots.get(self.index).and_then(Option::as_ref)
            .filter(|slot| Arc::ptr_eq(&slot.identity, &self.identity))
            .ok_or(TicketRecoveryError::InvalidKey)
    }

    fn sender(&self) -> Result<mpsc::SyncSender<TicketRecoveryCommand>, TicketRecoveryError> {
        let slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        self.slot(&slots)?.sender.clone().ok_or(TicketRecoveryError::WrongState)
    }

    /// Ask the original owner to observe its own fence, within the caller's
    /// deadline. A timeout or missing provider is never physical completion.
    pub fn reobserve(&self, until: Instant) -> Result<TicketRecoveryStatus, TicketRecoveryError> {
        if Instant::now() >= until { return Err(TicketRecoveryError::TimedOut); }
        let sender = {
            let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
            let slot = slots.get_mut(self.index).and_then(Option::as_mut)
                .filter(|slot| Arc::ptr_eq(&slot.identity, &self.identity))
                .ok_or(TicketRecoveryError::InvalidKey)?;
            if !matches!(slot.status, TicketRecoveryStatus::Unknown | TicketRecoveryStatus::Confirmed)
                || slot.observation_pending { return Err(TicketRecoveryError::WrongState); }
            let sender = slot.sender.clone().ok_or(TicketRecoveryError::WrongState)?;
            slot.observation_pending = true;
            sender
        };
        let (reply, receive) = mpsc::sync_channel(1);
        if let Err(error) = sender.try_send(TicketRecoveryCommand::Observe { until, reply }) {
            let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
            if let Some(slot) = slots.get_mut(self.index).and_then(Option::as_mut)
                .filter(|slot| Arc::ptr_eq(&slot.identity, &self.identity)) { slot.observation_pending = false; }
            return Err(match error { mpsc::TrySendError::Full(_) => TicketRecoveryError::WrongState, mpsc::TrySendError::Disconnected(_) => TicketRecoveryError::Disconnected });
        }
        receive.recv_timeout(until.saturating_duration_since(Instant::now()))
            .map_err(|e| match e { mpsc::RecvTimeoutError::Timeout => TicketRecoveryError::TimedOut, mpsc::RecvTimeoutError::Disconnected => TicketRecoveryError::Disconnected })
    }

    /// Borrow a capability to an actual undelivered event from this worker.
    /// The original remains in the worker until the recipient explicitly takes
    /// it; a type mismatch leaves it intact and proves no physical completion.
    pub fn take_event(&self, until: Instant) -> Result<Option<TicketUndeliveredEvent>, TicketRecoveryError> {
        if Instant::now() >= until { return Err(TicketRecoveryError::TimedOut); }
        if !self.receiver_released.load(Ordering::Acquire) { return Err(TicketRecoveryError::WrongState); }
        {
            let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
            let slot = slots.get_mut(self.index).and_then(Option::as_mut)
                .filter(|slot| Arc::ptr_eq(&slot.identity, &self.identity))
                .ok_or(TicketRecoveryError::InvalidKey)?;
            if slot.queued_event.as_ref().is_some_and(|event| !event.is_pending()) {
                slot.queued_event = None;
            }
            if slot.queued_event.is_none() {
                slot.queued_event = slot.queued_receiver.as_ref().and_then(|receiver| receiver.try_recv());
            }
            if let Some(event) = &slot.queued_event { return Ok(Some(event.clone())); }
        }
        if self.status()? == TicketRecoveryStatus::BeforeNative {
            let slots = registry().lock().unwrap_or_else(|e| e.into_inner());
            return Ok(self.slot(&slots)?.before_native_events.iter().find(|event| event.is_pending()).cloned());
        }
        if !matches!(self.status()?, TicketRecoveryStatus::Unknown | TicketRecoveryStatus::Confirmed) {
            return Err(TicketRecoveryError::WrongState);
        }
        let (reply, receive) = mpsc::sync_channel(1);
        self.sender()?.try_send(TicketRecoveryCommand::TakeEvent { until, reply })
            .map_err(|e| match e { mpsc::TrySendError::Full(_) => TicketRecoveryError::WrongState, mpsc::TrySendError::Disconnected(_) => TicketRecoveryError::Disconnected })?;
        receive.recv_timeout(until.saturating_duration_since(Instant::now()))
            .map_err(|e| match e { mpsc::RecvTimeoutError::Timeout => TicketRecoveryError::TimedOut, mpsc::RecvTimeoutError::Disconnected => TicketRecoveryError::Disconnected })
    }

    /// Acknowledgement custody is transferred once through its opaque handle;
    /// receiving a handle itself does not consume the original permit.
    pub fn take_pending_ack(&self, until: Instant) -> Result<Option<TicketUndeliveredEvent>, TicketRecoveryError> {
        if Instant::now() >= until { return Err(TicketRecoveryError::TimedOut); }
        if !matches!(self.status()?, TicketRecoveryStatus::Unknown | TicketRecoveryStatus::Confirmed) {
            return Err(TicketRecoveryError::WrongState);
        }
        let (reply, receive) = mpsc::sync_channel(1);
        self.sender()?.try_send(TicketRecoveryCommand::TakeAck { until, reply })
            .map_err(|e| match e { mpsc::TrySendError::Full(_) => TicketRecoveryError::WrongState, mpsc::TrySendError::Disconnected(_) => TicketRecoveryError::Disconnected })?;
        receive.recv_timeout(until.saturating_duration_since(Instant::now()))
            .map_err(|e| match e { mpsc::RecvTimeoutError::Timeout => TicketRecoveryError::TimedOut, mpsc::RecvTimeoutError::Disconnected => TicketRecoveryError::Disconnected })
    }

    /// Transfer the original retained owner state only after the same physical
    /// owner has confirmed completion. An unknown fence never exposes buffers.
    pub fn take_confirmed_state(&self, until: Instant) -> Result<Option<TicketUndeliveredEvent>, TicketRecoveryError> {
        if Instant::now() >= until { return Err(TicketRecoveryError::TimedOut); }
        if self.status()? != TicketRecoveryStatus::Confirmed { return Err(TicketRecoveryError::WrongState); }
        let (reply, receive) = mpsc::sync_channel(1);
        self.sender()?.try_send(TicketRecoveryCommand::TakeState { until, reply })
            .map_err(|e| match e { mpsc::TrySendError::Full(_) => TicketRecoveryError::WrongState, mpsc::TrySendError::Disconnected(_) => TicketRecoveryError::Disconnected })?;
        receive.recv_timeout(until.saturating_duration_since(Instant::now()))
            .map_err(|e| match e { mpsc::RecvTimeoutError::Timeout => TicketRecoveryError::TimedOut, mpsc::RecvTimeoutError::Disconnected => TicketRecoveryError::Disconnected })
    }

    /// A confirmed observation still retains original state until this
    /// explicit release and actual owner-thread join have both completed.
    pub fn release_confirmed(&self, proof: &TicketRecoveryConfirmation, until: Instant) -> Result<(), TicketRecoveryError> {
        if Instant::now() >= until { return Err(TicketRecoveryError::TimedOut); }
        if !self.receiver_released.load(Ordering::Acquire) { return Err(TicketRecoveryError::WrongState); }
        {
            let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
            let slot = slots.get_mut(self.index).and_then(Option::as_mut)
                .filter(|slot| Arc::ptr_eq(&slot.identity, &self.identity))
                .ok_or(TicketRecoveryError::InvalidKey)?;
            if slot.queued_event.as_ref().is_some_and(|event| !event.is_pending()) { slot.queued_event = None; }
            if slot.queued_event.is_none() {
                slot.queued_event = slot.queued_receiver.as_ref().and_then(|receiver| receiver.try_recv());
            }
            if slot.queued_receiver.is_none() || slot.queued_event.as_ref().is_some_and(TicketUndeliveredEvent::is_pending)
                || slot.queued_receiver.as_ref().is_some_and(|receiver| receiver.has_finish_report()) {
                return Err(TicketRecoveryError::WrongState);
            }
            let valid = slot.confirmation.as_ref().is_some_and(|last| {
                Arc::ptr_eq(&last.owner, &proof.owner)
                    && Arc::ptr_eq(&last.request, &proof.request)
                    && last.epoch == proof.epoch
                    && slot.original_epoch == proof.epoch
                    && last.work == proof.work
                    && (proof.work.is_none() || slot.expected_work == proof.work)
            });
            if !valid { return Err(TicketRecoveryError::WrongState); }
            if slot.release_in_progress { return Err(TicketRecoveryError::WrongState); }
            slot.release_in_progress = true;
        }
        let _release_guard = ReleaseGuard(self.clone());
        match self.status()? {
            TicketRecoveryStatus::Confirmed => {
                let (reply, receive) = mpsc::sync_channel(1);
                self.sender()?.try_send(TicketRecoveryCommand::Release { until, reply })
                    .map_err(|_| TicketRecoveryError::Disconnected)?;
                match receive.recv_timeout(until.saturating_duration_since(Instant::now()))
                    .map_err(|_| TicketRecoveryError::TimedOut)? {
                    TicketRecoveryStatus::Released => {},
                    TicketRecoveryStatus::Confirmed => return Err(TicketRecoveryError::RetainedStatePending),
                    _ => return Err(TicketRecoveryError::WrongState),
                }
            }
            TicketRecoveryStatus::Released => {}
            _ => return Err(TicketRecoveryError::WrongState),
        }
        self.join_original(until)?;
        let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        self.slot(&slots)?;
        slots[self.index] = None;
        Ok(())
    }

    fn join_original(&self, until: Instant) -> Result<(), TicketRecoveryError> {
        let handle = {
            let slots = registry().lock().unwrap_or_else(|e| e.into_inner());
            self.slot(&slots)?.thread.clone().ok_or(TicketRecoveryError::WrongState)?
        };
        loop {
            if handle.try_join()? {
                break;
            }
            if Instant::now() >= until { return Err(TicketRecoveryError::TimedOut); }
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok(())
    }

    /// Before-native failure may return the exact original caller-owned C,
    /// including observer ownership; it never creates a fence observation.
    pub fn take_before_native<C: Any + Send>(&self, until: Instant) -> Result<C, TicketRecoveryError> {
        if !self.receiver_released.load(Ordering::Acquire) { return Err(TicketRecoveryError::WrongState); }
        if self.status()? != TicketRecoveryStatus::BeforeNative { return Err(TicketRecoveryError::WrongState); }
        self.join_original(until)?;
        let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        let slot = slots.get_mut(self.index).and_then(Option::as_mut)
            .filter(|slot| Arc::ptr_eq(&slot.identity, &self.identity))
            .ok_or(TicketRecoveryError::InvalidKey)?;
        if slot.status != TicketRecoveryStatus::BeforeNative { return Err(TicketRecoveryError::WrongState); }
        if slot.queued_event.as_ref().is_some_and(|event| !event.is_pending()) { slot.queued_event = None; }
        if slot.queued_event.is_none() {
            slot.queued_event = slot.queued_receiver.as_ref().and_then(|receiver| receiver.try_recv());
        }
        if slot.queued_receiver.is_none() || slot.queued_event.as_ref().is_some_and(TicketUndeliveredEvent::is_pending)
            || slot.before_native_events.iter().any(TicketUndeliveredEvent::is_pending) {
            return Err(TicketRecoveryError::WrongState);
        }
        let boxed = slot.before_native_owner.take().ok_or(TicketRecoveryError::WrongState)?;
        match boxed.downcast::<C>() {
            Ok(owner) => { slots[self.index] = None; Ok(*owner) }
            Err(owner) => { slot.before_native_owner = Some(owner); Err(TicketRecoveryError::WrongOwnerType) }
        }
    }
}

/// Reserve before any native call; failure affects this attempt only.
pub struct TicketNativeRecoveryReservation {
    key: TicketNativeRecoveryKey,
    receiver: Option<mpsc::Receiver<TicketRecoveryCommand>>,
}

impl TicketNativeRecoveryReservation {
    #[cfg(test)]
    pub(crate) fn test_set_unstarted_unknown(&self) -> Result<(), TicketRecoveryError> {
        self.test_transition_unstarted(TicketRecoveryStatus::Reserved, TicketRecoveryStatus::Unknown)
    }
    #[cfg(test)]
    pub(crate) fn test_clear_unstarted_unknown(&self) -> Result<(), TicketRecoveryError> {
        self.test_transition_unstarted(TicketRecoveryStatus::Unknown, TicketRecoveryStatus::Reserved)
    }
    #[cfg(test)]
    fn test_transition_unstarted(&self, from: TicketRecoveryStatus, to: TicketRecoveryStatus) -> Result<(), TicketRecoveryError> {
        let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        let slot = slots.get_mut(self.key.index).and_then(Option::as_mut)
            .filter(|slot| Arc::ptr_eq(&slot.identity, &self.key.identity))
            .ok_or(TicketRecoveryError::InvalidKey)?;
        if slot.status != from || slot.thread.is_some() || slot.before_native_owner.is_some()
            || !slot.before_native_events.is_empty() || slot.confirmation.is_some()
            || slot.queued_receiver.is_some() || slot.queued_event.is_some() {
            return Err(TicketRecoveryError::WrongState);
        }
        slot.status = to;
        Ok(())
    }
    pub fn reserve() -> Result<(Self, TicketNativeRecoveryKey), TicketRecoveryError> {
        // Bound maintenance to this registry's sixteen slots. Only normal
        // worker-confirmed clean finishes qualify; unknown/explicit recovery
        // owners never become candidates merely because capacity is needed.
        let clean_keys: Vec<_> = {
            let slots = registry().lock().unwrap_or_else(|e| e.into_inner());
            slots.iter().enumerate().filter_map(|(index, slot)| {
                let slot = slot.as_ref()?;
                (slot.clean_finish && slot.status == TicketRecoveryStatus::Released)
                    .then(|| TicketNativeRecoveryKey { index, identity: Arc::clone(&slot.identity), receiver_released: Arc::clone(&slot.receiver_released) })
            }).collect()
        };
        for key in clean_keys { key.try_retire_clean(); }
        let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        let index = slots.iter().position(Option::is_none).ok_or(TicketRecoveryError::Capacity)?;
        let identity = Arc::new(());
        let receiver_released = Arc::new(AtomicBool::new(false));
        let key = TicketNativeRecoveryKey { index, identity: Arc::clone(&identity), receiver_released: Arc::clone(&receiver_released) };
        let (sender, receiver) = mpsc::sync_channel(1);
        slots[index] = Some(Slot { identity, status: TicketRecoveryStatus::Reserved, sender: Some(sender), thread: None, before_native_owner: None, before_native_events: Vec::new(), observation_pending: false, expected_inner: None, original_epoch: None, expected_work: None, confirmation: None, queued_receiver: None, queued_event: None, receiver_released, clean_finish: false, release_in_progress: false });
        Ok((Self { key: key.clone(), receiver: Some(receiver) }, key))
    }

    pub(crate) fn attach(&self, thread: Arc<TicketOwnerJoin>) -> Result<(), TicketRecoveryError> {
        let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        let slot = slots.get_mut(self.key.index).and_then(Option::as_mut)
            .filter(|slot| Arc::ptr_eq(&slot.identity, &self.key.identity))
            .ok_or(TicketRecoveryError::InvalidKey)?;
        if slot.status != TicketRecoveryStatus::Reserved { return Err(TicketRecoveryError::WrongState); }
        slot.thread = Some(thread);
        slot.status = TicketRecoveryStatus::Running;
        Ok(())
    }

    /// Bind exactly the original, pre-reserved physical owner and epoch
    /// before any native entry. A new key cannot substitute a different owner.
    pub fn bind_inner(&self, inner: &PhysicalRecoveryKey, epoch: u64) -> Result<(), TicketRecoveryError> {
        if self.key.status()? != TicketRecoveryStatus::Reserved { return Err(TicketRecoveryError::WrongState); }
        self.key.bind_inner_before_native(inner, epoch)
    }

    /// Seal only the original backend's outstanding lease and the matching
    /// active ticket. No value reconstructed after work is accepted here.
    pub(crate) fn bind_active_work(&self, inner: &PhysicalRecoveryKey, work: PhysicalWorkScope) -> Result<(), TicketRecoveryError> {
        let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        let slot = slots.get_mut(self.key.index).and_then(Option::as_mut)
            .filter(|slot| Arc::ptr_eq(&slot.identity, &self.key.identity))
            .ok_or(TicketRecoveryError::InvalidKey)?;
        let PhysicalWorkScope::Ticket { ticket, execution } = work else { return Err(TicketRecoveryError::WrongState); };
        if slot.status != TicketRecoveryStatus::Running || slot.expected_work.is_some()
            || slot.original_epoch != Some(ticket.authority.epoch.0)
            || ticket.id.epoch != execution.epoch
            || !slot.expected_inner.as_ref().is_some_and(|expected| expected.same_owner(inner)) {
            return Err(TicketRecoveryError::WrongState);
        }
        slot.expected_work = Some(work);
        Ok(())
    }

    pub(crate) fn before_native<C: Any + Send>(&self, owner: C, events: Vec<TicketUndeliveredEvent>) -> Result<(), (C, Vec<TicketUndeliveredEvent>)> {
        let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = slots.get_mut(self.key.index).and_then(Option::as_mut)
            .filter(|slot| Arc::ptr_eq(&slot.identity, &self.key.identity)
                && slot.status == TicketRecoveryStatus::Running
                && slot.before_native_owner.is_none()) {
            slot.before_native_owner = Some(Box::new(owner));
            slot.before_native_events = events;
            slot.status = TicketRecoveryStatus::BeforeNative;
            Ok(())
        } else {
            Err((owner, events))
        }
    }

    /// Runs on the original worker with all original resources still in its
    /// stack. Unknown and control timeouts retain the same state indefinitely.
    pub(crate) fn park(&mut self, initially_finished: bool, mut operate: impl FnMut(TicketRecoveryOperation<'_>) -> TicketRecoveryOperationResult, undelivered: &RefCell<Vec<TicketUndeliveredEvent>>) {
        let Some(receiver) = self.receiver.take() else {
            // An invalid duplicate park cannot release an unknown owner.
            loop { std::thread::park(); }
        };
        let mut confirmed = initially_finished && self.record_local_finish();
        self.set_status(if confirmed { TicketRecoveryStatus::Confirmed } else { TicketRecoveryStatus::Unknown });
        let mut inner_released = confirmed;
        while let Ok(command) = receiver.recv() {
            match command {
                TicketRecoveryCommand::Observe { until, reply } => {
                    if !confirmed && Instant::now() < until {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| operate(TicketRecoveryOperation::Observe(until))));
                        if let Ok(TicketRecoveryOperationResult::Observe(TicketNativeRecoveryProbe::Confirmed(proof))) = result {
                            confirmed = self.record_confirmation(proof, Instant::now() > until);
                        }
                    }
                    let status = if confirmed { TicketRecoveryStatus::Confirmed } else { TicketRecoveryStatus::Unknown };
                    self.finish_observation(status);
                    let _ = reply.try_send(if Instant::now() > until { TicketRecoveryStatus::Unknown } else { status });
                }
                TicketRecoveryCommand::Release { until, reply } => {
                    if confirmed && Instant::now() < until && undelivered.borrow().iter().all(|event| !event.is_pending()) {
                        if !inner_released {
                            if let Some(proof) = self.key.latest_confirmation().and_then(|proof| proof.inner) {
                                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| operate(TicketRecoveryOperation::Release(&proof, until))));
                                inner_released = matches!(result, Ok(TicketRecoveryOperationResult::Release(TicketNativeRelease::Released)));
                            }
                        }
                        let state_recovered = matches!(operate(TicketRecoveryOperation::CanRelease), TicketRecoveryOperationResult::CanRelease(true));
                        if inner_released && state_recovered && Instant::now() < until {
                            self.set_status(TicketRecoveryStatus::Released);
                            let _ = reply.try_send(TicketRecoveryStatus::Released);
                            return;
                        }
                    }
                    let _ = reply.try_send(if confirmed && inner_released { TicketRecoveryStatus::Confirmed } else { TicketRecoveryStatus::Unknown });
                }
                TicketRecoveryCommand::TakeEvent { until, reply } => {
                    let event = if Instant::now() < until {
                        undelivered.borrow().iter().find(|event| event.is_pending()).cloned()
                    } else { None };
                    let _ = reply.try_send(event);
                }
                TicketRecoveryCommand::TakeAck { until, reply } => {
                    let ack = if Instant::now() < until {
                        match operate(TicketRecoveryOperation::TakeAck) {
                            TicketRecoveryOperationResult::Ack(ack) => ack,
                            _ => None,
                        }
                    } else { None };
                    let _ = reply.try_send(ack);
                }
                TicketRecoveryCommand::TakeState { until, reply } => {
                    let state = if confirmed && Instant::now() < until {
                        match operate(TicketRecoveryOperation::TakeState { inner_released }) {
                            TicketRecoveryOperationResult::State(state) => state,
                            _ => None,
                        }
                    } else { None };
                    let _ = reply.try_send(state);
                }
            }
        }
        // The registry owns the sender while a slot is occupied, so a
        // disconnect is not treated as a fence or permission to free state.
        loop { std::thread::park(); }
    }

    fn set_status(&self, status: TicketRecoveryStatus) {
        let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = slots.get_mut(self.key.index).and_then(Option::as_mut)
            .filter(|slot| Arc::ptr_eq(&slot.identity, &self.key.identity)) { slot.status = status; }
    }

    fn finish_observation(&self, status: TicketRecoveryStatus) {
        let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = slots.get_mut(self.key.index).and_then(Option::as_mut)
            .filter(|slot| Arc::ptr_eq(&slot.identity, &self.key.identity)) {
            slot.status = status;
            slot.observation_pending = false;
        }
    }

    fn record_confirmation(&self, inner: PhysicalRecoveryConfirmation, late: bool) -> bool {
        let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        let Some(slot) = slots.get_mut(self.key.index).and_then(Option::as_mut)
            .filter(|slot| Arc::ptr_eq(&slot.identity, &self.key.identity)) else { return false; };
        let Some(epoch) = slot.original_epoch else { return false; };
        let Some(work) = slot.expected_work else { return false; };
        if !slot.expected_inner.as_ref().is_some_and(|key| key.confirms_work_scope(&inner, work)) { return false; }
        slot.confirmation = Some(TicketRecoveryConfirmation {
            owner: Arc::clone(&slot.identity), request: Arc::new(()), epoch: Some(epoch), inner: Some(inner), late, work: Some(work),
        });
        true
    }

    fn record_local_finish(&self) -> bool {
        let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        let Some(slot) = slots.get_mut(self.key.index).and_then(Option::as_mut)
            .filter(|slot| Arc::ptr_eq(&slot.identity, &self.key.identity)) else { return false; };
        slot.confirmation = Some(TicketRecoveryConfirmation {
            owner: Arc::clone(&slot.identity), request: Arc::new(()),
            epoch: slot.original_epoch, inner: None, late: false, work: None,
        });
        true
    }

    /// Only a reservation that has not been attached to a worker can be
    /// canceled without inspecting an owner or a physical fence.
    pub fn cancel_unstarted(self) -> Result<(), TicketRecoveryError> {
        let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        let slot = self.key.slot(&slots)?;
        if slot.status != TicketRecoveryStatus::Reserved { return Err(TicketRecoveryError::WrongState); }
        slots[self.key.index] = None;
        Ok(())
    }

    /// Called only after OS thread creation returned Err, before the start
    /// barrier could be opened. A live/installed thread is never canceled here.
    pub(crate) fn cancel_spawn_failure(self) -> Result<(), TicketRecoveryError> {
        let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        let slot = self.key.slot(&slots)?;
        if slot.status != TicketRecoveryStatus::Running
            || !slot.thread.as_ref().is_some_and(|thread| thread.not_installed()) {
            return Err(TicketRecoveryError::WrongState);
        }
        slots[self.key.index] = None;
        Ok(())
    }

    /// Only the original failed sender can establish non-delivery of its
    /// start token. This does not make a GPU-completion claim.
    pub(crate) fn mark_start_not_delivered(self) {
        self.set_status(TicketRecoveryStatus::StartNotDelivered);
    }

    /// Only the original worker calls this after an actual successful finish
    /// and when no terminal event remains undelivered.
    pub(crate) fn finish_clean(self) -> Result<(), TicketRecoveryError> {
        if self.key.status()? != TicketRecoveryStatus::Running { return Err(TicketRecoveryError::WrongState); }
        if !self.record_local_finish() { return Err(TicketRecoveryError::InvalidKey); }
        let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        let slot = slots.get_mut(self.key.index).and_then(Option::as_mut)
            .filter(|slot| Arc::ptr_eq(&slot.identity, &self.key.identity))
            .ok_or(TicketRecoveryError::InvalidKey)?;
        if slot.status != TicketRecoveryStatus::Running { return Err(TicketRecoveryError::WrongState); }
        slot.clean_finish = true;
        slot.status = TicketRecoveryStatus::Released;
        Ok(())
    }
}
