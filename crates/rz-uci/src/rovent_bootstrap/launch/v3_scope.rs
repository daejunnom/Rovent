//! Process-local custody for an original V3 launch scope.
//!
//! This is not a supervisor lease or a CUDA fence. A live child and an
//! authenticated external consumer remain necessary to retain it across IPC.
use std::any::Any;
use std::sync::{Arc, Mutex, OnceLock};

use crate::pals_native::ticketed::TicketNativeRecoveryKey;

const SCOPE_LIMIT: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum V3ScopeState {
    Reserved,
    Attempted,
    Interrupted,
    Parked,
}

/// Observation of this scope only; neither variant grants a device fence or
/// permission to release unrelated owners.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum V3ScopeTerminalEvidence {
    Unknown,
    BeforeNative,
    SessionReleased,
}

/// Observation of this original product's completed finish call. This is not
/// a device-wide CUDA fence, a result validation, or an external tree proof.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct V3ProductFinishObservation {
    pub process_id: u32,
    pub process_epoch: u64,
    pub input_lock_sha256: String,
}

pub struct V3ScopeObservation {
    pub state: V3ScopeState,
    pub recovery_key: Option<TicketNativeRecoveryKey>,
    pub parked_original: bool,
}

pub enum V3ScopeAccess<T> {
    Missing,
    Busy,
    Present(T),
}

/// Internal transfer result. A failed attempt leaves the original slot and
/// its pins in the registry. The callback must itself retain ownership of any
/// values it moves within `T`; arbitrary mutation cannot be rolled back.
pub(super) enum V3ScopeRecovery<P, E> {
    Missing,
    Busy,
    Poisoned,
    Expired,
    WrongState,
    WrongType,
    Pending(E),
    Panic(Box<dyn Any + Send>),
    /// Defensive invariant failure: the proof is returned to the caller and
    /// the original is left in its slot, never silently dropped.
    LostRegistration(P),
    Recovered {
        proof: P,
        original: Box<dyn Any + Send>,
        pins: Vec<Box<dyn Any + Send>>,
        recovery_key: Option<TicketNativeRecoveryKey>,
    },
}

/// A native-ticket observation inside a larger product scope. Even a native
/// `Confirmed` state is not proof that the entire product or external
/// process scope is physically closed.
#[derive(Debug)]
pub enum V3NativeReobservation {
    ScopeMissing,
    ScopeBusy,
    CustodyBusy {
        scope_state: V3ScopeState,
    },
    NativeKeyUnavailable {
        scope_state: V3ScopeState,
    },
    NativeObserved {
        scope_state: V3ScopeState,
        native_status: crate::pals_native::ticketed::TicketRecoveryStatus,
    },
    NativeError {
        scope_state: V3ScopeState,
        error: crate::pals_native::ticketed::TicketRecoveryError,
    },
}

struct SlotState {
    state: V3ScopeState,
    terminal: V3ScopeTerminalEvidence,
    product_finish: Option<V3ProductFinishObservation>,
    // These are the same Arc-owned descriptors and source owners used by the
    // launch, not copies reopened by path after a failure.
    _pins: Vec<Box<dyn Any + Send>>,
    original: Option<Box<dyn Any + Send>>,
    recovery_key: Option<TicketNativeRecoveryKey>,
    // A successful recovery callback may have performed irreversible work.
    // Preserve its exact proof if registry commit is temporarily busy.
    recovery_proof: Option<Box<dyn Any + Send>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum V3ScopeDomain {
    Cpu,
    // No verified physical device identity is available at this boundary.
    // Different ordinals remain in one conservative CUDA capacity group.
    CudaUnidentified,
    /// Reserved for a future authenticated supervisor admission. No public
    /// constructor exists, and an ordinal cannot create this capability.
    #[allow(dead_code)]
    CudaVerified(VerifiedPhysicalScope),
}
/// Can only be constructed in this module after a future integration has
/// checked the same supervisor, device and physical-owner identity. Merely
/// knowing UUID bytes is not authorization to create this value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct VerifiedPhysicalScope {
    identity: [u8; 16],
    _proof_is_not_yet_available: (),
}
struct Slot {
    domain: V3ScopeDomain,
    state: Mutex<SlotState>,
}

fn registry() -> &'static Mutex<Vec<Arc<Slot>>> {
    static REGISTRY: OnceLock<Mutex<Vec<Arc<Slot>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}

/// Capability for this process's exact original slot; not a fence token.
#[derive(Clone)]
pub struct RoventV3ScopeKey(Arc<Slot>);

impl RoventV3ScopeKey {
    /// None includes missing, busy and poisoned slots; it never grants release.
    pub fn product_finish_observation(&self) -> Option<V3ProductFinishObservation> {
        self.0.state.try_lock().ok()?.product_finish.clone()
    }
    /// Reads the original slot even after retirement. Registry absence alone
    /// never implies completion; contention and poisoning are unknown.
    pub fn terminal_evidence(&self) -> V3ScopeTerminalEvidence {
        match self.0.state.try_lock() {
            Ok(state) => state.terminal,
            Err(_) => V3ScopeTerminalEvidence::Unknown,
        }
    }

    pub fn observe(&self) -> Option<V3ScopeObservation> {
        match self.try_observe() {
            V3ScopeAccess::Present(observation) => Some(observation),
            V3ScopeAccess::Missing | V3ScopeAccess::Busy => None,
        }
    }

    pub fn try_observe(&self) -> V3ScopeAccess<V3ScopeObservation> {
        let slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        if !slots.iter().any(|slot| Arc::ptr_eq(slot, &self.0)) {
            return V3ScopeAccess::Missing;
        }
        let state = match self.0.state.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(error)) => error.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return V3ScopeAccess::Busy,
        };
        V3ScopeAccess::Present(V3ScopeObservation {
            state: state.state,
            recovery_key: state.recovery_key.clone(),
            parked_original: state.original.is_some(),
        })
    }

    /// Clone an already-registered typed capability (for example the same
    /// `Arc<Mutex<...>>` assembly owner). The original stays in the slot.
    pub fn cloned_pin<T: Any + Send + Sync>(&self) -> Option<Arc<T>> {
        match self.try_cloned_pin() {
            V3ScopeAccess::Present(pin) => Some(pin),
            V3ScopeAccess::Missing | V3ScopeAccess::Busy => None,
        }
    }

    pub fn try_cloned_pin<T: Any + Send + Sync>(&self) -> V3ScopeAccess<Arc<T>> {
        let slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        if !slots.iter().any(|slot| Arc::ptr_eq(slot, &self.0)) {
            return V3ScopeAccess::Missing;
        }
        let state = match self.0.state.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(error)) => error.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return V3ScopeAccess::Busy,
        };
        match state
            ._pins
            .iter()
            .find_map(|pin| pin.downcast_ref::<Arc<T>>().map(Arc::clone))
        {
            Some(pin) => V3ScopeAccess::Present(pin),
            None => V3ScopeAccess::Missing,
        }
    }

    /// Inspect the typed original while retaining it in the same slot. A busy
    /// or reentrant slot returns None. The callback must be short; it cannot
    /// move the original out and this method is not a release proof. There is
    /// intentionally no concurrent release path in this implementation.
    pub fn with_original<T: Any, R>(&self, f: impl FnOnce(&T) -> R) -> Option<R> {
        let slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        if !slots.iter().any(|slot| Arc::ptr_eq(slot, &self.0)) {
            return None;
        }
        drop(slots);
        let state = match self.0.state.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(error)) => error.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return None,
        };
        state.original.as_ref()?.downcast_ref::<T>().map(f)
    }

    /// Internal one-slot transfer. The caller must verify BeforeNative on the
    /// actual ticket owner inside `attempt`; this helper grants no proof by
    /// itself. An unsuccessful attempt leaves the slot and all pins in place.
    #[cfg(test)]
    fn try_recover_original<T: Any, P: Any + Send, E>(
        &self,
        attempt: impl FnOnce(&mut T) -> Result<P, E>,
    ) -> V3ScopeRecovery<P, E> {
        self.try_recover_original_until(
            std::time::Instant::now() + std::time::Duration::from_secs(60),
            attempt,
        )
    }

    pub(super) fn try_recover_original_until<T: Any, P: Any + Send, E>(
        &self,
        until: std::time::Instant,
        attempt: impl FnOnce(&mut T) -> Result<P, E>,
    ) -> V3ScopeRecovery<P, E> {
        if std::time::Instant::now() >= until {
            return V3ScopeRecovery::Expired;
        }
        {
            let slots = match registry().try_lock() {
                Ok(slots) => slots,
                Err(std::sync::TryLockError::WouldBlock) => return V3ScopeRecovery::Busy,
                Err(std::sync::TryLockError::Poisoned(_)) => return V3ScopeRecovery::Poisoned,
            };
            if !slots.iter().any(|slot| Arc::ptr_eq(slot, &self.0)) {
                return V3ScopeRecovery::Missing;
            }
        }
        let mut state = match self.0.state.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(_)) => return V3ScopeRecovery::Poisoned,
            Err(std::sync::TryLockError::WouldBlock) => return V3ScopeRecovery::Busy,
        };
        if state.state != V3ScopeState::Parked {
            return V3ScopeRecovery::WrongState;
        }
        if !state.original.as_ref().is_some_and(|o| o.is::<T>()) {
            return V3ScopeRecovery::WrongType;
        }
        if std::time::Instant::now() >= until {
            return V3ScopeRecovery::Expired;
        }
        // Only this slot is locked during external recovery. Other scope
        // observations and independent native attempts do not wait on it.
        let proof = if let Some(proof) = state.recovery_proof.take() {
            match proof.downcast::<P>() {
                Ok(proof) => *proof,
                Err(proof) => {
                    state.recovery_proof = Some(proof);
                    return V3ScopeRecovery::WrongType;
                }
            }
        } else {
            let original = state
                .original
                .as_mut()
                .and_then(|o| o.downcast_mut::<T>())
                .expect("original type checked while holding the same slot");
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| attempt(original))) {
                Ok(Ok(proof)) => proof,
                Ok(Err(error)) => return V3ScopeRecovery::Pending(error),
                Err(payload) => return V3ScopeRecovery::Panic(payload),
            }
        };
        if std::time::Instant::now() >= until {
            state.recovery_proof = Some(Box::new(proof));
            return V3ScopeRecovery::Expired;
        }
        let mut slots = match registry().try_lock() {
            Ok(slots) => slots,
            Err(std::sync::TryLockError::WouldBlock) => {
                state.recovery_proof = Some(Box::new(proof));
                return V3ScopeRecovery::Busy;
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                state.recovery_proof = Some(Box::new(proof));
                return V3ScopeRecovery::Poisoned;
            }
        };
        let Some(index) = slots.iter().position(|slot| Arc::ptr_eq(slot, &self.0)) else {
            return V3ScopeRecovery::LostRegistration(proof);
        };
        // No other operation can access this slot while its mutex is held.
        // Remove only this exact slot and transfer every original owner.
        let original = state
            .original
            .take()
            .expect("typed original held through callback");
        let pins = std::mem::take(&mut state._pins);
        let recovery_key = state.recovery_key.take();
        slots.remove(index);
        V3ScopeRecovery::Recovered {
            proof,
            original,
            pins,
            recovery_key,
        }
    }
}

/// Discover currently observable retained scopes when an earlier caller no
/// longer has its key. A busy slot is omitted, so absence is not proof that
/// no scope exists. These identities authorize no release.
pub fn unresolved_keys() -> Vec<RoventV3ScopeKey> {
    registry()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .filter_map(|slot| {
            let state = match slot.state.try_lock() {
                Ok(guard) => guard.state,
                Err(std::sync::TryLockError::Poisoned(error)) => error.into_inner().state,
                Err(std::sync::TryLockError::WouldBlock) => return None,
            };
            matches!(state, V3ScopeState::Parked | V3ScopeState::Interrupted)
                .then(|| RoventV3ScopeKey(Arc::clone(slot)))
        })
        .collect()
}

pub(super) struct ScopeReservation {
    key: RoventV3ScopeKey,
    armed: bool,
    settled: bool,
}

impl ScopeReservation {
    /// The exact registered scope; observing this key alone is not a fence.
    pub(super) fn key(&self) -> &RoventV3ScopeKey {
        &self.key
    }

    /// The original factory proved no native entry. This is not a reservation
    /// recovery proof for any other resource.
    pub(super) fn mark_before_native(&mut self) {
        if let Ok(mut state) = self.key.0.state.lock() {
            state.terminal = V3ScopeTerminalEvidence::BeforeNative;
        }
    }

    /// The original product reported session release; this is not a device
    /// fence or proof for a different product.
    pub(super) fn mark_session_released(&mut self) {
        if let Ok(mut state) = self.key.0.state.lock() {
            state.terminal = V3ScopeTerminalEvidence::SessionReleased;
        }
    }

    /// Called only after this product's physical finish returned confirmed and
    /// its enclosing UCI call did not unwind.
    pub(super) fn mark_product_finished(&mut self, epoch: u64, input_lock_sha256: &str) {
        if let Ok(mut state) = self.key.0.state.lock() {
            state.product_finish = Some(V3ProductFinishObservation {
                process_id: std::process::id(),
                process_epoch: epoch,
                input_lock_sha256: input_lock_sha256.to_owned(),
            });
        }
    }

    pub(super) fn reserve(
        domain: V3ScopeDomain,
        pins: Box<dyn Any + Send>,
    ) -> Result<Self, Box<dyn Any + Send>> {
        let mut slots = registry().lock().unwrap_or_else(|e| e.into_inner());
        if slots.iter().filter(|slot| slot.domain == domain).count() >= SCOPE_LIMIT {
            return Err(pins);
        }
        let slot = Arc::new(Slot {
            domain,
            state: Mutex::new(SlotState {
                state: V3ScopeState::Reserved,
                terminal: V3ScopeTerminalEvidence::Unknown,
                product_finish: None,
                _pins: vec![pins],
                original: None,
                recovery_key: None,
                recovery_proof: None,
            }),
        });
        slots.push(Arc::clone(&slot));
        Ok(Self {
            key: RoventV3ScopeKey(slot),
            armed: false,
            settled: false,
        })
    }

    pub(super) fn arm(&mut self) {
        self.armed = true;
        self.key
            .0
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .state = V3ScopeState::Attempted;
    }

    /// Attach the actual assembly owner to this same registered scope.
    /// This never reopens resources or issues a completion proof.
    pub(super) fn attach_pin(&mut self, pin: Box<dyn Any + Send>) {
        self.key
            .0
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            ._pins
            .push(pin);
    }

    pub(super) fn park(
        mut self,
        original: Box<dyn Any + Send>,
        recovery_key: Option<TicketNativeRecoveryKey>,
    ) -> RoventV3ScopeKey {
        let mut state = self.key.0.state.lock().unwrap_or_else(|e| e.into_inner());
        state.original = Some(original);
        state.recovery_key = recovery_key;
        state.state = V3ScopeState::Parked;
        self.settled = true;
        self.key.clone()
    }

    /// Only the launch's already-established pre-entry or confirmed physical
    /// closure paths may call this. It does not itself establish either fact.
    pub(super) fn settled_without_quarantine(&mut self) {
        self.armed = false;
    }
}

impl Drop for ScopeReservation {
    fn drop(&mut self) {
        if self.settled {
            return;
        }
        if self.armed {
            self.key
                .0
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .state = V3ScopeState::Interrupted;
            return;
        }
        registry()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|slot| !Arc::ptr_eq(slot, &self.key.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Counted(Arc<AtomicUsize>);

    #[test]
    fn recovery_deadline_and_poison_do_not_release_original_owner() {
        let mut reservation = ScopeReservation::reserve(V3ScopeDomain::Cpu, Box::new(()))
            .ok()
            .unwrap();
        reservation.arm();
        let key = reservation.park(Box::new(31_u32), None);
        assert!(matches!(
            key.try_recover_original_until::<u32, (), ()>(std::time::Instant::now(), |_| {
                panic!("expired callback must not run")
            }),
            V3ScopeRecovery::Expired
        ));
        assert_eq!(key.with_original::<u32, _>(|value| *value), Some(31));
        let slot = Arc::clone(&key.0);
        let _ = std::thread::spawn(move || {
            let _guard = slot.state.lock().unwrap();
            panic!("poison original slot for recovery check");
        })
        .join();
        assert!(matches!(
            key.try_recover_original::<u32, (), ()>(|_| panic!("poisoned callback must not run")),
            V3ScopeRecovery::Poisoned
        ));
    }

    #[test]
    fn recovery_commit_busy_retains_original_proof_and_does_not_repeat_callback() {
        let mut reservation = ScopeReservation::reserve(V3ScopeDomain::Cpu, Box::new(()))
            .ok()
            .unwrap();
        reservation.arm();
        let key = reservation.park(Box::new(19_u32), None);
        let (request_tx, request_rx) = std::sync::mpsc::channel();
        let (held_tx, held_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let holder = std::thread::spawn(move || {
            request_rx.recv().unwrap();
            let _guard = registry().lock().unwrap();
            held_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        let proof = Arc::new(41_u32);
        let original_proof = Arc::clone(&proof);
        assert!(matches!(
            key.try_recover_original::<u32, Arc<u32>, ()>(|owner| {
                *owner += 1;
                request_tx.send(()).unwrap();
                held_rx.recv().unwrap();
                Ok(proof)
            }),
            V3ScopeRecovery::Busy
        ));
        release_tx.send(()).unwrap();
        holder.join().unwrap();
        match key.try_recover_original::<u32, Arc<u32>, ()>(|_| panic!("callback repeated")) {
            V3ScopeRecovery::Recovered {
                proof, original, ..
            } => {
                assert!(Arc::ptr_eq(&proof, &original_proof));
                assert_eq!(*original.downcast::<u32>().ok().unwrap(), 20);
            }
            _ => panic!("the same original proof was not committed"),
        }
    }

    #[test]
    fn terminal_evidence_belongs_to_original_key_even_after_retirement() {
        let mut first = ScopeReservation::reserve(V3ScopeDomain::Cpu, Box::new(()))
            .ok()
            .unwrap();
        let first_key = first.key().clone();
        assert!(first_key.product_finish_observation().is_none());
        assert_eq!(
            first_key.terminal_evidence(),
            V3ScopeTerminalEvidence::Unknown
        );
        first.mark_before_native();
        drop(first);
        assert_eq!(
            first_key.terminal_evidence(),
            V3ScopeTerminalEvidence::BeforeNative
        );

        let mut second = ScopeReservation::reserve(V3ScopeDomain::Cpu, Box::new(()))
            .ok()
            .unwrap();
        let second_key = second.key().clone();
        assert!(second_key.product_finish_observation().is_none());
        assert_eq!(
            second_key.terminal_evidence(),
            V3ScopeTerminalEvidence::Unknown
        );
        second.mark_session_released();
        assert!(second_key.product_finish_observation().is_none());
        second.mark_product_finished(37, "original-lock");
        drop(second);
        let finish = second_key.product_finish_observation().unwrap();
        assert_eq!(finish.process_id, std::process::id());
        assert_eq!(finish.process_epoch, 37);
        assert_eq!(finish.input_lock_sha256, "original-lock");
        assert!(first_key.product_finish_observation().is_none());
        assert_eq!(
            second_key.terminal_evidence(),
            V3ScopeTerminalEvidence::SessionReleased
        );
        assert_eq!(
            first_key.terminal_evidence(),
            V3ScopeTerminalEvidence::BeforeNative
        );
    }
    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn unidentified_cuda_scope_saturation_does_not_block_cpu_scope() {
        let mut cuda_keys = Vec::new();
        for _ in 0..SCOPE_LIMIT {
            let mut owner =
                ScopeReservation::reserve(V3ScopeDomain::CudaUnidentified, Box::new(()))
                    .ok()
                    .expect("isolated CUDA mock capacity");
            owner.arm();
            cuda_keys.push(owner.park(Box::new(()), None));
        }
        assert!(ScopeReservation::reserve(V3ScopeDomain::CudaUnidentified, Box::new(())).is_err());
        let cpu = ScopeReservation::reserve(V3ScopeDomain::Cpu, Box::new(()))
            .ok()
            .expect("independent CPU scope must remain available");
        drop(cpu);
        for key in cuda_keys {
            // Test-owned unit values are not GPU resources or fence evidence.
            assert!(matches!(
                key.try_recover_original::<(), (), ()>(|_| Ok(())),
                V3ScopeRecovery::Recovered { .. }
            ));
        }
    }

    #[test]
    fn cloned_pin_uses_arc_identity_without_calling_payload_clone() {
        struct Payload;
        impl Clone for Payload {
            fn clone(&self) -> Self {
                panic!("payload clone must not run under registry lock")
            }
        }
        let original = Arc::new(Payload);
        let mut owner =
            ScopeReservation::reserve(V3ScopeDomain::Cpu, Box::new(Arc::clone(&original)))
                .ok()
                .unwrap();
        owner.arm();
        let key = owner.park(Box::new(()), None);
        let obtained = key.cloned_pin::<Payload>().unwrap();
        assert!(Arc::ptr_eq(&original, &obtained));
        assert!(matches!(
            key.try_recover_original::<(), (), ()>(|_| Ok(())),
            V3ScopeRecovery::Recovered { .. }
        ));
    }

    #[test]
    fn dropped_token_preserves_parked_original_and_same_key_discovery() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut owner =
            ScopeReservation::reserve(V3ScopeDomain::Cpu, Box::new(Counted(Arc::clone(&drops))))
                .ok()
                .unwrap();
        owner.arm();
        let key = owner.park(Box::new(Counted(Arc::clone(&drops))), None);
        let discovered = unresolved_keys()
            .into_iter()
            .find(|k| Arc::ptr_eq(&k.0, &key.0))
            .unwrap();
        drop(key);
        let observation = discovered.observe().unwrap();
        assert_eq!(observation.state, V3ScopeState::Parked);
        assert!(observation.parked_original);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn typed_pin_and_original_are_same_scope_without_reentrant_blocking() {
        let pin = Arc::new(Mutex::new(17_u32));
        let mut owner = ScopeReservation::reserve(V3ScopeDomain::Cpu, Box::new(()))
            .ok()
            .unwrap();
        owner.attach_pin(Box::new(Arc::clone(&pin)));
        owner.arm();
        let key = owner.park(Box::new(String::from("original")), None);
        let returned = key.cloned_pin::<Mutex<u32>>().unwrap();
        assert!(Arc::ptr_eq(&pin, &returned));
        assert!(key.cloned_pin::<Mutex<u64>>().is_none());
        assert_eq!(
            key.with_original::<String, _>(|value| {
                assert!(key.observe().is_none());
                assert!(matches!(
                    key.reobserve_native(std::time::Instant::now()),
                    V3NativeReobservation::ScopeBusy
                ));
                value.clone()
            })
            .as_deref(),
            Some("original")
        );
        assert!(matches!(
            key.reobserve_native(std::time::Instant::now()),
            V3NativeReobservation::NativeKeyUnavailable {
                scope_state: V3ScopeState::Parked
            }
        ));
    }

    #[test]
    fn before_native_recovery_preserves_failure_and_transfers_exact_slot_only_on_success() {
        let pin = Arc::new(Mutex::new(17_u32));
        let mut owner = ScopeReservation::reserve(V3ScopeDomain::Cpu, Box::new(Arc::clone(&pin)))
            .ok()
            .unwrap();
        owner.arm();
        let key = owner.park(Box::new(vec![1_u32]), None);

        let mut other = ScopeReservation::reserve(V3ScopeDomain::Cpu, Box::new(23_u32))
            .ok()
            .unwrap();
        other.arm();
        let other_key = other.park(Box::new(29_u32), None);

        assert!(matches!(
            key.try_recover_original::<String, (), ()>(|_| Ok(())),
            V3ScopeRecovery::WrongType
        ));
        assert!(matches!(
            key.try_recover_original::<Vec<u32>, (), _>(|events| {
                // Partial recovery remains inside the same original owner.
                events.push(2);
                Err("pending")
            }),
            V3ScopeRecovery::Pending("pending")
        ));
        assert_eq!(
            key.with_original::<Vec<u32>, _>(Clone::clone),
            Some(vec![1, 2])
        );
        assert!(matches!(
            key.with_original::<Vec<u32>, _>(
                |_| key.try_recover_original::<Vec<u32>, (), ()>(|_| Ok(()))
            ),
            Some(V3ScopeRecovery::Busy)
        ));
        let original_address = key
            .with_original::<Vec<u32>, _>(|v| v as *const _ as usize)
            .unwrap();
        let recovered = key.try_recover_original::<Vec<u32>, _, ()>(|events| {
            assert_eq!(events.as_slice(), &[1, 2]);
            Ok(41_u32)
        });
        match recovered {
            V3ScopeRecovery::Recovered {
                proof,
                original,
                mut pins,
                recovery_key,
            } => {
                assert_eq!(proof, 41);
                assert!(recovery_key.is_none());
                let original = original.downcast::<Vec<u32>>().ok().unwrap();
                assert_eq!((&*original) as *const _ as usize, original_address);
                let actual_pin = pins
                    .pop()
                    .unwrap()
                    .downcast::<Arc<Mutex<u32>>>()
                    .ok()
                    .unwrap();
                assert!(Arc::ptr_eq(&pin, &actual_pin));
            }
            _ => panic!("the verified original slot was not transferred"),
        }
        assert!(matches!(key.try_observe(), V3ScopeAccess::Missing));
        assert!(matches!(other_key.try_observe(), V3ScopeAccess::Present(_)));
    }

    #[test]
    fn before_native_recovery_callback_panic_retains_original_slot() {
        let mut owner = ScopeReservation::reserve(V3ScopeDomain::Cpu, Box::new(()))
            .ok()
            .unwrap();
        owner.arm();
        let key = owner.park(Box::new(13_u32), None);
        let result = key.try_recover_original::<u32, (), ()>(|_| std::panic::panic_any(71_u64));
        match result {
            V3ScopeRecovery::Panic(payload) => {
                assert_eq!(*payload.downcast::<u64>().ok().unwrap(), 71)
            }
            _ => panic!("panic payload was not returned"),
        }
        assert_eq!(key.with_original::<u32, _>(|value| *value), Some(13));
    }

    #[test]
    fn panic_after_arm_retains_pins_as_interrupted() {
        let drops = Arc::new(AtomicUsize::new(0));
        let owner =
            ScopeReservation::reserve(V3ScopeDomain::Cpu, Box::new(Counted(Arc::clone(&drops))))
                .ok()
                .unwrap();
        let key = owner.key.clone();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let mut owner = owner;
            owner.arm();
            panic!("mock native failure");
        }));
        let observation = key.observe().unwrap();
        assert_eq!(observation.state, V3ScopeState::Interrupted);
        assert!(!observation.parked_original);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn confirmed_pre_entry_releases_slot_and_pins() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut owner =
            ScopeReservation::reserve(V3ScopeDomain::Cpu, Box::new(Counted(Arc::clone(&drops))))
                .ok()
                .unwrap();
        let key = owner.key.clone();
        owner.arm();
        owner.settled_without_quarantine();
        drop(owner);
        assert!(key.observe().is_none());
        drop(key);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
