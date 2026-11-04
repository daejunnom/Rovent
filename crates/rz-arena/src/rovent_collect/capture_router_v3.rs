//! Route the original /3 captures through the one process-owned receiver.
//!
//! A root asking for its captures may encounter another root's input first.
//! Keep that input and its admission together until its own observer requests
//! it; dequeueing here is neither execution nor physical-completion evidence.
use super::native_capture::{
    RoventCapturedInputReceiver, RoventFailedCapturedInput, RoventNativeCaptureRegistration,
    RoventNativeInputObserver, rovent_native_capture_channel,
};
use super::producer_v3::RoventCapturedInputV3;
use super::{ArenaError, invalid};
use rz_uci::pals_adaptive::RoventCollectionRootAdmissionV3;
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, TryLockError},
};

// A finite routing allowance independent of the capture channel capacity.
// Reaching it yields Busy, preserving the queued native input for a later poll.
const MAX_ROUTED_CAPTURES: usize = 64;

pub(super) enum CapturePollV3 {
    Ready(RoventCapturedInputV3),
    Empty,
    Busy,
}

pub(super) enum FailurePollV3 {
    Ready(RoventFailedCapturedInput),
    Empty,
    Busy,
}

struct State {
    receiver: RoventCapturedInputReceiver,
    retained: VecDeque<(RoventCapturedInputV3, RoventCollectionRootAdmissionV3)>,
}

pub(super) struct CaptureRouterV3 {
    state: Mutex<State>,
    registration: RoventNativeCaptureRegistration,
}

impl CaptureRouterV3 {
    /// Both halves are produced by the same native channel. Callers cannot
    /// pair a registration with another receiver through this interface.
    pub(super) fn channel() -> (Arc<Self>, RoventNativeInputObserver) {
        let (registration, observer, receiver) = rovent_native_capture_channel();
        (
            Arc::new(Self {
                registration,
                state: Mutex::new(State {
                    receiver,
                    retained: VecDeque::new(),
                }),
            }),
            observer,
        )
    }

    pub(super) fn registration(&self) -> RoventNativeCaptureRegistration {
        self.registration.clone()
    }

    pub(super) fn try_take(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<CapturePollV3, ArenaError> {
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Ok(CapturePollV3::Busy),
            Err(TryLockError::Poisoned(_)) => {
                return Err(invalid("lock-v3 capture router poisoned"));
            }
        };
        if let Some(failure) = state.receiver.capture_failure_v3(admission)? {
            return Err(invalid(format!(
                "native lock-v3 input capture {:?} failed for {:?}: {:?}",
                failure.stage, failure.ticket.id, failure.cause
            )));
        }
        if let Some(index) = state
            .retained
            .iter()
            .position(|(_, owner)| owner.same_original_admission(admission))
        {
            let (captured, owner) = state.retained.remove(index).expect("located capture");
            if !captured.admission().same_original_admission(&owner) {
                state.retained.insert(index, (captured, owner));
                return Err(invalid(
                    "lock-v3 captured input and original admission differ",
                ));
            }
            return Ok(CapturePollV3::Ready(captured));
        }
        loop {
            if state.retained.len() >= MAX_ROUTED_CAPTURES {
                return Ok(CapturePollV3::Busy);
            }
            state
                .retained
                .try_reserve(1)
                .map_err(|_| invalid("lock-v3 capture router allocation failed"))?;
            let Some(input) = state.receiver.try_take_lock_v3(admission)? else {
                return Ok(CapturePollV3::Empty);
            };
            let (captured, owner) = input.into_original();
            if !captured.admission().same_original_admission(&owner) {
                state.retained.push_back((captured, owner));
                return Err(invalid(
                    "lock-v3 captured input and original admission differ",
                ));
            }
            if owner.same_original_admission(admission) {
                return Ok(CapturePollV3::Ready(captured));
            }
            state.retained.push_back((captured, owner));
        }
    }

    /// Caller retains the original failed input for this admission. This is
    /// custody transfer only and is not a native or teacher completion fence.
    pub(super) fn try_take_failure(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<FailurePollV3, ArenaError> {
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Ok(FailurePollV3::Busy),
            Err(TryLockError::Poisoned(_)) => {
                return Err(invalid("lock-v3 capture router poisoned"));
            }
        };
        Ok(
            match state.receiver.take_failed_input_for_cleanup_v3(admission)? {
                Some(input) => FailurePollV3::Ready(input),
                None => FailurePollV3::Empty,
            },
        )
    }

    /// Custody-only access after the caller has verified the original root's
    /// native fence. Unlike normal dispatch, this remains usable after a
    /// recorded capture failure. It never makes the input publishable.
    pub(super) fn try_take_for_cleanup(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<CapturePollV3, ArenaError> {
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Ok(CapturePollV3::Busy),
            Err(TryLockError::Poisoned(_)) => {
                return Err(invalid("lock-v3 capture router poisoned"));
            }
        };
        if let Some(index) = state
            .retained
            .iter()
            .position(|(_, owner)| owner.same_original_admission(admission))
        {
            let (captured, owner) = state.retained.remove(index).expect("located capture");
            if !captured.admission().same_original_admission(&owner) {
                state.retained.insert(index, (captured, owner));
                return Err(invalid("lock-v3 cleanup input and admission differ"));
            }
            return Ok(CapturePollV3::Ready(captured));
        }
        loop {
            if state.retained.len() >= MAX_ROUTED_CAPTURES {
                return Ok(CapturePollV3::Busy);
            }
            state
                .retained
                .try_reserve(1)
                .map_err(|_| invalid("lock-v3 capture router allocation failed"))?;
            let Some(input) = state.receiver.take_queued_lock_v3_for_cleanup() else {
                return Ok(CapturePollV3::Empty);
            };
            let (captured, owner) = input.into_original();
            if !captured.admission().same_original_admission(&owner) {
                state.retained.push_back((captured, owner));
                return Err(invalid("lock-v3 cleanup input and admission differ"));
            }
            if owner.same_original_admission(admission) {
                return Ok(CapturePollV3::Ready(captured));
            }
            state.retained.push_back((captured, owner));
        }
    }
}
