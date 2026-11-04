//! Own-R Collection launch custody.  Native and teacher completion are
//! separate; this owner never derives either from a receipt's presence.

use super::collection_v2_observer::{V2ObserverCustody, V2ObserverReclaimError};
use super::{RoventLaunchError, error};
use crate::pals_adaptive::{AdaptiveManagerError, RoventBoardCollectionHooksV3};
use crate::pals_native::ticketed::VirtualReviewCollectionObserver;
use crate::rovent_bootstrap::{RoventNativeBackend, RoventTeacherSourceSelection};
use rz_search::pals::engine::adaptive_port::AdaptivePortQuestion;
use std::any::Any;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

pub type RoventOwnCollectionHooksV3 =
    dyn RoventBoardCollectionHooksV3<RoventNativeBackend<AdaptivePortQuestion>>;

/// Keeps the exact hook and observer for this original Own-R launch.
pub struct RoventOwnCollectionLaunchOwnerV3 {
    teachers: Vec<RoventTeacherSourceSelection>,
    sealed_teacher_indices: Vec<usize>,
    hooks: Arc<RoventOwnCollectionHooksV3>,
    observer: Mutex<V2ObserverCustody>,
    proxy_issued: AtomicBool,
    native_session_finished: AtomicBool,
    settlement_claimed: AtomicBool,
    settlement_panic: Mutex<Option<Box<dyn Any + Send>>>,
    settlement_error: Mutex<Option<AdaptiveManagerError>>,
    observer_reclaimed: AtomicBool,
}

/// Same-launch native completion only; it is not a teacher or device fence.
pub struct RoventOwnCollectionNativeFenceV3 {
    owner: Arc<RoventOwnCollectionLaunchOwnerV3>,
}

/// Original hook custody bound to the same native session.
pub struct RoventOwnCollectionSettlementPermitV3 {
    owner: Arc<RoventOwnCollectionLaunchOwnerV3>,
    hook_custody: Option<Box<dyn Any + Send>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventOwnObserverReclaimErrorV3 {
    WrongOwner,
    ProxyStillOwned,
    Poisoned,
    AlreadyReclaimed,
}

impl RoventOwnCollectionNativeFenceV3 {
    pub fn matches_original_hooks(&self, hooks: &Arc<RoventOwnCollectionHooksV3>) -> bool {
        Arc::ptr_eq(&self.owner.hooks, hooks)
    }

    pub fn matches_original_owner(&self, owner: &Arc<RoventOwnCollectionLaunchOwnerV3>) -> bool {
        Arc::ptr_eq(&self.owner, owner)
    }
}

impl RoventOwnCollectionSettlementPermitV3 {
    pub fn take_hook_custody(&mut self) -> Option<Box<dyn Any + Send>> {
        if self.owner.observer_reclaimed.load(Ordering::Acquire) {
            self.hook_custody.take()
        } else {
            None
        }
    }
}

impl Drop for RoventOwnCollectionSettlementPermitV3 {
    fn drop(&mut self) {
        if let Some(custody) = self.hook_custody.take() {
            std::mem::forget(custody);
        }
    }
}

impl RoventOwnCollectionLaunchOwnerV3 {
    pub fn new(
        teachers: Vec<RoventTeacherSourceSelection>,
        sealed_teacher_indices: Vec<usize>,
        hooks: Arc<RoventOwnCollectionHooksV3>,
        observer: Box<dyn VirtualReviewCollectionObserver>,
    ) -> Self {
        Self {
            teachers,
            sealed_teacher_indices,
            hooks,
            observer: Mutex::new(V2ObserverCustody::new(observer)),
            proxy_issued: AtomicBool::new(false),
            native_session_finished: AtomicBool::new(false),
            settlement_claimed: AtomicBool::new(false),
            settlement_panic: Mutex::new(None),
            settlement_error: Mutex::new(None),
            observer_reclaimed: AtomicBool::new(false),
        }
    }

    pub(super) fn hooks(&self) -> &Arc<RoventOwnCollectionHooksV3> {
        &self.hooks
    }
    pub(super) fn teachers(&self) -> &[RoventTeacherSourceSelection] {
        &self.teachers
    }
    pub(super) fn sealed_teacher_indices(&self) -> &[usize] {
        &self.sealed_teacher_indices
    }

    /// Reclaim only before any proxy has been issued to a worker.
    pub fn take_unissued_observer(
        &self,
    ) -> Result<Box<dyn VirtualReviewCollectionObserver>, &'static str> {
        if self
            .proxy_issued
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err("original observer proxy already issued");
        }
        let result = match self.observer.lock() {
            Ok(mut owner) => owner
                .try_reclaim_after_verified_boundary()
                .map_err(|_| "original observer cannot be reclaimed"),
            Err(_) => Err("original observer custody poisoned"),
        };
        if result.is_err() {
            self.proxy_issued.store(false, Ordering::Release);
        } else {
            // No worker received a proxy; normal pre-start return can release
            // this launch's hook without entering the unresolved Drop path.
            self.observer_reclaimed.store(true, Ordering::Release);
        }
        result
    }

    pub(super) fn observer_proxy(
        &self,
    ) -> Result<Box<dyn VirtualReviewCollectionObserver>, RoventLaunchError> {
        if self
            .proxy_issued
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(error("CollectionObserver", "original proxy already issued"));
        }
        let proxy = match self.observer.lock() {
            Ok(owner) => owner.proxy(),
            Err(_) => {
                self.proxy_issued.store(false, Ordering::Release);
                return Err(error(
                    "CollectionObserver",
                    "original observer custody poisoned",
                ));
            }
        };
        match proxy {
            Some(proxy) => Ok(proxy),
            None => {
                self.proxy_issued.store(false, Ordering::Release);
                Err(error(
                    "CollectionObserver",
                    "original observer already reclaimed",
                ))
            }
        }
    }

    /// Called only by the original launch after confirmed session closure and
    /// a non-panicking UCI return. It does not settle teacher work.
    pub(super) fn mark_native_session_finished(&self) {
        self.native_session_finished.store(true, Ordering::Release);
    }

    pub fn native_session_fence(self: &Arc<Self>) -> Option<RoventOwnCollectionNativeFenceV3> {
        self.native_session_finished
            .load(Ordering::Acquire)
            .then(|| RoventOwnCollectionNativeFenceV3 {
                owner: Arc::clone(self),
            })
    }

    /// Only the installed hook may provide launch-wide custody. Pending is
    /// retryable; errors and panics remain attached to this original owner.
    pub fn try_settle_launch(
        self: &Arc<Self>,
    ) -> Result<Option<RoventOwnCollectionSettlementPermitV3>, &'static str> {
        let Some(fence) = self.native_session_fence() else {
            return Ok(None);
        };
        if self
            .settlement_claimed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err("launch settlement already claimed or unresolved");
        }
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.hooks.try_finish_launch(&fence)
        })) {
            Ok(Ok(Some(custody))) => Ok(Some(RoventOwnCollectionSettlementPermitV3 {
                owner: Arc::clone(self),
                hook_custody: Some(custody),
            })),
            Ok(Ok(None)) => {
                self.settlement_claimed.store(false, Ordering::Release);
                Ok(None)
            }
            Ok(Err(failure)) => {
                let mut retained = self
                    .settlement_error
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                if retained.is_none() {
                    *retained = Some(failure);
                }
                Err("original hook settlement failed and remains unresolved")
            }
            Err(payload) => {
                let mut retained = self
                    .settlement_panic
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                if retained.is_none() {
                    *retained = Some(payload);
                } else {
                    std::mem::forget(payload);
                }
                Err("original hook settlement panicked and remains unresolved")
            }
        }
    }

    pub fn settlement_error(&self) -> Option<AdaptiveManagerError> {
        self.settlement_error.lock().ok()?.clone()
    }

    pub fn try_reclaim_observer(
        self: &Arc<Self>,
        permit: &RoventOwnCollectionSettlementPermitV3,
    ) -> Result<Box<dyn VirtualReviewCollectionObserver>, RoventOwnObserverReclaimErrorV3> {
        if !Arc::ptr_eq(self, &permit.owner)
            || permit.hook_custody.is_none()
            || !self.native_session_finished.load(Ordering::Acquire)
        {
            return Err(RoventOwnObserverReclaimErrorV3::WrongOwner);
        }
        let mut observer = self
            .observer
            .lock()
            .map_err(|_| RoventOwnObserverReclaimErrorV3::Poisoned)?;
        let original = observer
            .try_reclaim_after_verified_boundary()
            .map_err(|failure| match failure {
                V2ObserverReclaimError::ProxyStillOwned => {
                    RoventOwnObserverReclaimErrorV3::ProxyStillOwned
                }
                V2ObserverReclaimError::Poisoned | V2ObserverReclaimError::NotPoisoned => {
                    RoventOwnObserverReclaimErrorV3::Poisoned
                }
                V2ObserverReclaimError::AlreadyReclaimed => {
                    RoventOwnObserverReclaimErrorV3::AlreadyReclaimed
                }
            })?;
        self.observer_reclaimed.store(true, Ordering::Release);
        Ok(original)
    }
}

impl Drop for RoventOwnCollectionLaunchOwnerV3 {
    fn drop(&mut self) {
        if !self.proxy_issued.load(Ordering::Acquire) {
            let owner = self.observer.get_mut().unwrap_or_else(|e| e.into_inner());
            let _ = owner.try_reclaim_after_verified_boundary();
        } else if !self.observer_reclaimed.load(Ordering::Acquire) {
            // Last-resort quarantine only.  Normal recovery must keep the
            // original launch owner reachable; leaking these values does not
            // establish physical completion or make recovery possible.
            std::mem::forget(Arc::clone(&self.hooks));
            if let Some(payload) = self
                .settlement_panic
                .get_mut()
                .unwrap_or_else(|e| e.into_inner())
                .take()
            {
                std::mem::forget(payload);
            }
            if let Some(failure) = self
                .settlement_error
                .get_mut()
                .unwrap_or_else(|e| e.into_inner())
                .take()
            {
                std::mem::forget(failure);
            }
        }
    }
}
