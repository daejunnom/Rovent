//! Original /3 Collection source and native preparation.
//!
//! These helpers keep the /3 lock, process root and startup control intact.
//! Preparation does not grant root, teacher, publication or completion
//! authority. The launch caller must retain the observer's original custody
//! until the same physical worker and teacher work have been settled.

use super::collection_v2_observer::{V2ObserverCustody, V2ObserverReclaimError};
use super::{ConnectedParent, RoventLaunchError, error};
use crate::pals_adaptive::NativeQuestionProjection;
use crate::pals_native::ticketed::VirtualReviewCollectionObserver;
use crate::rovent_bootstrap::pc_collection_v3::RoventPcCollectionHooksV3;
use crate::rovent_bootstrap::{
    PreparedRoventNativeV3, RoventCollectionPreloadSources, RoventNativeAssetSelection,
    RoventNativeResourcesV3, RoventNativeStartupRejectionV3, RoventPinnedNativeSources,
    RoventStartupControl, RoventTeacherSourceSelection, V3ObserverIntent,
};
use cap_std::fs::Dir;
use rz_experiments::RoventInputLockV3;
use rz_runtime::contracts::ContractSystemClock;
use std::any::Any;
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

/// The caller and the original /3 process scope retain the same owner.
/// Neither retaining it nor observing Ready proves collection completion.
pub struct RoventPcCollectionLaunchOwnerV3 {
    teachers: Vec<RoventTeacherSourceSelection>,
    sealed_teacher_indices: Vec<usize>,
    hooks: Arc<dyn RoventPcCollectionHooksV3>,
    observer: Mutex<V2ObserverCustody>,
    proxy_issued: AtomicBool,
    native_session_finished: AtomicBool,
    settlement_claimed: AtomicBool,
    settlement_panic: Mutex<Option<Box<dyn Any + Send>>>,
    settlement_error: Mutex<Option<crate::pals_adaptive::AdaptiveManagerError>>,
    retained_settlement_custody: Mutex<Option<Box<dyn Any + Send>>>,
    observer_reclaimed: AtomicBool,
}

/// This launch's original native session has confirmed physical shutdown.
/// It is not a teacher-settlement receipt or a device-wide CUDA fence.
pub struct RoventPcCollectionNativeFenceV3 {
    owner: Arc<RoventPcCollectionLaunchOwnerV3>,
}

/// An original hook's retained launch-wide proof, bound to this native owner.
/// The hook implementation is the trusted boundary for the semantics of its
/// opaque evidence; the native fence alone cannot issue this permit.
pub struct RoventPcCollectionSettlementPermitV3 {
    owner: Arc<RoventPcCollectionLaunchOwnerV3>,
    hook_custody: Option<Box<dyn Any + Send>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventPcObserverReclaimErrorV3 {
    WrongOwner,
    ProxyStillOwned,
    CustodyBusy,
    Poisoned,
    AlreadyReclaimed,
}

impl RoventPcCollectionSettlementPermitV3 {
    pub fn matches_original_owner(&self, owner: &Arc<RoventPcCollectionLaunchOwnerV3>) -> bool {
        Arc::ptr_eq(&self.owner, owner)
    }

    /// The original hook evidence remains owned until its observer is safely
    /// reclaimed. Returning it transfers custody rather than discarding it.
    pub fn take_hook_custody(&mut self) -> Option<Box<dyn Any + Send>> {
        if self.owner.observer_reclaimed.load(Ordering::Acquire) {
            self.hook_custody.take()
        } else {
            None
        }
    }
}

impl Drop for RoventPcCollectionSettlementPermitV3 {
    fn drop(&mut self) {
        if let Some(custody) = self.hook_custody.take() {
            let mut slot = self
                .owner
                .retained_settlement_custody
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            if slot.is_none() {
                *slot = Some(custody);
            } else {
                // Never overwrite an independently retained original.
                std::mem::forget(custody);
            }
        }
    }
}

impl RoventPcCollectionNativeFenceV3 {
    /// Verify the exact hook object installed in this native launch.
    pub fn matches_original_hooks(&self, hooks: &Arc<dyn RoventPcCollectionHooksV3>) -> bool {
        Arc::ptr_eq(&self.owner.hooks, hooks)
    }

    pub fn matches_original_owner(&self, owner: &Arc<RoventPcCollectionLaunchOwnerV3>) -> bool {
        Arc::ptr_eq(&self.owner, owner)
    }
}

impl RoventPcCollectionLaunchOwnerV3 {
    pub fn new(
        teachers: Vec<RoventTeacherSourceSelection>,
        sealed_teacher_indices: Vec<usize>,
        hooks: Arc<dyn RoventPcCollectionHooksV3>,
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
            retained_settlement_custody: Mutex::new(None),
            observer_reclaimed: AtomicBool::new(false),
        }
    }

    /// Tests the original installed hook object, not the type or contents of
    /// an equivalent-looking replacement. This grants no launch authority.
    pub fn matches_original_hooks(&self, hooks: &Arc<dyn RoventPcCollectionHooksV3>) -> bool {
        Arc::ptr_eq(&self.hooks, hooks)
    }

    pub(super) fn hooks(&self) -> &Arc<dyn RoventPcCollectionHooksV3> {
        &self.hooks
    }

    pub(super) fn teachers(&self) -> &[RoventTeacherSourceSelection] {
        &self.teachers
    }

    pub(super) fn sealed_teacher_indices(&self) -> &[usize] {
        &self.sealed_teacher_indices
    }

    /// Return the original observer only before a proxy has been issued.
    /// This establishes no worker was given this observer; it says nothing
    /// about other native work or the GPU.
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
            self.observer_reclaimed.store(true, Ordering::Release);
        }
        result
    }

    /// The original observer stays in caller/scope custody while its proxy is
    /// moved into the same native factory. Proxy creation is not a start or a
    /// completion witness.
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
            Ok(observer) => observer.proxy(),
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

    /// Called only by the original launch after its product has confirmed
    /// physical session completion and no panic escaped its UCI loop. This
    /// says nothing about separate teacher work or device-wide activity.
    pub(super) fn mark_native_session_finished(&self) {
        self.native_session_finished.store(true, Ordering::Release);
    }

    /// Issue a capability tied to this exact owner only after its native
    /// session physically finishes. Separate collection owners still need
    /// their own same-root and launch-wide settlement evidence.
    pub fn native_session_fence(self: &Arc<Self>) -> Option<RoventPcCollectionNativeFenceV3> {
        self.native_session_finished
            .load(Ordering::Acquire)
            .then(|| RoventPcCollectionNativeFenceV3 {
                owner: Arc::clone(self),
            })
    }

    /// Recover the same proof if its original permit was dropped. Busy and
    /// poison are not interpreted as proof absence.
    pub fn try_recover_settlement_permit(
        self: &Arc<Self>,
    ) -> Result<Option<RoventPcCollectionSettlementPermitV3>, &'static str> {
        let mut slot =
            self.retained_settlement_custody
                .try_lock()
                .map_err(|failure| match failure {
                    std::sync::TryLockError::WouldBlock => "settlement custody busy",
                    std::sync::TryLockError::Poisoned(_) => "settlement custody poisoned",
                })?;
        Ok(slot
            .take()
            .map(|custody| RoventPcCollectionSettlementPermitV3 {
                owner: Arc::clone(self),
                hook_custody: Some(custody),
            }))
    }

    /// Ask only the hook installed by this owner to seal its original
    /// launch-wide evidence. An error or panic is retained as unresolved;
    /// only an explicit pending result may be retried automatically.
    pub fn try_settle_launch(
        self: &Arc<Self>,
    ) -> Result<Option<RoventPcCollectionSettlementPermitV3>, &'static str> {
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
            Ok(Ok(Some(custody))) => Ok(Some(RoventPcCollectionSettlementPermitV3 {
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

    /// Preserve the typed original cause independently of the retry status.
    pub fn settlement_error(&self) -> Option<crate::pals_adaptive::AdaptiveManagerError> {
        self.settlement_error.lock().ok()?.clone()
    }

    /// Reclaim only this owner's observer after native completion and the
    /// exact installed hook's launch-wide custody have both been sealed.
    pub fn try_reclaim_observer(
        self: &Arc<Self>,
        permit: &RoventPcCollectionSettlementPermitV3,
    ) -> Result<Box<dyn VirtualReviewCollectionObserver>, RoventPcObserverReclaimErrorV3> {
        if !Arc::ptr_eq(self, &permit.owner)
            || permit.hook_custody.is_none()
            || !self.native_session_finished.load(Ordering::Acquire)
        {
            return Err(RoventPcObserverReclaimErrorV3::WrongOwner);
        }
        let mut observer = self.observer.try_lock().map_err(|failure| match failure {
            std::sync::TryLockError::WouldBlock => RoventPcObserverReclaimErrorV3::CustodyBusy,
            std::sync::TryLockError::Poisoned(_) => RoventPcObserverReclaimErrorV3::Poisoned,
        })?;
        let original = observer
            .try_reclaim_after_verified_boundary()
            .map_err(|failure| match failure {
                V2ObserverReclaimError::ProxyStillOwned => {
                    RoventPcObserverReclaimErrorV3::ProxyStillOwned
                }
                V2ObserverReclaimError::Poisoned | V2ObserverReclaimError::NotPoisoned => {
                    RoventPcObserverReclaimErrorV3::Poisoned
                }
                V2ObserverReclaimError::AlreadyReclaimed => {
                    RoventPcObserverReclaimErrorV3::AlreadyReclaimed
                }
            })?;
        self.observer_reclaimed.store(true, Ordering::Release);
        Ok(original)
    }

    /// Explicitly recover the original after an observer callback panic.
    /// The caller must retain the accompanying poison indication; this does
    /// not convert that callback failure into a successful collection.
    pub fn try_reclaim_poisoned_observer(
        self: &Arc<Self>,
        permit: &RoventPcCollectionSettlementPermitV3,
    ) -> Result<Box<dyn VirtualReviewCollectionObserver>, RoventPcObserverReclaimErrorV3> {
        if !Arc::ptr_eq(self, &permit.owner)
            || permit.hook_custody.is_none()
            || !self.native_session_finished.load(Ordering::Acquire)
        {
            return Err(RoventPcObserverReclaimErrorV3::WrongOwner);
        }
        let mut observer = self.observer.try_lock().map_err(|failure| match failure {
            std::sync::TryLockError::WouldBlock => RoventPcObserverReclaimErrorV3::CustodyBusy,
            std::sync::TryLockError::Poisoned(_) => RoventPcObserverReclaimErrorV3::Poisoned,
        })?;
        let (original, _) = observer
            .reclaim_poisoned_after_verified_boundary()
            .map_err(|failure| match failure {
                V2ObserverReclaimError::ProxyStillOwned => {
                    RoventPcObserverReclaimErrorV3::ProxyStillOwned
                }
                V2ObserverReclaimError::Poisoned | V2ObserverReclaimError::NotPoisoned => {
                    RoventPcObserverReclaimErrorV3::Poisoned
                }
                V2ObserverReclaimError::AlreadyReclaimed => {
                    RoventPcObserverReclaimErrorV3::AlreadyReclaimed
                }
            })?;
        self.observer_reclaimed.store(true, Ordering::Release);
        Ok(original)
    }
}

impl Drop for RoventPcCollectionLaunchOwnerV3 {
    fn drop(&mut self) {
        if !self.proxy_issued.load(Ordering::Acquire) {
            // No proxy was ever handed to a worker, so reclaiming here cannot
            // release an in-flight observer or assert device completion.
            let owner = self.observer.get_mut().unwrap_or_else(|e| e.into_inner());
            let _ = owner.try_reclaim_after_verified_boundary();
        }
        if self.proxy_issued.load(Ordering::Acquire)
            && !self.observer_reclaimed.load(Ordering::Acquire)
        {
            // Preserve the original hook and failure payload if every caller
            // loses the explicit recovery owner while work may remain live.
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
        if let Some(custody) = self
            .retained_settlement_custody
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            std::mem::forget(custody);
        }
    }
}

/// Open the teacher preload and the native inputs from the same original
/// lock, directory and startup control. No teacher or native work starts.
pub(super) fn open_collection_sources(
    lock: Arc<RoventInputLockV3>,
    root: &Dir,
    cwd: PathBuf,
    selection: RoventNativeAssetSelection,
    control: &RoventStartupControl,
    teachers: Vec<RoventTeacherSourceSelection>,
    sealed_teacher_indices: &[usize],
    strict_source: bool,
) -> Result<Arc<RoventPinnedNativeSources<RoventInputLockV3>>, RoventLaunchError> {
    let teacher_root = root.try_clone().map_err(|failure| error("Cwd", failure))?;
    let preload = if sealed_teacher_indices.is_empty() {
        RoventCollectionPreloadSources::open_v3(
            Arc::clone(&lock),
            teacher_root,
            &cwd,
            teachers,
            control,
        )
    } else {
        RoventCollectionPreloadSources::open_v3_with_sealed_policy(
            Arc::clone(&lock),
            teacher_root,
            &cwd,
            teachers,
            control,
            sealed_teacher_indices,
        )
    }
    .map_err(|failure| error("CollectionSources", failure))?;
    let source_root = root.try_clone().map_err(|failure| error("Cwd", failure))?;
    RoventPinnedNativeSources::open_v3_with_collection(
        lock,
        source_root,
        cwd,
        selection,
        control,
        Arc::new(preload),
        strict_source,
    )
    .map(Arc::new)
    .map_err(|failure| error("Assets", failure))
}

/// Start the original Collection-intent factory with the authenticated
/// parent received by this /3 launch. A missing system-CUDA parent is rejected
/// by the existing native-entry check; CPU behavior remains independent.
pub(super) fn prepare_collection_with_parent<Q: NativeQuestionProjection>(
    sources: Arc<RoventPinnedNativeSources<RoventInputLockV3>>,
    resources: RoventNativeResourcesV3,
    clock: ContractSystemClock,
    control: RoventStartupControl,
    observer: Box<dyn VirtualReviewCollectionObserver>,
    parent: Option<Arc<ConnectedParent>>,
) -> Result<PreparedRoventNativeV3<Q>, RoventNativeStartupRejectionV3<Q>> {
    super::super::prepare_native_v3_inner(
        sources,
        resources,
        clock,
        control,
        Some(observer),
        parent,
        V3ObserverIntent::Collection,
    )
}
