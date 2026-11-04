//! Original /3 Collection source and native preparation.
//!
//! These helpers keep the /3 lock, process root and startup control intact.
//! Preparation does not grant root, teacher, publication or completion
//! authority. The launch caller must retain the observer's original custody
//! until the same physical worker and teacher work have been settled.

use super::collection_v2_observer::V2ObserverCustody;
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
        }
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
}

impl Drop for RoventPcCollectionLaunchOwnerV3 {
    fn drop(&mut self) {
        if !self.proxy_issued.load(Ordering::Acquire) {
            // No proxy was ever handed to a worker, so reclaiming here cannot
            // release an in-flight observer or assert device completion.
            let owner = self.observer.get_mut().unwrap_or_else(|e| e.into_inner());
            let _ = owner.try_reclaim_after_verified_boundary();
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
