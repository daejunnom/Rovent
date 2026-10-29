//! Collection 전용 조립 callback을 위한 공유 진입점.
//!
//! 부모 launch가 원 lock, teacher preload, native service와 실제 프로세스
//! 경계를 확인한 뒤에만 이 payload를 만든다. 이 모듈은 질문이나 authority를
//! 만들지 않으며, 동일한 prepared owner와 원 FD의 소유권만 callback에 넘긴다.

use super::{LaunchRetirementEvidence, RoventLaunchError};
use crate::EngineIdentity;
use crate::engine::{EngineSettings, OwnerRegistry, ProcessClock};
use crate::pals_native::ticketed::VirtualReviewCollectionObserver;
use crate::rovent_bootstrap::nnuec_source::LoadedRoventNnuecSource;
use crate::rovent_bootstrap::product::{RoventNativeProductDriver, RoventProductStartupRejection};
use crate::rovent_bootstrap::product::RoventOwnPcrV2StartupRejection;
use crate::rovent_bootstrap::pc_session::RoventPcCollectionHooksV2;
use crate::rovent_bootstrap::nnuec_session::RoventNnuecCollectionHooksV2;
use crate::pals_adaptive::RoventBoardCollectionHooksV2;
use crate::rovent_bootstrap::{
    PreparedRoventNative, RoventCollectionPreloadSources, RoventPinnedExternalRSource,
    RoventStartupControl, RoventTeacherSourceSelection,
};
use crate::rovent_root_admission::RoventRootAdmissionPolicy;
use rz_experiments::{RoventInputLock, RoventInputLockV2};
use rz_search::pals::engine::{adaptive_port::AdaptivePortQuestion, pc_port::PcPortQuestion};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use super::collection_v2_observer::{V2ObserverCustody, V2ObserverReclaimError};

/// Explicit V2 P/C collection inputs. These owners are selected before the
/// original lock is opened; they never project a V2 input lock into V1.
pub(super) enum CollectionHooksV2 {
    Pc(Arc<dyn RoventPcCollectionHooksV2>),
    OwnNnuec(Arc<dyn RoventNnuecCollectionHooksV2>),
    OwnBoard(Arc<RoventOwnBoardCollectionHooksV2>),
}

pub type RoventOwnBoardCollectionHooksV2 = dyn RoventBoardCollectionHooksV2<
    crate::rovent_bootstrap::RoventNativeBackend<AdaptivePortQuestion>,
>;

pub(super) struct CollectionLaunchV2<'a> {
    pub(super) teacher_selections: Vec<RoventTeacherSourceSelection>,
    pub(super) observer: Box<dyn VirtualReviewCollectionObserver>,
    pub(super) hooks: CollectionHooksV2,
    pub(super) admit_resource: Box<dyn FnOnce(&Arc<RoventInputLockV2>) -> Result<(), RoventLaunchError> + 'a>,
    pub(super) startup_attempt: Arc<AtomicBool>,
    pub(super) native_settled: Arc<AtomicBool>,
    pub(super) board_product_custody: Arc<Mutex<Option<RoventOwnPcrV2StartupRejection>>>,
}

/// Startup/session failure returns the same hook Arc to the caller. It does
/// not assert that the hook's own external work has stopped. Unknown physical
/// completion remains quarantined if the failure is dropped without recovery.
pub struct RoventCollectionLaunchFailureV2<H: ?Sized> {
    pub error: RoventLaunchError,
    hooks: Option<Arc<H>>,
    panic_payload: Option<Box<dyn std::any::Any + Send>>,
    observer_before_start: Option<Box<dyn VirtualReviewCollectionObserver>>,
    retained_observer: Option<V2ObserverCustody>,
    observer_reclaim_error: Option<V2ObserverReclaimError>,
    board_product_custody: Option<RoventOwnPcrV2StartupRejection>,
    physical_unknown: bool,
}

pub type RoventPcCollectionLaunchFailureV2 = RoventCollectionLaunchFailureV2<dyn RoventPcCollectionHooksV2>;
pub type RoventOwnNnuecCollectionLaunchFailureV2 = RoventCollectionLaunchFailureV2<dyn RoventNnuecCollectionHooksV2>;
pub type RoventOwnBoardCollectionLaunchFailureV2 = RoventCollectionLaunchFailureV2<RoventOwnBoardCollectionHooksV2>;

/// Retains an actual Board product rejection while its physical owners are
/// unresolved. Borrowing permits inspection without releasing ownership.
pub struct RoventBoardProductCustody {
    original: Option<RoventOwnPcrV2StartupRejection>,
}

impl RoventBoardProductCustody {
    pub fn original(&self) -> &RoventOwnPcrV2StartupRejection {
        self.original.as_ref().expect("original Board product custody")
    }
}

impl Drop for RoventBoardProductCustody {
    fn drop(&mut self) {
        if let Some(original) = self.original.take() {
            // No API has established both native and teacher retirement yet.
            std::mem::forget(original);
        }
    }
}

impl<H: ?Sized> RoventCollectionLaunchFailureV2<H> {
    pub fn hooks(&self) -> &Arc<H> {
        self.hooks.as_ref().expect("original collection hooks retained")
    }

    /// Transfer the original hook custody to a caller that will retain it
    /// until its own teacher/process completion has been established.
    pub fn take_hook_custody(&mut self) -> Option<Arc<H>> {
        self.hooks.take()
    }

    /// Return the original observer after a verified no-worker or completed
    /// native boundary. Teacher and publication are separate obligations.
    pub fn take_settled_observer(&mut self) -> Option<Box<dyn VirtualReviewCollectionObserver>> {
        self.observer_before_start.take()
    }

    /// Retained custody is not evidence of completion or safe reclamation.
    pub fn observer_custody_retained(&self) -> bool {
        self.retained_observer.is_some()
    }

    pub fn observer_reclaim_failure(&self) -> Option<&'static str> {
        self.observer_reclaim_error.map(|error| match error {
            V2ObserverReclaimError::ProxyStillOwned => "proxy_still_owned",
            V2ObserverReclaimError::Poisoned => "poisoned",
            V2ObserverReclaimError::NotPoisoned => "not_poisoned",
            V2ObserverReclaimError::AlreadyReclaimed => "already_reclaimed",
        })
    }

    pub fn physical_completion_unknown(&self) -> bool { self.physical_unknown }

    /// Transfer the exact Board product rejection; this transfers custody but
    /// does not establish that its native or teacher owners have finished.
    pub fn take_board_product_custody(&mut self) -> Option<RoventBoardProductCustody> {
        self.board_product_custody.take().map(|original| RoventBoardProductCustody {
            original: Some(original),
        })
    }
}

impl<H: ?Sized> Drop for RoventCollectionLaunchFailureV2<H> {
    fn drop(&mut self) {
        if self.physical_unknown {
            if let Some(hooks) = self.hooks.take() {
                std::mem::forget(hooks);
            }
            if let Some(payload) = self.panic_payload.take() {
                std::mem::forget(payload);
            }
            if let Some(custody) = self.board_product_custody.take() {
                std::mem::forget(custody);
            }
        }
    }
}

/// Admit a P/C V2 collection using the actual V2 lock and original source
/// owners. Teacher retirement and publication are separate evidence.
pub fn run_pc_collection_v2_with_resource_admission<'a>(
    arguments: Vec<String>,
    teacher_selections: Vec<RoventTeacherSourceSelection>,
    observer: Box<dyn VirtualReviewCollectionObserver>,
    hooks: Arc<dyn RoventPcCollectionHooksV2>,
    admit_resource: impl FnOnce(&Arc<RoventInputLockV2>) -> Result<(), RoventLaunchError> + 'a,
) -> Result<(), RoventPcCollectionLaunchFailureV2> {
    run_collection_v2(arguments, teacher_selections, observer, Arc::clone(&hooks), CollectionHooksV2::Pc(hooks), admit_resource)
}

pub fn run_own_nnuec_collection_v2_with_resource_admission<'a>(
    arguments: Vec<String>,
    teacher_selections: Vec<RoventTeacherSourceSelection>,
    observer: Box<dyn VirtualReviewCollectionObserver>,
    hooks: Arc<dyn RoventNnuecCollectionHooksV2>,
    admit_resource: impl FnOnce(&Arc<RoventInputLockV2>) -> Result<(), RoventLaunchError> + 'a,
) -> Result<(), RoventOwnNnuecCollectionLaunchFailureV2> {
    run_collection_v2(arguments, teacher_selections, observer, Arc::clone(&hooks), CollectionHooksV2::OwnNnuec(hooks), admit_resource)
}

pub fn run_own_board_collection_v2_with_resource_admission<'a>(
    arguments: Vec<String>,
    teacher_selections: Vec<RoventTeacherSourceSelection>,
    observer: Box<dyn VirtualReviewCollectionObserver>,
    hooks: Arc<RoventOwnBoardCollectionHooksV2>,
    admit_resource: impl FnOnce(&Arc<RoventInputLockV2>) -> Result<(), RoventLaunchError> + 'a,
) -> Result<(), RoventOwnBoardCollectionLaunchFailureV2> {
    run_collection_v2(arguments, teacher_selections, observer, Arc::clone(&hooks), CollectionHooksV2::OwnBoard(hooks), admit_resource)
}

fn run_collection_v2<'a, H: ?Sized>(
    arguments: Vec<String>,
    teacher_selections: Vec<RoventTeacherSourceSelection>,
    observer: Box<dyn VirtualReviewCollectionObserver>,
    original_hooks: Arc<H>,
    hooks: CollectionHooksV2,
    admit_resource: impl FnOnce(&Arc<RoventInputLockV2>) -> Result<(), RoventLaunchError> + 'a,
) -> Result<(), RoventCollectionLaunchFailureV2<H>> {
    let startup_attempt = Arc::new(AtomicBool::new(false));
    let native_settled = Arc::new(AtomicBool::new(false));
    let board_product_slot = Arc::new(Mutex::new(None));
    let mut observer_custody = V2ObserverCustody::new(observer);
    let proxy = observer_custody.proxy().expect("new observer custody");
    let inner_startup_attempt = Arc::clone(&startup_attempt);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| super::run_pc_collection_v2_inner(arguments, CollectionLaunchV2 {
        teacher_selections,
        observer: proxy,
        hooks,
        admit_resource: Box::new(admit_resource),
        startup_attempt: inner_startup_attempt,
        native_settled: Arc::clone(&native_settled),
        board_product_custody: Arc::clone(&board_product_slot),
    })));
    let board_product_custody = board_product_slot.lock().unwrap_or_else(|p| p.into_inner()).take();
    match result {
        Ok(Ok(())) if board_product_custody.is_some() => Err(RoventCollectionLaunchFailureV2 {
            error: super::error("PhysicalCompletionUnknown", "a Board product rejection remained after apparent success"),
            hooks: Some(original_hooks),
            panic_payload: None,
            observer_before_start: None,
            retained_observer: Some(observer_custody),
            observer_reclaim_error: None,
            board_product_custody,
            physical_unknown: true,
        }),
        Ok(Ok(())) => match observer_custody.try_reclaim_after_verified_boundary() {
            Ok(original) => { drop(original); Ok(()) }
            Err(reclaim_error) => Err(RoventCollectionLaunchFailureV2 {
                error: super::error("PhysicalCompletionUnknown", "V2 collection observer custody remained active after session closure"),
                hooks: Some(original_hooks),
                panic_payload: None,
                observer_before_start: None,
                retained_observer: Some(observer_custody),
                observer_reclaim_error: Some(reclaim_error),
                board_product_custody,
                physical_unknown: true,
            }),
        },
        Ok(Err(error)) => {
            let safe_native = !startup_attempt.load(Ordering::Acquire)
                || native_settled.load(Ordering::Acquire);
            let (original, reclaim_error) = if safe_native && error.stage != "PhysicalCompletionUnknown" {
                match observer_custody.try_reclaim_after_verified_boundary() {
                    Ok(original) => (Some(original), None),
                    Err(reclaim_error) => (None, Some(reclaim_error)),
                }
            } else { (None, None) };
            let physical_unknown = !safe_native || error.stage == "PhysicalCompletionUnknown";
            let retained_observer = if original.is_none() { Some(observer_custody) } else { None };
            Err(RoventCollectionLaunchFailureV2 {
                error,
                hooks: Some(original_hooks),
                panic_payload: None,
                observer_before_start: original,
                retained_observer,
                observer_reclaim_error: reclaim_error,
                board_product_custody,
                physical_unknown,
            })
        }
        Err(payload) => Err(RoventCollectionLaunchFailureV2 {
            error: super::error("PhysicalCompletionUnknown", "V2 collection launch panicked"),
            hooks: Some(original_hooks),
            panic_payload: Some(payload),
            observer_before_start: None,
            retained_observer: Some(observer_custody),
            observer_reclaim_error: None,
            board_product_custody,
            physical_unknown: true,
        }),
    }
}

/// 부모 launch가 실제로 준비한 단일 native owner의 구체 질문 타입을 보존한다.
/// 다른 variant나 질문 타입으로 바꾸는 변환 경로는 제공하지 않는다.
pub enum RoventCollectionPreparedNative {
    Pc(Box<PreparedRoventNative<PcPortQuestion>>),
    PcNnuec {
        prepared: Box<PreparedRoventNative<PcPortQuestion>>,
        loaded: Box<LoadedRoventNnuecSource>,
    },
    OwnBoardPcr(Box<PreparedRoventNative<AdaptivePortQuestion>>),
    OwnNnuecPcr {
        prepared: Box<PreparedRoventNative<AdaptivePortQuestion>>,
        loaded: Box<LoadedRoventNnuecSource>,
    },
    ExternalPcr {
        prepared: Box<PreparedRoventNative<AdaptivePortQuestion>>,
        external: Box<RoventPinnedExternalRSource>,
        retirement_evidence: Box<LaunchRetirementEvidence>,
    },
}

/// 원 Collection preload와 native owner를 함께 이동하는 일회성 조립 입력.
/// 공개 필드는 별도의 source 검증이나 실제 실행 증거를 대신하지 않는다.
pub struct RoventCollectionPreparedLaunch {
    pub native: RoventCollectionPreparedNative,
    pub preload: Arc<RoventCollectionPreloadSources>,
    pub control: RoventStartupControl,
    pub clock: ProcessClock,
    pub owners: Arc<OwnerRegistry>,
    pub identity: EngineIdentity,
    pub root_policy: RoventRootAdmissionPolicy,
}

/// 부모 launch만 소비하는 요청. observer와 factory는 한 번만 이동하며,
/// factory의 실패는 기존 rejection 안에 동일한 실제 owner를 보존해야 한다.
pub(super) struct CollectionLaunch<'a> {
    pub(super) teacher_selections: Vec<RoventTeacherSourceSelection>,
    pub(super) observer: Box<dyn VirtualReviewCollectionObserver>,
    pub(super) assemble: Box<
        dyn FnOnce(
                RoventCollectionPreparedLaunch,
            ) -> Result<RoventNativeProductDriver, RoventProductStartupRejection>
            + 'a,
    >,
}

/// 부모의 공통 launch 절차에 명시적인 Collection 요청을 전달한다.
/// 조립 callback은 넘겨받은 원 owner를 소비하며, 실패하면 새 owner나 축약
/// 오류로 바꾸지 않고 동일한 owner를 RoventProductStartupRejection에 남긴다.
///
/// 원 source FD는 부모의 공통 launch가 전체 UCI 세션의 물리 종료까지
/// 보유한다. 이 진입점 자체는 teacher 실행이나 target 권한을 발급하지 않는다.
pub fn run_collection<'a>(
    arguments: Vec<String>,
    teacher_selections: Vec<RoventTeacherSourceSelection>,
    observer: Box<dyn VirtualReviewCollectionObserver>,
    assemble: impl FnOnce(
        RoventCollectionPreparedLaunch,
    ) -> Result<RoventNativeProductDriver, RoventProductStartupRejection>
    + 'a,
) -> Result<(), RoventLaunchError> {
    super::run_inner(
        arguments,
        Some(CollectionLaunch {
            teacher_selections,
            observer,
            assemble: Box::new(assemble),
        }),
        None,
        None,
    )
}

/// 원 lock/process boundary 확인 뒤 native 준비 전에 자원 owner를 인수한다.
/// OS scope는 callback의 상위 owner가 보유하며 Result<()>는 권한을 만들지 않는다.
pub fn run_collection_with_resource_admission<'a>(
    arguments: Vec<String>,
    teacher_selections: Vec<RoventTeacherSourceSelection>,
    observer: Box<dyn VirtualReviewCollectionObserver>,
    admit_resource: impl FnOnce(&Arc<RoventInputLock>) -> Result<(), RoventLaunchError> + 'a,
    assemble: impl FnOnce(
        RoventCollectionPreparedLaunch,
    ) -> Result<RoventNativeProductDriver, RoventProductStartupRejection>
    + 'a,
) -> Result<(), RoventLaunchError> {
    super::run_inner(
        arguments,
        Some(CollectionLaunch {
            teacher_selections,
            observer,
            assemble: Box::new(assemble),
        }),
        Some(Box::new(admit_resource)),
        None,
    )
}

/// 같은 원 제품과 FD 수명 안에서 유한한 실제 UCI runner를 사용한다.
/// runner는 공개 EngineProcess/event/serve 경로로 각 원 go를 발급하고
/// actual finish_session을 완료해야 한다. 반환값으로 물리 완료를 만들지 않는다.
pub fn run_collection_with_driver<'a>(
    arguments: Vec<String>,
    teacher_selections: Vec<RoventTeacherSourceSelection>,
    observer: Box<dyn VirtualReviewCollectionObserver>,
    admit_resource: impl FnOnce(&Arc<RoventInputLock>) -> Result<(), RoventLaunchError> + 'a,
    assemble: impl FnOnce(
        RoventCollectionPreparedLaunch,
    ) -> Result<RoventNativeProductDriver, RoventProductStartupRejection>
    + 'a,
    run: impl FnOnce(
        RoventNativeProductDriver,
        ProcessClock,
        Arc<OwnerRegistry>,
        EngineSettings,
    ) -> Result<(), RoventLaunchError>
    + 'a,
) -> Result<(), RoventLaunchError> {
    super::run_inner(
        arguments,
        Some(CollectionLaunch {
            teacher_selections,
            observer,
            assemble: Box::new(assemble),
        }),
        Some(Box::new(admit_resource)),
        Some(Box::new(run)),
    )
}
