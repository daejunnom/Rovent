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
use crate::rovent_bootstrap::pc_session::RoventPcCollectionHooksV2;
use crate::rovent_bootstrap::{
    PreparedRoventNative, RoventCollectionPreloadSources, RoventPinnedExternalRSource,
    RoventStartupControl, RoventTeacherSourceSelection,
};
use crate::rovent_root_admission::RoventRootAdmissionPolicy;
use rz_experiments::{RoventInputLock, RoventInputLockV2};
use rz_search::pals::engine::{adaptive_port::AdaptivePortQuestion, pc_port::PcPortQuestion};
use std::sync::Arc;

/// Explicit V2 P/C collection inputs. These owners are selected before the
/// original lock is opened; they never project a V2 input lock into V1.
pub(super) struct CollectionLaunchV2<'a> {
    pub(super) teacher_selections: Vec<RoventTeacherSourceSelection>,
    pub(super) observer: Box<dyn VirtualReviewCollectionObserver>,
    pub(super) hooks: Arc<dyn RoventPcCollectionHooksV2>,
    pub(super) admit_resource: Box<dyn FnOnce(&Arc<RoventInputLockV2>) -> Result<(), RoventLaunchError> + 'a>,
}

/// Admit a P/C V2 collection using the actual V2 lock and original source
/// owners. Teacher retirement and publication are separate evidence.
pub fn run_pc_collection_v2_with_resource_admission<'a>(
    arguments: Vec<String>,
    teacher_selections: Vec<RoventTeacherSourceSelection>,
    observer: Box<dyn VirtualReviewCollectionObserver>,
    hooks: Arc<dyn RoventPcCollectionHooksV2>,
    admit_resource: impl FnOnce(&Arc<RoventInputLockV2>) -> Result<(), RoventLaunchError> + 'a,
) -> Result<(), RoventLaunchError> {
    super::run_pc_collection_v2_inner(arguments, CollectionLaunchV2 {
        teacher_selections,
        observer,
        hooks,
        admit_resource: Box::new(admit_resource),
    })
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
