//! 선택적 실제 수집 접점이다. manager는 원 ticket·event·수락 proof를 빌려
//! 전달할 뿐이며, 구현체는 인코딩·forward·archive I/O를 동기로 실행하지 않는다.
use super::*;
use rz_search::pals::coordinator::AcceptedRoleStep;
mod origin;
pub use origin::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveCollectionStage {
    Dispatch,
    NativeEvent,
    Accepted,
    Abandoned,
    Settled,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptiveCollectionError {
    pub stage: AdaptiveCollectionStage,
    pub code: &'static str,
    pub detail: String,
}
impl AdaptiveCollectionError {
    pub fn new(
        stage: AdaptiveCollectionStage,
        code: &'static str,
        detail: impl Into<String>,
    ) -> Self {
        let mut detail = detail.into();
        let mut end = detail.len().min(2048);
        while !detail.is_char_boundary(end) {
            end -= 1;
        }
        detail.truncate(end);
        Self {
            stage,
            code,
            detail,
        }
    }
}
impl std::fmt::Display for AdaptiveCollectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "collection {:?}/{}: {}",
            self.stage, self.code, self.detail
        )
    }
}
impl std::error::Error for AdaptiveCollectionError {}
impl From<AdaptiveCollectionError> for AdaptiveManagerError {
    fn from(error: AdaptiveCollectionError) -> Self {
        Self::Collection(error)
    }
}

/// 현재 실제 Rovent manager가 원 registry의 최초 P publication에서 발급한다.
/// public JSON 또는 나중의 독립 proof로 생성할 수 없으며 AcceptedRoleStep으로
/// 변환하지 않는다. 원 ranking·view·Rules owner는 그대로 보존한다.
pub struct RoventRootRankingAcceptance {
    ranking: ProposerRankingPublished,
    admission: RoventCollectionRootAdmission,
}
impl RoventRootRankingAcceptance {
    pub(super) fn from_current_ranking(
        ranking: &ProposerRankingPublished,
        admission: &RoventCollectionRootAdmission,
    ) -> Result<Self, AdaptiveCollectionError> {
        admission.validate_ranking(ranking)?;
        Ok(Self {
            ranking: ranking.clone(),
            admission: admission.clone(),
        })
    }
    pub fn ranking(&self) -> &ProposerRankingPublished {
        &self.ranking
    }
    pub fn authority(&self) -> SearchAuthority {
        self.admission.authority()
    }
    pub fn admission(&self) -> &RoventCollectionRootAdmission {
        &self.admission
    }
}

/// hook는 원 manager root에 명시적으로 설치한다. corpus의 지속 상태는
/// caller가 보존하고, owner를 재사용한 다음 root에도 같은 상태를 재설치한다.
/// 오류는 원 runtime event·proof를 먼저 보존한 뒤 별도 typed failure로 기록한다.
pub trait AdaptiveCollectionObserver<Q, O>: Send {
    fn admitted(
        &mut self,
        admission: &RoventCollectionRootAdmission,
    ) -> Result<(), AdaptiveCollectionError>;
    fn dispatch(
        &mut self,
        ticket: &RoleTicketEnvelope,
        input: &FrameInput<ReviewView, Q>,
    ) -> Result<(), AdaptiveCollectionError>;
    fn native_event(&mut self, event: &TicketNativeEvent<O>)
    -> Result<(), AdaptiveCollectionError>;
    fn accepted(
        &mut self,
        accepted: &AcceptedRoleStep,
        input: &FrameInput<ReviewView, Q>,
        output: &O,
    ) -> Result<(), AdaptiveCollectionError>;
    fn root_ranking(
        &mut self,
        accepted: &RoventRootRankingAcceptance,
        input: &FrameInput<ReviewView, Q>,
        output: &O,
    ) -> Result<(), AdaptiveCollectionError>;
    /// 논리 폐기만 전달한다. 원 native owner의 물리 drain을 대신하지 않는다.
    fn abandoned(&mut self, ticket: &RoleTicketEnvelope) -> Result<(), AdaptiveCollectionError>;
    /// 실제 root/process fence 후 남은 수집 입력·예약만 정리한다.
    fn settled(&mut self, tickets: &[RoleTicketEnvelope]) -> Result<(), AdaptiveCollectionError>;
}
pub type AdaptiveCollectionObserverBox<Q, O> = Box<dyn AdaptiveCollectionObserver<Q, O>>;

/// Registry-issued root ranking paired with its original /2 root capability.
/// This cannot be constructed from serialized metadata or used as a teacher
/// result or physical-completion proof.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub struct RoventRootRankingAcceptanceV2 {
    ranking: ProposerRankingPublished,
    admission: RoventCollectionRootAdmissionV2,
}
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl RoventRootRankingAcceptanceV2 {
    pub(super) fn from_current_ranking(
        ranking: &ProposerRankingPublished,
        admission: &RoventCollectionRootAdmissionV2,
    ) -> Result<Self, AdaptiveCollectionError> {
        admission.validate_ranking(ranking)?;
        Ok(Self {
            ranking: ranking.clone(),
            admission: admission.clone(),
        })
    }
    pub fn ranking(&self) -> &ProposerRankingPublished {
        &self.ranking
    }
    pub fn admission(&self) -> &RoventCollectionRootAdmissionV2 {
        &self.admission
    }
}

/// The origin of a recorded search-consumption observation.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoleOnlySearchConsumptionV2 {
    DriverReturnedSuccess,
    DriverReadbackAfterError,
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
/// Observer for an original V2 root. Accepted is native-owner acceptance,
/// not a claim that the search has consumed the result. A teacher process and
/// other physical resources must be owned by the separate session lifecycle,
/// not by this observation-only object.
pub trait AdaptiveCollectionObserverV2<Q, O>: Send {
    fn admitted(
        &mut self,
        admission: &RoventCollectionRootAdmissionV2,
    ) -> Result<(), AdaptiveCollectionError>;
    fn dispatch(
        &mut self,
        ticket: &RoleTicketEnvelope,
        input: &FrameInput<ReviewView, Q>,
    ) -> Result<(), AdaptiveCollectionError>;
    fn native_event(&mut self, event: &TicketNativeEvent<O>)
    -> Result<(), AdaptiveCollectionError>;
    fn accepted(
        &mut self,
        accepted: &AcceptedRoleStep,
        input: &FrameInput<ReviewView, Q>,
        output: &O,
    ) -> Result<(), AdaptiveCollectionError>;
    /// Called after the manager records actual search consumption. This is a
    /// separate observation from native acceptance and carries its readback path.
    fn consumed(
        &mut self,
        accepted: &AcceptedRoleStep,
        observation: RoleOnlySearchConsumptionV2,
    ) -> Result<(), AdaptiveCollectionError>;
    /// Adaptive root ranking has its own registry proof and has already been
    /// consumed by the root when this callback is invoked.
    fn adaptive_root_ranking(
        &mut self,
        _accepted: &RoventRootRankingAcceptanceV2,
        _input: &FrameInput<ReviewView, Q>,
        _output: &O,
    ) -> Result<(), AdaptiveCollectionError> {
        Err(AdaptiveCollectionError::new(
            AdaptiveCollectionStage::Accepted,
            "AdaptiveRankingUnsupported",
            "observer does not support adaptive V2 ranking",
        ))
    }
    /// Reports an actual adaptive search-consumption transition after its
    /// separate native acceptance. It never creates acceptance authority.
    fn adaptive_consumed(
        &mut self,
        _accepted: &AcceptedRoleStep,
        _after_error_readback: bool,
    ) -> Result<(), AdaptiveCollectionError> {
        Err(AdaptiveCollectionError::new(
            AdaptiveCollectionStage::Accepted,
            "AdaptiveConsumptionUnsupported",
            "observer does not support adaptive V2 consumption",
        ))
    }
    fn abandoned(&mut self, ticket: &RoleTicketEnvelope) -> Result<(), AdaptiveCollectionError>;
    fn settled(&mut self, tickets: &[RoleTicketEnvelope]) -> Result<(), AdaptiveCollectionError>;
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub type AdaptiveCollectionObserverBoxV2<Q, O> = Box<dyn AdaptiveCollectionObserverV2<Q, O>>;
