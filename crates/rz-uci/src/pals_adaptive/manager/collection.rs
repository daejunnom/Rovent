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

/// Manager-observed original reservation, before native execution. This is
/// not a physical execution or completion proof.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub struct RoventCollectionDispatchWitnessV2<'a> {
    admission: &'a RoventCollectionRootAdmissionV2,
    ticket: &'a RoleTicketEnvelope,
}

/// The original ticket failure is retained intact so a temporary lock
/// contention cannot be mistaken for a revoked or malformed authority.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub enum RoventDispatchWitnessFailureV2 {
    NativeIdentity(AdaptiveCollectionError),
    Ticket(RoventTicketValidationErrorV2),
}

/// Exact live root and native owner selected by the manager for a whole-root
/// optional collection bypass. The marker is not a native completion proof.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
#[derive(Clone)]
pub struct RoventNoCaptureRootCapabilityV2 {
    admission: RoventCollectionRootAdmissionV2,
    verifier: rz_search::pals::coordinator::NativeAcceptanceVerifier,
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub(super) enum RoventNoCaptureIssueErrorV2 {
    Busy(rz_contracts::ContractError),
    Rejected(AdaptiveCollectionError),
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl RoventNoCaptureRootCapabilityV2 {
    pub(super) fn new(
        admission: RoventCollectionRootAdmissionV2,
        verifier: rz_search::pals::coordinator::NativeAcceptanceVerifier,
    ) -> Result<Self, RoventNoCaptureIssueErrorV2> {
        admission.validate_native_identity(&verifier)
            .map_err(RoventNoCaptureIssueErrorV2::Rejected)?;
        if !admission.root_admission().original_context().try_accepts()
            .map_err(|error| {
                if error.code == rz_contracts::ErrorCode::ResourceExhausted {
                    RoventNoCaptureIssueErrorV2::Busy(error)
                } else {
                    RoventNoCaptureIssueErrorV2::Rejected(AdaptiveCollectionError::new(
                        AdaptiveCollectionStage::Dispatch, "NoCaptureRootUnavailable", error.to_string()))
                }
            })?
        {
            return Err(RoventNoCaptureIssueErrorV2::Rejected(AdaptiveCollectionError::new(
                AdaptiveCollectionStage::Dispatch, "NoCaptureRootRevoked", "original root is not live")));
        }
        Ok(Self { admission, verifier })
    }

    pub fn same_original_admission(&self, other: &RoventCollectionRootAdmissionV2) -> bool {
        self.admission.same_original_admission(other)
    }

    /// Root identity is checked independently of the worker identity so a
    /// duplicate root cannot be hidden by presenting a different worker.
    pub fn same_original_root(&self, other: &Self) -> bool {
        Arc::ptr_eq(self.admission.root_admission(), other.admission.root_admission())
    }

    pub fn same_root_admission(&self, other: &RoventCollectionRootAdmissionV2) -> bool {
        Arc::ptr_eq(self.admission.root_admission(), other.root_admission())
    }

    /// Authority uniquely identifies the manager's active process/game/root
    /// scope even when a start materializes a second admission object.
    pub fn authority(&self) -> SearchAuthority {
        self.admission.root_admission().authority()
    }

    pub fn same_original_capability(&self, other: &Self) -> bool {
        self.admission.same_original_admission(&other.admission)
            && self.verifier.same_owner(&other.verifier)
    }

    pub fn validate_native_discard_owner<V, Q, P>(
        &self,
        context: &crate::pals_native::ticketed::TicketOwnerDispatchContext<'_, V, Q, P>,
    ) -> Result<(), AdaptiveCollectionError> {
        if !context.same_owner(&self.verifier) {
            return Err(AdaptiveCollectionError::new(
                AdaptiveCollectionStage::Dispatch, "NoCaptureWorkerMismatch", "different native owner"));
        }
        self.admission.validate_native_discard_owner(context)
    }

    /// Called on the actual root-run receipt; this does not attest teacher
    /// retirement or any other root's physical completion.
    pub fn matches_shutdown_receipt(&self, receipt: &super::AdaptiveShutdownReceipt) -> bool {
        receipt.authority == self.admission.root_admission().authority()
            && receipt.permits_root_reuse()
    }

    /// The P/C-only receipt carries its own native fence and root budget.
    /// This checks the original root and physical closure, not publication.
    pub fn matches_role_only_shutdown_receipt(
        &self,
        receipt: &super::role_only::RoleOnlyShutdownReceipt,
    ) -> bool {
        receipt.authority == self.admission.root_admission().authority()
            && receipt.root_budget == self.admission.root_budget()
            && receipt.permits_semantic_closure()
    }
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl<'a> RoventCollectionDispatchWitnessV2<'a> {
    pub(super) fn try_new(
        admission: &'a RoventCollectionRootAdmissionV2,
        verifier: &rz_search::pals::coordinator::NativeAcceptanceVerifier,
        ticket: &'a RoleTicketEnvelope,
    ) -> Result<Self, RoventDispatchWitnessFailureV2> {
        admission
            .validate_native_identity(verifier)
            .map_err(RoventDispatchWitnessFailureV2::NativeIdentity)?;
        admission
            .try_validate_ticket(ticket)
            .map_err(RoventDispatchWitnessFailureV2::Ticket)?;
        Ok(Self { admission, ticket })
    }
    pub(super) fn new(
        admission: &'a RoventCollectionRootAdmissionV2,
        verifier: &rz_search::pals::coordinator::NativeAcceptanceVerifier,
        ticket: &'a RoleTicketEnvelope,
    ) -> Result<Self, AdaptiveCollectionError> {
        admission.validate_native_owner(verifier)?;
        admission.validate_ticket(ticket)?;
        Ok(Self { admission, ticket })
    }
    pub fn admission(&self) -> &RoventCollectionRootAdmissionV2 {
        self.admission
    }
    pub fn ticket(&self) -> &RoleTicketEnvelope {
        self.ticket
    }
}

/// A single callback's original native acceptance, tied to the same live /2
/// root and worker. It is neither cloneable nor serializable and cannot be
/// constructed by an observer or reconstructed from a stored receipt.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub struct RoventCollectionAcceptedWitnessV2<'a> {
    admission: &'a RoventCollectionRootAdmissionV2,
    accepted: &'a AcceptedRoleStep,
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl<'a> RoventCollectionAcceptedWitnessV2<'a> {
    pub(super) fn new(
        admission: &'a RoventCollectionRootAdmissionV2,
        verifier: &rz_search::pals::coordinator::NativeAcceptanceVerifier,
        accepted: &'a AcceptedRoleStep,
        output: &rz_contracts::pals::RoleOutput,
    ) -> Result<Self, AdaptiveCollectionError> {
        admission.validate_native_owner(verifier)?;
        admission.validate_output(accepted.ticket(), output)?;
        if !accepted.validates_output(output) {
            return Err(AdaptiveCollectionError::new(
                AdaptiveCollectionStage::Accepted,
                "AcceptedOutputMismatch",
                "original accepted output differs",
            ));
        }
        Ok(Self {
            admission,
            accepted,
        })
    }

    pub fn admission(&self) -> &RoventCollectionRootAdmissionV2 {
        self.admission
    }

    pub fn accepted(&self) -> &AcceptedRoleStep {
        self.accepted
    }
}

/// A separate single callback witness made only after the manager has
/// recorded actual search consumption. It does not imply teacher retirement.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub struct RoventCollectionConsumedWitnessV2<'a> {
    acceptance: RoventCollectionAcceptedWitnessV2<'a>,
}

/// An event polled from the original native worker. This establishes identity
/// only; completion and search acceptance retain their separate contracts.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub struct RoventCollectionNativeEventWitnessV2<'a, O> {
    admission: &'a RoventCollectionRootAdmissionV2,
    event: &'a TicketNativeEvent<O>,
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl<'a, O> RoventCollectionNativeEventWitnessV2<'a, O> {
    pub(super) fn new(
        admission: &'a RoventCollectionRootAdmissionV2,
        verifier: &rz_search::pals::coordinator::NativeAcceptanceVerifier,
        event: &'a TicketNativeEvent<O>,
    ) -> Result<Self, AdaptiveCollectionError> {
        admission.validate_native_identity(verifier)?;
        Ok(Self { admission, event })
    }
    pub fn admission(&self) -> &RoventCollectionRootAdmissionV2 {
        self.admission
    }
    pub fn event(&self) -> &TicketNativeEvent<O> {
        self.event
    }
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl<'a> RoventCollectionConsumedWitnessV2<'a> {
    pub(super) fn new(acceptance: RoventCollectionAcceptedWitnessV2<'a>) -> Self {
        Self { acceptance }
    }

    pub fn admission(&self) -> &RoventCollectionRootAdmissionV2 {
        self.acceptance.admission()
    }

    pub fn accepted(&self) -> &AcceptedRoleStep {
        self.acceptance.accepted()
    }
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
/// Capture or disable optional collection for the original root. Neither
/// disposition changes native search or physical completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveCollectionDispatchV2 {
    Captured,
    /// The original capture owner is briefly busy. The manager retains the
    /// same ticket and retries inside its original deadline without executing.
    Busy,
    /// This root's optional collection is disabled from the original ticket
    /// onward. The observer must first retain that ticket, reason and root
    /// boundary and register the same disabled scope with native capture.
    DisabledBackpressure,
}

/// A busy original owner is retried only while the same root and ticket are
/// live; it is not an installed capture bypass or a physical completion.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveCollectionBoundaryV2 {
    Ready,
    Busy,
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
        ticket: RoventCollectionDispatchWitnessV2<'_>,
        input: &FrameInput<ReviewView, Q>,
    ) -> Result<AdaptiveCollectionDispatchV2, AdaptiveCollectionError>;
    /// Preserve the first collection-only failure and install the original
    /// root's native capture bypass before dispatching the supplied ticket.
    /// Existing per-ticket captures retain priority and must still be drained.
    /// A missing or unsafe boundary must return an error, not claim success.
    fn disable_after_failure(
        &mut self,
        _ticket: RoventCollectionDispatchWitnessV2<'_>,
        _original: &AdaptiveCollectionError,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        Err(AdaptiveCollectionError::new(
            AdaptiveCollectionStage::Dispatch,
            "DisableAfterFailureUnsupported",
            "original collection owner cannot establish a safe disabled boundary",
        ))
    }
    /// A previously prepared ticket may be dispatched after a later ticket
    /// disabled this root. Preserve its exact original native identity before
    /// allowing that earlier-sequence ticket to execute without a capture.
    fn allow_disabled_prior_ticket(
        &mut self,
        _ticket: RoventCollectionDispatchWitnessV2<'_>,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        Err(AdaptiveCollectionError::new(
            AdaptiveCollectionStage::Dispatch,
            "DisabledPriorTicketUnsupported",
            "original collection owner cannot register the earlier ticket",
        ))
    }
    fn native_event(
        &mut self,
        event: RoventCollectionNativeEventWitnessV2<'_, O>,
    ) -> Result<(), AdaptiveCollectionError>;
    fn accepted(
        &mut self,
        accepted: RoventCollectionAcceptedWitnessV2<'_>,
        input: &FrameInput<ReviewView, Q>,
        output: &O,
    ) -> Result<(), AdaptiveCollectionError>;
    /// Called after the manager records actual search consumption. This is a
    /// separate observation from native acceptance and carries its readback path.
    fn consumed(
        &mut self,
        accepted: RoventCollectionConsumedWitnessV2<'_>,
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
        _accepted: RoventCollectionConsumedWitnessV2<'_>,
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

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub(super) fn notify_adaptive_consumed_v2<Q, O>(
    observer: &mut Option<AdaptiveCollectionObserverBoxV2<Q, O>>,
    admission: Option<&RoventCollectionRootAdmissionV2>,
    verifier: Option<&rz_search::pals::coordinator::NativeAcceptanceVerifier>,
    accepted: &AcceptedRoleStep,
    output: &rz_contracts::pals::RoleOutput,
    after_error_readback: bool,
) -> Result<(), AdaptiveCollectionError> {
    let Some(observer) = observer.as_mut() else {
        return Ok(());
    };
    let missing = || {
        AdaptiveCollectionError::new(
            AdaptiveCollectionStage::Accepted,
            "OriginalOwnerUnavailable",
            "V2 consumption has no original admission or native verifier",
        )
    };
    let witness = RoventCollectionAcceptedWitnessV2::new(
        admission.ok_or_else(missing)?,
        verifier.ok_or_else(missing)?,
        accepted,
        output,
    )?;
    observer.adaptive_consumed(
        RoventCollectionConsumedWitnessV2::new(witness),
        after_error_readback,
    )
}
