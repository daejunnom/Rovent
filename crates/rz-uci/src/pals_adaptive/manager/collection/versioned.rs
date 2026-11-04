//! One manager flow for original /2 and /3 collection owners.
//!
//! These wrappers route existing observations; they never convert an owner or
//! manufacture a physical completion, ticket, or search-consumption proof.
use super::*;

type NativeVerifier = rz_search::pals::coordinator::NativeAcceptanceVerifier;

fn mismatch(stage: AdaptiveCollectionStage) -> AdaptiveCollectionError {
    AdaptiveCollectionError::new(
        stage,
        "CollectionVersionMismatch",
        "original admission, observer and witness versions differ",
    )
}

#[derive(Clone)]
pub(crate) enum VersionedCollectionAdmission {
    V2(RoventCollectionRootAdmissionV2),
    #[cfg(target_os = "linux")]
    V3(RoventCollectionRootAdmissionV3),
}

impl VersionedCollectionAdmission {
    pub(crate) fn authority(&self) -> SearchAuthority {
        match self {
            Self::V2(a) => a.root_admission().authority(),
            #[cfg(target_os = "linux")]
            Self::V3(a) => a.root_admission().authority(),
        }
    }
    pub(crate) fn hard_deadline(&self) -> rz_contracts::Deadline {
        match self {
            Self::V2(a) => a.root_admission().hard_deadline(),
            #[cfg(target_os = "linux")]
            Self::V3(a) => a.root_admission().hard_deadline(),
        }
    }
    pub(crate) fn root_budget(&self) -> Digest {
        match self {
            Self::V2(a) => a.root_budget(),
            #[cfg(target_os = "linux")]
            Self::V3(a) => a.root_budget(),
        }
    }
    pub(crate) fn mode(&self) -> ExecutionMode {
        match self {
            Self::V2(a) => a.native_origin().sources().lock().manifest.engine.mode,
            #[cfg(target_os = "linux")]
            Self::V3(a) => a.native_origin().sources().lock().manifest.engine.mode,
        }
    }
    pub(crate) fn validate_native_owner(
        &self,
        v: &NativeVerifier,
    ) -> Result<(), AdaptiveCollectionError> {
        match self {
            Self::V2(a) => a.validate_native_owner(v),
            #[cfg(target_os = "linux")]
            Self::V3(a) => a.validate_native_owner(v),
        }
    }
    pub(crate) fn validate_native_identity(
        &self,
        v: &NativeVerifier,
    ) -> Result<(), AdaptiveCollectionError> {
        match self {
            Self::V2(a) => a.validate_native_identity(v),
            #[cfg(target_os = "linux")]
            Self::V3(a) => a.validate_native_identity(v),
        }
    }
    pub(crate) fn validate_output(
        &self,
        ticket: &RoleTicketEnvelope,
        output: &rz_contracts::pals::RoleOutput,
    ) -> Result<(), AdaptiveCollectionError> {
        match self {
            Self::V2(a) => a.validate_output(ticket, output),
            #[cfg(target_os = "linux")]
            Self::V3(a) => a.validate_output(ticket, output),
        }
    }
    // On non-Linux targets only V2 exists, so the mismatch arm is unreachable.
    #[allow(unreachable_patterns)]
    pub(crate) fn same_original_admission(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::V2(a), Self::V2(b)) => a.same_original_admission(b),
            #[cfg(target_os = "linux")]
            (Self::V3(a), Self::V3(b)) => a.same_original_admission(b),
            _ => false,
        }
    }
    pub(crate) fn v2(&self) -> Option<&RoventCollectionRootAdmissionV2> {
        match self {
            Self::V2(a) => Some(a),
            #[cfg(target_os = "linux")]
            Self::V3(_) => None,
        }
    }
    #[cfg(target_os = "linux")]
    pub(crate) fn v3(&self) -> Option<&RoventCollectionRootAdmissionV3> {
        if let Self::V3(a) = self {
            Some(a)
        } else {
            None
        }
    }
}

#[cfg(target_os = "linux")]
pub struct RoventCollectionDispatchWitnessV3<'a> {
    admission: &'a RoventCollectionRootAdmissionV3,
    ticket: &'a RoleTicketEnvelope,
}
#[cfg(target_os = "linux")]
impl<'a> RoventCollectionDispatchWitnessV3<'a> {
    fn try_new(
        admission: &'a RoventCollectionRootAdmissionV3,
        verifier: &NativeVerifier,
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
    pub fn admission(&self) -> &RoventCollectionRootAdmissionV3 {
        self.admission
    }
    pub fn ticket(&self) -> &RoleTicketEnvelope {
        self.ticket
    }
}

#[cfg(target_os = "linux")]
pub struct RoventCollectionNativeEventWitnessV3<'a, O> {
    admission: &'a RoventCollectionRootAdmissionV3,
    event: &'a TicketNativeEvent<O>,
}
#[cfg(target_os = "linux")]
impl<'a, O> RoventCollectionNativeEventWitnessV3<'a, O> {
    fn new(
        admission: &'a RoventCollectionRootAdmissionV3,
        verifier: &NativeVerifier,
        event: &'a TicketNativeEvent<O>,
    ) -> Result<Self, AdaptiveCollectionError> {
        admission.validate_native_identity(verifier)?;
        Ok(Self { admission, event })
    }
    pub fn admission(&self) -> &RoventCollectionRootAdmissionV3 {
        self.admission
    }
    pub fn event(&self) -> &TicketNativeEvent<O> {
        self.event
    }
}

#[cfg(target_os = "linux")]
pub struct RoventCollectionAcceptedWitnessV3<'a> {
    admission: &'a RoventCollectionRootAdmissionV3,
    accepted: &'a AcceptedRoleStep,
}
#[cfg(target_os = "linux")]
impl<'a> RoventCollectionAcceptedWitnessV3<'a> {
    fn new(
        admission: &'a RoventCollectionRootAdmissionV3,
        verifier: &NativeVerifier,
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

    /// An already-issued native acceptance is an observation, not a new
    /// admission. Retry it against immutable owner and output identity even
    /// after the root has been revoked.
    fn observed(
        admission: &'a RoventCollectionRootAdmissionV3,
        verifier: &NativeVerifier,
        accepted: &'a AcceptedRoleStep,
        output: &rz_contracts::pals::RoleOutput,
    ) -> Result<Self, AdaptiveCollectionError> {
        admission.validate_native_identity(verifier)?;
        admission.validate_observed_output(accepted.ticket(), output)?;
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
    pub fn admission(&self) -> &RoventCollectionRootAdmissionV3 {
        self.admission
    }
    pub fn accepted(&self) -> &AcceptedRoleStep {
        self.accepted
    }
}

#[cfg(target_os = "linux")]
pub struct RoventCollectionConsumedWitnessV3<'a> {
    acceptance: RoventCollectionAcceptedWitnessV3<'a>,
}
#[cfg(target_os = "linux")]
impl RoventCollectionConsumedWitnessV3<'_> {
    pub fn admission(&self) -> &RoventCollectionRootAdmissionV3 {
        self.acceptance.admission()
    }
    pub fn accepted(&self) -> &AcceptedRoleStep {
        self.acceptance.accepted()
    }
}

#[cfg(target_os = "linux")]
pub struct RoventRootRankingAcceptanceV3 {
    ranking: ProposerRankingPublished,
    admission: RoventCollectionRootAdmissionV3,
}
#[cfg(target_os = "linux")]
impl RoventRootRankingAcceptanceV3 {
    fn from_current_ranking(
        ranking: &ProposerRankingPublished,
        admission: &RoventCollectionRootAdmissionV3,
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
    pub fn admission(&self) -> &RoventCollectionRootAdmissionV3 {
        &self.admission
    }
    /// Recheck only immutable root/output identity for this already-published
    /// ranking; this grants no new root admission after revocation.
    pub fn validate_observed_output(
        &self,
        output: &rz_contracts::pals::RoleOutput,
    ) -> Result<(), AdaptiveCollectionError> {
        self.admission
            .validate_observed_ranking(&self.ranking, output)
    }
}

/// Observation only: the collection observer owns no teacher or GPU fence.
/// Every `try_*` method returning `Busy` must leave that witness unapplied;
/// the caller retains and retries the same original evidence with this owner.
/// Independently committed captures may remain committed. A retry must not
/// repeat native acceptance or Search consumption.
#[cfg(target_os = "linux")]
pub trait AdaptiveCollectionObserverV3<Q, O>: Send {
    fn admitted(
        &mut self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<(), AdaptiveCollectionError>;
    /// Busy leaves this admission unapplied. The caller retains the same
    /// original admission and observer and may retry without reissuing them.
    fn try_admitted(
        &mut self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        self.admitted(admission)
            .map(|()| AdaptiveCollectionBoundaryV2::Ready)
    }
    /// `Captured` admits the original reservation for this exact ticket. The
    /// native capture occurs later and must be independently confirmed before
    /// the ticket's collection retirement.
    /// A validator ticket omitted by policy must remain explicitly unsupported
    /// until the native lifecycle has a distinct non-capture disposition;
    /// it must not masquerade as a capture or disable the entire root.
    fn dispatch(
        &mut self,
        ticket: RoventCollectionDispatchWitnessV3<'_>,
        input: &FrameInput<ReviewView, Q>,
    ) -> Result<AdaptiveCollectionDispatchV2, AdaptiveCollectionError>;
    fn disable_after_failure(
        &mut self,
        _ticket: RoventCollectionDispatchWitnessV3<'_>,
        _original: &AdaptiveCollectionError,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        Err(AdaptiveCollectionError::new(
            AdaptiveCollectionStage::Dispatch,
            "DisableAfterFailureUnsupported",
            "original collection owner cannot establish a safe disabled boundary",
        ))
    }
    fn allow_disabled_prior_ticket(
        &mut self,
        _ticket: RoventCollectionDispatchWitnessV3<'_>,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        Err(AdaptiveCollectionError::new(
            AdaptiveCollectionStage::Dispatch,
            "DisabledPriorTicketUnsupported",
            "original collection owner cannot register the earlier ticket",
        ))
    }
    fn native_event(
        &mut self,
        event: RoventCollectionNativeEventWitnessV3<'_, O>,
    ) -> Result<(), AdaptiveCollectionError>;
    fn accepted(
        &mut self,
        accepted: RoventCollectionAcceptedWitnessV3<'_>,
        input: &FrameInput<ReviewView, Q>,
        output: &O,
    ) -> Result<(), AdaptiveCollectionError>;
    fn consumed(
        &mut self,
        accepted: RoventCollectionConsumedWitnessV3<'_>,
        observation: RoleOnlySearchConsumptionV2,
    ) -> Result<(), AdaptiveCollectionError>;
    fn adaptive_root_ranking(
        &mut self,
        _accepted: &RoventRootRankingAcceptanceV3,
        _input: &FrameInput<ReviewView, Q>,
        _output: &O,
    ) -> Result<(), AdaptiveCollectionError> {
        Err(AdaptiveCollectionError::new(
            AdaptiveCollectionStage::Accepted,
            "AdaptiveRankingUnsupported",
            "observer does not support adaptive V3 ranking",
        ))
    }
    fn adaptive_consumed(
        &mut self,
        _accepted: RoventCollectionConsumedWitnessV3<'_>,
        _after_error_readback: bool,
    ) -> Result<(), AdaptiveCollectionError> {
        Err(AdaptiveCollectionError::new(
            AdaptiveCollectionStage::Accepted,
            "AdaptiveConsumptionUnsupported",
            "observer does not support adaptive V3 consumption",
        ))
    }
    fn abandoned(&mut self, ticket: &RoleTicketEnvelope) -> Result<(), AdaptiveCollectionError>;
    fn settled(&mut self, tickets: &[RoleTicketEnvelope]) -> Result<(), AdaptiveCollectionError>;

    /// `Busy` means the borrowed original evidence was not applied. The
    /// caller must retain that evidence and retry it in the same root; it is
    /// neither a physical completion nor a collection failure. Implementors
    /// may retain independently committed captures before returning Busy,
    /// but must not apply the supplied event until they return Ready.
    fn try_native_event(
        &mut self,
        event: RoventCollectionNativeEventWitnessV3<'_, O>,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        self.native_event(event)
            .map(|()| AdaptiveCollectionBoundaryV2::Ready)
    }

    fn try_accepted(
        &mut self,
        accepted: RoventCollectionAcceptedWitnessV3<'_>,
        input: &FrameInput<ReviewView, Q>,
        output: &O,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        self.accepted(accepted, input, output)
            .map(|()| AdaptiveCollectionBoundaryV2::Ready)
    }

    fn try_consumed(
        &mut self,
        accepted: RoventCollectionConsumedWitnessV3<'_>,
        observation: RoleOnlySearchConsumptionV2,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        self.consumed(accepted, observation)
            .map(|()| AdaptiveCollectionBoundaryV2::Ready)
    }

    fn try_adaptive_root_ranking(
        &mut self,
        accepted: &RoventRootRankingAcceptanceV3,
        input: &FrameInput<ReviewView, Q>,
        output: &O,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        self.adaptive_root_ranking(accepted, input, output)
            .map(|()| AdaptiveCollectionBoundaryV2::Ready)
    }

    fn try_adaptive_consumed(
        &mut self,
        accepted: RoventCollectionConsumedWitnessV3<'_>,
        after_error_readback: bool,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        self.adaptive_consumed(accepted, after_error_readback)
            .map(|()| AdaptiveCollectionBoundaryV2::Ready)
    }

    fn try_abandoned(
        &mut self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        self.abandoned(ticket)
            .map(|()| AdaptiveCollectionBoundaryV2::Ready)
    }

    fn try_settled(
        &mut self,
        tickets: &[RoleTicketEnvelope],
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        self.settled(tickets)
            .map(|()| AdaptiveCollectionBoundaryV2::Ready)
    }
}
#[cfg(target_os = "linux")]
pub type AdaptiveCollectionObserverBoxV3<Q, O> = Box<dyn AdaptiveCollectionObserverV3<Q, O>>;

pub(crate) enum VersionedCollectionDispatchWitness<'a> {
    V2(RoventCollectionDispatchWitnessV2<'a>),
    #[cfg(target_os = "linux")]
    V3(RoventCollectionDispatchWitnessV3<'a>),
}
impl<'a> VersionedCollectionDispatchWitness<'a> {
    pub(crate) fn try_new(
        admission: &'a VersionedCollectionAdmission,
        verifier: &NativeVerifier,
        ticket: &'a RoleTicketEnvelope,
    ) -> Result<Self, RoventDispatchWitnessFailureV2> {
        match admission {
            VersionedCollectionAdmission::V2(a) => {
                RoventCollectionDispatchWitnessV2::try_new(a, verifier, ticket).map(Self::V2)
            }
            #[cfg(target_os = "linux")]
            VersionedCollectionAdmission::V3(a) => {
                RoventCollectionDispatchWitnessV3::try_new(a, verifier, ticket).map(Self::V3)
            }
        }
    }
}
pub(crate) enum VersionedCollectionNativeEventWitness<'a, O> {
    V2(RoventCollectionNativeEventWitnessV2<'a, O>),
    #[cfg(target_os = "linux")]
    V3(RoventCollectionNativeEventWitnessV3<'a, O>),
}
impl<'a, O> VersionedCollectionNativeEventWitness<'a, O> {
    pub(crate) fn new(
        admission: &'a VersionedCollectionAdmission,
        verifier: &NativeVerifier,
        event: &'a TicketNativeEvent<O>,
    ) -> Result<Self, AdaptiveCollectionError> {
        match admission {
            VersionedCollectionAdmission::V2(a) => {
                RoventCollectionNativeEventWitnessV2::new(a, verifier, event).map(Self::V2)
            }
            #[cfg(target_os = "linux")]
            VersionedCollectionAdmission::V3(a) => {
                RoventCollectionNativeEventWitnessV3::new(a, verifier, event).map(Self::V3)
            }
        }
    }
}
pub(crate) enum VersionedCollectionAcceptedWitness<'a> {
    V2(RoventCollectionAcceptedWitnessV2<'a>),
    #[cfg(target_os = "linux")]
    V3(RoventCollectionAcceptedWitnessV3<'a>),
}
impl<'a> VersionedCollectionAcceptedWitness<'a> {
    pub(crate) fn new(
        admission: &'a VersionedCollectionAdmission,
        verifier: &NativeVerifier,
        accepted: &'a AcceptedRoleStep,
        output: &rz_contracts::pals::RoleOutput,
    ) -> Result<Self, AdaptiveCollectionError> {
        match admission {
            VersionedCollectionAdmission::V2(a) => {
                RoventCollectionAcceptedWitnessV2::new(a, verifier, accepted, output).map(Self::V2)
            }
            #[cfg(target_os = "linux")]
            VersionedCollectionAdmission::V3(a) => {
                RoventCollectionAcceptedWitnessV3::new(a, verifier, accepted, output).map(Self::V3)
            }
        }
    }

    pub(crate) fn observed(
        admission: &'a VersionedCollectionAdmission,
        verifier: &NativeVerifier,
        accepted: &'a AcceptedRoleStep,
        output: &rz_contracts::pals::RoleOutput,
    ) -> Result<Self, AdaptiveCollectionError> {
        match admission {
            VersionedCollectionAdmission::V2(a) => {
                RoventCollectionAcceptedWitnessV2::new(a, verifier, accepted, output).map(Self::V2)
            }
            #[cfg(target_os = "linux")]
            VersionedCollectionAdmission::V3(a) => {
                RoventCollectionAcceptedWitnessV3::observed(a, verifier, accepted, output)
                    .map(Self::V3)
            }
        }
    }
}
pub(crate) enum VersionedCollectionConsumedWitness<'a> {
    V2(RoventCollectionConsumedWitnessV2<'a>),
    #[cfg(target_os = "linux")]
    V3(RoventCollectionConsumedWitnessV3<'a>),
}
impl<'a> VersionedCollectionConsumedWitness<'a> {
    pub(crate) fn new(acceptance: VersionedCollectionAcceptedWitness<'a>) -> Self {
        match acceptance {
            VersionedCollectionAcceptedWitness::V2(a) => {
                Self::V2(RoventCollectionConsumedWitnessV2::new(a))
            }
            #[cfg(target_os = "linux")]
            VersionedCollectionAcceptedWitness::V3(a) => {
                Self::V3(RoventCollectionConsumedWitnessV3 { acceptance: a })
            }
        }
    }
}
pub(crate) enum VersionedRootRankingAcceptance {
    V2(RoventRootRankingAcceptanceV2),
    #[cfg(target_os = "linux")]
    V3(RoventRootRankingAcceptanceV3),
}
impl VersionedRootRankingAcceptance {
    pub(crate) fn from_current_ranking(
        ranking: &ProposerRankingPublished,
        admission: &VersionedCollectionAdmission,
    ) -> Result<Self, AdaptiveCollectionError> {
        match admission {
            VersionedCollectionAdmission::V2(a) => {
                RoventRootRankingAcceptanceV2::from_current_ranking(ranking, a).map(Self::V2)
            }
            #[cfg(target_os = "linux")]
            VersionedCollectionAdmission::V3(a) => {
                RoventRootRankingAcceptanceV3::from_current_ranking(ranking, a).map(Self::V3)
            }
        }
    }
    pub(crate) fn from_observed_ranking(
        ranking: &ProposerRankingPublished,
        admission: &VersionedCollectionAdmission,
        output: &rz_contracts::pals::RoleOutput,
    ) -> Result<Self, AdaptiveCollectionError> {
        match admission {
            VersionedCollectionAdmission::V2(a) => {
                a.validate_output(&ranking.source().ticket, output)?;
                RoventRootRankingAcceptanceV2::from_current_ranking(ranking, a).map(Self::V2)
            }
            #[cfg(target_os = "linux")]
            VersionedCollectionAdmission::V3(a) => {
                a.validate_observed_ranking(ranking, output)?;
                Ok(Self::V3(RoventRootRankingAcceptanceV3 {
                    ranking: ranking.clone(),
                    admission: a.clone(),
                }))
            }
        }
    }
}

pub(crate) enum VersionedCollectionObserver<Q, O> {
    V2(AdaptiveCollectionObserverBoxV2<Q, O>),
    #[cfg(target_os = "linux")]
    V3(AdaptiveCollectionObserverBoxV3<Q, O>),
}
// On non-Linux targets only V2 exists; mismatch arms protect the two-owner
// pairing on Linux but are intentionally unreachable in the V2-only build.
#[allow(unreachable_patterns)]
impl<Q, O> VersionedCollectionObserver<Q, O> {
    pub(crate) fn try_native_event(
        &mut self,
        witness: VersionedCollectionNativeEventWitness<'_, O>,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        match (self, witness) {
            (Self::V2(o), VersionedCollectionNativeEventWitness::V2(w)) => o
                .native_event(w)
                .map(|()| AdaptiveCollectionBoundaryV2::Ready),
            #[cfg(target_os = "linux")]
            (Self::V3(o), VersionedCollectionNativeEventWitness::V3(w)) => o.try_native_event(w),
            _ => Err(mismatch(AdaptiveCollectionStage::NativeEvent)),
        }
    }

    pub(crate) fn try_accepted(
        &mut self,
        witness: VersionedCollectionAcceptedWitness<'_>,
        input: &FrameInput<ReviewView, Q>,
        output: &O,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        match (self, witness) {
            (Self::V2(o), VersionedCollectionAcceptedWitness::V2(w)) => o
                .accepted(w, input, output)
                .map(|()| AdaptiveCollectionBoundaryV2::Ready),
            #[cfg(target_os = "linux")]
            (Self::V3(o), VersionedCollectionAcceptedWitness::V3(w)) => {
                o.try_accepted(w, input, output)
            }
            _ => Err(mismatch(AdaptiveCollectionStage::Accepted)),
        }
    }

    pub(crate) fn try_consumed(
        &mut self,
        witness: VersionedCollectionConsumedWitness<'_>,
        observation: RoleOnlySearchConsumptionV2,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        match (self, witness) {
            (Self::V2(o), VersionedCollectionConsumedWitness::V2(w)) => o
                .consumed(w, observation)
                .map(|()| AdaptiveCollectionBoundaryV2::Ready),
            #[cfg(target_os = "linux")]
            (Self::V3(o), VersionedCollectionConsumedWitness::V3(w)) => {
                o.try_consumed(w, observation)
            }
            _ => Err(mismatch(AdaptiveCollectionStage::Accepted)),
        }
    }

    pub(crate) fn try_abandoned(
        &mut self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        match self {
            Self::V2(o) => o
                .abandoned(ticket)
                .map(|()| AdaptiveCollectionBoundaryV2::Ready),
            #[cfg(target_os = "linux")]
            Self::V3(o) => o.try_abandoned(ticket),
        }
    }

    pub(crate) fn try_settled(
        &mut self,
        tickets: &[RoleTicketEnvelope],
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        match self {
            Self::V2(o) => o
                .settled(tickets)
                .map(|()| AdaptiveCollectionBoundaryV2::Ready),
            #[cfg(target_os = "linux")]
            Self::V3(o) => o.try_settled(tickets),
        }
    }

    pub(crate) fn admitted(
        &mut self,
        admission: &VersionedCollectionAdmission,
    ) -> Result<(), AdaptiveCollectionError> {
        match (self, admission) {
            (Self::V2(o), VersionedCollectionAdmission::V2(a)) => o.admitted(a),
            #[cfg(target_os = "linux")]
            (Self::V3(o), VersionedCollectionAdmission::V3(a)) => o.admitted(a),
            _ => Err(mismatch(AdaptiveCollectionStage::Dispatch)),
        }
    }
    pub(crate) fn dispatch(
        &mut self,
        witness: VersionedCollectionDispatchWitness<'_>,
        input: &FrameInput<ReviewView, Q>,
    ) -> Result<AdaptiveCollectionDispatchV2, AdaptiveCollectionError> {
        match (self, witness) {
            (Self::V2(o), VersionedCollectionDispatchWitness::V2(w)) => o.dispatch(w, input),
            #[cfg(target_os = "linux")]
            (Self::V3(o), VersionedCollectionDispatchWitness::V3(w)) => o.dispatch(w, input),
            _ => Err(mismatch(AdaptiveCollectionStage::Dispatch)),
        }
    }
    pub(crate) fn disable_after_failure(
        &mut self,
        witness: VersionedCollectionDispatchWitness<'_>,
        original: &AdaptiveCollectionError,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        match (self, witness) {
            (Self::V2(o), VersionedCollectionDispatchWitness::V2(w)) => {
                o.disable_after_failure(w, original)
            }
            #[cfg(target_os = "linux")]
            (Self::V3(o), VersionedCollectionDispatchWitness::V3(w)) => {
                o.disable_after_failure(w, original)
            }
            _ => Err(mismatch(AdaptiveCollectionStage::Dispatch)),
        }
    }
    pub(crate) fn allow_disabled_prior_ticket(
        &mut self,
        witness: VersionedCollectionDispatchWitness<'_>,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        match (self, witness) {
            (Self::V2(o), VersionedCollectionDispatchWitness::V2(w)) => {
                o.allow_disabled_prior_ticket(w)
            }
            #[cfg(target_os = "linux")]
            (Self::V3(o), VersionedCollectionDispatchWitness::V3(w)) => {
                o.allow_disabled_prior_ticket(w)
            }
            _ => Err(mismatch(AdaptiveCollectionStage::Dispatch)),
        }
    }
    pub(crate) fn native_event(
        &mut self,
        witness: VersionedCollectionNativeEventWitness<'_, O>,
    ) -> Result<(), AdaptiveCollectionError> {
        match (self, witness) {
            (Self::V2(o), VersionedCollectionNativeEventWitness::V2(w)) => o.native_event(w),
            #[cfg(target_os = "linux")]
            (Self::V3(o), VersionedCollectionNativeEventWitness::V3(w)) => o.native_event(w),
            _ => Err(mismatch(AdaptiveCollectionStage::NativeEvent)),
        }
    }
    pub(crate) fn accepted(
        &mut self,
        witness: VersionedCollectionAcceptedWitness<'_>,
        input: &FrameInput<ReviewView, Q>,
        output: &O,
    ) -> Result<(), AdaptiveCollectionError> {
        match (self, witness) {
            (Self::V2(o), VersionedCollectionAcceptedWitness::V2(w)) => {
                o.accepted(w, input, output)
            }
            #[cfg(target_os = "linux")]
            (Self::V3(o), VersionedCollectionAcceptedWitness::V3(w)) => {
                o.accepted(w, input, output)
            }
            _ => Err(mismatch(AdaptiveCollectionStage::Accepted)),
        }
    }
    pub(crate) fn consumed(
        &mut self,
        witness: VersionedCollectionConsumedWitness<'_>,
        observation: RoleOnlySearchConsumptionV2,
    ) -> Result<(), AdaptiveCollectionError> {
        match (self, witness) {
            (Self::V2(o), VersionedCollectionConsumedWitness::V2(w)) => o.consumed(w, observation),
            #[cfg(target_os = "linux")]
            (Self::V3(o), VersionedCollectionConsumedWitness::V3(w)) => o.consumed(w, observation),
            _ => Err(mismatch(AdaptiveCollectionStage::Accepted)),
        }
    }
    pub(crate) fn adaptive_root_ranking(
        &mut self,
        witness: &VersionedRootRankingAcceptance,
        input: &FrameInput<ReviewView, Q>,
        output: &O,
    ) -> Result<(), AdaptiveCollectionError> {
        match (self, witness) {
            (Self::V2(o), VersionedRootRankingAcceptance::V2(w)) => {
                o.adaptive_root_ranking(w, input, output)
            }
            #[cfg(target_os = "linux")]
            (Self::V3(o), VersionedRootRankingAcceptance::V3(w)) => {
                o.adaptive_root_ranking(w, input, output)
            }
            _ => Err(mismatch(AdaptiveCollectionStage::Accepted)),
        }
    }
    pub(crate) fn try_adaptive_root_ranking(
        &mut self,
        witness: &VersionedRootRankingAcceptance,
        input: &FrameInput<ReviewView, Q>,
        output: &O,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        match (self, witness) {
            (Self::V2(o), VersionedRootRankingAcceptance::V2(w)) => o
                .adaptive_root_ranking(w, input, output)
                .map(|()| AdaptiveCollectionBoundaryV2::Ready),
            #[cfg(target_os = "linux")]
            (Self::V3(o), VersionedRootRankingAcceptance::V3(w)) => {
                o.try_adaptive_root_ranking(w, input, output)
            }
            _ => Err(mismatch(AdaptiveCollectionStage::Accepted)),
        }
    }
    pub(crate) fn adaptive_consumed(
        &mut self,
        witness: VersionedCollectionConsumedWitness<'_>,
        after_error: bool,
    ) -> Result<(), AdaptiveCollectionError> {
        match (self, witness) {
            (Self::V2(o), VersionedCollectionConsumedWitness::V2(w)) => {
                o.adaptive_consumed(w, after_error)
            }
            #[cfg(target_os = "linux")]
            (Self::V3(o), VersionedCollectionConsumedWitness::V3(w)) => {
                o.adaptive_consumed(w, after_error)
            }
            _ => Err(mismatch(AdaptiveCollectionStage::Accepted)),
        }
    }
    pub(crate) fn try_adaptive_consumed(
        &mut self,
        witness: VersionedCollectionConsumedWitness<'_>,
        after_error: bool,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        match (self, witness) {
            (Self::V2(o), VersionedCollectionConsumedWitness::V2(w)) => o
                .adaptive_consumed(w, after_error)
                .map(|()| AdaptiveCollectionBoundaryV2::Ready),
            #[cfg(target_os = "linux")]
            (Self::V3(o), VersionedCollectionConsumedWitness::V3(w)) => {
                o.try_adaptive_consumed(w, after_error)
            }
            _ => Err(mismatch(AdaptiveCollectionStage::Accepted)),
        }
    }
    pub(crate) fn abandoned(
        &mut self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(), AdaptiveCollectionError> {
        match self {
            Self::V2(o) => o.abandoned(ticket),
            #[cfg(target_os = "linux")]
            Self::V3(o) => o.abandoned(ticket),
        }
    }
    pub(crate) fn settled(
        &mut self,
        tickets: &[RoleTicketEnvelope],
    ) -> Result<(), AdaptiveCollectionError> {
        match self {
            Self::V2(o) => o.settled(tickets),
            #[cfg(target_os = "linux")]
            Self::V3(o) => o.settled(tickets),
        }
    }
}

/// Forward a real manager-recorded consumption without granting a new
/// acceptance or hiding a missing original owner.
pub(crate) fn notify_adaptive_consumed_versioned<Q, O>(
    observer: &mut Option<VersionedCollectionObserver<Q, O>>,
    admission: Option<&VersionedCollectionAdmission>,
    verifier: Option<&NativeVerifier>,
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
            "consumption has no original admission or native verifier",
        )
    };
    let witness = VersionedCollectionAcceptedWitness::new(
        admission.ok_or_else(missing)?,
        verifier.ok_or_else(missing)?,
        accepted,
        output,
    )?;
    observer.adaptive_consumed(
        VersionedCollectionConsumedWitness::new(witness),
        after_error_readback,
    )
}

/// Observe an already completed search consumption without repeating it when
/// the same original observer is temporarily Busy.
pub(crate) fn try_notify_adaptive_consumed_versioned<Q, O>(
    observer: &mut Option<VersionedCollectionObserver<Q, O>>,
    admission: Option<&VersionedCollectionAdmission>,
    verifier: Option<&NativeVerifier>,
    accepted: &AcceptedRoleStep,
    output: &rz_contracts::pals::RoleOutput,
    after_error_readback: bool,
) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
    let Some(observer) = observer.as_mut() else {
        return Ok(AdaptiveCollectionBoundaryV2::Ready);
    };
    let missing = || {
        AdaptiveCollectionError::new(
            AdaptiveCollectionStage::Accepted,
            "OriginalOwnerUnavailable",
            "consumption has no original admission or native verifier",
        )
    };
    let witness = VersionedCollectionAcceptedWitness::observed(
        admission.ok_or_else(missing)?,
        verifier.ok_or_else(missing)?,
        accepted,
        output,
    )?;
    observer.try_adaptive_consumed(
        VersionedCollectionConsumedWitness::new(witness),
        after_error_readback,
    )
}
