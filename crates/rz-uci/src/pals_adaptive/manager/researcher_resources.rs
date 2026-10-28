//! Actual researcher resource kinds. A profiled external owner has neither an
//! internal native credit ledger nor an adaptive native worker controller.
use super::*;

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
use crate::rovent_root_admission::{RoventRootAdmission, source_seal::RoventRootAdmissionV2};
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
use rz_experiments::RoventInputLockV2;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
use rz_search::cpu_checker::{CheckerError, CheckerShutdown};
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
use rz_search::external_cpu::{
    ExternalCpuConfig, ExternalGoConstraint, ProfiledExternalCurrentRootFence,
    ProfiledExternalDispatch, ProfiledExternalJob, ProfiledExternalProcessCloseProof,
    ProfiledExternalRootSource,
};

/// Owns exactly one original versioned source for its entire physical lifetime.
/// The wrapper only delegates; it does not project a V2 root into a V1 root.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub(super) enum ProfiledExternalSource {
    V1(profiled_external_source::ProfiledExternalManagerSource),
    V2(profiled_external_source::ProfiledExternalManagerSource<RoventInputLockV2>),
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
macro_rules! delegate_profiled {
    ($name:ident(&self $(, $arg:ident : $arg_ty:ty)*) -> $ret:ty) => {
        pub(super) fn $name(&self, $($arg: $arg_ty),*) -> $ret {
            match self {
                Self::V1(source) => source.$name($($arg),*),
                Self::V2(source) => source.$name($($arg),*),
            }
        }
    };
    ($name:ident(&mut self $(, $arg:ident : $arg_ty:ty)*) -> $ret:ty) => {
        pub(super) fn $name(&mut self, $($arg: $arg_ty),*) -> $ret {
            match self {
                Self::V1(source) => source.$name($($arg),*),
                Self::V2(source) => source.$name($($arg),*),
            }
        }
    };
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl ProfiledExternalSource {
    pub(super) fn root_admission_v1(&self) -> Option<&Arc<RoventRootAdmission>> {
        match self {
            Self::V1(source) => Some(source.root_admission()),
            Self::V2(_) => None,
        }
    }
    pub(super) fn root_admission_v2(&self) -> Option<&Arc<RoventRootAdmissionV2>> {
        match self {
            Self::V1(_) => None,
            Self::V2(source) => Some(source.root_admission()),
        }
    }
    pub(super) fn root_authority(&self) -> rz_contracts::pals::SearchAuthority {
        match self {
            Self::V1(s) => s.root_admission().authority(),
            Self::V2(s) => s.root_admission().authority(),
        }
    }
    pub(super) fn root_hard_deadline(&self) -> rz_contracts::Deadline {
        match self {
            Self::V1(s) => s.root_admission().hard_deadline(),
            Self::V2(s) => s.root_admission().hard_deadline(),
        }
    }
    pub(super) fn root_original_context(&self) -> &crate::search_driver::SearchSessionContext {
        match self {
            Self::V1(s) => s.root_admission().original_context(),
            Self::V2(s) => s.root_admission().original_context(),
        }
    }
    pub(super) fn root_runtime_clock(&self) -> &rz_runtime::contracts::ContractSystemClock {
        match self {
            Self::V1(s) => s.root_admission().runtime_clock(),
            Self::V2(s) => s.root_admission().runtime_clock(),
        }
    }
    pub(super) fn root_budget_digest(&self) -> Result<rz_contracts::Digest, CheckerError> {
        match self {
            Self::V1(s) => Ok(s.root_admission().root_budget_digest()),
            Self::V2(s) => s
                .root_admission()
                .descriptor()
                .root_budget_digest()
                .map_err(|_| CheckerError::Invalid("invalid original V2 root budget")),
        }
    }
    delegate_profiled!(current_source(&self) -> &ProfiledExternalRootSource);
    delegate_profiled!(configuration(&self) -> &ExternalCpuConfig);
    delegate_profiled!(question_constraint(&self) -> ExternalGoConstraint);
    delegate_profiled!(soft_quantum(&self) -> Duration);
    delegate_profiled!(signal(&self) -> rz_runtime::CompletionSignal);
    delegate_profiled!(pending_dispatch(&self) -> Option<&ProfiledExternalDispatch>);
    delegate_profiled!(is_quarantined(&self) -> bool);
    delegate_profiled!(root_drained(&self) -> bool);
    delegate_profiled!(submit(&mut self, job: ProfiledExternalJob) -> Result<ProfiledExternalDispatch, CheckerError>);
    delegate_profiled!(take_completion(&mut self) -> Option<profiled_external_source::ProfiledExternalManagerCompletion>);
    delegate_profiled!(begin_root_quiesce(&mut self, until: Instant) -> Result<(), CheckerError>);
    delegate_profiled!(try_root_fence(&mut self) -> Option<Result<ProfiledExternalCurrentRootFence, CheckerError>>);
    delegate_profiled!(root_fence(&self) -> Option<&ProfiledExternalCurrentRootFence>);
    delegate_profiled!(request_shutdown(&mut self, until: Instant) -> ());
    delegate_profiled!(try_join(&mut self) -> Option<Result<CheckerShutdown, CheckerError>>);
    delegate_profiled!(shutdown_snapshot(&self) -> Option<&Result<CheckerShutdown, CheckerError>>);
    delegate_profiled!(process_close_proof(&self) -> Result<Option<ProfiledExternalProcessCloseProof>, CheckerError>);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveResearcherSourceKind {
    Native,
    ProfiledExternal,
}

pub(super) struct NativeResearcherResources {
    pub ledger: CpuCreditLedger,
    pub controller: AdaptiveController,
}

// A manager holds exactly one researcher selection. Keep its actual ledger or
// external owner in that slot without changing ownership on a variant switch.
#[allow(clippy::large_enum_variant)]
pub(super) enum ResearcherResources {
    Native(NativeResearcherResources),
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    ProfiledExternal(Box<ProfiledExternalSource>),
}

impl ResearcherResources {
    pub fn kind(&self) -> AdaptiveResearcherSourceKind {
        match self {
            Self::Native(_) => AdaptiveResearcherSourceKind::Native,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            Self::ProfiledExternal(_) => AdaptiveResearcherSourceKind::ProfiledExternal,
        }
    }
    pub fn native(&self) -> Result<&NativeResearcherResources, AdaptiveManagerError> {
        match self {
            Self::Native(native) => Ok(native),
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            Self::ProfiledExternal(_) => Err(AdaptiveManagerError::Invalid(
                "profiled external R has no native ledger/controller",
            )),
        }
    }
    pub fn native_mut(&mut self) -> Result<&mut NativeResearcherResources, AdaptiveManagerError> {
        match self {
            Self::Native(native) => Ok(native),
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            Self::ProfiledExternal(_) => Err(AdaptiveManagerError::Invalid(
                "profiled external R has no native ledger/controller",
            )),
        }
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn profiled(&self) -> Result<&ProfiledExternalSource, AdaptiveManagerError> {
        match self {
            Self::ProfiledExternal(source) => Ok(source),
            Self::Native(_) => Err(AdaptiveManagerError::Invalid(
                "native R has no profiled external owner",
            )),
        }
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn profiled_mut(&mut self) -> Result<&mut ProfiledExternalSource, AdaptiveManagerError> {
        match self {
            Self::ProfiledExternal(source) => Ok(source),
            Self::Native(_) => Err(AdaptiveManagerError::Invalid(
                "native R has no profiled external owner",
            )),
        }
    }
}

impl<Q, B, D> AdaptiveExecutionManager<Q, B, D>
where
    Q: Clone + Send + Sync + 'static,
    B: TicketNativeBackend<ReviewView, Q>,
    B::Output: ProposerRankingOutput,
    D: AdaptiveSearchDriver<Q>,
{
    pub fn researcher_source_kind(&self) -> AdaptiveResearcherSourceKind {
        self.researcher.kind()
    }
    pub(super) fn native_ledger(&self) -> Result<&CpuCreditLedger, AdaptiveManagerError> {
        Ok(&self.researcher.native()?.ledger)
    }
    pub(super) fn native_ledger_mut(
        &mut self,
    ) -> Result<&mut CpuCreditLedger, AdaptiveManagerError> {
        Ok(&mut self.researcher.native_mut()?.ledger)
    }
    pub(super) fn native_credit_outstanding(&self) -> Result<bool, AdaptiveManagerError> {
        match &self.researcher {
            ResearcherResources::Native(native) => Ok(native.ledger.global_usage()?.1 != 0),
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            ResearcherResources::ProfiledExternal(_) => Ok(false),
        }
    }
    pub(super) fn profiled_owner_closed(&self) -> bool {
        match &self.researcher {
            ResearcherResources::Native(_) => true,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            ResearcherResources::ProfiledExternal(source) => match self.close_mode {
                AdaptiveCloseMode::RootQuiesce => source.root_drained(),
                AdaptiveCloseMode::ProcessShutdown => {
                    source.shutdown_snapshot().is_some_and(|result| {
                        result.as_ref().is_ok_and(|receipt| {
                            receipt.cleanup_complete
                                && receipt.exit_observed
                                && receipt.stdout_drained
                                && receipt.stderr_drained
                                && !receipt.quarantined
                                && !receipt.ownership_lost
                        })
                    })
                }
            },
        }
    }
    pub(super) fn profiled_busy(&self) -> bool {
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if self.profiled_binding.is_some()
            || self.profiled_orphan.is_some()
            || self
                .deferred
                .iter()
                .any(|event| matches!(event, DeferredCommit::ProfiledExternal { .. }))
        {
            return true;
        }
        false
    }
    pub(super) fn profiled_frame_pending(&self, id: FrameId) -> bool {
        #[cfg(all(feature="pals-adaptive",feature="onnx-cpu"))]
        return self.profiled_binding.as_ref().is_some_and(|binding|binding.work.frame==id)
            || self.deferred.iter().any(|event|matches!(event,DeferredCommit::ProfiledExternal{binding,..} if binding.work.frame==id));
        #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
        {
            let _ = id;
            false
        }
    }
    pub(super) fn profiled_dispatch_available(&self) -> bool {
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        return self.researcher_source_kind() == AdaptiveResearcherSourceKind::ProfiledExternal
            && !self.profiled_busy()
            && self.researcher.profiled().is_ok_and(|source| {
                source.pending_dispatch().is_none()
                    && !source.is_quarantined()
                    && !source.root_drained()
            });
        #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
        {
            false
        }
    }
}
