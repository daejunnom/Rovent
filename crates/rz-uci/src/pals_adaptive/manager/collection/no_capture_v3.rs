//! Original V3 root and worker for an optional whole-root collection bypass.
//!
//! This capability does not admit a ticket, claim teacher scope, or attest
//! physical completion. It preserves the V3 owner rather than reconstructing
//! the older V2 root from serialized metadata.

use super::*;
use rz_search::pals::coordinator::NativeAcceptanceVerifier;

#[derive(Clone)]
pub struct RoventNoCaptureRootCapabilityV3 {
    admission: RoventCollectionRootAdmissionV3,
    verifier: NativeAcceptanceVerifier,
}

pub(crate) enum RoventNoCaptureIssueErrorV3 {
    Busy(rz_contracts::ContractError),
    Rejected(AdaptiveCollectionError),
}

impl RoventNoCaptureRootCapabilityV3 {
    pub(crate) fn new(
        admission: RoventCollectionRootAdmissionV3,
        verifier: NativeAcceptanceVerifier,
    ) -> Result<Self, RoventNoCaptureIssueErrorV3> {
        admission
            .validate_native_identity(&verifier)
            .map_err(RoventNoCaptureIssueErrorV3::Rejected)?;
        if !admission
            .root_admission()
            .original_context()
            .try_accepts()
            .map_err(|error| {
                if error.code == rz_contracts::ErrorCode::ResourceExhausted {
                    RoventNoCaptureIssueErrorV3::Busy(error)
                } else {
                    RoventNoCaptureIssueErrorV3::Rejected(AdaptiveCollectionError::new(
                        AdaptiveCollectionStage::Dispatch,
                        "NoCaptureRootUnavailable",
                        error.to_string(),
                    ))
                }
            })?
        {
            return Err(RoventNoCaptureIssueErrorV3::Rejected(
                AdaptiveCollectionError::new(
                    AdaptiveCollectionStage::Dispatch,
                    "NoCaptureRootRevoked",
                    "original root is not live",
                ),
            ));
        }
        Ok(Self {
            admission,
            verifier,
        })
    }

    pub fn same_original_admission(&self, other: &RoventCollectionRootAdmissionV3) -> bool {
        self.admission.same_original_admission(other)
    }

    pub fn same_original_root(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.admission.root_admission(),
            other.admission.root_admission(),
        )
    }

    pub fn same_root_admission(&self, other: &RoventCollectionRootAdmissionV3) -> bool {
        Arc::ptr_eq(self.admission.root_admission(), other.root_admission())
    }

    pub fn authority(&self) -> SearchAuthority {
        self.admission.root_admission().authority()
    }

    pub fn same_original_capability(&self, other: &Self) -> bool {
        self.admission.same_original_admission(&other.admission)
            && self.verifier.same_owner(&other.verifier)
    }

    /// Late discard checks immutable owner and ticket identity even after
    /// root revocation. It grants no new admission or completion authority.
    pub fn validate_native_discard_owner<V, Q, P>(
        &self,
        context: &crate::pals_native::ticketed::TicketOwnerDispatchContext<'_, V, Q, P>,
    ) -> Result<(), AdaptiveCollectionError> {
        if !context.same_owner(&self.verifier) {
            return Err(AdaptiveCollectionError::new(
                AdaptiveCollectionStage::Dispatch,
                "NoCaptureWorkerMismatch",
                "different native owner",
            ));
        }
        self.admission.validate_native_discard_owner(context)
    }

    pub fn matches_shutdown_receipt(
        &self,
        receipt: &super::super::AdaptiveShutdownReceipt,
    ) -> bool {
        receipt.authority == self.admission.root_admission().authority()
            && receipt.permits_root_reuse()
    }

    pub fn matches_role_only_shutdown_receipt(
        &self,
        receipt: &super::super::role_only::RoleOnlyShutdownReceipt,
    ) -> bool {
        receipt.authority == self.admission.root_admission().authority()
            && receipt.root_budget == self.admission.root_budget()
            && receipt.permits_semantic_closure()
    }
}
