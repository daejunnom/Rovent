//! First selected-view C preparation before a researcher exists.
//!
//! The move-only PreparingRoot owns the semantic Store and registration. It
//! alone mints the borrowed source below. No native role ticket, synthetic
//! FrameId, substitute researcher or empty bootstrap label is issued here.
use super::super::preparing_root::{PreparingRootReviewSource, ResearcherPreparingRoot};
use super::*;
use crate::cpu_value::CpuValueIdentity;
use crate::cpu_value::nnuec::v2::review::{
    BoundNnuecReviewContext, NnuecPreparationTerminalAck, Request, RequestIdentity,
};
use crate::cpu_worker::MaterializationRequest;
use std::sync::atomic::AtomicBool;

/// An immutable invocation seal, distinct from a native P/C role ticket.
/// The actual initial ReviewView keeps its original Store lease alive.
pub struct PreparingRootViewSeal {
    owner: Arc<()>,
    authority: SearchAuthority,
    root: StateId,
    original_deadline: Deadline,
    hard_deadline: Instant,
    position: PositionSnapshot,
    view: Arc<ReviewView>,
    model: CpuValueIdentity,
    cancellation: Arc<AtomicBool>,
    preparation_request: RequestIdentity,
}
impl PreparingRootViewSeal {
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn root_state(&self) -> StateId {
        self.root
    }
    pub fn view_seal(&self) -> [u8; 32] {
        self.view.seal_digest()
    }

    pub(crate) fn matches_preparation_request(&self, request: &Request) -> bool {
        self.preparation_request.same_request(&request.identity())
    }

    pub(crate) fn preparation_request_identity(&self) -> &RequestIdentity {
        &self.preparation_request
    }

    pub(crate) fn matches_preparation_terminal(
        &self,
        terminal: &NnuecPreparationTerminalAck,
    ) -> bool {
        terminal.matches_request(&self.preparation_request)
    }

    /// The borrowed source is minted by the same actual PreparingRoot, and
    /// cannot be assembled from caller JSON or public success flags.
    pub(crate) fn validate_root_source(
        &self,
        actual: &PreparingRootReviewSource<'_>,
    ) -> Result<(), AdaptiveNnuecPreparationError> {
        if !Arc::ptr_eq(&self.owner, actual.identity())
            || self.authority != actual.authority()
            || self.root != actual.root_state()
            || self.original_deadline != actual.original_deadline()
            || self.hard_deadline != actual.hard_deadline()
            || !self.position.same_state(&actual.position().snapshot())
            || !Arc::ptr_eq(&self.view, actual.view())
            || !Arc::ptr_eq(&self.cancellation, actual.cancellation())
            || self.model != *actual.source().model().identity()
            || self.view.root_state() != self.root
            || self.view.generation() != self.authority.root.0
            || !self.view.validates(actual.stores())
        {
            return Err(AdaptiveNnuecPreparationError::StateMismatch);
        }
        Ok(())
    }

    fn validate_bound(
        &self,
        actual: &PreparingRootReviewSource<'_>,
        bound: &Arc<BoundNnuecReviewContext>,
    ) -> Result<(), AdaptiveNnuecPreparationError> {
        self.validate_root_source(actual)?;
        let provenance = bound.provenance();
        if !bound.matches_request(&self.preparation_request)
            || provenance.model != self.model
            || provenance.root != self.root
            || provenance.generation != self.authority.root.0
            || provenance.sequence != self.view.sequence()
            || provenance.full_view_digest != self.view.digest()
            || provenance.view_seal != self.view.seal_digest()
            || provenance.record_scope_digest != self.view.record_scope_digest()
            || provenance.policy_digest != self.view.policy_digest()
            || !Arc::ptr_eq(bound.view(), &self.view)
            || !bound.validates(
                actual.stores(),
                self.root,
                self.authority.root.0,
                self.view.seal_digest(),
            )
        {
            return Err(AdaptiveNnuecPreparationError::StateMismatch);
        }
        Ok(())
    }
}

pub struct PreparingRootNnuecRequest {
    request: Request,
    seal: PreparingRootViewSeal,
}
impl PreparingRootNnuecRequest {
    pub fn into_parts(self) -> (Request, PreparingRootViewSeal) {
        (self.request, self.seal)
    }
}

impl<M: RoleModel> ResearcherPreparingRoot<M> {
    /// Only the existing preparation actor performs encoding and neural work.
    /// Selecting the first view is a semantic-owner operation, once per root.
    pub fn prepare_initial_nnuec_review(
        &mut self,
    ) -> Result<PreparingRootNnuecRequest, AdaptiveNnuecPreparationError> {
        let actual = self.initial_review_source()?;
        let request = Request::admit(
            actual.source().model(),
            Arc::clone(actual.view()),
            actual.stores(),
            actual.root_state(),
            actual.authority().root.0,
            actual.hard_deadline(),
            Arc::clone(actual.cancellation()),
        )?;
        let seal = PreparingRootViewSeal {
            owner: Arc::clone(actual.identity()),
            authority: actual.authority(),
            root: actual.root_state(),
            original_deadline: actual.original_deadline(),
            hard_deadline: actual.hard_deadline(),
            position: actual.position().snapshot(),
            view: Arc::clone(actual.view()),
            model: actual.source().model().identity().clone(),
            cancellation: Arc::clone(actual.cancellation()),
            preparation_request: request.identity(),
        };
        Ok(PreparingRootNnuecRequest { request, seal })
    }

    /// Reject a mismatched completion before changing the caller's semantic
    /// state. In particular this read-only source cannot select a newer view.
    /// The materialization actor, private Ready and same-owner handoff remain
    /// separate from this validation; no researcher is constructed here.
    pub fn materialize_initial_nnuec_review(
        &self,
        original: &PreparingRootViewSeal,
        bound: Arc<BoundNnuecReviewContext>,
    ) -> Result<MaterializationRequest, AdaptiveNnuecPreparationError> {
        let actual = self.prepared_review_source()?;
        original.validate_bound(&actual, &bound)?;
        Ok(actual
            .source()
            .prepare_materialization(bound, actual.stores(), original.view_seal())?)
    }
}
