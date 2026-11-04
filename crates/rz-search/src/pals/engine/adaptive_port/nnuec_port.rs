//! Capture and validate the original selected-view custody for NNUEC C.
//!
//! The preparation actor performs the neural work. This bridge neither
//! installs a researcher nor consumes a role result, changes a CPU profile,
//! or treats the prepared context as a search bound.
use super::*;
use crate::cpu_value::CpuValueIdentity;
use crate::cpu_value::nnuec::NnuecError;
use crate::cpu_value::nnuec::v2::Model;
use crate::cpu_value::nnuec::v2::capture::{
    CaptureError, NnuecCaptureLimits, NnuecCaptureReservation,
};
use crate::cpu_value::nnuec::v2::cpu::V2CpuValue;
use crate::cpu_value::nnuec::v2::review::{BoundNnuecReviewContext, PreparationError, Request};
use crate::cpu_worker::CpuWorkerError;
use crate::cpu_worker::PreparedValueInstallation;
use std::sync::atomic::AtomicBool;

#[derive(Debug)]
pub enum AdaptiveNnuecPreparationError {
    Semantic(PalsError),
    Preparation(PreparationError),
    Neural(NnuecError),
    Worker(CpuWorkerError),
    StateMismatch,
}
impl std::fmt::Display for AdaptiveNnuecPreparationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "adaptive NNUEC preparation: {self:?}")
    }
}
impl std::error::Error for AdaptiveNnuecPreparationError {}
impl From<PalsError> for AdaptiveNnuecPreparationError {
    fn from(error: PalsError) -> Self {
        Self::Semantic(error)
    }
}
impl From<PreparationError> for AdaptiveNnuecPreparationError {
    fn from(error: PreparationError) -> Self {
        Self::Preparation(error)
    }
}
impl From<NnuecError> for AdaptiveNnuecPreparationError {
    fn from(error: NnuecError) -> Self {
        Self::Neural(error)
    }
}
impl From<CpuWorkerError> for AdaptiveNnuecPreparationError {
    fn from(error: CpuWorkerError) -> Self {
        Self::Worker(error)
    }
}

#[derive(Debug)]
pub enum AdaptiveNnuecCaptureError {
    Preparation(AdaptiveNnuecPreparationError),
    Capture(CaptureError),
}
impl std::fmt::Display for AdaptiveNnuecCaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "adaptive NNUEC capture: {self:?}")
    }
}
impl std::error::Error for AdaptiveNnuecCaptureError {}

/// Minted only after the actual port validates the original R question. A
/// copied digest or a subsequently selected view cannot manufacture this
/// custody. Keeping the work also keeps its exact Q/view/store pin alive.
pub struct AdaptiveNnuecReviewSeal {
    work: CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    model: CpuValueIdentity,
}
impl AdaptiveNnuecReviewSeal {
    pub(super) fn work(&self) -> &CoordinatorWork<ReviewView, AdaptivePortQuestion> {
        &self.work
    }
    pub fn view(&self) -> &Arc<ReviewView> {
        &self.work.input.view
    }
    pub fn view_seal(&self) -> ReviewViewSeal {
        self.work.input.view_seal
    }
    pub fn frame(&self) -> FrameId {
        self.work.frame
    }
    pub fn question_digest(&self) -> ContractDigest {
        self.work.input.question_digest
    }
    pub fn full_view_digest(&self) -> ContractDigest {
        ContractDigest(self.work.input.view.digest())
    }
    pub fn original_deadline(&self) -> Deadline {
        self.work.original_deadline
    }
    pub fn authority(&self) -> SearchAuthority {
        self.work.frame.authority
    }
    pub fn model(&self) -> &CpuValueIdentity {
        &self.model
    }
}

pub struct AdaptiveNnuecReviewRequest {
    request: Request,
    seal: AdaptiveNnuecReviewSeal,
}
impl AdaptiveNnuecReviewRequest {
    pub fn into_parts(self) -> (Request, AdaptiveNnuecReviewSeal) {
        (self.request, self.seal)
    }
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// Must run before prepare_cpu_question fixes the actual worker profile.
    /// This only captures bounded immutable custody; encoding/forward remain
    /// on the existing preparation actor.
    pub fn prepare_cpu_nnuec_review(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        model: &Model,
        cancel: Arc<AtomicBool>,
    ) -> Result<AdaptiveNnuecReviewRequest, AdaptiveNnuecPreparationError> {
        self.check_work(work)?;
        AdaptivePortQuestion::validate_frontier_source(work, &self.engine.semantic.stores)?;
        if work.input.stage.role().is_some()
            || work.input.view.root_state() != self.root_state_id()
            || work.frame.authority != self.authority
            || self.engine.researcher.cpu_registered_value.as_ref() != Some(model.identity())
        {
            return Err(AdaptiveNnuecPreparationError::StateMismatch);
        }
        let request = Request::admit(
            model,
            Arc::clone(&work.input.view),
            &self.engine.semantic.stores,
            self.root_state_id(),
            self.authority.root.0,
            work.input.question.deadline,
            cancel,
        )?;
        Ok(AdaptiveNnuecReviewRequest {
            request,
            seal: AdaptiveNnuecReviewSeal {
                work: work.clone(),
                model: model.identity().clone(),
            },
        })
    }

    /// Bind the completed C to precisely the originally captured question.
    /// No latest-view substitution, role consumption or worker installation
    /// occurs here. The driver must use the installed-value ack/profile before
    /// it admits the CPU question, ledger reservation, job and actual Begin.
    pub fn bind_cpu_nnuec_review(
        &self,
        original: &AdaptiveNnuecReviewSeal,
        model: Arc<Model>,
        bound: Arc<BoundNnuecReviewContext>,
    ) -> Result<V2CpuValue, AdaptiveNnuecPreparationError> {
        self.validate_nnuec_review_completion(original, &model, &bound)?;
        Ok(V2CpuValue::from_bound(
            model,
            bound,
            &self.engine.semantic.stores,
            original.work.input.view.seal_digest(),
        )?)
    }

    /// Return only a prepared payload. The existing worker must independently
    /// prove idle/no-token ownership, install and return its private ack.
    pub fn install_cpu_nnuec_review(
        &self,
        original: &AdaptiveNnuecReviewSeal,
        model: Arc<Model>,
        bound: Arc<BoundNnuecReviewContext>,
    ) -> Result<PreparedValueInstallation, AdaptiveNnuecPreparationError> {
        self.validate_nnuec_review_completion(original, &model, &bound)?;
        Ok(PreparedValueInstallation::from_bound(
            model,
            bound,
            &self.engine.semantic.stores,
            original.work.input.view.seal_digest(),
        )?)
    }

    /// Opt-in custody of this original R question only. This does not bind a
    /// Job, run a score, publish a target or construct another RequestId.
    pub fn reserve_cpu_nnuec_capture(
        &self,
        original: &AdaptiveNnuecReviewSeal,
        model: Arc<Model>,
        bound: Arc<BoundNnuecReviewContext>,
        capture_sequence: u64,
        limits: NnuecCaptureLimits,
    ) -> Result<NnuecCaptureReservation, AdaptiveNnuecCaptureError> {
        self.validate_nnuec_review_completion(original, &model, &bound)
            .map_err(AdaptiveNnuecCaptureError::Preparation)?;
        NnuecCaptureReservation::reserve(
            model,
            bound,
            &original.work,
            &self.engine.semantic.stores,
            capture_sequence,
            limits,
        )
        .map_err(AdaptiveNnuecCaptureError::Capture)
    }

    pub(super) fn validate_nnuec_review_completion(
        &self,
        original: &AdaptiveNnuecReviewSeal,
        model: &Model,
        bound: &Arc<BoundNnuecReviewContext>,
    ) -> Result<(), AdaptiveNnuecPreparationError> {
        self.check_work(&original.work)?;
        AdaptivePortQuestion::validate_frontier_source(
            &original.work,
            &self.engine.semantic.stores,
        )?;
        let work = &original.work;
        let provenance = bound.provenance();
        if work.input.stage.role().is_some()
            || model.identity() != &original.model
            || self.engine.researcher.cpu_registered_value.as_ref() != Some(model.identity())
            || provenance.model != original.model
            || provenance.root != self.root_state_id()
            || provenance.generation != self.authority.root.0
            || provenance.sequence != work.input.view.sequence()
            || provenance.full_view_digest != work.input.view.digest()
            || provenance.view_seal != work.input.view.seal_digest()
            || provenance.record_scope_digest != work.input.view.record_scope_digest()
            || provenance.policy_digest != work.input.view.policy_digest()
            || !Arc::ptr_eq(bound.view(), &work.input.view)
            || !bound.validates(
                &self.engine.semantic.stores,
                self.root_state_id(),
                self.authority.root.0,
                work.input.view.seal_digest(),
            )
        {
            return Err(AdaptiveNnuecPreparationError::StateMismatch);
        }
        Ok(())
    }
}
