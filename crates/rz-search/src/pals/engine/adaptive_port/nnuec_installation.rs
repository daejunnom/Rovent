//! Per-question installation attestation. Numeric worker IDs and copied
//! profiles cannot substitute for the same live owner's private receipt.
use super::nnuec_port::{AdaptiveNnuecPreparationError, AdaptiveNnuecReviewSeal};
use super::*;
use crate::cpu_value::nnuec::v2::Model;
use crate::cpu_value::nnuec::v2::review::BoundNnuecReviewContext;
use crate::cpu_worker::{
    CPU_JOB_CAPACITY, CpuWorker, CpuWorkerProfile, PreparedValueInstallation, ValueInstallReceipt,
};

#[derive(Default)]
pub(super) struct NnuecQuestionInstallations {
    entries: HashMap<ContractDigest, InstalledQuestion>,
}

struct InstalledQuestion {
    work: CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    bound: Arc<BoundNnuecReviewContext>,
    receipt: ValueInstallReceipt,
    profile: CpuWorkerProfile,
}
impl InstalledQuestion {
    fn matches(&self, work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>) -> bool {
        self.work.frame == work.frame
            && self.work.original_first_move == work.original_first_move
            && self.work.original_deadline == work.original_deadline
            && self.work.input.stage == work.input.stage
            && self.work.input.question_digest == work.input.question_digest
            && self.work.input.view_seal == work.input.view_seal
            && self.work.input.public_context == work.input.public_context
            && Arc::ptr_eq(&self.work.input.question, &work.input.question)
            && Arc::ptr_eq(&self.work.input.view, &work.input.view)
    }
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// Validate the original positive researcher model before accepting a
    /// preparation configuration. No evaluator, actor or worker is created.
    pub fn validate_cpu_nnuec_model(
        &self,
        model: &Model,
    ) -> Result<(), AdaptiveNnuecPreparationError> {
        self.check_live()?;
        self.engine.validate_checker_namespace()?;
        if self.engine.researcher.cpu_registered_value.as_ref() != Some(model.identity())
            || self.engine.owned_identity()? != model.identity()
        {
            return Err(AdaptiveNnuecPreparationError::StateMismatch);
        }
        Ok(())
    }

    /// Admission is tied to this original Q and the current receipt authority
    /// of this exact live worker. It does not change engine-global metadata.
    #[allow(clippy::too_many_arguments)]
    pub fn accept_cpu_nnuec_installation(
        &mut self,
        original: &AdaptiveNnuecReviewSeal,
        model: Arc<Model>,
        bound: Arc<BoundNnuecReviewContext>,
        receipt: &ValueInstallReceipt,
        worker: &CpuWorker,
    ) -> Result<(), AdaptiveNnuecPreparationError> {
        self.validate_cpu_nnuec_model(&model)?;
        self.validate_nnuec_review_completion(original, &model, &bound)?;
        let work = original.work();
        let profile = worker.acknowledged_profile()?;
        let payload = PreparedValueInstallation::from_bound(
            model,
            Arc::clone(&bound),
            &self.engine.semantic.stores,
            work.input.view.seal_digest(),
        )?;
        let (_, conditions) = self.cpu_namespace_digests()?;
        let descriptor = self.engine.owned_descriptor()?;
        if !worker.validates_installation_receipt(receipt)
            || !receipt.matches_bound(&bound)
            || !receipt.is_installed()
            || receipt.installed_profile() != Some(profile)
            || receipt.binding_digest() != payload.binding_digest()
            || profile.evaluation_namespace != payload.binding_digest()
            || profile.search_conditions != conditions
            || profile.max_depth != descriptor.config.max_depth
            || !bound.authority_is_live()
        {
            return Err(AdaptiveNnuecPreparationError::StateMismatch);
        }
        if let Some(previous) = self
            .nnuec_installations
            .entries
            .get(&work.input.question_digest)
        {
            return if previous.matches(work)
                && Arc::ptr_eq(&previous.bound, &bound)
                && previous.profile == profile
                && previous.receipt.installation_id() == receipt.installation_id()
                && previous.receipt.binding_digest() == receipt.binding_digest()
                && worker.validates_installation_receipt(&previous.receipt)
            {
                Ok(())
            } else {
                Err(AdaptiveNnuecPreparationError::StateMismatch)
            };
        }
        if self.nnuec_installations.entries.len() >= CPU_JOB_CAPACITY {
            return Err(PalsError::Capacity.into());
        }
        if self
            .cpu_admissions
            .contains_key(&work.input.question_digest)
        {
            return Err(AdaptiveNnuecPreparationError::StateMismatch);
        }
        self.nnuec_installations.entries.insert(
            work.input.question_digest,
            InstalledQuestion {
                work: work.clone(),
                bound,
                receipt: receipt.clone(),
                profile,
            },
        );
        Ok(())
    }

    /// Frozen Q-specific namespace for CPU Job/Task admission. The caller still
    /// validates original Rules/work and dispatches only this selected owner.
    pub(super) fn installed_cpu_binding(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<Option<CpuWorkerProfile>, PalsError> {
        let Some(installed) = self
            .nnuec_installations
            .entries
            .get(&work.input.question_digest)
        else {
            return Ok(None);
        };
        if !installed.matches(work)
            || !installed.bound.authority_is_live()
            || !installed.bound.validates(
                &self.engine.semantic.stores,
                self.root_state_id(),
                self.authority.root.0,
                work.input.view.seal_digest(),
            )
            || installed.receipt.installed_profile() != Some(installed.profile)
            || !installed.receipt.matches_bound(&installed.bound)
            || installed.receipt.binding_digest() != installed.profile.evaluation_namespace
        {
            return Err(StoreError::InvalidConditions(
                "CPU question differs from original installed NNUEC view",
            )
            .into());
        }
        Ok(Some(installed.profile))
    }

    /// Drop only this port's extra immutable preparation custody. The actual
    /// worker may retain its evaluator/Bound until replacement or shutdown;
    /// this does not clear its TT, release a token or refund native credit.
    pub fn release_cpu_nnuec_installation(
        &mut self,
        original: &AdaptiveNnuecReviewSeal,
    ) -> Result<(), AdaptiveNnuecPreparationError> {
        let work = original.work();
        if work.frame.authority != self.authority
            || work.original_deadline != self.deadline
            || original.view().root_state() != self.root_state_id()
            || !original.view().validates(&self.engine.semantic.stores)
            || self.engine.researcher.cpu_registered_value.as_ref() != Some(original.model())
        {
            return Err(AdaptiveNnuecPreparationError::StateMismatch);
        }
        let Some(installed) = self
            .nnuec_installations
            .entries
            .get(&work.input.question_digest)
        else {
            return Ok(());
        };
        if !installed.matches(work)
            || self
                .cpu_admissions
                .get(&work.input.question_digest)
                .is_some_and(|admission| {
                    !admission.terminal
                        || admission.quarantined
                        || admission.grant.is_some()
                        || admission.continuation_retained
                })
        {
            return Err(AdaptiveNnuecPreparationError::StateMismatch);
        }
        self.nnuec_installations
            .entries
            .remove(&work.input.question_digest);
        Ok(())
    }
}
