//! Physical profiled-R reset admission. Declaration equality cannot arm this
//! path; only the retained worker's opaque current-source/reset proofs can.

use super::*;
use crate::rovent_external_launch::RoventExternalOwner;
use crate::rovent_root_admission::{RoventRootAdmission, source_seal::RoventRootAdmissionV2};
use crate::search_driver::SearchSessionContext;
use rz_runtime::contracts::ContractClock as _;
use rz_runtime::contracts::ContractSystemClock;
use rz_search::external_cpu::{
    ExternalEnginePurpose, ProfiledExternalResetProof, ProfiledExternalRootSource,
};
use std::sync::Arc;

/// Borrow only the root actually admitted by this physical owner. A /2 launch
/// must retain the same original lock allocation as the admitted /2 root.
enum OriginalRoot<'a> {
    V1(&'a RoventRootAdmission),
    V2(&'a RoventRootAdmissionV2),
}

impl OriginalRoot<'_> {
    fn authority(&self) -> SearchAuthority {
        match self {
            Self::V1(root) => root.authority(),
            Self::V2(root) => root.authority(),
        }
    }
    fn runtime_clock(&self) -> &ContractSystemClock {
        match self {
            Self::V1(root) => root.runtime_clock(),
            Self::V2(root) => root.runtime_clock(),
        }
    }
    fn original_context(&self) -> &SearchSessionContext {
        match self {
            Self::V1(root) => root.original_context(),
            Self::V2(root) => root.original_context(),
        }
    }
}

fn original_root<'a>(
    owner: &'a RoventExternalOwner,
    legacy_missing: &'static str,
) -> Result<OriginalRoot<'a>, CheckerError> {
    if let Some(lock) = owner.launch().original_lock_owner_v2() {
        let root = owner
            .actual_root_admission_v2()
            .ok_or(CheckerError::Unsupported(
                "profiled metadata original /2 root is absent",
            ))?;
        if !root
            .owner()
            .v2()
            .is_some_and(|root_lock| Arc::ptr_eq(lock, root_lock))
        {
            return Err(CheckerError::Invalid(
                "profiled metadata original /2 lock differs",
            ));
        }
        Ok(OriginalRoot::V2(root))
    } else {
        owner
            .actual_root_admission()
            .map(|root| OriginalRoot::V1(root))
            .ok_or(CheckerError::Unsupported(legacy_missing))
    }
}

#[derive(Debug)]
pub(super) struct ProfiledExternalMetadataBinding {
    source: ProfiledExternalRootSource,
    last_reset_sequence: u64,
    last_reset_authority: Option<SearchAuthority>,
}

impl AdaptiveCheckerMetadata {
    /// Bind the same actual process after startup and original-root admission.
    /// Rebinding the next root preserves the consumed reset high-water.
    pub fn bind_profiled_external_owner(
        &self,
        owner: &RoventExternalOwner,
        epoch: ProcessEpoch,
        implementation: Digest,
    ) -> Result<(), CheckerError> {
        let declaration = owner.launch().checker_declaration()?;
        let actual = AdaptiveCheckerMetadata::capture(&declaration)?;
        if self.identity != actual.identity
            || self.conditions != actual.conditions
            || self.capabilities != actual.capabilities
            || self.owned_descriptor.is_some()
            || self.resume_policy.is_some()
            || !matches!(self.identity, CheckerIdentity::ExternalUci(_))
        {
            return Err(CheckerError::Invalid(
                "profiled metadata declaration differs from original source",
            ));
        }
        let startup = owner.verified_startup()?.ok_or(CheckerError::Unsupported(
            "profiled metadata requires actual startup",
        ))?;
        let source = owner
            .worker()
            .current_root_source()?
            .ok_or(CheckerError::Unsupported(
                "profiled metadata current root is pending",
            ))?;
        source.validate_current()?;
        let root = original_root(owner, "profiled metadata original root is absent")?;
        if source.authority() != root.authority()
            || source.authority().epoch != epoch
            || source.authority().implementation != implementation
            || source.purpose() != ExternalEnginePurpose::ResearchChecker
            || source.profile() != owner.launch().profile().engine
            || source.process_identity() != startup.process_identity()
            || !owner.worker().clock_matches(source.clock())
            || source.clock().domain != root.runtime_clock().domain()
            || source.clock().origin != root.runtime_clock().origin()
        {
            return Err(CheckerError::Invalid(
                "profiled metadata actual process/root/clock differs",
            ));
        }
        let mut reset = self
            .reset
            .lock()
            .map_err(|_| CheckerError::Invalid("metadata reset owner poisoned"))?;
        if reset.pending.is_some() || reset.external.is_some() {
            return Err(CheckerError::Invalid(
                "profiled metadata reset or legacy owner already bound",
            ));
        }
        if let Some(binding) = &mut reset.profiled {
            let previous = &binding.source;
            if !previous.same_worker(&source)
                || previous.authority().epoch != source.authority().epoch
                || previous.authority().implementation != source.authority().implementation
                || previous.authority().root.0 > source.authority().root.0
                || previous.authority().game.0 > source.authority().game.0
            {
                return Err(CheckerError::Invalid(
                    "profiled metadata cannot adopt another physical owner",
                ));
            }
            binding.source = source;
        } else {
            reset.profiled = Some(ProfiledExternalMetadataBinding {
                source,
                last_reset_sequence: 0,
                last_reset_authority: None,
            });
        }
        Ok(())
    }

    /// One actual initial/new-game reset grants exactly one semantic delivery.
    /// A same-game next root never manufactures a new reset opportunity.
    pub fn arm_profiled_external_reset(
        &self,
        owner: &RoventExternalOwner,
        proof: &ProfiledExternalResetProof,
        deadline: Instant,
    ) -> Result<(), CheckerError> {
        let source = owner
            .worker()
            .current_root_source()?
            .ok_or(CheckerError::Unsupported(
                "profiled reset current source is pending",
            ))?;
        source.validate_current()?;
        let root = original_root(owner, "profiled reset original root absent")?;
        let observed = owner
            .worker()
            .current_reset_proof()
            .ok_or(CheckerError::Unsupported(
                "profiled actual reset was not observed",
            ))?;
        if !proof.matches_source(&source)
            || !observed.matches_source(&source)
            || proof.sequence() != observed.sequence()
            || proof.authority() != root.authority()
            || proof.sequence() == 0
            || proof.purpose() != ExternalEnginePurpose::ResearchChecker
            || proof.process_identity() != source.process_identity()
            || proof.clock().domain != root.runtime_clock().domain()
            || proof.clock().origin != root.runtime_clock().origin()
            || deadline > root.original_context().control.deadline
        {
            return Err(CheckerError::Invalid(
                "profiled reset proof is foreign or outside original root",
            ));
        }
        let mut reset = self
            .reset
            .lock()
            .map_err(|_| CheckerError::Invalid("metadata reset owner poisoned"))?;
        if reset.pending.is_some() {
            return Err(CheckerError::Invalid("metadata reset already armed"));
        }
        let next_sequence = reset
            .sequence
            .checked_add(1)
            .ok_or(CheckerError::Invalid("metadata reset sequence exhausted"))?;
        let binding = reset.profiled.as_mut().ok_or(CheckerError::Unsupported(
            "profiled metadata actual owner was not bound",
        ))?;
        if !source.same_source(&binding.source)
            || proof.sequence() <= binding.last_reset_sequence
            || match (binding.last_reset_authority, proof.previous_authority()) {
                (None, None) => !proof.is_initial(),
                (Some(old), Some(previous)) => {
                    proof.is_initial()
                        || previous.epoch != old.epoch
                        || previous.implementation != old.implementation
                        || previous.game.0 < old.game.0
                        || previous.root.0 < old.root.0
                        || proof.authority().game.0 <= previous.game.0
                        || proof.authority().root.0 <= previous.root.0
                }
                _ => true,
            }
        {
            return Err(CheckerError::Invalid(
                "profiled reset source/high-water is stale",
            ));
        }
        binding.last_reset_sequence = proof.sequence();
        binding.last_reset_authority = Some(proof.authority());
        reset.sequence = next_sequence;
        reset.pending = Some(MetadataResetArm::ProfiledExternal { deadline });
        Ok(())
    }
}
