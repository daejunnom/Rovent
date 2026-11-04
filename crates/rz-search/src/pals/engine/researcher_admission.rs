//! Admission-only researcher registration reused by old and Rovent owners.
//! Checks keep their original order; this module allocates no engine or Store.
use super::*;

pub(super) struct ResearcherAdmission {
    pub(super) model_registered_value: Option<ModelValueIdentity>,
    pub(super) checker_registered_identity: CheckerIdentity,
    pub(super) cpu_registered_value: Option<CpuValueIdentity>,
    pub(super) cpu_registered_condition: String,
}

impl<M: RoleModel> PalsEngine<M> {
    pub(super) fn admit_researcher(
        config: &PalsConfig,
        model: &M,
        cpu: &dyn CpuChecker,
        resolver: ResolverPolicy,
        policy: PostRepairRecheckPolicy,
    ) -> Result<ResearcherAdmission, PalsError> {
        Self::admit_researcher_with_profile(config, model, cpu, resolver, policy, None)
    }

    pub(super) fn admit_profiled_researcher(
        config: &PalsConfig,
        model: &M,
        cpu: &dyn CpuChecker,
        resolver: ResolverPolicy,
        policy: PostRepairRecheckPolicy,
        admission: &crate::external_cpu::ProfiledResearcherAdmission,
    ) -> Result<ResearcherAdmission, PalsError> {
        Self::admit_researcher_with_profile(config, model, cpu, resolver, policy, Some(admission))
    }

    fn admit_researcher_with_profile(
        config: &PalsConfig,
        model: &M,
        cpu: &dyn CpuChecker,
        resolver: ResolverPolicy,
        policy: PostRepairRecheckPolicy,
        profiled: Option<&crate::external_cpu::ProfiledResearcherAdmission>,
    ) -> Result<ResearcherAdmission, PalsError> {
        config.validate()?;
        cpu.identity().validate()?;
        if let Some(admission) = profiled {
            admission.validate_checker(cpu)?;
        }
        if resolver == ResolverPolicy::OwnRawRestricted
            && matches!(cpu.identity(), CheckerIdentity::ExternalUci(_))
        {
            return Err(CpuError::Unsupported("foreign checker with own raw resolver").into());
        }
        if policy != PostRepairRecheckPolicy::Disabled
            && !policy.uses_frozen_model_wdl()
            && matches!(cpu.identity(), CheckerIdentity::ExternalUci(_))
        {
            return Err(CpuError::Unsupported(
                "post-Repair recheck v1 requires completed own CPU evidence",
            )
            .into());
        }
        let capabilities = cpu.capabilities();
        if !capabilities.divergence || (!capabilities.root_moves && profiled.is_none()) {
            return Err(CpuError::Unsupported(
                "PALS independent CPU divergence/root-move discovery",
            )
            .into());
        }
        if capabilities.max_depth == 0
            || capabilities.max_depth > 64
            || capabilities.max_prefix_plies < config.line_plies.saturating_sub(1)
            || (capabilities.max_root_moves < 256
                && !(profiled.is_some()
                    && !capabilities.root_moves
                    && capabilities.max_root_moves == 0))
            || cpu.conditions().is_empty()
            || cpu.conditions().len()
                > if matches!(cpu.identity(), CheckerIdentity::ExternalUci(_)) {
                    8192
                } else {
                    2048
                }
            || cpu.conditions().chars().any(char::is_control)
        {
            return Err(CpuError::Unsupported("PALS CPU bounded search capabilities").into());
        }
        let (cpu_registered_value, model_registered_value) = match cpu.identity() {
            CheckerIdentity::Owned(identity) => {
                let descriptor = cpu
                    .owned_descriptor()
                    .ok_or(CheckerError::Invalid("owned configuration missing"))?;
                if descriptor.config.max_depth == 0
                    || descriptor.config.max_depth > 64
                    || descriptor.config.quiescence_ply > 32
                    || capabilities.max_depth < descriptor.config.max_depth
                    || descriptor.search_identity.is_empty()
                {
                    return Err(CpuError::Unsupported("PALS own CPU configuration").into());
                }
                (Some(identity.clone()), None)
            }
            CheckerIdentity::ExternalUci(_) => {
                if cpu.owned_descriptor().is_some() || capabilities.resume {
                    return Err(CheckerError::Invalid(
                        "foreign helper cannot disclose own/resume semantics",
                    )
                    .into());
                }
                let identity = model
                    .value_identity()
                    .ok_or(RoleError::Unavailable)?
                    .clone();
                identity.validate()?;
                if identity.model != model.identity()
                    || identity.semantics != MODEL_WDL_VALUE_SEMANTICS
                {
                    return Err(RoleError::InvalidOutput.into());
                }
                (None, Some(identity))
            }
        };
        let model_registered_value = if resolver == ResolverPolicy::ModelWdlRestricted
            || policy.uses_frozen_model_wdl()
        {
            let identity = model
                .value_identity()
                .ok_or(RoleError::Unavailable)?
                .clone();
            identity.validate()?;
            if identity.model != model.identity() || identity.semantics != MODEL_WDL_VALUE_SEMANTICS
            {
                return Err(RoleError::InvalidOutput.into());
            }
            Some(identity)
        } else {
            model_registered_value
        };
        let checker_registered_identity = cpu.identity().clone();
        let cpu_registered_condition = Self::cpu_condition_for(cpu);
        Ok(ResearcherAdmission {
            model_registered_value,
            checker_registered_identity,
            cpu_registered_value,
            cpu_registered_condition,
        })
    }
}
