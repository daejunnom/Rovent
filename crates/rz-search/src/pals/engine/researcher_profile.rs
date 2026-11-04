//! Opt-in original-root sequence limits for a registered researcher.
//!
//! Generation chunks keep legacy PalsConfig admission and codec bytes. This
//! profile changes only bounded exact semantic sequences and their consumers.
use super::researcher_admission::ResearcherAdmission;
use super::*;

#[derive(Clone, Debug)]
pub struct ResearcherSemanticProfile {
    pub config: PalsConfig,
    pub maximum_root_prefix_plies: u16,
}
impl ResearcherSemanticProfile {
    pub fn validate(&self) -> Result<(), PalsError> {
        self.config.validate()?;
        self.semantic_profile().validate()
    }
    pub fn semantic_profile(&self) -> RoventSemanticProfile {
        RoventSemanticProfile {
            semantic: PalsSemanticConfig::from(&self.config),
            maximum_root_prefix_plies: self.maximum_root_prefix_plies,
        }
    }
    pub(super) fn sequence_config(&self) -> PalsSemanticConfig {
        self.semantic_profile().sequence_config()
    }
}

impl<M: RoleModel> PalsEngine<M> {
    pub fn new_with_boxed_checker_and_rovent_profile(
        profile: ResearcherSemanticProfile,
        model: M,
        cpu: Box<dyn CpuChecker>,
        resolver: ResolverPolicy,
        policy: PostRepairRecheckPolicy,
    ) -> Result<Self, PalsError> {
        profile.validate()?;
        let mut admission =
            Self::admit_researcher(&profile.config, &model, cpu.as_ref(), resolver, policy)?;
        let semantic = PalsSemanticState::new_researcher_admitted(
            &profile,
            model,
            admission.model_registered_value.take(),
        )?;
        Ok(Self::from_admitted_researcher(
            profile.config,
            cpu,
            resolver,
            policy,
            Some(profile.maximum_root_prefix_plies),
            admission,
            semantic,
        ))
    }

    /// A source-pinned external declaration may support full legal root
    /// questions without supporting searchmoves. The actual external worker
    /// still rejects a restricted question; legacy admission remains strict.
    pub fn new_with_boxed_profiled_researcher_and_rovent_profile(
        profile: ResearcherSemanticProfile,
        model: M,
        cpu: Box<dyn CpuChecker>,
        resolver: ResolverPolicy,
        policy: PostRepairRecheckPolicy,
        declared: &crate::external_cpu::ProfiledResearcherAdmission,
    ) -> Result<Self, PalsError> {
        profile.validate()?;
        let mut admission = Self::admit_profiled_researcher(
            &profile.config,
            &model,
            cpu.as_ref(),
            resolver,
            policy,
            declared,
        )?;
        let semantic = PalsSemanticState::new_researcher_admitted(
            &profile,
            model,
            admission.model_registered_value.take(),
        )?;
        Ok(Self::from_admitted_researcher(
            profile.config,
            cpu,
            resolver,
            policy,
            Some(profile.maximum_root_prefix_plies),
            admission,
            semantic,
        ))
    }

    /// The caller has already checked model/checker admission in its original
    /// order. Moving a prepared semantic owner here creates no second Store.
    pub(super) fn from_admitted_researcher(
        config: PalsConfig,
        cpu: Box<dyn CpuChecker>,
        resolver: ResolverPolicy,
        policy: PostRepairRecheckPolicy,
        rovent_root_prefix_plies: Option<u16>,
        admission: ResearcherAdmission,
        semantic: PalsSemanticState<M>,
    ) -> Self {
        let researcher = RegisteredResearcher::from_admitted(
            cpu,
            admission.checker_registered_identity,
            admission.cpu_registered_value,
            admission.cpu_registered_condition,
        );
        Self {
            config,
            resolver,
            followup_lane: true,
            post_repair_recheck: policy,
            cpu_interleave: CpuInterleavePolicy::ReplaceOnFresh,
            held_cpu_task: None,
            rovent_root_prefix_plies,
            deferred_researcher_source: None,
            deferred_researcher_initial: None,
            researcher,
            semantic,
            last_cpu_checkpoint_cleanup_error: None,
            followup_execution: followup_execution::FollowupExecutionRecorder::default(),
        }
    }

    pub fn maximum_root_prefix_plies(&self) -> usize {
        self.rovent_root_prefix_plies
            .map(usize::from)
            .unwrap_or(self.config.line_plies)
    }

    /// Only an actual Ready handoff can install this declaration. It holds
    /// neither a Bound nor a view pin and supplies no worker/cleanup authority.
    pub fn deferred_researcher_source(&self) -> Option<&crate::cpu_worker::DeferredNnuecRSource> {
        self.deferred_researcher_source.as_ref()
    }

    pub fn rovent_semantic_profile(&self) -> Option<RoventSemanticProfile> {
        self.rovent_root_prefix_plies
            .map(|maximum_root_prefix_plies| RoventSemanticProfile {
                semantic: PalsSemanticConfig::from(&self.config),
                maximum_root_prefix_plies,
            })
    }

    pub(super) fn semantic_sequence_config(&self) -> PalsSemanticConfig {
        PalsSemanticConfig {
            line_plies: self.maximum_root_prefix_plies(),
            ..PalsSemanticConfig::from(&self.config)
        }
    }
}

#[cfg(test)]
#[path = "researcher_profile_tests.rs"]
mod tests;
