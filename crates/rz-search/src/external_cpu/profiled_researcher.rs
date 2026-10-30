//! Opaque startup declaration from a concrete, purpose-checked external owner.
//!
//! This descriptor identifies a research checker configuration. It grants no
//! Ready, dispatch, execution, physical completion, own bound or Rules proof.
//! Its private constructor cannot be supplied by an arbitrary `CpuChecker`.

use super::{
    CheckerCapabilities, CheckerError, CheckerIdentity, CpuChecker, ExternalEngineProfile,
    ExternalEnginePurpose, ExternalUciCpuChecker,
};

#[derive(Debug)]
pub struct ProfiledResearcherAdmission {
    identity: CheckerIdentity,
    conditions: String,
    capabilities: CheckerCapabilities,
    profile: ExternalEngineProfile,
    source_commit: &'static str,
}

impl ProfiledResearcherAdmission {
    fn from_profiled_checker(checker: &ExternalUciCpuChecker) -> Result<Self, CheckerError> {
        let (profile, purpose) = checker.engine_profile.ok_or(CheckerError::Unsupported(
            "research admission requires an explicitly profiled external checker",
        ))?;
        if purpose != ExternalEnginePurpose::ResearchChecker {
            return Err(CheckerError::Unsupported(
                "research admission rejects teacher and arena purposes",
            ));
        }
        checker.config.validate()?;
        profile.validate_identity(
            purpose,
            &checker.config.identity,
            checker
                .config
                .expected_uci_name
                .as_deref()
                .unwrap_or_default(),
        )?;
        checker.validate_namespace()?;
        let CheckerIdentity::ExternalUci(identity) = checker.identity() else {
            return Err(CheckerError::Invalid(
                "profiled research checker has a foreign identity kind",
            ));
        };
        if identity != &checker.config.identity {
            return Err(CheckerError::Invalid(
                "profiled research checker changed its configured identity",
            ));
        }
        Ok(Self {
            identity: checker.identity().clone(),
            conditions: checker.conditions().to_owned(),
            capabilities: checker.capabilities(),
            profile,
            source_commit: profile.source_commit(),
        })
    }

    /// Match the consumer's live immutable namespace, without starting or
    /// dispatching work. Equality is a configuration check, not evidence that
    /// another implementation owns this descriptor's concrete process.
    pub fn validate_checker(&self, checker: &dyn CpuChecker) -> Result<(), CheckerError> {
        checker.validate_namespace()?;
        if checker.identity() != &self.identity
            || checker.conditions() != self.conditions
            || checker.capabilities() != self.capabilities
        {
            return Err(CheckerError::Invalid(
                "research checker differs from its profiled admission",
            ));
        }
        Ok(())
    }

    pub fn capabilities(&self) -> CheckerCapabilities {
        self.capabilities
    }

    pub fn identity(&self) -> &CheckerIdentity {
        &self.identity
    }

    pub fn conditions(&self) -> &str {
        &self.conditions
    }

    pub fn profile(&self) -> ExternalEngineProfile {
        self.profile
    }

    pub fn source_commit(&self) -> &'static str {
        self.source_commit
    }
}

impl ExternalUciCpuChecker {
    /// Available only from the concrete profiled ResearchChecker owner. The
    /// existing `create_profiled` identity/options/config guards are repeated;
    /// this read-only operation produces no startup or physical witness.
    pub fn profiled_researcher_admission(
        &self,
    ) -> Result<ProfiledResearcherAdmission, CheckerError> {
        ProfiledResearcherAdmission::from_profiled_checker(self)
    }
}
