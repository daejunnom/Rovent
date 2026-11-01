//! Actual post-prepare model inputs for numerical-only admission.
//!
//! A first G0 initializer cannot supply hashes of files it has not created yet.
//! This scope is constructed only after those files exist. It carries no CPU
//! endpoint, startup lock, process readiness or authority to issue G1/Frozen
//! assets. The completed startup validator remains a separate consumer gate.
use crate::{
    ManifestError, PalsAdaptiveStartupConfigV5, PalsArtifactDomainV5, PalsEngineModeV5,
    PalsModelBackendV3, PalsModelSpecificationV5, PalsStartupAssetPurposeV5 as Purpose,
    PalsStartupAssetV5, compiled_source_components_v5, pals_private_snapshot_path_v5,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const NUMERICAL_COMPONENTS: [&str; 5] = [
    "model_native",
    "model_encoding",
    "nn_runtime",
    "model_reference",
    "model_export",
];

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsNumericalScopeV5 {
    pub mode: PalsEngineModeV5,
    pub model: PalsModelSpecificationV5,
    pub source_commit: String,
    pub source_components: BTreeMap<String, String>,
    pub assets: Vec<PalsStartupAssetV5>,
    pub training_executed: bool,
}

impl PalsNumericalScopeV5 {
    /// Exact projection of an already registered completed startup. This does
    /// not validate that startup or manufacture its missing endpoint/lock.
    pub fn from_config(config: &PalsAdaptiveStartupConfigV5) -> Self {
        Self {
            mode: config.endpoint.mode,
            model: config.endpoint.model.clone(),
            source_commit: config.endpoint.source_commit.clone(),
            source_components: config.source_components.clone(),
            assets: config.assets.clone(),
            training_executed: config.training_executed,
        }
    }

    pub fn asset(&self, purpose: Purpose) -> Result<&PalsStartupAssetV5, ManifestError> {
        let mut found = self.assets.iter().filter(|asset| asset.purpose == purpose);
        let first = found
            .next()
            .ok_or_else(|| fail("numerical asset purpose missing"))?;
        if found.next().is_some() {
            return Err(fail("numerical asset purpose ambiguous"));
        }
        Ok(first)
    }

    /// Validate actual producer/model/runtime inputs before an acceptance file
    /// exists. Raw captures and the acceptance may subsequently be added with
    /// their actual identities; no placeholder future digest is required.
    pub fn validate_pre_acceptance(&self) -> Result<(), ManifestError> {
        self.model.validate(self.mode)?;
        if self.training_executed
            || !lower_hex(&self.source_commit, 40)
            || self.assets.is_empty()
            || self.assets.len() > 96
            || self.source_components.len() > 128
        {
            return Err(fail(
                "numerical scope requires non-training actual source and bounded assets",
            ));
        }
        let compiled = compiled_source_components_v5()?;
        for key in NUMERICAL_COMPONENTS {
            let pin = self
                .source_components
                .get(key)
                .ok_or_else(|| fail("numerical source component missing"))?;
            if !lower_hex(pin, 64) || compiled.get(key) != Some(pin) {
                return Err(fail(
                    "numerical source component differs from compiled source",
                ));
            }
        }
        if self
            .source_components
            .iter()
            .any(|(key, pin)| compiled.get(key) != Some(pin))
        {
            return Err(fail(
                "numerical scope contains an unverified extra source component",
            ));
        }
        let mut snapshots = BTreeSet::new();
        let mut paths = BTreeSet::new();
        let mut total = 0u64;
        let mut purposes = BTreeSet::new();
        for asset in &self.assets {
            asset.artifact.validate()?;
            total = total
                .checked_add(asset.artifact.bytes)
                .ok_or_else(|| fail("numerical asset bytes overflow"))?;
            if !pals_private_snapshot_path_v5(&asset.snapshot_path)
                || !snapshots.insert(&asset.snapshot_path)
                || !paths.insert(&asset.artifact.path)
                || (!matches!(
                    asset.purpose,
                    Purpose::CudaLibrary | Purpose::NumericalRawEvidence
                ) && !purposes.insert(asset.purpose))
            {
                return Err(fail("numerical asset has unsafe or ambiguous identity"));
            }
        }
        for purpose in [
            Purpose::Weights,
            Purpose::ExportManifest,
            Purpose::RuntimeLibrary,
            Purpose::RulesDescriptor,
        ] {
            self.asset(purpose)?;
        }
        if self.asset(Purpose::Weights)?.artifact != self.model.weights {
            return Err(fail(
                "numerical weights differ from actual model specification",
            ));
        }
        let count = |purpose| self.assets.iter().filter(|a| a.purpose == purpose).count();
        let collection = self.mode == PalsEngineModeV5::Collection;
        let shared = count(Purpose::PublicGraph) == 1
            && count(Purpose::SharedPcGraph) == 1
            && count(Purpose::ProposerGraph) == 0
            && count(Purpose::CriticGraph) == 0;
        let separate = count(Purpose::PublicGraph) == 1
            && count(Purpose::SharedPcGraph) == 0
            && count(Purpose::ProposerGraph) == 1
            && count(Purpose::CriticGraph) == 1;
        if (!shared && (self.model.profile == "virtual_review_v1" || !separate))
            || count(Purpose::ValidatorGraph) != usize::from(collection)
        {
            return Err(fail(
                "numerical graph roster differs from actual model capability",
            ));
        }
        let cuda = self.model.backend == PalsModelBackendV3::OrtCuda;
        if count(Purpose::CudaControlInventory) != usize::from(cuda)
            || count(Purpose::CudaManifest) != usize::from(cuda)
            || (cuda && count(Purpose::CudaLibrary) == 0)
            || (!cuda && count(Purpose::CudaLibrary) != 0)
        {
            return Err(fail(
                "numerical native provider inventory differs from actual backend",
            ));
        }
        match self.model.artifact_domain {
            PalsArtifactDomainV5::Untrained => {
                self.asset(Purpose::InitializationEvidence)?;
                self.asset(Purpose::InitializationCheckpointMetadata)?;
                if count(Purpose::GenerationSpecification) != 0 {
                    return Err(fail("G0 cannot carry a generation specification"));
                }
            }
            PalsArtifactDomainV5::RealGeneration | PalsArtifactDomainV5::FrozenPcCollection => {
                self.asset(Purpose::GenerationSpecification)?;
                if count(Purpose::InitializationEvidence) != 0
                    || count(Purpose::InitializationCheckpointMetadata) != 0
                {
                    return Err(fail(
                        "trained metadata cannot relabel G0 initialization evidence",
                    ));
                }
            }
            PalsArtifactDomainV5::Diagnostic => {
                return Err(fail(
                    "diagnostic assets cannot become actual numerical candidates",
                ));
            }
        }
        match self.model.provenance.as_ref() {
            Some(pin) if self.asset(Purpose::Provenance)?.artifact == *pin => {}
            None if count(Purpose::Provenance) == 0 => {}
            _ => {
                return Err(fail(
                    "numerical provenance differs from the actual model domain",
                ));
            }
        }
        Ok(())
    }
}

fn lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}
fn fail(message: &str) -> ManifestError {
    ManifestError::Integrity(message.into())
}
