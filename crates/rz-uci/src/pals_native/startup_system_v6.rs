//! Source-owned PALS system-CUDA model input boundary.
//!
//! Admission checks the actual retained descriptors and model declarations;
//! it does not load ORT, prove numerical equivalence or authorize a game.
use crate::pals_adaptive_startup::{PalsSystemSourceObservationV6, PinnedPalsSystemStartupV6};
use crate::pals_native::startup_virtual::{exact_graph_snapshot_join, graph_purpose};
use crate::pals_native::virtual_encoding::{
    VirtualRulesDescriptorSummary, observe_pals_virtual_rules_descriptor_bytes,
    validate_pals_virtual_rules_descriptor_bytes,
};
use rz_eval::asset::parse_sha256;
use rz_eval::error::{BackendError, FailureKind, FailureStage};
use rz_eval::pals_model::{PalsModelProfile, virtual_review};
use rz_eval::pals_onnx::virtual_review::{
    VirtualReviewArtifactKind, VirtualReviewExecutionMode, VirtualReviewOwnedSources,
    validate_virtual_review_export_manifest_kind,
};
use rz_experiments::{
    PalsArtifactDomainV5, PalsEngineModeV5, PalsStartupAssetPurposeV5 as Purpose,
    PalsSystemSourcePolicyV6, compiled_source_components_v5,
};
use std::{
    sync::{Arc, atomic::AtomicBool},
    time::Instant,
};

fn error(detail: &'static str) -> BackendError {
    BackendError::new(
        FailureKind::IdentityMismatch,
        FailureStage::Admission,
        detail,
    )
}
fn transport(_: rz_experiments::ManifestError) -> BackendError {
    error("retained PALS system model asset failed verification")
}
fn hex(digest: [u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Source and numeric observations are separate from functional admission.
/// The consumer must retain these findings rather than silently treating
/// observational source drift as verified model compatibility.
#[must_use]
pub(crate) struct PalsSystemVirtualObservationV6 {
    pub rules: VirtualRulesDescriptorSummary,
    pub declared_model_source: String,
    pub declared_component_source: Option<String>,
    pub compiled_model_source: Option<String>,
    pub model_source_evidence: PalsSystemSourceObservationV6,
    pub numerical_acceptance: Option<[u8; 32]>,
}

/// Retains the real startup owner, same startup deadline and cancellation for
/// every source read. The backend must retain this Arc for its full lifetime.
pub(crate) struct PalsSystemVirtualSourcesV6 {
    owner: Arc<PinnedPalsSystemStartupV6>,
    until: Instant,
    cancelled: Arc<AtomicBool>,
}
impl PalsSystemVirtualSourcesV6 {
    pub(crate) fn new(
        owner: Arc<PinnedPalsSystemStartupV6>,
        until: Instant,
        cancelled: Arc<AtomicBool>,
    ) -> Self {
        Self {
            owner,
            until,
            cancelled,
        }
    }
    fn read(&self, purpose: Purpose, maximum: usize) -> Result<Vec<u8>, BackendError> {
        self.owner
            .read_asset_exact(purpose, maximum, self.until, &self.cancelled)
            .map_err(transport)
    }
    /// Validate model/export/Rules declarations against the original held
    /// bytes. Provenance, learned compatibility and independent numerical
    /// acceptance remain separate evidence and are not granted here.
    pub(crate) fn inspect(&self) -> Result<PalsSystemVirtualObservationV6, BackendError> {
        self.owner
            .verify_unchanged(self.until, &self.cancelled)
            .map_err(transport)?;
        let config = &self.owner.lock().config;
        let model = &config.endpoint.model;
        if model.profile != virtual_review::PROFILE
            || model.semantic_sha256
                != hex(PalsModelProfile::VirtualReviewV1.registered_semantics_sha256())
            || model.encoding_sha256 != hex(virtual_review::registered_encoding_sha256())
        {
            return Err(error("PALS system model or encoding declaration differs"));
        }
        let compiled = compiled_source_components_v5().map_err(transport)?;
        let compiled_model_source = compiled.get("model_native").cloned();
        let declared_component_source = config.source_components.get("model_native").cloned();
        let model_source_evidence = match (&compiled_model_source, &declared_component_source) {
            (Some(actual), Some(declared))
                if actual == declared && actual == &model.implementation_sha256 =>
            {
                PalsSystemSourceObservationV6::Verified
            }
            (None, _) | (_, None) => PalsSystemSourceObservationV6::Missing,
            _ => PalsSystemSourceObservationV6::Mismatch,
        };
        if config.source_policy == PalsSystemSourcePolicyV6::Strict
            && model_source_evidence != PalsSystemSourceObservationV6::Verified
        {
            return Err(error("strict PALS system model source is not verified"));
        }
        let mode = match config.endpoint.mode {
            PalsEngineModeV5::Deployment => VirtualReviewExecutionMode::Deployment,
            PalsEngineModeV5::Collection => VirtualReviewExecutionMode::Collection,
        };
        let kind = if config.native_options.approximate_warm {
            VirtualReviewArtifactKind::ApproxWarm
        } else {
            VirtualReviewArtifactKind::Fresh
        };
        let manifest_raw = self.read(Purpose::ExportManifest, 256 * 1024)?;
        let manifest = validate_virtual_review_export_manifest_kind(&manifest_raw, mode, kind)?;
        let rules_raw = self.read(Purpose::RulesDescriptor, 64 * 1024)?;
        let rules = match config.source_policy {
            PalsSystemSourcePolicyV6::Strict => {
                validate_pals_virtual_rules_descriptor_bytes(&rules_raw)?
            }
            PalsSystemSourcePolicyV6::Observe => {
                observe_pals_virtual_rules_descriptor_bytes(&rules_raw)?
            }
        };
        if manifest.checkpoint_sha256 != parse_sha256(&model.weights.sha256)?
            || manifest.model_profile != PalsModelProfile::VirtualReviewV1
            || manifest.validator_present != (config.endpoint.mode == PalsEngineModeV5::Collection)
            || manifest.config != rules.config
            || manifest.rules_input_semantic_sha256 != rules.semantic_sha256
            || manifest.rules_encoder_source_sha256 != rules.encoder_source_sha256
            || manifest.rules_profile_descriptor_sha256 != rules.descriptor_sha256()
        {
            return Err(error(
                "PALS system export, weights or Rules descriptor differs",
            ));
        }
        match model.artifact_domain {
            PalsArtifactDomainV5::Untrained if manifest.trained || manifest.training_steps != 0 => {
                return Err(error("untrained declaration contains trained export"));
            }
            PalsArtifactDomainV5::RealGeneration
                if !manifest.trained || manifest.training_steps == 0 =>
            {
                return Err(error("generation export does not match registered recipe"));
            }
            PalsArtifactDomainV5::FrozenPcCollection => {
                return Err(error("V6 frozen collection content admission unsupported"));
            }
            _ => {}
        }
        let manifest_asset = self
            .owner
            .asset(Purpose::ExportManifest)
            .map_err(transport)?;
        for graph in &manifest.graphs {
            let asset = self
                .owner
                .asset(graph_purpose(&graph.role)?)
                .map_err(transport)?;
            exact_graph_snapshot_join(
                &manifest_asset.declaration().snapshot_path,
                &asset.declaration().snapshot_path,
                &graph.file,
            )?;
            if parse_sha256(&asset.declaration().artifact.sha256)? != graph.sha256 {
                return Err(error("PALS system graph declaration differs from export"));
            }
        }
        Ok(PalsSystemVirtualObservationV6 {
            rules,
            declared_model_source: model.implementation_sha256.clone(),
            declared_component_source,
            compiled_model_source,
            model_source_evidence,
            numerical_acceptance: None,
        })
    }
}
impl VirtualReviewOwnedSources for PalsSystemVirtualSourcesV6 {
    fn read_manifest(&self, maximum: usize) -> Result<Vec<u8>, BackendError> {
        self.read(Purpose::ExportManifest, maximum)
    }
    fn read_graph(
        &self,
        role: &str,
        registered_file: &str,
        expected: [u8; 32],
        maximum: usize,
    ) -> Result<Vec<u8>, BackendError> {
        let asset = self.owner.asset(graph_purpose(role)?).map_err(transport)?;
        let manifest = self
            .owner
            .asset(Purpose::ExportManifest)
            .map_err(transport)?;
        exact_graph_snapshot_join(
            &manifest.declaration().snapshot_path,
            &asset.declaration().snapshot_path,
            registered_file,
        )?;
        if parse_sha256(&asset.declaration().artifact.sha256)? != expected {
            return Err(error("PALS system graph differs from original export"));
        }
        self.read(graph_purpose(role)?, maximum)
    }
}
