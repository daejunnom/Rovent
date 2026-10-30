//! Original /2 intervention FD custody and separately sealed learner input.
//! A historical comparison never reissues past native or optimizer authority.

use super::RoventUtilityInterventionTrainingPreparation;
use super::dataset::UtilityDatasetLineage;
use super::execution::RoventUtilityDatasetReference;
use super::intervention_execution::RoventUtilityInterventionExecutionRegistration;
use super::numerical_parts::ValidatedUtilityNumericalParts;
use crate::ArenaError;
use crate::rovent_collect::{
    RoventHistoricalUtilityInterventionFiles, RoventPublicationFile,
    RoventUtilityCohortAncestorSource,
};
use crate::rovent_training_process::{RoventTrainingFilePin, recovery::RoventRecoveryReadBudget};
use rz_experiments::pals_data_v3::{
    GenerationSplitV3, generation_digest, generation_identifier, generation_sha,
};
use rz_experiments::rovent_utility_intervention_preparation::RoventUtilityInterventionSource;
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::path::{Component, Path};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

pub const ROVENT_UTILITY_INTERVENTION_DATASET: &str = "rz-rovent-utility-training-dataset/2";
pub const ROVENT_UTILITY_INTERVENTION_CONTEXT_SOURCE: &str =
    "rz-rovent-utility-cohort-context-source/2";
pub const ROVENT_UTILITY_INTERVENTION_POOL: &str = "rz-rovent-utility-training-context-pool/2";
fn invalid(reason: impl Into<String>) -> ArenaError {
    ArenaError::Integrity(format!("Rovent intervention dataset: {}", reason.into()))
}
fn check(condition: bool, reason: &str) -> Result<(), ArenaError> {
    if condition {
        Ok(())
    } else {
        Err(invalid(reason))
    }
}
fn digest(domain: &str, value: &impl serde::Serialize) -> Result<String, ArenaError> {
    generation_digest(domain, value).map_err(ArenaError::Manifest)
}
fn relative(root: &Path, pin: &RoventTrainingFilePin) -> Result<RoventPublicationFile, ArenaError> {
    let path = pin
        .path
        .strip_prefix(root)
        .map_err(|_| invalid("original file escapes root"))?;
    check(
        !path.as_os_str().is_empty()
            && path.components().all(|c| matches!(c, Component::Normal(_))),
        "normalized original relative path required",
    )?;
    let path = path.to_str().ok_or_else(|| invalid("original path UTF8"))?;
    check(
        !path.contains('\\'),
        "original path spelling must not be normalized",
    )?;
    let value = RoventPublicationFile {
        path: path.into(),
        sha256: pin.sha256.clone(),
        bytes: pin.bytes,
    };
    value.validate()?;
    Ok(value)
}

/// Failed admission retains all descriptors already admitted by the original
/// reader. The parent must keep this owner with its failed-execution receipt.
pub struct RoventUtilityInterventionDatasetRejected {
    pub error: ArenaError,
    pub original: Option<Arc<RoventHistoricalUtilityInterventionFiles>>,
}
impl std::fmt::Debug for RoventUtilityInterventionDatasetRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoventUtilityInterventionDatasetRejected")
            .field("error", &self.error)
            .field("original_retained", &self.original.is_some())
            .finish()
    }
}
impl std::fmt::Display for RoventUtilityInterventionDatasetRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}
impl std::error::Error for RoventUtilityInterventionDatasetRejected {}

/// Only the original /2 owner plus exact source13 join can construct this.
pub(super) struct UtilityInterventionContextAuthority {
    sha256: String,
    source_projection: Value,
    known_pair_sha256: Vec<String>,
    unique_known_pairs: u64,
}
impl UtilityInterventionContextAuthority {
    pub(super) fn sha256(&self) -> &str {
        &self.sha256
    }
    pub(super) fn source_projection(&self) -> &Value {
        &self.source_projection
    }
    pub(super) fn unique_known_pairs(&self) -> u64 {
        self.unique_known_pairs
    }
    pub(super) fn pair_sha256s(&self) -> &[String] {
        &self.known_pair_sha256
    }
}

pub struct VerifiedUtilityInterventionDataset {
    reference: RoventUtilityDatasetReference,
    original: Arc<RoventHistoricalUtilityInterventionFiles>,
    pins: Vec<RoventTrainingFilePin>,
    context: UtilityInterventionContextAuthority,
    context_ids: Vec<String>,
    dataset_sha256: String,
    pool_sha256: String,
    preparation_sha256: String,
    ancestry_checked: AtomicBool,
}
impl VerifiedUtilityInterventionDataset {
    pub fn open_retaining(
        reference: &RoventUtilityDatasetReference,
        preparation: &Value,
        budget: &Arc<RoventRecoveryReadBudget>,
    ) -> Result<Self, Box<RoventUtilityInterventionDatasetRejected>> {
        let before = (|| {
            reference.validate()?;
            let preparation: RoventUtilityInterventionTrainingPreparation =
                serde_json::from_value(preparation.clone()).map_err(|e| invalid(e.to_string()))?;
            preparation.validate().map_err(ArenaError::Manifest)?;
            Ok((
                preparation,
                relative(&reference.root, &reference.publication)?,
                relative(&reference.root, &reference.durable_ack)?,
            ))
        })();
        let (preparation, publication, ack) = before.map_err(|error| {
            Box::new(RoventUtilityInterventionDatasetRejected {
                error,
                original: None,
            })
        })?;
        let original = match RoventHistoricalUtilityInterventionFiles::open_retaining(
            reference.root.clone(),
            publication,
            ack,
            Arc::clone(budget),
        ) {
            Ok(owner) => Arc::new(owner),
            Err(rejected) => {
                let rejected = *rejected;
                return Err(Box::new(RoventUtilityInterventionDatasetRejected {
                    error: rejected.error,
                    original: Some(Arc::new(rejected.original)),
                }));
            }
        };
        Self::from_original(reference, &preparation, Arc::clone(&original), budget).map_err(
            |error| {
                Box::new(RoventUtilityInterventionDatasetRejected {
                    error,
                    original: Some(original),
                })
            },
        )
    }
    /// Named alias keeps the retaining error type; no error conversion drops
    /// original descriptors while pretending the /2 admission simply vanished.
    pub fn open(
        reference: &RoventUtilityDatasetReference,
        preparation: &Value,
        budget: &Arc<RoventRecoveryReadBudget>,
    ) -> Result<Self, Box<RoventUtilityInterventionDatasetRejected>> {
        Self::open_retaining(reference, preparation, budget)
    }
    fn from_original(
        reference: &RoventUtilityDatasetReference,
        preparation: &RoventUtilityInterventionTrainingPreparation,
        original: Arc<RoventHistoricalUtilityInterventionFiles>,
        budget: &RoventRecoveryReadBudget,
    ) -> Result<Self, ArenaError> {
        let publication = original
            .publication()
            .ok_or_else(|| invalid("checked original publication absent"))?;
        let evidence = original
            .evidence()
            .ok_or_else(|| invalid("checked original /2 evidence absent"))?;
        let parts = ValidatedUtilityNumericalParts::from_intervention_typed(preparation)?;
        let registration = parts.registration();
        check(
            evidence.usage=="collection_candidate" && evidence.split==GenerationSplitV3::Train
            && [&evidence.opening_id,&evidence.game_id,&evidence.line_genealogy_id].into_iter().all(|id|generation_identifier(id))
            && [&evidence.transposition_sha256,&evidence.rules_state_sha256].into_iter().all(|sha|generation_sha(sha))
            && evidence.collection_source_sha256==registration.collection_source_sha256
            // This is the first original trial anchor only. Every other trial
            // retains its own separately checked cost epoch/action/clock.
            && evidence.cost_epoch_sha256==registration.cost_epoch_sha256
            && evidence.resource_profile_sha256==registration.resource_profile_sha256
            && evidence.frozen_base_weights_sha256==registration.frozen_base.weights_sha256
            && publication.cost_scope==registration.cost_scope,
            "original /2 eligibility/lineage/source/first cost anchor differs",
        )?;
        let actual_source:RoventUtilityInterventionSource=serde_json::from_value(json!({
            "cohort_id":evidence.cohort_id,"profile_sha256":evidence.profile_sha256,
            "publication":{"sha256":reference.publication.sha256,"bytes":reference.publication.bytes},
            "durable_ack":{"sha256":reference.durable_ack.sha256,"bytes":reference.durable_ack.bytes},
            "neutral_input_sha256":evidence.input_sha256,"neutral_target_sha256":evidence.neutral_target_sha256,
            "neutral_context_sha256":evidence.utility_context_sha256,"root_rules_sha256":evidence.rules_state_sha256,
            "role_queries":evidence.role_queries,"frozen_pc_sources":evidence.frozen_pc_sources,
            "ancestor_datasets":evidence.ancestor_datasets,"known_pair_sha256":evidence.known_pair_sha256,"all_pair_sha256":evidence.all_pair_sha256
        })).map_err(|e|invalid(format!("original exact source13: {e}")))?;
        check(
            actual_source == preparation.intervention_source,
            "passive source13 differs from original /2 checked files/ACK/neutral/arms",
        )?;
        check(
            evidence.known_pairs.len() == evidence.known_pair_sha256.len()
                && evidence.known_pairs.len() <= 3,
            "original known pair coverage differs",
        )?;
        let mut directed = BTreeSet::new();
        for pair in &evidence.known_pairs {
            check(
                pair[0] != pair[1]
                    && [4, 5, 6].contains(&pair[0])
                    && [4, 5, 6].contains(&pair[1])
                    && !directed.contains(&[pair[1], pair[0]])
                    && directed.insert(*pair),
                "original route pair unsupported/duplicate/contradictory",
            )?;
        }
        let mut projection = json!({"schema":ROVENT_UTILITY_INTERVENTION_CONTEXT_SOURCE,"data_family":"intervention_v2",
            "publication_sha256":reference.publication.sha256,"durable_ack_sha256":reference.durable_ack.sha256,
            "cohort_id":evidence.cohort_id,"profile_sha256":evidence.profile_sha256,
            "neutral_input_sha256":evidence.input_sha256,"neutral_target_sha256":evidence.neutral_target_sha256,
            "neutral_context_sha256":evidence.utility_context_sha256,"role_queries":evidence.role_queries,
            "collection_source_sha256":evidence.collection_source_sha256,"first_trial_cost_epoch_anchor_sha256":evidence.cost_epoch_sha256,
            "resource_profile_sha256":evidence.resource_profile_sha256,"frozen_base_weights_sha256":evidence.frozen_base_weights_sha256,
            "split":evidence.split,"opening_id":evidence.opening_id,"game_id":evidence.game_id,"line_genealogy_id":evidence.line_genealogy_id,
            "transposition_sha256":evidence.transposition_sha256,"root_rules_sha256":evidence.rules_state_sha256,
            "ancestor_dataset_publication_sha256":evidence.ancestor_dataset_publication_sha256,"all_pair_sha256":evidence.all_pair_sha256,
            "known_pair_sha256":evidence.known_pair_sha256,"known_pairs":evidence.known_pairs,"cost_scope":publication.cost_scope,"execution_authorized":false});
        let context_sha256 = digest(ROVENT_UTILITY_INTERVENTION_CONTEXT_SOURCE, &projection)?;
        projection["context_sha256"] = json!(context_sha256);
        let context = UtilityInterventionContextAuthority {
            sha256: context_sha256.clone(),
            source_projection: projection,
            known_pair_sha256: evidence.known_pair_sha256.clone(),
            unique_known_pairs: evidence.known_pairs.len() as u64,
        };
        let context_ids = vec![context_sha256];
        let pool_sha256 = digest(ROVENT_UTILITY_INTERVENTION_POOL, &context_ids)?;
        let dataset_sha256 = digest(
            ROVENT_UTILITY_INTERVENTION_DATASET,
            &json!({"data_family":"intervention_v2",
            "publication":{"sha256":reference.publication.sha256,"bytes":reference.publication.bytes},
            "durable_ack":{"sha256":reference.durable_ack.sha256,"bytes":reference.durable_ack.bytes},
            "contexts":context_ids,"pool_sha256":pool_sha256,"source_registration_sha256":parts.source_registration_sha256()}),
        )?;
        let ancestry_checked =
            AtomicBool::new(evidence.ancestor_dataset_publication_sha256.is_empty());
        let pins = original
            .original_files()
            .map(|(pin, _)| RoventTrainingFilePin {
                path: reference.root.join(&pin.path),
                sha256: pin.sha256.clone(),
                bytes: pin.bytes,
            })
            .collect();
        let owner = Self {
            reference: reference.clone(),
            original,
            pins,
            context,
            context_ids,
            dataset_sha256,
            pool_sha256,
            preparation_sha256: preparation.preparation_sha256.clone(),
            ancestry_checked,
        };
        owner.revalidate(budget)?;
        Ok(owner)
    }
    pub fn dataset_sha256(&self) -> &str {
        &self.dataset_sha256
    }
    pub fn pool_sha256(&self) -> &str {
        &self.pool_sha256
    }
    pub fn preparation_sha256(&self) -> &str {
        &self.preparation_sha256
    }
    pub fn context_sha256(&self) -> &str {
        self.context.sha256()
    }
    pub fn context_ids(&self) -> &[String] {
        &self.context_ids
    }
    pub fn reference(&self) -> &RoventUtilityDatasetReference {
        &self.reference
    }
    pub fn input_file_pins(&self) -> impl Iterator<Item = &RoventTrainingFilePin> {
        self.pins.iter()
    }
    pub fn revalidate(&self, budget: &RoventRecoveryReadBudget) -> Result<(), ArenaError> {
        self.original.revalidate_original(budget)
    }
    pub(super) fn source_ancestor_publications(&self) -> &[String] {
        &self
            .original
            .evidence()
            .expect("admitted original evidence")
            .ancestor_dataset_publication_sha256
    }
    pub(super) fn source_ancestor_descriptors(&self) -> &[RoventUtilityCohortAncestorSource] {
        &self
            .original
            .evidence()
            .expect("admitted original evidence")
            .ancestor_datasets
    }
    pub(super) fn lineage(&self) -> UtilityDatasetLineage<'_> {
        let e = self
            .original
            .evidence()
            .expect("admitted original evidence");
        UtilityDatasetLineage {
            split: e.split,
            opening_id: &e.opening_id,
            game_id: &e.game_id,
            line_genealogy_id: &e.line_genealogy_id,
            rules_state_sha256: &e.rules_state_sha256,
            transposition_sha256: &e.transposition_sha256,
        }
    }
    pub(super) fn validate_context_projection(
        &self,
        projection: &Value,
    ) -> Result<&UtilityInterventionContextAuthority, ArenaError> {
        check(
            self.ancestry_checked.load(Ordering::Acquire),
            "full original ancestor owner join absent",
        )?;
        check(
            *projection == self.context.source_projection,
            "projection differs from checked original /2 source",
        )?;
        Ok(&self.context)
    }
    pub(super) fn admit_ancestry(
        &self,
        owners: &super::ancestry::VerifiedUtilityAncestors,
        required: &BTreeSet<String>,
    ) -> Result<(), ArenaError> {
        owners.validate_intervention_current(self, required)?;
        self.ancestry_checked.store(true, Ordering::Release);
        Ok(())
    }
    pub fn validate_execution_sources_v3(
        &self,
        registration: &RoventUtilityInterventionExecutionRegistration,
    ) -> Result<(), ArenaError> {
        check(
            registration.current_dataset == self.reference,
            "new execution selected different original /2 dataset",
        )?;
        let publication = self
            .original
            .publication()
            .expect("admitted original publication");
        let evidence = self
            .original
            .evidence()
            .expect("admitted original evidence");
        check(
            registration.frozen_nnuec_base.sha256 == publication.nnuec_weights.sha256
                && registration.frozen_nnuec_base.bytes == publication.nnuec_weights.bytes,
            "original frozen NNUEC raw asset differs",
        )?;
        check(
            registration.frozen_pc_sources.len() == evidence.frozen_pc_sources.len(),
            "original frozen P/C source roster differs",
        )?;
        for declared in &registration.frozen_pc_sources {
            let original = evidence
                .frozen_pc_sources
                .iter()
                .find(|p| p.role == declared.role)
                .ok_or_else(|| invalid("original frozen P/C role missing"))?;
            check(
                declared.artifact.sha256 == original.artifact.sha256
                    && declared.artifact.bytes == original.artifact.bytes
                    && declared.model_semantics_sha256 == original.model_semantics_sha256
                    && declared.model_identity_sha256 == original.model_identity_sha256
                    && declared.encoding_sha256 == original.encoding_sha256,
                "original frozen P/C model/encoding/raw asset differs",
            )?;
        }
        Ok(())
    }
    pub fn observation(&self) -> Value {
        json!({"schema":"rz-rovent-historical-utility-dataset-admission/2","data_family":"intervention_v2","dataset_sha256":self.dataset_sha256,"pool_sha256":self.pool_sha256,"preparation_sha256":self.preparation_sha256,"context_ids":self.context_ids,"full_ancestor_owner_joined":self.ancestry_checked.load(Ordering::Acquire),"unique_known_pairs":self.context.unique_known_pairs,"source_kind":"historical_exact_intervention_pins","past_native_authority_reissued":false,"past_physical_completion_reissued":false,"execution_authorized":false,"original_file_custody":self.original.read_accounting(),"retained_bytes_scope":"original_file_bytes_not_process_peak_memory"})
    }
}
