//! Original mixed-source custody for one utility learner. Every source is
//! admitted by its original reader; declarations never become replay, native
//! execution, physical-completion or optimizer authority.

use super::dataset::UtilityDatasetLineage;
use super::dataset::VerifiedUtilityDataset;
use super::execution::RoventUtilityDatasetReference;
use super::execution_v2::RoventUtilityAncestorDatasetRegistration;
use super::intervention_dataset::{
    RoventUtilityInterventionDatasetRejected, VerifiedUtilityInterventionDataset,
};
use super::intervention_execution::RoventUtilityInterventionAncestorRegistration;
use crate::ArenaError;
use crate::rovent_training_process::recovery::{
    HeldHistoricalFile, HistoricalDatasetLineage, RoventHistoricalDatasetRegistration,
    RoventHistoricalRAuxiliaryDatasetRegistration, RoventRecoveryReadBudget, VerifiedRoventDataset,
    VerifiedRoventRAuxiliaryDataset, archive_pin, probe_utility_pc_source,
};
use crate::rovent_training_process::{RoventPcRecordFamily, check, invalid};
use rz_experiments::pals_data_v3::{GenerationSplitV3, generation_sha};
use rz_experiments::rovent_r_readout::RoventRReadoutManifest;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

const MAX_SOURCES: usize = 256;
const MAX_METADATA_BYTES: u64 = 16 << 20;
const MAX_PREPARATION_BYTES: u64 = 64 << 10;
const MAX_LINEAGE_KEYS: usize = 1_048_576;

struct PcSource {
    registration: RoventHistoricalDatasetRegistration,
    family: RoventPcRecordFamily,
    ancestor_indices: Vec<String>,
}
struct RSource {
    registration: RoventHistoricalRAuxiliaryDatasetRegistration,
    manifest: RoventRReadoutManifest,
    recipe: Value,
}

/// Retains complete original PC/R/utility owners and each original recipe or
/// preparation descriptor. No serde or public success constructor can replace
/// full source admission. Ancestors are never appended to the current sampler.
pub struct VerifiedUtilityAncestors {
    pc: Vec<Arc<VerifiedRoventDataset>>,
    r: Vec<Arc<VerifiedRoventRAuxiliaryDataset>>,
    utility: Vec<Arc<VerifiedUtilityDataset>>,
    intervention: Vec<Arc<VerifiedUtilityInterventionDataset>>,
    held: Vec<Arc<HeldHistoricalFile>>,
    dependencies: BTreeMap<String, BTreeSet<String>>,
}

/// Retains the partial source union and the exact rejected /2 FD owner.
/// No historical failure is flattened into a successful or empty inventory.
pub struct RoventUtilityInterventionAncestorsRejected {
    pub error: ArenaError,
    pub retained: Option<Box<VerifiedUtilityAncestors>>,
    pub failed_dataset: Option<Box<RoventUtilityInterventionDatasetRejected>>,
}
impl std::fmt::Debug for RoventUtilityInterventionAncestorsRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoventUtilityInterventionAncestorsRejected")
            .field("error", &self.error)
            .field("partial_source_union_retained", &self.retained.is_some())
            .field("failed_original_retained", &self.failed_dataset.is_some())
            .finish()
    }
}
impl std::fmt::Display for RoventUtilityInterventionAncestorsRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}
impl std::error::Error for RoventUtilityInterventionAncestorsRejected {}

impl VerifiedUtilityAncestors {
    pub fn open(
        registrations: &[RoventUtilityAncestorDatasetRegistration],
        budget: &Arc<RoventRecoveryReadBudget>,
    ) -> Result<Self, ArenaError> {
        check(
            registrations.len() <= MAX_SOURCES,
            "mixed utility source capacity",
        )?;
        let mut publications = BTreeSet::new();
        let mut pc_sources = BTreeMap::new();
        let mut r_sources = BTreeMap::new();
        let mut utility = Vec::new();
        let mut held = Vec::new();
        for registration in registrations {
            registration.validate()?;
            let declared_pc_family = registration.pc_family();
            check(
                publications.insert(registration.publication().sha256.clone()),
                "duplicate mixed ancestor raw publication",
            )?;
            match registration {
                RoventUtilityAncestorDatasetRegistration::PcLegacy { registration }
                | RoventUtilityAncestorDatasetRegistration::PcSubmittedInputV1 { registration } => {
                    let (family, index, ancestor_indices) =
                        probe_utility_pc_source(registration, budget)?;
                    check(
                        declared_pc_family == Some(family),
                        "original PC family differs from source tag",
                    )?;
                    check(
                        pc_sources
                            .insert(
                                index,
                                PcSource {
                                    registration: registration.clone(),
                                    family,
                                    ancestor_indices,
                                },
                            )
                            .is_none(),
                        "duplicate original PC canonical index",
                    )?;
                }
                RoventUtilityAncestorDatasetRegistration::RAuxiliary {
                    registration,
                    recipe,
                } => {
                    let (_, raw) =
                        budget.read(&registration.publication, MAX_METADATA_BYTES, true)?;
                    let publication: crate::rovent_collect::RoventRCorpusPublication =
                        decode(&raw)?;
                    publication.validate()?;
                    let (_, raw) = budget.read(
                        &archive_pin(&registration.root, &publication.manifest)?,
                        MAX_METADATA_BYTES,
                        true,
                    )?;
                    let manifest: RoventRReadoutManifest = decode(&raw)?;
                    check(
                        manifest.sha256().map_err(|e| invalid(e.to_string()))?
                            == publication.canonical_manifest_sha256,
                        "original R source manifest canonical seal differs",
                    )?;
                    let (recipe_file, raw) = budget.read(recipe, MAX_METADATA_BYTES, true)?;
                    let recipe = decode(&raw)?;
                    held.push(Arc::new(recipe_file));
                    check(
                        r_sources
                            .insert(
                                publication.canonical_index_sha256,
                                RSource {
                                    registration: registration.clone(),
                                    manifest,
                                    recipe,
                                },
                            )
                            .is_none(),
                        "duplicate original R canonical index",
                    )?;
                }
                RoventUtilityAncestorDatasetRegistration::UtilityCohort {
                    registration,
                    preparation,
                } => {
                    let (preparation_file, raw) =
                        budget.read(preparation, MAX_PREPARATION_BYTES, true)?;
                    let preparation: Value = decode(&raw)?;
                    super::registration_cohort::validate_utility_cohort_preparation(&preparation)?;
                    let owner = Arc::new(VerifiedUtilityDataset::open(
                        registration,
                        &preparation,
                        budget,
                    )?);
                    held.push(Arc::new(preparation_file));
                    utility.push(owner);
                }
            }
        }

        let mut pc = BTreeMap::new();
        let mut active = BTreeSet::new();
        for index in pc_sources.keys() {
            open_pc(index, &pc_sources, &mut pc, &mut active, budget)?;
        }
        let mut r = BTreeMap::new();
        for index in r_sources.keys() {
            open_r(index, &r_sources, &pc, &mut r, &mut active, budget)?;
        }

        let mut dependencies = BTreeMap::new();
        for (index, source) in &pc_sources {
            let owner = pc
                .get(index)
                .ok_or_else(|| invalid("missing admitted PC owner"))?;
            let parents = source
                .ancestor_indices
                .iter()
                .map(|key| {
                    pc.get(key)
                        .map(|owner| owner.registration().publication.sha256.clone())
                        .ok_or_else(|| invalid("missing original PC source closure"))
                })
                .collect::<Result<BTreeSet<_>, _>>()?;
            dependencies.insert(owner.registration().publication.sha256.clone(), parents);
        }
        for (index, source) in &r_sources {
            let owner = r
                .get(index)
                .ok_or_else(|| invalid("missing admitted R owner"))?;
            let mut parents = BTreeSet::new();
            for key in &source.manifest.pc_ancestor_index_sha256 {
                parents.insert(
                    pc.get(key)
                        .ok_or_else(|| invalid("missing R original PC closure"))?
                        .registration()
                        .publication
                        .sha256
                        .clone(),
                );
            }
            for key in &source.manifest.r_ancestor_index_sha256 {
                parents.insert(
                    r.get(key)
                        .ok_or_else(|| invalid("missing original R closure"))?
                        .publication_pin()
                        .sha256
                        .clone(),
                );
            }
            dependencies.insert(owner.publication_pin().sha256.clone(), parents);
        }
        for owner in &utility {
            let source = checked_publications(owner.source_ancestor_publications())?;
            dependencies.insert(owner.reference().publication.sha256.clone(), source);
        }
        check(
            dependencies.keys().cloned().collect::<BTreeSet<_>>() == publications,
            "mixed source full owner inventory differs from declarations",
        )?;
        let owner = Self {
            pc: pc.into_values().collect(),
            r: r.into_values().collect(),
            utility,
            intervention: Vec::new(),
            held,
            dependencies,
        };
        owner.validate_source_graph()?;
        owner.validate_lineage(None)?;
        // Each old utility context keeps its own source closure, not the new
        // learner's union or preparation. All descriptors remain retained by
        // the enclosing mixed owner and the subset shares original owners.
        for dataset in &owner.utility {
            let source = checked_publications(dataset.source_ancestor_publications())?;
            let original = owner.subset(&source)?;
            dataset.admit_ancestry(&original, &BTreeSet::new())?;
        }
        owner.revalidate(budget)?;
        Ok(owner)
    }

    /// Explicit /2 intake. Each original source keeps its own family and FD
    /// owner; only this entry can admit utility_intervention declarations.
    pub fn open_intervention(
        registrations: &[RoventUtilityInterventionAncestorRegistration],
        budget: &Arc<RoventRecoveryReadBudget>,
    ) -> Result<Self, Box<RoventUtilityInterventionAncestorsRejected>> {
        let legacy = registrations
            .iter()
            .filter_map(|row| row.legacy().cloned())
            .collect::<Vec<_>>();
        let mut owner = Self::open(&legacy, budget).map_err(|error| {
            Box::new(RoventUtilityInterventionAncestorsRejected {
                error,
                retained: None,
                failed_dataset: None,
            })
        })?;
        let mut failed_dataset = None;
        let admitted = (|| {
            check(
                registrations.len() <= MAX_SOURCES,
                "intervention source capacity",
            )?;
            let mut publications = BTreeSet::new();
            for declaration in registrations {
                declaration.validate()?;
                check(
                    publications.insert(declaration.publication().sha256.clone()),
                    "duplicate mixed intervention raw publication",
                )?;
                if let Some((reference, preparation)) = declaration.intervention() {
                    let (held, raw) = budget.read(preparation, MAX_PREPARATION_BYTES, true)?;
                    owner.held.push(Arc::new(held));
                    let value: Value = decode(&raw)?;
                    let dataset = match VerifiedUtilityInterventionDataset::open_retaining(
                        reference, &value, budget,
                    ) {
                        Ok(dataset) => Arc::new(dataset),
                        Err(rejected) => {
                            let error = invalid(format!(
                                "original intervention source admission failed: {}",
                                rejected.error
                            ));
                            failed_dataset = Some(rejected);
                            return Err(error);
                        }
                    };
                    // Retain the newly admitted owner before any subsequent
                    // graph validation can fail; the rejection then owns the
                    // complete original partial source union.
                    owner.intervention.push(dataset);
                    let parents = checked_publications(
                        owner
                            .intervention
                            .last()
                            .expect("retained source")
                            .source_ancestor_publications(),
                    )?;
                    check(
                        owner
                            .dependencies
                            .insert(reference.publication.sha256.clone(), parents)
                            .is_none(),
                        "duplicate intervention source owner",
                    )?;
                }
            }
            check(
                owner.publication_sha256s() == publications,
                "intervention source full owner inventory differs",
            )?;
            owner.validate_source_graph()?;
            for dataset in &owner.intervention {
                let source = checked_publications(dataset.source_ancestor_publications())?;
                let original = owner.subset(&source)?;
                dataset.admit_ancestry(&original, &BTreeSet::new())?;
            }
            owner.revalidate(budget)
        })();
        match admitted {
            Ok(()) => Ok(owner),
            Err(error) => Err(Box::new(RoventUtilityInterventionAncestorsRejected {
                error,
                retained: Some(Box::new(owner)),
                failed_dataset,
            })),
        }
    }

    pub fn pc_datasets(&self) -> &[Arc<VerifiedRoventDataset>] {
        &self.pc
    }
    pub fn r_datasets(&self) -> &[Arc<VerifiedRoventRAuxiliaryDataset>] {
        &self.r
    }
    pub fn utility_datasets(&self) -> &[Arc<VerifiedUtilityDataset>] {
        &self.utility
    }
    pub fn intervention_datasets(&self) -> &[Arc<VerifiedUtilityInterventionDataset>] {
        &self.intervention
    }
    pub fn intervention_references(&self) -> Vec<RoventUtilityDatasetReference> {
        self.intervention
            .iter()
            .map(|owner| owner.reference().clone())
            .collect()
    }
    pub fn utility_references(&self) -> Vec<RoventUtilityDatasetReference> {
        self.utility
            .iter()
            .map(|owner| owner.reference().clone())
            .collect()
    }
    pub fn publication_sha256s(&self) -> BTreeSet<String> {
        self.dependencies.keys().cloned().collect()
    }

    fn original_descriptors(&self) -> BTreeMap<&str, (&'static str, Value)> {
        let mut descriptors = BTreeMap::new();
        for owner in &self.pc {
            let kind = match owner.family() {
                RoventPcRecordFamily::LegacyRecordV1 => "pc_legacy",
                RoventPcRecordFamily::SubmittedInputV1 => "pc_submitted_input_v1",
            };
            descriptors.insert(
                owner.registration().publication.sha256.as_str(),
                (
                    kind,
                    json!({
                        "manifest_file_sha256": owner.manifest_file_sha256(),
                        "index_file_sha256": owner.index_file_sha256(),
                        "canonical_manifest_sha256": owner.manifest_sha256(),
                        "canonical_index_sha256": owner.index_sha256(),
                    }),
                ),
            );
        }
        for owner in &self.r {
            descriptors.insert(
                owner.publication_pin().sha256.as_str(),
                (
                    "r_auxiliary",
                    json!({
                        "manifest_file_sha256": owner.manifest_file_sha256(),
                        "index_file_sha256": owner.index_file_sha256(),
                        "canonical_manifest_sha256": owner.canonical_manifest_sha256(),
                        "canonical_index_sha256": owner.canonical_index_sha256(),
                    }),
                ),
            );
        }
        for owner in &self.utility {
            descriptors.insert(
                owner.reference().publication.sha256.as_str(),
                (
                    "utility_cohort",
                    json!({
                        "dataset_sha256": owner.dataset_sha256(),
                        "pool_sha256": owner.pool_sha256(),
                        "preparation_sha256": owner.preparation_sha256(),
                        "context_sha256": owner.context_sha256(),
                    }),
                ),
            );
        }
        for owner in &self.intervention {
            descriptors.insert(owner.reference().publication.sha256.as_str(),("utility_intervention",json!({"dataset_sha256":owner.dataset_sha256(),"pool_sha256":owner.pool_sha256(),"preparation_sha256":owner.preparation_sha256(),"context_sha256":owner.context_sha256()})));
        }
        descriptors
    }

    fn validate_source_descriptors(
        &self,
        current: &VerifiedUtilityDataset,
        expected_publications: &BTreeSet<String>,
    ) -> Result<(), ArenaError> {
        let originals = self.original_descriptors();
        let mut seen = BTreeSet::new();
        for descriptor in current.source_ancestor_descriptors() {
            descriptor.validate()?;
            check(
                seen.insert(descriptor.publication_sha256.as_str()),
                "repeated original ancestor source descriptor",
            )?;
            let (kind, metadata) = originals
                .get(descriptor.publication_sha256.as_str())
                .ok_or_else(|| invalid("ancestor descriptor lacks its full original owner"))?;
            check(
                *kind == descriptor.kind.as_str() && metadata == &descriptor.metadata,
                "original ancestor family or exact source metadata changed",
            )?;
        }
        check(
            seen == expected_publications.iter().map(String::as_str).collect(),
            "original ancestor source descriptor coverage differs",
        )
    }

    pub fn validate_current(
        &self,
        current: &VerifiedUtilityDataset,
        required_checkpoint_publications: &BTreeSet<String>,
    ) -> Result<(), ArenaError> {
        check(
            self.intervention.is_empty(),
            "legacy current dataset cannot adopt intervention source owners",
        )?;
        check(
            required_checkpoint_publications
                .iter()
                .all(|sha| generation_sha(sha)),
            "invalid original checkpoint dataset publication seal",
        )?;
        let mut expected = checked_publications(current.source_ancestor_publications())?;
        self.validate_source_descriptors(current, &expected)?;
        // Checkpoint-only sources cannot repair an incomplete declaration of
        // the current collector's own complete historical source graph.
        self.subset(&expected)?;
        expected.extend(required_checkpoint_publications.iter().cloned());
        check(
            !expected.contains(&current.reference().publication.sha256)
                && expected == self.publication_sha256s(),
            "mixed source coverage differs from current source and original checkpoint union",
        )?;
        self.validate_source_graph()?;
        self.validate_lineage(Some(current.lineage()))
    }

    pub fn validate_intervention_current(
        &self,
        current: &VerifiedUtilityInterventionDataset,
        required_checkpoint_publications: &BTreeSet<String>,
    ) -> Result<(), ArenaError> {
        check(
            required_checkpoint_publications
                .iter()
                .all(|sha| generation_sha(sha)),
            "invalid original intervention checkpoint publication",
        )?;
        let mut expected = checked_publications(current.source_ancestor_publications())?;
        let originals = self.original_descriptors();
        let mut seen = BTreeSet::new();
        for descriptor in current.source_ancestor_descriptors() {
            if descriptor.kind == "utility_intervention" {
                let body = descriptor
                    .metadata
                    .as_object()
                    .ok_or_else(|| invalid("intervention ancestor metadata object"))?;
                check(
                    body.len() == 4
                        && [
                            "dataset_sha256",
                            "pool_sha256",
                            "preparation_sha256",
                            "context_sha256",
                        ]
                        .into_iter()
                        .all(|key| {
                            body.get(key)
                                .and_then(Value::as_str)
                                .is_some_and(generation_sha)
                        })
                        && generation_sha(&descriptor.publication_sha256),
                    "exact intervention ancestor metadata4 required",
                )?;
            } else {
                descriptor.validate()?;
            }
            check(
                seen.insert(descriptor.publication_sha256.as_str()),
                "repeated intervention ancestor descriptor",
            )?;
            let (kind, metadata) = originals
                .get(descriptor.publication_sha256.as_str())
                .ok_or_else(|| invalid("intervention ancestor lacks full original owner"))?;
            check(
                *kind == descriptor.kind.as_str() && metadata == &descriptor.metadata,
                "original intervention ancestor family/metadata differs",
            )?;
        }
        check(
            seen == expected.iter().map(String::as_str).collect(),
            "intervention original descriptor coverage differs",
        )?;
        self.subset(&expected)?;
        expected.extend(required_checkpoint_publications.iter().cloned());
        check(
            !expected.contains(&current.reference().publication.sha256)
                && expected == self.publication_sha256s(),
            "intervention current/source/checkpoint union differs",
        )?;
        self.validate_source_graph()?;
        self.validate_lineage(Some(current.lineage()))
    }

    pub fn revalidate(&self, budget: &RoventRecoveryReadBudget) -> Result<(), ArenaError> {
        for owner in &self.pc {
            owner.revalidate(budget)?;
        }
        for owner in &self.r {
            owner.revalidate(budget)?;
        }
        for owner in &self.utility {
            owner.revalidate(budget)?;
        }
        for owner in &self.intervention {
            owner.revalidate(budget)?;
        }
        for original in &self.held {
            let (fresh, _) = budget.read(&original.pin, original.pin.bytes, false)?;
            check(
                fresh.identity.same_inode(original.identity),
                "original ancestor recipe/preparation identity changed",
            )?;
        }
        Ok(())
    }

    fn subset(&self, publications: &BTreeSet<String>) -> Result<Self, ArenaError> {
        check(
            publications.is_subset(&self.publication_sha256s()),
            "missing original utility source publication",
        )?;
        let dependencies = self
            .dependencies
            .iter()
            .filter(|(sha, _)| publications.contains(*sha))
            .map(|(sha, parents)| (sha.clone(), parents.clone()))
            .collect();
        let subset = Self {
            pc: self
                .pc
                .iter()
                .filter(|owner| publications.contains(&owner.registration().publication.sha256))
                .cloned()
                .collect(),
            r: self
                .r
                .iter()
                .filter(|owner| publications.contains(&owner.publication_pin().sha256))
                .cloned()
                .collect(),
            utility: self
                .utility
                .iter()
                .filter(|owner| publications.contains(&owner.reference().publication.sha256))
                .cloned()
                .collect(),
            intervention: self
                .intervention
                .iter()
                .filter(|owner| publications.contains(&owner.reference().publication.sha256))
                .cloned()
                .collect(),
            held: self.held.clone(),
            dependencies,
        };
        subset.validate_source_graph()?;
        Ok(subset)
    }

    fn validate_source_graph(&self) -> Result<(), ArenaError> {
        let mut done = BTreeSet::new();
        let mut active = BTreeSet::new();
        for publication in self.dependencies.keys() {
            visit_source(publication, &self.dependencies, &mut done, &mut active)?;
        }
        Ok(())
    }

    fn validate_lineage(
        &self,
        current: Option<UtilityDatasetLineage<'_>>,
    ) -> Result<(), ArenaError> {
        let Some(current) = current else {
            // Original PC/R readers retain their own eligibility/exclusion
            // decisions. Each historical utility owner is checked separately
            // against its own original source subset by admit_ancestry.
            return Ok(());
        };
        let mut components = UtilityLineageComponents::new();
        for owner in &self.pc {
            for row in owner.utility_lineage() {
                components.observe(row)?;
            }
        }
        for owner in &self.r {
            for row in owner.utility_lineage() {
                components.observe(row)?;
            }
        }
        for owner in &self.utility {
            let row = owner.lineage();
            components.observe(HistoricalDatasetLineage {
                split: row.split,
                opening_id: row.opening_id,
                game_id: row.game_id,
                line_genealogy_id: row.line_genealogy_id,
                rules_state_sha256: row.rules_state_sha256,
                transposition_sha256: row.transposition_sha256,
            })?;
        }
        for owner in &self.intervention {
            let row = owner.lineage();
            components.observe(HistoricalDatasetLineage {
                split: row.split,
                opening_id: row.opening_id,
                game_id: row.game_id,
                line_genealogy_id: row.line_genealogy_id,
                rules_state_sha256: row.rules_state_sha256,
                transposition_sha256: row.transposition_sha256,
            })?;
        }
        let row = current;
        let current_node = components.observe(HistoricalDatasetLineage {
            split: row.split,
            opening_id: row.opening_id,
            game_id: row.game_id,
            line_genealogy_id: row.line_genealogy_id,
            rules_state_sha256: row.rules_state_sha256,
            transposition_sha256: row.transposition_sha256,
        })?;
        let current_component = components.root(current_node);
        check(
            components.split_masks[current_component].count_ones() == 1,
            "current utility context belongs to a known cross-split source component",
        )
    }
}

fn open_pc(
    index: &str,
    sources: &BTreeMap<String, PcSource>,
    done: &mut BTreeMap<String, Arc<VerifiedRoventDataset>>,
    active: &mut BTreeSet<String>,
    budget: &RoventRecoveryReadBudget,
) -> Result<Arc<VerifiedRoventDataset>, ArenaError> {
    if let Some(owner) = done.get(index) {
        return Ok(Arc::clone(owner));
    }
    check(
        active.len() < MAX_SOURCES && active.insert(index.to_owned()),
        "PC source cycle/depth bound",
    )?;
    let source = sources
        .get(index)
        .ok_or_else(|| invalid("missing original PC ancestor index"))?;
    let ancestors = source
        .ancestor_indices
        .iter()
        .map(|key| open_pc(key, sources, done, active, budget))
        .collect::<Result<Vec<_>, _>>()?;
    let owner = Arc::new(match source.family {
        RoventPcRecordFamily::LegacyRecordV1 => {
            VerifiedRoventDataset::open(source.registration.clone(), ancestors, budget)?
        }
        RoventPcRecordFamily::SubmittedInputV1 => VerifiedRoventDataset::open_submitted_input_v1(
            source.registration.clone(),
            ancestors,
            budget,
        )?,
    });
    check(
        owner.family() == source.family && owner.index_sha256() == index,
        "original PC family/index changed after probe",
    )?;
    active.remove(index);
    done.insert(index.to_owned(), Arc::clone(&owner));
    Ok(owner)
}

fn open_r(
    index: &str,
    sources: &BTreeMap<String, RSource>,
    pc: &BTreeMap<String, Arc<VerifiedRoventDataset>>,
    done: &mut BTreeMap<String, Arc<VerifiedRoventRAuxiliaryDataset>>,
    active: &mut BTreeSet<String>,
    budget: &RoventRecoveryReadBudget,
) -> Result<Arc<VerifiedRoventRAuxiliaryDataset>, ArenaError> {
    if let Some(owner) = done.get(index) {
        return Ok(Arc::clone(owner));
    }
    check(
        active.len() < MAX_SOURCES && active.insert(index.to_owned()),
        "R source cycle/depth bound",
    )?;
    let source = sources
        .get(index)
        .ok_or_else(|| invalid("missing original R ancestor index"))?;
    let ancestors = source
        .manifest
        .r_ancestor_index_sha256
        .iter()
        .map(|key| open_r(key, sources, pc, done, active, budget))
        .collect::<Result<Vec<_>, _>>()?;
    let pc_datasets = source
        .manifest
        .pc_ancestor_index_sha256
        .iter()
        .map(|key| {
            pc.get(key)
                .cloned()
                .ok_or_else(|| invalid("missing original R PC source index"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let owner = Arc::new(VerifiedRoventRAuxiliaryDataset::open(
        source.registration.clone(),
        &source.recipe,
        pc_datasets,
        ancestors,
        budget,
    )?);
    active.remove(index);
    done.insert(index.to_owned(), Arc::clone(&owner));
    Ok(owner)
}

fn visit_source(
    publication: &str,
    dependencies: &BTreeMap<String, BTreeSet<String>>,
    done: &mut BTreeSet<String>,
    active: &mut BTreeSet<String>,
) -> Result<(), ArenaError> {
    if done.contains(publication) {
        return Ok(());
    }
    check(
        active.len() < MAX_SOURCES && active.insert(publication.to_owned()),
        "mixed source ancestry cycle/depth bound",
    )?;
    let parents = dependencies
        .get(publication)
        .ok_or_else(|| invalid("missing original mixed source publication"))?;
    for parent in parents {
        visit_source(parent, dependencies, done, active)?;
    }
    active.remove(publication);
    done.insert(publication.to_owned());
    Ok(())
}

fn checked_publications(values: &[String]) -> Result<BTreeSet<String>, ArenaError> {
    check(
        values.len() <= MAX_SOURCES
            && values.iter().all(|sha| generation_sha(sha))
            && values.windows(2).all(|pair| pair[0] < pair[1]),
        "invalid original full source publication set",
    )?;
    Ok(values.iter().cloned().collect())
}

/// Five domain-separated borrowed lineage keys connect original rows. Only
/// the current utility context's completed component is an intake decision;
/// unrelated excluded historical components remain original source evidence.
struct UtilityLineageComponents<'a> {
    keys: BTreeMap<(&'static str, &'a str), usize>,
    parents: Vec<usize>,
    ranks: Vec<u8>,
    split_masks: Vec<u8>,
}
impl<'a> UtilityLineageComponents<'a> {
    fn new() -> Self {
        Self {
            keys: BTreeMap::new(),
            parents: Vec::new(),
            ranks: Vec::new(),
            split_masks: Vec::new(),
        }
    }
    fn root(&mut self, mut node: usize) -> usize {
        while self.parents[node] != node {
            self.parents[node] = self.parents[self.parents[node]];
            node = self.parents[node];
        }
        node
    }
    fn join(&mut self, left: usize, right: usize) {
        let mut left = self.root(left);
        let mut right = self.root(right);
        if left == right {
            return;
        }
        if self.ranks[left] < self.ranks[right] {
            std::mem::swap(&mut left, &mut right);
        }
        self.parents[right] = left;
        self.split_masks[left] |= self.split_masks[right];
        if self.ranks[left] == self.ranks[right] {
            // At most MAX_LINEAGE_KEYS nodes gives rank at most 20.
            self.ranks[left] += 1;
        }
    }
    fn observe(&mut self, row: HistoricalDatasetLineage<'a>) -> Result<usize, ArenaError> {
        let keys = [
            ("opening", row.opening_id),
            ("game", row.game_id),
            ("genealogy", row.line_genealogy_id),
            ("rules_state", row.rules_state_sha256),
            ("transposition", row.transposition_sha256),
        ];
        check(
            keys.iter().all(|(_, key)| !key.is_empty()),
            "missing original mixed source lineage",
        )?;
        let node = match keys.iter().find_map(|key| self.keys.get(key).copied()) {
            Some(node) => self.root(node),
            None => {
                check(
                    self.parents.len() < MAX_LINEAGE_KEYS,
                    "mixed source lineage component capacity",
                )?;
                let node = self.parents.len();
                self.parents.push(node);
                self.ranks.push(0);
                self.split_masks.push(0);
                node
            }
        };
        self.split_masks[node] |= match row.split {
            GenerationSplitV3::Train => 1,
            GenerationSplitV3::Validation => 2,
            GenerationSplitV3::Holdout => 4,
        };
        for key in keys {
            if let Some(prior) = self.keys.get(&key).copied() {
                self.join(node, prior);
            } else {
                check(
                    self.keys.len() < MAX_LINEAGE_KEYS,
                    "mixed source lineage index capacity",
                )?;
                self.keys.insert(key, node);
            }
        }
        Ok(node)
    }
}

fn decode<T: DeserializeOwned>(raw: &[u8]) -> Result<T, ArenaError> {
    let text = std::str::from_utf8(raw).map_err(|_| invalid("original ancestor metadata UTF8"))?;
    rz_experiments::decode_json(text).map_err(|e| invalid(e.to_string()))
}
