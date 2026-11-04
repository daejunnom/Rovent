//! Exact historical utility-cohort dataset admission.
//!
//! This owner holds the externally pinned publication, its separate durable
//! collection-I/O receipt and every original payload descriptor. It is data
//! custody for a new learner process, never a reissued native trial, past
//! physical completion, executable action, or optimizer-update capability.
//! One publication is one cohort/context; this codec does not silently merge
//! independent cohorts or reinterpret root-local utility observations.

use super::execution::{RoventUtilityDatasetReference, RoventUtilityFrozenPcSource};
use super::execution_v2::RoventUtilityExecutionRegistrationV2;
use super::registration_cohort::validate_utility_cohort_preparation;
use crate::ArenaError;
use crate::pals_collect_v3::training_process::{FileIdentity, directory_file, open_directory};
use crate::rovent_collect::{
    RoventHistoricalUtilityCohortEvidence, RoventPublicationFile,
    RoventUtilityCohortAncestorSource, RoventUtilityCohortDurableAck,
    RoventUtilityCohortPublication, verify_historical_utility_cohort,
};
use crate::rovent_training_process::RoventTrainingFilePin;
use crate::rovent_training_process::recovery::{HeldHistoricalFile, RoventRecoveryReadBudget};
use rz_experiments::pals_data_v3::{
    GenerationSplitV3, generation_digest, generation_identifier, generation_sha,
};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Component, Path};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

pub const ROVENT_UTILITY_DATASET: &str = "rz-rovent-utility-training-dataset/1";
pub const ROVENT_UTILITY_CONTEXT_SOURCE: &str = "rz-rovent-utility-cohort-context-source/1";
pub const ROVENT_UTILITY_POOL: &str = "rz-rovent-utility-training-context-pool/1";
const MAX_METADATA: u64 = 64 << 10;
const MAX_PAYLOAD: u64 = 256 << 20;
const PUBLICATION_FILE: &str = "utility-cohort-publication.json";
const ACK_FILE: &str = "utility-cohort-durable-ack.json";

fn invalid(reason: impl Into<String>) -> ArenaError {
    ArenaError::Integrity(format!("Rovent utility dataset: {}", reason.into()))
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
fn raw_text(raw: &[u8]) -> Result<&str, ArenaError> {
    std::str::from_utf8(raw).map_err(|_| invalid("original cohort metadata is not UTF-8"))
}
fn relative_file(
    root: &Path,
    pin: &RoventTrainingFilePin,
) -> Result<RoventPublicationFile, ArenaError> {
    let path = pin
        .path
        .strip_prefix(root)
        .map_err(|_| invalid("file escapes original cohort root"))?;
    check(
        !path.as_os_str().is_empty()
            && path.components().all(|p| matches!(p, Component::Normal(_))),
        "normalized relative original cohort path required",
    )?;
    let path = path
        .to_str()
        .ok_or_else(|| invalid("cohort path is not UTF-8"))?;
    check(
        !path.contains('\\'),
        "cohort path spelling must not be normalized",
    )?;
    let relative = RoventPublicationFile {
        path: path.into(),
        sha256: pin.sha256.clone(),
        bytes: pin.bytes,
    };
    relative.validate()?;
    Ok(relative)
}
fn absolute_file(
    root: &Path,
    pin: &RoventPublicationFile,
) -> Result<RoventTrainingFilePin, ArenaError> {
    pin.validate()?;
    let relative = Path::new(&pin.path);
    check(
        !relative.is_absolute()
            && !pin.path.contains('\\')
            && relative
                .components()
                .all(|part| matches!(part, Component::Normal(_))),
        "payload escapes original cohort root",
    )?;
    Ok(RoventTrainingFilePin {
        path: root.join(relative),
        sha256: pin.sha256.clone(),
        bytes: pin.bytes,
    })
}

/// The parent constructs this only after the full E/CPU historical verifier
/// and original file-pin coverage succeed. No public constructor or serde
/// implementation can manufacture a valid context from support/known flags.
pub(super) struct UtilityContextAuthority {
    sha256: String,
    source_projection: Value,
    known_pair_sha256: Vec<String>,
    unique_known_pairs: u64,
}
pub(super) struct UtilityDatasetLineage<'a> {
    pub(super) split: GenerationSplitV3,
    pub(super) opening_id: &'a str,
    pub(super) game_id: &'a str,
    pub(super) line_genealogy_id: &'a str,
    pub(super) rules_state_sha256: &'a str,
    pub(super) transposition_sha256: &'a str,
}
impl UtilityContextAuthority {
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

/// Historical exact-byte dataset custody. All strong native/model/View caps
/// belonged to the old collector and are not reconstructed by this owner.
pub struct VerifiedUtilityDataset {
    reference: RoventUtilityDatasetReference,
    root: File,
    root_identity: FileIdentity,
    held: Vec<HeldHistoricalFile>,
    publication: RoventUtilityCohortPublication,
    durable_ack: RoventUtilityCohortDurableAck,
    evidence: RoventHistoricalUtilityCohortEvidence,
    context: UtilityContextAuthority,
    context_ids: Vec<String>,
    dataset_sha256: String,
    pool_sha256: String,
    preparation_sha256: String,
    ancestry_checked: AtomicBool,
}
impl VerifiedUtilityDataset {
    pub fn open(
        reference: &RoventUtilityDatasetReference,
        preparation: &Value,
        budget: &Arc<RoventRecoveryReadBudget>,
    ) -> Result<Self, ArenaError> {
        reference.validate()?;
        let preparation_sha256 = validate_utility_cohort_preparation(preparation)?;
        let root = directory_file(&open_directory(&reference.root)?)?;
        let metadata = root
            .metadata()
            .map_err(|_| invalid("cohort root metadata"))?;
        check(
            metadata.is_dir()
                && metadata.uid() == nix::unistd::geteuid().as_raw()
                && metadata.permissions().mode() & 0o777 == 0o700,
            "caller-pinned private original cohort directory required",
        )?;
        let root_identity = FileIdentity::from(&metadata);
        let publication_file = relative_file(&reference.root, &reference.publication)?;
        let ack_file = relative_file(&reference.root, &reference.durable_ack)?;
        check(
            publication_file.path == PUBLICATION_FILE && ack_file.path == ACK_FILE,
            "distinct original cohort publication/ACK filenames required",
        )?;
        let (publication_held, publication_raw) =
            budget.read(&reference.publication, MAX_METADATA, true)?;
        let publication =
            RoventUtilityCohortPublication::from_json_historical(raw_text(&publication_raw)?)?;
        let (ack_held, ack_raw) = budget.read(&reference.durable_ack, MAX_METADATA, true)?;
        let durable_ack = RoventUtilityCohortDurableAck::from_json_historical(
            raw_text(&ack_raw)?,
            &publication,
            &publication_file,
        )?;
        let mut held = vec![publication_held, ack_held];
        let mut payloads = BTreeMap::new();
        let mut total = 0_u64;
        for relative in publication.payloads() {
            let pin = absolute_file(&reference.root, relative)?;
            total = total
                .checked_add(pin.bytes)
                .filter(|n| *n <= MAX_PAYLOAD)
                .ok_or_else(|| invalid("original cohort payload byte overflow/budget"))?;
            let (file, bytes) = budget.read(&pin, MAX_PAYLOAD, true)?;
            check(
                payloads.insert(relative.path.clone(), bytes).is_none(),
                "original cohort payload path was repeated",
            )?;
            held.push(file);
        }
        check(
            total == publication.total_payload_bytes,
            "cohort payload coverage/count mismatch",
        )?;
        durable_ack.validate_original_trial_io(&publication, &payloads)?;
        // Exact same raw buffers verified through the held descriptors are
        // borrowed here. Value reserialization cannot replace native seals.
        let evidence = verify_historical_utility_cohort(&publication, &payloads)?;
        check(
            [
                &evidence.profile_sha256,
                &evidence.input_sha256,
                &evidence.target_sha256,
                &evidence.submission_target_sha256,
                &evidence.utility_context_sha256,
                &evidence.collection_source_sha256,
                &evidence.cost_epoch_sha256,
                &evidence.resource_profile_sha256,
                &evidence.frozen_base_weights_sha256,
            ]
            .into_iter()
            .all(|pin| generation_sha(pin))
                && !evidence.input_reference_json.is_empty()
                && evidence.input_reference_json.len() <= 8 << 20,
            "original frozen utility input and its distinct source seals are required",
        )?;
        check(
            evidence.cohort_id == publication.cohort_id
                && evidence.usage == "collection_candidate"
                && evidence.split == GenerationSplitV3::Train
                && generation_identifier(&evidence.opening_id)
                && generation_identifier(&evidence.game_id)
                && generation_identifier(&evidence.line_genealogy_id)
                && generation_sha(&evidence.transposition_sha256)
                && generation_sha(&evidence.rules_state_sha256)
                && evidence.collection_source_sha256
                    == preparation["registration"]["collection_source_sha256"]
                && evidence.cost_epoch_sha256 == preparation["registration"]["cost_epoch_sha256"]
                && evidence.resource_profile_sha256
                    == preparation["registration"]["resource_profile_sha256"]
                && evidence.frozen_base_weights_sha256
                    == preparation["registration"]["frozen_base"]["weights_sha256"],
            "original cohort train split/lineage/usage/source/cost/resource/frozen base differs from preparation",
        )?;
        check(
            evidence.known_pairs.len() == evidence.known_pair_sha256.len()
                && evidence.known_pairs.len() <= 3
                && evidence.all_pair_sha256.len() <= 3,
            "original known comparison/source counts differ",
        )?;
        let mut directed = BTreeSet::new();
        for pair in &evidence.known_pairs {
            check(
                pair[0] != pair[1]
                    && [4, 5, 6].contains(&pair[0])
                    && [4, 5, 6].contains(&pair[1])
                    && !directed.contains(&[pair[1], pair[0]])
                    && directed.insert(*pair),
                "unsupported, repeated or contradictory original route comparison",
            )?;
        }
        for pins in [&evidence.known_pair_sha256, &evidence.all_pair_sha256] {
            check(
                pins.iter().all(|p| generation_sha(p))
                    && pins.iter().collect::<BTreeSet<_>>().len() == pins.len(),
                "duplicate or invalid original pair seal",
            )?;
        }
        check(
            evidence
                .known_pair_sha256
                .iter()
                .all(|p| evidence.all_pair_sha256.contains(p)),
            "known comparison is outside original complete pair inventory",
        )?;
        let mut projection = json!({"schema":ROVENT_UTILITY_CONTEXT_SOURCE,
            "publication_sha256":reference.publication.sha256,"durable_ack_sha256":reference.durable_ack.sha256,
            "cohort_id":evidence.cohort_id,"profile_sha256":evidence.profile_sha256,
            "input_sha256":evidence.input_sha256,"target_sha256":evidence.target_sha256,
            "submission_target_sha256":evidence.submission_target_sha256,
            "utility_context_sha256":evidence.utility_context_sha256,
            "collection_source_sha256":evidence.collection_source_sha256,
            "cost_epoch_sha256":evidence.cost_epoch_sha256,"resource_profile_sha256":evidence.resource_profile_sha256,
            "cost_scope":publication.cost_scope,"frozen_base_weights_sha256":evidence.frozen_base_weights_sha256,
            "split":evidence.split,"opening_id":evidence.opening_id,"game_id":evidence.game_id,
            "line_genealogy_id":evidence.line_genealogy_id,"transposition_sha256":evidence.transposition_sha256,
            "rules_state_sha256":evidence.rules_state_sha256,
            "ancestor_dataset_publication_sha256":evidence.ancestor_dataset_publication_sha256,
            "all_pair_sha256":evidence.all_pair_sha256,"known_pair_sha256":evidence.known_pair_sha256,
            "known_pairs":evidence.known_pairs,"execution_authorized":false});
        let context_sha256 = digest(ROVENT_UTILITY_CONTEXT_SOURCE, &projection)?;
        projection["context_sha256"] = json!(context_sha256);
        let context = UtilityContextAuthority {
            sha256: context_sha256.clone(),
            source_projection: projection,
            known_pair_sha256: evidence.known_pair_sha256.clone(),
            unique_known_pairs: evidence.known_pairs.len() as u64,
        };
        let context_ids = vec![context_sha256];
        let pool_sha256 = digest(ROVENT_UTILITY_POOL, &context_ids)?;
        let dataset_sha256 = digest(
            ROVENT_UTILITY_DATASET,
            &json!({
            "publication":{"sha256":reference.publication.sha256,"bytes":reference.publication.bytes},
            "durable_ack":{"sha256":reference.durable_ack.sha256,"bytes":reference.durable_ack.bytes},
            "contexts":context_ids,"pool_sha256":pool_sha256,
            "source_registration_sha256":preparation["registration_sha256"]}),
        )?;
        let ancestry_checked =
            AtomicBool::new(evidence.ancestor_dataset_publication_sha256.is_empty());
        let owner = Self {
            reference: reference.clone(),
            root,
            root_identity,
            held,
            publication,
            durable_ack,
            evidence,
            context,
            context_ids,
            dataset_sha256,
            pool_sha256,
            preparation_sha256,
            ancestry_checked,
        };
        owner.revalidate(budget)?;
        Ok(owner)
    }
    pub fn dataset_sha256(&self) -> &str {
        &self.dataset_sha256
    }
    pub fn reference(&self) -> &RoventUtilityDatasetReference {
        &self.reference
    }
    pub(super) fn lineage(&self) -> UtilityDatasetLineage<'_> {
        UtilityDatasetLineage {
            split: self.evidence.split,
            opening_id: &self.evidence.opening_id,
            game_id: &self.evidence.game_id,
            line_genealogy_id: &self.evidence.line_genealogy_id,
            rules_state_sha256: &self.evidence.rules_state_sha256,
            transposition_sha256: &self.evidence.transposition_sha256,
        }
    }
    pub(super) fn source_ancestor_publications(&self) -> &[String] {
        &self.evidence.ancestor_dataset_publication_sha256
    }
    pub(super) fn source_ancestor_descriptors(&self) -> &[RoventUtilityCohortAncestorSource] {
        &self.evidence.ancestor_datasets
    }
    pub fn pool_sha256(&self) -> &str {
        &self.pool_sha256
    }
    pub fn preparation_sha256(&self) -> &str {
        &self.preparation_sha256
    }
    pub fn context_sha256(&self) -> &str {
        &self.context.sha256
    }
    pub fn context_ids(&self) -> &[String] {
        &self.context_ids
    }
    pub(super) fn validate_context_projection(
        &self,
        projection: &Value,
    ) -> Result<&UtilityContextAuthority, ArenaError> {
        check(
            self.ancestry_checked.load(Ordering::Acquire),
            "full original PC/R/utility ancestor owner join is not complete",
        )?;
        check(
            *projection == self.context.source_projection,
            "context differs from original exact cohort data",
        )?;
        Ok(&self.context)
    }
    pub fn input_file_pins(&self) -> impl Iterator<Item = &RoventTrainingFilePin> {
        self.held.iter().map(|f| &f.pin)
    }
    pub fn revalidate(&self, budget: &RoventRecoveryReadBudget) -> Result<(), ArenaError> {
        let named = directory_file(&open_directory(&self.reference.root)?)?;
        for file in [&self.root, &named] {
            let metadata = file
                .metadata()
                .map_err(|_| invalid("retained cohort root readback"))?;
            check(
                self.root_identity.same_inode(FileIdentity::from(&metadata)),
                "original named or held cohort root was replaced",
            )?;
        }
        for retained in &self.held {
            let identity = FileIdentity::from(
                &retained
                    .file
                    .metadata()
                    .map_err(|_| invalid("cohort retained file metadata"))?,
            );
            check(
                retained.identity.same_inode(identity),
                "retained cohort file identity changed",
            )?;
            budget.verify_held(&retained.file, &retained.pin, false)?;
        }
        Ok(())
    }
    pub fn validate_execution_sources_v2(
        &self,
        registration: &RoventUtilityExecutionRegistrationV2,
    ) -> Result<(), ArenaError> {
        self.validate_execution_parts(
            &registration.current_dataset,
            &registration.frozen_nnuec_base,
            &registration.frozen_pc_sources,
        )
    }
    fn validate_execution_parts(
        &self,
        current: &RoventUtilityDatasetReference,
        base: &RoventTrainingFilePin,
        pc_sources: &[RoventUtilityFrozenPcSource],
    ) -> Result<(), ArenaError> {
        check(
            *current == self.reference,
            "new execution selected a different exact cohort enrollment",
        )?;
        let archived_base = &self.publication.nnuec_weights;
        check(
            base.sha256 == archived_base.sha256 && base.bytes == archived_base.bytes,
            "frozen NNUEC raw file differs from original cohort asset",
        )?;
        check(
            pc_sources.len() == self.evidence.frozen_pc_sources.len(),
            "frozen P/C original source inventory differs",
        )?;
        for declared in pc_sources {
            let original = self
                .evidence
                .frozen_pc_sources
                .iter()
                .find(|p| p.role == declared.role)
                .ok_or_else(|| invalid("frozen P/C role is outside original cohort"))?;
            check(
                declared.artifact.sha256 == original.artifact.sha256
                    && declared.artifact.bytes == original.artifact.bytes
                    && declared.model_semantics_sha256 == original.model_semantics_sha256
                    && declared.model_identity_sha256 == original.model_identity_sha256
                    && declared.encoding_sha256 == original.encoding_sha256,
                "frozen P/C raw asset/model/encoding identities differ",
            )?;
        }
        Ok(())
    }
    pub(super) fn admit_ancestry(
        &self,
        owners: &super::ancestry::VerifiedUtilityAncestors,
        required_checkpoint_publications: &BTreeSet<String>,
    ) -> Result<(), ArenaError> {
        owners.validate_current(self, required_checkpoint_publications)?;
        self.ancestry_checked.store(true, Ordering::Release);
        Ok(())
    }
    pub fn observation(&self) -> Value {
        json!({"schema":"rz-rovent-historical-utility-dataset-admission/1",
            "dataset_sha256":self.dataset_sha256,"pool_sha256":self.pool_sha256,
            "preparation_sha256":self.preparation_sha256,"context_ids":self.context_ids,
            "record_count":1,"trial_count":self.publication.trial_count,"pair_count":self.publication.pair_count,
            "split":self.evidence.split,"opening_id":self.evidence.opening_id,"game_id":self.evidence.game_id,
            "line_genealogy_id":self.evidence.line_genealogy_id,"transposition_sha256":self.evidence.transposition_sha256,
            "rules_state_sha256":self.evidence.rules_state_sha256,
            "full_ancestor_owner_joined":self.ancestry_checked.load(Ordering::Acquire),
            "unique_known_pairs":self.context.unique_known_pairs,
            "source_kind":"historical_exact_cohort_pins","past_native_authority_reissued":false,
            "past_physical_completion_reissued":false,"execution_authorized":false,
            "frozen_pc_scope":"not_instantiated_pinned_original_source_preserved",
            "collection_io":self.durable_ack.collection_io,
            "retained_bytes":self.held.iter().map(|f|f.pin.bytes).sum::<u64>(),
            "retained_bytes_scope":"original_file_bytes_not_process_peak_memory"})
    }
}
