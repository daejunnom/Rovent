//! Historical R auxiliary data admission from an externally pinned publication.
//! This owner retains original file descriptors and replayed row metadata. It
//! never reconstructs a worker, Bound, native run, physical fence or trainer.
//! NNUEC checkpoint parsing checks parameters and declared capabilities only;
//! no model forward is performed by this reader.

use super::*;
use crate::rovent_collect::{
    ROVENT_R_RETIREMENT_OBSERVATION_DOMAIN, RoventNativeOriginEvidence, RoventRCorpusPublication,
    RoventRPublicationAdmission, RoventRootAdmissionEvidence, verify_historical_r_readout,
};
use rz_experiments::rovent_corpus::{MAX_CORPUS_BYTES, MAX_CORPUS_ROWS};
use rz_experiments::rovent_r_readout::*;
use rz_search::cpu_value::{
    CpuTrainingState,
    nnuec::{NnuecConfig, v2},
};
use sha2::{Digest as _, Sha256};
use std::collections::VecDeque;

pub const ROVENT_HISTORICAL_R_AUXILIARY_DATASET: &str =
    "rz-rovent-historical-r-auxiliary-dataset/1";
const RECIPE_DOMAIN: &str = "rz-rovent-generation-series/1";
const MAX_GLOBAL_METADATA_BYTES: u64 = 64 << 20;
const MAX_GLOBAL_EDGES: usize = 1_048_576;
// R's complete original PC closure (256) plus the actual current PC and its
// separate bounded ancestry (257). Row/metadata/edge quotas still apply.
const MAX_JOINT_PC_OWNERS: usize = 513;

/// An explicit request to verify archived bytes. Original model artifact paths
/// remain source declarations; only archive-relative files are opened here.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventHistoricalRAuxiliaryDatasetRegistration {
    pub root: PathBuf,
    pub publication: RoventTrainingFilePin,
    pub collection_source_commit: String,
    pub collection_source_sha256: String,
    pub collection_runtime_library_sha256: String,
    pub collection_model_semantics_sha256: String,
    pub collection_model_epoch: u64,
    pub collection_context_epoch: u64,
    pub collection_recipe_sha256: String,
    pub collection_stage_id: String,
    pub nnuec: rz_experiments::RoventNnuecPin,
    /// Parsed numerical parameter identity, distinct from weights.sha256.
    pub nnuec_parameter_sha256: String,
}
impl RoventHistoricalRAuxiliaryDatasetRegistration {
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            self.root.is_absolute()
                && self
                    .root
                    .components()
                    .all(|c| !matches!(c, Component::ParentDir | Component::CurDir))
                && self.publication.path == self.root.join("r-publication.json")
                && commit(&self.collection_source_commit)
                && [
                    &self.collection_source_sha256,
                    &self.collection_runtime_library_sha256,
                    &self.collection_model_semantics_sha256,
                    &self.collection_recipe_sha256,
                    &self.nnuec_parameter_sha256,
                    &self.nnuec.implementation_sha256,
                    &self.nnuec.configuration_sha256,
                    &self.nnuec.context_namespace_policy_sha256,
                    &self.nnuec.local_namespace_policy_sha256,
                    &self.nnuec.search_conditions_sha256,
                ]
                .iter()
                .all(|s| generation_sha(s))
                && self.collection_model_epoch > 0
                && self.collection_context_epoch > 0
                && generation_identifier(&self.collection_stage_id)
                && self.nnuec.schema == v2::SCHEMA
                && self.nnuec.semantics == v2::SEMANTICS
                && self.nnuec.encoding == v2::INPUT_SCHEMA
                && (1..=32 << 20).contains(&self.nnuec.weights.bytes)
                && (1..=256 << 10).contains(&self.nnuec.provenance.bytes),
            "strict historical R collection/model/raw-file registration",
        )?;
        self.nnuec
            .weights
            .validate()
            .map_err(|e| invalid(e.to_string()))?;
        self.nnuec
            .provenance
            .validate()
            .map_err(|e| invalid(e.to_string()))?;
        GenerationTrainingFilePinV3 {
            path: self.publication.path.clone(),
            sha256: self.publication.sha256.clone(),
            bytes: self.publication.bytes,
        }
        .validate(MAX_METADATA_BYTES)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RoventHistoricalRAuxiliaryOccurrence {
    pub process_epoch: u64,
    pub game_generation: u64,
    pub root_generation: u64,
    pub job_sequence: u64,
    pub worker_owner: u16,
    pub frame_sequence: u64,
    pub capture_sequence: u64,
    pub readout_sequence: u64,
}
/// Small immutable data observation. Construction is private to complete
/// publication admission, and this type carries no executable capability.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RoventHistoricalRAuxiliaryRow {
    pub record_sha256: String,
    pub origin_sha256: String,
    pub semantic_input_sha256: String,
    pub native_input_sha256: String,
    pub retirement_sha256: String,
    pub wdl_raw_receipt_sha256: Option<String>,
    pub valid_exact_trajectory: bool,
    pub occurrence: RoventHistoricalRAuxiliaryOccurrence,
}

/// No Deserialize or public success constructor. Empty retirements are checked
/// independently and remain in the original archive without creating rows.
pub struct VerifiedRoventRAuxiliaryDataset {
    registration: Arc<RoventHistoricalRAuxiliaryDatasetRegistration>,
    publication: Arc<RoventRCorpusPublication>,
    manifest: Arc<RoventRReadoutManifest>,
    index: Arc<RoventRReadoutIndex>,
    rows: Arc<BTreeMap<String, RoventHistoricalRAuxiliaryRow>>,
    held: Arc<Vec<HeldHistoricalFile>>,
    pc_datasets: Vec<Arc<VerifiedRoventDataset>>,
    ancestors: Vec<Arc<Self>>,
    lineage: Arc<Vec<LineageEntry>>,
    eligible_r_records: BTreeSet<String>,
    joint_current_pc: Option<Arc<VerifiedRoventDataset>>,
    joint_pc_datasets: Vec<Arc<VerifiedRoventDataset>>,
    joint_eligible_current_pc_origins: BTreeSet<String>,
    joint_eligible_current_pc_records: BTreeSet<String>,
    joint_pc_origin_inputs: BTreeMap<String, String>,
    joint_r_origin_inputs: BTreeMap<String, String>,
}
impl VerifiedRoventRAuxiliaryDataset {
    pub fn open(
        registration: RoventHistoricalRAuxiliaryDatasetRegistration,
        recipe: &Value,
        pc_datasets: Vec<Arc<VerifiedRoventDataset>>,
        ancestors: Vec<Arc<Self>>,
        budget: &RoventRecoveryReadBudget,
    ) -> Result<Self, ArenaError> {
        registration.validate()?;
        budget.reserve(0)?;
        check(
            pc_datasets.len() <= 256 && ancestors.len() <= 256,
            "bounded complete original PC/R ancestor owners",
        )?;
        let graph = RecipeGraph::parse(recipe, &registration.collection_recipe_sha256)?;
        let (publication_file, publication_raw) =
            budget.read(&registration.publication, MAX_METADATA_BYTES, true)?;
        let publication: RoventRCorpusPublication = decode(&publication_raw)?;
        publication.validate()?;
        check(
            publication.source_commit == registration.collection_source_commit
                && publication.source_sha256 == registration.collection_source_sha256
                && publication.runtime_sha256 == registration.collection_runtime_library_sha256,
            "R publication changed registered collection source/runtime",
        )?;
        let mut files = ArchiveFiles {
            root: &registration.root,
            held: vec![publication_file],
            paths: BTreeMap::new(),
            bytes: 0,
        };
        let manifest_raw = files.read(&publication.manifest, MAX_METADATA_BYTES, budget)?;
        let index_raw = files.read(&publication.index, MAX_METADATA_BYTES, budget)?;
        let manifest: RoventRReadoutManifest = decode(&manifest_raw)?;
        let index: RoventRReadoutIndex = decode(&index_raw)?;
        manifest.validate().map_err(|e| invalid(e.to_string()))?;
        index.validate().map_err(|e| invalid(e.to_string()))?;
        check(
            manifest.sha256().map_err(|e| invalid(e.to_string()))?
                == publication.canonical_manifest_sha256
                && index.sha256().map_err(|e| invalid(e.to_string()))?
                    == publication.canonical_index_sha256
                && index.manifest_sha256 == publication.canonical_manifest_sha256
                && index.record_count == publication.record_count
                && index.chunks.len() == publication.chunks.len(),
            "R manifest/index canonical seals/counts differ from original publication",
        )?;
        let c = &manifest.collection;
        check(
            c.source_commit == registration.collection_source_commit
                && c.source_sha256 == registration.collection_source_sha256
                && c.runtime_sha256 == registration.collection_runtime_library_sha256
                && c.model_semantics_sha256 == registration.collection_model_semantics_sha256
                && c.model_epoch == registration.collection_model_epoch
                && c.context_epoch == registration.collection_context_epoch
                && c.recipe_sha256 == registration.collection_recipe_sha256
                && c.stage_id == registration.collection_stage_id,
            "R collection declaration differs from caller pins",
        )?;
        graph.collection(c)?;
        check_ancestor_sets(&manifest, &pc_datasets, &ancestors)?;
        for owner in &pc_datasets {
            owner.revalidate(budget)?;
        }
        for owner in &ancestors {
            owner.revalidate_own(budget)?;
        }

        let mut admissions = BTreeMap::new();
        let mut parsed_models: BTreeMap<(String, String), ParsedNnuec> = BTreeMap::new();
        for admitted in &publication.admissions {
            let raw = files.read(&admitted.input_lock, MAX_METADATA_BYTES, budget)?;
            let lock = rz_experiments::RoventInputLock::from_json(
                std::str::from_utf8(&raw).map_err(|_| invalid("R full lock UTF8"))?,
            )
            .map_err(|e| invalid(e.to_string()))?;
            check(
                lock.canonical_sha256 == admitted.canonical_input_lock_sha256
                    && lock.manifest.engine.nnuec.as_ref() == Some(&registration.nnuec),
                "R full original lock/model declaration changed",
            )?;
            let bundle = admitted
                .runtime_bundle_manifest
                .as_ref()
                .map(|pin| {
                    check(
                        lock.manifest
                            .provenance
                            .iter()
                            .any(|p| p.sha256 == pin.sha256 && p.bytes == pin.bytes),
                        "R original bundle raw artifact absent from full lock provenance",
                    )?;
                    let raw = files.read(pin, 64 << 10, budget)?;
                    rz_eval::runtime_pin::CudaRuntimeBundleSpec::from_json(
                        std::str::from_utf8(&raw).map_err(|_| invalid("R bundle UTF8"))?,
                    )
                    .map_err(|e| invalid(e.to_string()))
                })
                .transpose()?;
            let key = (
                admitted.nnuec_weights.sha256.clone(),
                admitted.nnuec_provenance.sha256.clone(),
            );
            if !parsed_models.contains_key(&key) {
                let weights = files.read(&admitted.nnuec_weights, 32 << 20, budget)?;
                let provenance = files.read(&admitted.nnuec_provenance, 256 << 10, budget)?;
                let model =
                    ParsedNnuec::parse(&weights, &provenance, &registration, admitted, &lock)?;
                parsed_models.insert(key.clone(), model);
            }
            let model = parsed_models
                .get(&key)
                .ok_or_else(|| invalid("R parsed model owner missing"))?;
            model.join(admitted, &lock, &registration)?;
            check(
                admissions
                    .insert(
                        admitted.native_admission_sha256.clone(),
                        Admission {
                            lock,
                            bundle,
                            model_key: key,
                        },
                    )
                    .is_none(),
                "duplicate R original admission",
            )?;
        }
        let mut retired = BTreeMap::new();
        let mut used_admissions = BTreeSet::new();
        let mut capture_retirements = BTreeMap::new();
        for declaration in &publication.retirements {
            let raw = files.read(&declaration.file, MAX_METADATA_BYTES, budget)?;
            let observation: RetirementObservation = decode(&raw)?;
            check(
                observation.schema == ROVENT_R_RETIREMENT_OBSERVATION_DOMAIN
                    && observation.retirement_sha256 == declaration.retirement_sha256
                    && observation.readout_record_sha256 == declaration.readout_record_sha256,
                "R retirement observation changed full original readout coverage",
            )?;
            let admission = admissions
                .get(&observation.native_admission_sha256)
                .ok_or_else(|| invalid("R retirement has no original full lock"))?;
            verify_retirement(&observation, admission, &registration)?;
            let t = &observation.retirement;
            let a = &t.job.authority;
            let capture = (
                a.process_epoch,
                a.game_generation,
                a.root_generation,
                t.job.job_sequence,
                t.worker_owner,
                t.frame_sequence,
                t.capture_sequence,
            );
            check(
                capture_retirements
                    .insert(capture, observation.retirement_sha256.clone())
                    .is_none(),
                "same original R capture has duplicate/conflicting retirement observations",
            )?;
            used_admissions.insert(observation.native_admission_sha256.clone());
            check(
                retired
                    .insert(
                        observation.retirement_sha256.clone(),
                        RetirementCoverage {
                            original: observation,
                            rows: BTreeSet::new(),
                            attempts: BTreeSet::new(),
                            bytes: 0,
                        },
                    )
                    .is_none(),
                "duplicate R retirement observation",
            )?;
        }
        let origins = index
            .origins
            .iter()
            .map(|o| (o.record_sha256.as_str(), o))
            .collect::<BTreeMap<_, _>>();
        check(
            origins.len() == index.origins.len(),
            "duplicate R record index occurrence",
        )?;
        let mut rows = BTreeMap::new();
        let mut lineage = Vec::new();
        let mut execution_occurrences = BTreeSet::new();
        for (declared, pin) in index.chunks.iter().zip(&publication.chunks) {
            check(
                declared.path == pin.path
                    && declared.sha256 == pin.sha256
                    && declared.bytes == pin.bytes,
                "R publication/index chunk exact bytes changed",
            )?;
            let raw = files.read(pin, MAX_CHUNK_BYTES as u64, budget)?;
            let text = std::str::from_utf8(&raw).map_err(|_| invalid("R chunk UTF8"))?;
            let mut count = 0u64;
            for line in text.lines() {
                budget.reserve(0)?;
                check(
                    !line.is_empty()
                        && line.len() <= MAX_R_READOUT_BYTES
                        && count < declared.records,
                    "bounded R JSONL original row",
                )?;
                let row: RoventRReadoutRecord =
                    rz_experiments::decode_json(line).map_err(|e| invalid(e.to_string()))?;
                let indexed = origins
                    .get(row.record_sha256.as_str())
                    .ok_or_else(|| invalid("R chunk row missing from full index"))?;
                check(
                    **indexed == RoventRReadoutOrigin::from_record(&row),
                    "R index changed original occurrence",
                )?;
                let admitted = admissions
                    .get(&row.native_admission_sha256)
                    .ok_or_else(|| invalid("R row original admission missing"))?;
                let parsed = parsed_models
                    .get(&admitted.model_key)
                    .ok_or_else(|| invalid("R model metadata missing"))?;
                let evidence =
                    verify_historical_r_readout(&row, &manifest, &admitted.lock, &parsed.config)?;
                check(
                    evidence.nnuec_semantics == registration.nnuec.semantics
                        && evidence.nnuec_parameter_sha256 == registration.nnuec_parameter_sha256
                        && evidence.nnuec_weights_file_sha256 == registration.nnuec.weights.sha256
                        && evidence.nnuec_training_state == parsed.training,
                    "R historical row differs from parsed original model parameters",
                )?;
                let coverage = retired
                    .get_mut(&row.retirement_sha256)
                    .ok_or_else(|| invalid("R row has no original retirement"))?;
                check(
                    coverage.original.native_admission == row.native_admission
                        && coverage.original.native_admission_sha256 == row.native_admission_sha256
                        && coverage.original.retirement == row.retirement
                        && coverage
                            .original
                            .readout_record_sha256
                            .binary_search(&row.record_sha256)
                            .is_ok()
                        && coverage.rows.insert(row.record_sha256.clone())
                        && coverage.attempts.insert(row.question.readout_sequence),
                    "R row/retirement/source coverage mismatch or duplicate attempt",
                )?;
                coverage.bytes = coverage
                    .bytes
                    .checked_add(row.input.reference_input_bytes)
                    .ok_or_else(|| invalid("R retained input bytes overflow"))?;
                let occurrence = occurrence(&row);
                check(
                    execution_occurrences.insert(occurrence_key(&occurrence)),
                    "duplicate original R execution/score attempt",
                )?;
                let entry = LineageEntry::from_r(&row, c);
                graph.entry(&entry)?;
                lineage.push(entry);
                let record_sha = row.record_sha256.clone();
                check(
                    rows.insert(
                        record_sha,
                        RoventHistoricalRAuxiliaryRow {
                            record_sha256: row.record_sha256,
                            origin_sha256: row.origin_sha256,
                            semantic_input_sha256: row.input.semantic_input_sha256,
                            native_input_sha256: row.input.native_input_sha256,
                            retirement_sha256: row.retirement_sha256,
                            wdl_raw_receipt_sha256: row
                                .wdl
                                .as_ref()
                                .map(|w| w.raw_receipt_sha256.clone()),
                            valid_exact_trajectory: evidence.rules_wdl.is_some(),
                            occurrence,
                        },
                    )
                    .is_none(),
                    "duplicate R row across original chunks",
                )?;
                count += 1;
            }
            check(
                count == declared.records,
                "R original chunk row count mismatch",
            )?;
        }
        for coverage in retired.values() {
            check(
                coverage.rows.iter().cloned().collect::<Vec<_>>()
                    == coverage.original.readout_record_sha256
                    && coverage.rows.len()
                        == usize::from(coverage.original.retirement.retained_readouts)
                    && coverage.bytes == coverage.original.retirement.retained_input_bytes,
                "R full retirement/readout/empty/input-byte coverage did not close",
            )?;
        }
        check(
            rows.len() as u64 == publication.record_count
                && rows.len() == index.origins.len()
                && used_admissions == admissions.keys().cloned().collect()
                && files.bytes == publication.total_payload_bytes
                && files.bytes <= MAX_CORPUS_BYTES,
            "R full original payload/row/admission coverage did not close",
        )?;
        let eligible = global_training_rows(
            &graph,
            &lineage,
            &manifest,
            &pc_datasets,
            &ancestors,
            budget,
        )?;
        budget.reserve(0)?;
        let ArchiveFiles { held, .. } = files;
        Ok(Self {
            registration: Arc::new(registration),
            publication: Arc::new(publication),
            manifest: Arc::new(manifest),
            index: Arc::new(index),
            rows: Arc::new(rows),
            held: Arc::new(held),
            pc_datasets,
            ancestors,
            lineage: Arc::new(lineage),
            eligible_r_records: eligible.r_records,
            joint_current_pc: None,
            joint_pc_datasets: Vec::new(),
            joint_eligible_current_pc_origins: BTreeSet::new(),
            joint_eligible_current_pc_records: BTreeSet::new(),
            joint_pc_origin_inputs: BTreeMap::new(),
            joint_r_origin_inputs: BTreeMap::new(),
        })
    }
    pub fn registration(&self) -> &RoventHistoricalRAuxiliaryDatasetRegistration {
        &self.registration
    }
    pub(in crate::rovent_training_process) fn utility_lineage(
        &self,
    ) -> impl Iterator<Item = HistoricalDatasetLineage<'_>> {
        self.lineage.iter().map(|row| HistoricalDatasetLineage {
            split: row.split,
            opening_id: &row.opening_id,
            game_id: &row.game_id,
            line_genealogy_id: &row.genealogy_id,
            rules_state_sha256: &row.rules_state_sha256,
            transposition_sha256: &row.transposition_sha256,
        })
    }
    pub fn publication_pin(&self) -> &RoventTrainingFilePin {
        &self.registration.publication
    }
    pub fn manifest_file_sha256(&self) -> &str {
        &self.publication.manifest.sha256
    }
    pub fn index_file_sha256(&self) -> &str {
        &self.publication.index.sha256
    }
    pub(in crate::rovent_training_process) fn canonical_manifest_sha256(&self) -> &str {
        &self.publication.canonical_manifest_sha256
    }
    pub(in crate::rovent_training_process) fn canonical_index_sha256(&self) -> &str {
        &self.publication.canonical_index_sha256
    }
    pub fn ancestors(&self) -> &[Arc<Self>] {
        &self.ancestors
    }
    pub fn pc_datasets(&self) -> &[Arc<VerifiedRoventDataset>] {
        &self.pc_datasets
    }
    /// Binds the immutable historical source to one actual current PC owner.
    /// This does not add that PC dataset to the original publication ancestry
    /// or mutate another checkpoint's pool. File/row source custody is shared
    /// through Arc; only this joint scope's eligibility sets are newly built.
    pub fn for_current_pc(
        self: &Arc<Self>,
        current_pc: Arc<VerifiedRoventDataset>,
        recipe: &Value,
        budget: &RoventRecoveryReadBudget,
    ) -> Result<Arc<Self>, ArenaError> {
        let graph = RecipeGraph::parse(recipe, &self.registration.collection_recipe_sha256)?;
        check(
            current_pc.manifest.stage_id == self.manifest.collection.stage_id
                && current_pc.manifest.context_epoch == self.manifest.collection.context_epoch,
            "joint current P/C and R must have the same stage/context; their model epochs stay independent",
        )?;
        self.revalidate(budget)?;
        let mut pc_by_index: BTreeMap<String, Arc<VerifiedRoventDataset>> = BTreeMap::new();
        let mut pending = self
            .pc_datasets
            .iter()
            .cloned()
            .chain(std::iter::once(Arc::clone(&current_pc)))
            .collect::<Vec<_>>();
        while let Some(owner) = pending.pop() {
            let sha = owner.index.sha256().map_err(|e| invalid(e.to_string()))?;
            if pc_by_index.contains_key(&sha) {
                continue;
            }
            check(
                pc_by_index.len() < MAX_JOINT_PC_OWNERS,
                "joint current-PC original index closure exceeds its combined owner quota",
            )?;
            owner.revalidate(budget)?;
            pending.extend(owner.ancestors.iter().cloned());
            pc_by_index.insert(sha, owner);
        }
        let joint_pc_datasets = pc_by_index.into_values().collect::<Vec<_>>();
        let eligible = global_training_rows(
            &graph,
            &self.lineage,
            &self.manifest,
            &joint_pc_datasets,
            &self.ancestors,
            budget,
        )?;
        let current_origins = current_pc
            .index
            .origins
            .iter()
            .map(|o| o.origin_sha256.as_str())
            .collect::<BTreeSet<_>>();
        let current_records = current_pc
            .index
            .origins
            .iter()
            .map(|o| o.record_sha256.as_str())
            .collect::<BTreeSet<_>>();
        let joint_eligible_current_pc_origins = eligible
            .pc_origins
            .into_iter()
            .filter(|s| current_origins.contains(s.as_str()))
            .collect();
        let joint_eligible_current_pc_records = eligible
            .pc_records
            .into_iter()
            .filter(|s| current_records.contains(s.as_str()))
            .collect();
        Ok(Arc::new(Self {
            registration: Arc::clone(&self.registration),
            publication: Arc::clone(&self.publication),
            manifest: Arc::clone(&self.manifest),
            index: Arc::clone(&self.index),
            rows: Arc::clone(&self.rows),
            held: Arc::clone(&self.held),
            pc_datasets: self.pc_datasets.clone(),
            ancestors: self.ancestors.clone(),
            lineage: Arc::clone(&self.lineage),
            eligible_r_records: eligible.r_records,
            joint_current_pc: Some(current_pc),
            joint_pc_datasets,
            joint_eligible_current_pc_origins,
            joint_eligible_current_pc_records,
            joint_pc_origin_inputs: eligible.pc_origin_inputs,
            joint_r_origin_inputs: eligible.r_origin_inputs,
        }))
    }
    pub fn joint_current_pc_publication_sha256(&self) -> Result<&str, ArenaError> {
        self.joint_current_pc
            .as_ref()
            .map(|p| p.registration.publication.sha256.as_str())
            .ok_or_else(|| invalid("R auxiliary owner is not bound to a current PC dataset"))
    }
    pub fn joint_pc_datasets(&self) -> &[Arc<VerifiedRoventDataset>] {
        &self.joint_pc_datasets
    }
    pub fn joint_eligible_current_pc_origins(&self) -> Result<&BTreeSet<String>, ArenaError> {
        check(
            self.joint_current_pc.is_some(),
            "R current-PC joint eligibility is absent",
        )?;
        Ok(&self.joint_eligible_current_pc_origins)
    }
    pub fn joint_eligible_current_pc_records(&self) -> Result<&BTreeSet<String>, ArenaError> {
        check(
            self.joint_current_pc.is_some(),
            "R current-PC joint eligibility is absent",
        )?;
        Ok(&self.joint_eligible_current_pc_records)
    }
    /// Checks retained actual exposure against this joint source/split graph.
    /// The original checkpoint/update owner supplies independently verified,
    /// namespace-separated sets; no optimizer completion is invented here.
    pub fn validate_joint_retained_training(
        &self,
        pc_inputs: &BTreeSet<String>,
        pc_origins: &BTreeSet<String>,
        r_inputs: &BTreeSet<String>,
        r_origins: &BTreeSet<String>,
    ) -> Result<(), ArenaError> {
        check(
            self.joint_current_pc.is_some(),
            "retained training requires current-PC joint admission",
        )?;
        for (inputs, origins, known) in [
            (pc_inputs, pc_origins, &self.joint_pc_origin_inputs),
            (r_inputs, r_origins, &self.joint_r_origin_inputs),
        ] {
            check(
                inputs.len() <= MAX_CORPUS_ROWS
                    && origins.len() <= MAX_CORPUS_ROWS
                    && inputs.iter().chain(origins).all(|s| generation_sha(s)),
                "bounded separate original retained training seals",
            )?;
            let actual = origins.iter().map(|origin| known.get(origin).cloned().ok_or_else(|| invalid(
                "retained training origin is absent/excluded/holdout/future/sibling in its original PC/R namespace")))
                .collect::<Result<BTreeSet<_>, _>>()?;
            check(
                actual == *inputs,
                "joint retained original origin/input projection differs",
            )?;
        }
        Ok(())
    }
    pub fn auxiliary_row(
        &self,
        record_sha256: &str,
    ) -> Result<&RoventHistoricalRAuxiliaryRow, ArenaError> {
        let row = self
            .rows
            .get(record_sha256)
            .ok_or_else(|| invalid("R original row is absent from this dataset owner"))?;
        check(
            self.joint_current_pc.is_some() && self.eligible_r_records.contains(record_sha256),
            "validation/holdout/globally excluded R row cannot enter optimization",
        )?;
        Ok(row)
    }
    pub fn valid_exact_trajectory(&self, record_sha256: &str) -> Result<bool, ArenaError> {
        Ok(self.auxiliary_row(record_sha256)?.valid_exact_trajectory)
    }
    pub fn original_auxiliary_evidence(&self, record_sha256: &str) -> Result<Value, ArenaError> {
        let row = self.auxiliary_row(record_sha256)?;
        Ok(
            json!({"schema":"rz-rovent-r-auxiliary-evidence/1","source_domain":R_READOUT_RECORD_DOMAIN,
            "record_sha256":row.record_sha256,"origin_sha256":row.origin_sha256,
            "native_input_sha256":row.native_input_sha256,"semantic_input_sha256":row.semantic_input_sha256,
            "occurrence":row.occurrence,"retirement_sha256":row.retirement_sha256,
            "wdl_raw_sha256":row.wdl_raw_receipt_sha256,"execution_authorized":false}),
        )
    }
    pub fn expected_auxiliary_evidence(&self, record_sha256: &str) -> Result<Value, ArenaError> {
        self.original_auxiliary_evidence(record_sha256)
    }
    /// Python's pool follows original chunk/row order, not sorted record SHA
    /// or the independently published index-origin order. `lineage` is built
    /// in that same complete read order before small row metadata is retained.
    pub fn auxiliary_pool_sha256(&self) -> Result<String, ArenaError> {
        check(
            self.joint_current_pc.is_some(),
            "R auxiliary pool requires original current-PC joint admission",
        )?;
        let occurrences = self
            .lineage
            .iter()
            .filter(|e| {
                self.rows
                    .get(&e.record_sha256)
                    .is_some_and(|r| self.eligible_r_records.contains(&r.record_sha256))
            })
            .map(|e| {
                json!({"source_domain":R_READOUT_RECORD_DOMAIN,"record_sha256":e.record_sha256,
                "input_sha256":e.input_sha256,"origin_sha256":e.origin_sha256})
            })
            .collect::<Vec<_>>();
        generation_digest(
            "rz-rovent-r-auxiliary-pool/1",
            &json!({"index_sha256":self.publication.canonical_index_sha256,
            "occurrences":occurrences}),
        )
        .map_err(|e| invalid(e.to_string()))
    }
    pub fn observation(&self) -> Value {
        json!({"schema":ROVENT_HISTORICAL_R_AUXILIARY_DATASET,
            "publication_sha256":self.registration.publication.sha256,
            "manifest_file_sha256":self.manifest_file_sha256(),"index_file_sha256":self.index_file_sha256(),
            "canonical_manifest_sha256":self.publication.canonical_manifest_sha256,
            "canonical_index_sha256":self.publication.canonical_index_sha256,
            "record_count":self.rows.len(),"retirement_count":self.publication.retirements.len(),
            "current_pc_joint_admitted":self.joint_current_pc.is_some(),
            "optimization_eligible_count":if self.joint_current_pc.is_some(){self.eligible_r_records.len()}else{0},
            "valid_exact_trajectory_training_count":self.rows.values().filter(|r|self.joint_current_pc.is_some()&&self.eligible_r_records.contains(&r.record_sha256)&&r.valid_exact_trajectory).count(),
            "historical_data_integrity_only":true,"actual_rules_replay_verified":true,
            "backend_evidence_scope":"original_backend_observation_pinned_not_reexecuted",
            "export_manifest_scope":"original_lock_raw_artifact_pin_not_reparsed",
            "actual_provider_reexecuted":false,
            "model_forward_performed":false,"past_native_authority_reissued":false,
            "past_physical_completion_reissued":false,"execution_authorized":false})
    }
    pub fn revalidate(&self, budget: &RoventRecoveryReadBudget) -> Result<(), ArenaError> {
        // The original manifest is a complete transitive index closure. Check
        // each retained source once rather than recursively charging the same
        // ancestor archive once per path through that graph.
        self.revalidate_own(budget)?;
        for owner in &self.ancestors {
            owner.revalidate_own(budget)?;
        }
        for owner in if self.joint_current_pc.is_some() {
            &self.joint_pc_datasets
        } else {
            &self.pc_datasets
        } {
            owner.revalidate(budget)?;
        }
        Ok(())
    }
    fn revalidate_own(&self, budget: &RoventRecoveryReadBudget) -> Result<(), ArenaError> {
        for retained in self.held.iter() {
            budget.verify_held(&retained.file, &retained.pin, false)?;
            let (named, _) = budget.read(&retained.pin, retained.pin.bytes, false)?;
            check(
                named.identity == retained.identity,
                "R historical retained/named file identity changed",
            )?;
        }
        Ok(())
    }
}

struct ArchiveFiles<'a> {
    root: &'a Path,
    held: Vec<HeldHistoricalFile>,
    paths: BTreeMap<String, RoventPublicationFile>,
    bytes: u64,
}
impl ArchiveFiles<'_> {
    fn read(
        &mut self,
        pin: &RoventPublicationFile,
        maximum: u64,
        budget: &RoventRecoveryReadBudget,
    ) -> Result<Vec<u8>, ArenaError> {
        if let Some(old) = self.paths.get(&pin.path) {
            check(old == pin, "R archive path has conflicting raw file pins")?;
            let held = self
                .held
                .iter()
                .find(|h| h.pin.path == self.root.join(&pin.path))
                .ok_or_else(|| invalid("R retained source FD missing"))?;
            return budget.verify_held(&held.file, &held.pin, true);
        }
        let (held, raw) = budget.read(&archive_pin(self.root, pin)?, maximum, true)?;
        self.bytes = self
            .bytes
            .checked_add(pin.bytes)
            .filter(|n| *n <= MAX_CORPUS_BYTES)
            .ok_or_else(|| invalid("R archive full payload quota"))?;
        self.paths.insert(pin.path.clone(), pin.clone());
        self.held.push(held);
        Ok(raw)
    }
}
struct Admission {
    lock: rz_experiments::RoventInputLock,
    bundle: Option<rz_eval::runtime_pin::CudaRuntimeBundleSpec>,
    model_key: (String, String),
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RetirementObservation {
    schema: String,
    native_admission: Value,
    native_admission_sha256: String,
    retirement: RoventRCaptureRetirement,
    retirement_sha256: String,
    readout_record_sha256: Vec<String>,
}
struct RetirementCoverage {
    original: RetirementObservation,
    rows: BTreeSet<String>,
    attempts: BTreeSet<u64>,
    bytes: u64,
}
fn occurrence(row: &RoventRReadoutRecord) -> RoventHistoricalRAuxiliaryOccurrence {
    let q = &row.question;
    let a = &q.job.authority;
    RoventHistoricalRAuxiliaryOccurrence {
        process_epoch: a.process_epoch,
        game_generation: a.game_generation,
        root_generation: a.root_generation,
        job_sequence: q.job.job_sequence,
        worker_owner: q.worker_owner,
        frame_sequence: q.frame_sequence,
        capture_sequence: q.capture_sequence,
        readout_sequence: q.readout_sequence,
    }
}
fn occurrence_key(
    o: &RoventHistoricalRAuxiliaryOccurrence,
) -> (u64, u64, u64, u64, u16, u64, u64, u64) {
    (
        o.process_epoch,
        o.game_generation,
        o.root_generation,
        o.job_sequence,
        o.worker_owner,
        o.frame_sequence,
        o.capture_sequence,
        o.readout_sequence,
    )
}
fn commit(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NnuecProvenance {
    domain: String,
    weights_raw_sha256: String,
    semantic_weights_sha256: String,
    schema: String,
    semantics: String,
    encoding: String,
    implementation_sha256: String,
    configuration_sha256: String,
    context_namespace_policy_sha256: String,
    local_namespace_policy_sha256: String,
    training: CpuTrainingState,
    diagnostic: bool,
    provenance: Vec<rz_experiments::ArtifactRef>,
}
struct ParsedNnuec {
    config: NnuecConfig,
    training: CpuTrainingState,
    registration: Value,
    provenance: NnuecProvenance,
}
impl ParsedNnuec {
    fn parse(
        weights: &[u8],
        provenance: &[u8],
        registration: &RoventHistoricalRAuxiliaryDatasetRegistration,
        admitted: &RoventRPublicationAdmission,
        lock: &rz_experiments::RoventInputLock,
    ) -> Result<Self, ArenaError> {
        let checkpoint: v2::Checkpoint = decode(weights)?;
        checkpoint.validate().map_err(|e| invalid(e.to_string()))?;
        check(
            checkpoint
                .weights_sha256()
                .map_err(|e| invalid(e.to_string()))?
                == registration.nnuec_parameter_sha256,
            "R original checkpoint parameters differ from registered semantic parameter identity",
        )?;
        let config = checkpoint.base.config.clone();
        let training = checkpoint.base.training.clone();
        // Constructs a bounded parsed-parameter descriptor only. No board,
        // accumulator, evaluation, prediction or forward is requested.
        let model = v2::Model::new(checkpoint).map_err(|e| invalid(e.to_string()))?;
        let descriptor = model.registration().map_err(|e| invalid(e.to_string()))?;
        let mut actual = serde_json::to_value(&descriptor).map_err(|e| invalid(e.to_string()))?;
        // Historical implementation is an original source pin. A newer reader
        // must not relabel it with its own compiled implementation SHA. The
        // fixed semantic/configuration/policy fields are checked independently.
        actual["implementation_sha256"] = json!(registration.nnuec.implementation_sha256);
        let body: NnuecProvenance = decode(provenance)?;
        let result = Self {
            config,
            training,
            registration: actual,
            provenance: body,
        };
        result.join(admitted, lock, registration)?;
        Ok(result)
    }
    fn join(
        &self,
        admitted: &RoventRPublicationAdmission,
        lock: &rz_experiments::RoventInputLock,
        registration: &RoventHistoricalRAuxiliaryDatasetRegistration,
    ) -> Result<(), ArenaError> {
        let p = &registration.nnuec;
        let body = &self.provenance;
        check(
            admitted.nnuec_weights.sha256 == p.weights.sha256
                && admitted.nnuec_weights.bytes == p.weights.bytes
                && admitted.nnuec_provenance.sha256 == p.provenance.sha256
                && admitted.nnuec_provenance.bytes == p.provenance.bytes
                && admitted.nnuec_parameter_sha256 == registration.nnuec_parameter_sha256
                && admitted.nnuec_registration == self.registration
                && self.registration["schema"] == p.schema
                && self.registration["semantics"] == p.semantics
                && self.registration["encoding"] == p.encoding
                && self.registration["configuration_sha256"] == p.configuration_sha256
                && self.registration["context_namespace_policy_sha256"]
                    == p.context_namespace_policy_sha256
                && self.registration["local_namespace_policy_sha256"]
                    == p.local_namespace_policy_sha256
                && body.domain == "rz-rovent-nnuec-provenance/1"
                && body.weights_raw_sha256 == p.weights.sha256
                && body.semantic_weights_sha256 == registration.nnuec_parameter_sha256
                && body.schema == p.schema
                && body.semantics == p.semantics
                && body.encoding == p.encoding
                && body.implementation_sha256 == p.implementation_sha256
                && body.configuration_sha256 == p.configuration_sha256
                && body.context_namespace_policy_sha256 == p.context_namespace_policy_sha256
                && body.local_namespace_policy_sha256 == p.local_namespace_policy_sha256
                && body.training == self.training
                && body.diagnostic == p.diagnostic
                && body.provenance.len() <= 32,
            "R raw file / parsed parameter / provenance / original registration identity mismatch",
        )?;
        let mut paths = BTreeSet::new();
        let mut contents = BTreeSet::new();
        for artifact in &body.provenance {
            artifact.validate().map_err(|e| invalid(e.to_string()))?;
            check(
                paths.insert(&artifact.path)
                    && contents.insert((&artifact.sha256, artifact.bytes))
                    && lock.manifest.provenance.contains(artifact),
                "R model provenance missing from full original lock or duplicated",
            )?;
        }
        Ok(())
    }
}

fn verify_retirement(
    observation: &RetirementObservation,
    admission: &Admission,
    registration: &RoventHistoricalRAuxiliaryDatasetRegistration,
) -> Result<(), ArenaError> {
    let retirement = &observation.retirement;
    let job = &retirement.job;
    let a = &job.authority;
    check(
        generation_sha(&observation.native_admission_sha256)
            && generation_digest(
                "rz-rovent-collection-root-admission/1",
                &observation.native_admission,
            )? == observation.native_admission_sha256
            && generation_digest(R_READOUT_RETIREMENT_DOMAIN, retirement)?
                == observation.retirement_sha256
            && a.process_epoch > 0
            && generation_sha(&a.implementation_sha256)
            && job.job_sequence > 0
            && job.total_native_node_credit > 0
            && job.hard_deadline_clock_epoch == a.process_epoch
            && [
                &job.state_semantic_sha256,
                &job.question_sha256,
                &job.root_restriction_sha256,
                &job.evaluation_namespace_sha256,
                &job.search_conditions_sha256,
            ]
            .iter()
            .all(|s| generation_sha(s))
            && retirement.frame_sequence > 0
            && retirement.capture_sequence > 0
            && (1..=64).contains(&retirement.limits.maximum_readouts)
            && retirement.limits.every_nth_score > 0
            && (1..=MAX_R_READOUT_BYTES as u64).contains(&retirement.limits.maximum_input_bytes)
            && retirement.selected <= u64::from(retirement.limits.maximum_readouts)
            && retirement.selected <= retirement.attempted
            && u64::from(retirement.retained_readouts) <= retirement.selected
            && retirement.retained_input_bytes <= retirement.limits.maximum_input_bytes
            && observation.readout_record_sha256.len() == usize::from(retirement.retained_readouts)
            && observation
                .readout_record_sha256
                .iter()
                .all(|s| generation_sha(s))
            && observation
                .readout_record_sha256
                .windows(2)
                .all(|v| v[0] < v[1])
            && retirement
                .error
                .as_ref()
                .is_none_or(|e| generation_identifier(e)),
        "R original retirement Job/count/limit/seal/empty coverage",
    )?;
    if matches!(retirement.ending, RoventRCaptureEnding::NotStarted) {
        check(
            retirement.attempted == 0
                && retirement.selected == 0
                && retirement.retained_readouts == 0
                && retirement.retained_input_bytes == 0,
            "not-started R retirement cannot claim readout work",
        )?;
    }
    if let RoventRCaptureEnding::CpuTerminal { completion } = &retirement.ending {
        check(
            generation_identifier(completion),
            "R original CPU completion description",
        )?;
    }
    let root: RoventRootAdmissionEvidence =
        serde_json::from_value(observation.native_admission.clone())
            .map_err(|e| invalid(e.to_string()))?;
    verify_original_root(
        &root,
        &observation.native_admission,
        &admission.lock,
        admission.bundle.as_ref(),
        registration,
    )?;
    check(
        root.process_epoch == a.process_epoch
            && root.game_generation == a.game_generation
            && root.root_generation == a.root_generation
            && root.implementation_sha256 == a.implementation_sha256
            && job.hard_deadline_clock_epoch == root.deadline_clock
            && job.hard_deadline_ns <= root.deadline_tick
            && job.total_native_node_credit
                <= root
                    .root_descriptor
                    .input
                    .declared_ceilings
                    .cpu_credit_limit,
        "R retirement actual Job differs from its original root/deadline/credit",
    )
}

fn verify_original_root(
    root: &RoventRootAdmissionEvidence,
    raw: &Value,
    lock: &rz_experiments::RoventInputLock,
    bundle: Option<&rz_eval::runtime_pin::CudaRuntimeBundleSpec>,
    registration: &RoventHistoricalRAuxiliaryDatasetRegistration,
) -> Result<(), ArenaError> {
    root.root_descriptor
        .verify_against(lock)
        .map_err(|e| invalid(e.to_string()))?;
    let origin: &RoventNativeOriginEvidence = &root.native_origin;
    let engine = &lock.manifest.engine;
    let descriptor = &root.root_descriptor.input;
    let model = &engine.pc.proposer;
    let source = rz_experiments::rovent_source_aggregate_sha256(
        &origin.source_commit,
        &origin.source_components,
    )
    .map_err(|e| invalid(e.to_string()))?;
    let registered_semantics = hex(rz_contracts::Digest(
        rz_eval::pals_model::PalsModelProfile::VirtualReviewV1.registered_semantics_sha256(),
    ));
    let encoding_sha = format!(
        "{:x}",
        Sha256::digest(rz_eval::pals_model::virtual_review::ENCODING.as_bytes())
    );
    check(
        root.schema == "rz-rovent-collection-root-admission/1"
            && origin.schema == "rz-rovent-collection-native-origin/1"
            && origin.purpose == "collection"
            && generation_digest(&origin.schema, &raw["native_origin"])?
                == root.native_origin_sha256
            && root.process_epoch > 0
            && origin.process_epoch == root.process_epoch
            && origin.input_lock_sha256 == lock.canonical_sha256
            && descriptor.startup_lock_sha256 == lock.canonical_sha256
            && lock.manifest.purpose == rz_experiments::RoventPurpose::Collection
            && origin.source_commit == registration.collection_source_commit
            && engine.source_commit == origin.source_commit
            && origin.source_components == engine.source_components
            && source == origin.source_sha256
            && source == registration.collection_source_sha256
            && origin.runtime_sha256_domain == "loaded_runtime_library_file_sha256"
            && origin.runtime_sha256 == origin.runtime_library_sha256
            && origin.runtime_sha256 == registration.collection_runtime_library_sha256
            && engine
                .pc
                .runtime_artifacts
                .iter()
                .any(|a| a.sha256 == origin.runtime_library_sha256)
            && origin.native_configuration_sha256 == engine.pc.backend_configuration_sha256
            && origin.mode == format!("{:?}", engine.mode)
            && origin.teacher_profiles == engine.teachers
            && origin.weights_sha256 == model.weights.sha256
            && origin.model_implementation_sha256 == model.implementation_sha256
            && origin.model_semantics == model.semantics
            && origin.model_semantics == rz_eval::pals_model::virtual_review::MODEL_SEMANTICS
            && origin.model_semantics_sha256 == registration.collection_model_semantics_sha256
            && origin.model_semantics_sha256 == registered_semantics
            && origin.encoding == model.encoding
            && origin.encoding == rz_eval::pals_model::virtual_review::ENCODING
            && origin.encoding_sha256 == encoding_sha
            && origin.encoding_registry_sha256
                == hex(rz_contracts::Digest(
                    rz_eval::pals_model::virtual_review::registered_encoding_sha256(),
                ))
            && origin.representation_model_domain == "loaded_checkpoint_sha256"
            && origin.frozen_epoch > 0
            && lock
                .manifest
                .provenance
                .iter()
                .filter(|p| {
                    p.sha256 == origin.export_manifest_sha256 && (1..=256 << 10).contains(&p.bytes)
                })
                .count()
                == 1
            && [
                &origin.export_manifest_sha256,
                &origin.backend_sha256,
                &root.selective_policy_sha256,
                &root.implementation_sha256,
            ]
            .iter()
            .all(|s| generation_sha(s))
            && root.root_descriptor_sha256
                == hex(root
                    .root_descriptor
                    .digest()
                    .map_err(|e| invalid(e.to_string()))?)
            && root.root_budget_sha256 == root.root_descriptor.root_budget_sha256
            && root.process_epoch == descriptor.authority.epoch
            && root.game_generation == descriptor.authority.game
            && root.root_generation == descriptor.authority.root
            && root.implementation_sha256 == descriptor.authority.implementation_sha256
            && root.deadline_clock == descriptor.clock.hard.clock_epoch
            && root.deadline_tick == descriptor.clock.hard.at_ns
            && root.root_state.semantic == descriptor.state_and_history_sha256
            && root.root_moves == descriptor.allowed_root_moves
            && root.depth_plies == descriptor.original_go.depth_plies.map(|v| v.get())
            && root.mate_moves == descriptor.original_go.mate_moves.map(|v| v.get())
            && root.requested_nodes == descriptor.original_go.requested_nodes,
        "R original empty/nonempty root/full-lock/source/runtime/model join",
    )?;
    match (bundle, &origin.runtime_bundle_sha256) {
        (None, None) => check(
            matches!(engine.pc.device, rz_experiments::RoventDevice::Cpu),
            "R CUDA origin missing original bundle",
        ),
        (Some(spec), Some(digest)) => {
            let actual = hex(rz_contracts::Digest(
                spec.digest().map_err(|e| invalid(e.to_string()))?,
            ));
            check(
                !matches!(engine.pc.device, rz_experiments::RoventDevice::Cpu)
                    && actual == *digest
                    && spec.files.len() == engine.pc.runtime_artifacts.len()
                    && spec.files.iter().all(|expected| {
                        engine
                            .pc
                            .runtime_artifacts
                            .iter()
                            .filter(|actual| {
                                Path::new(&actual.path).file_name().and_then(|n| n.to_str())
                                    == Some(expected.filename.as_str())
                                    && actual.sha256 == expected.sha256
                                    && actual.bytes == expected.bytes
                            })
                            .count()
                            == 1
                    }),
                "R historical runtime bundle semantic digest/roster/aliases mismatch",
            )
        }
        _ => Err(invalid(
            "R original bundle presence differs from loaded origin",
        )),
    }
}

fn hex(digest: rz_contracts::Digest) -> String {
    digest.0.iter().map(|b| format!("{b:02x}")).collect()
}

fn check_ancestor_sets(
    manifest: &RoventRReadoutManifest,
    pc: &[Arc<VerifiedRoventDataset>],
    r: &[Arc<VerifiedRoventRAuxiliaryDataset>],
) -> Result<(), ArenaError> {
    let pc_set = pc
        .iter()
        .map(|p| p.index.sha256().map_err(|e| invalid(e.to_string())))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let r_set = r
        .iter()
        .map(|p| p.index.sha256().map_err(|e| invalid(e.to_string())))
        .collect::<Result<BTreeSet<_>, _>>()?;
    check(
        pc_set.len() == pc.len()
            && r_set.len() == r.len()
            && pc_set.iter().cloned().collect::<Vec<_>>() == manifest.pc_ancestor_index_sha256
            && r_set.iter().cloned().collect::<Vec<_>>() == manifest.r_ancestor_index_sha256,
        "R missing/extra/duplicate complete PC/R original ancestor indices",
    )?;
    for p in pc {
        check(
            p.manifest
                .ancestor_index_sha256
                .iter()
                .all(|s| pc_set.contains(s)),
            "R PC ancestor graph is not transitively closed",
        )?;
    }
    for p in r {
        check(
            p.manifest
                .pc_ancestor_index_sha256
                .iter()
                .all(|s| pc_set.contains(s))
                && p.manifest
                    .r_ancestor_index_sha256
                    .iter()
                    .all(|s| r_set.contains(s)),
            "R ancestor graph is not transitively closed",
        )?;
    }
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecipeBody {
    schema: String,
    series_id: String,
    student_id: String,
    learner_id: String,
    model_semantics_sha256: String,
    source_sha256: String,
    actors: Vec<RecipeActor>,
    parameter_ownership: RecipeOwnership,
    stages: Vec<RecipeStage>,
    opening_splits: BTreeMap<String, GenerationSplitV3>,
    global_holdout_openings: Vec<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecipeActor {
    actor_id: String,
    family: String,
    source_revision: String,
    source_sha256: String,
    binary_sha256: String,
    profile_sha256: String,
    options_sha256: String,
    assets_sha256: String,
    rights_sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecipeOwnership {
    pc_shared: String,
    p_private: String,
    c_private: String,
    v_private: String,
    nnuec: String,
    membership_sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecipeStage {
    stage_id: String,
    kind: String,
    parent_stage_id: Option<String>,
    teacher_actor_ids: Vec<String>,
    round_id: String,
    role_updates: BTreeMap<String, u32>,
}
struct RecipeGraph {
    body: RecipeBody,
    sha: String,
    stage_indices: BTreeMap<String, usize>,
    scopes: BTreeMap<String, BTreeSet<String>>,
}
impl RecipeGraph {
    fn parse(value: &Value, expected: &str) -> Result<Self, ArenaError> {
        check(
            generation_digest(RECIPE_DOMAIN, value)? == expected,
            "R original recipe domain seal mismatch",
        )?;
        let body: RecipeBody =
            serde_json::from_value(value.clone()).map_err(|e| invalid(e.to_string()))?;
        check(
            body.schema == RECIPE_DOMAIN
                && [&body.series_id, &body.student_id, &body.learner_id]
                    .iter()
                    .all(|s| plain_id(s))
                && generation_sha(&body.model_semantics_sha256)
                && generation_sha(&body.source_sha256)
                && body.actors.len() == 2
                && (3..=256).contains(&body.stages.len())
                && (1..=65536).contains(&body.opening_splits.len())
                && body.opening_splits.keys().all(|s| generation_identifier(s))
                && (1..=65536).contains(&body.global_holdout_openings.len()),
            "strict finite R series/actor/stage/split recipe",
        )?;
        let mut actor_ids = BTreeSet::new();
        let mut families = BTreeSet::new();
        for a in &body.actors {
            check(
                plain_id(&a.actor_id)
                    && actor_ids.insert(a.actor_id.clone())
                    && ["stockfish", "reckless"].contains(&a.family.as_str())
                    && families.insert(a.family.clone())
                    && commit(&a.source_revision)
                    && [
                        &a.source_sha256,
                        &a.binary_sha256,
                        &a.profile_sha256,
                        &a.options_sha256,
                        &a.assets_sha256,
                        &a.rights_sha256,
                    ]
                    .iter()
                    .all(|s| generation_sha(s)),
                "R frozen distinct Stockfish/Reckless recipe actors",
            )?;
        }
        let p = &body.parameter_ownership;
        // This is the historical series' established ownership grammar, which
        // also permits disabled/base P/C NNUEC supervision to share moments.
        // A current enabled R-auxiliary registration must independently join
        // nnuec to its real private membership owner and reject shared/P/C/V
        // owners; this parser must not rewrite old recipe/profile semantics.
        check(
            [
                &p.pc_shared,
                &p.p_private,
                &p.c_private,
                &p.v_private,
                &p.nnuec,
            ]
            .iter()
            .all(|s| plain_id(s))
                && p.pc_shared == p.p_private
                && p.pc_shared == p.c_private
                && p.v_private != p.pc_shared
                && p.v_private != p.nnuec
                && generation_sha(&p.membership_sha256),
            "R declared parameter ownership shape",
        )?;
        let holdout = body.global_holdout_openings.iter().collect::<BTreeSet<_>>();
        check(
            holdout.len() == body.global_holdout_openings.len()
                && holdout
                    .iter()
                    .all(|s| body.opening_splits.get(*s) == Some(&GenerationSplitV3::Holdout)),
            "R predeclared global holdout openings",
        )?;
        let mut stage_indices = BTreeMap::new();
        let mut scopes: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut common = None;
        let mut forks = BTreeSet::new();
        for (index, s) in body.stages.iter().enumerate() {
            let teachers = s.teacher_actor_ids.iter().cloned().collect::<BTreeSet<_>>();
            check(
                plain_id(&s.stage_id)
                    && plain_id(&s.round_id)
                    && !stage_indices.contains_key(&s.stage_id)
                    && s.parent_stage_id.as_ref().is_none_or(|s| plain_id(s))
                    && s.teacher_actor_ids.len() <= 2
                    && teachers.len() == s.teacher_actor_ids.len()
                    && teachers.is_subset(&actor_ids)
                    && s.role_updates.keys().map(String::as_str).eq([
                        "critic",
                        "proposer",
                        "validator",
                    ])
                    && s.role_updates.values().any(|v| *v > 0),
                "R unique explicit stage/teacher/update recipe",
            )?;
            let mut scope = if s.kind == "common_student" {
                check(
                    common.is_none()
                        && index == 0
                        && s.parent_stage_id.is_none()
                        && teachers == actor_ids,
                    "R common stage must initially consume both frozen teachers",
                )?;
                common = Some(s.stage_id.clone());
                BTreeSet::new()
            } else {
                let parent = s
                    .parent_stage_id
                    .as_ref()
                    .and_then(|p| scopes.get(p))
                    .ok_or_else(|| invalid("R stage parent is missing/future/cyclic"))?;
                match s.kind.as_str() {
                    "teacher_finetune" => {
                        check(
                            s.parent_stage_id == common && teachers.len() == 1,
                            "R teacher fork must share the common parent",
                        )?;
                        let teacher = teachers
                            .iter()
                            .next()
                            .ok_or_else(|| invalid("R teacher FT missing actor"))?;
                        check(
                            forks.insert(teacher.clone()),
                            "R duplicate teacher-specific fork",
                        )?;
                    }
                    "mixed" => check(
                        forks == actor_ids && teachers == actor_ids,
                        "R mixed stage requires both teacher forks",
                    )?,
                    "self" => check(
                        forks == actor_ids && teachers.is_empty(),
                        "R self stage must be teacher-free after both forks",
                    )?,
                    _ => return Err(invalid("R unsupported recipe stage kind")),
                }
                parent.clone()
            };
            scope.insert(s.stage_id.clone());
            scopes.insert(s.stage_id.clone(), scope);
            stage_indices.insert(s.stage_id.clone(), index);
        }
        check(forks == actor_ids, "R both teacher-specific forks required")?;
        Ok(Self {
            body,
            sha: expected.into(),
            stage_indices,
            scopes,
        })
    }
    fn stage(&self, id: &str) -> Result<&RecipeStage, ArenaError> {
        self.stage_indices
            .get(id)
            .and_then(|i| self.body.stages.get(*i))
            .ok_or_else(|| invalid("R row/collection stage absent from original recipe"))
    }
    fn collection(&self, c: &RoventCorpusManifest) -> Result<(), ArenaError> {
        self.collection_metadata(&HistoricalCollectionMetadata::legacy(c))
    }
    fn collection_metadata(&self, c: &HistoricalCollectionMetadata) -> Result<(), ArenaError> {
        let s = self.stage(&c.stage_id)?;
        check(
            c.recipe_sha256 == self.sha
                && c.source_sha256 == self.body.source_sha256
                && c.model_semantics_sha256 == self.body.model_semantics_sha256
                && c.round_id == s.round_id
                && c.student == self.body.student_id
                && c.learner == self.body.learner_id
                && c.one_learner
                && c.opening_splits == self.body.opening_splits
                && c.teachers.keys().cloned().collect::<BTreeSet<_>>()
                    == s.teacher_actor_ids.iter().cloned().collect(),
            "R/PC original collection differs from complete frozen recipe/stage/round/learner/split",
        )?;
        for (id, pin) in &c.teachers {
            let a = self
                .body
                .actors
                .iter()
                .find(|a| a.actor_id == *id)
                .ok_or_else(|| invalid("R corpus teacher actor missing from recipe"))?;
            check(
                pin.engine == a.family
                    && pin.source_commit == a.source_revision
                    && pin.binary_sha256 == a.binary_sha256
                    && pin.profile_sha256 == a.profile_sha256,
                "R frozen teacher family/source/binary/profile changed",
            )?;
        }
        Ok(())
    }
    fn entry(&self, e: &LineageEntry) -> Result<(), ArenaError> {
        let s = self.stage(&e.stage_id)?;
        check(
            e.round_id == s.round_id
                && e.teacher_actor_id
                    .as_ref()
                    .is_none_or(|id| s.teacher_actor_ids.contains(id))
                && self.body.opening_splits.get(&e.opening_id) == Some(&e.split)
                && [
                    &e.input_sha256,
                    &e.record_sha256,
                    &e.origin_sha256,
                    &e.producer_sha256,
                    &e.rules_state_sha256,
                    &e.transposition_sha256,
                ]
                .iter()
                .all(|s| generation_sha(s))
                && [&e.opening_id, &e.game_id, &e.genealogy_id, &e.actor_id]
                    .iter()
                    .all(|s| generation_identifier(s)),
            "R/PC original occurrence differs from frozen stage/round/teacher/split origin",
        )
    }
}
fn plain_id(s: &str) -> bool {
    generation_identifier(s)
        && s.is_ascii()
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

#[derive(Clone, Serialize)]
struct LineageEntry {
    source: &'static str,
    record_sha256: String,
    origin_sha256: String,
    input_sha256: String,
    producer_sha256: String,
    stage_id: String,
    round_id: String,
    teacher_actor_id: Option<String>,
    actor_id: String,
    opening_id: String,
    game_id: String,
    genealogy_id: String,
    rules_state_sha256: String,
    transposition_sha256: String,
    split: GenerationSplitV3,
    ancestor_origin_sha256: Vec<String>,
    pc_genealogy: Option<RoventRPcGenealogy>,
}
impl LineageEntry {
    fn from_r(r: &RoventRReadoutRecord, c: &RoventCorpusManifest) -> Self {
        Self {
            source: R_READOUT_RECORD_DOMAIN,
            record_sha256: r.record_sha256.clone(),
            origin_sha256: r.origin_sha256.clone(),
            input_sha256: r.input.semantic_input_sha256.clone(),
            producer_sha256: r.manifest_sha256.clone(),
            stage_id: c.stage_id.clone(),
            round_id: c.round_id.clone(),
            teacher_actor_id: r.lineage.teacher_actor_id.clone(),
            actor_id: r.lineage.actor_id.clone(),
            opening_id: r.lineage.opening_id.clone(),
            game_id: r.lineage.game_id.clone(),
            genealogy_id: r.lineage.line_genealogy_id.clone(),
            rules_state_sha256: r.bound.current_rules.anchor.rules_state_sha256.clone(),
            transposition_sha256: r.lineage.transposition_sha256.clone(),
            split: r.split,
            ancestor_origin_sha256: r.lineage.ancestor_origin_sha256.clone(),
            pc_genealogy: r.lineage.pc_genealogy.clone(),
        }
    }
    fn from_pc(
        r: &HistoricalPcLineageRow,
        c: &HistoricalCollectionMetadata,
        producer_manifest_sha256: &str,
    ) -> Self {
        let origin = &r.origin;
        Self {
            source: r.family.record_domain(),
            record_sha256: origin.record_sha256.clone(),
            origin_sha256: origin.origin_sha256.clone(),
            input_sha256: origin.input_sha256.clone(),
            // The P/C record has no manifest field. F's original lineage
            // projection uses the owner's canonical manifest seal, not the
            // distinct native source aggregate. The caller supplies the exact
            // seal already joined to the original external publication.
            producer_sha256: producer_manifest_sha256.to_owned(),
            stage_id: c.stage_id.clone(),
            round_id: c.round_id.clone(),
            teacher_actor_id: origin.teacher_actor_id.clone(),
            actor_id: origin.actor_id.clone(),
            opening_id: origin.opening_id.clone(),
            game_id: origin.game_id.clone(),
            genealogy_id: origin.line_genealogy_id.clone(),
            rules_state_sha256: r.rules_state_sha256.clone(),
            transposition_sha256: origin.transposition_sha256.clone(),
            split: origin.split,
            ancestor_origin_sha256: origin.ancestor_origin_sha256.clone(),
            pc_genealogy: None,
        }
    }
}

struct JointEligibility {
    r_records: BTreeSet<String>,
    pc_origins: BTreeSet<String>,
    pc_records: BTreeSet<String>,
    pc_origin_inputs: BTreeMap<String, String>,
    r_origin_inputs: BTreeMap<String, String>,
}
fn global_training_rows(
    graph: &RecipeGraph,
    current: &[LineageEntry],
    manifest: &RoventRReadoutManifest,
    pc: &[Arc<VerifiedRoventDataset>],
    r: &[Arc<VerifiedRoventRAuxiliaryDataset>],
    budget: &RoventRecoveryReadBudget,
) -> Result<JointEligibility, ArenaError> {
    let mut entries = Vec::new();
    let mut metadata_bytes = 0u64;
    let mut edges = 0usize;
    let mut push = |entry: LineageEntry| -> Result<(), ArenaError> {
        graph.entry(&entry)?;
        metadata_bytes = metadata_bytes
            .checked_add(
                serde_json::to_vec(&entry)
                    .map_err(|e| invalid(e.to_string()))?
                    .len() as u64,
            )
            .ok_or_else(|| invalid("R lineage metadata overflow"))?;
        edges = edges
            .checked_add(
                entry.ancestor_origin_sha256.len() + usize::from(entry.pc_genealogy.is_some()),
            )
            .ok_or_else(|| invalid("R lineage edge overflow"))?;
        check(
            entries.len() < MAX_CORPUS_ROWS
                && metadata_bytes <= MAX_GLOBAL_METADATA_BYTES
                && edges <= MAX_GLOBAL_EDGES,
            "R global lineage occurrence/metadata/edge budget",
        )?;
        entries.push(entry);
        Ok(())
    };
    let mut declarations: Vec<HistoricalCollectionMetadata> = Vec::new();
    for owner in pc {
        graph.collection_metadata(&owner.manifest)?;
        declarations.push(owner.manifest.clone());
        owner.revalidate(budget)?;
        let indexed = owner
            .index
            .origins
            .iter()
            .map(|o| (o.record_sha256.as_str(), o))
            .collect::<BTreeMap<_, _>>();
        let mut seen = BTreeSet::new();
        // This metadata was built only after the owner's original typed row,
        // raw native evidence and Rules replay passed. Its held chunk bytes
        // were just revalidated; never re-decode record/2 as record/1 here.
        for row in &owner.lineage_rows {
            budget.reserve(0)?;
            check(
                row.family == owner.family()
                    && indexed
                        .get(row.origin.record_sha256.as_str())
                        .is_some_and(|o| **o == row.origin)
                    && seen.insert(row.origin.record_sha256.clone()),
                "R PC original typed lineage/index occurrence mismatch",
            )?;
            push(LineageEntry::from_pc(
                row,
                &owner.manifest,
                owner.manifest_sha256(),
            ))?;
        }
        check(
            seen.len() == owner.index.origins.len(),
            "R PC source full original occurrence coverage",
        )?;
    }
    for owner in r {
        graph.collection(&owner.manifest.collection)?;
        declarations.push(HistoricalCollectionMetadata::legacy(
            &owner.manifest.collection,
        ));
        for entry in owner.lineage.iter() {
            push(entry.clone())?;
        }
    }
    graph.collection(&manifest.collection)?;
    declarations.push(HistoricalCollectionMetadata::legacy(&manifest.collection));
    for entry in current {
        push(entry.clone())?;
    }
    let mut by_source = BTreeMap::new();
    let mut source_records = BTreeSet::new();
    let mut by_origin: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, e) in entries.iter().enumerate() {
        check(
            by_source
                .insert((e.source, e.origin_sha256.as_str()), i)
                .is_none()
                && source_records.insert((e.source, e.record_sha256.as_str())),
            "R duplicate original source origin/record occurrence across owners",
        )?;
        by_origin
            .entry(e.origin_sha256.as_str())
            .or_default()
            .push(i);
    }
    let mut union = Disjoint::new(entries.len());
    let mut identities = BTreeMap::new();
    let mut children = vec![Vec::new(); entries.len()];
    let mut indegree = vec![0usize; entries.len()];
    for (i, e) in entries.iter().enumerate() {
        for (kind, value) in [
            ("input", &e.input_sha256),
            ("game", &e.game_id),
            ("opening", &e.opening_id),
            ("genealogy", &e.genealogy_id),
            ("rules", &e.rules_state_sha256),
            ("transposition", &e.transposition_sha256),
        ] {
            if let Some(prior) = identities.insert((kind, value.as_str()), i) {
                union.join(i, prior);
            }
        }
        let mut refs = BTreeSet::new();
        for sha in &e.ancestor_origin_sha256 {
            let matches = by_origin
                .get(sha.as_str())
                .ok_or_else(|| invalid("R unresolved original PC/R ancestor origin"))?;
            check(
                matches.len() == 1,
                "R source-ambiguous inherited ancestor origin",
            )?;
            refs.insert(matches[0]);
        }
        if let Some(link) = &e.pc_genealogy {
            let candidates = by_origin
                .get(link.origin_sha256.as_str())
                .ok_or_else(|| invalid("R causal PC genealogy lacks original PC source"))?;
            let candidates = candidates
                .iter()
                .copied()
                .filter(|index| is_pc_source(entries[*index].source))
                .collect::<Vec<_>>();
            check(
                candidates.len() == 1,
                "R causal PC genealogy source family is absent/ambiguous",
            )?;
            let index = candidates[0];
            let original = &entries[index];
            check(
                original.record_sha256 == link.record_sha256
                    && original.input_sha256 == link.input_sha256,
                "R causal PC genealogy changed exact original row",
            )?;
            refs.insert(index);
        }
        for parent in refs {
            check(
                parent != i
                    && graph
                        .scopes
                        .get(&e.stage_id)
                        .is_some_and(|scope| scope.contains(&entries[parent].stage_id)),
                "R ancestor occurrence belongs to self/future/sibling stage",
            )?;
            children[parent].push(i);
            indegree[i] += 1;
            union.join(i, parent);
        }
    }
    let mut ready = indegree
        .iter()
        .enumerate()
        .filter(|(_, n)| **n == 0)
        .map(|(i, _)| i)
        .collect::<VecDeque<_>>();
    let mut visited = 0;
    while let Some(index) = ready.pop_front() {
        visited += 1;
        for child in &children[index] {
            indegree[*child] -= 1;
            if indegree[*child] == 0 {
                ready.push_back(*child);
            }
        }
    }
    check(
        visited == entries.len(),
        "R PC/R original ancestor occurrence cycle",
    )?;
    let mut component_splits: BTreeMap<usize, BTreeSet<&str>> = BTreeMap::new();
    for (i, e) in entries.iter().enumerate() {
        component_splits
            .entry(union.root(i))
            .or_default()
            .insert(split_name(e.split));
    }
    let excluded = (0..entries.len())
        .filter(|i| component_splits[&union.root(*i)].len() > 1)
        .collect::<BTreeSet<_>>();
    // Declarations of already inherited training are stronger than candidate
    // rows: a contaminated/future/sibling ancestor is rejected, never repaired
    // by dropping only the current descendant from optimization.
    for c in declarations {
        let mut input_set = BTreeSet::new();
        for sha in &c.inherited_origin_sha256 {
            let matches = by_origin
                .get(sha.as_str())
                .ok_or_else(|| invalid("R inherited training origin missing"))?;
            check(
                matches.len() == 1,
                "R inherited training origin source ambiguous",
            )?;
            let i = matches[0];
            let e = &entries[i];
            check(
                !excluded.contains(&i)
                    && e.split == GenerationSplitV3::Train
                    && e.stage_id != c.stage_id
                    && graph
                        .scopes
                        .get(&c.stage_id)
                        .is_some_and(|s| s.contains(&e.stage_id)),
                "R inherited training includes holdout/cross-split/same-stage/future/sibling origin",
            )?;
            input_set.insert(e.input_sha256.clone());
        }
        check(
            input_set == c.inherited_input_sha256.iter().cloned().collect(),
            "R inherited original origin/input projection mismatch",
        )?;
    }
    let current_shas = current
        .iter()
        .map(|e| e.record_sha256.as_str())
        .collect::<BTreeSet<_>>();
    let r_records = entries
        .iter()
        .enumerate()
        .filter(|(i, e)| {
            e.source == R_READOUT_RECORD_DOMAIN
                && current_shas.contains(e.record_sha256.as_str())
                && e.split == GenerationSplitV3::Train
                && !excluded.contains(i)
        })
        .map(|(_, e)| e.record_sha256.clone())
        .collect();
    let scope = graph
        .scopes
        .get(&manifest.collection.stage_id)
        .ok_or_else(|| invalid("joint current stage scope absent"))?;
    let pc_entries = entries.iter().enumerate().filter(|(i, e)| {
        is_pc_source(e.source)
            && e.split == GenerationSplitV3::Train
            && !excluded.contains(i)
            && scope.contains(&e.stage_id)
    });
    let pc_origins = pc_entries
        .clone()
        .map(|(_, e)| e.origin_sha256.clone())
        .collect();
    let pc_records = pc_entries
        .clone()
        .map(|(_, e)| e.record_sha256.clone())
        .collect();
    let pc_origin_inputs = pc_entries
        .map(|(_, e)| (e.origin_sha256.clone(), e.input_sha256.clone()))
        .collect();
    let r_origin_inputs = entries
        .iter()
        .enumerate()
        .filter(|(i, e)| {
            e.source == R_READOUT_RECORD_DOMAIN
                && e.split == GenerationSplitV3::Train
                && !excluded.contains(i)
                && scope.contains(&e.stage_id)
        })
        .map(|(_, e)| (e.origin_sha256.clone(), e.input_sha256.clone()))
        .collect();
    Ok(JointEligibility {
        r_records,
        pc_origins,
        pc_records,
        pc_origin_inputs,
        r_origin_inputs,
    })
}
struct Disjoint {
    parents: Vec<usize>,
}
impl Disjoint {
    fn new(count: usize) -> Self {
        Self {
            parents: (0..count).collect(),
        }
    }
    fn root(&mut self, mut i: usize) -> usize {
        while self.parents[i] != i {
            self.parents[i] = self.parents[self.parents[i]];
            i = self.parents[i];
        }
        i
    }
    fn join(&mut self, a: usize, b: usize) {
        let a = self.root(a);
        let b = self.root(b);
        self.parents[a] = b;
    }
}
fn split_name(split: GenerationSplitV3) -> &'static str {
    match split {
        GenerationSplitV3::Train => "train",
        GenerationSplitV3::Validation => "validation",
        GenerationSplitV3::Holdout => "holdout",
    }
}
fn is_pc_source(source: &str) -> bool {
    source == RoventPcRecordFamily::LegacyRecordV1.record_domain()
        || source == RoventPcRecordFamily::SubmittedInputV1.record_domain()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn recipe() -> Value {
        let actor = |id: &str, family: &str| {
            json!({"actor_id":id,"family":family,"source_revision":"a".repeat(40),
            "source_sha256":"b".repeat(64),"binary_sha256":"c".repeat(64),"profile_sha256":"d".repeat(64),
            "options_sha256":"e".repeat(64),"assets_sha256":"f".repeat(64),"rights_sha256":"1".repeat(64)})
        };
        let stage = |id: &str,
                     kind: &str,
                     parent: Option<&str>,
                     teachers: Vec<&str>,
                     round: &str| {
            json!({
            "stage_id":id,"kind":kind,"parent_stage_id":parent,"teacher_actor_ids":teachers,"round_id":round,
            "role_updates":{"proposer":1,"critic":1,"validator":0}})
        };
        json!({"schema":RECIPE_DOMAIN,"series_id":"series","student_id":"student","learner_id":"learner",
            "model_semantics_sha256":"2".repeat(64),"source_sha256":"3".repeat(64),
            "actors":[actor("sf","stockfish"),actor("rk","reckless")],
            "parameter_ownership":{"pc_shared":"pc","p_private":"pc","c_private":"pc","v_private":"v",
                "nnuec":"nnuec","membership_sha256":"4".repeat(64)},
            "stages":[stage("common","common_student",None,vec!["sf","rk"],"round-common"),
                stage("sf-ft","teacher_finetune",Some("common"),vec!["sf"],"round-sf"),
                stage("rk-ft","teacher_finetune",Some("common"),vec!["rk"],"round-rk")],
            "opening_splits":{"train-opening":"train","validation-opening":"validation","holdout-opening":"holdout"},
            "global_holdout_openings":["holdout-opening"]})
    }
    fn graph(value: &Value) -> Result<RecipeGraph, ArenaError> {
        let sha = generation_digest(RECIPE_DOMAIN, value)?;
        RecipeGraph::parse(value, &sha)
    }
    fn entry() -> LineageEntry {
        LineageEntry {
            source: R_READOUT_RECORD_DOMAIN,
            record_sha256: "5".repeat(64),
            origin_sha256: "6".repeat(64),
            input_sha256: "7".repeat(64),
            producer_sha256: "8".repeat(64),
            stage_id: "common".into(),
            round_id: "round-common".into(),
            teacher_actor_id: None,
            actor_id: "student".into(),
            opening_id: "train-opening".into(),
            game_id: "game".into(),
            genealogy_id: "line".into(),
            rules_state_sha256: "9".repeat(64),
            transposition_sha256: "a".repeat(64),
            split: GenerationSplitV3::Train,
            ancestor_origin_sha256: vec![],
            pc_genealogy: None,
        }
    }
    #[test]
    fn strict_recipe_digest_and_parent_graph_are_checked_without_execution() {
        let mut value = recipe();
        graph(&value).unwrap();
        assert!(RecipeGraph::parse(&value, &"0".repeat(64)).is_err());
        value["stages"][1]["parent_stage_id"] = json!("rk-ft");
        assert!(graph(&value).is_err());
        value = recipe();
        value["stages"][2]["teacher_actor_ids"] = json!(["sf"]);
        assert!(graph(&value).is_err());
        value = recipe();
        value["stages"][1]["unknown"] = json!(true);
        assert!(graph(&value).is_err());
    }
    #[test]
    fn historical_disabled_owner_grammar_is_not_silently_rewritten() {
        let mut value = recipe();
        value["parameter_ownership"]["nnuec"] = json!("pc");
        graph(&value).unwrap();
        value["parameter_ownership"]["nnuec"] = json!("v");
        assert!(graph(&value).is_err());
    }
    #[test]
    fn original_stage_round_teacher_and_split_are_distinct_from_row_seals() {
        let graph = graph(&recipe()).unwrap();
        let mut row = entry();
        graph.entry(&row).unwrap();
        row.round_id = "round-sf".into();
        assert!(graph.entry(&row).is_err());
        row = entry();
        row.teacher_actor_id = Some("invented".into());
        assert!(graph.entry(&row).is_err());
        row = entry();
        row.split = GenerationSplitV3::Holdout;
        assert!(graph.entry(&row).is_err());
        assert!(!graph.scopes["sf-ft"].contains("rk-ft"));
        assert!(graph.scopes["sf-ft"].contains("common"));
    }
    #[test]
    fn repeated_semantic_input_is_not_an_occurrence_identity() {
        let first = entry();
        let mut second = entry();
        second.record_sha256 = "b".repeat(64);
        second.origin_sha256 = "c".repeat(64);
        assert_eq!(first.input_sha256, second.input_sha256);
        assert_ne!(first.record_sha256, second.record_sha256);
        let projection = |rows: &[LineageEntry]| {
            generation_digest("rz-rovent-r-auxiliary-pool/1",&json!({"index_sha256":"d".repeat(64),
            "occurrences":rows.iter().map(|r|json!({"source_domain":r.source,"record_sha256":r.record_sha256,
                "input_sha256":r.input_sha256,"origin_sha256":r.origin_sha256})).collect::<Vec<_>>()})).unwrap()
        };
        assert_ne!(
            projection(&[first.clone(), second.clone()]),
            projection(&[second, first])
        );
    }
}
