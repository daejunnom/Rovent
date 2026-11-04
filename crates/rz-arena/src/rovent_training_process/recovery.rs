//! Explicit historical training input admission. A recovered publication is
//! durable data evidence; it never recreates PublishedRoventRecord, a native
//! ticket/run witness, OriginalProcessOutput or a past physical completion.
//! This module is intentionally separate from the existing live /1 codecs.

use super::*;
use crate::pals_collect_v3::training_process::{
    FileIdentity, GenerationTrainingFilePinV3, ReadAccounting, ReadCredit, open_file, verify_bytes,
};
use crate::rovent_collect::{
    RoventCorpusPublication, RoventPublicationFile, verify_historical_record_evidence,
    verify_historical_submitted_record_evidence,
};
use rz_experiments::rovent_corpus::MAX_CHUNK_BYTES;
use std::{
    path::{Component, Path},
    sync::atomic::{AtomicU64, Ordering},
};

pub const ROVENT_HISTORICAL_DATASET: &str = "rz-rovent-historical-dataset-admission/1";
pub const ROVENT_RECOVERY_REGISTRATION: &str = "rz-rovent-recovered-training-registration/1";
pub const ROVENT_RECOVERY_STARTUP: &str = "rz-rovent-recovered-training-startup/1";
pub const ROVENT_RECOVERED_CHECKPOINT: &str = "rz-rovent-recovered-checkpoint-admission/1";
pub const ROVENT_RECOVERY_SOURCE_KIND: &str = "historical_exact_pins";
const MAX_METADATA_BYTES: u64 = 16 << 20;
mod corpus_family;
use corpus_family::*;
mod lock_v2_archive;
pub use lock_v2_archive::*;

/// Borrowed lineage of an already admitted original row; no replay or learner
/// authority can be created from this projection.
pub(super) struct HistoricalDatasetLineage<'a> {
    pub(super) split: GenerationSplitV3,
    pub(super) opening_id: &'a str,
    pub(super) game_id: &'a str,
    pub(super) line_genealogy_id: &'a str,
    pub(super) rules_state_sha256: &'a str,
    pub(super) transposition_sha256: &'a str,
}

pub(super) fn probe_utility_pc_source(
    registration: &RoventHistoricalDatasetRegistration,
    budget: &RoventRecoveryReadBudget,
) -> Result<(RoventPcRecordFamily, String, Vec<String>), ArenaError> {
    probe_registered_pc_dataset(registration, budget)
}

pub struct RoventRecoveryReadBudget {
    maximum: u64,
    charged: AtomicU64,
    uncertain: AtomicBool,
    until: Instant,
    cancel: Arc<AtomicBool>,
}
impl RoventRecoveryReadBudget {
    pub fn new(maximum: u64, until: Instant, cancel: Arc<AtomicBool>) -> Result<Self, ArenaError> {
        check(
            maximum > 0 && maximum <= MAX_TRAINER_OUTPUT_BYTES * 260 && Instant::now() < until,
            "finite original recovery read credit/deadline required",
        )?;
        Ok(Self {
            maximum,
            charged: AtomicU64::new(0),
            uncertain: AtomicBool::new(false),
            until,
            cancel,
        })
    }
    pub(super) fn reserve(&self, bytes: u64) -> Result<(), ArenaError> {
        check(
            Instant::now() < self.until && !self.cancel.load(Ordering::Acquire),
            "historical admission cancelled/expired",
        )?;
        self.charged
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |old| {
                old.checked_add(bytes).filter(|next| *next <= self.maximum)
            })
            .map_err(|_| {
                invalid("historical read credit exhausted; failed reservations are not refunded")
            })?;
        Ok(())
    }
    pub fn observation(&self) -> Value {
        json!({"charged_bytes":self.charged.load(Ordering::Acquire),"charged_bytes_scope":"conservative_reserved_upper_bound","unknown":self.uncertain.load(Ordering::Acquire),"maximum_bytes":self.maximum})
    }
    pub(super) fn read(
        &self,
        pin: &RoventTrainingFilePin,
        maximum: u64,
        retain: bool,
    ) -> Result<(HeldHistoricalFile, Vec<u8>), ArenaError> {
        let legacy = GenerationTrainingFilePinV3 {
            path: pin.path.clone(),
            sha256: pin.sha256.clone(),
            bytes: pin.bytes,
        };
        legacy.validate(maximum)?;
        self.reserve(pin.bytes)?;
        let file = open_file(&pin.path).inspect_err(|_| {
            self.uncertain.store(true, Ordering::Release);
        })?;
        let raw = self.verify_reserved(&file, pin, retain)?;
        let identity = FileIdentity::from(&file.metadata().map_err(|_| {
            self.uncertain.store(true, Ordering::Release);
            invalid("historical held file metadata")
        })?);
        Ok((
            HeldHistoricalFile {
                file,
                pin: pin.clone(),
                identity,
            },
            raw,
        ))
    }
    pub(crate) fn verify_held(
        &self,
        file: &File,
        pin: &RoventTrainingFilePin,
        retain: bool,
    ) -> Result<Vec<u8>, ArenaError> {
        GenerationTrainingFilePinV3 {
            path: pin.path.clone(),
            sha256: pin.sha256.clone(),
            bytes: pin.bytes,
        }
        .validate(MAX_TRAINER_OUTPUT_BYTES)?;
        self.reserve(pin.bytes)?;
        self.verify_reserved(file, pin, retain)
    }
    fn verify_reserved(
        &self,
        file: &File,
        pin: &RoventTrainingFilePin,
        retain: bool,
    ) -> Result<Vec<u8>, ArenaError> {
        let legacy = GenerationTrainingFilePinV3 {
            path: pin.path.clone(),
            sha256: pin.sha256.clone(),
            bytes: pin.bytes,
        };
        let account = ReadAccounting::new();
        let mut credit = ReadCredit {
            remaining: pin.bytes,
            maximum: pin.bytes,
            until: self.until,
            cancel: &self.cancel,
            account: Arc::clone(&account),
        };
        let result = verify_bytes(file, &legacy, retain, &mut credit);
        if result.is_err() || account.uncertain.load(Ordering::Acquire) {
            self.uncertain.store(true, Ordering::Release);
        }
        let raw = result?;
        if account.charged.load(Ordering::Acquire) != pin.bytes {
            self.uncertain.store(true, Ordering::Release);
            return Err(invalid("historical full file read accounting mismatch"));
        }
        Ok(raw)
    }
}
pub(super) struct HeldHistoricalFile {
    pub(super) file: File,
    pub(super) pin: RoventTrainingFilePin,
    pub(super) identity: FileIdentity,
}
fn decode<T: serde::de::DeserializeOwned>(raw: &[u8]) -> Result<T, ArenaError> {
    let text = std::str::from_utf8(raw).map_err(|_| invalid("historical metadata UTF8"))?;
    rz_experiments::decode_json(text).map_err(|e| invalid(e.to_string()))
}
pub(super) fn archive_pin(
    root: &Path,
    item: &RoventPublicationFile,
) -> Result<RoventTrainingFilePin, ArenaError> {
    item.validate()?;
    let relative = Path::new(&item.path);
    check(
        root.is_absolute()
            && !item.path.is_empty()
            && !relative.is_absolute()
            && !item.path.contains('\\')
            && relative
                .components()
                .all(|part| matches!(part, Component::Normal(_))),
        "historical artifact escapes registered archive root",
    )?;
    Ok(RoventTrainingFilePin {
        path: root.join(relative),
        sha256: item.sha256.clone(),
        bytes: item.bytes,
    })
}

/// Caller registration is a request to validate exact bytes, not a native run
/// capability. Collection and trainer/runtime/initial weights are separate.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventHistoricalRoleBinding {
    pub model_semantics_sha256: String,
    pub model_identity_sha256: String,
    pub weights_sha256: String,
    pub encoding_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventHistoricalDatasetRegistration {
    pub root: PathBuf,
    pub publication: RoventTrainingFilePin,
    pub collection_source_commit: String,
    pub collection_source_sha256: String,
    pub collection_runtime_library_sha256: String,
    pub model_semantics_sha256: String,
    pub model_epoch: u64,
    pub context_epoch: u64,
    pub recipe_sha256: String,
    pub stage_id: String,
    pub role_bindings: BTreeMap<String, RoventHistoricalRoleBinding>,
}

/// No Deserialize or public success constructor. Every retained row has been
/// reread from the externally pinned publication and replayed through Rules.
/// These rows are training data, never a reconstruction of native authority.
pub struct VerifiedRoventDataset {
    registration: RoventHistoricalDatasetRegistration,
    original: OriginalPcCorpus,
    publication: HistoricalPublicationMetadata,
    manifest: HistoricalCollectionMetadata,
    index: HistoricalIndexMetadata,
    rows: BTreeMap<String, NativeRowAuthority>,
    lineage_rows: Vec<HistoricalPcLineageRow>,
    held: Vec<HeldHistoricalFile>,
    ancestors: Vec<Arc<Self>>,
}
impl VerifiedRoventDataset {
    pub fn open(
        registration: RoventHistoricalDatasetRegistration,
        ancestors: Vec<Arc<Self>>,
        budget: &RoventRecoveryReadBudget,
    ) -> Result<Self, ArenaError> {
        Self::open_family(
            registration,
            ancestors,
            budget,
            RoventPcRecordFamily::LegacyRecordV1,
        )
    }
    /// The submitted-input/1, record/2, raw/3 family is explicitly selected.
    /// The legacy constructor never decodes it or converts its canonical seals.
    pub fn open_submitted_input_v1(
        registration: RoventHistoricalDatasetRegistration,
        ancestors: Vec<Arc<Self>>,
        budget: &RoventRecoveryReadBudget,
    ) -> Result<Self, ArenaError> {
        Self::open_family(
            registration,
            ancestors,
            budget,
            RoventPcRecordFamily::SubmittedInputV1,
        )
    }
    fn open_family(
        registration: RoventHistoricalDatasetRegistration,
        ancestors: Vec<Arc<Self>>,
        budget: &RoventRecoveryReadBudget,
        expected_family: RoventPcRecordFamily,
    ) -> Result<Self, ArenaError> {
        check(
            ancestors.len() <= 256
                && registration.publication.path == registration.root.join("publication.json")
                && generation_identifier(&registration.stage_id)
                && generation_sha(&registration.collection_source_sha256)
                && generation_sha(&registration.collection_runtime_library_sha256)
                && generation_sha(&registration.model_semantics_sha256)
                && generation_sha(&registration.recipe_sha256)
                && !registration.role_bindings.is_empty()
                && registration.role_bindings.len() <= 3,
            "strict historical dataset registration",
        )?;
        for (role, binding) in &registration.role_bindings {
            check(
                ROLES.contains(&role.as_str())
                    && [
                        &binding.model_semantics_sha256,
                        &binding.model_identity_sha256,
                        &binding.weights_sha256,
                        &binding.encoding_sha256,
                    ]
                    .iter()
                    .all(|s| generation_sha(s)),
                "historical per-role model/weights/encoding binding",
            )?;
        }
        let (mut held, published_raw) = budget
            .read(&registration.publication, MAX_METADATA_BYTES, true)
            .map(|(f, r)| (vec![f], r))?;
        let original_publication = OriginalPublication::decode(&published_raw)?;
        check(
            original_publication.family() == expected_family,
            "PC corpus requires the explicitly selected original family constructor",
        )?;
        let publication = original_publication.metadata();
        check(
            publication.source_commit == registration.collection_source_commit
                && publication.source_sha256 == registration.collection_source_sha256
                && publication.runtime_sha256 == registration.collection_runtime_library_sha256,
            "historical publication changed registered collection source/runtime",
        )?;
        let (manifest_file, manifest_raw) = budget.read(
            &archive_pin(&registration.root, &publication.manifest)?,
            MAX_METADATA_BYTES,
            true,
        )?;
        held.push(manifest_file);
        let (index_file, index_raw) = budget.read(
            &archive_pin(&registration.root, &publication.index)?,
            MAX_METADATA_BYTES,
            true,
        )?;
        held.push(index_file);
        let original_manifest = OriginalManifest::decode(&manifest_raw, expected_family)?;
        let original_index = OriginalIndex::decode(&index_raw, &original_manifest)?;
        let manifest = original_manifest.metadata();
        let index = original_index.metadata()?;
        check(
            original_manifest.sha256()? == publication.canonical_manifest_sha256
                && original_index.sha256()? == publication.canonical_index_sha256
                && manifest.source_commit == registration.collection_source_commit
                && manifest.source_sha256 == registration.collection_source_sha256
                && manifest.runtime_sha256 == registration.collection_runtime_library_sha256
                && manifest.model_semantics_sha256 == registration.model_semantics_sha256
                && manifest.model_epoch == registration.model_epoch
                && manifest.context_epoch == registration.context_epoch
                && manifest.recipe_sha256 == registration.recipe_sha256
                && manifest.stage_id == registration.stage_id
                && index.record_count == publication.record_count
                && index.chunks.len() == publication.chunks.len(),
            "historical publication/manifest/index source and canonical seals differ",
        )?;
        validate_original_ancestry(&manifest, &index, &ancestors)?;
        let mut rows = BTreeMap::new();
        let mut lineage_rows = Vec::new();
        let indexed = index
            .origins
            .iter()
            .map(|o| (o.record_sha256.as_str(), o))
            .collect::<BTreeMap<_, _>>();
        check(
            indexed.len() == index.origins.len(),
            "duplicate PC original record index occurrence",
        )?;
        let mut seen_records = BTreeSet::new();
        let mut admissions = BTreeSet::new();
        let mut total = publication
            .manifest
            .bytes
            .checked_add(publication.index.bytes)
            .ok_or_else(|| invalid("historical bytes overflow"))?;
        let mut lock_files = BTreeMap::new();
        let mut bundle_files = BTreeMap::new();
        let mut admission_locks = BTreeMap::new();
        for admitted in &publication.admissions {
            let lock = if let Some((_, owner)) = lock_files.get(&admitted.input_lock.path) {
                Arc::clone(owner)
            } else {
                let (file, raw) = budget.read(
                    &archive_pin(&registration.root, &admitted.input_lock)?,
                    MAX_METADATA_BYTES,
                    true,
                )?;
                let lock = Arc::new(
                    rz_experiments::RoventInputLock::from_json(
                        std::str::from_utf8(&raw)
                            .map_err(|_| invalid("historical input lock UTF8"))?,
                    )
                    .map_err(|e| invalid(e.to_string()))?,
                );
                total = total
                    .checked_add(admitted.input_lock.bytes)
                    .ok_or_else(|| invalid("historical lock bytes overflow"))?;
                lock_files.insert(
                    admitted.input_lock.path.clone(),
                    (admitted.input_lock.clone(), Arc::clone(&lock)),
                );
                held.push(file);
                lock
            };
            check(
                lock.canonical_sha256 == admitted.canonical_input_lock_sha256,
                "historical full lock canonical seal differs from publication",
            )?;
            let bundle = if let Some(pin) = &admitted.runtime_bundle_manifest {
                let spec = if let Some((_, spec)) = bundle_files.get(&pin.path) {
                    Arc::clone(spec)
                } else {
                    let (file, raw) =
                        budget.read(&archive_pin(&registration.root, pin)?, 64 << 10, true)?;
                    let spec = Arc::new(
                        rz_eval::runtime_pin::CudaRuntimeBundleSpec::from_json(
                            std::str::from_utf8(&raw)
                                .map_err(|_| invalid("historical runtime bundle UTF8"))?,
                        )
                        .map_err(|error| invalid(error.to_string()))?,
                    );
                    total = total
                        .checked_add(pin.bytes)
                        .ok_or_else(|| invalid("historical bundle bytes overflow"))?;
                    bundle_files.insert(pin.path.clone(), (pin.clone(), Arc::clone(&spec)));
                    held.push(file);
                    spec
                };
                Some((pin.clone(), spec))
            } else {
                None
            };
            admission_locks.insert(admitted.native_admission_sha256.clone(), (lock, bundle));
        }
        for (declared, pin) in index.chunks.iter().zip(&publication.chunks) {
            check(
                declared.path == pin.path
                    && declared.sha256 == pin.sha256
                    && declared.bytes == pin.bytes,
                "publication chunk exact bytes changed",
            )?;
            total = total
                .checked_add(pin.bytes)
                .ok_or_else(|| invalid("historical total bytes overflow"))?;
            let (file, bytes) = budget.read(
                &archive_pin(&registration.root, pin)?,
                MAX_CHUNK_BYTES as u64,
                true,
            )?;
            held.push(file);
            let text =
                std::str::from_utf8(&bytes).map_err(|_| invalid("PC original chunk UTF8"))?;
            let mut chunk_count = 0u64;
            for line in text.lines() {
                budget.reserve(0)?;
                check(
                    !line.is_empty()
                        && line.len() <= rz_experiments::pals_data_v3::GENERATION_MAX_ROW_BYTES
                        && chunk_count < declared.records,
                    "bounded original PC JSONL row",
                )?;
                let row = OriginalPcRow::decode(line, &original_manifest)?;
                let evidence = match (&row, &original_manifest) {
                    (OriginalPcRow::Legacy(row), OriginalManifest::Legacy(manifest)) => {
                        verify_historical_record_evidence(row, manifest)?
                    }
                    (OriginalPcRow::Submitted(row), OriginalManifest::Submitted(manifest)) => {
                        verify_historical_submitted_record_evidence(row, manifest)?
                    }
                    _ => return Err(invalid("PC original row/manifest typed family mismatch")),
                };
                let role = role_name(row.role());
                let binding = registration
                    .role_bindings
                    .get(role)
                    .ok_or_else(|| invalid("historical row role was not registered"))?;
                check(
                    evidence.collection_source_commit == registration.collection_source_commit
                        && evidence.collection_source_sha256
                            == registration.collection_source_sha256
                        && evidence.collection_runtime_library_sha256
                            == registration.collection_runtime_library_sha256
                        && evidence.model_semantics_sha256 == binding.model_semantics_sha256
                        && evidence.model_identity_sha256 == binding.model_identity_sha256
                        && evidence.weights_sha256 == binding.weights_sha256
                        && evidence.encoding_sha256 == binding.encoding_sha256,
                    "historical row/root/role model source differs from caller registration",
                )?;
                let (lock, bundle) = admission_locks
                    .get(&evidence.native_admission_sha256)
                    .ok_or_else(|| {
                        invalid("historical row admission is missing its full input lock")
                    })?;
                join_historical_input_lock(
                    lock,
                    bundle.as_ref().map(|(pin, spec)| (pin, spec.as_ref())),
                    &row,
                    &evidence,
                )?;
                admissions.insert(evidence.native_admission_sha256);
                let data = row.authority(&original_manifest)?;
                check(
                    indexed
                        .get(data.origin.record_sha256.as_str())
                        .is_some_and(|origin| **origin == data.origin)
                        && seen_records.insert(data.origin.record_sha256.clone()),
                    "PC original chunk changed/duplicated its exact index occurrence",
                )?;
                lineage_rows.push(row.lineage(&data));
                let origin = data.origin.origin_sha256.clone();
                check(
                    rows.insert(origin, data).is_none(),
                    "duplicate historical origin",
                )?;
                chunk_count += 1;
            }
            check(
                chunk_count == declared.records,
                "PC original chunk row count",
            )?;
        }
        check(
            total == publication.total_payload_bytes
                && total <= MAX_TRAINER_OUTPUT_BYTES
                && rows.len() as u64 == publication.record_count
                && seen_records.len() == index.origins.len()
                && admissions.into_iter().collect::<Vec<_>>()
                    == publication.native_admission_sha256,
            "historical full payload/row/admission set did not close",
        )?;
        Ok(Self {
            registration,
            original: OriginalPcCorpus {
                publication: original_publication,
                manifest: original_manifest,
                index: original_index,
            },
            publication,
            manifest,
            index,
            rows,
            lineage_rows,
            held,
            ancestors,
        })
    }
    pub fn registration(&self) -> &RoventHistoricalDatasetRegistration {
        &self.registration
    }
    pub fn family(&self) -> RoventPcRecordFamily {
        self.original.publication.family()
    }
    pub(super) fn utility_lineage(&self) -> impl Iterator<Item = HistoricalDatasetLineage<'_>> {
        self.lineage_rows
            .iter()
            .map(|row| HistoricalDatasetLineage {
                split: row.origin.split,
                opening_id: &row.origin.opening_id,
                game_id: &row.origin.game_id,
                line_genealogy_id: &row.origin.line_genealogy_id,
                rules_state_sha256: &row.rules_state_sha256,
                transposition_sha256: &row.origin.transposition_sha256,
            })
    }
    pub fn manifest_sha256(&self) -> &str {
        &self.publication.canonical_manifest_sha256
    }
    pub fn index_sha256(&self) -> &str {
        &self.publication.canonical_index_sha256
    }
    pub fn manifest_file_sha256(&self) -> &str {
        &self.publication.manifest.sha256
    }
    pub fn index_file_sha256(&self) -> &str {
        &self.publication.index.sha256
    }
    pub fn same_original_metadata(&self, other: &Self) -> bool {
        self.original == other.original
    }
    pub fn original_metadata_json(&self) -> Result<Value, ArenaError> {
        self.original.json()
    }
    pub fn observation(&self) -> Value {
        json!({"schema":ROVENT_HISTORICAL_DATASET,"source_kind":ROVENT_RECOVERY_SOURCE_KIND,
        "publication_sha256":self.registration.publication.sha256,"record_count":self.rows.len(),"past_native_authority_reissued":false,
        "past_physical_completion_reissued":false,"historical_evidence_only":true,"execution_authorized":false})
    }
    pub fn revalidate(&self, budget: &RoventRecoveryReadBudget) -> Result<(), ArenaError> {
        for retained in &self.held {
            let (new_file, _) = budget.read(&retained.pin, retained.pin.bytes, false)?;
            check(
                new_file.identity == retained.identity,
                "historical dataset file identity changed while retained",
            )?;
        }
        Ok(())
    }
}
fn join_historical_input_lock(
    lock: &rz_experiments::RoventInputLock,
    bundle: Option<(
        &RoventPublicationFile,
        &rz_eval::runtime_pin::CudaRuntimeBundleSpec,
    )>,
    row: &OriginalPcRow,
    evidence: &crate::rovent_collect::RoventHistoricalRecordEvidence,
) -> Result<(), ArenaError> {
    evidence
        .root_descriptor
        .verify_against(lock)
        .map_err(|error| invalid(error.to_string()))?;
    crate::rovent_collect::verify_historical_external_source_input_lock(evidence, lock)?;
    let manifest = &lock.manifest;
    let engine = &manifest.engine;
    let model = match row.role() {
        GenerationRoleV3::Proposer => &engine.pc.proposer,
        GenerationRoleV3::Critic => &engine.pc.critic,
        GenerationRoleV3::Validator => engine
            .pc
            .validator
            .as_ref()
            .ok_or_else(|| invalid("historical V row is absent from the full loaded lock"))?,
    };
    let origin = &row.native().raw_retirement["root_admission"]["native_origin"];
    check(
        manifest.purpose == rz_experiments::RoventPurpose::Collection
            && lock.canonical_sha256 == evidence.collection_input_lock_sha256
            && engine.source_commit == evidence.collection_source_commit
            && engine.source_components == evidence.collection_source_components
            && model.weights.sha256 == evidence.weights_sha256
            && origin["model_implementation_sha256"] == model.implementation_sha256
            && origin["model_semantics"] == model.semantics
            && origin["encoding"] == model.encoding
            && engine.pc.backend_configuration_sha256
                == evidence.collection_native_configuration_sha256
            && engine
                .pc
                .runtime_artifacts
                .iter()
                .any(|file| file.sha256 == evidence.collection_runtime_library_sha256)
            && origin["teacher_profiles"]
                == serde_json::to_value(&engine.teachers).map_err(|e| invalid(e.to_string()))?,
        "historical raw origin differs from the full registered source/model/runtime/teacher lock",
    )?;
    // Mode's public wire spelling is defined by the lock; the native owner uses
    // the shared Rust enum's Debug spelling. Compare the actual common value.
    check(
        origin["mode"] == format!("{:?}", engine.mode),
        "historical root mode differs from its full lock",
    )?;
    match (bundle, &evidence.collection_runtime_bundle_sha256) {
        (None, None) => check(
            matches!(engine.pc.device, rz_experiments::RoventDevice::Cpu),
            "CUDA history is missing its original runtime bundle manifest",
        ),
        (Some((pin, spec)), Some(digest)) => {
            let actual = spec.digest().map_err(|error| invalid(error.to_string()))?;
            check(
                !matches!(engine.pc.device, rz_experiments::RoventDevice::Cpu)
                    && actual
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect::<String>()
                        == *digest
                    && manifest
                        .provenance
                        .iter()
                        .any(|file| file.sha256 == pin.sha256 && file.bytes == pin.bytes)
                    && spec.files.len() == engine.pc.runtime_artifacts.len()
                    && spec.files.iter().all(|expected| {
                        engine
                            .pc
                            .runtime_artifacts
                            .iter()
                            .filter(|actual| {
                                Path::new(&actual.path)
                                    .file_name()
                                    .and_then(|name| name.to_str())
                                    == Some(expected.filename.as_str())
                                    && actual.sha256 == expected.sha256
                                    && actual.bytes == expected.bytes
                            })
                            .count()
                            == 1
                    }),
                "historical runtime bundle semantics/aliases/roster/provenance do not match the original lock",
            )
        }
        _ => Err(invalid(
            "historical runtime bundle presence disagrees with original runtime evidence",
        )),
    }
}
mod checkpoint;
pub use checkpoint::*;
mod registration;
pub use registration::*;
mod reclamation;
pub(in crate::rovent_training_process) use reclamation::VerifiedHistoricalUnresolvedScopes;
pub use reclamation::*;
pub mod cli;
mod runtime;
pub use runtime::*;
mod entry;
pub use entry::*;
mod checkpoint_publication;
pub(in crate::rovent_training_process) use checkpoint_publication::CheckpointArchiveIo;
pub use checkpoint_publication::*;
mod r_auxiliary;
pub use r_auxiliary::*;
mod r_dataset;
pub use r_dataset::*;
#[cfg(test)]
pub(crate) mod metadata_wire_tests;
mod r_auxiliary_ledger;
#[cfg(test)]
mod submitted_wire_tests;
