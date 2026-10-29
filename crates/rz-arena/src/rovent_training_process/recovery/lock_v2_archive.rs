//! Independent historical inspection for a /2-lock publication.
//!
//! This is deliberately not a `VerifiedRoventDataset`: it does not grant
//! trainer access, reconstruct a live admission, or claim original lock-file
//! custody for semantic snapshots.  Ancestry and learner role bindings still
//! require the separate training recovery path.

use super::*;
use rz_experiments::rovent_corpus_lock_v2::{
    publication::{RoventLockV2InputLockBytesOrigin, RoventLockV2Publication, RoventLockV2PublicationFile},
    record::{RoventCorpusIndexLockV2, RoventCorpusManifestLockV2, RoventTrainingRecordLockV2},
};
use rz_experiments::{RoventDevice, RoventInputLockV2};

/// Historical data only. Keeping the original file descriptors does not
/// recreate execution, worker, teacher, or physical-completion authority.
pub struct InspectedRoventLockV2Archive {
    publication: RoventLockV2Publication,
    manifest: RoventCorpusManifestLockV2,
    index: RoventCorpusIndexLockV2,
    held: Vec<HeldHistoricalFile>,
    semantic_lock_snapshots: usize,
}

fn lock_v2_pin(root: &Path, pin: &RoventLockV2PublicationFile) -> Result<RoventTrainingFilePin, ArenaError> {
    pin.validate()?;
    let relative = Path::new(&pin.path);
    check(
        root.is_absolute()
            && !relative.is_absolute()
            && relative.components().all(|part| matches!(part, Component::Normal(_))),
        "lock-v2 publication file escapes registered root",
    )?;
    Ok(RoventTrainingFilePin { path: root.join(relative), sha256: pin.sha256.clone(), bytes: pin.bytes })
}

fn text(raw: &[u8]) -> Result<&str, ArenaError> {
    std::str::from_utf8(raw).map_err(|_| invalid("lock-v2 historical UTF8"))
}

impl InspectedRoventLockV2Archive {
    /// Read exact registered bytes and validate the lock-v2 publication,
    /// indices, rows, Rules replay and runtime metadata. This is not learner
    /// admission: cross-publication ancestry and role binding remain separate.
    pub fn inspect(
        root: &Path,
        publication_pin: &RoventTrainingFilePin,
        budget: &RoventRecoveryReadBudget,
    ) -> Result<Self, ArenaError> {
        check(root.is_absolute() && publication_pin.path == root.join("publication.json"),
            "lock-v2 exact publication registration")?;
        let (file, raw) = budget.read(publication_pin, MAX_METADATA_BYTES, true)?;
        let mut held = vec![file];
        let publication = RoventLockV2Publication::from_json(text(&raw)?)?;
        let (file, raw) = budget.read(&lock_v2_pin(root, &publication.manifest)?, MAX_METADATA_BYTES, true)?;
        held.push(file);
        let manifest = RoventCorpusManifestLockV2::from_json(text(&raw)?)?;
        let (file, raw) = budget.read(&lock_v2_pin(root, &publication.index)?, MAX_METADATA_BYTES, true)?;
        held.push(file);
        let index = RoventCorpusIndexLockV2::from_json(text(&raw)?)?;
        let mut locks = BTreeMap::new();
        let mut bundle_specs = BTreeMap::new();
        let mut paths = BTreeSet::new();
        let mut semantic_lock_snapshots = 0;
        let mut reference_lock_path: Option<String> = None;
        for admission in &publication.admissions {
            if paths.insert(admission.input_lock.path.clone()) {
                let (file, raw) = budget.read(&lock_v2_pin(root, &admission.input_lock)?, MAX_METADATA_BYTES, true)?;
                held.push(file);
                let lock = RoventInputLockV2::from_json(text(&raw)?)?;
                check(lock.canonical_sha256 == admission.canonical_input_lock_sha256,
                    "lock-v2 original canonical lock differs from publication")?;
                locks.insert(admission.input_lock.path.clone(), lock);
            }
            if admission.input_lock_bytes_origin == RoventLockV2InputLockBytesOrigin::ValidatedSemanticSnapshot {
                semantic_lock_snapshots += 1;
            }
            let lock = locks.get(&admission.input_lock.path).ok_or_else(|| invalid("lock-v2 lock missing"))?;
            check(lock.canonical_sha256 == admission.canonical_input_lock_sha256,
                "lock-v2 reused lock canonical differs")?;
            if let Some(reference_path) = &reference_lock_path {
                check(locks.get(reference_path) == Some(lock),
                    "lock-v2 admissions do not use the same validated lock")?;
            } else {
                manifest.validate(lock)?;
                index.validate(&manifest, lock)?;
                check(manifest.sha256(lock)? == publication.canonical_manifest_sha256,
                    "lock-v2 publication manifest canonical mismatch")?;
                reference_lock_path = Some(admission.input_lock.path.clone());
            }
            if let Some(pin) = &admission.runtime_bundle_manifest {
                if !bundle_specs.contains_key(&pin.path) {
                    let (file, raw) = budget.read(&lock_v2_pin(root, pin)?, 64 << 10, true)?;
                    held.push(file);
                    let spec = rz_eval::runtime_pin::CudaRuntimeBundleSpec::from_json(text(&raw)?)
                        .map_err(|error| invalid(error.to_string()))?;
                    bundle_specs.insert(pin.path.clone(), spec);
                }
            }
        }
        check(index.sha256()? == publication.canonical_index_sha256
            && index.record_count == publication.record_count
            && manifest.registration.source_commit == publication.source_commit
            && manifest.registration.source_sha256 == publication.source_sha256
            && manifest.registration.runtime_sha256 == publication.runtime_sha256
            && index.chunks.len() == publication.chunks.len(),
            "lock-v2 publication index or source mismatch")?;
        let indexed = index.origins.iter().map(|origin| (origin.record_sha256.as_str(), origin)).collect::<BTreeMap<_, _>>();
        check(indexed.len() == index.origins.len(), "lock-v2 duplicate indexed row")?;
        let admissions = publication.admissions.iter().map(|entry| (entry.native_admission_sha256.as_str(), entry)).collect::<BTreeMap<_, _>>();
        check(admissions.len() == publication.admissions.len(), "lock-v2 duplicate admission")?;
        let mut records = BTreeSet::new();
        let mut used = BTreeSet::new();
        for (declared, pin) in index.chunks.iter().zip(&publication.chunks) {
            check(declared.path == pin.path && declared.sha256 == pin.sha256 && declared.bytes == pin.bytes,
                "lock-v2 chunk pin differs from index")?;
            let (file, raw) = budget.read(&lock_v2_pin(root, pin)?, MAX_CHUNK_BYTES as u64, true)?;
            held.push(file);
            let mut count = 0u64;
            for line in text(&raw)?.lines() {
                budget.reserve(0)?;
                check(!line.is_empty() && line.len() <= rz_experiments::pals_data_v3::GENERATION_MAX_ROW_BYTES && count < declared.records,
                    "lock-v2 bounded row")?;
                let record = RoventTrainingRecordLockV2::from_json(line)?;
                let row = &record.0;
                let admission_sha = row.native.raw_retirement["root_admission_sha256"].as_str()
                    .ok_or_else(|| invalid("lock-v2 row admission missing"))?;
                let admission = admissions.get(admission_sha).copied()
                    .ok_or_else(|| invalid("lock-v2 row admission absent from publication"))?;
                let lock = locks.get(&admission.input_lock.path).ok_or_else(|| invalid("lock-v2 row lock absent"))?;
                let evidence = crate::rovent_collect::verify_historical_lock_v2_record_evidence(&record, &manifest, lock)?;
                check(evidence.root_sha256 == admission_sha && evidence.input_lock_sha256 == admission.canonical_input_lock_sha256,
                    "lock-v2 row root/lock does not join publication")?;
                let observed = &row.native.raw_retirement["root_admission"]["native_origin"]["startup_observation"];
                let runtime = observed["runtime_library_sha256"].as_str().ok_or_else(|| invalid("lock-v2 runtime missing"))?;
                check(runtime == publication.runtime_sha256 && lock.manifest.engine.pc.runtime_artifacts.iter().any(|v| v.sha256 == runtime),
                    "lock-v2 runtime differs from original lock")?;
                match (&admission.runtime_bundle_manifest, observed["runtime_bundle_codec_sha256"].as_str()) {
                    (None, None) => check(matches!(lock.manifest.engine.pc.device, RoventDevice::Cpu), "lock-v2 CUDA bundle missing")?,
                    (Some(pin), Some(digest)) => {
                        let spec = bundle_specs.get(&pin.path).ok_or_else(|| invalid("lock-v2 bundle missing"))?;
                        let actual = spec.digest().map_err(|error| invalid(error.to_string()))?;
                        check(!matches!(lock.manifest.engine.pc.device, RoventDevice::Cpu)
                            && actual.iter().map(|b| format!("{b:02x}")).collect::<String>() == digest
                            && lock.manifest.provenance.iter().any(|v| v.sha256 == pin.sha256 && v.bytes == pin.bytes)
                            && spec.files.len() == lock.manifest.engine.pc.runtime_artifacts.len()
                            && spec.files.iter().all(|expected| lock.manifest.engine.pc.runtime_artifacts.iter().filter(|actual|
                                Path::new(&actual.path).file_name().and_then(|v| v.to_str()) == Some(expected.filename.as_str())
                                && actual.sha256 == expected.sha256 && actual.bytes == expected.bytes).count() == 1),
                            "lock-v2 runtime bundle differs from lock")?;
                    }
                    _ => return Err(invalid("lock-v2 runtime bundle presence mismatch")),
                }
                let expected = indexed.get(row.record_sha256.as_str()).ok_or_else(|| invalid("lock-v2 row missing from index"))?;
                let actual = RoventCorpusOrigin { record_sha256: row.record_sha256.clone(), input_sha256: row.input_sha256.clone(),
                    origin_sha256: row.origin_sha256.clone(), split: row.split, game_id: row.input.game_id.clone(),
                    opening_id: row.input.opening_id.clone(), line_genealogy_id: row.input.line_genealogy_id.clone(),
                    transposition_sha256: row.input.transposition_sha256.clone(), actor_id: row.lineage.actor_id.clone(),
                    teacher_actor_id: row.lineage.teacher_actor_id.clone(), ancestor_origin_sha256: row.lineage.ancestor_origin_sha256.clone() };
                check(**expected == actual && records.insert(row.record_sha256.clone()), "lock-v2 index row mismatch or duplicate")?;
                used.insert(admission_sha.to_owned());
                count += 1;
            }
            check(count == declared.records, "lock-v2 chunk row count mismatch")?;
        }
        check(records.len() as u64 == publication.record_count && records.len() == index.origins.len()
            && used.into_iter().collect::<Vec<_>>() == publication.native_admission_sha256,
            "lock-v2 complete record/admission coverage")?;
        Ok(Self { publication, manifest, index, held, semantic_lock_snapshots })
    }

    pub fn observation(&self) -> Value {
        json!({"schema":"rz-rovent-lock-v2-historical-inspection/1", "record_count":self.index.record_count,
            "publication_source_sha256":self.publication.source_sha256, "canonical_index_sha256":self.publication.canonical_index_sha256,
            "semantic_lock_snapshot_admissions":self.semantic_lock_snapshots, "corpus_id":self.manifest.registration.corpus_id,
            "cross_publication_ancestry_checked":false, "learner_role_bindings_checked":false,
            "trainer_admission":false, "past_native_authority_reissued":false, "physical_completion_reissued":false,
            "physical_completion_proven":false, "original_file_custody_proven":false})
    }

    pub fn revalidate(&self, budget: &RoventRecoveryReadBudget) -> Result<(), ArenaError> {
        for retained in &self.held {
            let (file, _) = budget.read(&retained.pin, retained.pin.bytes, false)?;
            check(file.identity == retained.identity, "lock-v2 historical file identity changed")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publication_pin_cannot_escape_archive_root() {
        let root = std::env::temp_dir().join("rovent-lock-v2-pin-test");
        let mut pin = RoventLockV2PublicationFile {
            path: "chunks/0001.jsonl".into(),
            sha256: "a".repeat(64),
            bytes: 1,
        };
        assert_eq!(lock_v2_pin(&root, &pin).unwrap().path, root.join("chunks/0001.jsonl"));
        for path in ["../outside", "chunks/../../outside", "/outside", "chunks\\outside"] {
            pin.path = path.into();
            assert!(lock_v2_pin(&root, &pin).is_err());
        }
    }
}
