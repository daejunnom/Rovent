//! Independent historical inspection for a /3-lock publication.
//!
//! This is deliberately not a `VerifiedRoventDataset`: it does not grant
//! trainer access, reconstruct a live admission, or claim original lock-file
//! custody for semantic snapshots.  Ancestry and learner role bindings still
//! require the separate training recovery path.

use super::*;
use rz_experiments::rovent_corpus_lock_v3::{
    publication::{
        RoventLockV3InputLockBytesOrigin, RoventLockV3Publication, RoventLockV3PublicationFile,
    },
    record::{RoventCorpusIndexLockV3, RoventCorpusManifestLockV3, RoventTrainingRecordLockV3},
};
use rz_experiments::{NativeRuntimeSourceV6, RoventInputLockV3};

/// Historical data only. Keeping the original file descriptors does not
/// recreate execution, worker, teacher, or physical-completion authority.
pub struct InspectedRoventLockV3Archive {
    publication: RoventLockV3Publication,
    manifest: RoventCorpusManifestLockV3,
    index: RoventCorpusIndexLockV3,
    held: Vec<HeldHistoricalFile>,
    semantic_lock_snapshots: usize,
}

/// Historical data with ancestry checked, but no loaded-role or trainer authority.
pub struct AncestryCheckedRoventLockV3Archive {
    archive: InspectedRoventLockV3Archive,
    original_ancestors: Vec<Arc<VerifiedRoventDataset>>,
    lock_v3_ancestors: Vec<Arc<AncestryCheckedRoventLockV3Archive>>,
    lock_v2_ancestors: Vec<Arc<super::lock_v2_archive::CheckedRoventLockV2HistoricalDataset>>,
}

/// Lexical only: reject protected names before metadata, symlink or file I/O.
/// This does not grant permission to read any other path.
pub(super) fn ensure_v3_nonsecret_path(path: &Path) -> Result<(), ArenaError> {
    for part in path.components() {
        let Component::Normal(part) = part else {
            continue;
        };
        let name = part
            .to_str()
            .ok_or_else(|| invalid("lock-v3 path is not UTF-8"))?
            .to_ascii_lowercase();
        let protected = name == ".env"
            || name.starts_with(".env.")
            || matches!(
                name.as_str(),
                ".ssh" | ".gnupg" | ".aws" | "authorized_keys" | "known_hosts"
            )
            || name.contains("credential")
            || name.contains("service-account")
            || name.contains("service_account")
            || name.contains("serviceaccount")
            || name.contains("api_key")
            || name.contains("api-key")
            || name.contains("apikey")
            || name.contains("private_key")
            || name.contains("private-key")
            || name.contains("secret")
            || name.contains("access_token")
            || name.contains("refresh_token")
            || ["id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"]
                .iter()
                .any(|key| name == *key || name.starts_with(&format!("{key}.")))
            || [".key", ".pem", ".p12", ".pfx", ".ppk"]
                .iter()
                .any(|suffix| name.ends_with(suffix));
        check(!protected, "lock-v3 protected path is not readable")?;
    }
    Ok(())
}

fn lock_v3_pin(
    root: &Path,
    pin: &RoventLockV3PublicationFile,
) -> Result<RoventTrainingFilePin, ArenaError> {
    ensure_v3_nonsecret_path(root)?;
    ensure_v3_nonsecret_path(Path::new(&pin.path))?;
    pin.validate()?;
    let relative = Path::new(&pin.path);
    check(
        root.is_absolute()
            && !relative.is_absolute()
            && relative
                .components()
                .all(|part| matches!(part, Component::Normal(_))),
        "lock-v3 publication file escapes registered root",
    )?;
    Ok(RoventTrainingFilePin {
        path: root.join(relative),
        sha256: pin.sha256.clone(),
        bytes: pin.bytes,
    })
}

fn text(raw: &[u8]) -> Result<&str, ArenaError> {
    std::str::from_utf8(raw).map_err(|_| invalid("lock-v3 historical UTF8"))
}

impl InspectedRoventLockV3Archive {
    /// Read exact registered bytes and validate the lock-v3 publication,
    /// indices, rows, Rules replay and runtime metadata. This is not learner
    /// admission: cross-publication ancestry and role binding remain separate.
    pub fn inspect(
        root: &Path,
        publication_pin: &RoventTrainingFilePin,
        budget: &RoventRecoveryReadBudget,
    ) -> Result<Self, ArenaError> {
        ensure_v3_nonsecret_path(root)?;
        ensure_v3_nonsecret_path(&publication_pin.path)?;
        check(
            root.is_absolute() && publication_pin.path == root.join("publication.json"),
            "lock-v3 exact publication registration",
        )?;
        let (file, raw) = budget.read(publication_pin, MAX_METADATA_BYTES, true)?;
        let mut held = vec![file];
        let publication = RoventLockV3Publication::from_json(text(&raw)?)?;
        let (file, raw) = budget.read(
            &lock_v3_pin(root, &publication.manifest)?,
            MAX_METADATA_BYTES,
            true,
        )?;
        held.push(file);
        let manifest = RoventCorpusManifestLockV3::from_json(text(&raw)?)?;
        let (file, raw) = budget.read(
            &lock_v3_pin(root, &publication.index)?,
            MAX_METADATA_BYTES,
            true,
        )?;
        held.push(file);
        let index = RoventCorpusIndexLockV3::from_json(text(&raw)?)?;
        let mut locks = BTreeMap::new();
        let mut paths = BTreeSet::new();
        let mut semantic_lock_snapshots = 0;
        let mut reference_lock_path: Option<String> = None;
        for admission in &publication.admissions {
            if paths.insert(admission.input_lock.path.clone()) {
                let (file, raw) = budget.read(
                    &lock_v3_pin(root, &admission.input_lock)?,
                    MAX_METADATA_BYTES,
                    true,
                )?;
                held.push(file);
                let lock = RoventInputLockV3::from_json(text(&raw)?)?;
                check(
                    lock.canonical_sha256 == admission.canonical_input_lock_sha256,
                    "lock-v3 original canonical lock differs from publication",
                )?;
                locks.insert(admission.input_lock.path.clone(), lock);
            }
            if admission.input_lock_bytes_origin
                == RoventLockV3InputLockBytesOrigin::ValidatedSemanticSnapshot
            {
                semantic_lock_snapshots += 1;
            }
            let lock = locks
                .get(&admission.input_lock.path)
                .ok_or_else(|| invalid("lock-v3 lock missing"))?;
            check(
                lock.canonical_sha256 == admission.canonical_input_lock_sha256,
                "lock-v3 reused lock canonical differs",
            )?;
            if let Some(reference_path) = &reference_lock_path {
                check(
                    locks.get(reference_path) == Some(lock),
                    "lock-v3 admissions do not use the same validated lock",
                )?;
            } else {
                manifest.validate(lock)?;
                index.validate(&manifest, lock)?;
                check(
                    manifest.sha256(lock)? == publication.canonical_manifest_sha256,
                    "lock-v3 publication manifest canonical mismatch",
                )?;
                reference_lock_path = Some(admission.input_lock.path.clone());
            }
            match &lock.manifest.system_runtime_lock.runtime {
                NativeRuntimeSourceV6::Cpu => check(
                    admission.cpu_runtime_library_sha256.as_deref()
                        == Some(publication.runtime_sha256.as_str())
                        && admission.system_runtime_identity_sha256.is_none(),
                    "lock-v3 CPU runtime publication mismatch",
                )?,
                NativeRuntimeSourceV6::SystemCuda(system) => check(
                    admission.system_runtime_identity_sha256.as_deref()
                        == Some(system.system_identity_sha256.as_str())
                        && admission.cpu_runtime_library_sha256.is_none()
                        && system
                            .libraries
                            .iter()
                            .filter(|item| {
                                item.role == "ort-core" && item.sha256 == publication.runtime_sha256
                            })
                            .count()
                            == 1,
                    "lock-v3 system CUDA publication mismatch",
                )?,
            }
        }
        check(
            index.sha256()? == publication.canonical_index_sha256
                && index.record_count == publication.record_count
                && manifest.registration.source_commit == publication.source_commit
                && manifest.registration.source_sha256 == publication.source_sha256
                && manifest.registration.runtime_sha256 == publication.runtime_sha256
                && index.chunks.len() == publication.chunks.len(),
            "lock-v3 publication index or source mismatch",
        )?;
        let indexed = index
            .origins
            .iter()
            .map(|origin| (origin.record_sha256.as_str(), origin))
            .collect::<BTreeMap<_, _>>();
        check(
            indexed.len() == index.origins.len(),
            "lock-v3 duplicate indexed row",
        )?;
        let admissions = publication
            .admissions
            .iter()
            .map(|entry| (entry.native_admission_sha256.as_str(), entry))
            .collect::<BTreeMap<_, _>>();
        check(
            admissions.len() == publication.admissions.len(),
            "lock-v3 duplicate admission",
        )?;
        let mut records = BTreeSet::new();
        let mut origins = BTreeSet::new();
        let mut used = BTreeSet::new();
        for (declared, pin) in index.chunks.iter().zip(&publication.chunks) {
            check(
                declared.path == pin.path
                    && declared.sha256 == pin.sha256
                    && declared.bytes == pin.bytes,
                "lock-v3 chunk pin differs from index",
            )?;
            let (file, raw) =
                budget.read(&lock_v3_pin(root, pin)?, MAX_CHUNK_BYTES as u64, true)?;
            held.push(file);
            let mut count = 0u64;
            for line in text(&raw)?.lines() {
                budget.reserve(0)?;
                check(
                    !line.is_empty()
                        && line.len() <= rz_experiments::pals_data_v3::GENERATION_MAX_ROW_BYTES
                        && count < declared.records,
                    "lock-v3 bounded row",
                )?;
                let record = RoventTrainingRecordLockV3::from_json(line)?;
                let row = &record.0;
                let admission_sha = row.native.raw_retirement["root_admission_sha256"]
                    .as_str()
                    .ok_or_else(|| invalid("lock-v3 row admission missing"))?;
                let admission = admissions
                    .get(admission_sha)
                    .copied()
                    .ok_or_else(|| invalid("lock-v3 row admission absent from publication"))?;
                let lock = locks
                    .get(&admission.input_lock.path)
                    .ok_or_else(|| invalid("lock-v3 row lock absent"))?;
                let evidence = crate::rovent_collect::verify_historical_lock_v3_record_evidence(
                    &record, &manifest, lock,
                )?;
                check(
                    evidence.root_sha256 == admission_sha
                        && evidence.input_lock_sha256 == admission.canonical_input_lock_sha256,
                    "lock-v3 row root/lock does not join publication",
                )?;
                let origin = &row.native.raw_retirement["root_admission"]["native_origin"];
                let observed = &origin["startup_observation"];
                check(
                    observed["runtime_library_sha256"].as_str()
                        == Some(publication.runtime_sha256.as_str()),
                    "lock-v3 row runtime differs from publication",
                )?;
                match &lock.manifest.system_runtime_lock.runtime {
                    NativeRuntimeSourceV6::Cpu => check(
                        observed["provider"].as_str() == Some("cpu")
                            && observed["provider_ready"].as_bool() == Some(true)
                            && admission.cpu_runtime_library_sha256.as_deref()
                                == Some(publication.runtime_sha256.as_str()),
                        "lock-v3 CPU runtime observation mismatch",
                    )?,
                    NativeRuntimeSourceV6::SystemCuda(system) => check(
                        observed["system_runtime_identity_sha256"].as_str()
                            == Some(system.system_identity_sha256.as_str())
                            && observed["provider_ready"].as_bool() == Some(true)
                            && admission.system_runtime_identity_sha256.as_deref()
                                == Some(system.system_identity_sha256.as_str()),
                        "lock-v3 CUDA runtime observation mismatch",
                    )?,
                }
                let expected = indexed
                    .get(row.record_sha256.as_str())
                    .ok_or_else(|| invalid("lock-v3 row missing from index"))?;
                let actual = RoventCorpusOrigin {
                    record_sha256: row.record_sha256.clone(),
                    input_sha256: row.input_sha256.clone(),
                    origin_sha256: row.origin_sha256.clone(),
                    split: row.split,
                    game_id: row.input.game_id.clone(),
                    opening_id: row.input.opening_id.clone(),
                    line_genealogy_id: row.input.line_genealogy_id.clone(),
                    transposition_sha256: row.input.transposition_sha256.clone(),
                    actor_id: row.lineage.actor_id.clone(),
                    teacher_actor_id: row.lineage.teacher_actor_id.clone(),
                    ancestor_origin_sha256: row.lineage.ancestor_origin_sha256.clone(),
                };
                check(
                    **expected == actual
                        && records.insert(row.record_sha256.clone())
                        && origins.insert(row.origin_sha256.clone()),
                    "lock-v3 index row mismatch or duplicate record/origin",
                )?;
                used.insert(admission_sha.to_owned());
                count += 1;
            }
            check(
                count == declared.records,
                "lock-v3 chunk row count mismatch",
            )?;
        }
        check(
            records.len() as u64 == publication.record_count
                && records.len() == index.origins.len()
                && used.into_iter().collect::<Vec<_>>() == publication.native_admission_sha256,
            "lock-v3 complete record/admission coverage",
        )?;
        Ok(Self {
            publication,
            manifest,
            index,
            held,
            semantic_lock_snapshots,
        })
    }

    pub fn observation(&self) -> Value {
        json!({"schema":"rz-rovent-lock-v3-historical-inspection/1", "record_count":self.index.record_count,
            "publication_source_sha256":self.publication.source_sha256, "canonical_index_sha256":self.publication.canonical_index_sha256,
            "semantic_lock_snapshot_admissions":self.semantic_lock_snapshots, "corpus_id":self.manifest.registration.corpus_id,
            "cross_publication_ancestry_checked":false, "learner_role_bindings_checked":false,
            "trainer_admission":false, "past_native_authority_reissued":false, "physical_completion_reissued":false,
            "physical_completion_proven":false, "original_file_custody_proven":false, "publication_ack_verified":false})
    }

    pub fn revalidate(&self, budget: &RoventRecoveryReadBudget) -> Result<(), ArenaError> {
        for retained in &self.held {
            ensure_v3_nonsecret_path(&retained.pin.path)?;
        }
        for retained in &self.held {
            let (file, _) = budget.read(&retained.pin, retained.pin.bytes, false)?;
            check(
                file.identity == retained.identity,
                "lock-v3 historical file identity changed",
            )?;
        }
        Ok(())
    }
}

impl AncestryCheckedRoventLockV3Archive {
    /// Reuses the original index seals and ancestor owners; no family is
    /// re-encoded, and this check grants no learner or current-run authority.
    pub fn inspect(
        root: &Path,
        publication_pin: &RoventTrainingFilePin,
        original_ancestors: Vec<Arc<VerifiedRoventDataset>>,
        lock_v3_ancestors: Vec<Arc<Self>>,
        lock_v2_ancestors: Vec<Arc<super::lock_v2_archive::CheckedRoventLockV2HistoricalDataset>>,
        budget: &RoventRecoveryReadBudget,
    ) -> Result<Self, ArenaError> {
        for ancestor in &original_ancestors {
            for retained in &ancestor.held {
                ensure_v3_nonsecret_path(&retained.pin.path)?;
            }
        }
        for ancestor in &lock_v3_ancestors {
            ancestor.ensure_nonsecret_paths()?;
        }
        for ancestor in &lock_v2_ancestors {
            ancestor.ensure_v3_nonsecret_paths()?;
        }
        check(
            original_ancestors
                .len()
                .saturating_add(lock_v3_ancestors.len())
                .saturating_add(lock_v2_ancestors.len())
                <= 256,
            "lock-v3 bounded historical ancestors",
        )?;
        let archive = InspectedRoventLockV3Archive::inspect(root, publication_pin, budget)?;
        // A retained ancestor's metadata is not proof that its original files
        // still have their pinned bytes and identity at this admission point.
        for ancestor in &original_ancestors {
            ancestor.revalidate(budget)?;
        }
        // The audit below requires the full transitive ancestor set, so each
        // retained /3 archive is checked once without recursive revalidation.
        for ancestor in &lock_v3_ancestors {
            ancestor.archive.revalidate(budget)?;
        }
        for ancestor in &lock_v2_ancestors {
            ancestor.revalidate(budget)?;
        }
        let v2_metadata = lock_v2_ancestors
            .iter()
            .map(|ancestor| ancestor.historical_ancestry_metadata())
            .collect::<Vec<_>>();
        let manifest = HistoricalCollectionMetadata::submitted(&archive.manifest.registration);
        let index = HistoricalIndexMetadata::from_validated_parts(
            archive.index.record_count,
            archive.index.chunks.clone(),
            archive.index.origins.clone(),
            archive.publication.canonical_index_sha256.clone(),
        );
        let lock_metadata = lock_v3_ancestors
            .iter()
            .map(|a| {
                (
                    HistoricalCollectionMetadata::submitted(&a.archive.manifest.registration),
                    HistoricalIndexMetadata::from_validated_parts(
                        a.archive.index.record_count,
                        a.archive.index.chunks.clone(),
                        a.archive.index.origins.clone(),
                        a.archive.publication.canonical_index_sha256.clone(),
                    ),
                )
            })
            .collect::<Vec<_>>();
        let mut ancestors = original_ancestors
            .iter()
            .map(|a| (a.index_sha256(), &a.manifest, &a.index))
            .collect::<Vec<_>>();
        ancestors.extend(
            lock_v3_ancestors
                .iter()
                .zip(&lock_metadata)
                .map(|(owner, (m, i))| {
                    (
                        owner.archive.publication.canonical_index_sha256.as_str(),
                        m,
                        i,
                    )
                }),
        );
        ancestors.extend(
            v2_metadata
                .iter()
                .map(|(manifest, index, digest)| (*digest, manifest, index)),
        );
        // The shared ancestry audit enforces exact, transitive source sets,
        // but does not itself rule out a cycle. Check every provided family,
        // including the current archive, without interpreting any new authority.
        let mut pending = ancestors
            .iter()
            .map(|(sha, source, _)| ((*sha).to_owned(), source.ancestor_index_sha256.clone()))
            .collect::<BTreeMap<_, _>>();
        check(
            pending.len() == ancestors.len()
                && !pending.contains_key(&archive.publication.canonical_index_sha256),
            "lock-v3 duplicate or self ancestor",
        )?;
        pending.insert(
            archive.publication.canonical_index_sha256.clone(),
            manifest.ancestor_index_sha256.clone(),
        );
        while !pending.is_empty() {
            let completed = pending
                .iter()
                .filter_map(|(sha, dependencies)| {
                    dependencies
                        .iter()
                        .all(|dependency| !pending.contains_key(dependency))
                        .then_some(sha.clone())
                })
                .collect::<Vec<_>>();
            check(
                !completed.is_empty(),
                "lock-v3 cyclic ancestor declarations",
            )?;
            for sha in completed {
                pending.remove(&sha);
            }
        }
        audit_original_ancestry(&manifest, &index, &ancestors)?;
        Ok(Self {
            archive,
            original_ancestors,
            lock_v3_ancestors,
            lock_v2_ancestors,
        })
    }

    pub fn observation(&self) -> Value {
        let mut result = self.archive.observation();
        result["cross_publication_ancestry_checked"] = json!(true);
        result["learner_role_bindings_checked"] = json!(false);
        result["ancestor_count"] = json!(
            self.original_ancestors.len()
                + self.lock_v3_ancestors.len()
                + self.lock_v2_ancestors.len()
        );
        // This does not cover P/C-to-R joins, current parent sample admission,
        // Python intake, or trainer-process and physical execution authority.
        result
    }

    pub fn revalidate(&self, budget: &RoventRecoveryReadBudget) -> Result<(), ArenaError> {
        self.ensure_nonsecret_paths()?;
        self.archive.revalidate(budget)?;
        for ancestor in &self.original_ancestors {
            ancestor.revalidate(budget)?;
        }
        for ancestor in &self.lock_v3_ancestors {
            ancestor.archive.revalidate(budget)?;
        }
        for ancestor in &self.lock_v2_ancestors {
            ancestor.revalidate(budget)?;
        }
        Ok(())
    }

    fn ensure_nonsecret_paths(&self) -> Result<(), ArenaError> {
        for retained in &self.archive.held {
            ensure_v3_nonsecret_path(&retained.pin.path)?;
        }
        for ancestor in &self.original_ancestors {
            for retained in &ancestor.held {
                ensure_v3_nonsecret_path(&retained.pin.path)?;
            }
        }
        for ancestor in &self.lock_v3_ancestors {
            ancestor.ensure_nonsecret_paths()?;
        }
        for ancestor in &self.lock_v2_ancestors {
            ancestor.ensure_v3_nonsecret_paths()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secret_names_are_rejected_lexically_before_read_credit() {
        let base = std::env::temp_dir().join("rovent-v3-lexical-test");
        for name in [
            ".env",
            ".ENV.local",
            ".ssh",
            "id_ed25519",
            "id_rsa.pub",
            "service-account.json",
            "service_account.json",
            "credentials.json",
            "api_key.txt",
            "client-secret.json",
            "private.pem",
            "key.p12",
        ] {
            assert!(ensure_v3_nonsecret_path(&base.join(name)).is_err());
        }
        assert!(ensure_v3_nonsecret_path(&base.join("chunks").join("0001.jsonl")).is_ok());

        let root = base.join(".ssh");
        let pin = RoventTrainingFilePin {
            path: root.join("publication.json"),
            sha256: "a".repeat(64),
            bytes: 1,
        };
        let budget = RoventRecoveryReadBudget::new(
            1024,
            Instant::now() + std::time::Duration::from_secs(60),
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
        assert!(InspectedRoventLockV3Archive::inspect(&root, &pin, &budget).is_err());
        assert_eq!(budget.observation()["charged_bytes"], 0);
    }

    #[test]
    fn publication_pin_cannot_escape_archive_root() {
        let root = std::env::temp_dir().join("rovent-lock-v3-pin-test");
        let mut pin = RoventLockV3PublicationFile {
            path: "chunks/0001.jsonl".into(),
            sha256: "a".repeat(64),
            bytes: 1,
        };
        assert_eq!(
            lock_v3_pin(&root, &pin).unwrap().path,
            root.join("chunks/0001.jsonl")
        );
        for path in [
            "../outside",
            "chunks/../../outside",
            "/outside",
            "chunks\\outside",
        ] {
            pin.path = path.into();
            assert!(lock_v3_pin(&root, &pin).is_err());
        }
    }
}
