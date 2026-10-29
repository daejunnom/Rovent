//! Durable publication building blocks for the input-lock /2 family.
//!
//! A captured source is constructed from the original live admission only.
//! This module does not construct a retired record, a writer, or a publication
//! authority from serialized input. The /2 record owner must connect it later.
use super::*;
use rz_experiments::rovent_corpus_lock_v2::{
    publication::{
        ROVENT_LOCK_V2_PUBLICATION_DOMAIN, RoventLockV2InputLockBytesOrigin,
        RoventLockV2Publication, RoventLockV2PublicationAdmission, RoventLockV2PublicationFile,
    },
    record::{RoventCorpusIndexLockV2, RoventCorpusManifestLockV2},
};
use rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2;
use std::sync::Arc;

/// Captures the lock's validated semantic serialization. It is deliberately
/// not described as the original lock file's bytes.
#[allow(dead_code)] // Connected by the forthcoming original /2 record owner.
pub(super) struct LockV2PublicationSource {
    admission_sha256: String,
    lock: rz_experiments::RoventInputLockV2,
    lock_bytes: Vec<u8>,
    bundle: Option<(rz_experiments::ArtifactRef, Vec<u8>)>,
}

#[allow(dead_code)] // This does not issue a publication capability.
impl LockV2PublicationSource {
    pub(super) fn capture(admission: &RoventCollectionRootAdmissionV2) -> Result<Self, ArenaError> {
        let origin = admission.native_origin();
        let lock = origin.original_input_lock();
        lock.validate()?;
        check(
            origin.raw_receipt()["input_lock_sha256"] == lock.canonical_sha256,
            "original /2 native origin and input lock differ",
        )?;
        let lock_bytes = serde_json::to_vec(lock).map_err(|error| invalid(error.to_string()))?;
        check(lock_bytes.len() <= 16 * 1024 * 1024, "/2 lock metadata cap")?;
        let bundle = origin
            .runtime_bundle_publication_bytes()
            .map_err(|error| invalid(error.to_string()))?;
        Ok(Self {
            admission_sha256: admission.raw_sha256().to_owned(),
            lock: lock.clone(),
            lock_bytes,
            bundle,
        })
    }

    pub(super) fn same_source(&self, other: &Self) -> bool {
        self.admission_sha256 == other.admission_sha256
            && self.lock == other.lock
            && self.lock_bytes == other.lock_bytes
            && self.bundle == other.bundle
    }
}

impl publication::PublicationSourceBytes for LockV2PublicationSource {
    fn canonical_lock_sha256(&self) -> &str {
        &self.lock.canonical_sha256
    }
    fn lock_bytes(&self) -> &[u8] {
        &self.lock_bytes
    }
    fn runtime_bundle(&self) -> Option<(&rz_experiments::ArtifactRef, &[u8])> {
        self.bundle
            .as_ref()
            .map(|(pin, bytes)| (pin, bytes.as_slice()))
    }
}

fn file(pin: publication::RoventPublicationFile) -> RoventLockV2PublicationFile {
    RoventLockV2PublicationFile {
        path: pin.path,
        sha256: pin.sha256,
        bytes: pin.bytes,
    }
}

/// Reuses the existing bounded, create-new, fsync and atomic-commit path.
/// This remains crate-private until a /2 writer consumes real retired rows and
/// holds their original admissions through physical completion.
#[allow(dead_code)]
pub(super) fn publish_lock_v2_payloads(
    directory: &cap_std::fs::Dir,
    until: std::time::Instant,
    cancel: &std::sync::atomic::AtomicBool,
    manifest: &RoventCorpusManifestLockV2,
    index: &RoventCorpusIndexLockV2,
    admissions: &BTreeMap<String, Arc<LockV2PublicationSource>>,
) -> Result<RoventLockV2Publication, ArenaError> {
    check(
        !admissions.is_empty(),
        "/2 publication requires original admissions",
    )?;
    for (key, source) in admissions {
        check(
            key == &source.admission_sha256,
            "/2 publication admission identity differs",
        )?;
        manifest.validate(&source.lock)?;
        index.validate(manifest, &source.lock)?;
    }
    let manifest_bytes =
        serde_json::to_vec(manifest).map_err(|error| invalid(error.to_string()))?;
    let index_bytes = serde_json::to_vec(index).map_err(|error| invalid(error.to_string()))?;
    publication::publish_original_payloads(
        directory,
        until,
        cancel,
        publication::RoventPublicationMaterial {
            manifest_bytes: &manifest_bytes,
            index_bytes: &index_bytes,
            chunks: &index.chunks,
            admissions,
        },
        |manifest_file, index_file, chunks, entries, total_payload_bytes| {
            let publication = RoventLockV2Publication {
                schema: ROVENT_LOCK_V2_PUBLICATION_DOMAIN.into(),
                manifest: file(manifest_file),
                index: file(index_file),
                chunks: chunks.into_iter().map(file).collect(),
                canonical_manifest_sha256: index.manifest_sha256.clone(),
                canonical_index_sha256: index.sha256()?,
                source_commit: manifest.registration.source_commit.clone(),
                source_sha256: manifest.registration.source_sha256.clone(),
                runtime_sha256: manifest.registration.runtime_sha256.clone(),
                record_count: index.record_count,
                total_payload_bytes,
                native_admission_sha256: admissions.keys().cloned().collect(),
                admissions: entries
                    .into_iter()
                    .map(|entry| RoventLockV2PublicationAdmission {
                        native_admission_sha256: entry.native_admission_sha256,
                        input_lock: file(entry.input_lock),
                        input_lock_bytes_origin:
                            RoventLockV2InputLockBytesOrigin::ValidatedSemanticSnapshot,
                        canonical_input_lock_sha256: entry.canonical_input_lock_sha256,
                        runtime_bundle_manifest: entry.runtime_bundle_manifest.map(file),
                    })
                    .collect(),
            };
            publication.validate()?;
            Ok(publication)
        },
    )
}
