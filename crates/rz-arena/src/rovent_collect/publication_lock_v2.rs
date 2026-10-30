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
    publish_lock_v2_payloads_authorized(directory, until, cancel, manifest, index, admissions, || Ok(()))
}

#[cfg(feature = "pals-collection-onnx")]
pub(super) fn publish_lock_v2_payloads_speculative(
    directory: &cap_std::fs::Dir,
    until: std::time::Instant,
    cancel: &std::sync::atomic::AtomicBool,
    manifest: &RoventCorpusManifestLockV2,
    index: &RoventCorpusIndexLockV2,
    admissions: &BTreeMap<String, Arc<LockV2PublicationSource>>,
    game_gate: &std::sync::atomic::AtomicU64,
) -> Result<RoventLockV2Publication, ArenaError> {
    use std::sync::atomic::Ordering;
    use super::collection_hooks_core_v2::{GAME_HAS_ABORT, GAME_IDLE, GAME_PHASE_MASK, GAME_PUBLICATION_GRANTED};
    let expected = game_gate.load(Ordering::Acquire);
    check(expected & (GAME_HAS_ABORT | GAME_PUBLICATION_GRANTED) == 0
        && expected & GAME_PHASE_MASK == GAME_IDLE,
        "speculative publication original game is aborted, busy, or already authorized")?;
    publish_lock_v2_payloads_authorized(directory, until, cancel, manifest, index, admissions, || {
        // This CAS is the logical publication authorization point. A UCI stop
        // that sets abort first prevents the final hard-link; a later stop is
        // ordered after authorization even if file visibility is later.
        game_gate.compare_exchange(expected, expected | GAME_PUBLICATION_GRANTED,
            Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| invalid("speculative publication game changed or was aborted"))?;
        Ok(())
    })
}

fn publish_lock_v2_payloads_authorized(
    directory: &cap_std::fs::Dir,
    until: std::time::Instant,
    cancel: &std::sync::atomic::AtomicBool,
    manifest: &RoventCorpusManifestLockV2,
    index: &RoventCorpusIndexLockV2,
    admissions: &BTreeMap<String, Arc<LockV2PublicationSource>>,
    authorize: impl FnOnce() -> Result<(), ArenaError>,
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
    publication::publish_original_payloads_authorized(
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
        authorize,
    )
}


#[cfg(all(test, target_os = "linux", feature = "pals-collection-onnx"))]
mod speculative_publication_boundary_tests {
    use super::*;
    use super::super::collection_hooks_core_v2::{
        GAME_HAS_ABORT, GAME_IDLE, GAME_PHASE_MASK, GAME_PUBLICATION_GRANTED,
    };
    use std::collections::BTreeMap;
    use std::sync::{Arc, atomic::{AtomicBool, AtomicU64, Ordering}};
    use std::time::{Duration, Instant};

    // This is only a source-byte fixture for the actual filesystem writer;
    // it cannot construct /2 native admission or a training corpus.
    struct Source { digest: String }
    impl publication::PublicationSourceBytes for Source {
        fn canonical_lock_sha256(&self) -> &str { &self.digest }
        fn lock_bytes(&self) -> &[u8] { b"{}" }
        fn runtime_bundle(&self) -> Option<(&rz_experiments::ArtifactRef, &[u8])> { None }
    }

    struct Scratch(std::path::PathBuf);
    impl Scratch {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let parent = std::env::temp_dir().canonicalize().expect("temp directory");
            loop {
                let id = NEXT.fetch_add(1, Ordering::Relaxed);
                let path = parent.join(format!("rovent-publication-boundary-{}-{id}", std::process::id()));
                match std::fs::create_dir(&path) {
                    Ok(()) => {
                        std::fs::create_dir(path.join("chunks")).expect("owned chunks directory");
                        // The writer creates input-locks itself using create_dir.
                        // Precreating it would fail before authorization.
                        return Self(path);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(error) => panic!("create owned scratch: {error}"),
                }
            }
        }
        fn directory(&self) -> cap_std::fs::Dir {
            cap_std::fs::Dir::open_ambient_dir(&self.0, cap_std::ambient_authority()).expect("scratch capability")
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
    }

    fn exercise_boundary(abort_before_grant: bool) {
        let scratch = Scratch::new();
        let directory = scratch.directory();
        let mut sources = BTreeMap::new();
        sources.insert("original-root".to_owned(), Arc::new(Source { digest: "a".repeat(64) }));
        let chunks: Vec<rz_experiments::rovent_corpus::RoventCorpusChunk> = Vec::new();
        let material = publication::RoventPublicationMaterial {
            manifest_bytes: b"{}", index_bytes: b"{}", chunks: &chunks, admissions: &sources,
        };
        let cancel = AtomicBool::new(false);
        let expected = (GAME_PHASE_MASK + 1) | GAME_IDLE;
        let gate = AtomicU64::new(expected);
        let authorization_reached = AtomicBool::new(false);
        let result = publication::publish_original_payloads_authorized(
            &directory, Instant::now() + Duration::from_secs(30), &cancel, material,
            |_, _, _, _, _| Ok(serde_json::json!({"test":"publication boundary only"})),
            || {
                authorization_reached.store(true, Ordering::Release);
                // Real pending bytes are already durable before authorization;
                // the visible success link must not exist yet.
                assert!(directory.open("publication.pending.json").is_ok());
                assert!(directory.open("publication.json").is_err());
                if abort_before_grant { gate.fetch_or(GAME_HAS_ABORT, Ordering::AcqRel); }
                gate.compare_exchange(expected, expected | GAME_PUBLICATION_GRANTED,
                    Ordering::AcqRel, Ordering::Acquire)
                    .map_err(|_| invalid("speculative publication game changed or was aborted"))?;
                if !abort_before_grant { gate.fetch_or(GAME_HAS_ABORT, Ordering::AcqRel); }
                Ok(())
            },
        );
        assert!(authorization_reached.load(Ordering::Acquire),
            "publisher did not reach final authorization: {result:?}");
        if abort_before_grant {
            assert!(result.is_err(), "abort before authorization must fail: {result:?}");
            assert!(directory.open("publication.pending.json").is_ok());
            assert!(directory.open("publication.json").is_err());
            assert_eq!(gate.load(Ordering::Acquire), expected | GAME_HAS_ABORT);
        } else {
            // Logical authorization preceded this simulated stop; the actual
            // link appears afterward. This is not a claim of physical pre-stop
            // file visibility or complete model/teacher execution.
            assert!(result.is_ok(), "publication after authorization failed: {result:?}");
            assert!(directory.open("publication.pending.json").is_err());
            assert!(directory.open("publication.json").is_ok());
            assert_eq!(gate.load(Ordering::Acquire), expected | GAME_PUBLICATION_GRANTED | GAME_HAS_ABORT);
        }
    }

    #[test]
    fn abort_before_authorization_preserves_pending_without_publication() {
        exercise_boundary(true);
    }

    #[test]
    fn abort_after_authorization_does_not_rewrite_the_prior_order() {
        exercise_boundary(false);
    }
}
