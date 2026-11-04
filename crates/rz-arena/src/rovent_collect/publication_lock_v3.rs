//! Original /3 source metadata for the shared atomic publication path.
//!
//! This snapshot never creates a live admission, a runtime bundle, or proof
//! that an accelerator or a teacher finished. The writer must retain the
//! original admission owner separately while it accepts rows.
use super::*;
use rz_experiments::RoventInputLockV3;
use rz_experiments::rovent_corpus_lock_v3::publication::{
    ROVENT_LOCK_V3_PUBLICATION_DOMAIN, RoventLockV3InputLockBytesOrigin, RoventLockV3Publication,
    RoventLockV3PublicationAdmission, RoventLockV3PublicationFile,
};
use rz_experiments::rovent_corpus_lock_v3::record::{
    RoventCorpusIndexLockV3, RoventCorpusManifestLockV3,
};
use rz_uci::pals_adaptive::RoventCollectionRootAdmissionV3;
use std::sync::Arc;

pub(super) struct LockV3PublicationSource {
    pub(super) admission_sha256: String,
    pub(super) lock: RoventInputLockV3,
    pub(super) system_runtime_identity_sha256: String,
    lock_bytes: Vec<u8>,
}

impl LockV3PublicationSource {
    /// Capture declaration bytes from the real in-process /3 admission.
    /// The lock contains the declared system runtime, not a CUDA file bundle.
    pub(super) fn capture(admission: &RoventCollectionRootAdmissionV3) -> Result<Self, ArenaError> {
        let origin = admission.native_origin();
        let lock = origin.original_input_lock();
        lock.validate()?;
        check(
            origin.raw_receipt()["input_lock_sha256"] == lock.canonical_sha256,
            "original /3 native origin and input lock differ",
        )?;
        let lock_bytes = serde_json::to_vec(lock).map_err(|error| invalid(error.to_string()))?;
        check(lock_bytes.len() <= 16 * 1024 * 1024, "/3 lock metadata cap")?;
        let runtime_identity =
            origin.raw_receipt()["startup_observation"]["system_runtime_identity_sha256"]
                .as_str()
                .ok_or_else(|| invalid("original /3 system runtime identity is unknown"))?
                .to_owned();
        Ok(Self {
            admission_sha256: admission.raw_sha256().to_owned(),
            lock: lock.clone(),
            system_runtime_identity_sha256: runtime_identity,
            lock_bytes,
        })
    }
}

impl publication::PublicationSourceBytes for LockV3PublicationSource {
    fn canonical_lock_sha256(&self) -> &str {
        &self.lock.canonical_sha256
    }

    fn lock_bytes(&self) -> &[u8] {
        &self.lock_bytes
    }

    fn runtime_bundle(&self) -> Option<(&rz_experiments::ArtifactRef, &[u8])> {
        // The /3 system runtime is recorded in the original lock and native
        // observation. There is intentionally no copied library bundle.
        None
    }
}

fn file(pin: publication::RoventPublicationFile) -> RoventLockV3PublicationFile {
    RoventLockV3PublicationFile {
        path: pin.path,
        sha256: pin.sha256,
        bytes: pin.bytes,
    }
}

pub(super) fn publish_lock_v3_payloads(
    directory: &cap_std::fs::Dir,
    until: std::time::Instant,
    cancel: &std::sync::atomic::AtomicBool,
    manifest: &RoventCorpusManifestLockV3,
    index: &RoventCorpusIndexLockV3,
    admissions: &BTreeMap<String, Arc<LockV3PublicationSource>>,
) -> Result<RoventLockV3Publication, ArenaError> {
    publish_lock_v3_payloads_authorized(
        directory,
        until,
        cancel,
        manifest,
        index,
        admissions,
        || Ok(()),
    )
}

pub(super) fn publish_lock_v3_payloads_speculative(
    directory: &cap_std::fs::Dir,
    until: std::time::Instant,
    cancel: &std::sync::atomic::AtomicBool,
    manifest: &RoventCorpusManifestLockV3,
    index: &RoventCorpusIndexLockV3,
    admissions: &BTreeMap<String, Arc<LockV3PublicationSource>>,
    game_gate: &std::sync::atomic::AtomicU64,
) -> Result<RoventLockV3Publication, ArenaError> {
    publish_lock_v3_payloads_speculative_impl(
        directory,
        until,
        cancel,
        manifest,
        index,
        admissions,
        game_gate,
        false,
        |_, _| Ok(()),
    )
}

/// Managed /3 callers provide their already-bound game authority, which is
/// invoked only at the original final commit point.
pub(super) fn publish_lock_v3_payloads_managed_with_authorize(
    directory: &cap_std::fs::Dir,
    until: std::time::Instant,
    cancel: &std::sync::atomic::AtomicBool,
    manifest: &RoventCorpusManifestLockV3,
    index: &RoventCorpusIndexLockV3,
    admissions: &BTreeMap<String, Arc<LockV3PublicationSource>>,
    authorize: impl FnOnce() -> Result<(), ArenaError>,
    hook: impl FnMut(
        publication::RoventPublicationHookEventV3,
        publication::RoventPublicationHookPhaseV3,
    ) -> Result<(), ArenaError>,
) -> Result<RoventLockV3Publication, ArenaError> {
    publish_lock_v3_payloads_authorized_with_hook(
        directory, until, cancel, manifest, index, admissions, authorize, true, hook,
    )
}

fn publish_lock_v3_payloads_speculative_impl(
    directory: &cap_std::fs::Dir,
    until: std::time::Instant,
    cancel: &std::sync::atomic::AtomicBool,
    manifest: &RoventCorpusManifestLockV3,
    index: &RoventCorpusIndexLockV3,
    admissions: &BTreeMap<String, Arc<LockV3PublicationSource>>,
    game_gate: &std::sync::atomic::AtomicU64,
    managed_paths: bool,
    hook: impl FnMut(
        publication::RoventPublicationHookEventV3,
        publication::RoventPublicationHookPhaseV3,
    ) -> Result<(), ArenaError>,
) -> Result<RoventLockV3Publication, ArenaError> {
    use super::collection_hooks_core_v2::{
        GAME_HAS_ABORT, GAME_IDLE, GAME_PHASE_MASK, GAME_PUBLICATION_GRANTED,
    };
    use std::sync::atomic::Ordering;
    let expected = game_gate.load(Ordering::Acquire);
    check(
        expected & (GAME_HAS_ABORT | GAME_PUBLICATION_GRANTED) == 0
            && expected & GAME_PHASE_MASK == GAME_IDLE,
        "speculative /3 publication game is aborted, busy, or already authorized",
    )?;
    publish_lock_v3_payloads_authorized_with_hook(
        directory,
        until,
        cancel,
        manifest,
        index,
        admissions,
        || {
            game_gate
                .compare_exchange(
                    expected,
                    expected | GAME_PUBLICATION_GRANTED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .map_err(|_| invalid("speculative /3 publication game changed or was aborted"))?;
            Ok(())
        },
        managed_paths,
        hook,
    )
}

fn publish_lock_v3_payloads_authorized(
    directory: &cap_std::fs::Dir,
    until: std::time::Instant,
    cancel: &std::sync::atomic::AtomicBool,
    manifest: &RoventCorpusManifestLockV3,
    index: &RoventCorpusIndexLockV3,
    admissions: &BTreeMap<String, Arc<LockV3PublicationSource>>,
    authorize: impl FnOnce() -> Result<(), ArenaError>,
) -> Result<RoventLockV3Publication, ArenaError> {
    publish_lock_v3_payloads_authorized_with_hook(
        directory,
        until,
        cancel,
        manifest,
        index,
        admissions,
        authorize,
        false,
        |_, _| Ok(()),
    )
}

fn publish_lock_v3_payloads_authorized_with_hook(
    directory: &cap_std::fs::Dir,
    until: std::time::Instant,
    cancel: &std::sync::atomic::AtomicBool,
    manifest: &RoventCorpusManifestLockV3,
    index: &RoventCorpusIndexLockV3,
    admissions: &BTreeMap<String, Arc<LockV3PublicationSource>>,
    authorize: impl FnOnce() -> Result<(), ArenaError>,
    managed_paths: bool,
    hook: impl FnMut(
        publication::RoventPublicationHookEventV3,
        publication::RoventPublicationHookPhaseV3,
    ) -> Result<(), ArenaError>,
) -> Result<RoventLockV3Publication, ArenaError> {
    check(
        !admissions.is_empty(),
        "/3 publication requires original admissions",
    )?;
    for (key, source) in admissions {
        check(
            key == &source.admission_sha256,
            "/3 publication admission identity differs",
        )?;
        manifest.validate(&source.lock)?;
        index.validate(manifest, &source.lock)?;
    }
    let manifest_bytes =
        serde_json::to_vec(manifest).map_err(|error| invalid(error.to_string()))?;
    let index_bytes = serde_json::to_vec(index).map_err(|error| invalid(error.to_string()))?;
    let material = publication::RoventPublicationMaterial {
        manifest_bytes: &manifest_bytes,
        index_bytes: &index_bytes,
        chunks: &index.chunks,
        admissions,
    };
    let build = |manifest_file,
                 index_file,
                 chunks: Vec<publication::RoventPublicationFile>,
                 entries: Vec<publication::RoventPublicationAdmission>,
                 total_payload_bytes| {
        let mut published_admissions = Vec::with_capacity(entries.len());
        for entry in entries {
            check(
                entry.runtime_bundle_manifest.is_none(),
                "/3 system runtime has no copied bundle",
            )?;
            let original = admissions
                .get(&entry.native_admission_sha256)
                .ok_or_else(|| invalid("/3 publication original admission missing"))?;
            published_admissions.push(RoventLockV3PublicationAdmission {
                native_admission_sha256: entry.native_admission_sha256,
                input_lock: file(entry.input_lock),
                input_lock_bytes_origin:
                    RoventLockV3InputLockBytesOrigin::ValidatedSemanticSnapshot,
                canonical_input_lock_sha256: entry.canonical_input_lock_sha256,
                system_runtime_identity_sha256: original.system_runtime_identity_sha256.clone(),
            });
        }
        let result = RoventLockV3Publication {
            schema: ROVENT_LOCK_V3_PUBLICATION_DOMAIN.into(),
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
            admissions: published_admissions,
        };
        result.validate()?;
        Ok(result)
    };
    if managed_paths {
        publication::publish_original_payloads_authorized_with_managed_hook(
            directory, until, cancel, material, build, authorize, hook,
        )
    } else {
        publication::publish_original_payloads_authorized_with_hook(
            directory, until, cancel, material, build, authorize, hook,
        )
    }
}
